"""Exact matrix-free tangents/adjoints for the frozen generic cells."""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
for directory in ('masked_state', 'masked_momentum', 'masked_medium', 'nca_inertial_wind_tunnel'):
    sys.path.insert(0, str(HERE.parent/directory))
from cells import make_cell as original_factory, laplacian
from masked_cells import masked_laplacian
from state_cells import make_cell as state_factory
from momentum_cells import make_cell as momentum_factory


def factory(arm):
    if arm == 'masked_state_nca':
        return state_factory()
    if arm == 'masked_momentum_nca':
        return momentum_factory()
    return original_factory(arm)


def base(model):
    return getattr(model, 'base', model)


def pack(state):
    return state[0] if state[1] is None else torch.cat(state, dim=1)


def unpack(s, model):
    c = base(model).channels
    return (s[:, :c], s[:, c:]) if base(model).inertial else (s, None)


def norm(x):
    # Float64 reduction avoids overflow when a float32 vector is still finite.
    return x.double().square().flatten(1).sum(1).sqrt()


def mul(x, values):
    return x*values.to(x.dtype).view(-1, 1, 1, 1)


def unit(x):
    n = norm(x)
    return mul(x, 1/n.clamp_min(1e-300)), n


def dot(a, b):
    return (a.double()*b.double()).flatten(1).sum(1)


def conv(x, weight):
    return F.conv2d(x, weight)


class LinearStep:
    """All derivatives hold the binary geometry fixed; source channels may vary."""
    def __init__(self, model, h, x):
        self.model = model
        b = base(model)
        self.c, self.inertial = b.channels, b.inertial
        self.mask = x[:, :1]
        self.masked = model.arm.startswith('masked_')
        self.eta = b.eta
        self.beta = b.coefficients()[0]
        w1, w2 = b.program[0].weight, b.program[2].weight
        self.wh, self.wl, self.wx = w1[:, :self.c], w1[:, self.c:2*self.c], w1[:, 2*self.c:]
        self.w2 = w2
        pre = b.program[0](torch.cat((h, self.lap(h), x), dim=1))
        self.slope = 1-pre.tanh().square()

    def lap(self, x):
        return masked_laplacian(x, self.mask) if self.masked else laplacian(x)

    def force(self, h):
        return self.eta*conv(self.slope*(conv(h, self.wh)+conv(self.lap(h), self.wl)), self.w2)

    def force_adj(self, h):
        q = self.eta*self.slope*conv(h, self.w2.transpose(0, 1))
        return conv(q, self.wh.transpose(0, 1))+self.lap(conv(q, self.wl.transpose(0, 1)))

    def apply(self, s):
        h = s[:, :self.c]
        f = self.force(h)
        if not self.inertial:
            return h+f
        v = self.beta*s[:, self.c:]+f
        return torch.cat((h+v, v), dim=1)

    def adj(self, s):
        h = s[:, :self.c]
        if not self.inertial:
            return h+self.force_adj(h)
        q = h+s[:, self.c:]
        return torch.cat((h+self.force_adj(q), self.beta*q), dim=1)

    def input_apply(self, dx):
        # dx[:,0] must be zero: changing the geometry also differentiates L_m.
        f = self.eta*conv(self.slope*conv(dx, self.wx), self.w2)
        return torch.cat((f, f), dim=1) if self.inertial else f


def product(steps, mask):
    def forward(v):
        v = v*mask
        for step in steps:
            v = step.apply(v)
        return v*mask

    def backward(v):
        v = v*mask
        for step in reversed(steps):
            v = step.adj(v)
        return v*mask
    return forward, backward


def estimate(forward, backward, shape, device, dtype, mask, iterations, starts, seed, check=lambda: None):
    """Per-example power iteration; estimates, never certified upper bounds."""
    generator = torch.Generator(device=device).manual_seed(seed)
    results, vectors, gains = [], [], []
    for _ in range(starts):
        v, _ = unit(torch.randn(shape, device=device, dtype=dtype, generator=generator)*mask)
        history = []
        for iteration in range(iterations):
            check()
            u, gain = unit(forward(v))
            v, _ = unit(backward(u))
            if not bool(torch.isfinite(v).all()):
                raise FloatingPointError('Nonfinite power vector')
            if iteration in (iterations//2-1, iterations-1):
                history.append(gain.cpu().tolist())
        pv = forward(v)
        u, gain = unit(pv)
        residual = norm(backward(u)-mul(v, gain))/gain.clamp_min(1e-300)
        vectors.append(v)
        gains.append(gain)
        results.append({'sigma': gain.cpu().tolist(), 'relative_singular_residual': residual.cpu().tolist(),
                        'iteration_checkpoints': history})
    stacked = torch.stack(gains)
    best = stacked.argmax(0)
    selected = torch.stack(vectors)[best, torch.arange(shape[0], device=device)]
    rows = []
    for i, choice in enumerate(best.cpu().tolist()):
        value = results[choice]
        spread = float((stacked[:, i].max()-stacked[:, i].min())/stacked[:, i].max().clamp_min(1e-300))
        residual = value['relative_singular_residual'][i]
        rows.append({'sigma_estimate': value['sigma'][i], 'relative_singular_residual': residual,
                     'start_relative_spread': spread, 'converged': residual <= .01 and spread <= .02,
                     'starts': [{key: (val[i] if key != 'iteration_checkpoints' else [a[i] for a in val])
                                 for key, val in r.items()} for r in results]})
    return rows, selected
