"""Four transparent cellular baselines; no PDE solver and no global operations.

The generic NCA here is a deterministic isotropic perception NCA [H, L H, X],
not an exact reproduction of Growing NCA's stochastic update/life-mask setup.
All methods use the same boundary convention and local inputs.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('nca_state_matched', 'momentum_nca', 'rd_nca', 'inertial_rd')


def laplacian(h: torch.Tensor) -> torch.Tensor:
    """Positive five-point Laplacian; replicated ghosts = no-flux grid edges.
    Symmetric PSD, eigenvalues in [0,8]. Unit grid spacing. No domain rescaling.
    """
    z = F.pad(h, (1,1,1,1), mode='replicate')
    return 4*h-z[:,:,1:-1,:-2]-z[:,:,1:-1,2:]-z[:,:,:-2,1:-1]-z[:,:,2:,1:-1]


def logit(x: float) -> float:
    if not 0 < x < 1:
        raise ValueError('logit expects an argument in (0,1)')
    return math.log(x/(1-x))


class Cell(nn.Module):
    def __init__(self, arm: str, channels: int = 16, hidden: int = 128,
                 inputs: int = 3, eta: float = .1, beta0: float = .9,
                 d0: float = .1):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f'Unknown arm: {arm}')
        if channels < 1 or hidden < 1 or eta <= 0:
            raise ValueError('Invalid model sizes or eta')
        self.arm, self.eta = arm, eta
        self.inertial = arm in ('momentum_nca','inertial_rd')
        self.structured = arm in ('rd_nca','inertial_rd')
        self.channels = 2*channels if arm == 'nca_state_matched' else channels
        c = self.channels
        self.encoder = nn.Conv2d(inputs, c, 1)
        self.readout = nn.Conv2d(c, 1, 1)
        inc = c+inputs if self.structured else 2*c+inputs
        self.program = nn.Sequential(nn.Conv2d(inc,hidden,1),nn.Tanh(),nn.Conv2d(hidden,c,1))
        nn.init.zeros_(self.program[-1].weight)
        nn.init.zeros_(self.program[-1].bias)
        # beta is allowed to specialize by channel. Scalar-beta results apply
        # exactly to the pure transport block; nonlinear channel mixing needs
        # the full block Jacobian, not separate scalar eigenvalue tests.
        if self.inertial:
            self.raw_beta = nn.Parameter(torch.full((1,c,1,1), logit(beta0/.995)))
        else:
            self.register_buffer('raw_beta', torch.zeros(1,c,1,1))
        if self.structured:
            cap = .98*2*(1+(beta0 if self.inertial else 0))/8
            self.raw_d = nn.Parameter(torch.full((1,c,1,1),logit(d0/cap)))
        else:
            self.register_buffer('raw_d',torch.zeros(1,c,1,1))

    def coefficients(self):
        b = .995*self.raw_beta.sigmoid() if self.inertial else torch.zeros_like(self.raw_beta)
        d = .98*2*(1+b)/8*self.raw_d.sigmoid() if self.structured else torch.zeros_like(b)
        return b,d

    def initial(self, x):
        h = self.encoder(x)
        return h, torch.zeros_like(h) if self.inertial else None

    def step(self, state, x):
        h,v = state
        lh = laplacian(h)
        features = torch.cat((h,x),dim=1) if self.structured else torch.cat((h,lh,x),dim=1)
        force = self.eta*self.program(features)
        b,d = self.coefficients()
        if self.structured:
            force = force-d*lh
        if self.inertial:
            v = b*v+force
            return h+v,v
        return h+force,None

    def logits(self,state):
        return self.readout(state[0])

    def rollout(self,x,steps,state=None):
        if steps < 0:
            raise ValueError('steps must be nonnegative')
        state = self.initial(x) if state is None else state
        for _ in range(steps):
            state = self.step(state,x)
        return state

    def metadata(self):
        b,d = self.coefficients()
        return {'arm':self.arm,'content_channels':self.channels,
                'persistent_scalars_per_cell':self.channels*(2 if self.inertial else 1),
                'hidden_width':self.program[0].out_channels,
                'parameters':sum(p.numel() for p in self.parameters()),
                'beta':b.detach().flatten().cpu().tolist(),
                'd':d.detach().flatten().cpu().tolist(), 'eta':self.eta}


def make_cell(arm,channels=16,reference_hidden=128):
    """Choose integer hidden width to approximately match trainable parameters with widths divisible by eight.
    This does not claim exact latency, activation-memory, or capacity matching.
    """
    target = sum(p.numel() for p in Cell('inertial_rd',channels,reference_hidden).parameters())
    if arm in ('rd_nca','inertial_rd'):
        hidden = reference_hidden
    else:
        def count(w):
            c = (2*channels if arm=='nca_state_matched' else channels)
            inc = 2*c+3
            return 4*c+(c+1)+(inc+1)*w+(w+1)*c+(c if arm=='momentum_nca' else 0)
        hidden = min(range(8,4*reference_hidden+1,8),key=lambda w:abs(count(w)-target))
    return Cell(arm,channels,hidden)
