"""Graph-safe relation action measurements; no intervention or task labels."""
from __future__ import annotations
import torch


def sufficient(incoming, mixed, x):
    with torch.no_grad():
        b, _, h, w = incoming.shape
        u = incoming.detach().reshape(b, 4, 6, h, w)
        delta = (mixed.detach()-incoming.detach()).reshape_as(u)
        opened = x[:, :1].bool().expand(b, 4, h, w)
        axes = (0, 2, 3)
        return torch.stack((opened.sum(axes, dtype=torch.float64),
            (u.square().sum(2)*opened).sum(axes, dtype=torch.float64),
            (delta.square().sum(2)*opened).sum(axes, dtype=torch.float64)), dim=1)


def parameters(model):
    if model.arm != 'rrc':
        return {'rho': None, 'pi': None, 'beta': 0., 'cyclic_mode_gains': [1.]*4}
    with torch.no_grad():
        pi = torch.softmax(model.relation_logits, dim=0).detach().cpu().tolist()
        rho = float(torch.sigmoid(model.relation_gate).detach().cpu())
    import cmath
    coefficients = [rho*p for p in pi]
    coefficients[0] += 1-rho
    gains = [abs(sum(coefficients[j]*cmath.exp(2j*cmath.pi*j*k/4)
                     for j in range(4))) for k in range(4)]
    return {'rho': rho, 'pi': pi, 'beta': rho*(1-pi[0]), 'cyclic_mode_gains': gains}


def gradient_norm(model):
    if model.arm != 'rrc':
        return None
    values = (model.relation_logits.grad, model.relation_gate.grad)
    assert all(v is not None for v in values)
    return float(torch.stack([v.detach().square().sum() for v in values]).sum().sqrt().cpu())


def rows(stats, **labels):
    result = []
    for lane, (count, energy, action) in enumerate(stats.detach().cpu().tolist()):
        result.append({**labels, 'lane': lane, 'lane_name': ('N','E','S','W')[lane],
            'open_cell_events': int(count), 'incoming_energy': energy,
            'relation_action_energy': action,
            'relative_action_l2': (action/energy)**.5 if energy else None})
    return result


class TrainingActivity:
    def __init__(self, model):
        self.stats = torch.zeros(4, 3, dtype=torch.float64, device=next(model.parameters()).device)

    def __call__(self, model, incoming, mixed, x):
        self.stats.add_(sufficient(incoming, mixed, x))

    def reset(self):
        self.stats.zero_()

    def result(self):
        return rows(self.stats, phase='cold_1_64_repeated_four_times')


class EvaluationActivity:
    def __init__(self, model):
        self.calls = {32: 0, 64: 0}
        self.stats = {(s,w,p): torch.zeros(4,3,dtype=torch.float64,
            device=next(model.parameters()).device)
            for s in (32,64) for w in ('original','flipped') for p in ('1_64','65_256')}

    def __call__(self, model, incoming, mixed, x):
        size = x.shape[-1]
        call = self.calls[size]
        self.calls[size] += 1
        step = call//2+1
        world = 'original' if call%2 == 0 else 'flipped'
        self.stats[(size,world,'1_64' if step<=64 else '65_256')].add_(sufficient(incoming,mixed,x))

    def result(self):
        assert self.calls == {32:512,64:512}, self.calls
        return [row for (s,w,p),stats in self.stats.items()
                for row in rows(stats,size=s,world=w,phase=p)]
