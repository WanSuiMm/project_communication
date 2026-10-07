"""Small graph-safe action records; no labels or learned controllers."""
from __future__ import annotations

import math

import torch


def sufficient(model, incoming, delta, conditional_delta, x):
    with torch.no_grad():
        # Cast before scaling and squaring: a finite float32 carrier can have a
        # square above float32 range even though its float64 energy is finite.
        u = incoming.detach().to(torch.float64).reshape(
            incoming.shape[0], 4, 6, *incoming.shape[2:])
        opened = x[:, :1].bool().expand(u.shape[0], 4, *u.shape[-2:])
        eta = float(model.eta)
        axes = (0, 2, 3)

        def masked_energy(values):
            # where (rather than multiplying by a 0/1 mask) also discards
            # nonfinite values in closed cells, since NaN * 0 is still NaN.
            per_cell = values.square().sum(2)
            return torch.where(opened, per_cell, 0.0).sum(axes)

        incoming_energy = masked_energy(u)
        # The cell contract supplies exact zeros for arms without these terms.
        # Avoid converting and reducing those full feature maps on every call.
        if model.arm == 'current':
            relation_energy = torch.zeros_like(incoming_energy)
        else:
            d = (delta.detach().to(torch.float64) * eta).reshape_as(u)
            relation_energy = masked_energy(d)
        if model.arm in ('current', 'constant'):
            conditioned_energy = torch.zeros_like(incoming_energy)
        else:
            c = (conditional_delta.detach().to(torch.float64) * eta).reshape_as(u)
            conditioned_energy = masked_energy(c)
        return torch.stack((opened.sum(axes, dtype=torch.float64), incoming_energy,
                            relation_energy, conditioned_energy), dim=1)


def parameters(model):
    if model.arm == 'current':
        return {'state_conditioned': False, 'relation_parameters': 0}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if k.startswith('relation_')}
    return {'state_conditioned': model.arm == 'conditioned',
            'relation_parameters': sum(v.numel() for v in state.values()),
            'tensors': {k: v.tolist() for k, v in state.items()}}


def gradient_norm(model):
    values = [v.grad for k, v in model.named_parameters() if k.startswith('relation_')]
    if not values:
        return None
    if any(v is None for v in values):
        raise RuntimeError('Relation parameter disconnected from task loss')
    return float(torch.stack([
        v.detach().to(torch.float64).square().sum() for v in values
    ]).sum().sqrt())


def rows(stats, **labels):
    result = []
    for lane, values in enumerate(stats.detach().cpu().tolist()):
        count, energy, action, conditional = values
        energy_ok, action_ok, conditional_ok = map(
            math.isfinite, (energy, action, conditional))

        def relative(numerator, numerator_ok):
            if not energy_ok or not numerator_ok or energy <= 0:
                return None, False
            # Taking roots before division avoids overflowing an intermediate
            # energy ratio when the resulting norm ratio is still finite.
            value = math.sqrt(numerator) / math.sqrt(energy)
            return (value, False) if math.isfinite(value) else (None, True)

        relative_action, relative_action_nonfinite = relative(action, action_ok)
        relative_conditional, relative_conditional_nonfinite = relative(
            conditional, conditional_ok)
        result.append({**labels, 'lane': lane, 'lane_name': ('N', 'E', 'S', 'W')[lane],
                       'open_cell_events': int(count),
                       'incoming_energy': energy if energy_ok else None,
                       'scaled_relation_write_energy': action if action_ok else None,
                       'scaled_state_conditioned_write_energy': conditional if conditional_ok else None,
                       'incoming_energy_nonfinite': not energy_ok,
                       'scaled_relation_write_energy_nonfinite': not action_ok,
                       'scaled_state_conditioned_write_energy_nonfinite': not conditional_ok,
                       'relative_relation_write_l2': relative_action,
                       'relative_relation_write_l2_nonfinite': relative_action_nonfinite,
                       'relative_conditioned_write_l2': relative_conditional,
                       'relative_conditioned_write_l2_nonfinite': relative_conditional_nonfinite})
    return result


class TrainingActivity:
    def __init__(self, model):
        self.stats = torch.zeros(4, 4, dtype=torch.float64, device=next(model.parameters()).device)

    def __call__(self, model, incoming, delta, conditional_delta, x):
        self.stats.add_(sufficient(model, incoming, delta, conditional_delta, x))

    def reset(self):
        self.stats.zero_()

    def result(self):
        return rows(self.stats, phase='cold_1_64_repeated_four_times')


class EvaluationActivity:
    def __init__(self, model):
        self.calls = {32: 0, 64: 0}
        self.stats = {(s, w, p): torch.zeros(4, 4, dtype=torch.float64,
                         device=next(model.parameters()).device)
                      for s in (32, 64) for w in ('original', 'flipped')
                      for p in ('1_64', '65_256')}

    def __call__(self, model, incoming, delta, conditional_delta, x):
        size = x.shape[-1]
        call = self.calls[size]
        self.calls[size] += 1
        step, world = call // 2 + 1, 'original' if call % 2 == 0 else 'flipped'
        phase = '1_64' if step <= 64 else '65_256'
        self.stats[(size, world, phase)].add_(
            sufficient(model, incoming, delta, conditional_delta, x))

    def result(self):
        if self.calls != {32: 512, 64: 512}:
            raise RuntimeError(f'Unexpected evaluation cadence: {self.calls}')
        return [row for (size, world, phase), stats in self.stats.items()
                for row in rows(stats, size=size, world=world, phase=phase)]
