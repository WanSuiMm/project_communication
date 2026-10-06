"""Read-only, graph-safe sufficient statistics for fixed clipping dose."""
from __future__ import annotations

import torch


def sufficient(details, x):
    """Per lane: event count, trigger count, raw RMS sum, removed RMS sum.

    Reductions use double precision; the writer and its gradients remain FP32.
    No dynamic masked indexing, host transfer or task label is needed.
    """
    with torch.no_grad():
        raw = details['raw_write'].detach()
        clipped = details['clipped_write'].detach()
        batch, _, height, width = raw.shape
        lanes = raw.reshape(batch, 4, 6, height, width)
        removed = (raw-clipped).reshape_as(lanes)
        r2 = lanes.square().mean(2)
        r = r2.sqrt()
        removed_r = removed.square().mean(2).sqrt()
        opened = x[:, :1].bool().expand_as(r)
        caps2 = details['caps'].detach().square().reshape(1, 4, 1, 1)
        axes = (0, 2, 3)
        return torch.stack((opened.sum(axes, dtype=torch.float64),
                            ((r2 > caps2) & opened).sum(axes, dtype=torch.float64),
                            (r*opened).sum(axes, dtype=torch.float64),
                            (removed_r*opened).sum(axes, dtype=torch.float64)), dim=1)


def rows(stats, *, actual_clipping, **labels):
    values = stats.detach().cpu().tolist()
    result = []
    for lane, (count, trigger, raw, removed) in enumerate(values):
        result.append({**labels, 'lane': lane, 'lane_name': ('N', 'E', 'S', 'W')[lane],
            'event_count': int(count), 'trigger_count': int(trigger),
            'raw_rms_sum': raw, 'removed_rms_sum': removed,
            'trigger_fraction': trigger/count if count else None,
            'removed_fraction': removed/raw if raw else None,
            'actual_clipping': actual_clipping,
            'interpretation': 'actual' if actual_clipping else 'counterfactual_fixed_cap'})
    return result


class TrainingDose:
    def __init__(self, model):
        self.actual_clipping = model.arm == 'hardclip'
        # Plain tensor outside state_dict: resets cannot change model provenance.
        self.stats = torch.zeros(4, 4, dtype=torch.float64, device=next(model.parameters()).device)

    def __call__(self, details, x):
        self.stats.add_(sufficient(details, x))

    def reset(self):
        self.stats.zero_()

    def result(self):
        return rows(self.stats, actual_clipping=self.actual_clipping,
                    phase='cold_1_64_repeated_four_times')


class EvaluationDose:
    def __init__(self, model):
        self.actual_clipping = model.arm == 'hardclip'
        self.calls = {32: 0, 64: 0}
        self.stats = {(size, world, phase): torch.zeros(4, 4, dtype=torch.float64,
                       device=next(model.parameters()).device)
                      for size in (32, 64) for world in ('original', 'flipped')
                      for phase in ('1_64', '65_256')}

    def __call__(self, details, x):
        size = x.shape[-1]
        call = self.calls[size]
        self.calls[size] += 1
        step = call//2+1
        world = 'original' if call%2 == 0 else 'flipped'
        phase = '1_64' if step <= 64 else '65_256'
        self.stats[(size, world, phase)].add_(sufficient(details, x))

    def result(self):
        assert self.calls == {32: 512, 64: 512}, self.calls
        return [row for (size, world, phase), stats in self.stats.items()
                for row in rows(stats, actual_clipping=self.actual_clipping,
                                size=size, world=world, phase=phase)]
