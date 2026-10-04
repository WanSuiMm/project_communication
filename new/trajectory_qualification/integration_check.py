"""Cheap CPU control-flow fixture for all three fresh-state transplant recipes.

Synthetic updates/evaluations exercise runner contracts, not scientific efficacy.
No model forward or optimizer update is performed in this fixture.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

import common as C
import fresh_recipe as F


def check(out):
    out.mkdir(parents=True, exist_ok=False)
    saved = {key: getattr(C, key) for key in ('setup_backend', 'initial', 'region_bank', 'train_updates', 'evaluate', 'compact')}
    saved_device_name = torch.cuda.get_device_name
    reference = json.loads((C.PRIOR / 'H_summary.json').read_text(encoding='utf-8'))

    def initial(seed):
        torch.manual_seed(seed)
        return C.StreamingCell()

    def updates(model, optimizer, train, rows, start=1, progress=lambda _: None):
        last = start + len(rows) - 1
        delta = float(np.asarray(rows).sum()) / 1e6
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(delta)
                optimizer.state[parameter] = {'step': torch.tensor(float(last)),
                    'exp_avg': torch.full_like(parameter, delta),
                    'exp_avg_sq': torch.full_like(parameter, delta * delta)}
        return [{'update': u, 'mean_trajectory_loss': 0., 'gradient_norm_before_clip': 0.,
                 'seconds': 0., 'backward_calls': 8, 'interior_detach_boundaries': 7,
                 'forward_steps': 64, 'loss_count': 8} for u in range(start, last + 1)]

    def evaluation(model, data, folder, name, budget):
        value = copy.deepcopy(reference)
        value['model_name'] = name
        C.write(folder / f'{name}_summary.json', value)
        return value

    def compact(value):
        result = saved['compact'](value)
        # A missing continuous stratum must not crash the final aggregate.
        result['sizes']['64']['frontier_effect'] = None
        return result

    try:
        C.setup_backend = lambda: None
        C.initial = initial
        C.region_bank = lambda *args, **kwargs: {'x': torch.zeros(2, 3, 4, 4)}
        C.train_updates = updates
        C.evaluate = evaluation
        C.compact = compact
        torch.cuda.get_device_name = lambda *args, **kwargs: 'CPU_SYNTHETIC_FIXTURE'
        results = {}
        for recipe in ('shadow_moments', 'shadow_parameters', 'shadow_joint'):
            value = F.run_stage(out / recipe, {'recipe': recipe})
            primary = value['primary_paired_result']
            assert value['status'] == 'COMPLETE' and primary['n_pairs'] == 16
            assert primary['baseline_passes'] == primary['treatment_passes'] == 16
            assert primary['exact_two_sided_mcnemar_binomial_p'] == 1.
            results[recipe] = 'PASS'
        return {'status': 'PASS', 'recipes': results,
                'scope': 'Synthetic CPU runner contracts only; zero scientific training/evaluation.'}
    finally:
        for key, value in saved.items():
            setattr(C, key, value)
        torch.cuda.get_device_name = saved_device_name
