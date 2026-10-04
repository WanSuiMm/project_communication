"""Shared frozen cell and training operations for the serial qualification."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for relative in ('new/nca_inertial_wind_tunnel', 'new/workspace_revision',
                 'new/short_bptt', 'new/streaming_carry', 'new/seed4_followup'):
    sys.path.insert(0, str(ROOT / relative))
from stream_cells import StreamingCell
from training import backward_trajectory
from tasks import bank as region_bank, subset
from phenotype import evaluate, _load_audit
from run_revision import now, sha, tensor_hash, write

PRIOR = ROOT / 'runs/bootstrap_20261004_seed4_01'
OPTIMIZER = {'lr': .001, 'weight_decay': .0001, 'betas': (.9, .999), 'eps': 1e-8}


def setup_backend():
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'Historical local CUDA backend required'
    assert str(torch.__version__).startswith('2.5.1'), 'Historical Torch2.5.1 required'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


def initial(seed):
    torch.manual_seed(seed)
    model = StreamingCell().cuda()
    assert sum(p.numel() for p in model.parameters()) == 5033
    return model


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), **OPTIMIZER)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def payload(model, optimizer, seed, update):
    return {'variant': 'stream', 'initialization_seed': seed,
            'completed_updates': update, 'state_dict': cpu_tree(model.state_dict()),
            'optimizer_state_dict': cpu_tree(optimizer.state_dict())}


def load_state(theta, moments, seed=4):
    assert theta['completed_updates'] == moments['completed_updates']
    model = initial(seed)
    model.load_state_dict(theta['state_dict'])
    optimizer = optimizer_for(model)
    optimizer.load_state_dict(copy.deepcopy(moments['optimizer_state_dict']))
    assert tensor_hash(model.state_dict()) == tensor_hash(theta['state_dict'])
    assert list(model.state_dict()) == list(theta['state_dict'])
    assert len(optimizer.state) == len(list(model.parameters()))
    for state in optimizer.state.values():
        assert int(state['step'].item()) == theta['completed_updates']
    return model, optimizer


def train_updates(model, optimizer, train, rows, start=1, progress=lambda _: None):
    curve = []
    model.train()
    for update, ids in enumerate(rows, start):
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss, state, cadence = backward_trajectory(model, subset(train, ids), 8)
        assert all(bool(torch.isfinite(v).all()) for v in state), 'Nonfinite state'
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert bool(torch.isfinite(norm)), 'Nonfinite gradient'
        optimizer.step()
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite parameter'
        torch.cuda.synchronize()
        row = {'update': update, 'mean_trajectory_loss': float(loss),
               'gradient_norm_before_clip': float(norm),
               'seconds': time.monotonic() - tick, **cadence}
        curve.append(row)
        if update == start or update % 25 == 0 or update == 300:
            progress({'phase': 'training', 'completed_updates': update, **row})
    return curve


def compact(summary):
    result = {'pass': summary['phenotype_gate']['pass'], 'sizes': {}}
    for size, row in summary['sizes'].items():
        ends = row['endpoints']
        result['sizes'][size] = {
            'strict_mean': {t: ends[t]['strict_16_32']['mean_map_coverage'] for t in ('64', '128', '256')},
            'strict_pooled': {t: ends[t]['strict_16_32']['pooled_coverage']['value'] for t in ('64', '128', '256')},
            'coverage': {t: ends[t]['all_changed']['pooled_coverage']['value'] for t in ('64', '128', '256')},
            'retention64_to256': row['transitions']['all_changed']['64_to_256']['pooled']['retention']['value'],
            'ever_regressed_fraction': row['ever_regressed_over_ever_correct']['value'],
            'frontier_effect': row['frontier']['mean_map_weighted_difference']}
    return result


def save_checkpoint(out, name, model, optimizer, seed, update=300):
    folder = out / 'checkpoints'
    folder.mkdir(exist_ok=True)
    path = folder / f'{name}_u{update:03}.pt'
    torch.save(payload(model, optimizer, seed, update), path)
    return {'path': path.relative_to(out).as_posix(), 'sha256': sha(path),
            'parameter_sha256': tensor_hash(model.state_dict())}


def compare_replay(previous, measured):
    counts = {'integer_leaves': 0, 'float_leaves': 0, 'maximum_absolute_error': 0.}
    _load_audit().replay_module.compare_tree(previous, measured, 'native.sizes', counts)
    return {'status': 'PASS', **counts}
