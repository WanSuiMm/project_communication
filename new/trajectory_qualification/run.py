"""Persistent serial controller: selected-state screen -> fresh reliability -> independent task."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from common import (ROOT, PRIOR, setup_backend, sha, tensor_hash, write, now,
                    load_state, train_updates, region_bank)
from state_cross import run_stage as state_cross, select_recipe, ReplayError, NATIVE
from fresh_recipe import run_stage as fresh_recipe, check_statistics
from distance_task import run_stage as distance_task, cpu_check, bank as distance_bank, backward_distance
from integration_check import check as check_fresh_contracts

PROTOCOL = 'trajectory_qualification_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_hashes():
    old = read(PRIOR / 'manifest.json')['source_sha256']
    for name, digest in old.items():
        assert sha(ROOT / name) == digest, f'Original source drift: {name}'
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT / 'new/trajectory_qualification').glob('*')
             if p.suffix in ('.py', '.md')]
    names += ['tools/launch_trajectory_qualification.ps1']
    return {**old, **{name: sha(ROOT / name) for name in sorted(names)}}


def reference_hashes():
    names = [f'checkpoints/{name}_u003.pt' for name in ('H', 'S20022')]
    names += [name + suffix for name in NATIVE.values() for suffix in ('.json', '_summary.json')]
    return {f'runs/bootstrap_20261004_seed4_01/{name}': sha(PRIOR / name) for name in names}


def selector_check():
    def rows(moment, parameter):
        values = {('H', 'H', 'H'): True, ('H', 'H', 'S'): True,
                  ('S', 'S', 'H'): True, ('S', 'S', 'S'): False}
        values.update({('S', 'H', s): moment for s in ('H', 'S')})
        values.update({('H', 'S', s): parameter for s in ('H', 'S')})
        return [{'theta': p, 'moments': m, 'suffix': s, 'endpoint': {'pass': v}}
                for (p, m, s), v in values.items()]
    for m, p, expected in ((True, True, 'shadow_moments'), (True, False, 'shadow_moments'),
                           (False, True, 'shadow_parameters'), (False, False, 'shadow_joint')):
        assert select_recipe(rows(m, p))['recipe'] == expected
    return {'status': 'PASS', 'fixtures': 4}


def check(out):
    assert not out.exists(), 'Check output already exists'
    sources = source_hashes()
    references = reference_hashes()
    states = {name: torch.load(PRIOR / 'checkpoints' / f'{name}_u003.pt',
                              map_location='cpu', weights_only=False)
              for name in ('H', 'S20022')}
    h, s = states.values()
    assert list(h['state_dict']) == list(s['state_dict'])
    for value in states.values():
        assert value['completed_updates'] == 3
        optimizer = value['optimizer_state_dict']
        assert len(optimizer['state']) == len(value['state_dict']) == 12
        assert len(optimizer['param_groups']) == 1
        group = optimizer['param_groups'][0]
        assert (group['lr'], group['weight_decay'], group['betas'], group['eps']) == (.001, .0001, (.9, .999), 1e-8)
        for parameter, name in zip(group['params'], value['state_dict']):
            entry = optimizer['state'][parameter]
            assert int(entry['step'].item()) == 3
            assert entry['exp_avg'].shape == entry['exp_avg_sq'].shape == value['state_dict'][name].shape
    assert h['optimizer_state_dict']['param_groups'] == s['optimizer_state_dict']['param_groups']
    result = {'protocol': PROTOCOL, 'source_sha256': sources, 'reference_sha256': references,
              'checkpoint_schema': {'status': 'PASS', 'parameters': 12, 'step': 3},
              'selector': selector_check(), 'statistics': check_statistics(), 'distance_oracle': cpu_check()}
    result['fresh_runner_contracts'] = check_fresh_contracts(out.parent / (out.stem + '_contracts'))
    setup_backend()
    model, optimizer = load_state(h, s)
    data = region_bank(32, 8, 10002, 'cuda')
    train_updates(model, optimizer, data, [list(range(8))], start=4)
    assert all(int(value['step'].item()) == 4 for value in optimizer.state.values())
    result['original_cuda_update'] = 'PASS'
    from common import initial, optimizer_for
    del model, optimizer, data
    gc.collect()
    model = initial(41001)
    optimizer = optimizer_for(model)
    data = distance_bank(8, 2, 81901, 'cuda')
    optimizer.zero_grad(set_to_none=True)
    loss, state, cadence = backward_distance(model, data)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert bool(torch.isfinite(norm)) and all(bool(torch.isfinite(v).all()) for v in state)
    optimizer.step()
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    result['distance_cuda_update'] = {'status': 'PASS', 'loss': float(loss), **cadence}
    result['status'] = 'PASS'
    result['runtime_limit_enforced'] = False
    result['checked_utc'] = now()
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, result)
    print(json.dumps({'status': 'PASS', 'qualification': out.relative_to(ROOT).as_posix()}), flush=True)


def freeze_plans(out):
    folder = out / 'plans'
    folder.mkdir()
    hashes = {}
    families = [('stage1_H', 20002), ('stage1_S', 20022)]
    families += [(f'stage2_seed{31001 + i}', 32001 + i) for i in range(16)]
    families += [(f'stage3_seed{41001 + i}', 42001 + i) for i in range(8)]
    for name, seed in families:
        path = folder / f'{name}.json'
        write(path, np.random.default_rng(seed).integers(0, 512, (300, 8)).tolist())
        hashes[path.relative_to(out).as_posix()] = sha(path)
    return hashes


def run(out, qualification):
    assert not out.exists(), 'Run output already exists'
    q = read(qualification)
    assert q['status'] == 'PASS' and q['protocol'] == PROTOCOL
    sources = source_hashes()
    references = reference_hashes()
    assert q['source_sha256'] == sources and q['reference_sha256'] == references
    out.mkdir(parents=True)
    started = time.monotonic()
    setup_backend()
    plans = freeze_plans(out)
    manifest = {'protocol': PROTOCOL, 'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'),
                'started_utc': now(), 'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
                'command': [sys.executable, *sys.argv], 'source_sha256': sources,
                'reference_sha256': references, 'qualification': qualification.relative_to(ROOT).as_posix(),
                'qualification_sha256': sha(qualification), 'schedule_plan_sha256': plans,
                'stage_order': [1, 2, 3], 'stage_expected_arms': {'1': 8, '2': 32, '3': 8},
                'stage2_depends_on_stage1': True, 'stage3_independent': True,
                'runtime_limit_enforced': False, 'maximum_seconds': None,
                'watchdog_enabled': False, 'continuous_monitoring': False}
    write(out / 'manifest.json', manifest)
    for name in sources:
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    stages = {}

    def progress(value):
        write(out / 'status.json', {'status': 'RUNNING', 'pid': os.getpid(),
                                    'elapsed_seconds': time.monotonic() - started,
                                    'updated_utc': now(), 'stages': stages, **value})
        print(json.dumps(value), flush=True)

    status = 'ERROR'
    try:
        progress({'stage': 1, 'phase': 'starting', 'completed_updates': 0})
        try:
            first = state_cross(out / 'stage1_state_cross', progress)
        except ReplayError as error:
            stages['1'] = {'status': 'CONTROL_UNQUALIFIED', 'error': repr(error)}
            stages['2'] = {'status': 'SKIPPED', 'reason': 'Stage1 native replay unqualified'}
            write(out / 'stage1_state_cross' / 'RESULTS.md', 'Stage1 control replay unqualified; stage2 skipped. Stage3 remains independent.\n')
        else:
            stages['1'] = {'status': 'COMPLETE', 'completed_arms': 8, 'selection': first['selection']}
            write(out / 'selection.json', first['selection'])
            progress({'stage': 2, 'phase': 'starting', 'completed_updates': 0, 'recipe': first['selection']['recipe']})
            try:
                second = fresh_recipe(out / 'stage2_fresh_recipe', first['selection'],
                                      lambda row: progress({'stage': 2, **{k: v for k, v in row.items() if k != 'status'}}))
            except Exception as error:
                message = str(error).lower()
                if isinstance(error, FloatingPointError) or any(token in message for token in
                        ('nonfinite', 'cuda error', 'device-side assert', 'out of memory')):
                    raise
                stages['2'] = {'status': 'ERROR', 'error': repr(error)}
                write(out / 'stage2_failure.json', {'error': repr(error), 'traceback': traceback.format_exc(),
                                                   'stage3_continues_independently': True})
            else:
                stages['2'] = {'status': 'COMPLETE', 'completed_arms': 32,
                               'decision': second['primary_paired_result']['verdict']}
        gc.collect()
        torch.cuda.empty_cache()
        progress({'stage': 3, 'phase': 'starting', 'completed_updates': 0})
        third = distance_task(out / 'stage3_distance', progress_callback=
                              lambda row: progress({'stage': 3, **{k: v for k, v in row.items() if k != 'status'}}))
        assert third['status'] == 'COMPLETE', 'Independent task execution incomplete'
        stages['3'] = {'status': 'COMPLETE', 'completed_arms': 8,
                       'decision': third.get('decision', third.get('status'))}
        assert source_hashes() == sources and reference_hashes() == references, 'Source/reference drift during suite'
        assert all(sha(out / name) == digest for name, digest in plans.items()), 'Frozen plan drift'
        if stages['1']['status'] != 'COMPLETE':
            status = 'COMPLETE_WITH_UNQUALIFIED_STAGE1'
        elif stages['2']['status'] != 'COMPLETE':
            status = 'COMPLETE_WITH_STAGE2_ERROR'
        else:
            status = 'COMPLETE'
    except Exception as error:
        write(out / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
    result = {'status': status, 'stages': stages, 'elapsed_seconds': time.monotonic() - started,
              'finished_utc': now(), 'runtime_limit_enforced': False}
    write(out / 'summary.json', result)
    write(out / 'status.json', {**result, 'pid': os.getpid()})
    lines = ['# Serial qualification suite', '', 'Execution status: ' + status, '',
             '1. [State cross](stage1_state_cross/RESULTS.md).',
             '2. [Fresh recipe](stage2_fresh_recipe/RESULTS.md), when stage1 qualified.',
             '3. [Independent distance task](stage3_distance/RESULTS.md).', '',
             'Scientific qualification failures are retained separately from execution completion.',
             'Read per-stage summaries first; checkpoints, arrays and machine records are secondary.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    if status == 'ERROR':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--qualification')
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    assert out.is_relative_to(ROOT / ('analyses' if args.check else 'runs'))
    if args.check:
        check(out)
    else:
        assert args.qualification
        qualification = (ROOT / args.qualification).resolve()
        assert qualification.is_relative_to(ROOT / 'analyses')
        run(out, qualification)


if __name__ == '__main__':
    main()
