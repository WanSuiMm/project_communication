"""Frozen native-carrier ladder; run from the standalone repository root."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'new/trajectory_qualification'))
import common as C
from fresh_recipe import exact_two_sided_binomial_p, wilson_interval


def local_module(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = local_module('_native_latent_width_cells', 'cells.py')
PROTOCOL = 'native_latent_width_v1'
ARM_NAMES = ('w24', 'w8', 'w24_capacity', 'w16', 'w4', 'w2_alternating')
PARAMETERS = dict(zip(ARM_NAMES, (5033, 2521, 2521, 3777, 1893, 1579)))
INIT_SEEDS = tuple(range(91001, 91017))
SCHEDULE_SEEDS = tuple(range(92001, 92017))
EVAL_SEEDS = {32: 99332, 64: 99364}
UPDATES = 300


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_hashes():
    # Bind the complete frozen evaluator import chain, not merely the new cell.
    names = set(read(ROOT / 'STREAMING_CARRY_PUBLICATION_MANIFEST.json')['source_sha256'])
    names.update({
        'STREAMING_CARRY_PUBLICATION_MANIFEST.json',
        'new/trajectory_qualification/common.py',
        'new/trajectory_qualification/fresh_recipe.py',
        'new/seed4_followup/phenotype.py',
        'new/frontier_audit/audit.py', 'new/frontier_audit/metrics.py',
        'new/frontier_audit/report.py', 'new/stream_path_audit/audit.py',
        'new/stream_path_audit/operators.py',
        'tools/launch_latent_width.ps1',
    })
    names.update(p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.iterdir()
                 if p.suffix in ('.py', '.md'))
    return {name: C.sha(ROOT / name) for name in sorted(names)}


def arm_order(block):
    offset = block % len(ARM_NAMES)
    return ARM_NAMES[offset:] + ARM_NAMES[:offset]


def model_for(arm, seed, device='cuda'):
    torch.manual_seed(seed)
    model = M.make_model(arm).to(device)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS[arm]
    return model


def checkpoint_payload(arm, model, optimizer, seed, update):
    value = C.payload(model, optimizer, seed, update)
    value.update({'variant': arm, 'arm': arm, 'metadata': model.metadata()})
    return value


def paired(rows, arm, baseline='w24'):
    a = {r['block']: r for r in rows if r['arm'] == baseline}
    b = {r['block']: r for r in rows if r['arm'] == arm}
    assert set(a) == set(b) == set(range(16))
    before = [bool(a[k]['phenotype_pass']) for k in sorted(a)]
    after = [bool(b[k]['phenotype_pass']) for k in sorted(b)]
    wins = sum(not x and y for x, y in zip(before, after))
    losses = sum(x and not y for x, y in zip(before, after))
    p = exact_two_sided_binomial_p(wins, losses)
    return {'baseline': baseline, 'arm': arm, 'blocks': 16,
            'baseline_successes': sum(before), 'arm_successes': sum(after),
            'wins': wins, 'losses': losses, 'net_gain': wins-losses,
            'delta': (wins-losses)/16, 'exact_two_sided_p': p,
            'baseline_wilson95': wilson_interval(sum(before), 16),
            'arm_wilson95': wilson_interval(sum(after), 16),
            'qualification_threshold_met': wins-losses >= 4 and p <= .05,
            'per_block': [{'block': k, 'baseline_pass': before[k], 'arm_pass': after[k]}
                          for k in range(16)]}


def holm(values):
    result = {}
    maximum = 0.
    for rank, (name, p) in enumerate(sorted(values.items(), key=lambda row: (row[1], row[0]))):
        maximum = max(maximum, min(1., (len(values)-rank)*p))
        result[name] = maximum
    return result


def stats_check():
    assert exact_two_sided_binomial_p(4, 0) == .125
    assert exact_two_sided_binomial_p(6, 0) == .03125
    assert exact_two_sided_binomial_p(0, 0) == 1.
    assert holm({'a': .01, 'b': .04, 'c': .03}) == {'a': .03, 'c': .06, 'b': .06}
    rows = [{'block': k, 'arm': arm, 'phenotype_pass': arm == 'w8' and k < 6}
            for k in range(16) for arm in ('w24', 'w8')]
    assert paired(rows, 'w8')['qualification_threshold_met']
    return {'status': 'PASS', 'four_wins_p': .125, 'six_wins_p': .03125,
            'holm_monotonicity': True, 'independent_units': 'training blocks'}


def check(out):
    assert not out.exists(), 'Check output already exists'
    C.setup_backend()
    source = source_hashes()
    tests = local_module('_native_latent_width_tests', 'test_cells.py')
    cpu = tests.check()
    data = C.region_bank(8, 2, 99408, 'cuda')
    cuda = {}
    for arm in ARM_NAMES:
        model = model_for(arm, 91999)
        optimizer = C.optimizer_for(model)
        optimizer.zero_grad(set_to_none=True)
        before = C.tensor_hash(model.state_dict())
        torch.cuda.synchronize()
        started = time.monotonic()
        loss, state, cadence = C.backward_trajectory(model, data, 8)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert cadence == {'backward_calls': 8, 'interior_detach_boundaries': 7,
                           'forward_steps': 64, 'loss_count': 8}
        assert bool(torch.isfinite(norm)) and all(bool(torch.isfinite(v).all()) for v in state)
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        optimizer.step()
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        assert before != C.tensor_hash(model.state_dict())
        assert all(int(s['step'].item()) == 1 for s in optimizer.state.values())
        saved = checkpoint_payload(arm, model, optimizer, 91999, 1)
        assert saved['variant'] == saved['arm'] == arm
        assert saved['metadata']['parameter_count'] == PARAMETERS[arm]
        torch.cuda.synchronize()
        cuda[arm] = {'status': 'PASS', 'parameters': PARAMETERS[arm],
                     'loss': float(loss), 'seconds_small_update': time.monotonic()-started,
                     'cadence': cadence, 'state_shapes': [list(v.shape) for v in state]}
        del model, optimizer, state, saved
    # Exercise both separately initialized paired trajectories and the unmodified
    # trace interface with a phase-bearing state; no formal efficacy measurement.
    audit = C._load_audit()
    previous = audit.budget
    audit.budget = lambda: None
    try:
        with torch.no_grad():
            trace, records = audit.trace(model_for('w2_alternating', 91999).eval(), data, 8)
        assert trace['correct'].shape == (257, 2, 8, 8)
        assert set(('64', '128', '256')).issubset(records)
    finally:
        audit.budget = previous
    assert source_hashes() == source
    out.parent.mkdir(parents=True, exist_ok=True)
    C.write(out, {'status': 'PASS', 'protocol': PROTOCOL, 'checked_utc': C.now(),
                  'source_sha256': source, 'cpu': cpu, 'statistics': stats_check(),
                  'cuda': cuda, 'phase_trace_interface': 'PASS',
                  'runtime_limit_enforced': False})
    print(json.dumps({'status': 'PASS', 'qualification': out.relative_to(ROOT).as_posix()}), flush=True)


@torch.no_grad()
def systems(model, data):
    result = {}
    model.eval()
    for size in (32, 64):
        x = data[size]['x'][:8].cuda()
        for _ in range(2):
            state = model.initial(x)
            for _ in range(64):
                state = model.step(state, x)
            logits = model.logits(state)
        del state, logits
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        measures = []
        for _ in range(5):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            state = model.initial(x)
            for _ in range(64):
                state = model.step(state, x)
            logits = model.logits(state)
            stop.record()
            torch.cuda.synchronize()
            measures.append(start.elapsed_time(stop))
        learned_bytes = sum(v.numel()*v.element_size() for v in state[:2])
        result[str(size)] = {'batch': 8, 'forward_steps': 64, 'warmups': 2, 'repeats': 5,
                             'forward64_ms': measures, 'median_forward64_ms': statistics.median(measures),
                             'median_step_ms': statistics.median(measures)/64,
                             'learned_persistent_bytes': learned_bytes,
                             'clock_global_bits': 1 if len(state) == 3 else 0,
                             'clock_tensor_storage_bytes': state[2].numel()*state[2].element_size() if len(state) == 3 else 0,
                             'incremental_peak_cuda_bytes': torch.cuda.max_memory_allocated()-base}
        del state, logits, x
    return result


def aggregate(rows):
    widths = ('w8', 'w16', 'w4', 'w2_alternating')
    contrasts = {arm: paired(rows, arm) for arm in widths}
    adjusted = holm({arm: row['exact_two_sided_p'] for arm, row in contrasts.items()})
    for arm, row in contrasts.items():
        row['width_family_holm_p'] = adjusted[arm]
        row['transport_confounded'] = arm == 'w2_alternating'
    primary = contrasts['w8']
    return {'protocol': PROTOCOL, 'status': 'COMPLETE', 'expected_arms': 96,
            'completed_arms': len(rows), 'primary': primary,
            'primary_verdict': 'W8_RELIABILITY_QUALIFIED' if primary['qualification_threshold_met'] else 'NO_W8_RELIABILITY_QUALIFICATION',
            'secondary_width_contrasts': contrasts,
            'capacity_diagnostic': paired(rows, 'w8', 'w24_capacity'),
            'counts': {arm: {'successes': sum(r['phenotype_pass'] for r in rows if r['arm'] == arm),
                             'trials': 16, 'parameters': PARAMETERS[arm]}
                       for arm in ARM_NAMES},
            'claim_boundary': 'Frozen single-task native width recipe; W2 changes transport; no minimal-dimensionality or universal BPTT claim.'}


def report(out, summary, rows):
    lines = ['# Native execution-state width qualification', '',
             'Execution: ' + summary['status'], '',
             'Primary verdict: ' + summary.get('primary_verdict', 'PENDING'), '',
             '| Arm | Full successes | Parameters |', '|---|---:|---:|']
    for arm in ARM_NAMES:
        done = [r for r in rows if r['arm'] == arm]
        lines.append(f"| {arm} | {sum(r['phenotype_pass'] for r in done)}/{len(done)} | {PARAMETERS[arm]} |")
    if 'primary' in summary:
        p = summary['primary']
        lines += ['', f"W8 vs W24: wins {p['wins']}, losses {p['losses']}; net {p['net_gain']}/16; exact two-sided p={p['exact_two_sided_p']:.6g}."]
    lines += ['', 'Read summary.json first, then perarm.json and metrics.csv. Per-arm evaluation/*_summary.json',
              'contains every unchanged Full component. Curves, Boolean NPZ traces and frontier CSVs are secondary.',
              'W2 alternates axes and is not a width-only contrast. Capacity control changes F/Q widths.',
              'Scientific qualification failure is separate from complete execution.']
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fields = ('block', 'initialization_seed', 'schedule_seed', 'arm', 'full_pass', 'size', 'horizon',
              'coverage', 'strict_mean', 'strict_pooled', 'retention64_to256', 'ever_regressed_fraction', 'frontier_effect')
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for size, value in row['evaluation_compact']['sizes'].items():
                for t in ('64', '128', '256'):
                    writer.writerow({'block': row['block'], 'initialization_seed': row['initialization_seed'],
                                     'schedule_seed': row['schedule_seed'], 'arm': row['arm'],
                                     'full_pass': row['phenotype_pass'], 'size': size, 'horizon': t,
                                     'coverage': value['coverage'][t], 'strict_mean': value['strict_mean'][t],
                                     'strict_pooled': value['strict_pooled'][t],
                                     **{k: value[k] for k in ('retention64_to256', 'ever_regressed_fraction', 'frontier_effect')}})


def run(out, qualification):
    assert not out.exists(), 'Run output already exists'
    q = read(qualification)
    assert q['status'] == 'PASS' and q['protocol'] == PROTOCOL
    hashes = source_hashes()
    assert q['source_sha256'] == hashes, 'Check/source binding changed'
    C.setup_backend()
    out.mkdir(parents=True)
    started = time.monotonic()
    rows = []
    plans = {}
    for block, (seed, schedule) in enumerate(zip(INIT_SEEDS, SCHEDULE_SEEDS)):
        plans[str(block)] = {'initialization_seed': seed, 'schedule_seed': schedule,
                             'arm_order': list(arm_order(block)),
                             'batch_indices': np.random.default_rng(schedule).integers(0, 512, (300, 8)).tolist()}
    C.write(out / 'plans.json', plans)
    plan_hash = C.sha(out / 'plans.json')
    train = C.region_bank(32, 512, 10002, 'cuda')
    evaluation = {size: C.region_bank(size, 32, seed) for size, seed in EVAL_SEEDS.items()}
    data_hashes = {'train': C.tensor_hash(train), **{f'evaluation{size}': C.tensor_hash(data) for size, data in evaluation.items()}}
    C.write(out / 'config.json', {'protocol': PROTOCOL, 'arms': {a: model_for(a, 91999, 'cpu').metadata() for a in ARM_NAMES},
                                'initialization_seeds': INIT_SEEDS, 'schedule_seeds': SCHEDULE_SEEDS,
                                'train_bank_seed': 10002, 'train_maps': 512, 'eval_bank_seeds': EVAL_SEEDS,
                                'eval_maps_each_size': 32, 'updates': 300, 'batch': 8, 'credit_horizon': 8,
                                'training_forward_horizon': 64, 'eval_horizons': [64, 128, 256],
                                'optimizer': C.OPTIMIZER, 'gradient_clip': 1., 'primary': 'w8 vs w24',
                                'runtime_limit_enforced': False})
    manifest = {'protocol': PROTOCOL, 'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'),
                'started_utc': C.now(), 'command': [sys.executable, *sys.argv], 'gpu': torch.cuda.get_device_name(),
                'torch': str(torch.__version__), 'numpy': str(np.__version__),
                'source_sha256': hashes, 'schedule_plan_sha256': plan_hash, 'data_sha256': data_hashes,
                'qualification_sha256': C.sha(qualification), 'expected_arms': 96, 'runtime_limit_enforced': False,
                'maximum_seconds': None, 'watchdog_enabled': False, 'continuous_monitoring': False,
                'git_review_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'backend': {'cudnn_benchmark': False, 'cudnn_deterministic': False, 'cudnn_tf32': True, 'matmul_tf32': False}}
    C.write(out / 'manifest.json', manifest)
    for name in hashes:
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)

    def progress(value):
        C.write(out / 'status.json', {'status': 'RUNNING', 'pid': os.getpid(),
                                    'elapsed_seconds': time.monotonic()-started, 'updated_utc': C.now(),
                                    'expected_arms': 96, 'completed_arms': len(rows), **value})
        print(json.dumps(value), flush=True)

    result = None
    try:
        for block, plan in plans.items():
            for arm in plan['arm_order']:
                block_id = int(block)
                folder = out / f"block{block_id:02d}" / arm
                folder.mkdir(parents=True)
                model = model_for(arm, plan['initialization_seed'])
                optimizer = C.optimizer_for(model)
                initial_hash = C.tensor_hash(model.state_dict())
                torch.save(checkpoint_payload(arm, model, optimizer, plan['initialization_seed'], 0), folder / 'initial.pt')
                progress({'block': block_id, 'arm': arm, 'phase': 'starting', 'completed_updates': 0})
                curve = C.train_updates(model, optimizer, train, plan['batch_indices'], progress=lambda event: progress({'block': block_id, 'arm': arm, **event}))
                assert len(curve) == 300 and curve[-1]['update'] == 300
                assert all(int(s['step'].item()) == 300 for s in optimizer.state.values())
                C.write(folder / 'training_curve.json', curve)
                checkpoint = folder / 'final_u300.pt'
                torch.save(checkpoint_payload(arm, model, optimizer, plan['initialization_seed'], 300), checkpoint)
                final_hash = C.tensor_hash(model.state_dict())
                progress({'block': block_id, 'arm': arm, 'phase': 'evaluation', 'completed_updates': 300})
                measured = C.evaluate(model, evaluation, folder / 'evaluation', arm, lambda: None)
                timing = systems(model, evaluation)
                assert C.tensor_hash(model.state_dict()) == final_hash, 'Evaluation mutated parameters'
                C.write(folder / 'systems.json', timing)
                row = {'block': block_id, 'initialization_seed': plan['initialization_seed'],
                       'schedule_seed': plan['schedule_seed'], 'arm': arm, 'metadata': model.metadata(),
                       'initial_parameter_sha256': initial_hash, 'final_parameter_sha256': final_hash,
                       'checkpoint_sha256': C.sha(checkpoint), 'checkpoint': checkpoint.relative_to(out).as_posix(),
                       'training_curve': (folder / 'training_curve.json').relative_to(out).as_posix(),
                       'training_seconds': sum(r['seconds'] for r in curve), 'systems': timing,
                       'evaluation_summary': (folder / 'evaluation' / f'{arm}_summary.json').relative_to(out).as_posix(),
                       'phenotype_pass': bool(measured['phenotype_gate']['pass']), 'evaluation_compact': C.compact(measured)}
                rows.append(row)
                C.write(out / 'perarm.json', rows)
                report(out, {'status': 'RUNNING'}, rows)
                progress({'block': block_id, 'arm': arm, 'phase': 'arm_complete', 'completed_updates': 300,
                          'phenotype_pass': row['phenotype_pass']})
                del model, optimizer, measured, curve
                gc.collect()
                torch.cuda.empty_cache()
        assert len(rows) == 96 and source_hashes() == hashes
        assert C.sha(out / 'plans.json') == plan_hash
        assert C.tensor_hash(train) == data_hashes['train']
        assert all(C.tensor_hash(data) == data_hashes[f'evaluation{size}'] for size, data in evaluation.items())
        result = aggregate(rows)
    except Exception as error:
        C.write(out / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        result = {'protocol': PROTOCOL, 'status': 'ERROR', 'completed_arms': len(rows), 'expected_arms': 96,
                  'error': repr(error), 'scientific_verdict': 'INCOMPLETE'}
    result.update({'finished_utc': C.now(), 'elapsed_seconds': time.monotonic()-started,
                   'runtime_limit_enforced': False})
    C.write(out / 'summary.json', result)
    C.write(out / 'status.json', {**result, 'pid': os.getpid()})
    report(out, result, rows)
    print(json.dumps(result), flush=True)
    if result['status'] == 'ERROR':
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
        path = (ROOT / args.qualification).resolve()
        assert path.is_relative_to(ROOT / 'analyses')
        run(out, path)


if __name__ == '__main__':
    main()
