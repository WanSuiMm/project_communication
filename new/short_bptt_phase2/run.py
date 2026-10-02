"""Additive K8/K16/K64 initialization replication; frozen Phase-II protocol."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import threading
import time
import traceback
import numpy as np
import torch
from training import ROOT, backward_trajectory, make_cell
from run_revision import now, sha, write, tensor_hash, score
from tasks import bank, subset

SEEDS = (2, 3, 4, 5)
KS = (8, 16, 64)
UPDATES = 300
ORDERS = ((8, 16, 64), (16, 64, 8), (64, 8, 16), (64, 16, 8))
DEADLINE = float('inf')


def budget():
    if time.monotonic() > DEADLINE:
        raise TimeoutError('Whole-run time cap')


def sources():
    base = json.loads((ROOT / 'BPTT_PUBLICATION_MANIFEST.json').read_text())
    prior = base['source_sha256']
    for name, expected in prior.items():
        assert sha(ROOT / name) == expected, name
    added = ['new/short_bptt_phase2/' + name for name in ('PROTOCOL.md', 'training.py', 'check.py', 'run.py')]
    added += ['tools/launch_bptt_phase2.ps1']
    return {**prior, **{name: sha(ROOT / name) for name in added}}


def paired(logits, flipped, data, selected):
    good = ((logits >= 0) == (data['y'] >= .5)) & ((flipped >= 0) == (data['y_flip'] >= .5))
    selected = selected.bool()
    counts = selected.sum((1, 2, 3))
    hits = (good & selected).sum((1, 2, 3))
    eligible = counts > 0
    values = hits.double() / counts.clamp_min(1)
    n = int(counts.sum())
    return {'mean': float(values[eligible].mean()) if bool(eligible.any()) else None,
            'pooled_accuracy': int(hits.sum()) / n if n else None,
            'pooled_pixels': n, 'pooled_correct': int(hits.sum()),
            'eligible_maps': int(eligible.sum()), 'per_map_pixels': counts.cpu().tolist(),
            'per_map_correct': hits.cpu().tolist(),
            'per_map': [v if e else None for v, e in zip(values.cpu().tolist(), eligible.cpu().tolist())]}


def selections(data, k, t):
    d, changed = data['distance'], data['changed'].bool()
    ranges = {'0_8': (d >= 0) & (d < 8), '8_16': (d >= 8) & (d < 16),
              'equal_16': d == 16, 'strict_16_32': (d > 16) & (d < 32),
              '32_64': (d >= 32) & (d < 64), '64_128': (d >= 64) & (d < 128),
              '128_inf': d >= 128, 'far_gt16': d > 16, 'far_gt32': d > 32,
              'r_le1': (d >= 0) & (d <= 2*k), 'r_1_2': (d > 2*k) & (d <= 4*k),
              'r_gt2': d > 4*k, 'outside_forward_lightcone': d > 2*t}
    return {name: changed & value for name, value in ranges.items()}


def coverage(data):
    selected = selections(data, 8, 64)['strict_16_32']
    counts = selected.sum((1, 2, 3))
    return {'eligible_maps': int((counts > 0).sum()), 'pixels': int(counts.sum()),
            'per_map_pixels': counts.cpu().tolist()}


@torch.no_grad()
def evaluate(model, data, k, preflight):
    a, b = model.initial(data['x']), model.initial(data['x_flip'])
    times = (64,) if preflight else (64, 128, 256)
    result = {}
    for t in range(1, max(times) + 1):
        if t % 8 == 1:
            budget()
        a, b = model.step(a, data['x']), model.step(b, data['x_flip'])
        if t not in times:
            continue
        assert all(bool(torch.isfinite(v).all()) for v in (*a, *b)), 'Nonfinite state'
        logits, flipped = model.logits(a), model.logits(b)
        assert bool(torch.isfinite(logits).all()) and bool(torch.isfinite(flipped).all()), 'Nonfinite logits'
        masks = selections(data, k, t)
        delta = float(((logits-flipped).abs() * masks['outside_forward_lightcone']).max())
        assert delta <= 1e-6, 'Light-cone violation'
        result[str(t)] = {'original': score(logits, data['y'], data['mask']),
                          'flipped': score(flipped, data['y_flip'], data['mask']),
                          'paired': paired(logits, flipped, data, data['changed']),
                          'bands': {name: paired(logits, flipped, data, mask) for name, mask in masks.items()},
                          'outside_forward_lightcone_max_logit_difference': delta}
        for branch in ('original', 'flipped'):
            assert all(np.isfinite(result[str(t)][branch][key]) for key in
                       ('balanced_accuracy', 'bce', 'accuracy')), 'Nonfinite evaluation metric'
    return result


def train_one(out, seed, k, rows, train, tests, preflight):
    budget()
    torch.manual_seed(seed)
    model = make_cell('ws_additive').cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    name = f'additive_K{k}_seed{seed}'
    record = {'architecture': 'ws_additive', 'seed': seed, 'gradient_horizon': k,
              'initial_parameter_sha256': tensor_hash(model.state_dict()),
              'train_data_sha256': tensor_hash(train), 'schedule_sha256': sha(out / 'schedule.json'),
              'parameter_count': sum(p.numel() for p in model.parameters()), 'status': 'TRAINING',
              'completed_updates': 0, 'training_curve': [], 'update_seconds': [], 'evaluation': {}}
    assert record['parameter_count'] == 5033
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    record['allocated_before_training_bytes'] = torch.cuda.memory_allocated()
    started = time.monotonic()
    clipped = 0
    try:
        for iteration, indices in enumerate(rows, 1):
            budget()
            batch = subset(train, indices)
            torch.cuda.synchronize()
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss, state, trace = backward_trajectory(model, batch, k, budget=budget)
            assert all(bool(torch.isfinite(v).all()) for v in state), 'Nonfinite training state'
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert bool(torch.isfinite(norm)), 'Nonfinite gradient'
            clipped += int(norm > 1.)
            optimizer.step()
            torch.cuda.synchronize()
            record['update_seconds'].append(time.monotonic()-tick)
            record['completed_updates'] = iteration
            if iteration == 1 or iteration % 100 == 0 or iteration == len(rows) or preflight:
                log = {'update': iteration, 'mean_trajectory_loss': float(loss),
                       'gradient_norm_before_clip': float(norm), 'update_seconds': record['update_seconds'][-1], **trace}
                record['training_curve'].append(log)
                write(out / f'{name}.json', record)
                write(out / 'status.json', {'status': 'RUNNING', 'phase': 'training', 'arm': name,
                                           'pid': os.getpid(), 'completed_updates': iteration, 'updated_utc': now()})
                print(json.dumps({'arm': name, **log}), flush=True)
        record['training_seconds'] = time.monotonic()-started
        record['training_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['training_peak_increment_bytes'] = record['training_peak_allocated_bytes']-record['allocated_before_training_bytes']
        record['gradient_clip_fraction'] = clipped/len(rows)
        record['final_parameter_sha256'] = tensor_hash(model.state_dict())
        torch.save({'architecture': 'ws_additive', 'gradient_horizon': k, 'seed': seed,
                    'completed_updates': len(rows), 'state_dict': {n: v.detach().cpu() for n, v in model.state_dict().items()}},
                   out / f'{name}.pt')
        record['status'] = 'TRAINED'
        write(out / f'{name}.json', record)
        model.eval()
        for size, data in tests.items():
            write(out / 'status.json', {'status': 'RUNNING', 'phase': 'evaluation', 'arm': name,
                                       'size': size, 'pid': os.getpid(), 'updated_utc': now()})
            record['evaluation'][str(size)] = evaluate(model, data, k, preflight)
        record['status'] = 'COMPLETE'
    except Exception as error:
        record['status'] = 'TIME_BUDGET' if isinstance(error, TimeoutError) else 'ERROR'
        record['error'], record['traceback'] = repr(error), traceback.format_exc()
        if not (out / f'{name}.pt').exists():
            torch.save({'partial': True, 'completed_updates': record['completed_updates'],
                        'state_dict': {n: v.detach().cpu() for n, v in model.state_dict().items()}}, out / f'{name}.pt')
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        write(out / f'{name}.json', record)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return record


def endpoint(evaluation):
    row = evaluation['32']['64']
    p = row['bands']['strict_16_32']
    return {'ba_original': row['original']['balanced_accuracy'], 'ba_flipped': row['flipped']['balanced_accuracy'],
            'primary_mean': p['mean'], 'primary_pooled': p['pooled_accuracy']}


def reach_predicate(row):
    return (row['ba_original'] >= .85 and row['ba_flipped'] >= .85
            and row['primary_mean'] is not None and row['primary_mean'] >= .80
            and row['primary_pooled'] is not None and row['primary_pooled'] >= .80)


def aggregate(out, results, preflight, cover):
    expected = 3 if preflight else 12
    updates = 3 if preflight else UPDATES
    complete = len(results) == expected and all(r['status'] == 'COMPLETE' and r['completed_updates'] == updates for r in results)
    # All runs use one data bank/schedule, and initialization matches across K.
    assert len({r['train_data_sha256'] for r in results}) <= 1
    assert len({r['schedule_sha256'] for r in results}) <= 1
    for seed in SEEDS:
        assert len({r['initial_parameter_sha256'] for r in results if r['seed'] == seed}) <= 1
    doc = {'complete': complete, 'coverage': cover, 'identities_verified': True,
           'statuses': [{key: r[key] for key in ('seed', 'gradient_horizon', 'status', 'completed_updates')} for r in results]}
    if preflight:
        cost = max(float(np.median(r['update_seconds'][1:])) for r in results) if complete else None
        estimate = 12*UPDATES*cost*1.20+180 if complete else None
        memory_ok = complete and max(r['training_peak_allocated_bytes'] for r in results) < 3*1024**3
        passed = complete and estimate <= 1800 and memory_ok and cover['eligible_maps'] >= 16 and cover['pixels'] >= 500
        doc.update({'decision': 'PREFLIGHT_PASSED' if passed else 'PREFLIGHT_FAILED',
                    'worst_update_seconds': cost, 'estimated_formal_seconds': estimate,
                    'memory_gate': memory_ok, 'updates_per_arm': UPDATES})
    else:
        rows = []
        for r in results:
            if r['status'] != 'COMPLETE':
                continue
            row = endpoint(r['evaluation'])
            hold = True
            for t in ('128', '256'):
                e = r['evaluation']['32'][t]
                p = e['bands']['strict_16_32']
                hold = hold and e['original']['balanced_accuracy'] >= row['ba_original']-.03
                hold = hold and e['flipped']['balanced_accuracy'] >= row['ba_flipped']-.03
                hold = hold and p['mean'] >= row['primary_mean']-.05 and p['pooled_accuracy'] >= row['primary_pooled']-.05
            rows.append({'seed': r['seed'], 'K': r['gradient_horizon'], **row,
                         'reach': reach_predicate(row), 'hold': bool(hold)})
        counts = {str(k): {'completed': sum(r['K'] == k for r in rows),
                          'reach': sum(r['K'] == k and r['reach'] for r in rows),
                          'reach_and_hold': sum(r['K'] == k and r['reach'] and r['hold'] for r in rows)} for k in KS}
        effects = []
        for seed in SEEDS:
            by_k = {r['K']: r for r in rows if r['seed'] == seed}
            for k in (8, 16):
                if k in by_k and 64 in by_k:
                    effects.append({'seed': seed, 'K': k, 'full_qualified': by_k[64]['reach'],
                                    **{key+'_effect_pp': 100*(by_k[k][key]-by_k[64][key]) for key in
                                       ('ba_original', 'ba_flipped', 'primary_mean', 'primary_pooled')}})
        def replication(key):
            return 'INCOMPLETE' if not complete else ('REPLICATED' if counts['8'][key] >= 3 else 'NOT_REPLICATED')
        doc.update({'execution_status': 'COMPLETE' if complete else 'INCOMPLETE', 'rows': rows, 'counts': counts,
                    'paired_effects': effects, 'K8_narrow_reach': replication('reach'),
                    'K8_narrow_sustained': replication('reach_and_hold'),
                    'comparison_status': 'INCOMPLETE' if not complete else (
                        'FULL_CONTROL_QUALIFIED' if counts['64']['reach'] >= 3 else 'BASELINE_UNQUALIFIED')})
        lines = ['# Phase-II initialization replication', '',
                 f"Execution: {doc['execution_status']}; K8 narrow reach: {doc['K8_narrow_reach']}; sustained: {doc['K8_narrow_sustained']}.",
                 f"Comparison: {doc['comparison_status']}. Fixed data/schedule; four initialization seeds.", '',
                 'Primary size32/T64, STRICT 16<d<32. This narrower endpoint was selected from Phase I and tested on new maps.',
                 'A narrow pass does not amend the old failed gate, prove robust farther propagation, or establish an optimal K.', '',
                 '| Seed | K | Original BA % | Flipped BA % | Paired mean % | Paired pooled % | Reach | Hold |',
                 '|---|---:|---:|---:|---:|---:|---|---|']
        for row in sorted(rows, key=lambda r: (r['seed'], r['K'])):
            lines.append(f"| {row['seed']} | {row['K']} | {100*row['ba_original']:.2f} | {100*row['ba_flipped']:.2f} | {100*row['primary_mean']:.2f} | {100*row['primary_pooled']:.2f} | {row['reach']} | {row['hold']} |")
        lines += ['', 'Per-arm JSON retains the old d>16 endpoint, farther/normalized bands, size64, all rollout horizons,',
                  'integer correct counts, denominators, timings, clipping and memory. Do not select only successful seeds.',
                  'No state pool, optimizer-crossing persistence, RelationFirst validation or general credit-assignment claim.']
        (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(out / 'aggregate.json', doc)
    return doc


def main():
    global DEADLINE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--preflight-dir')
    args = p.parse_args()
    source_hashes = sources()
    pf_binding = None
    if not args.preflight:
        if not args.preflight_dir:
            p.error('Formal run requires passed --preflight-dir')
        pf = Path(args.preflight_dir).resolve()
        assert json.loads((pf / 'status.json').read_text())['status'] == 'PREFLIGHT_PASSED'
        assert json.loads((pf / 'manifest.json').read_text())['source_sha256'] == source_hashes
        summary = json.loads((pf / 'aggregate.json').read_text())
        assert summary['decision'] == 'PREFLIGHT_PASSED' and summary['updates_per_arm'] == UPDATES
        pf_binding = {name: sha(pf / name) for name in ('status.json', 'manifest.json', 'aggregate.json')}
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    DEADLINE = start + (180 if args.preflight else 1800)
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    tests_cpu = {n: bank(n, 32, 40000+n) for n in (32, 64)}
    cover = coverage(tests_cpu[32])
    assert cover['eligible_maps'] >= 16 and cover['pixels'] >= 500, 'Frozen evaluation coverage failed'
    tests = ({32: subset(tests_cpu[32], [0, 1])} if args.preflight else tests_cpu)
    tests = {n: {key: value.cuda() for key, value in data.items()} for n, data in tests.items()}
    train = bank(32, 512, 10002, 'cuda')
    rows = np.random.default_rng(20002).integers(0, 512, (3 if args.preflight else UPDATES, 8)).tolist()
    write(out / 'schedule.json', rows)
    manifest = {'protocol': 'short_bptt_phase2_v1', 'training': True, 'preflight': args.preflight,
                'started_utc': now(), 'pid': os.getpid(), 'initialization_seeds': [2] if args.preflight else list(SEEDS),
                'gradient_horizons': list(KS), 'updates_per_arm': 3 if args.preflight else UPDATES,
                'forward_steps': 64, 'loss_times': list(range(8, 65, 8)), 'train_data_seed': 10002,
                'schedule_seed': 20002, 'schedule_sha256': sha(out / 'schedule.json'),
                'train_data_sha256': tensor_hash(train), 'evaluation_map_count': 32,
                'evaluation_data_sha256': {str(n): tensor_hash(d) for n, d in tests_cpu.items()},
                'executed_eval_data_sha256': {str(n): tensor_hash(d) for n, d in tests.items()},
                'primary_coverage': cover, 'source_sha256': source_hashes, 'preflight_sha256': pf_binding,
                'optimizer': {'type': 'AdamW', 'lr': .001, 'weight_decay': .0001, 'clip_norm': 1., 'batch': 8},
                'backend': {'cudnn_benchmark': False, 'cudnn_deterministic': False, 'cudnn_tf32': True, 'matmul_tf32': False},
                'torch': torch.__version__, 'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(),
                'maximum_seconds': 180 if args.preflight else 1800}
    write(out / 'manifest.json', manifest)
    for name in source_hashes:
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    def timeout():
        write(out / 'watchdog_timeout.json', {'status': 'TIME_BUDGET', 'time': now()})
        os._exit(124)
    watchdog = threading.Timer(max(1., start + (240 if args.preflight else 1860) - time.monotonic()), timeout)
    watchdog.daemon = True
    watchdog.start()
    results = []
    status = 'ERROR'
    try:
        for i, seed in enumerate(manifest['initialization_seeds']):
            for k in ORDERS[i]:
                r = train_one(out, seed, k, rows, train, tests, args.preflight)
                results.append(r)
                aggregate(out, results, args.preflight, cover)
                if r['status'] != 'COMPLETE':
                    raise RuntimeError(f"Arm stopped: {r['status']}")
        summary = aggregate(out, results, args.preflight, cover)
        status = summary['decision'] if args.preflight else ('COMPLETE' if summary['complete'] else 'INCOMPLETE')
    except Exception as error:
        status = 'TIME_BUDGET' if isinstance(error, TimeoutError) or any(r['status'] == 'TIME_BUDGET' for r in results) else 'ERROR'
        write(out / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
    finally:
        watchdog.cancel()
    write(out / 'status.json', {'status': status, 'pid': os.getpid(), 'completed_arms': sum(r['status'] == 'COMPLETE' for r in results),
                               'finished_utc': now(), 'elapsed_seconds': time.monotonic()-start})
    print(json.dumps({'status': status, 'elapsed_seconds': time.monotonic()-start}), flush=True)
    if status != ('PREFLIGHT_PASSED' if args.preflight else 'COMPLETE'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
