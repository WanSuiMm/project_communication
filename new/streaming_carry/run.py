"""Matched baseline versus lossless-path streaming, K8 development screen."""
import argparse
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

import numpy as np
import torch
from stream_cells import ROOT, VARIANTS, make_variant

sys.path.insert(0, str(ROOT / 'new/short_bptt_phase2'))
spec = importlib.util.spec_from_file_location('phase2_runner', ROOT / 'new/short_bptt_phase2/run.py')
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)
from training import backward_trajectory
from run_revision import now, sha, write, tensor_hash
from tasks import bank, subset

SEEDS = (2, 3, 4, 5)
UPDATES = 300
CAP = 1500


def sources():
    manifest = ROOT / 'DIRECT_CARRY_PUBLICATION_MANIFEST.json'
    prior = json.loads(manifest.read_text())['source_sha256']
    for name, expected in prior.items():
        assert sha(ROOT / name) == expected, f'Frozen source drift: {name}'
    names = ['new/streaming_carry/' + name for name in
             ('stream_cells.py', 'run.py', 'check.py', 'PROTOCOL.md')]
    names += ['tools/launch_streaming_carry.ps1', 'DIRECT_CARRY_PUBLICATION_MANIFEST.json']
    names += ['evidence/short_bptt_phase2_init2345/manifest.json']
    names += [f'evidence/short_bptt_phase2_init2345/raw/additive_K8_seed{s}.json' for s in SEEDS]
    return {**prior, **{name: sha(ROOT / name) for name in names}}


def historical_record(seed):
    return json.loads((ROOT / f'evidence/short_bptt_phase2_init2345/raw/additive_K8_seed{seed}.json').read_text())


def train_one(out, seed, variant, rows, train, tests, preflight):
    p2.budget()
    torch.manual_seed(seed)
    model = make_variant(variant).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    name = f'{variant}_K8_seed{seed}'
    record = {'variant': variant, 'architecture': model.metadata(), 'seed': seed,
              'gradient_horizon': 8, 'initial_parameter_sha256': tensor_hash(model.state_dict()),
              'train_data_sha256': tensor_hash(train), 'schedule_sha256': sha(out / 'schedule.json'),
              'parameter_count': sum(p.numel() for p in model.parameters()), 'status': 'TRAINING',
              'completed_updates': 0, 'training_curve': [], 'update_seconds': [], 'evaluation': {}}
    assert record['parameter_count'] == 5033
    old = historical_record(seed)
    for key in ('initial_parameter_sha256', 'train_data_sha256'):
        assert record[key] == old[key], f'Historical identity mismatch: {key}'
    if not preflight:
        assert record['schedule_sha256'] == old['schedule_sha256']
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    record['allocated_before_training_bytes'] = torch.cuda.memory_allocated()
    started = time.monotonic()
    clipped = 0
    try:
        for iteration, indices in enumerate(rows, 1):
            p2.budget()
            batch = subset(train, indices)
            torch.cuda.synchronize()
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss, state, trace = backward_trajectory(model, batch, 8, budget=p2.budget)
            assert all(bool(torch.isfinite(v).all()) for v in state), 'Nonfinite training state'
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert bool(torch.isfinite(norm)), 'Nonfinite gradient'
            clipped += int(norm > 1.)
            optimizer.step()
            torch.cuda.synchronize()
            record['update_seconds'].append(time.monotonic() - tick)
            record['completed_updates'] = iteration
            if iteration == 1 or iteration % 100 == 0 or iteration == len(rows) or preflight:
                log = {'update': iteration, 'mean_trajectory_loss': float(loss),
                       'gradient_norm_before_clip': float(norm),
                       'update_seconds': record['update_seconds'][-1], **trace}
                record['training_curve'].append(log)
                write(out / f'{name}.json', record)
                write(out / 'status.json', {'status': 'RUNNING', 'phase': 'training', 'arm': name,
                      'pid': os.getpid(), 'completed_updates': iteration, 'updated_utc': now()})
                print(json.dumps({'arm': name, **log}), flush=True)
        record['training_seconds'] = time.monotonic() - started
        record['training_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['training_peak_increment_bytes'] = record['training_peak_allocated_bytes'] - record['allocated_before_training_bytes']
        record['gradient_clip_fraction'] = clipped / len(rows)
        record['final_parameter_sha256'] = tensor_hash(model.state_dict())
        torch.save({'variant': variant, 'architecture': model.metadata(), 'gradient_horizon': 8,
                    'seed': seed, 'completed_updates': len(rows),
                    'state_dict': {n: v.detach().cpu() for n, v in model.state_dict().items()}}, out / f'{name}.pt')
        record['status'] = 'TRAINED'
        write(out / f'{name}.json', record)
        model.eval()
        for size, data in tests.items():
            write(out / 'status.json', {'status': 'RUNNING', 'phase': 'evaluation', 'arm': name,
                  'size': size, 'pid': os.getpid(), 'updated_utc': now()})
            record['evaluation'][str(size)] = p2.evaluate(model, data, 8, preflight)
        p2.budget()
        record['status'] = 'COMPLETE'
    except Exception as error:
        record['status'] = 'TIME_BUDGET' if isinstance(error, TimeoutError) else 'ERROR'
        record['error'], record['traceback'] = repr(error), traceback.format_exc()
        if not (out / f'{name}.pt').exists():
            torch.save({'partial': True, 'variant': variant, 'architecture': model.metadata(),
                        'completed_updates': record['completed_updates'],
                        'state_dict': {n: v.detach().cpu() for n, v in model.state_dict().items()}}, out / f'{name}.pt')
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        write(out / f'{name}.json', record)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return record


def predicates(record):
    row = p2.endpoint(record['evaluation'])
    hold = True
    for t in ('128', '256'):
        e = record['evaluation']['32'][t]
        p = e['bands']['strict_16_32']
        hold = hold and e['original']['balanced_accuracy'] >= row['ba_original'] - .03
        hold = hold and e['flipped']['balanced_accuracy'] >= row['ba_flipped'] - .03
        hold = hold and p['mean'] is not None and p['pooled_accuracy'] is not None
        if hold:
            hold = p['mean'] >= row['primary_mean'] - .05 and p['pooled_accuracy'] >= row['primary_pooled'] - .05
    reach = p2.reach_predicate(row)
    return {'seed': record['seed'], 'variant': record['variant'], **row,
            'reach': reach, 'hold': bool(hold), 'reach_and_hold': bool(reach and hold)}


def decision(rows, complete):
    counts = {v: {'reach': sum(r['variant'] == v and r['reach'] for r in rows),
                  'reach_and_hold': sum(r['variant'] == v and r['reach_and_hold'] for r in rows)} for v in VARIANTS}
    positive_kept = {v: all(any(r['seed'] == s and r['variant'] == v and r['reach_and_hold']
                                for r in rows) for s in (2, 5)) for v in VARIANTS}
    if not complete:
        status = 'INCOMPLETE'
    elif not positive_kept['baseline']:
        status = 'BASELINE_REPRODUCTION_DRIFT'
    elif (counts['stream']['reach_and_hold'] >= 3 and positive_kept['stream']
          and counts['stream']['reach_and_hold'] > counts['baseline']['reach_and_hold']):
        status = 'DEVELOPMENT_GO'
    else:
        status = 'DEVELOPMENT_NO_GO'
    return {'decision': status, 'counts': counts, 'historical_positive_seeds_preserved': positive_kept}


def curve_rows(results):
    rows = []
    for r in results:
        for size, evaluation in r['evaluation'].items():
            for t, e in evaluation.items():
                for band, p in e['bands'].items():
                    rows.append({'seed': r['seed'], 'variant': r['variant'], 'size': int(size), 'T': int(t),
                                 'band': band, 'mean': p['mean'], 'pooled': p['pooled_accuracy'],
                                 'pixels': p['pooled_pixels'], 'correct': p['pooled_correct'],
                                 'eligible_maps': p['eligible_maps'],
                                 'ba_original': e['original']['balanced_accuracy'],
                                 'ba_flipped': e['flipped']['balanced_accuracy']})
    return rows


def aggregate(out, results, preflight, cover):
    expected = {(s, v) for s in ((2,) if preflight else SEEDS) for v in VARIANTS}
    keys = [(r['seed'], r['variant']) for r in results]
    assert len(keys) == len(set(keys)) and set(keys) <= expected
    updates = 3 if preflight else UPDATES
    complete = set(keys) == expected and all(r['status'] == 'COMPLETE' and r['completed_updates'] == updates for r in results)
    assert len({r['train_data_sha256'] for r in results}) <= 1
    assert len({r['schedule_sha256'] for r in results}) <= 1
    for seed in SEEDS:
        assert len({r['initial_parameter_sha256'] for r in results if r['seed'] == seed}) <= 1
    doc = {'complete': complete, 'coverage': cover, 'identities_verified': True,
           'statuses': [{key: r[key] for key in ('seed', 'variant', 'status', 'completed_updates')} for r in results]}
    if preflight:
        cost = max(float(np.median(r['update_seconds'][1:])) for r in results) if complete else None
        estimate = 8 * UPDATES * cost * 1.20 + 120 if complete else None
        memory_ok = complete and max(r['training_peak_allocated_bytes'] for r in results) < 3 * 1024**3
        passed = complete and estimate <= CAP and memory_ok and cover['eligible_maps'] >= 16 and cover['pixels'] >= 500
        doc.update({'decision': 'PREFLIGHT_PASSED' if passed else 'PREFLIGHT_FAILED',
                    'worst_update_seconds': cost, 'estimated_formal_seconds': estimate,
                    'memory_gate': memory_ok, 'updates_per_arm': UPDATES})
    else:
        rows = [predicates(r) for r in results if r['status'] == 'COMPLETE']
        doc.update({'rows': rows, **decision(rows, complete)})
        curves = curve_rows(results)
        lookup = {(r['seed'], r['variant'], r['size'], r['T'], r['band']): r for r in curves}
        effects = []
        for r in curves:
            if r['variant'] != 'stream':
                continue
            b = lookup.get((r['seed'], 'baseline', r['size'], r['T'], r['band']))
            if b is None:
                continue
            assert r['pixels'] == b['pixels'] and r['eligible_maps'] == b['eligible_maps']
            effects.append({**{k: r[k] for k in ('seed', 'size', 'T', 'band', 'pixels', 'eligible_maps')},
                            **{k + '_effect_pp': None if r[k] is None or b[k] is None else 100 * (r[k] - b[k])
                               for k in ('mean', 'pooled', 'ba_original', 'ba_flipped')}})
        doc['paired_effects'] = effects
        for filename, data in (('curves.csv', curves), ('paired_effects.csv', effects)):
            if data:
                with (out / filename).open('w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=list(data[0]))
                    w.writeheader()
                    w.writerows(data)
        lines = ['# Streaming Carry development screen', '', f"Status: {doc['decision']}. Complete: {complete}.",
                 'Seeds2-5 and these maps were already inspected. This is development, not confirmation.', '',
                 '| Seed | Variant | BA original % | BA flipped % | Primary mean % | Primary pooled % | Reach | Hold |',
                 '|---|---|---:|---:|---:|---:|---|---|']
        for r in sorted(rows, key=lambda r: (r['seed'], r['variant'])):
            lines.append(f"| {r['seed']} | {r['variant']} | {100*r['ba_original']:.2f} | {100*r['ba_flipped']:.2f} | {100*r['primary_mean']:.2f} | {100*r['primary_pooled']:.2f} | {r['reach']} | {r['hold']} |")
        lines += ['', f"Reach+hold counts: baseline={doc['counts']['baseline']['reach_and_hold']}/4; stream={doc['counts']['stream']['reach_and_hold']}/4.",
                  'Primary: size32/T64, strict16<d<32. Hold: T128 AND T256 relative to T64.',
                  'Read curves.csv and paired_effects.csv for every seed, size, horizon and distance band.',
                  'Empty bands are null in JSON. Farther gains cannot rescue a failed primary.',
                  'Raw arm JSON retains per-map counts, denominators, clipping, synchronized timing and memory.',
                  'Only the fixed transport is lossless; the full learned recurrence may erase or amplify messages.',
                  'No automatic confirmation, architecture rescue or monitoring is scheduled.']
        (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write(out / 'aggregate.json', doc)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--preflight-dir')
    args = parser.parse_args()
    hashes = sources()
    binding = None
    if not args.preflight:
        if not args.preflight_dir:
            parser.error('Formal run requires passed --preflight-dir')
        pf = Path(args.preflight_dir).resolve()
        assert json.loads((pf / 'status.json').read_text())['status'] == 'PREFLIGHT_PASSED'
        assert json.loads((pf / 'manifest.json').read_text())['source_sha256'] == hashes
        summary = json.loads((pf / 'aggregate.json').read_text())
        assert summary['decision'] == 'PREFLIGHT_PASSED' and summary['updates_per_arm'] == UPDATES
        binding = {name: sha(pf / name) for name in ('status.json', 'manifest.json', 'aggregate.json')}
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    p2.DEADLINE = start + (180 if args.preflight else CAP)
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    tests_cpu = {n: bank(n, 32, 40000+n) for n in (32, 64)}
    eval_hashes = {str(n): tensor_hash(d) for n, d in tests_cpu.items()}
    old_manifest = json.loads((ROOT / 'evidence/short_bptt_phase2_init2345/manifest.json').read_text())
    assert eval_hashes == old_manifest['evaluation_data_sha256'], 'Evaluation bank changed'
    cover = p2.coverage(tests_cpu[32])
    assert cover['eligible_maps'] >= 16 and cover['pixels'] >= 500
    tests = {32: subset(tests_cpu[32], [0, 1])} if args.preflight else tests_cpu
    tests = {n: {key: value.cuda() for key, value in data.items()} for n, data in tests.items()}
    train = bank(32, 512, 10002, 'cuda')
    rows = np.random.default_rng(20002).integers(0, 512, (3 if args.preflight else UPDATES, 8)).tolist()
    write(out / 'schedule.json', rows)
    manifest = {'protocol': 'streaming_carry_development_v1', 'training': True, 'preflight': args.preflight,
                'started_utc': now(), 'pid': os.getpid(), 'initialization_seeds': [2] if args.preflight else list(SEEDS),
                'variants': list(VARIANTS), 'carrier_lanes': ['N', 'E', 'S', 'W'],
                'payload_channels_per_lane': 6, 'boundary': 'blocked-link lane reversal; walls identity',
                'gradient_horizon': 8,
                'updates_per_arm': 3 if args.preflight else UPDATES, 'forward_steps': 64,
                'loss_times': list(range(8, 65, 8)), 'train_data_seed': 10002, 'schedule_seed': 20002,
                'schedule_sha256': sha(out / 'schedule.json'), 'train_data_sha256': tensor_hash(train),
                'evaluation_map_count': 32, 'evaluation_data_sha256': eval_hashes,
                'executed_eval_data_sha256': {str(n): tensor_hash(d) for n, d in tests.items()},
                'primary_coverage': cover, 'source_sha256': hashes, 'preflight_sha256': binding,
                'optimizer': {'type': 'AdamW', 'lr': .001, 'weight_decay': .0001, 'clip_norm': 1., 'batch': 8},
                'backend': {'cudnn_benchmark': False, 'cudnn_deterministic': False, 'cudnn_tf32': True, 'matmul_tf32': False},
                'torch': torch.__version__, 'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(),
                'maximum_seconds': 180 if args.preflight else CAP}
    write(out / 'manifest.json', manifest)
    for name in hashes:
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    def timeout():
        write(out / 'watchdog_timeout.json', {'status': 'TIME_BUDGET', 'time': now()})
        os._exit(124)
    watchdog = threading.Timer(max(1., p2.DEADLINE + 60 - time.monotonic()), timeout)
    watchdog.daemon = True
    watchdog.start()
    results = []
    status = 'ERROR'
    try:
        for i, seed in enumerate(manifest['initialization_seeds']):
            order = VARIANTS if i % 2 == 0 else tuple(reversed(VARIANTS))
            for variant in order:
                r = train_one(out, seed, variant, rows, train, tests, args.preflight)
                results.append(r)
                aggregate(out, results, args.preflight, cover)
                if r['status'] != 'COMPLETE':
                    raise RuntimeError(f"Arm stopped: {r['status']}")
        summary = aggregate(out, results, args.preflight, cover)
        p2.budget()
        status = summary['decision'] if args.preflight else ('COMPLETE' if summary['complete'] else 'INCOMPLETE')
    except Exception as error:
        status = 'TIME_BUDGET' if isinstance(error, TimeoutError) or any(r['status'] == 'TIME_BUDGET' for r in results) else 'ERROR'
        write(out / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        # A cap or reporting error after the last arm cannot leave a GO label.
        if (out / 'aggregate.json').exists():
            partial = json.loads((out / 'aggregate.json').read_text())
            partial.update({'complete': False, 'decision': 'PREFLIGHT_FAILED' if args.preflight else 'INCOMPLETE',
                            'execution_status': status, 'execution_error': repr(error)})
            write(out / 'aggregate.json', partial)
        if not args.preflight:
            (out / 'RESULTS.md').write_text(
                '# Streaming Carry: INCOMPLETE\n\nExecution failed or exceeded its time cap. '
                'Read status.json, error.json and partial raw records; no scientific pass.\n',
                encoding='utf-8')
    finally:
        watchdog.cancel()
    write(out / 'status.json', {'status': status, 'pid': os.getpid(), 'completed_arms': sum(r['status'] == 'COMPLETE' for r in results),
                               'finished_utc': now(), 'elapsed_seconds': time.monotonic() - start})
    print(json.dumps({'status': status, 'elapsed_seconds': time.monotonic() - start}), flush=True)
    if status != ('PREFLIGHT_PASSED' if args.preflight else 'COMPLETE'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
