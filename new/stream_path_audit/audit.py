"""Bounded zero-training Streaming seed4 pathway audit; see PROTOCOL.md."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch

from operators import CONDITIONS, ROOT, StreamingCell, hops_per_step, step
from run_revision import open_rms, score, sha, tensor_hash, write
from tasks import bank

# Import the frozen metric implementation without changing any prior source.
import importlib.util
_spec = importlib.util.spec_from_file_location('stream_audit_phase2', ROOT / 'new/short_bptt_phase2/run.py')
_phase2 = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(ROOT / 'new/short_bptt_phase2'))
_spec.loader.exec_module(_phase2)

TIMES = (8, 16, 32, 64, 128, 256)
DEADLINE = float('inf')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def budget():
    if time.monotonic() >= DEADLINE:
        raise TimeoutError('Frozen five-minute audit cap')


def compare_tree(saved, measured, path, comparisons):
    if isinstance(saved, dict):
        assert set(saved) == set(measured), ('Replay fields', path)
        for key in saved:
            compare_tree(saved[key], measured[key], path+'/'+key, comparisons)
    elif isinstance(saved, list):
        assert len(saved) == len(measured), ('Replay length', path)
        for i, (a, b) in enumerate(zip(saved, measured)):
            compare_tree(a, b, path+'/'+str(i), comparisons)
    elif saved is None:
        assert measured is None, ('Replay null', path)
    elif isinstance(saved, int):
        assert saved == measured, ('Replay integer', path, saved, measured)
        comparisons['integer_leaves'] += 1
    elif isinstance(saved, float):
        relative = 1e-5 if path.endswith('/bce') else 0
        assert math.isclose(saved, measured, abs_tol=1e-6, rel_tol=relative), ('Replay numeric', path, saved, measured)
        comparisons['float_leaves'] += 1
        comparisons['maximum_absolute_error'] = max(comparisons['maximum_absolute_error'], abs(saved-measured))
    else:
        assert saved == measured, ('Replay value', path)


@torch.no_grad()
def evaluate(model, data, condition):
    switches = CONDITIONS[condition]
    a, b = model.initial(data['x']), model.initial(data['x_flip'])
    result = {}
    for t in range(1, max(TIMES)+1):
        budget()
        a = step(model, a, data['x'], *switches)
        b = step(model, b, data['x_flip'], *switches)
        if t % 8 == 0:
            assert all(bool(torch.isfinite(v).all()) for v in (*a, *b)), 'Nonfinite state'
        if t not in TIMES:
            continue
        logits, flipped = model.logits(a), model.logits(b)
        masks = _phase2.selections(data, 8, t)
        legacy_delta = float(((logits-flipped).abs() * masks['outside_forward_lightcone']).max())
        radius = hops_per_step(*switches)*t
        outside = data['changed'].bool() & (data['distance'] > radius)
        actual_delta = float(((logits-flipped).abs()*outside).max())
        assert actual_delta <= 1e-6, ('Light-cone violation', condition, t, actual_delta)
        evaluation = {
            'original': score(logits, data['y'], data['mask']),
            'flipped': score(flipped, data['y_flip'], data['mask']),
            'paired': _phase2.paired(logits, flipped, data, data['changed']),
            'bands': {name: _phase2.paired(logits, flipped, data, mask) for name, mask in masks.items()},
            'outside_forward_lightcone_max_logit_difference': legacy_delta,
        }
        for branch in ('original', 'flipped'):
            assert all(math.isfinite(evaluation[branch][k]) for k in ('balanced_accuracy', 'bce', 'accuracy'))
        if condition == 'neither':
            nonlocal_pair = _phase2.paired(logits, flipped, data, data['changed'].bool() & (data['distance'] > 0))
            assert nonlocal_pair['pooled_correct'] == 0, 'Both-off may not solve remote counterfactual labels'
        diagnostics = {'actual_hop_radius_upper_bound': radius,
                       'actual_outside_lightcone_max_logit_difference': actual_delta,
                       'actual_outside_lightcone_pixels': int(outside.sum())}
        for branch, state in (('original', a), ('flipped', b)):
            diagnostics[branch] = {
                name: {'mean': float(v.mean()), 'per_map': v.cpu().tolist()}
                for name, value in (('W_rms', state[0]), ('Z_rms', state[1]))
                for v in (open_rms(value, data['mask']),)
            }
        for name, mask in (('changed', data['changed']), ('open', data['mask'])):
            value = open_rms(logits-flipped, mask)
            diagnostics[name+'_source_flip_logit_rms'] = {'mean': float(value.mean()), 'per_map': value.cpu().tolist()}
        result[str(t)] = {'evaluation': evaluation, 'diagnostics': diagnostics}
    return result


def curve_rows(records):
    rows = []
    for condition, sizes in records.items():
        for size, sequence in sizes.items():
            for t, measured in sequence.items():
                e = measured['evaluation']
                for band, p in {'all_changed': e['paired'], **e['bands']}.items():
                    rows.append({'condition': condition, 'size': int(size), 'T': int(t), 'band': band,
                                 'mean': p['mean'], 'pooled': p['pooled_accuracy'],
                                 'correct': p['pooled_correct'], 'pixels': p['pooled_pixels'],
                                 'eligible_maps': p['eligible_maps'],
                                 'ba_original': e['original']['balanced_accuracy'],
                                 'ba_flipped': e['flipped']['balanced_accuracy']})
    return rows


def contrasts(records):
    rows = []
    for size, sequence in records['full'].items():
        for t, measured in sequence.items():
            all_metrics = lambda r: {'all_changed': r['evaluation']['paired'], **r['evaluation']['bands']}
            lookup = {name: all_metrics(records[name][size][t]) for name in CONDITIONS}
            for band, base in all_metrics(measured).items():
                if base['mean'] is None:
                    continue
                for condition in ('no_transport', 'no_perception', 'neither'):
                    q = lookup[condition][band]
                    assert base['per_map_pixels'] == q['per_map_pixels']
                    rows.append({'condition': condition, 'size': int(size), 'T': int(t), 'band': band,
                                 'pooled_effect_pp': 100*(q['pooled_accuracy']-base['pooled_accuracy']),
                                 'mean_effect_pp': 100*(q['mean']-base['mean']),
                                 'per_map_effect_pp': [None if a is None else 100*(b-a)
                                                       for a, b in zip(base['per_map'], q['per_map'])]})
                rows.append({'condition': 'factorial_interaction', 'size': int(size), 'T': int(t), 'band': band,
                             'pooled_effect_pp': 100*sum(sign*lookup[name][band]['pooled_accuracy'] for name, sign in
                                 (('full', 1), ('no_transport', -1), ('no_perception', -1), ('neither', 1))),
                             'mean_effect_pp': 100*sum(sign*lookup[name][band]['mean'] for name, sign in
                                 (('full', 1), ('no_transport', -1), ('no_perception', -1), ('neither', 1)))})
    return rows


def summarize(records, replay, elapsed):
    labels = {}
    for condition in ('no_transport', 'no_perception', 'neither'):
        differences = [100*(records[condition]['32'][str(t)]['evaluation']['bands']['strict_16_32']['pooled_accuracy']
                          -records['full']['32'][str(t)]['evaluation']['bands']['strict_16_32']['pooled_accuracy'])
                       for t in (64, 128, 256)]
        label = ('sustained_knockout_loss' if differences[0] <= -20 and differences[2] <= -20
                 else 'primary_retention' if all(abs(v) <= 5 for v in differences) else 'mixed')
        labels[condition] = {'descriptive_label': label, 'T64_128_256_pooled_effect_pp': differences}
    return {'status': 'COMPLETE', 'training': False, 'model_seed': 4, 'model_replications': 1,
            'elapsed_seconds': elapsed, 'replay': replay, 'pathway_sensitivity': labels,
            'interpretation': 'Selected frozen checkpoint under distribution-changing knockouts; no training-cause or architecture-superiority claim.'}


def report(records, summary):
    lines = ['# Streaming seed4: fixed-weight operator audit', '',
             'Completed zero-training single-checkpoint diagnostic. Original replay passed.', '',
             '| Condition | T | Size32 primary mean / pooled % | Primary correct / pixels | BA original / flipped % | Size64 d>32 pooled % |',
             '|---|---:|---:|---:|---:|---:|']
    for name in CONDITIONS:
        for t in (64, 128, 256):
            e = records[name]['32'][str(t)]['evaluation']
            p = e['bands']['strict_16_32']
            f = records[name]['64'][str(t)]['evaluation']['bands']['far_gt32']
            lines.append(f"| {name} | {t} | {100*p['mean']:.2f} / {100*p['pooled_accuracy']:.2f} | {p['pooled_correct']}/{p['pooled_pixels']} | {100*e['original']['balanced_accuracy']:.2f} / {100*e['flipped']['balanced_accuracy']:.2f} | {100*f['pooled_accuracy']:.2f} |")
    lines += ['', '## Descriptive sensitivity', '']
    for name, row in summary['pathway_sensitivity'].items():
        values = ' / '.join(f'{v:+.2f}' for v in row['T64_128_256_pooled_effect_pp'])
        lines.append(f"- {name}: {row['descriptive_label']}; primary pooled effects at T64/T128/T256: {values} pp.")
    lines += ['', 'Paired correctness requires both source alternatives to be correct at the same changed-region pixel.',
              'All four conditions use identical weights, input banks and macro-step counts, but hop bounds per step are 2/2/1/0.',
              'Removing transport also replaces the incoming feature read by F. Removing perception zeros both Laplacian slots in both F and Q.',
              'Knockouts change learned state/input distributions. A loss is a dependency of this frozen solution under this intervention, not proof of training causality, unique semantics or inability to retrain.',
              'This is selected seed4, n=1. Prior multi-seed DEVELOPMENT_NO_GO and seed4 positive evidence are unchanged.',
              '', f"Runtime: {summary['elapsed_seconds']:.2f} seconds. Read summary.json, curves.csv and contrasts.json first; raw_conditions.json is secondary."]
    return '\n'.join(lines)+'\n'


def main():
    global DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    assert out.is_relative_to(ROOT / 'runs'), 'Audit outputs belong under project runs/'
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    DEADLINE = start+300
    historical_run = ROOT / 'runs/streaming_carry_20261002_init2345'
    checkpoint_path = historical_run / 'stream_K8_seed4.pt'
    checkpoint_before = sha(checkpoint_path)
    records = {}
    timer = threading.Timer(330, lambda: (write(out/'status.json', {'status': 'WATCHDOG_TIMEOUT', 'finished_utc': now()}), os._exit(124)))
    timer.daemon = True
    timer.start()
    try:
        torch.set_num_threads(2)
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        publication = read(ROOT / 'STREAMING_CARRY_PUBLICATION_MANIFEST.json')
        old_manifest = read(historical_run / 'manifest.json')
        old = read(historical_run / 'stream_K8_seed4.json')
        assert old['status'] == 'COMPLETE' and old['completed_updates'] == 300 and old['seed'] == 4
        assert sha(historical_run/'stream_K8_seed4.json') == sha(ROOT/'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json')
        assert checkpoint_before == publication['checkpoint_sha256'][checkpoint_path.name]
        source_hashes = publication['source_sha256'].copy()
        for name, expected in source_hashes.items():
            assert sha(ROOT/name) == sha(historical_run/'source'/name) == expected, ('Historical source drift', name)
        for name in ('PROTOCOL.md', 'operators.py', 'check.py', 'audit.py'):
            relative = 'new/stream_path_audit/'+name
            source_hashes[relative] = sha(ROOT/relative)
        data_cpu = {size: bank(size, 32, 40000+size) for size in (32, 64)}
        data_hashes = {str(size): tensor_hash(data) for size, data in data_cpu.items()}
        assert data_hashes == old_manifest['evaluation_data_sha256'] == old_manifest['executed_eval_data_sha256']
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert checkpoint['variant'] == 'stream' and checkpoint['seed'] == 4 and checkpoint['completed_updates'] == 300
        model = StreamingCell().eval()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        before = tensor_hash(model.state_dict())
        assert before == old['final_parameter_sha256']
        model = model.cuda()
        manifest = {'protocol': 'stream_seed4_operator_audit_v1', 'training': False,
                    'started_utc': now(), 'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'),
                    'gpu': torch.cuda.get_device_name(0), 'device': 'cuda:0',
                    'command': [sys.executable, *sys.argv], 'maximum_seconds': 300,
                    'git_review_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'torch': torch.__version__, 'numpy': np.__version__, 'model_seed': 4,
                    'checkpoint_file_sha256': checkpoint_before, 'checkpoint_parameter_sha256': before,
                    'evaluation_data_sha256': data_hashes, 'evaluation_maps_per_size': 32,
                    'conditions': {name: {'transport': t, 'perception': p, 'hops_per_step_upper_bound': hops_per_step(t, p)} for name, (t, p) in CONDITIONS.items()},
                    'horizons': list(TIMES), 'source_sha256': source_hashes,
                    'historical_reference_sha256': sha(historical_run/'stream_K8_seed4.json'),
                    'backend': old_manifest['backend']}
        write(out/'manifest.json', manifest)
        for name in source_hashes:
            dest = out/'source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, dest)
        data = {size: {key: value.cuda() for key, value in d.items()} for size, d in data_cpu.items()}
        replay = {'status': 'PENDING', 'size_horizon_records': 0, 'integer_leaves': 0,
                  'float_leaves': 0, 'maximum_absolute_error': 0.0}
        for condition in CONDITIONS:
            records[condition] = {}
            for size in (32, 64):
                write(out/'status.json', {'status': 'RUNNING', 'condition': condition, 'size': size,
                                         'pid': os.getpid(), 'updated_utc': now()})
                measured = evaluate(model, data[size], condition)
                records[condition][str(size)] = measured
                if condition == 'full':
                    for t in ('64', '128', '256'):
                        compare_tree(old['evaluation'][str(size)][t], measured[t]['evaluation'], str(size)+'/'+t, replay)
                        replay['size_horizon_records'] += 1
                write(out/'raw_conditions.json', records)
                print(json.dumps({'condition': condition, 'size': size, 'status': 'COMPLETE'}), flush=True)
            if condition == 'full':
                assert replay['size_horizon_records'] == 6
                replay['status'] = 'PASS'
                write(out/'replay_validation.json', replay)
        assert tensor_hash(model.state_dict()) == before
        assert sha(checkpoint_path) == checkpoint_before
        budget()
        rows = curve_rows(records)
        with (out/'curves.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write(out/'contrasts.json', contrasts(records))
        summary = summarize(records, replay, time.monotonic()-start)
        summary['parameters_and_checkpoint_unchanged'] = True
        write(out/'summary.json', summary)
        (out/'RESULTS.md').write_text(report(records, summary), encoding='utf-8')
        write(out/'status.json', {'status': 'COMPLETE', 'finished_utc': now(), 'pid': os.getpid(),
                                 'completed_conditions': 4, 'elapsed_seconds': time.monotonic()-start})
        print(json.dumps(summary), flush=True)
    except Exception as error:
        status = 'TIME_BUDGET' if isinstance(error, TimeoutError) else 'ERROR'
        write(out/'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        write(out/'status.json', {'status': status, 'finished_utc': now(), 'pid': os.getpid(),
                                 'elapsed_seconds': time.monotonic()-start})
        raise
    finally:
        timer.cancel()


if __name__ == '__main__':
    main()
