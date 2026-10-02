"""Exactly nested stationary-sidecar K8 development screen, fixed three arms."""
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

from sidecar_cells import ROOT, VARIANTS, PARAMETERS, make_variant, core_state_dict

sys.path.insert(0, str(ROOT / 'new/nca_inertial_wind_tunnel'))
sys.path.insert(0, str(ROOT / 'new/short_bptt_phase2'))
spec = importlib.util.spec_from_file_location(
    'phase2_runner', ROOT / 'new/short_bptt_phase2/run.py')
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)

sys.path.insert(0, str(ROOT / 'new/workspace_revision'))
from run_revision import now, sha, write, tensor_hash
from tasks import bank, subset

SEEDS = (2, 3, 4, 5)
UPDATES = 300
PREFLIGHT_UPDATES = 3
PREFLIGHT_CAP = 180
FORMAL_CAP = 2400
WATCHDOG_GRACE = 60
ORDER = {
    2: ('stream', 'memory', 'stateless'),
    3: ('memory', 'stateless', 'stream'),
    4: ('stateless', 'stream', 'memory'),
    5: ('stateless', 'memory', 'stream'),
}
HISTORICAL_STREAM = ROOT / 'evidence/streaming_carry_init2345/raw'


def historical_record(seed, variant='stream'):
    if variant != 'stream':
        raise ValueError(f'No historical control for {variant}')
    path = HISTORICAL_STREAM / f'stream_K8_seed{seed}.json'
    return json.loads(path.read_text(encoding='utf-8'))

def sources():
    """Bind the frozen dependencies and the complete new executed source set."""
    base_path = ROOT / 'STREAMING_CARRY_PUBLICATION_MANIFEST.json'
    prior = json.loads(base_path.read_text(encoding='utf-8'))['source_sha256']
    if len(prior) != 35:
        raise RuntimeError(f'Expected 35 frozen source bindings, found {len(prior)}')
    for name, expected in prior.items():
        actual = sha(ROOT / name)
        if actual != expected:
            raise RuntimeError(f'Frozen source drift: {name}')

    added = ['STREAMING_CARRY_PUBLICATION_MANIFEST.json']
    added += ['new/stationary_sidecar/' + name for name in
              ('sidecar_cells.py', 'run.py', 'check.py', 'PROTOCOL.md')]
    added += ['tools/launch_stationary_sidecar.ps1']
    added += [f'evidence/streaming_carry_init2345/raw/stream_K8_seed{seed}.json'
              for seed in SEEDS]
    if set(added) & set(prior):
        raise RuntimeError('New evidence bindings overlap the frozen source set')
    return {**prior, **{name: sha(ROOT / name) for name in added}}


def common_parameter_hash(model):
    return tensor_hash(core_state_dict(model))


def gradient_groups(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if name.startswith(('g_', 'feedback_')):
            groups[name] = (0. if parameter.grad is None else
                            float(torch.linalg.vector_norm(parameter.grad)))
    return groups


@torch.no_grad()
def state_diagnostics(model, state):
    result = {'W_rms': float(state[0].square().mean().sqrt()),
              'Z_rms': float(state[1].square().mean().sqrt())}
    if len(state) == 3:
        result.update({
            'stored_H_rms': float(state[2].square().mean().sqrt()),
            'active_H_rms': float(model.last_active_h.square().mean().sqrt()),
            'scaled_F_injection_rms': .1 * float(model.last_injection_f.square().mean().sqrt()),
            'scaled_Q_injection_rms': .5 * float(model.last_injection_q.square().mean().sqrt()),
        })
    return result


def gradient_entry_check(record):
    if record['variant'] == 'stream':
        return True
    logs = record['training_curve']
    if not logs:
        return False
    first = logs[0]['side_gradient_norms']
    late = [row['side_gradient_norms'] for row in logs if row['update'] <= 3]
    required = ('g_in.weight', 'g_in.bias', 'g_out.weight', 'g_out.bias',
                'feedback_f.weight')
    return bool(first.get('feedback_q.weight', 0.) > 0 and
                all(any(row.get(name, 0.) > 0 for row in late) for name in required))

def cpu_initialization_reference():
    """Check same-seed common draws without touching CUDA or consuming RNG state."""
    reference = {}
    for seed in SEEDS:
        reference[str(seed)] = {}
        with torch.random.fork_rng(devices=[]):
            for variant in VARIANTS:
                cpu_rng = torch.Generator(device='cpu').manual_seed(seed)
                torch.set_rng_state(cpu_rng.get_state())
                model = make_variant(variant)
                state = model.state_dict()
                reference[str(seed)][variant] = {
                    'initial_parameter_sha256': tensor_hash(state),
                    'initial_common_parameter_sha256': common_parameter_hash(model),
                    'parameter_count': sum(p.numel() for p in model.parameters()),
                }
                del model

    for seed in SEEDS:
        row = reference[str(seed)]
        common = {row[v]['initial_common_parameter_sha256'] for v in VARIANTS}
        if len(common) != 1:
            raise RuntimeError(f'Original core initialization differs at seed {seed}')
        if row['memory']['initial_parameter_sha256'] != row['stateless']['initial_parameter_sha256']:
            raise RuntimeError(f'Side-arm full initialization differs at seed {seed}')
        for variant in VARIANTS:
            if row[variant]['parameter_count'] != PARAMETERS[variant]:
                raise RuntimeError(f'Unexpected parameter count for {variant}, seed {seed}')
        old = historical_record(seed)
        if row['stream']['initial_parameter_sha256'] != old['initial_parameter_sha256']:
            raise RuntimeError(f'Stream historical initialization mismatch at seed {seed}')

    return reference


def train_one(out, seed, variant, rows, train, tests, preflight, init_reference):
    p2.budget()
    torch.manual_seed(seed)
    model = make_variant(variant).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    name = f'{variant}_K8_seed{seed}'
    old = historical_record(seed) if variant == 'stream' else None
    common_hash = common_parameter_hash(model)
    record = {
        'variant': variant,
        'architecture': model.metadata(),
        'seed': seed,
        'gradient_horizon': 8,
        'initial_parameter_sha256': tensor_hash(model.state_dict()),
        'initial_common_parameter_sha256': common_hash,
        'initialization_reference_matches': (
            tensor_hash(model.state_dict()) ==
            init_reference[str(seed)][variant]['initial_parameter_sha256'] and
            common_hash ==
            init_reference[str(seed)][variant]['initial_common_parameter_sha256']),
        'historical_initial_parameter_sha256_matches': (
            None if old is None else
            tensor_hash(model.state_dict()) == old['initial_parameter_sha256']),
        'train_data_sha256': tensor_hash(train),
        'schedule_sha256': sha(out / 'schedule.json'),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'status': 'TRAINING',
        'completed_updates': 0,
        'training_curve': [],
        'update_seconds': [],
        'evaluation': {},
    }
    started = time.monotonic()
    clipped = 0
    try:
        if record['parameter_count'] != PARAMETERS[variant]:
            raise RuntimeError('Parameter count mismatch')
        if not record['initialization_reference_matches']:
            raise RuntimeError('Live initialization differs from CPU reference')
        if record['historical_initial_parameter_sha256_matches'] is False:
            raise RuntimeError('Historical control initialization mismatch')
        if old is not None:
            for key in ('train_data_sha256',):
                if record[key] != old[key]:
                    raise RuntimeError(f'Historical control identity mismatch: {key}')
            if not preflight and record['schedule_sha256'] != old['schedule_sha256']:
                raise RuntimeError('Historical control identity mismatch: schedule_sha256')

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        record['allocated_before_training_bytes'] = torch.cuda.memory_allocated()
        for iteration, indices in enumerate(rows, 1):
            p2.budget()
            batch = subset(train, indices)
            torch.cuda.synchronize()
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss, state, trace = p2.backward_trajectory(
                model, batch, 8, budget=p2.budget)
            if not all(bool(torch.isfinite(value).all()) for value in state):
                raise FloatingPointError('Nonfinite training state')
            side_grads = gradient_groups(model) if (iteration <= 3 or iteration % 100 == 0) else {}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError('Nonfinite gradient')
            clipped += int(norm > 1.)
            optimizer.step()
            torch.cuda.synchronize()
            record['update_seconds'].append(time.monotonic() - tick)
            record['completed_updates'] = iteration
            if iteration <= 3 or iteration % 100 == 0 or iteration == len(rows) or preflight:
                log = {
                    'update': iteration,
                    'mean_trajectory_loss': float(loss),
                    'gradient_norm_before_clip': float(norm),
                    'update_seconds': record['update_seconds'][-1],
                    'side_gradient_norms': side_grads,
                    **state_diagnostics(model, state),
                    **trace,
                }
                record['training_curve'].append(log)
                write(out / f'{name}.json', record)
                write(out / 'status.json', {
                    'status': 'RUNNING', 'phase': 'training', 'arm': name,
                    'pid': os.getpid(), 'completed_updates': iteration,
                    'updated_utc': now(),
                })
                print(json.dumps({'arm': name, **log}), flush=True)

        record['gradient_entry_passed'] = gradient_entry_check(record)
        if not record['gradient_entry_passed']:
            raise RuntimeError('Side-path gradient entry failed by update3')
        record['training_seconds'] = time.monotonic() - started
        record['training_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['training_peak_increment_bytes'] = (
            record['training_peak_allocated_bytes'] -
            record['allocated_before_training_bytes'])
        record['gradient_clip_fraction'] = clipped / len(rows)
        record['final_parameter_sha256'] = tensor_hash(model.state_dict())
        torch.save({
            'variant': variant,
            'architecture': model.metadata(),
            'gradient_horizon': 8,
            'seed': seed,
            'completed_updates': len(rows),
            'state_dict': {key: value.detach().cpu()
                           for key, value in model.state_dict().items()},
        }, out / f'{name}.pt')
        record['status'] = 'TRAINED'
        write(out / f'{name}.json', record)
        model.eval()
        for size, data in tests.items():
            p2.budget()
            write(out / 'status.json', {
                'status': 'RUNNING', 'phase': 'evaluation', 'arm': name,
                'size': size, 'pid': os.getpid(), 'updated_utc': now(),
            })
            record['evaluation'][str(size)] = p2.evaluate(model, data, 8, preflight)
        p2.budget()
        record['status'] = 'COMPLETE'
    except Exception as error:
        record['status'] = 'TIME_BUDGET' if isinstance(error, TimeoutError) else 'ERROR'
        record['error'], record['traceback'] = repr(error), traceback.format_exc()
        if not (out / f'{name}.pt').exists():
            torch.save({
                'partial': True,
                'variant': variant,
                'architecture': model.metadata(),
                'completed_updates': record['completed_updates'],
                'state_dict': {key: value.detach().cpu()
                               for key, value in model.state_dict().items()},
            }, out / f'{name}.pt')
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        write(out / f'{name}.json', record)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return record


def predicates(record):
    endpoint = p2.endpoint(record['evaluation'])
    hold = True
    for horizon in ('128', '256'):
        evaluation = record['evaluation']['32'][horizon]
        band = evaluation['bands']['strict_16_32']
        hold = hold and (
            evaluation['original']['balanced_accuracy'] >= endpoint['ba_original'] - .03)
        hold = hold and (
            evaluation['flipped']['balanced_accuracy'] >= endpoint['ba_flipped'] - .03)
        hold = hold and band['mean'] is not None and band['pooled_accuracy'] is not None
        if hold:
            hold = (
                band['mean'] >= endpoint['primary_mean'] - .05 and
                band['pooled_accuracy'] >= endpoint['primary_pooled'] - .05)
    reach = p2.reach_predicate(endpoint)
    return {
        'seed': record['seed'],
        'variant': record['variant'],
        **endpoint,
        'reach': reach,
        'hold': bool(hold),
        'reach_and_hold': bool(reach and hold),
    }


def control_history_reproduction(results, complete):
    by_key = {(row['seed'], row['variant']): row for row in results}
    checks = []
    for seed in SEEDS:
        for variant in ('stream',):
            current = by_key.get((seed, variant))
            if current is None or current['status'] != 'COMPLETE':
                checks.append({
                    'seed': seed, 'variant': variant,
                    'final_parameter_sha256_matches_historical': None,
                    'full_evaluation_matches_historical': None,
                    'reproduced': None,
                })
                continue
            old = historical_record(seed, variant)
            final_match = current.get('final_parameter_sha256') == old.get(
                'final_parameter_sha256')
            evaluation_match = current.get('evaluation') == old.get('evaluation')
            checks.append({
                'seed': seed,
                'variant': variant,
                'final_parameter_sha256_matches_historical': bool(final_match),
                'full_evaluation_matches_historical': bool(evaluation_match),
                'reproduced': bool(final_match and evaluation_match),
            })
    all_known = len(checks) == 4 and all(row['reproduced'] is not None for row in checks)
    all_reproduced = all_known and all(row['reproduced'] for row in checks)
    return {
        'expected_controls': 4,
        'checks': checks,
        'all_reproduced': bool(all_reproduced) if all_known else None,
        'complete_control_set': bool(all_known),
    }


def decision(rows, complete, control_reproduction):
    counts = {
        variant: {
            'reach': sum(row['variant'] == variant and row['reach'] for row in rows),
            'reach_and_hold': sum(
                row['variant'] == variant and row['reach_and_hold'] for row in rows),
        }
        for variant in VARIANTS
    }
    by_key = {(row['seed'], row['variant']): row for row in rows}
    anchors = {
        'stream_seed_4_reach_and_hold': (
            by_key.get((4, 'stream'), {}).get('reach_and_hold')),
    }
    candidate = [row for row in rows if row['variant'] == 'memory']
    candidate_seeds = {row['seed'] for row in candidate if row['reach_and_hold']}
    anchors_ok = all(value is True for value in anchors.values())
    if not complete:
        status = 'INCOMPLETE'
    elif not control_reproduction['all_reproduced'] or not anchors_ok:
        status = 'CONTROL_REPRODUCTION_DRIFT'
    elif (
        counts['memory']['reach_and_hold'] >= 3 and
        4 in candidate_seeds and
        counts['memory']['reach_and_hold'] > counts['stream']['reach_and_hold'] and
        counts['memory']['reach_and_hold'] > counts['stateless']['reach_and_hold']
    ):
        status = 'DEVELOPMENT_GO'
    else:
        status = 'DEVELOPMENT_NO_GO'
    return {
        'decision': status,
        'counts': counts,
        'control_anchor_checks': anchors,
    }

def curve_rows(results):
    rows = []
    for result in results:
        for size, evaluation in result['evaluation'].items():
            for horizon, metrics in evaluation.items():
                for band, value in metrics['bands'].items():
                    rows.append({
                        'seed': result['seed'],
                        'variant': result['variant'],
                        'size': int(size),
                        'T': int(horizon),
                        'band': band,
                        'mean': value['mean'],
                        'pooled': value['pooled_accuracy'],
                        'pixels': value['pooled_pixels'],
                        'correct': value['pooled_correct'],
                        'eligible_maps': value['eligible_maps'],
                        'ba_original': metrics['original']['balanced_accuracy'],
                        'ba_flipped': metrics['flipped']['balanced_accuracy'],
                    })
    return rows


def paired_effect_rows(curves):
    lookup = {
        (row['seed'], row['variant'], row['size'], row['T'], row['band']): row
        for row in curves
    }
    effects = []
    for candidate in curves:
        if candidate['variant'] != 'memory':
            continue
        for control in ('stream', 'stateless'):
            comparator = lookup.get((
                candidate['seed'], control, candidate['size'],
                candidate['T'], candidate['band']))
            if comparator is None:
                continue
            if (candidate['pixels'] != comparator['pixels'] or
                    candidate['eligible_maps'] != comparator['eligible_maps']):
                raise RuntimeError('Paired comparison coverage differs')
            effects.append({
                'comparison': f'memory-{control}',
                **{key: candidate[key] for key in (
                    'seed', 'size', 'T', 'band', 'pixels', 'eligible_maps')},
                **{
                    key + '_effect_pp': (
                        None if candidate[key] is None or comparator[key] is None else
                        100 * (candidate[key] - comparator[key]))
                    for key in ('mean', 'pooled', 'ba_original', 'ba_flipped')
                },
            })
    return effects


def write_result_markdown(out, doc):
    if not doc['complete']:
        (out / 'RESULTS.md').write_text(
            '# Stationary Sidecar: INCOMPLETE\n\n'
            'Execution failed or reached its time cap. Read status.json, '
            'error.json and partial arm records; no architecture verdict is available.\n',
            encoding='utf-8')
        return
    lines = [
        '# Stationary Sidecar screen', '',
        f"Execution: COMPLETE. Decision: {doc['decision']}.",
        'This is a developmental comparison on previously inspected seeds and maps.',
        'The memory/stateless pair has identical7989 parameter capacity; only extra H carry differs.',
        'Carry also changes activation accumulation/scale. Neither scale-independent memory causality nor FLOP equality is established.', '',
        '| Seed | Variant | Original BA % | Flipped BA % | Primary mean % | Primary pooled % | Reach | Hold |',
        '|---:|---|---:|---:|---:|---:|---|---|',
    ]
    for row in sorted(doc['rows'], key=lambda item: (item['seed'], item['variant'])):
        lines.append(
            f"| {row['seed']} | {row['variant']} | "
            f"{100 * row['ba_original']:.2f} | {100 * row['ba_flipped']:.2f} | "
            f"{100 * row['primary_mean']:.2f} | {100 * row['primary_pooled']:.2f} | "
            f"{row['reach']} | {row['hold']} |")
    lines += [
        '',
        'Reach+hold counts: ' + '; '.join(
            f"{variant}={doc['counts'][variant]['reach_and_hold']}/4"
            for variant in VARIANTS) + '.',
        'Primary: size32/T64, strict16<d<32. Hold: T128 and T256 relative to T64.',
        'Exact control reproduction requires each control seed’s final parameter SHA and full evaluation payload to match its historical record.',
        'See curves.csv and paired_effects.csv for all seeds, sizes, horizons and distance bands.',
        'Paired effects include memory-minus-stream and memory-minus-stateless rows.',
        'All arms preserve the original at-most-two-hop macro clock: K8 graph radius<=16.',
        'Empty bands are null. Farther gains cannot replace the frozen primary and hold gate.',
        'Raw arm JSON retains per-map counts, integer denominators, clipping, timing and memory.',
        'No reliability or population-level claim follows from four seeds and reused maps.',
    ]
    if doc['decision'] == 'CONTROL_REPRODUCTION_DRIFT':
        lines += ['', 'At least one required historical control identity, final parameter SHA, full evaluation, or control anchor did not reproduce.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def aggregate(out, results, preflight, cover):
    expected = {(seed, variant) for seed in ((2,) if preflight else SEEDS)
                for variant in VARIANTS}
    keys = [(result['seed'], result['variant']) for result in results]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise RuntimeError('Duplicate or unexpected arm records')
    updates = PREFLIGHT_UPDATES if preflight else UPDATES
    complete = (
        set(keys) == expected and
        all(result['status'] == 'COMPLETE' and
            result['completed_updates'] == updates for result in results))
    if len({result['train_data_sha256'] for result in results}) > 1:
        raise RuntimeError('Training data changed across arms')
    if len({result['schedule_sha256'] for result in results}) > 1:
        raise RuntimeError('Training schedule changed across arms')
    common_identity_ok = True
    seeds_in_results = {seed for seed, _ in keys}
    for seed in seeds_in_results:
        arm_rows = [result for result in results if result['seed'] == seed]
        if len({result['initial_common_parameter_sha256'] for result in arm_rows}) > 1:
            common_identity_ok = False
        memory = next((result for result in arm_rows
                       if result['variant'] == 'memory'), None)
        stateless = next((result for result in arm_rows
                         if result['variant'] == 'stateless'), None)
        if (memory is not None and stateless is not None and
                memory['initial_parameter_sha256'] != stateless['initial_parameter_sha256']):
            common_identity_ok = False

    initialization_checks_ok = all(
        result['initialization_reference_matches'] and
        result['historical_initial_parameter_sha256_matches'] is not False
        for result in results)
    identities_verified = bool(common_identity_ok and initialization_checks_ok)
    doc = {
        'complete': complete,
        'coverage': cover,
        'identities_verified': identities_verified,
        'statuses': [{key: result[key] for key in (
            'seed', 'variant', 'status', 'completed_updates')}
            for result in results],
    }
    if preflight:
        cost = (max(float(np.median(result['update_seconds'][-2:]))
                    for result in results) if complete else None)
        estimate = 12 * UPDATES * cost * 1.20 + 180 if complete else None
        memory_ok = (
            complete and
            max(result['training_peak_allocated_bytes'] for result in results) <
            3 * 1024**3)
        gradient_entry_ok = complete and all(result.get('gradient_entry_passed', False)
                                            for result in results)
        passed = (
            complete and identities_verified and gradient_entry_ok and estimate <= FORMAL_CAP and
            memory_ok and cover['eligible_maps'] >= 16 and cover['pixels'] >= 500)
        doc.update({
            'decision': 'PREFLIGHT_PASSED' if passed else 'PREFLIGHT_FAILED',
            'worst_last_two_median_update_seconds': cost,
            'estimated_formal_seconds': estimate,
            'memory_gate': memory_ok,
            'gradient_entry_gate': gradient_entry_ok,
            'updates_per_arm': UPDATES,
            'formal_hard_cap_seconds': FORMAL_CAP,
        })
    else:
        rows = [predicates(result) for result in results
                if result['status'] == 'COMPLETE']
        control_reproduction = control_history_reproduction(results, complete)
        doc.update({
            'rows': rows,
            'control_history_reproduction': control_reproduction,
            **decision(rows, complete, control_reproduction),
        })
        curves = curve_rows(results)
        effects = paired_effect_rows(curves)
        if complete and (len(curves) != 936 or len(effects) != 624):
            raise RuntimeError(
                f'Unexpected curve/effect coverage: {len(curves)}/{len(effects)}')
        doc['paired_effects'] = effects
        for filename, data in (
            ('curves.csv', curves), ('paired_effects.csv', effects)):
            if data:
                with (out / filename).open(
                        'w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                    writer.writeheader()
                    writer.writerows(data)
        write_result_markdown(out, doc)
    write(out / 'aggregate.json', doc)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--preflight-dir')
    args = parser.parse_args()

    hashes = sources()
    init_reference = cpu_initialization_reference()
    binding = None
    if not args.preflight:
        if not args.preflight_dir:
            parser.error('Formal run requires a passed --preflight-dir')
        preflight_dir = Path(args.preflight_dir).resolve()
        preflight_status = json.loads(
            (preflight_dir / 'status.json').read_text(encoding='utf-8'))
        if preflight_status['status'] != 'PREFLIGHT_PASSED':
            raise RuntimeError('Formal run requires a passed preflight')
        preflight_manifest = json.loads(
            (preflight_dir / 'manifest.json').read_text(encoding='utf-8'))
        if preflight_manifest['source_sha256'] != hashes:
            raise RuntimeError('Source bundle changed after preflight')
        preflight_summary = json.loads(
            (preflight_dir / 'aggregate.json').read_text(encoding='utf-8'))
        if (preflight_summary['decision'] != 'PREFLIGHT_PASSED' or
                preflight_summary['updates_per_arm'] != UPDATES or
                not preflight_summary['identities_verified']):
            raise RuntimeError('Preflight summary failed its fixed gate')
        binding = {
            name: sha(preflight_dir / name)
            for name in ('status.json', 'manifest.json', 'aggregate.json')
        }

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    p2.DEADLINE = started + (PREFLIGHT_CAP if args.preflight else FORMAL_CAP)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False

    tests_cpu = {size: bank(size, 32, 40000 + size) for size in (32, 64)}
    eval_hashes = {str(size): tensor_hash(data)
                   for size, data in tests_cpu.items()}
    old_manifest = json.loads((
        ROOT / 'evidence/short_bptt_phase2_init2345/manifest.json'
    ).read_text(encoding='utf-8'))
    if eval_hashes != old_manifest['evaluation_data_sha256']:
        raise RuntimeError('Evaluation bank changed')
    cover = p2.coverage(tests_cpu[32])
    if cover['eligible_maps'] < 16 or cover['pixels'] < 500:
        raise RuntimeError('Primary evaluation coverage gate failed')
    tests = ({32: subset(tests_cpu[32], [0, 1])}
             if args.preflight else tests_cpu)
    tests = {size: {key: value.cuda() for key, value in data.items()}
             for size, data in tests.items()}
    train = bank(32, 512, 10002, 'cuda')
    update_count = PREFLIGHT_UPDATES if args.preflight else UPDATES
    rows = np.random.default_rng(20002).integers(
        0, 512, (update_count, 8)).tolist()
    write(out / 'schedule.json', rows)

    manifest = {
        'protocol': 'stationary_sidecar_v1',
        'training': True,
        'preflight': args.preflight,
        'started_utc': now(),
        'pid': os.getpid(),
        'initialization_seeds': [2] if args.preflight else list(SEEDS),
        'variants': list(VARIANTS),
        'arm_order': {str(seed): list(order) for seed, order in ORDER.items()},
        'state_channels_by_variant': {
            'stream': {'W': 24, 'Z': 8},
            'memory': {'W': 24, 'Z': 8, 'H': 12},
            'stateless': {'W': 24, 'Z': 8, 'H': 12, 'H_reset_each_step': True}},
        'carrier_channels_per_arm': 24,
        'sidecar_write_mlp': {'input_channels': 67, 'hidden_channels': 32,
                             'output_channels': 12, 'input_includes_H': False},
        'sidecar_order': 'write H* from old core features; consume in F then Q; retain or reset',
        'sidecar_write_scale': .1,
        'sidecar_initialization': 'H0=0; G_in default; G_out weightN(0,.02),bias0; P_F/P_Q weight0,no bias',
        'macro_graph_radius_upper_bound': 2,
        'K8_graph_radius_upper_bound': 16,
        'transport': 'original masked port permutation with blocked-link bounce-back',
        'comparison_scope': 'additional carried side state versus same-capacity instantaneous branch; not scale-independent memory causality',
        'parameter_count_by_variant': PARAMETERS,

        'gradient_horizon': 8,
        'updates_per_arm': update_count,
        'forward_steps': 64,
        'loss_times': list(range(8, 65, 8)),
        'train_data_seed': 10002,
        'schedule_seed': 20002,
        'schedule_sha256': sha(out / 'schedule.json'),
        'train_data_sha256': tensor_hash(train),
        'evaluation_map_count_per_size': 32,
        'evaluation_data_sha256': eval_hashes,
        'executed_eval_data_sha256': {
            str(size): tensor_hash(data) for size, data in tests.items()},
        'primary_coverage': cover,
        'cpu_initialization_reference': init_reference,
        'source_sha256': hashes,
        'preflight_sha256': binding,
        'optimizer': {
            'type': 'AdamW', 'lr': .001, 'weight_decay': .0001,
            'clip_norm': 1., 'batch': 8,
        },
        'backend': {
            'cudnn_benchmark': False,
            'cudnn_deterministic': False,
            'cudnn_tf32': True,
            'matmul_tf32': False,
        },
        'torch': torch.__version__,
        'numpy': np.__version__,
        'gpu': torch.cuda.get_device_name(),
        'maximum_seconds': PREFLIGHT_CAP if args.preflight else FORMAL_CAP,
    }
    write(out / 'manifest.json', manifest)
    for name in hashes:
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)

    def timeout():
        write(out / 'watchdog_timeout.json', {
            'status': 'TIME_BUDGET', 'time': now(),
            'hard_cap_seconds': manifest['maximum_seconds'],
            'watchdog_grace_seconds': WATCHDOG_GRACE,
        })
        os._exit(124)

    watchdog = threading.Timer(
        max(1., p2.DEADLINE + WATCHDOG_GRACE - time.monotonic()), timeout)
    watchdog.daemon = True
    watchdog.start()
    results = []
    status = 'ERROR'
    try:
        for seed in manifest['initialization_seeds']:
            for variant in ORDER[seed]:
                result = train_one(
                    out, seed, variant, rows, train, tests,
                    args.preflight, init_reference)
                results.append(result)
                aggregate(out, results, args.preflight, cover)
                if result['status'] != 'COMPLETE':
                    raise RuntimeError(f"Arm stopped: {result['status']}")
        summary = aggregate(out, results, args.preflight, cover)
        p2.budget()
        status = (summary['decision'] if args.preflight else
                  ('COMPLETE' if summary['complete'] else 'INCOMPLETE'))
    except Exception as error:
        status = ('TIME_BUDGET' if isinstance(error, TimeoutError) or
                  any(result['status'] == 'TIME_BUDGET' for result in results)
                  else 'ERROR')
        write(out / 'error.json', {
            'error': repr(error), 'traceback': traceback.format_exc()})
        if (out / 'aggregate.json').exists():
            partial = json.loads(
                (out / 'aggregate.json').read_text(encoding='utf-8'))
            partial.update({
                'complete': False,
                'decision': 'PREFLIGHT_FAILED' if args.preflight else 'INCOMPLETE',
                'execution_status': status,
                'execution_error': repr(error),
            })
            write(out / 'aggregate.json', partial)
        if not args.preflight:
            write_result_markdown(out, {
                'complete': False, 'decision': 'INCOMPLETE'})
    finally:
        watchdog.cancel()

    write(out / 'status.json', {
        'status': status,
        'pid': os.getpid(),
        'completed_arms': sum(result['status'] == 'COMPLETE'
                              for result in results),
        'finished_utc': now(),
        'elapsed_seconds': time.monotonic() - started,
    })
    print(json.dumps({
        'status': status,
        'completed_arms': sum(result['status'] == 'COMPLETE'
                              for result in results),
        'elapsed_seconds': time.monotonic() - started,
    }), flush=True)
    required_status = 'PREFLIGHT_PASSED' if args.preflight else 'COMPLETE'
    if status != required_status:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
