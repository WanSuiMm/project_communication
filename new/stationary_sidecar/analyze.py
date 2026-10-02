"""CPU-only saved-result audit of the stationary sidecar; no model inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for relative in reversed((
        'new/nca_inertial_wind_tunnel',
        'new/short_bptt_phase2',
        'new/workspace_revision')):
    sys.path.insert(0, str(ROOT / relative))

from sidecar_cells import VARIANTS, PARAMETERS, make_variant, core_state_dict  # noqa: E402
from run_revision import tensor_hash  # noqa: E402
from tasks import bank, subset  # noqa: E402

SEEDS = (2, 3, 4, 5)
UPDATES = 300
HORIZONS = (64, 128, 256)
SIZES = (32, 64)


BANDS = (
    '0_8', '8_16', 'equal_16', 'strict_16_32', '32_64', '64_128',
    '128_inf', 'far_gt16', 'far_gt32', 'r_le1', 'r_1_2', 'r_gt2',
    'outside_forward_lightcone')
CONTROL_PATHS = {
    'stream': 'evidence/streaming_carry_init2345/raw/stream_K8_seed{seed}.json',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def close(actual, expected, label):
    if actual is None or expected is None:
        assert actual is None and expected is None, (label, actual, expected)
    else:
        assert math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6), (
            label, actual, expected)


def repo_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def common_parameter_hash(model):
    return tensor_hash(core_state_dict(model))

def initialization_reference():
    result = {}
    for seed in SEEDS:
        result[str(seed)] = {}
        with torch.random.fork_rng(devices=[]):
            for variant in VARIANTS:
                generator = torch.Generator(device='cpu').manual_seed(seed)
                torch.set_rng_state(generator.get_state())
                model = make_variant(variant)
                result[str(seed)][variant] = {
                    'initial_parameter_sha256': tensor_hash(model.state_dict()),
                    'initial_common_parameter_sha256': common_parameter_hash(model),
                    'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
                }
                del model
    for seed in SEEDS:
        row = result[str(seed)]
        assert len({row[v]['initial_common_parameter_sha256'] for v in VARIANTS}) == 1
        assert row['memory']['initial_parameter_sha256'] == row['stateless']['initial_parameter_sha256']
        assert all(row[v]['parameter_count'] == PARAMETERS[v] for v in VARIANTS)
    return result


def evaluation_masks(data, horizon):
    distance = data['distance'].numpy()
    changed = data['changed'].numpy().astype(bool)
    ranges = {
        '0_8': (distance >= 0) & (distance < 8),
        '8_16': (distance >= 8) & (distance < 16),
        'equal_16': distance == 16,
        'strict_16_32': (distance > 16) & (distance < 32),
        '32_64': (distance >= 32) & (distance < 64),
        '64_128': (distance >= 64) & (distance < 128),
        '128_inf': distance >= 128,
        'far_gt16': distance > 16,
        'far_gt32': distance > 32,
        'r_le1': (distance >= 0) & (distance <= 16),
        'r_1_2': (distance > 16) & (distance <= 32),
        'r_gt2': distance > 32,
        'outside_forward_lightcone': distance > 2 * horizon,
    }
    assert set(ranges) == set(BANDS)
    return {'paired': changed, **{name: changed & mask for name, mask in ranges.items()}}


def verify_metric(metric, expected_counts, label):
    counts = [int(value) for value in expected_counts]
    hits = metric['per_map_correct']
    assert metric['per_map_pixels'] == counts, label
    assert len(hits) == len(counts) == 32, label
    assert all(isinstance(value, int) and 0 <= value <= count
               for value, count in zip(hits, counts)), label
    per_map = [None if count == 0 else hit / count for hit, count in zip(hits, counts)]
    assert len(metric['per_map']) == 32, label
    for index, value in enumerate(per_map):
        close(metric['per_map'][index], value, (label, 'per_map', index))
    eligible = sum(count > 0 for count in counts)
    total_pixels, total_correct = sum(counts), sum(hits)
    valid = [value for value in per_map if value is not None]
    mean = float(np.mean(valid)) if valid else None
    pooled = total_correct / total_pixels if total_pixels else None
    assert metric['eligible_maps'] == eligible, label
    assert metric['pooled_pixels'] == total_pixels, label
    assert metric['pooled_correct'] == total_correct, label
    close(metric['mean'], mean, (label, 'mean'))
    close(metric['pooled_accuracy'], pooled, (label, 'pooled_accuracy'))


def verify_csv(path, expected, key_fields, fields):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == tuple(fields), (path.name, reader.fieldnames)
        rows = list(reader)
    expected_by_key = {tuple(str(row[field]) for field in key_fields): row for row in expected}
    actual_by_key = {tuple(row[field] for field in key_fields): row for row in rows}
    assert len(rows) == len(expected) == len(expected_by_key), path.name
    assert len(actual_by_key) == len(rows) and set(actual_by_key) == set(expected_by_key), path.name
    integer_fields = {'seed', 'size', 'T', 'pixels', 'correct', 'eligible_maps'}
    for key, row in expected_by_key.items():
        actual = actual_by_key[key]
        for field in fields:
            value = row[field]
            text = actual[field]
            if value is None:
                assert text == '', (path.name, key, field, text)
            elif isinstance(value, str):
                assert text == value, (path.name, key, field, text, value)
            elif field in integer_fields:
                assert int(text) == int(value), (path.name, key, field, text, value)
            else:
                close(float(text), float(value), (path.name, key, field))
    return len(rows)


def endpoint(record):
    primary = record['evaluation']['32']['64']
    band = primary['bands']['strict_16_32']
    return {
        'ba_original': primary['original']['balanced_accuracy'],
        'ba_flipped': primary['flipped']['balanced_accuracy'],
        'primary_mean': band['mean'],
        'primary_pooled': band['pooled_accuracy'],
    }


def gate_row(record):
    end = endpoint(record)
    reach = (end['ba_original'] >= .85 and end['ba_flipped'] >= .85 and
             end['primary_mean'] is not None and end['primary_mean'] >= .80 and
             end['primary_pooled'] is not None and end['primary_pooled'] >= .80)
    primary = record['evaluation']['32']['64']
    hold = True
    for horizon in ('128', '256'):
        evaluation = record['evaluation']['32'][horizon]
        band = evaluation['bands']['strict_16_32']
        hold = hold and (
            evaluation['original']['balanced_accuracy'] >= end['ba_original'] - .03 and
            evaluation['flipped']['balanced_accuracy'] >= end['ba_flipped'] - .03 and
            band['mean'] is not None and band['pooled_accuracy'] is not None)
        if hold:
            hold = (band['mean'] >= end['primary_mean'] - .05 and
                    band['pooled_accuracy'] >= end['primary_pooled'] - .05)
    return {
        'seed': record['seed'], 'variant': record['variant'], **end,
        'reach': bool(reach), 'hold': bool(hold),
        'reach_and_hold': bool(reach and hold),
    }


def expected_decision(rows, control_reproduction):
    counts = {
        variant: {
            'reach': sum(row['variant'] == variant and row['reach'] for row in rows),
            'reach_and_hold': sum(row['variant'] == variant and row['reach_and_hold'] for row in rows),
        }
        for variant in VARIANTS
    }
    by_key = {(row['seed'], row['variant']): row for row in rows}
    anchors = {
        'stream_seed_4_reach_and_hold': by_key[(4, 'stream')]['reach_and_hold'],
    }
    if not control_reproduction or not all(value is True for value in anchors.values()):
        verdict = 'CONTROL_REPRODUCTION_DRIFT'
    else:
        passing = {row['seed'] for row in rows
                   if row['variant'] == 'memory' and row['reach_and_hold']}
        verdict = ('DEVELOPMENT_GO'
                   if counts['memory']['reach_and_hold'] >= 3 and 4 in passing and
                   counts['memory']['reach_and_hold'] > counts['stream']['reach_and_hold'] and
                   counts['memory']['reach_and_hold'] > counts['stateless']['reach_and_hold']
                   else 'DEVELOPMENT_NO_GO')
    return verdict, counts, anchors


def gradient_entry_check(record):
    if record['variant'] == 'stream':
        return True
    logs = {row['update']: row['side_gradient_norms']
            for row in record['training_curve'] if row['update'] <= 3}
    assert set(logs) == {1, 2, 3}
    required = ('g_in.weight', 'g_in.bias', 'g_out.weight', 'g_out.bias',
                'feedback_f.weight')
    return logs[1]['feedback_q.weight'] > 0 and all(
        any(logs[t][name] > 0 for t in (2, 3)) for name in required)


def rollout_row(record, size, horizon, band='strict_16_32'):
    evaluation = record['evaluation'][str(size)][str(horizon)]
    metric = evaluation['bands'][band]
    return {'seed': record['seed'], 'variant': record['variant'],
            'size': size, 'T': horizon, 'band': band,
            'mean': metric['mean'], 'pooled': metric['pooled_accuracy'],
            'correct': metric['pooled_correct'], 'pixels': metric['pooled_pixels'],
            'ba_original': evaluation['original']['balanced_accuracy'],
            'ba_flipped': evaluation['flipped']['balanced_accuracy']}


def hold_failures(record):
    end = endpoint(record)
    failures = []
    for horizon in (128, 256):
        row = rollout_row(record, 32, horizon)
        for field, floor in (('ba_original', end['ba_original']-.03),
                             ('ba_flipped', end['ba_flipped']-.03),
                             ('mean', end['primary_mean']-.05),
                             ('pooled', end['primary_pooled']-.05)):
            if row[field] is None or row[field] < floor:
                failures.append({'T': horizon, 'metric': field,
                                 'value': row[field], 'minimum': floor})
    return failures

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--preflight', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run, preflight, out = map(repo_path, (args.run, args.preflight, args.out))
    assert not out.exists(), f'Output already exists: {out}'
    assert run.is_dir() and preflight.is_dir()
    torch.set_num_threads(2)

    manifest, status, aggregate = (read(run / name) for name in
                                   ('manifest.json', 'status.json', 'aggregate.json'))
    pf_manifest, pf_status, pf_aggregate = (read(preflight / name) for name in
                                            ('manifest.json', 'status.json', 'aggregate.json'))
    assert manifest['protocol'] == 'stationary_sidecar_v1'
    assert manifest['training'] is True and manifest['preflight'] is False
    assert manifest['initialization_seeds'] == list(SEEDS)
    assert manifest['variants'] == list(VARIANTS)
    assert manifest['updates_per_arm'] == UPDATES and manifest['gradient_horizon'] == 8
    assert manifest['parameter_count_by_variant'] == PARAMETERS
    assert len(manifest['source_sha256']) == 45
    assert manifest['K8_graph_radius_upper_bound'] == 16
    assert manifest['macro_graph_radius_upper_bound'] == 2
    assert manifest['arm_order'] == {'2': ['stream', 'memory', 'stateless'],
                                     '3': ['memory', 'stateless', 'stream'],
                                     '4': ['stateless', 'stream', 'memory'],
                                     '5': ['stateless', 'memory', 'stream']}
    assert status['status'] == 'COMPLETE' and status['completed_arms'] == 12
    assert status['elapsed_seconds'] < manifest['maximum_seconds'] == 2400
    assert aggregate['complete'] and aggregate['identities_verified']
    assert not (run / 'error.json').exists() and not (run / 'watchdog_timeout.json').exists()

    assert pf_status['status'] == 'PREFLIGHT_PASSED' and pf_status['completed_arms'] == 3
    assert pf_manifest['preflight'] is True and pf_manifest['initialization_seeds'] == [2]
    assert pf_manifest['source_sha256'] == manifest['source_sha256']
    assert pf_aggregate['decision'] == 'PREFLIGHT_PASSED' and pf_aggregate['identities_verified']
    assert pf_aggregate['updates_per_arm'] == UPDATES
    assert pf_aggregate['gradient_entry_gate'] and pf_aggregate['memory_gate']
    assert pf_aggregate['estimated_formal_seconds'] <= 2400
    assert manifest['preflight_sha256'] == {
        name: sha(preflight / name) for name in ('status.json', 'manifest.json', 'aggregate.json')}

    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT / name) == expected, ('current source', name)
        assert sha(run / 'source' / name) == expected, ('formal snapshot', name)
        assert sha(preflight / 'source' / name) == expected, ('preflight snapshot', name)

    training = bank(32, 512, 10002)
    evaluations = {size: bank(size, 32, 40000 + size) for size in SIZES}
    train_hash = tensor_hash(training)
    eval_hashes = {str(size): tensor_hash(data) for size, data in evaluations.items()}
    assert train_hash == manifest['train_data_sha256'] == pf_manifest['train_data_sha256']
    assert eval_hashes == manifest['evaluation_data_sha256'] == manifest['executed_eval_data_sha256']
    assert eval_hashes == pf_manifest['evaluation_data_sha256']
    pf_executed_hashes = {'32': tensor_hash(subset(evaluations[32], [0, 1]))}
    assert pf_executed_hashes == pf_manifest['executed_eval_data_sha256']

    schedule = np.random.default_rng(20002).integers(0, 512, (UPDATES, 8)).tolist()
    pf_schedule = np.random.default_rng(20002).integers(0, 512, (3, 8)).tolist()
    assert read(run / 'schedule.json') == schedule
    assert read(preflight / 'schedule.json') == pf_schedule
    assert sha(run / 'schedule.json') == manifest['schedule_sha256']
    assert sha(preflight / 'schedule.json') == pf_manifest['schedule_sha256']

    cover_mask = evaluation_masks(evaluations[32], 64)['strict_16_32']
    cover_counts = cover_mask.sum((1, 2, 3)).astype(int).tolist()
    coverage = {'eligible_maps': sum(value > 0 for value in cover_counts),
                'pixels': sum(cover_counts), 'per_map_pixels': cover_counts}
    assert coverage == manifest['primary_coverage'] == pf_manifest['primary_coverage']
    assert pf_aggregate['coverage'] == coverage

    init = initialization_reference()
    assert init == manifest['cpu_initialization_reference'] == pf_manifest['cpu_initialization_reference']
    current_records = {}
    checkpoint_hashes = {}
    parameter_hashes = {}
    historical_hashes = {}
    historical_checks = []
    for path in sorted(run.glob('*_K8_seed*.json')):
        record = read(path)
        seed, variant = int(record['seed']), record['variant']
        key = (seed, variant)
        assert key not in current_records and seed in SEEDS and variant in VARIANTS
        assert record['status'] == 'COMPLETE' and record['completed_updates'] == UPDATES
        assert record['gradient_horizon'] == 8 and record['parameter_count'] == PARAMETERS[variant]
        assert record['architecture'] == make_variant(variant).metadata()
        assert record['train_data_sha256'] == train_hash
        assert record['schedule_sha256'] == sha(run / 'schedule.json')
        assert record['initialization_reference_matches'] is True
        assert record['initial_parameter_sha256'] == init[str(seed)][variant]['initial_parameter_sha256']
        assert record['initial_common_parameter_sha256'] == init[str(seed)][variant]['initial_common_parameter_sha256']
        assert record['initial_parameter_sha256'] == manifest['cpu_initialization_reference'][str(seed)][variant]['initial_parameter_sha256']
        assert record['parameter_count'] == init[str(seed)][variant]['parameter_count']
        assert len(record['update_seconds']) == UPDATES
        assert len(record['training_curve']) >= 4
        for log in record['training_curve']:
            assert all(log[field] == value for field, value in {
                'forward_steps': 64, 'loss_count': 8, 'backward_calls': 8,
                'interior_detach_boundaries': 7}.items())

        assert record['gradient_entry_passed'] and gradient_entry_check(record)
        for log in record['training_curve']:
            for field in ('mean_trajectory_loss', 'gradient_norm_before_clip',
                          'W_rms', 'Z_rms', 'update_seconds'):
                assert math.isfinite(log[field]), (key, log['update'], field)
            for value in log['side_gradient_norms'].values():
                assert math.isfinite(value)
            if variant != 'stream':
                for field in ('active_H_rms', 'stored_H_rms', 'scaled_F_injection_rms',
                              'scaled_Q_injection_rms'):
                    assert math.isfinite(log[field]), (key, log['update'], field)
                if variant == 'stateless':
                    assert log['stored_H_rms'] == 0
        checkpoint_path = path.with_suffix('.pt')
        checkpoint_hashes[checkpoint_path.name] = sha(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert checkpoint['variant'] == variant and checkpoint['seed'] == seed
        assert checkpoint['completed_updates'] == UPDATES
        assert checkpoint['architecture'] == record['architecture']
        state = checkpoint['state_dict']
        assert sum(value.numel() for value in state.values()) == PARAMETERS[variant]
        assert all(bool(torch.isfinite(value).all()) for value in state.values())
        parameter_hash = tensor_hash(state)
        assert parameter_hash == record['final_parameter_sha256']
        parameter_hashes[f'{variant}_K8_seed{seed}'] = {
            'initial_parameter_sha256': record['initial_parameter_sha256'],
            'initial_common_parameter_sha256': record['initial_common_parameter_sha256'],
            'final_parameter_sha256': parameter_hash,
            'checkpoint_file_sha256': checkpoint_hashes[checkpoint_path.name],
        }

        if variant == 'stream':
            historical_path = ROOT / CONTROL_PATHS[variant].format(seed=seed)
            historical = read(historical_path)
            historical_hashes[historical_path.relative_to(ROOT).as_posix()] = sha(historical_path)
            checks = {
                'initial_parameter_sha256_matches': record['initial_parameter_sha256'] == historical['initial_parameter_sha256'],
                'final_parameter_sha256_matches': record['final_parameter_sha256'] == historical['final_parameter_sha256'],
                'full_evaluation_matches': record['evaluation'] == historical['evaluation'],
                'train_data_sha256_matches': record['train_data_sha256'] == historical['train_data_sha256'],
                'schedule_sha256_matches': record['schedule_sha256'] == historical['schedule_sha256'],
            }
            if variant == 'stream':
                assert record['historical_initial_parameter_sha256_matches'] is True
            assert all(checks.values()), (key, checks)
            historical_checks.append({
                'seed': seed, 'variant': variant, **checks,
                'reproduced': all(checks.values()),
            })
        current_records[key] = record

    expected_keys = {(seed, variant) for seed in SEEDS for variant in VARIANTS}
    assert set(current_records) == expected_keys
    assert len(checkpoint_hashes) == 12
    assert len(historical_checks) == 4 and all(row['reproduced'] for row in historical_checks)
    for seed in SEEDS:
        full = [current_records[(seed, variant)]['initial_parameter_sha256'] for variant in VARIANTS]
        common = [current_records[(seed, variant)]['initial_common_parameter_sha256'] for variant in VARIANTS]
        assert len(set(common)) == 1
        assert full[1] == full[2]

    # Check the twelve CPU-loaded checkpoints against the runner's final hashes;
    # do not instantiate a trained model or regenerate any outputs.
    ba_aggregates = paired_aggregates = 0
    curves = []
    gate_rows = []
    for (seed, variant), record in sorted(current_records.items()):
        assert set(record['evaluation']) == {str(size) for size in SIZES}
        assert len(record['evaluation']) == 2
        for size in SIZES:
            sequence = record['evaluation'][str(size)]
            assert set(sequence) == {str(horizon) for horizon in HORIZONS}
            data = evaluations[size]
            for horizon in HORIZONS:
                evaluation = sequence[str(horizon)]
                assert set(evaluation['bands']) == set(BANDS)
                masks = evaluation_masks(data, horizon)
                for branch in ('original', 'flipped'):
                    values = evaluation[branch]['per_map_ba']
                    assert len(values) == 32 and all(math.isfinite(value) for value in values)
                    close(evaluation[branch]['balanced_accuracy'], float(np.mean(values)),
                          (seed, variant, size, horizon, branch, 'BA mean'))
                    ba_aggregates += 1
                metrics = {'paired': evaluation['paired'], **evaluation['bands']}
                assert set(metrics) == set(masks)
                for name, metric in metrics.items():
                    counts = masks[name].sum((1, 2, 3)).astype(int).tolist()
                    verify_metric(metric, counts, (seed, variant, size, horizon, name))
                    paired_aggregates += 1
                partition = ('0_8', '8_16', 'equal_16', 'strict_16_32',
                             '32_64', '64_128', '128_inf')
                assert [sum(evaluation['bands'][band]['per_map_correct'][i]
                            for band in partition) for i in range(32)] == evaluation['paired']['per_map_correct']
                assert [sum(evaluation['bands'][band]['per_map_pixels'][i]
                            for band in partition) for i in range(32)] == evaluation['paired']['per_map_pixels']
                assert math.isfinite(evaluation['outside_forward_lightcone_max_logit_difference'])
                assert evaluation['outside_forward_lightcone_max_logit_difference'] <= 1e-6

            for horizon in HORIZONS:
                evaluation = sequence[str(horizon)]
                for band, metric in evaluation['bands'].items():
                    curves.append({
                        'seed': seed, 'variant': variant, 'size': size, 'T': horizon,
                        'band': band, 'mean': metric['mean'],
                        'pooled': metric['pooled_accuracy'], 'pixels': metric['pooled_pixels'],
                        'correct': metric['pooled_correct'], 'eligible_maps': metric['eligible_maps'],
                        'ba_original': evaluation['original']['balanced_accuracy'],
                        'ba_flipped': evaluation['flipped']['balanced_accuracy'],
                    })
        gate_rows.append(gate_row(record))

    assert (ba_aggregates, paired_aggregates, len(curves)) == (144, 1008, 936)
    curves_by_key = {(row['seed'], row['variant'], row['size'], row['T'], row['band']): row
                     for row in curves}
    assert len(curves_by_key) == len(curves)
    effects = []
    for row in curves:
        if row['variant'] != 'memory':
            continue
        for control in ('stream', 'stateless'):
            comparison = curves_by_key[(row['seed'], control, row['size'], row['T'], row['band'])]
            assert row['pixels'] == comparison['pixels']
            assert row['eligible_maps'] == comparison['eligible_maps']
            effects.append({
                'comparison': f"memory-{control}",
                **{field: row[field] for field in
                   ('seed', 'size', 'T', 'band', 'pixels', 'eligible_maps')},
                **{
                    field + '_effect_pp': (
                        None if row[field] is None or comparison[field] is None else
                        100 * (row[field] - comparison[field]))
                    for field in ('mean', 'pooled', 'ba_original', 'ba_flipped')
                },
            })
    assert len(effects) == 624

    curve_fields = ('seed', 'variant', 'size', 'T', 'band', 'mean', 'pooled', 'pixels',
                    'correct', 'eligible_maps', 'ba_original', 'ba_flipped')
    effect_fields = ('comparison', 'seed', 'size', 'T', 'band', 'pixels', 'eligible_maps',
                     'mean_effect_pp', 'pooled_effect_pp', 'ba_original_effect_pp',
                     'ba_flipped_effect_pp')
    curve_csv_rows = verify_csv(run / 'curves.csv', curves,
                                ('seed', 'variant', 'size', 'T', 'band'), curve_fields)
    effect_csv_rows = verify_csv(run / 'paired_effects.csv', effects,
                                 ('comparison', 'seed', 'size', 'T', 'band'), effect_fields)
    assert (curve_csv_rows, effect_csv_rows) == (936, 624)
    aggregate_effects = aggregate['paired_effects']
    assert len(aggregate_effects) == 624
    effects_by_key = {(row['comparison'], row['seed'], row['size'], row['T'], row['band']): row
                      for row in effects}
    assert len(effects_by_key) == len(effects)
    saved_effect_keys = [(row['comparison'], row['seed'], row['size'], row['T'], row['band'])
                         for row in aggregate_effects]
    assert len(set(saved_effect_keys)) == len(saved_effect_keys)
    assert set(saved_effect_keys) == set(effects_by_key)
    for row in aggregate_effects:
        key = (row['comparison'], row['seed'], row['size'], row['T'], row['band'])
        expected = effects_by_key[key]
        for field in effect_fields:
            if isinstance(expected[field], str):
                assert row[field] == expected[field], (key, field)
            else:
                close(row[field], expected[field], ('aggregate paired effect', key, field))

    computed_rows = [gate_row(current_records[(seed, variant)])
                     for seed in SEEDS for variant in VARIANTS]
    assert len(computed_rows) == 12
    aggregate_rows = {(row['seed'], row['variant']): row for row in aggregate['rows']}
    assert set(aggregate_rows) == expected_keys
    for row in computed_rows:
        saved = aggregate_rows[(row['seed'], row['variant'])]
        for field, value in row.items():
            if isinstance(value, float):
                close(saved[field], value, ('aggregate gate row', row['seed'], row['variant'], field))
            else:
                assert saved[field] == value, (row, saved)
    all_reproduced = all(row['reproduced'] for row in historical_checks)
    verdict, counts, anchors = expected_decision(computed_rows, all_reproduced)
    assert aggregate['decision'] == verdict == 'DEVELOPMENT_NO_GO'
    assert aggregate['counts'] == counts
    assert aggregate['control_anchor_checks'] == anchors
    assert aggregate['control_history_reproduction']['all_reproduced'] is True
    assert aggregate['control_history_reproduction']['complete_control_set'] is True
    saved_statuses = {(row['seed'], row['variant'], row['status'], row['completed_updates'])
                      for row in aggregate['statuses']}
    assert saved_statuses == {(seed, variant, 'COMPLETE', UPDATES) for seed, variant in expected_keys}

    assert counts == {
        'stream': {'reach': 1, 'reach_and_hold': 1},
        'memory': {'reach': 1, 'reach_and_hold': 0},
        'stateless': {'reach': 1, 'reach_and_hold': 0},
    }
    trajectories = [rollout_row(current_records[(seed, variant)], 32, horizon)
                    for seed in SEEDS for variant in VARIANTS for horizon in HORIZONS]
    seed4 = [row for row in trajectories if row['seed'] == 4]
    failures = [{'seed': seed, 'variant': variant,
                 'failed_hold_checks': hold_failures(current_records[(seed, variant)])}
                for seed in SEEDS for variant in VARIANTS]

    primary_effects = [row for row in effects
                       if row['size'] == 32 and row['T'] == 64 and row['band'] == 'strict_16_32']
    assert len(primary_effects) == 8
    rollout_summaries = []
    for size, band in ((32, 'strict_16_32'), (64, 'far_gt32')):
        for horizon in HORIZONS:
            for variant in VARIANTS:
                selected = [row for row in curves if row['size'] == size and row['T'] == horizon
                            and row['variant'] == variant and row['band'] == band]
                assert len(selected) == len(SEEDS)
                pixels = sum(row['pixels'] for row in selected)
                correct = sum(row['correct'] for row in selected)
                eligible = sum(row['eligible_maps'] for row in selected)
                rollout_summaries.append({
                    'size': size, 'T': horizon, 'band': band, 'variant': variant,
                    'seed_count': len(selected), 'correct': correct, 'pixels': pixels,
                    'pooled_accuracy': correct / pixels if pixels else None,
                    'eligible_maps': eligible,
                    'seed_values': [{
                        'seed': row['seed'], 'mean': row['mean'], 'pooled': row['pooled'],
                        'correct': row['correct'], 'pixels': row['pixels'],
                    } for row in sorted(selected, key=lambda item: item['seed'])],
                })

    systems = []
    for seed in SEEDS:
        for variant in VARIANTS:
            record = current_records[(seed, variant)]
            systems.append({
                'seed': seed, 'variant': variant,
                'training_seconds': record['training_seconds'],
                'peak_allocated_mib': record['training_peak_allocated_bytes'] / 2**20,
                'peak_increment_mib': record['training_peak_increment_bytes'] / 2**20,
                'gradient_clip_fraction': record['gradient_clip_fraction'],
                'median_update_seconds': float(np.median(record['update_seconds'])),
            })
    systems_by_variant = {}
    for variant in VARIANTS:
        rows = [row for row in systems if row['variant'] == variant]
        systems_by_variant[variant] = {
            'seed_count': len(rows),
            'median_training_seconds': float(np.median([row['training_seconds'] for row in rows])),
            'median_peak_allocated_mib': float(np.median([row['peak_allocated_mib'] for row in rows])),
            'median_gradient_clip_fraction': float(np.median([row['gradient_clip_fraction'] for row in rows])),
            'median_update_seconds': float(np.median([row['median_update_seconds'] for row in rows])),
        }

    validation = {
        'status': 'PASS', 'source_files': 45,
        'source_hashes_current_and_two_snapshots': 45,
        'checkpoint_state_hashes_cpu_only': 12,
        'historical_controls_exact_initial_final_evaluation_data_schedule': 4,
        'historical_control_final_and_full_evaluation_reproduced': 4,
        'training_schedule_evaluation_bindings': True,
        'preflight_bindings_and_source_snapshot': True,
        'BA_means': ba_aggregates,
        'paired_aggregates_and_independent_denominators': paired_aggregates,
        'arm_gate_decisions': len(computed_rows),
        'side_gradient_entry_verified': 8,
        'all_initial_core_parameters_matched': True,
        'full_initial_memory_stateless_parameters_matched': True,
        'curve_csv_rows': curve_csv_rows, 'paired_effect_csv_rows': effect_csv_rows,
        'runner_aggregate_effects': len(aggregate_effects),
        'new_training_or_model_inference': False,
        'checkpoint_check': 'weights_only CPU state_dict SHA against each saved final parameter SHA',
    }
    public_status = {key: value for key, value in status.items() if key != 'pid'}
    public_preflight_status = {key: value for key, value in pf_status.items() if key != 'pid'}
    parameter_hashes = dict(sorted(parameter_hashes.items()))
    summary = {
        'execution': public_status,
        'preflight': public_preflight_status,
        'decision': verdict, 'counts': counts, 'control_anchor_checks': anchors,
        'primary_rows': computed_rows,
        'primary_trajectories': trajectories,
        'seed4_trajectory': seed4,
        'hold_failure_reasons': failures,
        'primary_paired_effects': primary_effects,
        'historical_control_reproduction': historical_checks,
        'parameter_hashes': parameter_hashes,
        'rollout_summaries': rollout_summaries,
        'systems_by_arm': systems, 'systems_by_variant_descriptive': systems_by_variant,
        'validation': validation,
    }

    lines = [
        '# Stationary sidecar: completed development screen', '',
        f'**{verdict}.** All12 arms completed300 updates in '
        f"{status['elapsed_seconds']:.3f} seconds ({status['elapsed_seconds']/60:.2f} minutes).",
        'Four original Streaming controls reproduce historical final parameter hashes and full evaluations exactly.',
        'Reach+hold: stream1/4; memory0/4; stateless0/4. All three arms reach at seed4; only stream holds.', '',
        '## Frozen primary: size32, T64, strict16<d<32', '',
        '| Seed | Variant | BA original / flipped % | Per-map mean % | Pooled correct / pixels (%) | Reach | Hold | Reach+hold |',
        '|---:|---|---:|---:|---:|---|---|---|',
    ]
    for row in computed_rows:
        metric = current_records[(row['seed'], row['variant'])]['evaluation']['32']['64']['bands']['strict_16_32']
        lines.append(
            f"| {row['seed']} | {row['variant']} | {100*row['ba_original']:.2f} / "
            f"{100*row['ba_flipped']:.2f} | {100*row['primary_mean']:.2f} | "
            f"{metric['pooled_correct']}/{metric['pooled_pixels']} ({100*row['primary_pooled']:.2f}%) | "
            f"{row['reach']} | {row['hold']} | {row['reach_and_hold']} |")
    lines += [
        '', 'Reach requires BA>=85% in each source orientation and BOTH paired metrics>=80%.',
        'Hold bounds declines at BOTH T128/T256: BA<=3pp and paired metrics<=5pp relative to T64.',
        'Memory GO also requires>=3/4 reach+hold including seed4 and strictly more successes than both controls.',
        'Hold=True cannot qualify a failed reach endpoint.', '',
        '## Seed4: early gains do not preserve the longer computation', '',
        '| Variant | T | BA original / flipped % | Per-map mean % | Pooled correct / pixels (%) |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in seed4:
        lines.append(f"| {row['variant']} | {row['T']} | {100*row['ba_original']:.2f} / "
                     f"{100*row['ba_flipped']:.2f} | {100*row['mean']:.2f} | "
                     f"{row['correct']}/{row['pixels']} ({100*row['pooled']:.2f}%) |")
    lines += [
        '', 'Memory improves the T64 primary but loses it at T128 (0/2918), while stateless remains perfect at T128 then fails hold at T256.',
        'Thus a higher finite-horizon endpoint does not establish reliable beneficial continued rollout.',
        'Stateless seed2 improves through T128/T256, but its T64 endpoint remains below the frozen threshold.',
        'These are descriptive observations on selected previously inspected seeds; later endpoints cannot replace the frozen gate.',
        '', '## All primary trajectories', '',
        '| Seed | Variant | Pooled T64 % | Pooled T128 % | Pooled T256 % |',
        '|---:|---|---:|---:|---:|',
    ]
    for seed in SEEDS:
        for variant in VARIANTS:
            values = [current_records[(seed, variant)]['evaluation']['32'][str(t)]['bands']['strict_16_32']['pooled_accuracy'] for t in HORIZONS]
            lines.append(f"| {seed} | {variant} | " + ' | '.join(f'{100*v:.2f}' for v in values) + ' |')
    lines += [
        '', '## Far-space summaries', '',
        'Size64 d>32 values pool integer counts over the four seeds for description.',
        'The same32 maps per size are reused;128 map-seed evaluations are not128 independent task maps.',
        '| T | Variant | Correct / pixels | Pooled % |',
        '|---:|---|---:|---:|',
    ]
    for row in rollout_summaries:
        if row['size'] == 64:
            text = 'n/a' if row['pooled_accuracy'] is None else f"{100*row['pooled_accuracy']:.2f}"
            lines.append(f"| {row['T']} | {row['variant']} | {row['correct']}/{row['pixels']} | {text} |")
    lines += [
        '', '## Scope and interpretation', '',
        'Original W24/Z8, F/Q modules, masked streaming/Laplacians, perception clocks and training are preserved.',
        'G67->32->12 writes an additional H12. Bias-free P_F/P_Q feed H* into F/Q.',
        'Memory H*=H+.1G and stateless H*=.1G are the only structural difference between the7989-parameter side arms; stream has5033.',
        'No direct H/LH input columns are disabled in stateless. Both side arms consume current H* immediately.',
        'Neutral P_F=P_Q=0 nests the original W/Z/logits and raw core derivatives. Extra gradients can change global norm clipping and optimizer updates.',
        'All new gradient paths opened by update3. Two-hop macro/K8 radius16 bounds are retained.',
        'This recipe did not improve reproducible reach+hold. It does not reject stationary memory in general: original Z8 is already stationary.',
        'Carry changes temporal accumulation and activation scale together. No identified Jacobian, saturation, overwrite, optimization, or semantic mechanism follows from these trajectories.',
        'Stateless has no extra H carry but recurrent W/Z still retain information.',
        'Four reused initialization seeds are the independent units, conditional on one shared training bank/schedule; no population reliability, p-value, fresh confirmation, or3D claim.',
        '', '## Systems and saved-result verification', '',
        '| Variant | Median training seconds | Median peak allocated MiB | Median clipped % |',
        '|---|---:|---:|---:|',
    ]
    for variant in VARIANTS:
        row = systems_by_variant[variant]
        lines.append(f"| {variant} | {row['median_training_seconds']:.2f} | "
                     f"{row['median_peak_allocated_mib']:.2f} | {100*row['median_gradient_clip_fraction']:.2f} |")
    lines += [
        '', 'Verified45 source bindings in current checkout and both snapshots,12 CPU checkpoint state hashes,4 exact historical controls, banks/schedules/preflight,144 BA means,1008 paired metrics with independently regenerated denominators,936 curve rows,624 contrasts and all frozen gate decisions.',
        'These arithmetic and CPU checkpoint checks do not rerun inference or training. Timing/allocation/clipping/state RMS are descriptive; no matched-FLOP or stability claim.',
        'Read [analysis](analysis.json), [validation](validation.json), [curves](curves.csv) and [paired effects](paired_effects.csv) before the12 [raw records](raw/).',
        'Checkpoints, machine metadata and launch receipts remain local.', '',
    ]


    out.mkdir(parents=True)
    write(out / 'analysis.json', summary)
    write(out / 'validation.json', validation)
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')

    run_inputs = sorted(
        file for file in run.iterdir()
        if file.is_file() and file.suffix.lower() in ('.json', '.csv', '.md'))
    preflight_inputs = sorted(
        file for file in preflight.iterdir()
        if file.is_file() and file.suffix.lower() in ('.json', '.csv', '.md'))
    history_paths = [ROOT / name for name in sorted(historical_hashes)]
    output_hashes = {
        file.name: sha(file) for file in sorted(out.iterdir())
        if file.is_file() and file.name != 'provenance.json'
    }
    provenance = {
        'source_run': run.relative_to(ROOT).as_posix(),
        'source_preflight': preflight.relative_to(ROOT).as_posix(),
        'input_sha256': {
            file.relative_to(ROOT).as_posix(): sha(file)
            for file in run_inputs + preflight_inputs + history_paths
        },
        'source_sha256': manifest['source_sha256'],
        'checkpoint_file_sha256': checkpoint_hashes,
        'parameter_sha256': {
            key: value['final_parameter_sha256'] for key, value in parameter_hashes.items()
        },
        'original_private_manifest_sha256': sha(run / 'manifest.json'),
        'original_private_completion_sha256': sha(run / 'status.json'),
        'original_preflight_manifest_sha256': sha(preflight / 'manifest.json'),
        'original_preflight_completion_sha256': sha(preflight / 'status.json'),
        'analysis_source_sha256': {
            Path(__file__).resolve().relative_to(ROOT).as_posix(): sha(Path(__file__).resolve())
        },
        'output_sha256': output_hashes,
    }
    write(out / 'provenance.json', provenance)
    print(json.dumps({
        'decision': verdict, 'counts': counts, 'validation': validation,
        'analysis_output': out.relative_to(ROOT).as_posix(),
        'analysis_files': sum(path.is_file() for path in out.iterdir()),
    }, indent=2))


if __name__ == '__main__':
    main()
