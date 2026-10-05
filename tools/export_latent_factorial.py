"""Publish and verify saved latent-factorial data; no training or inference."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from export_continuation_interface import read, write, sha, safe, PRIVATE_TEXT

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/latent_factorial_20261005_01'
PUBLIC = ROOT / 'evidence/latent_factorial_20261005'
MANIFEST = ROOT / 'LATENT_FACTORIAL_PUBLICATION_MANIFEST.json'
QUALIFICATION = ROOT / 'analyses/latent_factorial_20261005_check_01.json'
ARMS = ('native', 'factorized', 'native_r2', 'factorized_r2')
SIZES = (32, 64)
HORIZONS = (64, 128, 256)
EXPECTED_ELAPSED_SECONDS = 4361.0470000000005


def runner():
    spec = importlib.util.spec_from_file_location(
        '_published_latent_factorial', ROOT / 'new/latent_factorial/run.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8-sig') as handle:
            return json.load(handle)
    return read(path)


def tensor_bank_hash(bank, r):
    tensors = {}
    for key, value in bank.items():
        if torch.is_tensor(value):
            tensors[key] = value
        else:
            tensors[key] = torch.from_numpy(np.asarray(value))
    return r.C.tensor_hash(tensors)


def verify_sources(manifest):
    assert 'runtime_source_sha256' not in manifest
    source_hashes = manifest['source_sha256']
    assert source_hashes
    for name, digest in source_hashes.items():
        assert sha(ROOT / name) == sha(RUN / 'source' / name) == digest, name
    assert sha(RUN / 'plans.json') == manifest['schedule_plan_sha256']
    assert sha(QUALIFICATION) == manifest['qualification_sha256']
    qualification = read(QUALIFICATION)
    assert qualification['source_sha256'] == source_hashes
    return qualification


def verify_qualification(qualification):
    assert qualification['status'] == 'PASS'
    assert qualification['protocol'] == 'latent_read_sidecar_factorial_v1'
    assert qualification['qualification_scope'] == (
        'Five actual-shape updates each arm; short replay, not full300 equivalence.')
    assert qualification['cpu']['status'] == 'PASS'
    assert qualification['cpu']['failures'] == qualification['cpu']['errors'] == 0
    assert qualification['three_state_trace'] == 'PASS'
    assert set(qualification['cuda']) == set(ARMS)
    for arm in ARMS:
        item = qualification['cuda'][arm]
        assert item['status'] == 'PASS'
        assert item['bitwise_eager_graph_updates'] == 5
        assert item['cadence'] == {
            'backward_calls': 8,
            'interior_detach_boundaries': 7,
            'forward_steps': 64,
            'loss_count': 8,
        }
    # This is the check runner's synthetic fixture, not the 128-arm endpoint.
    fixture = qualification['statistics']
    assert fixture['wins'] == 8 and fixture['losses'] == 0
    assert fixture['qualification_threshold_met']


def checkpoint_bindings(rows, r):
    records = {}
    for row in rows:
        block, arm = row['block'], row['arm']
        folder = RUN / f'block{block:02d}' / arm
        fields = {}
        for label, filename, updates, expected in (
            ('initial', 'initial.pt', 0, row['initial_parameter_sha256']),
            ('final', 'final_u300.pt', 300, row['final_parameter_sha256']),
        ):
            path = folder / filename
            payload = torch.load(path, map_location='cpu', weights_only=False)
            assert payload['arm'] == arm and payload['initialization_seed'] == row['initialization_seed']
            assert payload['completed_updates'] == updates
            assert payload['metadata'] == row['metadata']
            tensor_digest = r.C.tensor_hash(payload['state_dict'])
            assert tensor_digest == expected, (block, arm, label, tensor_digest, expected)
            file_digest = sha(path)
            if label == 'final':
                assert file_digest == row['checkpoint_sha256']
            fields[f'{label}_checkpoint_sha256'] = file_digest
            fields[f'{label}_parameter_sha256'] = tensor_digest
            del payload
        records[f'block{block:02d}/{arm}'] = fields
    assert len(records) == 128
    return records


def check_public_checkpoint_bindings(rows, records):
    assert set(records) == {f'block{row["block"]:02d}/{row["arm"]}' for row in rows}
    for row in rows:
        record = records[f'block{row["block"]:02d}/{row["arm"]}']
        assert record['final_checkpoint_sha256'] == row['checkpoint_sha256']
        assert record['initial_parameter_sha256'] == row['initial_parameter_sha256']
        assert record['final_parameter_sha256'] == row['final_parameter_sha256']
        assert all(len(record[key]) == 64 for key in (
            'initial_checkpoint_sha256', 'final_checkpoint_sha256',
            'initial_parameter_sha256', 'final_parameter_sha256'))


def checks(checkpoint_info):
    r = runner()
    summary = read(PUBLIC / 'summary.json')
    rows = read(PUBLIC / 'perarm.json')
    cfg = read(PUBLIC / 'config.json')
    plans = load_json(PUBLIC / 'plans.json.gz')
    assert summary['status'] == 'COMPLETE'
    assert summary['completed_arms'] == summary['expected_arms'] == 128
    assert math.isclose(summary['elapsed_seconds'], EXPECTED_ELAPSED_SECONDS, rel_tol=0, abs_tol=1e-8)
    assert len(rows) == 128
    check_public_checkpoint_bindings(rows, checkpoint_info)
    assert {(v['block'], v['arm']) for v in rows} == {
        (block, arm) for block in range(32) for arm in ARMS
    }
    aggregate = r.aggregate(rows)
    assert all(summary[key] == value for key, value in aggregate.items())
    assert cfg['protocol'] == r.PROTOCOL
    assert cfg['updates'] == 300 and cfg['credit_horizon'] == 8 and cfg['training_forward_horizon'] == 64
    assert cfg['initialization_seeds'] == list(r.INIT_SEEDS)
    assert cfg['schedule_seeds'] == list(r.SCHEDULE_SEEDS)
    assert cfg['arms'].keys() == set(ARMS)
    assert cfg['arm_metadata_example_seed'] == 97999
    assert cfg['runtime'] == 'cuda_graph_k8'
    assert cfg['finished_utc'] == summary['finished_utc']
    assert cfg['execution_elapsed_seconds'] == summary['elapsed_seconds']

    banks = {}
    for name, filename in (
        ('train', 'train.npz'),
        ('evaluation32', 'evaluation32.npz'),
        ('evaluation64', 'evaluation64.npz'),
    ):
        with np.load(PUBLIC / 'banks' / filename, allow_pickle=False) as values:
            banks[name] = {key: values[key] for key in values.files}
        assert tensor_bank_hash(banks[name], r) == cfg['data_sha256'][name]

    expected_metrics = {}
    traces = curves = curve_updates = per_map_summaries = 0
    full_successes = {arm: 0 for arm in ARMS}
    runtime_counts = {}
    for block in range(32):
        plan = plans[str(block)]
        assert plan['initialization_seed'] == 94001 + block
        assert plan['schedule_seed'] == 95001 + block
        assert plan['arm_order'] == list(ARMS[block % len(ARMS):] + ARMS[:block % len(ARMS)])
        expected_batches = np.random.default_rng(95001 + block).integers(0, 512, (300, 8))
        assert np.array_equal(expected_batches, plan['batch_indices'])

    for row in rows:
        block, arm = row['block'], row['arm']
        plan = plans[str(block)]
        assert row['initialization_seed'] == plan['initialization_seed'] == 94001 + block
        assert row['schedule_seed'] == plan['schedule_seed'] == 95001 + block
        assert arm in ARMS

        curve = load_json(PUBLIC / f'block{block:02d}' / arm / 'training_curve.json.gz')
        assert len(curve) == 300
        assert [v['update'] for v in curve] == list(range(1, 301))
        assert all(
            (v['forward_steps'], v['backward_calls'], v['loss_count'], v['interior_detach_boundaries'])
            == (64, 8, 8, 7)
            and all(math.isfinite(v[k]) for k in (
                'mean_trajectory_loss', 'gradient_norm_before_clip', 'seconds'))
            for v in curve
        )
        assert sum(v['seconds'] for v in curve) == row['training_seconds']
        runtime = read(PUBLIC / f'block{block:02d}' / arm / 'runtime.json')
        assert runtime['runtime'] == row['runtime'] == 'cuda_graph_k8'
        assert runtime['capture_setup_seconds'] == row['capture_setup_seconds']
        runtime_counts[row['runtime']] = runtime_counts.get(row['runtime'], 0) + 1
        assert read(PUBLIC / f'block{block:02d}' / arm / 'systems.json') == row['systems']

        measured = read(PUBLIC / row['evaluation_summary'])
        predicate = r.C.evaluate.__globals__['predicate'](measured)
        assert predicate == measured['phenotype_gate']
        assert r.C.compact(measured) == row['evaluation_compact']
        assert predicate['pass'] == row['phenotype_pass']
        full_successes[arm] += int(predicate['pass'])
        per_map_summaries += 1

        for size in SIZES:
            path = (PUBLIC / row['evaluation_summary']).with_name(f'{arm}_size{size}.npz')
            with np.load(path, allow_pickle=False) as values:
                correct, original, flipped = (
                    values[key] for key in ('correct', 'original_correct', 'flipped_correct'))
                assert correct.shape == original.shape == flipped.shape == (257, 32, size, size)
                assert correct.dtype == original.dtype == flipped.dtype == np.bool_
                assert np.array_equal(correct, original & flipped)
                changed = banks[f'evaluation{size}']['changed'][:, 0].astype(bool)
                distance = banks[f'evaluation{size}']['distance'][:, 0]
                observed = measured['sizes'][str(size)]
                for horizon in HORIZONS:
                    for label, mask in (
                        ('all_changed', changed),
                        ('strict_16_32', changed & (distance > 16) & (distance < 32)),
                    ):
                        pooled = observed['endpoints'][str(horizon)][label]['pooled_coverage']
                        assert pooled['denominator'] == int(mask.sum())
                        assert pooled['numerator'] == int((correct[horizon] & mask).sum())
                before = correct[64] & changed
                retention = observed['transitions']['all_changed']['64_to_256']['pooled']['retention']
                assert retention['denominator'] == int(before.sum())
                assert retention['numerator'] == int((before & correct[256]).sum())
                ever = correct.any(axis=0) & changed
                regressed = (np.maximum.accumulate(correct, axis=0) & ~correct).any(axis=0) & changed
                saved = observed['ever_regressed_over_ever_correct']
                assert saved['denominator'] == int(ever.sum())
                assert saved['numerator'] == int(regressed.sum())
            traces += 1

            compact = row['evaluation_compact']['sizes'][str(size)]
            for horizon in HORIZONS:
                expected_metrics[(block, arm, size, horizon)] = {
                    'initialization_seed': row['initialization_seed'],
                    'schedule_seed': row['schedule_seed'],
                    'full_pass': row['phenotype_pass'],
                    'coverage': compact['coverage'][str(horizon)],
                    'strict_mean': compact['strict_mean'][str(horizon)],
                    'strict_pooled': compact['strict_pooled'][str(horizon)],
                    **{key: compact[key] for key in (
                        'retention64_to256', 'ever_regressed_fraction', 'frontier_effect')},
                }
        curves += 1
        curve_updates += len(curve)

    with (PUBLIC / 'metrics.csv').open(encoding='utf-8', newline='') as handle:
        metrics = list(csv.DictReader(handle))
    assert len(metrics) == 768
    for value in metrics:
        key = (int(value['block']), value['arm'], int(value['size']), int(value['horizon']))
        expected = expected_metrics.pop(key)
        for name, item in expected.items():
            if isinstance(item, bool):
                assert value[name] == str(item)
            elif item is None:
                assert value[name] == ''
            else:
                assert float(value[name]) == item
    assert not expected_metrics
    assert runtime_counts == {'cuda_graph_k8': 128}
    assert traces == 256 and curves == 128 and curve_updates == 38400
    assert per_map_summaries == 128
    assert full_successes == {arm: 0 for arm in ARMS}
    assert summary['primary_verdict'] == 'NO_D_MINUS_A_RELIABILITY_QUALIFICATION'

    qualification = read(PUBLIC / 'runtime_qualification.json')
    verify_qualification(qualification)
    return {
        'status': 'PASS',
        'scope': 'Saved artifacts, hashes, CPU banks and arithmetic only; no inference, training or optimizer updates.',
        'completed_arms': 128,
        'training_blocks': 32,
        'metric_rows': 768,
        'trace_banks': traces,
        'curves': curves,
        'curve_updates_total': curve_updates,
        'paired_trace_identity_exact': True,
        'coverage_retention_regression_counts_recomputed': True,
        'full_predicates_recomputed_from_saved_metrics': True,
        'aggregate_exact_match_from_frozen_runner': True,
        'plans_banks_curves_and_checkpoints_checked': True,
        'runtime_backends': runtime_counts,
        'elapsed_seconds': summary['elapsed_seconds'],
        'primary_verdict': summary['primary_verdict'],
        'formal_full_successes': full_successes,
        'runtime_qualification_scope': qualification['qualification_scope'],
        'synthetic_preflight_statistics_are_not_formal_endpoint': True,
        'checkpoint_tensor_bindings': len(checkpoint_info),
        'checkpoint_files_and_tensor_hashes_verified_locally_during_export': True,
        'checkpoint_hash_records_crosschecked_against_public_perarm_rows': True,
        'matched_frontier_strata_not_recomputed': True,
    }


def build(resume=False):
    assert not MANIFEST.exists() and (not PUBLIC.exists() or resume), (
        'Refusing to overwrite published evidence')
    raw = read(RUN / 'summary.json')
    status = read(RUN / 'status.json')
    m = read(RUN / 'manifest.json')
    qualification = verify_sources(m)
    verify_qualification(qualification)
    assert raw['status'] == status['status'] == 'COMPLETE'
    assert raw['completed_arms'] == raw['expected_arms'] == 128
    assert status['completed_arms'] == 128
    assert math.isclose(raw['elapsed_seconds'], EXPECTED_ELAPSED_SECONDS, rel_tol=0, abs_tol=1e-8)

    PUBLIC.mkdir(parents=True, exist_ok=resume)
    bindings = {}
    files = [RUN / name for name in (
        'summary.json', 'config.json', 'perarm.json', 'metrics.csv', 'RESULTS.md', 'plans.json')]
    files.extend(
        path for path in RUN.rglob('*')
        if path.is_file() and path.relative_to(RUN).parts[0].startswith('block')
        and path.suffix in ('.json', '.csv', '.npz')
    )
    for path in sorted(set(files)):
        rel = path.relative_to(RUN)
        if path.suffix == '.json':
            safe(read(path))
        if path.suffix in ('.csv', '.md'):
            assert not PRIVATE_TEXT.search(path.read_text(encoding='utf-8-sig'))
        compress = (
            rel.name in ('plans.json', 'training_curve.json')
            or path.suffix == '.csv' and rel.parts[0].startswith('block')
        )
        destination = PUBLIC / (str(rel) + '.gz' if compress else rel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            destination.write_bytes(gzip.compress(path.read_bytes(), compresslevel=9, mtime=0))
        else:
            shutil.copyfile(path, destination)
        assert destination.stat().st_size < 50 * 1024**2, destination
        bindings[rel.as_posix()] = {
            'source_path': (RUN / rel).relative_to(ROOT).as_posix(),
            'sha256': sha(path),
            'public_path': destination.relative_to(ROOT).as_posix(),
            'encoding': 'gzip lossless' if compress else 'byte-exact copy',
        }

    cfg = read(RUN / 'config.json')
    cfg.update({key: m[key] for key in (
        'source_sha256', 'data_sha256', 'schedule_plan_sha256', 'backend', 'torch', 'numpy',
        'started_utc', 'git_review_base', 'qualification_sha256',
    )})
    cfg.update({
        'finished_utc': raw['finished_utc'],
        'execution_elapsed_seconds': raw['elapsed_seconds'],
        'elapsed_scope': (
            'Complete formal 128-arm run only, from run start to completion; excludes the '
            'five-update-per-arm preflight and preparation before run start.'
        ),
        'arm_metadata_example_seed': 97999,
        'checkpoint_delivery': (
            'Initial/final parameter hashes and checkpoint hashes are retained in perarm.json '
            'and the publication manifest; checkpoint files stay local.'
        ),
    })
    safe(cfg)
    write(PUBLIC / 'config.json', cfg)

    r = runner()
    (PUBLIC / 'banks').mkdir(exist_ok=resume)
    for name, size, count, seed, filename in (
        ('train', 32, 512, 10002, 'train.npz'),
        ('evaluation32', 32, 32, 99332, 'evaluation32.npz'),
        ('evaluation64', 64, 32, 99364, 'evaluation64.npz'),
    ):
        bank = r.C.region_bank(size, count, seed)
        assert r.C.tensor_hash(bank) == m['data_sha256'][name]
        np.savez_compressed(
            PUBLIC / 'banks' / filename,
            **{key: np.asarray(value) for key, value in bank.items()},
        )

    safe(qualification)
    shutil.copyfile(QUALIFICATION, PUBLIC / 'runtime_qualification.json')
    bindings['runtime_qualification.json'] = {
        'source_path': QUALIFICATION.relative_to(ROOT).as_posix(),
        'sha256': sha(QUALIFICATION),
        'public_path': (PUBLIC / 'runtime_qualification.json').relative_to(ROOT).as_posix(),
        'encoding': 'byte-exact copy',
    }

    note = '''

## Publication and timing

This package retains all 128 CUDA Graph K8 arms from 32 paired
initialization/schedule blocks: 128 complete 300-update curves, 256 paired
source-flip Boolean trace banks covering every integer step from 0 through 256
at sizes 32 and 64, all per-map evaluation summaries, runtime and systems
records, matched-frontier CSVs, the exact schedules, and 768 aggregate metric
rows. Curves, plans and per-block frontier CSVs use lossless gzip. Boolean NPZ
files remain byte-exact, and numeric precision is unchanged. The three
publication banks were regenerated on CPU from their frozen seeds and checked
against the recorded tensor hashes.

All 128 formal arms used the CUDA Graph K8 backend. The recorded 4361.047
seconds covers the completed 128-arm formal run only; it excludes the
five-update-per-arm preflight and preparation before the run started. The
preflight checked the actual batch-8, size-32 shape with five bitwise
eager-versus-graph updates per arm. Its qualification scope is a short replay,
not full 300-update equivalence. Its statistics field is the synthetic
eight-win fixture used to test the frozen threshold calculation, not the
formal efficacy endpoint. The formal endpoint is 0 Full successes in 32
blocks for every arm, with D-A wins=0, losses=0, net=0/32 and exact two-sided
p=1. This does not qualify the tested factorized-read/R2 recipe for reliability
on the fixed cohort; it does not reject all sidecars, latents or NCA.

Checkpoints, source snapshots, machine manifests, status/PID records, launch
receipts and logs stay local. Initial/final parameter tensor hashes and final
checkpoint file hashes are retained. Publication uses saved results and CPU
bank regeneration only; it performs no model inference, training or optimizer
updates.
'''
    source_results = (RUN / 'RESULTS.md').read_text(encoding='utf-8-sig')
    (PUBLIC / 'RESULTS.md').write_text(source_results + note, encoding='utf-8')
    assert not PRIVATE_TEXT.search((PUBLIC / 'RESULTS.md').read_text(encoding='utf-8'))

    (PUBLIC / 'REPRODUCTION.md').write_text('''# Latent factorial saved evidence

Start with RESULTS.md, summary.json, validation.json and config.json. Continue
with perarm.json and metrics.csv. The per-arm evaluation summaries contain the
Full components and per-map metrics. NPZ traces, training curves, runtime and
systems data, and frontier CSVs are the detailed records.

Saved-data verification from the repository root:

    python -X utf8 -B tools/export_latent_factorial.py --verify-only

The verifier needs NumPy and PyTorch and runs on CPU. It verifies source,
checkpoint and tensor hashes; regenerated banks; all 32 schedules; all 300
update clocks for each arm; exact pairing of all 256 Boolean trace banks;
coverage, retention and regression counts; the unchanged Full predicate;
all 768 metric rows; and the saved aggregate against the frozen runner. It
does not rerun logits or recompute matched-frontier strata. Read validation.json
for the exact verification scope.

For a new experiment, follow new/latent_factorial/PROTOCOL.md. The frozen
commands are:

    python -X utf8 -B new/latent_factorial/run.py --check --out analyses/NEW_FACTORIAL_CHECK.json
    pwsh -File tools/launch_latent_factorial.ps1 -RunName NEW_FACTORIAL_RUN -Qualification analyses/NEW_FACTORIAL_CHECK.json

A fresh run requires the historical PyTorch 2.5.1 environment and a compatible
CUDA device. The independent statistical unit is the paired training block
(32), not pixels, maps, rollout times or timing repetitions.
''', encoding='utf-8')

    for name in ('config.json', 'RESULTS.md'):
        bindings[name]['encoding'] = 'public provenance augmentation; raw source hash retained'
    checkpoint_info = checkpoint_bindings(read(PUBLIC / 'perarm.json'), r)
    v = checks(checkpoint_info)
    safe(v)
    write(PUBLIC / 'validation.json', v)
    publication = {
        'protocol': 'latent_factorial_saved_data_publication_v1',
        'run_id': RUN.name,
        'review_base': m['git_review_base'],
        'source_sha256': m['source_sha256'],
        'qualification_sha256': m['qualification_sha256'],
        'exporter_sha256': sha(__file__),
        'original_manifest_sha256': sha(RUN / 'manifest.json'),
        'original_status_sha256': sha(RUN / 'status.json'),
        'schedule_plan_sha256': m['schedule_plan_sha256'],
        'data_sha256': m['data_sha256'],
        'checkpoint_tensor_bindings': checkpoint_info,
        'raw_artifact_bindings': bindings,
        'published_sha256': {
            path.relative_to(ROOT).as_posix(): sha(path)
            for path in PUBLIC.rglob('*') if path.is_file()
        },
        'derived_only': [
            'CPU-regenerated train/evaluation banks exactly matching runtime tensor hashes',
            'saved-data validation counts and frozen-runner aggregate recomputation',
        ],
        'excluded': [
            'initial and final checkpoint files',
            'source snapshots',
            'machine manifests and status/PID records',
            'launch receipts and logs',
        ],
    }
    safe(publication)
    write(MANIFEST, publication)
    print(json.dumps({
        'status': 'PASS',
        'files': len(publication['published_sha256']),
        'MiB': round(sum(path.stat().st_size for path in PUBLIC.rglob('*') if path.is_file()) / 1024**2, 2),
        **v,
    }), flush=True)


def verify():
    m = read(MANIFEST)
    safe(m)
    assert sha(Path(__file__)) == m['exporter_sha256']
    for rel, digest in m['source_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    assert 'runtime_source_sha256' not in m

    actual = {
        path.relative_to(ROOT).as_posix(): sha(path)
        for path in PUBLIC.rglob('*') if path.is_file()
    }
    assert actual == m['published_sha256']
    qualification_path = PUBLIC / 'runtime_qualification.json'
    assert sha(qualification_path) == m['qualification_sha256']
    qualification = read(qualification_path)
    assert qualification['source_sha256'] == m['source_sha256']
    assert m['schedule_plan_sha256'] == hashlib.sha256(
        gzip.decompress((PUBLIC / 'plans.json.gz').read_bytes())).hexdigest()
    public_cfg = read(PUBLIC / 'config.json')
    assert public_cfg['source_sha256'] == m['source_sha256']
    assert public_cfg['data_sha256'] == m['data_sha256']
    assert public_cfg['schedule_plan_sha256'] == m['schedule_plan_sha256']
    assert public_cfg['qualification_sha256'] == m['qualification_sha256']
    for rel, record in m['raw_artifact_bindings'].items():
        public = ROOT / record['public_path']
        if record['encoding'] == 'gzip lossless':
            assert hashlib.sha256(gzip.decompress(public.read_bytes())).hexdigest() == record['sha256'], rel
        elif record['encoding'] == 'byte-exact copy':
            assert sha(public) == record['sha256'], rel
        else:
            assert record['encoding'] == 'public provenance augmentation; raw source hash retained'
            assert public.is_file()
    for path in PUBLIC.rglob('*.json'):
        safe(read(path))
    assert checks(m['checkpoint_tensor_bindings']) == read(PUBLIC / 'validation.json')
    print(json.dumps({
        'status': 'PASS',
        'files': len(actual),
        'arms': 128,
        'trace_banks': 256,
        'metric_rows': 768,
        'primary_verdict': read(PUBLIC / 'summary.json')['primary_verdict'],
    }), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--resume-export', action='store_true',
                        help='Resume an unpublished export whose manifest has not been written.')
    args = parser.parse_args()
    assert not (args.verify_only and args.resume_export)
    verify() if args.verify_only else build(args.resume_export)
