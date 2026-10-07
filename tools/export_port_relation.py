"""Publish/verify the completed port-relation run from saved records, without inference."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from export_continuous_coverage import (
    MAX_FILE_BYTES, bank_tensor_hash, compare_trace_metrics, decode_trace,
    joint_pass, load_bank, read, safe, sha, trace_metrics, write,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/port_relation_20261007_02'
PUBLIC = ROOT / 'evidence/port_relation_20261007_02'
MANIFEST = ROOT / 'PORT_RELATION_PUBLICATION_MANIFEST.json'
QUAL = ROOT / 'analyses/port_relation_qualification_20261007_03.json'

REPRODUCTION = '''# Completed port-relation screen: reproduction and evidence map

Read RESULTS.md, summary.json and final_metrics.csv first. The complete run has
24 formal trajectories,312 stages and624 complete Boolean banks, with no
NUMERICAL_FAILURE stage. No intermediate checkpoint replaces fixed u300.

From the repository root, a CPU saved-data check needs NumPy only:

```powershell
python -X utf8 -B tools/export_port_relation.py --verify-only
```

It verifies source/archive/file hashes, bank tensor hashes, schedule seeds,
completion-marker bindings, all312 evaluator-to-metric rows, the aggregate,
all624 packed shapes, all48 formal Boolean conjunctions/reach/retention gates,
and the24 curves'256 forward/32 backward/one optimizer cadence. Old Full is
retained from the saved evaluator; inference and optimizer updates are zero.

The raw/block00.zip through raw/block07.zip archives contain original bytes
with run-relative paths: all per-stage summaries, relation-activity records,
size32/64 packed traces, full-endpoint per-map CSVs, training curves and atomic
completion markers. Dense/perarm/plans JSON is losslessly gzipped. Numeric
banks are included. Checkpoint/optimizer contents, private launch metadata and
transient logs stay local; checkpoint hashes remain in records and markers.
PORT_RELATION_PUBLICATION_MANIFEST.json at the repository root binds archive
entries, sources, qualification, banks, plans and the recovery bridge.

For a NEW independent reproduction, install the repository requirements with
CUDA-enabled PyTorch2.5.1. Use unique output names and run from the repository
root (qualification below performs a small CUDA smoke, not the formal run):

```powershell
python -X utf8 -B new/port_relation/check_cells.py
python -X utf8 -B new/port_relation/check_activity.py
python -X utf8 -B new/port_relation/reporting.py --self-test
python -X utf8 -u -B new/port_relation/run.py --check --out analyses/port_relation_fresh_check.json
python -X utf8 -u -B new/port_relation/run.py --out runs/port_relation_fresh --qualification analyses/port_relation_fresh_check.json
```

Fresh execution generates the same bank seeds and300-update plans; no private
parent/checkpoint is needed. Windows independent execution can replace the last
line with `pwsh -NoProfile -File tools/launch_port_relation.ps1 -RunName port_relation_fresh -Qualification analyses/port_relation_fresh_check.json`.
The Windows helper requires an interactive scheduled-task session and available
CUDA/disk headroom. These instructions were source-inspected; a second formal
experiment was not launched for this publication. The completed experiment's
actual-shape qualification and saved-data checks are published separately.

The preserved protocol-header name port_relation_conditioned_v1 is an alias
for runtime ID port_relation_v1_full_writer_native_k8. The protocol's phrase
"backpropagate once" summarizes accumulation; the executed implementation
backpropagates32 K8 endpoint losses divided by32 and takes ONE optimizer step.
Model equations, endpoint, seeds and all256 physical forward steps are unchanged.

The original parent stopped after2 trajectories/33 stages on diagnostic float32
squaring overflow after finite T256 state/logit checks. The child inherited those
33 committed stages, restored conditioned u150 with Adam/RNG and replayed the
unsaved prefix. Qualification checks identical saved u175/u150 report dictionaries
and packed traces, including loading live parameters into cached CUDA Graphs.
This is not an uninterrupted bitwise claim; backend nondeterminism is retained.
Parent interruption and zero-training regression are not scientific endpoints.
Qualification timings are single engineering observations, not repeated systems
benchmarks. The resumed process lasted4799.797seconds (~80minutes), with original
partial process time recorded separately.
'''


def write_reproduction():
    (PUBLIC / 'REPRODUCTION.md').write_text(REPRODUCTION, encoding='utf-8')


def module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def gzread(relative):
    return json.loads(gzip.decompress((PUBLIC / relative).read_bytes()))


def verify(publication):
    safe(publication)
    for rel, expected in publication['published_sha256'].items():
        assert sha(ROOT / rel) == expected, rel
        assert (ROOT / rel).stat().st_size < MAX_FILE_BYTES, rel
    for rel, expected in publication['source_sha256'].items():
        assert sha(ROOT / rel) == expected, rel
    for rel, expected in publication['verification_source_sha256'].items():
        assert sha(ROOT / rel) == expected, rel
    rows, final = gzread('dense.json.gz'), gzread('perarm.json.gz')
    assert len(rows) == 312 and len(final) == 24
    assert len({(r['block'], r['arm'], r['update']) for r in rows}) == 312
    assert final == [r for r in rows if r['update'] == 300]
    reporter = module('new/port_relation/reporting.py', '_port_public_report')
    metrics = module('new/continuous_coverage/metrics.py', '_port_public_metrics')
    published = read(PUBLIC / 'summary.json')
    aggregate = reporter.aggregate(final, rows, expected_blocks=8)
    assert all(published[k] == v for k, v in aggregate.items())
    assert aggregate['status'] == 'COMPLETE'
    assert aggregate['overall_verdict'] == 'NO_CONDITIONED_RELIABILITY_QUALIFICATION'
    banks = {s: load_bank(PUBLIC / f'banks/evaluation{s}.npz') for s in (32, 64)}
    for label in ('train', 'evaluation32', 'evaluation64'):
        assert bank_tensor_hash(load_bank(PUBLIC / f'banks/{label}.npz')) == publication['data_sha256'][label]
    plans = gzread('plans.json.gz')
    for block in range(8):
        plan = plans[str(block)]
        assert plan['initialization_seed'] == 130001 + block
        assert plan['schedule_seed'] == 131001 + block
        assert np.asarray(plan['batch_indices']).shape == (300, 8)
    checked_traces = failures = checked_entries = 0
    for block in range(8):
        archive = PUBLIC / f'raw/block{block:02d}.zip'
        with zipfile.ZipFile(archive) as z:
            entries = publication['archive_entries'][archive.relative_to(ROOT).as_posix()]
            assert set(z.namelist()) == set(entries)
            for name, expected in entries.items():
                raw = z.read(name)
                assert hashlib.sha256(raw).hexdigest() == expected['sha256'], name
                if name.endswith('.json'):
                    safe(json.loads(raw))
                checked_entries += 1
            for row in (r for r in rows if r['block'] == block):
                stage = Path(row['evaluation_summary']).parent.as_posix()
                summary = json.loads(z.read(row['evaluation_summary']))
                assert metrics.compact_pair_metrics(summary) == row['metrics']
                assert metrics.joint_readiness(summary) == row['joint']
                marker = json.loads(z.read(row['checkpoint'] + '.complete.json'))
                assert marker['stage_complete'] and marker['record_sha256'] == digest(row)
                assert marker['checkpoint_sha256'] == row['checkpoint_sha256']
                for name, expected in marker['evaluation_sha256'].items():
                    assert entries[f'{stage}/{name}']['sha256'] == expected
                if row.get('numerical_failure'):
                    failures += 1
                    assert row['trace_complete'] is False and row['joint']['pass'] is False
                    continue
                assert row['trace_complete'] is True
                for size in (32, 64):
                    name = f'{stage}/size{size}_traces.npz'
                    with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as packed:
                        for field in ('correct', 'original_correct', 'flipped_correct'):
                            assert tuple(packed[field + '_shape']) == (257, 32, size, size)
                            assert packed[field + '_packed'].dtype == np.uint8
                            assert packed[field + '_packed'].size == (257 * 32 * size * size + 7) // 8
                    checked_traces += 1
                    if row['update'] == 300:
                        # decode_trace accepts a file-like object through NumPy.
                        decoded = decode_trace(io.BytesIO(z.read(name)), size)
                        assert np.array_equal(decoded['correct'], decoded['original_correct'] & decoded['flipped_correct'])
                        actual = trace_metrics(decoded['correct'], banks[size])
                        compare_trace_metrics(actual, row['metrics']['sizes'][str(size)], name)
                        if size == 32:
                            assert joint_pass(actual) == row['joint']['pass']
                curve = json.loads(z.read(f"block{block:02d}/{row['arm']}/training_curve.json"))
                if 'training_curve_prefix_sha256' in marker:
                    assert digest(curve[:row['update']]) == marker['training_curve_prefix_sha256']
            for arm in ('current', 'constant', 'conditioned'):
                curve = json.loads(z.read(f'block{block:02d}/{arm}/training_curve.json'))
                assert [r['update'] for r in curve] == list(range(1, 301))
                assert all(r['forward_steps'] == 256 and r['backward_calls'] == 32 and
                           r['optimizer_steps'] == 1 and r['credit_horizon'] == 8 for r in curve)
    with (PUBLIC / 'metrics.csv').open(encoding='utf-8') as handle:
        assert len(list(csv.DictReader(handle))) == 624
    result = {'status': 'PASS', 'saved_data_only': True, 'inference_or_optimizer_updates': 0,
              'formal_trajectories': 24, 'stage_records': 312, 'size_metric_rows': 624,
              'complete_packed_trace_banks': checked_traces, 'numerical_failure_stages': failures,
              'archive_entries_hash_checked': checked_entries, 'u300_trace_metrics_recomputed': 48,
              'limits': ['Old Full retained from saved evaluator, not rerun.',
                         'No inference or training; CUDA recovery need not be bitwise uninterrupted.']}
    return result


def build():
    assert not PUBLIC.exists() and not MANIFEST.exists(), 'Never overwrite a publication snapshot'
    original, status = read(RUN / 'manifest.json'), read(RUN / 'status.json')
    assert status['status'] == 'COMPLETE' and status['completed_arms'] == 24
    assert sha(QUAL) == original['qualification_sha256']
    for rel, expected in original['source_sha256'].items():
        assert sha(RUN / 'source' / rel) == sha(ROOT / rel) == expected, rel
    PUBLIC.mkdir(parents=True)
    write_reproduction()
    bindings = {}
    for name in ('config.json', 'summary.json', 'final_metrics.csv', 'metrics.csv', 'RESULTS.md',
                 'dense.json', 'perarm.json', 'plans.json'):
        source = RUN / name
        if name.endswith('.json'):
            safe(read(source))
        destination = PUBLIC / (name + '.gz' if name in ('dense.json', 'perarm.json', 'plans.json') else name)
        raw = source.read_bytes()
        destination.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0) if destination.suffix == '.gz' else raw)
        bindings[name] = {'sha256': sha(source), 'public_path': destination.relative_to(ROOT).as_posix(),
                          'encoding': 'gzip lossless' if destination.suffix == '.gz' else 'byte-exact copy'}
    for source in (RUN / 'banks').glob('*.npz'):
        target = PUBLIC / 'banks' / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        bindings[source.relative_to(RUN).as_posix()] = {
            'sha256': sha(source), 'public_path': target.relative_to(ROOT).as_posix(), 'encoding': 'byte-exact copy'}
    shutil.copyfile(QUAL, PUBLIC / 'runtime_qualification.json')
    parent_status = read(RUN.parent / 'port_relation_20261007_01/status.json')
    interruption = {k: parent_status[k] for k in ('status', 'completed_arms', 'dense_records', 'error', 'elapsed_seconds', 'finished_utc')}
    interruption.update(aggregation='INCOMPLETE', interpretation='Diagnostic float32 square overflow after finite state/logit checks; not proof of model-state inf.',
                        repair='Float64 observer arithmetic; cached evaluation and dense Boolean reductions separately qualified.',
                        inherited_committed_stages=original['inherited_committed_stages'], restored_checkpoint='block00/conditioned/u150',
                        backend_deterministic=False, bitwise_uninterrupted_equivalence_guaranteed=False)
    write(PUBLIC / 'interruption_and_recovery.json', interruption)
    archives = {}
    (PUBLIC / 'raw').mkdir()
    for block in range(8):
        dest = PUBLIC / f'raw/block{block:02d}.zip'
        entries = {}
        with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for source in sorted((RUN / f'block{block:02d}').rglob('*')):
                if not source.is_file() or source.suffix not in ('.json', '.csv', '.npz'):
                    continue
                rel = source.relative_to(RUN).as_posix()
                if source.suffix == '.json':
                    safe(read(source))
                z.write(source, rel)
                entries[rel] = {'sha256': sha(source), 'raw_bytes': source.stat().st_size}
        assert dest.stat().st_size < MAX_FILE_BYTES
        archives[dest.relative_to(ROOT).as_posix()] = entries
    publication = {'schema': 'port-relation-complete-publication-v1', 'run_id': RUN.name,
        'publication_status': 'COMPLETE', 'experiment_protocol': original['protocol'],
        'protocol_header_alias': 'port_relation_conditioned_v1', 'git_review_base': 'd3a95be5230b48532bad51370c6d1d021e4e9775',
        'source_sha256': original['source_sha256'], 'data_sha256': original['data_sha256'],
        'schedule_plan_sha256': original['schedule_plan_sha256'], 'qualification_sha256': original['qualification_sha256'],
        'recovery_binding': original['recovery_binding'], 'raw_artifact_bindings': bindings,
        'archive_entries': archives,
        'verification_source_sha256': {r: sha(ROOT / r) for r in ('tools/export_port_relation.py', 'tools/export_continuous_coverage.py')},
        'excluded': ['model/optimizer checkpoints (hashes and completion markers retained)',
                     'private machine manifests, receipts, PIDs and transient logs', 'duplicate source snapshots'],
        'published_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in PUBLIC.rglob('*') if p.is_file()}}
    validation = verify(publication)
    write(PUBLIC / 'validation.json', validation)
    publication['published_sha256']['evidence/port_relation_20261007_02/validation.json'] = sha(PUBLIC / 'validation.json')
    publication['published_bytes'] = sum((ROOT / p).stat().st_size for p in publication['published_sha256'])
    write(MANIFEST, publication)
    print(json.dumps({**validation, 'published_MiB': publication['published_bytes'] / 2**20}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        result = verify(read(MANIFEST))
        assert result == read(PUBLIC / 'validation.json')
        print(json.dumps(result))
    else:
        build()
