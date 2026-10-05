"""Publish/verify saved native-width results; no training or model inference."""
import argparse
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from export_continuation_interface import read, write, sha, safe, PRIVATE_TEXT

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/latent_width_20261005_accelerated_01'
PUBLIC = ROOT / 'evidence/latent_width_20261005'
MANIFEST = ROOT / 'LATENT_WIDTH_PUBLICATION_MANIFEST.json'
QUALIFICATION = ROOT / 'analyses/latent_graph_qualification_20261005_01.json'
PREFLIGHT = ROOT / 'analyses/latent_width_check_20261005_02.json'


def runner():
    spec = importlib.util.spec_from_file_location('_published_latent_width', ROOT / 'new/latent_width/run.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8-sig') as handle:
            return json.load(handle)
    return read(path)


def bank_hash(bank, r):
    return r.C.tensor_hash({k: torch.from_numpy(v) for k, v in bank.items()})


def checks():
    r = runner()
    summary, rows = read(PUBLIC / 'summary.json'), read(PUBLIC / 'perarm.json')
    cfg, plans = read(PUBLIC / 'config.json'), load_json(PUBLIC / 'plans.json.gz')
    assert summary['status'] == 'COMPLETE' and summary['completed_arms'] == summary['expected_arms'] == 96
    assert len(rows) == 96 and {(v['block'], v['arm']) for v in rows} == {(b, a) for b in range(16) for a in r.ARM_NAMES}
    aggregate = r.aggregate(rows)
    assert all(summary[key] == value for key, value in aggregate.items())
    assert cfg['updates'] == 300 and cfg['credit_horizon'] == 8 and cfg['training_forward_horizon'] == 64
    assert cfg['initialization_seeds'] == list(r.INIT_SEEDS) and cfg['schedule_seeds'] == list(r.SCHEDULE_SEEDS)
    banks = {}
    for size in (32, 64):
        with np.load(PUBLIC / 'banks' / f'evaluation{size}.npz', allow_pickle=False) as values:
            banks[size] = {k: values[k] for k in values.files}
        assert bank_hash(banks[size], r) == cfg['data_sha256'][f'evaluation{size}']
    with np.load(PUBLIC / 'banks/train.npz', allow_pickle=False) as values:
        assert bank_hash({k: values[k] for k in values.files}, r) == cfg['data_sha256']['train']
    expected_metrics = {}
    traces = 0
    for row in rows:
        block, arm = row['block'], row['arm']
        plan = plans[str(block)]
        assert plan['initialization_seed'] == row['initialization_seed'] == 91001 + block
        assert plan['schedule_seed'] == row['schedule_seed'] == 92001 + block
        assert plan['arm_order'] == list(r.arm_order(block))
        expected = np.random.default_rng(92001 + block).integers(0, 512, (300, 8))
        assert np.array_equal(expected, plan['batch_indices'])
        curve = load_json(PUBLIC / (row['training_curve'] + '.gz'))
        assert len(curve) == 300 and [v['update'] for v in curve] == list(range(1, 301))
        for v in curve:
            assert (v['forward_steps'], v['backward_calls'], v['loss_count'], v['interior_detach_boundaries']) == (64, 8, 8, 7)
            assert all(np.isfinite(v[k]) for k in ('mean_trajectory_loss', 'gradient_norm_before_clip', 'seconds'))
        measured = read(PUBLIC / row['evaluation_summary'])
        predicate = r.C.evaluate.__globals__['predicate'](measured)
        assert predicate == measured['phenotype_gate']
        assert r.C.compact(measured) == row['evaluation_compact']
        assert predicate['pass'] == row['phenotype_pass']
        assert read(PUBLIC / f'block{block:02d}/{arm}/systems.json') == row['systems']
        for size in (32, 64):
            path = (PUBLIC / row['evaluation_summary']).with_name(f'{arm}_size{size}.npz')
            with np.load(path, allow_pickle=False) as values:
                c, original, flipped = (values[k] for k in ('correct', 'original_correct', 'flipped_correct'))
            assert c.shape == original.shape == flipped.shape == (257, 32, size, size)
            assert c.dtype == original.dtype == flipped.dtype == np.bool_
            assert np.array_equal(c, original & flipped)
            changed = banks[size]['changed'][:, 0].astype(bool)
            distance = banks[size]['distance'][:, 0]
            observed = measured['sizes'][str(size)]
            for t in (64, 128, 256):
                for label, mask in [('all_changed', changed), ('strict_16_32', changed & (distance > 16) & (distance < 32))]:
                    count = int(mask.sum())
                    saved = observed['endpoints'][str(t)][label]['pooled_coverage']
                    assert saved['denominator'] == count and saved['numerator'] == int((c[t] & mask).sum())
            before = c[64] & changed
            retention = observed['transitions']['all_changed']['64_to_256']['pooled']['retention']
            assert retention['denominator'] == int(before.sum()) and retention['numerator'] == int((before & c[256]).sum())
            ever = c.any(0) & changed
            regressed = (np.maximum.accumulate(c, axis=0) & ~c).any(0) & changed
            saved = observed['ever_regressed_over_ever_correct']
            assert saved['denominator'] == int(ever.sum()) and saved['numerator'] == int(regressed.sum())
            traces += 1
            del c, original, flipped
            compact = row['evaluation_compact']['sizes'][str(size)]
            for t in ('64', '128', '256'):
                expected_metrics[(block, arm, size, int(t))] = {
                    'initialization_seed': row['initialization_seed'], 'schedule_seed': row['schedule_seed'],
                    'full_pass': row['phenotype_pass'], 'coverage': compact['coverage'][t],
                    'strict_mean': compact['strict_mean'][t], 'strict_pooled': compact['strict_pooled'][t],
                    **{k: compact[k] for k in ('retention64_to256', 'ever_regressed_fraction', 'frontier_effect')}}
    with (PUBLIC / 'metrics.csv').open(encoding='utf-8', newline='') as handle:
        metrics = list(csv.DictReader(handle))
    assert len(metrics) == 576
    for v in metrics:
        key = (int(v['block']), v['arm'], int(v['size']), int(v['horizon']))
        expected = expected_metrics.pop(key)
        for k, x in expected.items():
            if isinstance(x, bool): assert v[k] == str(x)
            elif x is None: assert v[k] == ''
            else: assert float(v[k]) == x
    assert not expected_metrics
    backends = {b: sum(v['execution_backend'] == b for v in rows) for b in ('eager_imported', 'cuda_graph_k8')}
    assert backends == {'eager_imported': 13, 'cuda_graph_k8': 83}
    q = read(PUBLIC / 'runtime_qualification.json')
    assert q['status'] == 'PASS' and q['exact300_all_arms'] and len(q['arms']) == 6
    for v in q['arms']:
        assert v['status'] == 'PASS' and v['optimizer_step_clocks_all_300']
        assert v['replay_final_parameter_sha256'] == v['expected_final_parameter_sha256']
        assert v['replay_optimizer_state_sha256'] == v['expected_optimizer_state_sha256']
    return {'status': 'PASS', 'scope': 'Saved data, arithmetic and hashes; no model inference or optimizer updates.',
            'completed_arms': 96, 'training_blocks': 16, 'metric_rows': 576, 'trace_banks': traces,
            'curves': 96, 'curve_updates_total': 28800, 'paired_traces_exact': True,
            'coverage_retention_regression_counts_recomputed': True, 'full_predicates_recomputed_from_saved_metrics': True,
            'paired_statistics_recomputed': True, 'plans_banks_and_curves_checked': True,
            'execution_backends': backends, 'primary_verdict': summary['primary_verdict'],
            'runtime_exact300_all6': True,
            'not_recomputed': ['model logits or balanced accuracies', 'matched-frontier strata from scratch', 'new training or CUDA equivalence replay']}


def build(resume=False):
    assert not MANIFEST.exists() and (not PUBLIC.exists() or resume), 'Refusing to overwrite published evidence'
    raw, status, m = (read(RUN / name) for name in ('summary.json', 'status.json', 'manifest.json'))
    assert raw['status'] == status['status'] == 'COMPLETE' and raw['completed_arms'] == 96
    sources = {**m['source_sha256'], **m['runtime_source_sha256']}
    for group, directory in ((m['source_sha256'], 'source'), (m['runtime_source_sha256'], 'runtime_source')):
        for name, digest in group.items():
            assert sha(ROOT / name) == sha(RUN / directory / name) == digest, name
    assert sha(RUN / 'plans.json') == m['schedule_plan_sha256']
    PUBLIC.mkdir(parents=True, exist_ok=resume)
    bindings = {}
    files = [RUN / name for name in ('summary.json', 'config.json', 'perarm.json', 'metrics.csv', 'RESULTS.md', 'plans.json')]
    files.extend(p for p in RUN.glob('block*/*/**/*') if p.is_file() and p.suffix in ('.json', '.csv', '.npz'))
    for path in sorted(set(files)):
        rel = path.relative_to(RUN)
        if path.suffix == '.json': safe(read(path))
        if path.suffix in ('.csv', '.md'): assert not PRIVATE_TEXT.search(path.read_text(encoding='utf-8-sig'))
        compress = rel.name in ('plans.json', 'training_curve.json') or path.suffix == '.csv' and rel.parts[0].startswith('block')
        destination = PUBLIC / (str(rel) + '.gz' if compress else rel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if compress: destination.write_bytes(gzip.compress(path.read_bytes(), compresslevel=9, mtime=0))
        else: shutil.copyfile(path, destination)
        assert destination.stat().st_size < 50 * 1024**2, destination
        bindings[rel.as_posix()] = {'sha256': sha(path), 'public_path': destination.relative_to(ROOT).as_posix(),
                                   'encoding': 'gzip byte-exact' if compress else 'byte-exact copy'}
    cfg = read(RUN / 'config.json')
    cfg.update({k: m[k] for k in ('source_sha256', 'runtime_source_sha256', 'data_sha256', 'schedule_plan_sha256',
                                 'backend', 'torch', 'numpy', 'started_utc', 'execution_amendment')})
    cfg.update({'finished_utc': raw['finished_utc'], 'accelerated_phase_elapsed_seconds': raw['elapsed_seconds'],
                'elapsed_scope': 'New accelerated worker only; excludes original work, qualification and handover preparation.',
                'imported_completed_arms': 13, 'new_completed_arms': 83,
                'checkpoint_delivery': 'Initial/final hashes retained in perarm.json; checkpoints stay local.'})
    safe(cfg); write(PUBLIC / 'config.json', cfg)
    r = runner()
    (PUBLIC / 'banks').mkdir(exist_ok=resume)
    for name, size, count, seed in [('train', 32, 512, 10002), ('evaluation32', 32, 32, 99332), ('evaluation64', 64, 32, 99364)]:
        bank = r.C.region_bank(size, count, seed)
        assert r.C.tensor_hash(bank) == m['data_sha256'][name]
        np.savez_compressed(PUBLIC / 'banks' / f'{name}.npz', **{k: np.asarray(v) for k, v in bank.items()})
    for src, name in ((QUALIFICATION, 'runtime_qualification.json'), (PREFLIGHT, 'preflight_validation.json')):
        safe(read(src)); shutil.copyfile(src, PUBLIC / name)
    original = ROOT / m['original_run']
    old_rows = read(original / 'perarm.json')
    assert len(old_rows) == 13
    rows = read(RUN / 'perarm.json')
    for old, imported in zip(old_rows, rows[:13]):
        assert imported['execution_backend'] == 'eager_imported'
        assert all(imported[k] == v for k, v in old.items())
        assert all(sha(original / f'block{old["block"]:02d}' / old['arm'] / p) == digest
                   for p, digest in imported['provenance']['imported_artifacts_sha256'].items())
    v = checks(); write(PUBLIC / 'validation.json', v)
    note = '''
## Publication and timing

All 96 training curves, per-map evaluation summaries, 192 paired Boolean
trace banks, matched-frontier tables, systems measurements, 576 aggregate
metric rows and exact batch schedules are retained. Curve/plan JSON and
frontier CSV use lossless gzip; NPZ files remain byte-exact. `perarm.json`
keeps original relative references; append `.gz` for a training-curve path.
Full numeric precision is unchanged. Banks were regenerated from frozen seeds
on CPU for publication and verified against the runtime tensor hashes.

13 eager arms were imported and 83 were trained with the qualified CUDA Graph
engine. The recorded 2771.016 seconds covers only the accelerated worker;
it is not the duration of all 96 arms including prior work and qualification.
Six block00 300-update replays exactly matched final parameters and Adam states.
That finite runtime check is separate from scientific qualification.

Checkpoints, host/PID manifests, launch receipts and logs stay local. Parameter
and checkpoint hashes remain available. Publication uses saved data and CPU
bank regeneration only; no new training, model inference or optimizer updates.
Concurrent W24 also has zero Full successes. This result rejects reliability
qualification for the frozen recipe; it does not reject all continuous latents
or establish an intrinsic carrier dimension. W2 changes transport.
'''
    (PUBLIC / 'RESULTS.md').write_text((RUN / 'RESULTS.md').read_text(encoding='utf-8') + note, encoding='utf-8')
    (PUBLIC / 'REPRODUCTION.md').write_text('''# Native-width saved evidence

Start with RESULTS.md, summary.json, validation.json and config.json, then
perarm.json and metrics.csv. Per-arm evaluation summaries contain every Full
component and all per-map metrics. NPZ and gzip data are secondary.

Saved-data verification from the standalone repository root:

    python -X utf8 -B tools/export_latent_width.py --verify-only

Requires NumPy and PyTorch; the verifier executes no model or CUDA operation.
It checks hashes, bank/schedule bindings, all 300-update curves, Boolean
pairing, coverage/retention/regression counts, saved Full predicates, all576
metric rows and paired statistics. It does not rerun logits, balanced accuracy
or matched-frontier strata. Read validation.json for the exact check scope.

For a fresh full run, use the source and commands in new/latent_width/PROTOCOL.md
with PyTorch2.5.1, NumPy1.26.4 and a compatible CUDA environment. Activate that
environment so the PowerShell launcher discovers the correct Python via PATH.
Run the combined sanity check and choose fresh output names; no historical
checkpoint is required. CUDA Graph continuation of an existing eager run uses
new/latent_runtime/ACCELERATION.md and requires its local reference checkpoints.
Runtime qualification is limited to the recorded six first-block replays.

Only the final update300 checkpoint is scored. The independent statistical
unit is an initialization/schedule block (16), not maps/cells/rollout times.
No efficacy selection or new sweeps were performed during publication.
''', encoding='utf-8')
    for name in ('config.json', 'RESULTS.md'):
        bindings[name]['encoding'] = 'Public provenance augmentation; original raw hash retained'
    publication = {'protocol': 'native_latent_width_publication_v1', 'run_id': RUN.name,
                   'review_base': m['git_review_base'], 'source_sha256': sources,
                   'exporter_sha256': sha(__file__), 'raw_artifact_bindings': bindings,
                   'original_manifest_sha256': sha(RUN / 'manifest.json'), 'original_status_sha256': sha(RUN / 'status.json'),
                   'imported_eager_arms_original_hashes_checked': True,
                   'published_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in PUBLIC.rglob('*') if p.is_file()},
                   'derived_only': ['CPU-regenerated banks exactly matching runtime tensor hashes'],
                   'excluded': ['checkpoints', 'machine manifests', 'source snapshots', 'launch receipts', 'logs', 'PIDs']}
    safe(publication); write(MANIFEST, publication)
    print(json.dumps({'status': 'PASS', 'files': len(publication['published_sha256']),
                      'MiB': round(sum(p.stat().st_size for p in PUBLIC.rglob('*') if p.is_file()) / 1024**2, 2), **v}), flush=True)


def verify():
    m = read(MANIFEST); safe(m)
    assert sha(__file__) == m['exporter_sha256']
    for group in ('source_sha256', 'published_sha256'):
        for rel, digest in m[group].items(): assert sha(ROOT / rel) == digest, rel
    for record in m['raw_artifact_bindings'].values():
        path = ROOT / record['public_path']
        # config/report add explicit public provenance; original raw hash is retained.
        if path.name in ('config.json', 'RESULTS.md'): continue
        if record['encoding'].startswith('gzip'):
            import hashlib
            assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == record['sha256']
        else: assert sha(path) == record['sha256']
    for path in PUBLIC.rglob('*.json'): safe(read(path))
    assert checks() == read(PUBLIC / 'validation.json')
    print(json.dumps({'status': 'PASS', 'files': len(m['published_sha256']), 'arms': 96, 'trace_banks': 192}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--resume-export', action='store_true', help='Resume an unpublished export with no publication manifest.')
    args = parser.parse_args()
    assert not (args.verify_only and args.resume_export)
    verify() if args.verify_only else build(args.resume_export)
