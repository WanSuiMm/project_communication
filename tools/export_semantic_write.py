"""Publish/verify completed semantic-write measurements without model inference."""
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
RUN = ROOT / 'runs/semantic_write_20261006_03'
PUBLIC = ROOT / 'evidence/semantic_write_20261006'
MANIFEST = ROOT / 'SEMANTIC_WRITE_PUBLICATION_MANIFEST.json'
QUAL = ROOT / 'analyses/semantic_write_check_20261006_04.json'


def runner():
    spec = importlib.util.spec_from_file_location('_public_semantic', ROOT / 'new/semantic_write_audit/run.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def masks(bank):
    changed = bank['changed'][:, 0].astype(bool)
    distance = bank['distance'][:, 0]
    return {'strict_far': changed & (distance > 16) & (distance < 32),
            'off_source': changed & (distance > 0), 'source': changed & (distance == 0),
            'all_changed': changed}


def validate_saved():
    r = runner()
    summary, rows, cfg = (read(PUBLIC / name) for name in ('summary.json', 'perunit.json', 'config.json'))
    assert summary['status'] == 'COMPLETE' and summary['completed_units'] == len(rows) == 396
    assert summary['training'] is False and summary['optimizer_updates'] == 0
    assert summary['imported_units'] == 75 and summary['newly_computed_units'] == 321
    expected = {(p['name'], size, c) for p in cfg['plans'] for size in (32, 64) for c in r.CONDITIONS}
    assert {(v['model'], v['size'], v['condition']) for v in rows} == expected and len(expected) == 396
    assert all(summary[k] == v for k, v in r.aggregate(rows).items())
    replay = read(PUBLIC / 'replay.json')
    assert len(replay) == 66 and len({(v['model'], v['size']) for v in replay}) == 66
    assert all(v['status'] == 'PASS' and sum(v['mismatched_bits'].values()) == 0
               for v in replay if v['model'] != 'historical_seed4')
    assert sum(v['status'] == 'PASS' for v in replay) == 64
    assert sum(v['status'] == 'NEW_COMMON_COHORT_REFERENCE' for v in replay) == 2
    banks = {}
    for size in (32, 64):
        with np.load(PUBLIC / f'banks/evaluation{size}.npz', allow_pickle=False) as values:
            banks[size] = {k: values[k] for k in values.files}
        assert r.C.tensor_hash({k: torch.from_numpy(v) for k, v in banks[size].items()}) == cfg['data_sha256'][str(size)]
    references = {}
    margin_arrays = diagnostic_arrays = map_rows = 0
    for row in rows:
        folder = PUBLIC / row['model'] / f"size{row['size']}"
        stem = row['condition']
        path = folder / f'{stem}.npz'
        meta = read(folder / f'{stem}.json')
        assert meta['arrays_sha256'] == sha(path) and meta['endpoints'] == row['endpoints']
        assert (meta['model'], meta['size'], meta['condition']) == (row['model'], row['size'], stem)
        with np.load(path, allow_pickle=False) as arrays:
            shape = tuple(int(v) for v in arrays['correct_shape'])
            assert shape == (257, 32, row['size'], row['size'])
            good = np.unpackbits(arrays['correct_packed'], bitorder='little', count=int(np.prod(shape))).reshape(shape).astype(bool)
            selected = masks(banks[row['size']])
            assert np.array_equal(arrays['sampled_margin_times'], r.TIMES)
            assert arrays['sampled_margins'].shape == (6, int(selected['all_changed'].sum()))
            assert arrays['sampled_margins'].dtype == np.float32
            assert arrays['intervention_stats'].shape == (256, 32, 2)
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            if stem == 'natural':
                assert arrays['write_diagnostics'].shape == (256, 4, 32, 9)
                assert arrays['source_margins'].shape == (257, 2, 32)
                assert arrays['source_proposed_increments'].shape == (256, 2, 32)
                diagnostic_arrays += 1
            margin_arrays += 1
        key = row['model'], row['size']
        if stem == 'natural':
            references[key] = good.copy()
        reference = references[key]
        if stem != 'natural':
            assert np.array_equal(good[:r.ACTIVATE[stem]+1], reference[:r.ACTIVATE[stem]+1])
        once = good.any(axis=0)
        regressed = (np.logical_or.accumulate(good, axis=0) & ~good).any(axis=0)
        expected_map = []
        ratio = lambda num, den: float(num/den) if den else None
        for name, mask in selected.items():
            base, natural_base = good[64] & mask, reference[64] & mask
            observed = {'pixels': int(mask.sum()),
                'coverage': {str(t): ratio((good[t] & mask).sum(), mask.sum()) for t in (64, 128, 256)},
                'retention_own64_to256': ratio((good[256] & base).sum(), base.sum()),
                'retention_natural64_to256': ratio((good[256] & natural_base).sum(), natural_base.sum()),
                'gained_vs_natural64': int((good[256] & mask & ~reference[64]).sum()),
                'lost_vs_natural64': int((~good[256] & natural_base).sum()),
                'ever_regressed_fraction': ratio((regressed & mask).sum(), (once & mask).sum())}
            assert observed == row['endpoints'][name], (key, stem, name)
            for i in range(32):
                for t in (64, 128, 256):
                    count, total = int((good[t, i] & mask[i]).sum()), int(mask[i].sum())
                    expected_map.append((name, i, t, count, total, ratio(count, total)))
        with gzip.open(folder / f'{stem}_per_map.csv.gz', 'rt', encoding='utf-8', newline='') as handle:
            actual = [(v['selection'], int(v['map']), int(v['time']), int(v['correct']), int(v['pixels']),
                       float(v['coverage']) if v['coverage'] else None) for v in csv.DictReader(handle)]
        assert actual == expected_map
        map_rows += len(actual)
        if stem == r.CONDITIONS[-1]:
            del references[key]
    return {'status': 'PASS', 'scope': 'Saved arrays, CPU bank hashes and arithmetic only; no inference or optimization.',
            'unique_units': len(rows), 'per_map_rows': map_rows, 'packed_trace_banks': len(rows),
            'sampled_margin_arrays': margin_arrays, 'natural_diagnostic_arrays': diagnostic_arrays,
            'runtime_factorial_natural_replays': 64, 'runtime_factorial_replay_mismatched_bits': 0,
            'historical_reference_uses_new_common_cohort': True,
            'aggregate_and_all_endpoint_counts_recomputed': True,
            'intervention_prefixes_exactly_natural': True,
            'logits_and_signed_write_diagnostics_not_rerun': True,
            'primary_verdict': summary['primary']['verdict']}


def build():
    assert not PUBLIC.exists() and not MANIFEST.exists(), 'Refusing to overwrite published evidence'
    m, status, summary = (read(RUN / name) for name in ('manifest.json', 'status.json', 'summary.json'))
    assert status['status'] == summary['status'] == 'COMPLETE' and status['completed_units'] == 396
    qualification = read(QUAL)
    assert qualification['status'] == 'PASS' and sha(QUAL) == m['qualification_sha256']
    assert qualification['source_sha256'] == m['source_sha256']
    for name, digest in m['source_sha256'].items():
        assert sha(ROOT / name) == sha(RUN / 'source' / name) == digest, name
    for plan in m['plans']:
        checkpoint = ROOT / plan['checkpoint'] if plan['block'] is None else ROOT / 'runs/latent_factorial_20261005_01' / plan['checkpoint']
        assert sha(checkpoint) == plan['checkpoint_sha256']
    PUBLIC.mkdir(parents=True)
    bindings = {}
    paths = [RUN / n for n in ('summary.json', 'perunit.json', 'replay.json', 'RESULTS.md')]
    paths += sorted(p for p in RUN.rglob('*') if p.is_file() and p.relative_to(RUN).parts[0].startswith(('block', 'historical_seed4', 'banks')))
    for path in paths:
        rel = path.relative_to(RUN)
        assert path.suffix in ('.json', '.csv', '.npz', '.md')
        if path.suffix == '.json':
            safe(read(path))
        if path.suffix in ('.csv', '.md'):
            assert not PRIVATE_TEXT.search(path.read_text(encoding='utf-8-sig'))
        destination = PUBLIC / (str(rel) + '.gz' if path.suffix == '.csv' else rel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.csv':
            destination.write_bytes(gzip.compress(path.read_bytes(), compresslevel=9, mtime=0))
        else:
            shutil.copyfile(path, destination)
        bindings[rel.as_posix()] = {'raw_sha256': sha(path), 'public_path': destination.relative_to(ROOT).as_posix(),
                                  'encoding': 'lossless gzip' if path.suffix == '.csv' else 'byte-exact copy'}
    cfg = read(RUN / 'config.json')
    cfg.update({key: m[key] for key in ('torch', 'source_sha256', 'data_sha256', 'plans', 'started_utc', 'git_review_base')})
    cfg.update({'training': False, 'optimizer_updates': 0, 'finished_utc': status['finished_utc'],
                'completed_units': 396, 'imported_units': 75, 'newly_computed_units': 321,
                'worker_elapsed_seconds': status['elapsed_seconds'],
                'elapsed_scope': 'Continuation worker only; excludes time previously spent computing imported units, preflight and publication.',
                'checkpoint_files_published': False, 'runtime_cap': False})
    safe(cfg)
    write(PUBLIC / 'config.json', cfg)
    safe(qualification)
    write(PUBLIC / 'runtime_qualification.json', qualification)
    imported = read(RUN / 'import_validation.json')
    assert imported['status'] == 'PASS' and imported['imported_units'] == 75
    resumed = {k: imported[k] for k in ('status', 'imported_units', 'source_run', 'source_manifest_sha256',
               'source_perunit_sha256', 'source_replay_sha256', 'scientific_function_AST_equivalence', 'file_sha256')}
    for name, digest in imported['file_sha256'].items():
        assert sha(RUN / name) == digest
    resumed['interruption'] = 'Transient Windows atomic JSON replacement PermissionError; original machine traceback remains local.'
    write(PUBLIC / 'import_validation.json', resumed)
    note = '''

## Interpretation and publication

The frozen primary requires both mean deltas >=0.05 and positive source-natural
in >=6/8 native blocks. Neither threshold is met. Native size32 source T256
coverage rises from0.5820 to1.0000, while unprotected strict16<d<32 T256 coverage
changes only0.3217 to0.3223. Source preservation alone did not rescue distant
computation for this intervention and these fixed checkpoints/cohorts.

Oracle protection uses distant ground truth and can inject labels via its gate;
its improvements cannot establish learned communication or rescue the primary.
Nullspace ablation changes future dynamics. The selected C24 seed4 reference
qualifies on this common cohort; it is not a matched-width replication. The
independent primary unit is the paired training block(n8), not a pixel/map/time.

All396 packed paired Boolean traces, FP32 sampled margins,66 natural signed-write
diagnostics, source trajectories, intervention arrays and all per-map endpoint
rows are retained. NPZ files are already losslessly compressed and byte-exact;
per-map CSV files use lossless gzip. No precision reduction. Protected retention
is a manipulation check. Checkpoints and machine/launch records remain local.

The1318.938-second timing covers the resumed worker only:75 completed units were
imported with validated hashes,321 were newly computed. It is not total elapsed
time for all396 measurements. Saved-data publication performs no model inference,
training or optimizer update. See validation.json for its exact scope.
'''
    with (PUBLIC / 'RESULTS.md').open('a', encoding='utf-8') as handle:
        handle.write(note)
    bindings['RESULTS.md']['encoding'] = 'public interpretation appended; raw hash retained'
    (PUBLIC / 'REPRODUCTION.md').write_text('''# Semantic-write saved evidence

Read RESULTS.md, summary.json, validation.json, then config.json and perunit.json.
All396 unit NPZ/JSON files and compressed per-map CSVs are secondary records.
Each NPZ stores little-endian packbits of the flattened paired Boolean trace,
its shape, FP32 sampled changed-region margins, and intervention arrays.
Natural units additionally store per-step signed-write diagnostics and source
margin/proposal arrays. Unit JSON files define layouts and column names.

CPU saved-data verification from this repository root:

    python -X utf8 -B tools/export_semantic_write.py --verify-only

Dependencies: NumPy and PyTorch (see requirements.txt). No CUDA, local run folders
or checkpoint files are needed for this verification. It checks hashes, banks,
all396 endpoint and per-map counts, retention/regression, natural prefixes and
the frozen aggregate. Recorded natural replay checks were performed at runtime;
publication does not repeat model logits or signed-write calculations.

Scientific code: new/semantic_write_audit/run.py; projection/wiring checks:
new/semantic_write_audit/checks.py; transient Windows-lock test: check_io.py.
Follow PROTOCOL.md and EXECUTION_FIX.md in that directory for a new experiment.
Running new model inference requires the33 local checkpoints bound in config.json
and the historical Torch2.5.1 CUDA backend. Checkpoint files stay local, so a
clone can verify saved measurements but cannot rerun inference unassisted.
The successful dispatch used a managed execution session and the direct command:

    python -X utf8 -u -B new/semantic_write_audit/run.py --out runs/NEW_RUN --qualification analyses/NEW_CHECK.json

The original detached PowerShell launcher is retained as runtime-bound source;
it did not survive the initial dispatch in this environment. Source snapshots
and machine receipts stay local. Neither publication nor verification trains.
''', encoding='utf-8')
    v = validate_saved()
    v['checkpoint_file_hashes_verified_locally_during_export'] = 33
    v['imported_files_byte_hashes_verified'] = len(imported['file_sha256'])
    write(PUBLIC / 'validation.json', v)
    publication = {'protocol': 'semantic_write_saved_data_publication_v1', 'run_id': RUN.name,
        'review_base': m['git_review_base'], 'source_sha256': m['source_sha256'], 'data_sha256': m['data_sha256'],
        'original_manifest_sha256': sha(RUN / 'manifest.json'), 'original_status_sha256': sha(RUN / 'status.json'),
        'exporter_sha256': sha(__file__), 'raw_artifact_bindings': bindings,
        'published_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in PUBLIC.rglob('*') if p.is_file()},
        'excluded': ['checkpoints', 'machine manifests/status/tracebacks', 'launch receipts/logs', 'duplicate source snapshots'],
        'training': False, 'optimizer_updates': 0}
    safe(publication)
    write(MANIFEST, publication)
    print(json.dumps({'status': 'PASS', 'files': len(publication['published_sha256']),
          'MiB': round(sum(p.stat().st_size for p in PUBLIC.rglob('*') if p.is_file())/1024**2, 2), **v}), flush=True)


def verify():
    m = read(MANIFEST)
    safe(m)
    assert sha(__file__) == m['exporter_sha256']
    for name, digest in m['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in m['published_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for record in m['raw_artifact_bindings'].values():
        path = ROOT / record['public_path']
        if record['encoding'] == 'lossless gzip':
            import hashlib
            assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == record['raw_sha256']
        elif record['encoding'] == 'byte-exact copy':
            assert sha(path) == record['raw_sha256']
    for path in PUBLIC.rglob('*.json'):
        safe(read(path))
    actual = validate_saved()
    saved = read(PUBLIC / 'validation.json')
    assert all(saved[k] == v for k, v in actual.items())
    print(json.dumps(actual), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verify() if args.verify_only else build()
