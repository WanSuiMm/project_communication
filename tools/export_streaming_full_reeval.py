"""Export and verify the six saved Full reevaluation units; never run a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/streaming_full_reeval_20261008_01'
PUBLIC = ROOT / 'evidence/streaming_full_reeval_20261008_01'
MANIFEST = ROOT / 'STREAMING_FULL_REEVAL_PUBLICATION_MANIFEST.json'
CHECK = ROOT / 'analyses/streaming_full_reeval_check_20261008_02.json'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


A = module('_full_public_safe', ROOT / 'tools/export_addressed_delta.py')
P = module('_full_public_predicate', ROOT / 'new/seed4_followup/phenotype.py')


def unpack(path, size):
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {key+suffix for key in
            ('correct', 'original_correct', 'flipped_correct') for suffix in ('_shape', '_packed')}
        for key in ('correct', 'original_correct', 'flipped_correct'):
            shape = tuple(int(x) for x in archive[key+'_shape'])
            assert shape == (257, 32, size, size)
            packed = archive[key+'_packed']
            assert packed.dtype == np.uint8 and packed.size == (np.prod(shape)+7)//8
            result[key] = np.unpackbits(packed, bitorder='little', count=int(np.prod(shape))).reshape(shape).astype(bool)
    assert np.array_equal(result['correct'], result['original_correct'] & result['flipped_correct'])
    return result


def verify_units(folder, summary, manifest):
    rows = summary['units']
    expected = {(m, c) for m in ('block00_u300', 'block02_u300', 'block00_u275')
                for c in ('historical_full', 'addressed_selection')}
    assert len(rows) == 6 and {(r['model'], r['cohort']) for r in rows} == expected
    assert summary['status'] == 'COMPLETE' and summary['completed_units'] == 6
    assert summary['training'] is False and summary['optimizer_updates'] == 0
    counts = {'units': 0, 'trace_banks': 0, 'map_summaries': 0, 'frontier_strata': 0, 'gates': 0}
    for row in rows:
        assert row['primary'] == (row['update'] == 300)
        unit = folder / row['unit']
        complete = A.read_json(unit / 'complete.json')
        assert complete['complete'] and complete['binding'] == row
        for name, digest in complete['artifacts_sha256'].items():
            assert name in ('summary.json', 'matched_frontier_strata.csv', 'size32_traces.npz', 'size64_traces.npz')
            assert A.sha(unit / name) == digest
        saved = A.read_json(unit / 'summary.json')
        assert saved['phenotype_gate'] == P.predicate(saved)
        assert saved['phenotype_gate']['pass'] == row['full_pass']
        assert saved['phenotype_gate']['reasons'] == row['reasons']
        counts['gates'] += len(saved['phenotype_gate']['checks'])
        with (unit / 'matched_frontier_strata.csv').open(newline='', encoding='utf-8') as handle:
            strata = list(csv.DictReader(handle))
        counts['frontier_strata'] += len(strata)
        for size in (32, 64):
            bank = A.load_bank(folder / f'banks/{row["cohort"]}/size{size}.npz')
            assert A.bank_tensor_hash(bank) == manifest['bank_sha256'][row['cohort']][str(size)]
            traces = unpack(unit / f'size{size}_traces.npz', size)
            correct = traces['correct']
            selected = bank['changed'][:, 0].astype(bool)
            distance = bank['distance'][:, 0]
            strict = selected & (distance > 16) & (distance < 32)
            sr = saved['sizes'][str(size)]
            for t in (64, 128, 256):
                for name, selection in (('all_changed', selected), ('strict_16_32', strict)):
                    assert P._coverage(correct[t], selection) == sr['endpoints'][str(t)][name]
            for start, end in ((64, 128), (64, 256), (128, 256)):
                for name, selection in (('all_changed', selected), ('strict_16_32', strict)):
                    assert P._transition(correct, selection, start, end) == sr['transitions'][name][f'{start}_to_{end}']
            first, _, relapse = P._first_and_stable(correct)
            assert P._ratio(int((selected & relapse).sum()), int((selected & (first >= 0)).sum())) == sr['ever_regressed_over_ever_correct']
            assert len(sr['per_map']) == 32
            assert sum(r['ever_regressed']['numerator'] for r in sr['per_map']) == sr['ever_regressed_over_ever_correct']['numerator']
            size_strata = [r for r in strata if int(r['size']) == size]
            assert len(size_strata) == sr['frontier']['common_strata']
            effects = []
            for map_index in range(32):
                mr = [r for r in size_strata if int(r['map_index']) == map_index]
                if mr:
                    weights = [float(r['matched_weight']) for r in mr]
                    value = sum(w*float(r['rate_difference']) for w, r in zip(weights, mr))/sum(weights)
                    effects.append(value)
            assert len(effects) == sr['frontier']['eligible_maps']
            assert np.isclose(np.mean(effects), sr['frontier']['mean_map_weighted_difference'], rtol=1e-12, atol=1e-12)
            counts['trace_banks'] += 1
            counts['map_summaries'] += 32
            del traces, correct, bank
        counts['units'] += 1
    return {'status': 'PASS', 'verification': 'saved data only; no model inference or optimization', **counts,
        'checks': ['unique frozen units and primary/diagnostic roles', 'completion-marker artifact hashes',
            'original Full predicate and reasons', 'bank tensor identity', 'lossless all-time trace shape and branch conjunction',
            'trace-derived coverage, retention, regression and per-map count consistency', 'CSV-derived frontier effects and support']}


def verify_public():
    publication = A.read_json(MANIFEST)
    A.safe(publication)
    expected = publication['published_artifacts_sha256']
    actual = {p.relative_to(ROOT).as_posix(): A.sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
    assert actual == expected, 'Published package file/hash drift'
    for name, digest in publication['source_sha256'].items():
        assert A.sha(ROOT / name) == digest, f'Source drift: {name}'
    for path in PUBLIC.rglob('*'):
        if path.is_file():
            assert path.stat().st_size < A.MAX_FILE_BYTES
            if path.suffix == '.json':
                A.safe(A.read_json(path))
            elif path.suffix in ('.csv', '.md'):
                A.safe(path.read_text(encoding='utf-8-sig'))
    validation = verify_units(PUBLIC, A.read_json(PUBLIC / 'summary.json'), A.read_json(PUBLIC / 'manifest.json'))
    assert validation == A.read_json(PUBLIC / 'validation.json')
    return {'status': 'PASS', **validation, 'files': len(actual),
        'bytes': sum(p.stat().st_size for p in PUBLIC.rglob('*') if p.is_file())}


def export():
    assert not PUBLIC.exists() and not MANIFEST.exists(), 'New publication destination required'
    state = A.read_json(RUN / 'status.json')
    assert state['status'] == 'COMPLETE' and state['completed_units'] == 6
    summary, raw_manifest = A.read_json(RUN / 'summary.json'), A.read_json(RUN / 'manifest.json')
    validation = verify_units(RUN, summary, raw_manifest)
    for name, digest in raw_manifest['source_sha256'].items():
        assert A.sha(ROOT / name) == digest
    qualification = A.read_json(CHECK)
    assert qualification['status'] == 'PASS' and A.sha(CHECK) == raw_manifest['qualification_sha256']
    PUBLIC.mkdir(parents=True)
    for name in ('summary.json', 'RESULTS.md'):
        shutil.copyfile(RUN / name, PUBLIC / name)
    for row in summary['units']:
        for name in ('summary.json', 'matched_frontier_strata.csv', 'size32_traces.npz', 'size64_traces.npz', 'complete.json'):
            target = PUBLIC / row['unit'] / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(RUN / row['unit'] / name, target)
    for name in ('historical_full', 'addressed_selection'):
        for size in (32, 64):
            target = PUBLIC / f'banks/{name}/size{size}.npz'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(RUN / f'banks/{name}/size{size}.npz', target)
    clean = {key: raw_manifest[key] for key in ('protocol', 'started_utc', 'training', 'optimizer_updates',
        'expected_units', 'bank_sha256', 'checkpoints', 'source_sha256', 'parent_manifest_sha256',
        'historical_manifest_sha256', 'qualification_sha256', 'torch', 'backend')}
    clean.update(execution_status='COMPLETE', finished_utc=state['finished_utc'], seconds=state['seconds'],
        checkpoint_contents_published=False, boolean_trace_persistence=True,
        claim_boundary='Selected-checkpoint descriptive Full reevaluation; selecting cohort reused; no formation-rate estimate.')
    A.safe(clean)
    A.write_json(PUBLIC / 'manifest.json', clean)
    A.write_json(PUBLIC / 'validation.json', validation)
    A.write_json(PUBLIC / 'qualification.json', qualification)
    with (PUBLIC / 'full_gates.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('model', 'primary', 'cohort', 'gate', 'value', 'threshold', 'pass'))
        writer.writeheader()
        for row in summary['units']:
            saved = A.read_json(PUBLIC / row['unit'] / 'summary.json')
            for gate, value in saved['phenotype_gate']['checks'].items():
                writer.writerow({'model': row['model'], 'primary': row['primary'], 'cohort': row['cohort'],
                    'gate': gate, **value})
    reproduction = '''# Saved-data reproduction

Read RESULTS.md and summary.json first; full_gates.csv lists every threshold.
The six unit folders retain unchanged summaries, completion bindings, all frontier
strata and losslessly compressed paired/original/flipped Boolean trajectories.
No trace times, cells, maps or negative gates were removed. Model weights and
private process receipts remain local; checkpoint and parameter hashes are retained.

The scientific run was zero-training and made no optimizer updates. Primary
u300 checkpoints were selected by the earlier Joint metric; u275 is diagnostic.
The addressed_selection cohort is reused selection data. No formation-rate estimate.

From repository root, Python with NumPy verifies the full public package:

```powershell
python -X utf8 -B tools/export_streaming_full_reeval.py --verify-only
```

For a trace NPZ, load with `numpy.load(path, allow_pickle=False)`. Each key
`correct`, `original_correct`, `flipped_correct` has `_shape` and `_packed`
arrays. Decode using `numpy.unpackbits(packed, bitorder="little", count=prod(shape))`
and reshape to `[257,32,size,size]`; every integer time0..256 is present.
Input map NPZ banks are published too, preserving their exact tensor hashes.

The verifier reuses the frozen Full predicate, recomputes trace-derived endpoint
coverage and transitions, ever-regression, and frontier effects from saved CSV.
It does not run inference. Repeating model evaluation requires the three private
checkpoint contents and Torch2.5.1/CUDA; see the frozen protocol and run.py.
'''
    (PUBLIC / 'REPRODUCTION.md').write_text(reproduction, encoding='utf-8')
    publication = {'schema': 'streaming-full-reeval-publication-v1', 'execution_status': 'COMPLETE',
        'training': False, 'optimizer_updates': 0, 'units': 6, 'primary_units': 4, 'diagnostic_units': 2,
        'source_sha256': {**raw_manifest['source_sha256'],
            'tools/export_streaming_full_reeval.py': A.sha(Path(__file__))},
        'original_manifest_sha256': A.sha(RUN / 'manifest.json'),
        'publication_scope': 'Code, unchanged summaries, all gates, all-time packed traces, frontier strata and exact banks; no weights or private execution records.',
        'published_artifacts_sha256': {p.relative_to(ROOT).as_posix(): A.sha(p) for p in PUBLIC.rglob('*') if p.is_file()}}
    A.write_json(MANIFEST, publication)
    return verify_public()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_public() if args.verify_only else export()), flush=True)
