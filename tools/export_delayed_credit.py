"""Publish/verify the completed cue-once screen using saved data only."""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/delayed_credit_20261008_01'
PUBLIC = ROOT / 'evidence/delayed_credit_20261008_01'
MANIFEST = ROOT / 'DELAYED_CREDIT_PUBLICATION_MANIFEST.json'
QUALIFICATION = ROOT / 'analyses/delayed_credit_qualification_20261008_01/qualification.json'
ARMS = ('dense_k8', 'terminal_k8', 'terminal_k64')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


A = module('_delayed_public_safe', ROOT / 'tools/export_addressed_delta.py')
E = module('_delayed_public_evaluation', ROOT / 'new/delayed_credit/evaluation.py')
P = module('_delayed_public_reporting', ROOT / 'new/delayed_credit/reporting.py')


def unpack(path, size):
    values = {}
    with np.load(path, allow_pickle=False) as archive:
        valid = int(archive['valid_through_step'])
        assert -1 <= valid <= 256
        assert np.array_equal(archive['valid_times'], np.arange(valid+1))
        for key in ('correct', 'original_correct', 'flipped_correct'):
            shape = tuple(int(v) for v in archive[key+'_shape'])
            assert shape == (valid+1, 32, size, size)
            packed = archive[key+'_packed']
            assert packed.dtype == np.uint8 and packed.size == (np.prod(shape)+7)//8
            values[key] = np.unpackbits(packed, bitorder='little', count=int(np.prod(shape))).reshape(shape).astype(bool)
    assert np.array_equal(values['correct'], values['original_correct'] & values['flipped_correct'])
    return values, valid


def verify_saved(folder, manifest, summary):
    assert summary['status'] == 'COMPLETE' and summary['complete'] and len(summary['rows']) == 12
    assert {(r['block'], r['arm']) for r in summary['rows']} == {(b,a) for b in range(4) for a in ARMS}
    assert P.aggregate(summary['rows'], 'COMPLETE') == summary
    for name, digest in manifest['source_sha256'].items():
        assert A.sha(ROOT / name) == digest, f'Frozen source drift: {name}'
    for name, digest in manifest['data_sha256'].items():
        assert A.bank_tensor_hash(A.load_bank(folder / f'banks/{name}.npz')) == digest
    counts = {'trajectories': 0, 'checkpoint_bindings': 0, 'training_updates': 0,
              'trace_banks': 0, 'per_map_records': 0, 'reach_checks': 0, 'hold_checks': 0}
    for row in summary['rows']:
        unit = folder / f'block{row["block"]:02d}/{row["arm"]}'
        record = A.read_json(unit / 'record.json')
        assert record == row and record['status'] == 'COMPLETE' and record['update'] == 300
        complete = A.read_json(unit / 'complete.json')
        assert complete['complete'] and A.sha(unit / 'record.json') == complete['record_sha256']
        evaluation = A.read_json(unit / 'evaluation/u300/summary.json')
        assert evaluation == record['evaluation']
        assert A.sha(unit / 'evaluation/u300/summary.json') == complete['evaluation_sha256']
        curve = A.read_json(unit / 'curves/u300.json')
        assert len(curve) == 300 and [r['update'] for r in curve] == list(range(1,301))
        assert A.digest(curve) == complete['curve_sha256']
        for update in (50,100,150,200,250,300):
            binding = A.read_json(unit / f'checkpoints/u{update:03d}.pt.complete.json')
            assert binding['complete'] and binding['update'] == update
            assert binding['curve_sha256'] == A.digest(curve[:update])
            assert binding['formal_endpoint'] == (update == 300)
            checkpoint = unit / f'checkpoints/u{update:03d}.pt'
            if checkpoint.exists():
                assert A.sha(checkpoint) == binding['checkpoint_sha256']
            if update == 300:
                assert binding['checkpoint_sha256'] == complete['checkpoint_sha256']
                assert binding['parameter_sha256'] == record['parameter_sha256']
            counts['checkpoint_bindings'] += 1
        maps = A.read_json(unit / 'evaluation/u300/per_map_metrics.json')['sizes']
        assert maps == evaluation['per_map_metrics']
        for size in (32,64):
            values, valid = unpack(unit / f'evaluation/u300/size{size}_traces.npz', size)
            bank = A.load_bank(folder / f'banks/evaluation{size}.npz')
            changed = bank['changed'][:,0].astype(bool)
            distance = bank['distance'][:,0]
            strict = changed & (distance > 16) & (distance < 32)
            correct = values['correct']
            saved = evaluation['sizes'][str(size)]
            assert valid == evaluation['trace_status']['sizes'][str(size)]['valid_through_step']
            for time in (64,128,256):
                endpoint = saved['endpoints'][str(time)]
                if valid < time:
                    assert endpoint is None
                    continue
                for key, selection in (('all_changed',changed),('strict_16_32',strict)):
                    assert E.E._coverage(correct[time], selection) == endpoint[key]
            for key, selection in (('all_changed',changed),('strict_16_32',strict)):
                for start,end in ((64,128),(64,256),(128,256)):
                    measured = E.E._transition(correct,selection,start,end) if valid >= end else None
                    assert measured == saved['transitions'][key][f'{start}_to_{end}']
            assert E._per_map_metrics(correct,bank,size,valid) == maps[str(size)]
            counts['trace_banks'] += 1
            counts['per_map_records'] += len(maps[str(size)])
        status = evaluation['trace_status']['sizes']['32']
        assert E._reach_gate(evaluation,status) == evaluation['reach_gate']
        assert E._hold_gate(evaluation,status) == evaluation['hold_gate']
        counts['reach_checks'] += len(evaluation['reach_gate']['checks'])
        counts['hold_checks'] += len(evaluation['hold_gate']['checks'])
        counts['trajectories'] += 1
        counts['training_updates'] += len(curve)
    return {'status':'PASS','verification':'saved arrays, records and hashes only; no model inference or optimization', **counts}


def verify_public():
    publication = A.read_json(MANIFEST)
    A.safe(publication)
    hashes = {p.relative_to(ROOT).as_posix():A.sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
    assert hashes == publication['published_artifacts_sha256']
    for name,digest in publication['source_sha256'].items():
        assert A.sha(ROOT/name) == digest
    for path in PUBLIC.rglob('*'):
        if not path.is_file():
            continue
        assert path.stat().st_size < A.MAX_FILE_BYTES
        if path.suffix == '.json':
            A.safe(A.read_json(path))
        elif path.suffix in ('.md','.csv'):
            A.safe(path.read_text(encoding='utf-8'))
    validation = verify_saved(PUBLIC,A.read_json(PUBLIC/'manifest.json'),A.read_json(PUBLIC/'summary.json'))
    assert validation == A.read_json(PUBLIC/'validation.json')
    return {**validation,'files':len(hashes),'bytes':sum(p.stat().st_size for p in PUBLIC.rglob('*') if p.is_file())}


def export():
    assert not PUBLIC.exists() and not MANIFEST.exists(), 'New destination required'
    status = A.read_json(RUN/'status.json')
    assert status['status'] == 'COMPLETE' and status['completed_trajectories'] == 12
    raw_manifest,summary = A.read_json(RUN/'manifest.json'),A.read_json(RUN/'summary.json')
    validation = verify_saved(RUN,raw_manifest,summary)
    qualification = A.read_json(QUALIFICATION)
    assert qualification['status'] == 'PASS'
    assert A.sha(QUALIFICATION) == raw_manifest['qualification_sha256']
    assert qualification['source_sha256'] == raw_manifest['source_sha256']
    assert qualification['config'] == raw_manifest['config']
    PUBLIC.mkdir(parents=True)
    paths = ['summary.json','RESULTS.md']
    paths += [f'banks/{name}.npz' for name in ('train','evaluation32','evaluation64')]
    paths += [f'schedules/block{b:02d}.json' for b in range(4)]
    for row in summary['rows']:
        unit = f'block{row["block"]:02d}/{row["arm"]}'
        paths += [f'{unit}/{name}' for name in ('record.json','complete.json','curves/u300.json',
            'evaluation/u300/summary.json','evaluation/u300/per_map_metrics.json',
            'evaluation/u300/size32_traces.npz','evaluation/u300/size64_traces.npz')]
        paths += [f'{unit}/checkpoints/u{u:03d}.pt.complete.json' for u in (50,100,150,200,250,300)]
    for name in paths:
        target = PUBLIC/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(RUN/name,target)
    clean = {k:raw_manifest[k] for k in ('protocol','config','started_utc','source_sha256','data_sha256',
        'qualification_sha256','expected_trajectories','torch','backend')}
    clean.update(execution_status='COMPLETE',finished_utc=status['finished_utc'],seconds=status['seconds'],
        checkpoint_contents_published=False,all_boolean_trace_prefixes_published=True,
        claim_boundary='Developmental screen; K64 positive control unqualified; no temporal-credit gap claim.')
    A.write_json(PUBLIC/'manifest.json',clean)
    A.write_json(PUBLIC/'qualification.json',qualification)
    A.write_json(PUBLIC/'validation.json',validation)
    with (PUBLIC/'final_metrics.csv').open('w',newline='',encoding='utf-8') as handle:
        fields = ['block','arm','update','size','time','original_ba','flipped_ba','strict_pooled',
                  'strict_mean_map','all_changed_pooled','primary_reach_pass','secondary_hold_pass','valid_through_step']
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for row in summary['rows']:
            evaluation = row['evaluation']
            for size in (32,64):
                for time in (64,128,256):
                    endpoint = evaluation['sizes'][str(size)]['endpoints'][str(time)]
                    endpoint = endpoint or {}
                    writer.writerow(dict(block=row['block'],arm=row['arm'],update=300,size=size,time=time,
                        original_ba=endpoint.get('original_ba'),flipped_ba=endpoint.get('flipped_ba'),
                        strict_pooled=endpoint.get('strict_16_32',{}).get('pooled_coverage',{}).get('value'),
                        strict_mean_map=endpoint.get('strict_16_32',{}).get('mean_map_coverage'),
                        all_changed_pooled=endpoint.get('all_changed',{}).get('pooled_coverage',{}).get('value'),
                        primary_reach_pass=evaluation['reach_gate']['pass'],secondary_hold_pass=evaluation['hold_gate']['pass'],
                        valid_through_step=evaluation['trace_status']['sizes'][str(size)]['valid_through_step']))
    (PUBLIC/'REPRODUCTION.md').write_text('''# Saved-data reproduction

Read RESULTS.md, summary.json and final_metrics.csv first. The 12 unit folders
retain unchanged endpoint records, all per-map metrics and packed Boolean traces
for every valid integer time. Each u300 curve contains all 300 updates; earlier
curve files are redundant prefixes and are represented by their completion hashes.
All 72 checkpoint bindings, four minibatch schedules and three exact banks remain.
Model/Adam/RNG checkpoint contents and private process receipts stay local.

From repository root with NumPy and PyTorch (CUDA is not needed for verification):

```powershell
python -X utf8 -B tools/export_delayed_credit.py --verify-only
```

NPZ decoding: for correct/original_correct/flipped_correct use numpy.unpackbits
on the _packed array, bitorder="little", count=prod(_shape), then reshape.
valid_times and valid_through_step specify the finite prefix; never interpret a
discarded nonfinite suffix as an incorrect Boolean prediction.

The frozen protocol and source are under new/delayed_credit. A fresh experiment
uses run.py --check in a new qualification directory, then run.py --out in a new
run directory with --qualification pointing to that qualification.json. Training
requires the established Torch2.5.1 CUDA environment. No new training or inference
is performed by this exporter. The K64 control failed qualification, so these
results do not identify a K8-versus-K64 learnability gap or its mechanism.
''',encoding='utf-8')
    publication = {'schema':'delayed-credit-publication-v0','execution_status':'COMPLETE',
        'verdict':summary['verdict'],'trajectories':12,'checkpoint_bindings':72,
        'source_sha256':{**raw_manifest['source_sha256'],'tools/export_delayed_credit.py':A.sha(Path(__file__))},
        'original_manifest_sha256':A.sha(RUN/'manifest.json'),
        'publication_scope':'Code, unchanged summaries, exact banks, full valid Boolean trajectories, all training updates and checkpoint provenance; no weights or private process records.',
        'published_artifacts_sha256':{p.relative_to(ROOT).as_posix():A.sha(p) for p in PUBLIC.rglob('*') if p.is_file()}}
    A.write_json(MANIFEST,publication)
    return verify_public()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    import json
    print(json.dumps(verify_public() if args.verify_only else export()),flush=True)
