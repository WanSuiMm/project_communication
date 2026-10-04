"""Publish and verify saved state-factorization artifacts without model execution."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

import numpy as np
from export_continuation_interface import read, write, sha, safe, PRIVATE_TEXT

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/state_factorization_20261004_01'
PUBLIC = ROOT/'evidence/state_factorization_20261004'
MANIFEST = ROOT/'STATE_FACTORIZATION_PUBLICATION_MANIFEST.json'
ARMS = ('FF','F-span','F-null','FS','SF','S-span','S-null','SS')
PROTOCOL = 'output_preserving_state_factorization_v1'


def helper():
    spec = importlib.util.spec_from_file_location('saved_factor_metrics',ROOT/'new/state_factorization/metrics.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def equal(a,b):
    if isinstance(b,dict):
        assert set(a)==set(b)
        for k in b: equal(a[k],b[k])
    elif isinstance(b,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    elif isinstance(b,float):
        assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12),(a,b)
    else: assert a==b,(a,b)


def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def tensor_hash(values):
    digest=hashlib.sha256()
    for key,array in sorted(values.items()):
        digest.update(key.encode());digest.update(str((array.shape,array.dtype)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def compact(raw):
    out = {k:raw[k] for k in ('protocol','status','training_updates','completed_cells','qualification',
                            'parameters_unchanged','elapsed_seconds','claim_boundary')}
    out['primary_size']=32;out['confirmation_size']=64
    out['support_note']={'32':{'prog_nonempty_maps':6,'prog_cells':231,'minimum_maps':16},
                         '64':{'keep_nonempty_maps':9,'keep_cells':393,'minimum_maps':16}}
    out['arms']={}
    for arm,r in raw['records'].items():
        out['arms'][arm]={}
        for size,record in r['sizes'].items():
            c=record['cohorts'];diag=r['handoff'][size]
            out['arms'][arm][size]={'rescue_proxy':r['rescue'][size],
                'keep_continuous':c['keep']['continuous64_256']['pooled'],
                'prog_immediate':c['prog']['immediate64']['pooled'],
                'prog_sustained':c['prog']['sustained241_256']['pooled'],
                'prog_delayed_sustained':c['prog']['delayed_sustained']['pooled'],
                'rescue_immediate':c['rescue']['immediate64']['pooled'],
                'all_changed':record['all_changed'],
                'readout_neutral_arm':diag['readout_neutral_arm'],
                'readout_property_qualified':diag['readout_property_qualified'],
                'max_absolute_logit_error':diag['max_absolute_logit_error'],
                'max_normalized_logit_error':diag['max_normalized_logit_error'],
                'open_decision_disagreements':diag['open_decision_disagreements']}
    return out


def scientific_checks(folder,raw_name='summary.json'):
    raw=read(folder/raw_name);metrics=helper()
    bindings=read(folder/('manifest.json' if (folder/'manifest.json').exists() else 'config.json'))
    assert raw['protocol']==PROTOCOL and raw['status']=='COMPLETE'
    assert raw['training_updates']==0 and raw['completed_cells']==16 and raw['parameters_unchanged']
    assert tuple(raw['records'])==ARMS
    native={};fixed={};banks={};saved_logits={}
    for size in (32,64):
        d=arrays(folder/'banks'/f'test{size}.npz');banks[size]=d
        assert d['x'].shape==(32,3,size,size) and np.isfinite(d['x']).all()
        assert tensor_hash(d)==bindings['banks'][str(size)]['data_sha256']
        native[size]={}
        for mid in ('S1','F1'):
            t=arrays(folder/'native'/f'{mid}_size{size}_trace.npz')
            assert t['correct'].dtype==np.bool_ and t['correct'].shape==(257,32,size,size)
            assert np.array_equal(t['correct'],t['original_correct']&t['flipped_correct'])
            native[size][mid]=t
        fixed[size]=metrics.cohorts(native[size]['S1']['correct'][64],native[size]['F1']['correct'][64],d['changed'][:,0].astype(bool))
        stored=arrays(folder/'cohorts'/f'size{size}.npz')
        assert set(stored)==set(fixed[size]) and all(np.array_equal(stored[k],v) for k,v in fixed[size].items())
        snapshots={k:arrays(folder/'states'/f'{k}_size{size}_T64.npz') for k in ('S1','F1')}
        assert all(s['W'].shape==(64,24,size,size) and s['Z'].shape==(64,8,size,size) for s in snapshots.values())
        assert all(np.isfinite(v).all() for s in snapshots.values() for v in s.values())
        for mid,snapshot in snapshots.items():
            assert tensor_hash(snapshot)==bindings['snapshot_sha256'][mid][str(size)]
        p=arrays(folder/f'projection_size{size}.npz');r=p['weight']
        assert r.shape==(8,) and float(r@r)>0
        assert np.allclose(p['projector'],np.outer(r,r)/(r@r),rtol=1e-12,atol=1e-12)
        assert np.allclose(p['projector']@p['projector'],p['projector'],rtol=1e-12,atol=1e-12)
        saved_logits[size]={}
        for arm in ARMS:
            record=raw['records'][arm]
            assert record==read(folder/'arms'/f'{arm}.json')
            t=arrays(folder/'arms'/f'{arm}_size{size}_trace.npz')
            assert t['correct'].shape==(193,32,size,size) and t['correct'].dtype==np.bool_
            assert np.array_equal(t['times'],np.arange(64,257))
            assert np.array_equal(t['correct'],t['original_correct']&t['flipped_correct'])
            observed=metrics.summarize(t['correct'],fixed[size])
            saved=record['sizes'][str(size)]
            equal(observed,{k:saved[k] for k in ('cohorts','all_changed')})
            gate=metrics.rescue_gate(observed,raw['records']['FF']['sizes'][str(size)])
            diag=record['handoff'][str(size)]
            if diag['readout_neutral_arm']:
                gate['output_neutrality_qualified']=diag['readout_property_qualified']
                gate['pass']=gate['pass'] and gate['output_neutrality_qualified']
            equal(gate,record['rescue'][str(size)])
            assert diag['instrumented_step_exact']
            if arm=='FF':
                assert all(np.array_equal(t[k],native[size]['F1'][k][64:]) for k in ('correct','original_correct','flipped_correct'))
            z=arrays(folder/'arms'/f'{arm}_size{size}_logits.npz')
            assert set(z)=={'T64','T128','T256'} and all(v.shape==(2,32,size,size) and np.isfinite(v).all() for v in z.values())
            saved_logits[size][arm]=z['T64']
            norms=arrays(folder/'arms'/f'{arm}_size{size}_norms.npz')
            assert norms['norms'].shape==(193,2,32,2) and np.isfinite(norms['norms']).all()
            assert np.array_equal(norms['times'],np.arange(64,257))
        mask=np.stack([d['mask'][:,0].astype(bool)]*2)
        for arm in ARMS:
            diag=raw['records'][arm]['handoff'][str(size)]
            expected=saved_logits[size]['FF' if diag['readout_neutral_arm'] else 'SS']
            actual=saved_logits[size][arm]
            err=np.abs(actual-expected)
            absolute=err.max(axis=(2,3));normalized=absolute/(1+np.abs(expected).max(axis=(2,3)))
            disagreements=int((((actual>=0)!=(expected>=0))&mask).sum())
            equal(float(absolute.max()),diag['max_absolute_logit_error'])
            equal(float(normalized.max()),diag['max_normalized_logit_error'])
            assert disagreements==diag['open_decision_disagreements']
            assert diag['readout_property_qualified']==(float(normalized.max())<=1e-6 and disagreements==0)
            if arm in ('FF','SF'):assert np.array_equal(actual,expected)
    for mid in ('S1','F1'):
        assert read(folder/'native'/f'{mid}_phenotype.json')['phenotype_gate']['pass']==raw['qualification']['native_full_flags'][mid]
    flags=raw['qualification']['native_full_flags']
    for size,key in (('32','primary_status'),('64','confirmation_status')):
        eligible=flags['S1'] and not flags['F1'] and raw['records']['SS']['rescue'][size]['pass']
        expected=('QUALIFIED_LOCALIZATION_EPISODE' if eligible else 'LOCALIZATION_EPISODE_UNQUALIFIED') if size=='32' else ('QUALIFIED' if eligible else 'UNQUALIFIED')
        assert raw['qualification'][key]==expected
    with (folder/'metrics.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==560
    for row in rows:
        v=raw['records'][row['arm']]['sizes'][row['size']]['cohorts'][row['cohort']][row['metric']]
        assert int(row['numerator'])==v['pooled']['numerator'] and int(row['denominator'])==v['pooled']['denominator']
        for field,value in (('pooled',v['pooled']['value']),('map_mean',v['map_mean'])):
            equal(None if row[field]=='' else float(row[field]),value)
        assert int(row['valid_maps'])==v['valid_maps']
    assert len(list(folder.rglob('*.npz')))==66
    return {'status':'PASS','scope':'Saved-data arithmetic and hashes only; no inference, training, or optimizer updates.',
            'intervention_cells':16,'npz_files':66,'metric_csv_rows':560,'fixed_cohorts_and_counts_recomputed':True,
            'rescue_gates_and_episode_recomputed':True,'FF_native_suffix_exact':True,'actual_neutrality_recomputed':True,
            'source_and_original_snapshot_bindings_checked':True,'native_full_flags':flags,
            'primary_status':raw['qualification']['primary_status'],'confirmation_status':raw['qualification']['confirmation_status'],
            'elapsed_seconds':raw['elapsed_seconds']}


def verify():
    m=read(MANIFEST);safe(m)
    for category in ('source_sha256','publication_tools_sha256','published_sha256'):
        for rel,digest in m[category].items():assert sha(ROOT/rel)==digest,rel
    for p in PUBLIC.rglob('*.json'):safe(read(p))
    result=scientific_checks(PUBLIC,'raw_summary.json')
    assert result==read(PUBLIC/'validation.json')
    assert compact(read(PUBLIC/'raw_summary.json'))==read(PUBLIC/'summary.json')
    print(json.dumps(result))


def build():
    assert not PUBLIC.exists() and not MANIFEST.exists(),'New publication directory required'
    m,status=read(RUN/'manifest.json'),read(RUN/'status.json')
    assert m['status']==status['status']=='COMPLETE' and status['completed_cells']==16
    for rel,digest in m['source_sha256'].items():
        assert sha(ROOT/rel)==sha(RUN/'source'/rel)==digest,rel
    validation=scientific_checks(RUN);raw=read(RUN/'summary.json')
    PUBLIC.mkdir(parents=True);raw_bindings={}
    for p in RUN.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(RUN)
        if 'source' in rel.parts or rel.as_posix() in ('manifest.json','status.json'):continue
        assert p.suffix in ('.json','.csv','.npz','.png','.pdf','.md'),str(rel)
        if p.suffix=='.json':safe(read(p))
        if p.suffix in ('.csv','.md'):assert not PRIVATE_TEXT.search(p.read_text(encoding='utf-8-sig'))
        dest=PUBLIC/({'summary.json':'raw_summary.json','RESULTS.md':'raw_RESULTS.md','config.json':'raw_config.json'}.get(rel.as_posix(),rel.as_posix()))
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        raw_bindings[rel.as_posix()]={'sha256':sha(p),'public_path':dest.relative_to(ROOT).as_posix()}
    write(PUBLIC/'summary.json',compact(raw));write(PUBLIC/'validation.json',validation)
    config=read(PUBLIC/'raw_config.json')
    config['source_sha256']=m['source_sha256'];config['snapshot_sha256']=m['snapshot_sha256']
    config['snapshot_cue_order']=m['snapshot_cue_order'];config['excluded_bank_sha256']=m['excluded_bank_sha256']
    config['started_utc']=m['started_utc'];config['finished_utc']=m['finished_utc']
    safe(config);write(PUBLIC/'config.json',config)
    text=(RUN/'RESULTS.md').read_text(encoding='utf-8')
    note='''
## Qualification denominators (publication annotation)

Native Full: S1=True, F1=False. All sixteen intervention rescue proxies are
unqualified by the frozen cohort-support rule. Size32 shared-unsolved covers
6/32 maps and231 cells; size64 shared-solved covers9/32 maps and393 cells.
The frozen minimum is16 nonempty maps and100 cells for BOTH keep/prog.
Thus the gate flags are not a claim that all continuous behaviors failed.
Size64 is confirmation and does not replace the primary eligibility check.

SF keeps the immediate F1 output unchanged. Its continuous keep rate is1.0
at both sizes; shared-unsolved sustained/delayed rates are.5152 (32) and.6730
(64). These selected-checkpoint measurements remain descriptive because the
frozen support qualification failed. No population or mechanism claim follows.

Read summary.json (compact derived aggregate), metrics.csv, validation.json
and config.json first. raw_summary.json (4.6MB) and raw_RESULTS.md preserve the
original run summaries unchanged. Arm JSON, native records and66 NPZ files
retain all per-map cohorts, counts, traces, logits, scales and native states.
'''
    (PUBLIC/'RESULTS.md').write_text(text+note,encoding='utf-8')
    (PUBLIC/'REPRODUCTION.md').write_text('''# State-factorization saved evidence

Start with RESULTS.md, summary.json, metrics.csv, validation.json, config.json
and state_factorization.png. Open raw_summary.json and per-arm JSON only after
the compact aggregate. Frozen protocol: new/state_factorization/PROTOCOL.md.

Saved-data verification from repository root (NumPy only; no model execution):

    python -X utf8 -B tools/export_state_factorization.py --verify-only

The verifier recomputes fixed cohorts from native paired traces, all16 arm
counts, support gates, immediate output comparisons and the560 CSV rows.
It checks FF/native suffix equality, source/data hashes and66 NPZ files.
Native parameter immutability and first-step equivalence are recorded checks,
not independently reproduced model executions by this verifier.

NPZ inventory:2 fresh banks,2 fixed-cohort masks,2 readout projectors,
4 native T64 state captures,4 native trace banks,4 native endpoint-logit banks,
16 intervention trace banks,16 intervention endpoint-logit banks and16 norm
trajectories. State captures are scientific activation arrays, not model
checkpoints. Projection is pointwise across Z channels using F1's weight.
Original/flip state order is all original maps followed by all flipped maps.

A full rerun requires the exact S1/F1 checkpoints identified by SHA in
config.json, historical Torch2.5.1, CUDA and matplotlib. Checkpoints and machine
receipts remain local. The original run and source snapshot were not changed.
The map-support failure remains frozen; no new maps, training or rescue run
was added during publication.
''',encoding='utf-8')
    write(MANIFEST,{'protocol':'state_factorization_publication_v1','run_id':RUN.name,
        'source_sha256':m['source_sha256'],
        'publication_tools_sha256':{p:sha(ROOT/p) for p in ('tools/export_state_factorization.py','tools/export_continuation_interface.py')},
        'published_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in PUBLIC.rglob('*') if p.is_file()},
        'raw_artifact_bindings':raw_bindings,'original_manifest_sha256':sha(RUN/'manifest.json'),
        'original_status_sha256':sha(RUN/'status.json'),
        'excluded':['model checkpoints','source snapshots','machine manifests','PIDs','launch receipts','logs']})
    verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    verify() if args.verify_only else build()
