"""Freeze and CPU-verify saved RRC evidence, including an interrupted partial run."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from export_continuous_coverage import (MAX_FILE_BYTES, bank_tensor_hash, compare_trace_metrics,
    copy_bound, decode_trace, gzip_json, joint_pass, load_bank, read, safe, sha, trace_metrics, write)

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/rrc_v0_20261007_01'
PUBLIC=ROOT/'evidence/rrc_v0_20261007_partial01'
MANIFEST=ROOT/'RRC_V0_PARTIAL01_PUBLICATION_MANIFEST.json'
QUALIFICATION=ROOT/'analyses/rrc_v0_qualification_20261007_02.json'
PROTOCOL='rrc_v0_prestream_relation_factorization_v1'


def module(relative,name):
    sys.path.insert(0,str(ROOT/'new'))
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def key(row):
    return row['block'],row['arm'],row['update']


def frozen_json(path,value):
    safe(value)
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    path.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))


def measure(publication):
    rows=gzip_json(PUBLIC/'dense.json.gz')
    final=gzip_json(PUBLIC/'perarm.json.gz')
    config=read(PUBLIC/'config.json')
    assert config['protocol']==PROTOCOL
    assert len(rows)==publication['completed_stage_records']
    assert len(final)==publication['completed_u300_trajectories']
    assert len({key(row) for row in rows})==len(rows)
    assert {key(r) for r in final}=={key(r) for r in rows if r['update']==300}
    banks={s:load_bank(PUBLIC/'banks'/f'evaluation{s}.npz') for s in (32,64)}
    for label in ('train','evaluation32','evaluation64'):
        bank=load_bank(PUBLIC/'banks'/f'{label}.npz')
        assert bank_tensor_hash(bank)==publication['data_sha256'][label]
    plans=gzip_json(PUBLIC/'plans.json.gz')
    assert set(plans)=={str(b) for b in range(8)}
    for b in range(8):
        plan=plans[str(b)]
        assert plan['initialization_seed']==120001+b and plan['schedule_seed']==121001+b
        indices=np.asarray(plan['batch_indices'])
        assert indices.shape==(300,8) and indices.min()>=0 and indices.max()<512
    metrics=module('new/continuous_coverage/metrics.py','_rrc_export_metrics')
    reporter=module('new/rrc_v0/reporting.py','_rrc_export_reporter')
    checked_final_traces=0
    for row in rows:
        stage=PUBLIC/Path(row['evaluation_summary']).parent
        summary=gzip_json(stage/'summary.json.gz')
        assert metrics.compact_pair_metrics(summary)==row['metrics'],key(row)
        assert metrics.joint_readiness(summary)==row['joint'],key(row)
        action=gzip_json(stage/'relation_activity.json.gz')
        assert len(action['rows'])==32
        if row['arm']=='rrc':
            assert action['parameters']['rho'] is not None
        else:
            assert all(r['relation_action_energy']==0 for r in action['rows'])
        for size in (32,64):
            path=stage/f'size{size}_traces.npz'
            with np.load(path,allow_pickle=False) as packed:
                for name in ('correct','original_correct','flipped_correct'):
                    assert tuple(packed[name+'_shape'])==(257,32,size,size)
                    assert packed[name+'_packed'].dtype==np.uint8
                    assert packed[name+'_packed'].size==(257*32*size*size+7)//8
            if row['update']==300:
                decoded=decode_trace(path,size)
                assert np.array_equal(decoded['correct'],decoded['original_correct'] & decoded['flipped_correct'])
                actual=trace_metrics(decoded['correct'],banks[size])
                compare_trace_metrics(actual,row['metrics']['sizes'][str(size)],str(key(row)))
                if size==32:
                    assert joint_pass(actual)==row['joint']['pass']
                checked_final_traces+=1
    for rel,limit in publication['training_curve_updates'].items():
        curve=gzip_json(PUBLIC/rel)
        assert len(curve)==limit and [r['update'] for r in curve]==list(range(1,limit+1))
        for row in curve:
            assert row['forward_steps']==256 and row['backward_calls']==32
            assert row['optimizer_steps']==1 and row['credit_horizon']==8
    aggregate=reporter.aggregate(final,rows,expected_blocks=8)
    published=read(PUBLIC/'summary.json')
    assert aggregate['primary_comparison']==published['primary_comparison']
    assert published['status']=='INCOMPLETE' and published['overall_verdict']=='INCOMPLETE'
    assert published['primary_comparison']['reliability_qualified'] is None
    csv_count=sum(1 for _ in (PUBLIC/'metrics.csv').open(encoding='utf-8'))-1
    assert csv_count==2*len(rows)
    return {'status':'PASS','saved_data_only':True,'training_or_inference_performed':False,
        'source_binding_status':'PASS','stage_summaries_checked':len(rows),
        'packed_trace_shapes_checked':2*len(rows),'u300_trace_metrics_recomputed':checked_final_traces,
        'aggregate_status':published['status'],'CSV_size_rows':csv_count,
        'limitations':['Old Full is retained from the saved evaluator, not rerun.',
            'No inference or optimizer update; uncompleted stages are excluded.']}


def verify_files(publication):
    safe(publication)
    actual={p.relative_to(ROOT).as_posix():sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
    assert actual==publication['published_sha256']
    for rel,digest in publication['source_sha256'].items():
        assert sha(ROOT/rel)==digest,rel
    for rel,digest in publication['verification_source_sha256'].items():
        assert sha(ROOT/rel)==digest,rel
    for rel,binding in publication['raw_artifact_bindings'].items():
        path=ROOT/binding['public_path']
        raw=gzip.decompress(path.read_bytes()) if binding['encoding']=='gzip lossless' else path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==binding['sha256'],rel
    assert not any(p.suffix in ('.pt','.pth','.log','.pid') for p in PUBLIC.rglob('*'))
    assert all((ROOT/rel).stat().st_size<MAX_FILE_BYTES for rel in actual)
    for rel in actual:
        path=ROOT/rel
        if path.suffix=='.json':
            safe(read(path))
        elif path.suffix=='.gz' and path.name.endswith('.json.gz'):
            safe(gzip_json(path))


def build():
    assert not PUBLIC.exists() and not MANIFEST.exists(),'Never overwrite a publication snapshot'
    # Capture the atomic completed-record index ONCE. Never publish live writes.
    rows=read(RUN/'dense.json')
    final=[r for r in rows if r['update']==300]
    status=read(RUN/'status.json')
    original=read(RUN/'manifest.json')
    config=read(RUN/'config.json')
    qualification=read(QUALIFICATION)
    assert config['protocol']==original['protocol']==PROTOCOL
    assert qualification['status']=='PASS' and sha(QUALIFICATION)==original['qualification_sha256']
    assert len(rows)<312 and len(final)<24,'This publication is explicitly partial'
    for rel,digest in original['source_sha256'].items():
        assert sha(RUN/'source'/rel)==digest and sha(ROOT/rel)==digest,rel
    PUBLIC.mkdir(parents=True)
    frozen_json(PUBLIC/'dense.json.gz',rows)
    frozen_json(PUBLIC/'perarm.json.gz',final)
    write(PUBLIC/'config.json',config)
    bindings={}
    for rel in ('plans.json','banks/train.npz','banks/evaluation32.npz','banks/evaluation64.npz'):
        encoding='gzip lossless' if rel.endswith('.json') else 'byte-exact copy'
        dest=PUBLIC/(rel+'.gz' if encoding=='gzip lossless' else rel)
        if rel.endswith('.json'):
            safe(read(RUN/rel))
        copy_bound(RUN/rel,dest,bindings,rel,encoding)
    checkpoint_bindings={}
    last={}
    for row in rows:
        cp=RUN/row['checkpoint']
        assert sha(cp)==row['checkpoint_sha256']
        checkpoint_bindings[row['checkpoint']]={'file_sha256':row['checkpoint_sha256'],
            'parameter_sha256':row['parameter_sha256']}
        stage=Path(row['evaluation_summary']).parent
        for p in (RUN/stage).iterdir():
            if not p.is_file() or p.suffix not in ('.json','.npz','.csv'):
                continue
            encoding='byte-exact copy' if p.suffix=='.npz' else 'gzip lossless'
            if p.suffix=='.json':
                safe(read(p))
            rel=p.relative_to(RUN).as_posix()
            dest=PUBLIC/(rel if encoding=='byte-exact copy' else rel+'.gz')
            copy_bound(p,dest,bindings,rel,encoding)
        pair=(row['block'],row['arm'])
        last[pair]=max(last.get(pair,0),row['update'])
    curve_updates={}
    for (block,arm),update in last.items():
        path=f'block{block:02d}/{arm}/training_curve.json.gz'
        source=RUN/path.removesuffix('.gz')
        curve=read(source) if source.exists() else []
        trimmed=[r for r in curve if r['update']<=update]
        assert len(trimmed)==update
        frozen_json(PUBLIC/path,trimmed)
        curve_updates[path]=update
    reporter=module('new/rrc_v0/reporting.py','_rrc_partial_reporter')
    aggregate=reporter.report(PUBLIC,final,rows,status={'status':'PARTIAL_INTERRUPTED_SNAPSHOT'})
    aggregate.update(protocol=PROTOCOL,snapshot_utc=datetime.now(timezone.utc).isoformat(),
        run_id=RUN.name,execution_observation='Last progress is stale; original tool session unavailable and original worker not found.',
        last_saved_progress_utc=status['updated_utc'],completed_arms=len(final),expected_arms=24,
        completed_dense_records=len(rows),expected_dense_records=312,final_result_available=False)
    write(PUBLIC/'summary.json',aggregate)
    write(PUBLIC/'runtime_qualification.json',qualification)
    text=(PUBLIC/'RESULTS.md').read_text(encoding='utf-8')
    prefix=('# Publication status: PARTIAL / INTERRUPTED\n\n'
        f'Fixed snapshot: **{len(final)}/24 u300 trajectories, {len(rows)}/312 complete stage evaluations**. '
        'This is not the final experiment. The saved worker status was RUNNING, but progress stopped; '
        'its original tool session is unavailable and the original worker was not found. '
        'Do not infer final no-qualification from these partial counts. No run was restarted for this upload.\n\n')
    (PUBLIC/'RESULTS.md').write_text(prefix+text,encoding='utf-8')
    (PUBLIC/'REPRODUCTION.md').write_text('''# RRC partial snapshot verification

Read RESULTS.md and summary.json first. This package is incomplete and includes
only fully recorded stages. Training curves are frozen through each last recorded
checkpoint; a partly written subsequent evaluation is excluded. Dense and curve
JSON is losslessly gzipped; traces retain every integer T0..T256 as packed bits.
Model/optimizer contents, private receipts, logs and original machine manifest
stay local. Checkpoint hashes bind the omitted checkpoint contents.

From the repository root:

```
python -X utf8 -B tools/export_rrc_v0.py --verify-only
```

This is a NumPy/CPU saved-data check, with no inference, training or optimizer
updates. It checks source/file/raw hashes, all saved evaluator summaries, trace
shapes, u300 paired correctness and reach/retention, bank/schedule bindings,
training cadence, and the unchanged partial reporter aggregate. It does not rerun
the saved old Full gate. The full experiment recipe is in
[PROTOCOL.md](../../new/rrc_v0/PROTOCOL.md).
''',encoding='utf-8')
    publication={'schema':'rrc-v0-partial-publication-v1','run_id':RUN.name,
        'experiment_protocol':PROTOCOL,'publication_status':'PARTIAL_INTERRUPTED',
        'git_review_base':original['git_review_base'],'completed_u300_trajectories':len(final),
        'completed_stage_records':len(rows),'snapshot_utc':aggregate['snapshot_utc'],
        'last_saved_progress_utc':status['updated_utc'],'source_sha256':original['source_sha256'],
        'data_sha256':original['data_sha256'],'schedule_plan_sha256':original['schedule_plan_sha256'],
        'qualification_sha256':original['qualification_sha256'],'checkpoint_bindings':checkpoint_bindings,
        'raw_artifact_bindings':bindings,'training_curve_updates':curve_updates,
        'verification_source_sha256':{p:sha(ROOT/p) for p in ('tools/export_rrc_v0.py','tools/export_continuous_coverage.py')},
        'excluded':['checkpoint and optimizer contents','private manifest, status, receipts, PIDs and logs',
            'unfinished evaluations and unsaved optimizer updates']}
    validation=measure(publication)
    write(PUBLIC/'validation.json',validation)
    published={p.relative_to(ROOT).as_posix():sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
    publication.update(published_sha256=published,published_file_count=len(published),
        published_bytes=sum((ROOT/p).stat().st_size for p in published))
    write(MANIFEST,publication)
    verify_files(publication)
    print(json.dumps({'status':'PASS','publication':'PARTIAL_INTERRUPTED','u300':len(final),
        'records':len(rows),'files':len(published),'MiB':publication['published_bytes']/2**20}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        publication=read(MANIFEST)
        verify_files(publication)
        assert measure(publication)==read(PUBLIC/'validation.json')
        print(json.dumps({'status':'PASS','publication':'PARTIAL_INTERRUPTED',
            'records':publication['completed_stage_records']}))
    else:
        build()


if __name__=='__main__':
    main()
