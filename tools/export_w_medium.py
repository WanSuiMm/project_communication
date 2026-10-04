"""Losslessly publish and verify the completed W-medium audit; NumPy only."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np

from export_continuation_interface import read, write, sha, safe, PRIVATE_TEXT
from export_state_factorization import arrays, equal, tensor_hash

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/w_medium_20261004_01'
PUBLIC=ROOT/'evidence/w_medium_20261004'
MANIFEST=ROOT/'W_MEDIUM_PUBLICATION_MANIFEST.json'
ARMS=('FF','SF','FS','SS','SF_norm_F','FF_norm_S','SF_payload','SF_payload_gauge',
      'SF_lane','SF_lane_gauge','SF_spatial','SF_spatial_gauge','SF_cue_swap',
      'SF_time32','SF_time48','SF_time80','SF_time96','SF_norm_F_cue_swap')
PROTOCOL='w_medium_qualification_v1'


def metric_helper():
    spec=importlib.util.spec_from_file_location('published_W_cohorts',ROOT/'new/state_factorization/metrics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def compact(raw):
    out={k:raw[k] for k in ('protocol','status','training_updates','completed_cells','maximum_cells',
                           'stage1_gate','native_full_flags','skipped_cells','parameters_unchanged',
                           'elapsed_seconds','claim_boundary')}
    out['primary_size']=32;out['confirmation_size']=64
    out['arms']={}
    for arm,row in raw['records'].items():
        out['arms'][arm]={}
        for size,r in row['sizes'].items():
            c=r['cohorts'];diag=row['handoff'][size]
            contrasts={key:{k:v for k,v in value.items() if k!='per_map_delta'}
                       for key,value in row['contrast_vs_SF'][size].items()}
            out['arms'][arm][size]={'rescue_proxy':row['rescue'][size],
                'keep_continuous':c['keep']['continuous64_256']['pooled'],
                'prog_immediate':c['prog']['immediate64']['pooled'],
                'prog_sustained':c['prog']['sustained241_256']['pooled'],
                'prog_delayed_sustained':c['prog']['delayed_sustained']['pooled'],
                'all_changed':{key:{k:v for k,v in rate.items() if k!='per_map'}
                               for key,rate in r['all_changed'].items()}, 'contrast_vs_SF':contrasts,
                'readout_neutral_arm':diag['readout_neutral_arm'],'exact_FF_output':diag['exact_FF_output'],
                'max_absolute_logit_error_vs_FF':diag['max_absolute_logit_error_vs_FF'],
                'open_decision_disagreements_vs_FF':diag['open_decision_disagreements_vs_FF'],
                'exact_gauge_equivalence_to_SF':diag.get('exact_gauge_equivalence_to_SF')}
    return out


def split_state(p):
    """32-map paired-cue chunks, separately by donor time; no quantization."""
    values=arrays(p);record={'original_sha256':sha(p),'tensor_sha256':tensor_hash(values),
        'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in values.items()},
        'cue_order':'all_original_maps_then_all_flipped_maps','maps_per_chunk':32,'chunks':[]}
    donor='times' in values
    times=values['times'].tolist() if donor else [None]
    if donor:record['times']=times
    folder=PUBLIC/'states'/p.stem;folder.mkdir(parents=True)
    for ti,t in enumerate(times):
        for start in range(0,128,32):
            stop=start+32;idx=np.r_[start:stop,start+128:stop+128]
            chunk={k:(v[ti,idx] if donor else v[idx]) for k,v in values.items() if k!='times'}
            name=f'{"T"+str(t)+"_" if donor else ""}maps{start:03d}_{stop-1:03d}.npz'
            dest=folder/name;np.savez_compressed(dest,**chunk)
            assert dest.stat().st_size<50*1024**2
            record['chunks'].append({'path':dest.relative_to(PUBLIC).as_posix(),'map_start':start,
                'map_stop':stop,'time_index':ti if donor else None,'time':t,'sha256':sha(dest)})
    return record


def state_values(folder,name,index):
    record=index[name]
    values={k:np.empty(v['shape'],np.dtype(v['dtype'])) for k,v in record['arrays'].items()}
    if 'times' in values:values['times'][:]=record['times']
    for chunk in record['chunks']:
        p=folder/chunk['path'];assert sha(p)==chunk['sha256']
        part=arrays(p);start,stop=chunk['map_start'],chunk['map_stop']
        idx=np.r_[start:stop,start+128:stop+128]
        for key,v in part.items():
            if chunk['time_index'] is None:values[key][idx]=v
            else:values[key][chunk['time_index'],idx]=v
    assert all(np.isfinite(v).all() for v in values.values())
    assert tensor_hash(values)==record['tensor_sha256'],name
    return values


def contrast(result,reference,seed):
    output={}
    for cohort,metric in (('keep','continuous64_256'),('prog','sustained241_256'),('prog','delayed_sustained')):
        a,b=(r['cohorts'][cohort][metric]['per_map'] for r in (result,reference))
        eligible=[(x['map_index'],x['value']-y['value']) for x,y in zip(a,b) if x['denominator']>0 and y['denominator']>0]
        values=np.asarray([v for _,v in eligible])
        if len(values):
            rng=np.random.default_rng(seed)
            ci=np.quantile(values[rng.integers(len(values),size=(2000,len(values)))].mean(1),[.025,.975]).tolist()
        else:ci=[None,None]
        output[f'{cohort}.{metric}']={'map_mean_delta':float(values.mean()) if len(values) else None,
            'eligible_maps':len(values),'bootstrap_95_interval':ci,'per_map_delta':[list(x) for x in eligible],
            'bootstrap_replicates':2000,'bootstrap_seed':seed,
            'interpretation':'descriptive paired map interval; not seed reliability or mechanistic significance'}
    return output


def scientific_checks():
    raw=read(PUBLIC/'raw_summary.json');cfg=read(PUBLIC/'config.json');metric=metric_helper()
    assert raw['protocol']==PROTOCOL and raw['status']=='COMPLETE'
    assert raw['training_updates']==0 and raw['completed_cells']==raw['maximum_cells']==36
    assert raw['skipped_cells']==0 and raw['parameters_unchanged'] and tuple(raw['records'])==ARMS
    index=read(PUBLIC/'state_chunks.json');assert len(index)==6
    supports={}
    for size in (32,64):
        key=str(size);d=arrays(PUBLIC/'banks'/f'test{size}.npz')
        assert d['x'].shape==(128,3,size,size) and tensor_hash(d)==cfg['banks'][key]['data_sha256']
        native={}
        for mid in ('S1','F1'):
            trace=arrays(PUBLIC/'native'/f'{mid}_size{size}_trace.npz')
            assert trace['correct'].shape==(257,128,size,size) and trace['correct'].dtype==np.bool_
            assert np.array_equal(trace['correct'],trace['original_correct']&trace['flipped_correct'])
            native[mid]=trace
            states=state_values(PUBLIC,f'{mid}_size{size}_T64.npz',index)
            assert states['W'].shape==(256,24,size,size) and states['Z'].shape==(256,8,size,size)
            assert tensor_hash(states)==cfg['snapshot_sha256'][mid][key]
            del states
        donor=state_values(PUBLIC,f'S1_size{size}_donor_times.npz',index)
        assert donor['W'].shape==(4,256,24,size,size) and np.array_equal(donor['times'],[32,48,80,96])
        del donor
        fixed=metric.cohorts(native['S1']['correct'][64],native['F1']['correct'][64],d['changed'][:,0].astype(bool))
        stored=arrays(PUBLIC/'cohorts'/f'size{size}.npz')
        assert set(fixed)==set(stored) and all(np.array_equal(v,stored[k]) for k,v in fixed.items())
        sf_trace=arrays(PUBLIC/'arms'/f'SF_size{size}_trace.npz')
        sf_logits=arrays(PUBLIC/'arms'/f'SF_size{size}_logits.npz')
        ff_logits=arrays(PUBLIC/'arms'/f'FF_size{size}_logits.npz')
        for arm in ARMS:
            row=raw['records'][arm];assert row==read(PUBLIC/'arms'/f'{arm}.json')
            t=arrays(PUBLIC/'arms'/f'{arm}_size{size}_trace.npz')
            assert t['correct'].dtype==np.bool_ and t['correct'].shape==(193,128,size,size)
            assert np.array_equal(t['times'],np.arange(64,257))
            assert np.array_equal(t['correct'],t['original_correct']&t['flipped_correct'])
            observed=metric.summarize(t['correct'],fixed)
            equal(observed,{k:row['sizes'][key][k] for k in ('cohorts','all_changed')})
            gate=metric.rescue_gate(observed,raw['records']['FF']['sizes'][key])
            diag=row['handoff'][key]
            if diag['readout_neutral_arm']:gate['pass']=gate['pass'] and diag['exact_FF_output']
            equal(gate,row['rescue'][key])
            assert diag['instrumented_first_step_exact']
            equal(contrast(row['sizes'][key],raw['records']['SF']['sizes'][key],97100+size+ARMS.index(arm)),row['contrast_vs_SF'][key])
            z=arrays(PUBLIC/'arms'/f'{arm}_size{size}_logits.npz')
            assert set(z)=={'T64','T128','T256'}
            assert all(v.shape==(2,128,size,size) and np.isfinite(v).all() for v in z.values())
            err=np.abs(z['T64']-ff_logits['T64']);mask=np.stack([d['mask'][:,0].astype(bool)]*2)
            errors=float(err.max());disagreements=int((((z['T64']>=0)!=(ff_logits['T64']>=0))&mask).sum())
            equal(errors,diag['max_absolute_logit_error_vs_FF'])
            assert disagreements==diag['open_decision_disagreements_vs_FF']
            assert diag['exact_FF_output']==(errors==0 and disagreements==0)
            if diag['readout_neutral_arm']:assert diag['exact_FF_output']
            if arm=='FF':assert all(np.array_equal(t[k],native['F1'][k][64:]) for k in native['F1'])
            if arm.endswith('_gauge'):
                assert all(np.array_equal(t[k],sf_trace[k]) for k in sf_trace)
                assert all(np.array_equal(v,sf_logits[k]) for k,v in z.items())
                assert diag['exact_gauge_equivalence_to_SF']
            norms=arrays(PUBLIC/'arms'/f'{arm}_size{size}_norms.npz')
            assert norms['norms'].shape==(193,2,128,2) and np.isfinite(norms['norms']).all()
            assert np.array_equal(norms['times'],np.arange(64,257))
        supports[key]=raw['records']['SF']['rescue'][key]['observed']
        del native,sf_trace,sf_logits,ff_logits,t,z,norms
    flags=raw['native_full_flags']
    for mid in ('S1','F1'):
        assert read(PUBLIC/'native'/f'{mid}_phenotype.json')['phenotype_gate']['pass']==flags[mid]
    gate=raw['stage1_gate'];assert gate==read(PUBLIC/'stage1_gate.json')
    expected={}
    for size in ('32','64'):
        for arm in ('SS','SF'):expected[f'{arm}_rescue_size{size}']=raw['records'][arm]['rescue'][size]['pass']
        expected[f'SF_exact_output_size{size}']=raw['records']['SF']['handoff'][size]['exact_FF_output']
    assert gate['checks']==expected and gate['pass']==all(expected.values())
    historical={size:flags['S1'] and not flags['F1'] and raw['records']['SS']['rescue'][size]['pass'] for size in ('32','64')}
    assert gate['historical_localization_episode_qualified']==historical
    with (PUBLIC/'metrics.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==1260
    for row in rows:
        v=raw['records'][row['arm']]['sizes'][row['size']]['cohorts'][row['cohort']][row['metric']]
        assert int(row['numerator'])==v['pooled']['numerator'] and int(row['denominator'])==v['pooled']['denominator']
        for field,value in (('pooled',v['pooled']['value']),('map_mean',v['map_mean'])):
            equal(None if row[field]=='' else float(row[field]),value)
        assert int(row['valid_maps'])==v['valid_maps']
    assert len(list(PUBLIC.rglob('*.npz')))==182
    return {'status':'PASS','scope':'Saved-data arithmetic, exact trace/logit comparisons and hashes; no model execution.',
        'intervention_cells':36,'metric_csv_rows':1260,'npz_files':182,'lossless_state_chunks':48,
        'fixed_cohorts_counts_gates_recomputed':True,'paired_map_bootstrap_recomputed':True,
        'FF_native_suffix_exact':True,'actual_W_output_neutrality_recomputed':True,
        'all_six_gauge_suffixes_and_endpoint_logits_exact':True,
        'state_captures_reconstructed_tensor_hashes_exact':True,'native_full_flags_recorded':flags,
        'stage1_gate':gate,'cohort_support':supports,'training_updates':0,'elapsed_seconds':raw['elapsed_seconds']}


def verify(write_validation=False):
    m=read(MANIFEST);safe(m)
    for category in ('source_sha256','publication_tools_sha256','published_sha256'):
        for rel,digest in m[category].items():assert sha(ROOT/rel)==digest,rel
    for p in PUBLIC.rglob('*.json'):safe(read(p))
    result=scientific_checks()
    assert compact(read(PUBLIC/'raw_summary.json'))==read(PUBLIC/'summary.json')
    if write_validation:write(PUBLIC/'validation.json',result)
    else:assert result==read(PUBLIC/'validation.json')
    print(json.dumps({k:result[k] for k in ('status','intervention_cells','metric_csv_rows','npz_files','lossless_state_chunks','elapsed_seconds')}),flush=True)


def build():
    assert not PUBLIC.exists() and not MANIFEST.exists(), 'New publication directory required'
    m,status=read(RUN/'manifest.json'),read(RUN/'status.json');raw=read(RUN/'summary.json')
    assert m['status']==status['status']==raw['status']=='COMPLETE' and status['completed_cells']==36
    for rel,digest in m['source_sha256'].items():assert sha(ROOT/rel)==sha(RUN/'source'/rel)==digest,rel
    PUBLIC.mkdir(parents=True);bindings={};index={}
    excluded=('manifest.json','status.json','startup_execution.json')
    for p in sorted(RUN.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(RUN)
        if 'source' in rel.parts or rel.as_posix() in excluded:continue
        assert p.suffix in ('.json','.csv','.npz','.png','.pdf','.md')
        if rel.parts[0]=='states':
            index[p.name]=split_state(p)
            bindings[rel.as_posix()]={'sha256':sha(p),'public_state_index_key':p.name,
                                   'tensor_sha256':index[p.name]['tensor_sha256']}
            continue
        if p.suffix=='.json':safe(read(p))
        if p.suffix in ('.csv','.md'):assert not PRIVATE_TEXT.search(p.read_text(encoding='utf-8-sig'))
        rename={'summary.json':'raw_summary.json','RESULTS.md':'raw_RESULTS.md','config.json':'raw_config.json',
                'validation.json':'preflight_validation.json'}
        dest=PUBLIC/rename.get(rel.as_posix(),rel.as_posix());dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)
        bindings[rel.as_posix()]={'sha256':sha(p),'public_path':dest.relative_to(ROOT).as_posix()}
    write(PUBLIC/'state_chunks.json',index);write(PUBLIC/'summary.json',compact(raw))
    cfg=read(PUBLIC/'raw_config.json')
    for key in ('source_sha256','snapshot_sha256','snapshot_cue_order','excluded_bank_sha256','started_utc','finished_utc'):
        cfg[key]=m[key]
    cfg['preflight_sha256']=sha(PUBLIC/'preflight_validation.json')
    cfg['state_capture_format']='Lossless32-map original/flip chunks; donor times separated; see state_chunks.json'
    safe(cfg);write(PUBLIC/'config.json',cfg)
    notes='''
## Publication notes

Fresh128 maps at each size. Fixed keep/prog support at size32 is118/33 maps
(20,620/3,573 cells), and at size64 is39/123 maps (2,884/103,230 cells).
Both SF and SS pass the unchanged rescue/support thresholds; all six gauge
suffixes exactly reproduce SF. Native Full is a separate recorded qualification.

Read the compact summary, metrics, configuration and validation first. The
37MB raw_summary and per-arm JSON are secondary and preserve original counts,
per-map differences and every gate. Negative results are retained. Spatial
scrambling has high progress but FAILS size64 preservation (.9029<.95).
No exclusive W/Z role, pure phase, task semantics or learning guarantee follows.

Scientific states are losslessly split into48 map/time chunks to keep each file
below50MiB; state_chunks.json records exact reconstruction and tensor hashes.
No activation values, frozen run files, gates or model parameters were changed.
Checkpoints, machine manifests, PIDs, launch receipts and logs remain local.
'''
    (PUBLIC/'RESULTS.md').write_text((RUN/'RESULTS.md').read_text()+notes,encoding='utf-8')
    (PUBLIC/'REPRODUCTION.md').write_text('''# W medium saved evidence

Read RESULTS.md, summary.json, metrics.csv, validation.json, config.json and
w_medium.png first. Then frozen protocol new/w_medium/PROTOCOL.md and source
map GPT_CONTEXT.md. raw_summary.json and all arrays are secondary.

Saved-data verification from repository root (NumPy; no Torch/model execution):

    python -X utf8 -B tools/export_w_medium.py --verify-only

The verifier reconstructs all six original state captures from48 paired-cue
map/time chunks and checks exact tensor hashes. It recomputes fixed cohorts,
all36 arm counts/support gates,1260 CSV rows, paired-map contrasts/2000-sample
bootstrap intervals, instantaneous W-only neutrality, native FF suffix equality
and all six coordinate-gauge suffix/logit equalities. Original raw files copied
without changes are bound by SHA256. Native Full and parameter/first-step
immutability remain recorded checks; this verifier does not execute models.

182 NPZ files:2 fresh banks,2 fixed cohorts,8 native trace/logit banks,
108 intervention trace/logit/norm banks,14 transform/index files and48 state
chunks. States are activations, not checkpoints. Every chunk stores32 original
maps then the same32 flipped maps. Reconstruct into original full-cue order
using state_chunks.json; donor-time chunks are indexed by32/48/80/96.

A full rerun requires the exact S1/F1 checkpoints listed by SHA in config.json,
Torch2.5.1, CUDA and matplotlib. Follow new/w_medium/PROTOCOL.md with NEW output
names. Source hashes bind the frozen implementation and preflight qualification.
The original local run is unchanged. No extra inference, training or optimizer
update was performed for publication. Maps are not independent training seeds.
''',encoding='utf-8')
    tools=('tools/export_w_medium.py','tools/export_continuation_interface.py','tools/export_state_factorization.py')
    publication={'protocol':'w_medium_publication_v1','run_id':RUN.name,'source_sha256':m['source_sha256'],
        'publication_tools_sha256':{rel:sha(ROOT/rel) for rel in tools},
        'published_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in PUBLIC.rglob('*') if p.is_file()},
        'raw_artifact_bindings':bindings,'original_manifest_sha256':sha(RUN/'manifest.json'),
        'original_status_sha256':sha(RUN/'status.json'),
        'excluded':['checkpoints','source snapshots','machine manifests','PIDs','startup records','launch receipts','logs']}
    write(MANIFEST,publication)
    verify(write_validation=True)
    publication['published_sha256']['evidence/w_medium_20261004/validation.json']=sha(PUBLIC/'validation.json')
    write(MANIFEST,publication)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true')
    args=p.parse_args();verify() if args.verify_only else build()
