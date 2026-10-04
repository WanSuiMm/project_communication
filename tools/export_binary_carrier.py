"""Publish complete saved binary-carrier measurements; no model execution."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from export_continuation_interface import read, write, sha, safe, PRIVATE_TEXT
from export_state_factorization import arrays, equal, tensor_hash
import export_w_medium as wm

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/binary_carrier_20261005_01'
PUBLIC=ROOT/'evidence/binary_carrier_20261005'
MANIFEST=ROOT/'BINARY_CARRIER_PUBLICATION_MANIFEST.json'
PROTOCOL='binary_carrier_causal_compression_v1'
ARMS=('FF','SF','SS','bit_once','bit_repeat8','swap_once','swap_repeat8',
      'midpoint_once','midpoint_repeat8','oracle_once','source_once','source_swap_once')
PRIMARY=('bit_once','bit_repeat8','source_once')
LAUNCHER='tools/launch_binary_carrier.ps1'


def raw_summary():
    with gzip.open(PUBLIC/'raw_summary.json.gz','rt',encoding='utf-8') as f:
        return json.load(f)


def recompress_chunk(path):
    temporary=path.with_suffix('.tmp')
    with zipfile.ZipFile(path) as old,zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as new:
        for item in old.infolist():
            new.writestr(item.filename,old.read(item.filename))
    temporary.replace(path)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def without_maps(value):
    if isinstance(value,dict):
        return {k:without_maps(v) for k,v in value.items() if k not in ('per_map','score_per_map','per_map_delta')}
    if isinstance(value,list):return [without_maps(v) for v in value]
    return value


def compact(raw):
    result={k:raw[k] for k in ('protocol','status','completed_cells','maximum_cells','anchor',
        'primary_gates','combined_gates','recurrent_training_updates','decoder_fits',
        'templates_optimized','parameters_unchanged','elapsed_seconds','claim_boundary')}
    result['decoder_probes']=without_maps(raw['decoder_probes']);result['arms']={}
    for arm,row in raw['records'].items():
        result['arms'][arm]={}
        for size,r in row['sizes'].items():
            result['arms'][arm][size]={'keep_continuous':without_maps(r['cohorts']['keep']['continuous64_256']),
                'prog_sustained':without_maps(r['cohorts']['prog']['sustained241_256']),
                'prog_delayed':without_maps(r['cohorts']['prog']['delayed_sustained']),
                'opposite_paired_T256':without_maps(r['opposite_paired_T256']),
                'rescue_proxy':row['rescue'][size],
                'exact_FF_immediate_output':row['handoff'][size]['exact_FF_output'],
                'contrast_vs_SF':without_maps(row['contrast_vs_SF'].get(size,{}))}
    return result


def state_values(name,index):
    if name in index:return wm.state_values(PUBLIC,name,index)
    return arrays(PUBLIC/'states'/name)


def initial_projection(size,index,fit,decoder,write_files=False):
    """Recover the omitted T64 projection event from frozen inputs, no inference."""
    d=arrays(PUBLIC/'banks'/f'test{size}.npz')
    w=state_values(f'S1_size{size}_T64.npz',index)['W']
    mask=np.concatenate((d['mask'],d['mask'])).astype(bool)
    bits=decoder.predict(w,fit);n=len(d['x'])
    templates=np.stack((fit['templates']['p0'],fit['templates']['p1']))
    out={}
    def rms(v):
        return np.sqrt((v.astype(np.float64)**2*mask).sum((1,2,3))/(mask.sum((1,2,3))*24)).reshape(2,n)
    for arm in ('bit_repeat8','swap_repeat8','midpoint_repeat8'):
        if arm=='midpoint_repeat8':
            replacement=templates.mean(0).astype(np.float32)[None,:,None,None]
            projected=np.where(mask,replacement,w).astype(np.float32)
        else:projected=decoder.apply(w,~bits if arm=='swap_repeat8' else bits,mask,fit)
        vals={'times':np.asarray([64]),'before_bits':bits.reshape(1,2,n,size,size),
            'after_bits':decoder.predict(projected,fit).reshape(1,2,n,size,size),
            'W_rms_before':rms(w)[None],'W_rms_after':rms(projected)[None],
            'removed_W_rms':rms(w-projected)[None]}
        p=PUBLIC/'derived'/f'{arm}_size{size}_T64_projection.npz'
        if write_files:p.parent.mkdir(exist_ok=True);np.savez_compressed(p,**vals)
        else:
            saved=arrays(p);assert saved.keys()==vals.keys()
            for k,v in vals.items():assert np.array_equal(v,saved[k]),(p,k)
        out[arm]=p.relative_to(PUBLIC).as_posix()
    return out


def checks():
    raw=raw_summary();cfg=read(PUBLIC/'config.json');metric=wm.metric_helper()
    assert raw['protocol']==PROTOCOL and raw['status']=='COMPLETE'
    assert raw['completed_cells']==raw['maximum_cells']==24
    assert raw['recurrent_training_updates']==0 and raw['decoder_fits']==1 and raw['parameters_unchanged']
    assert tuple(raw['records'])==ARMS
    index=read(PUBLIC/'state_chunks.json');assert len(index)==2
    dec=module('saved_binary_decoder',ROOT/'new/binary_carrier/decoder.py')
    fit=read(PUBLIC/'decoder/fit.json');assert fit['converged']
    seen,geometries=set(),set()
    for role,record in cfg['banks'].items():
        d=arrays(PUBLIC/'banks'/f'{role}.npz')
        assert tensor_hash(d)==record['data_sha256'] and sha(PUBLIC/'banks'/f'{role}.npz')==record['file_sha256']
        h={hashlib.sha256(v.tobytes()).hexdigest()
           for key in ('x','x_flip') for v in d[key]}
        g={hashlib.sha256(v[:1].tobytes()).hexdigest() for v in d['x']}
        assert not(h&seen) and not(g&geometries);seen.update(h);geometries.update(g)
        assert len(h)==2*record['count'] and len(g)==record['count']
        w=arrays(PUBLIC/'states'/f'{role}_S1_W64.npz')['W'] if role.startswith(('calibration','validation')) else state_values(f'S1_size{record["size"]}_T64.npz',index)['W']
        bits=dec.predict(w,fit);assert np.array_equal(bits,arrays(PUBLIC/'decoder'/f'{role}_bits.npz')['bits'])
        count=len(d['x']);y=np.concatenate((d['y'],d['y_flip']))[:,0]>=.5
        good=bits==y;pick=d['mask'][:,0].astype(bool);changed=d['changed'][:,0].astype(bool)
        expected=raw['decoder_probes'][role]
        for name,mask in (('open',pick),('changed',changed),('unchanged',pick&~changed)):
            for cue,hit in (('original',good[:count]),('flipped',good[count:]),('paired',good[:count]&good[count:])):
                equal(metric._mask_rate(mask,hit),expected[name][cue])
        xor=bits[:count]^bits[count:]
        for key,mask,hit in (('changed_flip_rate',changed,xor),('unchanged_no_flip_rate',pick&~changed,~xor),
                            ('open_agreement_with_changed',pick,xor==changed)):
            equal(metric._mask_rate(mask,hit),expected['xor'][key])
        if role=='calibration':
            masks=np.concatenate((d['mask'],d['mask']))[:,0].astype(bool)
            weight=np.repeat(1/masks.sum((1,2)),masks.sum((1,2)))/len(masks)
            features=w.transpose(0,2,3,1)[masks].astype(np.float64);labels=y[masks]
            for b in (0,1):
                p=(weight[labels==b,None]*features[labels==b]).sum(0)/weight[labels==b].sum()
                assert np.allclose(p,fit['templates'][f'p{b}'],rtol=1e-10,atol=1e-12)
        if role.startswith('test'):
            cohorts=arrays(PUBLIC/'cohorts'/f'size{record["size"]}.npz')
            for cue,hit in (('original',good[:count]),('flipped',good[count:]),('paired',good[:count]&good[count:])):
                equal(metric._mask_rate(cohorts['prog'],hit),expected['prog'][cue])
        del w,bits,good,d
    support={}
    for size in (32,64):
        key=str(size);d=arrays(PUBLIC/'banks'/f'test{size}.npz');native={}
        for mid in ('S1','F1'):
            native[mid]=arrays(PUBLIC/'native'/f'{mid}_size{size}_trace.npz')
            states=state_values(f'{mid}_size{size}_T64.npz',index)
            assert tensor_hash(states)==cfg['snapshot_sha256'][mid][key];del states
        fixed=metric.cohorts(native['S1']['correct'][64],native['F1']['correct'][64],d['changed'][:,0].astype(bool))
        stored=arrays(PUBLIC/'cohorts'/f'size{size}.npz')
        assert all(np.array_equal(v,stored[k]) for k,v in fixed.items())
        ff_logits=arrays(PUBLIC/'arms'/f'FF_size{size}_logits.npz')
        for arm in ARMS:
            row=raw['records'][arm];assert row==read(PUBLIC/'arms'/f'{arm}.json')
            t=arrays(PUBLIC/'arms'/f'{arm}_size{size}_trace.npz')
            assert t['correct'].shape==(193,128,size,size) and t['correct'].dtype==np.bool_
            assert np.array_equal(t['times'],np.arange(64,257))
            assert np.array_equal(t['correct'],t['original_correct']&t['flipped_correct'])
            observed=metric.summarize(t['correct'],fixed);r=row['sizes'][key]
            equal(observed,{k:r[k] for k in ('cohorts','all_changed')})
            equal(metric.rescue_gate(observed,raw['records']['FF']['sizes'][key]),row['rescue'][key])
            equal(metric._mask_rate(fixed['all_changed'],~t['original_correct'][-1]&~t['flipped_correct'][-1]),r['opposite_paired_T256'])
            if arm=='FF':assert all(np.array_equal(t[k],native['F1'][k][64:]) for k in native['F1'])
            logits=arrays(PUBLIC/'arms'/f'{arm}_size{size}_logits.npz')
            assert all(np.isfinite(v).all() for v in logits.values())
            err=float(np.abs(logits['T64']-ff_logits['T64']).max());diag=row['handoff'][key]
            equal(err,diag['max_absolute_logit_error_vs_FF'])
            if arm!='SS':assert err==0 and diag['exact_FF_output']
            norms=arrays(PUBLIC/'arms'/f'{arm}_size{size}_norms.npz')
            assert norms['norms'].shape==(193,2,128,2) and np.isfinite(norms['norms']).all()
            if arm!='FF' and key in row['contrast_vs_SF']:
                equal(wm.contrast(r,raw['records']['SF']['sizes'][key],98400+size+ARMS.index(arm)),row['contrast_vs_SF'][key])
            if arm.endswith('repeat8'):
                p=arrays(PUBLIC/'arms'/f'{arm}_size{size}_projections.npz')
                assert np.array_equal(p['times'],np.arange(72,257,8))
                assert p['before_bits'].shape==p['after_bits'].shape==(24,2,128,size,size)
                assert all(np.isfinite(v).all() for v in p.values())
                predictions=np.asarray(fit['template_predictions_float32'],bool)
                if arm.startswith('midpoint'):
                    template=np.mean(np.stack((fit['templates']['p0'],fit['templates']['p1'])).astype(np.float32),axis=0)
                    after=np.full_like(p['before_bits'],dec.predict(template[None,:,None,None],fit)[0,0,0])
                else:after=predictions[(~p['before_bits'] if arm.startswith('swap') else p['before_bits']).astype(np.int64)]
                mask=np.broadcast_to(d['mask'][:,0].astype(bool)[None,None],p['before_bits'].shape)
                assert np.array_equal(p['after_bits'][mask],after[mask])
                assert np.array_equal(p['after_bits'][~mask],p['before_bits'][~mask])
            del t,logits,norms
        initial_projection(size,index,fit,dec)
        support[key]=raw['records']['SF']['rescue'][key]['observed']
        for arm in PRIMARY:
            r=raw['records'][arm]['sizes'][key];sf=raw['records']['SF']['sizes'][key]
            gate=raw['primary_gates'][arm][key]
            k=r['cohorts']['keep']['continuous64_256'];p=r['cohorts']['prog']['sustained241_256']
            checks={'keep_ge_0_95':k['pooled']['value']>=.95,
                    'retains_90pct_SF_progress':p['pooled']['value']>=.9*sf['cohorts']['prog']['sustained241_256']['pooled']['value'],
                    'keep_support':k['valid_maps']>=16 and k['pooled']['denominator']>=100,
                    'prog_support':p['valid_maps']>=16 and p['pooled']['denominator']>=100,
                    'anchor_qualified':raw['anchor']['pass'],'immediate_exact':raw['records'][arm]['handoff'][key]['exact_FF_output']}
            assert checks==gate['checks'] and gate['pass']==all(checks.values())
    assert raw['anchor']==read(PUBLIC/'anchor.json') and raw['anchor']['pass']
    for size in ('32','64'):
        for arm in ('SS','SF'):assert raw['records'][arm]['rescue'][size]['pass']
    for arm in PRIMARY:assert raw['combined_gates'][arm]['pass']==all(raw['primary_gates'][arm][s]['pass'] for s in ('32','64'))
    with (PUBLIC/'metrics.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==840
    for row in rows:
        r=raw['records'][row['arm']]['sizes'][row['size']]['cohorts'][row['cohort']][row['metric']]
        assert int(row['numerator'])==r['pooled']['numerator'] and int(row['denominator'])==r['pooled']['denominator']
        equal(None if row['pooled']=='' else float(row['pooled']),r['pooled']['value'])
        equal(None if row['map_mean']=='' else float(row['map_mean']),r['map_mean'])
        assert int(row['valid_maps'])==r['valid_maps']
    assert compact(raw)==read(PUBLIC/'summary.json')
    return {'status':'PASS','scope':'Saved arrays and arithmetic only; no model execution or decoder refitting.',
        'measurement_cells':24,'metric_rows':840,'npz_files':len(list(PUBLIC.rglob('*.npz'))),
        'state_chunks':8,'fixed_cohorts_counts_gates_recomputed':True,'decoder_probe_XOR_recomputed':True,
        'class_centroids_recomputed':True,'FF_native_suffix_exact':True,'W_immediate_neutrality_recomputed':True,
        'repeated_projection_times_and_template_bits_checked':True,'T64_projection_records_reconstructed':True,
        'all_state_tensor_hashes_exact':True,'anchor':raw['anchor'],'combined_gates':raw['combined_gates'],
        'support':support,'recurrent_training_updates':0,'decoder_fits':1,'elapsed_seconds':raw['elapsed_seconds']}


def verify(write_validation=False):
    m=read(MANIFEST);safe(m)
    with gzip.open(PUBLIC/'raw_summary.json.gz','rb') as f:
        assert hashlib.sha256(f.read()).hexdigest()==m['raw_artifact_bindings']['summary.json']['sha256']
    for category in ('source_sha256','publication_tools_sha256','published_sha256'):
        for rel,digest in m[category].items():assert sha(ROOT/rel)==digest,rel
    for p in PUBLIC.rglob('*.json'):safe(read(p))
    result=checks()
    if write_validation:write(PUBLIC/'validation.json',result)
    else:assert result==read(PUBLIC/'validation.json')
    print(json.dumps({k:result[k] for k in ('status','measurement_cells','metric_rows','npz_files','state_chunks','elapsed_seconds')}),flush=True)


def build():
    assert not PUBLIC.exists() and not MANIFEST.exists(),'New publication directory required'
    m,status,raw=read(RUN/'manifest.json'),read(RUN/'status.json'),read(RUN/'summary.json')
    assert m['status']==status['status']==raw['status']=='COMPLETE' and status['completed_cells']==24
    for rel,digest in m['source_sha256'].items():
        assert sha(RUN/'source'/rel)==digest
        if rel!=LAUNCHER:assert sha(ROOT/rel)==digest,rel
    PUBLIC.mkdir(parents=True);bindings={};index={};wm.PUBLIC=PUBLIC
    excluded=('manifest.json','status.json','startup_execution.json')
    for p in sorted(RUN.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(RUN)
        if 'source' in rel.parts or rel.as_posix() in excluded:continue
        assert p.suffix in ('.json','.npz','.csv','.md')
        if p.suffix=='.json':safe(read(p))
        if p.suffix in ('.csv','.md'):assert not PRIVATE_TEXT.search(p.read_text(encoding='utf-8-sig'))
        if rel.as_posix()=='summary.json':
            dest=PUBLIC/'raw_summary.json.gz'
            dest.write_bytes(gzip.compress(p.read_bytes(),compresslevel=9,mtime=0))
            bindings[rel.as_posix()]={'sha256':sha(p),'public_path':dest.relative_to(ROOT).as_posix(),
                                     'encoding':'lossless gzip level9; decompressed bytes identical'}
            continue
        if p.suffix=='.npz' and p.stat().st_size>=50*1024**2:
            assert rel.parts[0]=='states'
            index[p.name]=wm.split_state(p)
            for chunk in index[p.name]['chunks']:
                dest=PUBLIC/chunk['path'];recompress_chunk(dest);chunk['sha256']=sha(dest)
            bindings[rel.as_posix()]={'sha256':sha(p),'state_index_key':p.name,'tensor_sha256':index[p.name]['tensor_sha256']}
            continue
        rename={'summary.json':'raw_summary.json','RESULTS.md':'raw_RESULTS.md',
                'config.json':'raw_config.json','validation.json':'preflight_validation.json'}
        dest=PUBLIC/rename.get(rel.as_posix(),rel.as_posix());dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        bindings[rel.as_posix()]={'sha256':sha(p),'public_path':dest.relative_to(ROOT).as_posix()}
    write(PUBLIC/'state_chunks.json',index);write(PUBLIC/'summary.json',compact(raw))
    cfg=read(PUBLIC/'raw_config.json')
    for key in ('source_sha256','snapshot_sha256','excluded_bank_sha256','started_utc','finished_utc','elapsed_seconds','decoder_sha256'):
        cfg[key]=m[key]
    cfg['source_sha256']={rel:sha(ROOT/rel) for rel in m['source_sha256']}
    cfg['executed_source_sha256']=m['source_sha256'];cfg['state_format']='Two size64 captures losslessly split into eight paired-cue32-map chunks; all other arrays copied unchanged.'
    cfg['projection_T64_format']='Reconstructed initial events under derived/; runtime event files contain T72..256.'
    safe(cfg);write(PUBLIC/'config.json',cfg)
    dec=module('initial_binary_decoder',ROOT/'new/binary_carrier/decoder.py');fit=read(PUBLIC/'decoder/fit.json')
    for size in (32,64):initial_projection(size,index,fit,dec,write_files=True)
    note='''
## Publication notes

All measurement arrays and per-map records are retained. The duplicate full
summary is losslessly gzip-compressed as raw_summary.json.gz. Small entry JSON,
CSV and Markdown remain directly readable. New state chunks use ZIP level9;
all numeric precision is unchanged. Two size64 W/Z
captures exceed the file limit and are losslessly split into eight chunks;
state_chunks.json gives exact reconstruction and tensor hashes. Raw results
and all earlier frozen runs remain unchanged.

The runtime repeated-projection NPZ files contain T72..256. Initial T64
projections are separately reconstructed from saved S1 W, decoder and templates
under derived/ (bits and pre/post norms; no model execution). These derived
records are labelled separately and do not change any endpoint or gate.

The public launcher resolves Python through PATH, replacing its local absolute
environment path. Every scientific source is identical to the executed copy;
config.json records executed/public launcher hashes. Checkpoints, PIDs, host
metadata, launch receipts and logs stay local. One decoder was fitted in the
experiment; publication performs no fitting, inference or optimizer updates.
'''
    (PUBLIC/'RESULTS.md').write_text((RUN/'RESULTS.md').read_text(encoding='utf-8')+note,encoding='utf-8')
    (PUBLIC/'REPRODUCTION.md').write_text('''# Binary carrier evidence

Read RESULTS.md, summary.json, validation.json, config.json, decoder/fit.json,
decoder/probes.json and metrics.csv first. raw_summary.json.gz is secondary.

Saved-data verification from this repository root (NumPy, no model execution):

    python -X utf8 -B tools/export_binary_carrier.py --verify-only

The verifier checks file/source hashes, map-disjoint banks, lossless state
reconstruction, centroid arithmetic, held-out decoder/XOR scores, fixed cohorts,
24 arm traces/gates, 840 metric rows, FF-native equality, immediate W-only
output equality, paired-map bootstrap records and repeated template projection
bits/clocks. Native Full and parameter immutability are recorded runtime checks.
See RESULTS.md for the separate reconstructed T64 projection artifacts.

For a full rerun, obtain exact S1/F1 checkpoint hashes from config.json, install
NumPy and Torch2.5.1 with CUDA, and follow new/binary_carrier/PROTOCOL.md using
new output names. Run check.py first, then tools/launch_binary_carrier.ps1.
The public launcher discovers Python via PATH. Checkpoints remain local.

Test banks have128 map pairs per size; each pair is one independent map unit.
The single selected S1/F1 pair does not establish training reliability. All
three combined primary gates fail; oracle is diagnostic and excluded from
method claims. Failed class centroids do not rule out every possible bit code.
''',encoding='utf-8')
    tools=('tools/export_binary_carrier.py','tools/export_w_medium.py','tools/export_continuation_interface.py','tools/export_state_factorization.py')
    publication={'protocol':'binary_carrier_publication_v1','run_id':RUN.name,
        'source_sha256':cfg['source_sha256'],'executed_source_sha256':m['source_sha256'],
        'source_portability_change':{'path':LAUNCHER,'change':'Resolve Python through PATH instead of local absolute environment path; scientific sources unchanged.'},
        'publication_tools_sha256':{rel:sha(ROOT/rel) for rel in tools},
        'published_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in PUBLIC.rglob('*') if p.is_file()},
        'raw_artifact_bindings':bindings,'original_manifest_sha256':sha(RUN/'manifest.json'),
        'original_status_sha256':sha(RUN/'status.json'),
        'derived_only':['T64 projection bits/norms recovered from saved states/templates'],
        'excluded':['checkpoints','source snapshots','machine manifests','PIDs','startup records','launch receipts','logs']}
    write(MANIFEST,publication);verify(write_validation=True)
    publication['published_sha256']['evidence/binary_carrier_20261005/validation.json']=sha(PUBLIC/'validation.json')
    write(MANIFEST,publication)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();verify() if a.verify_only else build()
