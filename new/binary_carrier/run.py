"""Frozen binary carrier causal compression; no recurrent training."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


medium = load_module('binary_medium', ROOT/'new/w_medium/run.py')
factor, base, metrics = medium.factor, medium.base, medium.metrics
decoder = load_module('binary_decoder', Path(__file__).with_name('decoder.py'))
sha, read, write = base.sha, base.read, base.write
PROTOCOL = 'binary_carrier_causal_compression_v1'
BANKS = {'calibration': (32,64,98032), 'validation32': (32,64,98132),
         'validation64': (64,64,98164), 'test32': (32,128,98232),
         'test64': (64,128,98264)}
ARMS = ('FF','SF','SS','bit_once','bit_repeat8','swap_once','swap_repeat8',
        'midpoint_once','midpoint_repeat8','oracle_once','source_once','source_swap_once')
PRIMARY = ('bit_once','bit_repeat8','source_once')
EXPECTED_CELLS = 2*len(ARMS)


def sources():
    previous = read(ROOT/'W_MEDIUM_PUBLICATION_MANIFEST.json')['source_sha256']
    for rel, digest in previous.items():
        assert sha(ROOT/rel) == digest, f'Frozen scientific source drift: {rel}'
    bindings = previous.copy()
    for p in Path(__file__).parent.glob('*'):
        if p.suffix in ('.py','.md'):
            bindings[p.relative_to(ROOT).as_posix()] = sha(p)
    p = ROOT/'tools/launch_binary_carrier.ps1'
    bindings[p.relative_to(ROOT).as_posix()] = sha(p)
    return dict(sorted(bindings.items()))


def hashes(d):
    return factor.input_hashes(d), {base.hashlib.sha256(v[:1].numpy().tobytes()).hexdigest() for v in d['x']}


def make_banks(out):
    seen, geometries, excluded = set(), set(), {}
    for name, folder in (('CONTINUATION_INTERFACE','continuation_interface_20261004'),
                         ('STATE_FACTORIZATION','state_factorization_20261004'),
                         ('W_MEDIUM','w_medium_20261004')):
        pub = read(ROOT/f'{name}_PUBLICATION_MANIFEST.json')
        for p in sorted((ROOT/'evidence'/folder/'banks').glob('*.npz')):
            rel = p.relative_to(ROOT).as_posix()
            assert sha(p) == pub['published_sha256'][rel]
            with np.load(p,allow_pickle=False) as arr:
                d = {k:torch.from_numpy(arr[k].copy()) for k in ('x','x_flip')}
            hi,hg = hashes(d); seen.update(hi); geometries.update(hg); excluded[rel] = sha(p)
    for size,seed in ((32,96932),(64,96964),(32,97932),(64,97964),(32,98332),(64,98364)):
        hi,hg = hashes(base.bank(size,2,seed)); seen.update(hi); geometries.update(hg)
    data, records = {}, {}
    (out/'banks').mkdir()
    for role,(size,count,seed) in BANKS.items():
        d = base.bank(size,count,seed); hi,hg = hashes(d)
        assert len(hi)==2*count and len(hg)==count
        assert not(hi & seen) and not(hg & geometries), f'Bank collision: {role}'
        seen.update(hi); geometries.update(hg)
        p = out/'banks'/f'{role}.npz'
        np.savez_compressed(p,**{k:v.numpy() for k,v in d.items()})
        data[role] = d
        records[role] = {'size':size,'count':count,'seed':seed,'file_sha256':sha(p),
                         'data_sha256':base.tensor_hash(d),'geometry_and_input_collisions':0}
    return data,records,excluded


def pair_arrays(d):
    x = np.concatenate((d['x'].numpy(),d['x_flip'].numpy()))
    y = np.concatenate((d['y'].numpy(),d['y_flip'].numpy()))
    m = np.concatenate((d['mask'].numpy(),d['mask'].numpy()))
    return x,y,m


def probe(w,d,fit,cohorts=None):
    _,y,m = pair_arrays(d); count=len(d['x'])
    bits = decoder.predict(w,fit)
    scores=np.einsum('c,bchw->bhw',np.asarray(fit['raw_coefficient'],np.float64),w.astype(np.float64))+float(fit['raw_intercept'])
    good = bits == (y[:,0]>=.5)
    masks = {'open':d['mask'][:,0].numpy().astype(bool),
             'changed':d['changed'][:,0].numpy().astype(bool)}
    masks['unchanged'] = masks['open'] & ~masks['changed']
    if cohorts is not None: masks['prog'] = cohorts['prog']
    result = {}
    for name,pick in masks.items():
        result[name] = {cue:metrics._mask_rate(pick,hit) for cue,hit in
                        (('original',good[:count]),('flipped',good[count:]),
                         ('paired',good[:count]&good[count:]))}
        result[name]['score_per_map']=[{'map_index':i,'cue':cue,'open_or_cohort_cells':int(pick[i].sum()),
             'quantiles_0_25_50_75_100':np.quantile(values[i][pick[i]],[0,.25,.5,.75,1]).tolist() if pick[i].any() else None,
             'minimum_absolute_score':float(np.abs(values[i][pick[i]]).min()) if pick[i].any() else None}
            for cue,values in (('original',scores[:count]),('flipped',scores[count:])) for i in range(count)]
    xor = bits[:count] ^ bits[count:]
    result['xor'] = {'changed_flip_rate':metrics._mask_rate(masks['changed'],xor),
                     'unchanged_no_flip_rate':metrics._mask_rate(masks['unchanged'],~xor),
                     'open_agreement_with_changed':metrics._mask_rate(masks['open'],xor==masks['changed'])}
    return result,bits


def compression_gate(result,sf,anchor,neutral=True):
    if 'cohorts' not in result:
        return {'pass':False,'status':'FAIL','reason':'nonfinite'}
    keep,kc,km=metrics._pooled(result,'keep','continuous64_256')
    prog,pc,pm=metrics._pooled(result,'prog','sustained241_256')
    raw,_,_=metrics._pooled(sf,'prog','sustained241_256')
    checks = {'keep_ge_0_95':keep is not None and keep>=.95,
              'retains_90pct_SF_progress':prog is not None and raw is not None and prog>=.9*raw,
              'keep_support':km>=16 and kc>=100,'prog_support':pm>=16 and pc>=100,
              'anchor_qualified':bool(anchor),'immediate_exact':bool(neutral)}
    qualified = checks['anchor_qualified'] and checks['keep_support'] and checks['prog_support']
    passed = all(checks.values())
    return {'pass':passed,'status':'PASS' if passed else ('FAIL' if qualified else 'UNQUALIFIED'),
            'checks':checks,'observed':{'keep':keep,'prog':prog,'SF_prog':raw,
              'progress_ratio':prog/raw if raw is not None and raw>0 and prog is not None else None,
              'keep_maps':km,'keep_cells':kc,'prog_maps':pm,'prog_cells':pc}}


class ProjectedConsumer(torch.nn.Module):
    """Ordinary F1 transitions followed by frozen W-only C every eight steps."""
    def __init__(self,model,fit,count,kind):
        super().__init__(); self.model=model; self.fit=fit; self.count=count; self.kind=kind
        self.calls=0; self.events=[]; self.a=self.b=self.p=None

    def logits(self,state):
        self.calls+=1
        return self.model.logits(state)

    def step(self,state,x):
        out=self.model.step(state,x)
        t=64+self.calls%193
        assert 65<=t<=256
        if t%8: return out
        w,z=out; mask=x[:,:1].bool()
        if not all(bool(torch.isfinite(v).all()) for v in out):
            raise FloatingPointError(f'Nonfinite pre-projection state at T{t}')
        if self.a is None:
            self.a=torch.as_tensor(self.fit['raw_coefficient'],device=w.device,dtype=torch.float64)
            self.b=float(self.fit['raw_intercept'])
            self.p=torch.as_tensor(np.stack((self.fit['templates']['p0'],self.fit['templates']['p1'])),device=w.device,dtype=w.dtype)
        bit=(w.double()*self.a[None,:,None,None]).sum(1)+self.b>=0
        if self.kind=='midpoint':
            replacement=self.p.mean(0)[None,:,None,None].expand_as(w)
        else:
            index=(~bit if self.kind=='swap' else bit).long()
            replacement=self.p[index].permute(0,3,1,2)
        projected=torch.where(mask,replacement,w)
        after=(projected.double()*self.a[None,:,None,None]).sum(1)+self.b>=0
        assert torch.equal(self.model.logits((w,z)),self.model.logits((projected,z)))
        def rms(v):
            return ((v.double().square()*mask).sum((1,2,3))/(mask.sum((1,2,3))*24)).sqrt().cpu().numpy()
        block=(self.calls-1)//193; start=block*base.BATCH; n=len(w)//2
        self.events.append({'t':t,'map_start':start,'map_count':n,
            'before_bits':bit.cpu().numpy(),'after_bits':after.cpu().numpy(),
            'W_rms_before':rms(w),'W_rms_after':rms(projected),'removed_W_rms':rms(w-projected),
            'Z_exact':True,'logits_exact':True})
        return projected,z

    def save(self,path):
        shape=(24,2,self.count,self.events[0]['before_bits'].shape[-2],self.events[0]['before_bits'].shape[-1])
        arrays={k:np.empty(shape,bool) for k in ('before_bits','after_bits')}
        norms={k:np.empty((24,2,self.count),np.float64) for k in ('W_rms_before','W_rms_after','removed_W_rms')}
        for e in self.events:
            j=(e['t']-72)//8; start=e['map_start']; n=e['map_count']
            for k,v in arrays.items(): v[j,:,start:start+n]=e[k].reshape(2,n,*shape[-2:])
            for k,v in norms.items(): v[j,:,start:start+n]=e[k].reshape(2,n)
        assert len(self.events)==24*((self.count+base.BATCH-1)//base.BATCH)
        assert self.calls==193*((self.count+base.BATCH-1)//base.BATCH)
        np.savez_compressed(path,**arrays,**norms,times=np.arange(72,257,8))
        return {'projection_times':list(range(64,257,8)),'exact_Z_and_immediate_logits':True,
                'observed_batch_events':len(self.events),'path':path.name,'sha256':sha(path)}


def states_for(s,f,d,fit):
    x,y,m=pair_arrays(d); bits=decoder.predict(s[0],fit)
    def code(b,swapped=False):
        return decoder.apply(s[0],~b if swapped else b,m,fit)
    neutral=np.stack((fit['templates']['p0'],fit['templates']['p1'])).mean(0).astype(np.float32)
    mean=np.where(m.astype(bool),neutral[None,:,None,None],s[0]).astype(np.float32)
    source=decoder.source_only(s[0],x,fit)
    sf=decoder.source_only(s[0],x[:,[0,2,1]],fit)
    once=code(bits); swapped=code(bits,True)
    return {'FF':f,'SF':(s[0],f[1]),'SS':s,
        'bit_once':(once,f[1]),'bit_repeat8':(once,f[1]),
        'swap_once':(swapped,f[1]),'swap_repeat8':(swapped,f[1]),
        'midpoint_once':(mean,f[1]),'midpoint_repeat8':(mean,f[1]),
        'oracle_once':(code(y[:,0]>=.5),f[1]),'source_once':(source,f[1]),'source_swap_once':(sf,f[1])}


def finish_report(out,records,anchor,probes,elapsed):
    gates={a:{size:compression_gate(records[a]['sizes'][size],records['SF']['sizes'][size],anchor['pass'],
                       records[a]['handoff'][size]['exact_FF_output']) for size in ('32','64')} for a in PRIMARY}
    combined={a:{'pass':all(r['pass'] for r in sizes.values()),
                  'status':'UNQUALIFIED' if any(r['status']=='UNQUALIFIED' for r in sizes.values())
                  else ('PASS' if all(r['pass'] for r in sizes.values()) else 'FAIL')} for a,sizes in gates.items()}
    summary={'protocol':PROTOCOL,'status':'COMPLETE','completed_cells':EXPECTED_CELLS,
        'maximum_cells':EXPECTED_CELLS,'anchor':anchor,'primary_gates':gates,'combined_gates':combined,
        'recurrent_training_updates':0,'decoder_fits':1,'templates_optimized':False,
        'parameters_unchanged':True,'elapsed_seconds':elapsed,'records':records,'decoder_probes':probes,
        'claim_boundary':'Selected fixed consumer, finite restart/projected continuation; no universal bit insufficiency, quotient, architecture or BPTT guarantee.'}
    write(out/'summary.json',summary)
    fields=['size','arm','cohort','metric','numerator','denominator','pooled','map_mean','valid_maps']
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for arm,row in records.items():
            for size,result in row['sizes'].items():
                for cohort,values in result.get('cohorts',{}).items():
                    for metric,item in values.items():
                        writer.writerow({'size':size,'arm':arm,'cohort':cohort,'metric':metric,
                            **{k:item['pooled'][v] for k,v in (('numerator','numerator'),('denominator','denominator'),('pooled','value'))},
                            'map_mean':item['map_mean'],'valid_maps':item['valid_maps']})
    lines=['# Binary Carrier Causal Compression','',f'Status COMPLETE; 24/24 cells; elapsed {elapsed}s.',
        'Recurrent updates: 0. Linear decoder fits: 1. Templates: fixed calibration centroids.',
        f"Anchor: {anchor['status']}. Combined gates: {combined}.",'',
        '| Size | Arm | Keep continuous | Prog sustained | Prog delayed | 90% gate |','|---:|---|---:|---:|---:|---|']
    for size in ('32','64'):
        for arm in ARMS:
            result=records[arm]['sizes'][size]
            vals=[result.get('cohorts',{}).get(c,{}).get(k,{}).get('pooled',{}).get('value')
                  for c,k in (('keep','continuous64_256'),('prog','sustained241_256'),('prog','delayed_sustained'))]
            formatted=['null' if v is None else f'{v:.4f}' for v in vals]
            lines.append(f"| {size} | {arm} | {' | '.join(formatted)} | {gates.get(arm,{}).get(size,{}).get('status','control')} |")
    lines+=['','Oracle-only outcomes are diagnostic and excluded from primary claims.',
        'Read decoder/probes.json and decoder/fit.json, then summary.json; arrays and per-map records are secondary.',
        'Repeated projections retain Z and apply at T64,72,...,256. Source-only retains Z_F64 and recipient X.',
        summary['claim_boundary']]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def run(out,qualification):
    base.setup(); bindings,rows=sources(),factor.catalog(); q=read(qualification)
    assert q['status']=='PASS' and q['protocol']==PROTOCOL
    assert q['source_sha256']==bindings and q['checkpoints']==rows
    assert out.parent==ROOT/'runs' and not out.exists()
    out.mkdir(); begun=time.monotonic(); completed=0
    manifest={'protocol':PROTOCOL,'status':'RUNNING','recurrent_training_updates':0,'decoder_fits':1,
        'pid':os.getpid(),'host':os.environ.get('COMPUTERNAME'),'command':sys.argv,'started_utc':base.now(),
        'source_sha256':bindings,'checkpoints':rows,'torch_version':str(torch.__version__),
        'gpu':torch.cuda.get_device_name(0),'runtime_limit_enforced':False,'watchdog_enabled':False,
        'continuous_monitoring':False,'qualification_sha256':sha(qualification)}
    write(out/'manifest.json',manifest); write(out/'validation.json',q)
    def status(phase,**extra):
        r={'protocol':PROTOCOL,'status':'RUNNING','phase':phase,'pid':os.getpid(),
           'completed_cells':completed,'maximum_cells':EXPECTED_CELLS,'updated_utc':base.now(),**extra}
        write(out/'status.json',r); print(r,flush=True)
    try:
        for rel in bindings:
            p=out/'source'/rel; p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/rel,p)
        status('banks'); banks,bank_meta,excluded=make_banks(out)
        manifest.update(banks=bank_meta,excluded_bank_sha256=excluded); write(out/'manifest.json',manifest)
        write(out/'config.json',{'protocol':PROTOCOL,'banks':bank_meta,'models':rows,'arms':ARMS,
            'handoff':64,'final_time':256,'projection_times':list(range(64,257,8)),
            'fit_role':'calibration','validation_selection':False,'neutral':'template_midpoint',
            'recurrent_training_updates':0,'decoder_fits':1,'batch_maps':8,'gate_progress_retention':.9,
            'backend':{'dtype':'float32','decoder_score_dtype':'float64','cudnn_benchmark':False,
                       'cudnn_deterministic':False,'cudnn_tf32':True,'matmul_tf32':False}})
        manifest['config_sha256']=sha(out/'config.json'); write(out/'manifest.json',manifest)
        models={mid:base.load_model(row) for mid,row in rows.items()}
        (out/'states').mkdir(); (out/'decoder').mkdir(); (out/'native').mkdir(); (out/'arms').mkdir(); (out/'cohorts').mkdir()
        status('calibration_capture'); cal=base.capture(models['S1'],banks['calibration'],(64,))[0][0]
        p=out/'states'/'calibration_S1_W64.npz'; np.savez_compressed(p,W=cal)
        write(out/'startup_execution.json',{'pid':os.getpid(),'protocol':PROTOCOL,'finite_capture':bool(np.isfinite(cal).all()),
            'capture_path':p.relative_to(out).as_posix(),'capture_sha256':sha(p),'finished_utc':base.now()})
        _,y,m=pair_arrays(banks['calibration']); status('decoder_fit'); fit=decoder.fit(cal,y,m)
        template_state=np.stack((fit['templates']['p0'],fit['templates']['p1'])).astype(np.float32)[:,:,None,None]
        template_bits=decoder.predict(template_state,fit)[:,0,0]
        fit['template_predictions_float32']=template_bits.tolist()
        fit['decoder_projection_idempotent_on_templates']=bool(np.array_equal(template_bits,[False,True]))
        write(out/'decoder'/'fit.json',fit)
        assert fit['converged'], 'Frozen logistic solver did not converge; no tuning permitted'
        manifest['decoder_sha256']=sha(out/'decoder'/'fit.json'); write(out/'manifest.json',manifest)
        probes={}; probes['calibration'],bits=probe(cal,banks['calibration'],fit)
        np.savez_compressed(out/'decoder'/'calibration_bits.npz',bits=bits)
        del cal
        for role in ('validation32','validation64'):
            status('decoder_validation',role=role)
            w=base.capture(models['S1'],banks[role],(64,))[0][0]
            np.savez_compressed(out/'states'/f'{role}_S1_W64.npz',W=w)
            probes[role],bits=probe(w,banks[role],fit)
            np.savez_compressed(out/'decoder'/f'{role}_bits.npz',bits=bits); del w
        states,native,flags={},{},{}
        data={size:banks[f'test{size}'] for size in (32,64)}
        for mid,model in models.items():
            states[mid]={}; native[mid]={}; scores={}
            for size,d in data.items():
                status('native',model=mid,size=size)
                trace,ends,state,logits=base.paired_trace(model,d)
                states[mid][size]=state; native[mid][size]=trace; scores[size]=ends
                np.savez_compressed(out/'states'/f'{mid}_size{size}_T64.npz',W=state[0],Z=state[1])
                np.savez_compressed(out/'native'/f'{mid}_size{size}_trace.npz',**trace)
                np.savez_compressed(out/'native'/f'{mid}_size{size}_logits.npz',**{'T'+t:v for t,v in logits.items()})
            ph=base.summarize_from_traces(native[mid],scores,data)
            fr=ph.pop('_matched_frontier_rows'); ph['phenotype_gate']=base.predicate(ph)
            base._write_frontier_csv(out/'native'/f'{mid}_frontier.csv',fr)
            write(out/'native'/f'{mid}_phenotype.json',ph); flags[mid]=ph['phenotype_gate']['pass']
        records={}; cohorts={}; prepared={}
        for size,d in data.items():
            cohorts[size]=metrics.cohorts(native['S1'][size]['correct'][64],native['F1'][size]['correct'][64],d['changed'][:,0].numpy().astype(bool))
            np.savez_compressed(out/'cohorts'/f'size{size}.npz',**cohorts[size])
            probes[f'test{size}'],bits=probe(states['S1'][size][0],d,fit,cohorts[size])
            np.savez_compressed(out/'decoder'/f'test{size}_bits.npz',bits=bits)
            prepared[size]=states_for(states['S1'][size],states['F1'][size],d,fit)
        write(out/'decoder'/'probes.json',probes)
        snapshot_hashes={mid:{size:base.tensor_hash({'W':torch.from_numpy(s[0]),'Z':torch.from_numpy(s[1])}) for size,s in rows_.items()} for mid,rows_ in states.items()}
        manifest['snapshot_sha256']=snapshot_hashes; write(out/'manifest.json',manifest)
        for arm in ARMS:
            row={'arm':arm,'sizes':{},'handoff':{},'rescue':{},'contrast_vs_SF':{},'oracle_only':arm=='oracle_once'}
            records[arm]=row
            for size,d in data.items():
                status('intervention',arm=arm,size=size); key=str(size); state=prepared[size][arm]
                try:
                    diag=medium.handoff(models['F1'],d,state,states['F1'][size],visible=arm=='SS')
                    row['handoff'][key]=diag
                    consumer=ProjectedConsumer(models['F1'],fit,len(d['x']),arm.split('_')[0]) if arm.endswith('repeat8') else models['F1']
                    trace,ends,_,logits,norms=factor.suffix(consumer,d,state)
                    if arm=='FF': assert all(np.array_equal(v,native['F1'][size][k][64:]) for k,v in trace.items())
                    if arm.endswith('repeat8'):
                        diag['repeated_projection']=consumer.save(out/'arms'/f'{arm}_size{size}_projections.npz')
                    result=metrics.summarize(trace['correct'],cohorts[size]); result['endpoint_scores']=ends
                    # Both cues incorrect on the changed component means coherent opposite labels.
                    result['opposite_paired_T256']=metrics._mask_rate(cohorts[size]['all_changed'],~trace['original_correct'][-1]&~trace['flipped_correct'][-1])
                    row['sizes'][key]=result
                    row['rescue'][key]=metrics.rescue_gate(result,records['FF']['sizes'][key])
                    if arm!='FF' and 'SF' in records and key in records['SF']['sizes']:
                        row['contrast_vs_SF'][key]=medium.paired_contrast(result,records['SF']['sizes'][key],98400+size+ARMS.index(arm))
                    np.savez_compressed(out/'arms'/f'{arm}_size{size}_trace.npz',**trace,times=np.arange(64,257))
                    np.savez_compressed(out/'arms'/f'{arm}_size{size}_logits.npz',**{'T'+t:v for t,v in logits.items()})
                    np.savez_compressed(out/'arms'/f'{arm}_size{size}_norms.npz',norms=norms,times=np.arange(64,257))
                except FloatingPointError as exc:
                    models['F1'].cpu()
                    if arm in ('FF','SF','SS'): raise
                    row['handoff'].setdefault(key,{'exact_FF_output':False,'status':'NONFINITE','reason':str(exc)})
                    row['sizes'][key]={'status':'NONFINITE','reason':str(exc)}
                    row['rescue'][key]={'pass':False,'status':'FAIL','reason':'nonfinite'}
                completed+=1; write(out/'arms'/f'{arm}.json',row); status('cell_complete',arm=arm,size=size)
        anchor=medium.stage1_gate(flags,records); write(out/'anchor.json',anchor)
        for mid,model in models.items():
            assert base.tensor_hash(model.state_dict())==rows[mid]['parameter_sha256']
            for size,s in states[mid].items():
                assert base.tensor_hash({'W':torch.from_numpy(s[0]),'Z':torch.from_numpy(s[1])})==snapshot_hashes[mid][size]
        assert sha(out/'decoder'/'fit.json')==manifest['decoder_sha256']
        summary=finish_report(out,records,anchor,probes,round(time.monotonic()-begun,3))
        manifest.update(status='COMPLETE',finished_utc=base.now(),elapsed_seconds=summary['elapsed_seconds'],parameters_unchanged=True)
        write(out/'manifest.json',manifest)
        write(out/'status.json',{'protocol':PROTOCOL,'status':'COMPLETE','phase':'complete','pid':os.getpid(),
            'completed_cells':completed,'maximum_cells':EXPECTED_CELLS,'updated_utc':base.now(),'elapsed_seconds':summary['elapsed_seconds']})
    except BaseException as exc:
        write(out/'status.json',{'protocol':PROTOCOL,'status':'ERROR','pid':os.getpid(),'completed_cells':completed,
            'updated_utc':base.now(),'error':repr(exc),'traceback':traceback.format_exc()})
        traceback.print_exc(); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True); p.add_argument('--qualification',required=True)
    args=p.parse_args(); run((ROOT/args.out).resolve(),(ROOT/args.qualification).resolve())
