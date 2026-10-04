"""Bounded zero-training W-only replication and medium sensitivity suite."""
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
sys.path.insert(0, str(ROOT/'new/state_factorization'))
import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


factor = load_module('w_medium_factor', ROOT/'new/state_factorization/run.py')
base, metrics = factor.base, factor.metrics
transforms = load_module('w_medium_transforms', Path(__file__).with_name('transforms.py'))
sha, read, write = base.sha, base.read, base.write
PROTOCOL = 'w_medium_qualification_v1'
BANK_SPECS = {32: (128, 97032), 64: (128, 97064)}
STAGE1 = ('FF', 'SF', 'FS', 'SS')
STAGE2 = ('SF_norm_F', 'FF_norm_S', 'SF_payload', 'SF_payload_gauge',
          'SF_lane', 'SF_lane_gauge', 'SF_spatial', 'SF_spatial_gauge',
          'SF_cue_swap', 'SF_time32', 'SF_time48', 'SF_time80', 'SF_time96',
          'SF_norm_F_cue_swap')
ARMS = STAGE1 + STAGE2
PAYLOAD_PERM = (2, 5, 0, 4, 1, 3)
LANE_PERM = (1, 2, 3, 0)
EXPECTED_CELLS = 2*len(ARMS)
DONOR_TIMES = (32, 48, 80, 96)


def sources():
    previous = read(ROOT/'STATE_FACTORIZATION_PUBLICATION_MANIFEST.json')['source_sha256']
    for rel, digest in previous.items():
        assert sha(ROOT/rel) == digest, f'Frozen scientific source drift: {rel}'
    bindings = previous.copy()
    for p in Path(__file__).parent.glob('*'):
        if p.suffix in ('.py', '.md'):
            bindings[p.relative_to(ROOT).as_posix()] = sha(p)
    p = ROOT/'tools/launch_w_medium.ps1'
    bindings[p.relative_to(ROOT).as_posix()] = sha(p)
    return dict(sorted(bindings.items()))


def make_banks(out):
    seen, exclusion = set(), {}
    for name, folder in (('CONTINUATION_INTERFACE', 'continuation_interface_20261004'),
                         ('STATE_FACTORIZATION', 'state_factorization_20261004')):
        pub = read(ROOT/f'{name}_PUBLICATION_MANIFEST.json')
        for p in sorted((ROOT/'evidence'/folder/'banks').glob('*.npz')):
            rel = p.relative_to(ROOT).as_posix()
            assert sha(p) == pub['published_sha256'][rel]
            with np.load(p, allow_pickle=False) as arr:
                d = {k: torch.from_numpy(arr[k].copy()) for k in ('x', 'x_flip')}
            seen.update(factor.input_hashes(d)); exclusion[rel] = sha(p)
    assert len(exclusion) == 8
    for size, seed in ((32,96932), (64,96964), (32,97932), (64,97964)):
        seen.update(factor.input_hashes(base.bank(size, 2, seed)))
    data, records = {}, {}
    (out/'banks').mkdir()
    for size, (count, seed) in BANK_SPECS.items():
        d = base.bank(size, count, seed)
        fresh = factor.input_hashes(d)
        assert len(fresh) == 2*count and not (seen & fresh), 'Fresh input collision'
        seen.update(fresh)
        path = out/'banks'/f'test{size}.npz'
        np.savez_compressed(path, **{k: v.numpy() for k, v in d.items()})
        data[size] = d
        records[size] = {'size': size, 'count': count, 'seed': seed,
                        'data_sha256': base.tensor_hash(d), 'file_sha256': sha(path),
                        'cross_role_collisions': 0}
    return data, records, exclusion


class ConjugatedConsumer(torch.nn.Module):
    """Exact P_W Phi P_W^-1; Z/readout and recipient inputs stay unchanged."""
    def __init__(self, model, spec, count):
        super().__init__()
        self.model = model
        self.spec, self.count, self.calls = spec, count, 0
        self.forward_index = self.inverse_index = None

    def _set_batch(self, device):
        block = self.calls//193
        start = block*base.BATCH
        ids = list(range(start, min(start+base.BATCH, self.count)))
        assert ids, 'Gauge batch observation order drift'
        forward = np.asarray(self.spec['forward_index'])
        inverse = np.asarray(self.spec['inverse_index'])
        if self.spec['kind'] == 'spatial':
            forward = np.concatenate((forward[ids], forward[ids]), axis=0)
            inverse = np.concatenate((inverse[ids], inverse[ids]), axis=0)
        self.forward_index = torch.as_tensor(forward.copy(), device=device, dtype=torch.long)
        self.inverse_index = torch.as_tensor(inverse.copy(), device=device, dtype=torch.long)

    def _apply_w(self, w, inverse=False):
        index = self.inverse_index if inverse else self.forward_index
        if self.spec['kind'] == 'channel':
            return w.index_select(1, index)
        return w.flatten(2).gather(2, index[:, None].expand(-1, w.shape[1], -1)).reshape_as(w)

    def logits(self, state):
        if self.calls % 193 == 0:
            self._set_batch(state[0].device)
        self.calls += 1
        return self.model.logits(state)

    def step(self, state, x):
        w, z = state
        wn, zn = self.model.step((self._apply_w(w, inverse=True), z), x)
        return self._apply_w(wn), zn


def stage1_gate(native_flags, records):
    checks = {}
    for size in ('32', '64'):
        for arm in ('SS', 'SF'):
            checks[f'{arm}_rescue_size{size}'] = bool(records[arm]['rescue'][size]['pass'])
        checks[f'SF_exact_output_size{size}'] = bool(records['SF']['handoff'][size]['exact_FF_output'])
    native_episode = bool(native_flags['S1']) and not bool(native_flags['F1'])
    historical = {size:native_episode and bool(records['SS']['rescue'][size]['pass']) for size in ('32','64')}
    return {'pass': all(checks.values()), 'checks': checks,
            'native_full_flags_descriptive':native_flags,
            'historical_localization_episode_qualified':historical,
            'status': 'QUALIFIED_W_REPLICATION_CONTROLS_ENABLED' if all(checks.values()) else 'STAGE1_UNQUALIFIED_CONTROLS_SKIPPED'}


def transform_states(s, f, d, donor_times):
    ws, wf, zf = s[0], f[0], f[1]
    norm_s, norm_record_s = transforms.norm_match(ws, wf, d['mask'].numpy())
    norm_f, norm_record_f = transforms.norm_match(wf, ws, d['mask'].numpy())
    payload, p_record = transforms.channel_permute(ws, PAYLOAD_PERM)
    lane, l_record = transforms.lane_permute(ws, LANE_PERM)
    spatial, s_record = transforms.spatial_scramble(ws, d['changed'][:,0].numpy().astype(bool), 97007+ws.shape[-1])
    swapped, c_record = transforms.cue_swap(ws)
    swapped_norm, cn_record = transforms.norm_match(swapped, wf, d['mask'].numpy())
    p_index = np.asarray([lane*6+PAYLOAD_PERM[j] for lane in range(4) for j in range(6)])
    l_index = np.asarray([lane*6+j for lane in LANE_PERM for j in range(6)])
    specs = {'payload': {'kind':'channel', 'forward_index':p_index, 'inverse_index':np.argsort(p_index)},
             'lane': {'kind':'channel', 'forward_index':l_index, 'inverse_index':np.argsort(l_index)},
             'spatial': {'kind':'spatial', 'forward_index':s_record['source_indices'],
                         'inverse_index':s_record['inverse_source_indices']}}
    rows = {'SF_norm_F':(norm_s,zf), 'FF_norm_S':(norm_f,zf),
            'SF_payload':(payload,zf), 'SF_payload_gauge':(payload,zf),
            'SF_lane':(lane,zf), 'SF_lane_gauge':(lane,zf),
            'SF_spatial':(spatial,zf), 'SF_spatial_gauge':(spatial,zf),
            'SF_cue_swap':(swapped,zf), 'SF_norm_F_cue_swap':(swapped_norm,zf)}
    rows.update({f'SF_time{t}':(donor_times[t],zf) for t in DONOR_TIMES})
    metadata = {'SF_norm_F':norm_record_s, 'FF_norm_S':norm_record_f,
                'SF_payload':p_record, 'SF_lane':l_record, 'SF_spatial':s_record,
                'SF_cue_swap':c_record, 'SF_norm_F_cue_swap':cn_record}
    return rows, specs, metadata


@torch.no_grad()
def handoff(model, d, state, f, visible=False, spec=None):
    model.cuda().eval()
    count = len(d['x']); error, disagreements = [], []
    scales = []
    for start in range(0, count, base.BATCH):
        ids = list(range(start, min(start+base.BATCH, count)))
        indexes = ids + [i+count for i in ids]
        x, _, mask = base.inputs(d, ids)
        w, z = (torch.from_numpy(v[indexes]).cuda() for v in state)
        if not all(bool(torch.isfinite(v).all()) for v in (w,z)):
            raise FloatingPointError('Nonfinite intervention handoff state')
        expected = model.logits(tuple(torch.from_numpy(v[indexes]).cuda() for v in f))
        actual = model.logits((w,z))
        if not all(bool(torch.isfinite(v).all()) for v in (actual,expected)):
            raise FloatingPointError('Nonfinite intervention handoff logits')
        error.extend((actual-expected).abs().flatten(1).max(1).values.cpu().tolist())
        disagreements.extend(((((actual>=0)!=(expected>=0)) & mask.bool()).flatten(1).sum(1)).cpu().tolist())
        def rms(value):
            return ((value.double().square()*mask).sum((1,2,3))/(mask.sum((1,2,3))*value.shape[1]).clamp_min(1)).sqrt()
        canonical_w = w
        if spec:
            index = np.asarray(spec['inverse_index'])
            if spec['kind'] == 'channel':
                canonical_w = w.index_select(1,torch.as_tensor(index.copy(),device='cuda',dtype=torch.long))
            else:
                index = np.concatenate((index[ids],index[ids]),axis=0)
                index = torch.as_tensor(index.copy(),device='cuda',dtype=torch.long)
                canonical_w = w.flatten(2).gather(2,index[:,None].expand(-1,w.shape[1],-1)).reshape_as(w)
        old = model._features(canonical_w,z,x)
        incoming = sys.modules['stream_cells'].stream(canonical_w,x[:,:1])
        features = torch.cat((incoming,old[:,model.workspace_channels:]),dim=1)
        force = model.f_out(torch.tanh(model.f_in(features)))
        wn = incoming+model.eta*force
        q = model._candidate(wn,z,x)
        actual_step = model.step((canonical_w,z),x)
        if not all(bool(torch.isfinite(v).all()) for v in (force,q,wn,*actual_step)):
            raise FloatingPointError('Nonfinite first intervention step')
        assert torch.equal(actual_step[0],wn) and torch.equal(actual_step[1],z+model.alpha*q), 'First-step instrumentation drift'
        rw,rz,rf,rq = (rms(value).cpu().numpy() for value in (w,z,force,q))
        for j, index in enumerate(indexes):
            scales.append({'map_index':index % count, 'cue':'original' if index<count else 'flipped',
                           'W_rms':float(rw[j]), 'Z_rms':float(rz[j]),
                           'F_rms':float(rf[j]), 'Q_rms':float(rq[j]),
                           'F_over_W':float(rf[j]/rw[j]) if rw[j]>0 else None,
                           'Q_over_Z':float(rq[j]/rz[j]) if rz[j]>0 else None})
    model.cpu()
    exact = max(error) == 0 and sum(disagreements) == 0
    if not visible:
        assert exact, 'W-only replacement changed immediate readout'
    return {'readout_neutral_arm':not visible, 'exact_FF_output':exact,
            'max_absolute_logit_error_vs_FF':max(error),
            'open_decision_disagreements_vs_FF':int(sum(disagreements)),
            'instrumented_first_step_exact':True,
            'force_write_coordinates':'canonical decoded consumer' if spec else 'fixed F1',
            'scale_per_cue_map':scales}


def edge_descriptor(w, mask):
    """Mean squared difference along open four-neighbor edges, per cue/map."""
    m = mask[:,0].numpy().astype(bool)
    m = np.concatenate((m,m)); v = w.astype(np.float64)
    sums, counts = np.zeros(len(w)), np.zeros(len(w), np.int64)
    for axis in (2,3):
        hi, lo = [slice(None)]*4, [slice(None)]*4
        hi[axis], lo[axis] = slice(1,None), slice(None,-1)
        a, b = [slice(None)]*3, [slice(None)]*3
        a[axis-1], b[axis-1] = slice(1,None), slice(None,-1)
        valid = m[tuple(a)] & m[tuple(b)]
        diff = v[tuple(hi)]-v[tuple(lo)]
        sums += (diff*diff*valid[:,None]).sum((1,2,3))
        counts += valid.sum((1,2))*w.shape[1]
    return {'open_edge_mean_square_per_cue_map':np.divide(sums, counts, out=np.zeros_like(sums), where=counts>0),
            'open_edge_channel_count_per_cue_map':counts}


def paired_contrast(result, reference, seed):
    output = {}
    for cohort, metric in (('keep','continuous64_256'), ('prog','sustained241_256'), ('prog','delayed_sustained')):
        a, b = (r['cohorts'][cohort][metric]['per_map'] for r in (result,reference))
        eligible = [(x['map_index'], x['value']-y['value']) for x,y in zip(a,b)
                    if x['denominator'] > 0 and y['denominator'] > 0]
        values = np.asarray([v for _,v in eligible])
        if len(values):
            rng = np.random.default_rng(seed)
            samples = values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
            interval = np.quantile(samples,[.025,.975])
        else:
            interval = [None,None]
        output[f'{cohort}.{metric}'] = {'map_mean_delta':float(values.mean()) if len(values) else None,
            'eligible_maps':len(values), 'bootstrap_95_interval':interval,
            'per_map_delta':eligible, 'bootstrap_replicates':2000, 'bootstrap_seed':seed,
            'interpretation':'descriptive paired map interval; not seed reliability or mechanistic significance'}
    return output


def persist_transform(out, size, metadata, specs):
    folder = out/'transforms'; folder.mkdir(exist_ok=True)
    for key, record in metadata.items():
        arrays = {k:v for k,v in record.items() if isinstance(v,np.ndarray)}
        scalar = {k:v for k,v in record.items() if not isinstance(v,np.ndarray)}
        if arrays:
            p = folder/f'{key}_size{size}.npz'; np.savez_compressed(p,**arrays)
            scalar['arrays'] = {'path':p.relative_to(out).as_posix(),'sha256':sha(p)}
        write(folder/f'{key}_size{size}.json',scalar)
    for kind,spec in specs.items():
        p = folder/f'gauge_{kind}_size{size}.npz'
        np.savez_compressed(p, forward_index=spec['forward_index'], inverse_index=spec['inverse_index'])


def report(out, records, gate, native_flags, completed, elapsed):
    for arm,row in records.items():
        for size,result in row['sizes'].items():
            reference=records['SF']['sizes'][size]
            if 'cohorts' in result and 'cohorts' in reference:
                row['contrast_vs_SF'][size]=paired_contrast(result,reference,97100+int(size)+ARMS.index(arm))
        write(out/'arms'/f'{arm}.json',row)
    fields = ['size','arm','cohort','metric','numerator','denominator','pooled','map_mean','valid_maps']
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for arm, row in records.items():
            for size,r in row['sizes'].items():
                for cohort,values in r.get('cohorts',{}).items():
                    for metric,item in values.items():
                        writer.writerow({'size':size,'arm':arm,'cohort':cohort,'metric':metric,
                            'numerator':item['pooled']['numerator'],'denominator':item['pooled']['denominator'],
                            'pooled':item['pooled']['value'],'map_mean':item['map_mean'],'valid_maps':item['valid_maps']})
    summary={'protocol':PROTOCOL,'status':'COMPLETE' if gate['pass'] else 'COMPLETE_STAGE1_ONLY',
             'training_updates':0,'completed_cells':completed,'maximum_cells':EXPECTED_CELLS,
             'stage1_gate':gate,'native_full_flags':native_flags,'records':records,
             'skipped_cells':0 if gate['pass'] else 2*len(STAGE2),
             'parameters_unchanged':True,'elapsed_seconds':elapsed,
             'claim_boundary':'Selected pair, finite conditional W sensitivity; no exclusive state roles, pure phase, carrier loss or training reliability.'}
    write(out/'summary.json',summary)
    lines=['# W medium qualification', '', f"Status: {summary['status']}; {completed}/{EXPECTED_CELLS} measured cells; zero training.",
           f"Stage1: {gate['status']}. Native Full: {native_flags}.", '',
           '| Size | Arm | Keep continuous | Prog sustained | Prog delayed | Proxy | Output |',
           '|---:|---|---:|---:|---:|---|---|']
    def fmt(value): return 'null' if value is None else f'{value:.4f}'
    for size in ('32','64'):
        for arm in ARMS:
            if arm not in records:
                lines.append(f'| {size} | {arm} | SKIPPED_STAGE1_GATE | | | | |'); continue
            row=records[arm]; r=row['sizes'][size]
            if 'cohorts' not in r:
                lines.append(f'| {size} | {arm} | NONFINITE_CONTINUATION | | | FAIL | |'); continue
            c=r['cohorts']
            values=[size,arm,fmt(c['keep']['continuous64_256']['pooled']['value']),
                    fmt(c['prog']['sustained241_256']['pooled']['value']),
                    fmt(c['prog']['delayed_sustained']['pooled']['value']), row['rescue'][size]['status'],
                    'exact FF' if row['handoff'][size]['readout_neutral_arm'] else 'visible Z']
            lines.append('| '+' | '.join(map(str,values))+' |')
    lines += ['', 'Read summary.json, metrics.csv and stage1_gate.json first. Native Full,',
              'map support and conditional rescue are distinct. Gauge arms change consumer',
              'coordinates as well as W; raw-arm losses are compatibility sensitivities.',
              'No mechanism class or learning algorithm is selected automatically.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    names=list(records)
    for i,size in enumerate(('32','64')):
        for j,(cohort,key) in enumerate((('keep','continuous64_256'),('prog','sustained241_256'))):
            vals=[records[a]['sizes'][size].get('cohorts',{}).get(cohort,{}).get(key,{}).get('pooled',{}).get('value') for a in names]
            axes[i,j].bar(names,[np.nan if v is None else v for v in vals])
            axes[i,j].set_ylim(0,1.05); axes[i,j].tick_params(axis='x',rotation=75)
            axes[i,j].set_title(f'Size{size}: {cohort} {key}')
    fig.suptitle('Fixed native cohorts, selected S1/F1 pair, no training')
    fig.savefig(out/'w_medium.png',dpi=160); fig.savefig(out/'w_medium.pdf'); plt.close(fig)
    return summary


def run(out, qualification):
    base.setup(); bindings,rows=sources(),factor.catalog()
    q=read(qualification)
    assert q['status']=='PASS' and q['protocol']==PROTOCOL
    assert q['source_sha256']==bindings and q['checkpoints']==rows
    assert out.parent==ROOT/'runs' and not out.exists(), 'New direct run directory required'
    out.mkdir(); begun=time.monotonic(); completed=0
    manifest={'protocol':PROTOCOL,'status':'RUNNING','training_updates':0,'pid':os.getpid(),
              'host':os.environ.get('COMPUTERNAME'),'command':sys.argv,'started_utc':base.now(),
              'source_sha256':bindings,'checkpoints':rows,'torch_version':str(torch.__version__),
              'gpu':torch.cuda.get_device_name(0),'runtime_limit_enforced':False,
              'watchdog_enabled':False,'continuous_monitoring':False}
    manifest['qualification']={'path':qualification.relative_to(ROOT).as_posix(),'sha256':sha(qualification)}
    write(out/'manifest.json',manifest)
    write(out/'validation.json',q)
    def status(phase, **extra):
        value={'protocol':PROTOCOL,'status':'RUNNING','phase':phase,'pid':os.getpid(),
               'completed_cells':completed,'maximum_cells':EXPECTED_CELLS,'updated_utc':base.now(),**extra}
        write(out/'status.json',value); print(value,flush=True)
    try:
        for rel in bindings:
            p=out/'source'/rel; p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/rel,p)
        status('banks'); data,banks,exclusion=make_banks(out)
        manifest.update(banks=banks,excluded_bank_sha256=exclusion); write(out/'manifest.json',manifest)
        write(out/'config.json',{'protocol':PROTOCOL,'stage1':STAGE1,'stage2':STAGE2,'banks':banks,
              'models':rows,'consumer':'F1','handoff':64,'last_step':256,'batch_maps':8,
              'donor_times':DONOR_TIMES,'payload_permutation':PAYLOAD_PERM,'lane_permutation':LANE_PERM,
              'training_updates':0,'runtime_limit_enforced':False,
              'backend':{'cudnn_benchmark':False,'cudnn_deterministic':False,'cudnn_tf32':True,'matmul_tf32':False}})
        models={k:base.load_model(row) for k,row in rows.items()}
        states,native,native_flags={},{},{}
        for mid,model in models.items():
            states[mid]={}; native[mid]={}; scores={}
            for size,d in data.items():
                status('native',model=mid,size=size)
                trace,endpoints,snapshot,logits=base.paired_trace(model,d)
                states[mid][size]=snapshot; native[mid][size]=trace; scores[size]=endpoints
                folder=out/'native'; folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'{mid}_size{size}_trace.npz',**trace)
                np.savez_compressed(folder/f'{mid}_size{size}_logits.npz',**{'T'+t:v for t,v in logits.items()})
                folder=out/'states'; folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'{mid}_size{size}_T64.npz',W=snapshot[0],Z=snapshot[1])
                if not (out/'startup_execution.json').exists():
                    executed=out/'native'/f'{mid}_size{size}_trace.npz'
                    write(out/'startup_execution.json',{'pid':os.getpid(),'protocol':PROTOCOL,
                          'model':mid,'size':size,'finite_native_trace':True,'finished_utc':base.now(),
                          'trace_path':executed.relative_to(out).as_posix(),'trace_sha256':sha(executed)})
                status('native_complete',model=mid,size=size)
            phenotype=base.summarize_from_traces(native[mid],scores,data)
            frontier=phenotype.pop('_matched_frontier_rows'); phenotype['phenotype_gate']=base.predicate(phenotype)
            base._write_frontier_csv(out/'native'/f'{mid}_frontier.csv',frontier)
            write(out/'native'/f'{mid}_phenotype.json',phenotype)
            native_flags[mid]=phenotype['phenotype_gate']['pass']
        cohorts={}; prepared={}
        for size,d in data.items():
            cohorts[size]=metrics.cohorts(native['S1'][size]['correct'][64],native['F1'][size]['correct'][64],d['changed'][:,0].numpy().astype(bool))
            folder=out/'cohorts'; folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/f'size{size}.npz',**cohorts[size])
            s,f=states['S1'][size],states['F1'][size]
            prepared[size]={'FF':f,'SF':(s[0],f[1]),'FS':(f[0],s[1]),'SS':s}
        snapshot_hashes={mid:{size:base.tensor_hash({'W':torch.from_numpy(s[0]),'Z':torch.from_numpy(s[1])})
                              for size,s in rows_.items()} for mid,rows_ in states.items()}
        manifest.update(snapshot_sha256=snapshot_hashes,snapshot_cue_order='all_original_maps_then_all_flipped_maps')
        write(out/'manifest.json',manifest)
        records={}
        reference_sf={}
        def measure(arm, size, state, spec=None):
            nonlocal completed
            d=data[size]; status('intervention',arm=arm,size=size)
            row=records.setdefault(arm,{'arm':arm,'sizes':{},'handoff':{},'rescue':{},'contrast_vs_SF':{}})
            folder=out/'arms'; folder.mkdir(exist_ok=True)
            try:
                diag=handoff(models['F1'],d,state,states['F1'][size],visible=arm in ('FS','SS'),spec=spec)
                diag['W_edge_descriptor']=edge_descriptor(state[0],d['mask'])
                row['handoff'][str(size)]=diag
                consumer=ConjugatedConsumer(models['F1'],spec,len(d['x'])) if spec else models['F1']
                trace,endpoints,_,logits,norms=factor.suffix(consumer,d,state)
                if arm=='FF':
                    assert all(np.array_equal(v,native['F1'][size][key][64:]) for key,v in trace.items()), 'FF/native mismatch'
                if spec:
                    ref_trace,ref_logits=reference_sf[size]
                    assert all(np.array_equal(v,ref_trace[k]) for k,v in trace.items()), 'Gauge trace mismatch'
                    assert all(np.array_equal(v,ref_logits[k]) for k,v in logits.items()), 'Gauge endpoint logits mismatch'
                    diag['exact_gauge_equivalence_to_SF']=True
                if arm=='SF': reference_sf[size]=(trace,logits)
                result=metrics.summarize(trace['correct'],cohorts[size]); result['endpoint_scores']=endpoints
                row['sizes'][str(size)]=result
                row['rescue'][str(size)]=metrics.rescue_gate(result,records['FF']['sizes'][str(size)])
                if diag['readout_neutral_arm']:
                    row['rescue'][str(size)]['pass'] &= diag['exact_FF_output']
                if arm not in ('FF',):
                    ref=records.get('SF',{}).get('sizes',{}).get(str(size))
                    if ref and 'cohorts' in ref:
                        row['contrast_vs_SF'][str(size)]=paired_contrast(result,ref,97100+size+ARMS.index(arm))
                np.savez_compressed(folder/f'{arm}_size{size}_trace.npz',**trace,times=np.arange(64,257))
                np.savez_compressed(folder/f'{arm}_size{size}_logits.npz',**{'T'+t:v for t,v in logits.items()})
                np.savez_compressed(folder/f'{arm}_size{size}_norms.npz',norms=norms,times=np.arange(64,257),
                                    cues=np.asarray(['original','flipped']),fields=np.asarray(['W_rms','Z_rms']))
            except FloatingPointError as exc:
                models['F1'].cpu()
                if spec or arm=='FF': raise
                row['handoff'].setdefault(str(size),{'status':'NONFINITE_HANDOFF','reason':str(exc),
                     'readout_neutral_arm':arm not in ('FS','SS'),'exact_FF_output':False})
                row['sizes'][str(size)]={'status':'NONFINITE_CONTINUATION','reason':str(exc)}
                row['rescue'][str(size)]={'pass':False,'status':'FAIL','reason':'nonfinite'}
            completed+=1; write(folder/f'{arm}.json',row); status('cell_complete',arm=arm,size=size)
        for arm in STAGE1:
            for size in data:
                measure(arm,size,prepared[size][arm])
        gate=stage1_gate(native_flags,records); write(out/'stage1_gate.json',gate); status('stage1_gate',gate=gate['status'])
        if gate['pass']:
            specs={}
            for size,d in data.items():
                status('donor_time_capture',size=size)
                cap=base.capture(models['S1'],d,DONOR_TIMES)
                donor_times={t:cap[0][i] for i,t in enumerate(DONOR_TIMES)}
                np.savez_compressed(out/'states'/f'S1_size{size}_donor_times.npz',W=cap[0],times=np.asarray(DONOR_TIMES))
                controls,specs[size],meta=transform_states(states['S1'][size],states['F1'][size],d,donor_times)
                prepared[size].update(controls); persist_transform(out,size,meta,specs[size])
                del cap
            for arm in STAGE2:
                for size in data:
                    spec=specs[size][arm.split('_')[1]] if arm.endswith('_gauge') else None
                    measure(arm,size,prepared[size][arm],spec)
        for mid,model in models.items():
            assert base.tensor_hash(model.state_dict())==rows[mid]['parameter_sha256'], 'Parameter mutation'
            for size,s in states[mid].items():
                assert base.tensor_hash({'W':torch.from_numpy(s[0]),'Z':torch.from_numpy(s[1])})==snapshot_hashes[mid][size], 'Native state mutation'
        summary=report(out,records,gate,native_flags,completed,round(time.monotonic()-begun,3))
        manifest.update(status=summary['status'],finished_utc=base.now(),elapsed_seconds=summary['elapsed_seconds'])
        write(out/'manifest.json',manifest)
        write(out/'status.json',{'protocol':PROTOCOL,'status':summary['status'],'phase':'complete','pid':os.getpid(),
              'completed_cells':completed,'maximum_cells':EXPECTED_CELLS,'skipped_cells':summary['skipped_cells'],
              'updated_utc':base.now(),'elapsed_seconds':summary['elapsed_seconds']})
    except BaseException as exc:
        write(out/'status.json',{'protocol':PROTOCOL,'status':'ERROR','pid':os.getpid(),
              'completed_cells':completed,'updated_utc':base.now(),'error':repr(exc),'traceback':traceback.format_exc()})
        traceback.print_exc(); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True); p.add_argument('--qualification',required=True)
    a=p.parse_args(); run((ROOT/a.out).resolve(),(ROOT/a.qualification).resolve())
