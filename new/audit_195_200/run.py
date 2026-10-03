"""Bound zero-training intervention suite for historical StreamingCell u195/u200."""
import argparse
from collections import deque
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REF = ROOT/'runs/transition_20261003_seed4_dense01'
PROTOCOL = 'audit195_200_v1'
for relative in ('new/nca_inertial_wind_tunnel', 'new/workspace_revision',
                 'new/streaming_carry'):
    sys.path.insert(0, str(ROOT/relative))
sys.path.insert(0,str(HERE))
from tasks import bank
from stream_cells import StreamingCell
from run_revision import sha, tensor_hash, write, now
from instrument import parts, semantic_decomposition
from state_interventions import (build_swaps, apply_swap, apply_z_projection_swap,
                                transplant_between, identity_plan)


class NonfiniteIntervention(Exception):
    def __init__(self, step, start, traces):
        self.step=step
        self.start=start
        self.traces={head:{key:value[:step-start+1].copy() for key,value in trace.items()}
                     for head,trace in traces.items()}
        super().__init__(f'Nonfinite intervention at/before macrostep{step}')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


metrics = module('audit195_metrics', HERE/'metrics.py')
probes = module('audit195_probes', HERE/'probes.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def bindings():
    old = read(REF/'manifest.json')['source_sha256']
    for name, expected in old.items():
        assert sha(ROOT/name) == expected, f'Historical source drift: {name}'
    names = [p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py')]
    names += ['new/audit_195_200/PROTOCOL.md', 'tools/launch_audit_195_200.ps1']
    source = {**old, **{n:sha(ROOT/n) for n in sorted(names)}}
    inputs, params = {}, {}
    for size in (32,64):
        inputs[f'bank_size{size}.npz'] = sha(REF/f'bank_size{size}.npz')
        for u in (195,200):
            row = read(REF/f'checkpoints/u{u}_size{size}.json')
            assert sha(REF/f'checkpoints/u{u}_size{size}.npz') == row['trace_sha256']
            data = torch.load(REF/f'checkpoints/u{u}.pt',map_location='cpu',weights_only=True)
            assert data['completed_updates'] == u and data['seed'] == 4
            assert tensor_hash(data['state_dict']) == row['parameter_sha256']
            params[str(u)] = row['parameter_sha256']
            for suffix in ('json','npz'):
                name = f'checkpoints/u{u}_size{size}.{suffix}'
                inputs[name] = sha(REF/name)
            inputs[f'checkpoints/u{u}.pt'] = sha(REF/f'checkpoints/u{u}.pt')
    return {'source_sha256':source, 'input_sha256':inputs, 'parameter_sha256':params}


def setup():
    torch.set_num_threads(2)
    assert torch.__version__.split('+')[0] == '2.5.1'
    assert torch.cuda.is_available()
    free,_ = torch.cuda.mem_get_info()
    assert free >= 3500*1024**2, 'Insufficient GPU headroom'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


def cpu_bank(size, confirmation):
    if confirmation:
        return bank(size,8,62000+size,'cpu')
    with np.load(REF/f'bank_size{size}.npz') as saved:
        return {k:torch.from_numpy(saved[k].copy()) for k in saved.files}


def cpu_state(state):
    return tuple(v.detach().cpu().clone() for v in state)


def gpu_state(state):
    return tuple(v.cuda() for v in state)


def blocks(key):
    for letter,prefixes in {'E':('encoder.',),'F':('f_in.','f_out.'),
                            'Q':('q_in.','q_out.'),'R':('readout.',)}.items():
        if key.startswith(prefixes):
            return letter
    raise ValueError(key)


class Runner:
    def __init__(self,out,bound,preflight):
        self.out,self.bound,self.preflight = out,bound,preflight
        self.rows = []
        self.started = time.perf_counter()
        self.states = {u:torch.load(REF/f'checkpoints/u{u}.pt',
                        map_location='cpu',weights_only=True)['state_dict'] for u in (195,200)}
        self.models = {u:self.model(self.states[u]) for u in (195,200)}
        self.controls = []
        self.progress('REPLAY',0)

    def model(self,weights):
        m=StreamingCell(streaming=True).cuda().eval()
        m.load_state_dict(weights)
        return m

    def progress(self,phase,completed=None):
        write(self.out/'status.json', {'status':'RUNNING','phase':phase,
            'completed_cases':len(self.rows) if completed is None else completed,
            'elapsed_seconds':time.perf_counter()-self.started,'pid':os.getpid(),
            'updated_utc':now(),'runtime_limit_enforced':False})

    @torch.no_grad()
    def rollout(self,model,data,state=None,heads=None,pulse_kind=None,pulse_mask=None,capture=False):
        start = 0 if state is None else 64
        a,b = (model.initial(data['x']),model.initial(data['x_flip'])) if state is None else tuple(gpu_state(s) for s in state)
        heads = heads or {str(model._audit_label):model.readout}
        shape=(257-start,len(data['x']),data['x'].shape[-2],data['x'].shape[-1])
        traces={k:{name:np.empty(shape,np.float32 if 'margin' in name else bool)
                      for name in ('correct','original_correct','flipped_correct','margin','original_margin','flipped_margin')}
                for k in heads}
        saved={}
        finite=torch.ones((),dtype=torch.bool,device='cuda')
        for t in range(start,257):
            if t>start:
                if pulse_kind is not None and t==65:
                    a=probes.pulse(model,a,data['x'],pulse_mask,pulse_kind)
                    b=probes.pulse(model,b,data['x_flip'],pulse_mask,pulse_kind)
                else:
                    a,b=model.step(a,data['x']),model.step(b,data['x_flip'])
            for v in (*a,*b):
                finite=finite & torch.isfinite(v).all()
            for name,head in heads.items():
                la,lb=head(a[1]),head(b[1])
                ca=(la>=0)==(data['y']>=.5)
                cb=(lb>=0)==(data['y_flip']>=.5)
                ma,mb=(2*data['y']-1)*la,(2*data['y_flip']-1)*lb
                for key,value in (('correct',ca&cb),('original_correct',ca),('flipped_correct',cb),
                                  ('margin',torch.minimum(ma,mb)),('original_margin',ma),('flipped_margin',mb)):
                    traces[name][key][t-start]=value[:,0].cpu().numpy()
                finite=finite & torch.isfinite(la).all() & torch.isfinite(lb).all()
            if capture and t in (64,96,128):
                saved[t]=(cpu_state(a),cpu_state(b))
            if t%8==0 or t==256:
                if not bool(finite):
                    if start==0 and model._audit_label in (195,200):
                        raise AssertionError(f'Nonfinite control state/logit at/before{t}')
                    raise NonfiniteIntervention(t,start,traces)
        return traces,saved

    def intervention(self,model,data,state=None,heads=None,pulse_kind=None,pulse_mask=None):
        try:
            return self.rollout(model,data,state,heads,pulse_kind,pulse_mask)[0]
        except NonfiniteIntervention as error:
            labels=heads or {str(model._audit_label):None}
            return {name:{'nonfinite_at_or_before':error.step,'failed_trace':error.traces[name],
                          'trace_start_step':error.start} for name in labels}

    def record(self,name,group,trace,cpu,cohorts=None,extra=None,full=False):
        if 'nonfinite_at_or_before' in trace:
            failed_path=self.out/'traces'/f'{name}_nonfinite.npz'
            np.savez_compressed(failed_path,**trace['failed_trace'])
            row={'case':name,'group':group,'bank':self.bank_name,'size':self.size,
                 'confirmation':self.confirmation,'summary':None,'extra':extra or {},
                 'scientific_outcome':'NONFINITE_INTERVENTION','nonfinite_at_or_before':trace['nonfinite_at_or_before'],
                 'trace_start_step':trace['trace_start_step'],'failed_trace_file':failed_path.relative_to(self.out).as_posix(),
                 'failed_trace_sha256':sha(failed_path),'bank_sha256':self.bank_sha}
            write(self.out/'cases'/f'{name}.json',row)
            self.rows.append(row);self.progress(group)
            return row
        # Every canonical case is a64..256 suffix, regardless of initialization.
        if len(trace['correct'])==257:
            trace={k:v[64:] for k,v in trace.items()}
        row={'case':name,'group':group,'bank':self.bank_name,'size':self.size,
             'confirmation':self.confirmation,'summary':metrics.summarize(trace,cpu,cohorts),
             'extra':extra or {},'steps':256,'zero_training':True,'bank_sha256':self.bank_sha}
        payload={'correct':trace['correct'],'original_correct':trace['original_correct'],
                 'flipped_correct':trace['flipped_correct'],
                 'margin_endpoints':trace['margin'][[0,64,192]],
                 'original_margin_endpoints':trace['original_margin'][[0,64,192]],
                 'flipped_margin_endpoints':trace['flipped_margin'][[0,64,192]]}
        if full:
            payload.update({k:v for k,v in trace.items() if 'margin' in k})
        if cohorts:
            payload.update({f'cohort_{k}':v for k,v in cohorts.items()})
        path=self.out/'traces'/f'{name}.npz'
        np.savez_compressed(path,**payload)
        row['trace_file']=path.relative_to(self.out).as_posix()
        row['trace_sha256']=sha(path)
        write(self.out/'cases'/f'{name}.json',row)
        self.rows.append(row)
        self.progress(group)
        return row

    def suffix(self,model,data,states,heads=None,pulse_kind=None,pulse_mask=None):
        return self.intervention(model,data,states,heads,pulse_kind,pulse_mask)

    def label_models(self):
        for u,m in self.models.items(): m._audit_label=u

    def replay(self,cpu,data):
        baseline,states={},{ }
        for u,m in self.models.items():
            traces,saved=self.rollout(m,data,capture=True)
            trace=traces[str(u)]
            if not self.confirmation:
                with np.load(REF/f'checkpoints/u{u}_size{self.size}.npz') as ref:
                    for key in ('correct','original_correct','flipped_correct'):
                        assert np.array_equal(trace[key],ref[key]),f'u{u}size{self.size} Boolean replay drift {key}'
                    error=float(np.max(np.abs(trace['margin']-ref['margin'])))
                    assert error<=2e-7,f'Margin replay drift {error}'
                self.controls.append({'update':u,'size':self.size,'Boolean_exact':True,'margin_max_error':error})
            baseline[u],states[u]=trace,saved
            self.record(f'{self.bank_name}_baseline{u}','baseline',trace,cpu,full=True)
            arrays={f'{world}_{part}_t{t}':saved[t][j][c].numpy()
                    for t in saved for j,world in enumerate(('original','flipped')) for c,part in enumerate(('W','Z'))}
            np.savez_compressed(self.out/'states'/f'{self.bank_name}_u{u}.npz',**arrays)
        return baseline,states

    @torch.no_grad()
    def preflight_interventions(self,cpu,data,baseline,states):
        checks=[]
        domain=cpu['changed'][:,0].numpy().astype(bool)&(cpu['distance'][:,0].numpy()>0)
        for u in (195,200):
            tr=baseline[u];correct=tr['correct'][64]
            age=np.zeros_like(correct,np.int32)
            for t in range(65):age=np.where(tr['correct'][t],age+1,0)
            for role in ('solved','frontier'):
                plan=build_swaps(cpu,correct,tr['margin'][64],age,role,73000+self.size+u,max_pairs_per_map=8)
                for base in states[u][64]:
                    new=apply_z_projection_swap(base,plan,self.models[u].readout.weight.detach().cpu(),'null')
                    weight=self.models[u].readout.weight.detach().cpu().reshape(1,8,1,1)
                    err=float(((new[1]-base[1])*weight).sum(1).abs().max())
                    assert err<=1e-6,f'Actual-state null intervention drift:{err}'
                    actual=float((self.models[u].logits(gpu_state(new))-self.models[u].logits(gpu_state(base))).abs().max())
                    assert actual<=1e-6,f'Actual GPU null readout drift:{actual}'
                    checks.append({'update':u,'role':role,'targets':plan.counts['swap_targets'],
                                   'null_linear_projection_error':err,'null_logit_error':actual})
            for j in (0,1):
                state=gpu_state(states[u][64][j]);x=data['x' if j==0 else 'x_flip']
                p=parts(self.models[u],state,x);expected=self.models[u].step(state,x)
                assert torch.equal(expected[0],p['W_new']) and torch.equal(expected[1],p['Z_new'])
                sem=semantic_decomposition(self.models[u],state,x,p)
                err=float((sem['dlogits_sum']-sem['dlogits']).abs().max())
                assert err<=2e-6
                zero=torch.zeros_like(data['mask'],dtype=torch.bool)
                mixed=torch.from_numpy(domain[:,None]).cuda()
                for kind in probes.PULSE_KINDS:
                    sham=probes.pulse(self.models[u],state,x,zero,kind)
                    assert all(torch.equal(a,b) for a,b in zip(sham,expected))
                    changed=probes.pulse(self.models[u],state,x,mixed,kind)
                    assert all(bool(torch.isfinite(v).all()) for v in changed)
                labels=data['y' if j==0 else 'y_flip']
                probes.summary(self.models[u],state,x,{'changed':mixed},labels)
                checks.append({'update':u,'world':j,'parts_exact':True,'telescope_error':err,'eight_pulses_passed':True})
        self.controls.append({'size':self.size,'actual_checkpoint_checks':checks})

    def experiments(self,cpu,data,baseline,states):
        domain=cpu['changed'][:,0].numpy().astype(bool)&cpu['mask'][:,0].numpy().astype(bool)&(cpu['distance'][:,0].numpy()>0)
        source=cpu['x'][:,1:2].ne(0)|cpu['x'][:,2:3].ne(0)
        cpu={**cpu,'source':source}
        common=domain.copy()
        for producer in (195,200):
            for head in (195,200):
                a,b=states[producer][64]
                with torch.no_grad():
                    ma=self.models[head].readout(a[1].cuda())
                    mb=self.models[head].readout(b[1].cuda())
                    correct=((ma>=0)==(data['y']>=.5))&((mb>=0)==(data['y_flip']>=.5))
                common &= correct[:,0].cpu().numpy()
        fixed={'common_solved64':common}
        # Full factorial lets all24 block orders be assessed without24 reruns.
        readout_controls={}
        for bits in itertools.product((0,1),repeat=4):
            selected=''.join(k for k,b in zip('EFQR',bits) if b) or 'none'
            if selected in ('none','EFQR'):
                u=195 if selected=='none' else 200
                tr=baseline[u]
            else:
                weights={k:self.states[200 if blocks(k) in selected else 195][k] for k in self.states[195]}
                m=self.model(weights);m._audit_label=selected
                tr=self.intervention(m,data)[selected]
                del m
            self.record(f'{self.bank_name}_factorial_{selected}','factorial',tr,cpu,fixed,
                        {'blocks_from200':selected,'parameter_coordinate_alignment':'historical shared path'})
            if selected in ('R','EFQ') and 'nonfinite_at_or_before' not in tr:
                readout_controls[selected]={k:v[64:].copy() for k,v in tr.items()}
        for i in range(1,10):
            lam=i/10
            weights={k:(1-lam)*self.states[195][k]+lam*self.states[200][k] for k in self.states[195]}
            m=self.model(weights);m._audit_label='interpolate'
            tr=self.intervention(m,data)['interpolate'];del m
            self.record(f'{self.bank_name}_lambda{i:02d}','interpolation',tr,cpu,fixed,{'lambda':lam})
        for producer,rule in itertools.product((195,200),repeat=2):
            tr=self.suffix(self.models[rule],data,states[producer][64],
                           {str(h):self.models[h].readout for h in (195,200)})
            for head in (195,200):
                if producer==rule==head:
                    assert all(np.array_equal(tr[str(head)][k],baseline[head][k][64:]) for k in tr[str(head)]),'Diagonal continuation drift'
                equivalent='R' if producer==rule==195 and head==200 else ('EFQ' if producer==rule==200 and head==195 else None)
                if equivalent in readout_controls:
                    assert all(np.array_equal(tr[str(head)][k],readout_controls[equivalent][k]) for k in tr[str(head)]),'Readout affected hidden dynamics'
                cross_cohorts={**fixed}
                if 'nonfinite_at_or_before' not in tr[str(head)]:
                    cross_cohorts['arm_own_solved64']=domain&tr[str(head)]['correct'][0]
                self.record(f'{self.bank_name}_crossS{producer}_D{rule}_R{head}','cross_continuation',tr[str(head)],cpu,cross_cohorts,
                            {'producer':producer,'continuation':rule,'readout':head})
        roles={}
        for u in (195,200):
            trace=baseline[u];correct=trace['correct'][64]
            age=np.zeros_like(correct,np.int32)
            for t in range(65):age=np.where(trace['correct'][t],age+1,0)
            solved=domain&correct
            adjacent=np.zeros_like(solved)
            adjacent[:,1:]|=solved[:,:-1];adjacent[:,:-1]|=solved[:,1:]
            adjacent[:,:,1:]|=solved[:,:,:-1];adjacent[:,:,:-1]|=solved[:,:,1:]
            roles[u]={'solved':solved,'frontier':domain&~correct&adjacent}
            sham=identity_plan(correct.shape)
            for state in states[u][64]:
                unchanged=apply_swap(state,sham,'both')
                assert all(torch.equal(a,b) for a,b in zip(unchanged,state))
            for role in ('solved','frontier'):
                plan=build_swaps(cpu,correct,trace['margin'][64],age,role,73000+self.size+u,
                                 max_pairs_per_map=8)
                plan_name=f'{self.bank_name}_u{u}_{role}'
                write(self.out/'plans'/f'{plan_name}.json',{'counts':plan.counts,'metadata':plan.metadata,
                      'target_flat':plan.target_flat.tolist(),'donor_flat':plan.donor_flat.tolist()})
                localized={**fixed,**rings(plan.mask,cpu['mask'][:,0].numpy().astype(bool)),
                           'selected_role':roles[u][role]}
                variants=[('W','W',None),('Z','Z',None),('both','both',None),('span','span',None),('null','null',None)]
                if not self.confirmation:
                    variants += [(f'lane{j}','W',range(j*6,(j+1)*6)) for j in range(4)]
                    variants += [(f'Z{j}','Z',[j]) for j in range(8)]
                for label,component,channels in variants:
                    changed=[]
                    for state in states[u][64]:
                        if component in ('span','null'):
                            swapped=apply_z_projection_swap(state,plan,self.models[u].readout.weight.detach().cpu(),component)
                        else:swapped=apply_swap(state,plan,component,channels)
                        changed.append(swapped)
                    norm=perturbation(states[u][64],changed)
                    if component=='null':
                        errors=[]
                        with torch.no_grad():
                            for base,new in zip(states[u][64],changed):
                                weight=self.models[u].readout.weight.detach().cpu().reshape(1,8,1,1)
                                errors.append(float(((new[1]-base[1])*weight).sum(1).abs().max()))
                        assert max(errors)<=1e-6,f'Null projection logit drift{errors}'
                        with torch.no_grad():
                            actual_errors=[float((self.models[u].logits(gpu_state(new))-self.models[u].logits(gpu_state(base))).abs().max())
                                           for base,new in zip(states[u][64],changed)]
                        assert max(actual_errors)<=1e-6,f'Actual null readout drift:{actual_errors}'
                        norm['null_linear_projection_max_error']=max(errors)
                        norm['null_logit_max_error']=max(actual_errors)
                    tr=self.suffix(self.models[u],data,changed)[str(u)]
                    self.record(f'{plan_name}_swap{label}','state_swap',tr,cpu,localized,
                                {'checkpoint':u,'role':role,'component':label,'plan':f'plans/{plan_name}.json',
                                 'perturbation':norm,'eligible_targets':plan.counts['swap_targets'],
                                 'interpretability':'NO_ELIGIBLE_SWAP' if not plan.counts['swap_targets'] else 'CONDITIONAL_OFF_TRAJECTORY'})
            # Update accounting at actual trajectory states, not perturbed states.
            alternate=self.model(self.states[u])
            alternate.readout.load_state_dict(self.models[200 if u==195 else 195].readout.state_dict())
            for t in (64,96,128):
                ct=baseline[u]['correct'][t]
                adjacent=np.zeros_like(ct)
                solved_t=domain&ct
                adjacent[:,1:]|=solved_t[:,:-1];adjacent[:,:-1]|=solved_t[:,1:]
                adjacent[:,:,1:]|=solved_t[:,:,:-1];adjacent[:,:,:-1]|=solved_t[:,:,1:]
                cohort={'solved':domain&ct,'wrong':domain&~ct,
                        'frontier':domain&~ct&adjacent,
                        'solved_new':domain&ct&~np.all(baseline[u]['correct'][max(0,t-8):t+1],axis=0),
                        'solved_old':domain&ct&np.all(baseline[u]['correct'][max(0,t-16):t+1],axis=0)}
                for j,world in enumerate(('original','flipped')):
                    state=gpu_state(states[u][t][j]);x=data['x' if j==0 else 'x_flip']
                    labels=data['y' if j==0 else 'y_flip']
                    for head,probe_model in ((u,self.models[u]),(200 if u==195 else 195,alternate)):
                        with torch.no_grad():
                            p=parts(probe_model,state,x)
                            expected=self.models[u].step(state,x)
                            assert torch.equal(expected[0],p['W_new']) and torch.equal(expected[1],p['Z_new'])
                            sem=semantic_decomposition(probe_model,state,x,p)
                            residual=float((sem['dlogits_sum']-sem['dlogits']).abs().max())
                            assert residual<=2e-6
                            masks={name:torch.from_numpy(value[:,None]).cuda() for name,value in cohort.items()}
                            row=probes.summary(probe_model,state,x,masks,labels)
                        write(self.out/'updates'/f'{self.bank_name}_u{u}_t{t}_{world}_R{head}.json',
                              {'checkpoint':u,'time':t,'world':world,'readout':head,'telescope_max_error':residual,'summary':row})
            del alternate
            for role in ('solved','frontier'):
                selected=roles[u][role]
                mask=torch.from_numpy(selected[:,None]).cuda()
                for kind in ('drop_F','drop_Q','identity_stream','Q_without_W','Q_without_Z',
                             'Q_without_LW','Q_without_LZ','Q_without_X'):
                    tr=self.suffix(self.models[u],data,states[u][64],pulse_kind=kind,pulse_mask=mask)[str(u)]
                    self.record(f'{self.bank_name}_u{u}_{role}_pulse{kind}','update_pulse',tr,cpu,
                                {**fixed,'selected_role':selected}, {'checkpoint':u,'role':role,'kind':kind,
                                'pulse_transition':'64->65','selected_cells':int(selected.sum())})
        for receiver,donor in ((195,200),(200,195)):
            for role in ('solved','frontier'):
                eligible=roles[receiver][role]&roles[donor][role]
                selected=sparse_cells(eligible,75000+self.size,16)
                for component in ('W','Z','both'):
                    changed=[transplant_between(a,b,selected,component)
                             for a,b in zip(states[receiver][64],states[donor][64])]
                    tr=self.suffix(self.models[receiver],data,changed)[str(receiver)]
                    self.record(f'{self.bank_name}_local{receiver}from{donor}_{role}_{component}',
                                'local_transplant',tr,cpu,{**fixed,**rings(selected,cpu['mask'][:,0].numpy().astype(bool))},
                                {'receiver':receiver,'donor':donor,'role':role,'component':component,
                                 'eligible_cells':int(eligible.sum()),'targets':int(selected.sum()),
                                 'perturbation':perturbation(states[receiver][64],changed)})

    def one_bank(self,size,confirmation,experiments):
        self.size,self.confirmation=size,confirmation
        self.bank_name=f'{"confirmation" if confirmation else "primary"}{size}'
        cpu=cpu_bank(size,confirmation)
        bank_path=self.out/f'bank_{self.bank_name}.npz'
        np.savez_compressed(bank_path,**{k:v.numpy() for k,v in cpu.items()})
        self.bank_sha=sha(bank_path)
        if not confirmation:
            expected=read(REF/'manifest.json')['audit_banks'][str(size)]['tensor_sha256']
            assert tensor_hash(cpu)==expected,'Evaluation bank tensor drift'
        manifest=read(self.out/'manifest.json')
        manifest.setdefault('audit_banks',{})[self.bank_name]={'maps':len(cpu['x']),
            'seed':(62000 if confirmation else 61000)+size,'file':bank_path.name,
            'sha256':self.bank_sha,'tensor_sha256':tensor_hash(cpu)}
        write(self.out/'manifest.json',manifest)
        data={k:v.cuda() for k,v in cpu.items()}
        self.progress('REPLAY')
        baseline,states=self.replay(cpu,data)
        if experiments:self.experiments(cpu,data,baseline,states)
        return cpu,data,baseline,states


def perturbation(base,new):
    return {f'{world}_{part}_rms':float((b-a).double().square().mean().sqrt())
            for world,(pair0,pair1) in zip(('original','flipped'),zip(base,new))
            for part,a,b in zip(('W','Z'),pair0,pair1)}


def sparse_cells(eligible,seed,maximum):
    rng=np.random.default_rng(seed);selected=np.zeros_like(eligible)
    for b in range(len(eligible)):
        choices=np.flatnonzero(eligible[b]);choices=rng.permutation(choices)[:maximum]
        selected[b].reshape(-1)[choices]=True
    return selected


def rings(targets,opened):
    d=np.full(targets.shape,-1,np.int32)
    for b in range(len(targets)):
        queue=deque((int(y),int(x)) for y,x in np.argwhere(targets[b]))
        for y,x in queue:d[b,y,x]=0
        while queue:
            y,x=queue.popleft()
            for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=yy<d.shape[1] and 0<=xx<d.shape[2] and opened[b,yy,xx] and d[b,yy,xx]<0:
                    d[b,yy,xx]=d[b,y,x]+1;queue.append((yy,xx))
    return {'targets':d==0,'downstream1_4':(d>=1)&(d<=4),
            'downstream5_16':(d>=5)&(d<=16),'downstream17_32':(d>=17)&(d<=32),'downstream33_plus':d>=33}


def report(out,rows,complete):
    write(out/'summary.json',{'protocol':PROTOCOL,'execution_complete':complete,
          'decision':'CAUSAL_INTERVENTIONS_COMPLETE' if complete else 'PENDING',
          'training_trajectories':1,'new_training':False,'cases':rows,
          'phase_transition_proven':False,'universal_type_closure_proven':False})
    lines=['# Joint195/200 audit','',f'Execution: {"COMPLETE" if complete else "RUNNING"}; {len(rows)} recorded cases.',
           'One selected training trajectory; primary and independent diagnostic task maps. No new training.',
           '', 'Read summary.json and cases/*.json for fixed-cohort counts and localized effects.',
           'Raw traces and state tensors are secondary. A parameter/state intervention identifies conditional',
           'sensitivity or sufficiency in this checkpoint background, not a universal mechanism.', '',
           '| Group | Primary cases | Confirmation cases |','|---|---:|---:|']
    for group in sorted({r['group'] for r in rows}):
        lines.append(f"| {group} | {sum(r['group']==group and not r['confirmation'] for r in rows)} | {sum(r['group']==group and r['confirmation'] for r in rows)} |")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--preflight',action='store_true')
    p.add_argument('--preflight-dir')
    args=p.parse_args()
    bound=bindings()
    out=(ROOT/args.out).resolve()
    assert out.is_relative_to(ROOT/'runs') and not out.exists(),'New project run directory required'
    if not args.preflight:
        pf=(ROOT/args.preflight_dir).resolve()
        assert pf.is_relative_to(ROOT/'runs')
        assert read(pf/'status.json')['status']=='PREFLIGHT_PASSED'
        assert read(pf/'manifest.json')['bindings']==bound
        assert read(pf/'aggregate.json')['pass']
    out.mkdir(parents=True)
    for sub in ('source','cases','traces','states','plans','updates'):(out/sub).mkdir()
    write(out/'status.json',{'status':'RUNNING','phase':'SETUP','completed_cases':0,'pid':os.getpid(),'started_utc':now()})
    try:
        for name in bound['source_sha256']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
            assert sha(target)==bound['source_sha256'][name]
        setup()
    except BaseException:
        write(out/'status.json',{'status':'ERROR','phase':'SETUP','pid':os.getpid(),'traceback':traceback.format_exc(),'updated_utc':now()})
        raise
    manifest={'protocol':PROTOCOL,'preflight':args.preflight,'bindings':bound,'pid':os.getpid(),
              'host':os.environ.get('COMPUTERNAME'),'command':sys.argv,'started_utc':now(),
              'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
              'threads':2,'backend':{'cudnn_benchmark':False,'cudnn_deterministic':False,'cudnn_tf32':True,'matmul_tf32':False},
              'maximum_seconds':None,'runtime_limit_enforced':False,'watchdog_enabled':False,'continuous_monitoring':False,
              'zero_training':True,'training_trajectories':1,'git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    write(out/'manifest.json',manifest)
    try:
        if args.preflight:
            fixtures=[]
            for name in ('check_instrument.py','check_state_interventions.py','check_metrics.py','check_probes.py','check_analyze.py'):
                result=subprocess.run([sys.executable,'-X','utf8','-B',str(HERE/name)],cwd=ROOT,
                                      capture_output=True,text=True,encoding='utf-8')
                fixtures.append({'script':name,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
                assert result.returncode==0,f'CPU fixture failed:{name}\n{result.stderr}'
            write(out/'cpu_checks.json',fixtures)
        runner=Runner(out,bound,args.preflight);runner.label_models()
        for size in (32,64):
            cpu,data,baseline,states=runner.one_bank(size,False,not args.preflight)
            if args.preflight:
                t0=time.perf_counter()
                tr=runner.suffix(runner.models[200],data,states[200][64])['200']
                assert all(np.array_equal(tr[k],baseline[200][k][64:]) for k in tr),'Suffix sham drift'
                domain=cpu['changed'].bool()&(cpu['distance']>0)
                altered=runner.suffix(runner.models[200],data,states[200][64],pulse_kind='drop_Q',pulse_mask=domain.cuda())['200']
                assert np.isfinite(altered['margin']).all()
                runner.record(f'primary{size}_smoke_pulse','smoke',altered,cpu)
                runner.preflight_interventions(cpu,data,baseline,states)
                runner.controls.append({'size':size,'suffix_sham_exact':True,'pulse_finite':True,
                                        'two_suffix_seconds':time.perf_counter()-t0})
        if args.preflight:
            measured=sum(r.get('two_suffix_seconds',0) for r in runner.controls)/4
            aggregate={'pass':True,'replay_controls':runner.controls,'cpu_fixtures_passed':5,
                       'estimated_formal_seconds':measured*492,'expected_formal_cases':492,'runtime_limit_enforced':False,
                       'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            write(out/'aggregate.json',aggregate)
            report(out,runner.rows,False)
            write(out/'status.json',{'status':'PREFLIGHT_PASSED','protocol':PROTOCOL,'completed_cases':len(runner.rows),'pid':os.getpid(),
                                  'elapsed_seconds':time.perf_counter()-runner.started})
        else:
            for size in (32,64):runner.one_bank(size,True,True)
            assert len(runner.rows)==492,'Prespecified case enumeration mismatch'
            report(out,runner.rows,True)
            module('audit195_analysis',HERE/'analyze.py').analyze(out)
            write(out/'aggregate.json',{'pass':True,'replay_controls':runner.controls,'cases':len(runner.rows),
                                      'elapsed_seconds':time.perf_counter()-runner.started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
            write(out/'status.json',{'status':'COMPLETE','completed_cases':len(runner.rows),'pid':os.getpid(),
                                  'elapsed_seconds':time.perf_counter()-runner.started,'completed_utc':now()})
        print(json.dumps({'status':read(out/'status.json')['status'],'out':str(out),'cases':len(runner.rows)}),flush=True)
    except BaseException:
        write(out/'status.json',{'status':'ERROR','pid':os.getpid(),'traceback':traceback.format_exc(),'updated_utc':now()})
        raise


if __name__=='__main__':main()
