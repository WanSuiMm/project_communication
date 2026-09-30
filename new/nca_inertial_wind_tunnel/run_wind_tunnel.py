"""A small train/evaluate harness for non-PDE spatial computation.

Examples:
  python run_wind_tunnel.py --smoke --device cpu
  python run_wind_tunnel.py --device cuda --arms all --seed 0 --steps 800

No dataset downloads. No automatic hyperparameter search. No background jobs.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from cells import ARMS, make_cell
from tasks import bank, subset, damage, balanced_loss, metrics, per_example_balanced_accuracy


DEADLINE = None


def check_budget():
    if DEADLINE is not None and time.monotonic() >= DEADLINE:
        raise TimeoutError('Frozen run wall-clock budget exhausted')


def write_json(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def sustained_threshold(curve,threshold=.95):
    horizons=sorted(int(t) for t in curve if str(t).isdigit())
    return next((t for t in horizons if all(
        curve[str(k)]['balanced_accuracy']>=threshold for k in horizons if k>=t)),None)


def write_summary(out,statuses,smoke,seed):
    rows=['# Local inertial NCA wind tunnel','',
          'SMOKE ONLY: these updates are not efficacy evidence.' if smoke else
          'Seed 0 exploratory screen. One model seed and 16 held-out maps per size; no population-level superiority claim.',
          '', '| Arm | Status | Updates | Train seconds |', '|---|---|---:|---:|']
    metrics_rows=[]
    for arm,status in statuses.items():
        files=list(out.glob(f'{arm}_seed*.json'))
        result=json.loads(files[0].read_text(encoding='utf-8')) if files else {}
        elapsed=result.get('training_seconds_including_python_and_first_compile')
        rows.append(f"| {arm} | {status} | {result.get('completed_updates',0)} | {round(elapsed,2) if elapsed else '-'} |")
        for size,ev in result.get('evaluation',{}).items():
            curve=ev.get('curve',{})
            if not curve:
                continue
            last=max(curve,key=int)
            paired=ev.get('paired_source_information',{}).get(last,{})
            metrics_rows.append({'arm':arm,'size':int(size),'last_horizon':int(last),
                'balanced_accuracy':curve[last]['balanced_accuracy'],
                'paired_correct':paired.get('both_counterfactuals_correct_on_changed_component'),
                'first_sustained_95_steps':ev.get('first_sustained_95_steps'),
                'seconds_per_query_at_sustained_95':ev.get('seconds_per_query_at_sustained_95'),
                'repair_eligible':ev.get('conditional_state_recovery',{}).get('0.0625',{}).get('clean_correct_eligible_count')})
    rows+=['','| Arm | Size | Last T | BA at last T | Paired correct | First sustained 95% T | Repair eligible |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for item in metrics_rows:
        def value(key):
            val=item[key]
            return 'null' if val is None else f'{val:.4f}' if isinstance(val,float) else str(val)
        rows.append('| '+' | '.join(value(key) for key in
            ('arm','size','last_horizon','balanced_accuracy','paired_correct','first_sustained_95_steps','repair_eligible'))+' |')
    rows+=['', 'A missing 95% threshold remains null. Repair eligibility is measured before damage at T=64.',
           'Per-arm JSON retains all horizons, paired distance bins, damage/revision controls, gradients and measured latency.',
           'A completed training run is not an architecture pass. Nonfinite and budget-limited runs retain their status.','']
    (out/'RESULTS.md').write_text('\n'.join(rows),encoding='utf-8')
    write_json(out/'aggregate.json',{'statuses':statuses,'metrics':metrics_rows,'seed':seed})


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def finite_state(state):
    return all(bool(torch.isfinite(v).all()) for v in state if v is not None)


def masked_accuracy(logits,y,mask):
    return float((((logits>=0)==(y>=.5)).float()*mask).sum()/mask.sum().clamp_min(1))


@torch.no_grad()
def evaluation(model,data,horizons,base_steps,recovery_steps):
    x,y,mask=data['x'],data['y'],data['mask']
    state=model.initial(x)
    curves={}
    saved=None
    for t in range(1,max(max(horizons),base_steps)+1):
        check_budget()
        state=model.step(state,x)
        if not finite_state(state):
            return {'status':'NONFINITE_EVAL','first_nonfinite_step':t,'curve':curves}
        if t in horizons:
            curves[str(t)]=metrics(model.logits(state),y,mask,data['distance'])
            curves[str(t)]['content_rms']=float(state[0].square().mean().sqrt())
            curves[str(t)]['velocity_rms']=(float(state[1].square().mean().sqrt())
                                           if state[1] is not None else None)
        if t==base_steps:
            saved=tuple(None if a is None else a.clone() for a in state)
    assert saved is not None
    base_logits=model.logits(saved)
    base_acc=metrics(base_logits,y,mask)['balanced_accuracy']
    ba=per_example_balanced_accuracy(base_logits,y,mask)
    eligible=ba>=.95
    eligible_count=int(eligible.sum())
    recovery={}
    for fraction in (.0625,.25):
        s=damage(saved,fraction,np.random.default_rng(421))
        cold=model.initial(x)
        clean=saved
        rows={'0':metrics(model.logits(s),y,mask)}
        for k in range(1,max(recovery_steps)+1):
            check_budget()
            s=model.step(s,x); cold=model.step(cold,x); clean=model.step(clean,x)
            if not (finite_state(s) and finite_state(cold) and finite_state(clean)):
                rows['status']='NONFINITE_RECOVERY'; rows['first_nonfinite_step']=k
                break
            if k in recovery_steps:
                rows[str(k)]={'damaged_continue':metrics(model.logits(s),y,mask),
                              'cold_restart_equal_extra_steps':metrics(model.logits(cold),y,mask),
                              'undamaged_continue':metrics(model.logits(clean),y,mask),
                              'damaged_continue_on_clean_correct_examples':
                                  metrics(model.logits(s)[eligible],y[eligible],mask[eligible])
                                  if eligible_count else None,
                              'cold_restart_on_clean_correct_examples':
                                  metrics(model.logits(cold)[eligible],y[eligible],mask[eligible])
                                  if eligible_count else None,
                              'undamaged_continue_on_clean_correct_examples':
                                  metrics(model.logits(clean)[eligible],y[eligible],mask[eligible])
                                  if eligible_count else None}
        recovery[str(fraction)]={'base_balanced_accuracy':base_acc,'clean_correct_eligible_count':eligible_count,'total_examples':len(x),'curve':rows}
    # Same geometry, a flipped source identity: output MUST change in that component.
    xf,yf,change=data['x_flip'],data['y_flip'],data['changed']
    warm=saved; cold=model.initial(xf)
    switch={'0':{'changed_component_accuracy':masked_accuracy(model.logits(warm),yf,change),
                 'old_target_accuracy_in_changed_component':masked_accuracy(model.logits(warm),y,change)}}
    for k in range(1,max(recovery_steps)+1):
        check_budget()
        warm=model.step(warm,xf); cold=model.step(cold,xf)
        if not (finite_state(warm) and finite_state(cold)):
            switch['status']='NONFINITE_SWITCH'; switch['first_nonfinite_step']=k
            break
        if k in recovery_steps:
            switch[str(k)]={'warm':metrics(model.logits(warm),yf,mask),
                            'cold_equal_extra_steps':metrics(model.logits(cold),yf,mask),
                            'changed_component_accuracy':masked_accuracy(model.logits(warm),yf,change),
                            'unchanged_component_accuracy':masked_accuracy(model.logits(warm),yf,mask-change)}
    # Compare paired source values at the SAME location, using the trained initialization.
    # This tests whether remote output carries source identity, not just energy.
    pair_state=model.initial(x)
    pair_flip=model.initial(xf)
    paired={}
    for t in range(1,max(horizons)+1):
        check_budget()
        pair_state=model.step(pair_state,x); pair_flip=model.step(pair_flip,xf)
        if not (finite_state(pair_state) and finite_state(pair_flip)):
            paired['status']='NONFINITE_PAIRED'; break
        if t in horizons:
            p0=model.logits(pair_state)>=0; p1=model.logits(pair_flip)>=0
            both=(p0==(y>=.5)) & (p1==(yf>=.5))
            paired[str(t)]={'both_counterfactuals_correct_on_changed_component':
                float((both.float()*change).sum()/change.sum().clamp_min(1)),
                'changed_component_pixels':int(change.sum())}
            bins={}
            for lo,hi in ((0,8),(8,16),(16,32),(32,64),(64,128),(128,100000)):
                selected=change*((data['distance']>=lo)&(data['distance']<hi))
                count=int(selected.sum())
                bins[f'{lo}_{hi}']={'pixels':count,'paired_correct_accuracy':
                    float((both.float()*selected).sum()/selected.sum()) if count else None}
            paired[str(t)]['distance_bins']=bins
    return {'status':'EVALUATED','curve':curves,
            'first_sustained_95_steps':sustained_threshold(curves),
            'threshold_scope':'Maintained only at all subsequent tested checkpoints through max horizon.',
            'conditional_state_recovery':recovery,'source_switch':switch,
            'paired_source_information':paired}


def gradient_probe(model,data,steps):
    """Task-loss sensitivity to the initial state, NOT a largest singular value."""
    x=data['x'][:1]; y=data['y'][:1]; mask=data['mask'][:1]
    raw=model.initial(x)
    state=tuple(None if a is None else a.detach().requires_grad_(True) for a in raw)
    vars_=tuple(a for a in state if a is not None)
    end=model.rollout(x,steps,state)
    loss=balanced_loss(model.logits(end),y,mask)
    if not bool(torch.isfinite(loss)):
        return {'steps':steps,'status':'NONFINITE_PROBE_LOSS'}
    grads=torch.autograd.grad(loss,vars_,allow_unused=False)
    if not all(bool(torch.isfinite(g).all()) for g in grads):
        return {'steps':steps,'status':'NONFINITE_PROBE_GRADIENT'}
    return {'steps':steps,'loss':float(loss.detach()),
            'initial_state_task_gradient_l2':float(torch.sqrt(sum(g.square().sum() for g in grads)))}


@torch.no_grad()
def benchmark(model,x,steps,repeats=5):
    device=x.device
    for _ in range(3):
        check_budget()
        _=model.logits(model.rollout(x,steps))
    synchronize(device)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    samples=[]
    for _ in range(repeats):
        check_budget()
        synchronize(device); start=time.perf_counter()
        _=model.logits(model.rollout(x,steps))
        synchronize(device); samples.append(time.perf_counter()-start)
    return {'batch':len(x),'grid_size':x.shape[-1],'steps':steps,
            'median_batch_seconds':float(np.median(samples)),
            'median_seconds_per_query':float(np.median(samples)/len(x)),
            'median_seconds_per_query_step':float(np.median(samples)/len(x)/steps),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'includes_readout':True,'includes_compilation':False,
            'note':'Warm repeated rollout latency; not a matched-accuracy time-to-solution claim.'}


def train_one(args,arm,device):
    out=Path(args.out)
    write_json(out/'status.json',{'status':'RUNNING','phase':'initializing_arm','arm':arm,'pid':os.getpid()})
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model=make_cell(arm,args.channels,args.hidden).to(device)
    initial_meta=model.metadata()
    if args.compile_cell:
        model.step=torch.compile(model.step)
    train=bank(args.size,args.train_examples,10000+args.seed,device)
    # Separate RNG from model initialization; every arm sees the same sample/order/horizon/damage.
    rng=np.random.default_rng(20000+args.seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    logs=[]; status='TRAINED'; clipped=0; completed=0
    synchronize(device)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    model.train()
    for iteration in range(args.steps):
        try:
            check_budget()
        except TimeoutError:
            status='TIME_BUDGET_TRAIN'; break
        idx=rng.integers(0,args.train_examples,args.batch)
        data=subset(train,idx.tolist())
        T=int(rng.choice(args.train_horizons))
        erase=bool(rng.random()<.5)
        state=model.initial(data['x'])
        optimizer.zero_grad(set_to_none=True)
        sampled_losses=[]
        for t in range(1,T+1):
            state=model.step(state,data['x'])
            if erase and t==T//2:
                state=damage(state,.0625,rng)
            # Two late readouts discourage success at only one oscillation phase.
            if t in {max(1,T-4),T}:
                sampled_losses.append(balanced_loss(model.logits(state),data['y'],data['mask']))
        loss=torch.stack(sampled_losses).mean()
        if not bool(torch.isfinite(loss)):
            status='NONFINITE_TRAIN_LOSS'; break
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(norm)):
            status='NONFINITE_TRAIN_GRAD'; break
        clipped+=int(norm>1.)
        optimizer.step()
        completed=iteration+1
        if iteration%max(1,args.log_every)==0 or iteration+1==args.steps:
            row={'iteration':iteration+1,'rollout':T,'loss':float(loss.detach()),
                 'gradient_norm_before_clip':float(norm),'damage':erase}
            logs.append(row)
            print(json.dumps({'arm':arm,**row}),flush=True)
            write_json(out/'status.json',{'status':'RUNNING','phase':'training','arm':arm,
                                         'completed_updates':completed,'pid':os.getpid()})
            with (out/'training.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps({'arm':arm,**row})+'\n')
    synchronize(device)
    elapsed=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
    model.eval()
    result={'run_type':'SMOKE_NOT_EFFICACY_EVIDENCE' if args.smoke or args.preflight else 'EXPLORATORY_LEARNED_WIND_TUNNEL',
            'status':status,'seed':args.seed,'torch_version':torch.__version__,
            'device':str(device),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
            'config':vars(args),'initial_model':initial_meta,'final_model':model.metadata(),
            'training_seconds_including_python_and_first_compile':elapsed,
            'train_peak_allocated_bytes':peak,
            'completed_updates':completed,
            'gradient_clip_fraction':clipped/max(1,completed),'training_curve':logs,
            'evaluation':{}}
    checkpoint={'arm':arm,'channels':args.channels,'reference_hidden':args.hidden,
                'completed_updates':completed,
                'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
    torch.save(checkpoint,out/f'{arm}_seed{args.seed}.pt')
    dest=out/f'{arm}_seed{args.seed}.json'
    write_json(dest,result)
    if status=='TRAINED':
        try:
            for size in args.eval_sizes:
                check_budget()
                write_json(out/'status.json',{'status':'RUNNING','phase':'evaluation',
                    'arm':arm,'grid_size':size,'pid':os.getpid()})
                test=bank(size,args.eval_examples,30000+size,device)
                ev=evaluation(model,test,args.eval_horizons,args.base_steps,args.recovery_steps)
                result['evaluation'][str(size)]=ev
                write_json(dest,result)
                if ev['status']=='EVALUATED':
                    ev['latency_by_horizon']={str(t):benchmark(model,test['x'][:args.batch],t,3)
                                               for t in args.eval_horizons}
                    threshold=ev['first_sustained_95_steps']
                    ev['seconds_per_query_at_sustained_95']=(
                        ev['latency_by_horizon'][str(threshold)]['median_seconds_per_query']
                        if threshold is not None else None)
                    write_json(dest,result)
            test=bank(args.size,min(args.batch,args.eval_examples),40000,device)
            check_budget()
            result['gradient_probes']=[gradient_probe(model,test,t) for t in args.probe_horizons]
            if any(ev['status']!='EVALUATED' for ev in result['evaluation'].values()):
                result['status']='NONFINITE_EVALUATION'
        except TimeoutError:
            result['status']='TIME_BUDGET_EVAL'
        except Exception as error:
            result['status']='EVALUATION_ERROR'
            result['error']=repr(error)
            result['traceback']=traceback.format_exc()
    write_json(dest,result)
    print(f'Saved {dest}',flush=True)
    del model,optimizer,train
    gc.collect()
    if device.type=='cuda':
        torch.cuda.empty_cache()
    return result['status']


def main():
    global DEADLINE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--arms',nargs='+',default=['all'])
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=800)
    p.add_argument('--channels',type=int,default=16)
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--size',type=int,default=32)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--train-examples',type=int,default=512)
    p.add_argument('--eval-examples',type=int,default=16)
    p.add_argument('--train-horizons',type=int,nargs='+',default=[32,48,64])
    p.add_argument('--eval-horizons',type=int,nargs='+',default=[16,32,64,128,256])
    p.add_argument('--eval-sizes',type=int,nargs='+',default=[32,64,128])
    p.add_argument('--base-steps',type=int,default=64)
    p.add_argument('--recovery-steps',type=int,nargs='+',default=[8,16,32,64])
    p.add_argument('--probe-horizons',type=int,nargs='+',default=[16,64])
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--log-every',type=int,default=100)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--out',required=True,help='New output directory; never overwrite prior evidence.')
    p.add_argument('--minutes',type=float,default=45.,help='Whole-run wall-clock scheduling cap.')
    p.add_argument('--compile-cell',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--preflight',action='store_true',help='Three GPU updates with full model/batch and longest training rollout; abbreviated evaluation.')
    args=p.parse_args()
    if args.smoke:
        args.steps=3; args.channels=4; args.hidden=16; args.size=8
        args.batch=2; args.train_examples=8; args.eval_examples=2
        args.train_horizons=[4,8]; args.eval_horizons=[4,8,16]
        args.eval_sizes=[8]; args.base_steps=8; args.recovery_steps=[2,4,8]
        args.probe_horizons=[4,8]; args.log_every=1
    if args.preflight:
        if args.smoke:
            p.error('Choose either --smoke or --preflight')
        args.steps=3; args.train_horizons=[64]; args.eval_examples=2
        args.eval_sizes=[32]; args.eval_horizons=[16,64]
        args.recovery_steps=[8]; args.probe_horizons=[16]; args.log_every=1
    arms=list(ARMS) if args.arms==['all'] else args.arms
    if any(a not in ARMS for a in arms):
        p.error(f'arms must be in {ARMS}')
    if args.steps<1 or args.batch<1 or args.threads<1 or args.minutes<=0:
        p.error('steps, batch, and threads must be positive')
    if any(t<1 for t in [*args.train_horizons,*args.eval_horizons,*args.recovery_steps,*args.probe_horizons,args.base_steps]):
        p.error('all horizons must be positive')
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable; do not interpret CPU timing as GPU timing')
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parent
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
              'config':vars(args),'torch':torch.__version__,'numpy':np.__version__,
              'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
              'code_sha256':{file.name:hashlib.sha256(file.read_bytes()).hexdigest()
                             for file in source.glob('*.py')}}
    archive=source.parent/'NCA_Inertial_Theory_WindTunnel.zip'
    manifest['original_zip_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None
    manifest['backend']={'cudnn_benchmark':torch.backends.cudnn.benchmark,
                         'cudnn_deterministic':torch.backends.cudnn.deterministic,
                         'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
                         'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32}
    snapshot=out/'source'; snapshot.mkdir()
    for file in source.iterdir():
        if file.is_file() and (file.suffix=='.py' or file.name in ('WIND_TUNNEL.md','THEORY.md','LOCAL_INTEGRATION.md','requirements.txt')):
            shutil.copy2(file,snapshot/file.name)
    write_json(out/'manifest.json',manifest)
    write_json(out/'status.json',{'status':'STARTED','pid':os.getpid(),'arms':arms})
    DEADLINE=time.monotonic()+args.minutes*60
    statuses={}
    for arm in arms:
        if time.monotonic()>=DEADLINE:
            statuses[arm]='NOT_STARTED_TIME_BUDGET'
        else:
            try:
                statuses[arm]=train_one(args,arm,device)
            except Exception as error:
                statuses[arm]='RUN_ERROR'
                write_json(out/f'{arm}_error.json',{'error':repr(error),'traceback':traceback.format_exc()})
                gc.collect()
                if device.type=='cuda':
                    torch.cuda.empty_cache()
        write_summary(out,statuses,args.smoke or args.preflight,args.seed)
    write_json(out/'status.json',{'status':'FINISHED','pid':os.getpid(),'statuses':statuses,
                                'finished_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'completed_statuses':statuses},indent=2))


if __name__=='__main__':
    main()
