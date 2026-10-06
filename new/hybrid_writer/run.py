"""Four matched lane-writer arms; fixed u300 and dense formation evidence."""
from __future__ import annotations
import argparse
import copy
import gc
import importlib.util
import json
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
HERE = Path(__file__).parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

CV = load('_hw_coverage', ROOT/'new/continuous_coverage/run.py')
C, R, E, M = CV.C, CV.R, CV.E, CV.M
H = load('_hw_cells', HERE/'cells.py')
P = load('_hw_reporting', HERE/'reporting.py')
PROTOCOL = 'hybrid_lane_write_budget_v1'
ARMS = ('neural', 'budget', 'hybrid', 'affine_hybrid')
BLOCKS, UPDATES = 8, 300
CHECKPOINTS = tuple(range(0, 301, 25))
INIT_SEEDS = tuple(range(110001, 110009))
SCHEDULE_SEEDS = tuple(range(111001, 111009))
EVAL_SEEDS = {32:112032, 64:112064}
LOSS_FN = CV.LOSS_FN
MODE = 'reset64x4'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def hashes():
    value = CV.source_hashes()
    names = [p for p in HERE.iterdir() if p.suffix in ('.py', '.md')]
    names.append(ROOT/'tools/launch_hybrid_writer.ps1')
    value.update({p.relative_to(ROOT).as_posix():C.sha(p) for p in names})
    return dict(sorted(value.items()))

def common_hash(model):
    return C.tensor_hash({k:v for k,v in model.state_dict().items()
                          if k.split('.')[0] in ('encoder','q_in','q_out','readout')})

def model_for(arm, seed, device='cuda'):
    return H.make_model(arm, seed, device)

def observed_evaluation(model, banks, folder, full=False):
    """Reuse the frozen evaluator; observe existing proposal/write tensors only."""
    telemetry, calls = [], {32:0, 64:0}
    def stats(values):
        if not len(values):
            return {'n':0, 'mean':None, 'max':None, 'quantiles':None}
        return {'n':len(values), 'mean':float(values.mean()), 'max':float(values.max()),
                'quantiles':dict(zip(('p10','p50','p90','p99'),
                                    map(float,np.quantile(values,[.1,.5,.9,.99]))))}
    def observer(details, x):
        size=x.shape[-1]; call=calls[size]; calls[size]+=1
        t=call//2+1
        if t not in (1,8,16,32,64,128,256):return
        b, _, height, width=details['delta'].shape
        selected=x[:,0].bool().cpu().numpy()
        delta=details['delta'].reshape(b,4,6,height,width).square().mean(2).sqrt().cpu().numpy()
        gate=details['gate']
        if gate is not None:
            if gate.ndim==4:gate=gate.unsqueeze(2)
            gate=gate.expand(b,4,1,height,width)[:,:,0].cpu().numpy()
        for lane in range(4):
            ds=stats(delta[:,lane][selected])
            if model.arm!='neural' and ds['max'] is not None:
                assert ds['max']<=.100001, 'Writer budget violated'
            telemetry.append({'size':size,'world':'original' if call%2==0 else 'flipped',
                'step':t,'lane':lane,'open_cells_write_rms':ds,
                'gate':stats(gate[:,lane][selected]) if gate is not None else None})
    assert model.writer_observer is None
    model.writer_observer=observer
    try: measured=E.evaluate(model,banks,folder,full=full)
    finally:model.writer_observer=None
    assert calls=={32:512,64:512},calls
    C.write(folder/'writer_telemetry.json',telemetry)
    return measured

def check(out):
    assert not out.exists(), 'Qualification already exists'
    C.setup_backend(); before=hashes(); started=time.monotonic()
    artifact=out.with_suffix('');artifact.mkdir(parents=True,exist_ok=False)
    # The meaningful CPU invariant check is also callable through its CLI.
    result=subprocess.run([sys.executable,'-X','utf8','-B',str(HERE/'check_cells.py')],
                          cwd=ROOT,capture_output=True,text=True)
    (artifact/'cells_stdout.txt').write_text(result.stdout,encoding='utf-8')
    assert result.returncode==0,result.stderr
    bank=C.region_bank(32,16,113032)
    data={k:v.cuda() for k,v in C.subset(bank,list(range(8))).items()}
    banks={s:C.region_bank(s,32,seed) for s,seed in EVAL_SEEDS.items()}
    rows=[]
    for arm in ARMS:
        base=model_for(arm,110099,'cpu')
        torch.manual_seed(114099)
        with torch.no_grad():
            for name,p in base.named_parameters():
                if name.startswith(('f_out.','q_out.','affine_proposal.')):p.normal_(std=.02)
                if 'beta' in name:p.normal_(std=.2)
        eager=copy.deepcopy(base).cuda(); captured_model=copy.deepcopy(base).cuda()
        before_capture=C.tensor_hash(captured_model.state_dict())
        opt_e=C.optimizer_for(eager);opt_g=C.optimizer_for(captured_model)
        torch.cuda.reset_peak_memory_stats()
        graph=R.CapturedSuperK8(captured_model,data,MODE,LOSS_FN)
        assert C.tensor_hash(captured_model.state_dict())==before_capture
        max_grad=0.;update_seconds=[]
        for update in range(3):
            eager.zero_grad(set_to_none=True)
            loss_e,state_e,finite_e=R.math_backward(eager,data,MODE,LOSS_FN)
            tick=time.monotonic();loss_g,state_g,finite_g=graph.run(data)
            torch.cuda.synchronize();replay_seconds=time.monotonic()-tick
            torch.testing.assert_close(loss_g,loss_e,atol=1e-6,rtol=1e-5)
            for (name,p),(other,q) in zip(eager.named_parameters(),captured_model.named_parameters()):
                assert name==other and p.grad is not None and q.grad is not None
                torch.testing.assert_close(q.grad,p.grad,atol=1e-6,rtol=1e-5)
                max_grad=max(max_grad,float((q.grad-p.grad).abs().max()))
            for a,b in zip(state_g,state_e):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            R.finish_update(eager,opt_e,finite_e);R.finish_update(captured_model,opt_g,finite_g)
            for a,b in zip(eager.parameters(),captured_model.parameters()):
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
                for key in ('exp_avg','exp_avg_sq','step'):
                    torch.testing.assert_close(opt_g.state[b][key],opt_e.state[a][key],atol=1e-6,rtol=1e-5)
            update_seconds.append(replay_seconds)
        tick=time.monotonic()
        observed_evaluation(captured_model,banks,artifact/arm,full=False)
        eval_seconds=time.monotonic()-tick
        rows.append({'arm':arm,'status':'PASS','parameter_count':sum(p.numel() for p in base.parameters()),
                     'three_update_gradient_max_absolute_error':max_grad,
                     'capture_setup_seconds':graph.setup_seconds,
                     'median_captured_backward_seconds':float(np.median(update_seconds)),
                     'actual_32_map_two_size_evaluation_seconds':eval_seconds,
                     'peak_cuda_allocated_MiB':torch.cuda.max_memory_allocated()/2**20})
        print(json.dumps(rows[-1]),flush=True)
        del base,eager,captured_model,graph,opt_e,opt_g,state_e,state_g
        gc.collect();torch.cuda.empty_cache()
    assert hashes()==before
    projected=BLOCKS*sum(r['capture_setup_seconds']+UPDATES*r['median_captured_backward_seconds']
                        +len(CHECKPOINTS)*r['actual_32_map_two_size_evaluation_seconds'] for r in rows)
    eval_bytes=sum(p.stat().st_size for p in artifact.rglob('*') if p.is_file())
    C.write(out,{'status':'PASS','protocol':PROTOCOL,'source_sha256':before,'checked_utc':C.now(),
        'arms':rows,'actual_training_shape':[8,3,32,32],'credit_horizon':8,
        'eager_graph_updates_each_arm':3,'atol':1e-6,'rtol':1e-5,
        'estimated_run_seconds_excluding_data_io_and_optimizer_overhead':projected,
        'estimated_packed_evidence_bytes':eval_bytes*BLOCKS*len(CHECKPOINTS),
        'qualification_seconds':time.monotonic()-started,'runtime_limit_enforced':False})
    print(json.dumps({'status':'PASS','estimated_hours':projected/3600,
                      'estimated_evidence_MiB':eval_bytes*BLOCKS*len(CHECKPOINTS)/2**20}),flush=True)

def run(out,qualification):
    assert not out.exists(),'Run already exists'
    q=read(qualification);source=hashes()
    assert q['status']=='PASS' and q['protocol']==PROTOCOL and q['source_sha256']==source
    C.setup_backend();out.mkdir(parents=True);started=time.monotonic()
    records,final=[],[];plans={}
    for block,(seed,schedule) in enumerate(zip(INIT_SEEDS,SCHEDULE_SEEDS)):
        models={a:model_for(a,seed,'cpu') for a in ARMS}
        shared={a:common_hash(m) for a,m in models.items()};assert len(set(shared.values()))==1
        offset=block%4
        plans[str(block)]={'initialization_seed':seed,'schedule_seed':schedule,
            'arm_order':list(ARMS[offset:]+ARMS[:offset]),'common_initial_sha256':shared['neural'],
            'initial_parameter_sha256':{a:C.tensor_hash(m.state_dict()) for a,m in models.items()},
            'batch_indices':np.random.default_rng(schedule).integers(0,512,(UPDATES,8)).tolist()}
    C.write(out/'plans.json',plans);plan_hash=C.sha(out/'plans.json')
    train_cpu=C.region_bank(32,512,10002);train={k:v.cuda() for k,v in train_cpu.items()}
    banks={s:C.region_bank(s,32,seed) for s,seed in EVAL_SEEDS.items()}
    data_hash={'train':C.tensor_hash(train_cpu),**{f'evaluation{s}':C.tensor_hash(v) for s,v in banks.items()}}
    (out/'banks').mkdir()
    for label,v in {'train':train_cpu,**{f'evaluation{s}':v for s,v in banks.items()}}.items():
        np.savez_compressed(out/'banks'/f'{label}.npz',**{k:a.numpy() for k,a in v.items()})
    config={'protocol':PROTOCOL,'arms':ARMS,'blocks':BLOCKS,'super_updates':UPDATES,
        'architecture':{a:model_for(a,110099,'cpu').metadata() for a in ARMS},
        'mode':MODE,'batch':8,'forward_steps_per_update':256,'loss_windows':32,'credit_horizon':8,
        'optimizer_steps_per_update':1,'parameters_fixed_within_super_update':True,
        'train_seed':10002,'eval_seeds':EVAL_SEEDS,'checkpoints':CHECKPOINTS,'formal_checkpoint':300,
        'optimizer':C.OPTIMIZER,'clip':1.,'runtime_limit_enforced':False}
    C.write(out/'config.json',config)
    manifest={'protocol':PROTOCOL,'pid':os.getpid(),'host':os.environ.get('COMPUTERNAME'),
        'started_utc':C.now(),'command':[sys.executable,*sys.argv],'gpu':torch.cuda.get_device_name(),
        'torch':str(torch.__version__),'numpy':str(np.__version__),'source_sha256':source,
        'data_sha256':data_hash,'schedule_plan_sha256':plan_hash,'qualification_sha256':C.sha(qualification),
        'expected_arms':32,'expected_dense_records':416,'runtime_limit_enforced':False,
        'maximum_seconds':None,'watchdog_enabled':False,'continuous_monitoring':False,
        'git_review_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    C.write(out/'manifest.json',manifest)
    for name in source:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    def progress(value):
        C.write(out/'status.json',{'status':'RUNNING','pid':os.getpid(),'updated_utc':C.now(),
            'elapsed_seconds':time.monotonic()-started,'completed_arms':len(final),'expected_arms':32,
            'completed_dense_records':len(records),**value})
        print(json.dumps(value),flush=True)
    result=None
    try:
        for block_text,plan in plans.items():
            block=int(block_text)
            for arm in plan['arm_order']:
                folder=out/f'block{block:02d}'/arm;(folder/'checkpoints').mkdir(parents=True)
                model=model_for(arm,plan['initialization_seed'])
                assert common_hash(model)==plan['common_initial_sha256']
                assert C.tensor_hash(model.state_dict())==plan['initial_parameter_sha256'][arm]
                opt=C.optimizer_for(model);curve=[];evaluation_seconds=0.
                def save_stage(update):
                    nonlocal evaluation_seconds
                    parameter_hash=C.tensor_hash(model.state_dict())
                    checkpoint=folder/'checkpoints'/f'u{update:03d}.pt'
                    payload=C.payload(model,opt,plan['initialization_seed'],update)
                    payload.update(protocol=PROTOCOL,arm=arm,schedule_seed=plan['schedule_seed'])
                    torch.save(payload,checkpoint)
                    progress({'block':block,'arm':arm,'phase':'evaluation','completed_updates':update})
                    tick=time.monotonic();stage=folder/'evaluation'/f'u{update:03d}'
                    measured=observed_evaluation(model,banks,stage,full=update==UPDATES)
                    elapsed=time.monotonic()-tick;evaluation_seconds+=elapsed
                    row={'block':block,'arm':arm,'update':update,'initialization_seed':plan['initialization_seed'],
                        'schedule_seed':plan['schedule_seed'],'checkpoint':checkpoint.relative_to(out).as_posix(),
                        'checkpoint_sha256':C.sha(checkpoint),'parameter_sha256':parameter_hash,
                        'evaluation_seconds':elapsed,'evaluation_summary':(stage/'summary.json').relative_to(out).as_posix(),
                        'writer_telemetry':(stage/'writer_telemetry.json').relative_to(out).as_posix(),
                        'metrics':M.compact_pair_metrics(measured),'joint':M.joint_readiness(measured),
                        'formal_endpoint':update==UPDATES,'checkpoint_selection':False}
                    if update==UPDATES:
                        row.update(training_seconds=sum(v['seconds'] for v in curve),
                            all_checkpoint_evaluation_seconds=evaluation_seconds,capture_setup_seconds=graph.setup_seconds)
                        final.append(row);C.write(out/'perarm.json',final)
                    records.append(row);C.write(out/'dense.json',records);C.write(folder/'training_curve.json',curve)
                    P.report(out,{'status':'RUNNING'},records,final);model.train()
                progress({'block':block,'arm':arm,'phase':'starting','completed_updates':0})
                save_stage(0);first=C.subset(train,plan['batch_indices'][0])
                graph=R.CapturedSuperK8(model,first,MODE,LOSS_FN)
                for update,ids in enumerate(plan['batch_indices'],1):
                    tick=time.monotonic();data=first if update==1 else C.subset(train,ids)
                    loss,state,finite=graph.run(data);norm=R.finish_update(model,opt,finite);torch.cuda.synchronize()
                    curve.append({'update':update,'mean_super_update_loss':float(loss),
                        'gradient_norm_before_clip':float(norm),'seconds':time.monotonic()-tick,
                        'cold_initializations':4,**CV.CADENCE})
                    if update==1 or update%25==0:
                        progress({'block':block,'arm':arm,'phase':'training','completed_updates':update,
                            'loss':float(loss),'gradient_norm_before_clip':float(norm)})
                    if update in CHECKPOINTS:
                        assert all(int(v['step'].item())==update for v in opt.state.values())
                        save_stage(update)
                    if update!=1:del data
                assert len(curve)==UPDATES
                progress({'block':block,'arm':arm,'phase':'arm_complete','completed_updates':UPDATES})
                del model,opt,graph,state,first,curve;gc.collect();torch.cuda.empty_cache()
        assert len(final)==32 and len(records)==416 and hashes()==source
        assert C.sha(out/'plans.json')==plan_hash and C.tensor_hash(train)==data_hash['train']
        assert all(C.tensor_hash(v)==data_hash[f'evaluation{s}'] for s,v in banks.items())
        result=P.aggregate(final,records,expected_blocks=BLOCKS)
        result.update(protocol=PROTOCOL,status='COMPLETE',completed_arms=32,expected_arms=32,dense_records=416)
        P.plot(out,records)
    except Exception as error:
        C.write(out/'error.json',{'error':repr(error),'traceback':traceback.format_exc()})
        result={'protocol':PROTOCOL,'status':'ERROR','completed_arms':len(final),'expected_arms':32,
            'dense_records':len(records),'error':repr(error),'primary_verdict':'INCOMPLETE'}
    result.update(finished_utc=C.now(),elapsed_seconds=time.monotonic()-started,runtime_limit_enforced=False)
    C.write(out/'summary.json',result);C.write(out/'status.json',{**result,'pid':os.getpid()})
    P.report(out,result,records,final);print(json.dumps(result),flush=True)
    if result['status']=='ERROR':raise SystemExit(1)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--check',action='store_true')
    parser.add_argument('--qualification');args=parser.parse_args()
    out=(ROOT/args.out).resolve();assert out.is_relative_to(ROOT/('analyses' if args.check else 'runs'))
    if args.check:
        try:check(out)
        except Exception as error:
            if not out.exists():
                out.parent.mkdir(parents=True,exist_ok=True)
                C.write(out,{'status':'ERROR','protocol':PROTOCOL,'error':repr(error),
                             'traceback':traceback.format_exc(),'runtime_limit_enforced':False})
            raise
    else:
        path=(ROOT/args.qualification).resolve();assert path.is_relative_to(ROOT/'analyses')
        run(out,path)

if __name__=='__main__':main()
