"""Paired current/factorized/RRC training with pre-stream spatial differences."""
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
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


CV = load('_rrc_coverage', ROOT/'new/continuous_coverage/run.py')
C, R, E, M = CV.C, CV.R, CV.E, CV.M
H = load('_rrc_cells', HERE/'cells.py')
P = load('_rrc_reporting', HERE/'reporting.py')
D = load('_rrc_activity', HERE/'activity.py')
PROTOCOL = 'rrc_v0_prestream_relation_factorization_v1'
ARMS = ('current', 'factorized', 'rrc')
COUNTS = {'current':5033, 'factorized':4983, 'rrc':4988}
BLOCKS, UPDATES = 8, 300
CHECKPOINTS = tuple(range(0,301,25))
INIT_SEEDS = tuple(range(120001,120009))
SCHEDULE_SEEDS = tuple(range(121001,121009))
EVAL_SEEDS = {32:122032,64:122064}
LOSS_FN, MODE = CV.LOSS_FN, 'reset64x4'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def hashes():
    value = CV.source_hashes()
    names = [p for p in HERE.iterdir() if p.suffix in ('.py','.md')]
    names += [ROOT/'tools/launch_rrc_v0.ps1',ROOT/'new/hybrid_writer/reporting.py']
    value.update({p.relative_to(ROOT).as_posix():C.sha(p) for p in names})
    return dict(sorted(value.items()))


def model_for(arm, seed, device='cuda'):
    model = H.make_model(arm,seed,device)
    assert sum(p.numel() for p in model.parameters()) == COUNTS[arm]
    return model


def component_hashes(model):
    state = model.state_dict()
    return {prefix:C.tensor_hash({k:v for k,v in state.items() if k.startswith(prefix)})
            for prefix in ('encoder.','q_in.','q_out.','readout.','f_in.','f_out.')}


def observed_evaluation(model,banks,folder,full=False):
    saved = model.relation_observer
    observer = D.EvaluationActivity(model)
    model.relation_observer = observer
    try:
        measured = E.evaluate(model,banks,folder,full=full)
        C.write(folder/'relation_activity.json',{'parameters':D.parameters(model),'rows':observer.result()})
    finally:
        model.relation_observer = saved
    return measured


def check(out):
    assert not out.exists(), 'Qualification already exists'
    C.setup_backend()
    before, started = hashes(), time.monotonic()
    artifact = out.with_suffix('')
    artifact.mkdir(parents=True,exist_ok=False)
    for filename in ('check_cells.py','reporting.py'):
        args = [sys.executable,'-X','utf8','-B',str(HERE/filename)]
        if filename == 'reporting.py':
            args.append('--self-test')
        result = subprocess.run(args,
            cwd=ROOT,capture_output=True,text=True)
        (artifact/(filename+'.stdout.txt')).write_text(result.stdout,encoding='utf-8')
        assert result.returncode == 0, result.stderr
    bank = C.region_bank(32,16,123032)
    data = {k:v.cuda() for k,v in C.subset(bank,list(range(8))).items()}
    banks = {s:C.region_bank(s,32,seed) for s,seed in EVAL_SEEDS.items()}
    checked = []
    for arm in ARMS:
        base = model_for(arm,120099,'cpu')
        torch.manual_seed(123099)
        # Qualification only: exercise content and relation gradients. Formal
        # initialization retains historical zero writer/Q output layers.
        with torch.no_grad():
            base.f_out.weight.normal_(std=.02)
            base.q_out.weight.normal_(std=.002)
            base.q_out.bias.zero_()
        eager,captured = copy.deepcopy(base).cuda(),copy.deepcopy(base).cuda()
        opt_e,opt_g = C.optimizer_for(eager),C.optimizer_for(captured)
        activity_e,activity_g = D.TrainingActivity(eager),D.TrainingActivity(captured)
        eager.relation_observer,captured.relation_observer = activity_e,activity_g
        before_capture = C.tensor_hash(captured.state_dict())
        torch.cuda.reset_peak_memory_stats()
        graph = R.CapturedSuperK8(captured,data,MODE,LOSS_FN)
        assert C.tensor_hash(captured.state_dict()) == before_capture
        max_grad,timings = 0.,[]
        for update in range(3):
            eager.zero_grad(set_to_none=True)
            activity_e.reset()
            loss_e,state_e,finite_e = R.math_backward(eager,data,MODE,LOSS_FN)
            activity_g.reset()
            tick = time.monotonic()
            loss_g,state_g,finite_g = graph.run(data)
            torch.cuda.synchronize()
            timings.append(time.monotonic()-tick)
            torch.testing.assert_close(loss_g,loss_e,atol=1e-6,rtol=1e-5)
            for (name,p),(other,q) in zip(eager.named_parameters(),captured.named_parameters()):
                assert name == other and p.grad is not None and q.grad is not None
                torch.testing.assert_close(q.grad,p.grad,atol=1e-6,rtol=1e-5)
                max_grad = max(max_grad,float((q.grad-p.grad).abs().max()))
            for a,b in zip(state_g,state_e):
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(activity_g.stats,activity_e.stats,atol=1e-6,rtol=1e-5)
            if arm == 'rrc':
                assert D.gradient_norm(captured) > 0, 'Relation gradient branch untested'
                assert float(activity_g.stats[:,2].sum()) > 0, 'Relation action untested'
            R.finish_update(eager,opt_e,finite_e)
            R.finish_update(captured,opt_g,finite_g)
            for a,b in zip(eager.parameters(),captured.parameters()):
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
                for key in ('exp_avg','exp_avg_sq','step'):
                    torch.testing.assert_close(opt_g.state[b][key],opt_e.state[a][key],atol=1e-6,rtol=1e-5)
        tick = time.monotonic()
        observed_evaluation(captured,banks,artifact/arm)
        evaluation_seconds = time.monotonic()-tick
        row = {'arm':arm,'status':'PASS','parameter_count':COUNTS[arm],
            'three_update_gradient_max_absolute_error':max_grad,
            'capture_setup_seconds':graph.setup_seconds,
            'median_captured_backward_seconds':float(np.median(timings)),
            'actual_32_map_two_size_evaluation_seconds':evaluation_seconds,
            'peak_cuda_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'relation_parameters':D.parameters(captured),'training_activity':activity_g.result()}
        checked.append(row)
        print(json.dumps(row),flush=True)
        del base,eager,captured,graph,opt_e,opt_g,state_e,state_g,activity_e,activity_g
        gc.collect()
        torch.cuda.empty_cache()
    assert hashes() == before
    projected = BLOCKS*sum(row['capture_setup_seconds']+UPDATES*row['median_captured_backward_seconds']
        +len(CHECKPOINTS)*row['actual_32_map_two_size_evaluation_seconds'] for row in checked)
    eval_bytes = sum(p.stat().st_size for p in artifact.rglob('*') if p.is_file())
    estimated_bytes = eval_bytes*BLOCKS*len(CHECKPOINTS)+312*150000
    C.write(out,{'status':'PASS','protocol':PROTOCOL,'source_sha256':before,
        'checked_utc':C.now(),'arms':checked,'actual_training_shape':[8,3,32,32],
        'credit_horizon':8,'eager_graph_updates_each_arm':3,'atol':1e-6,'rtol':1e-5,
        'estimated_run_seconds_excluding_data_io_and_optimizer_overhead':projected,
        'estimated_evidence_bytes':estimated_bytes,
        'qualification_seconds':time.monotonic()-started,'runtime_limit_enforced':False})
    print(json.dumps({'status':'PASS','estimated_hours':projected/3600,
        'estimated_evidence_MiB':estimated_bytes/2**20}),flush=True)


def run(out,qualification):
    assert not out.exists(), 'Run already exists'
    q,source = read(qualification),hashes()
    assert q['status']=='PASS' and q['protocol']==PROTOCOL and q['source_sha256']==source
    C.setup_backend()
    out.mkdir(parents=True)
    started = time.monotonic()
    records,final,plans = [],[],{}
    for block,(seed,schedule) in enumerate(zip(INIT_SEEDS,SCHEDULE_SEEDS)):
        models = {a:model_for(a,seed,'cpu') for a in ARMS}
        initial = {a:C.tensor_hash(m.state_dict()) for a,m in models.items()}
        components = {a:component_hashes(m) for a,m in models.items()}
        for prefix in ('encoder.','q_in.','q_out.','readout.'):
            assert len({v[prefix] for v in components.values()})==1
        for prefix in ('f_in.','f_out.'):
            assert components['factorized'][prefix]==components['rrc'][prefix]
        offset = block%3
        plans[str(block)] = {'initialization_seed':seed,'schedule_seed':schedule,
            'arm_order':list(ARMS[offset:]+ARMS[:offset]),'initial_parameter_sha256':initial,
            'component_sha256':components,
            'batch_indices':np.random.default_rng(schedule).integers(0,512,(UPDATES,8)).tolist()}
    C.write(out/'plans.json',plans)
    plan_hash = C.sha(out/'plans.json')
    train_cpu = C.region_bank(32,512,10002)
    train = {k:v.cuda() for k,v in train_cpu.items()}
    banks = {s:C.region_bank(s,32,seed) for s,seed in EVAL_SEEDS.items()}
    data_hash = {'train':C.tensor_hash(train_cpu),**{f'evaluation{s}':C.tensor_hash(v) for s,v in banks.items()}}
    (out/'banks').mkdir()
    for label,value in {'train':train_cpu,**{f'evaluation{s}':v for s,v in banks.items()}}.items():
        np.savez_compressed(out/'banks'/f'{label}.npz',**{k:a.numpy() for k,a in value.items()})
    config = {'protocol':PROTOCOL,'arms':ARMS,'blocks':BLOCKS,'super_updates':UPDATES,
        'parameter_counts':COUNTS,'carrier_channels':24,'latent_channels':8,
        'initialization_seeds':INIT_SEEDS,'schedule_seeds':SCHEDULE_SEEDS,
        'initial_relation':{'rho':.1,'pi':[.25]*4},'content_spatial_feature':'prestream_L(C_t)',
        'primary_contrast':['rrc','factorized'],'mode':MODE,'batch':8,
        'forward_steps_per_update':256,'loss_windows':32,'credit_horizon':8,
        'optimizer_steps_per_update':1,'parameters_fixed_within_super_update':True,
        'train_seed':10002,'eval_seeds':EVAL_SEEDS,'checkpoints':CHECKPOINTS,
        'formal_checkpoint':300,'optimizer':C.OPTIMIZER,'clip':1.,'runtime_limit_enforced':False}
    C.write(out/'config.json',config)
    manifest = {'protocol':PROTOCOL,'pid':os.getpid(),'host':os.environ.get('COMPUTERNAME'),
        'started_utc':C.now(),'command':[sys.executable,*sys.argv],'gpu':torch.cuda.get_device_name(),
        'torch':str(torch.__version__),'numpy':str(np.__version__),'source_sha256':source,
        'data_sha256':data_hash,'schedule_plan_sha256':plan_hash,
        'qualification_sha256':C.sha(qualification),'expected_arms':24,'expected_dense_records':312,
        'runtime_limit_enforced':False,'maximum_seconds':None,'watchdog_enabled':False,
        'continuous_monitoring':False,
        'git_review_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    C.write(out/'manifest.json',manifest)
    for name in source:
        dest = out/'source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    def progress(value):
        C.write(out/'status.json',{'status':'RUNNING','pid':os.getpid(),'updated_utc':C.now(),
            'elapsed_seconds':time.monotonic()-started,'completed_arms':len(final),'expected_arms':24,
            'completed_dense_records':len(records),**value})
        print(json.dumps(value),flush=True)
    try:
        for block_text,plan in plans.items():
            block = int(block_text)
            for arm in plan['arm_order']:
                folder = out/f'block{block:02d}'/arm
                (folder/'checkpoints').mkdir(parents=True)
                model = model_for(arm,plan['initialization_seed'])
                assert C.tensor_hash(model.state_dict())==plan['initial_parameter_sha256'][arm]
                opt,curve,evaluation_seconds = C.optimizer_for(model),[],0.
                activity = D.TrainingActivity(model)
                model.relation_observer = activity
                def save_stage(update):
                    nonlocal evaluation_seconds
                    parameter_hash = C.tensor_hash(model.state_dict())
                    checkpoint = folder/'checkpoints'/f'u{update:03d}.pt'
                    payload = C.payload(model,opt,plan['initialization_seed'],update)
                    payload.update(protocol=PROTOCOL,arm=arm,schedule_seed=plan['schedule_seed'])
                    torch.save(payload,checkpoint)
                    progress({'block':block,'arm':arm,'phase':'evaluation','completed_updates':update})
                    tick = time.monotonic()
                    stage = folder/'evaluation'/f'u{update:03d}'
                    measured = observed_evaluation(model,banks,stage,full=update==UPDATES)
                    elapsed = time.monotonic()-tick
                    evaluation_seconds += elapsed
                    row = {'block':block,'arm':arm,'update':update,
                        'initialization_seed':plan['initialization_seed'],'schedule_seed':plan['schedule_seed'],
                        'checkpoint':checkpoint.relative_to(out).as_posix(),'checkpoint_sha256':C.sha(checkpoint),
                        'parameter_sha256':parameter_hash,'evaluation_seconds':elapsed,
                        'evaluation_summary':(stage/'summary.json').relative_to(out).as_posix(),
                        'relation_activity':(stage/'relation_activity.json').relative_to(out).as_posix(),
                        'metrics':M.compact_pair_metrics(measured),'joint':M.joint_readiness(measured),
                        'formal_endpoint':update==UPDATES,'checkpoint_selection':False}
                    if update==UPDATES:
                        row.update(last_loss=curve[-1]['mean_super_update_loss'],training_seconds=sum(v['seconds'] for v in curve),
                            all_checkpoint_evaluation_seconds=evaluation_seconds,capture_setup_seconds=graph.setup_seconds)
                        final.append(row)
                        C.write(out/'perarm.json',final)
                    records.append(row)
                    C.write(out/'dense.json',records)
                    C.write(folder/'training_curve.json',curve)
                    P.report(out,final,records,status={'status':'RUNNING'})
                    model.train()
                progress({'block':block,'arm':arm,'phase':'starting','completed_updates':0})
                save_stage(0)
                first = C.subset(train,plan['batch_indices'][0])
                before_capture = C.tensor_hash(model.state_dict())
                graph = R.CapturedSuperK8(model,first,MODE,LOSS_FN)
                assert C.tensor_hash(model.state_dict())==before_capture
                for update,ids in enumerate(plan['batch_indices'],1):
                    tick = time.monotonic()
                    data = first if update==1 else C.subset(train,ids)
                    activity.reset()
                    loss,state,finite = graph.run(data)
                    relation_grad = D.gradient_norm(model)
                    norm = R.finish_update(model,opt,finite)
                    torch.cuda.synchronize()
                    curve.append({'update':update,'mean_super_update_loss':float(loss),
                        'gradient_norm_before_clip':float(norm),'seconds':time.monotonic()-tick,
                        'cold_initializations':4,**CV.CADENCE,'relation_gradient_norm_before_clip':relation_grad,
                        'relation_parameters_after_update':D.parameters(model),'relation_activity':activity.result()})
                    if update==1 or update%25==0:
                        progress({'block':block,'arm':arm,'phase':'training','completed_updates':update,
                            'loss':float(loss),'gradient_norm_before_clip':float(norm)})
                    if update in CHECKPOINTS:
                        assert all(int(v['step'].item())==update for v in opt.state.values())
                        save_stage(update)
                    if update!=1:
                        del data
                assert len(curve)==UPDATES
                progress({'block':block,'arm':arm,'phase':'arm_complete','completed_updates':UPDATES})
                model.relation_observer = None
                del model,opt,graph,state,first,curve,activity
                gc.collect()
                torch.cuda.empty_cache()
        assert len(final)==24 and len(records)==312 and hashes()==source
        assert C.sha(out/'plans.json')==plan_hash and C.tensor_hash(train)==data_hash['train']
        assert all(C.tensor_hash(v)==data_hash[f'evaluation{s}'] for s,v in banks.items())
        result = P.aggregate(final,records,expected_blocks=BLOCKS)
        assert result['status']=='COMPLETE'
        result.update(protocol=PROTOCOL,completed_arms=24,expected_arms=24,dense_records=312)
    except Exception as error:
        C.write(out/'error.json',{'error':repr(error),'traceback':traceback.format_exc()})
        result = {'protocol':PROTOCOL,'status':'ERROR','completed_arms':len(final),'expected_arms':24,
            'dense_records':len(records),'error':repr(error),'overall_verdict':'INCOMPLETE'}
    result.update(finished_utc=C.now(),elapsed_seconds=time.monotonic()-started,runtime_limit_enforced=False)
    C.write(out/'summary.json',result)
    C.write(out/'status.json',{**result,'pid':os.getpid()})
    P.report(out,final,records,status=result)
    print(json.dumps(result),flush=True)
    if result['status']=='ERROR':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--qualification')
    args = parser.parse_args()
    out = (ROOT/args.out).resolve()
    assert out.is_relative_to(ROOT/('analyses' if args.check else 'runs'))
    if args.check:
        try:
            check(out)
        except Exception as error:
            if not out.exists():
                out.parent.mkdir(parents=True,exist_ok=True)
                C.write(out,{'status':'ERROR','protocol':PROTOCOL,'error':repr(error),
                    'traceback':traceback.format_exc(),'runtime_limit_enforced':False})
            raise
    else:
        assert args.qualification, 'Qualification required'
        qualification = (ROOT/args.qualification).resolve()
        assert qualification.is_relative_to(ROOT/'analyses')
        run(out,qualification)


if __name__=='__main__':
    main()
