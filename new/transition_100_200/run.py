"""Dense historical seed4 checkpoint replay and offline continuation audit."""
import argparse
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
HERE = Path(__file__).resolve().parent
for relative in ('new/nca_inertial_wind_tunnel', 'new/workspace_revision',
                 'new/short_bptt', 'new/streaming_carry', 'new/frontier_audit'):
    sys.path.insert(0, str(ROOT/relative))
from tasks import bank, subset, balanced_loss, per_example_balanced_accuracy
from run_revision import now, sha, tensor_hash, write
from stream_cells import StreamingCell
from training import backward_trajectory
import audit as frontier

PROTOCOL = 'seed4_dense_transition_v1'
REFERENCE = ROOT/'runs/warmstart_20261003_paired01'
CHECKPOINTS = (0, *range(100, 201, 5), 300)
AUDIT_UPDATES = (*range(100, 201, 5), 300)
TIMES = (32, 64, 128, 256)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sources():
    old = read(ROOT/'SEED4_FOLLOWUP_PUBLICATION_MANIFEST.json')['source_sha256']
    assert len(old) == 56
    for name, expected in old.items():
        assert sha(ROOT/name) == expected, f'Historical source drift: {name}'
    extra = [p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py')]
    extra += ['new/transition_100_200/PROTOCOL.md', 'tools/launch_transition_100_200.ps1',
              'SEED4_FOLLOWUP_PUBLICATION_MANIFEST.json', 'WARMSTART_PUBLICATION_MANIFEST.json',
              'new/warmstart/diagnostics.py', 'evidence/streaming_carry_init2345/schedule.json',
              'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json',
              'evidence/streaming_carry_init2345/manifest.json']
    warm = read(ROOT/'WARMSTART_PUBLICATION_MANIFEST.json')
    if 'source_sha256' in warm:
        assert sha(ROOT/'new/warmstart/diagnostics.py') == warm['source_sha256']['new/warmstart/diagnostics.py']
    return {**old, **{name:sha(ROOT/name) for name in sorted(set(extra))}}


def references():
    record = read(REFERENCE/'baseline_seed4.json')
    public = read(ROOT/'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json')
    assert record['status'] == 'COMPLETE' and record['completed_updates'] == 300
    result = {'record_sha256':sha(REFERENCE/'baseline_seed4.json'), 'anchors':{}}
    for key in ('initial_parameter_sha256','final_parameter_sha256','train_data_sha256','schedule_sha256'):
        assert record[key] == public[key], key
        result[key] = record[key]
    for update in (0,100,200,300):
        item = record['checkpoints'][str(update)]
        path = REFERENCE/item['file']
        assert sha(path) == item['sha256'], f'Reference checkpoint file drift u{update}'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['completed_updates'] == update
        assert tensor_hash(checkpoint['state_dict']) == item['parameter_sha256']
        result['anchors'][str(update)] = dict(item)
    return result


def setup():
    torch.set_num_threads(2)
    assert torch.__version__.split('+')[0] == '2.5.1', 'Historical PyTorch version required'
    assert torch.cuda.is_available()
    free, _ = torch.cuda.mem_get_info()
    assert free >= 3500*1024**2, 'Insufficient GPU headroom'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


def rms(value):
    return float(value.double().square().mean().sqrt())


@torch.no_grad()
def collect(model, cpu, size):
    data = {k:v.cuda() for k,v in cpu.items()}
    shape = (257, len(data['x']), size, size)
    traces = {key:np.empty(shape, dtype=bool) for key in ('correct','original_correct','flipped_correct')}
    traces['margin'] = np.empty(shape, dtype=np.float32)
    state_rms, endpoints = [], {}
    a, b = model.initial(data['x']), model.initial(data['x_flip'])
    changed = data['changed'].bool()
    strict = changed & (data['distance']>16) & (data['distance']<32)
    finite = torch.ones((), dtype=torch.bool, device='cuda')
    for t in range(257):
        if t:
            a, b = model.step(a,data['x']), model.step(b,data['x_flip'])
        logits, flipped = model.logits(a), model.logits(b)
        good_a = (logits>=0)==(data['y']>=.5)
        good_b = (flipped>=0)==(data['y_flip']>=.5)
        both = good_a & good_b
        margin = torch.minimum((2*data['y']-1)*logits,(2*data['y_flip']-1)*flipped)
        for value in (*a,*b,logits,flipped,margin):
            finite = finite & torch.isfinite(value).all()
        outside = changed & (data['distance']>2*t)
        assert not bool((both & outside).any()), f'Paired correctness outside cone at{t}'
        assert float(((logits-flipped).abs()*outside).max()) <= 1e-6
        for key,value in (('correct',both),('original_correct',good_a),('flipped_correct',good_b),('margin',margin)):
            traces[key][t] = value[:,0].cpu().numpy()
        state_rms.append({'t':t, 'original_W':rms(a[0]), 'original_Z':rms(a[1]),
                          'flipped_W':rms(b[0]), 'flipped_Z':rms(b[1])})
        if t%8==0:
            assert bool(finite), f'Nonfinite trajectory at/before{t}'
        if t in TIMES:
            num = (both & strict).flatten(1).sum(1).cpu().numpy()
            den = strict.flatten(1).sum(1).cpu().numpy()
            per_map = [float(n/d) if d else None for n,d in zip(num,den)]
            eligible = [v for v in per_map if v is not None]
            selected_margin = margin[changed].double()
            endpoints[str(t)] = {
                'strict_coverage':{'per_map':per_map,'per_map_numerator':num.tolist(),
                    'per_map_denominator':den.tolist(),'equal_map_mean':float(np.mean(eligible)) if eligible else None,
                    'pooled_rate':float(num.sum()/den.sum()) if den.sum() else None},
                'original_BA_equal_map':float(per_example_balanced_accuracy(logits,data['y'],data['mask']).mean()),
                'flipped_BA_equal_map':float(per_example_balanced_accuracy(flipped,data['y_flip'],data['mask']).mean()),
                'original_BCE_balanced':float(balanced_loss(logits,data['y'],data['mask'])),
                'flipped_BCE_balanced':float(balanced_loss(flipped,data['y_flip'],data['mask'])),
                'paired_margin_quantiles':torch.quantile(selected_margin,torch.tensor([0.,.1,.5,.9,1.],device='cuda',dtype=torch.float64)).cpu().tolist(),
            }
    assert bool(finite)
    traces.update({key:cpu[key].numpy() for key in ('changed','distance','mask')})
    traces['source'] = ((cpu['x'][:,1:2]!=0)|(cpu['x'][:,2:3]!=0)).numpy()
    return traces, {'endpoints':endpoints,'state_rms':state_rms,'steps':256,'finite':True,'light_cone':True}


def compact_rate(row):
    return row.get('pooled_rate') if isinstance(row,dict) else row


def report(out, rows, complete, preflight=False):
    profile=[]
    for row in rows:
        interval=row.get('behavior_profile',{}).get('64_128',{})
        long=row.get('behavior_profile',{}).get('64_256',{})
        coverage=row['endpoints']['128']['strict_coverage']
        checks={'strict128_pooled':coverage['pooled_rate'] is not None and coverage['pooled_rate']>=.80,
                'strict128_equal_map':coverage['equal_map_mean'] is not None and coverage['equal_map_mean']>=.80,
                'g64_128':compact_rate(interval.get('acquisition')) is not None and compact_rate(interval['acquisition'])>=.20,
                'first_exit64_128':compact_rate(interval.get('first_exit')) is not None and compact_rate(interval['first_exit'])<=.01,
                'survival64_256':compact_rate(long.get('continuous_survival')) is not None and compact_rate(long['continuous_survival'])>=.95}
        profile.append({'update':row['update'],'size':row['size'],'checks':checks,'descriptive_screen':all(checks.values()),
                        'prespecified_primary_screen':row['size']==32,
                        'size64_is_secondary_projection':row['size']==64})
    dense=sorted((r for r in profile if r['size']==32 and 100<=r['update']<=200),key=lambda r:r['update'])
    onset=None
    if not preflight:
        for a,b,c in zip(dense,dense[1:],dense[2:]):
            if b['update']==a['update']+5 and c['update']==b['update']+5 and all(r['descriptive_screen'] for r in (a,b,c)):
                onset={'first_saved_update':a['update'],'confirmed_through':c['update'],
                       'resolution_updates':5,'causal_or_phase_transition_claim':False}
                break
    write(out/'summary.json', {'execution_complete':complete,'protocol':PROTOCOL,'selected_training_runs':1,
          'diagnostic_only':True,'scientific_endpoint':not preflight,'checkpoints':rows,
          'descriptive_screen_profiles':profile,'operational_onset':onset,
          'decision':('SMOKE_ONLY' if preflight else 'DESCRIPTIVE_AUDIT_COMPLETE') if complete else 'PENDING',
          'causal_handoff_tested':False,'latent_commitment_identified':False})
    lines = ['# Seed4 dense update100--200 audit','',
             f"Execution: {'COMPLETE' if complete else 'RUNNING'}. One selected historical training trajectory.",'',
             '| Update | Size | Strict T64 | T128 | T256 | G64->128 | First-exit64->128 | Survival64->256 |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [row['endpoints'][str(t)]['strict_coverage']['pooled_rate'] for t in (64,128,256)]
        cells = [f'{v:.4f}' if v is not None else 'null' for v in values]
        interval=row.get('behavior_profile',{}).get('64_128',{})
        long=row.get('behavior_profile',{}).get('64_256',{})
        for value in (compact_rate(interval.get('acquisition')),compact_rate(interval.get('first_exit')),
                      compact_rate(long.get('continuous_survival'))):
            cells.append(f'{value:.4f}' if value is not None else 'null')
        lines.append(f"| {row['update']} | {row['size']} | "+' | '.join(cells)+' |')
    lines += ['', f"Three-checkpoint persistent descriptive onset: {onset if onset is not None else 'NONE / pending'}. This is not the historical full gate.",
              'Primary preservation/turnover and censoring-aware first-acquisition summaries are in each checkpoints/uNNN_sizeNN.json.',
              'Raw NPZ traces and checkpoints are secondary. All diagnostics use fixed paired maps and include all macro steps0..256.',
              'Checkpoint changes locate behavioral onset; they do not prove a physical phase transition, latent commitment or contextual type closure.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if complete and not preflight:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
        for size in (32,64):
            data=sorted((r for r in rows if r['size']==size and r['update']<=200),key=lambda r:r['update'])
            x=[r['update'] for r in data]
            curves=[
                [r['endpoints']['128']['strict_coverage']['pooled_rate'] for r in data],
                [compact_rate(r['behavior_profile']['64_128']['first_exit']) for r in data],
                [compact_rate(r['behavior_profile']['64_128']['net_gain']) for r in data],
                [compact_rate(r['behavior_profile']['64_256']['continuous_survival']) for r in data]]
            for ax,y in zip(axes.flat,curves):
                ax.plot(x,y,marker='o',markersize=3,label=f'size{size}')
        for ax,title in zip(axes.flat,('Strict paired coverage T128','First-exit risk T64--128',
                                     'Net coverage gain T64--128','Continuous survival T64--256')):
            ax.set_title(title);ax.set_xlabel('Training update');ax.grid(alpha=.25);ax.legend()
        fig.suptitle('Selected seed4 replay: descriptive checkpoint profiles')
        fig.tight_layout();fig.savefig(out/'transition_profiles.png',dpi=180);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--qualification')
    parser.add_argument('--preflight-dir')
    args = parser.parse_args()
    hashes, refs = sources(), references()
    if args.preflight:
        qualification = Path(args.qualification).resolve()
        assert qualification.is_relative_to(ROOT/'analyses')
        q = read(qualification)
        assert q['status']=='PASS' and q['source_sha256']==hashes and q['reference_bindings']==refs
        bindings = {'qualification':sha(qualification)}
    else:
        pf = Path(args.preflight_dir).resolve()
        assert pf.is_relative_to(ROOT/'runs')
        pm, pa = read(pf/'manifest.json'),read(pf/'aggregate.json')
        assert read(pf/'status.json')['status']=='PREFLIGHT_PASSED' and pa['pass']
        assert pm['protocol']==PROTOCOL and pm['source_sha256']==hashes and pm['reference_bindings']==refs
        bindings = {name:sha(pf/name) for name in ('manifest.json','aggregate.json','status.json')}
    out = Path(args.out).resolve()
    assert out.is_relative_to(ROOT/'runs')
    out.mkdir(parents=True,exist_ok=False)
    (out/'checkpoints').mkdir()
    started = time.monotonic()
    rows=[]
    try:
        setup()
        metrics = load_module('dense_transition_metrics',HERE/'metrics.py')
        old_diagnostic = load_module('dense_old_diagnostic',ROOT/'new/warmstart/diagnostics.py')
        train=bank(32,512,10002,'cuda')
        schedule=np.random.default_rng(20002).integers(0,512,(300,8)).tolist()
        assert schedule==read(ROOT/'evidence/streaming_carry_init2345/schedule.json')
        write(out/'schedule.json',schedule)
        assert tensor_hash(train)==refs['train_data_sha256']
        assert sha(out/'schedule.json')==refs['schedule_sha256']
        cpu_banks={size:bank(size,16,61000+size) for size in (32,64)}
        torch.manual_seed(4)
        model=StreamingCell().cuda()
        assert sum(p.numel() for p in model.parameters())==5033
        assert tensor_hash(model.state_dict())==refs['initial_parameter_sha256']
        updates=3 if args.preflight else 300
        manifest={'protocol':PROTOCOL,'preflight':args.preflight,'started_utc':now(),'pid':os.getpid(),
            'host':os.environ.get('COMPUTERNAME'),'command':[sys.executable,*sys.argv],
            'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'numpy':np.__version__,
            'git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'threads':2,'backend':{'cudnn_benchmark':False,'cudnn_deterministic':False,'cudnn_tf32':True,'matmul_tf32':False},
            'source_sha256':hashes,'reference_bindings':refs,'qualification_bindings':bindings,
            'updates':updates,'gradient_horizon':8,'forward_steps':64,'seed':4,'train_seed':10002,'schedule_seed':20002,
            'train_data_sha256':tensor_hash(train),'schedule_sha256':sha(out/'schedule.json'),
            'audit_banks':{str(n):{'maps':16,'seed':61000+n,'tensor_sha256':tensor_hash(d)} for n,d in cpu_banks.items()},
            'checkpoints':list(CHECKPOINTS),'audit_updates':list(AUDIT_UPDATES),
            'maximum_seconds':None,'runtime_limit_enforced':False,'watchdog_enabled':False,'continuous_monitoring':False}
        old_manifest=read(REFERENCE/'manifest.json')
        assert manifest['audit_banks']['32']['tensor_sha256']==old_manifest['stage_diagnostic_data_sha256']
        write(out/'manifest.json',manifest)
        for name in hashes:
            target=out/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        for size,data in cpu_banks.items():
            np.savez_compressed(out/f'bank_size{size}.npz',**{k:v.numpy() for k,v in data.items()})
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        saved={}
        curve=[]
        def save(update):
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            parameter_hash=tensor_hash(state)
            if str(update) in refs['anchors']:
                assert parameter_hash==refs['anchors'][str(update)]['parameter_sha256'],f'Historical parameter drift u{update}'
            path=out/'checkpoints'/f'u{update:03}.pt'
            torch.save({'state_dict':state,'completed_updates':update,'seed':4},path)
            saved[str(update)]={'path':path.relative_to(out).as_posix(),'parameter_sha256':parameter_hash,'sha256':sha(path)}
            write(out/'training.json',{'completed_updates':update,'checkpoints':saved,'curve':curve})
        save(0)
        torch.cuda.reset_peak_memory_stats()
        train_started=time.monotonic()
        for update,indices in enumerate(schedule[:updates],1):
            tick=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss,state,cadence=backward_trajectory(model,subset(train,indices),8)
            assert cadence['backward_calls']==8 and cadence['loss_count']==8
            assert all(bool(torch.isfinite(v).all()) for v in state)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert bool(torch.isfinite(norm))
            if args.preflight and update==3:
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                gradients={key:float(getattr(model,key).weight.grad.norm()) for key in ('encoder','f_in','f_out','q_in','q_out','readout')}
                assert all(v>0 for v in gradients.values())
            optimizer.step()
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
            torch.cuda.synchronize()
            curve.append({'update':update,'loss':float(loss),'gradient_norm':float(norm),'seconds':time.monotonic()-tick})
            if update in CHECKPOINTS or (args.preflight and update==3):
                save(update)
            if update==1 or update%5==0 or update==updates:
                write(out/'status.json',{'status':'RUNNING','phase':'training','pid':os.getpid(),'completed_updates':update,'updated_utc':now()})
                print(json.dumps({'phase':'training',**curve[-1]}),flush=True)
        train_seconds=time.monotonic()-train_started
        write(out/'training.json',{'completed_updates':updates,'training_seconds':train_seconds,'checkpoints':saved,'curve':curve})
        del optimizer,train
        model.eval()
        if not args.preflight:
            old=read(ROOT/'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json')
            measured={}
            frontier.phase2.DEADLINE=float('inf')
            counts={'integer_leaves':0,'float_leaves':0,'maximum_absolute_error':0.}
            for size in (32,64):
                data=bank(size,32,40000+size,'cuda')
                measured[str(size)]=frontier.phase2.evaluate(model,data,8,False)
                del data
            frontier.replay_module.compare_tree(old['evaluation'],measured,'dense.historical',counts)
            write(out/'historical_replay.json',{'status':'PASS','counts':counts,'evaluation':measured})
        selected=(3,) if args.preflight else AUDIT_UPDATES
        audit_seconds=[]
        for update in selected:
            checkpoint=torch.load(out/'checkpoints'/f'u{update:03}.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
            parameter_hash=tensor_hash(model.state_dict())
            assert parameter_hash==saved[str(update)]['parameter_sha256']
            for size,cpu in cpu_banks.items():
                write(out/'status.json',{'status':'RUNNING','phase':'audit','pid':os.getpid(),'completed_updates':updates,
                     'checkpoint_update':update,'size':size,'completed_audits':len(rows),'expected_audits':len(selected)*2,'updated_utc':now()})
                tick=time.monotonic()
                traces,details=collect(model,cpu,size)
                assert np.array_equal(traces['correct'],traces['original_correct']&traces['flipped_correct'])
                summary=metrics.summarize(traces['correct'],traces['changed'],traces['distance'],source=traces['source'])
                anchor_diagnostic=None
                if not args.preflight and size==32 and update in (100,200,300):
                    previous=read(REFERENCE/f'baseline_seed4_u{update:03}_diagnostic.json')
                    reproduced=old_diagnostic.summarize_correct_traces(traces['correct'][:129],traces['changed'])
                    comparison={'integer_leaves':0,'float_leaves':0,'maximum_absolute_error':0.}
                    for key in ('changed_pixels_per_map','steps','survival','coverage_gain','relapse'):
                        frontier.replay_module.compare_tree(previous[key],reproduced[key],f'stage{update}.{key}',comparison)
                    anchor_diagnostic={'status':'PASS','comparisons':comparison,'reference_sha256':sha(REFERENCE/f'baseline_seed4_u{update:03}_diagnostic.json')}
                assert tensor_hash(model.state_dict())==parameter_hash,'Audit mutated parameters'
                path=out/'checkpoints'/f'u{update:03}_size{size}.npz'
                np.savez_compressed(path,**traces)
                row={'update':update,'size':size,'parameter_sha256':parameter_hash,'trace_sha256':sha(path),
                     'audit_bank_sha256':manifest['audit_banks'][str(size)]['tensor_sha256'],**details,'behavior':summary,
                     'behavior_profile':summary['intervals'],'historical_stage_replay':anchor_diagnostic}
                write(out/'checkpoints'/f'u{update:03}_size{size}.json',row)
                rows.append({k:v for k,v in row.items() if k not in ('state_rms','behavior')})
                report(out,rows,False,args.preflight)
                audit_seconds.append(time.monotonic()-tick)
                print(json.dumps({'phase':'audit','checkpoint_update':update,'size':size,'seconds':audit_seconds[-1]}),flush=True)
                del traces,summary,row
        assert sources()==hashes and references()==refs,'Bindings changed during run'
        peak=torch.cuda.max_memory_allocated()
        result={'pass':True,'protocol':PROTOCOL,'execution_complete':True,'completed_updates':updates,
            'completed_audits':len(rows),'training_seconds':train_seconds,'audit_seconds':audit_seconds,
            'peak_allocated_bytes':peak,'runtime_limit_enforced':False,'preflight':args.preflight}
        if args.preflight:
            assert peak < 3500*1024**2
            result.update({'gradient_groups':gradients,'estimated_formal_seconds':max(v['seconds'] for v in curve[1:])*300+sum(audit_seconds)*len(AUDIT_UPDATES)+30})
        write(out/'aggregate.json',result)
        report(out,rows,True,args.preflight)
        status='PREFLIGHT_PASSED' if args.preflight else 'COMPLETE'
    except Exception as error:
        status='ERROR'
        write(out/'error.json',{'error':repr(error),'traceback':traceback.format_exc()})
        (out/'RESULTS.md').write_text('# Seed4 dense transition audit\n\nExecution: ERROR. No qualified transition conclusion.\n',encoding='utf-8')
    write(out/'status.json',{'status':status,'pid':os.getpid(),'finished_utc':now(),'elapsed_seconds':time.monotonic()-started,
         'completed_audits':len(rows),'runtime_limit_enforced':False})
    print(json.dumps(read(out/'status.json')),flush=True)
    if status not in ('COMPLETE','PREFLIGHT_PASSED'):
        raise SystemExit(1)


if __name__=='__main__':
    main()
