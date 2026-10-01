"""Final one-arm completion of the generic recurrence by medium 2x2 screen."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from state_cells import ARM, make_cell, frozen_cells
import run_wind_tunnel as runner

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REFERENCES={
    'unmasked_state':ROOT/'evidence/inertial_seed0/arms/nca_state_matched_seed0.json',
    'unmasked_momentum':ROOT/'evidence/inertial_seed0/arms/momentum_nca_seed0.json',
    'masked_momentum':ROOT/'evidence/masked_momentum_seed0/arms/masked_momentum_nca_seed0.json',
}


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    old=read(ROOT/'INERTIAL_PUBLICATION_MANIFEST.json')
    recent=read(ROOT/'MOMENTUM_AUDIT_PUBLICATION_MANIFEST.json')
    hashes={**old['source_sha256'],**old['protocol_files_sha256'],**old['published_evidence_sha256'],
            **recent['source_sha256'],**recent['reference_sha256'],**recent['evidence_sha256']}
    for name,expected in hashes.items():
        if sha(ROOT/name)!=expected:raise RuntimeError(f'Frozen source/evidence changed: {name}')
    torch.manual_seed(0)
    original=frozen_cells.make_cell('nca_state_matched',16,128)
    next_rng=torch.rand(8)
    torch.manual_seed(0)
    model=make_cell()
    assert torch.equal(next_rng,torch.rand(8)), 'RNG mismatch'
    for k,v in original.state_dict().items():
        assert torch.equal(v,model.base.state_dict()[k]),k
    return old,hashes


def summarize(out,args,status):
    runner.write_summary(out,{ARM:status},args.preflight,args.seed)
    if args.preflight:return
    file=out/f'{ARM}_seed0.json'
    if not file.exists():
        runner.write_json(out/'comparison.json',{'status':status,'decision':'INVALID_OR_INCOMPLETE'})
        return
    data={k:read(p) for k,p in REFERENCES.items()}
    data['masked_state']=read(file)
    schedule=lambda d:[(r['iteration'],r['rollout'],r['damage']) for r in d['training_curve']]
    same_schedule=all(schedule(d)==schedule(data['masked_state']) for d in data.values())
    statuses=[]
    def inspect(v):
        if isinstance(v,dict):
            if 'status' in v:statuses.append(v['status'])
            for x in v.values():inspect(x)
        elif isinstance(v,list):
            for x in v:inspect(x)
    for d in data.values():inspect(d['evaluation'])
    valid=(status=='TRAINED' and same_schedule and all(d['completed_updates']==800 for d in data.values())
           and all(set(d['evaluation'])=={'32','64','128'} for d in data.values())
           and all(s=='EVALUATED' for s in statuses))
    rows=[]
    for size,ev in data['masked_state']['evaluation'].items():
        for t in ev.get('curve',{}):
            row={'size':int(size),'steps':int(t)}
            for key,d in data.items():
                e=d['evaluation'][size];c=e['curve'][t]
                row[key]={'ba':c['balanced_accuracy'],'bce':c['bce'],'h_rms':c['content_rms'],
                          'v_rms':c['velocity_rms'],'paired':e.get('paired_source_information',{}).get(t,{}).get('both_counterfactuals_correct_on_changed_component')}
            for metric in ('ba','paired'):
                s,us,m,um=[row[k][metric] for k in ('masked_state','unmasked_state','masked_momentum','unmasked_momentum')]
                row[f'momentum_minus_state_masked_{metric}_pp']=None if s is None else 100*(m-s)
                row[f'state_masking_effect_{metric}_pp']=None if s is None else 100*(s-us)
                row[f'momentum_masking_effect_{metric}_pp']=100*(m-um)
                row[f'interaction_{metric}_pp']=None if s is None else 100*((m-um)-(s-us))
            rows.append(row)
    primary=next((r for r in rows if r['size']==32 and r['steps']==64),None)
    decision='INVALID_OR_INCOMPLETE'
    if valid and len(rows)==15 and primary and primary['masked_state']['paired'] is not None:
        delta=[primary[f'momentum_minus_state_masked_{m}_pp'] for m in ('ba','paired')]
        decision=('MOMENTUM_JOINT_5PP_ADVANTAGE' if min(delta)>=5 else
                  'STATE_JOINT_5PP_ADVANTAGE' if max(delta)<=-5 else 'MIXED_OR_BELOW_JOINT_5PP')
    report={'protocol':'masked_state_final_seed0','status':status,'seed':0,'decision':decision,
            'logged_schedule_matches_all_four':same_schedule,'primary':primary,'all_horizons':rows,
            'limits':'Single seed; state scalars equal, parameter counts approximate; differing H/program widths prevent isolated velocity attribution. Small gaps are not equivalence. No further experiment scheduled.'}
    runner.write_json(out/'comparison.json',report)
    lines=['','## Final2x2 primary comparison:32x32/T64','',f'Descriptive outcome: `{decision}`.','',
           '| Generic model | Whole-grid BA % | Masked BA % | Whole-grid paired % | Masked paired % |',
           '|---|---:|---:|---:|---:|']
    def percent(v):return 'null' if v is None else f'{100*v:.2f}'
    if primary:
        for name,a,b in [('State-matched','unmasked_state','masked_state'),('Momentum','unmasked_momentum','masked_momentum')]:
            lines.append(f"| {name} | {percent(primary[a]['ba'])} | {percent(primary[b]['ba'])} | {percent(primary[a]['paired'])} | {percent(primary[b]['paired'])} |")
        lines+=['','| Contrast | BA difference pp | Paired difference pp |','|---|---:|---:|']
        for label,key in [('Masked Momentum minus masked state','momentum_minus_state_masked'),('State masking effect','state_masking_effect'),('Momentum masking effect','momentum_masking_effect'),('Difference-in-differences','interaction')]:
            vals=[primary[f'{key}_{m}_pp'] for m in ('ba','paired')]
            formatted=['null' if v is None else f'{v:+.2f}' for v in vals]
            lines.append(f'| {label} | {formatted[0]} | {formatted[1]} |')
    lines+=['',report['limits'],'Late horizons do not replace the primary endpoint. Inspect comparison.json for all sizes/horizons.',
            'Conditional repair cohorts can differ; timings from separate runs are not a controlled speed comparison.','']
    with (out/'RESULTS.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--minutes',type=float,default=25.);p.add_argument('--preflight',action='store_true')
    cli=p.parse_args()
    if cli.minutes<=0:p.error('minutes must be positive')
    old,hashes=verify()
    config=read(ROOT/'evidence/inertial_seed0/config.json');config.pop('conventions')
    config.update(arms=[ARM],out=cli.out,device=cli.device,minutes=cli.minutes,preflight=cli.preflight)
    if cli.preflight:
        config.update(steps=3,train_horizons=[64],eval_examples=2,eval_sizes=[32],eval_horizons=[16,64],recovery_steps=[8],probe_horizons=[16],log_every=1)
    args=argparse.Namespace(**config);device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    torch.set_num_threads(args.threads);settings=old['runtime']['backend']
    torch.backends.cudnn.benchmark=settings['cudnn_benchmark'];torch.backends.cudnn.deterministic=settings['cudnn_deterministic']
    torch.backends.cudnn.allow_tf32=settings['cudnn_allow_tf32'];torch.backends.cuda.matmul.allow_tf32=settings['matmul_allow_tf32']
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    paths=[ROOT/name for name in hashes if name.endswith('.py') or (name.endswith('.md') and not name.startswith('evidence/'))]
    paths+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    for path in paths:
        dest=out/'source'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    runner.write_json(out/'manifest.json',{'protocol':'masked_state_final_seed0','pid':os.getpid(),
        'started_utc':datetime.now(timezone.utc).isoformat(),'config':config,'backend':settings,
        'torch':torch.__version__,'numpy':np.__version__,'device':str(device),
        'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'initial_weights_and_rng_match_unmasked_state':True,
        'source_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in paths},
        'reference_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in REFERENCES.values()},
        'verified_frozen_file_count':len(hashes),'stop_after_this_arm':True})
    runner.write_json(out/'status.json',{'status':'STARTED','pid':os.getpid(),'arm':ARM})
    runner.make_cell=make_cell;runner.DEADLINE=time.monotonic()+args.minutes*60
    try:status=runner.train_one(args,ARM,device)
    except Exception as error:
        status='RUN_ERROR';runner.write_json(out/'error.json',{'error':repr(error),'traceback':traceback.format_exc()})
    summarize(out,args,status)
    runner.write_json(out/'status.json',{'status':'FINISHED','pid':os.getpid(),'statuses':{ARM:status},
        'finished_utc':datetime.now(timezone.utc).isoformat(),'stop_after_this_arm':True})
    print(json.dumps({'completed_statuses':{ARM:status},'stop_after_this_arm':True}),flush=True)


if __name__=='__main__':main()
