"""Two-arm masked-medium intervention using the unchanged frozen training loop."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FROZEN=HERE.parent/'nca_inertial_wind_tunnel'
sys.path.insert(0,str(FROZEN))
import cells as frozen_cells
import run_wind_tunnel as runner
from masked_cells import ARMS,make_cell

BASE_ARMS={'masked_rd_nca':'rd_nca','masked_inertial_rd':'inertial_rd'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def state_hash(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def verify_pairing():
    manifest=read(ROOT/'INERTIAL_PUBLICATION_MANIFEST.json')
    for name,expected in {**manifest['source_sha256'],**manifest['protocol_files_sha256'],
                           **manifest['published_evidence_sha256']}.items():
        if sha(ROOT/name)!=expected:
            raise RuntimeError(f'Frozen source/evidence changed: {name}')
    rows={}
    for arm,base_arm in BASE_ARMS.items():
        torch.manual_seed(0)
        original=frozen_cells.make_cell(base_arm,16,128)
        old_next=torch.rand(4)
        torch.manual_seed(0)
        masked=make_cell(arm,16,128)
        new_next=torch.rand(4)
        if not torch.equal(old_next,new_next) or state_hash(original)!=state_hash(masked.base):
            raise RuntimeError(f'Initialization/RNG pairing failed: {arm}')
        rows[arm]={'base_arm':base_arm,'initial_state_dict_sha256':state_hash(original),
                   'same_initial_parameters_and_rng':True,
                   'control_sha256':manifest['published_evidence_sha256'][f'evidence/inertial_seed0/arms/{base_arm}_seed0.json']}
    return manifest,rows


def compare(out,args,statuses):
    runner.write_summary(out,statuses,args.preflight,args.seed)
    if args.preflight:
        return
    pairs=[]
    for arm,status in statuses.items():
        file=out/f'{arm}_seed0.json'
        if not file.exists():
            pairs.append({'arm':arm,'status':status,'decision':'INVALID_OR_INCOMPLETE'})
            continue
        current=read(file)
        base=read(ROOT/f'evidence/inertial_seed0/arms/{BASE_ARMS[arm]}_seed0.json')
        old_schedule=[(v['iteration'],v['rollout'],v['damage']) for v in base['training_curve']]
        new_schedule=[(v['iteration'],v['rollout'],v['damage']) for v in current['training_curve']]
        pair={'arm':arm,'base_arm':BASE_ARMS[arm],'status':status,
              'logged_training_schedule_matches':old_schedule==new_schedule,
              'decision':'INVALID_OR_INCOMPLETE','all_horizons':[]}
        nested_status=[]
        def inspect(value):
            if isinstance(value,dict):
                if 'status' in value:
                    nested_status.append(value['status'])
                for item in value.values(): inspect(item)
            elif isinstance(value,list):
                for item in value: inspect(item)
        inspect(current['evaluation'])
        valid=(status=='TRAINED' and current['completed_updates']==800 and old_schedule==new_schedule
               and set(current['evaluation'])=={'32','64','128'}
               and all(s=='EVALUATED' for s in nested_status))
        for size,ev in current['evaluation'].items():
            old=base['evaluation'][size]
            for t,point in ev.get('curve',{}).items():
                cp=ev.get('paired_source_information',{}).get(t,{}).get('both_counterfactuals_correct_on_changed_component')
                op=old['paired_source_information'][t]['both_counterfactuals_correct_on_changed_component']
                pair['all_horizons'].append({'size':int(size),'steps':int(t),
                    'masked_ba':point['balanced_accuracy'],'unmasked_ba':old['curve'][t]['balanced_accuracy'],
                    'ba_delta_pp':100*(point['balanced_accuracy']-old['curve'][t]['balanced_accuracy']),
                    'masked_paired_correct':cp,'unmasked_paired_correct':op,
                    'paired_delta_pp':None if cp is None else 100*(cp-op)})
        primary=next((v for v in pair['all_horizons'] if v['size']==32 and v['steps']==64),None)
        if valid and len(pair['all_horizons'])==15 and primary and primary['paired_delta_pp'] is not None:
            pair['primary']=primary
            pair['decision']='JOINT_GAIN_5PP' if primary['ba_delta_pp']>=5 and primary['paired_delta_pp']>=5 else 'NO_JOINT_5PP_GAIN'
        pairs.append(pair)
    runner.write_json(out/'paired_comparison.json',{'protocol':'masked_medium_seed0','seed':0,
        'comparison_scope':'Operator replacement within each structured model; no generic-momentum superiority claim.',
        'pairs':pairs})
    lines=['','## Prespecified comparison: 32x32 / T64','',
           '| Masked arm | Status | BA delta, pp | Paired-correct delta, pp | Descriptive criterion |',
           '|---|---|---:|---:|---|']
    for pair in pairs:
        primary=pair.get('primary',{})
        def value(key):
            number=primary.get(key)
            return 'null' if number is None else f'{number:+.2f}'
        lines.append(f"| {pair['arm']} | {pair['status']} | {value('ba_delta_pp')} | {value('paired_delta_pp')} | {pair['decision']} |")
    lines+=['','One seed only. The 5pp joint criterion is descriptive, not statistical significance.',
            'Repair with zero eligible examples is unevaluable. Masking changes connectivity, degree and spectrum.',
            'Old and new timing runs are not a controlled speedup comparison. No extra run is scheduled.','']
    with (out/'RESULTS.md').open('a',encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--minutes',type=float,default=25.)
    p.add_argument('--preflight',action='store_true')
    cli=p.parse_args()
    if cli.minutes<=0:
        p.error('minutes must be positive')
    old_manifest,pairing=verify_pairing()
    config=read(ROOT/'evidence/inertial_seed0/config.json')
    config.pop('conventions')
    config.update(arms=list(ARMS),out=cli.out,device=cli.device,minutes=cli.minutes,preflight=cli.preflight)
    if cli.preflight:
        config.update(steps=3,train_horizons=[64],eval_examples=2,eval_sizes=[32],
                      eval_horizons=[16,64],recovery_steps=[8],probe_horizons=[16],log_every=1)
    args=argparse.Namespace(**config)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    settings=old_manifest['runtime']['backend']
    torch.backends.cudnn.benchmark=settings['cudnn_benchmark']
    torch.backends.cudnn.deterministic=settings['cudnn_deterministic']
    torch.backends.cudnn.allow_tf32=settings['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32=settings['matmul_allow_tf32']
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    paths=[ROOT/name for name in old_manifest['source_sha256']]
    paths+=[ROOT/name for name in old_manifest['protocol_files_sha256']]
    paths+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    snapshot=out/'source'
    for path in paths:
        destination=snapshot/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    manifest={'protocol':'masked_medium_seed0','started_utc':datetime.now(timezone.utc).isoformat(),
              'pid':os.getpid(),'config':config,'torch':torch.__version__,'numpy':np.__version__,
              'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
              'backend':settings,'pairing':pairing,
              'source_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path in paths},
              'control_publication_manifest_sha256':sha(ROOT/'INERTIAL_PUBLICATION_MANIFEST.json')}
    runner.write_json(out/'manifest.json',manifest)
    runner.write_json(out/'status.json',{'status':'STARTED','pid':os.getpid(),'arms':list(ARMS)})
    runner.make_cell=make_cell
    runner.DEADLINE=time.monotonic()+args.minutes*60
    statuses={}
    for arm in ARMS:
        if time.monotonic()>=runner.DEADLINE:
            statuses[arm]='NOT_STARTED_TIME_BUDGET'
        else:
            try:
                statuses[arm]=runner.train_one(args,arm,device)
            except Exception as error:
                statuses[arm]='RUN_ERROR'
                runner.write_json(out/f'{arm}_error.json',{'error':repr(error),'traceback':traceback.format_exc()})
                gc.collect()
                if device.type=='cuda': torch.cuda.empty_cache()
        compare(out,args,statuses)
    runner.write_json(out/'status.json',{'status':'FINISHED','pid':os.getpid(),'statuses':statuses,
                                       'finished_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'completed_statuses':statuses}),flush=True)


if __name__=='__main__':
    main()
