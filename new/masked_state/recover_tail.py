"""Recover only missing planned diagnostics; preserve the original run verbatim."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import time

import torch
import run_state as screen
from tasks import bank, metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/masked_state_20261001_seed0')
    p.add_argument('--out',required=True)
    args=p.parse_args()
    original=Path(args.run);out=Path(args.out)
    filename=f'{screen.ARM}_seed0.json'
    raw=screen.read(original/filename);manifest=screen.read(original/'manifest.json')
    if raw['status']!='TRAINED' or raw['completed_updates']!=800:
        raise RuntimeError('Recovery requires completed training; never resume training here')
    if set(raw['evaluation'])!={'32','64','128'} or any(e['status']!='EVALUATED' for e in raw['evaluation'].values()):
        raise RuntimeError('All scientific evaluation sizes must already be saved')
    for section in ('source_sha256','reference_sha256'):
        for rel,h in manifest[section].items():
            if screen.sha(screen.ROOT/rel)!=h:raise RuntimeError(f'Changed frozen file: {rel}')
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    screen.runner.DEADLINE=started+180
    config=raw['config'];torch.set_num_threads(config['threads'])
    settings=manifest['backend']
    torch.backends.cudnn.benchmark=settings['cudnn_benchmark'];torch.backends.cudnn.deterministic=settings['cudnn_deterministic']
    torch.backends.cudnn.allow_tf32=settings['cudnn_allow_tf32'];torch.backends.cuda.matmul.allow_tf32=settings['matmul_allow_tf32']
    checkpoint_path=original/f'{screen.ARM}_seed0.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    assert checkpoint['completed_updates']==800 and checkpoint['arm']==screen.ARM
    model=screen.make_cell(screen.ARM,checkpoint['channels'],checkpoint['reference_hidden']).to(config['device']).eval()
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    replay=[];recovered=[]
    for size,steps in ((32,64),(128,256)):
        screen.runner.check_budget()
        data=bank(size,config['eval_examples'],30000+size,config['device'])
        with torch.no_grad():
            state=model.rollout(data['x'],steps)
            m=metrics(model.logits(state),data['y'],data['mask'])
            m['content_rms']=float(state[0].square().mean().sqrt())
        expected=raw['evaluation'][str(size)]['curve'][str(steps)]
        errors={k:abs(m[k]-expected[k]) for k in ('balanced_accuracy','bce','content_rms')}
        if max(errors.values())>5e-5:raise RuntimeError(f'Checkpoint replay mismatch: {errors}')
        replay.append({'size':size,'steps':steps,'absolute_errors':errors})
    shutil.copyfile(original/filename,out/'original_arm.json')
    for size in config['eval_sizes']:
        ev=raw['evaluation'][str(size)]
        timings=ev.setdefault('latency_by_horizon',{})
        missing=[t for t in config['eval_horizons'] if str(t) not in timings]
        if missing:
            data=bank(size,config['eval_examples'],30000+size,config['device'])
            for t in missing:
                screen.runner.check_budget()
                timings[str(t)]=screen.runner.benchmark(model,data['x'][:config['batch']],t,3)
                recovered.append(f'evaluation.{size}.latency_by_horizon.{t}')
        threshold=ev['first_sustained_95_steps']
        if 'seconds_per_query_at_sustained_95' not in ev:
            ev['seconds_per_query_at_sustained_95']=None if threshold is None else timings[str(threshold)]['median_seconds_per_query']
            recovered.append(f'evaluation.{size}.seconds_per_query_at_sustained_95')
    if 'gradient_probes' not in raw:
        data=bank(config['size'],min(config['batch'],config['eval_examples']),40000,config['device'])
        probes=[]
        for t in config['probe_horizons']:
            screen.runner.check_budget()
            probe=screen.runner.gradient_probe(model,data,t)
            bad=[k for k,v in probe.items() if isinstance(v,float) and not math.isfinite(v)]
            if bad:
                probe={k:v for k,v in probe.items() if k not in bad}
                probe.update(status='NONFINITE_REPORTED_PROBE_VALUE',nonfinite_fields=bad)
            probes.append(probe)
        raw['gradient_probes']=probes;recovered.append('gradient_probes')
    screen.runner.write_json(out/filename,raw)
    summarize_args=argparse.Namespace(**{**config,'out':str(out)})
    screen.summarize(out,summarize_args,raw['status'])
    original_status=screen.read(original/'status.json')
    provenance={'status':'PLANNED_AUXILIARIES_RECOVERED','training_repeated':False,
        'original_execution_finalized':False,'original_status':{k:v for k,v in original_status.items() if k!='pid'},
        'cause_of_missing_finalization':'Not established; original log contained no terminal exception.',
        'original_arm_sha256':screen.sha(original/filename),'checkpoint_sha256':screen.sha(checkpoint_path),
        'source_manifest_sha256':screen.sha(original/'manifest.json'),
        'recovery_script_sha256':screen.sha(Path(__file__)),
        'checkpoint_replay':replay,'recovered_fields':recovered,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,
        'timing_limit':'Only missing size128 timings were collected in a separate process. No cross-run speed comparison.'}
    screen.runner.write_json(out/'recovery.json',provenance)
    note=('\n## Execution provenance\n\nOriginal training and all15 learning endpoints were saved, '
          'but the process did not leave a final completion marker. Original status remains unchanged. '
          'Missing size128 timings and initial-state gradient probes were recovered from the existing '
          'checkpoint without training. Two representative endpoints replayed within5e-5. '
          'See recovery.json and original_arm.json; timings are from separate sessions.\n')
    with (out/'RESULTS.md').open('a',encoding='utf-8') as f:f.write(note)
    print(provenance,flush=True)


if __name__=='__main__':main()
