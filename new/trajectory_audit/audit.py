"""Inference-only finite-horizon audit; no checkpoint or training modification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent/'masked_medium'))
from masked_cells import make_cell, masked_laplacian
from tasks import bank, metrics, components

CHECKPOINT = ROOT/'runs/masked_medium_20261001_seed0/masked_inertial_rd_seed0.pt'
REFERENCE = ROOT/'evidence/masked_medium_seed0/arms/masked_inertial_rd_seed0.json'
HORIZONS = (16, 32, 64, 128, 256)
DEADLINE = float('inf')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_time():
    if time.monotonic() > DEADLINE:
        raise TimeoutError('Audit 10-minute cap reached')


def array(x):
    return x.detach().cpu().numpy().astype(np.float64)


def rms(a):
    return float(np.sqrt(np.mean(a*a))) if a.size else None


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(np.dot(a, b)/norm, -1, 1)) if norm > 1e-20 else None


def open_vectors(logits, data):
    z, y, mask = array(logits), array(data['y']), array(data['mask'])
    weights = np.zeros_like(y)
    for i in range(len(y)):
        pos, neg = mask[i]*y[i], mask[i]*(1-y[i])
        count = int(pos.sum()>0)+int(neg.sum()>0)
        weights[i] = (pos/max(1,pos.sum())+neg/max(1,neg.sum()))/count/len(y)
    selected = mask > .5
    return z[selected], y[selected], weights[selected]


def fit_temperature(z, y, w):
    margin = (2*y-1)*z
    def derivative(alpha):
        inverse_sigmoid = np.exp(-np.logaddexp(0, alpha*margin))
        return float(np.sum(-w*margin*inverse_sigmoid))
    lo, hi = .001, 100.
    if derivative(lo) >= 0: alpha, boundary = lo, True
    elif derivative(hi) <= 0: alpha, boundary = hi, True
    else:
        for _ in range(50):
            mid = (lo+hi)/2
            if derivative(mid) > 0: hi = mid
            else: lo = mid
        alpha, boundary = (lo+hi)/2, False
    return {'temperature':1/alpha, 'bound_selected':boundary,
            'calibration_bce_raw':float(np.sum(w*np.logaddexp(0,-margin))),
            'calibration_bce_scaled':float(np.sum(w*np.logaddexp(0,-alpha*margin)))}


def component_stats(fields, masks, beta, eta, diffusion):
    records = []
    energies = {'h':0., 'v':0., 'projected_h':0., 'projected_v':0., 'residual_h':0., 'residual_v':0.}
    cancel = residual = state_residual = 0.
    for b in range(len(masks)):
        ids, groups = components(masks[b,0]>.5)
        for c, group in enumerate(groups):
            take = ids == c
            vectors = {k: value[b][:,take] for k,value in fields.items()}
            means = {k: value.mean(axis=1) for k,value in vectors.items()}
            count = len(group)
            for k in ('h','v'):
                energies[k] += float(np.sum(vectors[k]**2))
                energies['projected_'+k] += float(count*np.sum(means[k]**2))
                energies['residual_'+k] += float(np.sum((vectors[k]-means[k][:,None])**2))
            cancel = max(cancel, float(np.max(np.abs(means['lap']))))
            residual = max(residual, float(np.max(np.abs(means['next_v']-beta*means['v']-eta*means['reaction']))))
            state_residual = max(state_residual, float(np.max(np.abs(means['next_h']-means['h']-means['next_v']))))
            records.append({'map':b,'component':c,'pixels':count,
                            **{k+'_mean':v.tolist() for k,v in means.items()}})
    identities = {k:abs(energies[k]-energies['projected_'+k]-energies['residual_'+k])/max(energies[k],1e-30) for k in ('h','v')}
    assert max(identities.values()) < 1e-10, identities
    lap_scale = max(1.,float(np.max(np.abs(fields['lap']))))
    assert cancel/lap_scale < 1e-5, (cancel,lap_scale)
    summary = {'h_component_constant_energy_fraction':energies['projected_h']/max(energies['h'],1e-30),
               'v_component_constant_energy_fraction':energies['projected_v']/max(energies['v'],1e-30),
               'h_mean_projection_rms':float(np.sqrt(energies['projected_h']/max(1,masks.sum())/len(beta))),
               'v_mean_projection_rms':float(np.sqrt(energies['projected_v']/max(1,masks.sum())/len(beta))),
               'max_abs_component_mean_laplacian':cancel, 'relative_laplacian_cancellation':cancel/lap_scale,
               'max_abs_mean_velocity_equation_residual':residual,
               'max_abs_mean_state_equation_residual':state_residual,
               'projection_energy_identity_relative_errors':identities}
    return summary, records


@torch.inference_mode()
def rollout(model, data, full=False):
    state = model.initial(data['x']); result = {}; snapshots = {}; details = {}
    beta, diffusion = model.base.coefficients()
    beta, diffusion = array(beta).ravel(), array(diffusion).ravel()
    mask = array(data['mask']); take = np.broadcast_to(mask>.5,(len(mask),16,*mask.shape[-2:]))
    for t in range(1,257):
        check_time()
        state = model.step(state,data['x'])
        if t not in HORIZONS: continue
        if not all(bool(torch.isfinite(s).all()) for s in state): raise ValueError('NONFINITE_AUDIT_STATE')
        z = model.logits(state)
        zv,yv,w = open_vectors(z,data)
        if not full:
            result[str(t)] = fit_temperature(zv,yv,w)
            continue
        m = metrics(z,data['y'],data['mask'])
        h,v = array(state[0]),array(state[1])
        # Exactly retain the float32 evaluator norm for the reconstruction gate.
        m.update(content_rms=float(state[0].square().mean().sqrt()),velocity_rms=float(state[1].square().mean().sqrt()))
        margin = (2*yv-1)*zv; correct = (zv>=0)==(yv>=.5)
        bce = np.logaddexp(0,-margin)
        assert abs(float(np.sum(w*bce))-m['bce']) < 5e-5, 'BCE weighting mismatch'
        q = lambda a: float(np.median(a)) if a.size else None
        k = max(1,int(np.ceil(.1*len(margin))))
        m.update(logit_rms_open=rms(zv), margin_median=q(margin),
                 margin_correct_median=q(margin[correct]),margin_wrong_median=q(margin[~correct]),
                 margin_q10=float(np.quantile(margin,.1)),margin_worst_decile_mean=float(np.mean(np.partition(margin,k-1)[:k])),
                 wrong_fraction=float(np.mean(~correct)),strict_negative_margin_fraction=float(np.mean(margin<0)),zero_margin_fraction=float(np.mean(margin==0)),
                 bce_correct_contribution=float(np.sum(w[correct]*bce[correct])),bce_wrong_contribution=float(np.sum(w[~correct]*bce[~correct])),
                 h_rms_open=rms(h[take]),h_rms_wall=rms(h[~take]),v_rms_open=rms(v[take]),v_rms_wall=rms(v[~take]))
        reaction = model.base.program(torch.cat((state[0],data['x']),dim=1))
        lap = masked_laplacian(state[0],data['x'][:,:1])
        nh,nv = model.step(state,data['x'])
        c,details[str(t)] = component_stats({'h':h,'v':v,'reaction':array(reaction),'lap':array(lap),'next_h':array(nh),'next_v':array(nv)},mask,beta,model.base.eta,diffusion)
        m['components'] = c
        result[str(t)] = m
        snapshots[t] = {'h':h,'v':v,'z':zv,'y':yv,'w':w}
    if not full:
        return result, {}, {}, {}
    direction = {}
    for t in (16,32,64,128):
        a,b = snapshots[t],snapshots[2*t]; per_map=[]
        for i in range(len(mask)):
            h1,h2 = a['h'][i][take[i]],b['h'][i][take[i]]
            cs = cosine(h1,h2); n1,n2=np.linalg.norm(h1),np.linalg.norm(h2)
            per_map.append({'map':i,'h_cosine':cs,'normalized_direction_distance':None if cs is None else float(np.linalg.norm(h2/n2-h1/n1)),
                            'relative_h_change':None if n1<1e-20 else float(np.linalg.norm(h2-h1)/n1),
                            'v_cosine':cosine(a['v'][i][take[i]],b['v'][i][take[i]])})
        r1,r2=details[str(t)],details[str(2*t)]
        v1=np.array([r['v_mean'] for r in r1]);v2=np.array([r['v_mean'] for r in r2])
        weights=np.sqrt(np.array([r['pixels'] for r in r1]))[:,None]
        v1,v2=v1*weights,v2*weights
        denom=np.linalg.norm(v1)
        direction[f'{t}_{2*t}']={'per_map':per_map,'mean_h_cosine':float(np.mean([p['h_cosine'] for p in per_map if p['h_cosine'] is not None])),
                                 'component_mean_v_cosine':cosine(v1,v2),
                                 'component_mean_v_relative_change':None if denom<1e-20 else float(np.linalg.norm(v2-v1)/denom)}
    return result, snapshots, details, direction


def main():
    global DEADLINE
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();DEADLINE=started+600
    source=json.loads((ROOT/'MASKED_PUBLICATION_MANIFEST.json').read_text())
    for name,h in source['source_sha256'].items(): assert sha(ROOT/name)==h,name
    torch.set_num_threads(2);settings=source['backend']
    torch.backends.cudnn.benchmark=settings['cudnn_benchmark'];torch.backends.cudnn.deterministic=settings['cudnn_deterministic']
    torch.backends.cudnn.allow_tf32=settings['cudnn_allow_tf32'];torch.backends.cuda.matmul.allow_tf32=settings['matmul_allow_tf32']
    model=make_cell('masked_inertial_rd').cuda().eval()
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    reference=json.loads(REFERENCE.read_text())
    manifest={'checkpoint_sha256':sha(CHECKPOINT),'reference_sha256':sha(REFERENCE),
              'source_sha256':{**source['source_sha256'],**{f.relative_to(ROOT).as_posix():sha(f) for f in [HERE/'audit.py',HERE/'PROTOCOL.md']}},
              'backend':settings,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
              'test_examples':16,'calibration_examples':32,'sizes':[32,64,128],'horizons':list(HORIZONS),
              'test_seeds':{str(s):30000+s for s in [32,64,128]},'calibration_seeds':{str(s):51000+s for s in [32,64,128]},
              'started_utc':datetime.now(timezone.utc).isoformat(),'training':False}
    write(out/'manifest.json',manifest)
    write(out/'local_receipt.json',{'pid':os.getpid(),'host':socket.gethostname(),'checkpoint':str(CHECKPOINT),'output':str(out.resolve())})
    reports={};component_records={};reconstruction={};max_error=0.
    for size in (32,64,128):
        test=bank(size,16,30000+size,'cuda')
        rows,snaps,detail,directions=rollout(model,test,True)
        errors={}
        for t,row in rows.items():
            errors[t]={k:abs(row[k]-reference['evaluation'][str(size)]['curve'][t][k]) for k in ['balanced_accuracy','bce','content_rms','velocity_rms']}
        err=max(e for r in errors.values() for e in r.values());max_error=max(max_error,err)
        reconstruction[str(size)]=errors
        write(out/'reconstruction.json',{'max_absolute_error':max_error,'tolerance':5e-5,'by_size':reconstruction,'passed_so_far':max_error<=5e-5})
        if max_error>5e-5: raise ValueError('RECONSTRUCTION_GATE_FAILED; do not interpret new diagnostics')
        calibration=bank(size,32,51000+size,'cuda')
        temperatures=rollout(model,calibration,False)[0]
        for t,row in rows.items():
            cal=temperatures[t];s=snaps[int(t)];scaled=s['z']/cal['temperature']
            assert np.array_equal(scaled>=0,s['z']>=0),'Sign invariance failed'
            row['temperature']={**cal,'test_bce_scaled':float(np.sum(s['w']*np.logaddexp(0,-(2*s['y']-1)*scaled))),
                                'test_ba_unchanged':True,'test_ba_scaled':row['balanced_accuracy']}
        reports[str(size)]={'curve':rows,'direction':directions}
        component_records[str(size)]=detail
        write(out/'summary.json',{'status':'AUDIT_IN_PROGRESS','sizes':reports})
        print(json.dumps({'completed_size':size,'reconstruction_error':err}),flush=True)
        del snaps,test,calibration
    write(out/'components.json',component_records)
    write(out/'summary.json',{'status':'COMPLETE','training':False,'sizes':reports,'max_reconstruction_error':max_error,
                             'elapsed_seconds':time.monotonic()-started,
                             'limits':'Single checkpoint, finite horizons. Neither directional alignment nor neutral-mode drift proves asymptotic convergence or architecture benefit.'})
    lines=['# Masked inertial trajectory audit','','Inference only; exact frozen evaluation maps, independent32-map calibration per size. No model changes.',
           f'Reconstruction max absolute error: {max_error:.8g} (tolerance5e-5).','',
           '| Size | T | BA % | Raw BCE | Calibrated BCE | Temperature | H open RMS | H wall RMS | H constant-mode energy | V mean RMS | Wrong margin median |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for size,r in reports.items():
        for t,row in r['curve'].items():
            c=row['components'];cal=row['temperature']
            lines.append(f"| {size} | {t} | {100*row['balanced_accuracy']:.2f} | {row['bce']:.4f} | {cal['test_bce_scaled']:.4f} | {cal['temperature']:.3f} | {row['h_rms_open']:.3f} | {row['h_rms_wall']:.3f} | {c['h_component_constant_energy_fraction']:.4f} | {c['v_mean_projection_rms']:.4f} | {row['margin_wrong_median']:.4f} |")
    lines+=['','Temperature is fit on independent calibration maps; it cannot change BA or repair source revision.',
            'Component cancellation is an identity; nonzero reaction/velocity and projection energies are observations.',
            'The mean dynamics depends on the full field, so this is not a closed autonomous mean model.',
            'Finite-horizon direction similarity is not proof of convergence. Detailed vectors are in components.json.',
            'No architecture changes, extra training seeds, new benchmark or speed comparison are performed.','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','out':str(out),'seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':
    main()
