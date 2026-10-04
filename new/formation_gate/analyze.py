"""Retrospective, no-inference formation diagnostic using saved paired traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/'runs/transition_20261003_seed4_dense01'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def rate(n,d):
    return float(n/d) if d else None


def aggregate(n,d):
    n = np.asarray(n,dtype=np.int64); d = np.asarray(d,dtype=np.int64)
    valid = d>0
    return {'numerator':int(n.sum()),'denominator':int(d.sum()),'pooled':rate(n.sum(),d.sum()),
            'equal_map':float((n[valid]/d[valid]).mean()) if valid.any() else None,
            'eligible_maps':int(valid.sum()),'per_map_numerator':n.tolist(),'per_map_denominator':d.tolist()}


def counted(a):
    return a.reshape(len(a),-1).sum(1)


def interval(c,select,start,end):
    old = c[start]&select
    wrong = ~c[start]&select
    gain = wrong&c[end]
    destroyed = old&~c[end]
    survival = old&c[start:end+1].all(0)
    assert int(counted(c[end]&select).sum()-counted(old).sum()) == int(counted(gain).sum()-counted(destroyed).sum())
    return {'retention':aggregate(counted(old&c[end]),counted(old)),
            'survival':aggregate(counted(survival),counted(old)),
            'G':aggregate(counted(gain),counted(wrong)),
            'first_exit':aggregate(counted(old&~survival),counted(old)),
            'coverage':aggregate(counted(c[end]&select),counted(select))}


def predictors(c,m,select):
    early = interval(c,select,32,64)
    new = (~c[24])&c[32]&select
    old = c[24:33].all(0)&select
    continue_ = c[32:65].all(0)
    cohort = c[64]&select
    destructive = np.maximum(m[56:64]-m[57:65],0).max(0)
    reserve = m[64]-8*destructive
    safe = cohort&(reserve>0)
    early['new_survival'] = aggregate(counted(new&continue_),counted(new))
    early['old_survival'] = aggregate(counted(old&continue_),counted(old))
    early['margin_reserve'] = aggregate(counted(safe),counted(cohort))
    p = early['survival']['pooled']; g = early['G']['pooled']
    early['candidate_screen'] = bool(p is not None and g is not None and p>=.95 and g>=.20)
    return early


def long_screen(c,select,strict):
    mid = interval(c,select,64,128)
    far = interval(c,select,64,256)
    coverage = aggregate(counted(c[128]&strict),counted(strict))
    gates = {'coverage_pooled':coverage['pooled'] is not None and coverage['pooled']>=.80,
             'coverage_equal_map':coverage['equal_map'] is not None and coverage['equal_map']>=.80,
             'G':mid['G']['pooled'] is not None and mid['G']['pooled']>=.20,
             'first_exit':mid['first_exit']['pooled'] is not None and mid['first_exit']['pooled']<=.01,
             'survival':far['survival']['pooled'] is not None and far['survival']['pooled']>=.95}
    return {'screen':all(gates.values()),'gates':gates,'strict_T128':coverage,
            'G64_128':mid['G'],'first_exit64_128':mid['first_exit'],'survival64_256':far['survival']}


def confusion(pairs):
    result = {'TP':0,'FP':0,'TN':0,'FN':0}
    for pred,truth in pairs:
        result['TP' if pred and truth else 'FP' if pred else 'FN' if truth else 'TN']+=1
    return result


def ranking(rows,field):
    """Pairwise positive-versus-negative ranking, not a validated population AUC."""
    pos=[r[field] for r in rows if r['long_pass'] and r[field] is not None]
    neg=[r[field] for r in rows if not r['long_pass'] and r[field] is not None]
    if not pos or not neg:return None
    return float(np.mean([float(p>n)+.5*float(p==n) for p in pos for n in neg]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    out=Path(ap.parse_args().out);assert not out.exists();out.mkdir(parents=True)
    tick=time.monotonic()
    historical=json.loads((INPUT/'summary.json').read_text())
    assert historical['execution_complete']
    source={name:sha(ROOT/name) for name in ('new/formation_gate/analyze.py','new/formation_gate/PROTOCOL.md')}
    bindings={'source_sha256':source,'input_summary_sha256':sha(INPUT/'summary.json'),'traces':{}}
    records=[];flat=[]
    for ref in historical['checkpoints']:
        u,size=ref['update'],ref['size']
        path=INPUT/f'checkpoints/u{u:03d}_size{size}.npz'
        digest=sha(path);assert digest==ref['trace_sha256']
        bindings['traces'][path.relative_to(ROOT).as_posix()]=digest
        a=np.load(path);c=a['correct'];m=a['margin'];select=a['changed'][:,0].astype(bool);d=a['distance'][:,0]
        assert c.shape[0]==257 and np.isfinite(m).all()
        strict=select&(d>16)&(d<32)
        early=predictors(c[:65],m[:65],select)
        early_strict=predictors(c[:65],m[:65],strict)
        outcome=long_screen(c,select,strict)
        # Validate all saved long rates independently, including their counts.
        old=ref['behavior_profile']['64_128'];old_far=ref['behavior_profile']['64_256']
        for key,new in (('acquisition',outcome['G64_128']),('first_exit',outcome['first_exit64_128'])):
            assert abs(old[key]['pooled_rate']-new['pooled'])<1e-12
        assert abs(old_far['continuous_survival']['pooled_rate']-outcome['survival64_256']['pooled'])<1e-12
        row={'update':u,'size':size,'early':early,'early_strict':early_strict,'long':outcome}
        records.append(row)
        flat.append({'update':u,'size':size,'early_survival':early['survival']['pooled'],
                     'early_G':early['G']['pooled'],'early_new_survival':early['new_survival']['pooled'],
                     'early_old_survival':early['old_survival']['pooled'],
                     'margin_reserve':early['margin_reserve']['pooled'],
                     'early_pass':early['candidate_screen'],'long_pass':outcome['screen'],
                     'strict_T128':outcome['strict_T128']['pooled'],
                     'long_survival':outcome['survival64_256']['pooled'],
                     'long_first_exit':outcome['first_exit64_128']['pooled']})
        print(json.dumps({'update':u,'size':size,'early_pass':early['candidate_screen'],'long_pass':outcome['screen']}),flush=True)
    primary=[r for r in flat if r['size']==32 and r['update']<=200]
    lookup={r['update']:r for r in primary}
    same=confusion([(r['early_pass'],r['long_pass']) for r in primary])
    lagged=confusion([(r['early_pass'],lookup[r['update']+5]['long_pass']) for r in primary if r['update']+5 in lookup])
    rankings={f:ranking(primary,f) for f in ('early_survival','early_G','early_new_survival','margin_reserve')}
    summary={'status':'COMPLETE','protocol':'formation_gate_v1','no_training':True,'no_inference':True,
             'selected_training_trajectories':1,'records':records,'profiles':flat,
             'primary_dense':{'same_checkpoint_confusion':same,'next_update5_confusion':lagged,
                 'pairwise_rank_of_single_success':rankings,
                 'candidate_pass_updates':[r['update'] for r in primary if r['early_pass']],
                 'long_pass_updates':[r['update'] for r in primary if r['long_pass']]},
             'elapsed_seconds':time.monotonic()-tick}
    save(out/'summary.json',summary);save(out/'manifest.json',bindings)
    with (out/'profiles.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    text=['# Continuation formation: saved-trajectory diagnostic','',
          'One selected historical training trajectory; zero training and zero inference.',
          'u = training update; t = rollout step. Predictors read only t<=64.',
          'The long behavioral screen uses T128/T256. All21 dense checkpoints and300 control are retained.','',
          '| Update | Survival32–64 | G32–64 | Newly solved survival32–64 | Margin reserve at64 | Early screen | Strict T128 | Survival64–256 | Long screen |',
          '|---:|---:|---:|---:|---:|---|---:|---:|---|']
    for r in [r for r in flat if r['size']==32]:
        fmt=lambda x:f'{x:.4f}' if x is not None else 'undefined'
        text.append(f"| {r['update']} | {fmt(r['early_survival'])} | {fmt(r['early_G'])} | {fmt(r['early_new_survival'])} | {fmt(r['margin_reserve'])} | {r['early_pass']} | {fmt(r['strict_T128'])} | {fmt(r['long_survival'])} | {r['long_pass']} |")
    text+=['',f"Dense same-checkpoint confusion: {same}.",f"Predicting next saved update (+5) confusion: {lagged}.",
           f"Early candidate passes: {summary['primary_dense']['candidate_pass_updates']}.",
           f"Long screen passes: {summary['primary_dense']['long_pass_updates']}.",
           f"Single-positive retrospective pairwise ordering: {rankings}.",'',
           'These are retrospective diagnostics with one successful dense checkpoint. Thresholds were not fitted,',
           'but the historical outcomes were already known when these candidate definitions were proposed.',
           'No out-of-sample predictor, physical phase transition, abstract quotient, or causal commitment mechanism is established.',
           'Early retention/progress are short-horizon behavior proxies. Margin reserve is a heuristic, not a uniform certificate.',
           'A property of the rule at update175 cannot guarantee it survives the optimizer change to update180.',
           'The failure to discover an early predictor lowers the evidential weight of these particular mechanism candidates;',
           'it does not invalidate the conditional continuation theorem or all possible low-complexity representations.','',
           'All counts, strict-band variants and per-map denominators: [summary.json](summary.json).',
           'All sizes/checkpoints: [profiles.csv](profiles.csv).',
           'Frozen definitions: [protocol](../../new/formation_gate/PROTOCOL.md).']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(8,8),sharex=True)
    for f in ('early_survival','early_new_survival','margin_reserve'):
        axes[0].plot([r['update'] for r in primary],[r[f] for r in primary],marker='.',label=f)
    axes[0].legend(fontsize=8);axes[0].set_ylabel('Early retention / reserve')
    axes[1].plot([r['update'] for r in primary],[r['early_G'] for r in primary],marker='.',label='G32-64')
    axes[1].axhline(.20,color='gray',linestyle=':');axes[1].legend();axes[1].set_ylabel('Early acquisition')
    for f in ('strict_T128','long_survival'):
        axes[2].plot([r['update'] for r in primary],[r[f] for r in primary],marker='.',label=f)
    axes[2].legend();axes[2].set_ylabel('Long behavior');axes[2].set_xlabel('Training update u')
    for ax in axes:
        ax.set_ylim(0,1.03);ax.grid(alpha=.2)
        for u in (175,180,195,200):ax.axvline(u,color='gray',alpha=.2)
    fig.suptitle('One historical seed4 trajectory: early proxies and later behavior')
    fig.tight_layout();fig.savefig(out/'formation_profiles.png',dpi=160);plt.close(fig)
    assert all(sha(ROOT/n)==v for n,v in source.items())
    save(out/'status.json',{'status':'COMPLETE','records':len(records),'training':False,'inference':False,
                          'elapsed_seconds':time.monotonic()-tick})
    print(json.dumps({'status':'COMPLETE','primary_dense':summary['primary_dense']}),flush=True)


if __name__=='__main__':main()
