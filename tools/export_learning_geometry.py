"""Export/verify the two completed learning-geometry diagnostics; no model execution.

--verify-only uses the standard library and the public evidence only. Full export
also reads local saved feature arrays for an independent covariance/kernel check.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
INIT_RUN = ROOT/'runs/initial_geometry_20261004_update0_01'
FORM_RUN = ROOT/'runs/formation_gate_20261004_saved01'
INIT = ROOT/'evidence/initial_geometry_20261004'
FORM = ROOT/'evidence/formation_gate_20261004'
COMBINED = ROOT/'evidence/seed4_learning_geometry_20261004'
MANIFEST = ROOT/'LEARNING_GEOMETRY_PUBLICATION_MANIFEST.json'
SOURCE_NAMES = ('new/initial_geometry/audit.py','new/initial_geometry/summarize.py',
                'new/initial_geometry/PROTOCOL.md','new/formation_gate/analyze.py',
                'new/formation_gate/PROTOCOL.md','tools/export_learning_geometry.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def close(a,b):
    assert a is not None and b is not None and math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-10),(a,b)


def rate_check(r):
    n,d=r['per_map_numerator'],r['per_map_denominator']
    assert sum(n)==r['numerator'] and sum(d)==r['denominator']
    if sum(d):close(r['pooled'],sum(n)/sum(d))
    else:assert r['pooled'] is None
    vals=[a/b for a,b in zip(n,d) if b]
    if vals:close(r['equal_map'],sum(vals)/len(vals))
    else:assert r['equal_map'] is None
    assert len(vals)==r['eligible_maps']


def initial_profiles(summary):
    rows=[]
    for seed,r in summary['seeds'].items():
        for bank,b in r['banks'].items():
            for view,t,item in feature_items(b):
                sp=item['spectrum']
                rows.append({'seed':int(seed),'bank':bank,'size':b['size'],'view':view,'t':t,
                    'targets':item['target_count'],'available':item['available_count'],
                    'map_mean_relative':item['map_mean_relative'],
                    'available_map_mean_relative':item['available_map_mean_relative'],
                    'kernel_trace':sp['trace'],'kernel_rank':sp['rank_relative_1e8'],
                    'kernel_effective_rank':sp['effective_rank'],'kernel_condition':sp['nonzero_condition'],
                    'kernel_label_alignment':sp['label_alignment']})
    return rows


def feature_items(b):
    for t,item in b['features'].items():yield 'instant',int(t),item
    for w in b['windows']:
        for view in ('K8','full_history'):yield view,w['t'],w[view]


def write_csv(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def validate_initial_local(summary):
    import numpy as np
    checked=0;max_error=0.
    for seed,r in summary['seeds'].items():
        for name,b in r['banks'].items():
            path=INIT_RUN/f'{name}_seed{seed}_features.npz';a=np.load(path)
            k=len(a['map']);sign=a['sign']
            for view,t,item in feature_items(b):
                key=f'h{t}' if view=='instant' else f'H_{"K8" if view=="K8" else "full"}_{t}'
                h=a[key];diff=h[:k]-h[k:]
                eig=np.linalg.eigvalsh((diff.T@diff)*b['rnorm2']/k)[::-1].clip(0)
                expected=np.asarray(item['spectrum']['eigenvalues'])
                err=float(np.max(np.abs(eig-expected)));max_error=max(max_error,err)
                assert np.allclose(eig,expected,rtol=1e-7,atol=1e-10)
                orient=diff*sign[:,None]
                den=np.linalg.norm(diff.T@diff,'fro')*k
                alignment=float(np.square(orient.sum(0)).sum()/den) if den else None
                if den:close(alignment,item['spectrum']['label_alignment'])
                else:assert item['spectrum']['label_alignment'] is None
                norm=np.linalg.norm(diff,axis=1)
                scale=(np.linalg.norm(h[:k],axis=1)+np.linalg.norm(h[k:],axis=1))/2
                rel=norm/(scale+1e-12)
                close(float(np.mean([rel[a['map']==m].mean() for m in np.unique(a['map'])])),item['map_mean_relative'])
                if view!='instant':
                    available=a[f'available_{"K8" if view=="K8" else "full"}_{t}']
                    assert int(available.sum())==item['available_count']
                    assert not ((~available)&(norm>1e-8)).any()
                checked+=1
    return {'status':'PASS','scope':'saved feature arrays, no inference or training',
            'feature_views_checked':checked,'maximum_spectrum_error':max_error,
            'tangent_autograd_maximum_error':max(x['max_error'] for r in summary['seeds'].values() for x in r['sanity'])}


def validate_formation(s):
    checked=0
    for r in s['records']:
        for view in ('early','early_strict'):
            for name,metric in r[view].items():
                if isinstance(metric,dict) and 'per_map_numerator' in metric:rate_check(metric);checked+=1
            e=r[view]
            assert e['candidate_screen']==bool(e['survival']['pooled'] is not None and e['G']['pooled'] is not None
                and e['survival']['pooled']>=.95 and e['G']['pooled']>=.20)
        for name,metric in r['long'].items():
            if isinstance(metric,dict) and 'per_map_numerator' in metric:rate_check(metric);checked+=1
        assert r['long']['screen']==all(r['long']['gates'].values())
    rows=[r for r in s['profiles'] if r['size']==32 and r['update']<=200]
    assert len(rows)==21
    passes=[r['update'] for r in rows if r['early_pass']]
    assert passes==s['primary_dense']['candidate_pass_updates']
    def confusion(pairs):
        d=dict(TP=0,FP=0,TN=0,FN=0)
        for p,t in pairs:d['TP' if p and t else 'FP' if p else 'FN' if t else 'TN']+=1
        return d
    lookup={r['update']:r for r in rows}
    assert confusion([(r['early_pass'],r['long_pass']) for r in rows])==s['primary_dense']['same_checkpoint_confusion']
    assert confusion([(r['early_pass'],lookup[r['update']+5]['long_pass']) for r in rows if r['update']+5 in lookup])==s['primary_dense']['next_update5_confusion']
    return {'status':'PASS','scope':'published rate/count/gate arithmetic; not a new trajectory replay',
            'records_checked':len(s['records']),'rate_records_checked':checked,'confusion_tables_checked':2}


def build():
    si,sf=read(INIT_RUN/'summary.json'),read(FORM_RUN/'summary.json')
    assert si['status']==sf['status']=='COMPLETE'
    assert si['training_updates']==0 and sf['no_training'] and sf['no_inference']
    mi,mf=read(INIT_RUN/'manifest.json'),read(FORM_RUN/'manifest.json')
    for manifest in (mi,mf):
        for name,digest in manifest['source_sha256'].items():assert sha(ROOT/name)==digest
    for item in list(mi['banks'].values())+list(mi['checkpoints'].values()):
        assert sha(ROOT/item['file'])==item.get('sha256',item.get('file_sha256'))
    for name,digest in mf['traces'].items():assert sha(ROOT/name)==digest
    vi=validate_initial_local(si);vf=validate_formation(sf)
    for p in (INIT,FORM,COMBINED):p.mkdir(parents=True,exist_ok=True)
    compact=copy.deepcopy(si);permap=[]
    for seed,r in compact['seeds'].items():
        for bank,b in r['banks'].items():
            for view,t,item in feature_items(b):
                for row in item.pop('per_map_bands'):
                    permap.append({'seed':seed,'bank':bank,'view':view,'t':t,**row})
    write(INIT/'summary.json',compact);write_csv(INIT/'profiles.csv',initial_profiles(si))
    write_csv(INIT/'per_map_bands.csv',permap);write(INIT/'validation.json',vi)
    for filename in ('summary.json','profiles.csv','formation_profiles.png','INTERPRETATION.md'):
        shutil.copyfile(FORM_RUN/filename,FORM/filename)
    write(FORM/'validation.json',vf)
    ir=(INIT_RUN/'RESULTS.md').read_text().replace('Feature NPZs are secondary.',
        'Per-map distance-stratum rows are in [per_map_bands.csv](per_map_bands.csv); full feature NPZs remain local.')
    (INIT/'RESULTS.md').write_text(ir,encoding='utf-8')
    shutil.copyfile(FORM_RUN/'RESULTS.md',FORM/'RESULTS.md')
    (INIT/'REPRODUCTION.md').write_text('''# Reproduction and evidence scope

Public arithmetic/hash verification from the repository root (standard library):

    python -X utf8 -B tools/export_learning_geometry.py --verify-only

The public summaries preserve all four seeds, four banks, three instantaneous
feature views, eight K8 and eight full-history tangent views, and their spectra.
Per-map distance-stratum counts and relative separations are in per_map_bands.csv.
Full target-feature NPZs, checkpoints and original input archives remain local.
The export independently recomputed all304 relation Gram spectra and alignments
from saved feature arrays; it did not repeat model inference. Tangent/autograd
sanity was part of the original CPU diagnostic, not this publication.

Exact independent measurement uses the excluded historical update0 checkpoints
and paired map banks at the relative archive locations declared in the protocol:

    python -X utf8 -u -B new/initial_geometry/audit.py --out runs/NEW_INITIAL_GEOMETRY
    python -X utf8 -B new/initial_geometry/summarize.py --run runs/NEW_INITIAL_GEOMETRY

These commands require those archives; a fresh clone can verify the published
evidence without them. Do not mistake public verification for exact trajectory
reproduction. Dependencies are in ../../requirements.txt: Torch2.5.1, NumPy and
Matplotlib. The audit is CPU-only and takes no optimizer step. The existing
initialization hashes in ../streaming_carry_init2345/raw/ provide public identity
anchors. Review the [protocol](../../new/initial_geometry/PROTOCOL.md) first.
''',encoding='utf-8')
    (FORM/'REPRODUCTION.md').write_text('''# Reproduction and evidence scope

Public arithmetic/hash verification from the repository root (standard library):

    python -X utf8 -B tools/export_learning_geometry.py --verify-only

The complete44-record public summary retains per-map numerators/denominators,
all21 dense checkpoints plus300, both spatial sizes and strict-band variants.
The profiles CSV and figure are reading aids; source truth is summary.json.
Original Boolean/logit NPZs remain local. Public verification recomputes rate,
screen and confusion-table arithmetic; it does not reexecute a model or prove
closure. The original analysis checked raw archive hashes and independently
matched the prior long-horizon rates. There is one selected training trajectory.

Given the original saved dense archive, execute:

    python -X utf8 -u -B new/formation_gate/analyze.py --out runs/NEW_FORMATION_GATE

That command reads runs/transition_20261003_seed4_dense01 and requires its
excluded raw trace archive. It performs no inference or training. Full dense
trajectory reproduction is separately documented in
[the preceding reproduction notes](../transition_20261003/REPRODUCTION.md).
This publication did not launch that training. NumPy and Matplotlib dependencies
are listed in the root requirements.txt. Read the
[frozen diagnostic definitions](../../new/formation_gate/PROTOCOL.md).
''',encoding='utf-8')
    (COMBINED/'RESULTS.md').write_text('''# Seed4 learning geometry: mixed initialization evidence, no training-time precursor

Two completed2D diagnostics, with zero new training. Initialization geometry
uses CPU30.31s and four exact historical initializations; formation analysis
uses CPU8.86s and44 existing trace records, without new inference.

1. [Initialization results](../initial_geometry_20261004/RESULTS.md): seed4 has
   the most balanced raw lane RMS and the largest measured available relative
   source-flip separation at endpoint64. Centered balance and spectral rankings
   differ; seed3 has higher K8 paired kernel-label alignment on all four banks.
   All seeds share the same initial state-Jacobian isometry. The first task
   gradient primarily opens Q-out weight; encoder/F/Q-in/readout-weight gradients
   are zero, and Q-out gradient is rank one up to FP32 roundoff.
2. [Formation results](../formation_gate_20261004/RESULTS.md) and
   [interpretation](../formation_gate_20261004/INTERPRETATION.md): early rollout
   retention+progress passes at training updates140/145/190/200, while the long
   screen passes only200 in the dense window. Same-checkpoint comparison has
   TP1/FP3/TN17/FN0. Predicting update u+5 has TP0/FP3/TN16/FN1. The sampled
   early proxies do not identify a training-time precursor.
3. Raw feature matrices and full trajectories remain local; compact summaries,
   every negative seed/checkpoint, per-map denominators and exact code are public.
   Run `python -X utf8 -B tools/export_learning_geometry.py --verify-only`.

Training update u and rollout step t are distinct. The initialization audit runs
untrained parameters through t64; its weights are still at update0. Formation
predictors observe t<=64, while later behavior uses t128/256. An association
within a fixed checkpoint does not forecast future optimizer updates.

These are post hoc diagnostics on already-inspected seeds/maps, not a new
architecture qualification or prospective predictor. Initialization has four
seed units, one historical success; formation has one selected training
trajectory and one successful checkpoint in the dense window. No statistical
anomaly, causal initialization rule or reliability improvement is established.

The continuation-contract idea remains conditional: raw-domain closure, small
abstract transition error epsilon, robust progress margin gamma>epsilon, and
task-output error eta below a correct semantic margin m imply continued valid
execution. These audits have not constructed the required representation pi,
abstract rule F or uniform domain certificate. Therefore they lower the weight
of closure as an identified seed4 mechanism, while leaving that conditional
execution implication intact. Retention metrics alone do not identify a quotient.
Earlier architecture and warm-start negative verdicts remain unchanged.

Source/definition routes:
[initialization protocol](../../new/initial_geometry/PROTOCOL.md),
[formation protocol](../../new/formation_gate/PROTOCOL.md),
[initialization reproduction](../initial_geometry_20261004/REPRODUCTION.md),
[formation reproduction](../formation_gate_20261004/REPRODUCTION.md).
''',encoding='utf-8')
    files=sorted(p for d in (INIT,FORM,COMBINED) for p in d.iterdir() if p.is_file())
    source_hashes={**mi['source_sha256'],**mf['source_sha256'],**{p:sha(ROOT/p) for p in SOURCE_NAMES}}
    manifest={'schema':'learning-geometry-publication-v1','source_sha256':source_hashes,
        'public_evidence_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
        'completed_diagnostics':2,'publication_execution':'saved-artifact CPU only, no model execution',
        'input_summary_sha256':{'initial_geometry':sha(INIT_RUN/'summary.json'),'formation':sha(FORM_RUN/'summary.json')},
        'excluded_input_bindings':{'initial':{'checkpoints':mi['checkpoints'],'banks':mi['banks']},'formation_traces':mf['traces']},
        'transformations':['Initial summary: move per_map_bands to CSV; scalar/spectral scientific values unchanged.',
                           'Initial report: replace unavailable NPZ reading route with public per-map CSV.',
                           'Formation summary/profiles/figure/interpretation/report are byte-identical copies.',
                           'Machine receipts, checkpoints and full feature/trajectory arrays are excluded.']}
    write(MANIFEST,manifest)
    return verify()


def verify():
    m=read(MANIFEST)
    for names in ('source_sha256','public_evidence_sha256'):
        for p,digest in m[names].items():assert sha(ROOT/p)==digest, p
    si,sf=read(INIT/'summary.json'),read(FORM/'summary.json')
    assert set(si['seeds'])=={'2','3','4','5'} and si['status']=='COMPLETE' and si['training_updates']==0
    views=0
    for seed,r in si['seeds'].items():
        for name,b in r['banks'].items():
            assert len(b['windows'])==8 and set(b['features'])=={'0','8','64'}
            for view,t,item in feature_items(b):
                sp=item['spectrum'];eig=sp['eigenvalues'];close(sum(eig),sp['trace'])
                assert 0<=sp['rank_relative_1e8']<=16 and item['unavailable_nonzero_count']==0
                assert all(math.isfinite(x) and x>=0 for x in eig);views+=1
    assert views==304
    vf=validate_formation(sf)
    public_csv=list(csv.DictReader((INIT/'profiles.csv').open(encoding='utf-8')))
    wanted=initial_profiles(si);assert len(public_csv)==len(wanted)==304
    for actual,expected in zip(public_csv,wanted):
        for k,v in expected.items():
            if v is None:assert actual[k]==''
            elif isinstance(v,(int,float)):close(float(actual[k]),v)
            else:assert actual[k]==str(v)
    groups=defaultdict(list)
    for row in csv.DictReader((INIT/'per_map_bands.csv').open(encoding='utf-8')):
        groups[(row['seed'],row['bank'],row['view'],int(row['t']))].append(row)
    assert len(groups)==304
    for seed,r in si['seeds'].items():
        for name,b in r['banks'].items():
            for view,t,item in feature_items(b):
                rows=groups[(seed,name,view,t)]
                assert sum(int(x['n']) for x in rows)==item['target_count']
                assert sum(int(x['available']) for x in rows)==item['available_count']
                maps=defaultdict(list)
                for x in rows:
                    assert 0<=int(x['nonzero'])<=int(x['available'])<=int(x['n'])
                    maps[x['map']].append(x)
                means=[sum(float(x['mean_relative_separation'])*int(x['n']) for x in g)/sum(int(x['n']) for x in g)
                       for g in maps.values()]
                close(sum(means)/len(means),item['map_mean_relative'])
                active=[sum(float(x['available_mean_relative'])*int(x['available']) for x in g if int(x['available']))
                        /sum(int(x['available']) for x in g)
                        for g in maps.values() if sum(int(x['available']) for x in g)]
                if active:close(sum(active)/len(active),item['available_map_mean_relative'])
                else:assert item['available_map_mean_relative'] is None
    paths=[ROOT/p for p in m['public_evidence_sha256']]+[ROOT/p for p in SOURCE_NAMES]+[MANIFEST]
    bad=[]
    private=re.compile(r'(?i)[a-z]:[\\/]|\b(?:192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|10\.\d+\.\d+\.)|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----')
    for path in paths:
        if path.suffix=='.png':continue
        text=path.read_text(encoding='utf-8')
        assert not private.search(text),f'Private identifier in {path.relative_to(ROOT)}'
        if path.suffix=='.md':
            for link in re.findall(r'\]\(([^)]+)\)',text):
                if re.match(r'https?://|#|mailto:',link):continue
                dest=(path.parent/link.split('#')[0]).resolve()
                if not dest.is_relative_to(ROOT) or not dest.exists():bad.append((path.relative_to(ROOT).as_posix(),link))
    assert not bad,bad
    return {'status':'PASS','public_evidence_files':len(m['public_evidence_sha256']),
        'initial_feature_views':views,'initial_per_map_groups':len(groups),'formation_validation':vf,'source_hashes':len(m['source_sha256']),
        'privacy_scan':'PASS','markdown_links':'PASS','original_local_archives_required':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args();print(json.dumps(verify() if args.verify_only else build(),indent=2))
