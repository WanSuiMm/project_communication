"""Lossless publication of the completed block7 audit; CPU saved-data checks."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/block7_collapse_audit_20261006_01'
PUBLIC = ROOT/'evidence/block7_collapse_20261006'
MANIFEST = ROOT/'BLOCK7_COLLAPSE_PUBLICATION_MANIFEST.json'
PROTOCOL = 'block7_collapse_continuation_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def read(path):
    if Path(path).suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8-sig') as f:return json.load(f)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def unpack(path):
    with np.load(path,allow_pickle=False) as f:
        return {k[:-7]:np.unpackbits(f[k],bitorder='little',count=int(np.prod(f[k[:-7]+'_shape']))).reshape(tuple(f[k[:-7]+'_shape'])).astype(bool) for k in f.files if k.endswith('_packed')}

def summaries(matrix,steps):
    rows=[]
    for r in matrix:
        for view,groups in r['views'].items():
            for group,q in groups.items():
                x={k:r[k] for k in ('size','producer','consumer')}
                x.update(view=view,cohort=group,reference_pixels=q['producer_baseline_correct_reference']['count'],reference_maps=sum(z['count']>0 for z in q['producer_baseline_correct_reference']['maps']))
                for t in ('T64','T128','T256'):
                    x['coverage_'+t]=q['endpoint_coverage'][t]['pooled']['value']
                x['mapmean_T256']=q['endpoint_coverage']['T256']['mapmean']['value']
                for k in ('continuous_preservation','terminal_retention','sustained_progress_last16','progress_terminal','progress_ever','handoff_destruction','handoff_acquisition','same_view_first_step_destruction'):
                    for stat,value in q[k]['pooled'].items():x[k+'_'+stat]=value
                rows.append(x)
    srows=[]
    for r in steps:
        for group,q in r['metrics'].items():
            x={k:r[k] for k in ('size','producer','consumer','state_time','readout')};x['cohort']=group
            for k in ('acquisition','destruction','net_coverage_change'):
                for stat,value in q[k]['pooled'].items():x[k+'_'+stat]=value
            srows.append(x)
    return rows,srows

def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def build():
    assert not PUBLIC.exists() and not MANIFEST.exists(), 'Refuse to overwrite evidence'
    original=read(RUN/'manifest.json');status=read(RUN/'status.json');summary=read(RUN/'summary.json')
    assert summary['protocol']==PROTOCOL and status['status']==summary['status']=='COMPLETE'
    assert summary['completed_matrix_units']==32 and summary['completed_single_step_units']==12
    for name,expected in original['source_sha256'].items():
        assert sha(ROOT/name)==sha(RUN/'source'/name)==expected, name
    for name,expected in original['input_sha256'].items():assert sha(ROOT/name)==expected,name
    for row in original['checkpoints'].values():assert sha(ROOT/row['path'])==row['sha256']
    PUBLIC.mkdir(parents=True)
    raw_bindings={}
    def copy(name,dest=None,compress=False):
        source=RUN/name;target=PUBLIC/(dest or name);target.parent.mkdir(parents=True,exist_ok=True)
        if compress:target.write_bytes(gzip.compress(source.read_bytes(),compresslevel=9,mtime=0))
        else:shutil.copyfile(source,target)
        assert target.stat().st_size<90*2**20
        raw_bindings[name]={'source_sha256':sha(source),'public_path':target.relative_to(ROOT).as_posix(),'encoding':'gzip lossless' if compress else 'byte-exact'}
    copy('summary.json');copy('RESULTS.md','frozen_RESULTS.md')
    copy('matrix.json','matrix.json.gz',True);copy('single_step.json','single_step.json.gz',True)
    for folder,count in (('matrix_traces',32),('single_step_traces',12),('states',10)):
        files=sorted((RUN/folder).glob('*.npz'));assert len(files)==count
        for path in files:copy(path.relative_to(RUN).as_posix())
    matrix=read(RUN/'matrix.json');steps=read(RUN/'single_step.json')
    rows,srows=summaries(matrix,steps);assert len(rows)==192 and len(srows)==24
    save_csv(PUBLIC/'matrix_summary.csv',rows);save_csv(PUBLIC/'single_step_summary.csv',srows)
    provenance={k:original[k] for k in ('protocol','source_sha256','input_sha256','backend','review_base','runtime_limit_enforced','watchdog_enabled','continuous_monitoring','training_updates')}
    provenance['checkpoints']={k:{'filename':Path(v['path']).name,'sha256':v['sha256'],'parameter_sha256':v['parameter_sha256']} for k,v in original['checkpoints'].items()}
    provenance.update(run_id=RUN.name,checkpoint_contents_published=False,publication_inference_or_training=False)
    write(PUBLIC/'provenance.json',provenance)
    note=(ROOT/'analyses/block7_collapse_audit_20261006_01/INTERPRETATION.md').read_text()
    note=note.replace('../../runs/block7_collapse_audit_20261006_01/RESULTS.md','frozen_RESULTS.md').replace('../../runs/block7_collapse_audit_20261006_01/matrix.json','matrix.json.gz').replace('../../runs/block7_collapse_audit_20261006_01/single_step.json','single_step.json.gz').replace('../block7_collapse_audit_20261006_01_validation.json','validation.json')
    note+='\nReading route: [summary](summary.json), [matrix CSV](matrix_summary.csv), [single-step CSV](single_step_summary.csv), [provenance](provenance.json), [reproduction](REPRODUCTION.md).\n'
    (PUBLIC/'RESULTS.md').write_text(note,encoding='utf-8')
    validation=read(ROOT/'analyses/block7_collapse_audit_20261006_01_validation.json')
    validation.update(publication_fit_or_inference=False,frozen_evidence_changed=False,checkpoint_files_verified=4,source_snapshot_bindings_verified=8,packed_matrix_trajectories=32,packed_single_step_records=12,numeric_state_packages=10)
    write(PUBLIC/'validation.json',validation)
    reproduction='''# Reproduction and saved-data layout

Read RESULTS.md, summary.json, validation.json, then matrix_summary.csv and
single_step_summary.csv. Undefined CSV ratios are blank; raw JSON uses null.
This publication performs CPU saved-data checks only, with no model inference.

From repository root, with Python and NumPy:

    python -X utf8 -B tools/export_block7_collapse.py --verify-only

GPU audit code: new/collapse_audit/run.py; protocol and CPU metric checks are
beside it. Re-running inference requires the four original local checkpoint
files from continuous_coverage_20261006_02. Their hashes are in provenance.json;
checkpoint contents and all machine/session records remain local. The prior
published evaluation banks and native traces are reused by reference, with
their exact paths and hashes in provenance.json. No external workspace is used.

matrix.json.gz and single_step.json.gz are lossless copies of complete raw
records, including all per-map numerators/denominators. The matrix CSV contains
192 rows:32 cells x3 readout views x2 cohorts; single-step CSV has24 rows.
This audit evaluates one selected training trajectory, not independent rows.

matrix_traces/sizeSIZE_pPRODUCER_cCONSUMER.npz stores Boolean trajectories for
producer, consumer and fixed275 readout views. Each has correct,
original_correct, flipped_correct fields. Fields are stored as NAME_shape
(int32) and NAME_packed(uint8); unpack with numpy.unpackbits, bitorder=little,
count=product(shape), then reshape. Shape is [193,32,SIZE,SIZE], times64..256.
The paired correct field is the AND of the original and flipped fields.

single_step_traces/sizeSIZE_tTIME_cCONSUMER.npz uses before/after prefixes and
the same three correctness fields, shape[32,SIZE,SIZE]. TIME is64,128 or192.
Both observations use the fixed u275 readout.

states/ contains eight T64 producer packages and two u275 packages including
T64/T128/T192. Keys are tTIME_original_C/Z and tTIME_flipped_C/Z. Arrays are
FP32[32,24,SIZE,SIZE] for C and FP32[32,8,SIZE,SIZE] for Z. All54 NPZs are
byte-exact compressed copies; no quantization or scientific-data deletion.

Preservation uses the producer-native T64-correct changed cohort at every
time64..256. Sustained progress uses its T64-wrong cohort at all times241..256.
Three readout views only observe states, never affect the recurrent rule.
Native prefix/diagonal replay is exact at all eight checkpoint/size pairs.
Prior formal u300 qualification stays negative; Hybrid has not been run.
'''
    (PUBLIC/'REPRODUCTION.md').write_text(reproduction,encoding='utf-8')
    files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(PUBLIC.rglob('*')) if p.is_file()}
    write(MANIFEST,{'protocol':PROTOCOL,'publication_protocol':'block7_collapse_saved_publication_v1','review_base':original['review_base'],'run_id':RUN.name,'raw_artifact_bindings':raw_bindings,'published_sha256':files,'published_file_count':len(files),'published_bytes':sum(p.stat().st_size for p in PUBLIC.rglob('*') if p.is_file()),'exporter_sha256':sha(Path(__file__)),'excluded':['checkpoint contents','original machine manifest','PID/status/launch receipts','local smoke outputs']})

def verify():
    manifest=read(MANIFEST);provenance=read(PUBLIC/'provenance.json');summary=read(PUBLIC/'summary.json')
    assert summary['status']=='COMPLETE' and summary['training_updates']==summary['optimizer_updates']==0
    assert sha(Path(__file__))==manifest['exporter_sha256']
    assert set(manifest['published_sha256'])=={p.relative_to(ROOT).as_posix() for p in PUBLIC.rglob('*') if p.is_file()}
    for name,expected in manifest['published_sha256'].items():assert sha(ROOT/name)==expected,name
    for name,expected in provenance['source_sha256'].items():assert sha(ROOT/name)==expected,name
    for name,expected in provenance['input_sha256'].items():assert sha(ROOT/name)==expected,name
    matrix=read(PUBLIC/'matrix.json.gz');steps=read(PUBLIC/'single_step.json.gz')
    assert len(matrix)==32 and len(steps)==12
    checked=0
    for r in matrix:
        size,P,C=r['size'],r['producer'],r['consumer']
        a=unpack(PUBLIC/f'matrix_traces/size{size}_p{P}_c{C}.npz')
        baseline=unpack(ROOT/f'evidence/continuous_coverage_20261006/block07/reset64x4/evaluation/u{P:03}/size{size}_traces.npz')['correct'][64]
        with np.load(ROOT/f'evidence/continuous_coverage_20261006/banks/evaluation{size}.npz',allow_pickle=False) as b:
            changed=b['changed'][:,0].astype(bool);d=b['distance'][:,0]
        for view in ('producer','consumer','fixed275'):
            v=a[view+'_correct'];assert v.shape==(193,32,size,size)
            assert np.array_equal(v,a[view+'_original_correct']&a[view+'_flipped_correct'])
            for label,mask in (('all_changed',changed),('strict_16_32',changed&(d>16)&(d<32))):
                ref=baseline&mask;wrong=~baseline&mask;q=r['views'][view][label]
                for key,num,den in (('continuous_preservation',int((ref&v.all(0)).sum()),int(ref.sum())),('terminal_retention',int((ref&v[-1]).sum()),int(ref.sum())),('sustained_progress_last16',int((wrong&v[-16:].all(0)).sum()),int(wrong.sum()))):
                    z=q[key]['pooled'];assert z['numerator']==num and z['denominator']==den
                for t,i in (('T64',0),('T128',64),('T256',192)):
                    z=q['endpoint_coverage'][t]['pooled'];assert z['numerator']==int((v[i]&mask).sum()) and z['denominator']==int(mask.sum())
                checked+=1
    for r in steps:
        size,t,C=r['size'],r['state_time'],r['consumer'];a=unpack(PUBLIC/f'single_step_traces/size{size}_t{t}_c{C}.npz')
        for phase in ('before','after'):assert np.array_equal(a[phase+'_correct'],a[phase+'_original_correct']&a[phase+'_flipped_correct'])
        with np.load(ROOT/f'evidence/continuous_coverage_20261006/banks/evaluation{size}.npz',allow_pickle=False) as b:changed=b['changed'][:,0].astype(bool)
        before,after=a['before_correct'],a['after_correct'];q=r['metrics']['all_changed']
        assert q['destruction']['pooled']['numerator']==int((changed&before&~after).sum())
        assert q['acquisition']['pooled']['numerator']==int((changed&~before&after).sum())
    rows,srows=summaries(matrix,steps)
    for filename,expected in (('matrix_summary.csv',rows),('single_step_summary.csv',srows)):
        with (PUBLIC/filename).open(newline='',encoding='utf-8') as f:actual=list(csv.DictReader(f))
        assert actual==[{k:'' if v is None else str(v) for k,v in row.items()} for row in expected]
    assert len(summary['native_replay'])==8
    assert all(not any(r[k].values()) for r in summary['native_replay'] for k in ('prefix_mismatch_bits','suffix_mismatch_bits'))
    print(json.dumps({'status':'PASS','public_files':manifest['published_file_count'],'public_MiB':round(manifest['published_bytes']/2**20,3),'recomputed_view_cohort_summaries':checked,'matrix_units':32,'single_step_units':12,'publication_fit_or_inference':False}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    if not args.verify_only:build()
    verify()
