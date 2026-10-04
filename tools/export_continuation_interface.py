"""Publish or verify the completed zero-training continuation audit data."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/continuation_interface_20261004_01'
PUBLIC = ROOT/'evidence/continuation_interface_20261004'
MANIFEST = ROOT/'CONTINUATION_INTERFACE_PUBLICATION_MANIFEST.json'
PRIVATE_KEYS = {'pid','host','hostname','username','gpu_uuid','executable','command','launch_command','cwd'}
PRIVATE_TEXT = re.compile(r'C:\\Users\\|D:\\Storage\\|/home/|/data/users/|172\.25\.100\.245|192\.168\.193\.2|\bzxy1\b|\bchenx\b|BEGIN [^\n]*PRIVATE KEY|github_pat_|ghp_[A-Za-z0-9]|sk-[A-Za-z0-9]{20,}')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''): h.update(part)
    return h.hexdigest()


def safe(v):
    if isinstance(v,dict):
        assert not (set(k.lower() for k in v)&PRIVATE_KEYS)
        for x in v.values(): safe(x)
    elif isinstance(v,list):
        for x in v: safe(x)
    elif isinstance(v,str): assert not PRIVATE_TEXT.search(v)


def checks():
    s=read(PUBLIC/'summary.json')
    assert s['status']=='COMPLETE' and s['training'] is False
    assert len(s['cross_transfer_passes'])==50
    records=list((PUBLIC/'cross').glob('*.json'))
    records=[p for p in records if p.name!='decomposition.json']
    assert len(records)==50
    for p in records:
        r=read(p)
        assert s['cross_transfer_passes'][p.stem]==r['transfer_gate']['pass']
        assert r['transfer_gate']['pass']==all(r['transfer_gate']['checks'].values())
    a=read(PUBLIC/'alignment/validation.json')
    assert len(a)==25 and a==s['alignment_validation']
    g=read(PUBLIC/'gradients/metadata.json')
    assert g['parameters_unchanged'] and g['training_updates']==0 and len(g['models'])==5
    assert read(PUBLIC/'validation.json')['status']=='PASS'
    assert len(list((PUBLIC/'native').glob('*_trace.npz')))==10
    assert len(list((PUBLIC/'cross').glob('*_trace.npz')))==80
    assert len(list(PUBLIC.rglob('*.npz')))==274


def verify():
    m=read(MANIFEST); safe(m)
    assert sha(Path(__file__))==m['exporter_sha256']
    for rel,digest in m['published_sha256'].items(): assert sha(ROOT/rel)==digest,rel
    for rel,digest in m['source_sha256'].items(): assert sha(ROOT/rel)==digest,rel
    for p in PUBLIC.rglob('*.json'): safe(read(p))
    checks()
    print(json.dumps({'status':'PASS','files':len(m['published_sha256']),'cross_cells':50,'trace_banks':90,'npz_files':274}))


def build(validation):
    assert not PUBLIC.exists() and not MANIFEST.exists(),'Refusing to overwrite published evidence'
    m,s,status=read(RUN/'manifest.json'),read(RUN/'summary.json'),read(RUN/'status.json')
    v=read(validation); safe(v)
    assert v['status']=='PASS' and status['status']==m['status']==s['status']=='COMPLETE'
    assert status['completed_cells']==50
    for rel,digest in m['source_sha256'].items():
        assert sha(RUN/'source'/rel)==sha(ROOT/rel)==digest,rel
    raw={}
    PUBLIC.mkdir(parents=True)
    for p in RUN.rglob('*'):
        if not p.is_file(): continue
        rel=p.relative_to(RUN)
        if 'source' in rel.parts or rel.as_posix() in ('manifest.json','status.json'): continue
        assert p.suffix in ('.json','.npz','.csv','.md'),str(rel)
        if p.suffix=='.json': safe(read(p))
        if p.suffix in ('.csv','.md'): assert not PRIVATE_TEXT.search(p.read_text(encoding='utf-8-sig'))
        dst=PUBLIC/rel; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dst); raw[rel.as_posix()]=sha(p)
    config={k:m[k] for k in ('protocol','training','runtime_limit_enforced','torch_version','numpy_version','backend','banks','models','source_sha256','started_utc','finished_utc','elapsed_seconds')}
    safe(config); write(PUBLIC/'config.json',config); write(PUBLIC/'validation.json',v)
    a=s['alignment_validation']; flags=s['short_observable_comparisons']
    off=[x for k,x in a.items() if len(set(k.split('__')))==2]
    note=('\n## Recorded qualification flags\n\n'
          f"- Restricted off-diagonal maps qualified: {sum(x['restricted_map_qualified'] for x in off)}/{len(off)}.\n"
          f"- Matched short-observable flags: {sum(x['matched_observables_flag'] for x in flags.values())}/{len(flags)}.\n"
          '- Transfer proxies and restricted-map qualification are separate saved tests.\n'
          '- F2/F3 are nearest short-output candidates, not guaranteed matched observables.\n'
          '- This release contains data and code; no new mechanism interpretation.\n')
    with (PUBLIC/'RESULTS.md').open('a',encoding='utf-8') as f: f.write(note)
    (PUBLIC/'REPRODUCTION.md').write_text('''# Verification and archive requirements

From repository root:

    python -X utf8 -B tools/export_continuation_interface.py --verify-only

This standard-library check verifies published/source hashes and saved counts
and gate consistency. It does not run models. All scientific JSON, CSV and
NPZ files are retained at their original relative paths; RESULTS.md additionally
lists recorded qualification flags. Start with summary.json, selection.json,
alignment/validation.json, phase0/comparisons.json and cross/decomposition.json.
The larger state, logits, trace and gradient arrays are secondary evidence.

The frozen procedure and runnable source are in new/continuation_interface/.
Full rerun requires the original final checkpoint archives identified by the
file/parameter hashes in config.json, historical Torch2.5.1 and a CUDA device.
Checkpoints, source snapshots, machine manifests, launch receipts and PIDs stay
local. The new banks, calibration states, adapters and consumer traces are
published, enabling downstream saved-data analysis without those checkpoints.
No model training or optimizer update was performed in this audit or release.
''',encoding='utf-8')
    files={p.relative_to(ROOT).as_posix():sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
    write(MANIFEST,{'protocol':'continuation_interface_publication_v1','run_id':RUN.name,
                    'source_sha256':m['source_sha256'],'published_sha256':files,
                    'raw_sha256':raw,'original_manifest_sha256':sha(RUN/'manifest.json'),
                    'original_status_sha256':sha(RUN/'status.json'),
                    'exporter_sha256':sha(Path(__file__)),
                    'excluded':['checkpoint archives','source snapshots','machine metadata','launch receipts','PIDs']})
    verify()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true'); p.add_argument('--validation')
    a=p.parse_args()
    if a.verify_only: verify()
    elif a.validation: build(ROOT/a.validation)
    else: p.error('--validation is required to build')
