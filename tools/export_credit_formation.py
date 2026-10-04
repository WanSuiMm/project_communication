"""Publish saved formation data and verify it without running models."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil

import numpy as np
from export_continuation_interface import read, write, sha, safe, PRIVATE_TEXT

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/credit_formation_20261004_01'
PUBLIC = ROOT/'evidence/credit_formation_20261004'
MANIFEST = ROOT/'CREDIT_FORMATION_PUBLICATION_MANIFEST.json'
UPDATES = (140,145,175,180,190,195,200)


def equal_metrics(a,b):
    if isinstance(b,dict):
        assert set(a)==set(b)
        for k in b: equal_metrics(a[k],b[k])
    elif isinstance(b,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal_metrics(x,y)
    elif isinstance(b,float):
        assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12),(a,b)
    else: assert a==b,(a,b)


def scientific_checks(folder):
    s = read(folder/'summary.json')
    assert s['status'] == 'COMPLETE' and s['training'] is False
    assert s['training_updates'] == 0 and s['completed_checkpoints'] == 7
    assert s['updates'] == list(UPDATES)
    spec = importlib.util.spec_from_file_location('published_credit_metrics', ROOT/'new/credit_formation/credit_metrics.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    assert helper.formation_verdict(s['credit_metrics'],s['continuation']) == s['formation_screen']
    banks = {}
    for size in (32,64):
        with np.load(folder/'banks'/f'test{size}.npz',allow_pickle=False) as z:
            banks[size] = z['changed'][:,0].astype(bool)
    for update in UPDATES:
        p = folder/f'u{update:03}'
        metadata = read(p/'gradients/metadata.json')
        assert metadata['training_updates'] == 0 and metadata['parameters_unchanged']
        assert metadata['parameter_sha256_before'] == metadata['parameter_sha256_after']
        names, shapes = metadata['parameter_names'], metadata['parameter_shapes']
        vectors = []
        for bi,row in enumerate(metadata['batches']):
            assert row['batch'] == bi and row['indices'] == list(range(bi*8,bi*8+8))
            assert row['exact_trajectory_match'] and len(row['trajectory_sha256_by_step']) == 65
            assert abs(row['loss8']-row['loss64']) <= 1e-6
            with np.load(p/'gradients'/f'batch{bi:02}.npz',allow_pickle=False) as z:
                a,b = z['gradient_k8'].copy(),z['gradient_full64'].copy()
                assert z['parameter_names'].tolist() == names and z['indices'].tolist() == row['indices']
            equal_metrics(helper.summarize_pair(a,b,names,shapes),row['metrics'])
            vectors.append((a,b))
        assert len(vectors) == 4
        with np.load(p/'gradients/mean_gradients.npz',allow_pickle=False) as z:
            a,b = z['mean_gradient_k8'],z['mean_gradient_full64']
            assert np.array_equal(a,np.mean(np.stack([v[0] for v in vectors]),axis=0))
            assert np.array_equal(b,np.mean(np.stack([v[1] for v in vectors]),axis=0))
            equal_metrics(helper.summarize_pair(a,b,names,shapes),metadata['mean_metrics'])
            assert metadata['mean_metrics'] == s['credit_metrics'][str(update)]
        assert [v['metrics'] for v in metadata['batches']] == s['batch_credit_metrics'][str(update)]
        ph = read(p/'phenotype.json')
        assert ph['phenotype_gate']['pass'] == s['native_full_flags'][str(update)]
        handoff = read(p/'continuation.json')
        for size in (32,64):
            with np.load(p/f'size{size}_trace.npz',allow_pickle=False) as z:
                trace = z['correct']; assert trace.dtype == np.bool_
                assert trace.shape == (257,32,size,size)
                assert np.array_equal(trace,z['original_correct'] & z['flipped_correct'])
                selected = banks[size]
                old = trace[64] & selected; wrong = ~trace[64] & selected
                nums = {'preservation':(old & trace[64:].all(axis=0)).sum(axis=(1,2)),
                        'sustained_progress':(wrong & trace[241:257].all(axis=0)).sum(axis=(1,2))}
                dens = {'preservation':old.sum(axis=(1,2)),'sustained_progress':wrong.sum(axis=(1,2))}
                for metric in nums:
                    saved = handoff[str(size)]['bands']['all_changed'][metric]
                    n,d = int(nums[metric].sum()),int(dens[metric].sum())
                    value = n/d if d else None
                    assert saved['pooled'] == {'numerator':n,'denominator':d,'value':value}
                    assert value == s['continuation'][str(update)][str(size)][metric]
                    for i,row in enumerate(saved['per_map']):
                        assert row['numerator'] == int(nums[metric][i]) and row['denominator'] == int(dens[metric][i])
    with (folder/'metrics.csv').open(encoding='utf-8',newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 210
    for row in rows:
        update,tag,scope = row['update'],row['aggregation'],row['scope']
        source = (s['credit_metrics'][update] if tag=='mean_vector'
                  else s['batch_credit_metrics'][update][int(tag.removeprefix('batch'))])
        metric = source['overall'] if scope=='overall' else source['groups'][scope]
        for k in ('cosine','C_parallel','C_miss','norm_ratio','norm8','norm64'):
            assert (None if row[k]=='' else float(row[k])) == metric[k]
    assert len(list(folder.rglob('*.npz'))) == 65
    return {'status':'PASS','scope':'Saved-artifact validation; no model inference or training.',
            'completed_checkpoints':7,'gradient_pairs':28,'gradient_archives':35,
            'native_trace_banks':14,'npz_files':65,'csv_metric_rows':210,
            'gradient_metrics_recomputed':True,'gradient_comparison_rtol':1e-9,'gradient_comparison_atol':1e-12,
            'continuation_counts_recomputed':True,
            'formation_screen_recomputed':True,'forward_match_flags_and_parameter_hashes_checked':True,
            'elapsed_seconds':s['elapsed_seconds'],'formation_verdict':s['formation_screen']['verdict'],
            'native_full_passes':sum(s['native_full_flags'].values())}


def verify():
    m = read(MANIFEST); safe(m)
    for category in ('published_sha256','source_sha256','publication_tools_sha256'):
        for rel,digest in m[category].items(): assert sha(ROOT/rel) == digest,rel
    for p in PUBLIC.rglob('*.json'): safe(read(p))
    result = scientific_checks(PUBLIC)
    assert result == read(PUBLIC/'validation.json')
    print(json.dumps(result))


def build():
    assert not PUBLIC.exists() and not MANIFEST.exists(),'Refusing to overwrite published data'
    m,status = read(RUN/'manifest.json'),read(RUN/'status.json')
    assert m['status'] == status['status'] == 'COMPLETE' and status['completed_checkpoints'] == 7
    for rel,digest in m['source_sha256'].items():
        assert sha(RUN/'source'/rel) == sha(ROOT/rel) == digest,rel
    result = scientific_checks(RUN)
    raw = {}
    PUBLIC.mkdir(parents=True)
    for p in RUN.rglob('*'):
        if not p.is_file(): continue
        rel = p.relative_to(RUN)
        if 'source' in rel.parts or rel.as_posix() in ('manifest.json','status.json'): continue
        assert p.suffix in ('.json','.npz','.csv','.md','.png','.pdf')
        if p.suffix=='.json': safe(read(p))
        if p.suffix in ('.csv','.md'): assert not PRIVATE_TEXT.search(p.read_text(encoding='utf-8-sig'))
        dst = PUBLIC/rel; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dst); raw[rel.as_posix()] = sha(p)
    allowed = ('protocol','training','training_updates','runtime_limit_enforced','checkpoints','banks',
               'batches','torch_version','numpy_version','backend','started_utc','finished_utc','elapsed_seconds','source_sha256')
    config = {k:m[k] for k in allowed}; safe(config)
    write(PUBLIC/'config.json',config); write(PUBLIC/'validation.json',result)
    (PUBLIC/'REPRODUCTION.md').write_text('''# Reading and saved-data verification

Start with RESULTS.md, summary.json, metrics.csv and credit_formation.png.
Per-update continuation.json and phenotype.json retain cohorts, integer counts,
equal-map summaries, Full flags and reasons. The 65 NPZ files retain all banks,
per-batch/mean gradients, paired correctness traces and endpoint logits.

From repository root (NumPy required; no model/GPU execution):

    python -X utf8 -B tools/export_credit_formation.py --verify-only

The verifier checks source/publication hashes, recomputes gradient metrics,
mean vectors, continuation counts and the frozen synchrony verdict. Recorded
forward equality and no-update checks are retained in gradient metadata.
Full rerun requires the historical checkpoint archives identified by hashes in
config.json, Torch2.5.1, CUDA and matplotlib; see new/credit_formation/PROTOCOL.md.
Those checkpoints, source snapshots and machine launch receipts remain local.
Existing checkpoints and the original run have not been changed.

The behavior episode qualifies but overall/F/Q synchrony fails. All seven
current-bank native Full flags are false; update200 fails the strict T64 pooled
reach threshold (0.76648 versus0.80). This is distinct from the synchrony screen,
and does not erase the original checkpoint's result on its earlier banks.
The study is one selected trajectory on reused held-out banks; no population
or causal mechanism claim follows. No additional model run was made to publish.
''',encoding='utf-8')
    tools = ('tools/export_credit_formation.py','tools/export_continuation_interface.py')
    write(MANIFEST,{'protocol':'credit_formation_publication_v1','run_id':RUN.name,
        'published_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in PUBLIC.rglob('*') if p.is_file()},
        'source_sha256':m['source_sha256'],'publication_tools_sha256':{p:sha(ROOT/p) for p in tools},
        'raw_sha256':raw,'original_manifest_sha256':sha(RUN/'manifest.json'),
        'original_status_sha256':sha(RUN/'status.json'),
        'excluded':['checkpoints','source snapshots','machine manifests','launch receipts','PIDs']})
    verify()


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true'); a=p.parse_args()
    verify() if a.verify_only else build()
