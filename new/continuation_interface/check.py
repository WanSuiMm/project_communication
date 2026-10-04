"""One combined CPU qualification and two-map frozen GPU smoke."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

import alignment
import metrics
import gradients
import run


def check(out):
    assert out.parent == run.ROOT/'analyses' and not out.exists(), 'Use a new analyses receipt'
    started = time.monotonic()
    torch.set_num_threads(2)
    cpu = {'alignment': alignment.sanity_check(), 'metrics': metrics.sanity_check(),
           'gradients': gradients.sanity_check()}
    rng = np.random.default_rng(96008)
    w = rng.normal(size=(2,2,24,8,8)); z = rng.normal(size=(2,2,8,8,8))
    source = (w,z)
    pw = np.eye(6)[rng.permutation(6)]; pz = np.eye(8)[run.PERM]
    target = (np.einsum('oi,tblihw->tblohw',pw,w.reshape(2,2,4,6,8,8)).reshape(w.shape),
              np.einsum('oi,tbihw->tbohw',pz,z))
    mask = np.ones((2,1,8,8))
    recovered = alignment.fit_alignment(source,target,mask)
    diagnostic = alignment.alignment_diagnostics(source,target,mask,recovered)
    assert max(diagnostic['normalized_rms_residual'].values()) < 1e-3
    cpu['full_rank_alignment'] = diagnostic
    sources, rows = run.source_hashes(), run.catalog()
    run.setup()
    free, total = torch.cuda.mem_get_info()
    assert free >= 3_500_000_000, 'Insufficient GPU headroom'
    model = run.load_model(rows['S1'])
    data = run.bank(32,2,90032)
    tick = time.monotonic()
    gauge = run.gauge_control(model,data,steps=64)
    assert gauge['status'] == 'PASS', gauge
    # One self-handoff catches original/flipped batch-order and t64 indexing errors.
    native, _, state64, ends = run.paired_trace(model,data)
    handed, _, _, handed_ends = run.paired_trace(model,data,state64,start_time=64)
    assert np.array_equal(native['correct'][64:],handed['correct'])
    for t in ends:
        assert np.max(np.abs(ends[t]-handed_ends[t])) <= 1e-5*(1+np.max(np.abs(ends[t])))
    captured = run.capture(model,data,(8,64))
    assert all(np.allclose(captured[j][1],state64[j],atol=1e-6,rtol=1e-6) for j in (0,1))
    torch.cuda.synchronize()
    elapsed = time.monotonic()-tick
    assert run.tensor_hash(model.state_dict()) == rows['S1']['parameter_sha256']
    record = {'status':'PASS','protocol':run.PROTOCOL,'training_updates':0,
              'runtime_limit_enforced':False,'source_sha256':sources,
              'reference_checkpoint_sha256':{k:v['sha256'] for k,v in rows.items()},
              'cpu':cpu,'gpu_smoke':gauge,'self_handoff_exact':True,'capture_cue_order_pass':True,
              'two_map_smoke_seconds':elapsed,
              'elapsed_seconds':time.monotonic()-started,'torch_version':str(torch.__version__),
              'gpu':torch.cuda.get_device_name(0),'memory_free_bytes':free}
    run.write(out,record)
    print({'status':'PASS','two_map_smoke_seconds':elapsed,'elapsed_seconds':record['elapsed_seconds']})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    check((run.ROOT/a.out).resolve())
