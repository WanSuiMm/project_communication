"""Focused projection, cohort and two-map frozen-checkpoint qualification."""
import argparse
from pathlib import Path

import numpy as np
import torch

import run


def projection_checks():
    rng = np.random.default_rng(96000)
    a, b = (rng.normal(size=(4,8,3,5)).astype(np.float32) for _ in range(2))
    r = rng.normal(size=8).astype(np.float32); bias = .37
    span, null, record = run.projections(a,b,r)
    readout = lambda z: np.einsum('c,bchw->bhw',r.astype(np.float64),z.astype(np.float64))+bias
    assert np.allclose(readout(null),readout(a),rtol=1e-6,atol=1e-6)
    assert np.allclose(readout(span),readout(b),rtol=1e-6,atol=1e-6)
    assert np.allclose(span.astype(np.float64)+null-a,b,rtol=1e-6,atol=1e-6)
    assert record['float64_null_dot_max_abs'] < 1e-12
    try: run.projections(a,b,np.zeros(8))
    except ValueError: pass
    else: raise AssertionError('Zero readout accepted')
    return {'status':'PASS','arbitrary_axis_bias_and_reconstruction':True,'zero_readout_rejected':True}


def check(out):
    assert out.parent == run.ROOT/'analyses' and not out.exists(), 'New qualification file required'
    tests = run.load_module('state_factor_fixtures', Path(__file__).with_name('check_metrics.py')).run_checks()
    projections = projection_checks()
    run.base.setup()
    bindings, rows = run.sources(), run.catalog()
    free, _ = torch.cuda.mem_get_info()
    assert free >= 3500*1024**2
    models = {k:run.base.load_model(v) for k,v in rows.items()}
    cuda_records = {}
    for size,seed in ((32,96932),(64,96964)):
        d = run.base.bank(size,2,seed)
        native = {}; states = {}
        for mid,model in models.items():
            trace,_,state,_ = run.base.paired_trace(model,d)
            native[mid],states[mid] = trace,state
        cohort = run.metrics.cohorts(native['S1']['correct'][64],native['F1']['correct'][64],d['changed'][:,0].numpy().astype(bool))
        arms,_ = run.arm_states(states['S1'],states['F1'],models['F1'].readout.weight.detach().numpy())
        cuda_records[size] = {}
        for arm,state in arms.items():
            diag = run.handoff_diagnostics(models['F1'],d,state,states['F1'],states['S1'],arm)
            assert diag['readout_property_qualified'], f'Smoke readout property failed: {size}/{arm}'
            trace,_,_,_,norms = run.suffix(models['F1'],d,state)
            assert trace['correct'].shape == (193,2,size,size) and np.isfinite(norms).all()
            if arm == 'FF':
                assert all(np.array_equal(trace[k],native['F1'][k][64:]) for k in trace)
            run.metrics.summarize(trace['correct'],cohort)
            cuda_records[size][arm] = {k:diag[k] for k in ('max_absolute_logit_error','max_normalized_logit_error',
                                             'open_decision_disagreements','readout_property_qualified','instrumented_step_exact')}
    for mid,model in models.items():
        assert run.base.tensor_hash(model.state_dict()) == rows[mid]['parameter_sha256']
    result = {'protocol':run.PROTOCOL,'status':'PASS','training_updates':0,
              'source_sha256':bindings,'checkpoints':rows,'cpu_cohorts':tests,'cpu_projection':projections,
              'cuda':cuda_records,'FF_suffix_exact':True,'parameters_unchanged':True,
              'fresh_bank_seeds':{str(k):v[1] for k,v in run.BANK_SPECS.items()}}
    run.write(out,result)
    print({'status':'PASS','cuda_sizes':[32,64],'arms_per_size':8,'training_updates':0})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args();check((run.ROOT/args.out).resolve())
