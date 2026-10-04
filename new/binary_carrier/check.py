"""Meaningful fixtures plus two-map CUDA plumbing qualification."""
from pathlib import Path
import argparse
import copy
import tempfile

import numpy as np
import torch

import run


class Dummy(torch.nn.Module):
    def step(self,state,x): return state[0]+.01,state[1]+.02
    def logits(self,state): return state[1][:,:1]


def clocks():
    fit={'raw_coefficient':[1.]+[0.]*23,'raw_intercept':0.,
         'templates':{'p0':[-1.]*24,'p1':[1.]*24}}
    consumer=run.ProjectedConsumer(Dummy(),fit,10,'bit')
    for count in (8,2):
        w=torch.ones(2*count,24,2,3); z=torch.zeros(2*count,8,2,3)
        x=torch.ones(2*count,3,2,3); x[:,0,0,0]=0
        state=(w,z)
        for t in range(64,257):
            if t>64: state=consumer.step(state,x)
            consumer.logits(state)
            assert torch.equal(state[1],z+.02*(t-64)) or torch.allclose(state[1],z+.02*(t-64),atol=5e-6)
    assert consumer.calls==386 and len(consumer.events)==48
    for block,events in enumerate((consumer.events[:24],consumer.events[24:])):
        assert [e['t'] for e in events]==list(range(72,257,8))
        assert all(e['map_start']==8*block for e in events)
        assert all(e['map_count']==(8 if block==0 else 2) for e in events)
    with tempfile.TemporaryDirectory() as tmp:
        consumer.save(Path(tmp)/'projections.npz')
        with np.load(Path(tmp)/'projections.npz') as a:
            assert a['before_bits'].shape==(24,2,10,2,3)
    return {'status':'PASS','partial_second_batch':True,'projection_clock_72_to_256':True,'readout_Z_exact':True}


def gate_checks():
    def result(k,p,maps=16,cells=100):
        return {'cohorts':{c:{metric:{'pooled':{'value':v,'denominator':cells},'valid_maps':maps}}
                          for c,metric,v in (('keep','continuous64_256',k),('prog','sustained241_256',p))}}
    sf=result(1.,.6)
    assert run.compression_gate(result(.95,.54),sf,True)['pass']
    assert not run.compression_gate(result(.95,.539),sf,True)['pass']
    assert not run.compression_gate(result(.949,.6),sf,True)['pass']
    assert run.compression_gate(result(1.,1.,15),sf,True)['status']=='UNQUALIFIED'
    assert run.compression_gate(result(1.,1.),sf,False)['status']=='UNQUALIFIED'
    return {'status':'PASS','exact_90pct_boundary':True,'support_and_anchor_required':True}


def check(out):
    assert out.parent==run.ROOT/'analyses' and not out.exists()
    cpu=run.load_module('binary_decoder_checks',Path(__file__).with_name('check_decoder.py')).run_checks()
    assert cpu['status']=='PASS'
    clock=clocks(); gate=gate_checks()
    cohort=run.load_module('binary_cohort_checks',run.ROOT/'new/state_factorization/check_metrics.py').run_checks()
    run.base.setup(); free,_=torch.cuda.mem_get_info(); assert free>=3500*1024**2
    bindings,rows=run.sources(),run.factor.catalog(); models={k:run.base.load_model(v) for k,v in rows.items()}
    fit=None; cuda={}
    for size,seed in ((32,98332),(64,98364)):
        d=run.base.bank(size,2,seed); states,native={},{}
        for mid,model in models.items():
            trace,_,state,_=run.base.paired_trace(model,d)
            states[mid]=state; native[mid]=trace
        x,y,m=run.pair_arrays(d)
        if fit is None:
            fit=run.decoder.fit(states['S1'][0],y,m); assert fit['converged']
        prepared=run.states_for(states['S1'],states['F1'],d,fit)
        masks=run.metrics.cohorts(native['S1']['correct'][64],native['F1']['correct'][64],d['changed'][:,0].numpy().astype(bool))
        probe,bits=run.probe(states['S1'][0],d,fit,masks)
        assert probe['prog']['paired']['pooled']['denominator']==int(masks['prog'].sum())
        # Only oracle may depend on labels. Perturbing held-out labels cannot
        # change any primary/control restart constructed from W or source X.
        edited=copy.deepcopy(d); edited['y']=1-edited['y']; edited['y_flip']=1-edited['y_flip']
        other=run.states_for(states['S1'],states['F1'],edited,fit)
        for arm in run.ARMS:
            if arm!='oracle_once': assert all(np.array_equal(a,b) for a,b in zip(prepared[arm],other[arm]))
        cuda[size]={}
        for arm in run.ARMS:
            state=prepared[arm]
            diag=run.medium.handoff(models['F1'],d,state,states['F1'],visible=arm=='SS')
            consumer=run.ProjectedConsumer(models['F1'],fit,2,arm.split('_')[0]) if arm.endswith('repeat8') else models['F1']
            trace,_,_,_,norms=run.factor.suffix(consumer,d,state)
            run.metrics.summarize(trace['correct'],masks); assert np.isfinite(norms).all()
            if arm=='FF': assert all(np.array_equal(v,native['F1'][k][64:]) for k,v in trace.items())
            if arm.endswith('repeat8'):
                assert consumer.calls==193 and len(consumer.events)==24
                w=torch.from_numpy(states['S1'][0]).cuda().double()
                a=torch.as_tensor(fit['raw_coefficient'],device='cuda',dtype=torch.float64)
                gpu=(w*a[None,:,None,None]).sum(1)+float(fit['raw_intercept'])>=0
                assert np.array_equal(bits,gpu.cpu().numpy())
                diag['CPU_CUDA_decoder_bits_exact']=True
            cuda[size][arm]={k:diag[k] for k in ('exact_FF_output','readout_neutral_arm','instrumented_first_step_exact')}
    for mid,model in models.items(): assert run.base.tensor_hash(model.state_dict())==rows[mid]['parameter_sha256']
    result={'protocol':run.PROTOCOL,'status':'PASS','recurrent_training_updates':0,'decoder_smoke_fits':1,
        'source_sha256':bindings,'checkpoints':rows,'cpu_decoder':cpu,'cpu_clocks':clock,
        'cpu_gates':gate,'cpu_cohorts':cohort,'cuda':cuda,'parameters_unchanged':True,
        'smoke_seeds':{'32':98332,'64':98364},'smoke_outcomes_not_used_for_selection':True}
    run.write(out,result); print({'status':'PASS','sizes':[32,64],'arms_per_size':12},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True)
    args=p.parse_args(); check((run.ROOT/args.out).resolve())
