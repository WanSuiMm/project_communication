"""Focused CPU fixtures and two-map CUDA qualification for the W suite."""
from pathlib import Path
import argparse

import numpy as np
import torch

import run


def stop_gate_checks():
    rows={arm:{'rescue':{'32':{'pass':True},'64':{'pass':True}},
                'handoff':{'32':{'exact_FF_output':True},'64':{'exact_FF_output':True}}}
          for arm in ('SF','SS')}
    assert run.stage1_gate({'S1':True,'F1':False},rows)['pass']
    for arm in ('SF','SS'):
        for size in ('32','64'):
            rows[arm]['rescue'][size]['pass']=False
            assert not run.stage1_gate({'S1':True,'F1':False},rows)['pass']
            rows[arm]['rescue'][size]['pass']=True
    for flags in ({'S1':False,'F1':False},{'S1':True,'F1':True}):
        result=run.stage1_gate(flags,rows)
        assert result['pass'] and not any(result['historical_localization_episode_qualified'].values())
    rows['SF']['handoff']['64']['exact_FF_output']=False
    assert not run.stage1_gate({'S1':True,'F1':False},rows)['pass']
    return {'status':'PASS','no_primary_or_confirmation_fallback':True,'neutrality_required':True,
            'historical_full_separate_from_W_control_gate':True}


def gauge_index_checks():
    # A partial second batch has different map permutations, exposing accidental
    # use of the first batch or the wrong original/flip ordering.
    index=np.stack([np.roll(np.arange(6,dtype=np.int64),i % 6) for i in range(10)])
    spec={'kind':'spatial','forward_index':index,'inverse_index':np.argsort(index,axis=1)}
    consumer=run.ConjugatedConsumer(torch.nn.Identity(),spec,10)
    for calls,ids in ((0,list(range(8))),(193,[8,9])):
        consumer.calls=calls; consumer._set_batch('cpu')
        w=torch.arange(2*len(ids)*24*6,dtype=torch.float32).reshape(2*len(ids),24,2,3)
        expected=np.concatenate((index[ids],index[ids]))
        gathered=consumer._apply_w(w)
        assert torch.equal(gathered.flatten(2),w.flatten(2).gather(2,torch.from_numpy(expected)[:,None].expand(-1,24,-1)))
        assert torch.equal(consumer._apply_w(gathered,inverse=True),w)
    return {'status':'PASS','partial_second_batch_correct_maps':True,'original_flip_order_and_exact_inverse':True}


def check(out):
    assert out.parent==run.ROOT/'analyses' and not out.exists(), 'New qualification file required'
    fixture=run.load_module('w_medium_transform_checks',Path(__file__).with_name('check_transforms.py'))
    transform_checks=fixture.run_checks()
    metric_checks=run.load_module('w_medium_cohort_checks',run.ROOT/'new/state_factorization/check_metrics.py').run_checks()
    assert transform_checks['status']=='PASS' and metric_checks['status']=='PASS', 'CPU qualification failed'
    gate_checks=stop_gate_checks()
    gauge_checks=gauge_index_checks()
    run.base.setup()
    free,_=torch.cuda.mem_get_info(); assert free>=3500*1024**2
    bindings,rows=run.sources(),run.factor.catalog()
    models={k:run.base.load_model(v) for k,v in rows.items()}
    cuda={}
    for size,seed in ((32,97932),(64,97964)):
        d=run.base.bank(size,2,seed); native={}; states={}
        for mid,model in models.items():
            trace,_,state,_=run.base.paired_trace(model,d)
            native[mid],states[mid]=trace,state
        cap=run.base.capture(models['S1'],d,run.DONOR_TIMES)
        donors={t:cap[0][i] for i,t in enumerate(run.DONOR_TIMES)}
        control,specs,_=run.transform_states(states['S1'],states['F1'],d,donors)
        s,f=states['S1'],states['F1']
        all_states={'FF':f,'SF':(s[0],f[1]),'FS':(f[0],s[1]),'SS':s,**control}
        cohorts=run.metrics.cohorts(native['S1']['correct'][64],native['F1']['correct'][64],d['changed'][:,0].numpy().astype(bool))
        cuda[size]={}; sf_trace=sf_logits=None
        for arm in run.ARMS:
            spec=specs[arm.split('_')[1]] if arm.endswith('_gauge') else None
            state=all_states[arm]
            diag=run.handoff(models['F1'],d,state,f,visible=arm in ('FS','SS'),spec=spec)
            consumer=run.ConjugatedConsumer(models['F1'],spec,2) if spec else models['F1']
            trace,_,_,logits,norms=run.factor.suffix(consumer,d,state)
            assert trace['correct'].shape==(193,2,size,size) and np.isfinite(norms).all()
            run.metrics.summarize(trace['correct'],cohorts)
            if arm=='FF':
                assert all(np.array_equal(v,native['F1'][k][64:]) for k,v in trace.items())
            if arm=='SF': sf_trace,sf_logits=trace,logits
            if spec:
                assert all(np.array_equal(v,sf_trace[k]) for k,v in trace.items()), f'Gauge trace mismatch: {size}/{arm}'
                assert all(np.array_equal(v,sf_logits[k]) for k,v in logits.items()), f'Gauge logits mismatch: {size}/{arm}'
                assert consumer.calls==193
                diag['exact_gauge_equivalence_to_SF']=True
            cuda[size][arm]={k:diag[k] for k in ('readout_neutral_arm','exact_FF_output','max_absolute_logit_error_vs_FF',
                         'open_decision_disagreements_vs_FF','instrumented_first_step_exact')}
            if spec: cuda[size][arm]['exact_gauge_equivalence_to_SF']=True
        bad_w=all_states['SF'][0].copy(); bad_w[0,0,0,0]=np.inf
        try:
            run.handoff(models['F1'],d,(bad_w,f[1]),f)
        except FloatingPointError:
            cuda[size]['nonfinite_handoff_classified']=True
        else:
            raise AssertionError('Nonfinite handoff was not classified')
        finally:
            models['F1'].cpu()
    for mid,model in models.items():
        assert run.base.tensor_hash(model.state_dict())==rows[mid]['parameter_sha256']
    result={'protocol':run.PROTOCOL,'status':'PASS','training_updates':0,'source_sha256':bindings,
            'checkpoints':rows,'cpu_transforms':transform_checks,'cpu_cohorts':metric_checks,
            'cpu_stop_gate':gate_checks,'cuda':cuda,'FF_suffix_exact':True,'gauge_suffixes_exact':True,
            'cpu_gauge_batch_indices':gauge_checks,
            'parameters_unchanged':True,'smoke_seeds':{'32':97932,'64':97964}}
    run.write(out,result)
    print({'status':'PASS','sizes':[32,64],'arms_per_size':len(run.ARMS),'training_updates':0},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True)
    args=p.parse_args(); check((run.ROOT/args.out).resolve())
