"""Independent CPU-only data/reference bindings for a dense seed4 dispatch.

This supplements the already frozen runner without changing running code.
Its receipt is post-dispatch evidence, not a retroactive preflight binding.
No optimizer, GPU inference, job polling, or generated-run edits are performed.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('transition_binding_runner',ROOT/'new/transition_100_200/run.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    run=Path(args.run).resolve()
    out=Path(args.out).resolve()
    assert run.is_relative_to(ROOT/'runs') and out.is_relative_to(ROOT/'analyses')
    assert not out.exists()
    manifest=runner.read(run/'manifest.json')
    assert manifest['protocol']==runner.PROTOCOL and not manifest['preflight']
    assert manifest['source_sha256']==runner.sources()
    assert manifest['reference_bindings']==runner.references()
    assert manifest['gpu']=='NVIDIA GeForce RTX 4060 Laptop GPU'
    public=runner.read(ROOT/'evidence/streaming_carry_init2345/manifest.json')
    historical={}
    banks={}
    for size in (32,64):
        data=runner.bank(size,32,40000+size)
        digest=runner.tensor_hash(data)
        assert digest==public['evaluation_data_sha256'][str(size)]
        historical[str(size)]=digest
        bank_path=run/f'bank_size{size}.npz'
        with np.load(bank_path,allow_pickle=False) as saved:
            values={k:torch.from_numpy(saved[k]) for k in saved.files}
        saved_hash=runner.tensor_hash(values)
        assert saved_hash==manifest['audit_banks'][str(size)]['tensor_sha256']
        assert saved_hash==runner.tensor_hash(runner.bank(size,16,61000+size))
        banks[str(size)]={'tensor_sha256':saved_hash,'file_sha256':runner.sha(bank_path)}
    reference_manifest=runner.read(runner.REFERENCE/'manifest.json')
    assert banks['32']['tensor_sha256']==reference_manifest['stage_diagnostic_data_sha256']
    diagnostics={}
    for update in (100,200,300):
        path=runner.REFERENCE/f'baseline_seed4_u{update:03}_diagnostic.json'
        value=runner.read(path)
        assert value['checkpoint_parameter_sha256']==manifest['reference_bindings']['anchors'][str(update)]['parameter_sha256']
        assert value['training_update']==update and value['seed']==4
        assert value['num_maps']==16 and value['trace_steps']['last']==128
        diagnostics[str(update)]={'sha256':runner.sha(path),'checkpoint_parameter_sha256':value['checkpoint_parameter_sha256']}
    out.parent.mkdir(parents=True,exist_ok=True)
    runner.write(out,{'status':'PASS','finished_utc':runner.now(),'training':False,'gpu_execution':False,
        'binding_time':'post-dispatch before offline audit; not retroactive preflight',
        'manifest_sha256':runner.sha(run/'manifest.json'),'source_sha256':manifest['source_sha256'],
        'historical_evaluation_data_sha256':historical,'audit_banks':banks,
        'reference_manifest_sha256':runner.sha(runner.REFERENCE/'manifest.json'),
        'reference_stage_diagnostics':diagnostics,'verified_gpu_name':manifest['gpu'],
        'validation_script_sha256':runner.sha(Path(__file__))})
    print({'status':'PASS','historical_banks':2,'audit_banks':2,'stage_references':3,'output':str(out)})


if __name__=='__main__':
    main()
