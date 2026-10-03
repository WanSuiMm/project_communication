"""Replay the frozen dense experiment using published reference information.

The executed private-archive runner is left unchanged. This wrapper replaces
only reference lookup: fresh weights must match published parameter hashes,
and stage counts are compared against actual public JSON files. Original
checkpoint byte hashes are reported provenance, not reverified without those
excluded files. Preflight/formal manifests declare this distinction.

Use --check for CPU qualification; otherwise all arguments go to the frozen
runner. No training is launched by importing this module.
"""
import copy
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'evidence/transition_20261003/replay_reference'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    runner=load('public_transition_runner',ROOT/'new/transition_100_200/run.py')
    original_sources=runner.sources
    runner.REFERENCE=REFERENCE

    def public_sources():
        result=original_sources()
        extra=[Path(__file__).resolve(),*sorted(REFERENCE.glob('*.json'))]
        return {**result,**{p.relative_to(ROOT).as_posix():runner.sha(p) for p in extra}}

    def public_references():
        refs=copy.deepcopy(runner.read(REFERENCE/'bindings.json'))
        record=runner.read(REFERENCE/'baseline_seed4.json')
        original=runner.read(ROOT/'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json')
        for key in ('initial_parameter_sha256','final_parameter_sha256','train_data_sha256','schedule_sha256'):
            assert refs[key]==record[key]==original[key]
        stage_hashes={}
        for update in (100,200,300):
            path=REFERENCE/f'baseline_seed4_u{update:03}_diagnostic.json'
            stage=runner.read(path)
            assert stage['checkpoint_parameter_sha256']==refs['anchors'][str(update)]['parameter_sha256']
            assert stage['training_update']==update and stage['num_maps']==16
            stage_hashes[str(update)]=runner.sha(path)
        refs['record_sha256']=runner.sha(REFERENCE/'baseline_seed4.json')
        refs.update({'reference_mode':'published_parameter_hashes_and_public_stage_counts',
                     'original_checkpoint_files_verified':False,
                     'reference_manifest_sha256':runner.sha(REFERENCE/'manifest.json'),
                     'public_stage_reference_sha256':stage_hashes})
        return refs

    runner.sources=public_sources
    runner.references=public_references
    if '--check' in sys.argv:
        sys.argv.remove('--check')
        sys.modules['run']=runner
        checker=load('public_transition_checker',ROOT/'new/transition_100_200/check.py')
        checker.main()
    else:
        runner.main()


if __name__=='__main__':
    main()
