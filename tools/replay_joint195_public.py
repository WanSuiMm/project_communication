"""Run the unchanged joint audit against a separately reproduced dense archive.

First reproduce seed4 with tools/replay_transition_public.py. This wrapper
checks published parameter hashes, not unavailable original checkpoint bytes.
It does not start inference until explicitly executed.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reference-run',required=True)
    parser.add_argument('--check-bindings',action='store_true')
    args,remaining=parser.parse_known_args()
    ref=(ROOT/args.reference_run).resolve()
    assert ref.is_relative_to(ROOT/'runs'),'Reference must be a project run'
    spec=importlib.util.spec_from_file_location('public_joint195_runner',ROOT/'new/audit_195_200/run.py')
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    runner.REF=ref
    original=runner.bindings
    def public_bindings():
        result=original()
        public=runner.read(ROOT/'evidence/joint195_200_20261003/run_metadata.json')
        assert result['parameter_sha256']==public['parameter_sha256'],'Reproduced u195/u200 parameters mismatch'
        produced=runner.read(ref/'manifest.json')['audit_banks']
        for size in (32,64):
            assert produced[str(size)]['tensor_sha256']==public['audit_banks'][f'primary{size}']['tensor_sha256'],'Reproduced bank mismatch'
        result.update({'reference_mode':'reproduced_archive_and_published_parameter_hashes',
                       'original_input_archive_bytes_verified':False})
        for path in (Path(__file__).resolve(),ROOT/'evidence/joint195_200_20261003/run_metadata.json'):
            result['source_sha256'][path.relative_to(ROOT).as_posix()]=runner.sha(path)
        return result
    runner.bindings=public_bindings
    if args.check_bindings:
        result=public_bindings()
        print(json.dumps({'status':'PASS','parameter_sha256':result['parameter_sha256'],
                          'reference_mode':result['reference_mode'],'original_input_archive_bytes_verified':False}))
        return
    sys.argv=[sys.argv[0],*remaining]
    runner.main()


if __name__=='__main__':main()
