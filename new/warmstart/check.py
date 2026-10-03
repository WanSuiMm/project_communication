"""One durable CPU qualification for the frozen detached warm-start screen."""
import argparse
from pathlib import Path
import sys

from run import ROOT, local_module, sources, write, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    assert out.is_relative_to(ROOT/'analyses') and not out.exists()
    hashes = sources()
    diagnostic = local_module('diagnostics', 'diagnostics.py')
    sys.modules['diagnostics'] = diagnostic
    training_check = local_module('warmstart_training_check', 'check_training.py')
    diagnostic_check = local_module('warmstart_diagnostic_check', 'check_diagnostics.py')
    checks = {'training': training_check.run_checks(), 'diagnostics': diagnostic_check.run_checks()}
    assert all(v['status']=='PASS' for v in checks.values())
    assert sources()==hashes, 'Sources changed during qualification'
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, {'status':'PASS','training':False,'gpu_execution':False,
                'finished_utc':now(),'source_sha256':hashes,'checks':checks})
    print({'status':'PASS','source_bindings':len(hashes),'checks':checks,'output':str(out)})


if __name__ == '__main__':
    main()
