"""One durable CPU qualification of the three independent follow-ups."""
import argparse
from pathlib import Path

from train import ROOT, now, sources, write
import check_training
import check_phenotype
import check_causal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    assert out.is_relative_to(ROOT/'analyses') and not out.exists()
    hashes = sources()
    checks = {'training': check_training.run_checks(),
              'phenotype': check_phenotype.run_checks(),
              'causal': check_causal.run_checks()}
    assert all(result['status'] == 'PASS' for result in checks.values())
    assert sources() == hashes, 'Source changed during CPU qualification'
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'PASS', 'training': False, 'gpu_execution': False,
              'finished_utc': now(), 'source_sha256': hashes, 'checks': checks}
    write(out, result)
    print({'status': result['status'], 'source_bindings': len(hashes), 'output': str(out)})


if __name__ == '__main__':
    main()
