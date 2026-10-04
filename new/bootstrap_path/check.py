"""One lightweight schedule and first-step optimizer qualification."""
import argparse
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
from schedules import check_suite
from check_first_step import run_check
from run import source_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = (ROOT/args.out).resolve()
    assert out.is_relative_to(ROOT/'analyses') and not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {'schedules': check_suite(), 'first_step': run_check()}
    result['status'] = 'PASS'
    result['source_sha256'] = source_hashes()
    result['seconds'] = time.monotonic()-started
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'seconds': result['seconds'], 'out': args.out}))


if __name__ == '__main__':
    main()
