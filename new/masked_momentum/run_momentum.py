"""One frozen same-medium generic momentum control; no automatic follow-up."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from momentum_cells import ARM, make_cell, frozen_cells
import run_wind_tunnel as runner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'evidence/inertial_seed0/arms/momentum_nca_seed0.json'
CANDIDATE = ROOT/'evidence/masked_medium_seed0/arms/masked_inertial_rd_seed0.json'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    old = read(ROOT/'INERTIAL_PUBLICATION_MANIFEST.json')
    masked = read(ROOT/'MASKED_PUBLICATION_MANIFEST.json')
    hashes = {**old['source_sha256'], **old['protocol_files_sha256'],
              **old['published_evidence_sha256'], **masked['source_sha256'],
              **masked['evidence_sha256']}
    for name, expected in hashes.items():
        if sha(ROOT/name) != expected:
            raise RuntimeError(f'Frozen file changed: {name}')
    torch.manual_seed(0)
    base = frozen_cells.make_cell('momentum_nca', 16, 128)
    next_rng = torch.rand(8)
    torch.manual_seed(0)
    wrapped = make_cell()
    assert torch.equal(next_rng, torch.rand(8)), 'RNG mismatch'
    for name, value in base.state_dict().items():
        assert torch.equal(value, wrapped.base.state_dict()[name]), name
    return old, hashes


def compare(out, args, status):
    runner.write_summary(out, {ARM: status}, args.preflight, args.seed)
    if args.preflight:
        return
    path = out/f'{ARM}_seed0.json'
    if not path.exists():
        runner.write_json(out/'comparison.json', {'status': status, 'decision': 'INVALID_OR_INCOMPLETE'})
        return
    result, old, candidate = read(path), read(OLD), read(CANDIDATE)
    schedule = lambda d: [(r['iteration'], r['rollout'], r['damage']) for r in d['training_curve']]
    schedule_matches = schedule(result) == schedule(old) == schedule(candidate)
    nested_statuses = []
    def inspect(v):
        if isinstance(v, dict):
            if 'status' in v: nested_statuses.append(v['status'])
            for item in v.values(): inspect(item)
        elif isinstance(v, list):
            for item in v: inspect(item)
    inspect(result['evaluation'])
    valid = (status == 'TRAINED' and result['completed_updates'] == 800 and schedule_matches
             and set(result['evaluation']) == {'32', '64', '128'}
             and all(s == 'EVALUATED' for s in nested_statuses))
    rows = []
    for size, ev in result['evaluation'].items():
        for horizon, curve in ev.get('curve', {}).items():
            def values(d):
                e = d['evaluation'][size]
                return {'ba': e['curve'][horizon]['balanced_accuracy'],
                        'bce': e['curve'][horizon]['bce'],
                        'paired': e.get('paired_source_information', {}).get(horizon, {}).get('both_counterfactuals_correct_on_changed_component')}
            new, prev, cand = values(result), values(old), values(candidate)
            row = {'size': int(size), 'steps': int(horizon), 'masked_momentum': new,
                   'unmasked_momentum': prev, 'masked_inertial': cand}
            for metric in ('ba', 'paired'):
                row[f'candidate_minus_masked_momentum_{metric}_pp'] = None if new[metric] is None else 100*(cand[metric]-new[metric])
                row[f'masked_minus_unmasked_momentum_{metric}_pp'] = None if new[metric] is None else 100*(new[metric]-prev[metric])
            rows.append(row)
    primary = next((r for r in rows if r['size'] == 32 and r['steps'] == 64), None)
    decision = 'INVALID_OR_INCOMPLETE'
    if valid and len(rows) == 15 and primary and primary['masked_momentum']['paired'] is not None:
        deltas = [primary[f'candidate_minus_masked_momentum_{m}_pp'] for m in ('ba', 'paired')]
        decision = ('CANDIDATE_JOINT_5PP_ADVANTAGE' if min(deltas) >= 5 else
                    'MOMENTUM_JOINT_5PP_ADVANTAGE' if max(deltas) <= -5 else 'MIXED_OR_BELOW_JOINT_5PP')
    report = {'protocol': 'masked_momentum_seed0', 'status': status, 'seed': 0,
              'logged_schedule_matches_both_references': schedule_matches,
              'decision': decision, 'primary': primary, 'all_horizons': rows,
              'limits': 'Single-seed architecture package comparison, not isolated factorization, significance, repair superiority or matched-speedup evidence.'}
    runner.write_json(out/'comparison.json', report)
    lines = ['', '## Same-medium primary comparison: 32x32 / T64', '', f'Descriptive outcome: `{decision}`.', '',
             '| Model | BA (%) | Paired correctness (%) | BCE |', '|---|---:|---:|---:|']
    if primary:
        for key in ('masked_momentum', 'masked_inertial', 'unmasked_momentum'):
            v = primary[key]
            paired = 'null' if v['paired'] is None else f"{100*v['paired']:.2f}"
            lines.append(f"| {key} | {100*v['ba']:.2f} | {paired} | {v['bce']:.4f} |")
    lines += ['', report['limits'], 'Hidden widths and initial nonzero spatial coupling differ between generic and explicit cells.',
              'No automatic seeds, retuning or architecture changes. All horizons are in comparison.json.', '']
    with (out/'RESULTS.md').open('a', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--minutes', type=float, default=25.)
    p.add_argument('--preflight', action='store_true')
    cli = p.parse_args()
    if cli.minutes <= 0: p.error('minutes must be positive')
    old, hashes = verify()
    config = read(ROOT/'evidence/inertial_seed0/config.json'); config.pop('conventions')
    config.update(arms=[ARM], out=cli.out, device=cli.device, minutes=cli.minutes, preflight=cli.preflight)
    if cli.preflight:
        config.update(steps=3, train_horizons=[64], eval_examples=2, eval_sizes=[32],
                      eval_horizons=[16, 64], recovery_steps=[8], probe_horizons=[16], log_every=1)
    args = argparse.Namespace(**config)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    torch.set_num_threads(args.threads)
    settings = old['runtime']['backend']
    torch.backends.cudnn.benchmark = settings['cudnn_benchmark']
    torch.backends.cudnn.deterministic = settings['cudnn_deterministic']
    torch.backends.cudnn.allow_tf32 = settings['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32 = settings['matmul_allow_tf32']
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    paths = [ROOT/name for name in hashes if name.endswith('.py') or (name.endswith('.md') and not name.startswith('evidence/'))]
    paths += list(HERE.glob('*.py')) + [HERE/'PROTOCOL.md']
    for path in paths:
        dest = out/'source'/path.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    runner.write_json(out/'manifest.json', {'protocol': 'masked_momentum_seed0', 'pid': os.getpid(),
        'started_utc': datetime.now(timezone.utc).isoformat(), 'config': config, 'backend': settings,
        'torch': torch.__version__, 'numpy': np.__version__, 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'initial_weights_and_rng_match_unmasked_momentum': True,
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        'reference_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in (OLD, CANDIDATE)},
        'verified_frozen_file_count': len(hashes)})
    runner.write_json(out/'status.json', {'status': 'STARTED', 'pid': os.getpid(), 'arm': ARM})
    runner.make_cell = make_cell
    runner.DEADLINE = time.monotonic()+args.minutes*60
    try:
        status = runner.train_one(args, ARM, device)
    except Exception as error:
        status = 'RUN_ERROR'
        runner.write_json(out/'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
    compare(out, args, status)
    runner.write_json(out/'status.json', {'status': 'FINISHED', 'pid': os.getpid(), 'statuses': {ARM: status},
        'finished_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'completed_statuses': {ARM: status}}), flush=True)


if __name__ == '__main__':
    main()
