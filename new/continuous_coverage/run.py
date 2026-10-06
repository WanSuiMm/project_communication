"""Matched fixed-parameter 256-step super-updates and dense formation records."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'new/trajectory_qualification'))
import common as C


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


R = module('_continuous_coverage_runtime', 'runtime.py')
E = module('_continuous_coverage_evaluation', 'evaluation.py')
M = module('_continuous_coverage_metrics', 'metrics.py')
PROTOCOL = 'continuous_execution_coverage_v1'
ARMS = ('reset64x4', 'continuous256')
BLOCKS, UPDATES = 8, 300
INIT_SEEDS = tuple(range(96001, 96009))
SCHEDULE_SEEDS = tuple(range(97001, 97009))
EVAL_SEEDS = {32: 102032, 64: 102064}
CHECKPOINTS = tuple(range(0, UPDATES+1, 25))
CADENCE = {'forward_steps': 256, 'backward_calls': 32, 'loss_count': 32,
           'detach_after_loss': 32, 'optimizer_steps': 1, 'credit_horizon': 8}
LOSS_FN = C.backward_trajectory.__globals__['balanced_loss']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_hashes():
    names = set(read(ROOT / 'STREAMING_CARRY_PUBLICATION_MANIFEST.json')['source_sha256'])
    names.update({'STREAMING_CARRY_PUBLICATION_MANIFEST.json',
                  'new/trajectory_qualification/common.py',
                  'new/trajectory_qualification/fresh_recipe.py',
                  'new/seed4_followup/phenotype.py',
                  'new/frontier_audit/audit.py', 'new/frontier_audit/metrics.py',
                  'new/frontier_audit/report.py', 'new/stream_path_audit/audit.py',
                  'new/stream_path_audit/operators.py', 'tools/launch_continuous_coverage.ps1'})
    names.update(p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.iterdir()
                 if p.suffix in ('.py', '.md'))
    return {name: C.sha(ROOT / name) for name in sorted(names)}


def model_for(seed, device='cuda'):
    torch.manual_seed(seed)
    model = C.StreamingCell().to(device)
    assert sum(p.numel() for p in model.parameters()) == 5033
    return model


def write_csv(path, records):
    fields = ('block', 'initialization_seed', 'schedule_seed', 'arm', 'update', 'size',
              'R_strict_pooled_T64', 'R_strict_mean_T64', 'S_retention64_to256',
              'S_continuous_survival64_to256', 'retention_reference_pixels',
              'retention_reference_maps', 'ever_regressed_fraction',
              'coverage_gain64_to256', 'joint_readiness', 'full_pass')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for size, row in record['metrics']['sizes'].items():
                base = {key: record[key] for key in ('block', 'initialization_seed', 'schedule_seed', 'arm', 'update')}
                writer.writerow({**base, 'size': size,
                    'R_strict_pooled_T64': row['R_strict_pooled_T64'],
                    'R_strict_mean_T64': row['R_strict_mean_T64'],
                    'S_retention64_to256': row['S_retention64_to256'],
                    'S_continuous_survival64_to256': row.get('S_continuous_survival64_to256'),
                    'retention_reference_pixels': row['retention_reference_pixels'],
                    'retention_reference_maps': row['retention_reference_maps'],
                    'ever_regressed_fraction': row['ever_regressed_fraction'],
                    'coverage_gain64_to256': row['coverage_gain64_to256'],
                    'joint_readiness': record['joint']['pass'] if size == '32' else None,
                    'full_pass': record['metrics']['full_pass']})


def plot(out, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'reset64x4': '#d47d34', 'continuous256': '#2278b5'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, size in zip(axes, ('32', '64')):
        for block in range(BLOCKS):
            for arm in ARMS:
                seq = sorted((r for r in records if r['block'] == block and r['arm'] == arm), key=lambda r: r['update'])
                points = [r['metrics']['sizes'][size] for r in seq]
                valid = [p for p in points if p['S_retention64_to256'] is not None
                         and p['retention_reference_pixels'] >= 100
                         and p['retention_reference_maps'] >= 16]
                ax.plot([p['R_strict_pooled_T64'] for p in valid],
                        [p['S_retention64_to256'] for p in valid], '-o', markersize=2,
                        color=colors[arm], alpha=.42, linewidth=1,
                        label=arm if block == 0 else None)
                if (seq and seq[-1]['update'] == UPDATES and points[-1] in valid):
                    p = points[-1]
                    ax.scatter(p['R_strict_pooled_T64'], p['S_retention64_to256'],
                               marker='*', s=90, color=colors[arm], edgecolors='black', linewidths=.3)
        ax.axvline(.8, color='gray', linewidth=.7, linestyle='--')
        ax.axhline(.95, color='gray', linewidth=.7, linestyle='--')
        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel='Paired strict reach at T64',
               ylabel='Retention of T64-correct set to T256', title=f'Size {size}; stars = fixed u300')
        ax.legend(loc='lower right', fontsize=8)
    fig.suptitle('Predeclared training-checkpoint trajectories; unsupported S is not zero')
    fig.tight_layout()
    fig.savefig(out / 'reach_retention_trajectory.png', dpi=160)
    plt.close(fig)


def report(out, state, records, final_rows):
    lines = ['# Continuous execution-state coverage', '',
             'Execution: '+state['status'], '',
             'Primary: '+state.get('primary_verdict', 'PENDING'), '',
             'Fixed u300 is the formal endpoint; intermediate checkpoints are formation diagnostics.',
             'K8, same batch,256 forward steps,32 losses/32 backward windows and one AdamW step per update.', '',
             '| Arm | Completed trajectories | Joint readiness at u300 | Old Full at u300 |',
             '|---|---:|---:|---:|']
    for arm in ARMS:
        done = [r for r in final_rows if r['arm'] == arm]
        lines.append(f"| {arm} | {len(done)}/{BLOCKS} | {sum(r['joint']['pass'] for r in done)} | {sum(bool(r['metrics']['full_pass']) for r in done)} |")
    if 'primary' in state:
        p = state['primary']
        lines += ['', f"Joint wins={p['wins']}, losses={p['losses']}, net={p['net_gain']}/8; exact two-sided p={p['exact_two_sided_p']}."]
    lines += ['', 'Start with summary.json and metrics.csv; dense.json contains all frozen checkpoint points.',
              'Checkpoints and Boolean packed traces are secondary. No peak checkpoint or map subset replaces u300.',
              'Encoder-credit frequency differs as a consequence of reset; no unique visitation mechanism follows.']
    if records:
        lines += ['', f"Dense checkpoint records completed: {len(records)}/{BLOCKS*len(ARMS)*len(CHECKPOINTS)}."]
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_csv(out / 'metrics.csv', records)


def check(out):
    assert not out.exists(), 'Qualification output already exists'
    C.setup_backend()
    before = source_hashes()
    tests = module('_continuous_coverage_checks', 'checks.py')
    started = time.monotonic()
    checked = tests.qualify(C, R, E, M)
    assert source_hashes() == before
    out.parent.mkdir(parents=True, exist_ok=True)
    C.write(out, {**checked, 'protocol': PROTOCOL, 'checked_utc': C.now(),
                  'qualification_elapsed_seconds': time.monotonic()-started,
                  'source_sha256': before, 'runtime_limit_enforced': False})
    print(json.dumps({'status': checked['status'], 'qualification': out.relative_to(ROOT).as_posix(),
                      'check_seconds': time.monotonic()-started}), flush=True)
    if checked['status'] != 'PASS':
        raise SystemExit(1)


def run(out, qualification):
    assert not out.exists(), 'Run output already exists'
    q = read(qualification)
    hashes = source_hashes()
    assert q['status'] == 'PASS' and q['protocol'] == PROTOCOL
    assert q['source_sha256'] == hashes, 'Qualified sources changed'
    C.setup_backend()
    out.mkdir(parents=True)
    started = time.monotonic()
    records, final_rows = [], []
    plans = {}
    for block, (seed, schedule) in enumerate(zip(INIT_SEEDS, SCHEDULE_SEEDS)):
        order = list(ARMS if block % 2 == 0 else ARMS[::-1])
        initial_hash = C.tensor_hash(model_for(seed, 'cpu').state_dict())
        plans[str(block)] = {'initialization_seed': seed, 'schedule_seed': schedule,
                             'initial_parameter_sha256': initial_hash, 'arm_order': order,
                             'batch_indices': np.random.default_rng(schedule).integers(0, 512, (UPDATES, 8)).tolist()}
    C.write(out / 'plans.json', plans)
    plan_hash = C.sha(out / 'plans.json')
    train_cpu = C.region_bank(32, 512, 10002)
    train = {key: value.cuda() for key, value in train_cpu.items()}
    evaluation = {size: C.region_bank(size, 32, seed) for size, seed in EVAL_SEEDS.items()}
    data_hashes = {'train': C.tensor_hash(train_cpu), **{f'evaluation{size}': C.tensor_hash(data) for size, data in evaluation.items()}}
    (out / 'banks').mkdir()
    for label, bank in {'train': train_cpu, **{f'evaluation{size}': bank for size, bank in evaluation.items()}}.items():
        np.savez_compressed(out / 'banks' / f'{label}.npz', **{key: value.cpu().numpy() for key, value in bank.items()})
    C.write(out / 'config.json', {'protocol': PROTOCOL, 'architecture': model_for(96199, 'cpu').metadata(),
                                'arms': ARMS, 'blocks': BLOCKS, 'super_updates': UPDATES,
                                'same_minibatch_repeated_in_reset': True, 'batch': 8,
                                'training_forward_steps_per_update': 256, 'credit_horizon': 8,
                                'loss_count': 32, 'loss_normalization': 32,
                                'parameters_fixed_within_super_update': True, 'optimizer_steps_per_super_update': 1,
                                'train_maps': 512, 'train_seed': 10002, 'eval_seeds': EVAL_SEEDS,
                                'evaluation_maps_each_size': 32, 'checkpoints': CHECKPOINTS,
                                'formal_checkpoint': 300, 'optimizer': C.OPTIMIZER, 'clip': 1.,
                                'runtime_limit_enforced': False})
    manifest = {'protocol': PROTOCOL, 'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'),
                'started_utc': C.now(), 'command': [sys.executable, *sys.argv], 'gpu': torch.cuda.get_device_name(),
                'torch': str(torch.__version__), 'numpy': str(np.__version__),
                'source_sha256': hashes, 'schedule_plan_sha256': plan_hash, 'data_sha256': data_hashes,
                'qualification_sha256': C.sha(qualification), 'expected_arms': 16,
                'expected_dense_records': BLOCKS*len(ARMS)*len(CHECKPOINTS),
                'maximum_seconds': None, 'runtime_limit_enforced': False,
                'watchdog_enabled': False, 'continuous_monitoring': False,
                'git_review_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'backend': {'cudnn_benchmark': False, 'cudnn_deterministic': False,
                            'cudnn_tf32': True, 'matmul_tf32': False}}
    C.write(out / 'manifest.json', manifest)
    for name in hashes:
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)

    def progress(value):
        C.write(out / 'status.json', {'status': 'RUNNING', 'pid': os.getpid(),
                                    'updated_utc': C.now(), 'elapsed_seconds': time.monotonic()-started,
                                    'completed_arms': len(final_rows), 'expected_arms': 16,
                                    'completed_dense_records': len(records), **value})
        print(json.dumps(value), flush=True)

    result = None
    try:
        for block_text, plan in plans.items():
            block = int(block_text)
            for arm in plan['arm_order']:
                folder = out / f'block{block:02d}' / arm
                (folder / 'checkpoints').mkdir(parents=True)
                model = model_for(plan['initialization_seed'])
                assert C.tensor_hash(model.state_dict()) == plan['initial_parameter_sha256']
                optimizer = C.optimizer_for(model)
                curve = []
                evaluation_seconds = 0.

                def save_stage(update):
                    nonlocal evaluation_seconds
                    parameter_hash = C.tensor_hash(model.state_dict())
                    checkpoint = folder / 'checkpoints' / f'u{update:03d}.pt'
                    payload = C.payload(model, optimizer, plan['initialization_seed'], update)
                    payload.update({'protocol': PROTOCOL, 'arm': arm, 'schedule_seed': plan['schedule_seed']})
                    torch.save(payload, checkpoint)
                    progress({'block': block, 'arm': arm, 'phase': 'evaluation', 'completed_updates': update})
                    tick = time.monotonic()
                    eval_folder = folder / 'evaluation' / f'u{update:03d}'
                    measured = E.evaluate(model, evaluation, eval_folder, full=update == UPDATES)
                    elapsed = time.monotonic()-tick
                    evaluation_seconds += elapsed
                    row = {'block': block, 'arm': arm, 'update': update,
                           'initialization_seed': plan['initialization_seed'], 'schedule_seed': plan['schedule_seed'],
                           'checkpoint': checkpoint.relative_to(out).as_posix(), 'checkpoint_sha256': C.sha(checkpoint),
                           'parameter_sha256': parameter_hash, 'evaluation_seconds': elapsed,
                           'evaluation_summary': (eval_folder / 'summary.json').relative_to(out).as_posix(),
                           'metrics': M.compact_pair_metrics(measured), 'joint': M.joint_readiness(measured),
                           'formal_endpoint': update == UPDATES, 'checkpoint_selection': False}
                    if update == UPDATES:
                        row['training_seconds'] = sum(v['seconds'] for v in curve)
                        row['all_checkpoint_evaluation_seconds'] = evaluation_seconds
                        row['capture_setup_seconds'] = captured.setup_seconds
                        final_rows.append(row)
                        C.write(out / 'perarm.json', final_rows)
                    records.append(row)
                    C.write(out / 'dense.json', records)
                    C.write(folder / 'training_curve.json', curve)
                    report(out, {'status': 'RUNNING'}, records, final_rows)
                    model.train()

                progress({'block': block, 'arm': arm, 'phase': 'starting', 'completed_updates': 0})
                save_stage(0)
                first = C.subset(train, plan['batch_indices'][0])
                captured = R.CapturedSuperK8(model, first, arm, LOSS_FN)
                for update, ids in enumerate(plan['batch_indices'], 1):
                    tick = time.monotonic()
                    data = first if update == 1 else C.subset(train, ids)
                    loss, state, finite = captured.run(data)
                    norm = R.finish_update(model, optimizer, finite)
                    torch.cuda.synchronize()
                    curve.append({'update': update, 'mean_super_update_loss': float(loss),
                                  'gradient_norm_before_clip': float(norm), 'seconds': time.monotonic()-tick,
                                  'cold_initializations': 4 if arm == ARMS[0] else 1, **CADENCE})
                    if update == 1 or update % 25 == 0:
                        progress({'block': block, 'arm': arm, 'phase': 'training', 'completed_updates': update,
                                  'loss': float(loss), 'gradient_norm_before_clip': float(norm)})
                    if update in CHECKPOINTS:
                        assert all(int(value['step'].item()) == update for value in optimizer.state.values())
                        save_stage(update)
                    if update != 1:
                        del data
                assert len(curve) == UPDATES
                progress({'block': block, 'arm': arm, 'phase': 'arm_complete', 'completed_updates': UPDATES})
                del model, optimizer, captured, state, first, curve
                gc.collect()
                torch.cuda.empty_cache()
        assert len(final_rows) == BLOCKS*len(ARMS)
        assert len(records) == BLOCKS*len(ARMS)*len(CHECKPOINTS)
        assert source_hashes() == hashes and C.sha(out / 'plans.json') == plan_hash
        assert C.tensor_hash(train) == data_hashes['train']
        assert all(C.tensor_hash(bank) == data_hashes[f'evaluation{size}'] for size, bank in evaluation.items())
        result = M.aggregate_pairs(final_rows, expected_blocks=BLOCKS)
        result.update({'protocol': PROTOCOL, 'status': 'COMPLETE', 'completed_arms': len(final_rows),
                       'expected_arms': BLOCKS*len(ARMS), 'dense_records': len(records)})
        plot(out, records)
    except Exception as error:
        C.write(out / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        result = {'protocol': PROTOCOL, 'status': 'ERROR', 'completed_arms': len(final_rows),
                  'expected_arms': 16, 'dense_records': len(records),
                  'error': repr(error), 'scientific_verdict': 'INCOMPLETE'}
    result.update({'finished_utc': C.now(), 'elapsed_seconds': time.monotonic()-started,
                   'runtime_limit_enforced': False})
    C.write(out / 'summary.json', result)
    C.write(out / 'status.json', {**result, 'pid': os.getpid()})
    report(out, result, records, final_rows)
    print(json.dumps(result), flush=True)
    if result['status'] == 'ERROR':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--qualification')
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    assert out.is_relative_to(ROOT / ('analyses' if args.check else 'runs'))
    if args.check:
        check(out)
    else:
        assert args.qualification
        path = (ROOT / args.qualification).resolve()
        assert path.is_relative_to(ROOT / 'analyses')
        run(out, path)


if __name__ == '__main__':
    main()
