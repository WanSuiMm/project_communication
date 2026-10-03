"""Independent schedule and initialization-ray screens of original Streaming seed4."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch

from initialization import ROOT, initial_model
sys.path.insert(0, str(ROOT/'new/frontier_audit'))
import audit as frontier
from run_revision import now, sha, tensor_hash, write
from tasks import bank, subset
from training import backward_trajectory
from phenotype import evaluate

CAP = 2400
UPDATES = 300
DEADLINE = float('inf')
SCHEDULE_SEEDS = (20012, 20022, 20032, 20042)
DIRECTION_SEEDS = (70002, 70003, 70004, 70005)


def budget():
    if time.monotonic() >= DEADLINE:
        raise TimeoutError('Frozen training-screen time budget exhausted')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sources():
    prior = read(ROOT/'FRONTIER_AUDIT_PUBLICATION_MANIFEST.json')['source_sha256']
    assert len(prior) == 45
    for name, expected in prior.items():
        assert sha(ROOT/name) == expected, f'Frozen source drift: {name}'
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT/'new/seed4_followup').glob('*.py')]
    names += ['new/seed4_followup/PROTOCOL.md', 'tools/launch_seed4_followup.ps1',
              'FRONTIER_AUDIT_PUBLICATION_MANIFEST.json']
    return {**prior, **{name: sha(ROOT/name) for name in sorted(names)}}


def arms(preflight=False):
    control = {'name': 'control', 'experiment': 'control', 'epsilon': 0.,
               'direction_seed': None, 'schedule_seed': 20002}
    result = [control]
    for i, (schedule, direction) in enumerate(zip(SCHEDULE_SEEDS, DIRECTION_SEEDS)):
        result.append({'name': f'A_schedule{schedule}', 'experiment': 'A',
                       'epsilon': 0., 'direction_seed': None, 'schedule_seed': schedule})
        radii = (.01, .05) if i % 2 == 0 else (.05, .01)
        for eps in radii:
            result.append({'name': f'B_direction{direction}_eps{int(eps*100):02}',
                           'experiment': 'B', 'epsilon': eps,
                           'direction_seed': direction, 'schedule_seed': 20002})
    return [result[0], result[1], result[3]] if preflight else result


def control_replay(model, historical_tests, old, record):
    assert record['final_parameter_sha256'] == old['final_parameter_sha256'], 'Control final parameter drift'
    counts = {'integer_leaves': 0, 'float_leaves': 0, 'maximum_absolute_error': 0.}
    measured = {}
    for size, data in historical_tests.items():
        frontier.phase2.DEADLINE = DEADLINE
        measured[str(size)] = frontier.phase2.evaluate(model, data, 8, False)
    frontier.replay_module.compare_tree(old['evaluation'], measured, 'control.evaluation', counts)
    return {'status': 'PASS', 'size_horizon_records': 6, 'counts': counts,
            'final_parameter_sha256': record['final_parameter_sha256'], 'evaluation': measured}


def train_one(out, spec, rows, train, fresh_cpu, historical_tests, old, preflight):
    budget()
    model, noise = initial_model(spec['epsilon'], spec['direction_seed'])
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    name = spec['name']
    record = {**spec, 'status': 'TRAINING', 'completed_updates': 0, 'gradient_horizon': 8,
              'parameter_count': sum(p.numel() for p in model.parameters()), 'noise': noise,
              'architecture': model.metadata(),
              'initial_parameter_sha256': tensor_hash(model.state_dict()),
              'train_data_sha256': tensor_hash(train),
              'schedule_sha256': sha(out/f'schedule{spec["schedule_seed"]}.json'),
              'training_curve': [], 'update_seconds': []}
    assert record['parameter_count'] == 5033
    assert record['train_data_sha256'] == old['train_data_sha256']
    if spec['experiment'] != 'B':
        assert record['initial_parameter_sha256'] == old['initial_parameter_sha256']
    if spec['schedule_seed'] == 20002:
        assert record['schedule_sha256'] == old['schedule_sha256']
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    record['allocated_before_training_bytes'] = torch.cuda.memory_allocated()
    started = time.monotonic()
    clipped = 0
    try:
        for update, indices in enumerate(rows, 1):
            budget()
            batch = subset(train, indices)
            torch.cuda.synchronize()
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss, state, cadence = backward_trajectory(model, batch, 8, budget=budget)
            assert all(bool(torch.isfinite(v).all()) for v in state), 'Nonfinite training state'
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert bool(torch.isfinite(norm)), 'Nonfinite gradient'
            if preflight:
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                group_gradients = {key: float(getattr(model, key).weight.grad.norm()) for key in
                                   ('encoder', 'f_in', 'f_out', 'q_in', 'q_out', 'readout')}
                # Zero output layers delay entry into earlier layers; qualify by update3.
                if update == 3:
                    assert all(value > 0 for value in group_gradients.values()), 'Gradient entry failed'
                    record['gradient_entry_passed'] = True
            clipped += int(norm > 1.)
            optimizer.step()
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite updated parameter'
            torch.cuda.synchronize()
            record['update_seconds'].append(time.monotonic()-tick)
            record['completed_updates'] = update
            if update == 1 or update % 100 == 0 or update == len(rows) or preflight:
                log = {'update': update, 'mean_trajectory_loss': float(loss),
                       'gradient_norm_before_clip': float(norm),
                       'update_seconds': record['update_seconds'][-1], **cadence}
                if preflight:
                    log['gradient_group_norms'] = group_gradients
                record['training_curve'].append(log)
                write(out/f'{name}.json', record)
                write(out/'status.json', {'status': 'RUNNING', 'phase': 'training', 'arm': name,
                      'pid': os.getpid(), 'completed_updates': update, 'updated_utc': now()})
                print(json.dumps({'arm': name, **log}), flush=True)
        record.update({'training_seconds': time.monotonic()-started,
                       'training_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'gradient_clip_fraction': clipped/len(rows),
                       'final_parameter_sha256': tensor_hash(model.state_dict()), 'status': 'TRAINED'})
        torch.save({'variant': 'stream', **spec, 'completed_updates': len(rows),
                    'state_dict': {n: v.detach().cpu() for n, v in model.state_dict().items()}}, out/f'{name}.pt')
        record['checkpoint_file_sha256'] = sha(out/f'{name}.pt')
        write(out/f'{name}.json', record)
        if preflight and spec['experiment'] == 'control':
            tick = time.monotonic()
            evaluate(model, fresh_cpu, out, name, budget)
            record['smoke_endpoint_seconds'] = time.monotonic()-tick
            record['smoke_endpoint_steps'] = 256
            record['smoke_phenotype_is_scientific_endpoint'] = False
        if not preflight:
            model.eval()
            if spec['experiment'] == 'control':
                record['historical_replay'] = control_replay(model, historical_tests, old, record)
                write(out/f'{name}.json', record)
            write(out/'status.json', {'status': 'RUNNING', 'phase': 'fresh_evaluation', 'arm': name,
                  'pid': os.getpid(), 'updated_utc': now()})
            summary = evaluate(model, fresh_cpu, out, name, budget)
            record['phenotype_gate'] = summary['phenotype_gate']
            record['phenotype_summary_sha256'] = sha(out/f'{name}_summary.json')
        record['peak_allocated_bytes_including_evaluation'] = torch.cuda.max_memory_allocated()
        record['status'] = 'COMPLETE'
    except Exception as error:
        record['status'] = 'TIME_BUDGET' if isinstance(error, TimeoutError) else 'ERROR'
        record['error'], record['traceback'] = repr(error), traceback.format_exc()
        if not (out/f'{name}.pt').exists():
            torch.save({'partial': True, **spec, 'completed_updates': record['completed_updates'],
                        'state_dict': {n: v.detach().cpu() for n, v in model.state_dict().items()}}, out/f'{name}.pt')
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        write(out/f'{name}.json', record)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return record


def aggregate(out, records, preflight):
    expected = arms(preflight)
    keys = [r['name'] for r in records]
    assert len(keys) == len(set(keys)) and set(keys) <= {r['name'] for r in expected}
    updates = 3 if preflight else UPDATES
    complete = len(keys) == len(expected) and all(r['status'] == 'COMPLETE' and r['completed_updates'] == updates for r in records)
    identity = len({r['train_data_sha256'] for r in records}) <= 1
    identity &= len({r['initial_parameter_sha256'] for r in records if r['experiment'] != 'B'}) <= 1
    identity &= len({r['schedule_sha256'] for r in records if r['experiment'] != 'A'}) <= 1
    assert identity
    result = {'complete': complete, 'identities_verified': identity, 'updates_per_arm': UPDATES,
              'expected_arms': 13, 'completed_arms': sum(r['status'] == 'COMPLETE' for r in records),
              'arm_statuses': [{k: r[k] for k in ('name', 'experiment', 'status', 'completed_updates')} for r in records]}
    if preflight:
        cost = max(float(np.median(r['update_seconds'][1:])) for r in records) if complete else None
        evaluation_cost = records[0]['smoke_endpoint_seconds'] if complete else None
        estimate = 13*UPDATES*cost*1.15+13*evaluation_cost*1.25+120 if complete else None
        memory = complete and max(r['peak_allocated_bytes_including_evaluation'] for r in records) < 3*1024**3
        gradients = complete and all(r.get('gradient_entry_passed', False) for r in records)
        passed = bool(complete and identity and memory and gradients and estimate <= CAP)
        result.update({'decision': 'PREFLIGHT_PASSED' if passed else 'PREFLIGHT_FAILED',
                       'worst_update_seconds': cost, 'estimated_formal_seconds': estimate,
                       'measured_full_endpoint_seconds': evaluation_cost,
                       'memory_gate': memory, 'gradient_entry_gate': gradients})
    else:
        passes = {r['name']: r['phenotype_gate']['pass'] for r in records if r['status'] == 'COMPLETE'}
        a_rows = [r for r in records if r['experiment'] == 'A' and r['status'] == 'COMPLETE']
        a_count = sum(passes[r['name']] for r in a_rows)
        b = {}
        for eps in (.01, .05):
            selected = [r for r in records if r['experiment'] == 'B' and r['epsilon'] == eps and r['status'] == 'COMPLETE']
            count = sum(passes[r['name']] for r in selected)
            b[str(eps)] = {'completed_directions': len(selected), 'passes': count, 'unit': 'paired_direction',
                          'decision': ('SAMPLED_DIRECTIONAL_TOLERANCE' if count >= 3 else 'NOT_QUALIFIED') if len(selected) == 4 else 'INCOMPLETE'}
        result.update({'decision': 'SCREEN_COMPLETE' if complete else 'INCOMPLETE', 'phenotype_pass': passes,
                       'control_fresh_qualification': ('QUALIFIED' if passes.get('control') else 'BASELINE_FRESH_PHENOTYPE_UNQUALIFIED') if 'control' in passes else 'PENDING',
                       'A': {'completed_schedules': len(a_rows), 'passes': a_count, 'unit': 'schedule',
                             'decision': ('CONDITIONAL_SCHEDULE_ROBUSTNESS' if a_count >= 3 else 'NOT_QUALIFIED') if len(a_rows) == 4 else 'INCOMPLETE'},
                       'B': b, 'scope': 'Selected initialization, fixed train bank and recipe; no basin theorem or architecture reliability claim.'})
        lines = ['# Seed4 independent A/B screens', '', f"Execution complete: {complete}. Status: {result['decision']}.",
                 f"Fresh control: {result['control_fresh_qualification']}.", '',
                 '| Arm | Updates | Execution | Fresh phenotype |', '|---|---:|---|---|']
        lines += [f"| {r['name']} | {r['completed_updates']} | {r['status']} | {passes.get(r['name'], 'pending')} |" for r in records]
        lines += ['', f'A: {a_count}/{len(a_rows)} completed schedules passed; target>=3/4.',
                  'B: '+json.dumps(b), '',
                  'Read each *_summary.json for all gates, denominators and censored trajectories.',
                  'B has four sampled directions paired across radii. Maps/events/pixels are not model replicates.',
                  'These screens do not change previous architecture verdicts; no adaptive rescue or monitoring.']
        (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(out/'aggregate.json', result)
    return result


def main():
    global DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--preflight-dir')
    parser.add_argument('--qualification')
    args = parser.parse_args()
    hashes = sources()
    binding = None
    qualification = None
    if args.preflight:
        assert args.qualification, 'Passed CPU qualification required'
        qpath = Path(args.qualification).resolve()
        assert qpath.is_relative_to(ROOT/'analyses')
        q = read(qpath)
        assert q['status'] == 'PASS' and q['source_sha256'] == hashes
        qualification = {'path': qpath.relative_to(ROOT).as_posix(), 'sha256': sha(qpath)}
    else:
        assert args.preflight_dir, 'Passed preflight required'
        pf = Path(args.preflight_dir).resolve()
        assert pf.is_relative_to(ROOT/'runs')
        assert read(pf/'status.json')['status'] == 'PREFLIGHT_PASSED'
        pf_manifest = read(pf/'manifest.json')
        assert pf_manifest['source_sha256'] == hashes, 'Source changed since preflight'
        qualification = pf_manifest['cpu_qualification']
        qpath = (ROOT/qualification['path']).resolve()
        assert qpath.is_relative_to(ROOT/'analyses') and sha(qpath) == qualification['sha256']
        assert read(qpath)['source_sha256'] == hashes and read(qpath)['status'] == 'PASS'
        checked = read(pf/'aggregate.json')
        assert checked['decision'] == 'PREFLIGHT_PASSED' and checked['updates_per_arm'] == UPDATES
        binding = {name: sha(pf/name) for name in ('status.json', 'manifest.json', 'aggregate.json')}
    out = Path(args.out).resolve()
    assert out.is_relative_to(ROOT/'runs')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    DEADLINE = started+(180 if args.preflight else CAP)
    def timeout():
        write(out/'status.json', {'status': 'WATCHDOG_TIMEOUT', 'finished_utc': now(), 'pid': os.getpid()})
        os._exit(124)
    watchdog = threading.Timer(DEADLINE-time.monotonic()+60, timeout)
    watchdog.daemon = True
    watchdog.start()
    records, status = [], 'ERROR'
    try:
        torch.set_num_threads(2)
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        old = read(ROOT/'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json')
        old_manifest = read(ROOT/'runs/streaming_carry_20261002_init2345/manifest.json')
        assert sha(ROOT/'runs/streaming_carry_20261002_init2345/stream_K8_seed4.json') == sha(ROOT/'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json')
        selected = arms(args.preflight)
        schedules = {seed: np.random.default_rng(seed).integers(0, 512, (UPDATES, 8)).tolist() for seed in (20002, *SCHEDULE_SEEDS)}
        assert schedules[20002] == read(ROOT/'runs/streaming_carry_20261002_init2345/schedule.json')
        for seed, rows in schedules.items():
            write(out/f'schedule{seed}.json', rows)
        assert sha(out/'schedule20002.json') == old['schedule_sha256']
        train = bank(32, 512, 10002, 'cuda')
        assert tensor_hash(train) == old['train_data_sha256']
        fresh_cpu = {size: bank(size, 32, 50000+size) for size in (32, 64)}
        historical_cpu = {size: bank(size, 32, 40000+size) for size in (32, 64)}
        assert {str(n): tensor_hash(d) for n, d in historical_cpu.items()} == old_manifest['evaluation_data_sha256']
        historical_tests = {} if args.preflight else {n: {k: v.cuda() for k, v in data.items()} for n, data in historical_cpu.items()}
        manifest = {'protocol': 'seed4_independent_followup_v1', 'training': True, 'preflight': args.preflight,
                    'started_utc': now(), 'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'),
                    'command': [sys.executable, *sys.argv], 'gpu': torch.cuda.get_device_name(0),
                    'torch': torch.__version__, 'numpy': np.__version__, 'threads': 2,
                    'backend': {'cudnn_benchmark': False, 'cudnn_deterministic': False, 'cudnn_tf32': True, 'matmul_tf32': False},
                    'git_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'arms': selected, 'expected_formal_arms': 13, 'updates_per_arm': 3 if args.preflight else UPDATES,
                    'maximum_seconds': 180 if args.preflight else CAP, 'gradient_horizon': 8,
                    'forward_steps': 64, 'loss_times': list(range(8, 65, 8)), 'train_data_seed': 10002,
                    'train_data_sha256': tensor_hash(train), 'schedule_sha256': {str(seed): sha(out/f'schedule{seed}.json') for seed in schedules},
                    'fresh_evaluation_seeds': [50032, 50064], 'maps_per_size': 32,
                    'fresh_evaluation_data_sha256': {str(n): tensor_hash(d) for n, d in fresh_cpu.items()},
                    'historical_evaluation_data_sha256': {str(n): tensor_hash(d) for n, d in historical_cpu.items()},
                    'historical_reference_sha256': sha(ROOT/'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json'),
                    'optimizer': {'type': 'AdamW', 'lr': .001, 'weight_decay': .0001, 'clip_norm': 1., 'batch': 8},
                    'source_sha256': hashes, 'preflight_sha256': binding,
                    'cpu_qualification': qualification, 'continuous_monitoring': False}
        write(out/'manifest.json', manifest)
        for name in hashes:
            dest = out/'source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, dest)
        for spec in selected:
            rows = schedules[spec['schedule_seed']][:3 if args.preflight else UPDATES]
            record = train_one(out, spec, rows, train, fresh_cpu, historical_tests, old, args.preflight)
            records.append(record)
            aggregate(out, records, args.preflight)
            if record['status'] != 'COMPLETE':
                raise RuntimeError(f"Stopped arm {record['name']}: {record.get('error', record['status'])}")
        result = aggregate(out, records, args.preflight)
        budget()
        status = result['decision'] if args.preflight else 'COMPLETE'
    except Exception as error:
        status = 'TIME_BUDGET' if isinstance(error, TimeoutError) or any(r['status'] == 'TIME_BUDGET' for r in records) else 'ERROR'
        write(out/'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        if (out/'aggregate.json').exists():
            partial = read(out/'aggregate.json')
            partial.update({'complete': False, 'decision': 'PREFLIGHT_FAILED' if args.preflight else 'INCOMPLETE',
                            'execution_status': status, 'execution_error': repr(error)})
            write(out/'aggregate.json', partial)
        if not args.preflight:
            (out/'RESULTS.md').write_text('# Seed4 A/B: INCOMPLETE\n\nExecution stopped. Read error.json and partial arm records; no scientific qualification.\n', encoding='utf-8')
    finally:
        watchdog.cancel()
    write(out/'status.json', {'status': status, 'pid': os.getpid(), 'completed_arms': sum(r['status'] == 'COMPLETE' for r in records),
                            'finished_utc': now(), 'elapsed_seconds': time.monotonic()-started})
    print(json.dumps({'status': status, 'elapsed_seconds': time.monotonic()-started}), flush=True)
    if status != ('PREFLIGHT_PASSED' if args.preflight else 'COMPLETE'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
