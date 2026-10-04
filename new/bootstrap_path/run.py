"""Original seed4 recipe with frozen prefix/suffix and late schedule interventions."""
import argparse
import gc
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
for rel in ('new/nca_inertial_wind_tunnel', 'new/workspace_revision',
            'new/short_bptt', 'new/streaming_carry', 'new/seed4_followup'):
    sys.path.insert(0, str(ROOT/rel))
from stream_cells import StreamingCell
from training import backward_trajectory
from tasks import bank, subset
from run_revision import now, sha, tensor_hash, write
from phenotype import evaluate, _load_audit
from schedules import suite, check_suite, base_schedules, ALTERNATES, LENGTHS, UPDATES
from first_step import audit_first_step

PROTOCOL = 'seed4_bootstrap_path_v1'
CHECKPOINTS = (1, 3, 8, 175, 180, 300)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_hashes():
    prior = read(ROOT/'FRONTIER_AUDIT_PUBLICATION_MANIFEST.json')['source_sha256']
    for name, expected in prior.items():
        assert sha(ROOT/name) == expected, f'Frozen source drift: {name}'
    names = ['new/seed4_followup/phenotype.py', 'new/bootstrap_path/PROTOCOL.md',
             'tools/launch_bootstrap_path.ps1', 'FRONTIER_AUDIT_PUBLICATION_MANIFEST.json']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'new/bootstrap_path').glob('*.py')]
    return {**prior, **{name: sha(ROOT/name) for name in sorted(names)}}


def setup_backend():
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'CUDA required for historical replay'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    assert str(torch.__version__).startswith('2.5.1'), 'Historical Torch2.5.1 required'


def initial(seed=4):
    torch.manual_seed(seed)
    return StreamingCell().cuda()


def parameter_groups(model):
    return {name: float(getattr(model, name).weight.grad.norm()) for name in
            ('encoder', 'f_in', 'f_out', 'q_in', 'q_out', 'readout')}


def compact(summary):
    result = {'pass': summary['phenotype_gate']['pass'], 'sizes': {}}
    for size, row in summary['sizes'].items():
        ends = row['endpoints']
        result['sizes'][size] = {
            'strict_mean': {t: ends[t]['strict_16_32']['mean_map_coverage'] for t in ('64', '128', '256')},
            'strict_pooled': {t: ends[t]['strict_16_32']['pooled_coverage']['value'] for t in ('64', '128', '256')},
            'coverage': {t: ends[t]['all_changed']['pooled_coverage']['value'] for t in ('64', '128', '256')},
            'retention64_to256': row['transitions']['all_changed']['64_to_256']['pooled']['retention']['value'],
            'ever_regressed_fraction': row['ever_regressed_over_ever_correct']['value'],
            'frontier_effect': row['frontier']['mean_map_weighted_difference']}
    return result


def checkpoint(out, name, model, optimizer, update):
    path = out/'checkpoints'/f'{name}_u{update:03}.pt'
    path.parent.mkdir(exist_ok=True)
    torch.save({'variant': 'stream', 'initialization_seed': 4, 'completed_updates': update,
                'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(), 'optimizer_reset_at_splice': False}, path)
    return {'path': path.relative_to(out).as_posix(), 'sha256': sha(path),
            'parameter_sha256': tensor_hash(model.state_dict())}


def control_replay(model, record, old, historical_cpu):
    assert record['final_parameter_sha256'] == old['final_parameter_sha256'], 'H final parameters differ'
    audit = _load_audit()
    audit.phase2.DEADLINE = float('inf')
    measured = {str(size): audit.phase2.evaluate(model, {k: v.cuda() for k, v in data.items()}, 8, False)
                for size, data in historical_cpu.items()}
    counts = {'integer_leaves': 0, 'float_leaves': 0, 'maximum_absolute_error': 0.}
    audit.replay_module.compare_tree(old['evaluation'], measured, 'historical.H', counts)
    return {'status': 'PASS', 'size_horizon_records': 6, **counts}


def train_one(out, spec, train, fixed_cpu, old, historical_cpu, preflight):
    name = spec['name']
    model = initial()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    record = {k: v for k, v in spec.items() if k != 'rows'}
    record.update({'status': 'TRAINING', 'completed_updates': 0, 'training_curve': [],
                   'checkpoints': {}, 'optimizer_reset_at_splice': False,
                   'initial_parameter_sha256': tensor_hash(model.state_dict()),
                   'parameter_count': sum(p.numel() for p in model.parameters()),
                   'schedule_sha256': sha(out/'schedules'/f'{name}.json')})
    assert record['initial_parameter_sha256'] == old['initial_parameter_sha256']
    assert record['parameter_count'] == 5033
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    rows = spec['rows'][:3] if preflight else spec['rows']
    try:
        model.train()
        for update, ids in enumerate(rows, 1):
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss, state, cadence = backward_trajectory(model, subset(train, ids), 8)
            assert all(bool(torch.isfinite(v).all()) for v in state), 'Nonfinite state'
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert bool(torch.isfinite(norm)), 'Nonfinite gradient'
            gradients = parameter_groups(model) if preflight else None
            if preflight and update == 3:
                assert all(v > 0 and np.isfinite(v) for v in gradients.values()), 'Gradient entry failed'
                record['gradient_entry_pass'] = True
            optimizer.step()
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite parameter'
            torch.cuda.synchronize()
            log = {'update': update, 'mean_trajectory_loss': float(loss),
                   'gradient_norm_before_clip': float(norm), 'seconds': time.monotonic()-tick, **cadence}
            if preflight:
                log['gradient_groups'] = gradients
            record['training_curve'].append(log)
            record['completed_updates'] = update
            if not preflight and update == 1:
                for seed, base_rows in base_schedules().items():
                    if ids == base_rows[0].tolist():
                        measured_first = read(out/'first_step'/f'init4_schedule{seed}.json')
                        assert tensor_hash(model.state_dict()) == measured_first['after_parameter_sha256'], 'First-step audit/training delta drift'
                        record['actual_first_step_replay'] = {'status': 'PASS', 'schedule_seed': seed}
            if not preflight and update in CHECKPOINTS:
                record['checkpoints'][str(update)] = checkpoint(out, name, model, optimizer, update)
            if preflight or update == 1 or update % 25 == 0 or update in CHECKPOINTS:
                write(out/f'{name}.json', record)
                write(out/'status.json', {'status': 'RUNNING', 'phase': 'training', 'arm': name,
                      'pid': os.getpid(), 'completed_updates': update, 'updated_utc': now()})
                print(json.dumps({'arm': name, **log}), flush=True)
        record['training_seconds'] = time.monotonic()-started
        record['final_parameter_sha256'] = tensor_hash(model.state_dict())
        if not preflight and spec['kind'] == 'historical_control':
            record['historical_replay'] = control_replay(model, record, old, historical_cpu)
        if not preflight and spec['kind'] == 'alternate_control':
            prior = read(ROOT/f'runs/seed4_followup_20261003_training01/A_schedule{spec["alternate_seed"]}.json')
            assert record['final_parameter_sha256'] == prior['final_parameter_sha256'], 'S final parameter drift'
            record['prior_final_parameter_replay'] = 'PASS'
        if not preflight or name == 'H':
            write(out/'status.json', {'status': 'RUNNING', 'phase': 'evaluation', 'arm': name,
                  'pid': os.getpid(), 'completed_updates': len(rows), 'updated_utc': now()})
            tick = time.monotonic()
            summary = evaluate(model, fixed_cpu, out, name, lambda: None)
            record['evaluation_seconds'] = time.monotonic()-tick
            record['phenotype_gate'] = summary['phenotype_gate']
            record['endpoint'] = compact(summary)
            record['summary_sha256'] = sha(out/f'{name}_summary.json')
            if not preflight and spec['kind'] == 'alternate_control':
                previous = read(ROOT/f'runs/seed4_followup_20261003_training01/A_schedule{spec["alternate_seed"]}_summary.json')
                counts = {'integer_leaves': 0, 'float_leaves': 0, 'maximum_absolute_error': 0.}
                _load_audit().replay_module.compare_tree(previous['sizes'], summary['sizes'], 'alternate.sizes', counts)
                record['prior_phenotype_replay'] = {'status': 'PASS', **counts}
        record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['status'] = 'COMPLETE'
    except Exception as error:
        record.update({'status': 'ERROR', 'error': repr(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        write(out/f'{name}.json', record)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return record


def actual_first_steps(out, train):
    destination = out/'first_step'
    destination.mkdir()
    bases = base_schedules()
    results = []
    cases = [(seed, 20002) for seed in (2, 3, 4, 5)] + [(4, s) for s in ALTERNATES]
    for seed, schedule in cases:
        model = initial(seed)
        before = tensor_hash(model.state_dict())
        reference_path = ROOT/f'runs/warmstart_20261003_paired01/baseline_seed{seed}_u000.pt'
        reference = torch.load(reference_path,
                               map_location='cpu', weights_only=False)
        assert before == tensor_hash(reference['state_dict']), f'Initial seed{seed} drift'
        result = audit_first_step(model, subset(train, bases[schedule][0].tolist()),
                                  destination/f'init{seed}_schedule{schedule}.json')
        result.update({'initialization_seed': seed, 'schedule_seed': schedule,
                       'initial_parameter_sha256': before,
                       'initial_checkpoint_file_sha256': sha(reference_path),
                       'after_parameter_sha256': tensor_hash(model.state_dict())})
        write(destination/f'init{seed}_schedule{schedule}.json', result)
        results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write(destination/'summary.json', {'cases': results, 'optimizer_updates_per_case': 1,
          'claim_boundary': 'Actual first writes; not a mediator intervention or success predictor.'})


def aggregate(out, records, preflight=False):
    if preflight:
        complete = len(records) == 2 and all(r['status'] == 'COMPLETE' for r in records)
        passed = complete and all(r['gradient_entry_pass'] for r in records)
        memory = complete and max(r['peak_allocated_bytes'] for r in records) < 3*1024**3
        cost = max(float(np.median([v['seconds'] for v in r['training_curve'][1:]])) for r in records)
        estimate = 23*(UPDATES*cost*1.15+records[0]['evaluation_seconds']*1.25)+60
        result = {'pass': passed and memory, 'memory_pass': memory,
                  'estimated_formal_seconds': estimate, 'runtime_limit_enforced': False,
                  'expected_arms': 23, 'preflight_arms': 2, 'updates_per_arm': 3,
                  'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in records)}
    else:
        by_name = {r['name']: r for r in records}
        complete = len(records) == 23 and all(r['status'] == 'COMPLETE' for r in records)
        contrast = []
        for r in records:
            if 'endpoint' not in r:
                continue
            ep = r['endpoint']
            row = {k: r[k] for k in ('name', 'kind', 'alternate_seed', 'k')}
            row.update({'endpoint': ep, 'differences': {}})
            for ref in ('H', f'S{r["alternate_seed"]}' if r['alternate_seed'] else 'H'):
                if ref in by_name and 'endpoint' in by_name[ref]:
                    a = ep['sizes']['32']; b = by_name[ref]['endpoint']['sizes']['32']
                    row['differences'][ref] = {
                        'T64_strict_mean': a['strict_mean']['64']-b['strict_mean']['64'],
                        'T256_strict_mean': a['strict_mean']['256']-b['strict_mean']['256'],
                        'retention64_to256': a['retention64_to256']-b['retention64_to256']}
            contrast.append(row)
        signatures = {}
        for k in LENGTHS:
            names = [f'S{s}_{kind}{k}' for s in ALTERNATES
                     for kind in ('replace_early', 'preserve_early', 'replace_late')]
            observed = all(n in by_name and 'endpoint' in by_name[n] for n in names)
            signatures[str(k)] = {'complete': observed,
                'strong_conditional_prefix_pattern':
                all(not by_name[f'S{s}_replace_early{k}']['endpoint']['pass'] and
                    by_name[f'S{s}_preserve_early{k}']['endpoint']['pass'] and
                    by_name[f'S{s}_replace_late{k}']['endpoint']['pass'] for s in ALTERNATES)
                if observed else None}
        result = {'complete': complete, 'decision': 'SCREEN_COMPLETE' if complete else 'INCOMPLETE',
                  'expected_arms': 23, 'completed_arms': len(records), 'contrasts': contrast,
                  'prefix_signatures': signatures, 'independent_unit': 'two deliberately selected alternate schedules',
                  'historical_control_qualified': bool(by_name.get('H', {}).get('endpoint', {}).get('pass')),
                  'scope': 'One selected initialization, fixed reused test maps; no first-write mediation or architecture reliability claim.'}
        lines = ['# Seed4 bootstrap-path screen', '', f"Status: {result['decision']}; {len(records)}/23 arms.", '',
                 '| Arm | Full phenotype | size32 T64 strict mean | size32 T256 strict mean |',
                 '|---|---|---:|---:|']
        for row in contrast:
            e = row['endpoint']; s = e['sizes']['32']
            lines.append(f"| {row['name']} | {e['pass']} | {s['strict_mean']['64']:.6f} | {s['strict_mean']['256']:.6f} |")
        lines += ['', 'Prefix signatures: '+json.dumps(signatures), '',
                  'Read aggregate.json and first_step/summary.json, then per-arm summaries for all denominators.',
                  'Two alternate schedules are selected conditional units; maps and prefix lengths are not training replicates.',
                  'No rank-one bootstrap mechanism follows from a schedule effect alone.']
        (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(out/'aggregate.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--qualification')
    parser.add_argument('--preflight-dir')
    args = parser.parse_args()
    hashes = source_hashes()
    binding = None
    if args.preflight:
        qpath = (ROOT/args.qualification).resolve()
        assert qpath.is_relative_to(ROOT/'analyses')
        q = read(qpath)
        assert q['status'] == 'PASS' and q['source_sha256'] == hashes
        qualification = {'path': qpath.relative_to(ROOT).as_posix(), 'sha256': sha(qpath)}
    else:
        pf = (ROOT/args.preflight_dir).resolve()
        assert pf.is_relative_to(ROOT/'runs')
        pm = read(pf/'manifest.json')
        assert read(pf/'status.json')['status'] == 'PREFLIGHT_PASSED'
        assert pm['source_sha256'] == hashes and read(pf/'aggregate.json')['pass']
        qualification = pm['cpu_qualification']
        assert sha(ROOT/qualification['path']) == qualification['sha256']
        binding = {n: sha(pf/n) for n in ('manifest.json', 'aggregate.json', 'status.json')}
    out = (ROOT/args.out).resolve()
    assert out.is_relative_to(ROOT/'runs')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = []
    status = 'ERROR'
    try:
        setup_backend()
        check_suite()
        old = read(ROOT/'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json')
        specs = suite()
        if args.preflight:
            specs = [specs[0], next(s for s in specs if s['name'] == 'S20012_replace_early3')]
        schedule_directory = out/'schedules'; schedule_directory.mkdir()
        for spec in suite():
            write(schedule_directory/f'{spec["name"]}.json', spec['rows'])
        assert read(schedule_directory/'H.json') == read(ROOT/'runs/streaming_carry_20261002_init2345/schedule.json')
        assert sha(schedule_directory/'H.json') == old['schedule_sha256']
        train = bank(32, 512, 10002, 'cuda')
        assert tensor_hash(train) == old['train_data_sha256']
        fixed_cpu = {n: bank(n, 32, 50000+n) for n in (32, 64)}
        historical_cpu = {n: bank(n, 32, 40000+n) for n in (32, 64)}
        old_manifest = read(ROOT/'runs/seed4_followup_20261003_training01/manifest.json')
        assert {str(n): tensor_hash(d) for n, d in fixed_cpu.items()} == old_manifest['fresh_evaluation_data_sha256']
        assert {str(n): tensor_hash(d) for n, d in historical_cpu.items()} == old_manifest['historical_evaluation_data_sha256']
        references = ['evidence/streaming_carry_init2345/raw/stream_K8_seed4.json',
                      'runs/streaming_carry_20261002_init2345/schedule.json',
                      'runs/seed4_followup_20261003_training01/manifest.json']
        references += [f'runs/warmstart_20261003_paired01/baseline_seed{s}_u000.pt' for s in (2, 3, 4, 5)]
        references += [f'runs/seed4_followup_20261003_training01/A_schedule{s}{suffix}.json'
                       for s in ALTERNATES for suffix in ('', '_summary')]
        manifest = {'protocol': PROTOCOL, 'preflight': args.preflight, 'training': True,
            'started_utc': now(), 'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'),
            'command': [sys.executable, *sys.argv], 'gpu': torch.cuda.get_device_name(),
            'torch': torch.__version__, 'numpy': np.__version__, 'seed': 4,
            'git_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': hashes, 'cpu_qualification': qualification, 'preflight_sha256': binding,
            'reference_sha256': {name: sha(ROOT/name) for name in references},
            'arms': [{k: v for k, v in s.items() if k != 'rows'} for s in specs],
            'expected_formal_arms': 23, 'updates_per_arm': 3 if args.preflight else 300,
            'schedule_sha256': {s['name']: sha(schedule_directory/f'{s["name"]}.json') for s in suite()},
            'train_data_sha256': tensor_hash(train),
            'fixed_evaluation_data_sha256': {str(n): tensor_hash(d) for n, d in fixed_cpu.items()},
            'historical_evaluation_data_sha256': {str(n): tensor_hash(d) for n, d in historical_cpu.items()},
            'runtime_limit_enforced': False, 'maximum_seconds': None, 'watchdog_enabled': False,
            'continuous_monitoring': False, 'threads': 2,
            'backend': {'cudnn_benchmark': False, 'cudnn_deterministic': False, 'cudnn_tf32': True, 'matmul_tf32': False},
            'optimizer': {'type': 'AdamW', 'lr': .001, 'weight_decay': .0001, 'betas': [.9, .999], 'eps': 1e-8, 'clip_norm': 1.},
            'gradient_horizon': 8, 'forward_steps': 64, 'loss_times': list(range(8, 65, 8)),
            'optimizer_reset_at_splice': False, 'test_maps_reused': True}
        write(out/'manifest.json', manifest)
        for name in hashes:
            dest = out/'source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, dest)
        if not args.preflight:
            actual_first_steps(out, train)
        for spec in specs:
            record = train_one(out, spec, train, fixed_cpu, old, historical_cpu, args.preflight)
            records.append(record)
            if not args.preflight:
                aggregate(out, records)
                if spec['name'] == 'H' and not record['endpoint']['pass']:
                    status = 'BASELINE_UNQUALIFIED'
                    break
        assert all(sha(ROOT/name) == expected for name, expected in manifest['reference_sha256'].items()), 'Reference drift during run'
        if status != 'BASELINE_UNQUALIFIED':
            result = aggregate(out, records, args.preflight)
            status = ('PREFLIGHT_PASSED' if result['pass'] else 'PREFLIGHT_FAILED') if args.preflight else 'COMPLETE'
        else:
            result = read(out/'aggregate.json'); result['decision'] = status
            write(out/'aggregate.json', result)
            with (out/'RESULTS.md').open('a', encoding='utf-8') as handle:
                handle.write('\nBASELINE_UNQUALIFIED: H did not qualify; remaining efficacy arms were not run.\n')
    except Exception as error:
        status = 'ERROR'
        write(out/'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        (out/'RESULTS.md').write_text('# Bootstrap-path screen: ERROR\n\nExecution stopped; read error.json. No qualified comparison.\n', encoding='utf-8')
    write(out/'status.json', {'status': status, 'pid': os.getpid(), 'completed_arms': len(records),
          'expected_arms': 2 if args.preflight else 23, 'elapsed_seconds': time.monotonic()-started,
          'finished_utc': now()})
    print(json.dumps({'status': status, 'elapsed_seconds': time.monotonic()-started}), flush=True)
    if status not in ('PREFLIGHT_PASSED', 'COMPLETE'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
