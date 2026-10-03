"""Paired original StreamingCell: fresh K8 versus detached on-policy warm-start."""
import argparse
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
for name in reversed(('nca_inertial_wind_tunnel', 'workspace_revision', 'short_bptt',
                      'streaming_carry', 'frontier_audit', 'seed4_followup')):
    sys.path.insert(0, str(ROOT/'new'/name))
from tasks import bank, subset
from run_revision import now, sha, tensor_hash, write
from stream_cells import StreamingCell
import audit as frontier
from phenotype import evaluate

SEEDS = (2, 3, 4, 5)
UPDATES = 300
PROTOCOL_ID = 'detached_warmstart_v1_nocap'
DEADLINE = float('inf')


def local_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'new/warmstart'/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def budget():
    """Compatibility callback: the user removed the runtime limit before launch."""
    return None


def sources():
    prior = read(ROOT/'SEED4_FOLLOWUP_PUBLICATION_MANIFEST.json')['source_sha256']
    assert len(prior) == 56
    for name, expected in prior.items():
        assert sha(ROOT/name) == expected, f'Historical source drift: {name}'
    public = read(ROOT/'STREAMING_CARRY_PUBLICATION_MANIFEST.json')
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT/'new/warmstart').glob('*.py')]
    names += ['new/warmstart/PROTOCOL.md', 'tools/launch_warmstart.ps1',
              'SEED4_FOLLOWUP_PUBLICATION_MANIFEST.json',
              'evidence/streaming_carry_init2345/manifest.json',
              'evidence/streaming_carry_init2345/schedule.json']
    names += [f'evidence/streaming_carry_init2345/raw/stream_K8_seed{s}.json' for s in SEEDS]
    for name in names:
        if name.startswith('evidence/streaming_carry_init2345/'):
            assert sha(ROOT/name) == public['published_evidence_sha256'][name], name
    return {**prior, **{name: sha(ROOT/name) for name in sorted(set(names))}}


def arms(preflight=False):
    if preflight:
        return [{'seed': 2, 'variant': v, 'name': f'{v}_seed2'} for v in ('baseline', 'warmstart')]
    result = []
    for i, seed in enumerate(SEEDS):
        variants = ('baseline', 'warmstart') if i % 2 == 0 else ('warmstart', 'baseline')
        result += [{'seed': seed, 'variant': v, 'name': f'{v}_seed{seed}'} for v in variants]
    return result


def historical(seed):
    return read(ROOT/f'evidence/streaming_carry_init2345/raw/stream_K8_seed{seed}.json')


def save_stage(model, out, name, update, record, diagnostic_data, diagnostic, preflight):
    budget()
    state = {n: v.detach().cpu() for n, v in model.state_dict().items()}
    parameter_hash = tensor_hash(state)
    target = out/f'{name}_u{update:03}.pt'
    torch.save({'variant': record['variant'], 'seed': record['seed'],
                'completed_updates': update, 'state_dict': state}, target)
    record['checkpoints'][str(update)] = {'file': target.name, 'sha256': sha(target),
                                         'parameter_sha256': parameter_hash}
    if preflight and update == 0:
        return
    torch.cuda.synchronize()
    started = time.monotonic()
    value = diagnostic.diagnose(model, diagnostic_data, budget=budget)
    torch.cuda.synchronize()
    value.update({'training_update': update, 'arm': name, 'seed': record['seed'],
                  'checkpoint_parameter_sha256': parameter_hash,
                  'scientific_endpoint': False, 'exploratory': True,
                  'elapsed_seconds': time.monotonic()-started})
    assert tensor_hash(model.state_dict()) == parameter_hash, 'Diagnostics modified weights'
    write(out/f'{name}_u{update:03}_diagnostic.json', value)
    record['diagnostics'][str(update)] = {'file': f'{name}_u{update:03}_diagnostic.json',
                                        'elapsed_seconds': value['elapsed_seconds']}
    budget()


def replay(model, tests, old, record):
    assert record['final_parameter_sha256'] == old['final_parameter_sha256'], 'Historical baseline final SHA drift'
    counts = {'integer_leaves': 0, 'float_leaves': 0, 'maximum_absolute_error': 0.}
    measured = {}
    frontier.phase2.DEADLINE = DEADLINE
    for size, data in tests.items():
        budget()
        measured[str(size)] = frontier.phase2.evaluate(model, data, 8, False)
    frontier.replay_module.compare_tree(old['evaluation'], measured, 'baseline.evaluation', counts)
    return {'status': 'PASS', 'size_horizon_records': 6, 'counts': counts,
            'evaluation': measured, 'final_parameter_sha256': record['final_parameter_sha256']}


def train_one(out, spec, rows, ages, train, fresh_cpu, historical_gpu,
              diagnostic_data, helper, diagnostic, preflight):
    budget()
    torch.manual_seed(spec['seed'])
    model = StreamingCell().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    old = historical(spec['seed'])
    name = spec['name']
    record = {**spec, 'status': 'TRAINING', 'completed_updates': 0,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'architecture': model.metadata(), 'gradient_horizon': 8,
              'initial_parameter_sha256': tensor_hash(model.state_dict()),
              'train_data_sha256': tensor_hash(train), 'schedule_sha256': sha(out/'schedule.json'),
              'age_schedule_sha256': sha(out/'ages.json'), 'training_curve': [],
              'update_seconds': [], 'checkpoints': {}, 'diagnostics': {}}
    assert record['parameter_count'] == 5033
    for key in ('initial_parameter_sha256', 'train_data_sha256', 'schedule_sha256'):
        assert record[key] == old[key], f'Historical identity drift: {key}'
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    record['allocated_before_training_bytes'] = torch.cuda.memory_allocated()
    clipped, started = 0, time.monotonic()
    try:
        save_stage(model, out, name, 0, record, diagnostic_data, diagnostic, preflight)
        for update, (indices, age) in enumerate(zip(rows, ages), 1):
            budget()
            batch = subset(train, indices)
            torch.cuda.synchronize()
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss, state, cadence = helper.backward_warmstart(
                model, batch, age=int(age), keep_warm_state=spec['variant']=='warmstart', budget=budget)
            assert all(bool(torch.isfinite(v).all()) for v in state), 'Nonfinite suffix state'
            assert cadence['backward_calls'] == 8 and cadence['loss_count'] == 8
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert bool(torch.isfinite(norm)), 'Nonfinite gradient'
            if preflight:
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                gradients = {k: float(getattr(model, k).weight.grad.norm()) for k in
                             ('encoder', 'f_in', 'f_out', 'q_in', 'q_out', 'readout')}
                if update == 3:
                    assert all(v > 0 for v in gradients.values()), 'Gradient entry failed'
                    record['gradient_entry_passed'] = True
            clipped += int(norm > 1.)
            optimizer.step()
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite parameter'
            torch.cuda.synchronize()
            record['update_seconds'].append(time.monotonic()-tick)
            record['completed_updates'] = update
            if update == 1 or update % 100 == 0 or update == len(rows) or preflight:
                log = {'update': update, 'prefix_age_macro_steps': int(age),
                       'mean_suffix_loss': float(loss), 'gradient_norm_before_clip': float(norm),
                       'update_seconds': record['update_seconds'][-1], **cadence}
                if preflight:
                    log['gradient_group_norms'] = gradients
                record['training_curve'].append(log)
                write(out/f'{name}.json', record)
                write(out/'status.json', {'status': 'RUNNING', 'phase': 'training', 'arm': name,
                      'pid': os.getpid(), 'completed_updates': update, 'updated_utc': now()})
                print(json.dumps({'arm': name, **log}), flush=True)
            if update % 100 == 0 or (preflight and update == len(rows)):
                save_stage(model, out, name, update, record, diagnostic_data, diagnostic, preflight)
        record.update({'training_seconds': time.monotonic()-started,
                       'training_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'gradient_clip_fraction': clipped/len(rows),
                       'final_parameter_sha256': tensor_hash(model.state_dict()), 'status': 'TRAINED'})
        torch.save({'variant': spec['variant'], 'seed': spec['seed'], 'completed_updates': len(rows),
                    'state_dict': {n: v.detach().cpu() for n, v in model.state_dict().items()}}, out/f'{name}.pt')
        record['checkpoint_file_sha256'] = sha(out/f'{name}.pt')
        model.eval()
        if not preflight and spec['variant'] == 'baseline':
            record['historical_replay'] = replay(model, historical_gpu, old, record)
        write(out/'status.json', {'status': 'RUNNING', 'phase': 'phenotype', 'arm': name,
                                 'pid': os.getpid(), 'updated_utc': now()})
        tick = time.monotonic()
        summary = evaluate(model, fresh_cpu, out, name, budget)
        record['phenotype_seconds'] = time.monotonic()-tick
        record['phenotype_gate'] = summary['phenotype_gate']
        record['phenotype_summary_sha256'] = sha(out/f'{name}_summary.json')
        record['phenotype_is_scientific_endpoint'] = not preflight
        record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['status'] = 'COMPLETE'
    except Exception as error:
        record.update({'status': 'ERROR',
                       'error': repr(error), 'traceback': traceback.format_exc()})
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
    complete = len(records) == (2 if preflight else 8) and all(
        r['status']=='COMPLETE' and r['completed_updates']==(3 if preflight else UPDATES) for r in records)
    identities = True
    for seed in SEEDS:
        pairs = [r for r in records if r['seed']==seed]
        identities &= len({r['initial_parameter_sha256'] for r in pairs}) <= 1
        identities &= len({r['schedule_sha256'] for r in pairs}) <= 1
        identities &= len({r['age_schedule_sha256'] for r in pairs}) <= 1
    result = {'complete': complete, 'completed_arms': sum(r['status']=='COMPLETE' for r in records),
              'identities_verified': identities, 'updates_per_arm': UPDATES,
              'expected_arms': 8, 'independent_unit': 'paired initialization', 'n': 4}
    if preflight:
        speeds = [v for r in records for v in r['update_seconds'][1:]]
        endpoint = max([r.get('phenotype_seconds', 0) for r in records], default=0)
        diag = max([d['elapsed_seconds'] for r in records for d in r['diagnostics'].values()], default=0)
        # All timed updates use the worst prefix192, not the formal mean104.
        estimate = (max(speeds, default=float('inf'))*UPDATES*8 + endpoint*8 + diag*4*8 + 30)*1.20
        memory = all(r.get('peak_allocated_bytes', 2**63)-r['allocated_before_training_bytes'] < 3500*1024**2 for r in records)
        gradient = all(r.get('gradient_entry_passed') for r in records)
        passed = complete and identities and gradient and memory
        result.update({'decision': 'PREFLIGHT_PASSED' if passed else 'PREFLIGHT_FAILED',
                       'estimated_formal_seconds': estimate, 'gradient_entry_gate': gradient,
                       'memory_gate': memory, 'maximum_prefix_age': 192,
                       'runtime_limit_enforced': False, 'maximum_seconds': None,
                       'time_estimate_is_launch_gate': False,
                       'endpoint_seconds': endpoint, 'diagnostic_seconds': diag})
    else:
        passes = {r['name']: r.get('phenotype_gate', {}).get('pass', False) for r in records}
        counts = {v: sum(passes.get(f'{v}_seed{s}', False) for s in SEEDS) for v in ('baseline', 'warmstart')}
        controls = len([r for r in records if r['variant']=='baseline' and r.get('historical_replay', {}).get('status')=='PASS']) == 4
        qualified = controls and passes.get('baseline_seed4', False)
        positive = qualified and counts['warmstart'] >= 3 and counts['warmstart']-counts['baseline'] >= 2
        decision = 'INCOMPLETE' if not complete else (
            'BASELINE_FRESH_PHENOTYPE_UNQUALIFIED' if not qualified else
            'DEVELOPMENT_SIGNAL' if positive else 'DEVELOPMENT_NOT_QUALIFIED')
        result.update({'decision': decision, 'historical_controls_qualified': controls,
                       'known_seed4_fresh_qualified': passes.get('baseline_seed4', False),
                       'passes': counts, 'phenotype_pass': passes,
                       'paired_full_pass_change': counts['warmstart']-counts['baseline'],
                       'pairs': {str(s): {v: passes.get(f'{v}_seed{s}') for v in counts} for s in SEEDS},
                       'scope': 'Conditional four initializations, fixed bank/schedule; no phase transition, causal closure or reliability theorem.'})
    write(out/'aggregate.json', result)
    lines = ['# Detached warm-start screen', '', f"Status: {result['decision']}. Complete: {complete}.", '',
             '| Arm | Updates | Execution | Full phenotype |', '|---|---:|---|---|']
    for r in records:
        lines.append(f"| {r['name']} | {r['completed_updates']} | {r['status']} | {r.get('phenotype_gate', {}).get('pass', 'pending')} |")
    lines += ['', 'Primary gates and statistical units are frozen in new/warmstart/PROTOCOL.md.',
              'Smoke phenotype is not scientific. Stage diagnostics are exploratory; partial gains cannot rescue a failed full gate.',
              'Read aggregate.json and individual *_summary.json files first; trajectories/checkpoints/diagnostic JSON are secondary.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--qualification')
    parser.add_argument('--preflight-dir')
    args = parser.parse_args()
    hashes = sources()
    if args.preflight:
        qpath = Path(args.qualification).resolve()
        assert qpath.is_relative_to(ROOT/'analyses')
        q = read(qpath)
        assert q['status']=='PASS' and q['source_sha256']==hashes
        qualification = {'path': qpath.relative_to(ROOT).as_posix(), 'sha256': sha(qpath)}
        binding = None
    else:
        pf = Path(args.preflight_dir).resolve()
        assert pf.is_relative_to(ROOT/'runs')
        pm, pa = read(pf/'manifest.json'), read(pf/'aggregate.json')
        assert read(pf/'status.json')['status']=='PREFLIGHT_PASSED'
        assert pm['protocol']==PROTOCOL_ID and pm['source_sha256']==hashes
        assert pm['maximum_seconds'] is None and pm['runtime_limit_enforced'] is False
        assert pa['decision']=='PREFLIGHT_PASSED' and pa['expected_arms']==8 and pa['updates_per_arm']==300
        assert pa['time_estimate_is_launch_gate'] is False and pa['runtime_limit_enforced'] is False
        qualification = pm['cpu_qualification']
        qpath = ROOT/qualification['path']
        assert sha(qpath)==qualification['sha256'] and read(qpath)['source_sha256']==hashes
        binding = {name: sha(pf/name) for name in ('status.json', 'manifest.json', 'aggregate.json')}
    out = Path(args.out).resolve()
    assert out.is_relative_to(ROOT/'runs')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records, status = [], 'ERROR'
    try:
        torch.set_num_threads(2)
        assert torch.cuda.is_available()
        free, _ = torch.cuda.mem_get_info()
        assert free >= 3500*1024**2, 'Insufficient GPU headroom'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        helper = local_module('warmstart_training', 'training.py')
        diagnostic = local_module('warmstart_diagnostics', 'diagnostics.py')
        selected = arms(args.preflight)
        schedule = np.random.default_rng(20002).integers(0, 512, (UPDATES, 8)).tolist()
        assert schedule == read(ROOT/'evidence/streaming_carry_init2345/schedule.json')
        ages = np.random.default_rng(60002).choice([32, 64, 128, 192], size=UPDATES).tolist()
        write(out/'schedule.json', schedule)
        write(out/'ages.json', {'seed': 60002, 'choices': [32,64,128,192], 'ages': ages})
        train = bank(32, 512, 10002, 'cuda')
        fresh_cpu = {size: bank(size, 32, 60000+size) for size in (32,64)}
        hist_cpu = {size: bank(size, 32, 40000+size) for size in (32,64)}
        old_manifest = read(ROOT/'evidence/streaming_carry_init2345/manifest.json')
        assert {str(n): tensor_hash(d) for n,d in hist_cpu.items()} == old_manifest['evaluation_data_sha256']
        historical_gpu = {} if args.preflight else {n: {k:v.cuda() for k,v in d.items()} for n,d in hist_cpu.items()}
        diag_cpu = bank(32, 16, 61032)
        manifest = {'protocol': PROTOCOL_ID, 'training': True, 'preflight': args.preflight,
                    'started_utc': now(), 'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'),
                    'command': [sys.executable,*sys.argv], 'gpu': torch.cuda.get_device_name(0),
                    'torch': torch.__version__, 'numpy': np.__version__, 'threads': 2,
                    'backend': {'cudnn_benchmark': False, 'cudnn_deterministic': False,
                                'cudnn_tf32': True, 'matmul_tf32': False},
                    'git_base': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    'arms': selected, 'expected_formal_arms': 8, 'updates_per_arm': 3 if args.preflight else UPDATES,
                    'maximum_seconds': None, 'runtime_limit_enforced': False, 'watchdog_enabled': False,
                    'gradient_horizon': 8,
                    'trained_suffix_steps':64, 'loss_times_suffix':list(range(8,65,8)),
                    'warm_examples':4, 'fresh_examples':4, 'prefix_age_choices':[32,64,128,192],
                    'age_schedule_sha256':sha(out/'ages.json'), 'age_unit':'original two-hop macro step',
                    'stage_updates':[0,100,200,300], 'stage_diagnostic_seed':61032, 'stage_diagnostic_maps':16,
                    'stage_diagnostic_data_sha256':tensor_hash(diag_cpu), 'stage_diagnostic_steps':128,
                    'train_data_seed':10002, 'train_data_sha256':tensor_hash(train),
                    'schedule_seed':20002, 'schedule_sha256':sha(out/'schedule.json'),
                    'new_evaluation_seeds':[60032,60064], 'maps_per_size':32,
                    'new_evaluation_data_sha256':{str(n):tensor_hash(d) for n,d in fresh_cpu.items()},
                    'historical_evaluation_data_sha256':{str(n):tensor_hash(d) for n,d in hist_cpu.items()},
                    'optimizer':{'type':'AdamW','lr':.001,'weight_decay':.0001,'clip_norm':1.,'batch':8},
                    'source_sha256':hashes,'preflight_sha256':binding,'cpu_qualification':qualification,
                    'continuous_monitoring':False}
        write(out/'manifest.json',manifest)
        for name in hashes:
            target = out/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        for spec in selected:
            rows = schedule[:3] if args.preflight else schedule
            used_ages = [192]*3 if args.preflight else ages
            record = train_one(out,spec,rows,used_ages,train,fresh_cpu,historical_gpu,diag_cpu,helper,diagnostic,args.preflight)
            records.append(record)
            aggregate(out,records,args.preflight)
            if record['status']!='COMPLETE':
                raise RuntimeError(f"Stopped {record['name']}: {record.get('error',record['status'])}")
        result = aggregate(out,records,args.preflight)
        assert identities_valid(records), 'Paired initialization/data/schedule identities failed'
        assert sources()==hashes, 'Source changed during run'
        budget()
        status = result['decision'] if args.preflight else 'COMPLETE'
    except Exception as error:
        status = 'ERROR'
        write(out/'error.json',{'error':repr(error),'traceback':traceback.format_exc()})
        if (out/'aggregate.json').exists():
            partial = read(out/'aggregate.json')
            partial.update({'complete':False,'decision':'PREFLIGHT_FAILED' if args.preflight else 'INCOMPLETE',
                            'execution_status':status,'execution_error':repr(error)})
            write(out/'aggregate.json',partial)
        (out/'RESULTS.md').write_text(
            '# Detached warm-start screen\n\nExecution: ERROR. Scientific decision: INCOMPLETE.\n\n'
            'Read error.json and partial aggregate.json; no qualified claim is available.\n', encoding='utf-8')
    write(out/'status.json',{'status':status,'pid':os.getpid(),'completed_arms':sum(r['status']=='COMPLETE' for r in records),
                           'finished_utc':now(),'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'status':status,'elapsed_seconds':time.monotonic()-started}),flush=True)
    if status != ('PREFLIGHT_PASSED' if args.preflight else 'COMPLETE'):
        raise SystemExit(1)


def identities_valid(records):
    for seed in SEEDS:
        rows = [r for r in records if r['seed']==seed]
        for key in ('initial_parameter_sha256','train_data_sha256','schedule_sha256','age_schedule_sha256'):
            if len({r[key] for r in rows}) > 1:
                return False
    return True


if __name__ == '__main__':
    main()
