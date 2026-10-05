"""Frozen C8 read/sidecar factorial, repository-root entry point."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'new/trajectory_qualification'))
import common as C
from fresh_recipe import exact_two_sided_binomial_p, wilson_interval


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


M = module('_factorial_cells', Path(__file__).with_name('cells.py'))
R = module('_factorial_runtime', Path(__file__).with_name('runtime.py'))
OLD = module('_factorial_frozen_width_runner', ROOT / 'new/latent_width/run.py')
PROTOCOL = 'latent_read_sidecar_factorial_v1'
ARMS = ('native', 'factorized', 'native_r2', 'factorized_r2')
PARAMETERS = dict(zip(ARMS, (2521, 2649, 3275, 3403)))
BLOCKS = 32
UPDATES = 300
INIT_SEEDS = tuple(range(94001, 94033))
SCHEDULE_SEEDS = tuple(range(95001, 95033))
EVAL_SEEDS = {32: 99332, 64: 99364}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_hashes():
    names = set(OLD.source_hashes())
    names.add('tools/launch_latent_factorial.ps1')
    names.update(p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.iterdir()
                 if p.suffix in ('.py', '.md'))
    return {name: C.sha(ROOT / name) for name in sorted(names)}


def model_for(arm, seed, device='cuda'):
    model = M.make_model(arm, seed).to(device)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS[arm]
    return model


def optimizer_for(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if name.endswith('.U') else decay).append(p)
    groups = [{'params': decay}]
    if no_decay:
        assert len(no_decay) == 2
        groups.append({'params': no_decay, 'weight_decay': 0.})
    return torch.optim.AdamW(groups, **C.OPTIMIZER)


def payload(arm, model, optimizer, seed, update):
    return {**C.payload(model, optimizer, seed, update), 'variant': arm,
            'arm': arm, 'metadata': model.metadata()}


def paired(rows, arm, baseline='native'):
    a = {r['block']: r for r in rows if r['arm'] == baseline}
    b = {r['block']: r for r in rows if r['arm'] == arm}
    assert set(a) == set(b) == set(range(BLOCKS))
    wins = sum(not a[k]['phenotype_pass'] and b[k]['phenotype_pass'] for k in a)
    losses = sum(a[k]['phenotype_pass'] and not b[k]['phenotype_pass'] for k in a)
    p = exact_two_sided_binomial_p(wins, losses)
    return {'baseline': baseline, 'arm': arm, 'blocks': BLOCKS,
            'wins': wins, 'losses': losses, 'net_gain': wins-losses,
            'delta': (wins-losses)/BLOCKS, 'exact_two_sided_p': p,
            'qualification_threshold_met': wins-losses >= 8 and p <= .05,
            'per_block': [{'block': k, 'baseline_pass': bool(a[k]['phenotype_pass']),
                           'arm_pass': bool(b[k]['phenotype_pass'])} for k in range(BLOCKS)]}


def aggregate(rows):
    contrasts = {a: paired(rows, a) for a in ('factorized', 'native_r2')}
    adjusted = OLD.holm({a: v['exact_two_sided_p'] for a, v in contrasts.items()})
    for arm, value in contrasts.items():
        value['secondary_holm_p'] = adjusted[arm]
    indexed = {(r['block'], r['arm']): int(r['phenotype_pass']) for r in rows}
    interaction = [indexed[k, 'factorized_r2'] - indexed[k, 'native_r2']
                   - indexed[k, 'factorized'] + indexed[k, 'native'] for k in range(BLOCKS)]
    mean = float(np.mean(interaction))
    half = 2.0395134463964077 * float(np.std(interaction, ddof=1)) / BLOCKS**.5
    primary = paired(rows, 'factorized_r2')
    return {'protocol': PROTOCOL, 'status': 'COMPLETE', 'expected_arms': 128,
            'completed_arms': len(rows), 'primary': primary,
            'primary_verdict': 'D_MINUS_A_RELIABILITY_QUALIFIED' if primary['qualification_threshold_met']
                               else 'NO_D_MINUS_A_RELIABILITY_QUALIFICATION',
            'secondary': contrasts, 'interaction': {'mean_D_minus_C_minus_B_plus_A': mean,
            'descriptive_t95': [mean-half, mean+half], 'per_block': interaction,
            'confirmatory': False},
            'counts': {a: {'successes': sum(r['phenotype_pass'] for r in rows if r['arm'] == a),
                          'trials': BLOCKS, 'parameters': PARAMETERS[a],
                          'wilson95': wilson_interval(sum(r['phenotype_pass'] for r in rows if r['arm'] == a), BLOCKS)}
                       for a in ARMS},
            'claim_boundary': 'One frozen task/evaluation cohort, fresh paired training blocks; parameterization and R2 bundle, not memory necessity or universal short-credit sufficiency.'}


def report(out, summary, rows):
    lines = ['# C8 read / R2 factorial', '', 'Execution: '+summary['status'], '',
             'Primary verdict: '+summary.get('primary_verdict', 'PENDING'), '',
             '| Arm | Full successes / completed | Parameters |', '|---|---:|---:|']
    for arm in ARMS:
        done = [r for r in rows if r['arm'] == arm]
        lines.append(f"| {arm} | {sum(r['phenotype_pass'] for r in done)}/{len(done)} | {PARAMETERS[arm]} |")
    if 'primary' in summary:
        p = summary['primary']
        lines += ['', f"D-A wins={p['wins']}, losses={p['losses']}, net={p['net_gain']}/32; exact two-sided p={p['exact_two_sided_p']:.6g}."]
    lines += ['', 'Read summary.json, then perarm.json and metrics.csv. Complete Full components are in',
              'each arm evaluation/*_summary.json. Traces, curves and frontier CSVs are secondary.',
              'Completed execution and scientific qualification are distinct. R2 effects combine memory,',
              'computation and added parameters; a D win alone does not establish interaction.']
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fields = ('block', 'initialization_seed', 'schedule_seed', 'arm', 'full_pass', 'size', 'horizon',
              'coverage', 'strict_mean', 'strict_pooled', 'retention64_to256', 'ever_regressed_fraction', 'frontier_effect')
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for size, value in row['evaluation_compact']['sizes'].items():
                for t in ('64', '128', '256'):
                    writer.writerow({'block': row['block'], 'initialization_seed': row['initialization_seed'],
                                     'schedule_seed': row['schedule_seed'], 'arm': row['arm'],
                                     'full_pass': row['phenotype_pass'], 'size': size, 'horizon': t,
                                     **{k: value[k][t] for k in ('coverage', 'strict_mean', 'strict_pooled')},
                                     **{k: value[k] for k in ('retention64_to256', 'ever_regressed_fraction', 'frontier_effect')}})


@torch.no_grad()
def systems(model, evaluation):
    result = {}
    model.eval()
    for size in (32, 64):
        x = evaluation[size]['x'][:8].cuda()
        for _ in range(2):
            model.rollout(x, 64)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(5):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            state = model.rollout(x, 64)
            logits = model.logits(state)
            stop.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(stop))
            learned_bytes = sum(v.numel()*v.element_size() for v in state)
            del state, logits
        result[str(size)] = {'batch': 8, 'forward_steps': 64, 'warmups': 2, 'repeats': 5,
                             'forward64_ms': times, 'median_forward64_ms': statistics.median(times),
                             'median_step_ms': statistics.median(times)/64,
                             'learned_persistent_bytes': learned_bytes, 'clock_global_bits': 0,
                             'incremental_peak_cuda_bytes': torch.cuda.max_memory_allocated()-base}
        del x
    return result


def exact_tree(a, b, path='root'):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a.cpu(), b.cpu()), f'Bitwise mismatch: {path}'
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            exact_tree(a[k], b[k], path+'.'+str(k))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for k, (x, y) in enumerate(zip(a, b)):
            exact_tree(x, y, path+'.'+str(k))
    else:
        assert a == b, path


def check(out):
    assert not out.exists()
    C.setup_backend()
    hashes = source_hashes()
    tests = module('_factorial_tests', Path(__file__).with_name('test_cells.py'))
    cpu = tests.check()
    assert cpu['status'] == 'PASS', 'Cell checks failed'
    fixture = [{'block': k, 'arm': a, 'phenotype_pass': a == 'factorized_r2' and k < 8}
               for k in range(BLOCKS) for a in ARMS]
    fixture_summary = aggregate(fixture)
    assert fixture_summary['primary']['qualification_threshold_met']
    assert paired([dict(r, phenotype_pass=False) for r in fixture], 'factorized_r2')['exact_two_sided_p'] == 1.
    data = C.region_bank(32, 8, 97432, 'cuda')
    rows = np.random.default_rng(97499).integers(0, 8, (5, 8)).tolist()
    checks = {}
    for arm in ARMS:
        torch.cuda.reset_peak_memory_stats()
        model = model_for(arm, 97999)
        optimizer = optimizer_for(model)
        snapshots, entries = [], {}
        for update, ids in enumerate(rows, 1):
            optimizer.zero_grad(set_to_none=True)
            loss, state, cadence = C.backward_trajectory(model, C.subset(data, ids), 8)
            assert cadence == R.CADENCE
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            gradients = C.cpu_tree({name: p.grad for name, p in model.named_parameters()})
            for name, p in model.named_parameters():
                if name.endswith('.U') or name.startswith('g_out.'):
                    if bool((p.grad != 0).any()) and name not in entries:
                        entries[name] = update
            norm = R.finish_update(model, optimizer, all(bool(torch.isfinite(v).all()) for v in state))
            snapshots.append({'loss': C.cpu_tree(loss), 'gradients': gradients,
                              'norm': C.cpu_tree(norm), 'model': C.cpu_tree(model.state_dict()),
                              'optimizer': C.cpu_tree(optimizer.state_dict())})
        required = [name for name, p in model.named_parameters()
                    if name.endswith('.U') or name.startswith('g_out.')]
        assert all(name in entries for name in required), f'Dead new path: {arm}, {entries}'
        del model, optimizer, state, gradients
        gc.collect()
        graph_model = model_for(arm, 97999)
        graph_optimizer = optimizer_for(graph_model)
        graph = R.CapturedK8(graph_model, C.subset(data, rows[0]))
        update_times = []
        for update, ids in enumerate(rows, 1):
            tick = time.monotonic()
            loss, state, finite = graph.run(C.subset(data, ids))
            gradients = {name: p.grad for name, p in graph_model.named_parameters()}
            exact_tree(snapshots[update-1]['loss'], loss, arm+'.loss')
            exact_tree(snapshots[update-1]['gradients'], gradients, arm+'.gradients')
            norm = R.finish_update(graph_model, graph_optimizer, finite)
            exact_tree(snapshots[update-1]['norm'], norm, arm+'.clip')
            exact_tree(snapshots[update-1]['model'], graph_model.state_dict(), arm+'.theta')
            exact_tree(snapshots[update-1]['optimizer'], graph_optimizer.state_dict(), arm+'.Adam')
            torch.cuda.synchronize()
            update_times.append(time.monotonic()-tick)
        checks[arm] = {'status': 'PASS', 'bitwise_eager_graph_updates': 5,
                       'first_nonzero_new_gradients': entries, 'parameters': PARAMETERS[arm],
                       'capture_setup_seconds': graph.setup_seconds,
                       'median_update_seconds': statistics.median(update_times),
                       'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                       'state_shapes': [list(v.shape) for v in state], 'cadence': R.CADENCE}
        print(json.dumps({'qualification_arm': arm, **checks[arm]}), flush=True)
        del graph, graph_model, graph_optimizer, snapshots, gradients, state, loss, norm
        gc.collect()
        torch.cuda.empty_cache()
    # Exercise the unchanged trace interface with a real three-component R state.
    audit = C._load_audit()
    tiny = C.region_bank(8, 2, 97408, 'cuda')
    with torch.no_grad():
        trace, records = audit.trace(model_for('factorized_r2', 97999).eval(), tiny, 8)
    assert trace['correct'].shape == (257, 2, 8, 8)
    assert set(('64', '128', '256')).issubset(records)
    assert source_hashes() == hashes
    out.parent.mkdir(parents=True, exist_ok=True)
    C.write(out, {'status': 'PASS', 'protocol': PROTOCOL, 'checked_utc': C.now(),
                  'source_sha256': hashes, 'cpu': cpu, 'statistics': fixture_summary['primary'],
                  'cuda': checks, 'three_state_trace': 'PASS', 'runtime_limit_enforced': False,
                  'qualification_scope': 'Five actual-shape updates each arm; short replay, not full300 equivalence.'})
    print(json.dumps({'status': 'PASS', 'qualification': out.relative_to(ROOT).as_posix()}), flush=True)


def run(out, qualification):
    assert not out.exists()
    q = read(qualification)
    hashes = source_hashes()
    assert q['status'] == 'PASS' and q['protocol'] == PROTOCOL and q['source_sha256'] == hashes
    C.setup_backend()
    out.mkdir(parents=True)
    started = time.monotonic()
    rows = []
    plans = {str(k): {'initialization_seed': seed, 'schedule_seed': schedule,
                     'arm_order': list(ARMS[k % 4:] + ARMS[:k % 4]),
                     'batch_indices': np.random.default_rng(schedule).integers(0, 512, (UPDATES, 8)).tolist()}
             for k, (seed, schedule) in enumerate(zip(INIT_SEEDS, SCHEDULE_SEEDS))}
    C.write(out / 'plans.json', plans)
    plan_hash = C.sha(out / 'plans.json')
    train = C.region_bank(32, 512, 10002, 'cuda')
    evaluation = {size: C.region_bank(size, 32, seed) for size, seed in EVAL_SEEDS.items()}
    data_hashes = {'train': C.tensor_hash(train), **{f'evaluation{size}': C.tensor_hash(data) for size, data in evaluation.items()}}
    C.write(out / 'config.json', {'protocol': PROTOCOL,
        'arms': {a: model_for(a, 97999, 'cpu').metadata() for a in ARMS},
        'initialization_seeds': INIT_SEEDS, 'schedule_seeds': SCHEDULE_SEEDS,
        'train_bank_seed': 10002, 'train_maps': 512, 'eval_bank_seeds': EVAL_SEEDS,
        'eval_maps_each_size': 32, 'updates': UPDATES, 'batch': 8, 'credit_horizon': 8,
        'training_forward_horizon': 64, 'eval_horizons': [64, 128, 256],
        'optimizer': C.OPTIMIZER, 'U_weight_decay': 0., 'gradient_clip': 1.,
        'primary': 'factorized_r2 minus native', 'primary_net_gain_threshold': 8,
        'primary_exact_p_threshold': .05, 'runtime': 'cuda_graph_k8', 'runtime_limit_enforced': False})
    C.write(out / 'manifest.json', {'protocol': PROTOCOL, 'pid': os.getpid(),
        'host': os.environ.get('COMPUTERNAME'), 'started_utc': C.now(),
        'command': [sys.executable, *sys.argv], 'gpu': torch.cuda.get_device_name(),
        'torch': str(torch.__version__), 'numpy': str(np.__version__),
        'source_sha256': hashes, 'schedule_plan_sha256': plan_hash, 'data_sha256': data_hashes,
        'qualification_sha256': C.sha(qualification), 'expected_arms': 128,
        'runtime_limit_enforced': False, 'maximum_seconds': None, 'watchdog_enabled': False,
        'continuous_monitoring': False,
        'git_review_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'backend': {'cudnn_benchmark': False, 'cudnn_deterministic': False,
                    'cudnn_tf32': True, 'matmul_tf32': False}})
    for name in hashes:
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)

    def progress(value):
        C.write(out / 'status.json', {'status': 'RUNNING', 'pid': os.getpid(),
            'elapsed_seconds': time.monotonic()-started, 'updated_utc': C.now(),
            'expected_arms': 128, 'completed_arms': len(rows), **value})
        print(json.dumps(value), flush=True)

    try:
        for block, plan in plans.items():
            for arm in plan['arm_order']:
                folder = out / f'block{int(block):02d}' / arm
                folder.mkdir(parents=True)
                model = model_for(arm, plan['initialization_seed'])
                optimizer = optimizer_for(model)
                initial_hash = C.tensor_hash(model.state_dict())
                torch.save(payload(arm, model, optimizer, plan['initialization_seed'], 0), folder / 'initial.pt')
                progress({'block': int(block), 'arm': arm, 'phase': 'starting', 'completed_updates': 0})
                trained = R.train_graph_updates(model, optimizer, train, plan['batch_indices'],
                    progress=lambda event: progress({'block': int(block), 'arm': arm, **event}))
                curve = trained['curve']
                assert len(curve) == UPDATES and all(int(s['step'].item()) == UPDATES for s in optimizer.state.values())
                C.write(folder / 'training_curve.json', curve)
                C.write(folder / 'runtime.json', {k: v for k, v in trained.items() if k != 'curve'})
                checkpoint = folder / 'final_u300.pt'
                torch.save(payload(arm, model, optimizer, plan['initialization_seed'], UPDATES), checkpoint)
                final_hash = C.tensor_hash(model.state_dict())
                progress({'block': int(block), 'arm': arm, 'phase': 'evaluation', 'completed_updates': UPDATES})
                measured = C.evaluate(model, evaluation, folder / 'evaluation', arm, lambda: None)
                timing = systems(model, evaluation)
                assert C.tensor_hash(model.state_dict()) == final_hash
                C.write(folder / 'systems.json', timing)
                rows.append({'block': int(block), 'initialization_seed': plan['initialization_seed'],
                    'schedule_seed': plan['schedule_seed'], 'arm': arm, 'metadata': model.metadata(),
                    'initial_parameter_sha256': initial_hash, 'final_parameter_sha256': final_hash,
                    'checkpoint_sha256': C.sha(checkpoint), 'checkpoint': checkpoint.relative_to(out).as_posix(),
                    'training_seconds': sum(r['seconds'] for r in curve),
                    'capture_setup_seconds': trained['capture_setup_seconds'], 'runtime': trained['runtime'],
                    'systems': timing, 'phenotype_pass': bool(measured['phenotype_gate']['pass']),
                    'evaluation_summary': (folder / 'evaluation' / f'{arm}_summary.json').relative_to(out).as_posix(),
                    'evaluation_compact': C.compact(measured)})
                C.write(out / 'perarm.json', rows)
                report(out, {'status': 'RUNNING'}, rows)
                progress({'block': int(block), 'arm': arm, 'phase': 'arm_complete', 'completed_updates': UPDATES})
                del model, optimizer, measured, trained, curve
                gc.collect()
                torch.cuda.empty_cache()
        assert len(rows) == 128 and source_hashes() == hashes
        assert C.sha(out / 'plans.json') == plan_hash and C.tensor_hash(train) == data_hashes['train']
        assert all(C.tensor_hash(data) == data_hashes[f'evaluation{size}'] for size, data in evaluation.items())
        result = aggregate(rows)
    except Exception as error:
        C.write(out / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        result = {'protocol': PROTOCOL, 'status': 'ERROR', 'completed_arms': len(rows),
                  'expected_arms': 128, 'error': repr(error), 'scientific_verdict': 'INCOMPLETE'}
    result.update({'finished_utc': C.now(), 'elapsed_seconds': time.monotonic()-started,
                   'runtime_limit_enforced': False})
    C.write(out / 'summary.json', result)
    C.write(out / 'status.json', {**result, 'pid': os.getpid()})
    report(out, result, rows)
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
