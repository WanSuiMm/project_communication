"""Four-block addressed-write development screen with lightweight evidence."""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
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
HERE = Path(__file__).parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


V = load('_addressed_coverage_training', ROOT / 'new/continuous_coverage/run.py')
C, R = V.C, V.R
H = load('_addressed_cell_models', HERE / 'cells.py')
E = load('_addressed_light_evaluation', HERE / 'evaluation.py')
P = load('_addressed_pilot_report', HERE / 'reporting.py')
PROTOCOL = 'addressed_delta_native_k8_development_v1'
ARMS = ('current', 'additive', 'delta')
COUNTS = {'current': 5033, 'additive': 5689, 'delta': 5689}
BLOCKS, UPDATES = 4, 300
CHECKPOINTS = tuple(range(0, 301, 25))
INIT_SEEDS, SCHEDULE_SEEDS = tuple(range(140001, 140005)), tuple(range(141001, 141005))
MODE, LOSS_FN = 'reset64x4', V.LOSS_FN
BANK_SOURCE = ROOT / 'evidence/port_relation_20261007_02/banks'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def area_path(value, area):
    path, base = (ROOT / value).resolve(), (ROOT / area).resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError(f'Output must be inside {area}')
    return path


def config():
    return {'protocol': PROTOCOL, 'developmental_only': True, 'arms': list(ARMS),
        'blocks': BLOCKS, 'super_updates': UPDATES, 'parameter_counts': COUNTS,
        'carrier_channels': 24, 'latent_channels': 8, 'reads': 4, 'ports': 4,
        'payload_per_port': 6, 'write_step': .1, 'latent_step': .5,
        'initialization_seeds': list(INIT_SEEDS), 'schedule_seeds': list(SCHEDULE_SEEDS),
        'key_initialization': 'zero', 'value_initialization': 'normal PyTorch Conv2d initialization',
        'candidates_initial_tensors_identical': True, 'perception_copied_from_current_f_in': True,
        'unused_original_writer_removed': True, 'content_spatial_feature': 'prestream_L(C_t)',
        'primary_contrast': ['delta', 'additive'], 'contextual_reference': 'current',
        'mode': MODE, 'batch': 8, 'forward_steps_per_update': 256, 'loss_windows': 32,
        'credit_horizon': 8, 'optimizer_steps_per_update': 1,
        'parameters_fixed_within_super_update': True, 'train_seed': 10002, 'train_maps': 512,
        'train_size': 32, 'eval_seeds': {'32': 122032, '64': 122064}, 'eval_maps_per_size': 32,
        'checkpoints': list(CHECKPOINTS), 'formal_checkpoint': 300,
        'optimizer': C.OPTIMIZER, 'clip': 1., 'full_boolean_trace_persistence': False,
        'old_full_evaluated': False, 'runtime_limit_enforced': False}


def hashes():
    result = V.source_hashes()
    for folder in (HERE, ROOT / 'new/port_relation'):
        for path in folder.iterdir():
            if path.suffix in ('.py', '.md'):
                result[path.relative_to(ROOT).as_posix()] = C.sha(path)
    for rel in ('tools/launch_addressed_delta.ps1', 'tools/start_protected_job.ps1', 'tools/protected_job_worker.ps1'):
        result[rel] = C.sha(ROOT / rel)
    return dict(sorted(result.items()))


def model_for(arm, seed, device='cuda'):
    model = H.make_model(arm, seed, device)
    if sum(p.numel() for p in model.parameters()) != COUNTS[arm]:
        raise RuntimeError('Parameter count changed')
    return model


def load_banks(folder):
    result = {}
    for name in ('train', 'evaluation32', 'evaluation64'):
        with np.load(folder / f'{name}.npz', allow_pickle=False) as bank:
            result[name] = {k: torch.from_numpy(np.ascontiguousarray(bank[k])) for k in bank.files}
    expected = read(ROOT / 'PORT_RELATION_PUBLICATION_MANIFEST.json')['data_sha256']
    actual = {k: C.tensor_hash(v) for k, v in result.items()}
    if actual != expected:
        raise RuntimeError('Frozen bank tensor hashes changed')
    return result['train'], {s: result[f'evaluation{s}'] for s in (32, 64)}


def make_plans():
    plans = {}
    for block, (seed, schedule) in enumerate(zip(INIT_SEEDS, SCHEDULE_SEEDS)):
        generator = np.random.default_rng(schedule)
        models = {a: model_for(a, seed, 'cpu') for a in ARMS}
        for module_name in ('encoder', 'q_in', 'q_out', 'readout'):
            values = [C.tensor_hash(getattr(m, module_name).state_dict()) for m in models.values()]
            if len(set(values)) != 1:
                raise RuntimeError(f'Common initialization differs: {module_name}')
        if C.tensor_hash(models['additive'].state_dict()) != C.tensor_hash(models['delta'].state_dict()):
            raise RuntimeError('Candidate initial tensors differ')
        plans[str(block)] = {'initialization_seed': seed, 'schedule_seed': schedule,
            'batch_indices': generator.integers(0, 512, size=(UPDATES, 8)).tolist(),
            'arm_order': [ARMS[(i + block) % 3] for i in range(3)],
            'initial_parameter_sha256': {a: C.tensor_hash(m.state_dict()) for a, m in models.items()}}
    return plans


def qualification(out):
    if out.exists() or out.with_suffix('').exists():
        raise FileExistsError('New qualification output required')
    C.setup_backend()
    source, started = hashes(), time.monotonic()
    folder = out.with_suffix('')
    folder.mkdir(parents=True)
    for script, extra in (('check_cells.py', []), ('reporting.py', ['--self-test'])):
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(HERE / script), *extra],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        (folder / (script + '.stdout.txt')).write_text(result.stdout, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(result.stderr)
    train_cpu, banks = load_banks(BANK_SOURCE)
    data = {k: v.cuda() for k, v in C.subset(train_cpu, list(range(8))).items()}
    results = []
    for arm in ARMS:
        base = model_for(arm, 140099, 'cpu')
        eager, captured_model = copy.deepcopy(base).cuda(), copy.deepcopy(base).cuda()
        opt_e, opt_g = C.optimizer_for(eager), C.optimizer_for(captured_model)
        before = C.tensor_hash(captured_model.state_dict())
        torch.cuda.reset_peak_memory_stats()
        graph = R.CapturedSuperK8(captured_model, data, MODE, LOSS_FN)
        if C.tensor_hash(captured_model.state_dict()) != before:
            raise RuntimeError('Capture changed parameters')
        grad_error, timings = 0., []
        for update in range(1, 4):
            eager.zero_grad(set_to_none=True)
            loss_e, state_e, finite_e = R.math_backward(eager, data, MODE, LOSS_FN)
            tick = time.monotonic()
            loss_g, state_g, finite_g = graph.run(data)
            torch.cuda.synchronize()
            timings.append(time.monotonic() - tick)
            torch.testing.assert_close(loss_g, loss_e, atol=1e-6, rtol=1e-5)
            for (name, p), (other, q) in zip(eager.named_parameters(), captured_model.named_parameters()):
                if name != other or p.grad is None or q.grad is None:
                    raise RuntimeError('Captured parameter/gradient mismatch')
                torch.testing.assert_close(q.grad, p.grad, atol=1e-6, rtol=1e-5)
                grad_error = max(grad_error, float((q.grad - p.grad).abs().max()))
            for a, b in zip(state_g, state_e):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            if arm != 'current' and update == 3:
                for name in ('key_head', 'value_head'):
                    head = getattr(captured_model.carrier_writer, name)
                    if not bool(head.weight.grad.norm() > 0):
                        raise RuntimeError(f'Bootstrap did not activate {name}')
            R.finish_update(eager, opt_e, finite_e)
            R.finish_update(captured_model, opt_g, finite_g)
            for a, b in zip(eager.parameters(), captured_model.parameters()):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(opt_g.state[b][key], opt_e.state[a][key], atol=1e-6, rtol=1e-5)
        tick = time.monotonic()
        measured = E.evaluate(captured_model, banks)
        first_eval = time.monotonic() - tick
        tick = time.monotonic()
        repeated = E.evaluate(captured_model, banks)
        cached_eval = time.monotonic() - tick
        if measured != repeated:
            raise RuntimeError('Cached evaluation changed metrics or per-map rows')
        write(folder / f'{arm}_evaluation.json', measured)
        results.append({'arm': arm, 'status': 'PASS', 'parameter_count': COUNTS[arm],
            'three_update_gradient_max_absolute_error': grad_error,
            'capture_setup_seconds': graph.setup_seconds,
            'median_captured_backward_seconds': float(np.median(timings)),
            'first_two_size_evaluation_seconds': first_eval, 'cached_two_size_evaluation_seconds': cached_eval,
            'key_value_bootstrap_active_by_update3': arm != 'current',
            'peak_cuda_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20})
        print(json.dumps(results[-1]), flush=True)
        E.clear(captured_model)
        del base, eager, captured_model, graph, opt_e, opt_g, state_e, state_g
        gc.collect()
        torch.cuda.empty_cache()
    if hashes() != source:
        raise RuntimeError('Source changed during qualification')
    estimate = BLOCKS * sum(r['capture_setup_seconds'] + UPDATES * r['median_captured_backward_seconds']
        + r['first_two_size_evaluation_seconds'] + (len(CHECKPOINTS) - 1) * r['cached_two_size_evaluation_seconds'] for r in results)
    write(out, {'status': 'PASS', 'protocol': PROTOCOL, 'source_sha256': source, 'checked_utc': C.now(),
        'arms': results, 'actual_training_shape': [8, 3, 32, 32], 'qualification_optimizer_updates_each_arm': 3,
        'qualification_updates_are_not_formal_training': True, 'estimated_run_seconds_excluding_io': estimate,
        'estimated_local_evidence_bytes': 30_000_000, 'qualification_seconds': time.monotonic() - started,
        'timing_scope': 'single qualification observations, not repeated systems benchmark', 'runtime_limit_enforced': False})
    print(json.dumps({'status': 'PASS', 'estimated_hours': estimate / 3600}), flush=True)


def committed(parent, source):
    """Read only atomic, source/config-bound stages from an interrupted parent."""
    manifest = read(parent / 'manifest.json')
    if manifest['protocol'] != PROTOCOL or manifest['source_sha256'] != source or read(parent / 'config.json') != config():
        raise RuntimeError('Recovery parent scientific source/configuration differs')
    if C.sha(parent / 'plans.json') != manifest['schedule_plan_sha256']:
        raise RuntimeError('Recovery batch plan changed')
    rows = []
    for marker in sorted(parent.glob('block*/*/checkpoints/*.pt.complete.json')):
        mark = read(marker)
        stage = marker.parent.parent / 'evaluation' / f"u{mark['completed_update']:03d}"
        row = read(stage / 'record.json')
        checkpoint = (parent / row['checkpoint']).resolve()
        if not checkpoint.is_relative_to(parent.resolve()) or not mark['stage_complete']:
            raise RuntimeError('Invalid recovery checkpoint path/marker')
        if mark['record_sha256'] != digest(row) or mark['checkpoint_sha256'] != C.sha(checkpoint):
            raise RuntimeError('Recovery checkpoint/record binding changed')
        for name, expected in mark['evaluation_sha256'].items():
            if C.sha(stage / name) != expected:
                raise RuntimeError('Recovery stage artifact changed')
        curve = read(marker.parent.parent / 'training_curve.json')[:row['update']]
        if digest(curve) != mark['curve_prefix_sha256']:
            raise RuntimeError('Recovery training-curve prefix changed')
        rows.append(row)
    for block in range(BLOCKS):
        for arm in ARMS:
            selected = sorted(r['update'] for r in rows if r['block'] == block and r['arm'] == arm)
            if selected and selected != list(range(0, max(selected) + 1, 25)):
                raise RuntimeError('Recovery stage prefix has a gap')
    return sorted(rows, key=lambda r: (r['block'], ARMS.index(r['arm']), r['update']))


def save_stage(out, model, optimizer, block, arm, update, plan, curve, records, final):
    folder = out / f'block{block:02d}' / arm
    checkpoint = folder / 'checkpoints' / f'u{update:03d}.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix('.pt.tmp')
    payload = {'protocol': PROTOCOL, 'arm': arm, 'completed_updates': update,
        'initialization_seed': plan['initialization_seed'], 'schedule_seed': plan['schedule_seed'],
        'state_dict': C.cpu_tree(model.state_dict()), 'optimizer_state_dict': C.cpu_tree(optimizer.state_dict()),
        'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all(),
        'curve_prefix_sha256': digest(curve)}
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)
    stage = folder / 'evaluation' / f'u{update:03d}'
    stage.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        evaluation = E.evaluate(model, _BANKS)
    except FloatingPointError as error:
        if str(error) != 'Nonfinite paired rollout':
            raise
        evaluation = {'metrics': {'sizes': {str(s): {} for s in (32, 64)}},
            'joint': {'pass': False, 'reasons': ['NUMERICAL_FAILURE']},
            'evaluation_status': 'NUMERICAL_FAILURE', 'numerical_failure': True,
            'trace_complete': False, 'full_evaluated': False, 'map_metrics': [], 'error': str(error)}
    maps = evaluation.pop('map_metrics')
    write(stage / 'summary.json', evaluation)
    if maps:
        with (stage / 'map_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(maps[0]))
            writer.writeheader()
            writer.writerows(maps)
    row = {'block': block, 'arm': arm, 'update': update, 'initialization_seed': plan['initialization_seed'],
        'schedule_seed': plan['schedule_seed'], 'checkpoint': checkpoint.relative_to(out).as_posix(),
        'checkpoint_sha256': C.sha(checkpoint), 'parameter_sha256': C.tensor_hash(model.state_dict()),
        'evaluation_summary': (stage / 'summary.json').relative_to(out).as_posix(),
        'evaluation_seconds': time.monotonic() - started, 'formal_endpoint': update == UPDATES,
        'checkpoint_selection': False, **evaluation}
    write(stage / 'record.json', row)
    write(folder / 'training_curve.json', curve)
    # Marker precedes index writes: only committed stages can be recovered.
    write(checkpoint.with_name(checkpoint.name + '.complete.json'), {'stage_complete': True,
        'record_sha256': digest(row), 'checkpoint_sha256': row['checkpoint_sha256'],
        'curve_prefix_sha256': digest(curve), 'completed_update': update,
        'evaluation_sha256': {p.name: C.sha(p) for p in stage.iterdir() if p.is_file()}})
    records.append(row)
    if update == UPDATES:
        final.append(row)
    write(out / 'dense.json', records)
    write(out / 'perarm.json', final)
    P.report(out, final, records, execution_status={'status': 'RUNNING'})
    model.train()


def run(out, qualification_path, parent=None):
    global _BANKS
    if out.exists():
        raise FileExistsError('New run output required')
    checked, source = read(qualification_path), hashes()
    if checked.get('status') != 'PASS' or checked.get('protocol') != PROTOCOL or checked['source_sha256'] != source:
        raise RuntimeError('Passing current-source qualification required')
    C.setup_backend()
    train_cpu, _BANKS = load_banks(parent / 'banks' if parent else BANK_SOURCE)
    plans = read(parent / 'plans.json') if parent else make_plans()
    records = committed(parent, source) if parent else []
    final = [r for r in records if r['update'] == UPDATES]
    out.mkdir(parents=True)
    write(out / 'config.json', config())
    write(out / 'plans.json', plans)
    for rel in source:
        target = out / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    for name in ('train', 'evaluation32', 'evaluation64'):
        target = out / 'banks' / f'{name}.npz'
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile((parent / 'banks' if parent else BANK_SOURCE) / target.name, target)
    if parent:
        last = {}
        for row in records:
            checkpoint = Path(row['checkpoint'])
            target = out / checkpoint
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent / checkpoint, target)
            marker = Path(str(checkpoint) + '.complete.json')
            shutil.copyfile(parent / marker, out / marker)
            stage = Path(row['evaluation_summary']).parent
            shutil.copytree(parent / stage, out / stage)
            last[(row['block'], row['arm'])] = max(last.get((row['block'], row['arm']), 0), row['update'])
        for (block, arm), update in last.items():
            relative = f'block{block:02d}/{arm}/training_curve.json'
            write(out / relative, read(parent / relative)[:update])
        write(out / 'dense.json', records)
        write(out / 'perarm.json', final)
    write(out / 'manifest.json', {'protocol': PROTOCOL, 'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'),
        'started_utc': C.now(), 'command': sys.argv, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'source_sha256': source, 'config_sha256': C.sha(out / 'config.json'),
        'schedule_plan_sha256': C.sha(out / 'plans.json'), 'qualification_sha256': C.sha(qualification_path),
        'data_sha256': {'train': C.tensor_hash(train_cpu), **{f'evaluation{s}': C.tensor_hash(b) for s, b in _BANKS.items()}},
        'expected_arms': 12, 'expected_dense_records': 156, 'runtime_limit_enforced': False,
        'backend_deterministic': False, 'bitwise_uninterrupted_equivalence_guaranteed': False,
        'continuous_monitoring': False, 'old_full_evaluated': False, 'full_boolean_trace_persistence': False,
        'parent_run': parent.relative_to(ROOT).as_posix() if parent else None,
        'parent_manifest_sha256': C.sha(parent / 'manifest.json') if parent else None,
        'inherited_committed_stages': len(records)})
    train = {k: v.cuda() for k, v in train_cpu.items()}
    started = time.monotonic()

    def progress(extra):
        value = {'protocol': PROTOCOL, 'status': 'RUNNING', 'pid': os.getpid(), 'updated_utc': C.now(),
            'completed_arms': len(final), 'expected_arms': 12, 'dense_records': len(records),
            'expected_dense_records': 156, 'elapsed_seconds': time.monotonic() - started, **extra}
        write(out / 'status.json', value)
        print(json.dumps(extra), flush=True)

    try:
        for block in range(BLOCKS):
            plan = plans[str(block)]
            for arm in plan['arm_order']:
                old = sorted((r for r in records if r['block'] == block and r['arm'] == arm), key=lambda r: r['update'])
                if old and old[-1]['update'] == UPDATES:
                    continue
                model = model_for(arm, plan['initialization_seed'])
                optimizer = C.optimizer_for(model)
                if C.tensor_hash(model.state_dict()) != plan['initial_parameter_sha256'][arm]:
                    raise RuntimeError('Fresh initialization differs from plan')
                start = old[-1]['update'] + 1 if old else 1
                curve = read(out / f'block{block:02d}/{arm}/training_curve.json') if old else []
                if old:
                    payload = torch.load(out / old[-1]['checkpoint'], map_location='cpu', weights_only=False)
                    if (payload['protocol'], payload['arm'], payload['completed_updates'], payload['initialization_seed'], payload['schedule_seed']) != (
                            PROTOCOL, arm, start - 1, plan['initialization_seed'], plan['schedule_seed']):
                        raise RuntimeError('Restored checkpoint metadata differs')
                    model.load_state_dict(payload['state_dict'], strict=True)
                    optimizer.load_state_dict(payload['optimizer_state_dict'])
                    if C.tensor_hash(model.state_dict()) != old[-1]['parameter_sha256']:
                        raise RuntimeError('Restored parameter hash differs')
                    if any(int(v['step'].item()) != start - 1 for v in optimizer.state.values()):
                        raise RuntimeError('Restored Adam step count differs')
                    torch.set_rng_state(payload['torch_rng_state'])
                    torch.cuda.set_rng_state_all(payload['cuda_rng_state'])
                progress({'block': block, 'arm': arm, 'phase': 'starting', 'completed_updates': start - 1})
                if not old:
                    save_stage(out, model, optimizer, block, arm, 0, plan, curve, records, final)
                first = C.subset(train, plan['batch_indices'][start - 1])
                before = C.tensor_hash(model.state_dict())
                graph = R.CapturedSuperK8(model, first, MODE, LOSS_FN)
                if C.tensor_hash(model.state_dict()) != before:
                    raise RuntimeError('Capture changed parameters')
                for update, indices in enumerate(plan['batch_indices'][start - 1:], start):
                    data = C.subset(train, indices)
                    tick = time.monotonic()
                    loss, state, finite = graph.run(data)
                    norm = R.finish_update(model, optimizer, finite)
                    curve.append({'update': update, 'mean_super_update_loss': float(loss),
                        'gradient_norm_before_clip': float(norm), 'seconds': time.monotonic() - tick,
                        'cold_initializations': 4, 'forward_steps': 256, 'backward_calls': 32,
                        'loss_windows': 32, 'optimizer_steps': 1, 'credit_horizon': 8})
                    if update == 1 or update % 25 == 0:
                        progress({'block': block, 'arm': arm, 'phase': 'training', 'completed_updates': update,
                                  'loss': float(loss), 'gradient_norm_before_clip': float(norm)})
                    if update in CHECKPOINTS:
                        if any(int(v['step'].item()) != update for v in optimizer.state.values()):
                            raise RuntimeError('Adam step count differs from update')
                        save_stage(out, model, optimizer, block, arm, update, plan, curve, records, final)
                E.clear(model)
                del model, optimizer, graph, state, first, data, curve
                gc.collect()
                torch.cuda.empty_cache()
        if hashes() != source:
            raise RuntimeError('Scientific source changed during training')
        result = P.aggregate(final, records)
        if result['status'] != 'COMPLETE':
            raise RuntimeError('Final record grid incomplete')
        execution = {'status': 'COMPLETE'}
    except Exception as error:
        execution = {'status': 'ERROR', 'error': repr(error), 'overall_verdict': 'INCOMPLETE'}
        write(out / 'error.json', {**execution, 'traceback': traceback.format_exc()})
    execution.update(protocol=PROTOCOL, completed_arms=len(final), expected_arms=12,
        dense_records=len(records), expected_dense_records=156, finished_utc=C.now(),
        elapsed_seconds=time.monotonic() - started, runtime_limit_enforced=False)
    P.report(out, final, records, execution_status=execution)
    write(out / 'status.json', {**execution, 'pid': os.getpid()})
    print(json.dumps(execution), flush=True)
    if execution['status'] != 'COMPLETE':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--qualification')
    parser.add_argument('--resume-parent')
    args = parser.parse_args()
    out = area_path(args.out, 'analyses' if args.check else 'runs')
    if args.check:
        try:
            qualification(out)
        except Exception as error:
            if not out.exists():
                write(out, {'status': 'ERROR', 'protocol': PROTOCOL, 'error': repr(error),
                            'traceback': traceback.format_exc()})
            raise
    else:
        if not args.qualification:
            parser.error('--qualification required')
        parent = area_path(args.resume_parent, 'runs') if args.resume_parent else None
        run(out, area_path(args.qualification, 'analyses'), parent)


if __name__ == '__main__':
    main()
