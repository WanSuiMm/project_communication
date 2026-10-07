"""Matched native-writer port relations, with explicit committed-stage recovery."""
from __future__ import annotations

import argparse
import copy
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
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CV = load('_port_coverage', ROOT / 'new/continuous_coverage/run.py')
C, R, E, M = CV.C, CV.R, CV.E, CV.M
H = load('_port_cells', HERE / 'cells.py')
D = load('_port_activity', HERE / 'activity.py')
G = load('_port_evaluation_graph', HERE / 'evaluation_graph.py')
P = load('_port_reporting', HERE / 'reporting.py')
PROTOCOL = 'port_relation_v1_full_writer_native_k8'
ARMS = ('current', 'constant', 'conditioned')
COUNTS = {'current': 5033, 'constant': 5049, 'conditioned': 5177}
BLOCKS, UPDATES = 8, 300
CHECKPOINTS = tuple(range(0, 301, 25))
INIT_SEEDS = tuple(range(130001, 130009))
SCHEDULE_SEEDS = tuple(range(131001, 131009))
EVAL_SEEDS = {32: 122032, 64: 122064}
MODE, LOSS_FN = 'reset64x4', CV.LOSS_FN


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def area_path(value, area):
    path = (ROOT / value).resolve()
    base = (ROOT / area).resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError(f'Path must be a child of {area}')
    return path


def safe_child(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()) or path == base.resolve():
        raise ValueError('Artifact reference escapes parent')
    return path


def hashes():
    value = CV.source_hashes()
    names = [p for p in HERE.iterdir() if p.suffix in ('.py', '.md')]
    names += [ROOT / 'new/hybrid_writer/reporting.py',
              ROOT / 'tools/launch_port_relation.ps1',
              ROOT / 'tools/start_protected_job.ps1', ROOT / 'tools/protected_job_worker.ps1']
    value.update({p.relative_to(ROOT).as_posix(): C.sha(p) for p in names})
    return dict(sorted(value.items()))


def config():
    value = {'protocol': PROTOCOL, 'arms': ARMS, 'blocks': BLOCKS, 'super_updates': UPDATES,
             'parameter_counts': COUNTS, 'carrier_channels': 24, 'latent_channels': 8,
             'initialization_seeds': INIT_SEEDS, 'schedule_seeds': SCHEDULE_SEEDS,
             'relation_initialization': 'all coefficients zero',
             'content_spatial_feature': 'prestream_L(C_t)',
             'primary_contrast': ['conditioned', 'current'], 'mode': MODE, 'batch': 8,
             'forward_steps_per_update': 256, 'loss_windows': 32, 'credit_horizon': 8,
             'optimizer_steps_per_update': 1, 'parameters_fixed_within_super_update': True,
             'train_seed': 10002, 'train_maps': 512, 'eval_seeds': EVAL_SEEDS,
             'eval_maps_per_size': 32, 'checkpoints': CHECKPOINTS, 'formal_checkpoint': 300,
             'optimizer': C.OPTIMIZER, 'clip': 1., 'runtime_limit_enforced': False}
    return json.loads(json.dumps(value))


def model_for(arm, seed, device='cuda'):
    model = H.make_model(arm, seed, device)
    if sum(p.numel() for p in model.parameters()) != COUNTS[arm]:
        raise RuntimeError('Parameter count changed')
    return model


def observed_evaluation(model, banks, folder, full=False):
    saved = model.relation_observer
    parameter_hash = C.tensor_hash(model.state_dict())
    try:
        measured = G.evaluate(model, banks, folder, full=full)
        observer = G.get_activity(model)
        write(folder / 'relation_activity.json',
              {'parameters': D.parameters(model), 'rows': observer.result()})
        measured['evaluation_status'] = 'COMPLETE'
        measured['trace_complete'] = True
        write(folder / 'summary.json', measured)
    except FloatingPointError as error:
        if str(error) != 'Nonfinite paired rollout':
            raise
        observer = G.get_activity(model)
        measured = numerical_failure_metrics(full)
        measured.update(evaluation_status='NUMERICAL_FAILURE', trace_complete=False,
                        numerical_failure=True, error=str(error))
        write(folder / 'summary.json', measured)
        write(folder / 'numerical_failure.json',
              {'error': str(error), 'observer_calls': observer.calls,
               'trace_complete': False, 'counts_as_joint_ready': False})
        write(folder / 'relation_activity.json',
              {'parameters': D.parameters(model), 'rows': [], 'status': 'NUMERICAL_FAILURE'})
    finally:
        model.relation_observer = saved
    if C.tensor_hash(model.state_dict()) != parameter_hash:
        raise RuntimeError('Evaluation mutated parameters')
    return measured


def numerical_failure_metrics(full):
    fields = ('R_strict_pooled_T64', 'R_strict_mean_T64', 'S_retention64_to256',
              'S_retention_numerator', 'S_continuous_survival64_to256',
              'retention_reference_pixels', 'retention_reference_maps',
              'ever_regressed_fraction', 'ever_regressed_numerator',
              'ever_regressed_denominator', 'coverage_gain64_to256', 'frontier_effect')
    sizes = {}
    for size in EVAL_SEEDS:
        row = {key: None for key in fields}
        row.update(frontier_full=None, frontier_status='numerical_failure',
                   coverage={str(t): {key: None for key in ('mean_map', 'pooled', 'correct_pixels',
                             'pixels', 'eligible_maps')} for t in (64, 128, 256)})
        sizes[str(size)] = row
    return {'schema': 'continuous-coverage-metrics-v1', 'full_pass': False if full else None,
            'sizes': sizes}


def qualification(out, parent=None):
    if out.exists() or out.with_suffix('').exists():
        raise FileExistsError('Qualification output must be new')
    C.setup_backend()
    before, started = hashes(), time.monotonic()
    artifact = out.with_suffix('')
    artifact.mkdir(parents=True, exist_ok=False)
    for filename in ('check_cells.py', 'check_activity.py', 'reporting.py'):
        args = [sys.executable, '-X', 'utf8', '-B', str(HERE / filename)]
        if filename == 'reporting.py':
            args.append('--self-test')
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        (artifact / (filename + '.stdout.txt')).write_text(result.stdout, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(result.stderr)
    bank = C.region_bank(32, 16, 133032)
    data = {k: v.cuda() for k, v in C.subset(bank, list(range(8))).items()}
    banks = {s: C.region_bank(s, 32, seed) for s, seed in EVAL_SEEDS.items()}
    checked = []
    for arm in ARMS:
        base = model_for(arm, 130099, 'cpu')
        generator = torch.Generator(device='cpu').manual_seed(133099)
        # Qualification-only excitation. Formal initialization remains all zero.
        with torch.no_grad():
            base.f_out.weight.copy_(torch.randn(base.f_out.weight.shape, generator=generator) * .02)
            base.q_out.weight.copy_(torch.randn(base.q_out.weight.shape, generator=generator) * .002)
            for name, parameter in base.named_parameters():
                if name.startswith('relation_'):
                    parameter.copy_(torch.randn(parameter.shape, generator=generator) * .003)
        eager, captured = copy.deepcopy(base).cuda(), copy.deepcopy(base).cuda()
        opt_e, opt_g = C.optimizer_for(eager), C.optimizer_for(captured)
        ae, ag = D.TrainingActivity(eager), D.TrainingActivity(captured)
        eager.relation_observer, captured.relation_observer = ae, ag
        before_capture = C.tensor_hash(captured.state_dict())
        torch.cuda.reset_peak_memory_stats()
        graph = R.CapturedSuperK8(captured, data, MODE, LOSS_FN)
        if C.tensor_hash(captured.state_dict()) != before_capture:
            raise RuntimeError('Capture mutated parameters')
        max_grad, timings = 0., []
        for _ in range(2):
            eager.zero_grad(set_to_none=True)
            ae.reset()
            loss_e, state_e, finite_e = R.math_backward(eager, data, MODE, LOSS_FN)
            ag.reset()
            tick = time.monotonic()
            loss_g, state_g, finite_g = graph.run(data)
            torch.cuda.synchronize()
            timings.append(time.monotonic() - tick)
            torch.testing.assert_close(loss_g, loss_e, atol=1e-6, rtol=1e-5)
            for (name, p), (other, q) in zip(eager.named_parameters(), captured.named_parameters()):
                if name != other or p.grad is None or q.grad is None:
                    raise RuntimeError('Parameter/gradient mismatch')
                torch.testing.assert_close(q.grad, p.grad, atol=1e-6, rtol=1e-5)
                max_grad = max(max_grad, float((q.grad - p.grad).abs().max()))
            for a, b in zip(state_g, state_e):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(ag.stats, ae.stats, atol=1e-6, rtol=1e-5)
            if arm != 'current' and (D.gradient_norm(captured) <= 0 or float(ag.stats[:, 2].sum()) <= 0):
                raise RuntimeError('Relation branch not exercised')
            if arm == 'conditioned' and float(ag.stats[:, 3].sum()) <= 0:
                raise RuntimeError('State-conditioned branch not exercised')
            R.finish_update(eager, opt_e, finite_e)
            R.finish_update(captured, opt_g, finite_g)
            for a, b in zip(eager.parameters(), captured.parameters()):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
                for key in ('exp_avg', 'exp_avg_sq', 'step'):
                    torch.testing.assert_close(opt_g.state[b][key], opt_e.state[a][key], atol=1e-6, rtol=1e-5)
        tick = time.monotonic()
        measured = observed_evaluation(captured, banks, artifact / arm)
        if not measured['trace_complete']:
            raise RuntimeError('Qualification evaluation was nonfinite')
        evaluation_seconds = time.monotonic() - tick
        tick = time.monotonic()
        repeated = artifact / f'{arm}_cached'
        repeated_metrics = observed_evaluation(captured, banks, repeated)
        cached_evaluation_seconds = time.monotonic() - tick
        if measured != repeated_metrics:
            raise RuntimeError('Cached repeated evaluation changed metrics')
        compare_trace_banks(artifact / arm, repeated)
        row = {'arm': arm, 'status': 'PASS', 'parameter_count': COUNTS[arm],
               'two_update_gradient_max_absolute_error': max_grad,
               'capture_setup_seconds': graph.setup_seconds,
               'median_captured_backward_seconds': float(np.median(timings)),
               'actual_two_size_32_map_evaluation_seconds': evaluation_seconds,
               'cached_two_size_32_map_evaluation_seconds': cached_evaluation_seconds,
               'peak_cuda_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
               'relation_parameters': D.parameters(captured), 'training_activity': ag.result()}
        checked.append(row)
        print(json.dumps(row), flush=True)
        G.clear_cache(captured)
        del base, eager, captured, graph, opt_e, opt_g, ae, ag, state_e, state_g
        gc.collect()
        torch.cuda.empty_cache()
    recovery = recovery_binding(parent, before) if parent else None
    regression = recovery_regression(parent, artifact) if parent else None
    if hashes() != before:
        raise RuntimeError('Source changed during qualification')
    projected = BLOCKS * sum(v['capture_setup_seconds'] + UPDATES * v['median_captured_backward_seconds']
                            + v['actual_two_size_32_map_evaluation_seconds']
                            + (len(CHECKPOINTS) - 1) * v['cached_two_size_32_map_evaluation_seconds']
                            for v in checked)
    remaining = projected
    if parent:
        inherited = committed(parent, read(parent / 'manifest.json'))
        remaining = 0.
        by_arm = {v['arm']: v for v in checked}
        for block in range(BLOCKS):
            for arm in ARMS:
                updates = [r['update'] for r in inherited if (r['block'], r['arm']) == (block, arm)]
                last = max(updates) if updates else 0
                stages = sum(u > last for u in CHECKPOINTS) + int(not updates)
                if stages:
                    v = by_arm[arm]
                    remaining += (v['capture_setup_seconds']
                                  + (UPDATES - last) * v['median_captured_backward_seconds']
                                  + v['actual_two_size_32_map_evaluation_seconds']
                                  + (stages - 1) * v['cached_two_size_32_map_evaluation_seconds'])
    eval_bytes = sum(p.stat().st_size for arm in ARMS for p in (artifact / arm).rglob('*') if p.is_file())
    write(out, {'status': 'PASS', 'protocol': PROTOCOL, 'source_sha256': before,
                'checked_utc': C.now(), 'arms': checked, 'actual_training_shape': [8, 3, 32, 32],
                'credit_horizon': 8, 'eager_graph_updates_each_arm': 2,
                'estimated_run_seconds_excluding_io_optimizer_overhead': projected,
                'estimated_remaining_seconds_excluding_io_optimizer_overhead': remaining,
                'estimated_evidence_bytes': eval_bytes * BLOCKS * len(CHECKPOINTS) + 312 * 150000,
                'qualification_seconds': time.monotonic() - started, 'runtime_limit_enforced': False,
                'recovery_binding': recovery, 'recovery_regression': regression})
    print(json.dumps({'status': 'PASS', 'estimated_hours': projected / 3600}), flush=True)


def make_plans():
    plans = {}
    prefixes = ('encoder.', 'f_in.', 'f_out.', 'q_in.', 'q_out.', 'readout.')
    for block, (seed, schedule) in enumerate(zip(INIT_SEEDS, SCHEDULE_SEEDS)):
        models = {a: model_for(a, seed, 'cpu') for a in ARMS}
        initial = {a: C.tensor_hash(m.state_dict()) for a, m in models.items()}
        components = {a: {p: C.tensor_hash({k: v for k, v in m.state_dict().items() if k.startswith(p)})
                           for p in prefixes} for a, m in models.items()}
        if any(len({v[p] for v in components.values()}) != 1 for p in prefixes):
            raise RuntimeError('Common initialization differs')
        offset = block % 3
        plans[str(block)] = {'initialization_seed': seed, 'schedule_seed': schedule,
            'arm_order': list(ARMS[offset:] + ARMS[:offset]), 'initial_parameter_sha256': initial,
            'component_sha256': components,
            'batch_indices': np.random.default_rng(schedule).integers(0, 512, (UPDATES, 8)).tolist()}
    return plans


def load_banks(folder):
    result = {}
    for label in ('train', 'evaluation32', 'evaluation64'):
        with np.load(folder / (label + '.npz'), allow_pickle=False) as bank:
            result[label] = {k: torch.from_numpy(np.ascontiguousarray(bank[k])) for k in bank.files}
    return result.pop('train'), {s: result[f'evaluation{s}'] for s in EVAL_SEEDS}


def committed(parent, manifest):
    records = []
    for row in read(parent / 'dense.json'):
        checkpoint = safe_child(parent, row['checkpoint'])
        marker = checkpoint.with_name(checkpoint.name + '.complete.json')
        if not marker.exists():
            continue
        mark = read(marker)
        if mark.get('record_sha256') != digest(row) or mark.get('checkpoint_sha256') != C.sha(checkpoint):
            raise RuntimeError('Committed row/checkpoint binding changed')
        if row['arm'] not in ARMS or row['block'] not in range(BLOCKS) or row['update'] not in CHECKPOINTS:
            raise RuntimeError('Committed stage outside protocol')
        folder = safe_child(parent, str(Path(row['evaluation_summary']).parent))
        for name, expected in mark['evaluation_sha256'].items():
            if C.sha(safe_child(folder, name)) != expected:
                raise RuntimeError('Committed evaluation binding changed')
        if 'training_curve_prefix_sha256' in mark:
            curve = read(checkpoint.parent.parent / 'training_curve.json')
            prefix = [entry for entry in curve if entry['update'] <= row['update']]
            if digest(prefix) != mark['training_curve_prefix_sha256']:
                raise RuntimeError('Committed training curve prefix changed')
        records.append(row)
    keys = {(r['block'], r['arm'], r['update']) for r in records}
    if len(keys) != len(records):
        raise RuntimeError('Duplicate committed stages')
    for block in range(BLOCKS):
        for arm in ARMS:
            updates = sorted(r['update'] for r in records if r['block'] == block and r['arm'] == arm)
            if updates != list(CHECKPOINTS[:len(updates)]):
                raise RuntimeError('Committed stages are not a prefix')
    return records


def recovery_binding(parent, source):
    """Explicit source bridge for this telemetry repair; never waive model hashes."""
    manifest = read(parent / 'manifest.json')
    if manifest['protocol'] != PROTOCOL or read(parent / 'config.json') != config():
        raise RuntimeError('Recovery scientific protocol/config differs')
    if C.sha(parent / 'config.json') != manifest['config_sha256']:
        raise RuntimeError('Parent config binding changed')
    old_source = manifest['source_sha256']
    for name, expected in old_source.items():
        if C.sha(safe_child(parent / 'source', name)) != expected:
            raise RuntimeError('Parent source snapshot differs')
    changed = sorted(name for name in old_source.keys() | source.keys()
                     if old_source.get(name) != source.get(name))
    allowed = {'new/port_relation/activity.py', 'new/port_relation/check_activity.py',
               'new/port_relation/evaluation_graph.py',
               'new/port_relation/run.py', 'new/port_relation/EXECUTION.md'}
    if set(changed) - allowed:
        raise RuntimeError(f'Recovery patch changes scientific source: {set(changed) - allowed}')
    original_qualification = next((p for p in (ROOT / 'analyses').glob('port_relation_qualification_*.json')
                                   if C.sha(p) == manifest['qualification_sha256']), None)
    if original_qualification is None:
        raise RuntimeError('Original bound qualification is unavailable')
    original = read(original_qualification)
    if original.get('status') != 'PASS' or original.get('source_sha256') != old_source:
        raise RuntimeError('Original qualification binding differs')
    records = committed(parent, manifest)
    curves = {f'block{block:02d}/{arm}/training_curve.json':
                  C.sha(parent / f'block{block:02d}/{arm}/training_curve.json')
              for block, arm in {(row['block'], row['arm']) for row in records}}
    return {'schema': 'port-telemetry-repair-source-bridge-v1',
            'parent_run': parent.relative_to(ROOT).as_posix(),
            'parent_manifest_sha256': C.sha(parent / 'manifest.json'),
            'parent_config_sha256': C.sha(parent / 'config.json'),
            'parent_qualification_sha256': C.sha(original_qualification),
            'parent_source_sha256': old_source, 'current_source_sha256': source,
            'changed_sources': changed, 'parent_curve_sha256': curves,
            'committed_record_count': len(records),
            'scientific_model_loss_optimizer_schedule_unchanged': True}


def compare_trace_banks(previous, current):
    for size in EVAL_SEEDS:
        with np.load(previous / f'size{size}_traces.npz', allow_pickle=False) as old, \
             np.load(current / f'size{size}_traces.npz', allow_pickle=False) as new:
            if set(old.files) != set(new.files) or any(not np.array_equal(old[k], new[k]) for k in old.files):
                raise RuntimeError('Accelerated evaluation changed Boolean traces')
    old_summary, new_summary = read(previous / 'summary.json'), read(current / 'summary.json')
    for value in (old_summary, new_summary):
        value.pop('evaluation_status', None)
        value.pop('trace_complete', None)
    if old_summary != new_summary:
        raise RuntimeError('Accelerated evaluation changed frozen report metrics')


def recovery_regression(parent, artifact):
    """Re-evaluate the failed checkpoint without training or inheriting its stage."""
    error_path = parent / 'error.json'
    if not error_path.exists() or 'not JSON compliant: inf' not in read(error_path).get('error', ''):
        return None
    progress = read(error_path)['last_progress']
    block, arm, update = progress['block'], progress['arm'], progress['completed_updates']
    checkpoint = safe_child(parent, f'block{block:02d}/{arm}/checkpoints/u{update:03d}.pt')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    plan = read(parent / 'plans.json')[str(block)]
    if (payload['protocol'], payload['arm'], payload['completed_updates'],
        payload['initialization_seed'], payload['schedule_seed']) != (
            PROTOCOL, arm, update, plan['initialization_seed'], plan['schedule_seed']):
        raise RuntimeError('Failed-checkpoint regression metadata differs')
    model = model_for(arm, plan['initialization_seed'])
    model.load_state_dict(payload['state_dict'], strict=True)
    _, banks = load_banks(parent / 'banks')
    folder = artifact / 'failed_stage_regression'
    measured = observed_evaluation(model, banks, folder, full=update == UPDATES)
    if not measured['trace_complete']:
        raise RuntimeError('Failed-stage regression rollout was nonfinite')
    previous = safe_child(parent, f'block{block:02d}/{arm}/evaluation/.u{update:03d}.inprogress')
    compare_trace_banks(previous, folder)
    result = {'status': 'PASS', 'block': block, 'arm': arm, 'update': update,
              'checkpoint_sha256': C.sha(checkpoint), 'trace_complete': True,
              'packed_boolean_traces_identical': True,
              'strict_json_activity_finite': all(
                  not any(value for key, value in row.items() if key.endswith('_nonfinite'))
                  for row in read(folder / 'relation_activity.json')['rows']),
              'optimizer_updates': 0, 'stage_inherited_as_committed': False}
    if not result['strict_json_activity_finite']:
        raise RuntimeError('Failed-stage regression diagnostic remained nonfinite')
    # Load another saved parameter state into the SAME model, so the cached
    # evaluator must consume changed live weights rather than frozen values.
    records = committed(parent, read(parent / 'manifest.json'))
    old = max((row for row in records if (row['block'], row['arm']) == (block, arm)),
              key=lambda row: row['update'])
    previous_payload = torch.load(safe_child(parent, old['checkpoint']), map_location='cpu', weights_only=False)
    model.load_state_dict(previous_payload['state_dict'], strict=True)
    live = artifact / 'cached_live_parameter_regression'
    observed_evaluation(model, banks, live, full=old['update'] == UPDATES)
    compare_trace_banks(safe_child(parent, str(Path(old['evaluation_summary']).parent)), live)
    result['cached_graph_uses_live_parameters'] = True
    result['live_parameter_regression_update'] = old['update']
    G.clear_cache(model)
    del model, payload, previous_payload, banks
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({'failed_stage_regression': result}), flush=True)
    return result


def setup_output(out, qualification_path, parent):
    source, q = hashes(), read(qualification_path)
    if q.get('status') != 'PASS' or q.get('protocol') != PROTOCOL or q.get('source_sha256') != source:
        raise RuntimeError('Passing source-bound qualification required')
    if out.exists():
        raise FileExistsError('New output directory required')
    if parent is not None and (out.is_relative_to(parent) or parent.is_relative_to(out)):
        raise ValueError('Parent and child must be distinct and non-nested')
    records, plans, old_manifest = [], make_plans(), None
    if parent:
        old_manifest = read(parent / 'manifest.json')
        if read(parent / 'config.json') != config():
            raise RuntimeError('Recovery config differs')
        patch = (old_manifest['source_sha256'] != source or
                 old_manifest['qualification_sha256'] != C.sha(qualification_path))
        if patch and q.get('recovery_binding') != recovery_binding(parent, source):
            raise RuntimeError('Explicit passing recovery source bridge required')
        if old_manifest['gpu'] != torch.cuda.get_device_name():
            raise RuntimeError('Recovery GPU model differs')
        if read(parent / 'plans.json') != plans or C.sha(parent / 'plans.json') != old_manifest['schedule_plan_sha256']:
            raise RuntimeError('Recovery fixed schedule differs')
        for name, expected in old_manifest['source_sha256'].items():
            if C.sha(safe_child(parent / 'source', name)) != expected:
                raise RuntimeError('Recovery source snapshot differs')
        records = committed(parent, old_manifest)
        train_cpu, banks = load_banks(parent / 'banks')
        data_hash = {'train': C.tensor_hash(train_cpu), **{f'evaluation{s}': C.tensor_hash(v) for s, v in banks.items()}}
        if data_hash != old_manifest['data_sha256']:
            raise RuntimeError('Recovery banks differ')
    else:
        train_cpu = C.region_bank(32, 512, 10002)
        banks = {s: C.region_bank(s, 32, seed) for s, seed in EVAL_SEEDS.items()}
        data_hash = {'train': C.tensor_hash(train_cpu), **{f'evaluation{s}': C.tensor_hash(v) for s, v in banks.items()}}
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'config.json', config())
    write(out / 'plans.json', plans)
    (out / 'banks').mkdir()
    for label, value in {'train': train_cpu, **{f'evaluation{s}': v for s, v in banks.items()}}.items():
        np.savez_compressed(out / 'banks' / (label + '.npz'), **{k: a.numpy() for k, a in value.items()})
    for name in source:
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    if parent:
        for row in records:
            cp = safe_child(parent, row['checkpoint'])
            target = safe_child(out, row['checkpoint'])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cp, target)
            shutil.copyfile(cp.with_name(cp.name + '.complete.json'), target.with_name(target.name + '.complete.json'))
            stage = str(Path(row['evaluation_summary']).parent)
            shutil.copytree(safe_child(parent, stage), safe_child(out, stage))
        for block in range(BLOCKS):
            for arm in ARMS:
                stages = [r['update'] for r in records if r['block'] == block and r['arm'] == arm]
                if stages:
                    rel = f'block{block:02d}/{arm}/training_curve.json'
                    curve = [r for r in read(safe_child(parent, rel)) if r['update'] <= max(stages)]
                    if [r['update'] for r in curve] != list(range(1, max(stages) + 1)):
                        raise RuntimeError('Recovery curve is incomplete')
                    write(safe_child(out, rel), curve)
    final = [r for r in records if r['update'] == UPDATES]
    write(out / 'dense.json', records)
    write(out / 'perarm.json', final)
    manifest = {'protocol': PROTOCOL, 'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'),
        'started_utc': C.now(), 'command': [sys.executable, *sys.argv], 'gpu': torch.cuda.get_device_name(),
        'torch': str(torch.__version__), 'numpy': str(np.__version__), 'python': sys.version.split()[0],
        'source_sha256': source, 'config_sha256': C.sha(out / 'config.json'),
        'data_sha256': data_hash, 'schedule_plan_sha256': C.sha(out / 'plans.json'),
        'qualification_sha256': C.sha(qualification_path), 'expected_arms': 24, 'expected_dense_records': 312,
        'parent_run': parent.relative_to(ROOT).as_posix() if parent else None,
        'parent_manifest_sha256': C.sha(parent / 'manifest.json') if parent else None,
        'recovery_binding': q.get('recovery_binding') if parent else None,
        'inherited_committed_stages': len(records), 'runtime_limit_enforced': False,
        'watchdog_enabled': False, 'continuous_monitoring': False,
        'backend_deterministic': False, 'bitwise_uninterrupted_equivalence_guaranteed': False,
        'git_review_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write(out / 'manifest.json', manifest)
    return source, plans, train_cpu, banks, records, final


def save_stage(out, block, arm, update, model, optimizer, plan, curve, graph, banks, records, final,
               evaluation_seconds, progress, resumed):
    folder = out / f'block{block:02d}' / arm
    checkpoint = folder / 'checkpoints' / f'u{update:03d}.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        raise FileExistsError('Checkpoint output already exists')
    payload = C.payload(model, optimizer, plan['initialization_seed'], update)
    payload.update(protocol=PROTOCOL, arm=arm, schedule_seed=plan['schedule_seed'],
                   torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all())
    temp = checkpoint.with_name(checkpoint.name + '.tmp')
    torch.save(payload, temp)
    os.replace(temp, checkpoint)
    progress({'block': block, 'arm': arm, 'phase': 'evaluation', 'completed_updates': update})
    tick = time.monotonic()
    stage = folder / 'evaluation' / f'u{update:03d}'
    temporary_stage = folder / 'evaluation' / f'.u{update:03d}.inprogress'
    measured = observed_evaluation(model, banks, temporary_stage, full=update == UPDATES)
    os.replace(temporary_stage, stage)
    elapsed = time.monotonic() - tick
    evaluation_seconds += elapsed
    row = {'block': block, 'arm': arm, 'update': update,
        'initialization_seed': plan['initialization_seed'], 'schedule_seed': plan['schedule_seed'],
        'checkpoint': checkpoint.relative_to(out).as_posix(), 'checkpoint_sha256': C.sha(checkpoint),
        'parameter_sha256': C.tensor_hash(model.state_dict()), 'evaluation_seconds': elapsed,
        'evaluation_summary': (stage / 'summary.json').relative_to(out).as_posix(),
        'relation_activity': (stage / 'relation_activity.json').relative_to(out).as_posix(),
        'metrics': M.compact_pair_metrics(measured), 'joint': M.joint_readiness(measured),
        'formal_endpoint': update == UPDATES, 'checkpoint_selection': False,
        'evaluation_status': measured['evaluation_status'], 'trace_complete': measured['trace_complete'],
        'numerical_failure': measured.get('numerical_failure', False)}
    if update == UPDATES:
        row.update(last_loss=curve[-1]['mean_super_update_loss'],
            training_seconds=sum(v['seconds'] for v in curve), all_checkpoint_evaluation_seconds=evaluation_seconds,
            capture_setup_seconds=graph.setup_seconds,
            capture_setup_scope='resumed_session_only' if resumed else 'whole_arm_session')
        final.append(row)
    records.append(row)
    write(out / 'dense.json', records)
    write(out / 'perarm.json', final)
    write(folder / 'training_curve.json', curve)
    P.report(out, final, records, status={'status': 'RUNNING'})
    progress({'block': block, 'arm': arm, 'phase': 'stage_committing', 'completed_updates': update})
    write(checkpoint.with_name(checkpoint.name + '.complete.json'),
        {'stage_complete': True, 'record_sha256': digest(row), 'checkpoint_sha256': C.sha(checkpoint),
         'evaluation_sha256': {p.name: C.sha(p) for p in stage.iterdir() if p.is_file()},
         'training_curve_prefix_sha256': digest(curve),
         'completed_update': update})
    model.train()
    return evaluation_seconds


def run(out, qualification_path, parent=None):
    C.setup_backend()
    started = time.monotonic()
    source, plans, train_cpu, banks, records, final = setup_output(out, qualification_path, parent)
    train = {k: v.cuda() for k, v in train_cpu.items()}
    plan_hash = C.sha(out / 'plans.json')
    last = {}

    def progress(value):
        last.update(value)
        write(out / 'status.json', {'protocol': PROTOCOL, 'status': 'RUNNING', 'pid': os.getpid(),
            'updated_utc': C.now(), 'elapsed_seconds': time.monotonic() - started,
            'completed_arms': len(final), 'expected_arms': 24,
            'completed_dense_records': len(records), 'expected_dense_records': 312, **value})
        print(json.dumps(value), flush=True)

    try:
        for block in range(BLOCKS):
            plan = plans[str(block)]
            for arm in plan['arm_order']:
                folder = out / f'block{block:02d}' / arm
                old = sorted((r for r in records if r['block'] == block and r['arm'] == arm), key=lambda r: r['update'])
                if old and old[-1]['update'] == UPDATES:
                    continue
                model = model_for(arm, plan['initialization_seed'])
                optimizer = C.optimizer_for(model)
                resumed, start = bool(old), old[-1]['update'] + 1 if old else 1
                curve = read(folder / 'training_curve.json') if old else []
                evaluation_seconds = sum(r['evaluation_seconds'] for r in old)
                if old:
                    payload = torch.load(out / old[-1]['checkpoint'], map_location='cpu', weights_only=False)
                    if (payload['protocol'], payload['arm'], payload['completed_updates'], payload['initialization_seed'],
                        payload['schedule_seed']) != (PROTOCOL, arm, start - 1, plan['initialization_seed'], plan['schedule_seed']):
                        raise RuntimeError('Restored checkpoint metadata mismatch')
                    model.load_state_dict(payload['state_dict'], strict=True)
                    optimizer.load_state_dict(payload['optimizer_state_dict'])
                    if C.tensor_hash(model.state_dict()) != old[-1]['parameter_sha256']:
                        raise RuntimeError('Restored parameter hash mismatch')
                    torch.set_rng_state(payload['torch_rng_state'])
                    torch.cuda.set_rng_state_all(payload['cuda_rng_state'])
                    if any(int(v['step'].item()) != start - 1 for v in optimizer.state.values()):
                        raise RuntimeError('Restored Adam step mismatch')
                elif C.tensor_hash(model.state_dict()) != plan['initial_parameter_sha256'][arm]:
                    raise RuntimeError('Fresh initialization differs from plan')
                activity = D.TrainingActivity(model)
                model.relation_observer = activity
                progress({'block': block, 'arm': arm, 'phase': 'starting', 'completed_updates': start - 1})
                if not old:
                    evaluation_seconds = save_stage(out, block, arm, 0, model, optimizer, plan, curve,
                        None, banks, records, final, evaluation_seconds, progress, False)
                first = C.subset(train, plan['batch_indices'][start - 1])
                capture_before = C.tensor_hash(model.state_dict())
                adam_before = C.tensor_hash({f'{i}/{k}': v for i, state in enumerate(optimizer.state.values())
                                              for k, v in state.items() if torch.is_tensor(v)})
                graph = R.CapturedSuperK8(model, first, MODE, LOSS_FN)
                adam_after = C.tensor_hash({f'{i}/{k}': v for i, state in enumerate(optimizer.state.values())
                                             for k, v in state.items() if torch.is_tensor(v)})
                if capture_before != C.tensor_hash(model.state_dict()) or adam_before != adam_after:
                    raise RuntimeError('Capture mutated parameters or optimizer')
                for update in range(start, UPDATES + 1):
                    tick = time.monotonic()
                    data = first if update == start else C.subset(train, plan['batch_indices'][update - 1])
                    activity.reset()
                    loss, state, finite = graph.run(data)
                    relation_grad = D.gradient_norm(model)
                    norm = R.finish_update(model, optimizer, finite)
                    torch.cuda.synchronize()
                    curve.append({'update': update, 'mean_super_update_loss': float(loss),
                        'gradient_norm_before_clip': float(norm), 'seconds': time.monotonic() - tick,
                        'cold_initializations': 4, **CV.CADENCE,
                        'relation_gradient_norm_before_clip': relation_grad,
                        'relation_parameters_after_update': D.parameters(model), 'relation_activity': activity.result()})
                    if update == start or update % 25 == 0:
                        progress({'block': block, 'arm': arm, 'phase': 'training', 'completed_updates': update,
                                  'loss': float(loss), 'gradient_norm_before_clip': float(norm)})
                    if update in CHECKPOINTS:
                        if any(int(v['step'].item()) != update for v in optimizer.state.values()):
                            raise RuntimeError('Adam step count differs')
                        evaluation_seconds = save_stage(out, block, arm, update, model, optimizer, plan, curve,
                            graph, banks, records, final, evaluation_seconds, progress, resumed)
                    if update != start:
                        del data
                model.relation_observer = None
                G.clear_cache(model)
                del model, optimizer, graph, state, first, curve, activity
                gc.collect()
                torch.cuda.empty_cache()
        if hashes() != source or C.sha(out / 'plans.json') != plan_hash:
            raise RuntimeError('Source/schedule changed during training')
        result = P.aggregate(final, records)
        if result['status'] != 'COMPLETE':
            raise RuntimeError('Final aggregate remains incomplete')
        result.update(protocol=PROTOCOL, completed_arms=len(final), expected_arms=24,
                      dense_records=len(records), expected_dense_records=312)
    except Exception as error:
        write(out / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc(), 'last_progress': last})
        result = {'protocol': PROTOCOL, 'status': 'ERROR', 'completed_arms': len(final), 'expected_arms': 24,
                  'dense_records': len(records), 'error': repr(error), 'overall_verdict': 'INCOMPLETE'}
    result.update(finished_utc=C.now(), elapsed_seconds=time.monotonic() - started, runtime_limit_enforced=False)
    write(out / 'summary.json', result)
    P.report(out, final, records, status=result)
    write(out / 'status.json', {**result, 'pid': os.getpid()})
    print(json.dumps(result), flush=True)
    if result['status'] == 'ERROR':
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
            qualification(out, area_path(args.resume_parent, 'runs') if args.resume_parent else None)
        except Exception as error:
            if not out.exists():
                write(out, {'status': 'ERROR', 'protocol': PROTOCOL, 'error': repr(error),
                            'traceback': traceback.format_exc(), 'runtime_limit_enforced': False})
            raise
    else:
        if not args.qualification:
            parser.error('--qualification is required')
        parent = area_path(args.resume_parent, 'runs') if args.resume_parent else None
        run(out, area_path(args.qualification, 'analyses'), parent)


if __name__ == '__main__':
    main()
