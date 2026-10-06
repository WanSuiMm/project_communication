"""Fixed outcome-blind lane caps: paired K8 training and dense qualification."""
from __future__ import annotations

import argparse
import copy
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
HERE = Path(__file__).parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


CV = load('_hc_coverage', ROOT/'new/continuous_coverage/run.py')
C, R, E, M = CV.C, CV.R, CV.E, CV.M
H = load('_hc_cells', HERE/'cells.py')
P = load('_hc_reporting', HERE/'reporting.py')
D = load('_hc_dose', HERE/'dose.py')
PROTOCOL = 'hardclip_v1_fixed_tail_v1'
ARMS = ('neural', 'hardclip')
BLOCKS, UPDATES = 8, 300
CHECKPOINTS = tuple(range(0, 301, 25))
INIT_SEEDS = tuple(range(116001, 116009))
SCHEDULE_SEEDS = tuple(range(117001, 117009))
EVAL_SEEDS = {32: 118032, 64: 118064}
LOSS_FN = CV.LOSS_FN
MODE = 'reset64x4'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def hashes():
    value = CV.source_hashes()
    names = [p for p in HERE.iterdir() if p.suffix in ('.py', '.md')]
    names += [ROOT/'tools/launch_hardclip_v1.ps1', ROOT/'new/hybrid_writer/reporting.py']
    value.update({p.relative_to(ROOT).as_posix(): C.sha(p) for p in names})
    return dict(sorted(value.items()))


def calibration_binding(folder):
    summary, status = read(folder/'summary.json'), read(folder/'status.json')
    assert status['status'] == 'COMPLETE' and summary['status'] == 'CALIBRATED'
    assert summary['protocol'] == 'hardclip_v1_tail_calibration_v1'
    assert summary['completed_blocks'] == summary['expected_blocks'] == 8
    assert summary['target_removed_fraction'] == .01 and summary['max_trigger_fraction'] == .20
    assert len(summary['lanes']) == 4
    caps = []
    for lane, row in enumerate(summary['lanes']):
        assert row['lane'] == lane and row['sparse_enough']
        assert row['fp32_trigger_fraction'] <= .20
        assert abs(row['fp32_removed_fraction']-.01) <= 1e-7
        caps.append(row['frozen_fp32_write_cap'])
    assert np.array_equal(np.asarray(caps, np.float32), np.asarray(H.DEFAULT_CAPS, np.float32))
    return {'directory': folder.relative_to(ROOT).as_posix(), 'actual_write_caps': caps,
            'summary_sha256': C.sha(folder/'summary.json'),
            'status_sha256': C.sha(folder/'status.json'),
            'samples_sha256': C.sha(folder/'raw_write_rms.npz'),
            'inputs_sha256': C.sha(folder/'calibration_inputs.npz'),
            'parent_protocol': summary['protocol'], 'outcome_blind': True}


def model_for(arm, seed, caps, device='cuda'):
    model = H.make_model(arm, seed, device, caps=caps)
    assert sum(p.numel() for p in model.parameters()) == 5033
    return model


def observed_evaluation(model, banks, folder, full=False):
    saved_observer = model.writer_observer
    observer = D.EvaluationDose(model)
    model.writer_observer = observer
    try:
        measured = E.evaluate(model, banks, folder, full=full)
        C.write(folder/'clip_dose.json', observer.result())
    finally:
        model.writer_observer = saved_observer
    return measured


def check(out, calibration):
    assert not out.exists(), 'Qualification already exists'
    C.setup_backend()
    before, binding = hashes(), calibration_binding(calibration)
    caps = binding['actual_write_caps']
    started = time.monotonic()
    artifact = out.with_suffix('')
    artifact.mkdir(parents=True, exist_ok=False)
    for filename in ('check_cells.py', 'reporting.py'):
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(HERE/filename)],
                                cwd=ROOT, capture_output=True, text=True)
        (artifact/(filename+'.stdout.txt')).write_text(result.stdout, encoding='utf-8')
        assert result.returncode == 0, result.stderr
    bank = C.region_bank(32, 16, 119032)
    data = {k: v.cuda() for k, v in C.subset(bank, list(range(8))).items()}
    banks = {s: C.region_bank(s, 32, seed) for s, seed in EVAL_SEEDS.items()}
    checked = []
    for arm in ARMS:
        base = model_for(arm, 116099, caps, 'cpu')
        torch.manual_seed(119099)
        # Force active clipping in this implementation check only. Formal
        # initialization stays canonical, including zero f_out/q_out weights.
        with torch.no_grad():
            base.f_out.weight.normal_(std=.02)
            base.q_out.weight.normal_(std=.002)
            base.q_out.bias.zero_()
            base.f_out.bias.copy_(torch.tensor([b/.1*1.3 for b in caps]).repeat_interleave(6))
        eager, captured = copy.deepcopy(base).cuda(), copy.deepcopy(base).cuda()
        opt_e, opt_g = C.optimizer_for(eager), C.optimizer_for(captured)
        dose_e, dose_g = D.TrainingDose(eager), D.TrainingDose(captured)
        eager.writer_observer, captured.writer_observer = dose_e, dose_g
        before_capture = C.tensor_hash(captured.state_dict())
        torch.cuda.reset_peak_memory_stats()
        graph = R.CapturedSuperK8(captured, data, MODE, LOSS_FN)
        assert C.tensor_hash(captured.state_dict()) == before_capture
        max_grad, timings = 0., []
        for update in range(3):
            eager.zero_grad(set_to_none=True)
            dose_e.reset()
            loss_e, state_e, finite_e = R.math_backward(eager, data, MODE, LOSS_FN)
            dose_g.reset()
            tick = time.monotonic()
            loss_g, state_g, finite_g = graph.run(data)
            torch.cuda.synchronize()
            elapsed = time.monotonic()-tick
            torch.testing.assert_close(loss_g, loss_e, atol=1e-6, rtol=1e-5)
            for (name, p), (other, q) in zip(eager.named_parameters(), captured.named_parameters()):
                assert name == other and p.grad is not None and q.grad is not None
                torch.testing.assert_close(q.grad, p.grad, atol=1e-6, rtol=1e-5)
                max_grad = max(max_grad, float((q.grad-p.grad).abs().max()))
            for a, b in zip(state_g, state_e):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(dose_g.stats, dose_e.stats, atol=1e-6, rtol=1e-5)
            assert bool((dose_g.stats[:, 1] > 0).all()), 'Active clip branch untested'
            R.finish_update(eager, opt_e, finite_e)
            R.finish_update(captured, opt_g, finite_g)
            for a, b in zip(eager.parameters(), captured.parameters()):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
                for key in ('exp_avg', 'exp_avg_sq', 'step'):
                    torch.testing.assert_close(opt_g.state[b][key], opt_e.state[a][key], atol=1e-6, rtol=1e-5)
            assert captured.fixed_clip.max_write_rms.tolist() == caps
            timings.append(elapsed)
        tick = time.monotonic()
        observed_evaluation(captured, banks, artifact/arm)
        evaluation_seconds = time.monotonic()-tick
        row = {'arm': arm, 'status': 'PASS', 'parameter_count': 5033,
            'three_update_gradient_max_absolute_error': max_grad,
            'capture_setup_seconds': graph.setup_seconds,
            'median_captured_backward_seconds': float(np.median(timings)),
            'actual_32_map_two_size_evaluation_seconds': evaluation_seconds,
            'peak_cuda_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
            'active_branch_exercised_all_lanes': True,
            'training_dose': dose_g.result(), 'fixed_caps_unchanged': True}
        checked.append(row)
        print(json.dumps(row), flush=True)
        del base, eager, captured, graph, opt_e, opt_g, state_e, state_g, dose_e, dose_g
        gc.collect()
        torch.cuda.empty_cache()
    assert hashes() == before and calibration_binding(calibration) == binding
    projected = BLOCKS*sum(row['capture_setup_seconds']+UPDATES*row['median_captured_backward_seconds']
                           +len(CHECKPOINTS)*row['actual_32_map_two_size_evaluation_seconds'] for row in checked)
    eval_bytes = sum(p.stat().st_size for p in artifact.rglob('*') if p.is_file())
    C.write(out, {'status': 'PASS', 'protocol': PROTOCOL, 'source_sha256': before,
        'calibration_binding': binding, 'checked_utc': C.now(), 'arms': checked,
        'actual_training_shape': [8, 3, 32, 32], 'credit_horizon': 8,
        'eager_graph_updates_each_arm': 3, 'atol': 1e-6, 'rtol': 1e-5,
        'estimated_run_seconds_excluding_data_io_and_optimizer_overhead': projected,
        'estimated_evidence_bytes': eval_bytes*BLOCKS*len(CHECKPOINTS)+208*150000,
        'qualification_seconds': time.monotonic()-started, 'runtime_limit_enforced': False})
    print(json.dumps({'status': 'PASS', 'estimated_hours': projected/3600,
                     'estimated_evidence_MiB': (eval_bytes*BLOCKS*len(CHECKPOINTS)+208*150000)/2**20}), flush=True)


def run(out, qualification, calibration):
    assert not out.exists(), 'Run already exists'
    q, source = read(qualification), hashes()
    binding = calibration_binding(calibration)
    assert q['status'] == 'PASS' and q['protocol'] == PROTOCOL and q['source_sha256'] == source
    assert q['calibration_binding'] == binding
    caps = binding['actual_write_caps']
    C.setup_backend()
    out.mkdir(parents=True)
    started = time.monotonic()
    records, final, plans = [], [], {}
    for block, (seed, schedule) in enumerate(zip(INIT_SEEDS, SCHEDULE_SEEDS)):
        models = {a: model_for(a, seed, caps, 'cpu') for a in ARMS}
        initial = {a: C.tensor_hash(m.state_dict()) for a, m in models.items()}
        assert len(set(initial.values())) == 1
        offset = block%2
        plans[str(block)] = {'initialization_seed': seed, 'schedule_seed': schedule,
            'arm_order': list(ARMS[offset:]+ARMS[:offset]), 'initial_parameter_sha256': initial,
            'batch_indices': np.random.default_rng(schedule).integers(0, 512, (UPDATES, 8)).tolist()}
    C.write(out/'plans.json', plans)
    plan_hash = C.sha(out/'plans.json')
    train_cpu = C.region_bank(32, 512, 10002)
    train = {k: v.cuda() for k, v in train_cpu.items()}
    banks = {s: C.region_bank(s, 32, seed) for s, seed in EVAL_SEEDS.items()}
    data_hash = {'train': C.tensor_hash(train_cpu), **{f'evaluation{s}': C.tensor_hash(v) for s, v in banks.items()}}
    (out/'banks').mkdir()
    for label, value in {'train': train_cpu, **{f'evaluation{s}': v for s, v in banks.items()}}.items():
        np.savez_compressed(out/'banks'/f'{label}.npz', **{k: a.numpy() for k, a in value.items()})
    config = {'protocol': PROTOCOL, 'arms': ARMS, 'blocks': BLOCKS, 'super_updates': UPDATES,
        'parameter_count_each_arm': 5033, 'carrier_channels': 24, 'latent_channels': 8,
        'actual_write_caps': caps, 'calibration_binding': binding,
        'initialization_seeds': INIT_SEEDS, 'schedule_seeds': SCHEDULE_SEEDS,
        'mode': MODE, 'batch': 8, 'forward_steps_per_update': 256, 'loss_windows': 32,
        'credit_horizon': 8, 'optimizer_steps_per_update': 1,
        'parameters_fixed_within_super_update': True, 'train_seed': 10002,
        'eval_seeds': EVAL_SEEDS, 'checkpoints': CHECKPOINTS, 'formal_checkpoint': 300,
        'optimizer': C.OPTIMIZER, 'clip': 1., 'runtime_limit_enforced': False}
    C.write(out/'config.json', config)
    manifest = {'protocol': PROTOCOL, 'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'),
        'started_utc': C.now(), 'command': [sys.executable, *sys.argv], 'gpu': torch.cuda.get_device_name(),
        'torch': str(torch.__version__), 'numpy': str(np.__version__), 'source_sha256': source,
        'data_sha256': data_hash, 'schedule_plan_sha256': plan_hash,
        'qualification_sha256': C.sha(qualification), 'calibration_binding': binding,
        'expected_arms': 16, 'expected_dense_records': 208, 'runtime_limit_enforced': False,
        'maximum_seconds': None, 'watchdog_enabled': False, 'continuous_monitoring': False,
        'git_review_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    C.write(out/'manifest.json', manifest)
    for name in source:
        dest = out/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    def progress(value):
        C.write(out/'status.json', {'status': 'RUNNING', 'pid': os.getpid(), 'updated_utc': C.now(),
            'elapsed_seconds': time.monotonic()-started, 'completed_arms': len(final), 'expected_arms': 16,
            'completed_dense_records': len(records), **value})
        print(json.dumps(value), flush=True)
    try:
        for block_text, plan in plans.items():
            block = int(block_text)
            for arm in plan['arm_order']:
                folder = out/f'block{block:02d}'/arm
                (folder/'checkpoints').mkdir(parents=True)
                model = model_for(arm, plan['initialization_seed'], caps)
                assert C.tensor_hash(model.state_dict()) == plan['initial_parameter_sha256'][arm]
                opt, curve, evaluation_seconds = C.optimizer_for(model), [], 0.
                dose = D.TrainingDose(model)
                model.writer_observer = dose
                def save_stage(update):
                    nonlocal evaluation_seconds
                    parameter_hash = C.tensor_hash(model.state_dict())
                    checkpoint = folder/'checkpoints'/f'u{update:03d}.pt'
                    payload = C.payload(model, opt, plan['initialization_seed'], update)
                    payload.update(protocol=PROTOCOL, arm=arm, schedule_seed=plan['schedule_seed'],
                                   actual_write_caps=caps, calibration_summary_sha256=binding['summary_sha256'])
                    torch.save(payload, checkpoint)
                    progress({'block': block, 'arm': arm, 'phase': 'evaluation', 'completed_updates': update})
                    tick = time.monotonic()
                    stage = folder/'evaluation'/f'u{update:03d}'
                    measured = observed_evaluation(model, banks, stage, full=update == UPDATES)
                    elapsed = time.monotonic()-tick
                    evaluation_seconds += elapsed
                    row = {'block': block, 'arm': arm, 'update': update,
                        'initialization_seed': plan['initialization_seed'], 'schedule_seed': plan['schedule_seed'],
                        'checkpoint': checkpoint.relative_to(out).as_posix(), 'checkpoint_sha256': C.sha(checkpoint),
                        'parameter_sha256': parameter_hash, 'evaluation_seconds': elapsed,
                        'evaluation_summary': (stage/'summary.json').relative_to(out).as_posix(),
                        'clip_dose': (stage/'clip_dose.json').relative_to(out).as_posix(),
                        'metrics': M.compact_pair_metrics(measured), 'joint': M.joint_readiness(measured),
                        'formal_endpoint': update == UPDATES, 'checkpoint_selection': False}
                    if update == UPDATES:
                        row.update(training_seconds=sum(v['seconds'] for v in curve),
                            all_checkpoint_evaluation_seconds=evaluation_seconds, capture_setup_seconds=graph.setup_seconds)
                        final.append(row)
                        C.write(out/'perarm.json', final)
                    records.append(row)
                    C.write(out/'dense.json', records)
                    C.write(folder/'training_curve.json', curve)
                    P.report(out, {'status': 'RUNNING'}, records, final)
                    model.train()
                progress({'block': block, 'arm': arm, 'phase': 'starting', 'completed_updates': 0})
                save_stage(0)
                first = C.subset(train, plan['batch_indices'][0])
                before_capture = C.tensor_hash(model.state_dict())
                graph = R.CapturedSuperK8(model, first, MODE, LOSS_FN)
                assert C.tensor_hash(model.state_dict()) == before_capture
                for update, ids in enumerate(plan['batch_indices'], 1):
                    tick = time.monotonic()
                    data = first if update == 1 else C.subset(train, ids)
                    dose.reset()
                    loss, state, finite = graph.run(data)
                    norm = R.finish_update(model, opt, finite)
                    torch.cuda.synchronize()
                    curve.append({'update': update, 'mean_super_update_loss': float(loss),
                        'gradient_norm_before_clip': float(norm), 'seconds': time.monotonic()-tick,
                        'cold_initializations': 4, **CV.CADENCE, 'clip_dose': dose.result()})
                    if update == 1 or update%25 == 0:
                        progress({'block': block, 'arm': arm, 'phase': 'training', 'completed_updates': update,
                            'loss': float(loss), 'gradient_norm_before_clip': float(norm)})
                    if update in CHECKPOINTS:
                        assert all(int(v['step'].item()) == update for v in opt.state.values())
                        assert model.fixed_clip.max_write_rms.tolist() == caps
                        save_stage(update)
                    if update != 1:
                        del data
                assert len(curve) == UPDATES
                progress({'block': block, 'arm': arm, 'phase': 'arm_complete', 'completed_updates': UPDATES})
                model.writer_observer = None
                del model, opt, graph, state, first, curve, dose
                gc.collect()
                torch.cuda.empty_cache()
        assert len(final) == 16 and len(records) == 208 and hashes() == source
        assert calibration_binding(calibration) == binding
        assert C.sha(out/'plans.json') == plan_hash and C.tensor_hash(train) == data_hash['train']
        assert all(C.tensor_hash(v) == data_hash[f'evaluation{s}'] for s, v in banks.items())
        result = P.aggregate(final, records, expected_blocks=BLOCKS)
        assert result['status'] == 'COMPLETE'
        result.update(protocol=PROTOCOL, completed_arms=16, expected_arms=16, dense_records=208)
    except Exception as error:
        C.write(out/'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        result = {'protocol': PROTOCOL, 'status': 'ERROR', 'completed_arms': len(final), 'expected_arms': 16,
            'dense_records': len(records), 'error': repr(error), 'overall_verdict': 'INCOMPLETE'}
    result.update(finished_utc=C.now(), elapsed_seconds=time.monotonic()-started, runtime_limit_enforced=False)
    C.write(out/'summary.json', result)
    C.write(out/'status.json', {**result, 'pid': os.getpid()})
    P.report(out, result, records, final)
    print(json.dumps(result), flush=True)
    if result['status'] == 'ERROR':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--qualification')
    parser.add_argument('--calibration', default='runs/hardclip_v1_calibration_20261006_01')
    args = parser.parse_args()
    out = (ROOT/args.out).resolve()
    calibration = (ROOT/args.calibration).resolve()
    assert calibration.is_relative_to(ROOT/'runs')
    assert out.is_relative_to(ROOT/('analyses' if args.check else 'runs'))
    if args.check:
        try:
            check(out, calibration)
        except Exception as error:
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                C.write(out, {'status': 'ERROR', 'protocol': PROTOCOL, 'error': repr(error),
                    'traceback': traceback.format_exc(), 'runtime_limit_enforced': False})
            raise
    else:
        assert args.qualification, 'Qualification required'
        qualification = (ROOT/args.qualification).resolve()
        assert qualification.is_relative_to(ROOT/'analyses')
        run(out, qualification, calibration)


if __name__ == '__main__':
    main()
