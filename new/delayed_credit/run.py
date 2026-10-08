"""Three-arm delayed-credit development screen; no architecture changes."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'new/trajectory_qualification'))
import common as C


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


T = load('_delayed_credit_training', HERE / 'training.py')
E = load('_delayed_credit_evaluation', HERE / 'evaluation.py')
P = load('_delayed_credit_reporting', HERE / 'reporting.py')
ARMS = ('dense_k8', 'terminal_k8', 'terminal_k64')
PROTOCOL = 'delayed_credit_cue_once_streaming_v0'
INIT_SEEDS = tuple(range(150001, 150005))
SCHEDULE_SEEDS = tuple(range(151001, 151005))
UPDATES, BATCH = 300, 8
CHECKPOINTS = tuple(range(50, 301, 50))
EVAL_SEEDS = {32: 152032, 64: 152064}
TRAIN_PATH = ROOT / 'evidence/addressed_delta_20261008_02/banks/train.npz'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    C.write(path, value)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def config():
    return {'protocol': PROTOCOL, 'arms': list(ARMS), 'parameter_count': 5033,
        'architecture': 'original StreamingCell C24/Z8', 'steps_per_update': 64,
        'updates': UPDATES, 'batch': BATCH, 'train_maps': 512, 'train_size': 32,
        'train_seed': 10002, 'evaluation_seeds': EVAL_SEEDS, 'evaluation_maps_per_size': 32,
        'initialization_seeds': INIT_SEEDS, 'schedule_seeds': SCHEDULE_SEEDS,
        'checkpoint_grid': CHECKPOINTS, 'formal_checkpoint': 300,
        'midpoint_efficacy_evaluation': False, 'cue_initialization_only': True,
        'dense_loss_normalization': 'mean of eight losses', 'terminal_loss': 'L64 only',
        'optimizer': C.OPTIMIZER, 'clip_norm': 1., 'runtime_limit_enforced': False,
        'continuous_monitoring': False, 'old_full_evaluated': False,
        'primary': 'terminal_k64 versus terminal_k8 T64 reach on fresh size32 maps'}


def sources():
    files = set(HERE.glob('*.py')) | {HERE / 'PROTOCOL.md'}
    files |= {ROOT / name for name in (
        'new/streaming_carry/stream_cells.py', 'new/workspace_revision/revision_cells.py',
        'new/workspace_revision/run_revision.py', 'new/nca_inertial_wind_tunnel/tasks.py',
        'new/nca_inertial_wind_tunnel/cells.py', 'new/masked_medium/masked_cells.py',
        'new/trajectory_qualification/common.py', 'new/short_bptt/training.py',
        'new/continuous_coverage/evaluation.py', 'new/seed4_followup/phenotype.py',
        'tools/start_protected_job.ps1', 'tools/protected_job_worker.ps1')}
    hashes = {relative(path): C.sha(path) for path in sorted(files)}
    old = read(ROOT / 'ADDRESSED_DELTA_PUBLICATION_MANIFEST.json')['source_sha256']
    for name, h in hashes.items():
        if name in old:
            assert h == old[name], f'Frozen source drift: {name}'
    return hashes


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: torch.from_numpy(np.ascontiguousarray(archive[key])) for key in archive.files}


def prepare_banks():
    assert Path(sys.modules[C.region_bank.__module__].__file__).resolve() == ROOT / 'new/nca_inertial_wind_tunnel/tasks.py'
    train = load_npz(TRAIN_PATH)
    expected = read(ROOT / 'ADDRESSED_DELTA_PUBLICATION_MANIFEST.json')['data_sha256']['train']
    assert C.tensor_hash(train) == expected and len(train['x']) == 512
    tests = {size: C.region_bank(size, 32, seed) for size, seed in EVAL_SEEDS.items()}
    hashes = {'train': C.tensor_hash(train), **{f'evaluation{s}': C.tensor_hash(b) for s, b in tests.items()}}
    support = {}
    for size, data in tests.items():
        mask = data['changed'].bool() & (data['distance'] > 16) & (data['distance'] < 32)
        support[str(size)] = {'eligible_maps': int(mask.flatten(1).any(1).sum()), 'pixels': int(mask.sum())}
    assert support['32']['eligible_maps'] >= 16 and support['32']['pixels'] >= 100, 'Frozen map support unqualified'
    return train, tests, hashes, support


def model_for(seed, state=None):
    torch.manual_seed(seed)
    model = C.StreamingCell(streaming=True).cuda()
    if state is not None:
        model.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in model.parameters()) == 5033
    return model


def gradient_summary(model):
    groups = {'E': ('encoder',), 'F': ('f_in', 'f_out'), 'Q': ('q_in', 'q_out'), 'R': ('readout',)}
    values, counts = [], {}
    for name, modules in groups.items():
        params = [p for module in modules for p in getattr(model, module).parameters()]
        grads = [p.grad for p in params if p.grad is not None]
        counts[name] = {'connected_tensors': len(grads), 'parameter_tensors': len(params)}
        values.append(torch.stack([g.detach().square().sum() for g in grads]).sum().sqrt()
            if grads else torch.zeros((), device='cuda'))
    norms = torch.stack(values).cpu().tolist()
    return {name: {**counts[name], 'norm': float(norm)} for name, norm in zip(groups, norms)}


def expected_cadence(arm):
    dense = arm == 'dense_k8'
    return {'forward_steps': 64, 'loss_times': list(range(8, 65, 8)) if dense else [64],
        'loss_count': 8 if dense else 1, 'backward_calls': 8 if dense else 1,
        'interior_detach_boundaries': 0 if arm == 'terminal_k64' else 7,
        'credit_horizon': 64 if arm == 'terminal_k64' else 8, 'optimizer_steps_not_here': True}


def qualification(out):
    if out.exists():
        raise FileExistsError('New qualification directory required')
    C.setup_backend()
    train, tests, hashes, support = prepare_banks()
    source_hashes = sources()
    out.mkdir(parents=True)
    for name, bank in {'train': train, **{f'evaluation{s}': b for s, b in tests.items()}}.items():
        np.savez_compressed(out / f'{name}.npz', **{k: v.numpy() for k, v in bank.items()})
    batch = {k: v[:BATCH].cuda() for k, v in train.items()}
    reference = model_for(INIT_SEEDS[0])
    initial = C.cpu_tree(reference.state_dict())
    del reference
    initial_sha = C.tensor_hash(initial)
    states, final_losses, rows = {}, {}, []
    try:
        for arm in ARMS:
            gc.collect()
            torch.cuda.empty_cache()
            model = model_for(INIT_SEEDS[0], initial)
            optimizer = C.optimizer_for(model)
            before = C.tensor_hash(model.state_dict())
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            loss, state, cadence = T.backward_trajectory(model, batch, arm)
            torch.cuda.synchronize()
            elapsed = time.monotonic()-tick
            assert cadence == expected_cadence(arm)
            assert before == initial_sha == C.tensor_hash(model.state_dict())
            assert bool(torch.stack([torch.isfinite(v).all() for v in state]).all())
            grad = gradient_summary(model)
            assert (grad['E']['connected_tensors'] == 0) == (arm == 'terminal_k8')
            states[arm] = C.cpu_tree(state)
            final_losses[arm] = float(T.balanced_loss(model.logits(state), batch['y'], batch['mask']))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            assert bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all())
            if arm == 'terminal_k8':
                assert all(torch.equal(model.state_dict()[k].cpu(), initial[k])
                    for k in initial if k.startswith('encoder.'))
            rows.append({'arm': arm, 'status': 'PASS', 'seconds': elapsed,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'loss': float(loss),
                'terminal_loss': final_losses[arm], 'gradient_norm_before_clip': float(norm),
                'module_gradients': grad, 'cadence': cadence, 'optimizer_updates': 1})
            del model, optimizer, state
        differences = {arm: max(float((a-b).abs().max()) for a, b in
            zip(states[arm], states['terminal_k64'])) for arm in ARMS}
        assert all(v == 0. for v in differences.values()), 'Matched forward trajectories differ'
        assert len(set(final_losses.values())) == 1, 'Matched terminal scalar losses differ'
        # Actual evaluation batch/size: validate the cue-once interface without
        # inspecting task performance or using a formal endpoint.
        model = model_for(INIT_SEEDS[0], initial).eval()
        shapes = []
        with torch.no_grad():
            for size, bank in tests.items():
                x, flip = bank['x'].cuda(), bank['x_flip'].cuda()
                geometry = T.geometry_input(x)
                assert torch.equal(geometry, T.geometry_input(flip))
                assert not bool(geometry[:, 1:].any())
                a, b = model.initial(x), model.initial(flip)
                for _ in range(4):
                    a, b = model.step(a, geometry), model.step(b, geometry)
                assert bool(torch.stack([torch.isfinite(v).all() for v in (*a, *b)]).all())
                shapes.append(list(model.logits(a).shape))
                del x, flip, geometry, a, b
        result = {'status': 'PASS', 'protocol': PROTOCOL, 'config': config(),
            'source_sha256': source_hashes, 'data_sha256': hashes, 'map_support': support,
            'actual_shape_checks': shapes, 'arms': rows, 'forward_max_absolute_error': differences,
            'B_C_terminal_loss_identical': True, 'created_utc': C.now(),
            'scientific_endpoint_evaluated': False,
            'estimated_training_seconds': 4*UPDATES*sum(r['seconds'] for r in rows),
            'free_cuda_bytes_after_check': torch.cuda.mem_get_info()[0]}
        write(out / 'qualification.json', result)
        print(json.dumps({'status': 'PASS', 'qualification': relative(out / 'qualification.json'),
            'forward_max_absolute_error': differences,
            'estimated_training_minutes': result['estimated_training_seconds']/60,
            'peak_allocated_MiB': max(r['peak_allocated_bytes'] for r in rows)/2**20}), flush=True)
    except BaseException:
        write(out / 'qualification.json', {'status': 'ERROR', 'traceback': traceback.format_exc(),
            'completed_arm_checks': rows, 'source_sha256': source_hashes, 'data_sha256': hashes})
        raise


def save_stage(folder, model, optimizer, block, arm, update, curve):
    checkpoint = folder / f'checkpoints/u{update:03d}.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix('.pt.tmp')
    payload = {'protocol': PROTOCOL, 'block': block, 'arm': arm, 'completed_updates': update,
        'initialization_seed': INIT_SEEDS[block], 'schedule_seed': SCHEDULE_SEEDS[block],
        'state_dict': C.cpu_tree(model.state_dict()), 'optimizer_state_dict': C.cpu_tree(optimizer.state_dict()),
        'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all(),
        'curve_sha256': digest(curve)}
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    write(folder / f'curves/u{update:03d}.json', curve)
    write(Path(str(checkpoint)+'.complete.json'), {'complete': True, 'update': update,
        'checkpoint_sha256': C.sha(checkpoint), 'parameter_sha256': C.tensor_hash(model.state_dict()),
        'curve_sha256': digest(curve), 'optimizer_updates': update, 'formal_endpoint': update == 300})


def run(out, qualification_path):
    if out.exists():
        raise FileExistsError('New run directory required; existing evidence is read-only')
    C.setup_backend()
    check = read(qualification_path)
    assert check['status'] == 'PASS' and digest(check['config']) == digest(config())
    assert check['source_sha256'] == sources(), 'Qualification/source drift'
    bank_folder = qualification_path.parent
    train_cpu = load_npz(bank_folder / 'train.npz')
    tests = {s: load_npz(bank_folder / f'evaluation{s}.npz') for s in (32, 64)}
    hashes = {'train': C.tensor_hash(train_cpu), **{f'evaluation{s}': C.tensor_hash(b) for s, b in tests.items()}}
    assert hashes == check['data_sha256']
    out.mkdir(parents=True)
    manifest = {'protocol': PROTOCOL, 'config': config(), 'started_utc': C.now(),
        'source_sha256': check['source_sha256'], 'data_sha256': hashes,
        'qualification': relative(qualification_path), 'qualification_sha256': C.sha(qualification_path),
        'expected_trajectories': 12, 'runtime_limit_enforced': False, 'recurring_monitor': False,
        'pid': os.getpid(), 'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
        'backend': {'cudnn_benchmark': False, 'cudnn_deterministic': False,
            'cudnn_tf32': True, 'matmul_tf32': False}}
    write(out / 'manifest.json', manifest)
    for name in ('train', 'evaluation32', 'evaluation64'):
        target = out / f'banks/{name}.npz'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bank_folder / f'{name}.npz', target)
    for name in check['source_sha256']:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    rows, started = [], time.monotonic()
    P.report(out, rows, 'RUNNING')
    train = {k: v.cuda() for k, v in train_cpu.items()}
    current_model, folder = None, None
    try:
        for block, (seed, schedule_seed) in enumerate(zip(INIT_SEEDS, SCHEDULE_SEEDS)):
            generator = np.random.default_rng(schedule_seed)
            schedule = [generator.choice(512, BATCH, replace=False).tolist() for _ in range(UPDATES)]
            write(out / f'schedules/block{block:02d}.json', schedule)
            arm_order = ARMS[block % 3:] + ARMS[:block % 3]
            initial_hash = None
            for arm in arm_order:
                folder = out / f'block{block:02d}/{arm}'
                folder.mkdir(parents=True)
                model = model_for(seed)
                current_model = model
                current_hash = C.tensor_hash(model.state_dict())
                if initial_hash is None:
                    initial_hash = current_hash
                assert current_hash == initial_hash, 'Paired initialization mismatch'
                optimizer = C.optimizer_for(model)
                curve = []
                begin = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                write(out / 'status.json', {'status': 'RUNNING', 'phase': 'training', 'block': block,
                    'arm': arm, 'completed_updates': 0, 'completed_trajectories': len(rows),
                    'expected_trajectories': 12, 'updated_utc': C.now()})
                model.train()
                for update, ids in enumerate(schedule, 1):
                    tick = time.monotonic()
                    batch = C.subset(train, ids)
                    optimizer.zero_grad(set_to_none=True)
                    loss, state, cadence = T.backward_trajectory(model, batch, arm)
                    assert cadence == expected_cadence(arm)
                    assert bool(torch.stack([torch.isfinite(v).all() for v in state]).all())
                    grad = gradient_summary(model)
                    assert (grad['E']['connected_tensors'] == 0) == (arm == 'terminal_k8')
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    assert bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all())
                    torch.cuda.synchronize()
                    curve.append({'update': update, 'loss': float(loss), 'seconds': time.monotonic()-tick,
                        'gradient_norm_before_clip': float(norm), 'module_gradients': grad, **cadence})
                    if update % 25 == 0 or update == 1:
                        write(out / 'status.json', {'status': 'RUNNING', 'phase': 'training', 'block': block,
                            'arm': arm, 'completed_updates': update, 'completed_trajectories': len(rows),
                            'expected_trajectories': 12, 'updated_utc': C.now()})
                        print(json.dumps({'block': block, 'arm': arm, 'update': update,
                            'loss': float(loss), 'seconds': curve[-1]['seconds']}), flush=True)
                    if update in CHECKPOINTS:
                        save_stage(folder, model, optimizer, block, arm, update, curve)
                peak = torch.cuda.max_memory_allocated()
                training_seconds = time.monotonic()-begin
                write(out / 'status.json', {'status': 'RUNNING', 'phase': 'evaluation', 'block': block,
                    'arm': arm, 'completed_updates': 300, 'completed_trajectories': len(rows), 'updated_utc': C.now()})
                measured = E.evaluate(model, tests, folder / 'evaluation/u300')
                row = {'block': block, 'arm': arm, 'update': 300, 'completed_updates': 300,
                    'status': 'COMPLETE', 'initialization_seed': seed, 'schedule_seed': schedule_seed,
                    'initial_parameter_sha256': initial_hash, 'parameter_sha256': C.tensor_hash(model.state_dict()),
                    'training_seconds': training_seconds, 'training_peak_allocated_bytes': peak,
                    'gradient_clip_fraction': sum(r['gradient_norm_before_clip'] > 1 for r in curve)/UPDATES,
                    'evaluation': measured}
                write(folder / 'record.json', row)
                write(folder / 'complete.json', {'complete': True, 'record_sha256': C.sha(folder / 'record.json'),
                    'evaluation_sha256': C.sha(folder / 'evaluation/u300/summary.json'),
                    'curve_sha256': digest(curve), 'checkpoint_sha256': C.sha(folder / 'checkpoints/u300.pt')})
                rows.append(row)
                P.report(out, rows, 'RUNNING')
                del model, optimizer, state, batch
                current_model = None
                gc.collect()
                torch.cuda.empty_cache()
        assert all(C.sha(ROOT / name) == h for name, h in check['source_sha256'].items())
        P.report(out, rows, 'COMPLETE')
        write(out / 'status.json', {'status': 'COMPLETE', 'completed_trajectories': len(rows),
            'expected_trajectories': 12, 'finished_utc': C.now(), 'seconds': time.monotonic()-started})
    except BaseException as error:
        if current_model is not None and folder is not None:
            path = folder / 'partial_failure.pt'
            torch.save({'partial': True, 'protocol': PROTOCOL,
                'state_dict': C.cpu_tree(current_model.state_dict())}, path)
        P.report(out, rows, 'INCOMPLETE')
        write(out / 'status.json', {'status': 'ERROR', 'completed_trajectories': len(rows),
            'expected_trajectories': 12, 'error': repr(error), 'traceback': traceback.format_exc(), 'updated_utc': C.now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--out', required=True)
    parser.add_argument('--qualification')
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    base = ROOT / ('analyses' if args.check else 'runs')
    assert out.is_relative_to(base) and out != base
    if args.check:
        qualification(out)
    else:
        assert args.qualification
        run(out, (ROOT / args.qualification).resolve())


if __name__ == '__main__':
    main()
