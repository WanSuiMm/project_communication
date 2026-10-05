"""Replay block00 through the captured K8 engine and require exact u300 state."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / 'new/latent_width/run.py'
RUNTIME_PATH = Path(__file__).with_name('runtime.py')


def local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Both names are private to this utility so loading the qualifier cannot reuse
# a cells module imported by another runner in the same Python process.
R = local_module('_latent_graph_qualify_runner', RUNNER_PATH)
G = local_module('_latent_graph_qualify_runtime', RUNTIME_PATH)
C = R.C

PROTOCOL = 'native_latent_graph_exact300_v1'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_binding():
    values = dict(R.source_hashes())
    values['new/latent_runtime/runtime.py'] = C.sha(RUNTIME_PATH)
    values['new/latent_runtime/qualify.py'] = C.sha(__file__)
    return dict(sorted(values.items()))


def digest_tree(value):
    digest = hashlib.sha256()

    def add(node, path):
        if torch.is_tensor(node):
            tensor = node.detach().cpu().contiguous()
            array = tensor.numpy()
            digest.update(path.encode('utf-8'))
            digest.update(str((tuple(array.shape), array.dtype)).encode('ascii'))
            digest.update(array.tobytes())
        elif isinstance(node, dict):
            digest.update((path + ':dict').encode('utf-8'))
            for key in sorted(node, key=lambda item: repr(item)):
                add(node[key], f'{path}/{key!r}')
        elif isinstance(node, (tuple, list)):
            digest.update((path + ':' + type(node).__name__).encode('utf-8'))
            for index, child in enumerate(node):
                add(child, f'{path}/{index}')
        else:
            digest.update(path.encode('utf-8'))
            digest.update(json.dumps(node, sort_keys=True, allow_nan=False).encode('utf-8'))

    add(value, '$')
    return digest.hexdigest()


def compare_tree(actual, expected):
    result = {'equal': True, 'tensor_leaves': 0, 'mismatches': []}

    def mismatch(path):
        result['equal'] = False
        if len(result['mismatches']) < 50:
            result['mismatches'].append(path)

    def visit(left, right, path):
        if torch.is_tensor(left) or torch.is_tensor(right):
            result['tensor_leaves'] += 1
            if not (torch.is_tensor(left) and torch.is_tensor(right)):
                mismatch(path)
                return
            if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
                mismatch(path)
                return
            if not torch.equal(left.detach().cpu(), right.detach().cpu()):
                mismatch(path)
            return
        if isinstance(left, dict) or isinstance(right, dict):
            if not (isinstance(left, dict) and isinstance(right, dict)) or set(left) != set(right):
                mismatch(path)
                return
            for key in sorted(left, key=lambda item: repr(item)):
                visit(left[key], right[key], f'{path}/{key!r}')
            return
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                mismatch(path)
                return
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f'{path}/{index}')
            return
        if type(left) is not type(right) or left != right:
            mismatch(path)

    visit(actual, expected, '$')
    result['mismatches_truncated'] = not result['equal'] and len(result['mismatches']) == 50
    return result


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def qualify(original, out):
    original = Path(original).resolve()
    out = Path(out).resolve()
    assert original.is_relative_to(ROOT / 'runs'), 'Original must be an existing project run'
    assert out.is_relative_to(ROOT / 'analyses') and out.suffix == '.json'
    assert not out.exists(), 'Qualification output already exists'
    assert original.is_dir(), f'Missing original run: {original}'

    C.setup_backend()
    frozen_sources = R.source_hashes()
    manifest_path = original / 'manifest.json'
    plans_path = original / 'plans.json'
    config_path = original / 'config.json'
    manifest = read_json(manifest_path)
    plans = read_json(plans_path)
    config = read_json(config_path)
    assert manifest['protocol'] == 'native_latent_width_v1'
    assert manifest['source_sha256'] == frozen_sources, 'Original frozen source binding changed'
    assert config['protocol'] == 'native_latent_width_v1'
    assert config['train_bank_seed'] == 10002 and config['train_maps'] == 512
    assert config['updates'] == 300 and config['batch'] == 8
    plan = plans['0']
    arms = list(plan['arm_order'])
    assert set(arms) == set(R.ARM_NAMES) and len(arms) == 6
    assert len(plan['batch_indices']) == 300

    rows_path = original / 'perarm.json'
    prior_rows = read_json(rows_path)
    block_rows = [row for row in prior_rows if row['block'] == 0]
    references = {row['arm']: row for row in block_rows}
    assert len(block_rows) == 6
    assert set(references) == set(R.ARM_NAMES), 'Block00 does not have all six completed arms'
    checkpoint_paths = {}
    for arm in arms:
        folder = original / 'block00' / arm
        initial_path = folder / 'initial.pt'
        final_path = folder / 'final_u300.pt'
        assert initial_path.is_file() and final_path.is_file(), f'Missing completed block00 files for {arm}'
        assert references[arm]['checkpoint'] == f'block00/{arm}/final_u300.pt'
        checkpoint_paths[arm] = (initial_path, final_path)

    train = C.region_bank(32, 512, 10002, 'cuda')
    data_sha256 = C.tensor_hash(train)
    source_sha256 = source_binding()
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    per_arm = []

    out.parent.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        initial_path, final_path = checkpoint_paths[arm]
        reference_row = references[arm]
        row = {
            'arm': arm,
            'block': 0,
            'updates_requested': 300,
            'initial_file_sha256': C.sha(initial_path),
            'final_u300_file_sha256': C.sha(final_path),
            'expected_final_parameter_sha256': reference_row['final_parameter_sha256'],
        }
        model = None
        optimizer = None
        actual_model = None
        expected_model = None
        optimizer_actual = None
        optimizer_expected = None
        try:
            initial = load_checkpoint(initial_path)
            expected = load_checkpoint(final_path)
            assert initial['completed_updates'] == 0
            assert expected['completed_updates'] == 300
            assert initial['variant'] == initial['arm'] == arm
            assert expected['variant'] == expected['arm'] == arm
            assert expected['completed_updates'] == reference_row.get('completed_updates', 300)
            assert C.tensor_hash(expected['state_dict']) == reference_row['final_parameter_sha256']

            model = R.model_for(arm, int(plan['initialization_seed']))
            model.load_state_dict(initial['state_dict'])
            replay_initial_hash = C.tensor_hash(model.state_dict())
            assert replay_initial_hash == C.tensor_hash(initial['state_dict'])
            assert replay_initial_hash == reference_row['initial_parameter_sha256']

            # Start with the original AdamW recipe in a genuinely empty state;
            # the saved initial checkpoint is used for parameters only.
            optimizer = C.optimizer_for(model)
            empty_optimizer_state = optimizer.state_dict()
            assert empty_optimizer_state['state'] == {}
            assert not initial['optimizer_state_dict']['state']

            started_arm = time.monotonic()
            trained = G.train_graph_updates(
                model,
                optimizer,
                train,
                plan['batch_indices'],
                start=1,
                progress=lambda event: print(json.dumps({
                    'arm': arm,
                    'completed_updates': event['completed_updates'],
                    'phase': event['phase'],
                }), flush=True),
            )
            curve = trained['curve']
            assert len(curve) == 300 and curve[0]['update'] == 1 and curve[-1]['update'] == 300
            assert all(row['forward_steps'] == 64 and row['backward_calls'] == 8
                       and row['interior_detach_boundaries'] == 7 and row['loss_count'] == 8
                       for row in curve)

            actual_model = model.state_dict()
            expected_model = expected['state_dict']
            model_comparison = compare_tree(actual_model, expected_model)
            replay_final_hash = C.tensor_hash(actual_model)
            optimizer_actual = optimizer.state_dict()
            optimizer_expected = expected['optimizer_state_dict']
            optimizer_state_comparison = compare_tree(
                optimizer_actual['state'], optimizer_expected['state'])
            optimizer_groups_comparison = compare_tree(
                optimizer_actual['param_groups'], optimizer_expected['param_groups'])
            completed_steps = [int(state['step'].item()) for state in optimizer.state.values()]
            clock_ok = len(completed_steps) == len(list(model.parameters())) and all(
                step == 300 for step in completed_steps)
            exact = (
                len(curve) == 300
                and model_comparison['equal']
                and replay_final_hash == reference_row['final_parameter_sha256']
                and optimizer_state_comparison['equal']
                and optimizer_groups_comparison['equal']
                and clock_ok
            )
            row.update({
                'status': 'PASS' if exact else 'FAIL',
                'replay_initial_parameter_sha256': replay_initial_hash,
                'replay_final_parameter_sha256': replay_final_hash,
                'replay_optimizer_state_sha256': digest_tree(optimizer_actual['state']),
                'expected_optimizer_state_sha256': digest_tree(optimizer_expected['state']),
                'replay_optimizer_param_groups_sha256': digest_tree(optimizer_actual['param_groups']),
                'expected_optimizer_param_groups_sha256': digest_tree(optimizer_expected['param_groups']),
                'model_tensor_leaves_compared': model_comparison['tensor_leaves'],
                'model_mismatch_paths': model_comparison['mismatches'],
                'optimizer_state_comparison': optimizer_state_comparison,
                'optimizer_param_groups_comparison': optimizer_groups_comparison,
                'optimizer_step_count': len(completed_steps),
                'optimizer_step_clocks_all_300': clock_ok,
                'capture_setup_seconds': trained['capture_setup_seconds'],
                'replay_training_seconds': time.monotonic() - started_arm,
                'runtime': trained['runtime'],
            })
        except Exception as error:
            row.update({
                'status': 'ERROR',
                'error': repr(error),
                'traceback': traceback.format_exc(),
            })
        finally:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            if 'initial' in locals():
                del initial
            if 'expected' in locals():
                del expected
            if 'trained' in locals():
                del trained
            if 'curve' in locals():
                del curve
            del actual_model, expected_model, optimizer_actual, optimizer_expected
            gc.collect()
            torch.cuda.empty_cache()
        per_arm.append(row)
        print(json.dumps({
            'arm': arm,
            'status': row['status'],
            'updates': 300 if row['status'] != 'ERROR' else 0,
            'exact': row['status'] == 'PASS',
        }), flush=True)

    frozen_runner_sources_unchanged = R.source_hashes() == frozen_sources
    source_binding_unchanged = source_binding() == source_sha256
    data_unchanged = C.tensor_hash(train) == data_sha256
    errors = any(row['status'] == 'ERROR' for row in per_arm)
    all_exact = len(per_arm) == 6 and all(row['status'] == 'PASS' for row in per_arm)
    status = 'ERROR' if (errors or not frozen_runner_sources_unchanged
                         or not source_binding_unchanged or not data_unchanged) else ('PASS' if all_exact else 'FAIL')
    reference_files = {}
    for arm, row in zip(arms, per_arm):
        folder = f"{original.relative_to(ROOT).as_posix()}/block00/{arm}"
        reference_files[f'{folder}/initial.pt'] = row['initial_file_sha256']
        reference_files[f'{folder}/final_u300.pt'] = row['final_u300_file_sha256']
    summary = {
        'protocol': PROTOCOL,
        'status': status,
        'exact300_all_arms': all_exact and frozen_runner_sources_unchanged and source_binding_unchanged and data_unchanged,
        'qualification_scope': 'block00 replay only; no efficacy decision or new cohort',
        'original_run': original.relative_to(ROOT).as_posix(),
        'original_protocol': manifest['protocol'],
        'source_sha256': source_sha256,
        'source_binding_unchanged_after_replay': source_binding_unchanged,
        'frozen_runner_source_unchanged_after_replay': frozen_runner_sources_unchanged,
        'original_manifest_sha256': C.sha(manifest_path),
        'original_config_sha256': C.sha(config_path),
        'original_plan_sha256': C.sha(plans_path),
        'training_data_sha256': data_sha256,
        'training_data': {'size': 32, 'maps': 512, 'seed': 10002, 'tensor_sha256': data_sha256,
                          'unchanged_after_replay': data_unchanged},
        'schedule': {'block': 0, 'initialization_seed': plan['initialization_seed'],
                     'schedule_seed': plan['schedule_seed'], 'updates': len(plan['batch_indices']),
                     'batch_size': config['batch'], 'sha256': C.sha(plans_path)},
        'references_sha256': reference_files,
        'backend': {
            'torch': str(torch.__version__),
            'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(),
            'numpy': str(np.__version__),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        },
        'started_utc': started_utc,
        'elapsed_seconds': time.monotonic() - started,
        'arms': per_arm,
        'no_evaluation_or_checkpoint_written': True,
        'claim_boundary': 'Exact replay qualification for six block00 trajectories only; no efficacy conclusion.',
    }
    C.write(out, summary)
    print(json.dumps({'status': status, 'output': out.relative_to(ROOT).as_posix(),
                      'exact300_all_arms': summary['exact300_all_arms']}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = qualify(args.original, args.out)
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
