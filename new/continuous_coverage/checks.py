"""Bounded CPU, CUDA Graph, and trace-replay qualification for coverage runtime."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import gc
from io import BytesIO
import time

import numpy as np
import torch


CPU_SEED = 96199
BANK_SEED = 103008
MODES = ('reset64x4', 'continuous256')
UPDATES = 3
ATOL = 1e-6
RTOL = 1e-5


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _assert_exact(left, right, label):
    """Assert recursively bitwise equality and name the first differing leaf."""
    if torch.is_tensor(left) and torch.is_tensor(right):
        a, b = left.detach().cpu(), right.detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            detail = ''
            if a.shape == b.shape and a.dtype == b.dtype and a.numel():
                if a.is_floating_point() or a.is_complex():
                    delta = (a - b).abs()
                    detail = f' (max_abs={float(delta.max())})'
            raise AssertionError(f'{label}: tensor differs{detail}')
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
            raise AssertionError(f'{label}: array differs')
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise AssertionError(f'{label}: dictionary keys differ')
        for key in left:
            _assert_exact(left[key], right[key], f'{label}.{key}')
        return
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f'{label}: sequence structure differs')
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_exact(a, b, f'{label}[{index}]')
        return
    if left != right:
        raise AssertionError(f'{label}: {left!r} != {right!r}')


def _snapshot_parameters(model):
    return {name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()}


def _randomize_fq(model):
    with torch.no_grad():
        for module_name in ('f_out', 'q_out'):
            module = getattr(model, module_name)
            module.weight.normal_(std=.03)
            module.bias.normal_(std=.03)


def _loss_function(C):
    # Bind the same balanced BCE used by the historical 64-step K8 routine,
    # without importing a generic top-level `tasks` module into this checker.
    value = C.backward_trajectory.__globals__.get('balanced_loss')
    if value is None:
        raise RuntimeError('Historical balanced_loss is unavailable from common')
    return value


def _cpu_mode_check(C, R, base, data, loss_fn, mode):
    model = copy.deepcopy(base)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 5033:
        raise AssertionError(f'Expected original 5033-parameter StreamingCell, got {parameter_count}')

    counts = {
        'initial_calls': 0,
        'forward_steps': 0,
        'loss_calls': 0,
        'backward_calls': 0,
        'detached_window_inputs': [],
        'reset_segment_inputs_require_grad': [],
        'initial_workspace_requires_grad': [],
    }
    original_initial, original_step = model.initial, model.step

    def counted_initial(x):
        counts['initial_calls'] += 1
        state = original_initial(x)
        counts['initial_workspace_requires_grad'].append(bool(state[0].requires_grad))
        return state

    def counted_step(state, x):
        index = counts['forward_steps']
        if index and index % 8 == 0:
            if mode == 'reset64x4' and index % 64 == 0:
                counts['reset_segment_inputs_require_grad'].append(bool(state[0].requires_grad))
            else:
                counts['detached_window_inputs'].append(
                    all(not value.requires_grad and value.grad_fn is None for value in state)
                )
        counts['forward_steps'] += 1
        return original_step(state, x)

    def counted_loss(logits, y, mask):
        counts['loss_calls'] += 1
        return loss_fn(logits, y, mask)

    def backward_hook(gradient):
        counts['backward_calls'] += 1
        return gradient

    model.initial = counted_initial
    model.step = counted_step
    hook = model.readout.bias.register_hook(backward_hook)
    theta_before = _snapshot_parameters(model)
    try:
        total, state, finite = R.math_backward(model, data, mode, counted_loss)
    finally:
        hook.remove()
        model.initial = original_initial
        model.step = original_step

    if not bool(finite):
        raise FloatingPointError(f'{mode}: nonfinite CPU loss or state')
    expected_initials = 4 if mode == 'reset64x4' else 1
    expected_detach_inputs = 28 if mode == 'reset64x4' else 31
    expected_reset_starts = 3 if mode == 'reset64x4' else 0
    expected = {
        'initial_calls': expected_initials,
        'forward_steps': 256,
        'loss_calls': 32,
        'backward_calls': 32,
    }
    observed = {key: counts[key] for key in expected}
    if observed != expected:
        raise AssertionError(f'{mode}: cadence mismatch: expected {expected}, got {observed}')
    if len(counts['detached_window_inputs']) != expected_detach_inputs or not all(counts['detached_window_inputs']):
        raise AssertionError(f'{mode}: an 8-step carried state was not detached')
    if len(counts['reset_segment_inputs_require_grad']) != expected_reset_starts or not all(counts['reset_segment_inputs_require_grad']):
        raise AssertionError(f'{mode}: a reset segment did not start from model.initial(x)')
    if not all(counts['initial_workspace_requires_grad']):
        raise AssertionError(f'{mode}: initial workspace lost its encoder gradient path')
    if not all(not value.requires_grad and value.grad_fn is None for value in state):
        raise AssertionError(f'{mode}: returned recurrent state was not detached')
    _assert_exact(theta_before, _snapshot_parameters(model), f'{mode}.theta_unchanged')

    grad_by_name = {name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()}
    for name in ('f_out.weight', 'q_out.weight'):
        if name not in grad_by_name or not bool((grad_by_name[name] != 0).any()):
            raise AssertionError(f'{mode}: nonzero output weights did not receive gradient at {name}')

    return {
        'status': 'PASS',
        'parameter_count': parameter_count,
        **observed,
        'checked_interior_detached_boundaries': len(counts['detached_window_inputs']),
        'reset_segment_starts': len(counts['reset_segment_inputs_require_grad']),
        'returned_state_detached': True,
        'theta_unchanged': True,
        'f_out_and_q_out_gradients_nonzero': True,
        'mean_window_loss': float(total.detach()),
    }, total.detach().clone(), grad_by_name


def _close_report(actual, reference, label):
    actual_cpu = actual.detach().cpu()
    reference_cpu = reference.detach().cpu()
    difference = (actual_cpu - reference_cpu).abs()
    allowed = ATOL + RTOL * reference_cpu.abs()
    failures = int((difference > allowed).sum())
    if failures:
        raise AssertionError(
            f'{label}: {failures}/{actual_cpu.numel()} elements exceed '
            f'atol={ATOL}, rtol={RTOL}; max_abs={float(difference.max())}'
        )
    return float(difference.max()) if difference.numel() else 0.0


def _historical_reset_average_check(C, base, data, loss_fn, reset_loss, reset_grads):
    reference = copy.deepcopy(base)
    theta_before = _snapshot_parameters(reference)
    reference.zero_grad(set_to_none=True)
    segment_losses = []
    cadences = []
    for _ in range(4):
        loss, _state, cadence = C.backward_trajectory(
            reference, data, 8, steps=64, loss_every=8
        )
        segment_losses.append(loss.detach().clone())
        cadences.append(cadence)

    expected_cadence = {
        'backward_calls': 8,
        'interior_detach_boundaries': 7,
        'forward_steps': 64,
        'loss_count': 8,
    }
    if cadences != [expected_cadence] * 4:
        raise AssertionError(f'Historical 64 K8 cadence changed: {cadences!r}')
    _assert_exact(theta_before, _snapshot_parameters(reference), 'historical_64x4.theta_unchanged')

    reference_loss = torch.stack(segment_losses).mean()
    loss_max_abs = _close_report(reset_loss, reference_loss, 'reset256.loss_vs_mean_64k8')
    gradient_max_abs = 0.0
    gradient_count = 0
    for name, parameter in reference.named_parameters():
        if parameter.grad is None or name not in reset_grads:
            raise AssertionError(f'Historical/reset gradient missing at {name}')
        averaged = parameter.grad.detach() / 4
        gradient_max_abs = max(
            gradient_max_abs,
            _close_report(reset_grads[name], averaged, f'reset256.gradient_vs_mean_64k8.{name}'),
        )
        gradient_count += averaged.numel()

    return {
        'status': 'PASS',
        'historical_segments': 4,
        'steps_per_segment': 64,
        'k8_losses_per_segment': 8,
        'loss_aggregation': 'mean of four historical 64-step K8 means',
        'gradient_aggregation': 'mean of four historical 64-step K8 accumulated gradients',
        'atol': ATOL,
        'rtol': RTOL,
        'loss_max_abs_difference': loss_max_abs,
        'gradient_max_abs_difference': gradient_max_abs,
        'gradient_elements_checked': gradient_count,
        'theta_unchanged': True,
    }


def _cuda_equivalence_check(C, R, data, loss_fn):
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    device_clock = {
        'device_index': device_index,
        'device_name': torch.cuda.get_device_name(device_index),
        'reported_sm_clock_rate_khz': getattr(properties, 'clock_rate', None),
        'total_memory_bytes': int(properties.total_memory),
        'started_utc': _utc_now(),
    }
    mode_results = {}
    batches = {1: data}
    batch_seeds = {1: BANK_SEED}
    for update in range(2, UPDATES + 1):
        seed = BANK_SEED + update
        batches[update] = C.region_bank(32, 8, seed, 'cuda')
        batch_seeds[update] = seed

    for mode in MODES:
        eager_model = C.initial(CPU_SEED)
        graph_model = C.initial(CPU_SEED)
        _assert_exact(C.cpu_tree(eager_model.state_dict()),
                      C.cpu_tree(graph_model.state_dict()), f'{mode}.initial_theta')
        eager_optimizer = C.optimizer_for(eager_model)
        graph_optimizer = C.optimizer_for(graph_model)
        graph_theta_before = C.cpu_tree(graph_model.state_dict())
        graph_adam_before = C.cpu_tree(graph_optimizer.state_dict())

        torch.cuda.synchronize()
        allocated_before = int(torch.cuda.memory_allocated(device_index))
        torch.cuda.reset_peak_memory_stats(device_index)
        capture = R.CapturedSuperK8(graph_model, data, mode, loss_fn)
        torch.cuda.synchronize()
        _assert_exact(graph_theta_before, C.cpu_tree(graph_model.state_dict()), f'{mode}.capture_theta_unchanged')
        _assert_exact(graph_adam_before, C.cpu_tree(graph_optimizer.state_dict()), f'{mode}.capture_no_optimizer_step')

        eager_seconds, graph_seconds = [], []
        for update in range(1, UPDATES + 1):
            update_data = batches[update]
            torch.cuda.synchronize()
            started = time.perf_counter()
            eager_optimizer.zero_grad(set_to_none=True)
            eager_loss, eager_state, eager_finite = R.math_backward(
                eager_model, update_data, mode, loss_fn
            )
            torch.cuda.synchronize()
            eager_seconds.append(time.perf_counter() - started)

            torch.cuda.synchronize()
            started = time.perf_counter()
            graph_loss, graph_state, graph_finite = capture.run(update_data)
            torch.cuda.synchronize()
            graph_seconds.append(time.perf_counter() - started)

            if not bool(eager_finite) or not bool(graph_finite):
                raise FloatingPointError(f'{mode} update {update}: nonfinite eager/graph loss or state')
            _assert_exact(eager_loss, graph_loss, f'{mode}.u{update}.loss')
            _assert_exact(eager_state, graph_state, f'{mode}.u{update}.final_state')
            eager_grads = {name: parameter.grad for name, parameter in eager_model.named_parameters()}
            graph_grads = {name: parameter.grad for name, parameter in graph_model.named_parameters()}
            _assert_exact(C.cpu_tree(eager_grads), C.cpu_tree(graph_grads), f'{mode}.u{update}.gradients')

            eager_norm = R.finish_update(eager_model, eager_optimizer, eager_finite)
            graph_norm = R.finish_update(graph_model, graph_optimizer, graph_finite)
            _assert_exact(eager_norm, graph_norm, f'{mode}.u{update}.clip_norm')
            _assert_exact(C.cpu_tree(eager_model.state_dict()),
                          C.cpu_tree(graph_model.state_dict()), f'{mode}.u{update}.parameters')
            _assert_exact(C.cpu_tree(eager_optimizer.state_dict()),
                          C.cpu_tree(graph_optimizer.state_dict()), f'{mode}.u{update}.adam')

        mode_results[mode] = {
            'status': 'PASS',
            'updates': UPDATES,
            'batch_seed_by_update': {str(key): value for key, value in batch_seeds.items()},
            'forward_steps_per_update': 256,
            'backward_windows_per_update': 32,
            'capture_warmup_updates_with_optimizer': 0,
            'loss_gradients_parameters_adam_bitwise_equal': True,
            'capture_setup_seconds': float(capture.setup_seconds),
            'eager_update_seconds': [float(value) for value in eager_seconds],
            'graph_update_seconds': [float(value) for value in graph_seconds],
            'eager_total_seconds': float(sum(eager_seconds)),
            'graph_replay_total_seconds': float(sum(graph_seconds)),
            'memory_allocated_before_capture_bytes': allocated_before,
            'memory_allocated_after_checks_bytes': int(torch.cuda.memory_allocated(device_index)),
            'memory_peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device_index)),
            'memory_peak_reserved_bytes': int(torch.cuda.max_memory_reserved(device_index)),
        }

        del capture, eager_model, graph_model, eager_optimizer, graph_optimizer
        gc.collect()
        torch.cuda.empty_cache()

    device_clock['finished_utc'] = _utc_now()
    return {'status': 'PASS', 'clock': device_clock, 'modes': mode_results}


def _trace_replay_check(C, E):
    started = time.perf_counter()
    model = C.initial(CPU_SEED)
    _randomize_fq(model)
    model.eval()
    if not bool((model.f_out.weight != 0).any()) or not bool((model.q_out.weight != 0).any()):
        raise AssertionError('Trace replay model must have nonzero F/Qout weights')
    theta_before = C.cpu_tree(model.state_dict())
    data = C.region_bank(8, 2, BANK_SEED, 'cuda')

    new_values, new_records = E.trace(model, data, 8)
    audit = C._load_audit()
    old_budget = audit.budget
    try:
        audit.budget = lambda: None
        old_values, old_records = audit.trace(model, data, 8)
    finally:
        audit.budget = old_budget

    keys = ('correct', 'original_correct', 'flipped_correct')
    for key in keys:
        _assert_exact(new_values[key], old_values[key], f'trace_replay.{key}')
    if set(new_records) != {'64', '128', '256'}:
        raise AssertionError('New trace should retain only its three fixed endpoint records')
    expected_old_records = {str(step) for step in range(8, 257, 8)}
    if set(old_records) != expected_old_records:
        raise AssertionError('Trace endpoint record keys differ')
    for time_key in ('64', '128', '256'):
        for branch in ('original', 'flipped'):
            actual = new_records[time_key][branch]['balanced_accuracy']
            expected = old_records[time_key][branch]['balanced_accuracy']
            if actual != expected:
                raise AssertionError(
                    f'trace_replay.{time_key}.{branch}.balanced_accuracy differs: '
                    f'{actual!r} != {expected!r}'
                )
    _assert_exact(theta_before, C.cpu_tree(model.state_dict()), 'trace_replay.theta_unchanged')

    packed_roundtrip = True
    with BytesIO() as packed_buffer:
        E.pack(packed_buffer, new_values)
        packed_buffer.seek(0)
        with np.load(packed_buffer, allow_pickle=False) as archive:
            for key in keys:
                shape = tuple(int(value) for value in archive[key + '_shape'])
                count = int(np.prod(shape, dtype=np.int64))
                restored = np.unpackbits(archive[key + '_packed'], bitorder='little')[:count]
                restored = restored.reshape(shape).astype(bool, copy=False)
                if not np.array_equal(restored, new_values[key]):
                    packed_roundtrip = False
                    raise AssertionError(f'trace bitpack round-trip differs for {key}')

    return {
        'status': 'PASS',
        'size': 8,
        'maps': 2,
        'steps_including_initial_state': 257,
        'nonzero_f_out_and_q_out': True,
        'boolean_traces_equal': list(keys),
        'endpoint_balanced_accuracy_equal': True,
        'bitpack_roundtrip_equal': packed_roundtrip,
        'theta_unchanged': True,
        'elapsed_seconds': float(time.perf_counter() - started),
    }


def _section(call, *args):
    try:
        return call(*args)
    except Exception as error:
        return {
            'status': 'FAIL',
            'error_type': type(error).__name__,
            'error': str(error),
        }


def qualify(C, R, E, M) -> dict:
    """Run bounded runtime qualification; caller prepares the backend once.

    This covers arithmetic/control flow on CPU, eager-versus-graph update
    equivalence at the actual 8x32 shape, and the compact trace/bitpack replay.
    It does not run formal training or a full evaluation suite. ``M`` is
    accepted to keep the runner's module contract uniform; the checks compare
    raw tensors and recorded endpoint metrics directly.
    """
    started_utc = _utc_now()
    loss_fn = _section(_loss_function, C)
    if not callable(loss_fn):
        return {
            'status': 'FAIL',
            'started_utc': started_utc,
            'finished_utc': _utc_now(),
            'loss_function': loss_fn,
        }

    cpu_result = _section(_cpu_checks, C, R, loss_fn)
    cuda_result = _section(_cuda_checks, C, R, loss_fn)
    trace_result = _section(_trace_replay_check, C, E)
    sections = {'cpu': cpu_result[0] if isinstance(cpu_result, tuple) else cpu_result,
                'cuda_graph': cuda_result, 'trace_replay': trace_result}
    passed = all(value.get('status') == 'PASS' for value in sections.values())
    return {
        'status': 'PASS' if passed else 'FAIL',
        'started_utc': started_utc,
        'finished_utc': _utc_now(),
        'seed': CPU_SEED,
        'bank_seed': BANK_SEED,
        'updates_per_mode': UPDATES,
        'modes': list(MODES),
        'sections': sections,
    }


def _cpu_checks(C, R, loss_fn):
    torch.manual_seed(CPU_SEED)
    base = C.StreamingCell()
    _randomize_fq(base)
    parameter_count = sum(parameter.numel() for parameter in base.parameters())
    if parameter_count != 5033:
        raise AssertionError(f'Expected 5033 CPU parameters, got {parameter_count}')
    data = C.region_bank(8, 2, BANK_SEED, 'cpu')

    reset_report, reset_loss, reset_grads = _cpu_mode_check(
        C, R, base, data, loss_fn, 'reset64x4'
    )
    continuous_report, _continuous_loss, _continuous_grads = _cpu_mode_check(
        C, R, base, data, loss_fn, 'continuous256'
    )
    historical = _historical_reset_average_check(
        C, base, data, loss_fn, reset_loss, reset_grads
    )
    return {
        'status': 'PASS',
        'parameter_count': parameter_count,
        'data_shape': list(data['x'].shape),
        'modes': {'reset64x4': reset_report, 'continuous256': continuous_report},
        'reset256_vs_historical_64k8_average': historical,
    }


def _cuda_checks(C, R, loss_fn):
    data = C.region_bank(32, 8, BANK_SEED, 'cuda')
    return _cuda_equivalence_check(C, R, data, loss_fn)
