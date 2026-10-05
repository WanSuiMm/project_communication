"""Isolated eager/batched-check/CUDA-Graph K8 runtime comparison."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('_runtime_latent_runner', ROOT / 'new/latent_width/run.py')
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
spec.loader.exec_module(R)
C = R.C


def math_backward(model, data):
    """Same operations/ordering as the frozen K8 trainer, without host checks."""
    state = model.initial(data['x'])
    total = state[0].new_zeros(())
    losses = []
    for t in range(1, 65):
        state = model.step(state, data['x'])
        if t % 8 == 0:
            loss = C.backward_trajectory.__globals__['balanced_loss'](
                model.logits(state), data['y'], data['mask']) / 8
            total = total + loss.detach()
            losses.append(loss.detach())
            loss.backward()
            state = tuple(v.detach() for v in state)
    finite = torch.stack([torch.isfinite(v) for v in losses]).all()
    for value in state[:2]:
        finite = finite & torch.isfinite(value).all()
    return total, state, finite


class GraphedBackward:
    def __init__(self, model, data):
        self.model = model
        self.static = {key: data[key].clone() for key in ('x', 'y', 'mask')}
        started = time.monotonic()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                math_backward(model, self.static)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.state, self.finite = math_backward(model, self.static)
        torch.cuda.synchronize()
        self.setup_seconds = time.monotonic()-started
        self.grad_addresses = [p.grad.data_ptr() for p in model.parameters()]

    def run(self, data):
        assert [p.grad.data_ptr() for p in self.model.parameters()] == self.grad_addresses
        for key, value in self.static.items():
            value.copy_(data[key])
        self.graph.replay()
        return self.loss, self.state, self.finite


def backward(model, data, kind, graph=None):
    if kind == 'graph':
        return graph.run(data)
    model.zero_grad(set_to_none=True)
    if kind == 'eager':
        loss, state, _ = C.backward_trajectory(model, data, 8)
        assert all(bool(torch.isfinite(v).all()) for v in state)
        return loss, state, None
    return math_backward(model, data)


def finish(model, optimizer, finite, kind):
    if finite is not None:
        assert bool(finite), 'Nonfinite trajectory'
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert bool(torch.isfinite(norm)), 'Nonfinite gradient'
    optimizer.step()
    if kind == 'eager':
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    else:
        assert bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all())
    return norm


def compare(actual, expected):
    assert len(actual) == len(expected)
    exact = all(torch.equal(a, b) for a, b in zip(actual, expected))
    maximum = max(float((a-b).abs().max()) for a, b in zip(actual, expected))
    relative = max(float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-30)) for a, b in zip(actual, expected))
    close = all(torch.allclose(a, b, atol=1e-7, rtol=1e-5) for a, b in zip(actual, expected))
    return {'bitwise_equal': exact, 'within_tolerance': close,
            'maximum_absolute_error': maximum, 'maximum_relative_l2_error': relative}


def moments(model, optimizer):
    return [optimizer.state[p][name] for p in model.parameters() for name in ('step', 'exp_avg', 'exp_avg_sq')]


def arm_bench(arm, train, schedules):
    initial = R.model_for(arm, 91981)
    with torch.no_grad():
        initial.f_out.weight.normal_(std=.01)
        initial.q_out.weight.normal_(std=.01)
    state_dict = C.cpu_tree(initial.state_dict())
    del initial
    models = {kind: R.model_for(arm, 91981) for kind in ('eager', 'batched_checks', 'graph')}
    for model in models.values():
        model.load_state_dict(state_dict)
    optimizers = {kind: C.optimizer_for(model) for kind, model in models.items()}
    capture_data = C.subset(train, schedules[0])
    graph = GraphedBackward(models['graph'], capture_data)
    validations = []
    for update, ids in enumerate(schedules[:6], 1):
        data = C.subset(train, ids)
        raw = {}
        for kind, model in models.items():
            loss, state, finite = backward(model, data, kind, graph)
            raw[kind] = {'loss': loss.detach().clone(),
                         'state': tuple(v.clone() for v in state),
                         'gradients': tuple(p.grad.clone() for p in model.parameters())}
            finish(model, optimizers[kind], finite, kind)
        for kind in ('batched_checks', 'graph'):
            reference, candidate = raw['eager'], raw[kind]
            checks = {
                'loss': compare([candidate['loss']], [reference['loss']]),
                'state': compare(candidate['state'], reference['state']),
                'preclip_gradients': compare(candidate['gradients'], reference['gradients']),
                'parameters': compare(list(models[kind].parameters()), list(models['eager'].parameters())),
                'adam': compare(moments(models[kind], optimizers[kind]), moments(models['eager'], optimizers['eager'])),
            }
            validations.append({'update': update, 'kind': kind, 'checks': checks})
        del raw, data
    # One diagnostic pass, saving aggregate counts instead of a large trace.
    profile_data = C.subset(train, schedules[6])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        loss, state, finite = backward(models['eager'], profile_data, 'eager')
        finish(models['eager'], optimizers['eager'], finite, 'eager')
        torch.cuda.synchronize()
    events = profile.key_averages()
    selected = [{'key': v.key, 'count': v.count, 'self_cpu_ms': v.self_cpu_time_total/1000,
                 'self_device_ms': getattr(v, 'self_device_time_total', 0.)/1000}
                for v in events if any(s in v.key for s in ('cudaLaunchKernel', 'cudaStreamSynchronize', 'cudaDeviceSynchronize', '_local_scalar_dense', 'cudaMemcpy'))]
    # Warmup and timing are disposable. Do not interpret their parameters as
    # efficacy checkpoints, including the eager-only profiler update above.
    times = {kind: [] for kind in models}
    order = tuple(models)
    for round_id in range(9):
        data = C.subset(train, schedules[7+round_id])
        offset = round_id % 3
        for kind in order[offset:] + order[:offset]:
            torch.cuda.synchronize()
            start = time.monotonic()
            loss, state, finite = backward(models[kind], data, kind, graph)
            finish(models[kind], optimizers[kind], finite, kind)
            torch.cuda.synchronize()
            seconds = time.monotonic()-start
            if round_id >= 3:
                times[kind].append(seconds)
    medians = {kind: statistics.median(values) for kind, values in times.items()}
    exact = {kind: all(c['bitwise_equal'] for row in validations if row['kind'] == kind for c in row['checks'].values())
             for kind in ('batched_checks', 'graph')}
    close = {kind: all(c['within_tolerance'] for row in validations if row['kind'] == kind for c in row['checks'].values())
             for kind in ('batched_checks', 'graph')}
    result = {'arm': arm, 'parameters': R.PARAMETERS[arm], 'batch': 8, 'size': 32,
              'credit_horizon': 8, 'execution_horizon': 64, 'validation_updates': 6,
              'capture_setup_seconds': graph.setup_seconds,
              'bitwise_trajectory_equal': exact, 'trajectory_within_tolerance': close,
              'validation': validations, 'timings_seconds': times,
              'median_update_seconds': medians,
              'speedup_vs_eager': {k: medians['eager']/v for k, v in medians.items()},
              'eager_profiler_selected_events': selected,
              'prototype_peak_cuda_bytes': torch.cuda.max_memory_allocated(),
              'compared_under_concurrent_formal_job': True}
    del graph, models, optimizers, profile, events, state, loss, finite
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    assert out.is_relative_to(ROOT / 'analyses') and not out.exists()
    out.mkdir(parents=True)
    C.setup_backend()
    frozen_sources = R.source_hashes()
    started = time.monotonic()
    config = {'question': 'Runtime overhead reduction at unchanged finite K8 math',
              'arms': ['w24', 'w2_alternating'], 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(), 'formal_job_modified': False,
              'concurrent_formal_job': 'runs/latent_width_20261005_01',
              'frozen_source_sha256': frozen_sources,
              'benchmark_source_sha256': {p.relative_to(ROOT).as_posix(): C.sha(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py', '.md')}}
    C.write(out / 'config.json', config)
    result = {'status': 'ERROR', 'arms': []}
    try:
        train = C.region_bank(32, 24, 99432, 'cuda')
        schedules = np.random.default_rng(99433).integers(0, 24, (24, 8)).tolist()
        for arm in config['arms']:
            torch.cuda.reset_peak_memory_stats()
            row = arm_bench(arm, train, schedules)
            result['arms'].append(row)
            C.write(out / f'{arm}.json', row)
            print(json.dumps({'arm': arm, 'medians': row['median_update_seconds'], 'speedup': row['speedup_vs_eager'],
                              'bitwise_equal': row['bitwise_trajectory_equal']}), flush=True)
        assert R.source_hashes() == frozen_sources, 'Formal source changed'
        result['status'] = 'COMPLETE'
    except Exception as error:
        C.write(out / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        result['error'] = repr(error)
    result.update({'elapsed_seconds': time.monotonic()-started, 'formal_job_modified': False,
                   'claim_boundary': 'Concurrent-job runtime pilot and finite six-update numerical equivalence, not efficacy or universal exact replay.'})
    C.write(out / 'summary.json', result)
    lines = ['# Isolated K8 runtime pilot', '', 'Execution: ' + result['status'], '',
             '| Arm | Eager ms/update | Batched checks speedup | CUDA Graph speedup | CUDA Graph exact6 |',
             '|---|---:|---:|---:|---|']
    for row in result['arms']:
        lines.append(f"| {row['arm']} | {1000*row['median_update_seconds']['eager']:.1f} | {row['speedup_vs_eager']['batched_checks']:.2f}x | {row['speedup_vs_eager']['graph']:.2f}x | {row['bitwise_trajectory_equal']['graph']} |")
    lines += ['', 'The formal job and its bound source files were not changed.',
              'Benchmark ran concurrently with that job; speedups are a pilot.',
              'Close floating point replay is not bitwise replay. Six updates do not certify a full300-update trajectory.']
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    if result['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
