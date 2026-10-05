"""Captured CUDA Graph executor for the frozen native-width K8 update."""
from __future__ import annotations

from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'new/trajectory_qualification'))
import common as C


CADENCE = {
    'backward_calls': 8,
    'interior_detach_boundaries': 7,
    'forward_steps': 64,
    'loss_count': 8,
}


def math_backward(model, data):
    """Frozen K8 arithmetic with tensor-only finite checks for graph capture."""
    state = model.initial(data['x'])
    total = state[0].new_zeros(())
    losses = []
    balanced_loss = C.backward_trajectory.__globals__['balanced_loss']
    for t in range(1, 65):
        state = model.step(state, data['x'])
        if t % 8 == 0:
            loss = balanced_loss(model.logits(state), data['y'], data['mask']) / 8
            total = total + loss.detach()
            losses.append(loss.detach())
            loss.backward()
            state = tuple(value.detach() for value in state)
    finite = torch.stack([torch.isfinite(value) for value in losses]).all()
    for value in state[:2]:
        finite = finite & torch.isfinite(value).all()
    return total, state, finite


class CapturedK8:
    """One static-input, 64-step/eight-backward CUDA Graph for a model."""

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

        # The captured AccumulateGrad nodes own static grad buffers. Keep those
        # allocations attached for every replay; never clear them to None.
        model.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.state, self.finite = math_backward(model, self.static)
        torch.cuda.synchronize()
        self.setup_seconds = time.monotonic() - started
        self.grad_addresses = [parameter.grad.data_ptr() for parameter in model.parameters()]

    def run(self, data):
        assert [parameter.grad.data_ptr() for parameter in self.model.parameters()] == self.grad_addresses
        for key, value in self.static.items():
            value.copy_(data[key])
        self.graph.replay()
        return self.loss, self.state, self.finite


def train_graph_updates(model, optimizer, train, batchRows, start=1, progress=lambda _: None):
    """Train with captured K8 backward windows and the original eager update.

    Returns the original per-update curve schema, with capture setup time kept
    separately so callers can report it without charging it to update timings.
    """
    if len(batchRows) == 0:
        return {'curve': [], 'capture_setup_seconds': 0.0, 'runtime': 'cuda_graph_k8'}

    model.train()
    first_data = C.subset(train, batchRows[0])
    captured = CapturedK8(model, first_data)
    curve = []
    for update, ids in enumerate(batchRows, start):
        tick = time.monotonic()
        data = first_data if update == start else C.subset(train, ids)
        loss, state, finite = captured.run(data)
        if not bool(finite):
            raise FloatingPointError('Nonfinite trajectory loss or final state')

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        params_finite = torch.stack([
            torch.isfinite(parameter).all() for parameter in model.parameters()
        ]).all()
        if not bool(params_finite):
            raise FloatingPointError('Nonfinite parameter')
        torch.cuda.synchronize()

        row = {
            'update': update,
            'mean_trajectory_loss': float(loss),
            'gradient_norm_before_clip': float(norm),
            'seconds': time.monotonic() - tick,
            **CADENCE,
        }
        curve.append(row)
        if update == start or update % 25 == 0 or update == 300:
            progress({'phase': 'training', 'completed_updates': update, **row})

        if update != start:
            del data
    return {
        'curve': curve,
        'capture_setup_seconds': captured.setup_seconds,
        'runtime': 'cuda_graph_k8',
    }
