"""Captured K8 arithmetic, checking every learned state (including R2)."""
from __future__ import annotations

from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'new/trajectory_qualification'))
import common as C

CADENCE = {'backward_calls': 8, 'interior_detach_boundaries': 7,
           'forward_steps': 64, 'loss_count': 8}


def math_backward(model, data):
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
    for value in state:
        finite = finite & torch.isfinite(value).all()
    return total, state, finite


class CapturedK8:
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
        self.setup_seconds = time.monotonic() - started
        assert all(p.grad is not None for p in model.parameters()), 'Disconnected parameter'
        self.grad_addresses = [p.grad.data_ptr() for p in model.parameters()]

    def run(self, data):
        assert [p.grad.data_ptr() for p in self.model.parameters()] == self.grad_addresses
        for key, value in self.static.items():
            value.copy_(data[key])
        self.graph.replay()
        return self.loss, self.state, self.finite


def finish_update(model, optimizer, finite):
    if not bool(finite):
        raise FloatingPointError('Nonfinite loss or learned final state')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Nonfinite gradient')
    optimizer.step()
    params_finite = torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
    if not bool(params_finite):
        raise FloatingPointError('Nonfinite parameter')
    return norm


def train_graph_updates(model, optimizer, train, batch_rows, progress=lambda _: None):
    model.train()
    first = C.subset(train, batch_rows[0])
    captured = CapturedK8(model, first)
    curve = []
    for update, ids in enumerate(batch_rows, 1):
        tick = time.monotonic()
        data = first if update == 1 else C.subset(train, ids)
        loss, state, finite = captured.run(data)
        norm = finish_update(model, optimizer, finite)
        torch.cuda.synchronize()
        row = {'update': update, 'mean_trajectory_loss': float(loss),
               'gradient_norm_before_clip': float(norm),
               'seconds': time.monotonic()-tick, **CADENCE}
        curve.append(row)
        if update == 1 or update % 25 == 0 or update == len(batch_rows):
            progress({'phase': 'training', 'completed_updates': update, **row})
        if update != 1:
            del data
    return {'curve': curve, 'capture_setup_seconds': captured.setup_seconds,
            'runtime': 'cuda_graph_k8'}
