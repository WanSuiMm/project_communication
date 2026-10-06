"""Eager and CUDA Graph arithmetic for matched 256-step coverage updates."""
from __future__ import annotations

import time
from collections.abc import Callable

import torch


SUPER_UPDATE_STEPS = 256
WINDOW_STEPS = 8
LOSS_WINDOWS = SUPER_UPDATE_STEPS // WINDOW_STEPS
RESET_SEGMENT_STEPS = 64
MODES = ('reset64x4', 'continuous256')


def math_backward(model, data, mode, loss_fn: Callable[..., torch.Tensor]):
    """Run one fixed-parameter 256-step update and accumulate window gradients.

    ``loss_fn`` receives ``(logits, y, mask)`` and must return a scalar tensor.
    Each window contributes one thirty-second of its loss. The state is detached
    after every backward, while its numerical value carries into the next
    window. In reset mode that value is replaced with ``model.initial(x)`` at
    the start of each 64-step segment.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown coverage mode: {mode!r}')

    state = None
    total = None
    finite_terms = []

    for step in range(SUPER_UPDATE_STEPS):
        if state is None or (mode == 'reset64x4' and step % RESET_SEGMENT_STEPS == 0):
            state = model.initial(data['x'])
            finite_terms.extend(torch.isfinite(value).all() for value in state)
            if total is None:
                total = state[0].new_zeros(())

        state = model.step(state, data['x'])

        if (step + 1) % WINDOW_STEPS == 0:
            loss = loss_fn(model.logits(state), data['y'], data['mask']) / LOSS_WINDOWS
            total = total + loss.detach()
            loss.backward()

            # Keep every loss and every 8-step boundary state in the finite
            # gate using device operations only, so this also captures safely.
            finite_terms.append(torch.isfinite(loss).all())
            finite_terms.extend(torch.isfinite(value).all() for value in state)
            state = tuple(value.detach() for value in state)

    finite = torch.stack(finite_terms).all()
    return total, state, finite


class CapturedSuperK8:
    """Capture one 256-step update with 32 backward windows on a CUDA Graph."""

    def __init__(self, model, data, mode, loss_fn: Callable[..., torch.Tensor]):
        if mode not in MODES:
            raise ValueError(f'Unknown coverage mode: {mode!r}')
        self.model = model
        self.mode = mode
        self.loss_fn = loss_fn
        self.static = {key: data[key].clone() for key in ('x', 'y', 'mask')}
        started = time.monotonic()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                math_backward(model, self.static, mode, loss_fn)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        # Set-to-None once before capture so autograd allocates the graph's
        # persistent gradient buffers. Replays clear these buffers in place.
        model.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.state, self.finite = math_backward(
                model, self.static, mode, loss_fn
            )
        torch.cuda.synchronize()
        self.setup_seconds = time.monotonic() - started

        parameters = list(model.parameters())
        if any(parameter.grad is None for parameter in parameters):
            raise RuntimeError('A model parameter is disconnected from the captured loss')
        self.grad_addresses = tuple(parameter.grad.data_ptr() for parameter in parameters)

    def run(self, data):
        parameters = list(self.model.parameters())
        addresses = tuple(
            None if parameter.grad is None else parameter.grad.data_ptr()
            for parameter in parameters
        )
        if addresses != self.grad_addresses:
            raise RuntimeError('Captured gradient storage changed after graph setup')

        for key, value in self.static.items():
            value.copy_(data[key])
        for parameter in parameters:
            parameter.grad.zero_()

        self.graph.replay()
        return self.loss, self.state, self.finite


def finish_update(model, optimizer, finite):
    """Validate one complete update, clip its accumulated gradient, and step."""
    if not bool(finite):
        raise FloatingPointError('Nonfinite window loss or boundary state')

    if any(parameter.grad is None for parameter in model.parameters()):
        raise RuntimeError('A model parameter has no accumulated gradient')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Nonfinite accumulated gradient')

    optimizer.step()
    parameters_finite = torch.stack([
        torch.isfinite(parameter).all() for parameter in model.parameters()
    ]).all()
    if not bool(parameters_finite):
        raise FloatingPointError('Nonfinite parameter after optimizer step')
    return norm
