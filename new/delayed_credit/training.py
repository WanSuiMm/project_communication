"""Matched 64-step trajectories with dense or delayed temporal credit."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TASKS_PATH = _PROJECT_ROOT / "new" / "nca_inertial_wind_tunnel" / "tasks.py"
_TASKS_SPEC = spec_from_file_location("_delayed_credit_frozen_tasks", _TASKS_PATH)
if _TASKS_SPEC is None or _TASKS_SPEC.loader is None:
    raise ImportError(f"Cannot load frozen task definitions from {_TASKS_PATH}")
_TASKS = module_from_spec(_TASKS_SPEC)
_TASKS_SPEC.loader.exec_module(_TASKS)
balanced_loss = _TASKS.balanced_loss


ARMS = ("dense_k8", "terminal_k8", "terminal_k64")
STEPS = 64
LOSS_EVERY = 8


def geometry_input(x: torch.Tensor) -> torch.Tensor:
    """Return the static geometry input: mask followed by two zero channels."""
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("x must have shape [batch, 3, height, width]")
    return torch.cat((x[:, :1], torch.zeros_like(x[:, 1:])), dim=1)


def backward_trajectory(model, data, arm):
    """Backpropagate one frozen 64-step trajectory for a delayed-credit arm.

    The initial encoder receives all three task channels once. Each recurrent
    update receives the same mask-only geometry tensor. Parameters stay fixed
    throughout the trajectory; the caller owns zero_grad, clipping, and step.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown delayed-credit arm: {arm}")

    x = data["x"]
    state = model.initial(x)
    x_geometry = geometry_input(x)
    total = state[0].new_zeros(())
    backward_calls = 0
    interior_detach_boundaries = 0
    loss_times = []

    for t in range(1, STEPS + 1):
        state = model.step(state, x_geometry)
        dense_boundary = arm == "dense_k8" and t % LOSS_EVERY == 0
        terminal_boundary = t == STEPS and arm != "dense_k8"

        if dense_boundary or terminal_boundary:
            loss = balanced_loss(model.logits(state), data["y"], data["mask"])
            if arm == "dense_k8":
                loss = loss / (STEPS // LOSS_EVERY)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite trajectory loss at step {t}")

            total = total + loss.detach()
            loss_times.append(t)
            loss.backward()
            backward_calls += 1

        should_cut = (
            t < STEPS
            and t % LOSS_EVERY == 0
            and arm in ("dense_k8", "terminal_k8")
        )
        if should_cut:
            state = tuple(value.detach() for value in state)
            interior_detach_boundaries += 1

    detached_state = tuple(value.detach() for value in state)
    cadence = {
        "forward_steps": STEPS,
        "loss_times": loss_times,
        "loss_count": len(loss_times),
        "backward_calls": backward_calls,
        "interior_detach_boundaries": interior_detach_boundaries,
        "credit_horizon": LOSS_EVERY if arm != "terminal_k64" else STEPS,
        "optimizer_steps_not_here": True,
    }
    return total.detach(), detached_state, cadence
