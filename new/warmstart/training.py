"""Warm-start prefix training with the frozen eight-step BPTT cadence."""
from __future__ import annotations

import importlib.util
import operator
from pathlib import Path
import sys
from typing import Callable, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
WARMSTART_AGES = (0, 32, 64, 128, 192)
FORWARD_STEPS = 64
LOSS_EVERY = 8
BATCH_SIZE = 8
WARM_EXAMPLES = 4


def _add_project_import_paths() -> None:
    """Make this module usable from a root launcher or as a standalone check."""
    paths = (
        ROOT / "new/nca_inertial_wind_tunnel",
        ROOT / "new/workspace_revision",
        ROOT / "new/streaming_carry",
    )
    for path in paths:
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


_add_project_import_paths()


def _load_historical_training():
    """Load the frozen K8 implementation without colliding with other modules."""
    module_name = "historical_training"
    path = ROOT / "new/short_bptt/training.py"
    existing = sys.modules.get(module_name)
    if existing is not None and Path(existing.__file__).resolve() == path.resolve():
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load historical training helper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_HISTORICAL = _load_historical_training()


def _validate_inputs(data: Mapping[str, torch.Tensor], age: int, keep_warm_state: bool) -> int:
    try:
        age_steps = operator.index(age)
    except TypeError as error:
        raise ValueError("age must be one of 0, 32, 64, 128, or 192") from error
    if isinstance(age, bool) or age_steps not in WARMSTART_AGES:
        raise ValueError("age must be one of 0, 32, 64, 128, or 192")
    if not isinstance(keep_warm_state, bool):
        raise ValueError("keep_warm_state must be a bool")
    if not isinstance(data, Mapping) or not {"x", "y", "mask"}.issubset(data):
        raise ValueError("data must contain x, y, and mask tensors")
    if data["x"].ndim < 1 or data["x"].shape[0] != BATCH_SIZE:
        raise ValueError("warm-start trajectories require a batch of exactly 8 examples")
    for key in ("y", "mask"):
        if data[key].ndim < 1 or data[key].shape[0] != BATCH_SIZE:
            raise ValueError(f"data[{key!r}] must have the same 8-example batch")
    return age_steps


def _all_finite(state) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in state)


def _finite_flag(state) -> torch.Tensor:
    """Return a device-side scalar flag without synchronizing with the host."""
    flags = [torch.isfinite(value).all() for value in state]
    if not flags:
        raise ValueError("A model state must contain at least one tensor")
    result = flags[0]
    for flag in flags[1:]:
        result = result & flag
    return result


def _prefix_state(model, x: torch.Tensor, age: int,
                  budget: Callable[[], object]):
    """Compute one prefix under no-grad, checking its initial and stepped states."""
    with torch.no_grad():
        state = model.initial(x)
        if not _all_finite(state):
            raise FloatingPointError("Nonfinite warm prefix state")
        window_finite = None
        for step in range(1, age + 1):
            if (step - 1) % LOSS_EVERY == 0:
                budget()
            state = model.step(state, x)
            step_finite = _finite_flag(state)
            window_finite = (step_finite if window_finite is None
                             else window_finite & step_finite)
            if step % LOSS_EVERY == 0 or step == age:
                # Keep nonfinite observations sticky across the whole window,
                # even if a later state happens to become finite again.
                if not bool(window_finite):
                    raise FloatingPointError("Nonfinite warm prefix state")
                window_finite = None
    return tuple(value.detach() for value in state)


def _metadata(prefix_age: int, keep_warm_state: bool, trace: dict) -> dict:
    return {
        **trace,
        "prefix_age": prefix_age,
        "prefix_examples": WARM_EXAMPLES,
        "prefix_state_finite": True,
        "finite_prefix": True,
        "warm_state_kept": keep_warm_state,
        "gradient_horizon": LOSS_EVERY,
        "encoder_fresh_rows": BATCH_SIZE if not keep_warm_state else BATCH_SIZE - WARM_EXAMPLES,
    }


def _backward_mixed_trajectory(model, data, fresh_state, warm_state,
                               budget: Callable[[], object]):
    """Run the historical K8 loss/backward clock on fresh and detached states."""
    x = data["x"]
    state = tuple(torch.cat((fresh, warm.detach()), dim=0)
                  for fresh, warm in zip(fresh_state, warm_state))
    total = state[0].new_zeros(())
    backward_calls = cuts = 0
    loss_count = FORWARD_STEPS // LOSS_EVERY

    for step in range(1, FORWARD_STEPS + 1):
        if (step - 1) % LOSS_EVERY == 0:
            budget()
        state = model.step(state, x)
        if step % LOSS_EVERY == 0:
            loss = _HISTORICAL.balanced_loss(
                model.logits(state), data["y"], data["mask"]
            ) / loss_count
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite trajectory loss")
            total = total + loss.detach()
            loss.backward()
            backward_calls += 1
            state = tuple(value.detach() for value in state)
            if step < FORWARD_STEPS:
                cuts += 1

    trace = {
        "backward_calls": backward_calls,
        "interior_detach_boundaries": cuts,
        "forward_steps": FORWARD_STEPS,
        "loss_count": loss_count,
    }
    return total, tuple(value.detach() for value in state), trace


def backward_warmstart(model, data, age, keep_warm_state, budget=lambda: None):
    """Accumulate one matched 64-step K8 trajectory, optionally retaining a warm prefix.

    The last four examples always receive the same no-grad prefix computation.
    The discarded-state baseline then invokes the original K8 helper directly.
    The warm-state path combines four fresh, differentiable initial states with
    four detached prefix states. Parameters remain unchanged; the caller owns
    gradient clearing, clipping, and the optimizer step.
    """
    age_steps = _validate_inputs(data, age, keep_warm_state)
    if budget is None:
        budget = lambda: None

    warm_prefix = _prefix_state(model, data["x"][BATCH_SIZE - WARM_EXAMPLES:],
                                age_steps, budget)
    if not keep_warm_state:
        loss, state, trace = _HISTORICAL.backward_trajectory(
            model, data, LOSS_EVERY, steps=FORWARD_STEPS,
            loss_every=LOSS_EVERY, budget=budget,
        )
        return loss, state, _metadata(age_steps, False, trace)

    fresh_state = model.initial(data["x"][:BATCH_SIZE - WARM_EXAMPLES])
    loss, state, trace = _backward_mixed_trajectory(
        model, data, fresh_state, warm_prefix, budget
    )
    return loss, state, _metadata(age_steps, True, trace)

