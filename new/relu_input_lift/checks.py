"""CPU algebra/autograd checks and a bounded CUDA qualification for ReLU lift.

The CUDA path measures forward parity at T=64 without retaining a graph, then
checks K=8 gradients.  It deliberately does not build a full-T lifted graph.
Training-update benchmarking lives in the owning runner.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new.relu_input_lift.cells import ReLUInputLiftCell as NCACell


MODES = ("original", "au", "relu_lift")


def _make_triplet(
    *,
    seed: int,
    channels: int,
    hidden: int,
    dtype: torch.dtype,
    device: str | torch.device,
    projection_scale: float | None,
    feature_bias_scale: float = 0.0,
) -> dict[str, NCACell]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        template = NCACell(channels=channels, hidden=hidden)
    template = template.to(device=device, dtype=dtype)
    models = {mode: copy.deepcopy(template) for mode in MODES}

    if projection_scale is not None or feature_bias_scale:
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        with torch.no_grad():
            if projection_scale is not None:
                projection = torch.randn(
                    template.projection.weight.shape,
                    generator=generator,
                    dtype=torch.float64,
                ) * projection_scale
                projection = projection.to(device=device, dtype=dtype)
                for model in models.values():
                    model.projection.weight.copy_(projection)
            if feature_bias_scale:
                bias = torch.linspace(
                    -feature_bias_scale,
                    feature_bias_scale,
                    hidden,
                    dtype=torch.float64,
                ).to(device=device, dtype=dtype)
                for model in models.values():
                    model.feature.bias.copy_(bias)
    return models


def _seed_visible(
    seed: int,
    batch: int,
    channels: int,
    height: int,
    width: int,
    alpha_channel: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    visible = 0.025 * torch.randn(
        (batch, channels, height, width), generator=generator, dtype=torch.float64
    )
    alpha_noise = 0.004 * torch.randn(
        (batch, 1, height, width), generator=generator, dtype=torch.float64
    )
    visible[:, alpha_channel : alpha_channel + 1] = 0.8 + alpha_noise
    return visible.to(device=device, dtype=dtype)


def _fire_schedule(
    seed: int,
    steps: int,
    batch: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: str | torch.device,
    probability: float = 0.5,
) -> list[Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.rand((steps, batch, 1, height, width), generator=generator)
    return [item.to(device=device, dtype=dtype) for item in (draws < probability)]


def _forced_alive_schedule(
    steps: int, batch: int, height: int, width: int, device: str | torch.device
) -> list[Tensor]:
    schedule: list[Tensor] = []
    for step in range(steps):
        alive = torch.ones((batch, 1, height, width), dtype=torch.bool)
        for batch_index in range(batch):
            alive[
                batch_index,
                0,
                (step + batch_index) % height,
                (2 * step + batch_index + 1) % width,
            ] = False
        schedule.append(alive.to(device=device))
    return schedule


def _max_abs(left: Tensor, right: Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left.detach() - right.detach()).abs().max().cpu())


def _max_relative(left: Tensor, right: Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    scale = max(
        float(left.detach().abs().max().cpu()),
        float(right.detach().abs().max().cpu()),
    )
    return _max_abs(left, right) / max(scale, 1e-30)


def _close_stats(
    left: Tensor, right: Tensor, *, atol: float, rtol: float
) -> dict[str, Any]:
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return {
            "max_abs_error": None,
            "max_relative_error": None,
            "atol": atol,
            "rtol": rtol,
            "finite": False,
            "passed": False,
        }
    return {
        "max_abs_error": _max_abs(left, right),
        "max_relative_error": _max_relative(left, right),
        "atol": atol,
        "rtol": rtol,
        "finite": True,
        "passed": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
    }


def _parameter_vector(
    model: NCACell, names: tuple[str, ...] | None = None
) -> tuple[list[str], Tensor]:
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if names is None or name in names
    ]
    if not selected:
        parameter = next(model.parameters())
        return [], torch.empty(0, dtype=parameter.dtype, device=parameter.device)
    values = [
        (
            parameter.grad
            if parameter.grad is not None
            else torch.zeros_like(parameter)
        ).detach().reshape(-1)
        for _, parameter in selected
    ]
    return [name for name, _ in selected], torch.cat(values)


def _parameter_comparison(
    left_model: NCACell,
    right_model: NCACell,
    names: tuple[str, ...] | None,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    left_names, left = _parameter_vector(left_model, names)
    right_names, right = _parameter_vector(right_model, names)
    if left_names != right_names:
        raise AssertionError(f"parameter names differ: {left_names} != {right_names}")
    result = _close_stats(left, right, atol=atol, rtol=rtol)
    result["parameters"] = left_names
    result["left_grad_norm"] = float(torch.linalg.vector_norm(left).cpu())
    result["right_grad_norm"] = float(torch.linalg.vector_norm(right).cpu())
    result["non_vacuous"] = bool(
        result["left_grad_norm"] > 0.0 and result["right_grad_norm"] > 0.0
    )
    result["passed"] = bool(result["passed"] and result["non_vacuous"])
    return result


def _lift_formula_check(seed_visible: Tensor, fire: Tensor) -> dict[str, Any]:
    """Check the augmented bias coordinate and one explicit U/count update."""
    model = _make_triplet(
        seed=44101,
        channels=4,
        hidden=6,
        dtype=torch.float64,
        device="cpu",
        projection_scale=0.001,
        feature_bias_scale=0.02,
    )["relu_lift"]
    state = model.initialize(seed_visible, "relu_lift")
    assert isinstance(state, tuple) and len(state) == 3
    base, update, count = (part.clone() for part in state)

    augmented_matrix = torch.cat(
        (model.feature.weight[:, :, 0, 0], model.feature.bias[:, None]), dim=1
    )
    lifted_features = torch.einsum("jk,bjkhw->bjhw", augmented_matrix, update)
    projection_weight = model.projection.weight[:, :, 0, 0]
    explicit_visible = base + torch.einsum(
        "cj,bjhw->bchw", projection_weight[:, :-1], lifted_features
    ) + projection_weight[:, -1][None, :, None, None] * count
    initial_error = _max_abs(model.visible(state, "relu_lift"), explicit_visible)

    perceived = model.perceive(explicit_visible)
    p_aug = torch.cat(
        (perceived, torch.ones_like(perceived[:, :1])), dim=1
    )
    derivative = (model.feature(perceived) > 0).to(dtype=seed_visible.dtype)

    with torch.no_grad():
        next_state = model.step(state, fire, "relu_lift")
    assert isinstance(next_state, tuple) and len(next_state) == 3
    next_base, next_update, next_count = next_state
    assert model.last_masks is not None
    q = model.last_masks["alive"].to(dtype=seed_visible.dtype)
    expected_update = (
        update
        + fire[:, :, None, :, :]
        * derivative[:, :, None, :, :]
        * p_aug[:, None, :, :, :]
    ) * q[:, 0].unsqueeze(1).unsqueeze(1)
    expected_base = base * q
    expected_count = (count + fire) * q
    update_error = _max_abs(next_update, expected_update)
    base_error = _max_abs(next_base, expected_base)
    count_error = _max_abs(next_count, expected_count)
    stats = {
        "initial_reconstruction_max_abs_error": initial_error,
        "U_update_max_abs_error": update_error,
        "base_update_max_abs_error": base_error,
        "count_update_max_abs_error": count_error,
        "feature_bias_shape": list(model.feature.bias.shape),
        "feature_bias_nonzero": bool(torch.count_nonzero(model.feature.bias) > 0),
        "projection_constant_column_shape": list(model.projection.weight[:, -1].shape),
        "projection_constant_column_nonzero": bool(
            torch.count_nonzero(model.projection.weight[:, -1]) > 0
        ),
        "passed": bool(
            max(initial_error, update_error, base_error, count_error) <= 2e-12
            and torch.count_nonzero(model.feature.bias) > 0
            and torch.count_nonzero(model.projection.weight[:, -1]) > 0
        ),
    }
    return stats


def _zero_projection_initialization(seed_visible: Tensor) -> dict[str, Any]:
    model = NCACell(channels=4, hidden=6).to(dtype=torch.float64)
    state = model.initialize(seed_visible, "relu_lift")
    assert isinstance(state, tuple) and len(state) == 3
    base, update, count = state
    visible = model.visible(state, "relu_lift")
    projection_is_zero = bool(torch.count_nonzero(model.projection.weight) == 0)
    update_is_zero = bool(torch.count_nonzero(update) == 0)
    count_is_zero = bool(torch.count_nonzero(count) == 0)
    initial_matches_base = bool(torch.equal(visible, base))
    seed_matches_visible = bool(torch.equal(visible, seed_visible))
    return {
        "projection_is_zero": projection_is_zero,
        "U_shape": list(update.shape),
        "U_is_zero": update_is_zero,
        "count_shape": list(count.shape),
        "count_is_zero": count_is_zero,
        "visible_equals_base": initial_matches_base,
        "visible_equals_seed": seed_matches_visible,
        "feature_bias_present": model.feature.bias is not None,
        "feature_bias_shape": list(model.feature.bias.shape),
        "projection_weight_shape": list(model.projection.weight.shape),
        "passed": bool(
            projection_is_zero
            and update_is_zero
            and count_is_zero
            and initial_matches_base
            and seed_matches_visible
            and model.feature.bias is not None
        ),
    }


def _rollout_triplet(
    models: dict[str, NCACell],
    seed_visible: Tensor,
    fires: list[Tensor],
    *,
    alive_overrides: list[Tensor] | None = None,
    capture_alive_schedule: bool = False,
    atol: float = 2e-8,
    rtol: float = 2e-8,
) -> dict[str, Any]:
    states = {mode: models[mode].initialize(seed_visible, mode) for mode in MODES}
    mask_mismatches = 0
    largest_step_error = 0.0
    each_step_close = True
    active_cells: list[int] = []
    alive_schedule: list[Tensor] = []
    final_visible: dict[str, Tensor] = {}
    with torch.no_grad():
        for index, fire in enumerate(fires):
            alive_override = None if alive_overrides is None else alive_overrides[index]
            for mode in MODES:
                states[mode] = models[mode].step(
                    states[mode], fire, mode, alive_override=alive_override
                )
            final_visible = {
                mode: models[mode].visible(states[mode], mode) for mode in MODES
            }
            masks = {
                mode: models[mode].last_masks["alive"]
                for mode in MODES
                if models[mode].last_masks is not None
            }
            mask_mismatches += int(
                not torch.equal(masks["original"], masks["au"])
            )
            mask_mismatches += int(
                not torch.equal(masks["original"], masks["relu_lift"])
            )
            active_cells.append(int(masks["original"].sum().cpu()))
            if capture_alive_schedule:
                alive_schedule.append(masks["original"].detach().clone())
            for mode in ("au", "relu_lift"):
                error = _max_abs(final_visible["original"], final_visible[mode])
                largest_step_error = max(largest_step_error, error)
                each_step_close = bool(
                    each_step_close
                    and torch.allclose(
                        final_visible["original"],
                        final_visible[mode],
                        atol=atol,
                        rtol=rtol,
                    )
                )
    final = {
        mode: _close_stats(
            final_visible["original"], final_visible[mode], atol=atol, rtol=rtol
        )
        for mode in ("au", "relu_lift")
    }
    result = {
        "steps": len(fires),
        "shape": list(seed_visible.shape),
        "alive_mask_mismatches": mask_mismatches,
        "alive_cells_min_max": [min(active_cells), max(active_cells)],
        "largest_step_max_abs_error": largest_step_error,
        "each_step_close": each_step_close,
        "final_forward_vs_original": final,
        "passed": bool(
            mask_mismatches == 0
            and active_cells[-1] > 0
            and each_step_close
            and all(result["passed"] for result in final.values())
        ),
    }
    if capture_alive_schedule:
        result["_alive_schedule"] = alive_schedule
    return result


def _nonzero_outside_alive(state: Any, mode: str, alive: Tensor) -> int:
    dead = ~alive.bool()
    if mode == "original":
        return int(torch.count_nonzero(state.masked_select(dead.expand_as(state))).cpu())
    if mode == "au":
        base, accumulator = state
        base_count = torch.count_nonzero(base.masked_select(dead.expand_as(base)))
        accumulator_count = torch.count_nonzero(
            accumulator.masked_select(dead.expand_as(accumulator))
        )
        return int((base_count + accumulator_count).cpu())
    base, update, count = state
    dead_base = torch.count_nonzero(base.masked_select(dead.expand_as(base)))
    dead_update_mask = dead[:, 0].unsqueeze(1).unsqueeze(1).expand_as(update)
    dead_update = torch.count_nonzero(update.masked_select(dead_update_mask))
    dead_count = torch.count_nonzero(count.masked_select(dead.expand_as(count)))
    return int((dead_base + dead_update + dead_count).cpu())


def _component_nonzeros(state: Any) -> list[int]:
    if isinstance(state, Tensor):
        return [int(torch.count_nonzero(state).cpu())]
    return [int(torch.count_nonzero(part).cpu()) for part in state]


def _forced_mask_check(seed_visible: Tensor, fire: Tensor) -> dict[str, Any]:
    models = _make_triplet(
        seed=45211,
        channels=4,
        hidden=6,
        dtype=torch.float64,
        device="cpu",
        projection_scale=0.001,
    )
    partial = torch.ones((seed_visible.shape[0], 1, *seed_visible.shape[-2:]), dtype=torch.bool)
    partial[:, :, 0, :] = False
    partial[:, :, -1, -1] = False
    states = {mode: models[mode].initialize(seed_visible, mode) for mode in MODES}
    with torch.no_grad():
        for mode in MODES:
            states[mode] = models[mode].step(
                states[mode], fire, mode, alive_override=partial
            )
    visible = {
        mode: models[mode].visible(states[mode], mode) for mode in MODES
    }
    partial_dead_counts = {
        mode: _nonzero_outside_alive(states[mode], mode, partial) for mode in MODES
    }
    partial_parity = all(
        torch.allclose(visible["original"], visible[mode], atol=2e-8, rtol=2e-8)
        for mode in ("au", "relu_lift")
    )

    all_dead = torch.zeros_like(partial)
    all_dead_states = {
        mode: models[mode].initialize(seed_visible, mode) for mode in MODES
    }
    with torch.no_grad():
        for mode in MODES:
            all_dead_states[mode] = models[mode].step(
                all_dead_states[mode], fire, mode, alive_override=all_dead
            )
        after_forced_kill = {
            mode: _component_nonzeros(all_dead_states[mode]) for mode in MODES
        }
        for mode in MODES:
            all_dead_states[mode] = models[mode].step(
                all_dead_states[mode], torch.ones_like(fire), mode
            )
        after_actual_step = {
            mode: _component_nonzeros(all_dead_states[mode]) for mode in MODES
        }
    passed = bool(
        partial_parity
        and all(value == 0 for value in partial_dead_counts.values())
        and all(all(value == 0 for value in counts) for counts in after_forced_kill.values())
        and all(all(value == 0 for value in counts) for counts in after_actual_step.values())
    )
    return {
        "partial_alive_cells": int(partial.sum().cpu()),
        "partial_dead_state_nonzeros": partial_dead_counts,
        "partial_forced_mask_forward_parity": bool(partial_parity),
        "all_dead_state_component_nonzeros_after_forced_kill": after_forced_kill,
        "all_dead_state_component_nonzeros_after_actual_step": after_actual_step,
        "passed": passed,
    }


def _full_gradient_check(
    seed_visible: Tensor, fires: list[Tensor], probe: Tensor
) -> dict[str, Any]:
    models = _make_triplet(
        seed=46307,
        channels=4,
        hidden=6,
        dtype=torch.float64,
        device="cpu",
        projection_scale=0.001,
    )
    mask_mismatches = 0
    final_visible: dict[str, Tensor] = {}
    losses: dict[str, float] = {}
    for mode in MODES:
        state = models[mode].initialize(seed_visible, mode)
        for fire in fires:
            state = models[mode].step(state, fire, mode)
        final_visible[mode] = models[mode].visible(state, mode)
        loss = (final_visible[mode] * probe).sum() / probe.numel()
        losses[mode] = float(loss.detach().cpu())
        loss.backward()
    if any(models[mode].last_masks is None for mode in MODES):
        raise AssertionError("cell did not record the final alive mask")
    mask_mismatches += int(
        not torch.equal(
            models["original"].last_masks["alive"], models["au"].last_masks["alive"]
        )
    )
    mask_mismatches += int(
        not torch.equal(
            models["original"].last_masks["alive"],
            models["relu_lift"].last_masks["alive"],
        )
    )
    parameter_comparisons = {
        mode: _parameter_comparison(
            models["original"],
            models[mode],
            None,
            atol=2e-9,
            rtol=2e-8,
        )
        for mode in ("au", "relu_lift")
    }
    forward_comparisons = {
        mode: _close_stats(
            final_visible["original"], final_visible[mode], atol=2e-8, rtol=2e-8
        )
        for mode in ("au", "relu_lift")
    }
    return {
        "steps": len(fires),
        "losses": losses,
        "forward_vs_original": forward_comparisons,
        "full_parameter_gradients_vs_original": parameter_comparisons,
        "final_alive_mask_mismatches": mask_mismatches,
        "passed": bool(
            mask_mismatches == 0
            and all(result["passed"] for result in forward_comparisons.values())
            and all(result["passed"] for result in parameter_comparisons.values())
        ),
    }


def _flat_named_grads(model: NCACell) -> dict[str, Tensor]:
    return {
        name: (
            parameter.grad.detach().clone()
            if parameter.grad is not None
            else torch.zeros_like(parameter)
        )
        for name, parameter in model.named_parameters()
    }


def _k8_gradient_check(
    *,
    seed_visible: Tensor,
    fires: list[Tensor],
    alive_masks: list[Tensor],
    probe: Tensor,
    hidden: int,
    channels: int,
    dtype: torch.dtype,
    device: str | torch.device,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    models = _make_triplet(
        seed=47431,
        channels=channels,
        hidden=hidden,
        dtype=dtype,
        device=device,
        projection_scale=0.001,
    )
    prefix_steps = len(fires) - len(alive_masks)
    cuts: dict[str, Any] = {}
    with torch.no_grad():
        for mode in MODES:
            state = models[mode].initialize(seed_visible, mode)
            for fire in fires[:prefix_steps]:
                state = models[mode].step(state, fire, mode)
            cuts[mode] = models[mode].detach(state, mode)

    with torch.no_grad():
        cut_visible = {
            mode: models[mode].visible(cuts[mode], mode) for mode in MODES
        }
    cut_forward = {
        mode: _close_stats(
            cut_visible["original"], cut_visible[mode], atol=atol, rtol=rtol
        )
        for mode in ("au", "relu_lift")
    }

    initial_visible = cut_visible["original"].detach().clone().requires_grad_(True)
    delta_state: Any = initial_visible
    for fire, alive in zip(fires[prefix_steps:], alive_masks):
        delta_state = models["original"].step(
            delta_state, fire, "original", alive_override=alive
        )
    delta_loss = (models["original"].visible(delta_state, "original") * probe).sum() / probe.numel()
    delta = torch.autograd.grad(delta_loss, initial_visible)[0].detach()

    grads: dict[str, dict[str, Tensor]] = {}
    final_visible: dict[str, Tensor] = {}
    losses: dict[str, float] = {}
    for mode in MODES:
        models[mode].zero_grad(set_to_none=True)
        state = models[mode].detach(cuts[mode], mode)
        for fire, alive in zip(fires[prefix_steps:], alive_masks):
            state = models[mode].step(state, fire, mode, alive_override=alive)
        final_visible[mode] = models[mode].visible(state, mode)
        loss = (final_visible[mode] * probe).sum() / probe.numel()
        losses[mode] = float(loss.detach().cpu())
        loss.backward()
        grads[mode] = _flat_named_grads(models[mode])

    original_w = grads["original"]["projection.weight"]
    au_w = grads["au"]["projection.weight"]
    lift_w = grads["relu_lift"]["projection.weight"]
    w_new_equals_au = _close_stats(lift_w, au_w, atol=atol, rtol=rtol)
    w_new_equals_au["new_W_grad_norm"] = float(torch.linalg.vector_norm(lift_w).cpu())
    w_new_equals_au["au_W_grad_norm"] = float(torch.linalg.vector_norm(au_w).cpu())
    w_new_equals_au["original_W_grad_norm"] = float(torch.linalg.vector_norm(original_w).cpu())
    w_new_equals_au["non_vacuous"] = bool(
        w_new_equals_au["new_W_grad_norm"] > 0.0
        and w_new_equals_au["au_W_grad_norm"] > 0.0
    )
    w_new_equals_au["passed"] = bool(
        w_new_equals_au["passed"] and w_new_equals_au["non_vacuous"]
    )

    _, lift_u_tau, _ = cuts["relu_lift"]
    lift_u_tau = lift_u_tau.detach().clone()
    hidden_projection = models["relu_lift"].projection.weight[:, :hidden, 0, 0]
    hidden_delta = torch.einsum("cj,bchw->bjhw", hidden_projection, delta)
    expected_augmented_eta_delta = torch.einsum(
        "bjhw,bjkhw->jk", hidden_delta, lift_u_tau
    )
    actual_weight_delta = (
        grads["relu_lift"]["feature.weight"]
        - grads["original"]["feature.weight"]
    )[:, :, 0, 0]
    actual_bias_delta = (
        grads["relu_lift"]["feature.bias"]
        - grads["original"]["feature.bias"]
    )
    actual_augmented_eta_delta = torch.cat(
        (actual_weight_delta, actual_bias_delta[:, None]), dim=1
    )
    eta_identity = _close_stats(
        actual_augmented_eta_delta,
        expected_augmented_eta_delta,
        atol=atol,
        rtol=rtol,
    )
    eta_identity["expected_boundary_term_norm"] = float(
        torch.linalg.vector_norm(expected_augmented_eta_delta).cpu()
    )
    eta_identity["actual_difference_norm"] = float(
        torch.linalg.vector_norm(actual_augmented_eta_delta).cpu()
    )
    eta_identity["lift_accumulator_norm_at_cutoff"] = float(
        torch.linalg.vector_norm(lift_u_tau).cpu()
    )
    eta_identity["non_vacuous"] = bool(
        eta_identity["expected_boundary_term_norm"] > 1e-12
        and eta_identity["lift_accumulator_norm_at_cutoff"] > 0.0
    )
    eta_identity["passed"] = bool(
        eta_identity["passed"] and eta_identity["non_vacuous"]
    )

    gradient_norms: dict[str, dict[str, float]] = {}
    for mode in MODES:
        w_vector = grads[mode]["projection.weight"].reshape(-1)
        eta_vector = torch.cat(
            (grads[mode]["feature.weight"].reshape(-1), grads[mode]["feature.bias"].reshape(-1))
        )
        gradient_norms[mode] = {
            "W_norm": float(torch.linalg.vector_norm(w_vector).cpu()),
            "eta_norm": float(torch.linalg.vector_norm(eta_vector).cpu()),
        }

    return {
        "prefix_steps": prefix_steps,
        "suffix_steps": len(alive_masks),
        "shape": list(seed_visible.shape),
        "losses": losses,
        "cut_forward_vs_original": cut_forward,
        "K_gradient_norms": gradient_norms,
        "suffix_alive_masks_fixed": True,
        "W_new_equals_AU_K": w_new_equals_au,
        "eta_new_minus_original_boundary_identity": eta_identity,
        "passed": bool(
            all(result["passed"] for result in cut_forward.values())
            and w_new_equals_au["passed"]
            and eta_identity["passed"]
        ),
    }


def _cpu64_checks() -> dict[str, Any]:
    batch, channels, hidden, height, width = 1, 4, 6, 4, 5
    dtype = torch.float64
    seed_visible = _seed_visible(
        48101, batch, channels, height, width, 3, dtype, "cpu"
    )
    zero_init = _zero_projection_initialization(seed_visible)
    fires12 = _fire_schedule(48102, 12, batch, height, width, dtype, "cpu")
    models = _make_triplet(
        seed=48103,
        channels=channels,
        hidden=hidden,
        dtype=dtype,
        device="cpu",
        projection_scale=0.001,
    )
    actual_alive = _rollout_triplet(models, seed_visible, fires12)
    formula_fire = fires12[0]
    lift_formula = _lift_formula_check(seed_visible, formula_fire)
    forced = _forced_mask_check(seed_visible, fires12[1])

    probe_gen = torch.Generator(device="cpu").manual_seed(48104)
    probe = torch.randn(
        (batch, channels, height, width), generator=probe_gen, dtype=dtype
    )
    full_gradients = _full_gradient_check(seed_visible, fires12, probe)

    k_steps = 4
    k_fires = _fire_schedule(48105, 12, batch, height, width, dtype, "cpu")
    forced_suffix = _forced_alive_schedule(k_steps, batch, height, width, "cpu")
    k_probe_gen = torch.Generator(device="cpu").manual_seed(48106)
    k_probe = torch.randn(
        (batch, channels, height, width), generator=k_probe_gen, dtype=dtype
    )
    k8 = _k8_gradient_check(
        seed_visible=seed_visible,
        fires=k_fires,
        alive_masks=forced_suffix,
        probe=k_probe,
        hidden=hidden,
        channels=channels,
        dtype=dtype,
        device="cpu",
        atol=2e-9,
        rtol=2e-8,
    )

    return {
        "dtype": "float64",
        "shape": [batch, channels, height, width],
        "hidden": hidden,
        "steps": 12,
        "projection_perturbation_scale": 0.001,
        "zero_projection_initialization": zero_init,
        "augmented_bias_and_U_count_formula": lift_formula,
        "matched_actual_alive_rollout_original_AU_lift": actual_alive,
        "forced_alive_mask_and_state_clearing": forced,
        "full_gradient_original_AU_lift": full_gradients,
        "K4_forced_mask_gradient_identity": k8,
        "passed": bool(
            zero_init["passed"]
            and lift_formula["passed"]
            and actual_alive["passed"]
            and forced["passed"]
            and full_gradients["passed"]
            and k8["passed"]
        ),
    }


def _cuda32_check(device: torch.device) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    batch, channels, hidden, height, width, steps = 8, 16, 128, 32, 32, 64
    dtype = torch.float32
    seed_visible = _seed_visible(
        48201, batch, channels, height, width, 3, dtype, device
    )
    fires = _fire_schedule(48202, steps, batch, height, width, dtype, device)
    models = _make_triplet(
        seed=48203,
        channels=channels,
        hidden=hidden,
        dtype=dtype,
        device=device,
        projection_scale=0.001,
    )
    actual_alive = _rollout_triplet(
        models,
        seed_visible,
        fires,
        capture_alive_schedule=True,
        atol=3e-5,
        rtol=3e-5,
    )
    alive_schedule = actual_alive.pop("_alive_schedule")
    if not actual_alive["passed"]:
        torch.cuda.synchronize(device)
        actual_alive["elapsed_seconds"] = float(time.perf_counter() - started)
        actual_alive["peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        actual_alive["peak_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
        actual_alive["passed"] = False
        return {
            "device": str(device),
            "dtype": "float32",
            "steps": steps,
            "shape": list(seed_visible.shape),
            "actual_alive_forward": actual_alive,
            "K8_gradient_identity": None,
            "passed": False,
        }

    probe_gen = torch.Generator(device="cpu").manual_seed(48204)
    probe = torch.randn(
        (batch, channels, height, width), generator=probe_gen, dtype=torch.float32
    ).to(device=device)
    k8 = _k8_gradient_check(
        seed_visible=seed_visible,
        fires=fires,
        alive_masks=alive_schedule[-8:],
        probe=probe,
        hidden=hidden,
        channels=channels,
        dtype=dtype,
        device=device,
        atol=3e-5,
        rtol=1e-3,
    )
    torch.cuda.synchronize(device)
    elapsed = float(time.perf_counter() - started)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "device": str(device),
        "dtype": "float32",
        "steps": steps,
        "shape": list(seed_visible.shape),
        "actual_alive_forward": actual_alive,
        "K8_gradient_identity": k8,
        "K8_suffix_masks": "fixed masks captured from the original T64 pass and replayed across matched arms",
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "free_gpu_memory_bytes_after_check": int(free_bytes),
        "total_gpu_memory_bytes": int(total_bytes),
        "passed": bool(actual_alive["passed"] and k8["passed"]),
    }


def run_checks(device: str | torch.device = "cpu") -> dict[str, Any]:
    """Return strict-JSON diagnostics; CUDA adds a bounded B8/C16/T64 check."""
    device_obj = torch.device(device)
    if device_obj.type not in ("cpu", "cuda"):
        raise ValueError("device must be a CPU or CUDA device")
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device_obj}")
    torch.set_num_threads(2)

    cpu = _cpu64_checks()
    report: dict[str, Any] = {
        "schema": "relu_input_lift_checks_v1",
        "scope": "cell algebra and autograd qualification; no training update",
        "torch_version": str(torch.__version__),
        "cpu64": cpu,
        "cuda32": None,
        "passed": bool(cpu["passed"]),
    }
    if device_obj.type == "cuda":
        report["cuda32"] = _cuda32_check(device_obj)
        report["passed"] = bool(report["passed"] and report["cuda32"]["passed"])
    # Validate the public return contract before handing it to a runner.
    json.dumps(report, allow_nan=False)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_checks(args.device), indent=2, allow_nan=False))
