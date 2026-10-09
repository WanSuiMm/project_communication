"""Mathematical and implementation qualification for AU-NCA cells.

These checks are CPU-only by default.  Passing ``device='cuda'`` adds one
float32 forward and full-gradient check at the frozen B=8, C=16, 32x32, 64-step
shape; it does not launch training.
"""
from __future__ import annotations

import copy
import json
from typing import Any

import torch
from torch import Tensor

try:  # Supports both package import and ``python checks.py``.
    from .cells import NCACell
except ImportError:  # pragma: no cover - exercised by direct script execution
    from cells import NCACell


def _make_pair(seed: int, dtype: torch.dtype = torch.float64, device: str | torch.device = "cpu") -> tuple[NCACell, NCACell]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        original = NCACell()
    original = original.to(device=device, dtype=dtype)
    au = copy.deepcopy(original)

    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    nonzero_projection = torch.randn(
        original.projection.weight.shape, generator=generator, dtype=torch.float64
    ) * 0.001
    with torch.no_grad():
        original.projection.weight.copy_(nonzero_projection.to(device=device, dtype=dtype))
        au.projection.weight.copy_(original.projection.weight)
    return original, au


def _seed_visible(seed: int, batch: int, height: int, width: int, dtype: torch.dtype, device: str | torch.device) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    visible = 0.03 * torch.randn((batch, 16, height, width), generator=generator, dtype=torch.float64)
    alpha = 0.65 + 0.02 * torch.randn((batch, 1, height, width), generator=generator, dtype=torch.float64)
    visible[:, 3:4] = alpha
    return visible.to(device=device, dtype=dtype)


def _fire_schedule(seed: int, steps: int, batch: int, height: int, width: int, dtype: torch.dtype, device: str | torch.device) -> list[Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.rand((steps, batch, 1, height, width), generator=generator)
    return [item.to(device=device, dtype=dtype) for item in (draws < 0.5)]


def _piecewise_alive_schedule(steps: int, batch: int, height: int, width: int, device: str | torch.device) -> list[Tensor]:
    schedule: list[Tensor] = []
    for step in range(steps):
        per_batch = []
        for batch_index in range(batch):
            mask = torch.ones((height, width), dtype=torch.bool)
            corner = (batch_index + step // 2) % 4
            if corner == 0:
                mask[:2, :2] = False
            elif corner == 1:
                mask[:2, -2:] = False
            elif corner == 2:
                mask[-2:, :2] = False
            else:
                mask[-2:, -2:] = False
            per_batch.append(mask)
        schedule.append(torch.stack(per_batch, dim=0).unsqueeze(1).to(device=device))
    return schedule


def _max_abs(left: Tensor, right: Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left.detach() - right.detach()).abs().max().cpu())


def _max_relative(left: Tensor, right: Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    scale = max(float(left.detach().abs().max().cpu()), float(right.detach().abs().max().cpu()))
    return _max_abs(left, right) / max(scale, 1e-300)


def _close_stats(left: Tensor, right: Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    max_abs = _max_abs(left, right)
    max_relative = _max_relative(left, right)
    close = torch.allclose(left, right, atol=atol, rtol=rtol)
    return {"max_abs_error": max_abs, "max_relative_error": max_relative, "atol": atol, "rtol": rtol, "passed": bool(close)}


def _state_visible(model: NCACell, state: Tensor | tuple[Tensor, Tensor], mode: str) -> Tensor:
    return model.visible(state, mode)


def _loss(model: NCACell, state: Tensor | tuple[Tensor, Tensor], mode: str, probe: Tensor) -> Tensor:
    return (_state_visible(model, state, mode) * probe).sum() / probe.numel()


def _parameter_vector(model: NCACell, names: tuple[str, ...] | None = None) -> tuple[list[str], Tensor]:
    selected: list[tuple[str, Tensor]] = []
    for name, parameter in model.named_parameters():
        if names is None or name in names:
            selected.append((name, parameter))
    if not selected:
        return [], torch.empty(0, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
    return [name for name, _ in selected], torch.cat(
        [
            (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).detach().reshape(-1)
            for _, parameter in selected
        ]
    )


def _group_diagnostics(original: NCACell, au: NCACell, group_names: tuple[str, ...]) -> dict[str, Any]:
    left_names, left = _parameter_vector(original, group_names)
    right_names, right = _parameter_vector(au, group_names)
    if left_names != right_names:
        raise AssertionError(f"parameter group changed between arms: {left_names} != {right_names}")
    norm_left = float(torch.linalg.vector_norm(left).cpu())
    norm_right = float(torch.linalg.vector_norm(right).cpu())
    denominator = norm_left * norm_right
    cosine = float(torch.dot(left, right).cpu() / denominator) if denominator > 0.0 else None
    return {
        "parameters": left_names,
        "original_grad_norm": norm_left,
        "au_grad_norm": norm_right,
        "gradient_cosine": cosine,
        "max_abs_error": _max_abs(left, right),
        "max_relative_error": _max_relative(left, right),
    }


def _all_parameter_error(original: NCACell, au: NCACell) -> dict[str, float]:
    left_names, left = _parameter_vector(original)
    right_names, right = _parameter_vector(au)
    if left_names != right_names:
        raise AssertionError(f"trainable parameter names differ: {left_names} != {right_names}")
    return {"max_abs_error": _max_abs(left, right), "max_relative_error": _max_relative(left, right)}


def _actual_64_step_and_full_gradient_check() -> dict[str, Any]:
    original, au = _make_pair(seed=180721)
    batch, height, width, steps = 2, 8, 9, 64
    seed = _seed_visible(180722, batch, height, width, torch.float64, "cpu")
    fires = _fire_schedule(180723, steps, batch, height, width, torch.float64, "cpu")
    probe_gen = torch.Generator(device="cpu").manual_seed(180724)
    probe = torch.randn((batch, 16, height, width), generator=probe_gen, dtype=torch.float64)

    original_state = original.initialize(seed, "original")
    au_state = au.initialize(seed, "au")
    largest_forward_error = 0.0
    mask_mismatches = 0
    active_cells: list[int] = []
    for fire in fires:
        original_state = original.step(original_state, fire, "original")
        au_state = au.step(au_state, fire, "au")
        original_visible = original.visible(original_state, "original")
        au_visible = au.visible(au_state, "au")
        largest_forward_error = max(largest_forward_error, _max_abs(original_visible, au_visible))
        mask_mismatches += int(not torch.equal(original.last_masks["alive"], au.last_masks["alive"]))
        active_cells.append(int(original.last_masks["alive"].sum().cpu()))

    forward = _close_stats(original_visible, au_visible, atol=5e-10, rtol=5e-10)
    forward["largest_step_max_abs_error"] = largest_forward_error
    forward["alive_mask_mismatches"] = mask_mismatches
    forward["alive_cells_min_max"] = [min(active_cells), max(active_cells)]
    forward["passed"] = bool(forward["passed"] and mask_mismatches == 0 and active_cells[-1] > 0)

    _loss(original, original_state, "original", probe).backward()
    _loss(au, au_state, "au", probe).backward()
    all_gradients = _all_parameter_error(original, au)
    all_gradients.update({"atol": 2e-9, "rtol": 2e-8})
    all_gradients["passed"] = bool(
        torch.allclose(
            _parameter_vector(original)[1],
            _parameter_vector(au)[1],
            atol=all_gradients["atol"],
            rtol=all_gradients["rtol"],
        )
    )
    group_gradients = {
        "W": _group_diagnostics(original, au, ("projection.weight",)),
        "eta": _group_diagnostics(original, au, ("feature.weight", "feature.bias")),
    }
    group_gradients["W"]["non_vacuous"] = bool(
        group_gradients["W"]["original_grad_norm"] > 0.0
        and group_gradients["W"]["au_grad_norm"] > 0.0
    )
    group_gradients["eta"]["non_vacuous"] = bool(
        group_gradients["eta"]["original_grad_norm"] > 0.0
        and group_gradients["eta"]["au_grad_norm"] > 0.0
    )
    group_gradients["W"]["passed"] = bool(
        group_gradients["W"]["max_abs_error"] <= 2e-9
        + 2e-8 * max(group_gradients["W"]["original_grad_norm"], group_gradients["W"]["au_grad_norm"])
        and group_gradients["W"]["non_vacuous"]
    )
    group_gradients["eta"]["passed"] = bool(
        group_gradients["eta"]["max_abs_error"] <= 2e-9
        + 2e-8 * max(group_gradients["eta"]["original_grad_norm"], group_gradients["eta"]["au_grad_norm"])
        and group_gradients["eta"]["non_vacuous"]
    )
    return {"steps": steps, "shape": [batch, 16, height, width], "forward": forward, "full_gradients": all_gradients, "group_gradients": group_gradients}


def _truncated_k8_identity_check() -> dict[str, Any]:
    original, au = _make_pair(seed=290831)
    batch, height, width, prefix_steps, suffix_steps = 2, 7, 8, 56, 8
    seed = _seed_visible(290832, batch, height, width, torch.float64, "cpu")
    prefix_fires = _fire_schedule(290833, prefix_steps, batch, height, width, torch.float64, "cpu")
    suffix_fires = _fire_schedule(290834, suffix_steps, batch, height, width, torch.float64, "cpu")
    suffix_masks = _piecewise_alive_schedule(suffix_steps, batch, height, width, "cpu")
    probe_gen = torch.Generator(device="cpu").manual_seed(290835)
    probe = torch.randn((batch, 16, height, width), generator=probe_gen, dtype=torch.float64)

    original_cut: Tensor = original.initialize(seed, "original")  # type: ignore[assignment]
    au_cut: tuple[Tensor, Tensor] = au.initialize(seed, "au")  # type: ignore[assignment]
    with torch.no_grad():
        for fire in prefix_fires:
            original_cut = original.step(original_cut, fire, "original")  # type: ignore[assignment]
            au_cut = au.step(au_cut, fire, "au")  # type: ignore[assignment]
    cut_visible = original.visible(original_cut, "original")
    au_cut_visible = au.visible(au_cut, "au")
    cut_forward = _close_stats(cut_visible, au_cut_visible, atol=5e-10, rtol=5e-10)

    # Compute delta_tau = d(loss_tail)/d(s_tau) with fixed suffix masks.
    initial_visible = cut_visible.detach().clone().requires_grad_(True)
    delta_state: Tensor | tuple[Tensor, Tensor] = initial_visible
    for fire, mask in zip(suffix_fires, suffix_masks):
        delta_state = original.step(delta_state, fire, "original", alive_override=mask)
    delta = torch.autograd.grad(_loss(original, delta_state, "original", probe), initial_visible)[0]

    # The AU base and accumulator are both truncated at tau; reconstruction
    # through W remains live, creating the sole additional W gradient term.
    original.zero_grad(set_to_none=True)
    au.zero_grad(set_to_none=True)
    original_tail: Tensor | tuple[Tensor, Tensor] = original.detach(original_cut, "original")
    au_tail: Tensor | tuple[Tensor, Tensor] = au.detach(au_cut, "au")
    for fire, mask in zip(suffix_fires, suffix_masks):
        original_tail = original.step(original_tail, fire, "original", alive_override=mask)
        au_tail = au.step(au_tail, fire, "au", alive_override=mask)
    _loss(original, original_tail, "original", probe).backward()
    _loss(au, au_tail, "au", probe).backward()

    original_w = original.projection.weight.grad
    au_w = au.projection.weight.grad
    if original_w is None or au_w is None:
        raise AssertionError("projection gradient was unexpectedly absent")
    e_tau = au_cut[1].detach()
    expected_difference = torch.einsum("bchw,bjhw->cj", delta, e_tau).view_as(original_w)
    observed_difference = au_w - original_w
    w_identity = _close_stats(observed_difference, expected_difference, atol=2e-9, rtol=2e-8)
    expected_difference_norm = float(torch.linalg.vector_norm(expected_difference).cpu())
    w_identity["expected_difference_norm"] = expected_difference_norm
    w_identity["non_vacuous"] = expected_difference_norm > 1e-12
    w_identity["passed"] = bool(w_identity["passed"] and w_identity["non_vacuous"])
    eta_equal = _group_diagnostics(original, au, ("feature.weight", "feature.bias"))
    eta_equal["atol"] = 2e-9
    eta_equal["rtol"] = 2e-8
    eta_equal["non_vacuous"] = bool(
        eta_equal["original_grad_norm"] > 0.0 and eta_equal["au_grad_norm"] > 0.0
    )
    eta_equal["passed"] = bool(
        torch.allclose(
            _parameter_vector(original, ("feature.weight", "feature.bias"))[1],
            _parameter_vector(au, ("feature.weight", "feature.bias"))[1],
            atol=eta_equal["atol"],
            rtol=eta_equal["rtol"],
        )
        and eta_equal["non_vacuous"]
    )
    return {
        "cutoff": prefix_steps,
        "suffix_steps": suffix_steps,
        "shape": [batch, 16, height, width],
        "piecewise_alive_mask_unique_values": sorted({int(value) for mask in suffix_masks for value in mask.unique().tolist()}),
        "accumulator_norm_at_cutoff": float(torch.linalg.vector_norm(e_tau).cpu()),
        "cut_state_forward": cut_forward,
        "W_gradient_difference_identity": w_identity,
        "W_observed_difference_norm": float(torch.linalg.vector_norm(observed_difference).cpu()),
        "W_expected_delta_outer_e_norm": expected_difference_norm,
        "eta_gradient_equality": eta_equal,
    }


def _seed_firing_and_death_check() -> dict[str, Any]:
    original = NCACell()
    au = copy.deepcopy(original)
    seed = torch.zeros((1, 16, 5, 5), dtype=torch.float32)
    seed[0, 3, 2, 2] = 0.5
    fire = torch.ones((1, 1, 5, 5), dtype=torch.float32)
    # A constant negative output bias moves alpha below the post-update
    # threshold; the final alive intersection must clear every state component.
    with torch.no_grad():
        for model in (original, au):
            model.feature.weight.zero_()
            model.feature.bias.zero_()
            model.projection.weight.zero_()
            model.projection.weight[3, -1, 0, 0] = -1.0

    original_state = original.step(original.initialize(seed, "original"), fire, "original")
    au_state = au.step(au.initialize(seed, "au"), fire, "au")
    visible_original = original.visible(original_state, "original")
    visible_au = au.visible(au_state, "au")
    pre_alive_count = int(original.last_masks["pre_alive"].sum())
    post_alive_count = int(original.last_masks["post_alive"].sum())
    final_alive_count = int(original.last_masks["alive"].sum())
    original_cleared = int(torch.count_nonzero(original_state)) == 0
    base_cleared = int(torch.count_nonzero(au_state[0])) == 0
    accumulator_cleared = int(torch.count_nonzero(au_state[1])) == 0
    parity = torch.equal(visible_original, visible_au)
    passed = pre_alive_count > 0 and post_alive_count == 0 and final_alive_count == 0 and original_cleared and base_cleared and accumulator_cleared and parity
    return {
        "pre_alive_cells": pre_alive_count,
        "post_alive_cells": post_alive_count,
        "final_alive_cells": final_alive_count,
        "original_state_nonzeros_after_death": int(torch.count_nonzero(original_state)),
        "au_base_nonzeros_after_death": int(torch.count_nonzero(au_state[0])),
        "au_accumulator_nonzeros_after_death": int(torch.count_nonzero(au_state[1])),
        "visible_parity": bool(parity),
        "passed": bool(passed),
    }


def _cuda_float32_check(device: str | torch.device) -> dict[str, Any]:
    device = torch.device(device)
    original, au = _make_pair(seed=390941, dtype=torch.float32, device=device)
    batch, height, width, steps = 8, 32, 32, 64
    seed = _seed_visible(390942, batch, height, width, torch.float32, device)
    fires = _fire_schedule(390943, steps, batch, height, width, torch.float32, device)
    probe_gen = torch.Generator(device="cpu").manual_seed(390944)
    probe = torch.randn((batch, 16, height, width), generator=probe_gen).to(device=device)
    original_state = original.initialize(seed, "original")
    au_state = au.initialize(seed, "au")
    largest_forward_error = 0.0
    mask_mismatches = 0
    for fire in fires:
        original_state = original.step(original_state, fire, "original")
        au_state = au.step(au_state, fire, "au")
        original_visible = original.visible(original_state, "original")
        au_visible = au.visible(au_state, "au")
        largest_forward_error = max(largest_forward_error, _max_abs(original_visible, au_visible))
        mask_mismatches += int(not torch.equal(original.last_masks["alive"], au.last_masks["alive"]))
    forward = _close_stats(original_visible, au_visible, atol=3e-5, rtol=3e-5)
    forward["largest_step_max_abs_error"] = largest_forward_error
    forward["alive_mask_mismatches"] = mask_mismatches
    forward["passed"] = bool(forward["passed"] and mask_mismatches == 0)
    _loss(original, original_state, "original", probe).backward()
    _loss(au, au_state, "au", probe).backward()
    gradients = _all_parameter_error(original, au)
    gradients.update({"atol": 3e-5, "rtol": 3e-4})
    gradients["passed"] = bool(
        torch.allclose(
            _parameter_vector(original)[1],
            _parameter_vector(au)[1],
            atol=gradients["atol"],
            rtol=gradients["rtol"],
        )
    )
    gradients["groups"] = {
        "W": _group_diagnostics(original, au, ("projection.weight",)),
        "eta": _group_diagnostics(original, au, ("feature.weight", "feature.bias")),
    }
    return {
        "device": str(device),
        "dtype": "float32",
        "steps": steps,
        "shape": [batch, 16, height, width],
        "forward": forward,
        "full_gradients": gradients,
        "passed": bool(forward["passed"] and gradients["passed"]),
    }


def run_checks(device: str | torch.device = "cpu") -> dict[str, Any]:
    """Run CPU64 mathematical checks and, for CUDA, one CUDA32 shape check."""
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device_obj}")
    if device_obj.type not in ("cpu", "cuda"):
        raise ValueError("device must be a CPU or CUDA device")
    torch.set_num_threads(2)

    default_cell = NCACell()
    shape_and_counts = {
        "feature_weight_shape": list(default_cell.feature.weight.shape),
        "feature_bias_shape": list(default_cell.feature.bias.shape),
        "projection_weight_shape": list(default_cell.projection.weight.shape),
        "trainable_parameter_count": default_cell.trainable_parameter_count,
        "state_channels": default_cell.state_channels,
        "fixed_perception_trainable_parameters": sum(parameter.numel() for parameter in default_cell.perception.parameters()),
        "projection_zero_initialized": bool(torch.count_nonzero(default_cell.projection.weight) == 0),
    }
    assert shape_and_counts["feature_weight_shape"] == [128, 48, 1, 1]
    assert shape_and_counts["feature_bias_shape"] == [128]
    assert shape_and_counts["projection_weight_shape"] == [16, 129, 1, 1]
    assert shape_and_counts["trainable_parameter_count"] == 8336
    assert shape_and_counts["state_channels"] == {"original": 16, "au_base": 16, "au_accumulator": 129}
    assert shape_and_counts["fixed_perception_trainable_parameters"] == 0
    assert shape_and_counts["projection_zero_initialized"]

    actual_rollout = _actual_64_step_and_full_gradient_check()
    truncated = _truncated_k8_identity_check()
    death = _seed_firing_and_death_check()
    report: dict[str, Any] = {
        "passed": bool(
            actual_rollout["forward"]["passed"]
            and actual_rollout["full_gradients"]["passed"]
            and actual_rollout["group_gradients"]["W"]["passed"]
            and actual_rollout["group_gradients"]["eta"]["passed"]
            and truncated["cut_state_forward"]["passed"]
            and truncated["W_gradient_difference_identity"]["passed"]
            and truncated["eta_gradient_equality"]["passed"]
            and death["passed"]
        ),
        "scope": "cell algebra and autograd qualification; no training",
        "torch_version": torch.__version__,
        "default_architecture": shape_and_counts,
        "nonzero_projection_scale": 0.001,
        "actual_mask_64_step_forward_and_full_gradients": actual_rollout,
        "truncated_k8_piecewise_mask_identity": truncated,
        "seed_firing_and_alive_death": death,
        "cuda32_actual_shape_check": None,
    }
    if device_obj.type == "cuda":
        report["cuda32_actual_shape_check"] = _cuda_float32_check(device_obj)
        report["passed"] = bool(report["passed"] and report["cuda32_actual_shape_check"]["passed"])
    return report


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, allow_nan=False))

