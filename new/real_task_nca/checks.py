"""Small CPU64 algebra/autograd checks for the conditioned NCA cell.

This is an implementation qualification only.  It performs no optimizer step,
dataset access, download, or training run.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

try:  # Supports both package imports and ``python checks.py``.
    from .cells import ConditionalNCACell, CellState
except ImportError:  # pragma: no cover - direct script execution
    from cells import ConditionalNCACell, CellState


def _close(left: Tensor, right: Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    error = float((left.detach() - right.detach()).abs().max().cpu()) if left.numel() else 0.0
    passed = bool(torch.allclose(left, right, atol=atol, rtol=rtol))
    return {"max_abs_error": error, "atol": atol, "rtol": rtol, "passed": passed}


def _flat_grads(
    model: ConditionalNCACell,
    loss: Tensor,
    extra_inputs: tuple[Tensor, ...] = (),
) -> tuple[list[str], list[Tensor]]:
    named = list(model.named_parameters())
    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in named] + list(extra_inputs),
        allow_unused=True,
    )
    result = [
        torch.zeros_like(parameter) if grad is None else grad.detach()
        for (_, parameter), grad in zip(named, gradients)
    ]
    result.extend(
        torch.zeros_like(value) if grad is None else grad.detach()
        for value, grad in zip(extra_inputs, gradients[len(named) :])
    )
    names = [name for name, _ in named] + [f"input_{i}" for i in range(len(extra_inputs))]
    return names, result


def _seed_inputs(
    *, seed: int, batch: int, height: int, width: int, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor, list[Tensor], Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probability = 0.15 + 0.7 * torch.rand(
        (batch, 1, height, width), generator=generator, dtype=dtype
    )
    # Include exact endpoints to exercise the finite-logit clamp.
    probability[..., 0, 0] = 0.0
    probability[..., -1, -1] = 1.0
    rgb = 0.2 * torch.randn((batch, 3, height, width), generator=generator, dtype=dtype)
    fov = torch.ones((batch, 1, height, width), dtype=torch.bool)
    fov[..., 0, :2] = False
    fov[..., -1, -2:] = False
    fov[..., 2, 3] = False
    fires = [
        (torch.rand((batch, 1, height, width), generator=generator) < 0.55)
        for _ in range(11)
    ]
    probe = torch.randn((batch, 16, height, width), generator=generator, dtype=dtype)
    return probability, rgb, fov, fires, probe


def _make_models(seed: int) -> tuple[ConditionalNCACell, ConditionalNCACell]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        original = ConditionalNCACell().to(dtype=torch.float64)
    au = copy.deepcopy(original)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    projection = 0.01 * torch.randn(
        original.projection.weight.shape, generator=generator, dtype=torch.float64
    )
    with torch.no_grad():
        original.projection.weight.copy_(projection)
        au.projection.weight.copy_(projection)
    return original, au


def _rollout_with_grad(
    model: ConditionalNCACell,
    probability: Tensor,
    rgb: Tensor,
    fov: Tensor,
    fires: list[Tensor],
    probe: Tensor,
    mode: str,
) -> tuple[Tensor, list[str], list[Tensor]]:
    probability_leaf = probability.detach().clone().requires_grad_(True)
    rgb_leaf = rgb.detach().clone().requires_grad_(True)
    image_features = model.encode_image(rgb_leaf)
    state = model.initialize(probability_leaf, fov, mode)
    for fire in fires:
        state = model.step(state, image_features, fire, fov, mode)
    visible = model.visible(state, mode)
    loss = (visible * probe).sum() / probe.numel()
    names, grads = _flat_grads(model, loss, (probability_leaf, rgb_leaf))
    return visible.detach(), names, grads


def _max_grad_error(left: list[Tensor], right: list[Tensor]) -> float:
    if len(left) != len(right):
        raise ValueError("gradient lists must have the same length")
    return max(
        (float((a - b).abs().max().cpu()) if a.numel() else 0.0)
        for a, b in zip(left, right)
    )


def _cpu64_check() -> dict[str, Any]:
    batch, height, width, total_steps = 1, 6, 7, 11
    dtype = torch.float64
    probability, rgb, fov, fires, probe = _seed_inputs(
        seed=71301, batch=batch, height=height, width=width, dtype=dtype
    )
    original, au = _make_models(seed=71302)
    projection_zero_at_initialization = bool(torch.count_nonzero(
        ConditionalNCACell().projection.weight
    ).item() == 0)

    original_visible, original_names, original_grads = _rollout_with_grad(
        original, probability, rgb, fov, fires, probe, "original"
    )
    au_visible, au_names, au_grads = _rollout_with_grad(
        au, probability, rgb, fov, fires, probe, "au"
    )
    if original_names != au_names:
        raise AssertionError("original and AU parameter ordering differs")

    forward = _close(original_visible, au_visible, atol=2e-11, rtol=2e-10)
    gradient_error = _max_grad_error(original_grads, au_grads)
    gradient = {
        "max_abs_error": gradient_error,
        "atol": 2e-10,
        "rtol": 2e-9,
        "passed": gradient_error <= 2e-10 + 2e-9 * max(
            max(float(value.abs().max().cpu()) for value in original_grads),
            max(float(value.abs().max().cpu()) for value in au_grads),
        ),
        "compared": len(original_grads),
    }

    outside = (~fov).expand(-1, 16, -1, -1)
    fov_max = max(
        float(original_visible.masked_select(outside).abs().max().cpu()),
        float(au_visible.masked_select(outside).abs().max().cpu()),
    )
    fov_check = {"outside_fov_max_abs": fov_max, "passed": fov_max == 0.0}

    k = 8
    prefix_steps = total_steps - k
    cut_original = copy.deepcopy(original)
    cut_au = copy.deepcopy(au)
    with torch.no_grad():
        original_features = cut_original.encode_image(rgb)
        au_features = cut_au.encode_image(rgb)
        original_state = cut_original.initialize(probability, fov, "original")
        au_state = cut_au.initialize(probability, fov, "au")
        for index in range(prefix_steps):
            original_state = cut_original.step(
                original_state, original_features, fires[index], fov, "original"
            )
            au_state = cut_au.step(au_state, au_features, fires[index], fov, "au")
        original_cut = cut_original.detach(original_state, "original")
        au_cut = cut_au.detach(au_state, "au")
        cut_visible = cut_au.visible(au_cut, "au").detach()
        history_norm = float(torch.linalg.vector_norm(au_cut[1]).cpu())

    # K=8 continues directly from the detached AU pair, including its complete
    # accumulator.  The original arm starts from the matching detached visible state.
    original_suffix = copy.deepcopy(original)
    au_suffix = copy.deepcopy(au)
    original_suffix_features = original_suffix.encode_image(rgb)
    au_suffix_features = au_suffix.encode_image(rgb)
    original_state = original_cut
    au_state = au_cut
    for fire in fires[prefix_steps:]:
        original_state = original_suffix.step(
            original_state, original_suffix_features, fire, fov, "original"
        )
        au_state = au_suffix.step(au_state, au_suffix_features, fire, fov, "au")
    original_final = original_suffix.visible(original_state, "original")
    au_final = au_suffix.visible(au_state, "au")
    suffix_forward = _close(original_final, au_final, atol=2e-11, rtol=2e-10)
    original_loss = (original_final * probe).sum() / probe.numel()
    au_loss = (au_final * probe).sum() / probe.numel()
    original_suffix_names, original_suffix_grads = _flat_grads(
        original_suffix, original_loss
    )
    au_suffix_names, au_suffix_grads = _flat_grads(au_suffix, au_loss)
    if original_suffix_names != au_suffix_names:
        raise AssertionError("K=8 parameter ordering differs")

    # Split W into a history-read copy and the suffix-update copy.  This keeps
    # the AU history intact while making the direct boundary term explicit.
    split = copy.deepcopy(au)
    with torch.no_grad():
        split_base, split_history = au_cut
        split_base = split_base.detach()
        split_history = split_history.detach()
    history_weight = torch.nn.Parameter(split.projection.weight.detach().clone())
    split_features = split.encode_image(rgb)
    base = split_base
    history = split_history
    suffix_updates = torch.zeros_like(split_history)
    fov_value = fov.to(dtype=dtype)
    for fire in fires[prefix_steps:]:
        visible = (
            base
            + F.conv2d(history, history_weight)
            + split.projection(suffix_updates)
        )
        update = split._augmented_update(visible, split_features)
        base = fov_value * base
        history = fov_value * history
        suffix_updates = fov_value * (
            suffix_updates + fire.to(dtype=dtype) * update
        )
    split_final = (
        base
        + F.conv2d(history, history_weight)
        + split.projection(suffix_updates)
    )
    split_loss = (split_final * probe).sum() / probe.numel()
    history_gradient = torch.autograd.grad(
        split_loss, history_weight, retain_graph=True
    )[0].detach()
    split_names, split_grads = _flat_grads(split, split_loss)

    split_forward = _close(au_final.detach(), split_final.detach(), atol=2e-11, rtol=2e-10)
    projection_index = split_names.index("projection.weight")
    actual_projection_gradient = au_suffix_grads[projection_index]
    split_projection_gradient = split_grads[projection_index]
    decomposed_projection_gradient = split_projection_gradient + history_gradient
    boundary_error = _close(
        actual_projection_gradient,
        decomposed_projection_gradient,
        atol=2e-10,
        rtol=2e-9,
    )
    suffix_projection_error = _close(
        original_suffix_grads[projection_index],
        split_projection_gradient,
        atol=2e-10,
        rtol=2e-9,
    )
    history_gradient_norm = float(torch.linalg.vector_norm(history_gradient).cpu())
    split_non_projection_indices = [
        index for index, name in enumerate(split_names) if name != "projection.weight"
    ]
    non_projection_error = _max_grad_error(
        [au_suffix_grads[index] for index in split_non_projection_indices],
        [split_grads[index] for index in split_non_projection_indices],
    )
    k8_boundary = {
        "prefix_steps": prefix_steps,
        "suffix_steps": k,
        "history_preserved_at_cut": True,
        "cut_accumulator_norm": history_norm,
        "boundary_direct_term_norm": history_gradient_norm,
        "actual_vs_split_forward": split_forward,
        "actual_projection_grad_vs_suffix_plus_boundary": boundary_error,
        "suffix_projection_grad_vs_original_K8": suffix_projection_error,
        "non_projection_gradient_max_abs_error": non_projection_error,
        "passed": bool(
            history_norm > 0.0
            and history_gradient_norm > 0.0
            and split_forward["passed"]
            and boundary_error["passed"]
            and suffix_projection_error["passed"]
            and non_projection_error <= 2e-10
        ),
    }

    return {
        "dtype": "float64",
        "shape": [batch, 16, height, width],
        "hidden": 128,
        "perception_width": 64,
        "steps": total_steps,
        "projection_randomized_for_nonzero_dynamics": True,
        "projection_zero_at_initialization": projection_zero_at_initialization,
        "original_vs_au_forward": forward,
        "original_vs_au_full_gradients": gradient,
        "fixed_fov": fov_check,
        "K8_boundary_direct_term": k8_boundary,
        "passed": bool(
            projection_zero_at_initialization
            and forward["passed"]
            and gradient["passed"]
            and fov_check["passed"]
            and suffix_forward["passed"]
            and k8_boundary["passed"]
        ),
    }


def run_checks() -> dict[str, Any]:
    """Run the bounded CPU64 cell algebra/autograd check and return JSON data."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 2))
    try:
        cpu64 = _cpu64_check()
    finally:
        torch.set_num_threads(previous_threads)
    report: dict[str, Any] = {
        "schema": "real_task_nca_checks_v1",
        "scope": "small CPU64 numerical qualification; no dataset or training",
        "torch_version": str(torch.__version__),
        "cpu64": cpu64,
        "passed": bool(cpu64["passed"]),
    }
    json.dumps(report, allow_nan=False)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="optional path for qualification JSON")
    args = parser.parse_args()
    serialized = json.dumps(run_checks(), indent=2, allow_nan=False) + "\n"
    if args.out is None:
        print(serialized, end="")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(serialized, encoding="utf-8")
    print(json.dumps({"out": str(args.out), "passed": json.loads(serialized)["passed"]}))


if __name__ == "__main__":
    main()
