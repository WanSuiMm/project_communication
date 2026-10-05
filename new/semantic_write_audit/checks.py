"""Small CPU checks for the zero-training semantic-write interventions.

The module is intentionally independent of training and data loading.  The
``intervention`` callback has the project contract

    intervention(q, z, readout_weight, alpha, x, y, condition, bias=None)
        -> (q_modified, stats)

An optional ``step_with_intervention`` callback can be supplied to check the
actual C8 and C8+R2 cell wiring.  It has the contract

    step_with_intervention(model, state, x, y, condition)
        -> (next_state, raw_q, stats)
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = (
    "natural",
    "source_negative_parallel",
    "oracle_solved_negative_parallel",
    "nullspace_zero",
    "source_orthogonal_matched",
)


def _weight_kernel(readout_weight: Tensor) -> Tensor:
    weight = readout_weight.detach()
    if weight.ndim == 1:
        weight = weight.reshape(1, -1, 1, 1)
    elif weight.ndim == 2:
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
    if weight.ndim != 4 or weight.shape[0] != 1 or weight.shape[2:] != (1, 1):
        raise AssertionError("checks expect a scalar 1x1-convolution readout")
    return weight


def _readout(value: Tensor, readout_weight: Tensor, bias: Tensor | None) -> Tensor:
    return F.conv2d(value, _weight_kernel(readout_weight), bias)


def _parallel(value: Tensor, readout_weight: Tensor) -> Tensor:
    weight = _weight_kernel(readout_weight).reshape(-1)
    denom = torch.dot(weight, weight)
    if not bool(denom > 0):
        raise AssertionError("readout weight must be nonzero for projection checks")
    coeff = torch.einsum("c,bchw->bhw", weight, value) / denom
    return coeff.unsqueeze(1) * weight.reshape(1, -1, 1, 1)


def _source_and_oracle_triggers(
    q: Tensor,
    z: Tensor,
    readout_weight: Tensor,
    alpha: float,
    x: Tensor,
    y: Tensor,
    bias: Tensor | None,
) -> tuple[Tensor, Tensor]:
    logits = _readout(z, readout_weight, bias)
    signed_label = 2.0 * y - 1.0
    signed_increment = signed_label * alpha * _readout(q, readout_weight, None)
    oracle_correct = (logits >= 0) == (y >= 0.5)
    oracle = (x[:, :1] > 0) & oracle_correct & (signed_increment < 0)

    source_sign = x[:, 1:2] - x[:, 2:3]
    source_mask = (x[:, 1:2] + x[:, 2:3]) > 0
    source_increment = source_sign * alpha * _readout(q, readout_weight, None)
    source_correct = (logits >= 0) == (source_sign > 0)
    source = source_mask & source_correct & (source_increment < 0)
    return source, oracle


def _matched_orthogonal_direction(
    q: Tensor, readout_weight: Tensor
) -> tuple[Tensor, Tensor]:
    """Return Q-perp, plus a deterministic unit fallback at zero-Q-perp sites."""
    q_parallel = _parallel(q, readout_weight)
    q_perp = q - q_parallel
    perp_norm = torch.linalg.vector_norm(q_perp, dim=1, keepdim=True)
    unit = q_perp / perp_norm.clamp_min(torch.finfo(q.dtype).tiny)

    weight = _weight_kernel(readout_weight).reshape(-1).to(dtype=q.dtype, device=q.device)
    denom = torch.dot(weight, weight)
    # Match the intervention implementation: project e_j off w, where j is
    # the smallest-magnitude readout coordinate (argmin is deterministic).
    j = int(torch.argmin(weight.abs()).item())
    basis = torch.zeros_like(weight)
    basis[j] = 1.0
    fallback = basis - (torch.dot(weight, basis) / denom) * weight
    fallback = fallback / torch.linalg.vector_norm(fallback)
    fallback = fallback.reshape(1, -1, 1, 1).expand_as(q)
    unit = torch.where(perp_norm > 1e-8, unit, fallback)
    return q_perp, unit


def _assert_close(actual: Tensor, expected: Tensor, label: str) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    except AssertionError as exc:
        raise AssertionError(f"{label}: {exc}") from exc


def _invoke(
    intervention: Callable,
    q: Tensor,
    z: Tensor,
    weight: Tensor,
    alpha: float,
    x: Tensor,
    y: Tensor,
    condition: str,
    bias: Tensor | None,
) -> tuple[Tensor, object]:
    result = intervention(q, z, weight, alpha, x, y, condition, bias=bias)
    if not isinstance(result, tuple) or len(result) != 2:
        raise AssertionError("intervention must return (q_modified, stats)")
    q_modified, stats = result
    if not isinstance(q_modified, Tensor) or q_modified.shape != q.shape:
        raise AssertionError("intervention must return a write tensor matching q")
    return q_modified, stats


def _make_fixture() -> tuple[Tensor, Tensor, Tensor, Tensor, float, tuple[Tensor, Tensor]]:
    """Return a six-site fixture with source, solved, unsolved and wall sites."""
    dtype = torch.float32
    weight = torch.zeros(1, 8, 1, 1, dtype=dtype)
    weight[0, 0, 0, 0] = 1.0
    bias = torch.tensor([0.1], dtype=dtype)
    alpha = 0.25

    # Site meanings: solved +source, solved -source, unsolved +source,
    # solved traversable non-source (harmful write), solved non-source
    # (helpful write for label 0), and a solved wall site.
    z = torch.zeros(1, 8, 1, 6, dtype=dtype)
    z[0, 0, 0] = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0, 1.0])
    q = torch.zeros_like(z)
    q[0, 0, 0] = torch.tensor([-0.4, 0.4, -0.4, -0.5, -0.3, -0.6])
    q[0, 1, 0, 1] = 0.3
    q[0, 2, 0, 2] = 0.25
    q[0, 3, 0, 3] = 0.2
    q[0, 2, 0, 4] = -0.1
    q[0, 4, 0, 5] = 0.15

    x = torch.zeros(1, 3, 1, 6, dtype=dtype)
    x[0, 0, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    x[0, 1, 0, 0] = 1.0
    x[0, 2, 0, 1] = 1.0
    x[0, 1, 0, 2] = 1.0
    y = torch.tensor([[[[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]]], dtype=dtype)
    return q, z, weight, x, alpha, (y, bias)


def _check_intervention_math(intervention: Callable) -> list[str]:
    q, z, weight, x, alpha, extra = _make_fixture()
    y, bias = extra
    source_trigger, oracle_trigger = _source_and_oracle_triggers(
        q, z, weight, alpha, x, y, bias
    )
    expected_source = torch.tensor([[[[True, True, False, False, False, False]]]])
    expected_oracle = torch.tensor([[[[True, True, False, True, False, False]]]])
    _assert_close(source_trigger.float(), expected_source.float(), "source trigger fixture")
    _assert_close(oracle_trigger.float(), expected_oracle.float(), "oracle trigger fixture")

    # Match the scorer's >= 0 tie convention.  At logit zero, the positive
    # source/label is currently correct and its negative write is clamped;
    # the negative source/label is currently wrong and must remain untouched,
    # even when its proposed write would further hurt that source class.
    tie_weight = torch.zeros(1, 8, 1, 1)
    tie_weight[0, 0, 0, 0] = 1.0
    tie_bias = torch.zeros(1)
    tie_z = torch.zeros(1, 8, 1, 3)
    tie_q = torch.zeros_like(tie_z)
    tie_q[0, 0, 0] = torch.tensor([-1.0, 1.0, -1.0])
    tie_x = torch.zeros(1, 3, 1, 3)
    tie_x[:, 0] = 1.0
    tie_x[0, 1, 0, 0] = 1.0
    tie_x[0, 2, 0, 1:] = 1.0
    tie_y = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    tie_source, tie_oracle = _source_and_oracle_triggers(
        tie_q, tie_z, tie_weight, 1.0, tie_x, tie_y, tie_bias
    )
    _assert_close(
        tie_source.float(),
        torch.tensor([[[[1.0, 0.0, 0.0]]]]),
        "tie rule source trigger",
    )
    _assert_close(
        tie_oracle.float(),
        torch.tensor([[[[1.0, 0.0, 0.0]]]]),
        "tie rule oracle trigger",
    )
    tie_source_q, _ = _invoke(
        intervention, tie_q, tie_z, tie_weight, 1.0, tie_x, tie_y,
        "source_negative_parallel", tie_bias
    )
    tie_oracle_q, _ = _invoke(
        intervention, tie_q, tie_z, tie_weight, 1.0, tie_x, tie_y,
        "oracle_solved_negative_parallel", tie_bias
    )
    expected_tie_q = tie_q.clone()
    expected_tie_q[0, 0, 0, 0] = 0.0
    _assert_close(tie_source_q, expected_tie_q, "tie rule source clamp")
    _assert_close(tie_oracle_q, expected_tie_q, "tie rule oracle clamp")

    q_parallel = _parallel(q, weight)
    source_mask = source_trigger.expand_as(q)
    oracle_mask = oracle_trigger.expand_as(q)
    expected_by_condition = {
        "natural": q,
        "source_negative_parallel": q - q_parallel * source_mask,
        "oracle_solved_negative_parallel": q - q_parallel * oracle_mask,
        "nullspace_zero": q_parallel,
    }

    results: dict[str, Tensor] = {}
    passed: list[str] = []
    for condition in CONDITIONS:
        q_modified, _ = _invoke(
            intervention, q, z, weight, alpha, x, y, condition, bias
        )
        results[condition] = q_modified

        delta_logits = _readout(z + alpha * q_modified, weight, bias) - _readout(
            z, weight, bias
        )
        affine_delta = alpha * _readout(q_modified, weight, None)
        _assert_close(delta_logits, affine_delta, f"affine logit increment: {condition}")

        if condition in expected_by_condition:
            _assert_close(
                q_modified,
                expected_by_condition[condition],
                f"write geometry: {condition}",
            )

        if condition in ("source_negative_parallel", "oracle_solved_negative_parallel"):
            trigger = source_trigger if condition == "source_negative_parallel" else oracle_trigger
            pre_margin = (
                (x[:, 1:2] - x[:, 2:3]) * _readout(z, weight, bias)
                if condition == "source_negative_parallel"
                else (2.0 * y - 1.0) * _readout(z, weight, bias)
            )
            margin_sign = (
                x[:, 1:2] - x[:, 2:3]
                if condition == "source_negative_parallel"
                else 2.0 * y - 1.0
            )
            post_margin = margin_sign * _readout(z + alpha * q_modified, weight, bias)
            if bool(torch.any(post_margin[trigger] + 2e-6 < pre_margin[trigger])):
                raise AssertionError(f"protected margin decreased: {condition}")

        passed.append(condition)

    # Source-triggered policies are causal to the source cue and current
    # state; changing oracle labels must not change their gates or writes.
    for condition in ("source_negative_parallel", "source_orthogonal_matched"):
        original, original_stats = _invoke(
            intervention, q, z, weight, alpha, x, y, condition, bias
        )
        flipped, flipped_stats = _invoke(
            intervention, q, z, weight, alpha, x, 1.0 - y, condition, bias
        )
        _assert_close(original, flipped, f"source gate must ignore y: {condition}")
        if isinstance(original_stats, dict) and "active" in original_stats:
            _assert_close(
                original_stats["active"].float(),
                flipped_stats["active"].float(),
                f"source active mask must ignore y: {condition}",
            )

    # Q-perp removal must preserve the immediate affine logit and leave a
    # nonzero state change on this fixture, which is the source of future risk.
    null_logits = _readout(z + alpha * results["nullspace_zero"], weight, bias)
    natural_logits = _readout(z + alpha * q, weight, bias)
    _assert_close(null_logits, natural_logits, "Q-perp removal immediate logits")
    if not bool(torch.linalg.vector_norm(results["nullspace_zero"] - q) > 1e-6):
        raise AssertionError("nullspace_zero did not remove any Q-perp component")

    # A simple nonlinear consumer witnesses that immediate equality does not
    # imply future equality: it reads a latent coordinate orthogonal to w.
    future_natural = torch.tanh(z + alpha * q).sum(dim=1, keepdim=True)
    future_null = torch.tanh(z + alpha * results["nullspace_zero"]).sum(dim=1, keepdim=True)
    if torch.allclose(future_natural, future_null, rtol=2e-5, atol=2e-6):
        raise AssertionError("Q-perp fixture failed to separate a nonlinear future consumer")

    # The matched control must use a Q-perp direction of exactly the removed
    # Q-parallel norm, and that perturbation must have zero immediate readout.
    q_modified = results["source_orthogonal_matched"]
    _, unit_perp = _matched_orthogonal_direction(q, weight)
    expected_orthogonal = q - unit_perp * torch.linalg.vector_norm(
        q_parallel, dim=1, keepdim=True
    ) * source_mask
    _assert_close(q_modified, expected_orthogonal, "matched orthogonal write")
    orth_delta = q_modified - q
    _assert_close(
        _readout(orth_delta, weight, None),
        torch.zeros_like(_readout(orth_delta, weight, None)),
        "matched orthogonal immediate readout",
    )
    delta_norm = torch.linalg.vector_norm(orth_delta, dim=1, keepdim=True)
    target_norm = torch.linalg.vector_norm(q_parallel, dim=1, keepdim=True) * source_trigger
    _assert_close(delta_norm, target_norm, "matched orthogonal perturbation norm")

    # The zero-Q-perp fallback is deterministic across calls.
    repeated, _ = _invoke(
        intervention, q, z, weight, alpha, x, y, "source_orthogonal_matched", bias
    )
    _assert_close(repeated, q_modified, "deterministic orthogonal fallback")

    # Repeat the clamp arithmetic in float64 to catch a small negative margin
    # regression hidden by float32 rounding.
    q64, z64, weight64, x64, y64, bias64 = (
        q.double(), z.double(), weight.double(), x.double(), y.double(), bias.double()
    )
    source64, oracle64 = _source_and_oracle_triggers(
        q64, z64, weight64, alpha, x64, y64, bias64
    )
    for condition, trigger, sign in (
        ("source_negative_parallel", source64, x64[:, 1:2] - x64[:, 2:3]),
        ("oracle_solved_negative_parallel", oracle64, 2.0 * y64 - 1.0),
    ):
        q64_modified, _ = _invoke(
            intervention, q64, z64, weight64, alpha, x64, y64, condition, bias64
        )
        before = sign * _readout(z64, weight64, bias64)
        after = sign * _readout(z64 + alpha * q64_modified, weight64, bias64)
        if bool(torch.any(after[trigger] < before[trigger])):
            raise AssertionError(f"float64 protected margin decreased: {condition}")
    passed.extend(("immediate_logit_affinity", "protected_margin_clamp",
                   "source_gate_ignores_y", "float64_clamp_regression", "classifier_tie_rule",
                   "qperp_immediate_and_future", "matched_orthogonal_control"))
    return passed


def _load_latent_factorial_module():
    path = ROOT / "new" / "latent_factorial" / "cells.py"
    module_name = "reaction_transport_semantic_write_audit_latent_factorial"
    prior = sys.modules.get(module_name)
    if prior is not None:
        return prior
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load latent-factorial cell at {path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _check_cell_wiring(intervention: Callable, step_with_intervention: Callable) -> list[str]:
    cells = _load_latent_factorial_module()
    passed: list[str] = []
    for arm in ("native", "native_r2"):
        model = cells.make_model(arm, seed=314159).cpu().eval()
        # The frozen cell initializes F_out/Q_out/G_out at zero.  Give those
        # heads a tiny deterministic, nonzero fixture so this wiring check
        # exercises both the recurrent write and the R2 sidecar without any
        # optimizer updates or checkpoint mutation.
        with torch.no_grad():
            model.readout.weight.zero_()
            model.readout.weight[0, 0, 0, 0] = 1.0
            model.readout.bias.fill_(0.1)

            model.f_in.weight.zero_()
            model.f_in.bias.zero_()
            # BASE feature order is incoming C, current Z, L(C), L(Z), X.
            model.f_in.weight[0, model.workspace_channels + 1, 0, 0] = 1.0
            model.f_out.weight.zero_()
            model.f_out.weight[0, 0, 0, 0] = 0.2
            model.f_out.bias.zero_()

            model.q_out.weight.zero_()
            if model.q_out.weight.shape[1] > 0:
                model.q_out.weight[1:].copy_(torch.linspace(
                    -0.002, 0.002, model.q_out.weight[1:].numel(),
                    dtype=model.q_out.weight.dtype,
                ).reshape_as(model.q_out.weight[1:]))
            model.q_out.bias.copy_(torch.tensor(
                [-0.2, 0.15, -0.1, 0.08, -0.06, 0.11, 0.03, -0.14],
                dtype=model.q_out.bias.dtype,
            ))
            if model.has_r2:
                model.g_out.weight.copy_(torch.linspace(
                    -0.01, 0.01, model.g_out.weight.numel(),
                    dtype=model.g_out.weight.dtype,
                ).reshape_as(model.g_out.weight))
                model.g_out.bias.copy_(torch.tensor(
                    [0.03, -0.02], dtype=model.g_out.bias.dtype
                ))

        height, width = 3, 4
        x = torch.zeros(1, 3, height, width, dtype=torch.float32)
        x[:, 0] = 1.0
        x[0, 0, 1, 2] = 0.0
        x[0, 1, 0, 0] = 1.0
        x[0, 2, 2, 3] = 1.0
        yy, xx = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij"
        )
        y = ((xx + yy) % 2).to(torch.float32)[None, None]
        state = list(model.initial(x))
        with torch.no_grad():
            state[0].copy_(torch.linspace(
                -0.2, 0.25, state[0].numel(), dtype=state[0].dtype
            ).reshape_as(state[0]))
            state[1].copy_(torch.linspace(
                -0.15, 0.2, state[1].numel(), dtype=state[1].dtype
            ).reshape_as(state[1]))
            # Ensure both source classes start on the correct side of the
            # readout so the positive source cue has a harmful negative Q.
            state[1][0, 0, 0, 0] = 0.5
            state[1][0, 0, 2, 3] = -0.5
            if len(state) == 3:
                state[2].copy_(torch.linspace(
                    -0.1, 0.1, state[2].numel(), dtype=state[2].dtype
                ).reshape_as(state[2]))
        state = tuple(state)

        reference = model.step(state, x)
        natural, raw_q, _ = step_with_intervention(model, state, x, y, "natural")
        if len(natural) != len(reference):
            raise AssertionError(f"{arm}: natural step returned the wrong state arity")
        for index, (actual, expected) in enumerate(zip(natural, reference)):
            _assert_close(actual, expected, f"{arm}: natural step state[{index}]")
        if torch.allclose(reference[0], state[0], rtol=2e-5, atol=2e-6):
            raise AssertionError(f"{arm}: fixture did not exercise the C update")
        if len(state) == 3 and torch.allclose(
            reference[2], state[2], rtol=2e-5, atol=2e-6
        ):
            raise AssertionError(f"{arm}: fixture did not exercise the R2 update")
        passed.append(f"{arm}_natural_step")

        for condition in ("source_negative_parallel", "nullspace_zero"):
            changed, observed_raw_q, _ = step_with_intervention(
                model, state, x, y, condition
            )
            _assert_close(observed_raw_q, raw_q, f"{arm}: raw Q stable for {condition}")
            q_modified, _ = _invoke(
                intervention,
                raw_q,
                state[1],
                model.readout.weight,
                model.alpha,
                x,
                y,
                condition,
                model.readout.bias,
            )
            _assert_close(
                changed[1],
                state[1] + model.alpha * q_modified,
                f"{arm}: latent write applied for {condition}",
            )
            invariant_indices = (0, 2) if len(state) == 3 else (0,)
            for index in invariant_indices:
                _assert_close(
                    changed[index],
                    reference[index],
                    f"{arm}: intervention isolation state[{index}]",
                )
        passed.append(f"{arm}_intervention_isolation")

        natural_logits = model.logits(natural)
        null_state, _, _ = step_with_intervention(model, state, x, y, "nullspace_zero")
        _assert_close(
            model.logits(null_state), natural_logits, f"{arm}: Q-perp immediate logits"
        )
        future_natural = model.step(natural, x)
        future_null = model.step(null_state, x)
        if all(
            torch.allclose(a, b, rtol=2e-5, atol=2e-6)
            for a, b in zip(future_natural, future_null)
        ):
            raise AssertionError(f"{arm}: Q-perp removal had no future state effect")
        passed.append(f"{arm}_future_non_equivalence")
    return passed


def run_checks(
    intervention: Callable,
    step_with_intervention: Callable | None = None,
) -> dict[str, object]:
    """Run cheap CPU geometry and optional cell-wiring assertions.

    No training, optimizer, data, or accelerator is used.  Supplying the
    project step callback additionally checks the frozen C8 and C8+R2 cell
    update and confirms that Q interventions leave C and R unchanged.
    """
    passed = _check_intervention_math(intervention)
    if step_with_intervention is not None:
        passed.extend(_check_cell_wiring(intervention, step_with_intervention))
    return {
        "status": "passed",
        "device": "cpu",
        "training_updates": 0,
        "checks": passed,
        "cell_wiring_checked": step_with_intervention is not None,
    }
