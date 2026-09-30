"""Bounded numerical and oracle checks for the new A0 models.

These checks do not modify or replace the frozen v1 transport tests or results.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .a0_models import A0Model, SharedValueAttention, VARIANTS, transport_pair
from .data import make_batch
from .transport import apply_transport, prepare_transport


def _device_and_fork(device):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for A0 checks but is unavailable")
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    return device, torch.random.fork_rng(devices=devices)


def _finite_values(values):
    return bool(torch.isfinite(values).all().item())


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _max_abs(values):
    return float(values.detach().abs().max().item()) if values.numel() else 0.0


def _close_or_fail(actual, expected, *, atol, label):
    error = _max_abs(actual - expected)
    if not error <= atol:
        raise AssertionError(f"{label}: max absolute error {error} exceeds {atol}")
    return error


def _make_terminal_edges(batch, groups, shape, *, dtype, device):
    edges = []
    for axis, length in enumerate(shape):
        edge = torch.ones((batch, groups, *shape), dtype=dtype, device=device)
        terminal = [slice(None)] * edge.ndim
        terminal[axis + 2] = length - 1
        edge[tuple(terminal)] = 0
        edges.append(edge)
    return edges


def _pair_transport_checks():
    reports = {}
    for dim, shape in ((2, (3, 4)), (3, (2, 3, 4))):
        batch, groups, per_group = 2, 3, 2
        q = torch.randn(batch, groups * per_group, *shape, dtype=torch.float64)
        confidence = torch.rand(batch, groups, *shape, dtype=torch.float64)
        edges = []
        for axis, length in enumerate(shape):
            edge = 0.4 * torch.rand(batch, groups, *shape, dtype=torch.float64)
            terminal = [slice(None)] * edge.ndim
            terminal[axis + 2] = length - 1
            edge[tuple(terminal)] = 0
            edges.append(edge)
        tau = torch.tensor([0.7, 1.4, 2.8], dtype=torch.float64)
        prepared = prepare_transport(edges, tau)
        paired_q, paired_confidence = transport_pair(q, confidence, prepared)
        separate_q = apply_transport(q, prepared)
        separate_confidence = apply_transport(confidence, prepared)
        q_error = _close_or_fail(paired_q, separate_q, atol=1e-11,
                                 label=f"{dim}D packed numerator")
        c_error = _close_or_fail(paired_confidence, separate_confidence, atol=1e-11,
                                 label=f"{dim}D packed confidence")
        reports[f"{dim}d"] = {
            "packed_numerator_max_error": q_error,
            "packed_confidence_max_error": c_error,
        }
    return reports


def _normalized_gradcheck():
    batch, groups, per_group, shape = 1, 2, 2, (2, 3)
    q = torch.randn(batch, groups * per_group, *shape, dtype=torch.float64,
                    requires_grad=True)
    confidence = (0.6 + torch.rand(batch, groups, *shape, dtype=torch.float64)).requires_grad_()
    edge0 = 0.08 * torch.rand(batch, groups, *shape, dtype=torch.float64)
    edge1 = 0.08 * torch.rand(batch, groups, *shape, dtype=torch.float64)
    edge0[:, :, -1, :] = 0
    edge1[:, :, :, -1] = 0
    edge0.requires_grad_()
    edge1.requires_grad_()
    tau = torch.tensor([0.8, 1.7], dtype=torch.float64, requires_grad=True)
    epsilon = 1e-6

    def normalized(q_value, confidence_value, first_edge, second_edge, tau_value):
        prepared = prepare_transport([first_edge, second_edge], tau_value)
        numerator, denominator = transport_pair(q_value, confidence_value, prepared)
        return numerator / (denominator.repeat_interleave(per_group, dim=1) + epsilon)

    passed = torch.autograd.gradcheck(
        normalized, (q, confidence, edge0, edge1, tau),
        eps=1e-6, atol=3e-6, rtol=2e-4, fast_mode=True)
    if not passed:
        raise AssertionError("normalized paired-transport gradcheck failed")
    return {"passed": True, "inputs": ["q", "confidence", "edges", "tau"],
            "fast_mode": True, "epsilon": epsilon}


def _attention_mean_check():
    attention = SharedValueAttention(dim=2, width=5, message_width=8, groups=2).double()
    nn.init.zeros_(attention.query_key.weight)
    nn.init.zeros_(attention.query_key.bias)
    state = torch.randn(2, 5, 2, 3, dtype=torch.float64)
    emitted = torch.randn(2, 8, 2, 3, dtype=torch.float64)
    actual = attention(state, emitted)
    values = emitted.reshape(2, 2, 4, -1)
    expected = values.mean(dim=-1, keepdim=True).expand_as(values).reshape_as(emitted)
    error = _close_or_fail(actual, expected, atol=1e-12,
                           label="zero-query/key supplied-value attention mean")
    return {"spatial_mean_max_error": error}


def _new_input(dim, device):
    shape = (4, 5) if dim == 2 else (3, 3, 4)
    batch = 2
    x = torch.zeros((batch, 5, *shape), dtype=torch.float32, device=device)
    x[:, 3] = 1
    x[:, 4] = torch.randint(0, 2, (batch, *shape), device=device).float().mul_(2).sub_(1)
    flat = x.flatten(start_dim=2)
    n = math.prod(shape)
    source_index = torch.tensor([1, n - 2], device=device)
    target_index = torch.tensor([n - 1, n // 2], device=device)
    rows = torch.arange(batch, device=device)
    flat[rows, 0, source_index] = torch.tensor([-1.0, 1.0], device=device)
    flat[rows, 1, source_index] = 1.0
    flat[rows, 2, target_index] = 1.0
    return x, target_index


def _model_checks(device):
    width, message_width, groups, steps, seed = 8, 8, 2, 2, 8031
    init_reports = {}
    initial_medium_reports = {}
    for dim in (2, 3):
        for medium in ("constant", "learned"):
            raw_name, normalized_name = medium + "_raw", medium + "_normalized"
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + dim)
                raw = A0Model(dim, raw_name, width, message_width, groups, steps)
                torch.manual_seed(seed + dim)
                normalized = A0Model(dim, normalized_name, width, message_width, groups, steps)
            raw_state, normalized_state = raw.state_dict(), normalized.state_dict()
            if raw_state.keys() != normalized_state.keys():
                raise AssertionError(f"{dim}D {medium} raw/normalized state keys differ")
            unequal = [key for key in raw_state
                       if not torch.equal(raw_state[key], normalized_state[key])]
            if unequal:
                raise AssertionError(f"{dim}D {medium} paired initialization differs at {unequal}")
            required = ("confidence.", "readout.", "cells.", "stem.", "init_state.", "log_tau")
            if not all(any(key.startswith(prefix) for key in raw_state) for prefix in required):
                raise AssertionError(f"{dim}D {medium} common initialization families are incomplete")
            init_reports[f"{dim}d_{medium}"] = {
                "all_state_dict_equal": True,
                "checked_common_families": ["confidence", "readout", "cells", "stem", "init_state", "log_tau"],
            }

        for mode in ("raw", "normalized"):
            constant_name = "constant_" + mode
            learned_name = "learned_" + mode
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 100 + dim)
                constant = A0Model(dim, constant_name, width, message_width, groups, steps).to(device)
                torch.manual_seed(seed + 100 + dim)
                learned = A0Model(dim, learned_name, width, message_width, groups, steps).to(device)
            shared_keys = constant.state_dict().keys() & learned.state_dict().keys()
            unequal = [key for key in shared_keys
                       if not torch.equal(constant.state_dict()[key], learned.state_dict()[key])]
            if unequal:
                raise AssertionError(f"{dim}D {mode} common medium initialization differs at {unequal}")
            x, target_index = _new_input(dim, device)
            constant.eval()
            learned.eval()
            with torch.no_grad():
                constant_logits = constant(x, target_index)
                learned_logits = learned(x, target_index)
            error = _close_or_fail(learned_logits, constant_logits, atol=1e-7,
                                   label=f"{dim}D initial constant/learned {mode} output")
            initial_medium_reports[f"{dim}d_{mode}"] = {
                "shared_state_dict_equal": True,
                "initial_output_max_error": error,
            }

    forward_reports = {}
    for dim in (2, 3):
        x, target_index = _new_input(dim, device)
        for variant in VARIANTS:
            model = A0Model(dim, variant, width, message_width, groups, steps).to(device)
            model.eval()
            logits, diagnostics = model(x, target_index, diagnostics=True)
            if not _finite_values(logits):
                raise AssertionError(f"{dim}D {variant} produced nonfinite evaluation logits")
            if not _finite_tree(diagnostics):
                raise AssertionError(f"{dim}D {variant} produced nonfinite diagnostics")
            (logits.square().mean() + 0.1 * logits.mean()).backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(_finite_values(g) for g in grads):
                raise AssertionError(f"{dim}D {variant} produced missing or nonfinite gradients")
            confidence_grad = sum(float(p.grad.detach().abs().sum().item())
                                  for p in model.confidence.parameters() if p.grad is not None)
            if not math.isfinite(confidence_grad) or confidence_grad <= 0:
                raise AssertionError(f"{dim}D {variant} confidence gradient is zero or nonfinite")
            edge_grad = 0.0
            if variant.startswith("learned"):
                edge_grad = sum(float(p.grad.detach().abs().sum().item())
                                for p in model.edge_builders.parameters() if p.grad is not None)
                if not math.isfinite(edge_grad) or edge_grad <= 0:
                    raise AssertionError(f"{dim}D {variant} learned-edge gradient is zero or nonfinite")
            forward_reports[f"{dim}d_{variant}"] = {
                "eval_logits_finite": True,
                "diagnostics_finite": True,
                "parameter_gradients_finite": True,
                "confidence_gradient_l1": confidence_grad,
                "learned_edge_gradient_l1": edge_grad,
            }
            del model, logits
    return {"raw_normalized_initialization": init_reports,
            "initial_constant_learned_medium_match": initial_medium_reports,
            "eval_forward_backward": forward_reports}


def run_checks(device="cuda"):
    """Run the A0 software/numerical checks and return a JSON-safe report."""
    device, rng = _device_and_fork(device)
    with rng:
        torch.manual_seed(50271)
        report = {
            "device": str(device),
            "packed_pair_transport": _pair_transport_checks(),
            "normalized_pair_gradcheck": _normalized_gradcheck(),
            "shared_value_attention": _attention_mean_check(),
            "models": _model_checks(device),
        }
    return report


def _gather_target(field, target_index):
    groups = field.shape[1]
    return field.flatten(2).gather(
        2, target_index[:, None, None].expand(-1, groups, 1)).squeeze(-1)


def _sign_report(values, source_sign):
    decoded = torch.isfinite(values) & (values != 0)
    correct = decoded & ((values > 0) == (source_sign > 0))
    total = values.numel()
    decoded_count = int(decoded.sum().item())
    correct_count = int(correct.sum().item())
    return {
        "sign_accuracy": correct_count / total,
        "decoded_accuracy": (correct_count / decoded_count if decoded_count else None),
        "decoded_count": decoded_count,
        "ties_not_decoded": total - decoded_count,
    }


def oracle_diagnostics(device="cuda"):
    """Return fixed-seed source-oracle propagation metrics for every axis."""
    device, _ = _device_and_fork(device)
    groups = 4
    epsilon = 1e-6
    tau_values = (1.0, 16.0, 256.0, 4096.0)
    conditions = (("train_d16_long28", 16, 28),
                  ("long140_d16", 16, 140),
                  ("long140_d32", 32, 140),
                  ("long140_d64", 64, 140),
                  ("long140_d128", 128, 140))
    rows = []
    for dim in (2, 3):
        for condition, distance, long_size in conditions:
            for axis in range(dim):
                batch = make_batch(dim, "A", batch_size=8, distance=distance,
                                   seed=424242 + axis, device=device,
                                   long_size=long_size, axis=axis)
                source_value = batch["x"][:, 0:1]
                confidence_marker = batch["x"][:, 1:2]
                q = source_value.expand(-1, groups, *source_value.shape[2:]).contiguous()
                confidence = confidence_marker.expand(
                    -1, groups, *confidence_marker.shape[2:]).contiguous()
                edges = _make_terminal_edges(8, groups, batch["x"].shape[2:],
                                             dtype=batch["x"].dtype, device=device)
                tau = torch.tensor(tau_values, dtype=batch["x"].dtype, device=device)
                prepared = prepare_transport(edges, tau)
                numerator, denominator = transport_pair(q, confidence, prepared)
                normalized = numerator / (denominator + epsilon)
                raw_target = _gather_target(numerator, batch["target_index"])
                denominator_target = _gather_target(denominator, batch["target_index"])
                normalized_target = _gather_target(normalized, batch["target_index"])
                source_sign = (2 * batch["source_labels"] - 1).to(raw_target.dtype)[:, None]
                for group, tau_value in enumerate(tau_values):
                    raw = raw_target[:, group]
                    den = denominator_target[:, group]
                    norm = normalized_target[:, group]
                    raw_sign = _sign_report(raw, source_sign[:, 0])
                    norm_sign = _sign_report(norm, source_sign[:, 0])
                    row = {
                        "dim": dim,
                        "axis": axis,
                        "case": condition,
                        "distance": distance,
                        "long_size": long_size,
                        "seed": 424242 + axis,
                        "batch_size": 8,
                        "group": group,
                        "tau": tau_value,
                        "epsilon": epsilon,
                        "raw_sign_accuracy": raw_sign["sign_accuracy"],
                        "raw_decoded_accuracy": raw_sign["decoded_accuracy"],
                        "raw_decoded_count": raw_sign["decoded_count"],
                        "raw_ties_not_decoded": raw_sign["ties_not_decoded"],
                        "normalized_sign_accuracy": norm_sign["sign_accuracy"],
                        "normalized_decoded_accuracy": norm_sign["decoded_accuracy"],
                        "normalized_decoded_count": norm_sign["decoded_count"],
                        "normalized_ties_not_decoded": norm_sign["ties_not_decoded"],
                        "raw_abs_amplitude_min": float(raw.abs().min().item()),
                        "raw_abs_amplitude_mean": float(raw.abs().mean().item()),
                        "denominator_min": float(den.min().item()),
                        "denominator_mean": float(den.mean().item()),
                        "normalized_abs_error_max": float((norm - source_sign[:, 0]).abs().max().item()),
                        "normalized_abs_error_mean": float((norm - source_sign[:, 0]).abs().mean().item()),
                        "raw_zero_count": int((raw == 0).sum().item()),
                        "denominator_zero_count": int((den == 0).sum().item()),
                        "denominator_at_or_below_epsilon_count": int((den <= epsilon).sum().item()),
                        "normalized_zero_count": int((norm == 0).sum().item()),
                        "raw_nonfinite_count": int((~torch.isfinite(raw)).sum().item()),
                        "denominator_nonfinite_count": int((~torch.isfinite(den)).sum().item()),
                        "normalized_nonfinite_count": int((~torch.isfinite(norm)).sum().item()),
                    }
                    rows.append(row)
                    if tau_value == 4096.0:
                        nonfinite_count = (row["raw_nonfinite_count"]
                                           + row["denominator_nonfinite_count"]
                                           + row["normalized_nonfinite_count"])
                        if row["raw_sign_accuracy"] != 1.0 or nonfinite_count != 0:
                            raise AssertionError(
                                f"tau=4096 oracle qualification failed: {row}")
    return rows
