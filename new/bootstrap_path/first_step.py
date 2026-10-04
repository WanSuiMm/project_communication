"""One-update AdamW audit for the StreamingCell bootstrap path."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LR = 0.001
WEIGHT_DECAY = 0.0001
CLIP_NORM = 1.0
BETAS = (0.9, 0.999)
EPS = 1e-8


def _load_file(name: str, path: Path, *, register: bool = True):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_exact_dependencies():
    """Load the named project modules without trusting cached top-level names."""
    aliases = ("masked_cells", "revision_cells", "tasks")
    sentinel = object()
    saved_modules = {name: sys.modules.get(name, sentinel) for name in aliases}
    saved_path = list(sys.path)
    private_names = ("_bootstrap_path_stream_cells", "_bootstrap_path_training")
    try:
        wanted = [
            ROOT / "new/streaming_carry",
            ROOT / "new/short_bptt",
            ROOT / "new/workspace_revision",
            ROOT / "new/nca_inertial_wind_tunnel",
            ROOT / "new/masked_medium",
        ]
        sys.path[:] = [str(p) for p in wanted] + [p for p in saved_path if p not in {str(q) for q in wanted}]
        _load_file("masked_cells", ROOT / "new/masked_medium/masked_cells.py")
        _load_file("revision_cells", ROOT / "new/workspace_revision/revision_cells.py")
        _load_file("tasks", ROOT / "new/nca_inertial_wind_tunnel/tasks.py")
        stream_mod = _load_file(private_names[0], ROOT / "new/streaming_carry/stream_cells.py")
        training_mod = _load_file(private_names[1], ROOT / "new/short_bptt/training.py")
        return stream_mod.StreamingCell, training_mod.backward_trajectory
    finally:
        sys.path[:] = saved_path
        for name in private_names:
            sys.modules.pop(name, None)
        for name, prior in saved_modules.items():
            if prior is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def _norm(value: torch.Tensor) -> float:
    return float(value.detach().double().norm().cpu())


def _maxabs(value: torch.Tensor) -> float:
    return float(value.detach().abs().max().cpu()) if value.numel() else 0.0


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _singular_summary(matrix: torch.Tensor) -> dict:
    a = matrix.detach().reshape(matrix.shape[0], -1).double()
    values = torch.linalg.svdvals(a).cpu().numpy()
    energy = values * values
    total = float(energy.sum())
    tail = float(energy[1:].sum() / total) if total else 0.0
    rank = int((values > (values[0] * 1e-8 if len(values) and values[0] else 0)).sum())
    return {
        "singular_values": values.tolist(),
        "relative_tail_energy_after_first": tail,
        "rank_relative_1e-8": rank,
        "frobenius_norm": float(np.sqrt(total)),
    }


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.detach().double()
    expected = expected.detach().double()
    denom = max(float(actual.norm()), float(expected.norm()), 1e-30)
    return float((actual - expected).norm() / denom)


def _spectrum_rows(rows: np.ndarray) -> dict:
    rows = np.asarray(rows, dtype=np.float64)
    singular = np.linalg.svd(rows, compute_uv=False)
    energy = singular * singular
    total = float(energy.sum())
    return {
        "singular_values": singular.tolist(),
        "relative_tail_energy_after_first": float(energy[1:].sum() / total) if total else 0.0,
        "rank_relative_1e-8": int((singular > (singular[0] * 1e-8 if len(singular) and singular[0] else 0)).sum()),
        "rows": int(rows.shape[0]),
        "columns": int(rows.shape[1]),
        "squared_energy": total,
    }


def _write_diagnostics(z_write: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> tuple[dict, np.ndarray, np.ndarray]:
    valid = mask[:, 0].bool()
    features = z_write.permute(0, 2, 3, 1)[valid].detach().double().cpu().numpy()
    labels = y[:, 0][valid].detach().double().cpu().numpy().reshape(-1)
    if not len(features):
        raise ValueError("The supplied batch has no traversable pixels")
    centered = features - features.mean(axis=0, keepdims=True)
    target = labels - labels.mean()
    if float(target @ target) > 0:
        coefficients, *_ = np.linalg.lstsq(centered, target, rcond=None)
        projected = centered @ coefficients
        projection_fraction = float((projected @ projected) / (target @ target))
        projection_fraction = min(1.0, max(0.0, projection_fraction))
    else:
        projection_fraction = None
    return ({
        "finite": bool(np.isfinite(features).all() and np.isfinite(labels).all()),
        "open_pixel_rows": int(len(features)),
        "l2_norm": float(np.linalg.norm(features)),
        "mean_absolute_write": float(np.abs(features).mean()),
        "feature_spectrum_uncentered": _spectrum_rows(features),
        "feature_spectrum_centered": _spectrum_rows(centered),
        "semantic_projection_fraction": projection_fraction,
        "semantic_projection_definition": (
            "Centered binary labels projected by least squares onto centered post-update Z writes "
            "over traversable pixels; squared projected-label norm divided by centered-label norm."
        ),
    }, features, labels)


def audit_first_step(model, data, out_path=None) -> dict:
    """Run K8/64-step backward, clip to 1, and apply exactly one default AdamW step.

    ``out_path`` may name an NPZ file or a directory (which receives
    ``first_step_arrays.npz``). The model is updated in place.
    """
    if not all(k in data for k in ("x", "y", "mask")):
        raise KeyError("data must contain x, y, and mask")
    params = list(model.named_parameters())
    if not params:
        raise ValueError("model has no parameters")
    device = params[0][1].device
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
    if not all(torch.is_tensor(batch[k]) and _finite(batch[k]) for k in ("x", "y", "mask")):
        raise FloatingPointError("x, y, and mask must be finite tensors")

    _cell_type, backward_trajectory = _load_exact_dependencies()
    model.zero_grad(set_to_none=True)
    loss, _, trace = backward_trajectory(model, batch, gradient_horizon=8, steps=64, loss_every=8)
    if not _finite(loss):
        raise FloatingPointError("Nonfinite trajectory loss")
    before = {name: p.detach().clone() for name, p in params}
    raw_grads = {name: (None if p.grad is None else p.grad.detach().clone()) for name, p in params}
    if any(g is not None and not _finite(g) for g in raw_grads.values()):
        raise FloatingPointError("Nonfinite trajectory gradient")
    preclip_norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], CLIP_NORM)
    clipped_grads = {name: (None if p.grad is None else p.grad.detach().clone()) for name, p in params}
    if any(g is not None and not _finite(g) for g in clipped_grads.values()):
        raise FloatingPointError("Nonfinite clipped gradient")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer.step()

    parameter_rows = {}
    arrays = {}
    module_accum = {}
    for name, parameter in params:
        theta0 = before[name]
        raw = raw_grads[name]
        grad = clipped_grads[name]
        delta = parameter.detach() - theta0
        state = optimizer.state.get(parameter, {})
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        step_value = state.get("step")
        step_count = int(step_value.item()) if torch.is_tensor(step_value) else (int(step_value) if step_value is not None else None)

        if grad is None:
            expected_delta = torch.zeros_like(theta0)
            decay_delta = torch.zeros_like(theta0)
        else:
            decay_start = theta0 * (1.0 - LR * WEIGHT_DECAY)
            decay_delta = decay_start - theta0
            if exp_avg is not None and exp_avg_sq is not None:
                m_hat = exp_avg / (1.0 - BETAS[0] ** step_count)
                v_hat = exp_avg_sq / (1.0 - BETAS[1] ** step_count)
                expected_parameter = decay_start - LR * m_hat / (v_hat.sqrt() + EPS)
            else:
                expected_parameter = decay_start
            expected_delta = expected_parameter - theta0
        grad_norm = _norm(grad) if grad is not None else None
        row = {
            "shape": list(parameter.shape),
            "parameter_l2_before": _norm(theta0),
            "gradient_l2_unclipped": _norm(raw) if raw is not None else None,
            "gradient_l2_applied": grad_norm,
            "gradient_max_abs_applied": _maxabs(grad) if grad is not None else None,
            "delta_l2_actual": _norm(delta),
            "delta_max_abs_actual": _maxabs(delta),
            "decoupled_weight_decay_delta_l2": _norm(decay_delta),
            "delta_to_gradient_norm_ratio": (_norm(delta) / grad_norm) if grad_norm else None,
            "optimizer_step": step_count,
            "exp_avg_l2": _norm(exp_avg) if exp_avg is not None else None,
            "exp_avg_sq_l2": _norm(exp_avg_sq) if exp_avg_sq is not None else None,
            "first_step_formula_relative_error": _relative_error(delta, expected_delta),
            "finite": bool(_finite(parameter) and _finite(delta) and (grad is None or _finite(grad))),
        }
        parameter_rows[name] = row
        group = name.split(".", 1)[0]
        acc = module_accum.setdefault(group, {"parameter_count": 0, "gradient_l2_unclipped_sq": 0.0,
                                               "gradient_l2_applied_sq": 0.0, "delta_l2_sq": 0.0})
        acc["parameter_count"] += parameter.numel()
        if raw is not None:
            acc["gradient_l2_unclipped_sq"] += _norm(raw) ** 2
        if grad is not None:
            acc["gradient_l2_applied_sq"] += _norm(grad) ** 2
        acc["delta_l2_sq"] += _norm(delta) ** 2
        arrays[f"parameter_before__{name}"] = theta0.detach().cpu().numpy()
        arrays[f"gradient_unclipped__{name}"] = raw.detach().cpu().numpy() if raw is not None else np.zeros(tuple(parameter.shape), np.float32)
        arrays[f"gradient_applied__{name}"] = grad.detach().cpu().numpy() if grad is not None else np.zeros(tuple(parameter.shape), np.float32)
        arrays[f"parameter_delta__{name}"] = delta.detach().cpu().numpy()
        if exp_avg is not None:
            arrays[f"exp_avg__{name}"] = exp_avg.detach().cpu().numpy()
            arrays[f"exp_avg_sq__{name}"] = exp_avg_sq.detach().cpu().numpy()

    for row in module_accum.values():
        row["gradient_l2_unclipped"] = float(np.sqrt(row.pop("gradient_l2_unclipped_sq")))
        row["gradient_l2_applied"] = float(np.sqrt(row.pop("gradient_l2_applied_sq")))
        row["delta_l2_actual"] = float(np.sqrt(row.pop("delta_l2_sq")))

    qweight = next((p for n, p in params if n == "q_out.weight"), None)
    if qweight is None or qweight.ndim < 2:
        raise ValueError("Expected q_out.weight with output and input dimensions")
    qg = clipped_grads["q_out.weight"].reshape(qweight.shape[0], -1)
    if tuple(qg.shape) != (8, 16):
        raise ValueError(f"Expected q_out.weight matrix [8,16], got {tuple(qg.shape)}")
    qd = (qweight.detach() - before["q_out.weight"]).reshape(qweight.shape[0], -1)
    qbefore = before["q_out.weight"].reshape(qweight.shape[0], -1)
    qdata_delta = qd + LR * WEIGHT_DECAY * qbefore
    qg64, qd64 = qg.double(), qdata_delta.double()
    denom = float(qg64.square().sum())
    scalar = float((qd64 * qg64).sum() / denom) if denom else None
    proportional_error = (float((qd64 - scalar * qg64).norm() / max(float(qd64.norm()), 1e-30))
                          if scalar is not None else None)
    exact_data_update = -LR * qg / (qg.abs() + EPS)
    qgrad_geometry = _singular_summary(qg)
    qdelta_geometry = _singular_summary(qd)
    exact_geometry = _singular_summary(exact_data_update)
    u, _, vh = torch.linalg.svd(qg.double(), full_matrices=False)
    r = next((p.detach().reshape(-1).double() for n, p in params if n == "readout.weight"), None)
    if r is not None and r.numel() == qg.shape[0] and float(r.norm()) > 0:
        readout_energy_fraction = float(((r / r.norm()) @ qg.double()).square().sum() / qg.double().square().sum().clamp_min(1e-300))
        readout_top_left_cosine = float(abs(torch.dot(u[:, 0], r / r.norm())))
    else:
        readout_energy_fraction = None
        readout_top_left_cosine = None
    sign_outer = torch.sign(u[:, :1]) @ torch.sign(vh[:1, :])
    nonzero = qg != 0
    sign_match = float((torch.sign(qg[nonzero]) == sign_outer[nonzero]).double().mean()) if bool(nonzero.any()) else None
    sign_approx = -LR * sign_outer

    with torch.no_grad():
        state0 = model.initial(batch["x"])
        state1 = model.step(state0, batch["x"])
        z_write = state1[1] - state0[1]
        write_summary, open_writes, open_labels = _write_diagnostics(z_write, batch["y"], batch["mask"])
    arrays["q_out_weight_gradient_matrix"] = qg.detach().cpu().numpy()
    arrays["q_out_weight_delta_matrix"] = qd.detach().cpu().numpy()
    arrays["q_out_weight_exact_firststep_matrix"] = exact_data_update.detach().cpu().numpy()
    arrays["z_write_open_pixels"] = open_writes
    arrays["labels_open_pixels"] = open_labels

    focus = {name: parameter_rows[name] for name in ("q_out.weight", "q_out.bias", "readout.bias")
             if name in parameter_rows}
    summary = {
        "protocol": "bootstrap_path_first_step_v1",
        "loss": float(loss.detach().cpu()),
        "trajectory": {k: int(v) for k, v in trace.items()},
        "optimizer": {"name": "AdamW", "steps": 1, "lr": LR, "weight_decay": WEIGHT_DECAY,
                      "betas": list(BETAS), "eps": EPS, "clip_norm": CLIP_NORM,
                      "global_gradient_norm_unclipped": float(preclip_norm.detach().cpu()),
                      "global_gradient_norm_applied": float(np.sqrt(sum(_norm(g) ** 2 for g in clipped_grads.values() if g is not None)))},
        "parameters": parameter_rows,
        "focus_parameters": focus,
        "module_groups": module_accum,
        "q_out_weight_geometry": {
            "matrix_definition": "q_out.weight reshaped as [8,16] (output channels by hidden channels)",
            "matrix_shape": list(qg.shape),
            "applied_gradient": qgrad_geometry,
            "actual_parameter_delta": qdelta_geometry,
            "exact_epsilon_stabilized_firststep": exact_geometry,
            "actual_delta_relative_error_to_best_scalar_times_gradient": proportional_error,
            "best_scalar_times_gradient": scalar,
            "exact_formula_relative_error": parameter_rows["q_out.weight"]["first_step_formula_relative_error"],
            "readout_r_gradient_direction_identity": {
                "gradient_energy_in_readout_direction": readout_energy_fraction,
                "top_left_singular_vector_absolute_cosine_with_r": readout_top_left_cosine,
            },
            "sign_factorization": {
                "sign_pattern_agreement_with_top_gradient_singular_factors": sign_match,
                "relative_error_exact_adam_direction_vs_rank_one_sign_outer_product": _relative_error(exact_data_update, sign_approx),
            },
        },
        "post_update_forward_step": {"z_write": write_summary},
        "interpretation_boundary": {
            "claims_general_mechanism": False,
            "scope": "One initialized model, one fixed batch, one optimizer step, and one post-update forward step.",
            "claim_limit": "Gradient rank and optimizer-update rank are reported separately; a rank-one gradient alone does not establish a rank-one parameter update or a general mechanism.",
        },
    }
    # Fail before returning a report that cannot be represented as strict JSON.
    import json
    json.dumps(summary, allow_nan=False)
    if out_path is not None:
        out = Path(out_path)
        if out.suffix.lower() == ".npz":
            arrays_path = out
        elif out.suffix:
            arrays_path = out.with_suffix(".npz")
        else:
            out.mkdir(parents=True, exist_ok=True)
            arrays_path = out / "first_step_arrays.npz"
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arrays_path, **arrays)
        summary["arrays_file"] = arrays_path.name
    return summary
