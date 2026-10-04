"""Deterministic NumPy-only binary carrier decoder.

Calibration arrays are ordered as all original maps followed by their flips.
Each map/cue has equal total weight: each of its open cells receives
``1 / open_count`` before the global normalization used by the mean logistic
loss. The fitted linear decoder is in raw W coordinates. ``p0`` and ``p1``
are the matching weighted open-cell class centroids used for reconstruction.
"""
from __future__ import annotations

from typing import Any

import numpy as np


CHANNELS = 24
L2 = 1e-3
MAX_ITERATIONS = 100
GRADIENT_TOLERANCE = 1e-7


def _as_float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _validate_fit_inputs(w: Any, y: Any, mask: Any):
    workspace = _as_float_array(w, "w")
    labels = _as_float_array(y, "y")
    open_input = _as_float_array(mask, "mask")
    if workspace.ndim != 4 or workspace.shape[1] != CHANNELS:
        raise ValueError("w must have shape [2B,24,H,W]")
    expected = (workspace.shape[0], 1, *workspace.shape[-2:])
    if workspace.shape[0] <= 0 or workspace.shape[0] % 2:
        raise ValueError("w must contain originals followed by an equal number of flips")
    if labels.shape != expected or open_input.shape != expected:
        raise ValueError("y and mask must both have shape [2B,1,H,W]")
    open_cells = open_input[:, 0] > 0.5
    counts = open_cells.sum(axis=(1, 2)).astype(np.int64)
    if np.any(counts == 0):
        raise ValueError("each calibration map/cue must contain at least one open cell")
    targets = labels[:, 0]
    open_labels = targets[open_cells]
    if not np.all(np.isclose(open_labels, 0.0) | np.isclose(open_labels, 1.0)):
        raise ValueError("open-cell y labels must be binary")
    return workspace, targets >= 0.5, open_cells, counts


def _sigmoid(score: np.ndarray) -> np.ndarray:
    # exp(-logaddexp(0,-x)) is stable for both large positive and negative x.
    return np.exp(-np.logaddexp(0.0, -score))


def _objective_gradient_hessian(design, target, weights, beta, ridge, hessian=True):
    score = design @ beta
    probability = _sigmoid(score)
    loss = float(np.sum(weights * (np.logaddexp(0.0, score) - target * score)))
    loss += 0.5 * ridge * float(beta[:-1] @ beta[:-1])
    residual = weights * (probability - target)
    gradient = design.T @ residual
    gradient[:-1] += ridge * beta[:-1]
    if not hessian:
        return loss, gradient, None
    curvature = weights * probability * (1.0 - probability)
    matrix = design.T @ (design * curvature[:, None])
    matrix[:-1, :-1] += ridge * np.eye(beta.size - 1, dtype=np.float64)
    return loss, gradient, matrix


def _weighted_centroid(features, target, weights, label):
    selected = target == label
    mass = float(weights[selected].sum())
    if mass <= 0.0:
        return None, 0.0
    return (weights[selected, None] * features[selected]).sum(axis=0) / mass, mass


def fit(w: Any, y: Any, mask: Any) -> dict[str, Any]:
    """Fit the fixed 24-channel, no-extra-feature logistic decoder.

    The weighted objective is mean binary cross-entropy plus
    ``0.5 * 1e-3 * ||coef||^2``. Its intercept is unpenalized. Newton/IRLS
    uses deterministic Armijo backtracking and at most 100 accepted steps.
    Nonconvergence is returned in the fit record rather than hidden.
    """
    workspace, all_labels, open_cells, open_counts = _validate_fit_inputs(w, y, mask)
    maps = workspace.shape[0]
    features = workspace.transpose(0, 2, 3, 1)[open_cells]
    target = all_labels[open_cells].astype(np.float64)
    # Boolean indexing follows row-major map/cue, then spatial order.
    sample_weights = np.repeat(1.0 / open_counts, open_counts) / maps
    weight_sum = float(sample_weights.sum())
    if not np.isclose(weight_sum, 1.0, rtol=1e-12, atol=1e-12):
        raise ArithmeticError("map/cue sample weights failed to normalize")

    mean = np.sum(sample_weights[:, None] * features, axis=0)
    variance = np.sum(sample_weights[:, None] * np.square(features - mean), axis=0)
    scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), 1e-6)
    standardized = (features - mean) / scale
    design = np.column_stack((standardized, np.ones(len(standardized), dtype=np.float64)))

    beta = np.zeros(CHANNELS + 1, dtype=np.float64)
    loss_trace: list[float] = []
    converged = False
    status = "max_iterations"
    gradient_inf = float("inf")
    for _ in range(MAX_ITERATIONS):
        loss, gradient, hessian = _objective_gradient_hessian(
            design, target, sample_weights, beta, L2, hessian=True
        )
        if not loss_trace:
            loss_trace.append(loss)
        gradient_inf = float(np.max(np.abs(gradient)))
        if gradient_inf < GRADIENT_TOLERANCE:
            converged, status = True, "converged"
            break
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        directional_derivative = float(gradient @ direction)
        if not np.isfinite(direction).all() or not np.isfinite(directional_derivative) or directional_derivative <= 0.0:
            status = "invalid_newton_direction"
            break
        step = 1.0
        accepted = False
        for _backtrack in range(64):
            candidate = beta - step * direction
            candidate_loss, _, _ = _objective_gradient_hessian(
                design, target, sample_weights, candidate, L2, hessian=False
            )
            if candidate_loss <= loss - 1e-4 * step * directional_derivative:
                beta = candidate
                loss_trace.append(candidate_loss)
                accepted = True
                break
            step *= 0.5
        if not accepted:
            status = "line_search_failed"
            break
    else:
        status = "max_iterations"

    final_loss, final_gradient, _ = _objective_gradient_hessian(
        design, target, sample_weights, beta, L2, hessian=False
    )
    if not loss_trace or loss_trace[-1] != final_loss:
        loss_trace.append(final_loss)
    gradient_inf = float(np.max(np.abs(final_gradient)))
    if gradient_inf < GRADIENT_TOLERANCE:
        converged, status = True, "converged"

    standardized_coef = beta[:CHANNELS]
    raw_coef = standardized_coef / scale
    raw_intercept = float(beta[-1] - mean @ raw_coef)
    p0, mass0 = _weighted_centroid(features, target, sample_weights, 0.0)
    p1, mass1 = _weighted_centroid(features, target, sample_weights, 1.0)
    per_map_counts = []
    for map_index in range(maps):
        map_open = open_cells[map_index]
        map_labels = all_labels[map_index][map_open]
        per_map_counts.append({
            "open": int(open_counts[map_index]),
            "class0": int(np.count_nonzero(~map_labels)),
            "class1": int(np.count_nonzero(map_labels)),
        })
    class_counts = {
        "class0": int(np.count_nonzero(target == 0.0)),
        "class1": int(np.count_nonzero(target == 1.0)),
    }
    return {
        "raw_coefficient": raw_coef,
        "raw_intercept": raw_intercept,
        # Short aliases keep the fitted object convenient in interactive use.
        "weights": raw_coef,
        "intercept": raw_intercept,
        "mean": mean,
        "scale": scale,
        "standardized_weights": standardized_coef,
        "standardized_intercept": float(beta[-1]),
        "templates": {"p0": p0, "p1": p1},
        "p0": p0,
        "p1": p1,
        "template_weight_mass": {"p0": mass0, "p1": mass1},
        "converged": converged,
        "status": status,
        "iterations": max(0, len(loss_trace) - 1),
        "gradient_inf_norm": gradient_inf,
        "loss_trace": loss_trace,
        "counts": {
            "map_cues": maps,
            "open_cells": int(len(target)),
            "class_counts": class_counts,
            "per_map_cue": per_map_counts,
        },
        "sample_weighting": {
            "per_cell_before_global_normalization": "1/open_count_for_each_map_cue",
            "per_map_cue_total": 1.0,
            "global_weight_sum": weight_sum,
            "class_weighting": "none",
        },
        "l2": L2,
        "max_iterations": MAX_ITERATIONS,
        "gradient_tolerance": GRADIENT_TOLERANCE,
    }


def _validate_w(w: Any) -> np.ndarray:
    workspace = np.asarray(w)
    if workspace.ndim != 4 or workspace.shape[1] != CHANNELS:
        raise ValueError("w must have shape [2B,24,H,W]")
    if workspace.dtype.kind not in "biuf":
        raise ValueError("w must contain real numeric values")
    if not np.isfinite(workspace).all():
        raise ValueError("w contains nonfinite values")
    if workspace.dtype.kind in "biu":
        workspace = workspace.astype(np.float64)
    return workspace


def predict(w: Any, fitdict: dict[str, Any]) -> np.ndarray:
    """Return Boolean predictions ``[2B,H,W]`` using raw score >= 0."""
    workspace = _validate_w(w)
    coef = np.asarray(fitdict["raw_coefficient"], dtype=np.float64)
    intercept = float(fitdict["raw_intercept"])
    if coef.shape != (CHANNELS,) or not np.isfinite(coef).all() or not np.isfinite(intercept):
        raise ValueError("fitdict must contain finite 24-vector weights and intercept")
    score = np.einsum("bchw,c->bhw", workspace.astype(np.float64, copy=False), coef, optimize=True) + intercept
    return score >= 0.0


def _centroids(fitdict: dict[str, Any]):
    templates = fitdict.get("templates", {})
    p0 = templates.get("p0")
    p1 = templates.get("p1")
    if p0 is None or p1 is None:
        raise ValueError("both calibration class centroids are required")
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    if p0.shape != (CHANNELS,) or p1.shape != (CHANNELS,):
        raise ValueError("p0 and p1 must each have 24 channels")
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError("p0 and p1 must be finite")
    return p0, p1


def _open_mask(mask: Any, maps: int, height: int, width: int) -> np.ndarray:
    value = _as_float_array(mask, "mask")
    if value.shape == (maps, 1, height, width):
        value = value[:, 0]
    if value.shape != (maps, height, width):
        raise ValueError("mask must have shape [2B,1,H,W] or [2B,H,W]")
    return value > 0.5


def apply(w: Any, bits: Any, mask: Any, fitdict: dict[str, Any]) -> np.ndarray:
    """Replace open W cells by their predicted class centroid; preserve walls."""
    workspace = _validate_w(w)
    maps, _, height, width = workspace.shape
    open_cells = _open_mask(mask, maps, height, width)
    binary = np.asarray(bits)
    if binary.shape == (maps, 1, height, width):
        binary = binary[:, 0]
    if binary.shape != (maps, height, width):
        raise ValueError("bits must have shape [2B,H,W] (or [2B,1,H,W])")
    if not np.all(np.isin(binary, (0, 1, False, True))):
        raise ValueError("bits must be Boolean or binary")
    binary = binary.astype(bool, copy=False)
    p0, p1 = _centroids(fitdict)
    output = workspace.copy()
    channels_last = output.transpose(0, 2, 3, 1)
    channels_last[open_cells & ~binary] = p0
    channels_last[open_cells & binary] = p1
    return output


def source_only(w: Any, x: Any, fitdict: dict[str, Any]) -> np.ndarray:
    """Build a source-only W state using X=(mask,+seed,-seed), never labels.

    Positive seeds receive p1, negative seeds receive p0, and every other
    open cell receives their midpoint. Wall values are copied from input W.
    """
    workspace = _validate_w(w)
    inputs = _as_float_array(x, "x")
    maps, _, height, width = workspace.shape
    if inputs.shape != (maps, 3, height, width):
        raise ValueError("x must have shape [2B,3,H,W] with channels mask,+seed,-seed")
    open_cells = inputs[:, 0] > 0.5
    positive_seed = (inputs[:, 1] > 0.5) & open_cells
    negative_seed = (inputs[:, 2] > 0.5) & open_cells
    if np.any(positive_seed & negative_seed):
        raise ValueError("a cell cannot be both a positive and negative seed")
    p0, p1 = _centroids(fitdict)
    midpoint = 0.5 * (p0 + p1)
    output = workspace.copy()
    channels_last = output.transpose(0, 2, 3, 1)
    channels_last[open_cells] = midpoint
    channels_last[positive_seed] = p1
    channels_last[negative_seed] = p0
    return output
