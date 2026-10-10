"""Per-image binary vessel metrics for the frozen DRIVE developmental screen.

Inputs may be NumPy arrays or torch tensors. Metrics are always computed for
one original image at a time; callers aggregate those image-level values.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize


def _as_2d_array(value: Any, name: str) -> np.ndarray:
    """Convert a CPU/GPU tensor or array to a finite, normalized 2-D array."""
    if all(hasattr(value, attr) for attr in ("detach", "cpu", "numpy")):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be convertible to a numeric array") from exc

    while array.ndim > 2:
        singleton_axes = [axis for axis, length in enumerate(array.shape) if length == 1]
        if not singleton_axes:
            raise ValueError(f"{name} must describe one image, got shape {array.shape}")
        if 0 in singleton_axes:
            axis = 0
        elif array.ndim - 1 in singleton_axes:
            axis = array.ndim - 1
        else:
            axis = singleton_axes[0]
        array = np.squeeze(array, axis=axis)

    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must have a non-empty HxW shape, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if array.min() < 0.0 or array.max() > 1.0:
        raise ValueError(f"{name} values must be normalized to [0, 1]")
    return array


def _fov_boundary(fov: np.ndarray) -> np.ndarray:
    """Return FOV pixels adjacent to its 4-connected exterior."""
    padded = np.pad(fov, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]
    return fov & (~north | ~south | ~west | ~east)


def _betti_numbers(mask: np.ndarray, fov: np.ndarray) -> tuple[int, int]:
    """Count 8-connected foreground components and enclosed background holes.

    Holes use the dual 4-connectivity convention. A background component is a
    hole only when it does not touch the boundary of the fixed FOV.
    """
    foreground = mask & fov
    components = label(foreground, background=0, connectivity=2)
    beta0 = int(components.max(initial=0))

    background = fov & ~foreground
    if not background.any():
        return beta0, 0

    background_components = label(background, background=0, connectivity=1)
    boundary = _fov_boundary(fov)
    exterior_labels = set(
        int(value)
        for value in np.unique(background_components[background & boundary])
        if value != 0
    )
    all_background_labels = np.unique(background_components[background])
    beta1 = sum(
        int(value != 0 and int(value) not in exterior_labels)
        for value in all_background_labels
    )
    return beta0, int(beta1)


def _cldice(prediction: np.ndarray, target: np.ndarray, fov: np.ndarray) -> float:
    """Compute hard clDice after skeletonizing each complete image mask."""
    pred_skeleton = skeletonize(prediction) & fov
    target_skeleton = skeletonize(target) & fov
    pred_skeleton_count = int(pred_skeleton.sum())
    target_skeleton_count = int(target_skeleton.sum())

    if pred_skeleton_count == 0 and target_skeleton_count == 0:
        return 1.0 if not (prediction & fov).any() and not (target & fov).any() else 0.0

    topology_precision = (
        float((pred_skeleton & target).sum()) / pred_skeleton_count
        if pred_skeleton_count
        else 0.0
    )
    topology_sensitivity = (
        float((target_skeleton & prediction).sum()) / target_skeleton_count
        if target_skeleton_count
        else 0.0
    )
    denominator = topology_precision + topology_sensitivity
    if denominator == 0.0:
        return 0.0
    return float(2.0 * topology_precision * topology_sensitivity / denominator)


def segmentation_metrics(prob: Any, target: Any, fov: Any) -> dict[str, float | int]:
    """Return per-image Dice, clDice, precision, recall, and Betti errors.

    Probabilities, targets, and FOV masks are normalized to [0, 1]. The
    prediction and target are thresholded at 0.5. Skeletonization is performed
    on each complete binary image before FOV masking; all metric counts are
    then restricted to the fixed FOV. If both foreground sets are empty in the
    FOV, overlap metrics are 1; if only one is empty, they are 0. This keeps
    the API finite and deterministic for empty masks.
    """
    probabilities = _as_2d_array(prob, "prob")
    target_values = _as_2d_array(target, "target")
    fov_values = _as_2d_array(fov, "fov")
    if probabilities.shape != target_values.shape or probabilities.shape != fov_values.shape:
        raise ValueError(
            "prob, target, and fov must have identical spatial shapes; got "
            f"{probabilities.shape}, {target_values.shape}, and {fov_values.shape}"
        )

    prediction_mask = probabilities >= 0.5
    target_mask = target_values >= 0.5
    fov_mask = fov_values >= 0.5
    prediction = prediction_mask & fov_mask
    truth = target_mask & fov_mask

    true_positive = int((prediction & truth).sum())
    predicted_positive = int(prediction.sum())
    target_positive = int(truth.sum())

    if predicted_positive == 0 and target_positive == 0:
        dice = precision = recall = 1.0
    else:
        dice = (2.0 * true_positive) / (predicted_positive + target_positive)
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / target_positive if target_positive else 0.0

    pred_beta0, pred_beta1 = _betti_numbers(prediction_mask, fov_mask)
    target_beta0, target_beta1 = _betti_numbers(target_mask, fov_mask)
    return {
        "dice": float(dice),
        "cldice": _cldice(prediction_mask, target_mask, fov_mask),
        "precision": float(precision),
        "recall": float(recall),
        "betti0_error": abs(pred_beta0 - target_beta0),
        "betti1_error": abs(pred_beta1 - target_beta1),
    }
