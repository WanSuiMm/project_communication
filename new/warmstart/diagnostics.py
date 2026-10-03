"""Compact paired correctness turnover diagnostics for warm-start rollouts.

Correctness is paired across the original and source-flipped worlds and is
counted only on the changed component. Pixel and time observations are
descriptive; maps remain the reporting units.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch


TRACE_STEPS = 128
ENDPOINTS = (32, 64, 128)
SURVIVAL_START = 64
SURVIVAL_END = 128


def _binary_array(name: str, value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == np.bool_:
        return array
    if not np.issubdtype(array.dtype, np.number) or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be boolean or contain only 0/1")
    return array.astype(bool, copy=False)


def _changed_array(changed: Any, batch: int, height: int, width: int) -> np.ndarray:
    array = _binary_array("changed", changed)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.shape != (batch, height, width):
        raise ValueError("changed must have shape [B,H,W] or [B,1,H,W] matching correct")
    return array


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _equal_map_rate(numerators: np.ndarray, denominators: np.ndarray) -> tuple[float | None, int]:
    eligible = denominators > 0
    if not np.any(eligible):
        return None, 0
    values = numerators[eligible].astype(np.float64) / denominators[eligible]
    return float(values.mean()), int(np.count_nonzero(eligible))


def _aggregate_rate(numerators: np.ndarray, denominators: np.ndarray) -> dict[str, Any]:
    numerator = int(numerators.sum())
    denominator = int(denominators.sum())
    mean, eligible_maps = _equal_map_rate(numerators, denominators)
    return {
        "equal_map_mean": mean,
        "equal_map_eligible_maps": eligible_maps,
        "pooled_numerator": numerator,
        "pooled_denominator": denominator,
        "pooled_rate": _rate(numerator, denominator),
    }


def _normalise_trace(correct: Any) -> np.ndarray:
    array = _binary_array("correct", correct)
    if array.ndim != 4:
        raise ValueError("correct must have shape [T+1,B,H,W]")
    return array


def summarize_correct_traces(
    correct: Any,
    changed: Any,
    survival_start: int = SURVIVAL_START,
    survival_end: int = SURVIVAL_END,
) -> dict[str, Any]:
    """Summarize paired-correct traces using integer acquisition/loss counts.

    ``correct`` is [T+1,B,H,W], already requiring both worlds to be right.
    ``changed`` is the [B,H,W] flipped component. The generalized survival
    bounds make this helper usable with small hand-computed fixtures.
    """
    trace = _normalise_trace(correct)
    steps, maps, height, width = trace.shape
    changed_map = _changed_array(changed, maps, height, width)
    last_step = steps - 1
    if not (0 <= survival_start <= survival_end <= last_step):
        raise ValueError("survival bounds must lie inside the observed trace")

    pixels = changed_map.reshape(maps, -1).sum(axis=1, dtype=np.int64)
    correct_counts: list[np.ndarray] = []
    step_rows: list[dict[str, Any]] = []
    for step in range(steps):
        selected_correct = trace[step] & changed_map
        current = selected_correct.reshape(maps, -1).sum(axis=1, dtype=np.int64)
        correct_counts.append(current)
        q_rate = _aggregate_rate(current, pixels)
        q = {
            "equal_map_mean": q_rate["equal_map_mean"],
            "equal_map_eligible_maps": q_rate["equal_map_eligible_maps"],
            "pooled_correct": q_rate["pooled_numerator"],
            "pooled_pixels": q_rate["pooled_denominator"],
            "pooled_accuracy": q_rate["pooled_rate"],
        }
        row: dict[str, Any] = {
            "t": int(step),
            "correct": int(current.sum()),
            "changed_pixels": int(pixels.sum()),
            "q": q,
            "acquired": None,
            "wrong_old": None,
            "g_acquisition_rate": None,
            "destroyed": None,
            "correct_old": None,
            "d_destruction_rate": None,
            "delta_correct": None,
            "accounting_error": None,
        }
        if step:
            previous = trace[step - 1] & changed_map
            acquired_map = ((~previous) & selected_correct).reshape(maps, -1).sum(axis=1, dtype=np.int64)
            destroyed_map = (previous & (~selected_correct)).reshape(maps, -1).sum(axis=1, dtype=np.int64)
            wrong_old_map = pixels - correct_counts[step - 1]
            correct_old_map = correct_counts[step - 1]
            acquired = int(acquired_map.sum())
            destroyed = int(destroyed_map.sum())
            delta = int(current.sum() - correct_old_map.sum())
            accounting_error = int(delta - (acquired - destroyed))
            if accounting_error != 0:
                raise AssertionError((step, delta, acquired, destroyed, accounting_error))
            row.update({
                "acquired": acquired,
                "wrong_old": int(wrong_old_map.sum()),
                "g_acquisition_rate": _aggregate_rate(acquired_map, wrong_old_map),
                "destroyed": destroyed,
                "correct_old": int(correct_old_map.sum()),
                "d_destruction_rate": _aggregate_rate(destroyed_map, correct_old_map),
                "delta_correct": delta,
                "accounting_error": accounting_error,
            })
        step_rows.append(row)

    start = trace[survival_start] & changed_map
    end = trace[survival_end] & changed_map
    sustained = changed_map & np.all(trace[survival_start:survival_end + 1], axis=0)
    start_counts = start.reshape(maps, -1).sum(axis=1, dtype=np.int64)
    sustained_counts = sustained.reshape(maps, -1).sum(axis=1, dtype=np.int64)
    retained_counts = (start & end).reshape(maps, -1).sum(axis=1, dtype=np.int64)
    survival_rows = []
    for index in range(maps):
        survival_rows.append({
            "map_index": int(index),
            "changed_pixels": int(pixels[index]),
            "correct_at_start": int(start_counts[index]),
            "correct_at_start_and_every_step": int(sustained_counts[index]),
            "correct_at_endpoint": int(retained_counts[index]),
            "multi_step_survival_rate": _rate(int(sustained_counts[index]), int(start_counts[index])),
            "endpoint_retention_rate": _rate(int(retained_counts[index]), int(start_counts[index])),
        })

    coverage_acquired = changed_map & (~(trace[survival_start])) & trace[survival_end]
    coverage_destroyed = changed_map & trace[survival_start] & (~trace[survival_end])
    coverage_acquired_counts = coverage_acquired.reshape(maps, -1).sum(axis=1, dtype=np.int64)
    coverage_destroyed_counts = coverage_destroyed.reshape(maps, -1).sum(axis=1, dtype=np.int64)
    coverage_wrong_counts = pixels - start_counts
    coverage_start_correct = start_counts
    coverage_net = int(correct_counts[survival_end].sum() - correct_counts[survival_start].sum())
    coverage_acquired_total = int(coverage_acquired_counts.sum())
    coverage_destroyed_total = int(coverage_destroyed_counts.sum())
    if coverage_net != coverage_acquired_total - coverage_destroyed_total:
        raise AssertionError((coverage_net, coverage_acquired_total, coverage_destroyed_total))

    ever = np.zeros((maps, height, width), dtype=bool)
    relapsed = np.zeros_like(ever)
    for step in range(steps):
        relapsed |= ever & (~trace[step])
        ever |= trace[step]
    ever &= changed_map
    relapsed &= changed_map
    ever_counts = ever.reshape(maps, -1).sum(axis=1, dtype=np.int64)
    relapse_counts = relapsed.reshape(maps, -1).sum(axis=1, dtype=np.int64)

    survival_equal, survival_maps = _equal_map_rate(sustained_counts, start_counts)
    retention_equal, retention_maps = _equal_map_rate(retained_counts, start_counts)
    survival_pooled_start = int(start_counts.sum())
    retention_pooled = int(retained_counts.sum())
    coverage = {
        "from_step": int(survival_start),
        "to_step": int(survival_end),
        "acquired": coverage_acquired_total,
        "wrong_at_start": int(coverage_wrong_counts.sum()),
        "g_acquisition_rate": _aggregate_rate(coverage_acquired_counts, coverage_wrong_counts),
        "destroyed": coverage_destroyed_total,
        "correct_at_start": int(coverage_start_correct.sum()),
        "d_destruction_rate": _aggregate_rate(coverage_destroyed_counts, coverage_start_correct),
        "delta_correct": coverage_net,
        "accounting_error": 0,
    }
    return {
        "schema_version": "warmstart-turnover-v1",
        "scope": (
            "Descriptive paired correctness on the changed component. d is the per-step destruction rate; "
            "it is complementary to retention among previously correct pixels, so those summaries are not independent."
        ),
        "trace_steps": {"first": 0, "last": int(last_step)},
        "changed_pixels_per_map": [int(value) for value in pixels],
        "steps": step_rows,
        "survival": {
            "from_step": int(survival_start),
            "through_step": int(survival_end),
            "per_map": survival_rows,
            "multi_step_survival": {
                "equal_map_mean": survival_equal,
                "equal_map_eligible_maps": survival_maps,
                "pooled_correct_at_start_and_every_step": int(sustained_counts.sum()),
                "pooled_correct_at_start": survival_pooled_start,
                "pooled_rate": _rate(int(sustained_counts.sum()), survival_pooled_start),
            },
            "endpoint_retention": {
                "equal_map_mean": retention_equal,
                "equal_map_eligible_maps": retention_maps,
                "pooled_correct_at_start_and_endpoint": retention_pooled,
                "pooled_correct_at_start": survival_pooled_start,
                "pooled_rate": _rate(retention_pooled, survival_pooled_start),
            },
        },
        "coverage_gain": coverage,
        "relapse": {
            "definition": "changed pixels correct at least once, then wrong at a later observed step",
            "ever_correct": int(ever_counts.sum()),
            "relapsed": int(relapse_counts.sum()),
            "equal_map_rate": _aggregate_rate(relapse_counts, ever_counts),
            "per_map": [
                {
                    "map_index": int(index),
                    "ever_correct": int(ever_counts[index]),
                    "relapsed": int(relapse_counts[index]),
                    "rate": _rate(int(relapse_counts[index]), int(ever_counts[index])),
                }
                for index in range(maps)
            ],
        },
    }


def _device_of(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def _rms(value: torch.Tensor) -> float:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Nonfinite state while computing W/Z RMS")
    flat_norm = torch.linalg.vector_norm(value.detach().to(torch.float64))
    return float((flat_norm / math.sqrt(value.numel())).item())


def _balanced_accuracy(logits: torch.Tensor, target: torch.Tensor, open_mask: torch.Tensor) -> dict[str, Any]:
    prediction = logits[:, 0] >= 0
    truth = target[:, 0] >= 0.5
    opened = open_mask[:, 0].bool()
    good = prediction == truth
    per_map: list[float | None] = []
    positive_pixels: list[int] = []
    positive_correct: list[int] = []
    negative_pixels: list[int] = []
    negative_correct: list[int] = []
    for index in range(target.shape[0]):
        positive = opened[index] & truth[index]
        negative = opened[index] & (~truth[index])
        npix = int(positive.sum().item())
        nnix = int(negative.sum().item())
        ncorrect = int((negative & good[index]).sum().item())
        pcorrect = int((positive & good[index]).sum().item())
        positive_pixels.append(npix)
        positive_correct.append(pcorrect)
        negative_pixels.append(nnix)
        negative_correct.append(ncorrect)
        recalls = []
        if npix:
            recalls.append(pcorrect / npix)
        if nnix:
            recalls.append(ncorrect / nnix)
        per_map.append(float(np.mean(recalls)) if recalls else None)
    eligible = [value for value in per_map if value is not None]
    return {
        "per_map": per_map,
        "eligible_maps": len(eligible),
        "mean_across_maps": float(np.mean(eligible)) if eligible else None,
        "per_map_positive_pixels": positive_pixels,
        "per_map_positive_correct": positive_correct,
        "per_map_negative_pixels": negative_pixels,
        "per_map_negative_correct": negative_correct,
    }


def _paired_band(correct: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    correct_counts = (correct & selected).reshape(selected.shape[0], -1).sum(axis=1, dtype=np.int64)
    pixel_counts = selected.reshape(selected.shape[0], -1).sum(axis=1, dtype=np.int64)
    per_map = [
        None if int(pixels) == 0 else float(int(hits) / int(pixels))
        for hits, pixels in zip(correct_counts, pixel_counts)
    ]
    eligible = [value for value in per_map if value is not None]
    pooled_correct = int(correct_counts.sum())
    pooled_pixels = int(pixel_counts.sum())
    return {
        "per_map_correct": [int(value) for value in correct_counts],
        "per_map_pixels": [int(value) for value in pixel_counts],
        "per_map": per_map,
        "eligible_maps": len(eligible),
        "mean_across_maps": float(np.mean(eligible)) if eligible else None,
        "pooled_correct": pooled_correct,
        "pooled_pixels": pooled_pixels,
        "pooled_accuracy": _rate(pooled_correct, pooled_pixels),
    }


def _data_tensor(data: dict[str, Any], name: str, device: torch.device) -> torch.Tensor:
    if name not in data:
        raise ValueError(f"data_cpu is missing required field {name!r}")
    return torch.as_tensor(data[name], device=device)


@torch.no_grad()
def diagnose(
    model: torch.nn.Module,
    data_cpu: dict[str, Any],
    budget: Callable[[], Any] = lambda: None,
) -> dict[str, Any]:
    """Run a no-gradient, paired 128-step diagnostic and return JSON-ready data."""
    device = _device_of(model)
    x = _data_tensor(data_cpu, "x", device)
    x_flip = _data_tensor(data_cpu, "x_flip", device)
    y = _data_tensor(data_cpu, "y", device)
    y_flip = _data_tensor(data_cpu, "y_flip", device)
    changed = _data_tensor(data_cpu, "changed", device).bool()
    open_mask = _data_tensor(data_cpu, "mask", device).bool()
    distance = _data_tensor(data_cpu, "distance", device)

    if x.ndim != 4 or x.shape != x_flip.shape or x.shape[1] != 3:
        raise ValueError("x and x_flip must have matching shape [B,3,H,W]")
    expected = (x.shape[0], 1, x.shape[2], x.shape[3])
    for name, tensor in (("y", y), ("y_flip", y_flip), ("changed", changed),
                         ("mask", open_mask), ("distance", distance)):
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if bool((changed & (~open_mask)).any()):
        raise ValueError("changed component must be contained in the open mask")

    changed_cpu = changed[:, 0].detach().cpu().numpy()
    distance_cpu = distance[:, 0].detach().cpu().numpy()
    changed_count = int(changed.sum().item())
    if changed_count == 0:
        raise ValueError("changed component is empty across all maps")

    paired_trace = np.zeros((TRACE_STEPS + 1, x.shape[0], x.shape[2], x.shape[3]), dtype=bool)
    endpoint_worlds: dict[int, dict[str, torch.Tensor]] = {}
    state_rms: dict[str, Any] = {}
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        state_original = model.initial(x)
        state_flipped = model.initial(x_flip)
        for step in range(TRACE_STEPS + 1):
            budget()
            if step:
                state_original = model.step(state_original, x)
                state_flipped = model.step(state_flipped, x_flip)
            logits_original = model.logits(state_original)
            logits_flipped = model.logits(state_flipped)
            if not bool(torch.isfinite(logits_original).all() and torch.isfinite(logits_flipped).all()):
                raise FloatingPointError(f"Nonfinite logits at step {step}")
            good_original = ((logits_original >= 0) == (y >= 0.5))
            good_flipped = ((logits_flipped >= 0) == (y_flip >= 0.5))
            paired_trace[step] = (good_original & good_flipped)[:, 0].detach().cpu().numpy()

            if step in ENDPOINTS:
                endpoint_worlds[step] = {
                    "logits_original": logits_original.detach(),
                    "logits_flipped": logits_flipped.detach(),
                }
                w_original, z_original = state_original
                w_flipped, z_flipped = state_flipped
                state_rms[str(step)] = {
                    "original": {"W_rms": _rms(w_original), "Z_rms": _rms(z_original)},
                    "flipped": {"W_rms": _rms(w_flipped), "Z_rms": _rms(z_flipped)},
                }
    finally:
        for module, was_training in modes:
            module.training = was_training

    trace_summary = summarize_correct_traces(paired_trace, changed_cpu)
    strict_band = changed_cpu & (distance_cpu > 16) & (distance_cpu < 32)
    endpoint_metrics: dict[str, Any] = {}
    for step in ENDPOINTS:
        worlds = endpoint_worlds[step]
        original_logits = worlds["logits_original"]
        flipped_logits = worlds["logits_flipped"]
        original_open = _balanced_accuracy(original_logits, y, open_mask)
        flipped_open = _balanced_accuracy(flipped_logits, y_flip, open_mask)
        endpoint_metrics[str(step)] = {
            "paired_strict_16_32": _paired_band(paired_trace[step], strict_band),
            "whole_open_balanced_accuracy_original": original_open,
            "whole_open_balanced_accuracy_flipped": flipped_open,
        }

    return {
        **trace_summary,
        "paired_correctness": "both original and source-flipped predictions correct, gated by changed",
        "endpoint_metrics": endpoint_metrics,
        "state_rms": state_rms,
        "state_rms_note": "Finite descriptive W/Z RMS at t32,64,128; no latent fixed-point criterion is applied.",
        "num_maps": int(x.shape[0]),
        "map_shape": [int(x.shape[2]), int(x.shape[3])],
    }
