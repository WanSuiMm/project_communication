"""Seeded input banks and schedules for the spatial nonlinear-lift task.

Inputs have shape ``[samples, steps, side, side, 3]``.  Each sample has its
own latent cues, static spatial fields, and fresh per-step noise.  The first
half of the requested horizon is one source-cue episode; the remaining steps
form one query-cue episode.

This module deliberately contains no teacher or student model.  In
particular, the teacher constants below are for label generation only and
must not be supplied as student inputs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor


INPUT_CHANNELS = 3

# Fixed hidden teacher parameters.  These are metadata for generating labels,
# not part of the observed student input.
TEACHER_PARAMETERS: dict[str, tuple[float, ...]] = {
    "a": (0.8, -0.6, 0.2),
    "b": (0.1, 0.15, 0.9),
    "gamma": (0.05, 0.4, 0.8),
}


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _seed_value(seed: int) -> int:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed must be an integer")
    return int(seed)


def make_bank(samples: int, side: int, steps: int, seed: int) -> dict[str, Any]:
    """Create a deterministic bank of independent, two-phase inputs.

    During the first ``steps // 2`` steps, channels 0 and 1 carry one source
    cue episode formed from a per-map Gaussian latent, a static per-map spatial
    Gaussian field, and fresh time noise.  Channel 2 contains only fresh
    noise.  The remaining steps form one query cue episode: channel 2 carries
    an independent query with the same latent/static/noise decomposition,
    while channels 0 and 1 contain fresh low-amplitude noise.  Thus all
    channels remain active in both phases, while the query does not reveal the
    earlier source.

    The returned bank has no targets.  Its floating-point arrays are float32;
    ``sample_id`` is int64 and ``metadata`` records the construction settings.
    The latent and field arrays are included for diagnostics and should not be
    passed to a student model.
    """
    n = _positive_int("samples", samples)
    width = _positive_int("side", side)
    length = _positive_int("steps", steps)
    actual_seed = _seed_value(seed)
    rng = np.random.default_rng(actual_seed)

    # Independent per-map latent variables and static spatial Gaussian fields.
    source_latent = rng.standard_normal((n, 2), dtype=np.float32)
    query_latent = rng.standard_normal((n,), dtype=np.float32)
    source_field = rng.standard_normal((n, width, width, 2), dtype=np.float32)
    query_field = rng.standard_normal((n, width, width), dtype=np.float32)

    e = np.empty((n, length, width, width, INPUT_CHANNELS), dtype=np.float32)
    phase_length = length // 2
    for t in range(length):
        step_noise = rng.standard_normal(
            (n, width, width, INPUT_CHANNELS), dtype=np.float32
        )
        if t < phase_length:
            e[:, t, :, :, :2] = (
                0.7 * source_latent[:, None, None, :]
                + 0.3 * source_field
                + 0.2 * step_noise[:, :, :, :2]
            )
            e[:, t, :, :, 2] = 0.15 * step_noise[:, :, :, 2]
        else:
            e[:, t, :, :, :2] = 0.15 * step_noise[:, :, :, :2]
            e[:, t, :, :, 2] = (
                0.7 * query_latent[:, None, None]
                + 0.3 * query_field
                + 0.2 * step_noise[:, :, :, 2]
            )

    return {
        "e": e,
        "source_latent": source_latent,
        "query_latent": query_latent,
        "source_field": source_field,
        "query_field": query_field,
        "sample_id": np.arange(n, dtype=np.int64),
        "metadata": {
            "seed": actual_seed,
            "samples": n,
            "side": width,
            "steps": length,
            "phase_length": phase_length,
            "cycle_length": length,
            "periodic_spatial_boundary": True,
        },
    }


def make_schedule(
    updates: int, batch_size: int, samples: int, seed: int
) -> np.ndarray:
    """Draw one without-replacement minibatch per update from a fixed bank."""
    n_updates = _positive_int("updates", updates)
    batch = _positive_int("batch_size", batch_size)
    n_samples = _positive_int("samples", samples)
    actual_seed = _seed_value(seed)
    if batch > n_samples:
        raise ValueError("batch_size cannot exceed samples for no-replacement draws")

    rng = np.random.default_rng(actual_seed)
    schedule = np.empty((n_updates, batch), dtype=np.int64)
    for update in range(n_updates):
        schedule[update] = rng.choice(n_samples, size=batch, replace=False)
    return schedule


def bank_to_torch(bank: Mapping[str, Any], device: str | torch.device) -> dict[str, Any]:
    """Move floating NumPy arrays in a bank to ``device`` and keep metadata.

    Integer arrays and non-array metadata are preserved as supplied.  This is
    useful for adding GPU-generated target arrays to a bank before conversion.
    """
    converted: dict[str, Any] = {}
    for key, value in bank.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            converted[key] = torch.as_tensor(value, device=device)
        else:
            converted[key] = value
    return converted


def batch_from_indices(
    bank: Mapping[str, Any], indices: Sequence[int] | np.ndarray | Tensor
) -> dict[str, Any]:
    """Select a minibatch from every sample-aligned array/tensor in ``bank``.

    The returned mapping also contains ``indices`` as an int64 NumPy array,
    allowing a runner to record the exact schedule row used for the batch.
    Nested metadata and values without the bank's sample dimension are kept
    intact.
    """
    if "e" not in bank:
        raise KeyError("bank must contain the input array 'e'")
    sample_count = int(bank["e"].shape[0])
    if isinstance(indices, Tensor):
        raw_indices = indices.detach().to(device="cpu").numpy()
    else:
        raw_indices = np.asarray(indices)
    if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
        raise TypeError("indices must be a one-dimensional integer sequence")
    selected = raw_indices.astype(np.int64, copy=False)
    if np.any(selected < 0) or np.any(selected >= sample_count):
        raise IndexError("indices contain a value outside the bank")

    result: dict[str, Any] = {}
    for key, value in bank.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == sample_count:
            result[key] = value[selected]
        elif isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == sample_count:
            tensor_indices = torch.as_tensor(selected, dtype=torch.long, device=value.device)
            result[key] = value.index_select(0, tensor_indices)
        else:
            result[key] = value
    result["indices"] = selected.copy()
    return result


__all__ = [
    "INPUT_CHANNELS",
    "TEACHER_PARAMETERS",
    "make_bank",
    "make_schedule",
    "bank_to_torch",
    "batch_from_indices",
]
