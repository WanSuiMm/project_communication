"""Pure NumPy zero-training interventions for W-first streaming-carry runs.

W is [2 * maps, 24, H, W]. Its batch order is all original maps followed by
their flipped counterparts; its channel order is four six-channel lanes in
the order N, E, S, W (as defined by new/streaming_carry/stream_cells.py).
All index metadata uses gather semantics: output[..., i] = input[..., source[i]].
"""

from __future__ import annotations

from typing import Any

import numpy as np


LANE_ORDER = ("N", "E", "S", "W")
LANES = 4
PAYLOAD_CHANNELS_PER_LANE = 6
CHANNELS = LANES * PAYLOAD_CHANNELS_PER_LANE

# Fixed, data-independent defaults. Permutations act within the corresponding
# coordinate system and never sum or interpolate values.
DEFAULT_CHANNEL_PERMUTATION = (1, 2, 3, 4, 5, 0)
DEFAULT_LANE_PERMUTATION = (1, 2, 3, 0)


def _state(W: np.ndarray) -> np.ndarray:
    arr = np.asarray(W)
    if arr.dtype != np.float32:
        raise TypeError(f"W must have dtype float32, got {arr.dtype}")
    if arr.ndim != 4 or arr.shape[1] != CHANNELS:
        raise ValueError("W must have shape [2*maps,24,H,W]")
    if arr.shape[0] == 0 or arr.shape[0] % 2:
        raise ValueError("W batch must contain original maps followed by flips")
    if arr.shape[2] == 0 or arr.shape[3] == 0:
        raise ValueError("W spatial dimensions must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError("W must contain only finite values")
    return arr


def _binary_mask(mask: np.ndarray, maps: int, height: int, width: int) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.shape != (maps, 1, height, width):
        raise ValueError(
            f"mask must have shape [{maps},1,{height},{width}], got {arr.shape}"
        )
    if arr.dtype.kind not in "bifu":
        raise TypeError("mask must be a boolean or numeric binary array")
    if arr.dtype.kind in "fu" and not np.isfinite(arr).all():
        raise ValueError("mask must contain only finite binary values")
    if not np.logical_or(arr == 0, arr == 1).all():
        raise ValueError("mask must contain only 0/1 values")
    return arr.astype(bool, copy=False)


def _strict_permutation(values: Any, size: int, name: str) -> tuple[int, ...]:
    raw = np.asarray(values)
    if raw.shape != (size,) or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer permutation of length {size}")
    perm = tuple(int(x) for x in raw.tolist())
    if sorted(perm) != list(range(size)):
        raise ValueError(f"{name} must be a bijection over 0..{size - 1}")
    if perm == tuple(range(size)):
        raise ValueError(f"{name} must be nonidentity")
    return perm


def _inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for destination, source in enumerate(permutation):
        inverse[source] = destination
    return tuple(inverse)


def norm_match(
    donor: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match per-map open-cell RMS, then scale each entire donor map.

    `mask` describes the unflipped maps and is duplicated in original-then-flip
    order. RMS is taken over all 24 channels and open spatial cells for each
    batch row. The resulting scalar is applied to every cell, including closed
    cells, as required by the intervention protocol.
    """
    donor_arr = _state(donor)
    reference_arr = _state(reference)
    if donor_arr.shape != reference_arr.shape:
        raise ValueError("donor and reference must have identical shapes")
    maps = donor_arr.shape[0] // 2
    open_mask = _binary_mask(mask, maps, *donor_arr.shape[2:])[:, 0]
    open_counts = open_mask.sum(axis=(1, 2), dtype=np.int64)
    if np.any(open_counts == 0):
        raise ValueError("each map must have at least one open cell")
    open_pair = np.concatenate((open_mask, open_mask), axis=0)[:, None, :, :]
    denom = CHANNELS * np.concatenate((open_counts, open_counts)).astype(np.float64)

    def open_rms(array: np.ndarray) -> np.ndarray:
        square_sum = np.square(array.astype(np.float64), dtype=np.float64)
        square_sum *= open_pair
        return np.sqrt(square_sum.sum(axis=(1, 2, 3), dtype=np.float64) / denom)

    donor_rms = open_rms(donor_arr)
    reference_rms = open_rms(reference_arr)
    if np.any(donor_rms == 0.0) or np.any(reference_rms == 0.0):
        raise ValueError("zero open-mask RMS cannot be norm-matched")
    alpha = reference_rms / donor_rms
    if not np.isfinite(alpha).all():
        raise FloatingPointError("non-finite norm-match scale")
    matched = (
        donor_arr.astype(np.float64) * alpha[:, None, None, None]
    ).astype(np.float32)
    if not np.isfinite(matched).all():
        raise FloatingPointError("norm-matched W overflowed float32")
    record = {
        "kind": "norm_match",
        "metric": "RMS over 24 channels and open cells per batch row",
        "batch_order": "original maps followed by flipped maps",
        "alpha_by_batch": alpha.copy(),
        "alpha_by_map_variant": alpha.reshape(2, maps).T.copy(),
        "donor_rms_by_batch": donor_rms.copy(),
        "reference_rms_by_batch": reference_rms.copy(),
        "open_cell_count_by_map": open_counts.copy(),
        "closed_cells_scaled": True,
    }
    return matched, record


def apply_channel_gather(W: np.ndarray, source_channel_indices: Any) -> np.ndarray:
    """Apply a full 24-channel gather map to every spatial vector."""
    arr = _state(W)
    raw = np.asarray(source_channel_indices)
    if raw.shape != (CHANNELS,) or raw.dtype.kind not in "iu":
        raise ValueError("source_channel_indices must be an integer vector of length 24")
    indices = tuple(int(x) for x in raw.tolist())
    if sorted(indices) != list(range(CHANNELS)):
        raise ValueError("source_channel_indices must be a bijection over 0..23")
    return arr[:, indices, :, :].copy()


def channel_permute(
    W: np.ndarray, permutation: Any = DEFAULT_CHANNEL_PERMUTATION
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one fixed six-payload-channel permutation in every lane."""
    arr = _state(W)
    payload_perm = _strict_permutation(permutation, PAYLOAD_CHANNELS_PER_LANE, "permutation")
    full = tuple(
        lane * PAYLOAD_CHANNELS_PER_LANE + payload
        for lane in range(LANES)
        for payload in payload_perm
    )
    inverse = _inverse_permutation(full)
    transformed = apply_channel_gather(arr, full)
    return transformed, {
        "kind": "channel_permute",
        "lane_order": LANE_ORDER,
        "payload_permutation": payload_perm,
        "inverse_payload_permutation": _inverse_permutation(payload_perm),
        "source_channel_indices": full,
        "inverse_source_channel_indices": inverse,
        "gather_convention": "output channel i reads input channel source_channel_indices[i]",
    }


def lane_permute(
    W: np.ndarray, permutation: Any = DEFAULT_LANE_PERMUTATION
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one fixed lane permutation, preserving each lane's six values."""
    arr = _state(W)
    lane_perm = _strict_permutation(permutation, LANES, "permutation")
    full = tuple(
        source_lane * PAYLOAD_CHANNELS_PER_LANE + payload
        for source_lane in lane_perm
        for payload in range(PAYLOAD_CHANNELS_PER_LANE)
    )
    inverse = _inverse_permutation(full)
    transformed = apply_channel_gather(arr, full)
    return transformed, {
        "kind": "lane_permute",
        "lane_order": LANE_ORDER,
        "lane_permutation": lane_perm,
        "inverse_lane_permutation": _inverse_permutation(lane_perm),
        "source_channel_indices": full,
        "inverse_source_channel_indices": inverse,
        "gather_convention": "output channel i reads input channel source_channel_indices[i]",
    }


def apply_spatial_gather(W: np.ndarray, source_indices: np.ndarray) -> np.ndarray:
    """Gather every channel vector using one permutation per original map.

    `source_indices` has shape [maps,H*W]. The same row is applied to the
    matching original and flipped map. Each row uses destination-to-source
    gather semantics; use its argsort as the exact inverse gather.
    """
    arr = _state(W)
    maps, height, width = arr.shape[0] // 2, arr.shape[2], arr.shape[3]
    indices = np.asarray(source_indices)
    cell_count = height * width
    if indices.shape != (maps, cell_count) or indices.dtype.kind not in "iu":
        raise ValueError(f"source_indices must be an integer array [{maps},{cell_count}]")
    target = np.arange(cell_count, dtype=np.int64)
    for row in indices:
        if not np.array_equal(np.sort(row), target):
            raise ValueError("each source_indices row must be a spatial bijection")
    out = arr.copy()
    flat = arr.reshape(2 * maps, CHANNELS, cell_count)
    flat_out = out.reshape(2 * maps, CHANNELS, cell_count)
    for map_index in range(maps):
        gather = indices[map_index]
        flat_out[map_index] = flat[map_index][:, gather]
        flat_out[map_index + maps] = flat[map_index + maps][:, gather]
    return out


def _changed_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of flattened cell indices."""
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if seen[start_y, start_x]:
            continue
        seen[start_y, start_x] = True
        stack = [(int(start_y), int(start_x))]
        cells: list[int] = []
        while stack:
            y, x = stack.pop()
            cells.append(y * width + x)
            if y > 0 and mask[y - 1, x] and not seen[y - 1, x]:
                seen[y - 1, x] = True
                stack.append((y - 1, x))
            if y + 1 < height and mask[y + 1, x] and not seen[y + 1, x]:
                seen[y + 1, x] = True
                stack.append((y + 1, x))
            if x > 0 and mask[y, x - 1] and not seen[y, x - 1]:
                seen[y, x - 1] = True
                stack.append((y, x - 1))
            if x + 1 < width and mask[y, x + 1] and not seen[y, x + 1]:
                seen[y, x + 1] = True
                stack.append((y, x + 1))
        components.append(np.asarray(cells, dtype=np.int64))
    return components


def spatial_scramble(
    W: np.ndarray, changed_mask: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Shuffle cell vectors only within each map's 4-connected changed region.

    A single deterministic map-specific gather is shared by the original and
    flipped map and across all 24 channels. Unchanged cells gather themselves.
    A nontrivial component is forced to move, avoiding a silent identity draw.
    """
    arr = _state(W)
    maps, height, width = arr.shape[0] // 2, arr.shape[2], arr.shape[3]
    changed = np.asarray(changed_mask)
    if changed.dtype != np.bool_:
        raise TypeError("changed_mask must have boolean dtype")
    if changed.shape != (maps, height, width):
        raise ValueError(f"changed_mask must have shape [{maps},{height},{width}]")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")

    cell_count = height * width
    identity = np.arange(cell_count, dtype=np.int64)
    source_indices = np.tile(identity, (maps, 1))
    rng = np.random.default_rng(int(seed))
    nontrivial_components = 0
    for map_index in range(maps):
        components = _changed_components(changed[map_index])
        for cells in components:
            if len(cells) < 2:
                continue
            shuffled = rng.permutation(cells)
            if np.array_equal(shuffled, cells):
                # A cyclic shift is a guaranteed nonidentity permutation within
                # the same connected component.
                shift = int(rng.integers(1, len(cells)))
                shuffled = np.roll(cells, shift)
            source_indices[map_index, cells] = shuffled
            nontrivial_components += 1
    if nontrivial_components == 0:
        raise ValueError("changed_mask has no multi-cell component to scramble")

    inverse_indices = np.argsort(source_indices, axis=1).astype(np.int64, copy=False)
    transformed = apply_spatial_gather(arr, source_indices)
    if not np.isfinite(transformed).all():
        raise FloatingPointError("spatial scramble produced non-finite values")
    return transformed, {
        "kind": "spatial_scramble",
        "seed": int(seed),
        "connectivity": 4,
        "source_indices": source_indices,
        "inverse_source_indices": inverse_indices.copy(),
        "gather_convention": "destination i reads source source_indices[map,i]",
        "unchanged_cells_are_identity": True,
        "permutation_shared_by_original_and_flip": True,
        "permutation_shared_by_all_24_channels": True,
        "nontrivial_component_count": nontrivial_components,
    }


def cue_swap(W: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Exchange the first and second batch halves; this operation is involutive."""
    arr = _state(W)
    maps = arr.shape[0] // 2
    return np.concatenate((arr[maps:], arr[:maps]), axis=0).copy(), {
        "kind": "cue_swap",
        "batch_halves_exchanged": True,
        "inverse_is_self": True,
    }
