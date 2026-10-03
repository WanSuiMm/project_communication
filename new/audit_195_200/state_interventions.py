"""Deterministic, role-matched state interventions for the 195/200 audit.

The pairing is built once from public task geometry and the paired diagnostic
at T=64.  The same :class:`SwapPlan` must be applied to both source worlds.
It defines a matched local intervention cohort; it does not establish a
universal semantic decomposition of W or Z.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np


Component = Literal["W", "Z", "both"]


@dataclass(frozen=True)
class SwapPlan:
    """A deterministic mapping between eligible cells, flattened over B,H,W."""

    target_flat: np.ndarray
    donor_flat: np.ndarray
    mask: np.ndarray
    counts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        targets = np.asarray(self.target_flat, dtype=np.int64).reshape(-1)
        donors = np.asarray(self.donor_flat, dtype=np.int64).reshape(-1)
        selected = np.asarray(self.mask, dtype=bool)
        if targets.shape != donors.shape:
            raise ValueError("target_flat and donor_flat must have identical lengths")
        if selected.ndim != 3:
            raise ValueError("plan.mask must have shape [B,H,W]")
        if int(selected.sum()) != len(targets):
            raise ValueError("plan.mask must select exactly the mapped targets")
        if len(targets):
            if np.any(targets < 0) or np.any(targets >= selected.size):
                raise ValueError("target_flat contains an out-of-range BHW index")
            if np.any(donors < 0) or np.any(donors >= selected.size):
                raise ValueError("donor_flat contains an out-of-range BHW index")
            if len(np.unique(targets)) != len(targets):
                raise ValueError("target_flat entries must be unique")
            if len(np.unique(donors)) != len(donors):
                raise ValueError("donor_flat entries must be unique")
            if np.any(targets == donors):
                raise ValueError("swap plans must not contain fixed points")
            if not np.array_equal(np.sort(targets), np.sort(donors)):
                raise ValueError("every donor must also be a target")
            mapping = dict(zip((int(v) for v in targets), (int(v) for v in donors)))
            if any(mapping.get(mapping[target]) != target for target in mapping):
                raise ValueError("swap plans must be reciprocal involutions")
            if not np.array_equal(np.sort(targets), np.flatnonzero(selected.reshape(-1))):
                raise ValueError("plan.mask must match target_flat")
        object.__setattr__(self, "target_flat", targets)
        object.__setattr__(self, "donor_flat", donors)
        object.__setattr__(self, "mask", selected)


_DISTANCE_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("1_16", 1, 16),
    ("17_31", 17, 31),
    ("32_63", 32, 63),
    ("64_inf", 64, None),
)
_MARGIN_BINS = ("0_025", "025_1", "1_inf")
_AGE_BINS = ("0", "1_7", "8_31", "32_inf")


def _numpy(value: Any, name: str) -> np.ndarray:
    """Convert a NumPy or Torch value to a detached host array."""
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive conversion path
        raise TypeError(f"{name} must be array-like") from exc


def _map_array(value: Any, name: str, *, shape: tuple[int, int, int] | None = None,
               dtype: Any | None = None) -> np.ndarray:
    arr = _numpy(value, name)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape [B,H,W] or [B,1,H,W]")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {shape}")
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _broadcast_map(value: Any, name: str, shape: tuple[int, int, int], *,
                   binary: bool = False) -> np.ndarray:
    arr = _numpy(value, name)
    batch, height, width = shape
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.shape == (batch,):
        arr = arr[:, None, None]
    elif arr.shape == (batch, 1, 1):
        pass
    elif arr.ndim == 3 and arr.shape == (batch, height, width):
        pass
    try:
        out = np.broadcast_to(arr, shape)
    except ValueError as exc:
        raise ValueError(f"{name} cannot broadcast to [B,H,W]={shape}") from exc
    if binary:
        return out.astype(bool, copy=False) if out.dtype == np.bool_ else (out >= 0.5)
    return out


def open_degree(mask: Any) -> np.ndarray:
    """Return the number of open four-neighbors at every [B,H,W] cell."""
    opened = _map_array(mask, "mask", dtype=bool)
    degree = np.zeros(opened.shape, dtype=np.uint8)
    degree[:, 1:, :] += opened[:, :-1, :]
    degree[:, :-1, :] += opened[:, 1:, :]
    degree[:, :, 1:] += opened[:, :, :-1]
    degree[:, :, :-1] += opened[:, :, 1:]
    return degree


def _four_neighbor_any(value: np.ndarray) -> np.ndarray:
    adjacent = np.zeros(value.shape, dtype=bool)
    adjacent[:, 1:, :] |= value[:, :-1, :]
    adjacent[:, :-1, :] |= value[:, 1:, :]
    adjacent[:, :, 1:] |= value[:, :, :-1]
    adjacent[:, :, :-1] |= value[:, :, 1:]
    return adjacent


def _distance_bin(distance: int) -> int:
    for index, (_, low, high) in enumerate(_DISTANCE_BINS):
        if distance >= low and (high is None or distance <= high):
            return index
    raise ValueError(f"eligible distance {distance} is outside the defined bins")


def _margin_bin(abs_margin: float) -> int:
    if abs_margin < 0.25:
        return 0
    if abs_margin < 1.0:
        return 1
    return 2


def _age_bin(age: int) -> int:
    if age == 0:
        return 0
    if age <= 7:
        return 1
    if age <= 31:
        return 2
    return 3


def _bank_arrays(bank: dict[str, Any]) -> tuple[np.ndarray, ...]:
    required = ("x", "y", "y_flip", "mask", "changed", "distance")
    missing = [key for key in required if key not in bank]
    if missing:
        raise KeyError(f"bank is missing required keys: {', '.join(missing)}")
    x = _numpy(bank["x"], "bank['x']")
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("bank['x'] must have shape [B,3,H,W]")
    shape = (int(x.shape[0]), int(x.shape[2]), int(x.shape[3]))
    opened = _map_array(bank["mask"], "bank['mask']", shape=shape, dtype=bool)
    changed = _map_array(bank["changed"], "bank['changed']", shape=shape, dtype=bool)
    distance = _map_array(bank["distance"], "bank['distance']", shape=shape)
    y = _broadcast_map(bank["y"], "bank['y']", shape, binary=True)
    y_flip = _broadcast_map(bank["y_flip"], "bank['y_flip']", shape, binary=True)
    return opened, changed, distance, y, y_flip


def build_swaps(bank: dict[str, Any], correct: Any, margin: Any, age: Any,
                role: Literal["solved", "frontier"], seed: int,
                donor_mode: Literal["within_map", "global"] = "within_map",
                max_pairs_per_map: int | None = 8) -> SwapPlan:
    """Build deterministic, disjoint reciprocal pairs within each stratum.

    ``solved`` selects paired-correct changed non-source cells. ``frontier``
    selects paired-incorrect cells adjacent to a paired-correct changed,
    open, non-source four-neighbor.  Both roles require the cell itself to be
    open, changed, and at positive BFS distance.  Donors come only from the
    same eligible role and exact matching stratum.
    """
    if role not in ("solved", "frontier"):
        raise ValueError("role must be 'solved' or 'frontier'")
    if donor_mode not in ("within_map", "global"):
        raise ValueError("donor_mode must be 'within_map' or 'global'")
    if int(seed) < 0:
        raise ValueError("seed must be nonnegative")
    if max_pairs_per_map is not None and (
        isinstance(max_pairs_per_map, bool)
        or not isinstance(max_pairs_per_map, (int, np.integer))
        or int(max_pairs_per_map) < 0
    ):
        raise ValueError("max_pairs_per_map must be a nonnegative integer or None")
    opened, changed, distance, y, y_flip = _bank_arrays(bank)
    shape = opened.shape
    correctness = _map_array(correct, "correct", shape=shape, dtype=bool)
    signed_margin = _map_array(margin, "margin", shape=shape, dtype=np.float64)
    current_age = _map_array(age, "age", shape=shape)
    if not np.all(np.isfinite(distance)) or not np.all(np.isfinite(signed_margin)):
        raise ValueError("distance and margin must be finite")
    if np.any(current_age < 0) or not np.all(np.isfinite(current_age)):
        raise ValueError("age must contain finite nonnegative values")
    if np.any(changed & (~opened | (distance < 0))):
        raise ValueError("changed cells must be open and have nonnegative distance")
    if np.any(changed & (distance != np.floor(distance))):
        raise ValueError("changed-cell distances must be integer BFS distances")
    if np.any(current_age != np.floor(current_age)):
        raise ValueError("age values must be integers")

    non_source = distance > 0
    eligible_domain = opened & changed & non_source
    if role == "solved":
        eligible = eligible_domain & correctness
    else:
        frontier_support = correctness & eligible_domain
        eligible = eligible_domain & ~correctness & _four_neighbor_any(frontier_support)

    degree = open_degree(opened)
    abs_margin = np.abs(signed_margin)
    if not np.all(np.isfinite(abs_margin)):
        raise ValueError("absolute margin must be finite")
    groups: dict[tuple[int, ...], list[int]] = {}
    bsz, height, width = shape
    for batch, row, col in np.argwhere(eligible):
        batch_i, row_i, col_i = int(batch), int(row), int(col)
        flat = batch_i * height * width + row_i * width + col_i
        key: tuple[int, ...] = (
            batch_i if donor_mode == "within_map" else -1,
            int(y[batch_i, row_i, col_i]),
            int(y_flip[batch_i, row_i, col_i]),
            int(degree[batch_i, row_i, col_i]),
            _distance_bin(int(distance[batch_i, row_i, col_i])),
            _margin_bin(float(abs_margin[batch_i, row_i, col_i])),
        )
        if role == "solved":
            key += (_age_bin(int(current_age[batch_i, row_i, col_i])),)
        groups.setdefault(key, []).append(flat)

    rng = np.random.default_rng(int(seed))
    candidate_pairs: list[tuple[int, int]] = []
    singleton_sites = 0
    singleton_strata = 0
    matched_sizes: list[int] = []
    for key in sorted(groups):
        cells = np.asarray(groups[key], dtype=np.int64)
        if cells.size < 2:
            singleton_sites += int(cells.size)
            singleton_strata += 1
            continue
        shuffled = rng.permutation(cells)
        for index in range(0, shuffled.size - 1, 2):
            candidate_pairs.append((int(shuffled[index]), int(shuffled[index + 1])))
        matched_sizes.append(int(cells.size))

    # Enforce a sparse per-map budget after matching.  For explicitly global
    # matching, a cross-map pair consumes one target slot on each map.
    pair_order = rng.permutation(len(candidate_pairs)) if candidate_pairs else np.empty(0, dtype=np.int64)
    selected_pairs: list[tuple[int, int]] = []
    selected_per_map = np.zeros(bsz, dtype=np.int64)
    for pair_index in pair_order:
        left, right = candidate_pairs[int(pair_index)]
        left_map = left // (height * width)
        right_map = right // (height * width)
        if max_pairs_per_map is not None:
            cap = int(max_pairs_per_map)
            if left_map == right_map:
                if selected_per_map[left_map] + 1 > cap:
                    continue
            elif selected_per_map[left_map] + 1 > cap or selected_per_map[right_map] + 1 > cap:
                continue
        selected_pairs.append((left, right))
        if left_map == right_map:
            selected_per_map[left_map] += 1
        else:
            selected_per_map[left_map] += 1
            selected_per_map[right_map] += 1

    targets: list[int] = []
    donors: list[int] = []
    for left, right in selected_pairs:
        targets.extend((left, right))
        donors.extend((right, left))
    target_array = np.asarray(targets, dtype=np.int64)
    donor_array = np.asarray(donors, dtype=np.int64)
    plan_mask = np.zeros(shape, dtype=bool)
    if target_array.size:
        plan_mask.reshape(-1)[target_array] = True
    eligible_count = int(eligible.sum())
    swapped_count = int(target_array.size)
    per_map = []
    for map_id in range(bsz):
        map_cells = eligible[map_id]
        map_targets = (target_array // (height * width)) == map_id
        per_map.append({
            "map_index": map_id,
            "eligible_cells": int(map_cells.sum()),
            "swap_targets": int(map_targets.sum()),
            "unmatched_cells": int(map_cells.sum() - map_targets.sum()),
            "pairs": int(selected_per_map[map_id]),
            "eligible": int(map_cells.sum()),
            "swapped": int(map_targets.sum()),
            "singleton_skipped": int(map_cells.sum() - map_targets.sum()),
        })
    counts = {
        "eligible_cells": eligible_count,
        "unmatched_cells": eligible_count - swapped_count,
        "swap_targets": swapped_count,
        "pairs": len(selected_pairs),
        "eligible": eligible_count,
        "swapped": swapped_count,
        "singleton_skipped": singleton_sites,
        "singleton_strata": singleton_strata,
        "matched_strata": sum(1 for size in matched_sizes if size >= 2),
        "candidate_pairs_before_cap": len(candidate_pairs),
        "stratum_sizes": matched_sizes,
        "per_map": per_map,
    }
    metadata = {
        "role": role,
        "seed": int(seed),
        "donor_mode": donor_mode,
        "max_pairs_per_map": None if max_pairs_per_map is None else int(max_pairs_per_map),
        "pairing": "randomized disjoint reciprocal pairs; each target is the other's donor",
        "matching": [
            "map" if donor_mode == "within_map" else "no_map_constraint",
            "class_y", "class_y_flip", "four_neighbor_open_degree",
            "distance_band", "absolute_margin_bin",
        ] + (["current_correct_run_age_bin"] if role == "solved" else []),
        "distance_bins": {name: f"{low}..{high if high is not None else 'inf'}"
                          for name, low, high in _DISTANCE_BINS},
        "absolute_margin_bins": ["[0,0.25)", "[0.25,1)", "[1,inf)"],
        "age_bins": ["0", "1..7", "8..31", "32..inf"],
        "teacher_or_diagnostic_role_mask_exposed_to_model": False,
        "claim_scope": "matched local intervention cohort; not universal context or semantic-subspace equivalence",
    }
    return SwapPlan(target_array, donor_array, plan_mask, counts, metadata)


def identity_plan(shape: tuple[int, int, int] | Any,
                  role: str = "sham", *, reason: str = "identity sham") -> SwapPlan:
    """Return an empty plan suitable for an unchanged sham intervention."""
    if isinstance(shape, tuple):
        if len(shape) != 3 or any(int(v) < 0 for v in shape):
            raise ValueError("shape must be a nonnegative (B,H,W) tuple")
        resolved = tuple(int(v) for v in shape)
    else:
        resolved = _map_array(shape, "shape mask").shape
    return SwapPlan(
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
        np.zeros(resolved, dtype=bool),
        counts={"eligible_cells": 0, "unmatched_cells": 0, "swap_targets": 0,
                "pairs": 0, "eligible": 0, "swapped": 0, "singleton_skipped": 0,
                "singleton_strata": 0, "matched_strata": 0, "stratum_sizes": []},
        metadata={"role": role, "identity": True, "reason": reason,
                  "teacher_or_diagnostic_role_mask_exposed_to_model": False},
    )


def _torch_module():
    try:
        import torch
        return torch
    except ImportError:  # pragma: no cover - this project normally uses Torch
        return None


def _component_names(component: Component) -> tuple[str, ...]:
    if component == "both":
        return ("W", "Z")
    if component in ("W", "Z"):
        return (component,)
    raise ValueError("component must be 'W', 'Z', or 'both'")


def _channels(channels: Any, count: int) -> np.ndarray:
    if channels is None:
        return np.arange(count, dtype=np.int64)
    if isinstance(channels, slice):
        selected = np.arange(count, dtype=np.int64)[channels]
    elif isinstance(channels, (int, np.integer)):
        selected = np.asarray([int(channels)], dtype=np.int64)
    else:
        try:
            selected = np.asarray(list(channels), dtype=np.int64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise TypeError("channels must be an integer, slice, or iterable of integers") from exc
    if selected.size == 0:
        raise ValueError("channels selection must not be empty")
    if np.any(selected < 0) or np.any(selected >= count):
        raise ValueError(f"channels must be in [0,{count})")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("channels selection must not contain duplicates")
    return selected


def _indices_for_plan(plan: SwapPlan, batch: int, height: int, width: int,
                      device: Any | None = None) -> tuple[Any, Any]:
    if plan.mask.shape != (batch, height, width):
        raise ValueError(f"plan shape {plan.mask.shape} does not match state {(batch, height, width)}")
    targets = plan.target_flat
    donors = plan.donor_flat
    torch = _torch_module()
    if device is not None and torch is not None:
        target_index = torch.as_tensor(targets, dtype=torch.long, device=device)
        donor_index = torch.as_tensor(donors, dtype=torch.long, device=device)
        return target_index, donor_index
    return targets, donors


def _copy_selected_vectors(base: Any, donor: Any, targets: Any, donors: Any,
                           channels: Any) -> Any:
    """Copy donor vectors into target positions without mutating either input."""
    torch = _torch_module()
    if torch is not None and isinstance(base, torch.Tensor):
        if not isinstance(donor, torch.Tensor):
            donor = torch.as_tensor(donor, device=base.device, dtype=base.dtype)
        if base.ndim != 4 or donor.shape != base.shape:
            raise ValueError("state components must be matching [B,C,H,W] tensors")
        out = base.clone()
        if len(targets) == 0:
            return out
        bsz, count, height, width = base.shape
        tb, ty, tx = targets // (height * width), (targets // width) % height, targets % width
        db, dy, dx = donors // (height * width), (donors // width) % height, donors % width
        channel_index = torch.as_tensor(channels, dtype=torch.long, device=base.device)
        out[tb[:, None], channel_index[None, :], ty[:, None], tx[:, None]] = \
            donor[db[:, None], channel_index[None, :], dy[:, None], dx[:, None]]
        return out
    base_array = np.asarray(base)
    donor_array = _numpy(donor, "donor state")
    if base_array.ndim != 4 or donor_array.shape != base_array.shape:
        raise ValueError("state components must be matching [B,C,H,W] arrays")
    out = base_array.copy()
    if len(targets) == 0:
        return out
    _, _, height, width = base_array.shape
    targets = np.asarray(targets, dtype=np.int64)
    donors = np.asarray(donors, dtype=np.int64)
    tb, ty, tx = targets // (height * width), (targets // width) % height, targets % width
    db, dy, dx = donors // (height * width), (donors // width) % height, donors % width
    channel_index = np.asarray(channels, dtype=np.int64)
    out[tb[:, None], channel_index[None, :], ty[:, None], tx[:, None]] = \
        donor_array[db[:, None], channel_index[None, :], dy[:, None], dx[:, None]]
    return out


def apply_swap(state: tuple[Any, Any], plan: SwapPlan,
               component: Component = "both", channels: Any = None) -> tuple[Any, Any]:
    """Clone ``(W,Z)`` and replace selected target vectors from plan donors.

    ``channels`` indexes the selected component channels.  For ``both`` it is
    applied to W and Z independently; use ``None`` to copy all channels.
    """
    if not isinstance(state, (tuple, list)) or len(state) != 2:
        raise ValueError("state must be a (W,Z) pair")
    source_w, source_z = state
    outputs: dict[str, Any] = {"W": _numpy(source_w, "W").copy()
                               if not hasattr(source_w, "clone") else source_w.clone(),
                               "Z": _numpy(source_z, "Z").copy()
                               if not hasattr(source_z, "clone") else source_z.clone()}
    for name in _component_names(component):
        value = source_w if name == "W" else source_z
        shape = tuple(value.shape)
        if len(shape) != 4:
            raise ValueError(f"{name} must have shape [B,C,H,W]")
        batch, count, height, width = (int(v) for v in shape)
        target_idx, donor_idx = _indices_for_plan(plan, batch, height, width,
                                                 getattr(value, "device", None))
        selection = _channels(channels, count)
        outputs[name] = _copy_selected_vectors(value, value, target_idx, donor_idx, selection)
    return outputs["W"], outputs["Z"]


def _mask_array(mask: Any, shape: tuple[int, int, int]) -> np.ndarray:
    return _map_array(mask, "mask", shape=shape, dtype=bool)


def transplant_between(base_state: tuple[Any, Any], donor_state: tuple[Any, Any],
                       mask: Any, component: Component = "both",
                       channels: Any = None) -> tuple[Any, Any]:
    """Copy donor 195/200 state vectors at the same selected map locations."""
    if not isinstance(base_state, (tuple, list)) or len(base_state) != 2:
        raise ValueError("base_state must be a (W,Z) pair")
    if not isinstance(donor_state, (tuple, list)) or len(donor_state) != 2:
        raise ValueError("donor_state must be a (W,Z) pair")
    base_w, base_z = base_state
    donor_w, donor_z = donor_state
    shape = tuple(int(v) for v in base_w.shape)
    if len(shape) != 4 or tuple(base_z.shape)[0] != shape[0] or tuple(base_z.shape)[2:] != shape[2:]:
        raise ValueError("base W and Z must share [B,H,W] dimensions")
    spatial_shape = (shape[0], shape[2], shape[3])
    selected = _mask_array(mask, spatial_shape)
    flat = np.flatnonzero(selected.reshape(-1)).astype(np.int64)
    torch = _torch_module()
    if torch is not None and isinstance(base_w, torch.Tensor):
        target_idx = torch.as_tensor(flat, dtype=torch.long, device=base_w.device)
    else:
        target_idx = flat
    outputs: dict[str, Any] = {"W": base_w.clone() if hasattr(base_w, "clone") else np.asarray(base_w).copy(),
                               "Z": base_z.clone() if hasattr(base_z, "clone") else np.asarray(base_z).copy()}
    for name in _component_names(component):
        base = base_w if name == "W" else base_z
        donor = donor_w if name == "W" else donor_z
        if tuple(donor.shape) != tuple(base.shape):
            raise ValueError(f"donor {name} shape must match base {name}")
        _, count, height, width = (int(v) for v in base.shape)
        donor_indices = target_idx
        selection = _channels(channels, count)
        outputs[name] = _copy_selected_vectors(base, donor, target_idx, donor_indices, selection)
    return outputs["W"], outputs["Z"]


def apply_z_projection_swap(state: tuple[Any, Any], plan: SwapPlan,
                            readout_weight: Any,
                            subspace: Literal["null", "span"] = "null") -> tuple[Any, Any]:
    """Swap the null- or span-readout projection of donor-minus-target Z.

    The readout direction is the normalized flattened linear readout weight.
    A null intervention adds ``(I-P_R)(Z_donor-Z_target)``; a span
    intervention adds ``P_R(Z_donor-Z_target)``.  This is a coordinate defined
    by the supplied readout, not a claim about universal semantic subspaces.
    """
    if subspace not in ("null", "span"):
        raise ValueError("subspace must be 'null' or 'span'")
    if not isinstance(state, (tuple, list)) or len(state) != 2:
        raise ValueError("state must be a (W,Z) pair")
    w, z = state
    if len(tuple(z.shape)) != 4:
        raise ValueError("Z must have shape [B,C,H,W]")
    batch, channels, height, width = (int(v) for v in z.shape)
    target_idx, donor_idx = _indices_for_plan(plan, batch, height, width,
                                             getattr(z, "device", None))
    w_out = w.clone() if hasattr(w, "clone") else np.asarray(w).copy()
    z_out = z.clone() if hasattr(z, "clone") else np.asarray(z).copy()
    if len(target_idx) == 0:
        return w_out, z_out
    torch = _torch_module()
    is_torch = torch is not None and isinstance(z, torch.Tensor)
    if is_torch:
        vector = torch.as_tensor(readout_weight, device=z.device, dtype=z.dtype).reshape(-1)
        if vector.numel() != channels:
            raise ValueError(f"readout weight must have {channels} entries")
        norm = torch.linalg.vector_norm(vector)
        if not bool(torch.isfinite(norm)) or float(norm) <= 0:
            raise ValueError("readout weight must be finite and nonzero")
        direction = vector / norm
        channel_idx = torch.arange(channels, dtype=torch.long, device=z.device)
    else:
        vector = _numpy(readout_weight, "readout_weight").astype(np.asarray(z).dtype, copy=False).reshape(-1)
        if vector.size != channels:
            raise ValueError(f"readout weight must have {channels} entries")
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("readout weight must be finite and nonzero")
        direction = vector / norm
        channel_idx = np.arange(channels, dtype=np.int64)

    h_w = height * width
    if is_torch:
        tb, ty, tx = target_idx // h_w, (target_idx // width) % height, target_idx % width
        db, dy, dx = donor_idx // h_w, (donor_idx // width) % height, donor_idx % width
        receiver = z[tb[:, None], channel_idx[None, :], ty[:, None], tx[:, None]]
        donor_values = z[db[:, None], channel_idx[None, :], dy[:, None], dx[:, None]]
        delta = donor_values - receiver
        along_readout = torch.sum(delta * direction[None, :], dim=1, keepdim=True)
        projected = along_readout * direction[None, :]
        update = projected if subspace == "span" else delta - projected
        z_out[tb[:, None], channel_idx[None, :], ty[:, None], tx[:, None]] = receiver + update
    else:
        target_idx = np.asarray(target_idx, dtype=np.int64)
        donor_idx = np.asarray(donor_idx, dtype=np.int64)
        tb, ty, tx = target_idx // h_w, (target_idx // width) % height, target_idx % width
        db, dy, dx = donor_idx // h_w, (donor_idx // width) % height, donor_idx % width
        receiver = np.asarray(z)[tb[:, None], channel_idx[None, :], ty[:, None], tx[:, None]]
        donor_values = np.asarray(z)[db[:, None], channel_idx[None, :], dy[:, None], dx[:, None]]
        delta = donor_values - receiver
        along_readout = np.sum(delta * direction[None, :], axis=1, keepdims=True)
        projected = along_readout * direction[None, :]
        update = projected if subspace == "span" else delta - projected
        z_out[tb[:, None], channel_idx[None, :], ty[:, None], tx[:, None]] = receiver + update
    return w_out, z_out


def pair_margin(logits: Any, logits_flip: Any, y: Any, y_flip: Any) -> np.ndarray:
    """Return the minimum label-signed margin across the paired worlds."""
    logit_a = _numpy(logits, "logits")
    if logit_a.ndim == 4 and logit_a.shape[1] == 1:
        logit_a = logit_a[:, 0]
    if logit_a.ndim != 3:
        raise ValueError("logits must have shape [B,H,W] or [B,1,H,W]")
    shape = tuple(int(v) for v in logit_a.shape)
    logit_b = _map_array(logits_flip, "logits_flip", shape=shape)
    label_a = _broadcast_map(y, "y", shape, binary=True)
    label_b = _broadcast_map(y_flip, "y_flip", shape, binary=True)
    margin_a = np.where(label_a, logit_a, -logit_a)
    margin_b = np.where(label_b, logit_b, -logit_b)
    return np.minimum(margin_a, margin_b)


def paired_correct(logits: Any, logits_flip: Any, y: Any, y_flip: Any) -> np.ndarray:
    """Return cellwise correctness in both original and flipped source worlds."""
    logit_a = _numpy(logits, "logits")
    if logit_a.ndim == 4 and logit_a.shape[1] == 1:
        logit_a = logit_a[:, 0]
    if logit_a.ndim != 3:
        raise ValueError("logits must have shape [B,H,W] or [B,1,H,W]")
    shape = tuple(int(v) for v in logit_a.shape)
    logit_b = _map_array(logits_flip, "logits_flip", shape=shape)
    label_a = _broadcast_map(y, "y", shape, binary=True)
    label_b = _broadcast_map(y_flip, "y_flip", shape, binary=True)
    return ((logit_a >= 0) == label_a) & ((logit_b >= 0) == label_b)

