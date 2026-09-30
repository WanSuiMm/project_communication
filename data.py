"""Deterministic synthetic batches for narrow reaction-transport gates.

Spatial tensors are ordered by their ordinary dimension indices.  The main
axis is shared by a batch and is seed-selected unless ``axis`` is supplied.
2D gates use a 14-cell transverse slab; 3D gates use a narrow 4x4 transverse
cross-section with four straight corner corridors.  ``x`` has five float32
channels: signed source values, source markers, target markers, visible-region
mask, and independent local detail bits.  Oracle edges encode undirected
nearest-neighbor conductances once in the positive-axis slot; terminal slots
are zero.  Position/path metadata is for checks only and is not model input.
"""

from __future__ import annotations

import math

import torch


def make_batch(
    dim: int,
    gate: str,
    batch_size: int,
    distance: int,
    seed: int,
    device: str | torch.device = "cpu",
    long_size: int | None = None,
    axis: int | None = None,
) -> dict:
    """Build one deterministic batch for gate A, B, or C.

    Gate A has one source and a fully visible slab.  Gates B and C have four
    disjoint thin corridors with one independent source bit per corridor; the
    target is placed on one randomly selected corridor.  Gate C additionally
    includes the target's independent detail bit in its four-way label.
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    gate = gate.upper()
    if gate not in ("A", "B", "C"):
        raise ValueError("gate must be A, B, or C")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if distance < 1:
        raise ValueError("distance must be positive so source and target are distinct")

    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    if axis is None:
        axis = int(torch.randint(dim, (), generator=rng).item())
    elif not isinstance(axis, int) or not 0 <= axis < dim:
        raise ValueError(f"axis must be None or an integer in [0, {dim})")

    transverse_sizes = [14] if dim == 2 else [4, 4]
    min_length = distance + 8
    main_length = max(min_length, distance + 12 if long_size is None else int(long_size))
    spatial = []
    cross_iter = iter(transverse_sizes)
    for k in range(dim):
        spatial.append(main_length if k == axis else next(cross_iter))
    transverse_axes = [k for k in range(dim) if k != axis]

    # Four well-separated lanes in 2D; four corners in the narrow 3D cross-section.
    if dim == 2:
        lanes = [(1,), (4,), (9,), (12,)]
    else:
        lanes = [(0, 0), (0, 3), (3, 0), (3, 3)]
    n_paths = 1 if gate == "A" else 4

    # Leave two cells before and after the source-target interval when possible.
    starts = torch.randint(2, main_length - distance - 2, (batch_size,), generator=rng)
    source_positions = torch.zeros((batch_size, n_paths, dim), dtype=torch.long)
    source_positions[:, :, axis] = starts[:, None]
    if gate == "A":
        random_cross = [
            torch.randint(size, (batch_size,), generator=rng)
            for size in transverse_sizes
        ]
        for j, k in enumerate(transverse_axes):
            source_positions[:, 0, k] = random_cross[j]
    else:
        lane_tensor = torch.tensor(lanes, dtype=torch.long)
        for j, k in enumerate(transverse_axes):
            source_positions[:, :, k] = lane_tensor[None, :, j]

    batch_rows = torch.arange(batch_size, dtype=torch.long)
    if gate == "A":
        path_ids = torch.zeros((batch_size,), dtype=torch.long)
    else:
        path_ids = torch.randint(4, (batch_size,), generator=rng)
    source_bits = torch.randint(2, (batch_size, n_paths), generator=rng)
    target_positions = source_positions[batch_rows, path_ids].clone()
    target_positions[:, axis] += distance

    # Region IDs are internal geometry only.  The model sees a binary mask,
    # while IDs keep oracle conductances from joining different corridors.
    region_ids = torch.full(tuple(spatial), -1, dtype=torch.long)
    if gate == "A":
        region_ids.fill_(0)
    else:
        for path_id, lane in enumerate(lanes):
            location = tuple(
                slice(None) if k == axis else lane[transverse_axes.index(k)]
                for k in range(dim)
            )
            region_ids[location] = path_id

    x = torch.zeros((batch_size, 5, *spatial), dtype=torch.float32)
    x[:, 3] = (region_ids >= 0).to(torch.float32).unsqueeze(0).expand(batch_size, *spatial)
    local_bits = torch.randint(2, (batch_size, *spatial), generator=rng)
    x[:, 4] = (2 * local_bits - 1).to(torch.float32)

    strides = [math.prod(spatial[k + 1 :]) for k in range(dim)]

    def flatten_positions(points: torch.Tensor) -> torch.Tensor:
        return sum(points[..., k] * strides[k] for k in range(dim))

    source_flat = flatten_positions(source_positions)
    target_flat = flatten_positions(target_positions)
    x_flat = x.flatten(start_dim=2)
    source_values = 2 * source_bits - 1
    x_flat[batch_rows[:, None], 0, source_flat] = source_values.to(torch.float32)
    x_flat[batch_rows[:, None], 1, source_flat] = 1.0
    x_flat[batch_rows, 2, target_flat] = 1.0

    detail_labels = local_bits.reshape(batch_size, -1)[batch_rows, target_flat]
    source_labels = source_bits[batch_rows, path_ids]
    labels = 2 * source_labels + detail_labels if gate == "C" else source_labels.clone()

    oracle_edges = []
    for k, size in enumerate(spatial):
        edge = torch.zeros((1, 1, *spatial), dtype=torch.float32)
        low = [slice(None)] * dim
        high = [slice(None)] * dim
        low[k] = slice(0, size - 1)
        high[k] = slice(1, size)
        low_ids = region_ids[tuple(low)]
        high_ids = region_ids[tuple(high)]
        edge[(0, 0, *low)] = ((low_ids >= 0) & (low_ids == high_ids)).to(torch.float32)
        oracle_edges.append(edge.expand(batch_size, 1, *spatial).clone())

    target_device = torch.device(device)
    return {
        "x": x.to(target_device),
        "target_index": target_flat.to(target_device),
        "labels": labels.to(target_device, dtype=torch.long),
        "source_labels": source_labels.to(target_device, dtype=torch.long),
        "detail_labels": detail_labels.to(target_device, dtype=torch.long),
        "oracle_edges": [edge.to(target_device) for edge in oracle_edges],
        "source_positions": source_positions.to(target_device),
        "target_positions": target_positions.to(target_device),
        "path_ids": path_ids.to(target_device),
    }
