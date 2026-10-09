"""Deterministic procedural target and seed for the AU-NCA screen."""
from __future__ import annotations

import torch
from torch import Tensor


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def make_target(size: int = 32) -> torch.FloatTensor:
    """Return one RGBA flower target with shape [1, 4, size, size].

    The background is transparent. Petals, a yellow center, a curved green
    stem, and one asymmetric leaf are solid, so RGB is already premultiplied
    by alpha. Every non-background pixel lies within Chebyshev radius 9 of
    the seed cell when size is at least 19.
    """
    side = _positive_int("size", size)
    if side < 19:
        raise ValueError("size must be at least 19 to contain the radius-9 target")

    center = side // 2
    y, x = torch.meshgrid(
        torch.arange(side, dtype=torch.float32),
        torch.arange(side, dtype=torch.float32),
        indexing="ij",
    )
    dx = x - float(center)
    dy = y - float(center)

    rgba = torch.zeros((1, 4, side, side), dtype=torch.float32)
    occupied = torch.zeros((side, side), dtype=torch.bool)

    petals = (
        (0.0, -4.0, (0.92, 0.16, 0.30)),
        (4.0, 0.0, (0.96, 0.39, 0.08)),
        (0.0, 4.0, (0.70, 0.16, 0.83)),
        (-4.0, 0.0, (0.16, 0.48, 0.92)),
    )
    for offset_x, offset_y, color in petals:
        petal = (dx - offset_x).square() + (dy - offset_y).square() <= 3.2**2
        rgba[0, :3, petal] = torch.tensor(color, dtype=torch.float32)[:, None]
        occupied |= petal

    # A narrow, slightly curved stem reaches the radius-9 boundary.
    stem = (dy >= 4.0) & (dy <= 9.0) & ((dx - 0.12 * (dy - 4.0)).abs() <= 0.55)
    rgba[0, :3, stem] = torch.tensor((0.10, 0.58, 0.18), dtype=torch.float32)[:, None]
    occupied |= stem

    # The leaf is tilted and offset to the right, making the silhouette
    # asymmetric while keeping its farthest pixel inside radius 9.
    leaf_dx = dx - 2.0
    leaf_dy = dy - 6.5
    leaf_u = leaf_dx - 0.35 * leaf_dy
    leaf_v = leaf_dy + 0.20 * leaf_dx
    leaf = (leaf_u / 2.65).square() + (leaf_v / 1.45).square() <= 1.0
    rgba[0, :3, leaf] = torch.tensor((0.16, 0.72, 0.24), dtype=torch.float32)[:, None]
    occupied |= leaf

    center_disk = dx.square() + dy.square() <= 2.0**2
    rgba[0, :3, center_disk] = torch.tensor((1.0, 0.76, 0.08), dtype=torch.float32)[:, None]
    occupied |= center_disk

    rgba[0, 3, occupied] = 1.0
    return rgba


def make_seed(
    batch: int,
    size: int = 32,
    channels: int = 16,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create one-pixel living seeds with zero RGB and one-valued hidden channels.

    Alpha and hidden channels 4 onward are one at the center cell. Every
    other value, including RGB and all off-center state, is zero.
    """
    count = _positive_int("batch", batch)
    side = _positive_int("size", size)
    width = _positive_int("channels", channels)
    if width < 4:
        raise ValueError("channels must include the four RGBA channels")
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")

    state = torch.zeros((count, width, side, side), device=device, dtype=dtype)
    center = side // 2
    state[:, 3, center, center] = 1
    if width > 4:
        state[:, 4:, center, center] = 1
    return state


__all__ = ["make_target", "make_seed"]
