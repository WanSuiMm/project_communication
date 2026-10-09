"""Exactly equivalent original and lifted spatial polynomial cells.

The ``original`` arm carries its two scalar fields directly.  The ``lifted``
arm carries unprojected moments and applies the same trainable vectors only in
``project``.  Both arms implement the same cyclic 2D recurrence for a fixed
feature sequence ``e[B,T,H,W,3]``; the lifted carrier uses 42 scalars per cell
instead of two.  A single Python integer records the number of completed
steps, so the counter is not replicated over the spatial grid.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


INPUT_CHANNELS = 3
ORIGINAL_STATE_CHANNELS = 2
LIFTED_STATE_CHANNELS = 3 + 3 + 3 * 3 + 3 * 3 * 3


class OriginalState(NamedTuple):
    """Direct scalar state on a batch of cyclic grids and a shared step count."""

    u: Tensor  # [B,H,W]
    z: Tensor  # [B,H,W]
    count: int  # completed updates, shared by every cell


class LiftedState(NamedTuple):
    """Raw polynomial moments and a shared step count for late projection."""

    U: Tensor  # [B,H,W,3]
    M0: Tensor  # [B,H,W,3]
    M1: Tensor  # [B,H,W,3,3]
    M2: Tensor  # [B,H,W,3,3,3]
    count: int  # completed updates, shared by every cell


CellState = OriginalState | LiftedState


def _check_features(e: Tensor) -> None:
    if e.ndim != 4 or e.shape[-1] != INPUT_CHANNELS:
        raise ValueError(f"e must have shape [B,H,W,{INPUT_CHANNELS}]")
    if any(size <= 0 for size in e.shape[:3]):
        raise ValueError("batch and spatial dimensions must be non-empty")


def _check_count(count: int) -> None:
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise TypeError("state.count must be a non-negative Python integer")


def _p(value: Tensor) -> Tensor:
    """Roll one row forward on axis H, leaving batch and channels untouched."""
    return torch.roll(value, shifts=1, dims=1)


def _q(value: Tensor) -> Tensor:
    """Roll one column backward on axis W, leaving batch and channels untouched."""
    return torch.roll(value, shifts=-1, dims=2)


class SpatialPolynomialCell(nn.Module):
    """A cyclic 2D state-nonlinear degree-two cell with two exact forms.

    ``e`` is a fixed external feature sequence with individual slices shaped
    ``[B,H,W,3]``.  The trainable ``a``, ``b`` and ``gamma`` vectors are
    initialized identically for both ``kind`` values under a given seed.
    ``original`` stores ``u`` and ``z`` directly; ``lifted`` stores raw
    moments and contracts them with those same vectors in ``project``.  No
    separate decoder or per-cell step counter is used.
    """

    def __init__(
        self,
        kind: str,
        seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if kind not in ("original", "lifted"):
            raise ValueError(f"Unknown spatial-cell kind: {kind!r}")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch dtype")
        self.kind = kind
        self.seed = int(seed)

        # A forked RNG stream makes construction deterministic without
        # advancing the caller's random stream.  Parameter creation order and
        # initialization are shared by both implementations.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.a = nn.Parameter(torch.empty(INPUT_CHANNELS, dtype=dtype))
            self.b = nn.Parameter(torch.empty(INPUT_CHANNELS, dtype=dtype))
            self.gamma = nn.Parameter(torch.empty(INPUT_CHANNELS, dtype=dtype))
            nn.init.normal_(self.a, mean=0.0, std=0.4)
            nn.init.normal_(self.b, mean=0.0, std=0.4)
            with torch.no_grad():
                self.gamma.copy_(
                    torch.tensor([0.05, 0.3, 0.2], dtype=dtype)
                    + 0.05 * torch.randn(INPUT_CHANNELS, dtype=dtype)
                )

    def initialize(self, e0: Tensor) -> CellState:
        """Create a zero state matching one feature slice's shape and dtype."""
        _check_features(e0)
        batch, height, width, _ = e0.shape
        shape = (batch, height, width)
        count = 0
        if self.kind == "original":
            return OriginalState(
                u=e0.new_zeros(shape),
                z=e0.new_zeros(shape),
                count=count,
            )
        return LiftedState(
            U=e0.new_zeros((*shape, INPUT_CHANNELS)),
            M0=e0.new_zeros((*shape, INPUT_CHANNELS)),
            M1=e0.new_zeros((*shape, INPUT_CHANNELS, INPUT_CHANNELS)),
            M2=e0.new_zeros(
                (*shape, INPUT_CHANNELS, INPUT_CHANNELS, INPUT_CHANNELS)
            ),
            count=count,
        )

    def step(self, state: CellState, e: Tensor) -> CellState:
        """Advance one zero-based time step of the cyclic spatial recurrence."""
        _check_features(e)
        _check_count(state.count)
        batch, height, width, _ = e.shape
        shape = (batch, height, width)
        denominator = max(1, state.count)

        if self.kind == "original":
            if not isinstance(state, OriginalState):
                raise TypeError("original cell requires an OriginalState")
            if state.u.shape != shape or state.z.shape != shape:
                raise ValueError("original state fields must have shape [B,H,W]")

            p_u = _p(state.u)
            v = p_u / denominator
            e_a = torch.einsum("bhwc,c->bhw", e, self.a)
            e_b = torch.einsum("bhwc,c->bhw", e, self.b)
            u = p_u + e_a
            z = _q(state.z) + e_b * (
                self.gamma[0] + self.gamma[1] * v + self.gamma[2] * v.square()
            )
            return OriginalState(u=u, z=z, count=state.count + 1)

        if not isinstance(state, LiftedState):
            raise TypeError("lifted cell requires a LiftedState")
        if (
            state.U.shape != (*shape, INPUT_CHANNELS)
            or state.M0.shape != (*shape, INPUT_CHANNELS)
            or state.M1.shape != (*shape, INPUT_CHANNELS, INPUT_CHANNELS)
            or state.M2.shape
            != (*shape, INPUT_CHANNELS, INPUT_CHANNELS, INPUT_CHANNELS)
        ):
            raise ValueError("lifted moment shapes must match [B,H,W,3^(order)]")

        pre_U = _p(state.U)
        v = pre_U / denominator
        U = pre_U + e
        M0 = _q(state.M0) + e
        M1 = _q(state.M1) + torch.einsum("bhwi,bhwj->bhwij", v, e)
        M2 = _q(state.M2) + torch.einsum("bhwi,bhwj,bhwk->bhwijk", v, v, e)
        return LiftedState(U=U, M0=M0, M1=M1, M2=M2, count=state.count + 1)

    def project(self, state: CellState) -> tuple[Tensor, Tensor]:
        """Return the original scalar fields ``(u,z)`` for either state form."""
        _check_count(state.count)
        if self.kind == "original":
            if not isinstance(state, OriginalState):
                raise TypeError("original cell requires an OriginalState")
            return state.u, state.z

        if not isinstance(state, LiftedState):
            raise TypeError("lifted cell requires a LiftedState")
        u = torch.einsum("bhwc,c->bhw", state.U, self.a)
        z0 = torch.einsum("bhwc,c->bhw", state.M0, self.b)
        z1 = torch.einsum("bhwij,i,j->bhw", state.M1, self.a, self.b)
        z2 = torch.einsum("bhwijk,i,j,k->bhw", state.M2, self.a, self.a, self.b)
        z = self.gamma[0] * z0 + self.gamma[1] * z1 + self.gamma[2] * z2
        return u, z

    def predict(self, state: CellState) -> Tensor:
        """Decode the dense grid with the current completed-step normalizer."""
        _, z = self.project(state)
        return torch.tanh(z / max(1, state.count))

    @staticmethod
    def detach(state: CellState) -> CellState:
        """Detach every tensor while retaining the shared Python step count."""
        if isinstance(state, OriginalState):
            return OriginalState(state.u.detach(), state.z.detach(), state.count)
        if isinstance(state, LiftedState):
            return LiftedState(
                state.U.detach(),
                state.M0.detach(),
                state.M1.detach(),
                state.M2.detach(),
                state.count,
            )
        raise TypeError(f"Unsupported spatial-cell state: {type(state).__name__}")


__all__ = [
    "INPUT_CHANNELS",
    "ORIGINAL_STATE_CHANNELS",
    "LIFTED_STATE_CHANNELS",
    "OriginalState",
    "LiftedState",
    "CellState",
    "SpatialPolynomialCell",
]
