"""Raw-moment reparameterization of the learnable GCR ordered cell.

The recurrent state stores the unprojected first and second moments (8 + 64
channels).  They are projected through the inherited ``phi`` weights only
when the current endpoint is decoded, which is algebraically equivalent to
the original GCR recurrence.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Mapping, NamedTuple

import torch
from torch import Tensor


_LEGACY_MODULE_NAME = "_reactiontransport_learnable_gcr_cells"


def _load_legacy_cells():
    """Load the sibling implementation under a collision-proof module name."""
    existing = sys.modules.get(_LEGACY_MODULE_NAME)
    if existing is not None:
        return existing

    source = Path(__file__).resolve().parents[1] / "learnable_gcr" / "cells.py"
    spec = importlib.util.spec_from_file_location(_LEGACY_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the legacy GCR cell from {source}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution so normal module introspection works and no
    # generic name such as ``cells`` can collide with another project module.
    sys.modules[_LEGACY_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_LEGACY_MODULE_NAME, None)
        raise
    return module


_legacy_cells = _load_legacy_cells()

INPUT_CHANNELS = _legacy_cells.INPUT_CHANNELS
FEATURE_CHANNELS = _legacy_cells.FEATURE_CHANNELS
WORKSPACE_CHANNELS = _legacy_cells.WORKSPACE_CHANNELS
RAW_WORKSPACE_CHANNELS = INPUT_CHANNELS + INPUT_CHANNELS * INPUT_CHANNELS
COUNT_NORMALIZER = _legacy_cells.COUNT_NORMALIZER
OrderedCell = _legacy_cells.OrderedCell


class ReparamState(NamedTuple):
    """Per-node raw moments and the current endpoint's scalar decoder state."""

    count: Tensor  # [B, N, 1]
    workspace: Tensor  # [B, N, 72], with U[8] then row-major V[8, 8]
    z: Tensor  # [B, 1]


class ReparamGCR(OrderedCell):
    """GCR cell with an 8+64 raw-moment carrier and endpoint-only projection."""

    def __init__(self, seed: int, rho: float = 0.5):
        # Reuse the old constructor verbatim so phi/interpreter parameter names,
        # initialization tensors, parameter count, and writer=None are exact.
        super().__init__("gcr", seed, rho)
        # Keep a useful model label.  Recurrence behavior is fully overridden
        # below and does not branch on this value.
        self.kind = "reparam_gcr"

    def initialize(self, batch: Mapping[str, Tensor]) -> ReparamState:
        """Create zero raw-moment state on the input's device and dtype."""
        x = batch["x"]
        if x.ndim != 3 or x.shape[-1] != INPUT_CHANNELS:
            raise ValueError(f"x must have shape [B,N,{INPUT_CHANNELS}]")
        batch_size, node_count, _ = x.shape
        if node_count == 0:
            raise ValueError("the predecessor graph must contain at least one node")
        return ReparamState(
            count=x.new_zeros((batch_size, node_count, 1)),
            workspace=x.new_zeros((batch_size, node_count, RAW_WORKSPACE_CHANNELS)),
            z=x.new_zeros((batch_size, 1)),
        )

    def project_workspace(self, raw: Tensor) -> Tensor:
        """Project raw ``[U, V]`` moments to the original 4+16 channels.

        ``raw`` may have any number of leading dimensions, including none.
        The projection remains attached to ``phi.weight`` for autograd.
        """
        if raw.ndim < 1 or raw.shape[-1] != RAW_WORKSPACE_CHANNELS:
            raise ValueError(f"raw must have final dimension {RAW_WORKSPACE_CHANNELS}")
        u = raw[..., :INPUT_CHANNELS]
        v = raw[..., INPUT_CHANNELS:].reshape(
            *raw.shape[:-1], INPUT_CHANNELS, INPUT_CHANNELS
        )
        weight = self.phi.weight
        s1 = torch.matmul(u, weight.transpose(-1, -2))
        s2 = torch.matmul(torch.matmul(weight, v), weight.transpose(-1, -2))
        return torch.cat((s1, s2.flatten(start_dim=-2)), dim=-1)

    def step(
        self,
        state: ReparamState,
        batch: Mapping[str, Tensor],
    ) -> ReparamState:
        """Advance raw moments locally, then decode the selected endpoint."""
        x = batch["x"]
        pred = batch["pred"]
        mask = batch["mask"]
        endpoint = batch["endpoint"]
        if x.ndim != 3 or x.shape[-1] != INPUT_CHANNELS:
            raise ValueError(f"x must have shape [B,N,{INPUT_CHANNELS}]")
        batch_size, node_count, _ = x.shape
        if node_count == 0:
            raise ValueError("the predecessor graph must contain at least one node")
        if pred.shape != (batch_size, node_count):
            raise ValueError("pred must have shape [B,N]")
        if pred.dtype != torch.long:
            raise TypeError("pred must have dtype torch.long")
        if mask.shape != (batch_size, node_count) or mask.dtype != torch.bool:
            raise TypeError("mask must be bool with shape [B,N]")
        if endpoint.shape != (batch_size,):
            raise ValueError("endpoint must have shape [B]")
        if state.count.shape != (batch_size, node_count, 1):
            raise ValueError("state.count must have shape [B,N,1]")
        if state.workspace.shape != (batch_size, node_count, RAW_WORKSPACE_CHANNELS):
            raise ValueError(f"state.workspace must have shape [B,N,{RAW_WORKSPACE_CHANNELS}]")
        if state.z.shape != (batch_size, 1):
            raise ValueError("state.z must have shape [B,1]")

        # Invalid, missing, out-of-range, or masked predecessors contribute
        # zero count and moments, matching the original cell's graph semantics.
        valid_index = (pred >= 0) & (pred < node_count)
        safe_pred = pred.clamp(min=0, max=max(node_count - 1, 0))
        predecessor_active = self._gather_predecessors(mask.unsqueeze(-1), safe_pred).squeeze(-1)
        valid_pred = valid_index & predecessor_active
        predecessor_count = self._gather_predecessors(state.count, safe_pred)
        predecessor_workspace = self._gather_predecessors(state.workspace, safe_pred)
        predecessor_count = torch.where(
            valid_pred.unsqueeze(-1), predecessor_count, torch.zeros_like(predecessor_count)
        )
        predecessor_workspace = torch.where(
            valid_pred.unsqueeze(-1), predecessor_workspace, torch.zeros_like(predecessor_workspace)
        )

        active = mask.unsqueeze(-1)
        count = torch.where(active, predecessor_count + 1.0, torch.zeros_like(predecessor_count))

        u_predecessor = predecessor_workspace[..., :INPUT_CHANNELS]
        v_predecessor = predecessor_workspace[..., INPUT_CHANNELS:]
        u = u_predecessor + x
        pair = torch.einsum("bni,bnj->bnij", u_predecessor, x).flatten(start_dim=-2)
        v = v_predecessor + pair
        workspace = torch.cat((u, v), dim=-1)
        workspace = torch.where(active, workspace, torch.zeros_like(workspace))

        safe_endpoint = endpoint.to(device=x.device, dtype=torch.long)
        endpoint_count = self._gather_predecessors(count, safe_endpoint.unsqueeze(1)).squeeze(1)
        endpoint_raw = self._gather_predecessors(workspace, safe_endpoint.unsqueeze(1)).squeeze(1)
        normalized_workspace = self.project_workspace(endpoint_raw) / endpoint_count.clamp_min(1.0)
        endpoint_log_count = torch.log1p(endpoint_count) / torch.log(
            endpoint_count.new_tensor(COUNT_NORMALIZER)
        )
        interpreter_input = torch.cat((endpoint_log_count, normalized_workspace), dim=-1)
        endpoint_proposal = self.interpreter(interpreter_input)

        z = self.rho * state.z + (1.0 - self.rho) * endpoint_proposal
        return ReparamState(count=count, workspace=workspace, z=z)

    @staticmethod
    def detach(state: ReparamState) -> ReparamState:
        """Detach each carried tensor at an explicit caller-chosen boundary."""
        return ReparamState(
            count=state.count.detach(),
            workspace=state.workspace.detach(),
            z=state.z.detach(),
        )


__all__ = ["ReparamState", "ReparamGCR"]
