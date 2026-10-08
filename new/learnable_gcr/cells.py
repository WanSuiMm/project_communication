"""Ordered graph cells for the learnable GCR comparison.

Each node reads the state of its single valid predecessor.  The GCR arm keeps
exact first- and second-order prefix sums; the full-writer arm proposes a
learned residual update to the same 20-dimensional workspace.
"""
from __future__ import annotations

from typing import Mapping, NamedTuple

import torch
from torch import Tensor, nn


INPUT_CHANNELS = 8
FEATURE_CHANNELS = 4
WORKSPACE_CHANNELS = FEATURE_CHANNELS + FEATURE_CHANNELS * FEATURE_CHANNELS
INTERPRETER_INPUTS = WORKSPACE_CHANNELS + 1
WRITER_INPUTS = WORKSPACE_CHANNELS + FEATURE_CHANNELS + 1
COUNT_NORMALIZER = 97.0
WRITER_RESIDUAL_SCALE = 0.1
WRITER_SEED_OFFSET = 1_000_003


class OrderedState(NamedTuple):
    """Per-node prefix state and the current endpoint's scalar decoder state."""

    count: Tensor  # [B, N, 1]
    workspace: Tensor  # [B, N, 20]
    z: Tensor  # [B, 1]


class OrderedCell(nn.Module):
    """A shared-parameter encoder/interpreter with one of two prefix writers.

    ``batch`` is a mapping containing ``x`` [B,N,8], ``pred`` [B,N] (long,
    with -1 for a missing predecessor), ``mask`` [B,N] (bool), and ``endpoint``
    [B] (long node indices).  ``step`` updates every node from its predecessor
    state.  Supplying ``p`` reuses already-computed encoder features [B,N,4];
    it is left attached to its autograd graph, if any.
    """

    def __init__(self, kind: str, seed: int, rho: float = 0.5):
        super().__init__()
        if kind not in ("gcr", "full_writer"):
            raise ValueError(f"Unknown ordered-cell kind: {kind!r}")
        if not 0.0 <= float(rho) <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        self.kind = kind
        self.seed = int(seed)
        self.rho = float(rho)

        # Build common modules in the same order and RNG stream for both arms.
        # Forking keeps model construction from advancing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.phi = nn.Linear(INPUT_CHANNELS, FEATURE_CHANNELS, bias=False)
            self.interpreter = nn.Sequential(
                nn.Linear(INTERPRETER_INPUTS, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        self.writer: nn.Module | None = None
        if kind == "full_writer":
            # An isolated stream leaves phi and interpreter bit-identical to
            # the GCR arm for a given seed.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + WRITER_SEED_OFFSET)
                self.writer = nn.Sequential(
                    nn.Linear(WRITER_INPUTS, 16),
                    nn.ReLU(),
                    nn.Linear(16, WORKSPACE_CHANNELS),
                )
            # Keep ordinary nn.Linear initialization at the output head; the
            # fixed 0.1 residual scale controls the initial write magnitude.

    def feature(self, x: Tensor) -> Tensor:
        """Encode node inputs as trainable four-dimensional features."""
        if x.ndim != 3 or x.shape[-1] != INPUT_CHANNELS:
            raise ValueError(f"x must have shape [B,N,{INPUT_CHANNELS}]")
        return self.phi(x)

    def initialize(self, batch: Mapping[str, Tensor]) -> OrderedState:
        """Create zero prefix state on the batch input's device and dtype."""
        x = batch["x"]
        if x.ndim != 3 or x.shape[-1] != INPUT_CHANNELS:
            raise ValueError(f"x must have shape [B,N,{INPUT_CHANNELS}]")
        batch_size, node_count, _ = x.shape
        if node_count == 0:
            raise ValueError("the predecessor graph must contain at least one node")
        return OrderedState(
            count=x.new_zeros((batch_size, node_count, 1)),
            workspace=x.new_zeros((batch_size, node_count, WORKSPACE_CHANNELS)),
            z=x.new_zeros((batch_size, 1)),
        )

    @staticmethod
    def _gather_predecessors(value: Tensor, safe_pred: Tensor) -> Tensor:
        """Gather a per-node tensor at each predecessor index."""
        batch_size, node_count = safe_pred.shape
        index = safe_pred.reshape(batch_size, node_count, *([1] * (value.ndim - 2)))
        index = index.expand(batch_size, node_count, *value.shape[2:])
        return value.gather(1, index)

    def step(
        self,
        state: OrderedState,
        batch: Mapping[str, Tensor],
        p: Tensor | None = None,
    ) -> OrderedState:
        """Advance every active node and refresh the endpoint decoder state."""
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
        if state.workspace.shape != (batch_size, node_count, WORKSPACE_CHANNELS):
            raise ValueError("state.workspace must have shape [B,N,20]")
        if state.z.shape != (batch_size, 1):
            raise ValueError("state.z must have shape [B,1]")

        if p is None:
            # Keep this path differentiable through phi.  In particular, do
            # not detach features as a convenience for truncated BPTT.
            p = self.feature(x)
        elif p.shape != (batch_size, node_count, FEATURE_CHANNELS):
            raise ValueError("p must have shape [B,N,4]")

        # Clamp only for safe gathering.  A negative index, an out-of-range
        # index, or a masked predecessor contributes an all-zero state.
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

        if self.kind == "gcr":
            s1_predecessor = predecessor_workspace[..., :FEATURE_CHANNELS]
            s2_predecessor = predecessor_workspace[..., FEATURE_CHANNELS:]
            s1 = s1_predecessor + p
            pair = torch.einsum("bni,bnj->bnij", s1_predecessor, p).flatten(start_dim=-2)
            s2 = s2_predecessor + pair
            workspace = torch.cat((s1, s2), dim=-1)
        else:
            if self.writer is None:  # Keep the invariant clear to type checkers.
                raise RuntimeError("full_writer arm is missing its writer module")
            log_count = torch.log1p(count) / torch.log(count.new_tensor(COUNT_NORMALIZER))
            writer_input = torch.cat((predecessor_workspace, p, log_count), dim=-1)
            workspace = predecessor_workspace + WRITER_RESIDUAL_SCALE * self.writer(writer_input)

        workspace = torch.where(active, workspace, torch.zeros_like(workspace))

        safe_endpoint = endpoint.to(device=x.device, dtype=torch.long)
        endpoint_count = self._gather_predecessors(count, safe_endpoint.unsqueeze(1)).squeeze(1)
        endpoint_workspace = self._gather_predecessors(workspace, safe_endpoint.unsqueeze(1)).squeeze(1)
        normalized_workspace = endpoint_workspace / endpoint_count.clamp_min(1.0)
        endpoint_log_count = torch.log1p(endpoint_count) / torch.log(
            endpoint_count.new_tensor(COUNT_NORMALIZER)
        )
        interpreter_input = torch.cat((endpoint_log_count, normalized_workspace), dim=-1)
        endpoint_proposal = self.interpreter(interpreter_input)

        # z is only decoded for the supervised endpoint. Its EMA state is
        # carried between steps; no per-node decoder is instantiated.
        z = self.rho * state.z + (1.0 - self.rho) * endpoint_proposal
        return OrderedState(count=count, workspace=workspace, z=z)

    def predict(self, state: OrderedState) -> Tensor:
        """Return the tanh-bounded endpoint prediction with shape [B]."""
        return torch.tanh(state.z).squeeze(-1)

    def param_counts(self) -> dict[str, int]:
        """Report common, writer, and total trainable parameter counts."""
        common = sum(p.numel() for p in self.phi.parameters()) + sum(
            p.numel() for p in self.interpreter.parameters()
        )
        writer = 0 if self.writer is None else sum(p.numel() for p in self.writer.parameters())
        return {"phi": sum(p.numel() for p in self.phi.parameters()),
                "interpreter": sum(p.numel() for p in self.interpreter.parameters()),
                "writer": writer,
                "common": common,
                "total": common + writer}

    @staticmethod
    def detach(state: OrderedState) -> OrderedState:
        """Detach each carried tensor at an explicit caller-chosen boundary."""
        return OrderedState(
            count=state.count.detach(),
            workspace=state.workspace.detach(),
            z=state.z.detach(),
        )


class FixedEvidenceMLP(nn.Module):
    """Seeded diagnostic MLP for precomputed 73-dimensional fixed evidence."""

    def __init__(self, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.network = nn.Sequential(
                nn.Linear(73, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

    def forward(self, features: Tensor) -> Tensor:
        """Map [log-count, mean8, pair64] features to a [B] prediction."""
        if features.shape[-1] != 73:
            raise ValueError("fixed evidence features must have 73 channels")
        return torch.tanh(self.network(features)).squeeze(-1)
