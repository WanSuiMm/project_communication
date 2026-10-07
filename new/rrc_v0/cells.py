"""Current, factorized, and relation-refined streaming cells (RRC-v0)."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, Tuple

import torch
from torch import Tensor, nn

NEW_ROOT = Path(__file__).resolve().parents[1]
if str(NEW_ROOT / "streaming_carry") not in sys.path:
    sys.path.insert(0, str(NEW_ROOT / "streaming_carry"))

from stream_cells import StreamingCell, stream  # noqa: E402
from revision_cells import masked_laplacian  # noqa: E402


ARMS = ("current", "factorized", "rrc")
LANES = ("N", "E", "S", "W")
LANE_CHANNELS = 6
FACTOR_HIDDEN = 96
RELATION_RHO_INIT = 0.1
RELATION_GAMMA_INIT = torch.logit(torch.tensor(RELATION_RHO_INIT)).item()

RelationObserver = Callable[[nn.Module, Tensor, Tensor, Tensor], None]


def cyclic_lane_shift(carrier: Tensor, shift: int) -> Tensor:
    """Apply P_shift with output lane d receiving input lane (d+shift) mod 4.

    In N/E/S/W order, P0=[N,E,S,W], P1=[E,S,W,N],
    P2=[S,W,N,E], and P3=[W,N,E,S].
    """
    if carrier.ndim != 4 or carrier.shape[1] % 4:
        raise ValueError("Carrier must be [B,C,H,W] with four equal lanes")
    lane_width = carrier.shape[1] // 4
    lanes = carrier.split(lane_width, dim=1)
    offset = int(shift) % 4
    return torch.cat([lanes[(lane + offset) % 4] for lane in range(4)], dim=1)


def relation_mix(carrier: Tensor, relation_logits: Tensor, relation_gate: Tensor) -> Tensor:
    """Return H=U+rho*(sum_j softmax(alpha)_j P_j U-U)."""
    if carrier.ndim != 4 or carrier.shape[1] % 4:
        raise ValueError("Carrier must be [B,C,H,W] with four equal lanes")
    if relation_logits.shape != (4,):
        raise ValueError("relation_logits must have shape [4]")
    if relation_gate.numel() != 1:
        raise ValueError("relation_gate must be scalar")

    pi = torch.softmax(relation_logits, dim=0)
    rho = torch.sigmoid(relation_gate)
    average = torch.zeros_like(carrier)
    for shift in range(4):
        average = average + pi[shift] * cyclic_lane_shift(carrier, shift)
    return carrier + rho * (average - carrier)


class CurrentCell(StreamingCell):
    """Historical StreamingCell with a no-op relation observer hook."""

    def __init__(self) -> None:
        super().__init__(streaming=True)
        self.arm = "current"
        # Plain callable attribute: it is deliberately absent from state_dict.
        self.relation_observer: RelationObserver | None = None

    def _observe_relation(self, incoming_u: Tensor, relation_h: Tensor, x: Tensor) -> None:
        observer = self.relation_observer
        if observer is not None:
            observer(self, incoming_u, relation_h, x)

    def step(self, state: Tuple[Tensor, Tensor], x: Tensor) -> Tuple[Tensor, Tensor]:
        self._validate_x(x)
        workspace, latent = state
        # This is the canonical pre-stream perception used by StreamingCell.
        old_features = self._features(workspace, latent, x)
        incoming_u = stream(workspace, x[:, :1])
        self._observe_relation(incoming_u, incoming_u, x)
        features = torch.cat(
            (incoming_u, old_features[:, self.workspace_channels :]), dim=1
        )
        force = self.f_out(torch.tanh(self.f_in(features)))
        workspace_new = incoming_u + self.eta * force
        candidate = self._candidate(workspace_new, latent, x)
        latent_new = latent + self.alpha * candidate
        return workspace_new, latent_new

    def metadata(self) -> dict:
        metadata = super().metadata()
        metadata.update(
            {
                "arm": self.arm,
                "architecture": "historical_streaming_cell_with_relation_observer",
                "pure_transport_isometry": True,
                "only_pure_stream_preserves_norm": True,
                "relation_map": "identity; observer receives H=U",
                "relation_parameter_count": 0,
                "relation_observer_signature": "relation_observer(model,U,H,x)",
                "prestream_spatial_clock_2hop_preserved": True,
                "full_recurrence_stability_guarantee": False,
            }
        )
        return metadata


class FactorizedCell(StreamingCell):
    """Four-lane shared force network, optionally followed by RRC mixing."""

    def __init__(self, arm: str) -> None:
        if arm not in ("factorized", "rrc"):
            raise ValueError(f"Unknown factorized cell arm: {arm}")
        super().__init__(streaming=True)
        self.arm = arm
        self.workspace_hidden = FACTOR_HIDDEN
        self.lane_channels = LANE_CHANNELS

        # Each lane sees [incoming lane, pre-stream Laplacian lane, Z, L(Z), X].
        # A single pair of convolutions is shared across all four lane calls.
        per_lane_features = 2 * LANE_CHANNELS + 2 * self.latent_channels + 3
        self.f_in = nn.Conv2d(per_lane_features, FACTOR_HIDDEN, kernel_size=1)
        self.f_out = nn.Conv2d(FACTOR_HIDDEN, LANE_CHANNELS, kernel_size=1)
        nn.init.zeros_(self.f_out.weight)
        nn.init.zeros_(self.f_out.bias)

        self.relation_observer: RelationObserver | None = None
        if arm == "rrc":
            # alpha=0 gives uniform pi; gamma=logit(.1) gives rho=.1.
            self.relation_logits = nn.Parameter(torch.zeros(4))
            self.relation_gate = nn.Parameter(
                torch.tensor(RELATION_GAMMA_INIT, dtype=torch.float32)
            )

    def _observe_relation(self, incoming_u: Tensor, relation_h: Tensor, x: Tensor) -> None:
        observer = self.relation_observer
        if observer is not None:
            observer(self, incoming_u, relation_h, x)

    def _relation(self, incoming_u: Tensor) -> Tensor:
        if self.arm == "rrc":
            return relation_mix(incoming_u, self.relation_logits, self.relation_gate)
        return incoming_u

    def _factorized_force(
        self,
        incoming: Tensor,
        prestream_laplacian: Tensor,
        latent: Tensor,
        latent_laplacian: Tensor,
        x: Tensor,
    ) -> Tensor:
        batch, _, height, width = incoming.shape
        lane_features = []
        for lane in range(4):
            lo = lane * self.lane_channels
            hi = lo + self.lane_channels
            lane_features.append(
                torch.cat(
                    (
                        incoming[:, lo:hi],
                        prestream_laplacian[:, lo:hi],
                        latent,
                        latent_laplacian,
                        x,
                    ),
                    dim=1,
                )
            )
        packed = torch.stack(lane_features, dim=1).reshape(
            batch * 4, -1, height, width
        )
        lane_force = self.f_out(torch.tanh(self.f_in(packed)))
        return lane_force.reshape(batch, 4, self.lane_channels, height, width).reshape(
            batch, self.workspace_channels, height, width
        )

    def step(self, state: Tuple[Tensor, Tensor], x: Tensor) -> Tuple[Tensor, Tensor]:
        self._validate_x(x)
        workspace, latent = state
        mask = x[:, :1]
        incoming_u = stream(workspace, mask)
        relation_h = self._relation(incoming_u)
        self._observe_relation(incoming_u, relation_h, x)

        # Keep the first phase's spatial perception before pure streaming.
        prestream_laplacian = masked_laplacian(workspace, mask)
        latent_laplacian = masked_laplacian(latent, mask)
        force = self._factorized_force(
            relation_h, prestream_laplacian, latent, latent_laplacian, x
        )
        workspace_new = relation_h + self.eta * force

        # Canonical second phase: Q sees C_new and retains the whole Z bypass.
        candidate = self._candidate(workspace_new, latent, x)
        latent_new = latent + self.alpha * candidate
        return workspace_new, latent_new

    def metadata(self) -> dict:
        metadata = super().metadata()
        is_rrc = self.arm == "rrc"
        metadata.update(
            {
                "arm": self.arm,
                "architecture": "rrc_v0_factorized_shared_lane_force",
                "carrier_lanes": list(LANES),
                "payload_channels_per_lane": self.lane_channels,
                "workspace_input_channels": 2 * self.lane_channels
                + 2 * self.latent_channels
                + 3,
                "workspace_hidden": FACTOR_HIDDEN,
                "workspace_output_channels_per_lane": self.lane_channels,
                "workspace_force_features": [
                    "incoming lane (6)",
                    "pre-stream masked Laplacian(C_t) lane (6)",
                    "Z_t (8)",
                    "masked Laplacian(Z_t) (8)",
                    "X (3)",
                ],
                "workspace_force_sharing": "same 31->96->6 Conv1x1 network on all four lanes",
                "workspace_update": (
                    "C_new = U + eta*F_shared(U_lane,L(C_t)_lane,Z,L(Z),X)"
                    if not is_rrc
                    else "C_new = H + eta*F_shared(H_lane,L(C_t)_lane,Z,L(Z),X)"
                ),
                "latent_update": "Z_new = Z + 0.5*Q(C_new,Z,L(C_new),L(Z),X)",
                "candidate_input_channels": 2 * self.workspace_channels
                + 2 * self.latent_channels
                + 3,
                "pure_stream_operator": "masked port permutation with blocked-link bounce-back",
                "pure_transport_isometry": True,
                "only_pure_stream_preserves_norm": True,
                "relation_map": (
                    "H=U (exact identity)"
                    if not is_rrc
                    else "H=U+rho*(sum_j softmax(alpha)_j P_j U-U)"
                ),
                "relation_permutation_convention": (
                    "P_j output lane d receives input lane (d+j) mod 4; "
                    "N/E/S/W: P0=[N,E,S,W], P1=[E,S,W,N], "
                    "P2=[S,W,N,E], P3=[W,N,E,S]"
                ),
                "relation_initialization": (
                    None
                    if not is_rrc
                    else {
                        "relation_logits_alpha": [0.0, 0.0, 0.0, 0.0],
                        "initial_pi": [0.25, 0.25, 0.25, 0.25],
                        "relation_gate_gamma": RELATION_GAMMA_INIT,
                        "initial_rho": RELATION_RHO_INIT,
                        "learnable": True,
                        "outcome_blind": True,
                    }
                ),
                "relation_parameter_count": 5 if is_rrc else 0,
                "relation_observer_signature": "relation_observer(model,U,H,x)",
                "relation_map_nonexpansive": True,
                "relation_map_dissipative": is_rrc,
                "prestream_spatial_clock_2hop_preserved": True,
                "full_recurrence_stability_guarantee": False,
                "whole_Q_Z_bypass_retained": True,
                "parameter_count": sum(p.numel() for p in self.parameters()),
                "trainable_parameter_count": sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                ),
                "buildinfo": {
                    "input_channels": 3,
                    "workspace_channels": 24,
                    "latent_channels": 8,
                    "shared_force": "31->96->6 applied to four lanes",
                    "candidate_network": "67->16->8",
                    "readout": "8->1",
                    "initialization_seeded_by_make_model": True,
                },
            }
        )
        return metadata


def make_model(arm: str, seed: int, device: str | torch.device = "cpu") -> nn.Module:
    """Create one arm with canonical E/Q/readout initialization for ``seed``.

    Current keeps the historical StreamingCell initialization exactly. The two
    factorized arms consume the same construction sequence, so their shared F
    and the common encoder, Q, and readout tensors match bit for bit.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown RRC-v0 arm: {arm}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if arm == "current":
            model: nn.Module = CurrentCell()
        else:
            model = FactorizedCell(arm)
    return model.to(device)

