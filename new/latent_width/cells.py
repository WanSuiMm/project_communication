"""Latent-width comparison cells with alternating two-port transport.

The two-channel arm is exploratory: its two learned carrier channels represent
the active axis's +/- ports, selected by an explicit per-rollout phase clock.
It does not have four fixed directional lanes.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_REVISION = ROOT / "new" / "workspace_revision"
STREAMING_CARRY = ROOT / "new" / "streaming_carry"
for _path in (WORKSPACE_REVISION, STREAMING_CARRY):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Keep the established implementations as the source of truth for the
# four-lane arms and for the shared workspace/latent parameterization.
from revision_cells import RevisionCell  # noqa: E402
from stream_cells import StreamingCell  # noqa: E402


LATENT_CHANNELS = 8
ARMS = [
    {
        "name": "w24",
        "workspace_channels": 24,
        "workspace_hidden": 40,
        "candidate_hidden": 16,
    },
    {
        "name": "w8",
        "workspace_channels": 8,
        "workspace_hidden": 40,
        "candidate_hidden": 16,
    },
    {
        "name": "w24_capacity",
        "workspace_channels": 24,
        "workspace_hidden": 16,
        "candidate_hidden": 12,
    },
    {
        "name": "w16",
        "workspace_channels": 16,
        "workspace_hidden": 40,
        "candidate_hidden": 16,
    },
    {
        "name": "w4",
        "workspace_channels": 4,
        "workspace_hidden": 40,
        "candidate_hidden": 16,
    },
    {
        "name": "w2_alternating",
        "workspace_channels": 2,
        "workspace_hidden": 40,
        "candidate_hidden": 16,
    },
]

_PHASE_DIRECTIONS = {
    # Channel order is north/south, then west/east.
    0: ((-1, 0), (1, 0)),
    1: ((0, -1), (0, 1)),
}


def _phase_value(phase: int | Tensor) -> int:
    if isinstance(phase, Tensor):
        if phase.device.type != "cpu" or phase.ndim != 0 or phase.dtype != torch.int64:
            raise ValueError("phase must be a CPU scalar int64 tensor")
        value = int(phase.item())
    elif isinstance(phase, int) and not isinstance(phase, bool):
        value = phase
    else:
        raise TypeError("phase must be an integer or CPU scalar int64 tensor")
    if value not in _PHASE_DIRECTIONS:
        raise ValueError(f"phase must be 0 or 1, got {value}")
    return value


def phase_transport(carrier: Tensor, mask: Tensor, phase: int | Tensor) -> Tensor:
    """Apply one masked two-port permutation along the active grid axis.

    Each open site's directional port moves to its open neighbor. A blocked
    link, wall, or exterior boundary bounces the port into the opposite port
    at the same site. Wall ports remain fixed. The operation is a permutation
    for binary masks and never wraps around the grid.
    """
    phase_index = _phase_value(phase)
    if carrier.ndim != 4 or carrier.shape[1] != 2:
        raise ValueError("carrier must have shape [batch, 2, height, width]")
    if mask.shape != (carrier.shape[0], 1, *carrier.shape[2:]):
        raise ValueError("mask shape must be [batch, 1, height, width]")

    batch, _, height, width = carrier.shape
    padded_carrier = F.pad(carrier, (1, 1, 1, 1))
    padded_mask = F.pad(mask.bool(), (1, 1, 1, 1))
    open_site = mask.bool()
    result = []
    for channel, (dy, dx) in enumerate(_PHASE_DIRECTIONS[phase_index]):
        # Pull from the upstream neighbor (site - direction), with a one-cell
        # padded border so exterior links are treated as blocked.
        upstream_y = slice(1 - dy, 1 - dy + height)
        upstream_x = slice(1 - dx, 1 - dx + width)
        upstream = padded_carrier[:, channel : channel + 1, upstream_y, upstream_x]
        upstream_open = padded_mask[:, :, upstream_y, upstream_x]
        opposite = carrier[:, 1 - channel : 2 - channel]
        incoming = torch.where(upstream_open, upstream, opposite)
        result.append(
            torch.where(open_site, incoming, carrier[:, channel : channel + 1])
        )
    return torch.cat(result, dim=1)


def inverse_phase_transport(carrier: Tensor, mask: Tensor, phase: int | Tensor) -> Tensor:
    """Apply the inverse permutation, equal to B T B for port reversal B."""
    reversed_ports = carrier[:, (1, 0)]
    return phase_transport(reversed_ports, mask, phase)[:, (1, 0)]


class TwoPhaseStreamingCell(RevisionCell):
    """Two-channel exploratory carrier with an explicit alternating clock.

    State is ``(workspace, latent, phase)``. ``phase`` is a CPU scalar
    ``torch.int64`` tensor so ordinary chunk detachment preserves the clock.
    Phase zero transports vertical +/- ports; phase one transports horizontal
    +/- ports. The clock belongs to the rollout state, not global mutable data.
    """

    def __init__(
        self,
        workspace_hidden: int = 40,
        candidate_hidden: int = 16,
    ) -> None:
        super().__init__(
            "ws_additive",
            workspace_channels=2,
            latent_channels=LATENT_CHANNELS,
            workspace_hidden=workspace_hidden,
            candidate_hidden=candidate_hidden,
        )

    def initial(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        workspace, latent = super().initial(x)
        phase = torch.zeros((), dtype=torch.int64, device="cpu")
        return workspace, latent, phase

    def step(
        self,
        state: tuple[Tensor, Tensor, Tensor],
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self._validate_x(x)
        if len(state) != 3:
            raise ValueError("two-phase state must be (workspace, latent, phase)")
        workspace, latent, phase = state
        if not isinstance(phase, Tensor):
            raise ValueError("phase state component must be a CPU scalar int64 tensor")
        phase_index = _phase_value(phase)

        # The force sees the same pre-transport L(C), L(Z) clock as
        # StreamingCell; only its first feature block is replaced by T(C).
        old_features = self._features(workspace, latent, x)
        incoming = phase_transport(workspace, x[:, :1], phase_index)
        force_features = torch.cat(
            (incoming, old_features[:, self.workspace_channels :]), dim=1
        )
        force = self.f_out(torch.tanh(self.f_in(force_features)))
        workspace_new = incoming + self.eta * force

        # Q uses the post-force workspace and the current latent state.
        candidate = self._candidate(workspace_new, latent, x)
        latent_new = latent + self.alpha * candidate
        next_phase = phase.new_tensor(1 - phase_index)
        return workspace_new, latent_new, next_phase

    def logits(self, state: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        return self.readout(state[1])

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "arm": "w2_alternating",
            "architecture": "latent_width_two_phase_streaming_carry",
            "status": "exploratory",
            "exploratory": True,
            "learned_carrier_channels": 2,
            "stationary_latent_channels": LATENT_CHANNELS,
            "carrier_representation": (
                "two learned channels interpreted as the active axis's +/- ports"
            ),
            "carrier_layout": "two alternating +/- ports; no four fixed directional lanes",
            "clock_bits": 1,
            "phase_clock": {
                "state_component": 2,
                "type": "CPU scalar torch.int64",
                "initial_phase": 0,
                "phase_0": "vertical +/- ports",
                "phase_1": "horizontal +/- ports",
                "scope": "explicit rollout state; no global mutable clock",
            },
            "transport_operator": (
                "alternating-axis masked two-port permutation with blocked-link "
                "bounce-back"
            ),
            "workspace_update": (
                "C_new = T_phase(C) + eta*F(T_phase(C),Z,L(C),L(Z),X)"
            ),
            "force_perception_clock": (
                "L(C),L(Z) before transport; first feature is incoming T_phase(C)"
            ),
            "transport_parameters": 0,
            "pure_transport_isometry": True,
            "full_update_lossless": False,
            "stability_guarantee": False,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }


def make_model(arm: str | dict[str, Any]) -> StreamingCell | TwoPhaseStreamingCell:
    """Construct one named arm from the ordered ``ARMS`` specifications."""
    name = arm.get("name") if isinstance(arm, dict) else arm
    if not isinstance(name, str):
        raise TypeError("arm must be an arm name or its specification dictionary")
    spec = next((entry for entry in ARMS if entry["name"] == name), None)
    if spec is None:
        raise ValueError(f"Unknown latent-width arm: {name}")

    workspace_channels = spec["workspace_channels"]
    if workspace_channels == 2:
        return TwoPhaseStreamingCell(
            workspace_hidden=spec["workspace_hidden"],
            candidate_hidden=spec["candidate_hidden"],
        )
    return StreamingCell(
        streaming=True,
        workspace_channels=workspace_channels,
        latent_channels=LATENT_CHANNELS,
        workspace_hidden=spec["workspace_hidden"],
        candidate_hidden=spec["candidate_hidden"],
    )
