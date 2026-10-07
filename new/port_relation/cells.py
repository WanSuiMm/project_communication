"""Streaming carry cells with a learned relation over the four payload lanes."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, Tuple

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = ROOT / "new" / "streaming_carry"
if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

from stream_cells import StreamingCell, stream  # noqa: E402


ARMS = ("current", "constant", "conditioned")
LANES = 4
PAYLOAD_PER_LANE = 6
RelationObserver = Callable[[nn.Module, Tensor, Tensor, Tensor, Tensor], None]


def _apply_lane_matrix(u: Tensor, matrix: Tensor) -> Tensor:
    """Apply per-pixel K[a,b] to six-channel payload lanes of U."""
    if u.ndim != 4 or u.shape[1] != LANES * PAYLOAD_PER_LANE:
        raise ValueError("U must have shape [B,24,H,W] with four six-channel lanes")
    batch, _, height, width = u.shape
    if matrix.shape != (batch, LANES, LANES, height, width):
        raise ValueError("matrix must have shape [B,4,4,H,W]")
    lanes = u.reshape(batch, LANES, PAYLOAD_PER_LANE, height, width)
    # Output lane a receives the sum over source lane b. Payload channel c is
    # carried independently, so there is no mixing between the six payloads.
    moved = torch.einsum("najhw,njchw->nachw", matrix, lanes)
    return moved.reshape_as(u)


class PortRelationCell(StreamingCell):
    """Historical StreamingCell plus an optional local lane-relation action.

    ``conditioned`` represents K(Z) as a zero-initialized 8->16 Conv1x1:
    its bias is the constant part and its weights supply the tanh(Z)-dependent
    part. This keeps the prescribed 144 added parameters without adding a
    second constant matrix.
    """

    def __init__(self, arm: str) -> None:
        if arm not in ARMS:
            raise ValueError(f"Unknown port-relation arm: {arm}")
        # Keep the historical shared module construction and initialization
        # order unchanged. Relation-only modules are created after it.
        super().__init__(streaming=True)
        self.arm = arm
        # Plain instrumentation hook: never enters parameters or state_dict.
        self.relation_observer: RelationObserver | None = None

        if arm == "constant":
            self.relation_matrix = nn.Parameter(torch.zeros(LANES, LANES))
        elif arm == "conditioned":
            self.relation_map = nn.Conv2d(
                self.latent_channels, LANES * LANES, kernel_size=1, bias=True
            )
            nn.init.zeros_(self.relation_map.weight)
            nn.init.zeros_(self.relation_map.bias)

    def _relation_action(self, u: Tensor, z: Tensor) -> tuple[Tensor | None, Tensor | None]:
        """Return full K(Z)U and, when observed, only its Z-dependent part."""
        if self.arm == "current":
            return None, None

        batch, _, height, width = u.shape
        if self.arm == "constant":
            matrix = self.relation_matrix.view(1, LANES, LANES, 1, 1).expand(
                batch, LANES, LANES, height, width
            )
            return _apply_lane_matrix(u, matrix), None

        # Channel order is row-major [output lane a, source lane b]. The bias
        # is theta0; subtracting it only for observer telemetry isolates the
        # state-dependent action while keeping the training path to one K(Z)U.
        raw = self.relation_map(torch.tanh(z))
        matrix = raw.reshape(batch, LANES, LANES, height, width)
        full_action = _apply_lane_matrix(u, matrix)
        if self.relation_observer is None:
            return full_action, None
        # This second action is telemetry only; do not retain another autograd
        # graph or intermediate backward buffers for it.
        with torch.no_grad():
            constant = self.relation_map.bias.view(1, LANES, LANES, 1, 1)
            state_matrix = matrix.detach() - constant
            state_action = _apply_lane_matrix(u.detach(), state_matrix)
        return full_action, state_action

    def _observe(
        self,
        u: Tensor,
        relation_delta: Tensor | None,
        conditioned_delta: Tensor | None,
        x: Tensor,
    ) -> None:
        observer = self.relation_observer
        if observer is None:
            return
        if relation_delta is None:
            relation_delta = torch.zeros_like(u)
        if conditioned_delta is None:
            conditioned_delta = torch.zeros_like(u)
        observer(self, u, relation_delta, conditioned_delta, x)

    def step(self, state: Tuple[Tensor, Tensor], x: Tensor) -> Tuple[Tensor, Tensor]:
        self._validate_x(x)
        c, z = state
        # Preserve the canonical pre-stream L(C), L(Z) feature clock.
        old_features = self._features(c, z, x)
        u = stream(c, x[:, :1])
        relation_delta, conditioned_delta = self._relation_action(u, z)
        self._observe(u, relation_delta, conditioned_delta, x)

        features = torch.cat((u, old_features[:, self.workspace_channels :]), dim=1)
        force = self.f_out(torch.tanh(self.f_in(features)))
        if self.arm == "current":
            # Keep the historical expression exactly, including operation order.
            c_new = u + self.eta * force
        else:
            assert relation_delta is not None
            c_new = u + self.eta * (force + relation_delta)

        # Original Q path: its whole input and Z bypass are unchanged.
        q = self._candidate(c_new, z, x)
        z_new = z + self.alpha * q
        return c_new, z_new

    def metadata(self) -> dict:
        metadata = super().metadata()
        relation_map = {
            "current": "identity; no learned relation action",
            "constant": "K[a,b] applied to six-channel lane payloads",
            "conditioned": "K(Z)=Conv1x1(tanh(Z)); output channel a*4+b is K[a,b]",
        }[self.arm]
        metadata.update(
            {
                "arm": self.arm,
                "architecture": "streaming_port_relation",
                "carrier_lanes": ["N", "E", "S", "W"],
                "payload_channels_per_lane": PAYLOAD_PER_LANE,
                "relation_map": relation_map,
                "relation_parameter_count": sum(
                    parameter.numel() for parameter in relation_parameters(self).values()
                ),
                "relation_observer_signature": (
                    "relation_observer(model,U,relation_delta,conditioned_delta,x)"
                ),
                "workspace_update": "C_new = U + eta*(F(U,Z,L(C),L(Z),X) + K(Z)U)",
                "conditioned_delta_definition": (
                    "K(Z)U - K(0)U; only the tanh(Z)-dependent contribution"
                ),
                "prestream_spatial_clock_2hop_preserved": True,
                "full_recurrence_stability_guarantee": False,
            }
        )
        return metadata


def make_model(
    arm: str, seed: int, device: str | torch.device = "cpu"
) -> PortRelationCell:
    """Build an arm with the same common tensors for every arm at ``seed``."""
    if arm not in ARMS:
        raise ValueError(arm)
    # Relation modules are constructed only after StreamingCell has drawn all
    # common tensors. fork_rng also restores the caller's RNG state afterward.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = PortRelationCell(arm)
    return model.to(device)


def relation_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """Return the trainable relation tensors, empty for the current arm."""
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if name == "relation_matrix" or name.startswith("relation_map.")
    }
