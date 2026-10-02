"""A local recurrent state with a learned, ephemeral spatial interface."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "new/workspace_revision", ROOT / "new/streaming_carry"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from revision_cells import RevisionCell  # noqa: E402
from stream_cells import StreamingCell, stream  # noqa: E402


VARIANTS = ("baseline", "stream", "interface")
INPUT_CHANNELS = 3
WORKSPACE_CHANNELS = 24
LATENT_CHANNELS = 8
EMITTER_INPUT_CHANNELS = WORKSPACE_CHANNELS + LATENT_CHANNELS + INPUT_CHANNELS
MESSAGE_CHANNELS = WORKSPACE_CHANNELS
FEATURE_CHANNELS = WORKSPACE_CHANNELS + LATENT_CHANNELS + MESSAGE_CHANNELS + INPUT_CHANNELS
WORKSPACE_HIDDEN = 31
CANDIDATE_HIDDEN = 21
PAYLOAD_CHANNELS_PER_DIRECTION = MESSAGE_CHANNELS // 4


class InterfaceCell(RevisionCell):
    """Use learned four-lane messages while keeping W and Z locally resident.

    Each phase emits and transports a fresh message field. The field is consumed
    by that phase's residual update and is never stored in the recurrent state.
    """

    def __init__(self) -> None:
        # Preserve the baseline encoder and readout initialization draws exactly.
        super().__init__("ws_additive")

        self.workspace_hidden = WORKSPACE_HIDDEN
        self.candidate_hidden = CANDIDATE_HIDDEN
        self.interface_emitter = nn.Conv2d(
            EMITTER_INPUT_CHANNELS, MESSAGE_CHANNELS, kernel_size=1
        )
        self.f_in = nn.Conv2d(FEATURE_CHANNELS, WORKSPACE_HIDDEN, kernel_size=1)
        self.f_out = nn.Conv2d(WORKSPACE_HIDDEN, WORKSPACE_CHANNELS, kernel_size=1)
        self.q_in = nn.Conv2d(FEATURE_CHANNELS, CANDIDATE_HIDDEN, kernel_size=1)
        self.q_out = nn.Conv2d(CANDIDATE_HIDDEN, LATENT_CHANNELS, kernel_size=1)

        nn.init.zeros_(self.f_out.weight)
        nn.init.zeros_(self.f_out.bias)
        nn.init.zeros_(self.q_out.weight)
        nn.init.zeros_(self.q_out.bias)

    def _message(self, workspace: Tensor, latent: Tensor, x: Tensor) -> Tensor:
        emitted = self.interface_emitter(torch.cat((workspace, latent, x), dim=1))
        return stream(emitted, x[:, :1])

    def step(self, state: Tuple[Tensor, Tensor], x: Tensor) -> Tuple[Tensor, Tensor]:
        self._validate_x(x)
        workspace, latent = state

        incoming = self._message(workspace, latent, x)
        workspace_features = torch.cat((workspace, latent, incoming, x), dim=1)
        force = self.f_out(torch.tanh(self.f_in(workspace_features)))
        workspace_new = workspace + self.eta * force

        incoming_new = self._message(workspace_new, latent, x)
        candidate_features = torch.cat((workspace_new, latent, incoming_new, x), dim=1)
        candidate = self.q_out(torch.tanh(self.q_in(candidate_features)))
        latent_new = latent + self.alpha * candidate
        return workspace_new, latent_new

    def metadata(self) -> dict:
        metadata = super().metadata()
        metadata.update(
            {
                "arm": "interface",
                "architecture": "learned_ephemeral_interface",
                "workspace_hidden": WORKSPACE_HIDDEN,
                "candidate_hidden": CANDIDATE_HIDDEN,
                "workspace_input_channels": FEATURE_CHANNELS,
                "candidate_input_channels": FEATURE_CHANNELS,
                "interface_emitter_input_channels": EMITTER_INPUT_CHANNELS,
                "interface_emitter_output_channels": MESSAGE_CHANNELS,
                "interface_emitter_parameters": sum(
                    parameter.numel() for parameter in self.interface_emitter.parameters()
                ),
                "learned_interface_parameters": sum(
                    parameter.numel() for parameter in self.interface_emitter.parameters()
                ),
                "interface_emitter_shared_across_phases": True,
                "interface_emitter": "shared Conv2d(concat(W,Z,X),24,kernel_size=1)",
                "interface_directions": ["N", "E", "S", "W"],
                "payload_channels_per_direction": PAYLOAD_CHANNELS_PER_DIRECTION,
                "persistent_state": ["W", "Z"],
                "persistent_state_channels": WORKSPACE_CHANNELS + LATENT_CHANNELS,
                "local_identity_paths": ["W", "Z"],
                "transient_interface_channels": MESSAGE_CHANNELS,
                "persistent_interface_message": False,
                "message_lifetime": "one phase; discarded after the local residual update",
                "transport_operator": (
                    "masked four-lane port permutation with blocked-link bounce-back"
                ),
                "transport_parameters": 0,
                "communication_phases_per_macro_step": 2,
                "max_graph_hops_per_macro_step": 2,
                "workspace_update": "W_new = W + eta*F(W,Z,T(E(W,Z,X)),X)",
                "latent_update": "Z_new = Z + alpha*Q(W_new,Z,T(E(W_new,Z,X)),X)",
                "masked_laplacian_used": False,
                "direct_neighbor_state_access": False,
                "learned_gates": False,
                "stability_guarantee": False,
            }
        )
        return metadata


def make_variant(variant: str) -> RevisionCell:
    if variant == "baseline":
        return RevisionCell("ws_additive")
    if variant == "stream":
        return StreamingCell()
    if variant == "interface":
        return InterfaceCell()
    raise ValueError(f"Unknown local-interface variant: {variant}")
