"""Stationary computation and task state with a persistent moving carrier."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "new/workspace_revision", ROOT / "new/streaming_carry"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from revision_cells import RevisionCell  # noqa: E402
from stream_cells import StreamingCell, stream  # noqa: E402

VARIANTS = ("baseline", "stream", "roles")
H_CHANNELS = 12
C_CHANNELS = 12
Z_CHANNELS = 8
FEATURE_CHANNELS = H_CHANNELS + C_CHANNELS + Z_CHANNELS + 3
RESIDUAL_HIDDEN = 72
State = tuple[Tensor, Tensor, Tensor]


class PersistentRoleCell(nn.Module):
    """Two evaluations of one shared pointwise collide-then-stream rule.

    C at a phase boundary is the incoming carrier. H/Z consume it before
    the phase streams the modified carrier to the next spatial location.
    """

    def __init__(self) -> None:
        super().__init__()
        # Preserve the original encoder/readout draws, including RNG advancement.
        original = RevisionCell("ws_additive")
        self.encoder = original.encoder
        self.readout = original.readout
        self.r_in = nn.Conv2d(FEATURE_CHANNELS, RESIDUAL_HIDDEN, 1)
        self.r_out = nn.Conv2d(RESIDUAL_HIDDEN, 32, 1)
        nn.init.zeros_(self.r_out.weight)
        nn.init.zeros_(self.r_out.bias)
        self.eta_h = 0.1
        self.eta_c = 0.1
        self.eta_z = 0.5

    @staticmethod
    def _validate_x(x: Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("X must have shape [batch,3,height,width]")

    def initial(self, x: Tensor) -> State:
        self._validate_x(x)
        encoded = self.encoder(x)
        h, c = encoded.split((H_CHANNELS, C_CHANNELS), dim=1)
        z = encoded.new_zeros((x.shape[0], Z_CHANNELS, *x.shape[-2:]))
        return h, c, z

    def phase(self, state: State, x: Tensor) -> State:
        h, c, z = state
        residual = self.r_out(torch.tanh(self.r_in(torch.cat((h, c, z, x), 1))))
        dh, dc, dz = residual.split((H_CHANNELS, C_CHANNELS, Z_CHANNELS), dim=1)
        return (h + self.eta_h * dh,
                stream(c + self.eta_c * dc, x[:, :1]),
                z + self.eta_z * dz)

    def step(self, state: State, x: Tensor) -> State:
        self._validate_x(x)
        return self.phase(self.phase(state, x), x)

    def rollout(self, x: Tensor, steps: int, state: State | None = None) -> State:
        self._validate_x(x)
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        state = self.initial(x) if state is None else state
        for _ in range(steps):
            state = self.step(state, x)
        return state

    def logits(self, state: State) -> Tensor:
        return self.readout(state[2])

    def metadata(self) -> dict:
        return {
            "arm": "roles",
            "architecture": "persistent_local_carrier_task_roles",
            "input_channels": 3,
            "persistent_state": ["H", "C", "Z"],
            "persistent_state_channels": {"H": 12, "C": 12, "Z": 8},
            "total_persistent_channels": 32,
            "stationary_states": ["H", "Z"],
            "carrier_lanes": ["N", "E", "S", "W"],
            "payload_channels_per_direction": 3,
            "residual_mlp": {"input": 35, "hidden": 72, "output": 32,
                             "activation": "Tanh", "shared_across_phases": True},
            "phase_update": "(H+.1*dH,T(C+.1*dC),Z+.5*dZ); d=R(H,C,Z,X)",
            "communication_phases_per_macro_step": 2,
            "phase_order": "pointwise collision then stream",
            "carrier_max_graph_hops_per_macro_step": 2,
            "readout_radius_after_p_phases_from_initial_carrier": "max(0,p-1)",
            "K8_carrier_radius_upper_bound": 16,
            "K8_readout_radius_upper_bound": 15,
            "transport_operator": "masked port permutation with blocked-link bounce-back",
            "transport_parameters": 0,
            "masked_laplacian_used": False,
            "direct_neighbor_state_access": False,
            "initialization": {
                "H_C": "split old encoder3->24 into H12 then C12; encode once",
                "Z": "zeros", "residual_last_layer": "zeros",
                "common_encoder_readout": "same draws as original RevisionCell",
            },
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "pure_base_map_isometry": True,
            "full_update_lossless": False,
            "stability_guarantee": False,
            "cross_detach_credit_restored": False,
        }


def make_variant(variant: str) -> nn.Module:
    if variant == "baseline":
        return RevisionCell("ws_additive")
    if variant == "stream":
        return StreamingCell()
    if variant == "roles":
        return PersistentRoleCell()
    raise ValueError(f"Unknown persistent-role variant: {variant}")
