"""Two-phase masked-workspace NCA cells for the revision comparison."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import torch
from torch import Tensor, nn

MASKED_MEDIUM = Path(__file__).resolve().parents[1] / "masked_medium"
if str(MASKED_MEDIUM) not in sys.path:
    sys.path.insert(0, str(MASKED_MEDIUM))

from masked_cells import masked_laplacian  # noqa: E402


ARMS = ("ws_additive", "ws_revision")
INPUT_CHANNELS = 3
ALPHA = 0.5
ETA = 0.1


class RevisionCell(nn.Module):
    """Workspace update followed by a masked latent candidate update.

    The defaults are the frozen comparison architecture. Widths are exposed
    for small implementation checks; formal runs should use the defaults.
    """

    def __init__(
        self,
        arm: str,
        workspace_channels: int = 24,
        latent_channels: int = 8,
        workspace_hidden: int = 40,
        candidate_hidden: int = 16,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown workspace-revision arm: {arm}")
        widths = (workspace_channels, latent_channels, workspace_hidden, candidate_hidden)
        if any(width <= 0 for width in widths):
            raise ValueError("All channel widths must be positive")

        self.arm = arm
        self.workspace_channels = workspace_channels
        self.latent_channels = latent_channels
        self.workspace_hidden = workspace_hidden
        self.candidate_hidden = candidate_hidden
        self.alpha = ALPHA
        self.eta = ETA

        # Keep module creation order identical across arms so the initial RNG
        # draws and all trainable parameters match exactly.
        self.encoder = nn.Conv2d(INPUT_CHANNELS, workspace_channels, kernel_size=1)
        feature_channels = 2 * workspace_channels + 2 * latent_channels + INPUT_CHANNELS
        self.f_in = nn.Conv2d(feature_channels, workspace_hidden, kernel_size=1)
        self.f_out = nn.Conv2d(workspace_hidden, workspace_channels, kernel_size=1)
        self.q_in = nn.Conv2d(feature_channels, candidate_hidden, kernel_size=1)
        self.q_out = nn.Conv2d(candidate_hidden, latent_channels, kernel_size=1)
        self.readout = nn.Conv2d(latent_channels, 1, kernel_size=1)

        nn.init.zeros_(self.f_out.weight)
        nn.init.zeros_(self.f_out.bias)
        nn.init.zeros_(self.q_out.weight)
        nn.init.zeros_(self.q_out.bias)
        nn.init.zeros_(self.readout.bias)

    def initial(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        self._validate_x(x)
        workspace = self.encoder(x)
        latent = workspace.new_zeros(
            (workspace.shape[0], self.latent_channels, *workspace.shape[-2:])
        )
        return workspace, latent

    def _validate_x(self, x: Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != INPUT_CHANNELS:
            raise ValueError(f"x must have shape [batch, {INPUT_CHANNELS}, height, width]")

    def _features(self, workspace: Tensor, latent: Tensor, x: Tensor) -> Tensor:
        mask = x[:, :1]
        return torch.cat(
            (
                workspace,
                latent,
                masked_laplacian(workspace, mask),
                masked_laplacian(latent, mask),
                x,
            ),
            dim=1,
        )

    def _candidate(self, workspace: Tensor, latent: Tensor, x: Tensor) -> Tensor:
        features = self._features(workspace, latent, x)
        return self.q_out(torch.tanh(self.q_in(features)))

    def step(self, state: Tuple[Tensor, Tensor], x: Tensor) -> Tuple[Tensor, Tensor]:
        self._validate_x(x)
        workspace, latent = state
        features = self._features(workspace, latent, x)
        force = self.f_out(torch.tanh(self.f_in(features)))
        workspace_new = workspace + self.eta * force

        # The second communication phase sees the updated workspace and the
        # current latent state. Q is a raw linear output with no clamp/tanh.
        candidate = self._candidate(workspace_new, latent, x)
        if self.arm == "ws_additive":
            latent_new = latent + self.alpha * candidate
        else:
            latent_new = latent + self.alpha * (candidate - latent)
        return workspace_new, latent_new

    def rollout(
        self,
        x: Tensor,
        steps: int,
        state: Tuple[Tensor, Tensor] | None = None,
    ) -> Tuple[Tensor, Tensor]:
        self._validate_x(x)
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        state = self.initial(x) if state is None else state
        for _ in range(steps):
            state = self.step(state, x)
        return state

    def logits(self, state: Tuple[Tensor, Tensor]) -> Tensor:
        _, latent = state
        return self.readout(latent)

    def metadata(self) -> dict:
        return {
            "arm": self.arm,
            "architecture": "workspace_then_latent_candidate",
            "input_channels": INPUT_CHANNELS,
            "workspace_channels": self.workspace_channels,
            "latent_channels": self.latent_channels,
            "workspace_hidden": self.workspace_hidden,
            "candidate_hidden": self.candidate_hidden,
            "workspace_input_channels": 2 * self.workspace_channels
            + 2 * self.latent_channels
            + INPUT_CHANNELS,
            "candidate_input_channels": 2 * self.workspace_channels
            + 2 * self.latent_channels
            + INPUT_CHANNELS,
            "communication_phases_per_macro_step": 2,
            "transport_operator": "masked_laplacian(state, x[:, :1])",
            "workspace_update": "W_new = W + eta * F(W, Z, L(W), L(Z), X)",
            "latent_update": (
                "Z_new = Z + alpha * Q"
                if self.arm == "ws_additive"
                else "Z_new = Z + alpha * (Q - Z)"
            ),
            "alpha": self.alpha,
            "eta": self.eta,
            "candidate_output_activation": None,
            "learned_gates": False,
            "readout": "Conv2d(Z, 1, kernel_size=1)",
            "initialization": {
                "workspace": (
                    f"Conv2d(X, {self.workspace_channels}, kernel_size=1), "
                    "default PyTorch init"
                ),
                "latent": "zeros",
                "F_last_layer": "zero initialized",
                "Q_last_layer": "zero initialized",
                "readout_weights": "default PyTorch init",
                "readout_bias": "zero initialized",
            },
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "stability_guarantee": False,
        }


def make_cell(arm: str) -> RevisionCell:
    return RevisionCell(arm)
