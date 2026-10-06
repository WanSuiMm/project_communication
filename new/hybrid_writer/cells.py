"""Hybrid v0 writer cells for streamed workspace carriers."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[2]
STREAMING_DIR = ROOT / "new" / "streaming_carry"
if str(STREAMING_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMING_DIR))

from stream_cells import StreamingCell  # noqa: E402


ARMS = ("neural", "budget", "hybrid", "affine_hybrid")
VARIANTS = ARMS
WORKSPACE_CHANNELS = 24
LATENT_CHANNELS = 8
LANES = 4
PAYLOAD_PER_LANE = 6
INPUT_CHANNELS = 3
WRITER_INPUT_CHANNELS = 2 * WORKSPACE_CHANNELS + 2 * LATENT_CHANNELS + INPUT_CHANNELS
GATE_NODES = (0.0, 0.5, 1.0)
RMS_OFFSET = 0.5
RMS_EPSILON = 1e-12


def hat_basis(phi: Tensor) -> Tensor:
    """Return the nonnegative partition-of-unity hats at 0, 0.5, and 1.

    The node axis is inserted at dimension 2, so ``phi`` shaped
    ``[B, 4, H, W]`` maps to ``[B, 4, 3, H, W]``.
    """
    t = phi.clamp(0.0, 1.0)
    left = (1.0 - 2.0 * t).clamp_min(0.0)
    middle = 1.0 - (2.0 * t - 1.0).abs()
    right = (2.0 * t - 1.0).clamp_min(0.0)
    return torch.stack((left, middle, right), dim=2)


class HybridWriterCell(StreamingCell):
    """Stream a 24-channel carrier, then apply one of four local writers."""

    def __init__(self, arm: str):
        if arm not in ARMS:
            raise ValueError(f"Unknown hybrid-writer arm: {arm}")
        super().__init__(streaming=True)
        if self.workspace_channels != WORKSPACE_CHANNELS or self.latent_channels != LATENT_CHANNELS:
            raise ValueError("Hybrid v0 uses the canonical C24/Z8 widths")
        self.arm = arm
        self.writer_input_channels = WRITER_INPUT_CHANNELS

        if arm == "affine_hybrid":
            # Remove both modules from the module tree so the inactive MLP has
            # no parameters in optimizers, checkpoints, or metadata counts.
            del self.f_in
            del self.f_out
            self.affine_proposal = nn.Conv2d(
                self.writer_input_channels, self.workspace_channels, kernel_size=1
            )
            nn.init.zeros_(self.affine_proposal.weight)
            nn.init.zeros_(self.affine_proposal.bias)

        if arm == "budget":
            self.beta_d = nn.Parameter(torch.zeros(LANES))
        elif arm in ("hybrid", "affine_hybrid"):
            self.beta_dpq = nn.Parameter(torch.zeros(LANES, 3, 3))

        # Telemetry hook only; deliberately a plain attribute, not a parameter
        # or persistent buffer. It is called once per step before the Q update.
        self.writer_observer: Callable[[dict[str, Tensor | tuple[Tensor, Tensor] | None], Tensor], None] | None = None

    def _proposal(self, chi: Tensor) -> Tensor:
        if self.arm == "affine_hybrid":
            return self.affine_proposal(chi)
        return self.f_out(torch.tanh(self.f_in(chi)))

    @staticmethod
    def _lane_view(value: Tensor) -> Tensor:
        batch, channels, height, width = value.shape
        if channels != WORKSPACE_CHANNELS:
            raise ValueError(f"Carrier/proposal must have {WORKSPACE_CHANNELS} channels")
        return value.reshape(batch, LANES, PAYLOAD_PER_LANE, height, width)

    @staticmethod
    def _lane_rms(value: Tensor) -> Tensor:
        lanes = HybridWriterCell._lane_view(value)
        return torch.sqrt(lanes.square().mean(dim=2) + RMS_EPSILON)

    def _gate(
        self, incoming: Tensor, pre_stream_lc: Tensor
    ) -> tuple[Tensor | None, tuple[Tensor, Tensor] | None]:
        batch, _, height, width = incoming.shape
        if self.arm == "neural":
            return None, None
        if self.arm == "budget":
            lane_gate = self.beta_d.sigmoid().view(1, LANES, 1, 1)
            return lane_gate.expand(batch, -1, height, width), None

        phi1_rms = self._lane_rms(incoming)
        phi1 = phi1_rms / (1.0 + phi1_rms)
        phi2_rms = self._lane_rms(pre_stream_lc)
        phi2 = phi2_rms / (1.0 + phi2_rms)
        h1 = hat_basis(phi1)
        h2 = hat_basis(phi2)
        coefficients = self.beta_dpq.sigmoid()
        lane_gate = torch.einsum("bdphw,bdqhw,dpq->bdhw", h1, h2, coefficients)
        return lane_gate, (phi1, phi2)

    def _scale_proposal(self, proposal: Tensor, gate: Tensor | None) -> Tensor:
        if self.arm == "neural":
            return self.eta * proposal
        if gate is None:
            raise ValueError("A bounded writer must have a gate")
        lanes = self._lane_view(proposal)
        denominator = torch.sqrt(RMS_OFFSET**2 + lanes.square().mean(dim=2, keepdim=True))
        delta = self.eta * gate.unsqueeze(2) * lanes / denominator
        return delta.reshape_as(proposal)

    def details(self, state, x: Tensor) -> dict[str, Tensor | tuple[Tensor, Tensor] | None]:
        """Return the five frozen writer telemetry values used by ``step``.

        ``incoming`` and ``proposal`` are ``[B,24,H,W]``. ``gate`` is
        ``None`` for the neural arm and ``[B,4,H,W]`` for bounded arms.
        Hybrid ``phi`` is the pair ``(phi1, phi2)``, each ``[B,4,H,W]``.
        """
        self._validate_x(x)
        carrier, latent = state
        old_features = self._features(carrier, latent, x)
        incoming = self._transport(carrier, x)
        # The old local features stay pre-stream; only the carrier feature is
        # replaced by T(C), preserving the existing two-phase clock.
        chi = torch.cat((incoming, old_features[:, self.workspace_channels :]), dim=1)
        proposal = self._proposal(chi)
        pre_stream_lc = old_features[
            :, self.workspace_channels + self.latent_channels :
            2 * self.workspace_channels + self.latent_channels
        ]
        gate, phi = self._gate(incoming, pre_stream_lc)
        delta = self._scale_proposal(proposal, gate)
        return {
            "incoming": incoming,
            "proposal": proposal,
            "delta": delta,
            "gate": gate,
            "phi": phi,
        }

    @staticmethod
    def _transport(carrier: Tensor, x: Tensor) -> Tensor:
        # Keep this call aligned with StreamingCell's established operator.
        from stream_cells import stream

        return stream(carrier, x[:, :1])

    def step(self, state, x: Tensor):
        self._validate_x(x)
        carrier, latent = state
        detail = self.details(state, x)
        observer = self.writer_observer
        if observer is not None:
            observer(detail, x)
        carrier_new = detail["incoming"] + detail["delta"]
        candidate = self._candidate(carrier_new, latent, x)
        latent_new = latent + self.alpha * candidate
        return carrier_new, latent_new

    def metadata(self) -> dict:
        metadata = super().metadata()
        gate_parameters = 0 if self.arm == "neural" else (4 if self.arm == "budget" else 36)
        writer_parameters = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(("f_in.", "f_out.", "affine_proposal."))
        )
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        base_initialization = dict(metadata["initialization"])
        base_initialization["F_last_layer"] = (
            "absent" if self.arm == "affine_hybrid" else "zero initialized"
        )
        base_initialization["writer_proposal"] = (
            "Conv2d(67,24,kernel_size=1), zero initialized"
            if self.arm == "affine_hybrid"
            else "canonical StreamingCell MLP; final layer zero initialized"
        )
        base_initialization["gate_parameters"] = (
            "absent" if self.arm == "neural" else "zero initialized"
        )
        metadata.update(
            {
                "arm": self.arm,
                "architecture": "hybrid_writer_v0",
                "writer_input_channels": self.writer_input_channels,
                "proposal": (
                    "Affine(67,24) with zero output initialization"
                    if self.arm == "affine_hybrid"
                    else "Conv(67,40)-tanh-Conv(40,24)"
                ),
                "gate": {
                    "neural": "none",
                    "budget": "sigmoid(beta_d), one coefficient per lane",
                    "hybrid": "bilinear hat interpolation of sigmoid(beta_dpq)",
                    "affine_hybrid": "bilinear hat interpolation of sigmoid(beta_dpq)",
                }[self.arm],
                "gate_nodes": list(GATE_NODES),
                "gate_initial_beta": 0.0,
                "gate_initial_value": None if self.arm == "neural" else 0.5,
                "gate_parameter_count": gate_parameters,
                "writer_parameter_count": writer_parameters,
                "parameter_count": total_parameters,
                "parameter_counts": {
                    "total": total_parameters,
                    "common": total_parameters - writer_parameters - gate_parameters,
                    "writer_proposal": writer_parameters,
                    "gate": gate_parameters,
                },
                "initialization": base_initialization,
                "learned_gates": self.arm != "neural",
                "rms_offset": RMS_OFFSET,
                "rms_epsilon": RMS_EPSILON,
                "workspace_update": (
                    "C_new = T(C) + 0.1*m"
                    if self.arm == "neural"
                    else "C_new = T(C) + 0.1*a*m/sqrt(0.5^2 + mean_lane(m^2))"
                ),
                "bounded_rms_delta": self.arm != "neural",
                "bounded_rms_delta_limit": 0.1 if self.arm != "neural" else None,
                "proposal_jacobian_at_zero": 0.1,
                "transport_parameters": 0,
                "stationary_state": "Z",
                "latent_update": "Z_new = Z + 0.5*Q(C_new,Z,X)",
                "writer_observer_default": None,
            }
        )
        return metadata


def make_model(arm: str, seed: int, device="cpu") -> HybridWriterCell:
    """Build an arm from the canonical seeded StreamingCell parameter source.

    Common and MLP parameters are copied by name from a freshly seeded
    ``StreamingCell``. This makes initialization independent of the target
    arm's module creation order.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown hybrid-writer arm: {arm}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        canonical = StreamingCell()
        model = HybridWriterCell(arm)
        for name in ("encoder", "q_in", "q_out", "readout"):
            getattr(model, name).load_state_dict(getattr(canonical, name).state_dict())
        if arm != "affine_hybrid":
            for name in ("f_in", "f_out"):
                getattr(model, name).load_state_dict(getattr(canonical, name).state_dict())
    return model.to(device)
