"""Neural and fixed lane hard-clipped streamed carrier cells."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
STREAMING_DIR = ROOT / "new" / "streaming_carry"
HARDCLIP_DIR = ROOT / "new" / "hardclip_v1"
for directory in (STREAMING_DIR, HARDCLIP_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from primitive import FixedLaneHardClip  # noqa: E402
from stream_cells import StreamingCell, stream  # noqa: E402


ARMS = ("neural", "hardclip")
DEFAULT_CAPS = (
    0.39645159244537354,
    0.4826018810272217,
    0.4742255210876465,
    0.5157895088195801,
)


class HardClipCell(StreamingCell):
    """Canonical C24/Z8 StreamingCell with an optional fixed-write clip.

    Both arms carry the same cap buffer and the same trainable modules. The
    observer is deliberately a plain attribute so it is absent from model
    parameters and checkpoints. It receives writer details before the Q
    candidate update; observers should treat the tensors as read-only.
    """

    def __init__(self, arm: str, caps=DEFAULT_CAPS):
        if arm not in ARMS:
            raise ValueError(f"Unknown hardclip-v1 arm: {arm}")
        super().__init__(streaming=True)
        if self.workspace_channels != 24 or self.latent_channels != 8:
            raise ValueError("HardClip-v1 requires canonical C24/Z8 widths")
        self.arm = arm
        self.fixed_clip = FixedLaneHardClip(caps, eta=self.eta)
        self.writer_observer: Callable[[dict[str, Tensor | None], Tensor], None] | None = None

    def details(self, state, x: Tensor) -> dict[str, Tensor | None]:
        """Build one step's writer terms for the observer and carrier update.

        ``raw_write_rms`` is intentionally not materialized here. The observer
        can calculate it from ``raw_write`` while it collects dose statistics;
        the clipping primitive already performs the reduction needed by the
        hardclip arm. Neural computes its counterfactual ``clipped_write`` only
        while an observer is installed.
        """
        self._validate_x(x)
        carrier, latent = state
        old_features = self._features(carrier, latent, x)
        incoming = self._transport(carrier, x)
        # Preserve StreamingCell's two-phase clock: local features use the old
        # carrier, while the writer sees transported carrier in channel block 0.
        features = torch.cat(
            (incoming, old_features[:, self.workspace_channels :]), dim=1
        )
        proposal = self.f_out(torch.tanh(self.f_in(features)))
        raw_write = self.eta * proposal

        if self.arm == "hardclip":
            clipped_write = self.fixed_clip(proposal)
            delta = clipped_write
        else:
            delta = raw_write
            clipped_write = (
                self.fixed_clip(proposal) if self.writer_observer is not None else None
            )

        return {
            "incoming": incoming,
            "proposal": proposal,
            "raw_write": raw_write,
            "delta": delta,
            "clipped_write": clipped_write,
            # A small independent snapshot prevents telemetry from changing the
            # model's persistent cap buffer through this details dictionary.
            "caps": self.fixed_clip.max_write_rms.detach().clone(),
        }

    @staticmethod
    def _transport(carrier: Tensor, x: Tensor) -> Tensor:
        return stream(carrier, x[:, :1])

    def step(self, state, x: Tensor):
        # The unobserved neural arm delegates directly to the frozen canonical
        # implementation, preserving its exact forward and backward path.
        if self.arm == "neural" and self.writer_observer is None:
            return super().step(state, x)

        detail = self.details(state, x)
        if self.writer_observer is not None:
            self.writer_observer(detail, x)

        carrier_new = detail["incoming"] + detail["delta"]
        carrier, latent = state
        candidate = self._candidate(carrier_new, latent, x)
        latent_new = latent + self.alpha * candidate
        return carrier_new, latent_new


def make_model(arm: str, seed: int, device="cpu", caps=DEFAULT_CAPS) -> HardClipCell:
    """Build either arm from a seeded canonical StreamingCell by module name."""
    if arm not in ARMS:
        raise ValueError(f"Unknown hardclip-v1 arm: {arm}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        canonical = StreamingCell()
        model = HardClipCell(arm, caps=caps)
        for name in ("encoder", "f_in", "f_out", "q_in", "q_out", "readout"):
            getattr(model, name).load_state_dict(getattr(canonical, name).state_dict())
    return model.to(device)

