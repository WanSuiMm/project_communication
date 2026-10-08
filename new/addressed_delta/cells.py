"""Current and addressed-delta carrier cells for the bounded comparison."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STREAMING_ROOT = ROOT / "new" / "streaming_carry"
for path in (STREAMING_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stream_cells import StreamingCell, stream  # noqa: E402
from writer import AddressedDeltaWriter  # noqa: E402

ARMS = ("current", "additive", "delta")
COUNTS = {"current": 5033, "additive": 5689, "delta": 5689}


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class CurrentCell(StreamingCell):
    """The historical StreamingCell, with only a stable arm label added."""

    def __init__(self) -> None:
        super().__init__(streaming=True)
        self.arm = "current"
        # Keeps simple observer-based runner code compatible; it has no effect
        # on the historical forward path or the model state_dict.
        self.relation_observer = None

    def metadata(self) -> dict:
        metadata = super().metadata()
        metadata.update(arm=self.arm, parameter_count=_parameter_count(self))
        return metadata


class AddressedDeltaCell(StreamingCell):
    """Replace only the old F carrier write; retain the original Q path."""

    def __init__(self, arm: str) -> None:
        if arm not in ("additive", "delta"):
            raise ValueError(f"Unknown addressed-delta arm: {arm}")
        # This preserves the baseline initialization order for encoder, F,
        # Q, and readout. The writer perception is then copied from f_in.
        super().__init__(streaming=True)
        self.arm = arm
        self.relation_observer = None

        writer = AddressedDeltaWriter(subtract_read=(arm == "delta"))
        with torch.no_grad():
            writer.perception.weight.copy_(self.f_in.weight)
            writer.perception.bias.copy_(self.f_in.bias)
        self.carrier_writer = writer

        # The writer owns the carrier write now. Drop the old F modules so
        # their parameters are neither optimized nor counted.
        del self.f_in
        del self.f_out

    def step(self, state: Tuple[Tensor, Tensor], x: Tensor) -> Tuple[Tensor, Tensor]:
        self._validate_x(x)
        c, z = state
        # Keep the frozen feature clock: L(C) and L(Z) are computed before
        # streaming, while the first writer feature is the incoming U.
        old_features = self._features(c, z, x)
        incoming = stream(c, x[:, :1])
        features = torch.cat((incoming, old_features[:, self.workspace_channels :]), dim=1)
        c_next = self.carrier_writer(incoming, features)

        # Preserve the historical post-writer Q and readout interface.
        q = self._candidate(c_next, z, x)
        z_next = z + self.alpha * q
        return c_next, z_next

    def metadata(self) -> dict:
        metadata = super().metadata()
        metadata.update(
            arm=self.arm,
            architecture="streaming_addressed_delta_carrier",
            carrier_writer={
                "type": "normalized_multi_read_delta" if self.arm == "delta" else "normalized_multi_read_additive",
                "input_channels": 67,
                "hidden_channels": 40,
                "ports": 4,
                "payload_channels": 6,
                "reads": 4,
                "step_size": 0.1,
                "subtract_read": self.arm == "delta",
                "initialization": "zero key head; normally initialized value head",
            },
            workspace_update=(
                "C_new = U + 0.1 K^T(V - K U)"
                if self.arm == "delta"
                else "C_new = U + 0.1 K^T V"
            ),
            parameter_count=_parameter_count(self),
            removed_modules=("f_in", "f_out"),
            original_pre_stream_feature_clock=True,
            full_recurrence_stability_guarantee=False,
        )
        return metadata


def make_model(arm: str, seed: int, device: str | torch.device = "cpu") -> StreamingCell:
    """Construct one arm from a reproducible baseline initialization."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    # fork_rng keeps block initialization local. Every arm receives the same
    # explicit baseline encoder/Q/readout tensors; candidate perception also
    # copies the baseline f_in before the old F modules are removed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        baseline = StreamingCell(streaming=True)
        if arm == "current":
            model = CurrentCell()
            model.load_state_dict(baseline.state_dict())
        else:
            model = AddressedDeltaCell(arm)
            for name in ("encoder", "q_in", "q_out", "readout"):
                getattr(model, name).load_state_dict(getattr(baseline, name).state_dict())
            model.carrier_writer.perception.load_state_dict(baseline.f_in.state_dict())
    model = model.to(device)
    expected = COUNTS[arm]
    if _parameter_count(model) != expected:
        raise RuntimeError(f"{arm} parameter count changed: {_parameter_count(model)} != {expected}")
    return model
