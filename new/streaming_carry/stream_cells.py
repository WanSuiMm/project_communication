"""Four payload lanes with bijective masked streaming and residual updates."""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'new/workspace_revision'))
from revision_cells import RevisionCell

VARIANTS = ('baseline', 'stream')
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W
OPPOSITE = (2, 3, 0, 1)


def reverse_lanes(carrier):
    return torch.cat([carrier.chunk(4, dim=1)[d] for d in OPPOSITE], dim=1)


def stream(carrier, mask):
    """Pull each lane from upstream, or bounce the blocked outgoing lane.

    Binary masks only. An open sender (i,d) maps to (i+delta_d,d) when
    that neighbor is open, otherwise to (i,opposite(d)). Wall ports stay
    fixed. Each destination port has exactly one predecessor, including
    at exterior boundaries and isolated open nodes. No wrapping or loss.
    """
    if carrier.ndim != 4 or carrier.shape[1] % 4:
        raise ValueError('Carrier must have four equal directional lanes')
    if mask.shape != (carrier.shape[0], 1, *carrier.shape[2:]):
        raise ValueError('Mask shape must be [B,1,H,W]')
    lanes = carrier.chunk(4, dim=1)
    c = carrier.shape[1] // 4
    padded = F.pad(carrier, (1, 1, 1, 1))
    m = F.pad(mask.bool(), (1, 1, 1, 1))
    upstream = ((slice(2, None), slice(1, -1)),
                (slice(1, -1), slice(None, -2)),
                (slice(None, -2), slice(1, -1)),
                (slice(1, -1), slice(2, None)))
    result = []
    for d, (ys, xs) in enumerate(upstream):
        moved = padded[:, d*c:(d+1)*c, ys, xs]
        incoming = torch.where(m[:, :, ys, xs], moved, lanes[OPPOSITE[d]])
        result.append(torch.where(mask.bool(), incoming, lanes[d]))
    return torch.cat(result, dim=1)


def inverse_stream(carrier, mask):
    """T^-1 = B T B, where B reverses all lane directions."""
    return reverse_lanes(stream(reverse_lanes(carrier), mask))


class StreamingCell(RevisionCell):
    def __init__(self, streaming=True, **kwargs):
        super().__init__('ws_additive', **kwargs)
        if self.workspace_channels % 4:
            raise ValueError('Workspace/carrier width must divide into four lanes')
        self.streaming = streaming

    def step(self, state, x):
        self._validate_x(x)
        w, z = state
        # L(W), L(Z) are pre-stream residual perception, in parallel with T.
        # Computing L(TW) here would add a third hop to the macro-step.
        old_features = self._features(w, z, x)
        incoming = stream(w, x[:, :1]) if self.streaming else w
        features = torch.cat((incoming, old_features[:, self.workspace_channels:]), dim=1)
        force = self.f_out(torch.tanh(self.f_in(features)))
        w_new = incoming + self.eta * force
        q = self._candidate(w_new, z, x)
        return w_new, z + self.alpha * q

    def metadata(self):
        return {**super().metadata(), 'architecture': 'streaming_carry',
                'streaming': self.streaming, 'carrier_lanes': ['N', 'E', 'S', 'W'],
                'payload_channels_per_lane': self.workspace_channels // 4,
                'stationary_state': 'Z', 'transport_parameters': 0,
                'transport_operator': 'masked port permutation with blocked-link bounce-back',
                'workspace_update': 'W_new = T(W) + eta*F(T(W),Z,L(W),L(Z),X)',
                'force_perception_clock': 'L(W),L(Z) before stream; first feature is incoming T(W)',
                'pure_transport_isometry': True, 'full_update_lossless': False,
                'stability_guarantee': False}


def make_variant(variant):
    if variant == 'baseline':
        return RevisionCell('ws_additive')
    if variant == 'stream':
        return StreamingCell()
    raise ValueError(variant)
