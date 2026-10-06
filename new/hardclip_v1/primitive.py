"""Fixed lane hard clipping and an outcome-blind threshold calibration.

This is a standalone implementation draft, not a modification of the GitHub
repository. Feed the existing MLP proposal to FixedLaneHardClip; add its result
to the existing streamed carrier. Encoder, transport, Q, Z and readout stay
unchanged. All caps are ACTUAL write RMS, not proposal RMS.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import torch
from torch import Tensor, nn


class FixedLaneHardClip(nn.Module):
    """C24: four directional lanes, six payload coordinates per lane.

    The returned tensor is the C correction, not the next carrier. There are
    no trainable parameters. Caps persist as a buffer in the state dictionary.
    Autograd differentiates the genuine clipping map, without a straight-through
    estimator or a detached scale. The inactive branch is exactly eta*proposal.
    """

    def __init__(self, max_write_rms: Sequence[float], eta: float = 0.1):
        super().__init__()
        caps = torch.as_tensor(max_write_rms, dtype=torch.float32)
        if caps.shape != (4,) or not bool(torch.isfinite(caps).all()):
            raise ValueError("max_write_rms must contain four finite numbers")
        if not bool((caps > 0).all()) or not np.isfinite(eta) or eta <= 0:
            raise ValueError("Caps and eta must be strictly positive")
        self.register_buffer("max_write_rms", caps.clone())
        self.eta = float(eta)

    def forward(self, proposal: Tensor) -> Tensor:
        if proposal.ndim != 4 or proposal.shape[1] != 24:
            raise ValueError("proposal must have shape [batch,24,height,width]")
        if proposal.dtype not in (torch.float32, torch.float64):
            raise TypeError("This reference implementation supports FP32/FP64")
        if (proposal.device != self.max_write_rms.device
                or proposal.dtype != self.max_write_rms.dtype):
            raise ValueError("Move the writer buffer to the model device/dtype first")
        batch, _, height, width = proposal.shape
        raw = (self.eta * proposal).reshape(batch, 4, 6, height, width)
        r2 = raw.square().mean(dim=2, keepdim=True)
        b = self.max_write_rms.reshape(1, 4, 1, 1, 1)
        b2 = b.square()
        # Positive floor is inside the unused scale branch; it does not alter
        # any below-threshold write and prevents the zero-norm sqrt singularity.
        scale = b / torch.sqrt(torch.maximum(r2, b2))
        used = torch.where(r2 > b2, raw * scale, raw)
        return used.reshape_as(proposal)


def calibrate_write_caps(
    raw_write_rms: np.ndarray,
    *,
    removed_fraction: float = 0.01,
    max_trigger_fraction: float = 0.20,
) -> dict:
    """Calibrate four frozen caps from unmodified reference writes.

    Input: [N,4] RMS(0.1*m), sampled on open cells, with the same number of
    samples for each reference block/time. Do not pass labels or select models
    using success. No outcome-based threshold search is performed.

    For each lane choose the largest b with
        sum(max(r-b,0))/sum(r) >= removed_fraction
    to numerical precision (equality at the root). Since both removed amount
    and trigger frequency decrease with b, this root minimizes trigger frequency
    among caps achieving the requested amount. A dense intervention is flagged;
    the caller must not silently retune the target after observing that flag.
    """
    values = np.asarray(raw_write_rms, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise ValueError("raw_write_rms must be a nonempty [N,4] array")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("RMS samples must be finite and nonnegative")
    if not 0 < removed_fraction < 1 or not 0 < max_trigger_fraction < 1:
        raise ValueError("Both fractions must be strictly between zero and one")
    rows = []
    for lane in range(4):
        r = values[:, lane]
        total = float(r.sum())
        if total <= 0:
            raise ValueError(f"Lane {lane} has no nonzero reference writes")
        lo, hi = 0.0, float(r.max())
        for _ in range(48):
            mid = (lo + hi) / 2.0
            removed = float(np.maximum(r - mid, 0.0).sum()) / total
            if removed > removed_fraction:
                lo = mid
            else:
                hi = mid
        b = (lo + hi) / 2.0
        dose = float(np.maximum(r - b, 0.0).sum()) / total
        trigger = float(np.mean(r > b))
        rows.append({"lane": lane, "max_write_rms": b,
                     "proposal_rms_cap_at_eta_0p1": b / 0.1,
                     "removed_fraction": dose,
                     "trigger_fraction": trigger,
                     "sparse_enough": trigger <= max_trigger_fraction})
    return {"status": ("CALIBRATED" if all(x["sparse_enough"] for x in rows)
                        else "NO_SPARSE_TAIL_AT_PREDECLARED_DOSE"),
            "target_removed_fraction": removed_fraction,
            "max_trigger_fraction": max_trigger_fraction,
            "sample_count_per_lane": len(values), "lanes": rows}


def _self_test() -> None:
    torch.manual_seed(17)
    caps = (0.4, 0.5, 0.5, 0.55)
    for dtype in (torch.float32, torch.float64):
        writer = FixedLaneHardClip(caps).to(dtype=dtype)
        zero = torch.zeros(2, 24, 3, 3, dtype=dtype, requires_grad=True)
        y = writer(zero)
        g, = torch.autograd.grad(y.sum(), zero)
        assert torch.equal(y, torch.zeros_like(y))
        torch.testing.assert_close(g, torch.full_like(g, 0.1), rtol=0, atol=0)
        x = (torch.randn(2, 24, 3, 3, dtype=dtype) * 0.001).requires_grad_()
        assert torch.equal(writer(x), 0.1 * x)
        big = (torch.randn(2, 24, 3, 3, dtype=dtype) * 100).requires_grad_()
        result = writer(big)
        rms = result.reshape(2, 4, 6, 3, 3).square().mean(2).sqrt()
        assert bool((rms <= writer.max_write_rms[None, :, None, None] + 1e-6).all())
        grad, = torch.autograd.grad(result.square().sum(), big)
        assert bool(torch.isfinite(grad).all())
    writer = FixedLaneHardClip(caps).double()
    x = torch.randn(1, 24, 1, 1, dtype=torch.float64) * 15
    x.requires_grad_()
    assert torch.autograd.gradcheck(writer, (x,), atol=1e-5, rtol=1e-3)
    # A sparse high-amplitude tail: exactly 10% events at 5, the rest at 1.
    r = np.repeat(np.r_[np.ones(900), np.full(100, 5.)][:, None], 4, axis=1)
    out = calibrate_write_caps(r)
    assert out["status"] == "CALIBRATED"
    assert all(abs(row["max_write_rms"] - 4.86) < 1e-10 for row in out["lanes"])
    # A plateau: any nonzero dose acts on the entire distribution.
    out = calibrate_write_caps(np.ones((100, 4)))
    assert out["status"] == "NO_SPARSE_TAIL_AT_PREDECLARED_DOSE"
    print("PASS: FP32/FP64 identity, zero gradients, bounds, finite gradients, "
          "FP64 gradcheck, sparse-tail and plateau calibration fixtures")


if __name__ == "__main__":
    _self_test()
