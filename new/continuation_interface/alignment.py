"""Linear state alignment and exact latent-channel gauge transforms.

State arrays use ``[T, B, C, H, W]``. For the workspace, the 24 channels are
four ordered lanes of six channels; the fitted 6x6 map is shared by all lanes.
The convention is column-oriented: ``target = A @ source``.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


WORKSPACE_CHANNELS = 24
LANES = 4
LANE_WIDTH = 6
LATENT_CHANNELS = 8


def _validate_calibration(source: tuple, target: tuple, mask: np.ndarray):
    if len(source) != 2 or len(target) != 2:
        raise ValueError("source and target must be (W, Z) tuples")
    src = tuple(np.asarray(value, dtype=np.float64) for value in source)
    tgt = tuple(np.asarray(value, dtype=np.float64) for value in target)
    if src[0].ndim != 5 or src[1].ndim != 5:
        raise ValueError("calibration states must have shape [T,B,C,H,W]")
    if any(a.shape != b.shape for a, b in zip(src, tgt)):
        raise ValueError("source and target state shapes must match")
    if src[0].shape[2] != WORKSPACE_CHANNELS:
        raise ValueError("W must have 24 channels (four lanes of six)")
    if src[1].shape[2] != LATENT_CHANNELS:
        raise ValueError("Z must have 8 channels")
    if src[0].shape[:2] != src[1].shape[:2] or src[0].shape[-2:] != src[1].shape[-2:]:
        raise ValueError("W and Z calibration dimensions must agree")
    mask = np.asarray(mask)
    batch, height, width = src[0].shape[1], *src[0].shape[-2:]
    if mask.shape != (batch, 1, height, width):
        raise ValueError("mask must have shape [B,1,H,W]")
    return src, tgt, mask[:, 0].astype(bool, copy=False)


def _samples(values: np.ndarray, open_cells: np.ndarray, workspace: bool) -> np.ndarray:
    """Return rows of channel vectors from every time/map/open cell/lane."""
    times, batch, channels, height, width = values.shape
    if workspace:
        # [T,B,4,6,H,W] -> [T,B,H,W,4,6], retaining all four lanes as samples.
        arranged = values.reshape(times, batch, LANES, LANE_WIDTH, height, width)
        arranged = arranged.transpose(0, 1, 4, 5, 2, 3)
        lane_mask = np.broadcast_to(
            open_cells[None, :, :, :, None], (times, batch, height, width, LANES)
        )
        return arranged[lane_mask]
    # [T,B,8,H,W] -> [T,B,H,W,8].
    arranged = values.transpose(0, 1, 3, 4, 2)
    cell_mask = np.broadcast_to(open_cells[None, :, :, :], (times, batch, height, width))
    return arranged[cell_mask]


def _fit_matrix(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, dict[str, Any]]:
    if x.shape[0] == 0:
        raise ValueError("calibration mask contains no open cells")
    dim = x.shape[1]
    gram = x.T @ x
    cross = y.T @ x
    trace_scale = float(np.trace(gram) / dim)
    # All-zero/underflowed calibration states still get a well-defined scale.
    scale = trace_scale if np.isfinite(trace_scale) and trace_scale > 1e-15 else 1.0
    lam = float(ridge) * scale
    regularized = gram + lam * np.eye(dim, dtype=np.float64)
    try:
        matrix = np.linalg.solve(regularized, cross.T).T
    except np.linalg.LinAlgError:
        matrix = cross @ np.linalg.pinv(regularized)
    return matrix, {
        "samples": int(x.shape[0]),
        "ridge_requested": float(ridge),
        "ridge_scale": scale,
        "ridge_lambda": lam,
    }


def fit_alignment(source: tuple, target: tuple, mask: np.ndarray, ridge: float = 1e-4) -> dict:
    """Fit no-bias float64 ridge maps for W and Z on open calibration cells.

    Workspace observations from all four six-channel lanes share one map.
    Time, batch/map, spatial position, and lane observations are pooled.
    """
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    src, tgt, open_cells = _validate_calibration(source, target, mask)
    xw = _samples(src[0], open_cells, workspace=True)
    yw = _samples(tgt[0], open_cells, workspace=True)
    xz = _samples(src[1], open_cells, workspace=False)
    yz = _samples(tgt[1], open_cells, workspace=False)
    aw, fit_w = _fit_matrix(xw, yw, ridge)
    az, fit_z = _fit_matrix(xz, yz, ridge)
    return {"W": aw, "Z": az, "fit": {"W": fit_w, "Z": fit_z}}


def identity_alignment() -> dict[str, np.ndarray]:
    """Return identity maps in the module's column-vector convention."""
    return {
        "W": np.eye(LANE_WIDTH, dtype=np.float64),
        "Z": np.eye(LATENT_CHANNELS, dtype=np.float64),
    }


def apply_alignment(state: tuple[torch.Tensor, torch.Tensor], alignment: dict) -> tuple:
    """Apply shared lane map to W and the 8x8 map to Z on their own device."""
    if len(state) != 2:
        raise ValueError("state must be a (W, Z) tuple")
    workspace, latent = state
    if workspace.ndim != 4 or workspace.shape[1] != WORKSPACE_CHANNELS:
        raise ValueError("W must have shape [B,24,H,W]")
    if latent.ndim != 4 or latent.shape[1] != LATENT_CHANNELS:
        raise ValueError("Z must have shape [B,8,H,W]")
    if workspace.shape[0] != latent.shape[0] or workspace.shape[-2:] != latent.shape[-2:]:
        raise ValueError("W and Z batch/spatial dimensions must agree")
    aw = torch.as_tensor(alignment["W"], device=workspace.device, dtype=workspace.dtype)
    az = torch.as_tensor(alignment["Z"], device=latent.device, dtype=latent.dtype)
    if tuple(aw.shape) != (LANE_WIDTH, LANE_WIDTH) or tuple(az.shape) != (LATENT_CHANNELS, LATENT_CHANNELS):
        raise ValueError("alignment matrices must have shapes W=[6,6], Z=[8,8]")
    batch, _, height, width = workspace.shape
    lanes = workspace.reshape(batch, LANES, LANE_WIDTH, height, width)
    mapped_w = torch.einsum("oi,blihw->blohw", aw, lanes).reshape_as(workspace)
    mapped_z = torch.einsum("oi,bihw->bohw", az, latent)
    return mapped_w, mapped_z


def _matrix_pair(value: Any) -> dict[str, np.ndarray] | None:
    if isinstance(value, dict) and "W" in value and "Z" in value:
        return {key: np.asarray(value[key], dtype=np.float64) for key in ("W", "Z")}
    return None


def _cycle_one(forward: np.ndarray, reverse: np.ndarray) -> dict[str, float]:
    fwd = np.asarray(forward, dtype=np.float64)
    rev = np.asarray(reverse, dtype=np.float64)
    if fwd.ndim != 2 or fwd.shape[0] != fwd.shape[1] or rev.shape != fwd.shape:
        raise ValueError("cycle maps must be equally sized square matrices")
    error = rev @ fwd - np.eye(fwd.shape[0], dtype=np.float64)
    denom_spec = max(float(np.linalg.norm(np.eye(fwd.shape[0]), ord=2)), 1e-12)
    denom_frob = max(float(np.linalg.norm(np.eye(fwd.shape[0], dtype=np.float64))), 1e-12)
    return {
        "relative_spectral_error": float(np.linalg.norm(error, ord=2) / denom_spec),
        "relative_frobenius_error": float(np.linalg.norm(error, ord="fro") / denom_frob),
    }


def cycle_diagnostics(forward: Any, reverse: Any) -> dict:
    """Measure ``A_reverse @ A_forward - I`` for one map or W/Z map pairs."""
    fwd_pair, rev_pair = _matrix_pair(forward), _matrix_pair(reverse)
    if fwd_pair is None or rev_pair is None:
        return _cycle_one(forward, reverse)
    return {key: _cycle_one(fwd_pair[key], rev_pair[key]) for key in ("W", "Z")}


def alignment_diagnostics(source: tuple, target: tuple, mask: np.ndarray, alignment: dict) -> dict:
    """Report open-cell normalized RMS residuals and matrix-only descriptors."""
    src, tgt, open_cells = _validate_calibration(source, target, mask)
    matrices = {key: np.asarray(alignment[key], dtype=np.float64) for key in ("W", "Z")}
    descriptors = {}
    residuals = {}
    for key, workspace in (("W", True), ("Z", False)):
        x = _samples(src[0 if key == "W" else 1], open_cells, workspace)
        y = _samples(tgt[0 if key == "W" else 1], open_cells, workspace)
        matrix = matrices[key]
        if matrix.shape != (x.shape[1], x.shape[1]):
            raise ValueError(f"{key} alignment matrix has the wrong shape")
        err = y - x @ matrix.T
        rms_target = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
        rms_error = float(np.sqrt(np.mean(np.square(err)))) if err.size else 0.0
        residuals[key] = rms_error / max(rms_target, 1e-12)
        singular = np.linalg.svd(matrix, compute_uv=False)
        descriptors[key] = {
            "condition_number": float(np.linalg.cond(matrix)),
            "singular_values": singular.tolist(),
        }
    return {"normalized_rms_residual": residuals, "matrix_diagnostics": descriptors}


def exact_gauge_clone(model: torch.nn.Module, permutation) -> torch.nn.Module:
    """Deep-copy a cell and conjugate its latent channels by ``z' = z[permutation]``.

    For Conv2d weight tensors, indexing input columns by ``permutation`` is
    right multiplication by P^-1. Output rows use P. The feature layout is
    exactly ``[W, Z, L(W), L(Z), X]`` in both workspace and candidate MLPs.
    """
    perm = np.asarray(
        permutation.detach().cpu().numpy() if torch.is_tensor(permutation) else permutation,
        dtype=np.int64,
    )
    latent_channels = int(getattr(model, "latent_channels", LATENT_CHANNELS))
    workspace_channels = int(getattr(model, "workspace_channels", WORKSPACE_CHANNELS))
    if perm.shape != (latent_channels,) or not np.array_equal(np.sort(perm), np.arange(latent_channels)):
        raise ValueError("permutation must contain every latent-channel index exactly once")
    if workspace_channels != WORKSPACE_CHANNELS or latent_channels != LATENT_CHANNELS:
        raise ValueError("this gauge transform is defined for W=24 and Z=8")
    clone = deepcopy(model)
    idx = torch.as_tensor(perm, dtype=torch.long)
    z_start, lz_start = workspace_channels, 2 * workspace_channels + latent_channels
    with torch.no_grad():
        for layer in (clone.f_in, clone.q_in):
            if layer.weight.shape[1] != 2 * workspace_channels + 2 * latent_channels + 3:
                raise ValueError("unexpected feature layout in cell input layer")
            idx_z = idx.to(layer.weight.device)
            layer.weight[:, z_start : z_start + latent_channels] = layer.weight[
                :, z_start : z_start + latent_channels
            ].index_select(1, idx_z)
            layer.weight[:, lz_start : lz_start + latent_channels] = layer.weight[
                :, lz_start : lz_start + latent_channels
            ].index_select(1, idx_z)
        idx_out = idx.to(clone.q_out.weight.device)
        clone.q_out.weight.copy_(clone.q_out.weight.index_select(0, idx_out))
        clone.q_out.bias.copy_(clone.q_out.bias.index_select(0, idx_out))
        idx_readout = idx.to(clone.readout.weight.device)
        clone.readout.weight.copy_(clone.readout.weight.index_select(1, idx_readout))
    return clone


def sanity_check() -> dict[str, Any]:
    """Check an exact latent gauge on an active 8x8 CPU StreamingCell rollout."""
    stream_dir = Path(__file__).resolve().parents[1] / "streaming_carry"
    if str(stream_dir) not in sys.path:
        sys.path.insert(0, str(stream_dir))
    from stream_cells import StreamingCell

    torch.manual_seed(31)
    model = StreamingCell().cpu().eval()
    with torch.no_grad():
        # Activate both residual heads; the repository defaults zero-initialize them.
        torch.nn.init.normal_(model.f_out.weight, mean=0.0, std=0.015)
        torch.nn.init.normal_(model.f_out.bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(model.q_out.weight, mean=0.0, std=0.015)
        torch.nn.init.normal_(model.q_out.bias, mean=0.0, std=0.01)
    permutation = np.asarray([3, 0, 7, 2, 1, 6, 5, 4], dtype=np.int64)
    transformed = exact_gauge_clone(model, permutation).eval()
    gauge = identity_alignment()
    gauge["Z"] = np.eye(LATENT_CHANNELS, dtype=np.float64)[permutation]

    generator = torch.Generator(device="cpu").manual_seed(47)
    x = torch.randn((2, 3, 8, 8), generator=generator)
    mask = torch.ones((2, 1, 8, 8), dtype=torch.float32)
    mask[:, :, 0, :] = 0
    mask[:, :, 4, 3] = 0
    x[:, :1] = mask
    state = model.initial(x)
    transformed_state = transformed.initial(x)
    max_state_error = 0.0
    max_output_error = 0.0
    with torch.no_grad():
        for _ in range(8):
            state = model.step(state, x)
            transformed_state = transformed.step(transformed_state, x)
            expected = apply_alignment(state, gauge)
            max_state_error = max(
                max_state_error,
                max(float((a - b).abs().max()) for a, b in zip(expected, transformed_state)),
            )
            max_output_error = max(
                max_output_error,
                float((model.logits(state) - transformed.logits(transformed_state)).abs().max()),
            )
    if max(max_state_error, max_output_error) > 1e-5:
        raise AssertionError(
            f"gauge mismatch: state={max_state_error:.3g}, output={max_output_error:.3g}"
        )
    return {
        "passed": True,
        "steps": 8,
        "batch": 2,
        "spatial_shape": [8, 8],
        "max_abs_state_error": max_state_error,
        "max_abs_output_error": max_output_error,
        "atol": 1e-5,
    }
