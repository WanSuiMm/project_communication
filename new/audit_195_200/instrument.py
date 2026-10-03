"""Tensor instrumentation for the zero-training 195/200 mechanism audit.

The returned components follow StreamingCell.step's original two-hop order.
They support accounting checks and do not establish a causal mechanism.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


STREAMING_DIR = Path(__file__).resolve().parents[1] / "streaming_carry"
if str(STREAMING_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMING_DIR))

from stream_cells import stream  # noqa: E402


def _feature_blocks(model, w: Tensor, z: Tensor, x: Tensor):
    features = model._features(w, z, x)
    workspace_channels = model.workspace_channels
    latent_channels = model.latent_channels
    lw_start = workspace_channels + latent_channels
    lz_start = 2 * workspace_channels + latent_channels
    x_start = 2 * workspace_channels + 2 * latent_channels
    blocks = {
        "W": features[:, :workspace_channels],
        "Z": features[:, workspace_channels:lw_start],
        "LW": features[:, lw_start:lz_start],
        "LZ": features[:, lz_start:x_start],
        "X": features[:, x_start:],
    }
    return features, blocks


def parts(model, state, x: Tensor) -> dict[str, Tensor]:
    """Return named one-step tensors in the same order as model.step.

    dW is the learned residual increment after incoming transport; dZ is the
    additive latent increment. Defaults are eta=0.1 and alpha=0.5.
    """
    model._validate_x(x)
    w, z = state
    old_features, feature_blocks = _feature_blocks(model, w, z, x)

    incoming = stream(w, x[:, :1]) if model.streaming else w
    # Preserve the original clock: LW and LZ are computed before transport.
    force_features = torch.cat(
        (incoming, old_features[:, model.workspace_channels :]), dim=1
    )
    force = model.f_out(torch.tanh(model.f_in(force_features)))
    d_w = model.eta * force
    w_new = incoming + d_w

    # The candidate is the second communication phase and sees W_new.
    q = model._candidate(w_new, z, x)
    d_z = model.alpha * q
    z_new = z + d_z

    return {
        "incoming": incoming,
        "LW": feature_blocks["LW"],
        "LZ": feature_blocks["LZ"],
        "F": force,
        "W_new": w_new,
        "Q": q,
        "Z_new": z_new,
        "dW": d_w,
        "dZ": d_z,
    }


def _readout_weight(model, value: Tensor) -> Tensor:
    """Apply the existing linear readout weights without its affine bias."""
    readout = model.readout
    input_value = value
    padding = readout.padding
    if readout.padding_mode != "zeros":
        input_value = F.pad(
            input_value,
            readout._reversed_padding_repeated_twice,
            mode=readout.padding_mode,
        )
        padding = 0
    return F.conv2d(
        input_value,
        readout.weight,
        bias=None,
        stride=readout.stride,
        padding=padding,
        dilation=readout.dilation,
        groups=readout.groups,
    )


def semantic_decomposition(
    model, state, x: Tensor, parts_dict: Mapping[str, Tensor] | None = None
) -> dict[str, Tensor]:
    """Return the exact logit delta and an order-dependent telescoping split.

    The split follows baseline Q(W,Z), then transport Q(TW,Z), then the F-updated
    workspace Q(W_new,Z). It uses this model's readout; labels remain caller-owned.
    The terms are mediation accounting, not independent interventions or causal
    effects. Their magnitudes alone imply no gate or mechanism claim.
    """
    model._validate_x(x)
    w, z = state
    step_parts = parts(model, state, x) if parts_dict is None else parts_dict
    q_base = model._candidate(w, z, x)
    q_transport = model._candidate(step_parts["incoming"], z, x)
    q_force = model._candidate(step_parts["W_new"], z, x)

    dlogits = model.alpha * _readout_weight(model, step_parts["Q"])
    dlogits_transport = model.alpha * _readout_weight(
        model, q_transport - q_base
    )
    dlogits_f = model.alpha * _readout_weight(model, q_force - q_transport)
    dlogits_baseline = model.alpha * _readout_weight(model, q_base)
    dlogits_sum = dlogits_transport + dlogits_f + dlogits_baseline
    return {
        "dlogits": dlogits,
        "dlogits_transport": dlogits_transport,
        "dlogits_F": dlogits_f,
        "dlogits_baseline": dlogits_baseline,
        "dlogits_sum": dlogits_sum,
    }


def preactivation_parts(model, w: Tensor, z: Tensor, x: Tensor) -> dict[str, Tensor]:
    """Split q_in preactivation by W, Z, LW, LZ, and X input blocks.

    The Conv2d bias is added once to X. These terms sum before tanh; tanh makes
    them nonadditive after activation, so they are not semantic attributions.
    """
    model._validate_x(x)
    layer = model.q_in
    if layer.kernel_size != (1, 1) or layer.groups != 1:
        raise ValueError("q_in must be an ungrouped 1x1 Conv2d for block splitting")
    if layer.padding_mode != "zeros":
        raise ValueError("q_in must use zero padding for block splitting")

    _, blocks = _feature_blocks(model, w, z, x)
    result: dict[str, Tensor] = {}
    offset = 0
    for name in ("W", "Z", "LW", "LZ", "X"):
        block = blocks[name]
        end = offset + block.shape[1]
        result[name] = F.conv2d(
            block,
            layer.weight[:, offset:end],
            bias=None,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )
        offset = end
    if layer.bias is not None:
        result["X"] = result["X"] + layer.bias[None, :, None, None]
    return result
