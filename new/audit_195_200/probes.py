"""Single-step masked probes and summaries for the 195/200 audit.

Probes alter only the requested cells of one next state. They are accounting
interventions for one supplied input; compare separate worlds at the call site.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from instrument import parts, preactivation_parts, semantic_decomposition


PULSE_KINDS = (
    "drop_F",
    "drop_Q",
    "identity_stream",
    "Q_without_W",
    "Q_without_Z",
    "Q_without_LW",
    "Q_without_LZ",
    "Q_without_X",
)
Q_BLOCKS = {
    "Q_without_W": "W",
    "Q_without_Z": "Z",
    "Q_without_LW": "LW",
    "Q_without_LZ": "LZ",
    "Q_without_X": "X",
}


def _validate_mask(mask: Tensor, x: Tensor) -> Tensor:
    expected = (x.shape[0], 1, *x.shape[-2:])
    if tuple(mask.shape) != expected:
        raise ValueError(f"mask must have shape {expected}, got {tuple(mask.shape)}")
    return mask.to(device=x.device, dtype=torch.bool)


def _blend(original, altered, mask: Tensor):
    return tuple(torch.where(mask, changed, base) for base, changed in zip(original, altered))


def _zero_q_block(model, candidate_features: Tensor, block_name: str) -> Tensor:
    widths = (
        model.workspace_channels,
        model.latent_channels,
        model.workspace_channels,
        model.latent_channels,
        candidate_features.shape[1]
        - 2 * model.workspace_channels
        - 2 * model.latent_channels,
    )
    names = ("W", "Z", "LW", "LZ", "X")
    start = 0
    blocks = []
    for name, width in zip(names, widths):
        end = start + width
        block = candidate_features[:, start:end]
        blocks.append(torch.zeros_like(block) if name == block_name else block)
        start = end
    if start != candidate_features.shape[1]:
        raise ValueError("Candidate feature widths do not match the model")
    # q_in is applied after this zeroing, so its learned bias remains present.
    return model.q_out(torch.tanh(model.q_in(torch.cat(blocks, dim=1))))


def _pulse_from_base(model, state, x: Tensor, mask: Tensor, kind: str,
                     original, step_parts, candidate_features=None):
    w, z = state
    if kind == "drop_F":
        # Remove F only on selected cells before the second-hop neighborhood read.
        w_altered = torch.where(mask, step_parts["incoming"], original[0])
        q_altered = model._candidate(w_altered, z, x)
        z_altered = z + model.alpha * q_altered
    elif kind == "drop_Q":
        w_altered, z_altered = original[0], z
    elif kind == "identity_stream":
        # Use the pre-transport state and its already-computed LW/LZ features.
        force_features = torch.cat(
            (w, z, step_parts["LW"], step_parts["LZ"], x), dim=1
        )
        force = model.f_out(torch.tanh(model.f_in(force_features)))
        identity_w = w + model.eta * force
        # Keep the unmodified one-step workspace outside the intervention mask.
        w_altered = torch.where(mask, identity_w, original[0])
        q_altered = model._candidate(w_altered, z, x)
        z_altered = z + model.alpha * q_altered
    elif kind in Q_BLOCKS:
        if candidate_features is None:
            candidate_features = model._features(original[0], z, x)
        q_altered = _zero_q_block(model, candidate_features, Q_BLOCKS[kind])
        w_altered = original[0]
        z_altered = z + model.alpha * q_altered
    else:
        raise ValueError(f"Unknown pulse kind: {kind}")
    return _blend(original, (w_altered, z_altered), mask)


def pulse(model, state, x: Tensor, mask: Tensor, kind: str):
    """Apply one named intervention on mask cells, with the original step outside.

    Q_without_* zeros exactly one post-F candidate input block before q_in;
    X denotes all input channels and q_in's bias is retained. The tanh means
    block ablations do not form additive semantic attributions.
    """
    model._validate_x(x)
    if kind not in PULSE_KINDS:
        raise ValueError(f"Unknown pulse kind: {kind}")
    selected = _validate_mask(mask, x)
    original = model.step(state, x)
    step_parts = parts(model, state, x)
    candidate_features = (
        model._features(original[0], state[1], x) if kind in Q_BLOCKS else None
    )
    return _pulse_from_base(
        model, state, x, selected, kind, original, step_parts, candidate_features
    )


def _scalar_field(value: Tensor) -> Tensor:
    if value.ndim != 4:
        raise ValueError("Summary tensors must be BCHW")
    if value.shape[1] == 1:
        return value[:, 0]
    return value.square().mean(dim=1).sqrt()


def _magnitude_stats(field: Tensor, selected: Tensor) -> dict[str, float | int | None]:
    values = field[selected]
    if values.numel() == 0:
        return {"count": 0, "mean": None, "rms": None, "max": None}
    return {
        "count": int(values.numel()),
        "mean": values.mean().item(),
        "rms": values.square().mean().sqrt().item(),
        "max": values.max().item(),
    }


def _signed_stats(field: Tensor, selected: Tensor) -> dict[str, float | int | None]:
    values = field[selected]
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "rms": None,
            "max": None,
            "max_abs": None,
        }
    return {
        "count": int(values.numel()),
        "mean": values.mean().item(),
        "rms": values.square().mean().sqrt().item(),
        "max": values.max().item(),
        "max_abs": values.abs().max().item(),
    }


def _per_map_delta(delta: Tensor, selected: Tensor) -> list[dict[str, float | int | None]]:
    output = []
    for index in range(delta.shape[0]):
        values = delta[index][selected[index]]
        if values.numel() == 0:
            output.append({"map_index": index, "count": 0,
                           "fraction_negative": None,
                           "q05": None, "q50": None, "q95": None})
            continue
        quantiles = torch.quantile(
            values, values.new_tensor((0.05, 0.5, 0.95))
        )
        output.append({
            "map_index": index,
            "count": int(values.numel()),
            "fraction_negative": (values < 0).float().mean().item(),
            "q05": quantiles[0].item(),
            "q50": quantiles[1].item(),
            "q95": quantiles[2].item(),
        })
    return output


def _correctness_metrics(before: Tensor, after: Tensor, labels: Tensor,
                         selected: Tensor) -> dict[str, float | int | None]:
    label = labels >= 0.5
    before_correct = (before >= 0) == label
    after_correct = (after >= 0) == label
    baseline_correct = selected & before_correct
    denominator = int(baseline_correct.sum().item())
    selected_count = int(selected.sum().item())
    crossed = baseline_correct & ~after_correct
    return {
        "count": selected_count,
        "baseline_correct_count": denominator,
        "baseline_correct_fraction": (
            denominator / selected_count if selected_count else None
        ),
        "conditional_correctness_on_baseline_correct": (
            (baseline_correct & after_correct).sum().item() / denominator
            if denominator else None
        ),
        "margin_crossing_rate_on_baseline_correct": (
            crossed.sum().item() / denominator if denominator else None
        ),
    }


@torch.no_grad()
def summary(model, state, x: Tensor, cohorts: Mapping[str, Tensor], labels: Tensor):
    """Summarize one supplied input by cohort and one-step feature knockout.

    State-channel parts are reduced to per-pixel channel RMS, then summarized
    over each cohort. Semantic deltas are label-signed margin changes; raw logit
    deltas are also retained. Knockout scores use fixed caller labels and
    condition on cells correct after the unmodified step. This describes one
    world only; callers compare original/flipped worlds.
    """
    model._validate_x(x)
    if not hasattr(cohorts, "items"):
        raise TypeError("cohorts must map names to [B,1,H,W] masks")

    original = model.step(state, x)
    step_parts = parts(model, state, x)
    semantic = semantic_decomposition(model, state, x, step_parts)
    before_logits = model.logits(original)
    if before_logits.ndim != 4 or before_logits.shape[1] != 1:
        raise ValueError("summary expects a one-channel binary logit map")
    if labels.ndim == 3:
        labels = labels.unsqueeze(1)
    if tuple(labels.shape) != tuple(before_logits.shape):
        raise ValueError("labels must match the [B,1,H,W] logit shape")
    labels = labels.to(device=before_logits.device)

    w, z = state
    candidate_features = model._features(original[0], z, x)
    part_tensors = {
        "incoming": step_parts["incoming"],
        "LW": step_parts["LW"],
        "LZ": step_parts["LZ"],
        "F": step_parts["F"],
        "Q": step_parts["Q"],
        "learned_dW": step_parts["dW"],
        "total_dW": original[0] - w,
        "dZ": step_parts["dZ"],
    }
    part_fields = {name: _scalar_field(value) for name, value in part_tensors.items()}
    raw_semantic_fields = {
        name: semantic[name][:, 0]
        for name in (
            "dlogits",
            "dlogits_baseline",
            "dlogits_transport",
            "dlogits_F",
        )
    }
    label_sign = labels[:, 0].to(dtype=semantic["dlogits"].dtype) * 2 - 1
    semantic_fields = {
        name: value * label_sign for name, value in raw_semantic_fields.items()
    }
    q_preactivation_parts = preactivation_parts(model, original[0], z, x)
    q_preactivation_fields = {
        name: _scalar_field(value)
        for name, value in q_preactivation_parts.items()
    }
    q_kinds = tuple(Q_BLOCKS)
    output = {}
    for name, raw_mask in cohorts.items():
        selected = _validate_mask(raw_mask, x)[:, 0]
        part_stats = {
            component: _magnitude_stats(field, selected)
            for component, field in part_fields.items()
        }
        semantic_stats = {
            component: _signed_stats(field, selected)
            for component, field in semantic_fields.items()
        }
        raw_semantic_stats = {
            component: _signed_stats(field, selected)
            for component, field in raw_semantic_fields.items()
        }
        q_preactivation_stats = {
            component: _magnitude_stats(field, selected)
            for component, field in q_preactivation_fields.items()
        }
        knockout_stats = {}
        for kind in q_kinds:
            altered = _pulse_from_base(
                model,
                state,
                x,
                selected[:, None],
                kind,
                original,
                step_parts,
                candidate_features,
            )
            after_logits = model.logits(altered)
            knockout_stats[kind] = _correctness_metrics(
                before_logits[:, 0], after_logits[:, 0], labels[:, 0], selected
            )

        output[name] = {
            "count": int(selected.sum().item()),
            "parts": part_stats,
            "semantic": semantic_stats,
            "semantic_margin": semantic_stats,
            "semantic_raw": raw_semantic_stats,
            "q_in_preactivation": q_preactivation_stats,
            "q_in_preactivation_note": (
                "q_in bias is included once in X; these are pre-tanh pieces, "
                "not additive semantic attributions after tanh."
            ),
            "dmargin_per_map": _per_map_delta(semantic_fields["dlogits"], selected),
            "dlogit_per_map": _per_map_delta(raw_semantic_fields["dlogits"], selected),
            "feature_knockouts": knockout_stats,
        }
    return output
