"""One small CPU check for masked audit pulses and cohort summaries."""
from __future__ import annotations

import torch
from torch import nn

from instrument import parts, semantic_decomposition
from probes import PULSE_KINDS, pulse, summary
from stream_cells import StreamingCell


def _assert_equal(actual, expected, name: str) -> None:
    if not torch.equal(actual, expected):
        difference = (actual - expected).abs().max().item()
        raise AssertionError(f"{name}: max absolute difference {difference:.3g}")


def _assert_finite(value, name: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{name}[{index}]")
    elif isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise AssertionError(f"{name} contains nonfinite values")
    elif isinstance(value, float):
        if not torch.isfinite(torch.tensor(value)):
            raise AssertionError(f"{name} is nonfinite")


def main() -> None:
    torch.manual_seed(195201)
    model = StreamingCell(
        workspace_channels=8,
        latent_channels=4,
        workspace_hidden=11,
        candidate_hidden=7,
    ).cpu()
    model.eval()
    with torch.no_grad():
        for layer in (model.f_out, model.q_out, model.readout):
            nn.init.normal_(layer.weight, mean=0.0, std=0.08)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=0.02)

    batch, height, width = 1, 5, 6
    x = torch.rand(batch, 3, height, width)
    medium = torch.ones(batch, 1, height, width)
    medium[0, 0, 2, 2] = 0.0
    medium[0, 0, 1, 4] = 0.0
    x[:, :1] = medium
    state = (
        torch.randn(batch, model.workspace_channels, height, width),
        torch.randn(batch, model.latent_channels, height, width),
    )
    selected = torch.zeros(batch, 1, height, width, dtype=torch.bool)
    selected[:, :, 1:4, 1:5] = True
    selected &= medium.bool()

    baseline = model.step(state, x)
    empty = torch.zeros_like(selected)
    for kind in PULSE_KINDS:
        altered = pulse(model, state, x, selected, kind)
        _assert_finite(altered, kind)
        no_op = pulse(model, state, x, empty, kind)
        _assert_equal(no_op[0], baseline[0], f"empty-mask W, {kind}")
        _assert_equal(no_op[1], baseline[1], f"empty-mask Z, {kind}")

    drop_q = pulse(model, state, x, selected, "drop_Q")
    expected_drop_q_z = torch.where(selected, state[1], baseline[1])
    _assert_equal(drop_q[0], baseline[0], "drop_Q workspace")
    _assert_equal(drop_q[1], expected_drop_q_z, "drop_Q latent")

    # The second hop sees the locally altered workspace and unmodified neighbors.
    step_parts = parts(model, state, x)
    expected_drop_f_w = torch.where(selected, step_parts["incoming"], baseline[0])
    expected_drop_f_q = model._candidate(expected_drop_f_w, state[1], x)
    expected_drop_f_z = torch.where(
        selected, state[1] + model.alpha * expected_drop_f_q, baseline[1]
    )
    drop_f = pulse(model, state, x, selected, "drop_F")
    _assert_equal(drop_f[0], expected_drop_f_w, "drop_F mixed workspace")
    _assert_equal(drop_f[1], expected_drop_f_z, "drop_F mixed-neighbor Q")

    old_features = model._features(state[0], state[1], x)
    identity_force = model.f_out(torch.tanh(model.f_in(old_features)))
    expected_identity_w = torch.where(
        selected, state[0] + model.eta * identity_force, baseline[0]
    )
    expected_identity_q = model._candidate(expected_identity_w, state[1], x)
    expected_identity_z = torch.where(
        selected, state[1] + model.alpha * expected_identity_q, baseline[1]
    )
    identity = pulse(model, state, x, selected, "identity_stream")
    _assert_equal(identity[0], expected_identity_w, "identity_stream mixed workspace")
    _assert_equal(identity[1], expected_identity_z, "identity_stream mixed-neighbor Q")

    semantic = semantic_decomposition(model, state, x, step_parts)
    _assert_equal(
        torch.isclose(
            semantic["dlogits_sum"],
            semantic["dlogits"],
            atol=1e-6,
            rtol=1e-6,
        ),
        torch.ones_like(semantic["dlogits"], dtype=torch.bool),
        "semantic telescope",
    )

    solved = selected.clone()
    solved[:, :, 1, :] = False
    frontier = selected & ~solved
    labels = torch.ones(batch, 1, height, width)
    labels[solved] = 0.0
    report = summary(
        model,
        state,
        x,
        {"solved": solved, "frontier": frontier},
        labels,
    )
    _assert_finite(report, "summary")
    if set(report) != {"solved", "frontier"}:
        raise AssertionError("summary must return one entry per cohort")
    for cohort in report.values():
        if set(cohort["feature_knockouts"]) != {
            "Q_without_W", "Q_without_Z", "Q_without_LW", "Q_without_LZ", "Q_without_X"
        }:
            raise AssertionError("summary is missing a feature knockout")
    solved_summary = report["solved"]
    margin_mean = solved_summary["semantic_margin"]["dlogits"]["mean"]
    raw_mean = solved_summary["semantic_raw"]["dlogits"]["mean"]
    if abs(margin_mean + raw_mean) > 1e-6:
        raise AssertionError("negative labels must flip the signed logit-margin delta")
    if "X" not in solved_summary["q_in_preactivation"]:
        raise AssertionError("q_in preactivation summary is missing the X contribution")
    if "bias is included once in X" not in solved_summary["q_in_preactivation_note"]:
        raise AssertionError("q_in bias-on-X note is missing")

    print("audit_195_200_probes: PASS (8 pulse kinds, empty masks, drop_Q, summary)")


if __name__ == "__main__":
    main()
