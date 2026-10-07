"""CPU regression checks for relation-activity telemetry precision and rows."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from activity import gradient_norm, rows, sufficient  # noqa: E402


def _inputs(height: int = 2, width: int = 3):
    generator = torch.Generator(device="cpu").manual_seed(731)
    incoming = torch.randn((2, 24, height, width), generator=generator)
    delta = torch.randn((2, 24, height, width), generator=generator)
    conditioned = torch.randn((2, 24, height, width), generator=generator)
    x = torch.zeros((2, 3, height, width), dtype=torch.float32)
    x[:, 0] = torch.tensor(
        [[[1, 0, 1], [0, 1, 1]], [[1, 1, 0], [1, 0, 1]]], dtype=torch.float32
    )[:, :height, :width]
    return incoming, delta, conditioned, x


def _float64_reference(model, incoming, delta, conditioned, x):
    u = incoming.detach().to(torch.float64).reshape(
        incoming.shape[0], 4, 6, *incoming.shape[2:]
    )
    d = (delta.detach().to(torch.float64) * float(model.eta)).reshape_as(u)
    c = (conditioned.detach().to(torch.float64) * float(model.eta)).reshape_as(u)
    opened = x[:, :1].bool().expand(u.shape[0], 4, *u.shape[-2:])
    axes = (0, 2, 3)

    def energy(value):
        return torch.where(opened, value.square().sum(2), 0.0).sum(axes)

    return torch.stack(
        (opened.sum(axes, dtype=torch.float64), energy(u), energy(d), energy(c)),
        dim=1,
    )


def _old_float32_reference(model, incoming, delta, conditioned, x):
    u = incoming.detach().reshape(incoming.shape[0], 4, 6, *incoming.shape[2:])
    d = (model.eta * delta.detach()).reshape_as(u)
    c = (model.eta * conditioned.detach()).reshape_as(u)
    opened = x[:, :1].bool().expand(u.shape[0], 4, *u.shape[-2:])
    axes = (0, 2, 3)
    return torch.stack(
        (
            opened.sum(axes, dtype=torch.float64),
            (u.square().sum(2) * opened).sum(axes, dtype=torch.float64),
            (d.square().sum(2) * opened).sum(axes, dtype=torch.float64),
            (c.square().sum(2) * opened).sum(axes, dtype=torch.float64),
        ),
        dim=1,
    )


def check_ordinary_values_and_no_mutation() -> None:
    model = SimpleNamespace(arm="conditioned", eta=0.1)
    incoming, delta, conditioned, x = _inputs()
    inputs = (incoming, delta, conditioned)
    snapshots = [value.clone() for value in inputs]
    for value in inputs:
        value.requires_grad_()

    actual = sufficient(model, incoming, delta, conditioned, x)
    expected = _float64_reference(model, incoming, delta, conditioned, x)
    old = _old_float32_reference(model, incoming, delta, conditioned, x)
    assert torch.equal(actual, expected), "ordinary stats differ from exact float64 reference"
    torch.testing.assert_close(actual, old, atol=2e-7, rtol=2e-7)
    assert actual.requires_grad is False
    for value, snapshot in zip(inputs, snapshots):
        assert value.grad is None
        assert torch.equal(value, snapshot), "telemetry mutated an input"
    json.dumps(rows(actual), allow_nan=False)


def check_float32_max_and_closed_mask() -> None:
    model = SimpleNamespace(arm="conditioned", eta=0.1)
    maximum = torch.finfo(torch.float32).max
    incoming = torch.full((1, 24, 1, 2), maximum, dtype=torch.float32)
    delta = torch.full_like(incoming, maximum)
    conditioned = torch.full_like(incoming, maximum)
    x = torch.zeros((1, 3, 1, 2), dtype=torch.float32)
    x[:, 0, 0, 0] = 1

    actual = sufficient(model, incoming, delta, conditioned, x)
    expected = _float64_reference(model, incoming, delta, conditioned, x)
    assert torch.equal(actual, expected), "large stats differ from exact float64 reference"
    assert torch.isfinite(actual).all(), "finite float32 maxima overflowed telemetry"
    max64 = torch.tensor(maximum, dtype=torch.float64)
    assert actual[0, 1].item() == (max64.square() * 6).item()
    assert actual[0, 2].item() == ((max64 * model.eta).square() * 6).item()
    assert actual[0, 3].item() == ((max64 * model.eta).square() * 6).item()
    assert actual[:, 0].tolist() == [1.0, 1.0, 1.0, 1.0]
    json.dumps(rows(actual), allow_nan=False)


def check_nonfinite_diagnostics_and_missing_zero_arms() -> None:
    model = SimpleNamespace(arm="conditioned", eta=0.1)
    incoming = torch.ones((1, 24, 1, 2), dtype=torch.float32)
    delta = torch.full_like(incoming, 2.0)
    conditioned = torch.full_like(incoming, 3.0)
    x = torch.zeros((1, 3, 1, 2), dtype=torch.float32)
    x[:, 0, 0, 0] = 1
    incoming[0, 0, 0, 0] = math.inf  # open incoming value: must stay visible
    delta[0, 6, 0, 0] = math.inf  # open lane 1: must stay visible
    conditioned[0, 12, 0, 0] = math.nan  # open lane 2: must stay visible
    incoming[0, 1, 0, 1] = math.nan  # closed values are excluded
    delta[0, 7, 0, 1] = math.inf
    conditioned[0, 13, 0, 1] = math.nan

    actual = sufficient(model, incoming, delta, conditioned, x)
    rows_out = rows(actual)
    assert math.isinf(actual[0, 1].item())
    assert math.isinf(actual[1, 2].item())
    assert math.isnan(actual[2, 3].item())
    assert math.isfinite(actual[1, 1].item())
    assert math.isfinite(actual[0, 2].item())
    assert math.isfinite(actual[0, 3].item())
    assert rows_out[0]["incoming_energy"] is None
    assert rows_out[0]["incoming_energy_nonfinite"] is True
    assert rows_out[1]["scaled_relation_write_energy"] is None
    assert rows_out[1]["scaled_relation_write_energy_nonfinite"] is True
    assert rows_out[2]["scaled_state_conditioned_write_energy"] is None
    assert rows_out[2]["scaled_state_conditioned_write_energy_nonfinite"] is True
    assert rows_out[3]["incoming_energy_nonfinite"] is False
    json.dumps(rows_out, allow_nan=False)

    # Current and constant arms provide exact zero tensors for absent actions.
    # Their reduced rows remain zero while avoiding full-map conversions.
    zero = torch.zeros_like(incoming)
    regular_incoming = torch.ones_like(incoming)
    for arm in ("current", "constant"):
        reduced = sufficient(
            SimpleNamespace(arm=arm, eta=0.1),
            regular_incoming,
            zero,
            zero,
            x,
        )
        missing_index = 2 if arm == "current" else 3
        assert torch.equal(reduced[:, missing_index], torch.zeros_like(reduced[:, missing_index]))


def check_large_gradient_norm() -> None:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relation_weight = torch.nn.Parameter(torch.ones(1))

    model = Model()
    maximum = torch.finfo(torch.float32).max
    model.relation_weight.grad = torch.tensor([maximum], dtype=torch.float32)
    before_parameter = model.relation_weight.detach().clone()
    before_gradient = model.relation_weight.grad.clone()
    actual = gradient_norm(model)
    expected = float(torch.tensor(maximum, dtype=torch.float64))
    assert math.isfinite(actual) and actual == expected
    assert torch.equal(model.relation_weight, before_parameter)
    assert torch.equal(model.relation_weight.grad, before_gradient)


def main() -> None:
    check_ordinary_values_and_no_mutation()
    check_float32_max_and_closed_mask()
    check_nonfinite_diagnostics_and_missing_zero_arms()
    check_large_gradient_norm()
    print("activity telemetry CPU checks passed")


if __name__ == "__main__":
    main()
