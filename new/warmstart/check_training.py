"""Focused CPU checks for the warm-start K8 training helper."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import torch


_TRAINING_PATH = Path(__file__).with_name("training.py")
_TRAINING_SPEC = importlib.util.spec_from_file_location("warmstart_training", _TRAINING_PATH)
if _TRAINING_SPEC is None or _TRAINING_SPEC.loader is None:
    raise ImportError(f"Cannot load warm-start training helper from {_TRAINING_PATH}")
training = importlib.util.module_from_spec(_TRAINING_SPEC)
sys.modules["warmstart_training"] = training
_TRAINING_SPEC.loader.exec_module(training)

from stream_cells import make_variant  # noqa: E402
from tasks import bank  # noqa: E402


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _same_state(left, right) -> bool:
    return all(torch.equal(a, b) for a, b in zip(left, right))


def _first_examples(data, count=4):
    return {key: value[:count] for key, value in data.items()}


def _finite_gradients(model) -> bool:
    return all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters())


def _parameters_unchanged(model, snapshot) -> bool:
    current = model.state_dict()
    return all(torch.equal(snapshot[name], value) for name, value in current.items())


def _make_active_streaming_cell():
    model = make_variant("stream")
    # Activate the update paths so the encoder and both temporal state blocks
    # participate in a nontrivial gradient comparison.
    with torch.no_grad():
        model.f_out.weight.normal_(std=0.03)
        model.q_out.weight.normal_(std=0.03)
    return model


class _TransientRecoveryCell:
    """Tiny fixture that produces one nonfinite state, then recovers."""

    def __init__(self):
        self.steps = 0
        self.initial_calls = 0

    def initial(self, x):
        self.initial_calls += 1
        return (torch.zeros_like(x[:, :1]),)

    def step(self, state, _x):
        self.steps += 1
        if self.steps == 3:
            return (torch.full_like(state[0], float("nan")),)
        return (torch.zeros_like(state[0]),)


def _assert_encoder_is_fresh_half_only(model, data, observed):
    fresh_model = copy.deepcopy(model)
    subset = _first_examples(data)
    training._HISTORICAL.backward_trajectory(
        fresh_model, subset, 8, steps=64, loss_every=8, budget=lambda: None
    )
    for name in ("encoder.weight", "encoder.bias"):
        actual = dict(observed.named_parameters())[name].grad
        expected = dict(fresh_model.named_parameters())[name].grad / 2
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def run_checks() -> dict:
    """Run a small set of deterministic CPU checks and return a PASS record."""
    old_threads = torch.get_num_threads()
    old_rng = torch.random.get_rng_state()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(1847)
        data = bank(8, 8, 1853, device="cpu")
        source = _make_active_streaming_cell()
        initial_parameters = {name: value.detach().clone()
                              for name, value in source.state_dict().items()}
        checks = []

        # The discarded-prefix baseline performs the requested no-grad prefix,
        # then preserves the original K8 losses, final state, and every gradient.
        baseline = copy.deepcopy(source)
        reference = copy.deepcopy(source)
        loss, final_state, trace = training.backward_warmstart(
            baseline, data, age=32, keep_warm_state=False, budget=lambda: None
        )
        ref_loss, ref_state, ref_trace = training._HISTORICAL.backward_trajectory(
            reference, data, 8, steps=64, loss_every=8, budget=lambda: None
        )
        _check(torch.equal(loss, ref_loss), "discarded-prefix loss differs from historical K8")
        _check(_same_state(final_state, ref_state),
               "discarded-prefix final state differs from historical K8")
        _check(trace["backward_calls"] == 8 and trace["interior_detach_boundaries"] == 7,
               "discarded-prefix K8 backward cadence changed")
        _check(trace["prefix_age"] == 32 and trace["prefix_examples"] == 4
               and trace["prefix_state_finite"] and not trace["warm_state_kept"],
               "discarded-prefix metadata is incomplete")
        _check(_parameters_unchanged(baseline, initial_parameters),
               "discarded-prefix helper changed model parameters")
        for (name, actual), (_, expected) in zip(baseline.named_parameters(),
                                                 reference.named_parameters()):
            _check(torch.equal(actual.grad, expected.grad),
                   f"discarded-prefix gradient differs for {name}")
        checks.append("discarded_prefix_matches_historical_loss_state_and_all_gradients")

        # At age zero the warm half begins from the original numerical state,
        # while its detached encoder path is deliberately absent from gradients.
        warm_zero = copy.deepcopy(source)
        full_zero = copy.deepcopy(source)
        zero_loss, zero_state, zero_trace = training.backward_warmstart(
            warm_zero, data, age=0, keep_warm_state=True, budget=lambda: None
        )
        full_loss, full_state, _ = training._HISTORICAL.backward_trajectory(
            full_zero, data, 8, steps=64, loss_every=8, budget=lambda: None
        )
        _check(torch.equal(zero_loss, full_loss), "warm age-zero loss differs from fresh K8")
        _check(_same_state(zero_state, full_state),
               "warm age-zero final state differs from fresh K8")
        _check(zero_trace["encoder_fresh_rows"] == 4 and zero_trace["finite_prefix"],
               "warm age-zero metadata does not describe the detached half-batch")
        _assert_encoder_is_fresh_half_only(source, data, warm_zero)
        for (name, actual), (_, expected) in zip(warm_zero.named_parameters(),
                                                 full_zero.named_parameters()):
            if not name.startswith("encoder."):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)
        _check(any(not torch.equal(dict(warm_zero.named_parameters())[name].grad,
                                   dict(full_zero.named_parameters())[name].grad)
                   for name in ("encoder.weight", "encoder.bias")),
               "warm age-zero encoder gradients were not cut for the warm half")
        checks.append("warm_age_zero_matches_forward_values_and_cuts_warm_encoder_gradient")

        # A nonzero prefix must remain finite, detached, and unable to contribute
        # encoder gradients; the four fresh examples still train the encoder.
        warm32 = copy.deepcopy(source)
        warm_snapshot = {name: value.detach().clone()
                         for name, value in warm32.state_dict().items()}
        warm_loss, warm_state, warm_trace = training.backward_warmstart(
            warm32, data, age=32, keep_warm_state=True, budget=lambda: None
        )
        _check(bool(torch.isfinite(warm_loss)), "warm age-32 loss is nonfinite")
        _check(all(bool(torch.isfinite(value).all()) and not value.requires_grad
                   and value.grad_fn is None for value in warm_state),
               "warm age-32 returned state is nonfinite or attached")
        _check(_finite_gradients(warm32), "warm age-32 gradients are missing or nonfinite")
        _check(warm_trace["backward_calls"] == 8
               and warm_trace["interior_detach_boundaries"] == 7
               and warm_trace["forward_steps"] == 64 and warm_trace["loss_count"] == 8,
               "warm age-32 K8 cadence changed")
        _check(warm_trace["gradient_horizon"] == 8
               and warm_trace["prefix_age"] == 32
               and warm_trace["prefix_examples"] == 4
               and warm_trace["encoder_fresh_rows"] == 4
               and warm_trace["prefix_state_finite"],
               "warm age-32 metadata is incomplete")
        _assert_encoder_is_fresh_half_only(source, data, warm32)
        _check(_parameters_unchanged(warm32, warm_snapshot),
               "warm age-32 helper changed model parameters")
        checks.append("warm_age_32_has_finite_gradients_and_only_fresh_encoder_rows")

        transient = _TransientRecoveryCell()
        recovered = False
        try:
            training.backward_warmstart(
                transient, data, age=32, keep_warm_state=True, budget=lambda: None
            )
        except FloatingPointError as error:
            recovered = "Nonfinite warm prefix state" in str(error)
        _check(recovered and transient.steps == 8 and transient.initial_calls == 1,
               "transient nonfinite prefix was forgotten or training continued past its first window")
        checks.append("transient_nonfinite_prefix_remains_sticky_until_eight_step_check")

        bad_data = dict(data)
        bad_data["x"] = data["x"].clone()
        bad_data["x"][4, 0, 0, 0] = float("nan")
        rejected = False
        try:
            training.backward_warmstart(
                copy.deepcopy(source), bad_data, age=32, keep_warm_state=True,
                budget=lambda: None,
            )
        except FloatingPointError as error:
            rejected = "Nonfinite warm prefix state" in str(error)
        _check(rejected, "nonfinite warm-prefix state was not rejected")
        checks.append("nonfinite_prefix_rejected_before_training")

        return {"status": "PASS", "checks": checks}
    finally:
        torch.random.set_rng_state(old_rng)
        torch.set_num_threads(old_threads)


def main() -> None:
    print(json.dumps(run_checks(), indent=2))


if __name__ == "__main__":
    main()
