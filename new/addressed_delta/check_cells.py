"""CPU integration checks for the addressed-delta carrier comparison."""
from __future__ import annotations

import json

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cells import ARMS, COUNTS, AddressedDeltaCell, CurrentCell, make_model
from stream_cells import StreamingCell, stream
from writer import addressed_write


def _same_state(left: nn.Module, right: nn.Module) -> None:
    left_state, right_state = left.state_dict(), right.state_dict()
    assert tuple(left_state) == tuple(right_state)
    for name in left_state:
        assert torch.equal(left_state[name], right_state[name]), name


def _max_grad_difference(left: nn.Module, right: nn.Module) -> float:
    left_params, right_params = dict(left.named_parameters()), dict(right.named_parameters())
    assert tuple(left_params) == tuple(right_params)
    difference = 0.0
    for name, parameter in left_params.items():
        other = right_params[name]
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is None and other.grad is None, name
            continue
        difference = max(difference, float((parameter.grad - other.grad).abs().max()))
    return difference


def _grad_norm(parameters) -> float:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    return float(torch.sqrt(sum(gradient.detach().square().sum() for gradient in gradients)))


def _batch(seed: int = 71031) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((2, 3, 4, 5), generator=generator)
    x[:, 0] = 1.0  # all cells are open; masked operators have a simple domain
    yy = torch.arange(4).view(1, 1, 4, 1)
    xx = torch.arange(5).view(1, 1, 1, 5)
    target = ((yy + xx) % 2).float().expand(2, 1, 4, 5).clone()
    return x, target


def _check_current_exact_multistep() -> None:
    seed = 31991
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = StreamingCell(streaming=True)
    current = make_model("current", seed)
    assert isinstance(current, CurrentCell)
    _same_state(reference, current)
    x, _ = _batch(31992)
    expected = reference.initial(x)
    actual = current.initial(x)
    for _ in range(4):
        expected = reference.step(expected, x)
        actual = current.step(actual, x)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        assert torch.equal(reference.logits(expected), current.logits(actual))


def _check_candidate_initialization_and_common_path() -> None:
    seed = 41027
    baseline = make_model("current", seed)
    additive = make_model("additive", seed)
    delta = make_model("delta", seed)
    assert isinstance(additive, AddressedDeltaCell)
    assert isinstance(delta, AddressedDeltaCell)

    for name in ("encoder", "q_in", "q_out", "readout"):
        base_module = getattr(baseline, name)
        for suffix, tensor in base_module.state_dict().items():
            candidate_tensor = getattr(additive, name).state_dict()[suffix]
            assert torch.equal(tensor, candidate_tensor), f"{name}.{suffix}"
    assert torch.equal(additive.carrier_writer.perception.weight, baseline.f_in.weight)
    assert torch.equal(additive.carrier_writer.perception.bias, baseline.f_in.bias)
    _same_state(additive, delta)
    assert additive.carrier_writer.subtract_read is False
    assert delta.carrier_writer.subtract_read is True
    assert not hasattr(additive, "f_in") and not hasattr(additive, "f_out")
    assert not hasattr(delta, "f_in") and not hasattr(delta, "f_out")
    assert not any(name.startswith(("f_in.", "f_out.")) for name, _ in additive.named_parameters())
    assert not any(name.startswith(("f_in.", "f_out.")) for name, _ in delta.named_parameters())

    x, _ = _batch(41028)
    states = [model.initial(x) for model in (baseline, additive, delta)]
    for _ in range(4):
        old_features = baseline._features(*states[0], x)
        incoming = stream(states[0][0], x[:, :1])
        writer_features = torch.cat((incoming, old_features[:, 24:]), dim=1)
        for model in (additive, delta):
            assert torch.equal(model.carrier_writer(incoming, writer_features), incoming)

        next_states = [model.step(state, x) for model, state in zip((baseline, additive, delta), states)]
        for model, next_state in zip((additive, delta), next_states[1:]):
            assert all(torch.equal(a, b) for a, b in zip(next_states[0], next_state))
            assert torch.equal(baseline.logits(next_states[0]), model.logits(next_state))
        states = next_states


def _check_active_reference_equations() -> None:
    x, _ = _batch(51039)
    results = {}
    for arm in ("additive", "delta"):
        model = make_model(arm, 51040)
        assert isinstance(model, AddressedDeltaCell)
        generator = torch.Generator(device="cpu").manual_seed(51041)
        with torch.no_grad():
            model.carrier_writer.key_head.weight.copy_(
                torch.randn(model.carrier_writer.key_head.weight.shape, generator=generator) * 0.02
            )
            model.carrier_writer.key_head.bias.copy_(
                torch.randn(model.carrier_writer.key_head.bias.shape, generator=generator) * 0.02
            )

        state = model.initial(x)
        old_features = model._features(*state, x)
        incoming = stream(state[0], x[:, :1])
        features = torch.cat((incoming, old_features[:, 24:]), dim=1)
        keys, values = model.carrier_writer.controls(features)
        b, _, h, w = incoming.shape
        lanes = incoming.reshape(b, 4, 6, h, w)
        if arm == "delta":
            residual = values - torch.einsum("brphw,bpdhw->brdhw", keys, lanes)
        else:
            residual = values
        expected = lanes + 0.1 * torch.einsum("brphw,brdhw->bpdhw", keys, residual)
        expected = expected.reshape_as(incoming)
        actual_state = model.step(state, x)
        torch.testing.assert_close(actual_state[0], expected, atol=1e-7, rtol=1e-6)
        assert torch.count_nonzero(keys) > 0
        results[arm] = float((actual_state[0] - expected).abs().max())

    # Exercise the copied primitive independently of the cell wrapper too.
    torch.manual_seed(51042)
    u = torch.randn((2, 4, 6, 2, 3), dtype=torch.float64)
    k = torch.randn((2, 4, 4, 2, 3), dtype=torch.float64)
    v = torch.randn((2, 4, 6, 2, 3), dtype=torch.float64)
    from writer import normalize_reads

    normalized = normalize_reads(k)
    expected = u + 0.1 * torch.einsum(
        "brphw,brdhw->bpdhw",
        normalized,
        v - torch.einsum("brphw,bpdhw->brdhw", normalized, u),
    )
    actual = addressed_write(u, normalized, v, 0.1, subtract_read=True)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def _check_bce_bootstrap_gradients() -> dict:
    x, target = _batch(61051)
    additive = make_model("additive", 61052)
    delta = make_model("delta", 61052)
    _same_state(additive, delta)
    assert torch.count_nonzero(additive.q_out.weight) == 0
    assert torch.count_nonzero(additive.q_out.bias) == 0

    options = {"lr": 0.001, "weight_decay": 0.0001, "betas": (0.9, 0.999), "eps": 1e-8}
    optimizers = (torch.optim.AdamW(additive.parameters(), **options),
                  torch.optim.AdamW(delta.parameters(), **options))
    trajectory = []
    for update in range(1, 4):
        for model, optimizer in zip((additive, delta), optimizers):
            optimizer.zero_grad(set_to_none=True)
            state = model.initial(x)
            for _ in range(3):
                state = model.step(state, x)
            loss = F.binary_cross_entropy_with_logits(model.logits(state), target)
            loss.backward()

        if update <= 2:
            assert _max_grad_difference(additive, delta) == 0.0
        for model in (additive, delta):
            assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters())

        key_norms = [
            _grad_norm(model.carrier_writer.key_head.parameters())
            for model in (additive, delta)
        ]
        value_norms = [
            _grad_norm(model.carrier_writer.value_head.parameters())
            for model in (additive, delta)
        ]
        q_norms = [_grad_norm((model.q_out.weight, model.q_out.bias))
                   for model in (additive, delta)]
        assert min(q_norms) > 0.0
        if update == 1:
            assert max(key_norms + value_norms) == 0.0
        elif update == 2:
            assert min(key_norms) > 0.0
            assert max(value_norms) == 0.0
        else:
            assert min(value_norms) > 0.0

        trajectory.append({
            "update": update,
            "q_out_gradient_norm": q_norms,
            "key_head_gradient_norm": key_norms,
            "value_head_gradient_norm": value_norms,
            "paired_gradient_max_abs_difference": _max_grad_difference(additive, delta),
        })
        for optimizer in optimizers:
            optimizer.step()

    return {"optimizer": "AdamW(lr=0.001, weight_decay=0.0001, betas=(0.9,0.999), eps=1e-8)",
            "loss": "binary_cross_entropy_with_logits over a 3-step cell rollout",
            "initial_q_out_is_zero": True,
            "updates": trajectory}


def run_checks() -> dict:
    torch.set_num_threads(2)
    _check_current_exact_multistep()
    _check_candidate_initialization_and_common_path()
    _check_active_reference_equations()
    bootstrap = _check_bce_bootstrap_gradients()

    counts = {arm: sum(parameter.numel() for parameter in make_model(arm, 70061).parameters())
              for arm in ARMS}
    assert counts == COUNTS
    metadata = {arm: make_model(arm, 70062).metadata() for arm in ARMS}
    assert {arm: metadata[arm]["parameter_count"] for arm in ARMS} == COUNTS
    return {
        "status": "PASS",
        "scope": "CPU cell integration and short BCE autograd checks only",
        "torch": torch.__version__,
        "arms": ARMS,
        "parameter_counts": counts,
        "checks": [
            "current_exact_4_step_parity_with_historical_StreamingCell",
            "candidate_initial_identity_and_common_encoder_Q_readout_outputs",
            "additive_delta_same_initial_tensors_and_first_2_update_gradients",
            "active_key_additive_and_delta_reference_equations",
            "original_zero_Q_out_BCE_bootstrap_over_3_AdamW_updates",
            "removed_f_in_f_out_parameters_are_not_registered_or_optimized",
        ],
        "bootstrap": bootstrap,
        "task_training_campaign": False,
        "cuda_qualified": False,
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, allow_nan=False))
