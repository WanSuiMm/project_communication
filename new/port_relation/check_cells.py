"""CPU implementation checks for the three port-relation cell arms."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
STREAMING_ROOT = HERE.parent / "streaming_carry"
if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

from cells import ARMS, make_model, relation_parameters  # noqa: E402
from stream_cells import StreamingCell, stream  # noqa: E402


COMMON_PREFIXES = ("encoder.", "f_in.", "f_out.", "q_in.", "q_out.", "readout.")


def tensor_hash(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def module_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {
        name: tensor_hash(value)
        for name, value in model.state_dict().items()
        if name.startswith(COMMON_PREFIXES)
    }


def make_x(seed: int = 511) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand((2, 3, 3, 5), generator=generator)
    masks = torch.tensor(
        [
            [[1, 1, 0, 1, 1], [1, 0, 1, 1, 0], [1, 1, 1, 0, 1]],
            [[1, 0, 1, 1, 1], [1, 1, 1, 0, 1], [0, 1, 1, 1, 1]],
        ],
        dtype=torch.float32,
    )
    x[:, :1] = masks[:, None]
    return x


def activate_common_tails(models: list[torch.nn.Module], seed: int = 512) -> None:
    """Give F and Q nonzero final layers while keeping them matched exactly."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    names = ("f_out.weight", "f_out.bias", "q_out.weight", "q_out.bias")
    values = {
        name: torch.randn(dict(models[0].named_parameters())[name].shape, generator=generator)
        * 0.025
        for name in names
    }
    with torch.no_grad():
        for model in models:
            parameters = dict(model.named_parameters())
            for name, value in values.items():
                parameters[name].copy_(value)


def trajectory_loss(model: torch.nn.Module, x: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    state = model.initial(x)
    generator = torch.Generator(device="cpu").manual_seed(513)
    c_probe = torch.randn(state[0].shape, generator=generator)
    z_probe = torch.randn(state[1].shape, generator=generator)
    loss = state[0].new_zeros(())
    for step in range(4):
        state = model.step(state, x)
        loss = loss + (state[0] * c_probe).sum() * (0.13 + 0.02 * step)
        loss = loss + (state[1] * z_probe).sum() * (0.21 + 0.03 * step)
    return state, loss


def check_shared_initialization_and_zero_relation() -> dict[str, dict[str, str]]:
    seed = 514
    models = [make_model(arm, seed) for arm in ARMS]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = StreamingCell(streaming=True)

    hashes = {model.arm: module_hashes(model) for model in models}
    assert hashes["current"] == hashes["constant"] == hashes["conditioned"]
    assert all(
        torch.equal(reference.state_dict()[name], models[0].state_dict()[name])
        for name in reference.state_dict()
    )
    assert relation_parameters(models[0]) == {}
    assert set(relation_parameters(models[1])) == {"relation_matrix"}
    assert set(relation_parameters(models[2])) == {
        "relation_map.weight",
        "relation_map.bias",
    }

    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    assert counts == [5033, 5049, 5177]
    assert all(
        torch.count_nonzero(parameter) == 0
        for model in models[1:]
        for parameter in relation_parameters(model).values()
    )

    activate_common_tails([reference, *models])
    x = make_x()
    reference.zero_grad(set_to_none=True)
    reference_state, reference_loss = trajectory_loss(reference, x)
    reference_loss.backward()
    reference_grads = dict(reference.named_parameters())
    for model in models:
        model.zero_grad(set_to_none=True)
        state, loss = trajectory_loss(model, x)
        assert all(torch.equal(a, b) for a, b in zip(reference_state, state))
        assert torch.equal(reference_loss, loss)
        loss.backward()
        parameters = dict(model.named_parameters())
        for name in reference_grads:
            old_grad, new_grad = reference_grads[name].grad, parameters[name].grad
            assert (old_grad is None) == (new_grad is None), name
            if old_grad is not None:
                assert torch.equal(old_grad, new_grad), name

    return hashes


def _manual_apply(u: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Slow scalar reference; matrix is [B,4,4,H,W], ordered K[a,b]."""
    batch, _, height, width = u.shape
    lanes = u.reshape(batch, 4, 6, height, width)
    result = torch.zeros_like(u).reshape(batch, 4, 6, height, width)
    for n in range(batch):
        for a in range(4):
            for payload in range(6):
                for y in range(height):
                    for x in range(width):
                        value = u.new_zeros(())
                        for b in range(4):
                            value = value + matrix[n, a, b, y, x] * lanes[n, b, payload, y, x]
                        result[n, a, payload, y, x] = value
    return result.reshape_as(u)


def _manual_conditioned_matrices(
    model: torch.nn.Module, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = z.shape
    full = z.new_zeros((batch, 4, 4, height, width))
    state = torch.zeros_like(full)
    weight = model.relation_map.weight.detach()[:, :, 0, 0]
    bias = model.relation_map.bias.detach()
    z_tanh = torch.tanh(z)
    for n in range(batch):
        for a in range(4):
            for b in range(4):
                output = a * 4 + b
                for y in range(height):
                    for x in range(width):
                        value = z.new_zeros(())
                        for channel in range(8):
                            value = value + weight[output, channel] * z_tanh[n, channel, y, x]
                        state[n, a, b, y, x] = value
                        full[n, a, b, y, x] = value + bias[output]
    return full, state


def check_relation_reference_and_observer() -> None:
    x = make_x(seed=515)[:1, :, :2, :3]
    generator = torch.Generator(device="cpu").manual_seed(516)
    c = torch.randn((1, 24, 2, 3), generator=generator)
    z = torch.randn((1, 8, 2, 3), generator=generator)
    u = stream(c, x[:, :1])

    constant = make_model("constant", 517)
    with torch.no_grad():
        constant.relation_matrix.copy_(
            torch.tensor(
                [[0.1, -0.2, 0.3, 0.05], [0.4, 0.15, -0.1, 0.2],
                 [-0.3, 0.25, 0.07, 0.11], [0.05, -0.15, 0.35, -0.22]]
            )
        )
    constant_matrix = constant.relation_matrix.view(1, 4, 4, 1, 1).expand(1, 4, 4, 2, 3)
    seen_constant: list[tuple[torch.Tensor, ...]] = []
    constant.relation_observer = lambda model, *args: seen_constant.append(args)
    constant.step((c, z), x)
    observed_u, observed_delta, observed_conditioned, observed_x = seen_constant[0]
    assert observed_x is x and torch.equal(observed_u, u)
    torch.testing.assert_close(
        observed_delta, _manual_apply(u, constant_matrix), rtol=1e-6, atol=1e-7
    )
    assert torch.count_nonzero(observed_conditioned) == 0

    conditioned = make_model("conditioned", 517)
    with torch.no_grad():
        generator = torch.Generator(device="cpu").manual_seed(518)
        conditioned.relation_map.weight.copy_(
            torch.randn(conditioned.relation_map.weight.shape, generator=generator) * 0.04
        )
        conditioned.relation_map.bias.copy_(
            torch.linspace(-0.15, 0.15, 16)
        )
    full_matrix, state_matrix = _manual_conditioned_matrices(conditioned, z)
    seen_conditioned: list[tuple[torch.Tensor, ...]] = []
    conditioned.relation_observer = lambda model, *args: seen_conditioned.append(args)
    conditioned.step((c, z), x)
    observed_u, observed_delta, observed_conditioned, observed_x = seen_conditioned[0]
    assert observed_x is x and torch.equal(observed_u, u)
    torch.testing.assert_close(
        observed_delta, _manual_apply(u, full_matrix), rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        observed_conditioned, _manual_apply(u, state_matrix), rtol=1e-6, atol=1e-7
    )

    current = make_model("current", 517)
    seen_current: list[tuple[torch.Tensor, ...]] = []
    current.relation_observer = lambda model, *args: seen_current.append(args)
    current.step((c, z), x)
    _, current_delta, current_conditioned, _ = seen_current[0]
    assert torch.count_nonzero(current_delta) == 0
    assert torch.count_nonzero(current_conditioned) == 0


def _assert_live_relation_gradients(model: torch.nn.Module) -> None:
    relation = relation_parameters(model)
    assert relation
    for name, parameter in relation.items():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert float(parameter.grad.norm()) > 0.0, name


def check_nonzero_relation_gradients() -> None:
    x = make_x(seed=519)
    for arm in ("constant", "conditioned"):
        model = make_model(arm, 520)
        generator = torch.Generator(device="cpu").manual_seed(521)
        with torch.no_grad():
            for name in ("f_out.weight", "f_out.bias", "q_out.weight", "q_out.bias"):
                parameter = dict(model.named_parameters())[name]
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.03)
            if arm == "constant":
                model.relation_matrix.copy_(torch.randn((4, 4), generator=generator) * 0.04)
            else:
                model.relation_map.weight.copy_(
                    torch.randn(model.relation_map.weight.shape, generator=generator) * 0.04
                )
                model.relation_map.bias.copy_(torch.randn((16,), generator=generator) * 0.04)

        c = torch.randn((2, 24, 3, 5), generator=generator)
        z = torch.randn((2, 8, 3, 5), generator=generator) * 0.2
        q = model._candidate(c, z, x)
        assert float(q.norm()) > 0.0 and float(z.norm()) > 0.0
        model.zero_grad(set_to_none=True)
        c_new, z_new = model.step((c, z), x)
        c_probe = torch.randn(c_new.shape, generator=generator)
        z_probe = torch.randn(z_new.shape, generator=generator)
        loss = (c_new * c_probe).sum() + (z_new * z_probe).sum()
        loss.backward()
        _assert_live_relation_gradients(model)


def check_cold_gradient_delay() -> None:
    """Cold zero Q blocks relation gradients once; the next update can open them."""
    x = make_x(seed=522)
    target = x[:, 1:2]
    for arm in ("constant", "conditioned"):
        model = make_model(arm, 523)
        first_state = model.step(model.initial(x), x)
        first_loss = F.binary_cross_entropy_with_logits(model.logits(first_state), target)
        first_loss.backward()
        for name, parameter in relation_parameters(model).items():
            assert parameter.grad is not None, name
            assert bool(torch.isfinite(parameter.grad).all()), name
            assert torch.count_nonzero(parameter.grad) == 0, name

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        assert float(model.q_out.weight.norm()) > 0.0
        optimizer.zero_grad(set_to_none=True)
        state = model.initial(x)
        state = model.step(state, x)
        state = model.step(state, x)
        second_loss = F.binary_cross_entropy_with_logits(model.logits(state), target)
        second_loss.backward()
        _assert_live_relation_gradients(model)


def main() -> None:
    torch.set_num_threads(1)
    hashes = check_shared_initialization_and_zero_relation()
    check_relation_reference_and_observer()
    check_nonzero_relation_gradients()
    check_cold_gradient_delay()
    counts = {
        arm: {
            "parameters": sum(parameter.numel() for parameter in make_model(arm, 524).parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in make_model(arm, 524).parameters()
                if parameter.requires_grad
            ),
        }
        for arm in ARMS
    }
    print(
        json.dumps(
            {
                "status": "PASS",
                "parameter_counts": counts,
                "common_initialization_sha256": hashes,
                "checks": {
                    "current_matches_historical_streaming_cell": True,
                    "zero_relation_multistep_outputs_and_common_gradients_exact": True,
                    "nonzero_K_manual_lane_payload_reference": True,
                    "conditioned_K_and_state_only_observer_reference": True,
                    "relation_parameter_gradients_finite_and_nonzero": True,
                    "cold_first_gradient_zero_then_second_update_activates_relation": True,
                    "rectangular_binary_mask": True,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
