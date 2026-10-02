"""CPU checks for the learned local-interface cell and its K8 gradient clock."""
from __future__ import annotations

from collections import deque
import copy
import json
from pathlib import Path
import sys

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "new/nca_inertial_wind_tunnel"))
sys.path.insert(0, str(ROOT / "new/short_bptt_phase2"))

from interface_cells import (  # noqa: E402
    CANDIDATE_HIDDEN,
    FEATURE_CHANNELS,
    InterfaceCell,
    MESSAGE_CHANNELS,
    VARIANTS,
    make_variant,
)
from revision_cells import RevisionCell  # noqa: E402
from stream_cells import StreamingCell  # noqa: E402
from tasks import bank, balanced_loss  # noqa: E402
from training import backward_trajectory  # noqa: E402


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPPOSITE = (2, 3, 0, 1)


def _explicit_stream(carrier: Tensor, mask: Tensor) -> Tensor:
    """Scatter every outgoing port independently and verify a bijection."""
    batch, channels, height, width = carrier.shape
    assert channels % 4 == 0
    lane_channels = channels // 4
    out = torch.zeros_like(carrier)
    writes = torch.zeros(batch, 4, height, width, dtype=torch.int32)
    for b in range(batch):
        for y in range(height):
            for x in range(width):
                for direction, (dy, dx) in enumerate(DIRECTIONS):
                    yy, xx, destination_lane = y, x, direction
                    if bool(mask[b, 0, y, x]):
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and bool(mask[b, 0, ny, nx])
                        ):
                            yy, xx = ny, nx
                        else:
                            destination_lane = OPPOSITE[direction]
                    assert writes[b, destination_lane, yy, xx].item() == 0
                    out[
                        b,
                        destination_lane * lane_channels : (destination_lane + 1)
                        * lane_channels,
                        yy,
                        xx,
                    ] = carrier[
                        b,
                        direction * lane_channels : (direction + 1) * lane_channels,
                        y,
                        x,
                    ]
                    writes[b, destination_lane, yy, xx] += 1
    assert bool((writes == 1).all())
    return out


def _reference_step(
    model: InterfaceCell, state: tuple[Tensor, Tensor], x: Tensor
) -> tuple[Tensor, Tensor]:
    """Independent interface step using explicit port scatter transport."""
    workspace, latent = state
    mask = x[:, :1]
    emitted = model.interface_emitter(torch.cat((workspace, latent, x), dim=1))
    incoming = _explicit_stream(emitted, mask)
    workspace_features = torch.cat((workspace, latent, incoming, x), dim=1)
    force = model.f_out(torch.tanh(model.f_in(workspace_features)))
    workspace_new = workspace + model.eta * force

    emitted_new = model.interface_emitter(torch.cat((workspace_new, latent, x), dim=1))
    incoming_new = _explicit_stream(emitted_new, mask)
    candidate_features = torch.cat((workspace_new, latent, incoming_new, x), dim=1)
    candidate = model.q_out(torch.tanh(model.q_in(candidate_features)))
    return workspace_new, latent + model.alpha * candidate


def _set_nonzero_residuals(model: InterfaceCell, seed: int = 900) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for layer in (model.f_out, model.q_out):
            layer.weight.normal_(std=0.035)
            layer.bias.normal_(std=0.02)


def _common_parameters(model: torch.nn.Module) -> dict[str, Tensor]:
    state = model.state_dict()
    names = ("encoder.weight", "encoder.bias", "readout.weight", "readout.bias")
    return {name: state[name] for name in names}


def _check_variants_shapes_and_common_initialization() -> None:
    x = torch.zeros(1, 3, 6, 7)
    x[:, 0, 1:5, 1:6] = 1.0
    for variant in VARIANTS:
        torch.manual_seed(14)
        model = make_variant(variant)
        assert sum(parameter.numel() for parameter in model.parameters()) == 5033
        assert model.metadata()["parameter_count"] == 5033
        state = model.initial(x)
        next_state = model.step(state, x)
        assert state[0].shape == next_state[0].shape == (1, 24, 6, 7)
        assert state[1].shape == next_state[1].shape == (1, 8, 6, 7)
        assert model.logits(next_state).shape == (1, 1, 6, 7)

    for seed in (2, 3, 4, 5):
        torch.manual_seed(seed)
        baseline = RevisionCell("ws_additive")
        torch.manual_seed(seed)
        streaming = StreamingCell()
        torch.manual_seed(seed)
        interface = InterfaceCell()
        expected = _common_parameters(baseline)
        for candidate in (streaming, interface):
            actual = _common_parameters(candidate)
            assert all(torch.equal(expected[name], actual[name]) for name in expected)
        assert sum(parameter.numel() for parameter in interface.parameters()) == 5033

    torch.manual_seed(15)
    model = InterfaceCell()
    metadata = model.metadata()
    assert model.interface_emitter.in_channels == 35
    assert model.interface_emitter.out_channels == MESSAGE_CHANNELS == 24
    assert model.workspace_hidden == 31
    assert model.candidate_hidden == CANDIDATE_HIDDEN == 21
    assert model.f_in.in_channels == model.q_in.in_channels == FEATURE_CHANNELS == 59
    assert metadata["workspace_input_channels"] == metadata["candidate_input_channels"] == 59
    assert metadata["persistent_state_channels"] == 32
    assert metadata["arm"] == "interface"
    assert metadata["transient_interface_channels"] == 24
    assert metadata["persistent_interface_message"] is False
    assert metadata["max_graph_hops_per_macro_step"] == 2


def _check_independent_step_outputs_and_gradients() -> None:
    torch.manual_seed(57)
    model = InterfaceCell()
    _set_nonzero_residuals(model, 58)
    x = torch.zeros(1, 3, 5, 6)
    x[:, 0, 1:4, 1:5] = 1.0
    x[:, 0, 2, 3] = 0.0
    x[:, 1, 1, 1] = 1.0
    state = (
        torch.randn(1, 24, 5, 6, requires_grad=True),
        torch.randn(1, 8, 5, 6, requires_grad=True),
    )
    actual = model.step(state, x)
    reference = _reference_step(model, state, x)
    for got, expected in zip(actual, reference):
        torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)

    probes = tuple(torch.randn_like(value) for value in actual)
    actual_loss = sum((value * probe).sum() for value, probe in zip(actual, probes))
    reference_loss = sum((value * probe).sum() for value, probe in zip(reference, probes))
    inputs_and_parameters = tuple(state) + tuple(model.parameters())
    actual_gradients = torch.autograd.grad(
        actual_loss, inputs_and_parameters, allow_unused=True
    )
    reference_gradients = torch.autograd.grad(
        reference_loss, inputs_and_parameters, allow_unused=True
    )
    for got, expected in zip(actual_gradients, reference_gradients):
        assert (got is None) == (expected is None)
        if got is not None:
            torch.testing.assert_close(got, expected, rtol=1e-6, atol=2e-7)


def _check_identity_and_emitter_influence() -> None:
    x = torch.zeros(1, 3, 7, 8)
    x[:, 0, 1:6, 1:7] = 1.0
    x[:, 1, 3, 2] = 1.0
    torch.manual_seed(81)
    identity = InterfaceCell()
    state0 = identity.initial(x)
    assert torch.count_nonzero(
        identity.interface_emitter(torch.cat((state0[0], state0[1], x), dim=1))
    ) > 0
    state = state0
    with torch.no_grad():
        for _ in range(8):
            state = identity.step(state, x)
            assert torch.equal(state[0], state0[0])
            assert torch.equal(state[1], state0[1])

    torch.manual_seed(82)
    active = InterfaceCell()
    _set_nonzero_residuals(active, 83)
    silent = copy.deepcopy(active)
    with torch.no_grad():
        silent.interface_emitter.weight.zero_()
        silent.interface_emitter.bias.zero_()
    probe_state = (
        torch.randn(1, 24, 7, 8),
        torch.randn(1, 8, 7, 8),
    )
    active_result = active.step(probe_state, x)
    silent_result = silent.step(probe_state, x)
    assert all(
        torch.count_nonzero(left - right).item() > 0
        for left, right in zip(active_result, silent_result)
    ), "the emitted interface did not influence both local residual updates"


def _bfs_distances(mask: Tensor, source: tuple[int, int]) -> Tensor:
    height, width = mask.shape[-2:]
    distances = torch.full((height, width), -1, dtype=torch.int64)
    sy, sx = source
    assert bool(mask[0, 0, sy, sx])
    distances[sy, sx] = 0
    queue = deque([(sy, sx)])
    while queue:
        y, x = queue.popleft()
        for dy, dx in DIRECTIONS:
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < height
                and 0 <= nx < width
                and bool(mask[0, 0, ny, nx])
                and distances[ny, nx] < 0
            ):
                distances[ny, nx] = distances[y, x] + 1
                queue.append((ny, nx))
    return distances


def _check_two_hop_lightcone_and_components() -> None:
    mask = torch.zeros(1, 1, 12, 16)
    mask[:, :, 1:11, 1:10] = 1.0
    mask[:, :, 2:9, 12:15] = 1.0
    x = torch.zeros(1, 3, 12, 16)
    x[:, :1] = mask
    source = (2, 2)
    x[0, 1, source[0], source[1]] = 1.0
    x_flip = x.clone()
    x_flip[0, 1, source[0], source[1]] = 0.0
    x_flip[0, 2, source[0], source[1]] = 1.0
    distances = _bfs_distances(mask, source)
    second_component = torch.zeros_like(mask, dtype=torch.bool)
    second_component[:, :, 2:9, 12:15] = True

    torch.manual_seed(91)
    model = InterfaceCell()
    _set_nonzero_residuals(model, 92)
    left = model.initial(x)
    right = model.initial(x_flip)
    checked_outside = 0
    with torch.no_grad():
        for step in range(1, 5):
            left = model.step(left, x)
            right = model.step(right, x_flip)
            outside = ((distances < 0) | (distances > 2 * step))[None, None]
            checked_outside += int(outside.sum())
            for a, b in zip(left, right):
                assert torch.equal(a.masked_select(outside.expand_as(a)), b.masked_select(outside.expand_as(b)))
            logits_left = model.logits(left)
            logits_right = model.logits(right)
            assert torch.equal(
                logits_left.masked_select(outside), logits_right.masked_select(outside)
            )
            for a, b in zip(left, right):
                assert torch.equal(
                    a.masked_select(second_component.expand_as(a)),
                    b.masked_select(second_component.expand_as(b)),
                )
    assert checked_outside > 0
    assert int((distances > 8).sum()) > 0


def _reference_k8_trajectory(
    model: InterfaceCell, data: dict[str, Tensor]
) -> tuple[Tensor, tuple[Tensor, Tensor], dict[str, int]]:
    """Independent K8 loop using the explicit-scatter reference step."""
    steps, loss_every, horizon = 64, 8, 8
    state = model.initial(data["x"])
    pending = []
    total = state[0].new_zeros(())
    backward_calls = 0
    cuts = 0
    for step in range(1, steps + 1):
        state = _reference_step(model, state, data["x"])
        if step % loss_every == 0:
            loss = balanced_loss(model.logits(state), data["y"], data["mask"])
            loss = loss / (steps // loss_every)
            assert bool(torch.isfinite(loss))
            pending.append(loss)
            total = total + loss.detach()
        if step % horizon == 0:
            torch.stack(pending).sum().backward()
            pending.clear()
            backward_calls += 1
            state = tuple(value.detach() for value in state)
            cuts += int(step < steps)
    trace = {
        "backward_calls": backward_calls,
        "interior_detach_boundaries": cuts,
        "forward_steps": steps,
        "loss_count": steps // loss_every,
    }
    return total, state, trace


def _check_k8_reference_and_unchanged_weights() -> None:
    torch.manual_seed(2)
    actual_model = InterfaceCell()
    _set_nonzero_residuals(actual_model, 902)
    reference_model = copy.deepcopy(actual_model)
    data = bank(8, 2, 722)
    actual_before = {name: value.clone() for name, value in actual_model.state_dict().items()}
    reference_before = {name: value.clone() for name, value in reference_model.state_dict().items()}

    actual_loss, actual_state, actual_trace = backward_trajectory(
        actual_model, data, gradient_horizon=8
    )
    reference_loss, reference_state, reference_trace = _reference_k8_trajectory(
        reference_model, data
    )
    torch.testing.assert_close(actual_loss, reference_loss, rtol=1e-6, atol=1e-7)
    assert actual_trace == reference_trace == {
        "backward_calls": 8,
        "interior_detach_boundaries": 7,
        "forward_steps": 64,
        "loss_count": 8,
    }
    for actual, expected in zip(actual_state, reference_state):
        assert actual.grad_fn is None and not actual.requires_grad
        assert expected.grad_fn is None and not expected.requires_grad
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    for (actual_name, actual_parameter), (reference_name, reference_parameter) in zip(
        actual_model.named_parameters(), reference_model.named_parameters()
    ):
        assert actual_name == reference_name
        assert actual_parameter.grad is not None and reference_parameter.grad is not None
        assert bool(torch.isfinite(actual_parameter.grad).all())
        torch.testing.assert_close(
            actual_parameter.grad, reference_parameter.grad, rtol=2e-5, atol=2e-6
        )
    torch.testing.assert_close(
        actual_model.interface_emitter.weight.grad,
        reference_model.interface_emitter.weight.grad,
        rtol=2e-5,
        atol=2e-6,
    )
    assert torch.count_nonzero(actual_model.interface_emitter.weight.grad).item() > 0
    assert all(
        torch.equal(actual_before[name], value)
        for name, value in actual_model.state_dict().items()
    )
    assert all(
        torch.equal(reference_before[name], value)
        for name, value in reference_model.state_dict().items()
    )


def main() -> None:
    torch.set_num_threads(2)
    _check_variants_shapes_and_common_initialization()
    _check_independent_step_outputs_and_gradients()
    _check_identity_and_emitter_influence()
    _check_two_hop_lightcone_and_components()
    _check_k8_reference_and_unchanged_weights()
    print(
        json.dumps(
            {
                "status": "PASS",
                "variants_and_parameters": 5033,
                "common_encoder_readout_seeds": [2, 3, 4, 5],
                "independent_reference_step_outputs_and_gradients": True,
                "zero_residual_local_identity_with_nonzero_emission": True,
                "emitter_affects_both_local_residual_pipeline": True,
                "two_hop_bfs_lightcone_steps": [1, 2, 3, 4],
                "disconnected_component_isolation": True,
                "K8_reference_gradient_clock": 8,
                "K8_forward_steps": 64,
                "K8_weights_unchanged": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
