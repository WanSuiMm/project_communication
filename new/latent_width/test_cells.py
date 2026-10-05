"""CPU-only invariants for the latent-width comparison cells."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch.nn import functional as F


def _load_cells_module():
    # Do not import this file as top-level ``cells``: masked_cells imports the
    # frozen NCA implementation under that name.
    module_name = "_latent_width_cells_under_test"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    path = Path(__file__).resolve().with_name("cells.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load cell module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


CELLS = _load_cells_module()


def _directions(phase: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if phase == 0:
        return ((-1, 0), (1, 0))
    if phase == 1:
        return ((0, -1), (0, 1))
    raise ValueError(phase)


def _destination(
    mask: torch.Tensor,
    batch: int,
    y: int,
    x: int,
    channel: int,
    phase: int,
) -> tuple[int, int, int]:
    if not bool(mask[batch, 0, y, x]):
        return y, x, channel
    dy, dx = _directions(phase)[channel]
    ny, nx = y + dy, x + dx
    height, width = mask.shape[-2:]
    if 0 <= ny < height and 0 <= nx < width and bool(mask[batch, 0, ny, nx]):
        return ny, nx, channel
    return y, x, 1 - channel


def _outgoing_scatter(
    carrier: torch.Tensor, mask: torch.Tensor, phase: int
) -> torch.Tensor:
    """Independent definition: scatter every source port to its destination."""
    result = torch.empty_like(carrier)
    written = torch.zeros_like(carrier, dtype=torch.bool)
    for batch in range(carrier.shape[0]):
        for y in range(carrier.shape[2]):
            for x in range(carrier.shape[3]):
                for channel in range(2):
                    dy, dx, dc = _destination(mask, batch, y, x, channel, phase)
                    assert not bool(written[batch, dc, dy, dx])
                    result[batch, dc, dy, dx] = carrier[batch, channel, y, x]
                    written[batch, dc, dy, dx] = True
    assert bool(written.all())
    return result


def _inverse_scatter(
    carrier: torch.Tensor, mask: torch.Tensor, phase: int
) -> torch.Tensor:
    """Independent inverse made from the outgoing source-to-destination map."""
    result = torch.empty_like(carrier)
    written = torch.zeros_like(carrier, dtype=torch.bool)
    for batch in range(carrier.shape[0]):
        for y in range(carrier.shape[2]):
            for x in range(carrier.shape[3]):
                for channel in range(2):
                    dy, dx, dc = _destination(mask, batch, y, x, channel, phase)
                    assert not bool(written[batch, channel, y, x])
                    result[batch, channel, y, x] = carrier[batch, dc, dy, dx]
                    written[batch, channel, y, x] = True
    assert bool(written.all())
    return result


def test_arm_order_parameter_counts_and_metadata() -> None:
    expected = [
        ("w24", 24, 40, 16, 5033),
        ("w8", 8, 40, 16, 2521),
        ("w24_capacity", 24, 16, 12, 2521),
        ("w16", 16, 40, 16, 3777),
        ("w4", 4, 40, 16, 1893),
        ("w2_alternating", 2, 40, 16, 1579),
    ]
    actual_specs = [
        (
            spec["name"],
            spec["workspace_channels"],
            spec["workspace_hidden"],
            spec["candidate_hidden"],
        )
        for spec in CELLS.ARMS
    ]
    assert actual_specs == [entry[:4] for entry in expected]
    for name, workspace, f_hidden, q_hidden, count in expected:
        model = CELLS.make_model(name)
        assert sum(parameter.numel() for parameter in model.parameters()) == count
        assert model.metadata()["parameter_count"] == count
        assert model.workspace_channels == workspace
        assert model.workspace_hidden == f_hidden
        assert model.candidate_hidden == q_hidden
    exploratory = CELLS.make_model("w2_alternating").metadata()
    assert exploratory["status"] == "exploratory"
    assert exploratory["learned_carrier_channels"] == 2
    assert exploratory["stationary_latent_channels"] == 8
    assert exploratory["clock_bits"] == 1
    assert "no four fixed directional lanes" in exploratory["carrier_layout"]
    assert "no global mutable clock" in exploratory["phase_clock"]["scope"]


def test_phase_transport_exhaustive_binary_2x3() -> None:
    carrier = torch.arange(12, dtype=torch.float64).reshape(1, 2, 2, 3) + 0.25
    for phase in (0, 1):
        for bits in range(1 << 6):
            mask = torch.tensor(
                [(bits >> index) & 1 for index in range(6)], dtype=torch.float64
            ).reshape(1, 1, 2, 3)
            actual = CELLS.phase_transport(carrier, mask, phase)
            expected = _outgoing_scatter(carrier, mask, phase)
            assert torch.equal(actual, expected), (phase, bits)

            inverse = CELLS.inverse_phase_transport(carrier, mask, phase)
            expected_inverse = _inverse_scatter(carrier, mask, phase)
            assert torch.equal(inverse, expected_inverse), (phase, bits)
            assert torch.equal(
                CELLS.inverse_phase_transport(actual, mask, phase), carrier
            ), (phase, bits)
            assert torch.allclose(
                torch.linalg.vector_norm(actual),
                torch.linalg.vector_norm(carrier),
                rtol=0.0,
                atol=1e-12,
            ), (phase, bits)

            for y in range(2):
                for x in range(3):
                    if not bool(mask[0, 0, y, x]):
                        assert torch.equal(
                            actual[0, :, y, x], carrier[0, :, y, x]
                        ), (phase, bits, y, x)


def test_phase_transport_adjoint_matches_inverse() -> None:
    masks = (
        torch.ones((1, 1, 2, 3), dtype=torch.float64),
        torch.zeros((1, 1, 2, 3), dtype=torch.float64),
        torch.tensor([[[[1, 1, 0], [1, 0, 1]]]], dtype=torch.float64),
    )
    generator = torch.Generator(device="cpu").manual_seed(20261005)
    for phase in (0, 1):
        for mask in masks:
            value = torch.randn((1, 2, 2, 3), generator=generator, dtype=torch.float64)
            probe = torch.randn((1, 2, 2, 3), generator=generator, dtype=torch.float64)
            value.requires_grad_()
            transported = CELLS.phase_transport(value, mask, phase)
            adjoint, = torch.autograd.grad((transported * probe).sum(), value)
            inverse = CELLS.inverse_phase_transport(probe, mask, phase)
            assert torch.equal(adjoint, inverse)


def test_w2_phase_clock_survives_detach_and_resets_independently() -> None:
    model = CELLS.make_model("w2_alternating")
    x = torch.zeros((1, 3, 2, 3), dtype=torch.float32)
    x[:, 0] = 1
    first = model.initial(x)
    second = model.initial(x)
    assert first[2].device.type == "cpu" and first[2].dtype == torch.int64
    assert first[2].ndim == 0 and first[2].item() == 0
    assert second[2].item() == 0 and second[2].data_ptr() != first[2].data_ptr()

    state = first
    for step_index in range(6):
        assert state[2].item() == step_index % 2
        state = model.step(state, x)
        assert state[2].item() == (step_index + 1) % 2
        if step_index in (0, 1, 3, 4):
            state = tuple(part.detach() for part in state)
    assert torch.equal(model.logits(state), model.readout(state[1]))
    assert second[2].item() == 0


def test_w24_matches_existing_streaming_cell() -> None:
    torch.manual_seed(811)
    actual = CELLS.make_model("w24")
    reference = CELLS.StreamingCell(
        streaming=True,
        workspace_channels=24,
        latent_channels=8,
        workspace_hidden=40,
        candidate_hidden=16,
    )
    assert type(actual) is CELLS.StreamingCell
    reference.load_state_dict(actual.state_dict())
    assert actual.metadata() == reference.metadata()

    x = torch.rand((2, 3, 3, 4), dtype=torch.float32)
    x[:, 0] = (x[:, 0] > 0.35).to(x.dtype)
    state_actual = actual.initial(x)
    state_reference = reference.initial(x)
    for left, right in zip(state_actual, state_reference):
        assert torch.equal(left, right)
    stepped_actual = actual.step(state_actual, x)
    stepped_reference = reference.step(state_reference, x)
    for left, right in zip(stepped_actual, stepped_reference):
        assert torch.equal(left, right)


def _masked_laplacian_reference(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    padded_h = F.pad(h, (1, 1, 1, 1), mode="replicate")
    padded_mask = F.pad(mask, (1, 1, 1, 1), mode="replicate")
    north_h = padded_h[:, :, :-2, 1:-1]
    south_h = padded_h[:, :, 2:, 1:-1]
    west_h = padded_h[:, :, 1:-1, :-2]
    east_h = padded_h[:, :, 1:-1, 2:]
    north_m = padded_mask[:, :, :-2, 1:-1]
    south_m = padded_mask[:, :, 2:, 1:-1]
    west_m = padded_mask[:, :, 1:-1, :-2]
    east_m = padded_mask[:, :, 1:-1, 2:]
    return mask * (
        north_m * (h - north_h)
        + south_m * (h - south_h)
        + west_m * (h - west_h)
        + east_m * (h - east_h)
    )


def _reference_w2_step(
    model,
    workspace: torch.Tensor,
    latent: torch.Tensor,
    x: torch.Tensor,
    phase: int,
):
    mask = x[:, :1]
    incoming = _outgoing_scatter(workspace, mask, phase)
    old_force_features = torch.cat(
        (
            incoming,
            latent,
            _masked_laplacian_reference(workspace, mask),
            _masked_laplacian_reference(latent, mask),
            x,
        ),
        dim=1,
    )
    force = model.f_out(torch.tanh(model.f_in(old_force_features)))
    workspace_new = incoming + model.eta * force
    candidate_features = torch.cat(
        (
            workspace_new,
            latent,
            _masked_laplacian_reference(workspace_new, mask),
            _masked_laplacian_reference(latent, mask),
            x,
        ),
        dim=1,
    )
    candidate = model.q_out(torch.tanh(model.q_in(candidate_features)))
    latent_new = latent + model.alpha * candidate
    return workspace_new, latent_new, force, candidate


def test_w2_feature_clock_nonzero_tails_and_gradients_match_reference() -> None:
    torch.manual_seed(2408)
    model = CELLS.make_model("w2_alternating")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.04)

    mask = torch.tensor([[[[1, 1, 0], [1, 1, 1]]]], dtype=torch.float32)
    image = torch.randn((1, 2, 2, 3), dtype=torch.float32) * 0.2
    x_value = torch.cat((mask, image), dim=1)
    workspace_value = torch.randn((1, 2, 2, 3), dtype=torch.float32) * 0.2
    latent_value = torch.randn((1, 8, 2, 3), dtype=torch.float32) * 0.2
    active_parameters = [
        model.f_in.weight,
        model.f_in.bias,
        model.f_out.weight,
        model.f_out.bias,
        model.q_in.weight,
        model.q_in.bias,
        model.q_out.weight,
        model.q_out.bias,
    ]

    for phase in (0, 1):
        workspace = workspace_value.clone().requires_grad_()
        latent = latent_value.clone().requires_grad_()
        x = x_value.clone().requires_grad_()
        actual_workspace, actual_latent, next_phase = model.step(
            (workspace, latent, torch.tensor(phase, dtype=torch.int64)), x
        )
        (reference_workspace, reference_latent, force, candidate) = _reference_w2_step(
            model, workspace, latent, x, phase
        )
        assert next_phase.device.type == "cpu" and next_phase.item() == 1 - phase
        assert torch.count_nonzero(force).item() > 0
        assert torch.count_nonzero(candidate).item() > 0
        incoming = _outgoing_scatter(workspace, x[:, :1], phase)
        assert not torch.equal(
            _masked_laplacian_reference(workspace, x[:, :1]),
            _masked_laplacian_reference(incoming, x[:, :1]),
        )

        actual_loss = actual_workspace.square().mean() + 0.37 * actual_latent.square().mean()
        reference_loss = (
            reference_workspace.square().mean()
            + 0.37 * reference_latent.square().mean()
        )
        targets = [workspace, latent, x, *active_parameters]
        actual_gradients = torch.autograd.grad(actual_loss, targets)
        reference_gradients = torch.autograd.grad(reference_loss, targets)
        for actual_gradient, reference_gradient in zip(
            actual_gradients, reference_gradients
        ):
            assert torch.allclose(
                actual_gradient,
                reference_gradient,
                rtol=2e-5,
                atol=2e-7,
            )
        assert all(torch.count_nonzero(gradient).item() > 0 for gradient in actual_gradients)


def check() -> dict[str, object]:
    """Run the compact CPU invariants and return a runner-friendly receipt."""
    test_arm_order_parameter_counts_and_metadata()
    test_phase_transport_exhaustive_binary_2x3()
    test_phase_transport_adjoint_matches_inverse()
    test_w2_phase_clock_survives_detach_and_resets_independently()
    test_w24_matches_existing_streaming_cell()
    test_w2_feature_clock_nonzero_tails_and_gradients_match_reference()
    return {
        "status": "PASS",
        "device": "cpu",
        "arms": [spec["name"] for spec in CELLS.ARMS],
        "parameter_counts": {
            spec["name"]: CELLS.make_model(spec["name"]).metadata()["parameter_count"]
            for spec in CELLS.ARMS
        },
        "exhaustive_masks_per_phase": 64,
        "transport_phases": 2,
        "w24_streaming_equivalence": True,
        "w2_gradient_reference": True,
    }
