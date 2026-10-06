"""CPU implementation checks for Hybrid Writer v0."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "new"))
sys.path.insert(0, str(ROOT / "new" / "streaming_carry"))

from hybrid_writer.cells import ARMS, HybridWriterCell, hat_basis, make_model  # noqa: E402
from stream_cells import StreamingCell  # noqa: E402


def _fixture(seed: int = 121, batch: int = 2, height: int = 4, width: int = 5):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch, 3, height, width, generator=generator)
    x[:, :1] = (torch.rand(batch, 1, height, width, generator=generator) > 0.2).float()
    carrier = torch.randn(batch, 24, height, width, generator=generator)
    latent = torch.randn(batch, 8, height, width, generator=generator)
    return (carrier, latent), x


def _check_hats() -> None:
    phi = torch.tensor([0.0, 0.125, 0.5, 0.875, 1.0]).view(1, 1, 1, 5)
    basis = hat_basis(phi)
    assert basis.shape == (1, 1, 3, 1, 5)
    assert bool((basis >= 0).all())
    torch.testing.assert_close(basis.sum(dim=2), torch.ones_like(phi))
    torch.testing.assert_close(basis[0, 0, :, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(basis[0, 0, :, 0, 2], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(basis[0, 0, :, 0, 4], torch.tensor([0.0, 0.0, 1.0]))


def _check_initialization_and_controls() -> dict:
    models = {arm: make_model(arm, seed=37, device="cpu") for arm in ARMS}
    state, x = _fixture(seed=88)
    reference = models["neural"]
    initial = reference.initial(x)
    for arm, model in models.items():
        assert model.metadata()["parameter_count"] == sum(p.numel() for p in model.parameters())
        assert model.writer_observer is None
        assert "writer_observer" not in dict(model.named_parameters())
        assert "writer_observer" not in dict(model.named_buffers())
        for name in ("encoder", "q_in", "q_out", "readout"):
            for left, right in zip(getattr(reference, name).parameters(), getattr(model, name).parameters()):
                assert torch.equal(left, right), f"{arm} common parameter mismatch: {name}"
        got_initial = model.initial(x)
        assert all(torch.equal(a, b) for a, b in zip(initial, got_initial))
        detail = model.details(initial, x)
        assert tuple(detail) == ("incoming", "proposal", "delta", "gate", "phi")
        assert torch.count_nonzero(detail["proposal"]) == 0
        assert torch.count_nonzero(detail["delta"]) == 0

    for arm in ("neural", "budget", "hybrid"):
        for name in ("f_in", "f_out"):
            for left, right in zip(
                getattr(models["neural"], name).parameters(), getattr(models[arm], name).parameters()
            ):
                assert torch.equal(left, right), f"{arm} canonical MLP mismatch: {name}"
    affine_parameter_names = tuple(name for name, _ in models["affine_hybrid"].named_parameters())
    assert not any(name.startswith(("f_in.", "f_out.")) for name in affine_parameter_names)
    assert "affine_proposal.weight" in affine_parameter_names

    next_states = {arm: model.step(initial, x) for arm, model in models.items()}
    for arm, values in next_states.items():
        assert all(torch.equal(a, b) for a, b in zip(next_states["neural"], values)), (
            f"initial control diverged for {arm} despite m=0"
        )

    calls = []
    models["hybrid"].writer_observer = lambda detail, observed_x: calls.append((detail, observed_x))
    models["hybrid"].step(initial, x)
    assert len(calls) == 1 and calls[0][1] is x
    assert tuple(calls[0][0]) == ("incoming", "proposal", "delta", "gate", "phi")
    return {
        "canonical_common_parameters_equal": True,
        "initial_states_and_step_functions_equal_when_proposal_zero": True,
        "affine_inactive_mlp_absent": True,
        "observer_called_once_with_frozen_keys": True,
    }


def _check_neural_exact_baseline() -> None:
    model = make_model("neural", seed=49, device="cpu")
    baseline = StreamingCell()
    with torch.no_grad():
        generator = torch.Generator(device="cpu").manual_seed(950)
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.08)
        for name in ("encoder", "f_in", "f_out", "q_in", "q_out", "readout"):
            getattr(baseline, name).load_state_dict(getattr(model, name).state_dict())
    state, x = _fixture(seed=167, batch=1, height=5, width=4)
    actual = model.step(state, x)
    expected = baseline.step(state, x)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected)), "neural arm differs from StreamingCell"


def _check_proposal_jacobian() -> None:
    state, x = _fixture(seed=419, batch=1, height=2, width=2)
    eye = torch.eye(24, dtype=torch.float64) * 0.1
    for arm in ARMS:
        model = make_model(arm, seed=11, device="cpu")
        detail = model.details(state, x)
        gate = detail["gate"]
        if gate is not None:
            gate = gate[:, :, :1, :1].to(torch.float64)

        def apply(flat):
            proposal = flat.reshape(1, 24, 1, 1)
            return model._scale_proposal(proposal, gate).reshape(-1)

        zero = torch.zeros(24, dtype=torch.float64, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(apply, zero)
        torch.testing.assert_close(jacobian, eye, rtol=0.0, atol=1e-12)
    return None


def _set_large_proposal(model: HybridWriterCell) -> None:
    module = model.affine_proposal if model.arm == "affine_hybrid" else model.f_out
    with torch.no_grad():
        module.weight.zero_()
        module.bias.fill_(1_000_000.0)
        if model.arm == "budget":
            model.beta_d.fill_(12.0)
        elif model.arm in ("hybrid", "affine_hybrid"):
            model.beta_dpq.fill_(12.0)


def _check_bounded_updates() -> None:
    state, x = _fixture(seed=121, batch=1, height=3, width=4)
    for arm in ("budget", "hybrid", "affine_hybrid"):
        model = make_model(arm, seed=5, device="cpu")
        _set_large_proposal(model)
        detail = model.details(state, x)
        assert float(detail["proposal"].abs().max()) > 1e5
        lane_rms = detail["delta"].reshape(1, 4, 6, 3, 4).square().mean(dim=2).sqrt()
        assert float(lane_rms.max()) <= 0.1 + 1e-6, f"{arm} RMS delta bound violated"


def _check_writer_gradients() -> None:
    state, x = _fixture(seed=772, batch=2, height=4, width=3)
    generator = torch.Generator(device="cpu").manual_seed(991)
    results = {}
    for arm in ARMS:
        model = make_model(arm, seed=19, device="cpu")
        with torch.no_grad():
            proposal_module = model.affine_proposal if arm == "affine_hybrid" else model.f_out
            proposal_module.weight.copy_(torch.randn(proposal_module.weight.shape, generator=generator) * 0.05)
            proposal_module.bias.copy_(torch.randn(proposal_module.bias.shape, generator=generator) * 0.05)
            if arm != "affine_hybrid":
                model.f_in.weight.copy_(torch.randn(model.f_in.weight.shape, generator=generator) * 0.05)
                model.f_in.bias.copy_(torch.randn(model.f_in.bias.shape, generator=generator) * 0.05)
            if arm == "budget":
                model.beta_d.copy_(torch.linspace(-0.8, 0.9, 4))
            elif arm in ("hybrid", "affine_hybrid"):
                model.beta_dpq.copy_(torch.linspace(-0.9, 0.7, 36).reshape(4, 3, 3))

        model.zero_grad(set_to_none=True)
        detail = model.details(state, x)
        assert bool(detail["proposal"].abs().sum() > 0), f"{arm} fixture has m=0"
        probe = torch.randn(detail["delta"].shape, generator=generator)
        (detail["delta"] * probe).sum().backward()
        writer = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(("f_in.", "f_out.", "affine_proposal."))
        ]
        assert any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and bool(torch.count_nonzero(parameter.grad) > 0)
            for parameter in writer
        ), f"{arm} writer received no finite nonzero gradient"
        gate_parameters = [
            parameter for name, parameter in model.named_parameters() if name.startswith("beta_")
        ]
        if gate_parameters:
            assert any(
                parameter.grad is not None and bool(torch.count_nonzero(parameter.grad) > 0)
                for parameter in gate_parameters
            ), f"{arm} gate received no gradient"
        results[arm] = float(sum(p.grad.square().sum() for p in writer if p.grad is not None).sqrt())
    return results


def run_checks() -> dict:
    torch.set_num_threads(1)
    _check_hats()
    controls = _check_initialization_and_controls()
    _check_neural_exact_baseline()
    _check_proposal_jacobian()
    _check_bounded_updates()
    gradients = _check_writer_gradients()
    return {
        "status": "PASS",
        "device": "cpu",
        "checks": {
            "hat_basis_nonnegative_partition_of_unity": True,
            **controls,
            "neural_step_bitexact_streaming_baseline_after_randomization": True,
            "proposal_jacobian_at_zero_is_0_1_identity": True,
            "bounded_arm_large_proposal_rms_delta_at_most_0_1": True,
            "finite_nonzero_writer_gradients_when_proposal_nonzero": gradients,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, sort_keys=True))
