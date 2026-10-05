"""CPU implementation checks for the latent-factorial cells."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
_spec = spec_from_file_location(
    "reaction_transport_latent_factorial_cells_under_test", HERE / "cells.py"
)
if _spec is None or _spec.loader is None:
    raise ImportError("Could not load latent-factorial cells.py")
cells = module_from_spec(_spec)
sys.modules[_spec.name] = cells
_spec.loader.exec_module(cells)


def _assert_exact(test: unittest.TestCase, left: torch.Tensor, right: torch.Tensor) -> None:
    test.assertEqual(left.shape, right.shape)
    test.assertTrue(torch.equal(left, right), "tensors should be exact copied values")


def _set_nonzero_heads(models: list[torch.nn.Module], seed: int = 9182) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = {
        "f_out.weight": torch.randn((8, 40, 1, 1), generator=generator) * 0.08,
        "f_out.bias": torch.randn((8,), generator=generator) * 0.03,
        "q_out.weight": torch.randn((8, 16, 1, 1), generator=generator) * 0.08,
        "q_out.bias": torch.randn((8,), generator=generator) * 0.03,
    }
    for model in models:
        with torch.no_grad():
            for name, value in values.items():
                module_name, parameter_name = name.split(".")
                getattr(getattr(model, module_name), parameter_name).copy_(value)


def _inputs(seed: int = 3201) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mask = (torch.rand((2, 1, 5, 6), generator=generator) > 0.28).float()
    x = torch.cat((mask, torch.randn((2, 2, 5, 6), generator=generator)), dim=1)
    c = torch.randn((2, 8, 5, 6), generator=generator)
    z = torch.randn((2, 8, 5, 6), generator=generator)
    return x, c, z, generator


class LatentFactorialCellTests(unittest.TestCase):
    def test_seeded_copy_counts_and_metadata(self) -> None:
        models = {arm: cells.make_model(arm, seed=43) for arm in cells.ARMS}
        expected_counts = {
            "native": 2521,
            "factorized": 2649,
            "native_r2": 3275,
            "factorized_r2": 3403,
        }
        for arm, model in models.items():
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected_counts[arm])
            metadata = model.metadata()
            self.assertEqual(metadata["parameter_count"], expected_counts[arm])
            self.assertFalse(metadata["stability_guarantee"])
            self.assertIn("geometry", metadata)
            self.assertIn("clocks", metadata)
            self.assertIn("state_shapes", metadata)
            self.assertIn("parameter_shapes", metadata)
            self.assertEqual(metadata["latent_update"], "Z_new = Z + alpha * Q")
            expected_input_channels = 37 if model.has_r2 else 35
            self.assertEqual(metadata["workspace_input_channels"], expected_input_channels)
            self.assertEqual(metadata["candidate_input_channels"], expected_input_channels)
            self.assertEqual(metadata["stationary_state"], "Z and R" if model.has_r2 else "Z")
            if model.has_r2:
                self.assertIn("R_plus", metadata["workspace_update"])
                self.assertEqual(metadata["initialization_seeds"]["r2_extension_seed_offset"], 104729)

        native, factorized = models["native"], models["factorized"]
        factorized_params = dict(factorized.named_parameters())
        for name, parameter in native.named_parameters():
            _assert_exact(self, parameter, factorized_params[name])
        self.assertTrue(torch.equal(factorized.f_in.U, torch.eye(8)))
        self.assertTrue(torch.equal(factorized.q_in.U, torch.eye(8)))
        torch.testing.assert_close(
            factorized.f_in.effective_weight(), native.f_in.weight, rtol=0, atol=0
        )
        torch.testing.assert_close(
            factorized.q_in.effective_weight(), native.q_in.weight, rtol=0, atol=0
        )
        self.assertIn("f_in.U", dict(factorized.named_parameters()))
        self.assertIn("q_in.U", dict(factorized.named_parameters()))

        native_r2, factorized_r2 = models["native_r2"], models["factorized_r2"]
        r2_params = dict(native_r2.named_parameters())
        for name, parameter in native.named_parameters():
            copied = r2_params[name]
            if name in ("f_in.weight", "q_in.weight"):
                _assert_exact(self, parameter, copied[:, :cells.BASE_FEATURE_CHANNELS])
            else:
                _assert_exact(self, parameter, copied)

        factorized_r2_params = dict(factorized_r2.named_parameters())
        for name, parameter in native_r2.named_parameters():
            _assert_exact(self, parameter, factorized_r2_params[name])
        self.assertTrue(torch.equal(factorized_r2.f_in.U, torch.eye(8)))
        self.assertTrue(torch.equal(factorized_r2.q_in.U, torch.eye(8)))
        torch.testing.assert_close(
            factorized_r2.f_in.effective_weight()[:, :cells.BASE_FEATURE_CHANNELS],
            native_r2.f_in.weight[:, :cells.BASE_FEATURE_CHANNELS], rtol=0, atol=0,
        )
        torch.testing.assert_close(
            factorized_r2.q_in.effective_weight()[:, :cells.BASE_FEATURE_CHANNELS],
            native_r2.q_in.weight[:, :cells.BASE_FEATURE_CHANNELS], rtol=0, atol=0,
        )

    def test_nontrivial_matched_transition_at_zero_r(self) -> None:
        models = [cells.make_model(arm, seed=71) for arm in cells.ARMS]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            reference = cells._StreamingCell(
                streaming=True,
                workspace_channels=8,
                latent_channels=8,
                workspace_hidden=40,
                candidate_hidden=16,
            )
        native_params = dict(models[0].named_parameters())
        for name, parameter in reference.named_parameters():
            _assert_exact(self, parameter, native_params[name])
        _set_nonzero_heads(models + [reference])
        x, c, z, generator = _inputs()
        r = torch.zeros((2, 2, 5, 6))
        outputs = []
        for model in models:
            state = (c, z, r) if model.has_r2 else (c, z)
            outputs.append(model.step(state, x))
        reference_output = reference.step((c, z), x)
        torch.testing.assert_close(outputs[0][0], reference_output[0], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(outputs[0][1], reference_output[1], rtol=1e-6, atol=1e-6)
        for output in outputs[1:]:
            torch.testing.assert_close(output[0], outputs[0][0], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(output[1], outputs[0][1], rtol=1e-6, atol=1e-6)
        for output in outputs[2:]:
            torch.testing.assert_close(output[2], r, rtol=0, atol=0)

    def test_factorized_effect_orientation_gradient_and_stream_order(self) -> None:
        model = cells.make_model("factorized", seed=101)
        module = model.f_in
        generator = torch.Generator(device="cpu").manual_seed(778)
        with torch.no_grad():
            module.U.copy_(torch.randn((8, 8), generator=generator) * 0.2)
            module.U.add_(torch.eye(8))
        features = torch.randn((2, 35, 4, 5), generator=generator)
        base_matrix = module.weight.detach()[..., 0, 0]
        u = module.U.detach()
        effective_matrix = base_matrix.clone()
        effective_matrix[:, :8] = base_matrix[:, :8] @ u
        effective_matrix[:, 16:24] = base_matrix[:, 16:24] @ u
        expected = F.conv2d(
            features, effective_matrix[:, :, None, None], module.bias
        )
        torch.testing.assert_close(module(features), expected, rtol=0, atol=0)

        wrong_orientation = base_matrix.clone()
        wrong_orientation[:, :8] = base_matrix[:, :8] @ u.T
        wrong_orientation[:, 16:24] = base_matrix[:, 16:24] @ u.T
        wrong = F.conv2d(features, wrong_orientation[:, :, None, None], module.bias)
        self.assertFalse(torch.allclose(expected, wrong, rtol=1e-5, atol=1e-6))

        b = torch.randn((40, 8), generator=generator)
        u_grad = torch.randn((8, 8), generator=generator, requires_grad=True)
        effective_grad = torch.randn((40, 8), generator=generator)
        loss = ((b @ u_grad) * effective_grad).sum()
        actual_gradient = torch.autograd.grad(loss, u_grad)[0]
        torch.testing.assert_close(actual_gradient, b.T @ effective_grad)

        # Mixing directional lanes before and after T(C) differs, so the learned
        # U belongs after transport as implemented by the factorized first layer.
        mask = (torch.rand((1, 1, 4, 5), generator=generator) > 0.35).float()
        carrier = torch.randn((1, 8, 4, 5), generator=generator)
        apply_u = lambda value: torch.einsum("ij,bjhw->bihw", u, value)
        after_stream = apply_u(cells._REFERENCE_STREAMING_MODULE.stream(carrier, mask))
        before_stream = cells._REFERENCE_STREAMING_MODULE.stream(apply_u(carrier), mask)
        self.assertFalse(torch.allclose(after_stream, before_stream, rtol=1e-5, atol=1e-6))

    def test_r2_stationarity_gradients_and_readout(self) -> None:
        model = cells.make_model("native_r2", seed=202)
        generator = torch.Generator(device="cpu").manual_seed(551)
        with torch.no_grad():
            model.f_out.weight.copy_(torch.randn(model.f_out.weight.shape, generator=generator) * 0.08)
            model.f_out.bias.copy_(torch.randn(model.f_out.bias.shape, generator=generator) * 0.02)
            model.q_out.weight.copy_(torch.randn(model.q_out.weight.shape, generator=generator) * 0.08)
            model.q_out.bias.copy_(torch.randn(model.q_out.bias.shape, generator=generator) * 0.02)
        self.assertEqual(float(model.g_out.weight.abs().sum()), 0.0)
        self.assertEqual(float(model.g_out.bias.abs().sum()), 0.0)

        x, c, z, _ = _inputs(seed=905)
        r = torch.randn((2, 2, 5, 6), generator=generator)
        state = (c, z, r)
        for _ in range(3):
            state = model.step(state, x)
            torch.testing.assert_close(state[2], r, rtol=0, atol=0)

        # With nonzero downstream F/Q heads, the zero-initialized G_out still
        # receives a gradient through R_plus. G_in remains graph-connected.
        state0 = (c, z, torch.zeros_like(r))
        first = model.step(state0, x)
        loss = first[0].square().sum() + first[1].square().sum()
        g_out_grad, g_in_grad = torch.autograd.grad(
            loss, (model.g_out.weight, model.g_in.weight), allow_unused=True
        )
        self.assertIsNotNone(g_out_grad)
        self.assertIsNotNone(g_in_grad)
        self.assertGreater(float(g_out_grad.abs().sum()), 0.0)

        with torch.no_grad():
            model.g_out.weight.copy_(torch.randn(model.g_out.weight.shape, generator=generator) * 0.04)
        opened = model.step(state0, x)
        opened_loss = opened[0].square().sum() + opened[1].square().sum()
        opened_g_in_grad = torch.autograd.grad(opened_loss, model.g_in.weight)[0]
        self.assertGreater(float(opened_g_in_grad.abs().sum()), 0.0)

        r_for_readout = r.detach().requires_grad_(True)
        logits_a = model.logits((c, z, r_for_readout))
        logits_b = model.logits((c, z, r_for_readout + 10.0))
        torch.testing.assert_close(logits_a, logits_b, rtol=0, atol=0)
        r_grad = torch.autograd.grad(logits_a.sum(), r_for_readout, allow_unused=True)
        self.assertIsNone(r_grad[0])


def check() -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LatentFactorialCellTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return {
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
    }


if __name__ == "__main__":
    torch.set_num_threads(2)
    outcome = check()
    print(json.dumps(outcome, sort_keys=True))
    if outcome["status"] != "PASS":
        raise SystemExit(1)

