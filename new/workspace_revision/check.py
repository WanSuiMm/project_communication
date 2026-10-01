"""Small CPU-only implementation checks for the workspace-revision cells."""
from __future__ import annotations

import json

import torch
from torch import Tensor, nn

from revision_cells import ARMS, RevisionCell, make_cell, masked_laplacian


def _assert_close(actual: Tensor, expected: Tensor, label: str) -> None:
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7), label


def _randomize_final_layers(cell: RevisionCell, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for layer in (cell.f_out, cell.q_out):
            nn.init.normal_(layer.weight, mean=0.0, std=0.04)
            nn.init.normal_(layer.bias, mean=0.0, std=0.03)


def _input_with_mask(height: int = 8, width: int = 8) -> Tensor:
    x = torch.zeros(1, 3, height, width)
    x[:, 0, 1:-1, 1:-1] = 1.0
    x[0, 1, 2, 2] = 1.0
    x[0, 2, height - 3, width - 3] = 1.0
    return x


def _check_initialization_and_first_step() -> None:
    torch.manual_seed(412)
    additive = make_cell("ws_additive")
    rng_after_additive = torch.random.get_rng_state().clone()
    next_additive = torch.rand(5)

    torch.manual_seed(412)
    revision = make_cell("ws_revision")
    rng_after_revision = torch.random.get_rng_state().clone()
    next_revision = torch.rand(5)

    assert torch.equal(rng_after_additive, rng_after_revision), "arms consume different RNG draws"
    assert torch.equal(next_additive, next_revision), "arms leave different RNG states"
    assert additive.state_dict().keys() == revision.state_dict().keys()
    assert all(
        torch.equal(additive.state_dict()[name], revision.state_dict()[name])
        for name in additive.state_dict()
    ), "initial parameters differ across arms"
    for cell in (additive, revision):
        assert sum(parameter.numel() for parameter in cell.parameters()) == 5033
        assert cell.metadata()["parameter_count"] == 5033
        assert cell.metadata()["workspace_input_channels"] == 67
        assert cell.metadata()["candidate_input_channels"] == 67

    # Make F and Q active while retaining the common initialization. With Z=0,
    # the two latent rules must agree on their first macro-step.
    _randomize_final_layers(additive, seed=96)
    _randomize_final_layers(revision, seed=96)
    x = _input_with_mask()
    state_a = additive.initial(x)
    state_r = revision.initial(x)
    assert torch.count_nonzero(state_a[1]) == 0
    assert torch.count_nonzero(state_r[1]) == 0
    after_a = additive.step(state_a, x)
    after_r = revision.step(state_r, x)
    for value_a, value_r in zip(after_a, after_r):
        _assert_close(value_a, value_r, "first zero-Z step differs across arms")
    assert torch.count_nonzero(after_a[0] - state_a[0]) > 0, "F did not update W"
    assert torch.count_nonzero(after_a[1]) > 0, "Q did not update zero-initialized Z"


def _manual_candidate(cell: RevisionCell, state: tuple[Tensor, Tensor], x: Tensor) -> tuple[Tensor, Tensor]:
    workspace, latent = state
    mask = x[:, :1]
    workspace_features = torch.cat(
        (workspace, latent, masked_laplacian(workspace, mask), masked_laplacian(latent, mask), x),
        dim=1,
    )
    force = cell.f_out(torch.tanh(cell.f_in(workspace_features)))
    workspace_new = workspace + 0.1 * force
    candidate_features = torch.cat(
        (
            workspace_new,
            latent,
            masked_laplacian(workspace_new, mask),
            masked_laplacian(latent, mask),
            x,
        ),
        dim=1,
    )
    candidate = cell.q_out(torch.tanh(cell.q_in(candidate_features)))
    return workspace_new, candidate


def _check_update_algebra() -> None:
    x = _input_with_mask()
    h, w = x.shape[-2:]
    workspace = torch.linspace(-0.3, 0.4, 24 * h * w).reshape(1, 24, h, w)
    latent = torch.linspace(-0.2, 0.25, 8 * h * w).reshape(1, 8, h, w)
    state = (workspace, latent)

    for arm in ARMS:
        torch.manual_seed(720)
        cell = make_cell(arm)
        _randomize_final_layers(cell, seed=721)
        with torch.no_grad():
            cell.q_out.bias.fill_(2.0)
        workspace_new, candidate = _manual_candidate(cell, state, x)
        assert candidate.abs().max().item() > 1.0, "candidate appears clipped"

        actual_workspace, actual_latent = cell.step(state, x)
        _assert_close(actual_workspace, workspace_new, f"{arm}: workspace update mismatch")
        if arm == "ws_additive":
            expected_latent = latent + 0.5 * candidate
        else:
            expected_latent = latent + 0.5 * (candidate - latent)
        _assert_close(actual_latent, expected_latent, f"{arm}: latent update mismatch")

    # A candidate equal to the current state is a fixed point for revision.
    fixed = torch.full((1, 1, 1, 1), 1.25)
    fixed_candidate = fixed.clone()
    _assert_close(fixed + 0.5 * (fixed_candidate - fixed), fixed, "revision moved a fixed point")

    # With a constant positive candidate, additive accumulation grows as
    # alpha*n while revision approaches that candidate. This algebraic example
    # rules out an unconditional stability guarantee for additive integration.
    constant_candidate = torch.ones(1)
    additive_state = torch.zeros(1)
    revision_state = torch.zeros(1)
    steps = 256
    for _ in range(steps):
        additive_state = additive_state + 0.5 * constant_candidate
        revision_state = revision_state + 0.5 * (constant_candidate - revision_state)
    assert additive_state.item() == 128.0
    assert revision_state.item() <= 1.0 and revision_state.item() > 0.999999


def _check_masked_component_isolation() -> None:
    x_left = torch.zeros(1, 3, 12, 12)
    x_left[0, 0, 2:9, 1:5] = 1.0
    x_left[0, 0, 2:9, 6:11] = 1.0
    x_left[0, 1, 4, 2] = 1.0
    x_right_source_flip = x_left.clone()
    x_right_source_flip[0, 1, 4, 2] = 0.0
    x_right_source_flip[0, 2, 4, 2] = 1.0

    torch.manual_seed(508)
    cell = make_cell("ws_revision")
    _randomize_final_layers(cell, seed=509)
    first = cell.initial(x_left)
    first_step = cell.step(first, x_left)
    assert torch.count_nonzero(first_step[0] - first[0]) > 0, "F is inactive in isolation check"
    assert torch.count_nonzero(first_step[1]) > 0, "Q is inactive in isolation check"

    final_left = cell.rollout(x_left, steps=5)
    final_flipped = cell.rollout(x_right_source_flip, steps=5)
    right_component = torch.zeros_like(x_left[:, :1], dtype=torch.bool)
    right_component[:, :, 2:9, 6:11] = True
    for left_value, flipped_value in zip(final_left, final_flipped):
        _assert_close(
            left_value.masked_select(right_component),
            flipped_value.masked_select(right_component),
            "source change crossed a disconnected masked component",
        )
    logits_left = cell.logits(final_left)
    logits_flipped = cell.logits(final_flipped)
    _assert_close(
        logits_left.masked_select(right_component),
        logits_flipped.masked_select(right_component),
        "readout changed in the untouched component",
    )


def _assert_finite_nonzero_parameter_gradient(loss: Tensor, cell: RevisionCell, label: str) -> None:
    parameters = tuple(parameter for parameter in cell.parameters() if parameter.requires_grad)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    present = tuple(gradient for gradient in gradients if gradient is not None)
    assert present, f"{label}: no parameter gradients"
    assert all(torch.isfinite(gradient).all() for gradient in present), f"{label}: nonfinite gradient"
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in present), (
        f"{label}: all parameter gradients are zero"
    )


def _check_reach_and_detached_repair_gradients() -> None:
    x = _input_with_mask(8, 8)
    torch.manual_seed(831)
    cell = make_cell("ws_revision")
    _randomize_final_layers(cell, seed=832)

    reached = cell.rollout(x, steps=2)
    reach_loss = cell.logits(reached).square().mean()
    _assert_finite_nonzero_parameter_gradient(reach_loss, cell, "reach branch")

    keep = torch.ones_like(reached[0][:, :1])
    keep[:, :, 2:4, 2:4] = 0.0
    detached_damaged = tuple(value.detach() * keep for value in reached)
    assert all(not value.requires_grad for value in detached_damaged)
    repaired = cell.rollout(x, steps=2, state=detached_damaged)
    repair_loss = (cell.logits(repaired) - 0.25).square().mean()
    _assert_finite_nonzero_parameter_gradient(repair_loss, cell, "detached repair branch")


def _check_readout_uses_only_latent() -> None:
    torch.manual_seed(940)
    cell = make_cell("ws_revision")
    workspace = torch.randn(1, 24, 4, 5, requires_grad=True)
    latent = torch.randn(1, 8, 4, 5, requires_grad=True)
    logits = cell.logits((workspace, latent))
    changed_workspace_logits = cell.logits((workspace + 37.0, latent))
    _assert_close(logits, changed_workspace_logits, "readout depends on W")
    workspace_gradient, latent_gradient = torch.autograd.grad(
        logits.sum(), (workspace, latent), allow_unused=True
    )
    assert workspace_gradient is None, "readout produced a W gradient"
    assert latent_gradient is not None and torch.count_nonzero(latent_gradient) > 0


def run_checks() -> dict:
    _check_initialization_and_first_step()
    _check_update_algebra()
    _check_masked_component_isolation()
    _check_reach_and_detached_repair_gradients()
    _check_readout_uses_only_latent()
    return {
        "status": "PASS",
        "device": "cpu",
        "arms": list(ARMS),
        "default_parameter_count_per_arm": 5033,
        "checks": [
            "same_initial_parameters_and_rng",
            "first_zero_latent_step_agrees",
            "manual_update_algebra_and_raw_candidate",
            "fixed_point_and_unbounded_additive_counterexample",
            "masked_disconnected_component_isolation_with_active_F_and_Q",
            "finite_nonzero_reach_and_detached_repair_gradients",
            "readout_uses_only_Z",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), sort_keys=True, separators=(",", ":")))
