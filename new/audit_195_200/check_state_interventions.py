"""Small deterministic CPU fixtures for 195/200 state interventions."""

from __future__ import annotations

import numpy as np
import torch

from state_interventions import (
    apply_swap,
    apply_z_projection_swap,
    build_swaps,
    identity_plan,
    open_degree,
    paired_correct,
    pair_margin,
    transplant_between,
)


def _bank(batch: int = 2, height: int = 9, width: int = 9):
    opened = np.ones((batch, 1, height, width), dtype=bool)
    changed = np.ones_like(opened)
    distance = np.full((batch, 1, height, width), 20, dtype=np.int16)
    # Walls and source cells must never enter the target set.
    opened[:, :, 0, 0] = False
    changed[:, :, 0, 0] = False
    distance[:, :, 0, 0] = -1
    distance[0, 0, 4, 4] = 0
    changed[0, 0, 4, 4] = False
    y = np.zeros((batch, 1, height, width), dtype=np.float32)
    y_flip = np.ones_like(y)
    return {
        "x": np.zeros((batch, 3, height, width), dtype=np.float32),
        "y": y,
        "y_flip": y_flip,
        "mask": opened,
        "changed": changed,
        "distance": distance,
    }


def _state(batch: int, height: int, width: int, offset: float = 0.0):
    sites = np.arange(batch * height * width, dtype=np.float32).reshape(batch, 1, height, width)
    w = np.broadcast_to(sites + np.arange(24, dtype=np.float32)[None, :, None, None] / 10 + offset,
                        (batch, 24, height, width)).copy()
    z = np.broadcast_to(sites + np.arange(8, dtype=np.float32)[None, :, None, None] / 20 + offset,
                        (batch, 8, height, width)).copy()
    return w, z


def _coords(flat: np.ndarray, height: int, width: int):
    return (flat // (height * width), (flat // width) % height, flat % width)


def check():
    bank = _bank()
    batch, _, height, width = bank["x"].shape
    shape = (batch, height, width)
    correct = np.ones(shape, dtype=bool)
    margin = np.full(shape, 0.1, dtype=np.float32)
    age = np.full(shape, 5, dtype=np.int16)

    # Four-neighbor degree uses open cells only and treats the map boundary as closed.
    degree = open_degree(bank["mask"])
    assert degree.shape == shape and degree[0, 0, 0] == 2
    assert degree[0, 4, 4] == 4

    plan = build_swaps(bank, correct, margin, age, "solved", seed=37)
    repeat = build_swaps(bank, correct, margin, age, "solved", seed=37)
    assert np.array_equal(plan.target_flat, repeat.target_flat)
    assert np.array_equal(plan.donor_flat, repeat.donor_flat)
    assert plan.counts["eligible_cells"] > plan.counts["swap_targets"] > 0
    assert plan.counts["swap_targets"] <= 2 * 8 * batch
    assert plan.counts["pairs"] * 2 == plan.counts["swap_targets"]
    assert plan.counts["unmatched_cells"] == (
        plan.counts["eligible_cells"] - plan.counts["swap_targets"]
    )
    assert all(row["swap_targets"] <= 16 for row in plan.counts["per_map"])
    assert np.all(plan.target_flat != plan.donor_flat)
    assert set(plan.target_flat.tolist()) == set(plan.donor_flat.tolist())
    mapping = dict(zip(plan.target_flat.tolist(), plan.donor_flat.tolist()))
    assert all(mapping[mapping[target]] == target for target in mapping)
    target_maps, _, _ = _coords(plan.target_flat, height, width)
    donor_maps, _, _ = _coords(plan.donor_flat, height, width)
    assert np.array_equal(target_maps, donor_maps)
    assert not plan.mask[0, 0, 0] and not plan.mask[0, 4, 4]
    assert np.all(bank["changed"][:, 0][plan.mask])
    assert np.all(bank["distance"][:, 0][plan.mask] > 0)
    assert np.all(bank["mask"][:, 0][plan.mask])

    full_plan = build_swaps(bank, correct, margin, age, "solved", seed=37,
                            max_pairs_per_map=None)
    assert full_plan.counts["pairs"] == full_plan.counts["candidate_pairs_before_cap"]
    assert full_plan.counts["swap_targets"] == 2 * full_plan.counts["pairs"]
    zero_plan = build_swaps(bank, correct, margin, age, "solved", seed=37,
                            max_pairs_per_map=0)
    assert zero_plan.counts["swap_targets"] == 0
    assert zero_plan.counts["unmatched_cells"] == zero_plan.counts["eligible_cells"]

    original = _state(batch, height, width)
    flipped = _state(batch, height, width, offset=10000.0)
    original_swapped = apply_swap(original, plan)
    flipped_swapped = apply_swap(flipped, plan)
    assert np.array_equal(original[0], _state(batch, height, width)[0])
    for base, changed_state in zip(original, original_swapped):
        assert np.array_equal(base[0, :, 0, 0], changed_state[0, :, 0, 0])
    assert np.array_equal(original[0][0, :, 4, 4], original_swapped[0][0, :, 4, 4])
    assert np.array_equal(original[1][0, :, 4, 4], original_swapped[1][0, :, 4, 4])
    # Every target receives its planned donor vector; the same geometry is used
    # for both source worlds, and the selected-cell multiset is preserved.
    target_w = _coords(plan.target_flat, height, width)
    donor_w = _coords(plan.donor_flat, height, width)
    for base, result in ((original, original_swapped), (flipped, flipped_swapped)):
        for component_index, (source, output) in enumerate(zip(base, result)):
            for channel in range(source.shape[1]):
                assert np.array_equal(
                    output[target_w[0], channel, target_w[1], target_w[2]],
                    source[donor_w[0], channel, donor_w[1], donor_w[2]],
                )
                before = source.transpose(0, 2, 3, 1)[plan.mask]
                after = output.transpose(0, 2, 3, 1)[plan.mask]
                assert np.array_equal(np.sort(before[:, channel]), np.sort(after[:, channel]))
    # The two worlds use exactly the same target and donor indices.
    assert np.array_equal(original_swapped[0][target_w[0], 0, target_w[1], target_w[2]],
                          original[0][donor_w[0], 0, donor_w[1], donor_w[2]])
    assert np.array_equal(flipped_swapped[0][target_w[0], 0, target_w[1], target_w[2]],
                          flipped[0][donor_w[0], 0, donor_w[1], donor_w[2]])

    sham = identity_plan(shape)
    sham_state = apply_swap(original, sham)
    assert np.array_equal(sham_state[0], original[0])
    assert np.array_equal(sham_state[1], original[1])

    # Component/channel selection cannot affect unselected state coordinates.
    lane = apply_swap(original, plan, component="W", channels=range(6, 12))
    assert np.array_equal(lane[1], original[1])
    lane_diff = lane[0] != original[0]
    expected_lane = np.zeros_like(lane_diff)
    expected_lane[:, 6:12] = plan.mask[:, None]
    assert np.all(~lane_diff | expected_lane)
    z_channels = apply_swap(original, plan, component="Z", channels=[2, 4])
    assert np.array_equal(z_channels[0], original[0])
    z_diff = z_channels[1] != original[1]
    expected_z = np.zeros_like(z_diff)
    expected_z[:, [2, 4]] = plan.mask[:, None]
    assert np.all(~z_diff | expected_z)

    # Same-location cross-checkpoint transplant changes only its supplied cohort.
    donor_state = _state(batch, height, width, offset=50000.0)
    transplant_mask = plan.mask.copy()
    transplanted = transplant_between(original, donor_state, transplant_mask,
                                       component="Z", channels=[1])
    assert np.array_equal(transplanted[0], original[0])
    trans_diff = transplanted[1] != original[1]
    expected_trans = np.zeros_like(trans_diff)
    expected_trans[:, 1] = transplant_mask
    assert np.array_equal(trans_diff, expected_trans)
    assert np.array_equal(transplanted[1][0, 1][transplant_mask[0]],
                          donor_state[1][0, 1][transplant_mask[0]])

    # The frontier selector only emits wrong cells with a correct, changed,
    # open, non-source four-neighbor. Two compatible cells form one pair.
    frontier_correct = correct.copy()
    frontier_correct[0, 3, 3] = False
    frontier_correct[0, 3, 5] = False
    frontier = build_swaps(bank, frontier_correct, margin, age,
                           "frontier", seed=5)
    assert frontier.counts["swap_targets"] == 2
    assert set(frontier.target_flat.tolist()) == {
        3 * width + 3,
        3 * width + 5,
    }

    # A singleton stratum is reported and skipped.
    one_bank = _bank(batch=1, height=5, width=5)
    one_bank["mask"][:] = False
    one_bank["changed"][:] = False
    one_bank["distance"][:] = -1
    one_bank["mask"][0, 0, 2, 2] = True
    one_bank["changed"][0, 0, 2, 2] = True
    one_bank["distance"][0, 0, 2, 2] = 10
    one_correct = np.zeros((1, 5, 5), dtype=bool)
    one_correct[0, 2, 2] = True
    one_plan = build_swaps(one_bank, one_correct, np.zeros_like(one_correct),
                           np.zeros_like(one_correct, dtype=np.int16),
                           "solved", seed=0)
    assert one_plan.counts["eligible_cells"] == 1
    assert one_plan.counts["swap_targets"] == 0
    assert one_plan.counts["unmatched_cells"] == 1
    assert one_plan.counts["singleton_skipped"] == 1

    # Readout-relative null swaps preserve the linear readout while changing Z;
    # span swaps carry the projected component. These coordinates are defined
    # only by this supplied readout vector.
    torch_state = (
        torch.randn((batch, 24, height, width), dtype=torch.float32),
        torch.randn((batch, 8, height, width), dtype=torch.float32),
    )
    readout = torch.tensor([0.5, -0.2, 0.8, 1.1, -0.7, 0.3, 0.4, -0.9])
    null_state = apply_z_projection_swap(torch_state, plan, readout, subspace="null")
    span_state = apply_z_projection_swap(torch_state, plan, readout, subspace="span")
    assert torch.equal(null_state[0], torch_state[0])
    assert torch.equal(span_state[0], torch_state[0])
    null_delta = null_state[1] - torch_state[1]
    span_delta = span_state[1] - torch_state[1]
    null_logits = torch.einsum("c,bchw->bhw", readout, null_state[1])
    base_logits = torch.einsum("c,bchw->bhw", readout, torch_state[1])
    assert float((null_logits - base_logits).abs().max()) <= 1e-6
    assert float(null_delta.square().sum().sqrt()) > 1e-4
    assert float(span_delta.square().sum().sqrt()) > 1e-4
    direction = readout / readout.norm()
    target_tensor = torch.as_tensor(plan.mask)
    null_target_delta = null_delta.permute(0, 2, 3, 1)[target_tensor]
    span_target_delta = span_delta.permute(0, 2, 3, 1)[target_tensor]
    assert float((null_target_delta @ direction).abs().max()) <= 1e-6
    span_residual = span_target_delta - (span_target_delta @ direction)[:, None] * direction
    assert float(span_residual.abs().max()) <= 1e-6
    assert torch.equal(null_delta.permute(0, 2, 3, 1)[~target_tensor],
                       torch.zeros_like(null_delta.permute(0, 2, 3, 1)[~target_tensor]))

    # Optional paired-logit helpers follow the exact sign convention.
    logits_a = np.array([[[1.0, -2.0]]], dtype=np.float32)
    logits_b = np.array([[[2.0, -1.0]]], dtype=np.float32)
    labels_a = np.array([[[1, 0]]], dtype=np.float32)
    labels_b = np.array([[[1, 1]]], dtype=np.float32)
    assert np.array_equal(paired_correct(logits_a, logits_b, labels_a, labels_b),
                          np.array([[[True, False]]]))
    assert np.array_equal(pair_margin(logits_a, logits_b, labels_a, labels_b),
                          np.array([[[1.0, -1.0]]], dtype=np.float32))

    print({
        "status": "PASS",
        "solved_eligible": plan.counts["eligible_cells"],
        "solved_pairs": plan.counts["pairs"],
        "solved_unmatched": plan.counts["unmatched_cells"],
        "frontier_pairs": frontier.counts["pairs"],
        "singleton_skipped": one_plan.counts["singleton_skipped"],
        "null_readout_max_abs_error": float((null_logits - base_logits).abs().max()),
    })


if __name__ == "__main__":
    check()

