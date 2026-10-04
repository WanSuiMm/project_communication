"""Small deterministic CPU checks for the pure NumPy W interventions."""

from __future__ import annotations

import json

import numpy as np

if __package__:
    from .transforms import (
        CHANNELS,
        apply_channel_gather,
        apply_spatial_gather,
        channel_permute,
        cue_swap,
        lane_permute,
        norm_match,
        spatial_scramble,
    )
else:
    from transforms import (
        CHANNELS,
        apply_channel_gather,
        apply_spatial_gather,
        channel_permute,
        cue_swap,
        lane_permute,
        norm_match,
        spatial_scramble,
    )


def _expect_error(function, exception_types=(ValueError, TypeError)) -> None:
    try:
        function()
    except exception_types:
        return
    raise AssertionError("malformed input was not rejected")


def _rms_by_row(W: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(W.astype(np.float64)), axis=(1, 2, 3)))


def run_checks() -> dict[str, object]:
    rng = np.random.default_rng(20261004)
    maps, height, width = 2, 5, 6
    W = rng.normal(size=(2 * maps, CHANNELS, height, width)).astype(np.float32)
    reference = rng.normal(size=W.shape).astype(np.float32)
    reference[:maps] *= np.asarray([1.7, 0.6], dtype=np.float32)[:, None, None, None]
    reference[maps:] *= np.asarray([0.8, 2.2], dtype=np.float32)[:, None, None, None]

    open_mask = np.zeros((maps, 1, height, width), dtype=bool)
    open_mask[0, 0, :3, :4] = True
    open_mask[1, 0, 1:5, 2:6] = True

    checks: dict[str, bool] = {}
    matched, norm_record = norm_match(W, reference, open_mask)
    paired_open = np.concatenate((open_mask[:, 0], open_mask[:, 0]), axis=0)[:, None]
    matched_open_rms = []
    reference_open_rms = []
    for batch_index in range(2 * maps):
        selected = paired_open[batch_index].astype(bool)
        matched_values = matched[batch_index][:, selected[0]]
        reference_values = reference[batch_index][:, selected[0]]
        matched_open_rms.append(np.sqrt(np.mean(np.square(matched_values.astype(np.float64)))))
        reference_open_rms.append(np.sqrt(np.mean(np.square(reference_values.astype(np.float64)))))
    checks["norm_match_open_rms"] = bool(np.allclose(matched_open_rms, reference_open_rms, rtol=2e-6, atol=1e-7))
    checks["norm_match_records_per_map_variant"] = bool(
        norm_record["alpha_by_map_variant"].shape == (maps, 2)
        and norm_record["closed_cells_scaled"] is True
        and np.allclose(norm_record["alpha_by_batch"], norm_record["alpha_by_map_variant"].T.reshape(-1))
    )
    expected_single_cast = (
        W.astype(np.float64)
        * norm_record["alpha_by_batch"][:, None, None, None]
    ).astype(np.float32)
    early_rounded_scale = (
        W * norm_record["alpha_by_batch"].astype(np.float32)[:, None, None, None]
    ).astype(np.float32)
    checks["norm_match_single_float64_multiply_then_float32_cast"] = bool(
        np.array_equal(matched, expected_single_cast)
    )
    checks["single_cast_fixture_detects_early_scale_rounding"] = bool(
        np.any(expected_single_cast != early_rounded_scale)
    )
    closed = ~paired_open
    checks["norm_match_scales_closed_cells"] = bool(
        np.allclose(
            matched[closed.repeat(CHANNELS, axis=1)].reshape(-1),
            (W * norm_record["alpha_by_batch"][:, None, None, None])[closed.repeat(CHANNELS, axis=1)].reshape(-1),
            rtol=1e-6,
            atol=1e-7,
        )
    )

    permuted_channels, channel_record = channel_permute(W)
    restored_channels = apply_channel_gather(
        permuted_channels, channel_record["inverse_source_channel_indices"]
    )
    channel_permutation = np.asarray(channel_record["source_channel_indices"])
    checks["channel_permutation_bijective_and_nonidentity"] = bool(
        np.array_equal(np.sort(channel_permutation), np.arange(CHANNELS))
        and not np.array_equal(channel_permutation, np.arange(CHANNELS))
        and np.array_equal(
            channel_permutation.reshape(4, 6) % 6,
            np.tile(np.asarray(channel_record["payload_permutation"]), (4, 1)),
        )
    )
    checks["channel_inverse_exact"] = bool(np.array_equal(restored_channels, W))
    checks["channel_permutation_preserves_row_norm"] = bool(
        np.allclose(_rms_by_row(permuted_channels), _rms_by_row(W), rtol=1e-7, atol=1e-7)
    )

    permuted_lanes, lane_record = lane_permute(W)
    restored_lanes = apply_channel_gather(
        permuted_lanes, lane_record["inverse_source_channel_indices"]
    )
    checks["lane_permutation_bijective_and_nonidentity"] = bool(
        np.array_equal(
            np.sort(np.asarray(lane_record["lane_permutation"])), np.arange(4)
        )
        and lane_record["lane_permutation"] != tuple(range(4))
    )
    checks["lane_inverse_exact"] = bool(np.array_equal(restored_lanes, W))
    checks["lane_permutation_preserves_row_norm"] = bool(
        np.allclose(_rms_by_row(permuted_lanes), _rms_by_row(W), rtol=1e-7, atol=1e-7)
    )

    changed = np.zeros((maps, height, width), dtype=bool)
    changed[0, 0:2, 0:3] = True  # one six-cell component
    changed[0, 4, 5] = True  # singleton remains fixed
    changed[1, 2, 1:5] = True  # a separate four-cell component
    scrambled, spatial_record = spatial_scramble(W, changed, seed=17)
    source = spatial_record["source_indices"]
    inverse = spatial_record["inverse_source_indices"]
    restored_spatial = apply_spatial_gather(scrambled, inverse)
    identity = np.arange(height * width)
    changed_flat = changed.reshape(maps, -1)
    original_flat = W.reshape(2 * maps, CHANNELS, -1)
    scrambled_flat = scrambled.reshape(2 * maps, CHANNELS, -1)
    checks["spatial_inverse_exact"] = bool(np.array_equal(restored_spatial, W))
    checks["spatial_not_identity_per_map"] = bool(
        all(np.any(source[m] != identity) for m in range(maps))
    )
    outside_unchanged = True
    shared_gather_correct = True
    for map_index in range(maps):
        outside = ~changed_flat[map_index]
        for batch_index in (map_index, map_index + maps):
            outside_unchanged &= np.array_equal(
                scrambled_flat[batch_index][:, outside], original_flat[batch_index][:, outside]
            )
            shared_gather_correct &= np.array_equal(
                scrambled_flat[batch_index], original_flat[batch_index][:, source[map_index]]
            )
    checks["spatial_outside_unchanged_exact"] = bool(outside_unchanged)
    checks["spatial_original_flip_mapping_shared"] = bool(
        shared_gather_correct and spatial_record["permutation_shared_by_original_and_flip"]
        and spatial_record["permutation_shared_by_all_24_channels"]
    )
    component_labels = np.full((maps, height * width), -1, dtype=np.int64)
    component_labels[0, np.ravel_multi_index(np.mgrid[0:2, 0:3], (height, width)).reshape(-1)] = 0
    component_labels[0, 4 * width + 5] = 1
    component_labels[1, np.arange(2 * width + 1, 2 * width + 5)] = 0
    stays_in_component = True
    for map_index in range(maps):
        for destination in np.flatnonzero(changed_flat[map_index]):
            source_cell = source[map_index, destination]
            stays_in_component &= (
                component_labels[map_index, destination]
                == component_labels[map_index, source_cell]
            )
    checks["spatial_stays_inside_changed_components"] = bool(stays_in_component)
    checks["spatial_preserves_row_norm"] = bool(
        np.allclose(_rms_by_row(scrambled), _rms_by_row(W), rtol=1e-7, atol=1e-7)
    )
    checks["spatial_metadata_is_bijective"] = bool(
        all(np.array_equal(np.sort(row), identity) for row in source)
        and np.array_equal(inverse, np.argsort(source, axis=1))
    )

    swapped, swap_record = cue_swap(W)
    restored_swap, _ = cue_swap(swapped)
    checks["cue_swap_involution"] = bool(
        swap_record["inverse_is_self"] and np.array_equal(restored_swap, W)
    )

    malformed_count = 0
    malformed = [
        lambda: norm_match(W.astype(np.float64), reference, open_mask),
        lambda: norm_match(W, reference, np.ones((maps, height, width), dtype=bool)),
        lambda: channel_permute(W, (0, 1, 2, 3, 4, 5)),
        lambda: lane_permute(W, (0, 0, 2, 3)),
        lambda: spatial_scramble(W, changed.astype(np.uint8), seed=17),
        lambda: spatial_scramble(W, np.zeros_like(changed), seed=17),
    ]
    for case in malformed:
        _expect_error(case)
        malformed_count += 1
    zero_donor = W.copy()
    zero_donor[0][:, open_mask[0, 0]] = 0.0
    _expect_error(lambda: norm_match(zero_donor, reference, open_mask))
    malformed_count += 1
    checks["malformed_inputs_rejected"] = malformed_count == len(malformed) + 1

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "malformed_cases_rejected": malformed_count,
        "max_open_rms_abs_error": float(
            np.max(np.abs(np.asarray(matched_open_rms) - np.asarray(reference_open_rms)))
        ),
        "spatial_nontrivial_components": spatial_record["nontrivial_component_count"],
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, sort_keys=True))
