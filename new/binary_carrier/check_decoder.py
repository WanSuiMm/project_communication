"""Small deterministic CPU fixtures for the binary carrier decoder."""
from __future__ import annotations

import numpy as np

from decoder import apply, fit, predict, source_only


def run_checks() -> dict:
    rng = np.random.default_rng(1701)
    maps, height, width = 6, 4, 5
    open_cells = np.ones((maps, height, width), dtype=bool)
    open_cells[:, 0, 0] = False
    labels = np.zeros((maps, height, width), dtype=np.float64)
    for m in range(maps):
        labels[m] = (np.indices((height, width)).sum(axis=0) + m) % 2
    labels[~open_cells] = 0
    w = rng.normal(0.0, 0.05, size=(maps, 24, height, width))
    w[:, 0] = np.where(labels > 0.5, 1.0, -1.0)
    # Give walls distinct values so exact preservation is easy to detect.
    w[:, :, 0, 0] = (100.0 + np.arange(maps))[:, None]
    y = labels[:, None]
    mask = open_cells[:, None].astype(np.float64)

    fitted = fit(w, y, mask)
    if not fitted["converged"]:
        raise AssertionError(f"decoder did not converge: {fitted['status']}")
    bits = predict(w, fitted)
    accuracy = float(np.mean(bits[open_cells] == labels[open_cells].astype(bool)))
    templates = fitted["templates"]
    if accuracy < 0.99 or fitted["raw_coefficient"][0] <= 0.0:
        raise AssertionError("decoder failed the separable calibration fixture")
    if templates["p0"][0] >= templates["p1"][0]:
        raise AssertionError("weighted calibration centroids have the wrong class order")
    if fitted["counts"]["map_cues"] != maps or not np.isclose(
        fitted["sample_weighting"]["global_weight_sum"], 1.0
    ):
        raise AssertionError("map/cue weighting metadata is inconsistent")

    w32 = w.astype(np.float32)
    reconstruction = apply(w32, bits, mask.astype(np.float32), fitted)
    if reconstruction.dtype != np.float32:
        raise AssertionError("apply must preserve the input W dtype")
    reconstructed_last = reconstruction.transpose(0, 2, 3, 1)
    original_last32 = w32.transpose(0, 2, 3, 1)
    if not np.array_equal(reconstructed_last[~open_cells], original_last32[~open_cells]):
        raise AssertionError("apply changed a wall value")
    p1_actual = reconstructed_last[open_cells & bits]
    p0_actual = reconstructed_last[open_cells & ~bits]
    np.testing.assert_array_equal(
        p1_actual, np.broadcast_to(templates["p1"].astype(np.float32), p1_actual.shape)
    )
    np.testing.assert_array_equal(
        p0_actual, np.broadcast_to(templates["p0"].astype(np.float32), p0_actual.shape)
    )
    if not np.array_equal(apply(reconstruction, bits, mask, fitted), reconstruction):
        raise AssertionError("template reconstruction must be idempotent for fixed bits")
    template_state = np.stack((templates["p0"], templates["p1"])).astype(np.float32)[:, :, None, None]
    template_predictions = predict(template_state, fitted)[:, 0, 0]
    if not np.array_equal(template_predictions, np.asarray([False, True])):
        raise AssertionError("fitted class templates must predict their own binary labels")

    x = np.zeros((maps, 3, height, width), dtype=np.float64)
    x[:, 0] = open_cells
    x[:, 1, 1, 1] = 1.0
    x[:, 2, 2, 2] = 1.0
    # source_only has no y argument: labels cannot enter its construction path.
    baseline = source_only(w32, x.astype(np.float32), fitted)
    if baseline.dtype != np.float32:
        raise AssertionError("source_only must preserve the input W dtype")
    baseline_last = baseline.transpose(0, 2, 3, 1)
    midpoint = 0.5 * (templates["p0"] + templates["p1"])
    np.testing.assert_allclose(
        baseline_last[:, 1, 1], np.broadcast_to(templates["p1"], (maps, 24)), atol=1e-12
    )
    np.testing.assert_allclose(
        baseline_last[:, 2, 2], np.broadcast_to(templates["p0"], (maps, 24)), atol=1e-12
    )
    np.testing.assert_allclose(
        baseline_last[:, 1, 2], np.broadcast_to(midpoint, (maps, 24)), atol=1e-12
    )
    if not np.array_equal(baseline_last[~open_cells], original_last32[~open_cells]):
        raise AssertionError("source_only changed a wall value")

    tie_fit = {"raw_coefficient": np.zeros(24), "raw_intercept": 0.0}
    if not predict(w, tie_fit).all():
        raise AssertionError("predict must classify an exact zero score as positive")
    repeated = fit(w, y, mask)
    np.testing.assert_array_equal(repeated["raw_coefficient"], fitted["raw_coefficient"])
    np.testing.assert_array_equal(repeated["loss_trace"], fitted["loss_trace"])
    return {
        "status": "PASS",
        "maps_and_cues": maps,
        "open_accuracy": accuracy,
        "fit_iterations": fitted["iterations"],
        "gradient_inf_norm": fitted["gradient_inf_norm"],
        "wall_preservation": True,
        "source_only_uses_x_only": True,
        "zero_score_is_positive": True,
        "float32_output_preserved": True,
        "template_prediction": template_predictions.tolist(),
        "template_reconstruction_idempotent": True,
    }


if __name__ == "__main__":
    print(run_checks())
