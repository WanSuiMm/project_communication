"""Independent geodesic-distance qualification task for StreamingCell.

This module deliberately imports the unchanged streaming cell and the original
mask sampler. It adds a task and measurement harness only; it does not change
the cell, training recipe, or any earlier evidence.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import json
import math
import os
import sys
import time
from typing import Callable, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
STREAMING_DIR = ROOT / "new" / "streaming_carry"
TASKS_DIR = ROOT / "new" / "nca_inertial_wind_tunnel"
for _path in (str(STREAMING_DIR), str(TASKS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stream_cells import StreamingCell  # noqa: E402
from tasks import components, neighbors, sample  # noqa: E402
try:  # Root's serial runner imports these same cached helpers first.
    from .common import sha, tensor_hash
except ImportError:  # Also support direct import as ``distance_task``.
    from common import sha, tensor_hash


TASK_NAME = "streaming_multisource_geodesic_distance_v1"
TRAIN_SIZE = 32
TRAIN_COUNT = 512
TRAIN_DATA_SEED = 81032
INIT_SEEDS = tuple(range(41001, 41009))
SCHEDULE_SEEDS = tuple(range(42001, 42009))
EVAL_SEEDS = {32: 82032, 64: 82064}
EVAL_COUNT = 32
UPDATES = 300
BATCH_SIZE = 8
FORWARD_STEPS = 64
LOSS_EVERY = 8
EVAL_STEPS = 256
EVAL_CHECKPOINTS = (0, 8, 16, 32, 64, 128, 256)
DISTANCE_SCALE = 32.0
MODEL_PARAMS = 5033


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def oracle_distance(mask: np.ndarray, sources: list[tuple[int, int]]) -> np.ndarray:
    """Exact unweighted multi-source geodesic distance; walls are -1."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not sources:
        raise ValueError("mask must be 2-D and sources must be nonempty")
    n = mask.shape[0]
    if mask.shape != (n, n):
        raise ValueError("only square masks are supported")
    distance = np.full(mask.shape, -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    for y, x in sources:
        if not (0 <= y < n and 0 <= x < n and mask[y, x]):
            raise ValueError(f"source {(y, x)} must lie on an open cell")
        if distance[y, x] < 0:
            distance[y, x] = 0
            queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for yy, xx in neighbors(y, x, n):
            if mask[yy, xx] and distance[yy, xx] < 0:
                distance[yy, xx] = distance[y, x] + 1
                queue.append((yy, xx))
    if np.any(mask & (distance < 0)):
        raise ValueError("every open cell must be connected to a source")
    return distance


def _local_bellman_check(mask: np.ndarray, source_map: np.ndarray,
                         distance: np.ndarray) -> None:
    """Verify source boundary values and the exact local min-plus equation."""
    n = mask.shape[0]
    if np.any(distance[~mask] != -1) or np.any(distance[source_map] != 0):
        raise AssertionError("oracle boundary conditions failed")
    for y, x in zip(*np.nonzero(mask & ~source_map)):
        adjacent = [distance[yy, xx] for yy, xx in neighbors(int(y), int(x), n)
                    if mask[yy, xx]]
        if not adjacent or int(distance[y, x]) != 1 + min(adjacent):
            raise AssertionError("oracle violates the local min-plus equation")


def _minplus_steps(mask: np.ndarray, source_map: np.ndarray,
                   steps: int) -> np.ndarray:
    """Run synchronous unit-cost min-plus wavefront propagation."""
    inf = np.float32(np.inf)
    current = np.where(source_map, 0.0, inf).astype(np.float32)
    current[~mask] = inf
    n = mask.shape[0]
    for _ in range(steps):
        padded = np.pad(current, ((1, 1), (1, 1)),
                        mode="constant", constant_values=inf)
        neighbor_min = np.minimum.reduce((
            padded[:-2, 1:-1], padded[2:, 1:-1],
            padded[1:-1, :-2], padded[1:-1, 2:],
        ))
        updated = np.minimum(current, neighbor_min + 1.0)
        updated[~mask] = inf
        current = updated
    return current


def _make_example(n: int, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], list[list[list[int]]]]:
    """Retain only an original sampled mask; replace all old task semantics."""
    original = sample(n, rng)
    mask = np.asarray(original["mask"][0] > 0.5, dtype=bool)
    _, groups = components(mask)
    source_coords: list[list[list[int]]] = []
    source_map = np.zeros((n, n), dtype=bool)
    for cells in groups:
        if len(cells) < 2:
            raise ValueError("each connected component needs two distinct sources")
        picks = rng.choice(len(cells), size=2, replace=False)
        pair = [[int(cells[int(i)][0]), int(cells[int(i)][1])] for i in picks]
        source_coords.append(pair)
        for y, x in pair:
            source_map[y, x] = True
    distance = oracle_distance(mask, [tuple(p) for group in source_coords for p in group])
    source_count = int(source_map.sum())
    if source_count != 2 * len(groups):
        raise AssertionError("source set must contain exactly two cells per component")
    x = np.zeros((3, n, n), dtype=np.float32)
    x[0] = mask
    x[1] = source_map
    # Channel 2 remains identically zero by design.
    target = np.zeros((1, n, n), dtype=np.float32)
    target[0, mask] = distance[mask].astype(np.float32) / DISTANCE_SCALE
    row = {
        "x": x,
        "mask": mask[None].astype(np.float32),
        "distance": distance[None].astype(np.int32),
        "target": target,
    }
    return row, source_coords


def bank(n: int, count: int, seed: int, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Sample an immutable task bank with the original geometry generator."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    examples = [_make_example(n, rng)[0] for _ in range(count)]
    return {key: torch.from_numpy(np.stack([row[key] for row in examples])).to(device)
            for key in examples[0]}


def source_coordinates(n: int, count: int, seed: int) -> list[list[list[list[int]]]]:
    """Recreate the deterministic source coordinates for provenance output."""
    rng = np.random.default_rng(seed)
    return [_make_example(n, rng)[1] for _ in range(count)]


def validate_oracle_bank(data: Mapping[str, torch.Tensor], steps: int = EVAL_STEPS,
                         exact_minplus: bool = False) -> dict[str, object]:
    """Check BFS, Bellman equations and 256-hop reachability for a whole bank.

    When exact_minplus is true, explicitly run the iterative oracle for every
    map. Otherwise the exact BFS maximum-distance bound proves that 256 local
    propagation steps suffice for this bank.
    """
    masks = _as_numpy(data["mask"])[:, 0] > 0.5
    sources = _as_numpy(data["x"])[:, 1] > 0.5
    distances = _as_numpy(data["distance"])[:, 0]
    max_distance = 0
    checked = 0
    for i, (mask, source_map, distance) in enumerate(zip(masks, sources, distances)):
        source_coords = [tuple(map(int, p)) for p in np.argwhere(source_map)]
        bfs = oracle_distance(mask, source_coords)
        if not np.array_equal(bfs, distance):
            raise AssertionError(f"BFS target mismatch in bank map {i}")
        _local_bellman_check(mask, source_map, distance)
        map_max = int(distance[mask].max(initial=0))
        max_distance = max(max_distance, map_max)
        if map_max > steps:
            raise AssertionError(f"map {i} needs more than {steps} local propagation steps")
        if exact_minplus:
            reached = _minplus_steps(mask, source_map, steps)
            if not np.array_equal(reached[mask], distance[mask].astype(np.float32)):
                raise AssertionError(f"min-plus oracle did not solve bank map {i} in {steps} steps")
            checked += 1
    return {"maps": len(masks), "maps_explicitly_minplus_checked": checked,
            "max_geodesic_distance_cells": max_distance,
            "full_solution_by_hops": steps, "bfs_and_local_bellman": "PASS"}


def run_cpu_checks() -> dict[str, object]:
    """Cheap deterministic algorithm check; no learned model or training."""
    mask = np.zeros((7, 7), dtype=bool)
    mask[1, 1:6] = True
    mask[2:6, 3] = True
    sources = [(1, 1), (5, 3)]
    source_map = np.zeros_like(mask)
    for y, x in sources:
        source_map[y, x] = True
    distance = oracle_distance(mask, sources)
    expected = np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, 0, 1, 2, 3, 4, -1],
        [-1, -1, -1, 3, -1, -1, -1],
        [-1, -1, -1, 2, -1, -1, -1],
        [-1, -1, -1, 1, -1, -1, -1],
        [-1, -1, -1, 0, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ], dtype=np.int32)
    if not np.array_equal(distance, expected):
        raise AssertionError("hand-computed multi-source fixture failed")
    _local_bellman_check(mask, source_map, distance)
    reached = _minplus_steps(mask, source_map, 6)
    if not np.array_equal(reached[mask], distance[mask].astype(np.float32)):
        raise AssertionError("tiny min-plus propagation fixture failed")
    x = np.zeros((1, 3, 7, 7), dtype=np.float32)
    x[0, 0] = mask
    x[0, 1] = source_map
    tiny = {
        "x": torch.from_numpy(x),
        "mask": torch.from_numpy(mask[None, None].astype(np.float32)),
        "distance": torch.from_numpy(distance[None, None]),
        "target": torch.from_numpy(np.where(mask, distance / DISTANCE_SCALE, 0.0)[None, None].astype(np.float32)),
    }
    torch.manual_seed(1001)
    model = StreamingCell(streaming=True)
    if sum(p.numel() for p in model.parameters()) != MODEL_PARAMS:
        raise AssertionError("StreamingCell parameter count changed")
    loss, final_state, cadence = backward_distance(model, tiny, steps=16, loss_every=8)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if cadence["backward_calls"] != 2 or cadence["interior_detach_boundaries"] != 1:
        raise AssertionError("K=8 loss/backward cadence changed")
    if any(not bool(torch.isfinite(grad).all()) for grad in gradients):
        raise AssertionError("tiny distance loss produced a nonfinite gradient")
    if any(value.requires_grad for value in final_state):
        raise AssertionError("both carried states must detach at K=8")
    return {"status": "PASS", "fixture": "hand-computed_two-source_branched_corridor",
            "fixture_hops": 6, "fixture_local_bellman": "PASS",
            "fixture_minplus": "PASS", "tiny_cpu_loss_cadence": cadence,
            "tiny_cpu_loss": float(loss), "tiny_cpu_finite_gradients": len(gradients),
            "both_states_detached": True}


def cpu_check() -> dict[str, object]:
    """API alias used by the shared serial suite driver."""
    return run_cpu_checks()


def _masked_map_mse(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    raw = (pred - target).square() * mask
    return raw.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)


def backward_distance(model: StreamingCell, data: Mapping[str, torch.Tensor],
                      steps: int = FORWARD_STEPS,
                      loss_every: int = LOSS_EVERY) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, int]]:
    """Accumulate equal checkpoint losses and detach both states at K=8."""
    if steps <= 0 or loss_every <= 0 or steps % loss_every:
        raise ValueError("steps must be a positive multiple of loss_every")
    x, target, mask = data["x"], data["target"], data["mask"]
    state = model.initial(x)
    detached_losses: list[torch.Tensor] = []
    backward_calls = 0
    for t in range(1, steps + 1):
        state = model.step(state, x)
        if t % loss_every == 0:
            per_map = _masked_map_mse(model.logits(state), target, mask)
            loss = per_map.mean() / (steps // loss_every)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite distance-task loss")
            # Every K=8 chunk contributes its own backward pass; parameter
            # gradients accumulate until the single optimizer update at T=64.
            loss.backward()
            detached_losses.append(loss.detach() * (steps // loss_every))
            backward_calls += 1
            state = tuple(value.detach() for value in state)
    return torch.stack(detached_losses).mean(), state, {
        "backward_calls": backward_calls,
        "interior_detach_boundaries": max(0, backward_calls - 1),
        "forward_steps": steps,
        "loss_count": backward_calls,
    }


def _data_on_device(data: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=False) for key, value in data.items()}


def _checkpoint_metrics(pred_cells: np.ndarray, distance: np.ndarray,
                        mask: np.ndarray) -> dict[str, object]:
    """Map-balanced and pooled raw-cell metrics at one checkpoint."""
    valid = mask.astype(bool)
    within = (np.abs(pred_cells - distance) <= 1.0) & valid
    maes: list[float] = []
    coverages: list[float] = []
    per_map_counts: list[int] = []
    for i in range(len(valid)):
        cells = valid[i]
        count = int(cells.sum())
        per_map_counts.append(count)
        maes.append(float(np.abs(pred_cells[i][cells] - distance[i][cells]).mean())
                    if count else math.nan)
        coverages.append(float(within[i][cells].mean()) if count else math.nan)
    all_error = np.abs(pred_cells[valid] - distance[valid])
    return {
        "mean_map_mae_cells": float(np.nanmean(maes)),
        "pooled_mae_cells": float(all_error.mean()),
        "mean_map_within1_coverage": float(np.nanmean(coverages)),
        "pooled_within1_coverage": float(within[valid].mean()),
        "per_map_mae_cells": maes,
        "per_map_within1_coverage": coverages,
        "per_map_open_cells": per_map_counts,
    }


@torch.no_grad()
def _evaluate_one_size(model: StreamingCell, fixed_cpu: Mapping[str, torch.Tensor],
                       out: Path, name: str) -> dict[str, object]:
    device = next(model.parameters()).device
    data = _data_on_device(fixed_cpu, device)
    x = data["x"]
    mask = _as_numpy(fixed_cpu["mask"])[:, 0] > 0.5
    distance = _as_numpy(fixed_cpu["distance"])[:, 0]
    source_map = _as_numpy(fixed_cpu["x"])[:, 1] > 0.5
    n = int(x.shape[-1])
    b = len(x)
    trace = np.zeros((EVAL_STEPS + 1, b, n, n), dtype=np.bool_)
    state_norms: dict[str, dict[str, float]] = {}
    curve: dict[str, dict[str, object]] = {}
    state = model.initial(x)
    completed = 0
    for t in range(EVAL_STEPS + 1):
        if not all(bool(torch.isfinite(value).all()) for value in state):
            break
        logits = model.logits(state)
        if not bool(torch.isfinite(logits).all()):
            break
        pred_cells = _as_numpy(logits[:, 0]) * DISTANCE_SCALE
        valid = mask
        trace[t] = (np.abs(pred_cells - distance) <= 1.0) & valid
        if t in EVAL_CHECKPOINTS:
            met = _checkpoint_metrics(pred_cells, distance, valid)
            # The strict d>32 subset is the predeclared size-64 diagnostic;
            # reporting it at size 32 too makes empty support explicit.
            band = valid & (distance > 32)
            band_mae: list[float | None] = []
            band_coverage: list[float | None] = []
            band_counts: list[int] = []
            within = trace[t]
            for i in range(b):
                take = band[i]
                count = int(take.sum())
                band_counts.append(count)
                if count:
                    band_mae.append(float(np.abs(pred_cells[i][take] - distance[i][take]).mean()))
                    band_coverage.append(float(within[i][take].mean()))
                else:
                    band_mae.append(None)
                    band_coverage.append(None)
            band_error = np.abs(pred_cells[band] - distance[band])
            band_within = within[band]
            met["d_gt_32"] = {
                "eligible_maps": sum(count > 0 for count in band_counts),
                "pooled_cells": int(sum(band_counts)),
                "mean_eligible_map_mae_cells": (
                    float(np.mean([v for v in band_mae if v is not None]))
                    if any(v is not None for v in band_mae) else None),
                "pooled_mae_cells": float(band_error.mean()) if len(band_error) else None,
                "mean_eligible_map_within1_coverage": (
                    float(np.mean([v for v in band_coverage if v is not None]))
                    if any(v is not None for v in band_coverage) else None),
                "pooled_within1_coverage": float(band_within.mean()) if len(band_within) else None,
                "per_map_mae_cells": band_mae,
                "per_map_within1_coverage": band_coverage,
                "per_map_cells": band_counts,
            }
            curve[str(t)] = met
            state_norms[str(t)] = {
                "workspace_rms": float(state[0].square().mean().sqrt()),
                "latent_rms": float(state[1].square().mean().sqrt()),
            }
        completed = t
        if t < EVAL_STEPS:
            state = model.step(state, x)
    trace = trace[:completed + 1]
    trace_path = out / f"{name}_size{n}_correct_trace.npz"
    np.savez_compressed(trace_path, correct_within1=trace,
                        distance_cells=distance.astype(np.int16),
                        traversable_mask=mask.astype(np.bool_),
                        source_set=source_map.astype(np.bool_),
                        horizons=np.arange(completed + 1, dtype=np.int16))
    status = "EVALUATED" if completed == EVAL_STEPS and len(curve) == len(EVAL_CHECKPOINTS) else "NONFINITE_EVAL"
    return {
        "status": status,
        "size": n,
        "maps": b,
        "completed_steps": completed,
        "checkpoints": curve,
        "state_norms": state_norms,
        "trace_file": trace_path.name,
        "trace_note": "correct_within1[t,map,y,x] is true iff absolute raw cell error <=1; walls are false",
    }


def _retention(correct_a: np.ndarray, correct_b: np.ndarray,
               mask: np.ndarray) -> dict[str, object]:
    at_a = correct_a & mask
    at_b = correct_b & mask
    denominator = int(at_a.sum())
    numerator = int((at_a & at_b).sum())
    per_map: list[float | None] = []
    for i in range(len(at_a)):
        count = int(at_a[i].sum())
        per_map.append(float((at_a[i] & at_b[i]).sum() / count) if count else None)
    return {
        "retained_correct_pixels": numerator,
        "correct64_pixels": denominator,
        "pooled_retention": numerator / denominator if denominator else None,
        "mean_map_retention": (float(np.mean([v for v in per_map if v is not None]))
                               if any(v is not None for v in per_map) else None),
        "eligible_maps": sum(v is not None for v in per_map),
        "per_map_retention": per_map,
    }


def _trace_arrays(out: Path, name: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(out / f"{name}_size{n}_correct_trace.npz") as data:
        return data["correct_within1"], data["traversable_mask"]


def _gate(evaluation: Mapping[str, object], trace_dir: Path) -> dict[str, object]:
    sizes = evaluation["sizes"]
    checks: dict[str, bool] = {}
    failures: list[str] = []
    s32 = sizes.get("32", {})
    s64 = sizes.get("64", {})
    c32 = s32.get("checkpoints", {})
    c64 = s64.get("checkpoints", {})
    if "64" not in c32:
        checks["size32_mae64_le_2"] = False
        failures.append("size32 checkpoint T64 is missing")
    else:
        value = c32["64"]["mean_map_mae_cells"]
        checks["size32_mae64_le_2"] = value <= 2.0
        if not checks["size32_mae64_le_2"]:
            failures.append(f"size32 mean-map MAE at T64 is {value:.6g} cells (>2)")
    for size, rows in ((32, c32), (64, c64)):
        label = str(size)
        if "64" not in rows or "256" not in rows:
            checks[f"size{size}_mae256_no_worse_than_mae64_plus_0_25"] = False
            checks[f"size{size}_retention_ge_0_95"] = False
            failures.append(f"size{size} T64/T256 retention checkpoints are missing")
            continue
        mae64 = rows["64"]["mean_map_mae_cells"]
        mae256 = rows["256"]["mean_map_mae_cells"]
        checks[f"size{size}_mae256_no_worse_than_mae64_plus_0_25"] = mae256 <= mae64 + 0.25
        if not checks[f"size{size}_mae256_no_worse_than_mae64_plus_0_25"]:
            failures.append(f"size{size} mean-map T256 MAE exceeds T64 by more than 0.25 cells")
        correct64, mask64 = _trace_arrays(trace_dir, str(evaluation["name"]), size)
        retention = _retention(correct64[64], correct64[256], mask64)
        retention_value = retention["pooled_retention"]
        checks[f"size{size}_retention_ge_0_95"] = retention_value is not None and retention_value >= 0.95
        if not checks[f"size{size}_retention_ge_0_95"]:
            failures.append(f"size{size} pooled correct64-to-correct256 retention is empty or below 0.95")
        rows["retention_correct64_to_correct256"] = retention
    band = c64.get("256", {}).get("d_gt_32", {})
    band64 = c64.get("64", {}).get("d_gt_32", {})
    eligible = int(band.get("eligible_maps", 0))
    checks["size64_d_gt_32_eligible_maps_ge_16"] = eligible >= 16
    if not checks["size64_d_gt_32_eligible_maps_ge_16"]:
        failures.append(f"size64 d>32 band has {eligible} eligible maps (<16)")
    mae64, mae256 = band64.get("mean_eligible_map_mae_cells"), band.get("mean_eligible_map_mae_cells")
    checks["size64_d_gt_32_mae_gain_ge_0_5"] = (
        mae64 is not None and mae256 is not None and mae256 <= mae64 - 0.5)
    if not checks["size64_d_gt_32_mae_gain_ge_0_5"]:
        failures.append("size64 d>32 mean-map MAE does not improve by at least 0.5 cells or is empty")
    cov64 = band64.get("mean_eligible_map_within1_coverage")
    cov256 = band.get("mean_eligible_map_within1_coverage")
    checks["size64_d_gt_32_within1_gain_ge_0_05"] = (
        cov64 is not None and cov256 is not None and cov256 >= cov64 + 0.05)
    if not checks["size64_d_gt_32_within1_gain_ge_0_05"]:
        failures.append("size64 d>32 mean-map within-1 coverage gain is below 0.05 or empty")
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "failures": failures,
            "gate_scope": "independent distance-task proxy only; not the original paired-flip Full gate"}


def evaluate_distance(model: StreamingCell, fixed_cpu: Mapping[object, object],
                      out: str | Path, name: str) -> dict[str, object]:
    """Evaluate a full held-out bank (or a size->bank mapping) and save traces."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if fixed_cpu and all(isinstance(k, (str, int)) and isinstance(v, Mapping)
                         for k, v in fixed_cpu.items()):
        banks = {int(k): v for k, v in fixed_cpu.items()}
    else:
        banks = {int(fixed_cpu["x"].shape[-1]): fixed_cpu}  # type: ignore[index,union-attr]
    sizes: dict[str, object] = {}
    for n in sorted(banks):
        sizes[str(n)] = _evaluate_one_size(model, banks[n], out, name)  # type: ignore[arg-type]
    result: dict[str, object] = {"name": name, "status": "EVALUATED", "sizes": sizes}
    if len(sizes) == 2 and all(value["status"] == "EVALUATED" for value in sizes.values()):
        result["gate"] = _gate(result, out)
    if any(value["status"] != "EVALUATED" for value in sizes.values()):
        result["status"] = "NONFINITE_EVAL"
    _atomic_json(out / f"{name}_evaluation.json", result)
    return result


def _write_results(out: Path, arms: list[dict[str, object]],
                   cpu_checks: dict[str, object], bank_checks: dict[str, object],
                   execution_status: str = "COMPLETE",
                   failure: str | None = None) -> dict[str, object]:
    complete = len(arms) == len(INIT_SEEDS) and all(a.get("status") == "COMPLETE" for a in arms)
    gates = [a.get("evaluation", {}).get("gate", {}).get("status") for a in arms]
    training_qualified = 0
    for arm in arms:
        value = (arm.get("evaluation", {}).get("sizes", {}).get("32", {})
                 .get("checkpoints", {}).get("64", {}).get("mean_map_mae_cells"))
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and value <= 2.0:
            training_qualified += 1
    gate_passes = sum(value == "PASS" for value in gates)
    if execution_status != "COMPLETE":
        decision = "INCOMPLETE"
    elif training_qualified == 0:
        decision = "SECOND_TASK_TRAINING_UNQUALIFIED"
    elif gate_passes == 0:
        decision = "NO_SECOND_TASK_LONG_ROLLOUT_QUALIFICATION"
    else:
        decision = "SECOND_TASK_POSITIVE_SCREEN"
    summary = {
        "protocol": TASK_NAME,
        "status": "COMPLETE" if complete and execution_status == "COMPLETE" else execution_status,
        "decision": decision,
        "task_gate_scope": "proxy only; does not qualify the original paired-flip Full gate",
        "arm_count": len(arms),
        "complete_arms": sum(a.get("status") == "COMPLETE" for a in arms),
        "size32_T64_training_qualified_arms": training_qualified,
        "gate_passes": gate_passes,
        "gate_failures": sum(value == "FAIL" for value in gates),
        "cpu_oracle_checks": cpu_checks,
        "map_bank_oracle_checks": bank_checks,
        "failure": failure,
        "arms": arms,
        "interpretation": "All eight arms are baseline-only independent task runs. No trained weights transfer between arms or stages.",
    }
    _atomic_json(out / "summary.json", summary)
    lines = ["# Independent multi-source geodesic-distance qualification",
             "", "All eight arms use the unchanged StreamingCell and fixed recipe. This is an independent task validator/proxy; it does not test a selected Stage-2 recipe and it does not qualify the original paired-flip Full gate.",
             "", f"Execution: {summary['status']}. Decision: {decision}. Completed arms: {summary['complete_arms']}/{summary['arm_count']}. Distance-task proxy passes: {summary['gate_passes']}/{summary['arm_count']}.",
             "", "| Init seed | Updates | Size-32 MAE T64 | Size-64 MAE T64 | Size-64 MAE T256 | d>32 eligible | d>32 MAE T64 → T256 | Gate |",
             "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm in arms:
        ev = arm.get("evaluation", {})
        sizes = ev.get("sizes", {})
        r32 = sizes.get("32", {}).get("checkpoints", {})
        r64 = sizes.get("64", {}).get("checkpoints", {})
        a = r32.get("64", {}).get("mean_map_mae_cells")
        b = r64.get("64", {}).get("mean_map_mae_cells")
        c = r64.get("256", {}).get("mean_map_mae_cells")
        d64 = r64.get("64", {}).get("d_gt_32", {})
        d256 = r64.get("256", {}).get("d_gt_32", {})
        gate = ev.get("gate", {}).get("status", "INCOMPLETE")
        def fmt(value: object) -> str:
            return "—" if value is None else f"{float(value):.3f}"
        lines.append(f"| {arm.get('init_seed')} | {arm.get('completed_updates', 0)} | {fmt(a)} | {fmt(b)} | {fmt(c)} | {d256.get('eligible_maps', 0)} | {fmt(d64.get('mean_eligible_map_mae_cells'))} → {fmt(d256.get('mean_eligible_map_mae_cells'))} | {gate} |")
    lines += ["", "Gate definitions use mean-map MAE and mean-map within-1 coverage unless marked pooled. Correctness is absolute raw error ≤1 cell; retention is pooled correct64 pixels still correct at T256. Empty support fails.",
              "The NPZ files retain correctness for every step 0–256, each map's oracle distance, traversable mask and source set. JSON files retain per-map and pooled metrics, eligibility, state norms and arm curves.",
              "A positive screen is not an architecture GO. No efficacy-based seed, map, update, or checkpoint selection is performed.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run_stage(out: str | Path,
              progress_callback: Callable[[dict[str, object]], None] = lambda _row: None) -> dict[str, object]:
    """Run the fixed eight-arm independent distance task on the active device."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cpu_checks = run_cpu_checks()
    if not torch.cuda.is_available():
        raise RuntimeError("distance qualification requires the sequential GPU suite device")
    device = torch.device("cuda")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    config = {
        "task": TASK_NAME,
        "architecture": "unchanged StreamingCell(streaming=True)",
        "model": {"workspace_channels": 24, "latent_channels": 8,
                  "parameter_count": MODEL_PARAMS},
        "train": {"size": TRAIN_SIZE, "maps": TRAIN_COUNT,
                  "data_seed": TRAIN_DATA_SEED, "init_seeds": list(INIT_SEEDS),
                  "schedule_seeds": list(SCHEDULE_SEEDS), "updates": UPDATES,
                  "batch_size": BATCH_SIZE, "sampling": "with replacement",
                  "forward_steps": FORWARD_STEPS, "loss_steps": list(range(8, 65, 8)),
                  "gradient_horizon": 8, "both_state_tensors_detached_at_each_loss_step": True,
                  "loss": "per-map masked MSE on distance/32; average maps equally and eight checkpoints equally"},
        "optimizer": {"type": "AdamW", "lr": 0.001, "weight_decay": 0.0001,
                      "clip_norm": 1.0},
        "evaluation": {"maps_per_size": EVAL_COUNT, "seeds": {str(k): v for k, v in EVAL_SEEDS.items()},
                       "sizes": [32, 64], "steps": EVAL_STEPS,
                       "reported_checkpoints": list(EVAL_CHECKPOINTS),
                       "full_correctness_trace_steps": [0, EVAL_STEPS],
                       "target_scale": DISTANCE_SCALE,
                       "within1_cell_definition": "absolute raw prediction error <= 1 cell",
                       "band": "strict d>32; eligible map has at least one band cell"},
        "gate": {"size32_mean_map_mae_T64_le_cells": 2.0,
                 "both_sizes_mean_map_mae_T256_le_T64_plus_cells": 0.25,
                 "both_sizes_pooled_correct64_to_correct256_retention_ge": 0.95,
                 "size64_d_gt_32_mean_eligible_map_mae_gain_ge_cells": 0.5,
                 "size64_d_gt_32_mean_eligible_map_coverage_gain_ge": 0.05,
                 "size64_d_gt_32_min_eligible_maps": 16,
                 "empty_groups_fail": True},
        "claim_boundary": "Independent baseline-only task test. It estimates existence/frequency of this behavior on another task; no weight transfer and no qualification of the original paired-flip Full gate.",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "backend": {"cudnn_benchmark": False, "cudnn_deterministic": False,
                    "cudnn_tf32": True, "matmul_tf32": False},
    }
    _atomic_json(out / "manifest.json", config)
    status = {"status": "RUNNING", "phase": "sampling", "pid": os.getpid(),
              "completed_arms": 0, "started_unix": time.time()}
    _atomic_json(out / "status.json", status)
    progress_callback(dict(status))

    train_cpu = bank(TRAIN_SIZE, TRAIN_COUNT, TRAIN_DATA_SEED)
    train_check = validate_oracle_bank(train_cpu, exact_minplus=False)
    eval_cpu = {n: bank(n, EVAL_COUNT, EVAL_SEEDS[n]) for n in (32, 64)}
    eval_checks = {str(n): validate_oracle_bank(d, exact_minplus=True) for n, d in eval_cpu.items()}
    config["train_data_sha256"] = tensor_hash(train_cpu)
    config["evaluation_data_sha256"] = {str(n): tensor_hash(data) for n, data in eval_cpu.items()}
    _atomic_json(out / "manifest.json", config)
    bank_checks = {"train": train_check, "evaluation": eval_checks,
                   "full_solution_statement": "All maps have exact BFS max distance <=256; held-out banks were also explicitly propagated by min-plus for 256 steps."}
    _atomic_json(out / "oracle_checks.json", bank_checks)
    train_device = _data_on_device(train_cpu, device)
    _atomic_json(out / "source_coordinates.json", {
        str(n): source_coordinates(n, EVAL_COUNT, EVAL_SEEDS[n]) for n in (32, 64)})
    arms: list[dict[str, object]] = []
    checkpoint_dir = out / "checkpoints"
    curves_dir = out / "curves"
    checkpoint_dir.mkdir(exist_ok=True)
    curves_dir.mkdir(exist_ok=True)
    for arm_index, (init_seed, schedule_seed) in enumerate(zip(INIT_SEEDS, SCHEDULE_SEEDS), start=1):
        torch.manual_seed(init_seed)
        model = StreamingCell(streaming=True).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != MODEL_PARAMS or model.workspace_channels != 24 or model.latent_channels != 8:
            raise AssertionError("StreamingCell architecture no longer matches frozen W24/Z8, 5033-param contract")
        initial_parameter_hash = tensor_hash(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        rng = np.random.default_rng(schedule_seed)
        schedule = rng.integers(0, TRAIN_COUNT, size=(UPDATES, BATCH_SIZE), dtype=np.int64)
        schedule_path = out / f"schedule_seed{schedule_seed}.npy"
        np.save(schedule_path, schedule, allow_pickle=False)
        schedule_sha256 = sha(schedule_path)
        arm_id = f"seed{init_seed}"
        curve_path = curves_dir / f"{arm_id}_training.json"
        arm_row: dict[str, object] = {
            "arm_id": arm_id, "init_seed": init_seed, "schedule_seed": schedule_seed,
            "status": "TRAINING", "completed_updates": 0, "parameter_count": parameter_count,
            "initial_parameter_sha256": initial_parameter_hash,
            "schedule_sha256": schedule_sha256,
            "training_curve_file": str(curve_path.relative_to(out)),
            "checkpoint_file": str((checkpoint_dir / f"{arm_id}.pt").relative_to(out)),
        }
        arm_started = time.perf_counter()
        training_curve: list[dict[str, object]] = []
        numerical_failure: str | None = None
        for update in range(UPDATES):
            indices = schedule[update].tolist()
            minibatch = {key: value[indices] for key, value in train_device.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, final_state, cadence = backward_distance(model, minibatch)
            if not all(bool(torch.isfinite(value).all()) for value in final_state):
                numerical_failure = f"nonfinite carried state after attempted update {update + 1}"
                arm_row.update({"status": "NONFINITE_STATE", "failed_update": update + 1})
                break
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(grad_norm)):
                numerical_failure = f"nonfinite gradient norm at attempted update {update + 1}"
                arm_row.update({"status": "NONFINITE_GRADIENT", "failed_update": update + 1})
                break
            optimizer.step()
            if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                arm_row.update({"status": "NONFINITE_PARAMETER",
                                "completed_updates": update + 1,
                                "failed_update": update + 1})
                training_curve.append({"update": update + 1, "loss": float(loss),
                                       "gradient_norm_before_clip": float(grad_norm),
                                       "cadence": cadence, "status": "NONFINITE_PARAMETER"})
                numerical_failure = f"nonfinite parameter after optimizer update {update + 1}"
                break
            training_curve.append({"update": update + 1, "loss": float(loss),
                                   "gradient_norm_before_clip": float(grad_norm),
                                   "cadence": cadence})
            arm_row["completed_updates"] = update + 1
            if (update + 1) % 10 == 0 or update + 1 == UPDATES:
                _atomic_json(curve_path, training_curve)
                status = {"status": "RUNNING", "phase": "training", "arm": arm_id,
                          "completed_updates": update + 1, "total_updates": UPDATES,
                          "completed_arms": len(arms), "current_arm_index": arm_index,
                          "pid": os.getpid()}
                _atomic_json(out / "status.json", status)
                progress_callback(dict(status))
        _atomic_json(curve_path, training_curve)
        arm_row["training_seconds"] = time.perf_counter() - arm_started
        if arm_row["status"] == "TRAINING" and arm_row["completed_updates"] == UPDATES:
            arm_row["status"] = "TRAINED"
        checkpoint_path = checkpoint_dir / f"{arm_id}.pt"
        final_parameter_hash = tensor_hash(model.state_dict())
        torch.save({"arm_id": arm_id, "init_seed": init_seed,
                    "schedule_seed": schedule_seed, "completed_updates": arm_row["completed_updates"],
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}},
                   checkpoint_path)
        arm_row["final_parameter_sha256"] = final_parameter_hash
        arm_row["checkpoint_sha256"] = sha(checkpoint_path)
        if numerical_failure is not None:
            arms.append(arm_row)
            _atomic_json(out / "arms.json", arms)
            failure = {"status": "NUMERICAL_FAILURE", "arm": arm_id,
                       "failed_update": arm_row.get("failed_update"),
                       "reason": numerical_failure}
            _atomic_json(out / "failure.json", failure)
            _write_results(out, arms, cpu_checks, bank_checks,
                           execution_status="NUMERICAL_FAILURE", failure=numerical_failure)
            status = {"status": "NUMERICAL_FAILURE", "phase": "training",
                      "arm": arm_id, "completed_arms": len(arms), "pid": os.getpid()}
            _atomic_json(out / "status.json", status)
            progress_callback(dict(status))
            raise FloatingPointError(numerical_failure)
        if arm_row["status"] == "TRAINED":
            model.eval()
            evaluation = evaluate_distance(model, eval_cpu, out, arm_id)
            arm_row["evaluation"] = evaluation
            arm_row["status"] = "COMPLETE" if evaluation["status"] == "EVALUATED" else evaluation["status"]
        arms.append(arm_row)
        _atomic_json(out / "arms.json", arms)
        if arm_row["status"] != "COMPLETE":
            reason = f"evaluation failed for {arm_id}: {arm_row['status']}"
            failure = {"status": "NUMERICAL_FAILURE", "arm": arm_id,
                       "reason": reason, "evaluation": arm_row.get("evaluation")}
            _atomic_json(out / "failure.json", failure)
            _write_results(out, arms, cpu_checks, bank_checks,
                           execution_status="NUMERICAL_FAILURE", failure=reason)
            status = {"status": "NUMERICAL_FAILURE", "phase": "evaluation",
                      "arm": arm_id, "completed_arms": len(arms), "pid": os.getpid()}
            _atomic_json(out / "status.json", status)
            progress_callback(dict(status))
            raise FloatingPointError(reason)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        status = {"status": "RUNNING", "phase": "arm_complete", "arm": arm_id,
                  "completed_arms": len(arms), "total_arms": len(INIT_SEEDS), "pid": os.getpid()}
        _atomic_json(out / "status.json", status)
        progress_callback(dict(status))
    summary = _write_results(out, arms, cpu_checks, bank_checks)
    final_status = {"status": summary["status"], "pid": os.getpid(),
                    "completed_arms": summary["complete_arms"], "total_arms": len(INIT_SEEDS),
                    "finished_unix": time.time()}
    _atomic_json(out / "status.json", final_status)
    progress_callback(dict(final_status))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = run_stage(args.out, lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"status": report["status"], "gate_passes": report["gate_passes"]}, flush=True))
