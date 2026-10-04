"""CPU-only validation of the saved trajectory-qualification artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_DEFAULT = ROOT / "runs" / "trajectory_20261004_serial01"
ADAPTED_LAUNCHER = "tools/launch_trajectory_qualification.ps1"
CADENCE = {"backward_calls": 8, "interior_detach_boundaries": 7,
           "forward_steps": 64, "loss_count": 8}
TOL = 2e-7
sys.path.insert(0, str(ROOT / "new" / "nca_inertial_wind_tunnel"))
from tasks import bank as region_bank  # CPU-only deterministic map construction


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, message: str):
    if not ok:
        raise AssertionError(message)


def near(actual, expected, label: str, tol: float = TOL):
    if actual is None or expected is None:
        require(actual is expected, f"{label}: {actual!r} != {expected!r}")
    else:
        require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
                abs(float(actual) - float(expected)) <= tol,
                f"{label}: {actual!r} != {expected!r}")


def same_tree(a, b, label: str):
    require(type(a) is type(b), f"{label}: type mismatch")
    if isinstance(a, dict):
        require(a.keys() == b.keys(), f"{label}: keys differ")
        for key in a:
            same_tree(a[key], b[key], f"{label}.{key}")
    elif isinstance(a, list):
        require(len(a) == len(b), f"{label}: lengths differ")
        for i, (x, y) in enumerate(zip(a, b)):
            same_tree(x, y, f"{label}[{i}]")
    elif isinstance(a, float):
        near(a, b, label, 0.0)
    else:
        require(a == b, f"{label}: {a!r} != {b!r}")


def cadence(curve, updates, first):
    require(len(curve) == updates and
            [r["update"] for r in curve] == list(range(first, first + updates)),
            "training curve has wrong update cadence")
    for row in curve:
        c = row.get("cadence", row)
        for key, value in CADENCE.items():
            require(c.get(key) == value, f"training cadence {key} mismatch")


def check_pair_trace(trace_path: Path, full: dict, compact: dict | None,
                     changed: np.ndarray, label: str):
    sizes = full["sizes"]
    count = 0
    with np.load(trace_path, allow_pickle=False) as z:
        require(set(("correct", "original_correct", "flipped_correct")) <= set(z.files),
                f"{label}: missing paired Boolean arrays")
        correct, original, flipped = (z[k] for k in
                                      ("correct", "original_correct", "flipped_correct"))
        require(correct.dtype == original.dtype == flipped.dtype == np.dtype(bool),
                f"{label}: correctness arrays are not Boolean")
        n = int(trace_path.stem.split("size")[-1])
        expected = (257, 32, n, n)
        require(correct.shape == original.shape == flipped.shape == expected,
                f"{label}: expected Boolean shape {expected}")
        require(np.array_equal(correct, original & flipped), f"{label}: paired correctness mismatch")
        row = sizes[str(n)]
        require(changed.shape == (32, n, n), f"{label}: regenerated map mask shape mismatch")
        denominators = np.asarray([m["changed_pixels"] for m in row["per_map"]], dtype=np.int64)
        actual_denominators = changed.reshape(32, -1).sum(1, dtype=np.int64)
        require(len(denominators) == 32 and np.array_equal(denominators, actual_denominators) and
                int(denominators.sum()) == row["changed_pixels"],
                f"{label}: changed-pixel denominators mismatch")
        for t in (64, 128, 256):
            per_map = (correct[t] & changed).reshape(32, -1).sum(1, dtype=np.int64)
            total = int(per_map.sum())
            meta = row["endpoints"][str(t)]["all_changed"]
            require(total == meta["correct_pixels"] and row["changed_pixels"] == meta["pixels"],
                    f"{label}: T{t} paired coverage counts mismatch")
            pooled = total / int(denominators.sum())
            mean_map = float(np.mean(per_map / denominators))
            near(meta["pooled_coverage"]["value"], pooled, f"{label}.T{t}.pooled")
            near(meta["mean_map_coverage"], mean_map, f"{label}.T{t}.mean_map")
            if compact is not None:
                near(compact["sizes"][str(n)]["coverage"][str(t)], pooled,
                     f"{label}.compact.T{t}")
        before, after = correct[64] & changed, correct[256] & changed
        denominator = int(before.sum())
        retained = int((before & after).sum())
        transitions = row["transitions"]["all_changed"]["64_to_256"]["pooled"]
        require(denominator == transitions["from_correct"] and retained == transitions["retained"],
                f"{label}: retention counts mismatch")
        retention = retained / denominator if denominator else None
        near(transitions["retention"]["value"], retention, f"{label}.retention")
        relapse = np.any((correct[:-1] & changed) & ~(correct[1:] & changed), axis=0) & changed
        ever = np.any(correct & changed, axis=0) & changed
        relapse_num = int(relapse.sum())
        ever_num = int(ever.sum())
        meta = row["ever_regressed_over_ever_correct"]
        require(relapse_num == meta["numerator"] and ever_num == meta["denominator"],
                f"{label}: regression counts mismatch")
        regression = relapse_num / ever_num if ever_num else None
        near(meta["value"], regression, f"{label}.regression")
        if compact is not None:
            cm = compact["sizes"][str(n)]
            near(cm["retention64_to256"], retention, f"{label}.compact.retention")
            near(cm["ever_regressed_fraction"], regression, f"{label}.compact.regression")
        count = 1
    return count


def wilson(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / trials
    d = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / d
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / d
    return center - half, center + half


def exact_mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def validate(run: Path) -> dict:
    manifest = read(run / "manifest.json")
    require(manifest["protocol"] == "trajectory_qualification_v1", "wrong protocol")
    expected_sources = manifest["source_sha256"]
    live_matches, adapted = 0, False
    for rel, digest in expected_sources.items():
        frozen = run / "source" / Path(rel)
        require(sha(frozen) == digest, f"snapshot binding mismatch: {rel}")
        current = ROOT / Path(rel)
        if sha(current) == digest:
            live_matches += 1
        else:
            require(rel == ADAPTED_LAUNCHER, f"live source drift: {rel}")
            text = current.read_text(encoding="utf-8-sig")
            require("Get-Command python" in text, "launcher edit is not the portable Python adapter")
            adapted = True
    refs = manifest["reference_sha256"]
    for rel, digest in refs.items():
        require(sha(ROOT / Path(rel)) == digest, f"reference binding mismatch: {rel}")
    qpath = ROOT / Path(manifest["qualification"])
    require(sha(qpath) == manifest["qualification_sha256"], "qualification hash mismatch")
    q = read(qpath)
    require(q["status"] == "PASS" and q["source_sha256"] == expected_sources and
            q["reference_sha256"] == refs, "qualification contents mismatch")
    plan_hashes = manifest["schedule_plan_sha256"]
    require(len(plan_hashes) == 26, "expected 26 frozen schedule plans")
    plans = {}
    for rel, digest in plan_hashes.items():
        p = run / rel
        require(sha(p) == digest, f"schedule-plan hash mismatch: {rel}")
        plans[Path(rel).stem] = read(p)

    status = read(run / "status.json")
    require(status["status"] == "COMPLETE", "serial execution incomplete")
    for stage, arms in (("1", 8), ("2", 32), ("3", 8)):
        require(status["stages"][stage]["status"] == "COMPLETE" and
                status["stages"][stage]["completed_arms"] == arms,
                f"stage {stage} completion count mismatch")

    s1 = run / "stage1_state_cross"
    a1 = read(s1 / "summary.json")["arms"]
    require(len(a1) == 8 and all(a["status"] == "COMPLETE" for a in a1), "stage1 arms incomplete")
    for name, plan in (("suffixH", plans["stage1_H"]), ("suffixS", plans["stage1_S"])):
        require(read(s1 / f"{name}_schedule.json") == plan, f"stage1 {name} schedule differs from plan")
    # These are the four fixed evaluation banks used by stages 1 and 2.
    banks1 = {n: region_bank(n, 32, 50000 + n, device="cpu")["changed"][:, 0].numpy().astype(bool)
              for n in (32, 64)}
    banks2 = {n: region_bank(n, 32, 83000 + n, device="cpu")["changed"][:, 0].numpy().astype(bool)
              for n in (32, 64)}
    native_replays = 0
    traces = 0
    native_names = {"thetaH_mH_suffixH": "H", "thetaH_mH_suffixS": "S20022_preserve_early3",
                    "thetaS_mS_suffixH": "S20022_replace_early3", "thetaS_mS_suffixS": "S20022"}
    for arm in a1:
        curve = arm["training_curve"]
        cadence(curve, 297, 4)
        full = read(s1 / f"{arm['name']}_summary.json")
        require(arm["endpoint"]["pass"] == full["phenotype_gate"]["pass"], "stage1 endpoint gate mismatch")
        if arm["name"] in native_names:
            ref = ROOT / "runs" / "bootstrap_20261004_seed4_01"
            name = native_names[arm["name"]]
            raw, old = read(ref / f"{name}.json"), read(ref / f"{name}_summary.json")
            require(arm["final_parameter_sha256"] == raw["final_parameter_sha256"], "native final parameter mismatch")
            replay = arm["native_replay"]
            require(replay["status"] == "PASS" and replay["maximum_absolute_error"] == 0 and
                    replay["integer_leaves"] > 0 and replay["float_leaves"] > 0 and
                    full["phenotype_gate"]["pass"] == old["phenotype_gate"]["pass"],
                    "native replay record mismatch")
            same_tree(full["sizes"], old["sizes"], f"native replay {name}")
            native_replays += 1
        for n in (32, 64):
            traces += check_pair_trace(s1 / f"{arm['name']}_size{n}.npz", full,
                                       arm["endpoint"], banks1[n], arm["name"])
    selection = read(s1 / "selection.json")
    require(selection == status["stages"]["1"]["selection"] and
            selection["recipe"] == "shadow_joint", "stage1 selector mismatch")

    s2 = run / "stage2_fresh_recipe"
    m2, perarm, sched = read(s2 / "manifest.json"), read(s2 / "perarm.json"), read(s2 / "schedules.json")
    require(m2["status"] == "COMPLETE" and len(m2["completed_pairs"]) == 16 and
            len(m2["completed_arms"]) == 32 and len(perarm) == 32, "stage2 completion mismatch")
    for pair in sched["pairs"]:
        plan = plans[f"stage2_seed{pair['initialization_seed']}"]
        require(pair["batch_indices"] == plan, "stage2 saved schedule differs from frozen plan")
    by_pair = {}
    for arm in perarm:
        cadence(arm["training_curve"], 300, 1)
        by_pair.setdefault(arm["pair_id"], {})[arm["arm"]] = arm
    require(len(by_pair) == 16, "stage2 independent pair count mismatch")
    pass_counts = {"baseline": 0, "treatment": 0}
    wins = losses = ties_both_pass = ties_both_fail = 0
    for pair_id, pair in by_pair.items():
        base, treat = pair["baseline"], pair["treatment"]
        require(base["initial_payload_sha256"] == treat["initial_payload_sha256"] and
                base["schedule_seed"] == treat["schedule_seed"], "stage2 pair binding mismatch")
        audit = treat["u3_components"]
        require(base["u3_components"] == audit, "stage2 pair state audit mismatch")
        pre, post, shadow = audit["baseline_before"], audit["treatment_after"], audit["shadow"]
        require(pre == audit["treatment_before"] == audit["baseline_after"], "stage2 pre-u3 baseline/treatment mismatch")
        require(post["theta_sha256"] == shadow["theta_sha256"] and
                post["adam_moments_sha256"] == shadow["adam_moments_sha256"] and
                post["adam_step_values"] == [3] and shadow["adam_step_values"] == [3] and
                post["adam_step_sha256"] == pre["adam_step_sha256"] == shadow["adam_step_sha256"] and
                post["optimizer_param_groups_sha256"] == pre["optimizer_param_groups_sha256"] == shadow["optimizer_param_groups_sha256"] and
                audit["selected_recipe"] == "shadow_joint", "stage2 shadow joint import mismatch")
        for arm in (base, treat):
            full = read(s2 / arm["evaluation_summary"])
            require(arm["phenotype_pass"] == full["phenotype_gate"]["pass"], "stage2 gate metadata mismatch")
            pass_counts[arm["arm"]] += int(arm["phenotype_pass"])
            for trace in arm["evaluation_traces"]:
                n = int(Path(trace).stem.rsplit("size", 1)[1])
                traces += check_pair_trace(s2 / trace, full, arm["evaluation_compact"],
                                           banks2[n], pair_id + arm["arm"])
        b, t = int(base["phenotype_pass"]), int(treat["phenotype_pass"])
        wins += int(t > b); losses += int(b > t)
        ties_both_pass += int(t == b == 1); ties_both_fail += int(t == b == 0)
    primary = read(s2 / "summary.json")["primary_paired_result"]
    p = exact_mcnemar(wins, losses)
    require(primary["baseline_passes"] == pass_counts["baseline"] and
            primary["treatment_passes"] == pass_counts["treatment"] and
            primary["treatment_only_passes"] == wins and primary["baseline_only_passes"] == losses and
            primary["ties_both_pass"] == ties_both_pass and primary["ties_both_fail"] == ties_both_fail,
            "stage2 paired endpoint counts mismatch")
    near(primary["delta_treatment_minus_baseline"],
         (pass_counts["treatment"] - pass_counts["baseline"]) / 16, "stage2 paired delta")
    near(primary["exact_two_sided_mcnemar_binomial_p"], p, "stage2 exact p", 1e-12)
    for arm in ("baseline", "treatment"):
        lo, hi = wilson(pass_counts[arm], 16)
        interval = primary[f"{arm}_pass_rate_wilson_95"]
        near(interval["lower"], lo, f"{arm} Wilson lower", 1e-12)
        near(interval["upper"], hi, f"{arm} Wilson upper", 1e-12)

    s3 = run / "stage3_distance"
    summary3, oracle = read(s3 / "summary.json"), read(s3 / "oracle_checks.json")
    require(summary3["status"] == "COMPLETE" and len(summary3["arms"]) == 8, "stage3 completion mismatch")
    distance_gate_passes = 0
    maxdist = {"32": None, "64": None}
    for arm in summary3["arms"]:
        curve = read(s3 / arm["training_curve_file"].replace("\\", "/"))
        cadence(curve, 300, 1)
        plan = plans[f"stage3_seed{arm['init_seed']}"]
        saved_schedule = np.load(s3 / f"schedule_seed{arm['schedule_seed']}.npy", allow_pickle=False)
        require(saved_schedule.shape == (300, 8) and np.array_equal(saved_schedule, np.asarray(plan)),
                "stage3 saved schedule differs from frozen plan")
        ev = arm["evaluation"]
        derived = {}
        for n in (32, 64):
            with np.load(s3 / f"{arm['arm_id']}_size{n}_correct_trace.npz", allow_pickle=False) as z:
                c, d, mask, source, horizons = (z[k] for k in
                    ("correct_within1", "distance_cells", "traversable_mask", "source_set", "horizons"))
                shape = (32, n, n)
                require(c.shape == (257, *shape) and c.dtype == np.dtype(bool), "stage3 trace shape/dtype mismatch")
                require(d.shape == mask.shape == source.shape == shape and mask.dtype == source.dtype == np.dtype(bool),
                        "stage3 oracle array shape/dtype mismatch")
                require(np.array_equal(horizons, np.arange(257)), "stage3 trace horizons mismatch")
                require(np.all(source <= mask) and np.all(d[source] == 0) and
                        np.all(source.reshape(32, -1).sum(1) > 0), "stage3 source/oracle map mismatch")
                maximum = int(d[mask].max())
                require(maxdist[str(n)] in (None, maximum), "stage3 oracle maximum varies across arms")
                maxdist[str(n)] = maximum
                eligible = mask & (d > 32)
                eligible_counts = eligible.reshape(32, -1).sum(1)
                gate_metrics = {}
                for t in (64, 256):
                    cp = c[t]
                    map_open = mask.reshape(32, -1).sum(1)
                    map_correct = (cp & mask).reshape(32, -1).sum(1)
                    overall_mean = float(np.mean(map_correct / map_open))
                    overall_pooled = int(map_correct.sum()) / int(map_open.sum())
                    band_open = eligible_counts
                    band_correct = (cp & eligible).reshape(32, -1).sum(1)
                    band_eligible = band_open > 0
                    band_mean = float(np.mean(band_correct[band_eligible] / band_open[band_eligible])) if np.any(band_eligible) else None
                    band_pooled = int(band_correct.sum()) / int(band_open.sum()) if int(band_open.sum()) else None
                    checkpoint = ev["sizes"][str(n)]["checkpoints"][str(t)]
                    near(checkpoint["mean_map_within1_coverage"], overall_mean, f"stage3 {n} T{t} mean coverage")
                    near(checkpoint["pooled_within1_coverage"], overall_pooled, f"stage3 {n} T{t} pooled coverage")
                    band_meta = checkpoint["d_gt_32"]
                    require(band_meta["eligible_maps"] == int(band_eligible.sum()) and
                            band_meta["pooled_cells"] == int(band_open.sum()), "stage3 d>32 oracle eligibility mismatch")
                    near(band_meta["mean_eligible_map_within1_coverage"], band_mean, f"stage3 {n} d>32 T{t} mean")
                    near(band_meta["pooled_within1_coverage"], band_pooled, f"stage3 {n} d>32 T{t} pooled")
                    gate_metrics[str(t)] = (cp, mask, eligible)
                c64, mask64, band64 = gate_metrics["64"]
                c256 = gate_metrics["256"][0]
                den = int((c64 & mask64).sum())
                kept = int((c64 & c256 & mask64).sum())
                retention_meta = ev["sizes"][str(n)]["checkpoints"]["retention_correct64_to_correct256"]
                require(retention_meta["correct64_pixels"] == den and retention_meta["retained_correct_pixels"] == kept,
                        "stage3 retention counts mismatch")
                near(retention_meta["pooled_retention"], kept / den, f"stage3 {n} retention")
                derived[str(n)] = {"retention": kept / den,
                    "d_gt_32_mean_coverage_gain": ev["sizes"][str(n)]["checkpoints"]["256"]["d_gt_32"]["mean_eligible_map_within1_coverage"] - ev["sizes"][str(n)]["checkpoints"]["64"]["d_gt_32"]["mean_eligible_map_within1_coverage"],
                    "eligible_maps": int(band64.reshape(32, -1).sum(1).astype(bool).sum())}
        checks = ev["gate"]["checks"]
        mae32_64 = ev["sizes"]["32"]["checkpoints"]["64"]["mean_map_mae_cells"]
        mae32_256 = ev["sizes"]["32"]["checkpoints"]["256"]["mean_map_mae_cells"]
        mae64_64 = ev["sizes"]["64"]["checkpoints"]["64"]["mean_map_mae_cells"]
        mae64_256 = ev["sizes"]["64"]["checkpoints"]["256"]["mean_map_mae_cells"]
        band64 = ev["sizes"]["64"]["checkpoints"]
        recomputed = {
            "size32_mae64_le_2": mae32_64 <= 2.0,
            "size32_mae256_no_worse_than_mae64_plus_0_25": mae32_256 <= mae32_64 + .25,
            "size32_retention_ge_0_95": derived["32"]["retention"] >= .95,
            "size64_mae256_no_worse_than_mae64_plus_0_25": mae64_256 <= mae64_64 + .25,
            "size64_retention_ge_0_95": derived["64"]["retention"] >= .95,
            "size64_d_gt_32_eligible_maps_ge_16": derived["64"]["eligible_maps"] >= 16,
            "size64_d_gt_32_mae_gain_ge_0_5": band64["64"]["d_gt_32"]["mean_eligible_map_mae_cells"] - band64["256"]["d_gt_32"]["mean_eligible_map_mae_cells"] >= .5,
            "size64_d_gt_32_within1_gain_ge_0_05": derived["64"]["d_gt_32_mean_coverage_gain"] >= .05,
        }
        require(checks == recomputed and ev["gate"]["status"] == ("PASS" if all(recomputed.values()) else "FAIL"),
                "stage3 gate recomputation mismatch")
        distance_gate_passes += int(all(recomputed.values()))
    for n, maximum in maxdist.items():
        require(maximum == oracle["evaluation"][n]["max_geodesic_distance_cells"],
                f"stage3 size{n} oracle maximum mismatch")
    require(summary3["gate_passes"] == distance_gate_passes == 0, "stage3 task gate count mismatch")
    return {
        "status": "PASS", "run_id": run.name,
        "execution": {"stage_arms": {"1": 8, "2": 32, "3": 8}, "status": "COMPLETE"},
        "bindings": {"source_files": len(expected_sources), "snapshots_match": len(expected_sources),
                     "live_sources_match_except_approved_launcher": live_matches,
                     "approved_portable_launcher_adapter": adapted,
                     "reference_files_match": len(refs), "schedule_plans_match": len(plan_hashes),
                     "qualification_matches": True},
        "stage1": {"native_controls_replayed": native_replays, "selected_recipe": selection["recipe"]},
        "stage2": {"pairs": 16, "baseline_full_passes": pass_counts["baseline"],
                   "treatment_full_passes": pass_counts["treatment"], "treatment_only": wins,
                   "baseline_only": losses, "paired_delta": (pass_counts["treatment"] - pass_counts["baseline"]) / 16,
                   "exact_two_sided_p": p, "verdict": "NO_RELIABILITY_QUALIFICATION"},
        "paired_trace_validation": {"arms": 40, "size_traces": traces,
                                    "per_trace": "Boolean equality, (257,32,n,n) shape, endpoint coverage, retention and regression match saved metadata"},
        "cpu_evaluation_banks_regenerated": 4,
        "training_curves": {"stage1_suffix_curves": 8, "stage2_full_curves": 32,
                            "stage3_full_curves": 8, "cadence": CADENCE},
        "stage3": {"status": "SECOND_TASK_TRAINING_UNQUALIFIED", "gate_passes": distance_gate_passes,
                   "oracle_max_distance_cells": maxdist,
                   "mae_source": "saved evaluation summaries; Boolean traces cannot independently verify MAE"},
        "scope": "Saved-artifact CPU validation only; no training, inference, or frontier reanalysis. Scientific failure labels are retained and do not imply architecture impossibility.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--out", type=Path, default=ROOT / "analyses" / "trajectory_publication_validation_20261004.json")
    args = parser.parse_args()
    run = (ROOT / args.run).resolve() if not args.run.is_absolute() else args.run.resolve()
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    try:
        result = validate(run)
    except Exception as error:
        result = {"status": "FAIL", "run_id": run.name, "error": f"{type(error).__name__}: {error}"}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out": out.relative_to(ROOT).as_posix()}))


if __name__ == "__main__":
    main()
