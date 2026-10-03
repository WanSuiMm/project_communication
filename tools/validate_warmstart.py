"""CPU-only saved-artifact validation for the detached warm-start screen.

This validator reads saved checkpoints and traces, reconstructs deterministic
CPU banks/schedules, and independently recomputes fresh-map phenotype metrics.
It does not run model forward, backward, training, or CUDA operations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/warmstart_20261003_paired01"
OUT_DIR = ROOT / "analyses/warmstart_20261003_publication_validation"
OUT_PATH = OUT_DIR / "validation.json"
SEEDS = (2, 3, 4, 5)
STAGES = (0, 100, 200, 300)
SIZES = (32, 64)
FLOAT_TOL = 2e-7


# Reuse the established saved-artifact hash, comparison, and trace readers.
sys.path.insert(0, str(ROOT / "new/streaming_carry"))
import validate_seed4_followup_training as common  # noqa: E402
from run_revision import tensor_hash  # noqa: E402
from stream_cells import StreamingCell  # noqa: E402
from tasks import bank  # noqa: E402

phenotype = common.phenotype


class Audit:
    """Keep the published report compact while retaining all failures."""

    def __init__(self) -> None:
        self.groups: dict[str, dict[str, int]] = {}
        self.failures: list[dict[str, Any]] = []
        self.total = 0

    def check(self, group: str, name: str, passed: bool, detail: Any = None) -> bool:
        row = self.groups.setdefault(group, {"passed": 0, "failed": 0})
        row["passed" if passed else "failed"] += 1
        self.total += 1
        if not passed and len(self.failures) < 80:
            failure = {"group": group, "name": name}
            if detail is not None:
                failure["detail"] = common._json_safe(detail)
            self.failures.append(failure)
        return passed


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _close(a: Any, b: Any, tol: float = FLOAT_TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _safe_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _nested_select(value: dict[str, Any], size: int, step: int) -> Any:
    return value.get(str(size), {}).get(str(step))


def _check_preflight_and_qualification(
    audit: Audit, manifest: dict[str, Any], source_hashes: dict[str, str]
) -> dict[str, Any]:
    preflight_path = ROOT / "runs/warmstart_20261003_preflight02"
    expected_preflight = manifest.get("preflight_sha256", {})
    preflight_actual: dict[str, str | None] = {}
    preflight_doc: dict[str, Any] = {}
    preflight_status: dict[str, Any] = {}
    preflight_aggregate: dict[str, Any] = {}
    for name in ("status.json", "manifest.json", "aggregate.json"):
        path = preflight_path / name
        actual = common._sha(path) if path.is_file() else None
        preflight_actual[name] = actual
        audit.check("preflight", f"{name} hash binding", actual == expected_preflight.get(name),
                    {"expected": expected_preflight.get(name), "actual": actual})
    if (preflight_path / "manifest.json").is_file():
        preflight_doc = common._read(preflight_path / "manifest.json")
    if (preflight_path / "status.json").is_file():
        preflight_status = common._read(preflight_path / "status.json")
    if (preflight_path / "aggregate.json").is_file():
        preflight_aggregate = common._read(preflight_path / "aggregate.json")
    audit.check("preflight", "source and no-cap protocol bindings",
                preflight_doc.get("protocol") == "detached_warmstart_v1_nocap" and
                preflight_doc.get("preflight") is True and preflight_doc.get("training") is True and
                preflight_doc.get("maximum_seconds") is None and
                preflight_doc.get("runtime_limit_enforced") is False and
                preflight_doc.get("watchdog_enabled") is False and
                preflight_doc.get("source_sha256") == source_hashes,
                {"status": preflight_doc.get("protocol"),
                 "source_entries": len(preflight_doc.get("source_sha256", {}))})
    audit.check("preflight", "passed status and aggregate",
                preflight_status.get("status") == "PREFLIGHT_PASSED" and
                preflight_status.get("completed_arms") == 2 and
                preflight_aggregate.get("complete") is True and
                preflight_aggregate.get("decision") == "PREFLIGHT_PASSED" and
                preflight_aggregate.get("completed_arms") == 2 and
                preflight_aggregate.get("expected_arms") == 8 and
                preflight_aggregate.get("updates_per_arm") == 300 and
                preflight_aggregate.get("maximum_seconds") is None and
                preflight_aggregate.get("runtime_limit_enforced") is False,
                {"status": preflight_status.get("status"),
                 "completed_arms": preflight_status.get("completed_arms"),
                 "decision": preflight_aggregate.get("decision")})

    qualification_binding = manifest.get("cpu_qualification", {})
    qualification_path = ROOT / Path(qualification_binding.get("path", ""))
    qualification_sha = common._sha(qualification_path) if qualification_path.is_file() else None
    qualification = common._read(qualification_path) if qualification_path.is_file() else {}
    audit.check("qualification", "CPU qualification hash and CPU-only status",
                qualification_sha == qualification_binding.get("sha256") and
                qualification.get("status") == "PASS" and
                qualification.get("training") is False and
                qualification.get("gpu_execution") is False and
                qualification.get("source_sha256") == source_hashes,
                {"path": qualification_binding.get("path"),
                 "expected_sha256": qualification_binding.get("sha256"),
                 "actual_sha256": qualification_sha,
                 "status": qualification.get("status"),
                 "training": qualification.get("training"),
                 "gpu_execution": qualification.get("gpu_execution"),
                 "source_entries": len(qualification.get("source_sha256", {}))})
    audit.check("qualification", "preflight qualification agrees with run binding",
                preflight_doc.get("cpu_qualification") == qualification_binding,
                {"preflight_binding": preflight_doc.get("cpu_qualification"),
                 "run_binding": qualification_binding})
    return {
        "preflight": {
            "path": "runs/warmstart_20261003_preflight02",
            "hashes": preflight_actual,
            "status": preflight_status.get("status"),
            "completed_smoke_arms": preflight_status.get("completed_arms"),
            "decision": preflight_aggregate.get("decision"),
            "source_entries": len(preflight_doc.get("source_sha256", {})),
        },
        "cpu_qualification": {
            "path": qualification_binding.get("path"),
            "sha256": qualification_sha,
            "status": qualification.get("status"),
            "training": qualification.get("training"),
            "gpu_execution": qualification.get("gpu_execution"),
            "source_entries": len(qualification.get("source_sha256", {})),
        },
    }


def _reconstruct_banks_and_schedules(
    audit: Audit, manifest: dict[str, Any], run: Path
) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]], dict[int, dict[str, torch.Tensor]], dict[str, Any]]:
    schedule = common._read(run / "schedule.json")
    expected_schedule = np.random.default_rng(20002).integers(0, 512, (300, 8)).tolist()
    published_schedule = common._read(ROOT / "evidence/streaming_carry_init2345/schedule.json")
    schedule_sha = common._sha(run / "schedule.json")
    audit.check("deterministic inputs", "batch schedule reconstructed and matches published schedule",
                schedule == expected_schedule and schedule == published_schedule and
                schedule_sha == manifest.get("schedule_sha256") and
                manifest.get("schedule_seed") == 20002,
                {"rows": len(schedule), "seed": manifest.get("schedule_seed"),
                 "sha256": schedule_sha})

    age_doc = common._read(run / "ages.json")
    expected_ages = np.random.default_rng(60002).choice([32, 64, 128, 192], size=300).tolist()
    age_sha = common._sha(run / "ages.json")
    audit.check("deterministic inputs", "prefix age schedule reconstructed",
                age_doc == {"seed": 60002, "choices": [32, 64, 128, 192], "ages": expected_ages} and
                age_sha == manifest.get("age_schedule_sha256"),
                {"count": len(age_doc.get("ages", [])), "seed": age_doc.get("seed"),
                 "sha256": age_sha})

    training = bank(32, 512, 10002, device="cpu")
    fresh = {size: bank(size, 32, 60000 + size, device="cpu") for size in SIZES}
    historical = {size: bank(size, 32, 40000 + size, device="cpu") for size in SIZES}
    diagnostic = bank(32, 16, 61032, device="cpu")
    evidence_manifest = common._read(ROOT / "evidence/streaming_carry_init2345/manifest.json")

    training_hash = tensor_hash(training)
    audit.check("deterministic inputs", "training bank reconstructed",
                training_hash == manifest.get("train_data_sha256") and
                manifest.get("train_data_seed") == 10002,
                {"sha256": training_hash, "seed": manifest.get("train_data_seed")})
    fresh_hashes = {str(size): tensor_hash(data) for size, data in fresh.items()}
    audit.check("deterministic inputs", "fresh evaluation banks reconstructed",
                fresh_hashes == manifest.get("new_evaluation_data_sha256") and
                manifest.get("new_evaluation_seeds") == [60032, 60064] and
                manifest.get("maps_per_size") == 32,
                {"hashes": fresh_hashes, "seeds": manifest.get("new_evaluation_seeds")})
    historical_hashes = {str(size): tensor_hash(data) for size, data in historical.items()}
    audit.check("deterministic inputs", "historical evaluation banks reconstructed",
                historical_hashes == manifest.get("historical_evaluation_data_sha256") and
                historical_hashes == evidence_manifest.get("evaluation_data_sha256"),
                {"hashes": historical_hashes})
    diagnostic_hash = tensor_hash(diagnostic)
    per_map_changed = [int(value) for value in diagnostic["changed"].sum(dim=(1, 2, 3)).tolist()]
    audit.check("deterministic inputs", "stage diagnostic bank reconstructed",
                diagnostic_hash == manifest.get("stage_diagnostic_data_sha256") and
                manifest.get("stage_diagnostic_seed") == 61032 and
                manifest.get("stage_diagnostic_maps") == 16 and
                manifest.get("stage_diagnostic_steps") == 128,
                {"sha256": diagnostic_hash, "maps": len(per_map_changed),
                 "changed_pixels_per_map": per_map_changed})
    return ({"schedule_sha256": schedule_sha, "age_schedule_sha256": age_sha,
             "training_data_sha256": training_hash, "fresh_evaluation_data_sha256": fresh_hashes,
             "historical_evaluation_data_sha256": historical_hashes,
             "stage_diagnostic_data_sha256": diagnostic_hash,
             "stage_changed_pixels_per_map": per_map_changed,
             "schedule_rows": len(schedule), "ages": age_doc.get("ages", [])},
            fresh, historical, {"training": training, "diagnostic": diagnostic})


def _source_bindings(audit: Audit, run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    source_hashes = manifest.get("source_sha256", {})
    snapshot_files = [path for path in (run / "source").rglob("*") if path.is_file()]
    snapshot_relatives = {path.relative_to(run / "source").as_posix() for path in snapshot_files}
    rows = []
    for relative, expected in sorted(source_hashes.items()):
        current = ROOT / Path(relative)
        snapshot = run / "source" / Path(relative)
        current_hash = common._sha(current) if current.is_file() else None
        snapshot_hash = common._sha(snapshot) if snapshot.is_file() else None
        current_ok = current_hash == expected
        snapshot_ok = snapshot_hash == expected
        audit.check("source bindings", f"current file {relative}", current_ok,
                    {"expected": expected, "actual": current_hash})
        audit.check("source bindings", f"run snapshot {relative}", snapshot_ok,
                    {"expected": expected, "actual": snapshot_hash})
        rows.append({"path": relative, "sha256": expected,
                     "current_matches": current_ok, "snapshot_matches": snapshot_ok})
    audit.check("source bindings", "exact 70-entry manifest and snapshot coverage",
                len(source_hashes) == 70 and len(snapshot_files) == 70 and
                snapshot_relatives == set(source_hashes),
                {"manifest_entries": len(source_hashes), "snapshot_files": len(snapshot_files),
                 "extra_snapshot_paths": sorted(snapshot_relatives - set(source_hashes))[:8],
                 "missing_snapshot_paths": sorted(set(source_hashes) - snapshot_relatives)[:8]})
    return {"manifest_entry_count": len(source_hashes), "snapshot_file_count": len(snapshot_files),
            "all_current_and_snapshot_hashes_match": all(
                row["current_matches"] and row["snapshot_matches"] for row in rows),
            "files": rows}


def _initial_identities(audit: Audit, records: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    old_rng_state = torch.random.get_rng_state()
    initial_by_seed: dict[int, str] = {}
    templates: dict[int, dict[str, torch.Tensor]] = {}
    architectures: dict[int, dict[str, Any]] = {}
    published: dict[int, dict[str, Any]] = {}
    try:
        for seed in SEEDS:
            # Seed only PyTorch's CPU generator; no CUDA RNG or context is touched.
            torch.random.default_generator.manual_seed(seed)
            model = StreamingCell()
            state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            initial_sha = tensor_hash(state)
            initial_by_seed[seed] = initial_sha
            templates[seed] = state
            architectures[seed] = model.metadata()
            historical = common._read(ROOT / f"evidence/streaming_carry_init2345/raw/stream_K8_seed{seed}.json")
            published[seed] = historical
            paired = [records.get(f"{variant}_seed{seed}", {}) for variant in ("baseline", "warmstart")]
            checks = [row.get("initial_parameter_sha256") == initial_sha for row in paired]
            checks += [row.get("initial_parameter_sha256") == historical.get("initial_parameter_sha256")
                       for row in paired]
            checks += [row.get("architecture") == architectures[seed] for row in paired]
            audit.check("initial identities", f"seed {seed} initialization and architecture",
                        all(checks) and historical.get("seed") == seed and
                        historical.get("initial_parameter_sha256") == initial_sha,
                        {"initial_parameter_sha256": initial_sha,
                         "arm_initial_match": checks[:2], "historical_match": checks[2:4],
                         "architecture_match": checks[4:]})
    finally:
        torch.random.set_rng_state(old_rng_state)
    return ({str(seed): value for seed, value in initial_by_seed.items()}, templates)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    # Tensor-only checkpoints are loaded on CPU, without executing arbitrary pickle objects.
    return torch.load(path, map_location="cpu", weights_only=True)


def _check_state(
    audit: Audit,
    path: Path,
    expected_file_sha: str | None,
    expected_parameter_sha: str | None,
    expected_identity: dict[str, Any],
    template: dict[str, torch.Tensor],
    checkpoint_group: str,
) -> dict[str, Any]:
    file_sha = common._sha(path) if path.is_file() else None
    file_match = file_sha is not None and file_sha == expected_file_sha
    audit.check("checkpoints", f"{checkpoint_group} file hash", file_match,
                {"path": _safe_relative(path), "expected": expected_file_sha, "actual": file_sha})
    payload = _load_checkpoint(path)
    state = payload.get("state_dict")
    state_ok = isinstance(state, dict) and bool(state) and all(isinstance(value, torch.Tensor) for value in state.values())
    shape_ok = state_ok and set(state) == set(template)
    finite = state_ok and all(bool(torch.isfinite(value).all()) for value in state.values())
    if shape_ok:
        shape_ok = all(tuple(state[key].shape) == tuple(template[key].shape) and
                       state[key].dtype == template[key].dtype for key in template)
    parameter_sha = tensor_hash(state) if state_ok else None
    parameter_match = parameter_sha is not None and parameter_sha == expected_parameter_sha
    metadata_match = all(payload.get(key) == value for key, value in expected_identity.items())
    audit.check("checkpoints", f"{checkpoint_group} CPU state shape and finiteness", bool(shape_ok and finite),
                {"path": _safe_relative(path), "tensor_count": len(state) if state_ok else 0,
                 "all_finite": bool(finite), "shape_match": bool(shape_ok)})
    audit.check("checkpoints", f"{checkpoint_group} parameter hash", parameter_match,
                {"path": _safe_relative(path), "expected": expected_parameter_sha,
                 "actual": parameter_sha})
    audit.check("checkpoints", f"{checkpoint_group} identity metadata", metadata_match,
                {"path": _safe_relative(path), "expected": expected_identity,
                 "actual": {key: payload.get(key) for key in expected_identity}})
    return {"path": _safe_relative(path), "file_sha256": file_sha,
            "parameter_sha256": parameter_sha, "file_sha_match": bool(file_match),
            "parameter_sha_match": bool(parameter_match), "identity_match": bool(metadata_match),
            "state_shape_match": bool(shape_ok), "all_finite": bool(finite),
            "parameter_count": sum(int(value.numel()) for value in state.values()) if state_ok else 0}


def _check_checkpoints(
    audit: Audit, run: Path, records: dict[str, dict[str, Any]], templates: dict[int, dict[str, torch.Tensor]],
    initial_by_seed: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_results: list[dict[str, Any]] = []
    final_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        for variant in ("baseline", "warmstart"):
            name = f"{variant}_seed{seed}"
            record = records[name]
            template = templates[seed]
            audit.check("arm completion", f"{name} complete at 300 updates",
                        record.get("name") == name and record.get("seed") == seed and
                        record.get("variant") == variant and record.get("status") == "COMPLETE" and
                        record.get("completed_updates") == 300 and record.get("parameter_count") == 5033 and
                        record.get("gradient_horizon") == 8 and
                        record.get("phenotype_is_scientific_endpoint") is True,
                        {"status": record.get("status"), "updates": record.get("completed_updates"),
                         "parameter_count": record.get("parameter_count")})
            checkpoint_meta = record.get("checkpoints", {})
            audit.check("checkpoints", f"{name} has exactly four declared stages",
                        set(checkpoint_meta) == {str(stage) for stage in STAGES},
                        {"stages": sorted(checkpoint_meta)})
            for update in STAGES:
                metadata = checkpoint_meta.get(str(update), {})
                path = run / metadata.get("file", f"{name}_u{update:03}.pt")
                expected_parameter_sha = metadata.get("parameter_sha256")
                if update == 0:
                    expected_parameter_sha = initial_by_seed[str(seed)]
                    audit.check("checkpoints", f"{name} stage 0 matches paired initialization",
                                metadata.get("parameter_sha256") == expected_parameter_sha,
                                {"recorded": metadata.get("parameter_sha256"), "expected": expected_parameter_sha})
                stage = _check_state(
                    audit, path, metadata.get("sha256"), expected_parameter_sha,
                    {"variant": variant, "seed": seed, "completed_updates": update},
                    template, f"{name} stage {update}")
                stage.update({"arm": name, "update": update})
                stage_results.append(stage)
            final_meta = {"variant": variant, "seed": seed, "completed_updates": 300}
            final = _check_state(
                audit, run / f"{name}.pt", record.get("checkpoint_file_sha256"),
                record.get("final_parameter_sha256"), final_meta, template, f"{name} final")
            final.update({"arm": name, "update": 300})
            final_results.append(final)
            audit.check("checkpoints", f"{name} final equals stage 300 parameter state",
                        final.get("parameter_sha256") == next(
                            row["parameter_sha256"] for row in stage_results
                            if row["arm"] == name and row["update"] == 300),
                        {"final": final.get("parameter_sha256"),
                         "stage300": next(row["parameter_sha256"] for row in stage_results
                                          if row["arm"] == name and row["update"] == 300)})
    return stage_results, final_results


def _check_historical_controls(
    audit: Audit, records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    total_records = 0
    for seed in SEEDS:
        name = f"baseline_seed{seed}"
        record = records[name]
        published = common._read(ROOT / f"evidence/streaming_carry_init2345/raw/stream_K8_seed{seed}.json")
        replay = record.get("historical_replay", {})
        saved_eval = replay.get("evaluation", {})
        original_eval = published.get("evaluation", {})
        final_hash_match = record.get("final_parameter_sha256") == published.get("final_parameter_sha256")
        audit.check("historical controls", f"{name} final parameter SHA matches published original",
                    final_hash_match,
                    {"saved": record.get("final_parameter_sha256"),
                     "published": published.get("final_parameter_sha256")})
        audit.check("historical controls", f"{name} saved replay declares six records",
                    replay.get("status") == "PASS" and replay.get("size_horizon_records") == 6 and
                    replay.get("final_parameter_sha256") == published.get("final_parameter_sha256") and
                    replay.get("counts", {}).get("maximum_absolute_error") == 0.0,
                    {"status": replay.get("status"),
                     "size_horizon_records": replay.get("size_horizon_records")})
        mismatches = []
        checked = 0
        for size in SIZES:
            for step in (64, 128, 256):
                checked += 1
                total_records += 1
                expected = _nested_select(original_eval, size, step)
                actual = _nested_select(saved_eval, size, step)
                # These saved replay records were generated against the published
                # originals; require exact numeric equality, not a tolerance.
                detail = [] if actual == expected else [f"size{size}/t{step}: exact record mismatch"]
                if detail:
                    mismatches.extend(detail)
                audit.check("historical controls", f"{name} size{size}/t{step} saved replay",
                            not detail, detail or {"match": True})
        rows.append({"seed": seed, "arm": name, "final_parameter_sha256": record.get("final_parameter_sha256"),
                     "published_final_sha_match": final_hash_match, "replay_status": replay.get("status"),
                     "size_horizon_records_compared": checked, "all_six_replays_match": not mismatches,
                     "mismatch_examples": mismatches[:5]})
    return {"baseline_arms": len(rows), "saved_size_horizon_records_compared": total_records,
            "all_historical_baselines_match": all(row["published_final_sha_match"] and
                                                   row["all_six_replays_match"] for row in rows),
            "baselines": rows}


def _gate_projection(gate: dict[str, Any]) -> dict[str, Any]:
    # Reason strings format FP32-derived values; compare the numeric check tree
    # with tolerance and the booleans/thresholds exactly instead.
    return {key: gate.get(key) for key in ("pass", "checks", "interpretation")}


def _validate_fresh_phenotypes(
    audit: Audit, run: Path, records: dict[str, dict[str, Any]], fresh: dict[int, dict[str, torch.Tensor]]
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    results: list[dict[str, Any]] = []
    gate_passes: dict[str, bool] = {}
    for seed in SEEDS:
        for variant in ("baseline", "warmstart"):
            name = f"{variant}_seed{seed}"
            record = records[name]
            summary_path = run / f"{name}_summary.json"
            saved_summary = common._read(summary_path)
            summary_sha = common._sha(summary_path)
            audit.check("fresh phenotypes", f"{name} saved summary hash",
                        summary_sha == record.get("phenotype_summary_sha256"),
                        {"expected": record.get("phenotype_summary_sha256"), "actual": summary_sha})
            traces: dict[str, dict[str, np.ndarray]] = {}
            trace_report = {}
            for size in SIZES:
                path = run / f"{name}_size{size}.npz"
                arrays = common._trace_arrays(path, size)
                traces[str(size)] = arrays
                trace_report[str(size)] = {
                    "path": _safe_relative(path), "sha256": common._sha(path),
                    "shape": list(arrays["correct"].shape), "dtype": str(arrays["correct"].dtype),
                    "paired_conjunction_verified": True,
                }
            endpoint_records = common._records_from_traces(traces, fresh)
            recomputed = phenotype.summarize_from_traces(traces, endpoint_records, fresh)
            gate = phenotype.predicate(recomputed)
            gate_passes[name] = bool(gate["pass"])

            size_mismatches = common._tree_mismatches(
                common._json_safe(recomputed["sizes"]), saved_summary.get("sizes"), limit=20)
            audit.check("fresh phenotypes", f"{name} trace-derived sizes and numeric summaries",
                        not size_mismatches,
                        size_mismatches or {"float_rel_abs_tolerance": FLOAT_TOL,
                                            "integer_counts_and_denominators": "exact"})
            summary_gate = saved_summary.get("phenotype_gate", {})
            record_gate = record.get("phenotype_gate", {})
            gate_summary_mismatches = common._tree_mismatches(
                _gate_projection(gate), _gate_projection(summary_gate), limit=20)
            gate_record_mismatches = common._tree_mismatches(
                _gate_projection(gate), _gate_projection(record_gate), limit=20)
            audit.check("fresh phenotypes", f"{name} full gate matches saved summary",
                        not gate_summary_mismatches,
                        gate_summary_mismatches or {"pass": gate["pass"], "gate_booleans_exact": True})
            audit.check("fresh phenotypes", f"{name} full gate matches arm record",
                        not gate_record_mismatches,
                        gate_record_mismatches or {"pass": gate["pass"], "gate_booleans_exact": True})

            matched_path = run / f"{name}_matched_frontier_strata.csv"
            matched_stats, csv_mismatches = common._matched_csv_audit(
                matched_path, recomputed["_matched_frontier_rows"])
            audit.check("fresh phenotypes", f"{name} integer matched-frontier CSV",
                        not csv_mismatches,
                        csv_mismatches or {"sha256": common._sha(matched_path),
                                           "rows": sum(row["csv_rows"] for row in matched_stats.values())})
            frontier_mismatches = []
            for size in SIZES:
                key = str(size)
                expected_frontier = recomputed["sizes"][key]["frontier"]
                actual_frontier = matched_stats[key]
                left = [{field: row[field] for field in
                         ("map_index", "common_strata", "weight_sum", "weighted_difference")}
                        for row in actual_frontier["per_map"]]
                right = [{field: row[field] for field in
                          ("map_index", "common_strata", "weight_sum", "weighted_difference")}
                         for row in expected_frontier["per_map"]]
                frontier_mismatches += common._tree_mismatches(left, right, path=f"size{size}/per_map", limit=10)
                frontier_mismatches += common._tree_mismatches(
                    {field: actual_frontier[field] for field in
                     ("common_strata", "eligible_maps", "mean_map_weighted_difference")},
                    {field: expected_frontier[field] for field in
                     ("common_strata", "eligible_maps", "mean_map_weighted_difference")},
                    path=f"size{size}/aggregate", limit=10)
            audit.check("fresh phenotypes", f"{name} frontier denominators and effects match traces",
                        not frontier_mismatches, frontier_mismatches[:15] or {"maps": 64})
            results.append({
                "arm": name, "seed": seed, "variant": variant,
                "saved_summary_sha256": summary_sha,
                "trace_files": trace_report,
                "phenotype_pass": bool(gate["pass"]),
                "failed_gate_count": sum(not row["pass"] for row in gate["checks"].values()),
                "recomputed_gate": _gate_projection(common._json_safe(gate)),
                "matched_frontier_csv": {
                    "path": _safe_relative(matched_path), "sha256": common._sha(matched_path),
                    "rows": sum(row["csv_rows"] for row in matched_stats.values()),
                    "sizes": {size: {key: value for key, value in matched_stats[size].items()
                                     if key != "per_map"} for size in ("32", "64")},
                },
            })
    return results, gate_passes


def _validate_stage_diagnostic(
    audit: Audit, path: Path, arm: str, seed: int, variant: str, update: int,
    expected_sha: str, per_map_changed: list[int]
) -> dict[str, Any]:
    doc = common._read(path)
    identity_ok = doc.get("arm") == arm and doc.get("seed") == seed and \
        doc.get("training_update") == update and \
        doc.get("checkpoint_parameter_sha256") == expected_sha and \
        doc.get("scientific_endpoint") is False and doc.get("exploratory") is True
    audit.check("stage diagnostics", f"{arm} update{update} checkpoint binding", identity_ok,
                {"arm": doc.get("arm"), "seed": doc.get("seed"),
                 "update": doc.get("training_update"),
                 "parameter_sha256": doc.get("checkpoint_parameter_sha256"),
                 "exploratory": doc.get("exploratory")})

    rows = doc.get("steps", [])
    exact = len(rows) == 129 and doc.get("trace_steps") == {"first": 0, "last": 128} and \
        doc.get("changed_pixels_per_map") == per_map_changed and doc.get("num_maps") == 16 and \
        doc.get("map_shape") == [32, 32]
    q_checks = identity_checks = 0
    previous_correct: int | None = None
    total_changed = sum(per_map_changed)
    for step, row in enumerate(rows):
        q = row.get("q", {})
        correct = row.get("correct")
        q_checks += 1
        exact &= row.get("t") == step and _is_int(row.get("changed_pixels")) and \
            row.get("changed_pixels") == total_changed and _is_int(correct) and \
            0 <= correct <= total_changed and q.get("pooled_correct") == correct and \
            q.get("pooled_pixels") == total_changed and \
            _close(q.get("pooled_accuracy"), _ratio(int(correct), total_changed))
        if step == 0:
            exact &= all(row.get(key) is None for key in
                         ("acquired", "wrong_old", "g_acquisition_rate", "destroyed",
                          "correct_old", "d_destruction_rate", "delta_correct", "accounting_error"))
        else:
            acquired, wrong_old = row.get("acquired"), row.get("wrong_old")
            destroyed, correct_old = row.get("destroyed"), row.get("correct_old")
            delta, accounting = row.get("delta_correct"), row.get("accounting_error")
            gained = row.get("g_acquisition_rate", {})
            lost = row.get("d_destruction_rate", {})
            expected_delta = int(correct) - int(previous_correct)
            identity_checks += 1
            integer_identity = all(_is_int(value) for value in
                                   (acquired, wrong_old, destroyed, correct_old, delta, accounting)) and \
                correct_old == previous_correct and wrong_old == total_changed - int(previous_correct) and \
                0 <= acquired <= wrong_old and 0 <= destroyed <= correct_old and \
                delta == acquired - destroyed == expected_delta and accounting == 0
            exact &= integer_identity
            exact &= gained.get("pooled_numerator") == acquired and \
                gained.get("pooled_denominator") == wrong_old and \
                _close(gained.get("pooled_rate"), _ratio(int(acquired), int(wrong_old)))
            exact &= lost.get("pooled_numerator") == destroyed and \
                lost.get("pooled_denominator") == correct_old and \
                _close(lost.get("pooled_rate"), _ratio(int(destroyed), int(correct_old)))
        previous_correct = int(correct)
    audit.check("stage diagnostics", f"{arm} update{update} 129-step integer turnover identities",
                bool(exact) and identity_checks == 128,
                {"steps": len(rows), "integer_identities": identity_checks,
                 "expected_identities": 128})

    survival = doc.get("survival", {})
    per_map = survival.get("per_map", [])
    multi = survival.get("multi_step_survival", {})
    endpoint = survival.get("endpoint_retention", {})
    start_correct = int(rows[64]["q"]["pooled_correct"]) if len(rows) > 64 else -1
    end_correct = int(rows[128]["q"]["pooled_correct"]) if len(rows) > 128 else -1
    per_map_valid = len(per_map) == 16 and all(
        row.get("map_index") == index and _is_int(row.get("correct_at_start")) and
        _is_int(row.get("correct_at_start_and_every_step")) and
        _is_int(row.get("correct_at_endpoint")) and
        row["correct_at_start_and_every_step"] <= row["correct_at_endpoint"] and
        row["correct_at_endpoint"] <= row["correct_at_start"] <= row["changed_pixels"] and
        row.get("changed_pixels") == per_map_changed[index] and
        row["correct_at_endpoint"] <= row["correct_at_start"]
        for index, row in enumerate(per_map))
    denom = sum(int(row["correct_at_start"]) for row in per_map) if per_map_valid else -1
    sustained = sum(int(row["correct_at_start_and_every_step"]) for row in per_map) if per_map_valid else -1
    endpoint_retained_sum = sum(int(row["correct_at_endpoint"]) for row in per_map) if per_map_valid else -1
    endpoint_rates = []
    survival_rates = []
    inferred_retained_counts = []
    if per_map_valid:
        for row in per_map:
            denominator = int(row["correct_at_start"])
            rate = row.get("endpoint_retention_rate")
            if denominator == 0:
                if rate is not None:
                    per_map_valid = False
                inferred_retained_counts.append(0)
                continue
            if not isinstance(rate, (int, float)) or not math.isfinite(float(rate)):
                per_map_valid = False
                inferred_retained_counts.append(-1)
                continue
            retained = int(row["correct_at_endpoint"])
            survivor = int(row["correct_at_start_and_every_step"])
            multi_rate = row.get("multi_step_survival_rate")
            if retained < 0 or retained > denominator or not _close(rate, _ratio(retained, denominator), 1e-12) or \
                    not _close(multi_rate, _ratio(survivor, denominator), 1e-12):
                per_map_valid = False
            inferred_retained_counts.append(retained)
            endpoint_rates.append(float(rate))
            survival_rates.append(float(multi_rate))
    retention_numerator = sum(inferred_retained_counts) if per_map_valid else -1
    retention_mean = float(np.mean(endpoint_rates)) if endpoint_rates else None
    survival_mean = float(np.mean(survival_rates)) if survival_rates else None
    survival_ok = survival.get("from_step") == 64 and survival.get("through_step") == 128 and \
        per_map_valid and denom == start_correct and endpoint_retained_sum == retention_numerator and \
        multi.get("pooled_correct_at_start") == start_correct and \
        multi.get("pooled_correct_at_start_and_every_step") == sustained and \
        endpoint.get("pooled_correct_at_start") == start_correct and \
        endpoint.get("pooled_correct_at_start_and_endpoint") == retention_numerator and \
        _close(multi.get("pooled_rate"), _ratio(sustained, start_correct)) and \
        _close(multi.get("equal_map_mean"), survival_mean) and \
        multi.get("equal_map_eligible_maps") == len(survival_rates) and \
        _close(endpoint.get("pooled_rate"), _ratio(retention_numerator, start_correct)) and \
        _close(endpoint.get("equal_map_mean"), retention_mean) and \
        endpoint.get("equal_map_eligible_maps") == len(endpoint_rates)
    audit.check("stage diagnostics", f"{arm} update{update} survival/retention denominators",
                bool(survival_ok),
                {"start_step64_correct_denominator": start_correct,
                 "per_map_start_denominator_sum": denom,
                 "multi_step_retained_numerator": sustained,
                 "per_map_endpoint_retained_numerator": endpoint_retained_sum,
                 "inferred_endpoint_retained_numerator": retention_numerator,
                 "saved_endpoint_retained_numerator": endpoint.get("pooled_correct_at_start_and_endpoint"),
                 "step128_all_changed_correct_count": end_correct})

    rms = doc.get("state_rms", {})
    rms_count = 0
    rms_ok = set(rms) == {"32", "64", "128"}
    for step in ("32", "64", "128"):
        for branch in ("original", "flipped"):
            values = rms.get(step, {}).get(branch, {})
            for field in ("W_rms", "Z_rms"):
                value = values.get(field)
                rms_count += 1
                rms_ok &= isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0
    audit.check("stage diagnostics", f"{arm} update{update} finite recorded RMS",
                bool(rms_ok) and rms_count == 12,
                {"finite_nonnegative_scalars": rms_count, "expected": 12})
    return {"path": _safe_relative(path), "arm": arm, "seed": seed, "variant": variant,
            "update": update, "checkpoint_parameter_sha256": expected_sha,
            "turnover_steps_checked": identity_checks, "survival_denominator": start_correct,
            "multi_step_survival_numerator": sustained,
            "endpoint_retention_numerator": retention_numerator,
            "endpoint_retention_rate": endpoint.get("pooled_rate"),
            "finite_rms_scalars": rms_count, "exploratory": True,
            "full_boolean_trace_saved": False}


def _validate_stage_diagnostics(
    audit: Audit, run: Path, records: dict[str, dict[str, Any]], per_map_changed: list[int]
) -> list[dict[str, Any]]:
    result = []
    for seed in SEEDS:
        for variant in ("baseline", "warmstart"):
            name = f"{variant}_seed{seed}"
            checkpoints = records[name].get("checkpoints", {})
            for update in STAGES:
                stage_meta = checkpoints.get(str(update), {})
                result.append(_validate_stage_diagnostic(
                    audit, run / f"{name}_u{update:03}_diagnostic.json", name, seed, variant,
                    update, stage_meta.get("parameter_sha256", ""), per_map_changed))
    return result


def _check_aggregate(
    audit: Audit, saved: dict[str, Any], historical: dict[str, Any], gate_passes: dict[str, bool]
) -> dict[str, Any]:
    baseline_passes = sum(gate_passes.get(f"baseline_seed{seed}", False) for seed in SEEDS)
    warmstart_passes = sum(gate_passes.get(f"warmstart_seed{seed}", False) for seed in SEEDS)
    historical_ok = historical["all_historical_baselines_match"]
    baseline_qualified = historical_ok and gate_passes.get("baseline_seed4", False)
    decision = ("BASELINE_FRESH_PHENOTYPE_UNQUALIFIED" if not baseline_qualified else
                "DEVELOPMENT_SIGNAL" if warmstart_passes >= 3 and
                warmstart_passes - baseline_passes >= 2 else "DEVELOPMENT_NOT_QUALIFIED")
    expected_passes = {"baseline": baseline_passes, "warmstart": warmstart_passes}
    expected_pairs = {str(seed): {variant: gate_passes.get(f"{variant}_seed{seed}", False)
                                  for variant in ("baseline", "warmstart")} for seed in SEEDS}
    expected_by_arm = {name: gate_passes[name] for name in sorted(gate_passes)}
    checks = {
        "complete": saved.get("complete") is True,
        "completed_arms": saved.get("completed_arms") == 8,
        "expected_arms": saved.get("expected_arms") == 8,
        "updates_per_arm": saved.get("updates_per_arm") == 300,
        "independent_unit": saved.get("independent_unit") == "paired initialization",
        "n": saved.get("n") == 4,
        "historical_controls_qualified": saved.get("historical_controls_qualified") == historical_ok,
        "known_seed4_fresh_qualified": saved.get("known_seed4_fresh_qualified") == gate_passes.get("baseline_seed4"),
        "passes": saved.get("passes") == expected_passes,
        "phenotype_pass": saved.get("phenotype_pass") == expected_by_arm,
        "paired_full_pass_change": saved.get("paired_full_pass_change") == warmstart_passes - baseline_passes,
        "pairs": saved.get("pairs") == expected_pairs,
        "decision": saved.get("decision") == decision,
    }
    for key, passed in checks.items():
        audit.check("aggregate decision", f"saved aggregate {key}", passed,
                    {"saved": saved.get(key), "independently_derived": {
                        "passes": expected_passes, "phenotype_pass": expected_by_arm,
                        "pairs": expected_pairs, "paired_full_pass_change": warmstart_passes-baseline_passes,
                        "decision": decision, "historical_controls_qualified": historical_ok,
                        "known_seed4_fresh_qualified": gate_passes.get("baseline_seed4"),
                    }.get(key, checks[key])})
    return {"independent_unit": "paired initialization", "n": 4,
            "baseline_passes": baseline_passes, "warmstart_passes": warmstart_passes,
            "paired_full_pass_change": warmstart_passes - baseline_passes,
            "historical_controls_qualified": historical_ok,
            "known_seed4_fresh_qualified": gate_passes.get("baseline_seed4"),
            "decision": decision, "saved_decision": saved.get("decision"),
            "all_saved_aggregate_fields_match": all(checks.values()),
            "pairs": expected_pairs,
            "scope": saved.get("scope")}


def validate() -> dict[str, Any]:
    audit = Audit()
    manifest = common._read(RUN / "manifest.json")
    status = common._read(RUN / "status.json")
    aggregate = common._read(RUN / "aggregate.json")
    records = {f"{variant}_seed{seed}": common._read(RUN / f"{variant}_seed{seed}.json")
               for seed in SEEDS for variant in ("baseline", "warmstart")}

    status_ok = status.get("status") == "COMPLETE" and status.get("completed_arms") == 8
    aggregate_complete = aggregate.get("complete") is True and aggregate.get("completed_arms") == 8 and \
        aggregate.get("expected_arms") == 8 and aggregate.get("updates_per_arm") == 300 and \
        aggregate.get("identities_verified") is True
    audit.check("completion", "run status COMPLETE 8/8", status_ok,
                {"status": status.get("status"), "completed_arms": status.get("completed_arms")})
    audit.check("completion", "aggregate 8/8 at 300 updates", aggregate_complete,
                {"complete": aggregate.get("complete"), "completed_arms": aggregate.get("completed_arms"),
                 "expected_arms": aggregate.get("expected_arms"), "updates_per_arm": aggregate.get("updates_per_arm"),
                 "identities_verified": aggregate.get("identities_verified")})

    protocol_ok = manifest.get("protocol") == "detached_warmstart_v1_nocap" and \
        manifest.get("training") is True and manifest.get("preflight") is False and \
        manifest.get("expected_formal_arms") == 8 and manifest.get("updates_per_arm") == 300 and \
        manifest.get("maximum_seconds") is None and manifest.get("runtime_limit_enforced") is False and \
        manifest.get("watchdog_enabled") is False and manifest.get("stage_updates") == list(STAGES) and \
        manifest.get("gradient_horizon") == 8 and manifest.get("trained_suffix_steps") == 64 and \
        manifest.get("loss_times_suffix") == list(range(8, 65, 8)) and \
        manifest.get("warm_examples") == 4 and manifest.get("fresh_examples") == 4 and \
        manifest.get("prefix_age_choices") == [32, 64, 128, 192] and \
        manifest.get("optimizer") == {"type": "AdamW", "lr": 0.001, "weight_decay": 0.0001,
                                        "clip_norm": 1.0, "batch": 8}
    audit.check("protocol", "completed no-cap protocol configuration", protocol_ok,
                {key: manifest.get(key) for key in
                 ("protocol", "training", "preflight", "updates_per_arm", "maximum_seconds",
                  "runtime_limit_enforced", "watchdog_enabled", "gradient_horizon",
                  "trained_suffix_steps", "warm_examples", "fresh_examples", "prefix_age_choices", "optimizer")})
    expected_names = {f"{variant}_seed{seed}" for seed in SEEDS for variant in ("baseline", "warmstart")}
    manifest_names = {item.get("name") for item in manifest.get("arms", [])}
    audit.check("protocol", "manifest lists exact eight paired arms",
                manifest_names == expected_names and len(manifest.get("arms", [])) == 8,
                {"manifest_arm_names": sorted(manifest_names), "expected_arm_names": sorted(expected_names)})

    source_result = _source_bindings(audit, RUN, manifest)
    qualification_result = _check_preflight_and_qualification(audit, manifest, manifest.get("source_sha256", {}))
    deterministic, fresh_banks, _historical_banks, bank_details = _reconstruct_banks_and_schedules(
        audit, manifest, RUN)

    # Each arm must carry the reconstructed common data, batch, and age identities.
    for name, record in sorted(records.items()):
        bindings = {
            "train_data_sha256": deterministic["training_data_sha256"],
            "schedule_sha256": deterministic["schedule_sha256"],
            "age_schedule_sha256": deterministic["age_schedule_sha256"],
        }
        mismatches = {key: {"saved": record.get(key), "reconstructed": expected}
                      for key, expected in bindings.items() if record.get(key) != expected}
        for index in (0, 99, 199, 299):
            update = index + 1
            curve = next((row for row in record.get("training_curve", []) if row.get("update") == update), None)
            expected_age = deterministic["ages"][index]
            if curve is None or curve.get("prefix_age_macro_steps") != expected_age:
                mismatches[f"training_curve_age_update_{update}"] = {
                    "saved": None if curve is None else curve.get("prefix_age_macro_steps"),
                    "reconstructed": expected_age}
        audit.check("paired schedules", f"{name} bank/schedule/age bindings",
                    not mismatches, mismatches or {"bindings_match": True,
                                                   "logged_ages_checked_at_updates": [1, 100, 200, 300]})

    initial_hashes, templates = _initial_identities(audit, records)
    stage_checkpoints, final_checkpoints = _check_checkpoints(
        audit, RUN, records, templates, initial_hashes)
    historical_result = _check_historical_controls(audit, records)
    phenotype_results, gate_passes = _validate_fresh_phenotypes(audit, RUN, records, fresh_banks)
    stage_results = _validate_stage_diagnostics(
        audit, RUN, records, deterministic["stage_changed_pixels_per_map"])
    aggregate_result = _check_aggregate(audit, aggregate, historical_result, gate_passes)

    all_ok = not audit.failures and status_ok and aggregate_complete and protocol_ok
    return {
        "schema": "warmstart-publication-saved-result-validation-v1",
        "status": "PASS" if all_ok else "FAIL",
        "run_id": RUN.name,
        "validation_mode": "CPU saved-artifact validation; no model rollout, inference, backward, training, or CUDA calls",
        "qualification_and_preflight": qualification_result,
        "completed_configuration": {
            "protocol": manifest.get("protocol"), "training": manifest.get("training"),
            "preflight": manifest.get("preflight"), "expected_arms": manifest.get("expected_formal_arms"),
            "completed_arms": status.get("completed_arms"), "updates_per_arm": manifest.get("updates_per_arm"),
            "maximum_seconds": manifest.get("maximum_seconds"),
            "runtime_limit_enforced": manifest.get("runtime_limit_enforced"),
            "watchdog_enabled": manifest.get("watchdog_enabled"),
            "gradient_horizon": manifest.get("gradient_horizon"),
            "trained_suffix_steps": manifest.get("trained_suffix_steps"),
            "loss_times_suffix": manifest.get("loss_times_suffix"),
            "warm_examples": manifest.get("warm_examples"), "fresh_examples": manifest.get("fresh_examples"),
            "prefix_age_choices": manifest.get("prefix_age_choices"),
            "optimizer": manifest.get("optimizer"), "backend": manifest.get("backend"),
        },
        "source_bindings": source_result,
        "deterministic_inputs": {key: value for key, value in deterministic.items() if key != "ages"} |
            {"age_count": len(deterministic["ages"]),
             "age_histogram": {str(age): deterministic["ages"].count(age) for age in [32, 64, 128, 192]}},
        "initial_parameter_sha256_by_seed": initial_hashes,
        "coverage": {"final_checkpoints": len(final_checkpoints),
                     "stage_checkpoints": len(stage_checkpoints),
                     "historical_final_parameter_hashes": historical_result["baseline_arms"],
                     "historical_size_horizon_records": historical_result["saved_size_horizon_records_compared"],
                     "fresh_phenotype_arms": len(phenotype_results),
                     "formal_boolean_trace_files": 2 * len(phenotype_results),
                     "stage_diagnostic_summaries": len(stage_results),
                     "stage_turnover_identities": sum(row["turnover_steps_checked"] for row in stage_results),
                     "finite_stage_rms_values": sum(row["finite_rms_scalars"] for row in stage_results)},
        "check_groups": audit.groups,
        "check_count": audit.total,
        "failures": audit.failures,
        "checkpoints": {"final": final_checkpoints, "stages": stage_checkpoints},
        "historical_controls": historical_result,
        "fresh_phenotypes": phenotype_results,
        "stage_diagnostics": stage_results,
        "aggregate_decision": aggregate_result,
        "limitations": [
            "Stage diagnostics retain per-step integer summaries, not their full 128-step Boolean traces; turnover identities, denominators, survival aggregates, checkpoint bindings, and RMS finiteness were checked from saved summaries, but the stage survival curves cannot be independently replayed from raw stage traces.",
            "The phenotype gate was independently recomputed from each saved fresh-evaluation Boolean trace and reconstructed deterministic CPU bank; no model inference or GPU call was made.",
            "The paired screen has four initializations under one fixed training bank and batch schedule; its saved decision remains a conditional development result.",
        ],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        report = validate()
    except Exception as error:
        report = {
            "schema": "warmstart-publication-saved-result-validation-v1",
            "status": "FAIL", "run_id": RUN.name,
            "validation_mode": "CPU saved-artifact validation; no model rollout, inference, backward, training, or CUDA calls",
            "fatal_error": f"{type(error).__name__}: {error}",
        }
    common._json_write(OUT_PATH, common._json_safe(report))
    print(json.dumps({"status": report.get("status"), "check_count": report.get("check_count"),
                      "failure_count": len(report.get("failures", [])),
                      "path": _safe_relative(OUT_PATH)}, indent=2))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
