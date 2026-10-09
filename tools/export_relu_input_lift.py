"""Export and independently verify the compact ReLU input-lift evidence package.

Export reads the completed private run and checkpoint tensors on CPU, then copies
only the allowlisted evidence. The offline verifier reads the public package,
the already-public AU-NCA control package, and repository source files; it never
opens the private run or checkpoint payloads and performs no training/inference.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from tools import export_au_nca as au


RUN = ROOT / "runs/relu_input_lift_20261009_01"
PUBLIC = ROOT / "evidence/relu_input_lift_20261009_01"
PUBLICATION = ROOT / "RELU_INPUT_LIFT_PUBLICATION_MANIFEST.json"
REFERENCE = ROOT / "evidence/au_nca_20261009_01"
REFERENCE_PUBLICATION = ROOT / "AU_NCA_PUBLICATION_MANIFEST.json"
RUN_ID = "relu_input_lift_20261009_01"
REFERENCE_RUN_ID = "au_nca_20261009_01"
ARM = "relu_lift_k8"
HISTORICAL_ARMS = ("original_k64", "original_k8", "au_k8")
BRANCHES = ("intact_t64", "intact_t128", "intact_t256", "damaged_t128", "damaged_t256")
TRAINING_FIELDS = ("update", "loss", "W_gradient_norm_before_normalization", "eta_gradient_norm_before_normalization")
CHECKPOINT_UPDATES = (1000, 2000, 3000)
EXPECTED_SOURCE_NAMES = {
    *(f"new/relu_input_lift/{name}" for name in ("run.py", "cells.py", "checks.py", "PROTOCOL.md", "THEORY.md")),
    *(f"new/au_nca/{name}" for name in ("run.py", "cells.py", "tasks.py", "checks.py", "PROTOCOL.md", "THEORY.md")),
    "tools/start_protected_job.ps1",
    "tools/protected_job_worker.ps1",
}
EXPECTED_DATA_NAMES = {"target_rgba.npy", "block00_inputs.npz", "block01_inputs.npz", "block02_inputs.npz"}
TOL = 1e-12


def read_json(path: Path) -> Any:
    return au.read_json(path)


def write_json(path: Path, value: Any) -> None:
    au.write_json(path, value)


def sha256_file(path: Path) -> str:
    return au.sha256_file(path)


def copy_file(source: Path, destination: Path) -> None:
    au.copy_file(source, destination)


def privacy_check(path: Path) -> None:
    au.privacy_check(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return au.read_jsonl(path)


def assert_close(actual: Any, expected: Any, label: str, *, atol: float = TOL, rtol: float = TOL) -> float:
    return au.assert_close(actual, expected, label, atol=atol, rtol=rtol)


def check_relative_source_map(source_hashes: dict[str, str]) -> None:
    if set(source_hashes) != EXPECTED_SOURCE_NAMES:
        raise AssertionError("source hash map does not contain the frozen 13-source set")
    for name, digest in source_hashes.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AssertionError(f"invalid source hash binding: {name}")


def check_data_hash_map(data_hashes: dict[str, str]) -> None:
    if set(data_hashes) != EXPECTED_DATA_NAMES:
        raise AssertionError("data hash map does not contain the fixed target and three input banks")
    for name, digest in data_hashes.items():
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AssertionError(f"invalid data hash binding: {name}")


def load_reference_bindings(run_binding: dict[str, Any]) -> dict[str, Any]:
    """Pin the previously published controls without copying their result data."""
    if not REFERENCE_PUBLICATION.is_file() or not REFERENCE.is_dir():
        raise AssertionError("the already-public AU-NCA reference package is missing")
    reference_publication = read_json(REFERENCE_PUBLICATION)
    reference_manifest = read_json(REFERENCE / "manifest.json")
    reference_qualification = read_json(REFERENCE / "qualification.json")
    if reference_publication.get("schema") != "au-nca-publication-v1" or reference_publication.get("run_id") != REFERENCE_RUN_ID:
        raise AssertionError("unexpected AU-NCA publication identity")
    if reference_manifest.get("run_id") != REFERENCE_RUN_ID or reference_manifest.get("status") != "COMPLETE":
        raise AssertionError("AU-NCA reference package is not the frozen completed package")
    if reference_manifest.get("config") != reference_qualification.get("config"):
        raise AssertionError("AU-NCA reference configuration bindings disagree")
    if reference_qualification.get("status") != "PASS":
        raise AssertionError("AU-NCA historical qualification is not PASS")
    if sha256_file(REFERENCE / "qualification.json") != reference_manifest.get("qualification_sha256"):
        raise AssertionError("AU-NCA reference qualification hash mismatch")
    au.verify_source_hashes(reference_manifest["source_hashes"])

    bound_data = run_binding.get("data_hashes", {})
    if bound_data != reference_manifest.get("data_hashes"):
        raise AssertionError("new run and historical controls do not bind identical target/input banks")
    if run_binding.get("historical_not_rerun") is not True:
        raise AssertionError("historical control binding must state that controls were not rerun")

    public_files = reference_publication.get("published_artifacts", {})
    source_artifacts = run_binding.get("artifact_hashes", {})
    expected_result_names = {
        f"block{block:02d}/{arm}/result.json"
        for block in range(3)
        for arm in HISTORICAL_ARMS
    }
    expected_source_names = {"aggregate.json", "manifest.json", "qualification.json", *expected_result_names}
    if set(source_artifacts) != expected_source_names:
        raise AssertionError("frozen run does not bind the complete nine-result historical control set")

    artifact_hashes: dict[str, str] = {}
    for relative_name, expected_hash in source_artifacts.items():
        public_relative = f"evidence/{REFERENCE_RUN_ID}/{relative_name}"
        entry = public_files.get(public_relative)
        path = REFERENCE / Path(relative_name)
        if not entry:
            raise AssertionError(f"public historical artifact is missing: {relative_name}")
        # The old package's public manifest was generated during its export, so
        # it differs from the private run manifest bound by the original run.
        # Pin that public manifest to its publication record; all result and
        # aggregate hashes must still match the run's frozen artifact bindings.
        published_hash = entry.get("sha256")
        if relative_name != "manifest.json" and published_hash != expected_hash:
            raise AssertionError(f"public historical artifact differs from frozen binding: {relative_name}")
        if sha256_file(path) != published_hash:
            raise AssertionError(f"public historical artifact differs from frozen binding: {relative_name}")
        artifact_hashes[public_relative] = published_hash

    for relative_name in ("RESULTS.md", "final_metrics.csv", "per_episode_metrics.csv"):
        public_relative = f"evidence/{REFERENCE_RUN_ID}/{relative_name}"
        entry = public_files.get(public_relative)
        if not entry:
            raise AssertionError(f"historical public artifact missing from publication map: {relative_name}")
        path = REFERENCE / relative_name
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
            raise AssertionError(f"historical public artifact hash mismatch: {relative_name}")
        artifact_hashes[public_relative] = expected_hash

    old_aggregate = read_json(REFERENCE / "aggregate.json")
    if old_aggregate.get("status") != "COMPLETE" or len(old_aggregate.get("arms", [])) != 9:
        raise AssertionError("historical aggregate is incomplete")
    old_decision = au.compute_decision(old_aggregate["arms"], reference_manifest["config"])
    if old_decision != old_aggregate.get("decision"):
        raise AssertionError("historical AU-NCA decision does not recompute")
    old_results = {
        (int(row["block"]), row["arm"]): row
        for row in old_aggregate["arms"]
    }
    if set(old_results) != {(block, arm) for block in range(3) for arm in HISTORICAL_ARMS}:
        raise AssertionError("historical aggregate does not contain all nine expected controls")
    for (block, arm), expected in old_results.items():
        relative_name = f"block{block:02d}/{arm}/result.json"
        actual = read_json(REFERENCE / relative_name)
        if actual != expected or actual.get("status") != "COMPLETE":
            raise AssertionError(f"historical arm result differs from its aggregate: {relative_name}")

    return {
        "schema": "relu-input-lift-reference-bindings-v1",
        "historical_run_id": REFERENCE_RUN_ID,
        "historical_public_package": f"evidence/{REFERENCE_RUN_ID}",
        "historical_results_markdown": f"evidence/{REFERENCE_RUN_ID}/RESULTS.md",
        "historical_control_results_not_rerun": True,
        "shared_target_and_input_hashes": bound_data,
        "historical_result_artifacts": {
            path: digest for path, digest in artifact_hashes.items()
            if path.endswith("/result.json")
        },
        "supporting_public_artifacts": {
            path: digest for path, digest in artifact_hashes.items()
            if not path.endswith("/result.json")
        },
        "control_aggregate_sha256": artifact_hashes[f"evidence/{REFERENCE_RUN_ID}/aggregate.json"],
        "no_historical_predictions_or_input_banks_duplicated": True,
    }


def validate_private_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_manifest = read_json(RUN / "manifest.json")
    qualification = read_json(RUN / "qualification.json")
    aggregate = read_json(RUN / "aggregate.json")
    status = read_json(RUN / "status.json")
    config = run_manifest["config"]
    source_hashes = run_manifest["source_hashes"]
    data_hashes = run_manifest["data_hashes"]
    if config != qualification["config"] or config != aggregate["config"]:
        raise AssertionError("run, qualification, and aggregate configurations differ")
    if source_hashes != qualification["source_hashes"]:
        raise AssertionError("qualification source bindings differ from the frozen run manifest")
    if qualification.get("status") != "PASS" or aggregate.get("status") != "COMPLETE" or status.get("status") != "COMPLETE":
        raise AssertionError("run is not qualified and complete")
    if aggregate.get("completed_units") != 3 or aggregate.get("planned_units") != 3:
        raise AssertionError("new run completion count is not 3/3")
    if status.get("completed_units") != 3 or status.get("planned_units") != 3:
        raise AssertionError("saved status completion count is not 3/3")
    if len(aggregate.get("arms", [])) != 3:
        raise AssertionError("aggregate does not contain all three new arms")
    arm_map = {(int(result["block"]), result["arm"]): result for result in aggregate["arms"]}
    if set(arm_map) != {(block, ARM) for block in range(3)}:
        raise AssertionError("aggregate new-arm identities are incomplete or duplicated")
    for block in range(3):
        result = read_json(RUN / f"block{block:02d}/{ARM}/result.json")
        if result != arm_map[(block, ARM)] or result.get("status") != "COMPLETE":
            raise AssertionError(f"per-block result does not match completed aggregate: block{block:02d}")
    if aggregate["decision"].get("verdict") != "NO_QUALIFIED_RELU_LIFT_BENEFIT":
        raise AssertionError("unexpected frozen ReLU input-lift verdict")
    if not math.isfinite(float(aggregate.get("elapsed_seconds_this_execution", float("nan")))):
        raise AssertionError("aggregate execution duration is not finite")
    if qualification.get("qualification_reuse", {}).get("source"):
        reuse_source = Path(qualification["qualification_reuse"]["source"])
        if reuse_source.is_absolute() or ".." in reuse_source.parts:
            raise AssertionError("qualification reuse source must be a safe relative path")

    check_relative_source_map(source_hashes)
    check_data_hash_map(data_hashes)
    au.verify_source_hashes(source_hashes)
    for relative_name, expected_hash in data_hashes.items():
        path = RUN / relative_name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise AssertionError(f"frozen input hash mismatch: {relative_name}")
    source_qualification_hash = sha256_file(RUN / "qualification.json")
    if source_qualification_hash != run_manifest.get("qualification_sha256"):
        raise AssertionError("qualification hash differs from frozen run manifest")
    for path in (RUN / "qualification.json", RUN / "aggregate.json", RUN / "status.json", RUN / "RESULTS.md"):
        privacy_check(path)

    reference_bindings = load_reference_bindings(run_manifest["reference_binding"])
    return run_manifest, qualification, aggregate, status, reference_bindings


def parameter_checkpoint_rows(run_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - export environment dependent
        raise RuntimeError("PyTorch is required only for CPU checkpoint metadata during export") from exc
    rows: list[dict[str, Any]] = []
    for block in range(3):
        folder = RUN / f"block{block:02d}/{ARM}/checkpoints"
        hashes: dict[int, str] = {}
        labels = [f"u{update:04d}.pt" for update in CHECKPOINT_UPDATES] + ["latest.pt"]
        for label in labels:
            path = folder / label
            if not path.is_file():
                raise AssertionError(f"missing checkpoint binding source: block{block:02d}/{label}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            expected_update = int(label[1:5]) if label.startswith("u") else 3000
            if checkpoint.get("config") != run_manifest["config"]:
                raise AssertionError(f"checkpoint config mismatch: block{block:02d}/{label}")
            if checkpoint.get("source_hashes") != run_manifest["source_hashes"]:
                raise AssertionError(f"checkpoint source hash mismatch: block{block:02d}/{label}")
            if checkpoint.get("update") != expected_update:
                raise AssertionError(f"checkpoint update mismatch: block{block:02d}/{label}")
            parameter_sha, parameter_count = au.parameter_hash(checkpoint["model"], torch)
            rows.append({
                "block": block,
                "arm": ARM,
                "checkpoint": label,
                "update": expected_update,
                "checkpoint_sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "parameter_sha256": parameter_sha,
                "parameter_count": parameter_count,
                "weights_published": False,
            })
            hashes[expected_update] = parameter_sha
            del checkpoint
            gc.collect()
        if hashes[3000] != next(row["parameter_sha256"] for row in rows if row["block"] == block and row["checkpoint"] == "latest.pt"):
            raise AssertionError(f"latest and named u3000 checkpoint parameter hashes differ: block{block:02d}")
    if len(rows) != 12:
        raise AssertionError(f"expected 12 checkpoint metadata bindings, got {len(rows)}")
    return rows


def sanitize_qualification(qualification: dict[str, Any], reference_bindings: dict[str, Any]) -> dict[str, Any]:
    """Retain qualification evidence while pointing its historical binding at the public package."""
    result = json.loads(json.dumps(qualification))
    reference = result.get("reference_binding", {})
    reference["reference"] = f"evidence/{REFERENCE_RUN_ID}"
    artifact_map = {
        **reference_bindings.get("historical_result_artifacts", {}),
        **reference_bindings.get("supporting_public_artifacts", {}),
    }
    relative_artifact_map = {
        path.removeprefix(f"evidence/{REFERENCE_RUN_ID}/"): digest
        for path, digest in artifact_map.items()
    }
    for name in reference.get("artifact_hashes", {}):
        if name not in relative_artifact_map:
            raise AssertionError(f"qualification refers to an unpinned historical artifact: {name}")
        reference["artifact_hashes"][name] = relative_artifact_map[name]
    result["reference_binding"] = reference
    return result


def write_checkpoint_bindings(path: Path, rows: list[dict[str, Any]]) -> None:
    write_json(path, {
        "schema": "relu-input-lift-checkpoint-bindings-v1",
        "checkpoint_weights_published": False,
        "bindings": rows,
    })


def build_results_markdown(aggregate: dict[str, Any]) -> str:
    arms = {(int(row["block"]), row["arm"]): row for row in aggregate["arms"] + aggregate["historical_controls"]}
    decision = aggregate["decision"]
    rows = [
        "# ReLU activation-conditioned input lift v0",
        "",
        "Execution **COMPLETE** (3/3 new arms); verdict **NO_QUALIFIED_RELU_LIFT_BENEFIT**.",
        "",
        "The primary endpoint is intact T64 NMSE at update 3000. Historical AU-NCA controls were reused from the pinned public package and were not retrained.",
        "",
        "| Block | Arm | T64 NMSE | T64 alpha IoU | T128 NMSE | T256 NMSE | Damaged T256 NMSE |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for block in range(3):
        for arm in (*HISTORICAL_ARMS, ARM):
            result = arms[(block, arm)]
            evaluation = result["evaluation"]
            values = [
                evaluation["intact_t64"]["nmse"]["mean"],
                evaluation["intact_t64"]["alpha_iou"]["mean"],
                evaluation["intact_t128"]["nmse"]["mean"],
                evaluation["intact_t256"]["nmse"]["mean"],
                evaluation["damaged_t256"]["nmse"]["mean"],
            ]
            rows.append(f"| {block} | {arm} | " + " | ".join(f"{float(value):.6f}" for value in values) + " |")
    gains = [float(row["au_minus_new"]) for row in decision["paired_primary"]]
    rows.extend([
        "",
        f"Mean paired AU-K8 minus ReLU-lift T64 NMSE is {float(decision['mean_au_minus_new']):.6f}; positive values favor ReLU-lift. It improves on AU-K8 in {decision['improving_blocks']}/3 blocks, below the frozen requirement of at least two improving blocks and mean gain at least 0.03.",
        "",
        "The independent unit is a paired initialization/schedule block (n=3); 32 evaluation episodes are repeated measurements within a block. The historical controls share the same target and packed input banks, pinned by SHA-256 in reference_bindings.json.",
        "",
        "T128/T256 same-state continuation, damaged-state evaluation, and gradient diagnostics are secondary. The result does not establish general NCA reliability or a universal BPTT solution.",
        "",
        f"The aggregate records elapsed_seconds_this_execution={float(aggregate['elapsed_seconds_this_execution']):.4f}; this is the scientific run's saved duration field.",
        "",
        "See [historical AU-NCA results](../au_nca_20261009_01/RESULTS.md), final_metrics.csv, per_episode.csv, and validation.json.",
        "",
    ])
    if len(gains) != 3:
        raise AssertionError("expected three paired primary blocks")
    return "\n".join(rows)


def write_reproduction(path: Path) -> None:
    path.write_text(
        "# ReLU input-lift evidence reproduction\n\n"
        "Start with RESULTS.md, final_metrics.csv, per_episode.csv, aggregate.json, validation.json, and reference_bindings.json. "
        "The package includes the fixed target, three packed input banks, all three new arm results, final saved predictions, "
        "intermediate diagnostics at updates 1000 and 2000, all 9000 training records, continuation curves, gradient diagnostics, "
        "and 12 metadata-only checkpoint bindings. It excludes checkpoint tensors, optimizer/pool state, worker logs, local receipts, and machine identifiers.\n\n"
        "From the project repository root, verify the package offline:\n\n"
        "```powershell\npython -X utf8 -B tools/export_relu_input_lift.py --verify-only\n```\n\n"
        "Verification recomputes metrics from the saved predictions and target, checks the saved training and continuation records, "
        "recomputes the frozen decision, and verifies current source hashes plus the SHA-pinned public AU-NCA controls. It does not open "
        "the private run or checkpoint files, initialize CUDA, train, or run model inference. Historical controls are linked from their "
        "existing public package; their prediction arrays and input banks are not duplicated here.\n\n"
        "## Fresh training\n\n"
        "Checkpoint weights are not shipped. A fresh ReLU input-lift run needs a local frozen AU-NCA reference: first create that reference "
        "with the qualification and training steps in [the AU-NCA protocol](../../new/au_nca/PROTOCOL.md), then pass its run directory "
        "with the ReLU runner's --reference option while qualifying and training under [the ReLU input-lift protocol](../../new/relu_input_lift/PROTOCOL.md). "
        "The published scores use the historical controls pinned in reference_bindings.json; a newly generated reference is a new experiment.\n",
        encoding="utf-8",
    )


def export_allowlist(
    run_manifest: dict[str, Any],
    qualification: dict[str, Any],
    aggregate: dict[str, Any],
    status: dict[str, Any],
    reference_bindings: dict[str, Any],
    checkpoint_rows: list[dict[str, Any]],
) -> None:
    if PUBLIC.exists() or PUBLICATION.exists():
        raise FileExistsError("ReLU input-lift public output already exists; preserve and inspect it before replacing")
    PUBLIC.mkdir(parents=True)
    for name in ("target_rgba.npy", "block00_inputs.npz", "block01_inputs.npz", "block02_inputs.npz", "aggregate.json", "status.json"):
        copy_file(RUN / name, PUBLIC / name)
    source_qualification_hash = sha256_file(RUN / "qualification.json")
    public_qualification = sanitize_qualification(qualification, reference_bindings)
    write_json(PUBLIC / "qualification.json", public_qualification)
    write_json(PUBLIC / "reference_bindings.json", reference_bindings)
    write_checkpoint_bindings(PUBLIC / "checkpoint_bindings.json", checkpoint_rows)

    public_manifest = {
        "schema": "relu-input-lift-public-evidence-v1",
        "run_id": RUN_ID,
        "status": "COMPLETE",
        "completed_units": 3,
        "planned_units": 3,
        "verdict": aggregate["decision"]["verdict"],
        "config": run_manifest["config"],
        "source_hashes": run_manifest["source_hashes"],
        "data_hashes": run_manifest["data_hashes"],
        "qualification_sha256": sha256_file(PUBLIC / "qualification.json"),
        "source_qualification_sha256": source_qualification_hash,
        "reference_bindings_sha256": sha256_file(PUBLIC / "reference_bindings.json"),
        "elapsed_seconds_this_execution": float(aggregate["elapsed_seconds_this_execution"]),
        "historical_controls_not_rerun": True,
        "private_artifacts_excluded": [
            "checkpoint tensor contents", "optimizer and pool state", "local dispatch receipts",
            "guard receipts and worker logs", "status lock files outside the saved scientific status", "machine identifiers",
        ],
        "checkpoint_bindings_note": "Local checkpoint payloads were inspected on CPU during export; public bindings contain hashes, sizes, update numbers, and canonical parameter hashes only.",
    }
    write_json(PUBLIC / "manifest.json", public_manifest)

    for block in range(3):
        private_arm = RUN / f"block{block:02d}/{ARM}"
        public_arm = PUBLIC / f"block{block:02d}/{ARM}"
        public_arm.mkdir(parents=True, exist_ok=True)
        for name in ("result.json", "training.jsonl", "predictions.npz", "continuation_curves.json", "gradient_diagnostics.json"):
            copy_file(private_arm / name, public_arm / name)
        for update in (1000, 2000):
            for suffix in ("json", "npz"):
                name = f"diagnostic_u{update:04d}.{suffix}"
                copy_file(private_arm / name, public_arm / name)

    (PUBLIC / "RESULTS.md").write_text(build_results_markdown(aggregate), encoding="utf-8")
    write_reproduction(PUBLIC / "REPRODUCTION.md")


def recompute_relu_decision(results: list[dict[str, Any]], controls: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Recompute the pure decision rule recorded by new/relu_input_lift/run.py."""
    rows: list[dict[str, Any]] = []
    for block in range(config["blocks"]):
        values = {
            result["arm"]: result.get("evaluation", {}).get("intact_t64", {}).get("nmse", {}).get("mean")
            for result in controls + results
            if int(result["block"]) == block
        }
        required = (ARM, "au_k8", "original_k64", "original_k8")
        if all(values.get(arm) is not None for arm in required):
            row = {arm: float(values[arm]) for arm in required}
            rows.append({
                "block": block,
                "nmse": row,
                "au_minus_new": row["au_k8"] - row[ARM],
                "half_remaining_gap_recovered": row[ARM] <= 0.5 * (row["au_k8"] + row["original_k64"]),
            })
    if len(results) < config["blocks"]:
        return {"verdict": "INCOMPLETE", "paired_primary": rows}
    if len(rows) < config["blocks"]:
        return {"verdict": "NUMERICAL_FAILURE_DEVELOPMENTAL", "paired_primary": rows}
    gains = [row["au_minus_new"] for row in rows]
    mean_gain = float(np.mean(gains))
    benefit = (
        mean_gain >= config["benefit_mean_gain_min"]
        and sum(gain > 0 for gain in gains) >= config["benefit_improving_blocks_min"]
        and sum(gain < -0.02 for gain in gains) <= config["benefit_worsening_over_02_blocks_max"]
    )
    strong = (
        benefit
        and float(np.mean([row["nmse"][ARM] for row in rows])) <= config["strong_mean_nmse_max"]
        and sum(row["half_remaining_gap_recovered"] for row in rows) >= config["strong_half_remaining_gap_blocks_min"]
    )
    verdict = "RELU_LIFT_RECOVERY_DEVELOPMENTAL" if strong else (
        "RELU_LIFT_BENEFIT_DEVELOPMENTAL" if benefit else "NO_QUALIFIED_RELU_LIFT_BENEFIT"
    )
    return {"verdict": verdict, "mean_au_minus_new": mean_gain, "improving_blocks": sum(gain > 0 for gain in gains), "paired_primary": rows}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def historical_metric_rows(reference_bindings: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old_final = read_csv(REFERENCE / "final_metrics.csv")
    old_episode = read_csv(REFERENCE / "per_episode_metrics.csv")
    if len(old_final) != 45 or len(old_episode) != 1440:
        raise AssertionError("historical public metric tables have unexpected row counts")
    final_rows = []
    for source in old_final:
        row = {
            "evidence_role": "historical_control",
            "source_run": REFERENCE_RUN_ID,
            "block": int(source["block"]),
            "arm": source["arm"],
            "branch": source["branch"],
            "episodes": int(source["episodes"]),
        }
        for metric in ("mse_mean", "mse_std", "nmse_mean", "nmse_std", "alpha_iou_mean", "alpha_iou_std"):
            row[metric] = float(source[metric])
        final_rows.append(row)
    episode_rows = []
    for source in old_episode:
        episode_rows.append({
            "evidence_role": "historical_control",
            "source_run": REFERENCE_RUN_ID,
            "block": int(source["block"]),
            "arm": source["arm"],
            "branch": source["branch"],
            "episode": int(source["episode"]),
            "mse": float(source["mse"]),
            "nmse": float(source["nmse"]),
            "alpha_iou": float(source["alpha_iou"]),
        })
    old_aggregate = read_json(REFERENCE / "aggregate.json")
    old_results = {(int(row["block"]), row["arm"]): row for row in old_aggregate["arms"]}
    expected_final = {(row["block"], row["arm"], row["branch"]): row for row in final_rows}
    expected_episode = {(row["block"], row["arm"], row["branch"], row["episode"]): row for row in episode_rows}
    if len(expected_final) != 45 or len(expected_episode) != 1440:
        raise AssertionError("historical metric keys are duplicated")
    for key, result in old_results.items():
        for branch in BRANCHES:
            evaluation = result["evaluation"][branch]
            row = expected_final[(key[0], key[1], branch)]
            if row["episodes"] != 32:
                raise AssertionError(f"historical episode count mismatch: {key}/{branch}")
            for metric in ("mse", "nmse", "alpha_iou"):
                assert_close(row[f"{metric}_mean"], evaluation[metric]["mean"], f"historical {key}/{branch}/{metric}.mean")
                assert_close(row[f"{metric}_std"], evaluation[metric]["std"], f"historical {key}/{branch}/{metric}.std")
                for episode, value in enumerate(evaluation[metric]["per_episode"]):
                    episode_row = expected_episode[(key[0], key[1], branch, episode)]
                    assert_close(episode_row[metric], value, f"historical {key}/{branch}/{metric}/episode{episode}")
    return final_rows, episode_rows


def verify_reference_bindings(reference_bindings: dict[str, Any], manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if reference_bindings.get("schema") != "relu-input-lift-reference-bindings-v1":
        raise AssertionError("unexpected historical reference binding schema")
    if reference_bindings.get("historical_run_id") != REFERENCE_RUN_ID or reference_bindings.get("historical_public_package") != f"evidence/{REFERENCE_RUN_ID}":
        raise AssertionError("historical public reference path mismatch")
    if reference_bindings.get("historical_control_results_not_rerun") is not True:
        raise AssertionError("historical controls must remain labeled as not rerun")
    if reference_bindings.get("shared_target_and_input_hashes") != manifest.get("data_hashes"):
        raise AssertionError("new and historical target/input hash bindings differ")
    if reference_bindings.get("no_historical_predictions_or_input_banks_duplicated") is not True:
        raise AssertionError("historical package duplication boundary is missing")

    reference_publication = read_json(REFERENCE_PUBLICATION)
    expected_artifacts = {
        **reference_bindings.get("historical_result_artifacts", {}),
        **reference_bindings.get("supporting_public_artifacts", {}),
    }
    if len(expected_artifacts) != 15:
        raise AssertionError("reference binding does not pin the nine results and supporting public artifacts")
    public_files = reference_publication.get("published_artifacts", {})
    for relative_name, expected_hash in expected_artifacts.items():
        if not relative_name.startswith(f"evidence/{REFERENCE_RUN_ID}/"):
            raise AssertionError(f"unsafe public reference path: {relative_name}")
        entry = public_files.get(relative_name)
        local_path = ROOT / Path(relative_name)
        if not entry or entry.get("sha256") != expected_hash or sha256_file(local_path) != expected_hash:
            raise AssertionError(f"historical reference pin mismatch: {relative_name}")
    aggregate_hash = sha256_file(REFERENCE / "aggregate.json")
    if reference_bindings.get("control_aggregate_sha256") != aggregate_hash:
        raise AssertionError("historical aggregate reference hash mismatch")

    old_manifest = read_json(REFERENCE / "manifest.json")
    old_aggregate = read_json(REFERENCE / "aggregate.json")
    old_qualification = read_json(REFERENCE / "qualification.json")
    if old_manifest.get("data_hashes") != manifest.get("data_hashes"):
        raise AssertionError("historical manifest data hashes differ")
    if old_qualification.get("status") != "PASS" or old_manifest.get("status") != "COMPLETE":
        raise AssertionError("historical qualification or status is not complete")
    old_decision = au.compute_decision(old_aggregate["arms"], old_manifest["config"])
    if old_decision != old_aggregate.get("decision"):
        raise AssertionError("historical control decision does not recompute")
    history = {(int(row["block"]), row["arm"]): row for row in old_aggregate["arms"]}
    if set(history) != {(block, arm) for block in range(3) for arm in HISTORICAL_ARMS}:
        raise AssertionError("historical aggregate does not have all nine frozen controls")
    final_rows, episode_rows = historical_metric_rows(reference_bindings)
    return final_rows, episode_rows


def write_metric_tables(folder: Path, final_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]]) -> None:
    final_fields = ("evidence_role", "source_run", "block", "arm", "branch", "episodes", "mse_mean", "mse_std", "nmse_mean", "nmse_std", "alpha_iou_mean", "alpha_iou_std")
    with (folder / "final_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        for row in final_rows:
            writer.writerow({key: format(row[key], ".17g") if isinstance(row[key], float) else row[key] for key in final_fields})
    episode_fields = ("evidence_role", "source_run", "block", "arm", "branch", "episode", "mse", "nmse", "alpha_iou")
    with (folder / "per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=episode_fields)
        writer.writeheader()
        for row in episode_rows:
            writer.writerow({key: format(row[key], ".17g") if isinstance(row[key], float) else row[key] for key in episode_fields})


def verify_metric_tables(folder: Path, expected_final: list[dict[str, Any]], expected_episode: list[dict[str, Any]]) -> None:
    final_rows = read_csv(folder / "final_metrics.csv")
    episode_rows = read_csv(folder / "per_episode.csv")
    if len(final_rows) != 60 or len(episode_rows) != 1920:
        raise AssertionError("final or per-episode metric table has an unexpected row count")
    key_fields = ("evidence_role", "source_run", "block", "arm", "branch")
    expected = {tuple(str(row[key]) for key in key_fields): row for row in expected_final}
    if len(expected) != 60:
        raise AssertionError("duplicate expected final metric table keys")
    seen = set()
    for saved in final_rows:
        key = tuple(str(int(saved[field]) if field == "block" else saved[field]) for field in key_fields)
        if key in seen or key not in expected:
            raise AssertionError(f"unexpected or duplicate final metric row: {key}")
        seen.add(key)
        want = expected[key]
        if int(saved["episodes"]) != int(want["episodes"]):
            raise AssertionError(f"final metric episode count mismatch: {key}")
        for field in ("mse_mean", "mse_std", "nmse_mean", "nmse_std", "alpha_iou_mean", "alpha_iou_std"):
            assert_close(float(saved[field]), float(want[field]), f"final_metrics.csv {key}/{field}")
    episode_key_fields = (*key_fields, "episode")
    expected_eps = {tuple(str(row[key]) for key in episode_key_fields): row for row in expected_episode}
    if len(expected_eps) != 1920:
        raise AssertionError("duplicate expected episode table keys")
    seen_eps = set()
    for saved in episode_rows:
        key = tuple(str(int(saved[field]) if field in ("block", "episode") else saved[field]) for field in episode_key_fields)
        if key in seen_eps or key not in expected_eps:
            raise AssertionError(f"unexpected or duplicate per-episode row: {key}")
        seen_eps.add(key)
        want = expected_eps[key]
        for field in ("mse", "nmse", "alpha_iou"):
            assert_close(float(saved[field]), float(want[field]), f"per_episode.csv {key}/{field}")


def validate_public_payload(folder: Path, *, check_current_sources: bool) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_json(folder / "manifest.json")
    aggregate = read_json(folder / "aggregate.json")
    qualification = read_json(folder / "qualification.json")
    status = read_json(folder / "status.json")
    reference_bindings = read_json(folder / "reference_bindings.json")
    if manifest.get("schema") != "relu-input-lift-public-evidence-v1" or manifest.get("run_id") != RUN_ID:
        raise AssertionError("unexpected public manifest schema or run id")
    if manifest.get("status") != "COMPLETE" or manifest.get("completed_units") != 3 or manifest.get("planned_units") != 3:
        raise AssertionError("public manifest does not record 3/3 completion")
    if aggregate.get("status") != "COMPLETE" or len(aggregate.get("arms", [])) != 3:
        raise AssertionError("public aggregate is incomplete")
    if aggregate.get("config") != manifest.get("config") or qualification.get("config") != manifest.get("config"):
        raise AssertionError("public configuration bindings disagree")
    if status.get("status") != "COMPLETE" or status.get("decision") != aggregate.get("decision"):
        raise AssertionError("public scientific status differs from aggregate")
    if status.get("completed_units") != 3 or status.get("planned_units") != 3:
        raise AssertionError("public scientific status does not show 3/3 completion")
    if qualification.get("status") != "PASS" or qualification.get("source_hashes") != manifest.get("source_hashes"):
        raise AssertionError("public qualification/source binding mismatch")
    if qualification.get("reference_binding", {}).get("reference") != f"evidence/{REFERENCE_RUN_ID}":
        raise AssertionError("qualification has an unnormalized historical reference path")
    if qualification.get("reference_binding", {}).get("data_hashes") != manifest.get("data_hashes"):
        raise AssertionError("qualification and public manifest shared data hashes differ")
    qualification_reference_hashes = qualification.get("reference_binding", {}).get("artifact_hashes", {})
    reference_artifact_hashes = {
        path.removeprefix(f"evidence/{REFERENCE_RUN_ID}/"): digest
        for path, digest in {
            **reference_bindings.get("historical_result_artifacts", {}),
            **reference_bindings.get("supporting_public_artifacts", {}),
        }.items()
    }
    if any(reference_artifact_hashes.get(name) != digest for name, digest in qualification_reference_hashes.items()):
        raise AssertionError("sanitized qualification historical artifact pins differ from reference_bindings.json")
    if sha256_file(folder / "qualification.json") != manifest.get("qualification_sha256"):
        raise AssertionError("public qualification file hash mismatch")
    if sha256_file(folder / "reference_bindings.json") != manifest.get("reference_bindings_sha256"):
        raise AssertionError("public reference binding file hash mismatch")
    check_relative_source_map(manifest["source_hashes"])
    check_data_hash_map(manifest["data_hashes"])
    if check_current_sources:
        au.verify_source_hashes(manifest["source_hashes"])
    for path in (folder / "manifest.json", folder / "aggregate.json", folder / "status.json", folder / "qualification.json", folder / "reference_bindings.json", folder / "RESULTS.md", folder / "REPRODUCTION.md"):
        privacy_check(path)

    for relative_name, digest in manifest["data_hashes"].items():
        path = folder / relative_name
        if not path.is_file() or sha256_file(path) != digest:
            raise AssertionError(f"public data hash mismatch: {relative_name}")
    bank_count = au.check_input_banks(folder, {"data_hashes": manifest["data_hashes"]})
    target = np.load(folder / "target_rgba.npy", allow_pickle=False)
    if target.shape != (1, 4, 32, 32) or not np.isfinite(target).all():
        raise AssertionError("invalid public target RGBA array")

    controls_final_rows, controls_episode_rows = verify_reference_bindings(reference_bindings, manifest)
    result_map = {(int(row["block"]), row["arm"]): row for row in aggregate["arms"]}
    if set(result_map) != {(block, ARM) for block in range(3)}:
        raise AssertionError("aggregate new-arm identities are incomplete or duplicated")
    max_error = 0.0
    new_final_rows: list[dict[str, Any]] = []
    new_episode_rows: list[dict[str, Any]] = []
    for block in range(3):
        base = folder / f"block{block:02d}/{ARM}"
        result = read_json(base / "result.json")
        if result != result_map[(block, ARM)] or result.get("status") != "COMPLETE":
            raise AssertionError(f"block result differs from aggregate: block{block:02d}")
        if result.get("block") != block or result.get("arm") != ARM or result.get("evaluation", {}).get("episodes") != 32:
            raise AssertionError(f"block result identity or evaluation count is invalid: block{block:02d}")
        if result.get("init_seed") != manifest["config"]["init_seeds"][block] or result.get("schedule_seed") != manifest["config"]["schedule_seeds"][block]:
            raise AssertionError(f"block seed binding mismatch: block{block:02d}")
        privacy_check(base / "result.json")

        with np.load(base / "predictions.npz", allow_pickle=False) as packed:
            if set(packed.files) != set(BRANCHES):
                raise AssertionError(f"final prediction names differ: block{block:02d}")
            predictions = {key: packed[key] for key in BRANCHES}
        for branch in BRANCHES:
            prediction = predictions[branch]
            if prediction.shape != (32, 4, 32, 32) or not np.isfinite(prediction).all():
                raise AssertionError(f"invalid saved predictions: block{block:02d}/{branch}")
            values = au.calculate_metrics(prediction, target[0])
            max_error = max(max_error, au.compare_saved_metrics(values, result["evaluation"][branch], f"block{block:02d}/{branch}"))
            metric_row = {
                "evidence_role": "new_arm",
                "source_run": RUN_ID,
                "block": block,
                "arm": ARM,
                "branch": branch,
                "episodes": 32,
            }
            for metric in ("mse", "nmse", "alpha_iou"):
                metric_row[f"{metric}_mean"] = float(values[metric].mean())
                metric_row[f"{metric}_std"] = float(values[metric].std(ddof=1))
            new_final_rows.append(metric_row)
            for episode in range(32):
                new_episode_rows.append({
                    "evidence_role": "new_arm",
                    "source_run": RUN_ID,
                    "block": block,
                    "arm": ARM,
                    "branch": branch,
                    "episode": episode,
                    "mse": float(values["mse"][episode]),
                    "nmse": float(values["nmse"][episode]),
                    "alpha_iou": float(values["alpha_iou"][episode]),
                })

        curve = read_json(base / "continuation_curves.json")
        if len(curve) != 1024:
            raise AssertionError(f"expected 1024 continuation rows: block{block:02d}")
        curve_map: dict[tuple[int, int], dict[str, Any]] = {}
        for entry in curve:
            fields = ("episode", "step", "mse", "nmse", "alpha_iou", "state_rms", "alive_fraction")
            if not all(math.isfinite(float(entry[field])) for field in fields):
                raise AssertionError(f"nonfinite continuation value: block{block:02d}")
            key = (int(entry["episode"]), int(entry["step"]))
            if key in curve_map:
                raise AssertionError(f"duplicate continuation row {key}: block{block:02d}")
            curve_map[key] = entry
        expected_curve = {(episode, step) for episode in range(32) for step in range(8, 257, 8)}
        if set(curve_map) != expected_curve:
            raise AssertionError(f"continuation grid mismatch: block{block:02d}")
        for step, branch in ((64, "intact_t64"), (128, "intact_t128"), (256, "intact_t256")):
            for metric in ("mse", "nmse", "alpha_iou"):
                max_error = max(max_error, assert_close(
                    [curve_map[(episode, step)][metric] for episode in range(32)],
                    result["evaluation"][branch][metric]["per_episode"],
                    f"continuation block{block:02d}/{branch}/{metric}",
                ))

        training = read_jsonl(base / "training.jsonl")
        if [row.get("update") for row in training] != list(range(1, 3001)):
            raise AssertionError(f"training records are not unique updates 1..3000: block{block:02d}")
        for update, row in enumerate(training, 1):
            if set(row) != set(TRAINING_FIELDS) or row.get("update") != update:
                raise AssertionError(f"unexpected training fields/update: block{block:02d}/u{update}")
            if not all(math.isfinite(float(row[field])) for field in TRAINING_FIELDS[1:]):
                raise AssertionError(f"nonfinite training metric: block{block:02d}/u{update}")
        privacy_check(base / "training.jsonl")

        for update in (1000, 2000):
            diagnostic = read_json(base / f"diagnostic_u{update:04d}.json")
            if diagnostic.get("episodes") != 8 or diagnostic.get("same_state_continuation") is not False:
                raise AssertionError(f"intermediate diagnostic metadata mismatch: block{block:02d}/u{update}")
            with np.load(base / f"diagnostic_u{update:04d}.npz", allow_pickle=False) as packed:
                if set(packed.files) != {"intact_t64"}:
                    raise AssertionError(f"intermediate prediction names differ: block{block:02d}/u{update}")
                diagnostic_prediction = packed["intact_t64"]
            if diagnostic_prediction.shape != (8, 4, 32, 32) or not np.isfinite(diagnostic_prediction).all():
                raise AssertionError(f"invalid intermediate prediction bank: block{block:02d}/u{update}")
            values = au.calculate_metrics(diagnostic_prediction, target[0])
            max_error = max(max_error, au.compare_saved_metrics(values, diagnostic["intact_t64"], f"diagnostic block{block:02d}/u{update}"))
        gradient_diagnostics = read_json(base / "gradient_diagnostics.json")
        if not isinstance(gradient_diagnostics, dict) or not gradient_diagnostics.get("same_parameters_inputs_loss"):
            raise AssertionError(f"gradient diagnostics are incomplete: block{block:02d}")
        privacy_check(base / "gradient_diagnostics.json")

    recomputed_decision = recompute_relu_decision(aggregate["arms"], aggregate["historical_controls"], manifest["config"])
    if recomputed_decision != aggregate.get("decision"):
        raise AssertionError("public verdict/decision does not recompute from saved outcomes")
    if manifest.get("verdict") != recomputed_decision["verdict"] or manifest["verdict"] != "NO_QUALIFIED_RELU_LIFT_BENEFIT":
        raise AssertionError("public verdict does not match the frozen decision")
    if len(aggregate.get("historical_controls", [])) != 9:
        raise AssertionError("public aggregate does not include nine pinned historical controls")
    historical_map = {(int(row["block"]), row["arm"]): row for row in aggregate["historical_controls"]}
    old_aggregate = read_json(REFERENCE / "aggregate.json")
    old_map = {(int(row["block"]), row["arm"]): row for row in old_aggregate["arms"]}
    if historical_map != old_map:
        raise AssertionError("embedded historical controls differ from the public pinned controls")

    final_rows = new_final_rows + controls_final_rows
    episode_rows = new_episode_rows + controls_episode_rows
    tables_exist = (folder / "final_metrics.csv").exists(), (folder / "per_episode.csv").exists()
    if tables_exist[0] != tables_exist[1]:
        raise AssertionError("metric tables must be published together")
    if all(tables_exist):
        verify_metric_tables(folder, final_rows, episode_rows)

    bindings = read_json(folder / "checkpoint_bindings.json")
    if bindings.get("schema") != "relu-input-lift-checkpoint-bindings-v1" or bindings.get("checkpoint_weights_published") is not False:
        raise AssertionError("checkpoint binding metadata is malformed or exposes weights")
    binding_rows = bindings.get("bindings", [])
    expected_binding_keys = {(block, ARM, label, update) for block in range(3) for label, update in (("u1000.pt", 1000), ("u2000.pt", 2000), ("u3000.pt", 3000), ("latest.pt", 3000))}
    actual_binding_keys: set[tuple[int, str, str, int]] = set()
    latest_hashes: dict[int, str] = {}
    final_hashes: dict[int, str] = {}
    for row in binding_rows:
        key = (int(row["block"]), row["arm"], row["checkpoint"], int(row["update"]))
        actual_binding_keys.add(key)
        if not re.fullmatch(r"[0-9a-f]{64}", row["checkpoint_sha256"]) or not re.fullmatch(r"[0-9a-f]{64}", row["parameter_sha256"]):
            raise AssertionError("malformed checkpoint hash binding")
        if int(row["size_bytes"]) <= 0 or int(row["parameter_count"]) != 8336 or row["weights_published"] is not False:
            raise AssertionError("invalid or unsafe checkpoint metadata")
        if row["checkpoint"] == "latest.pt":
            latest_hashes[int(row["block"])] = row["parameter_sha256"]
        if row["checkpoint"] == "u3000.pt":
            final_hashes[int(row["block"])] = row["parameter_sha256"]
    if len(binding_rows) != 12 or actual_binding_keys != expected_binding_keys or latest_hashes != final_hashes:
        raise AssertionError("checkpoint binding set or latest/u3000 parameter hash comparison failed")

    return {
        "status": "PASS",
        "scope": "Offline public evidence integrity; no private run/checkpoint payload, training, model inference, or CUDA.",
        "new_arms_complete": 3,
        "historical_controls_pinned_not_rerun": 9,
        "input_banks": bank_count,
        "canonical_training_updates": 9000,
        "training_update_range_per_block": "1..3000, unique and ordered",
        "final_metric_rows": len(final_rows),
        "new_final_metric_rows": len(new_final_rows),
        "historical_final_metric_rows": len(controls_final_rows),
        "per_episode_metric_rows": len(episode_rows),
        "new_per_episode_metric_rows": len(new_episode_rows),
        "historical_per_episode_metric_rows": len(controls_episode_rows),
        "intermediate_diagnostics": 6,
        "intermediate_prediction_episodes": 48,
        "continuation_curve_records": 3072,
        "checkpoint_bindings": len(binding_rows),
        "checkpoint_weights_published": False,
        "max_metric_recomputation_abs_error": max_error,
        "decision_recomputed": recomputed_decision,
        "elapsed_seconds_this_execution": float(aggregate["elapsed_seconds_this_execution"]),
        "verified_source_hashes": manifest["source_hashes"],
        "verified_data_hashes": manifest["data_hashes"],
        "public_qualification_sha256": manifest["qualification_sha256"],
        "source_qualification_sha256": manifest["source_qualification_sha256"],
        "reference_bindings_sha256": manifest["reference_bindings_sha256"],
        "historical_control_result_hashes": reference_bindings["historical_result_artifacts"],
    }, final_rows, episode_rows


def package_file_map(folder: Path) -> tuple[dict[str, dict[str, Any]], int]:
    return au.package_file_map(folder)


def export() -> dict[str, Any]:
    run_manifest, qualification, aggregate, status, reference_bindings = validate_private_inputs()
    checkpoint_rows = parameter_checkpoint_rows(run_manifest)
    export_allowlist(run_manifest, qualification, aggregate, status, reference_bindings, checkpoint_rows)
    validation, final_rows, episode_rows = validate_public_payload(PUBLIC, check_current_sources=True)
    write_metric_tables(PUBLIC, final_rows, episode_rows)
    validation, final_rows, episode_rows = validate_public_payload(PUBLIC, check_current_sources=True)
    write_json(PUBLIC / "validation.json", validation)
    artifacts, package_bytes = package_file_map(PUBLIC)
    publication = {
        "schema": "relu-input-lift-publication-v1",
        "run_id": RUN_ID,
        "published_artifacts": artifacts,
        "evidence_package_bytes": package_bytes,
        "validation": validation,
        "checkpoint_contents_published": False,
        "private_run_preserved": True,
    }
    write_json(PUBLICATION, publication)
    return verify_public()


def refresh_publication() -> dict[str, Any]:
    """Refresh derived summaries for an already-created public package only."""
    if not PUBLIC.is_dir() or not PUBLICATION.is_file():
        raise FileNotFoundError("an existing public package and publication manifest are required")
    write_reproduction(PUBLIC / "REPRODUCTION.md")
    validation, _final_rows, _episode_rows = validate_public_payload(PUBLIC, check_current_sources=True)
    write_json(PUBLIC / "validation.json", validation)
    artifacts, package_bytes = package_file_map(PUBLIC)
    publication = read_json(PUBLICATION)
    if publication.get("schema") != "relu-input-lift-publication-v1" or publication.get("run_id") != RUN_ID:
        raise AssertionError("refusing to refresh an unrelated publication manifest")
    publication["published_artifacts"] = artifacts
    publication["evidence_package_bytes"] = package_bytes
    publication["validation"] = validation
    write_json(PUBLICATION, publication)
    return verify_public()


def verify_public() -> dict[str, Any]:
    publication = read_json(PUBLICATION)
    if publication.get("schema") != "relu-input-lift-publication-v1" or publication.get("run_id") != RUN_ID:
        raise AssertionError("unexpected ReLU input-lift publication identity")
    artifacts, package_bytes = package_file_map(PUBLIC)
    if artifacts != publication.get("published_artifacts"):
        raise AssertionError("public package file hashes/sizes differ from publication manifest")
    if package_bytes != publication.get("evidence_package_bytes"):
        raise AssertionError("public package byte count differs from publication manifest")
    validation, _final_rows, _episode_rows = validate_public_payload(PUBLIC, check_current_sources=True)
    if validation != read_json(PUBLIC / "validation.json") or validation != publication.get("validation"):
        raise AssertionError("offline public verification differs from stored validation")
    return {
        "status": "PASS",
        "files": len(artifacts),
        "evidence_package_bytes": package_bytes,
        "canonical_training_updates": validation["canonical_training_updates"],
        "final_metric_rows": validation["final_metric_rows"],
        "per_episode_metric_rows": validation["per_episode_metric_rows"],
        "intermediate_diagnostics": validation["intermediate_diagnostics"],
        "checkpoint_bindings": validation["checkpoint_bindings"],
        "checkpoint_weights_published": False,
        "historical_controls_pinned_not_rerun": validation["historical_controls_pinned_not_rerun"],
        "verdict": validation["decision_recomputed"]["verdict"],
        "method": "public saved arrays, JSON, CSV, source/data hashes, and pinned historical public artifacts only; no private run/checkpoint files, training, model inference, or CUDA",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-only", action="store_true", help="verify the public package without opening the private run or checkpoint files")
    modes.add_argument("--refresh-publication", action="store_true", help="refresh derived public reproduction and validation summaries without opening the private run")
    args = parser.parse_args(argv)
    try:
        summary = verify_public() if args.verify_only else (refresh_publication() if args.refresh_publication else export())
    except Exception as exc:
        print(f"ReLU input-lift evidence export/verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
