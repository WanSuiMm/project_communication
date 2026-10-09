"""Export and verify the compact public AU-NCA v0 evidence package.

Exporting reads the completed private run and checkpoints on CPU, then copies
only the allowlisted evidence. ``--verify-only`` reads the public package and
repository source files; it never opens the private run or checkpoint files.
Neither path performs model inference or training.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/au_nca_20261009_01"
PUBLIC = ROOT / "evidence/au_nca_20261009_01"
PUBLICATION = ROOT / "AU_NCA_PUBLICATION_MANIFEST.json"
ARMS = ("original_k64", "original_k8", "au_k8")
BRANCHES = ("intact_t64", "intact_t128", "intact_t256", "damaged_t128", "damaged_t256")
PARAMETER_KEYS = ("feature.weight", "feature.bias", "projection.weight")
TRAINING_FIELDS = ("update", "loss", "W_gradient_norm_before_normalization", "eta_gradient_norm_before_normalization")
CHECKPOINT_UPDATES = (1000, 2000, 3000)
TOL = 1e-12

PRIVATE_TEXT = re.compile(
    r"(?i)(?:[A-Z]:\\(?:Users|Storage)\\|[A-Z]:/(?:Users|Storage)/|"
    r"\b(?:worker_pid|child_pid|hostname|gpu_uuid|username)\b|"
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b)"
)
ABSOLUTE_PATH = re.compile(r"(?i)(?:[A-Z]:\\|/home/|/mnt/[a-z]/)")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def public_path(block: int, arm: str, filename: str) -> Path:
    return PUBLIC / f"block{block:02d}" / arm / filename


def assert_close(actual: Any, expected: Any, label: str, *, atol: float = TOL, rtol: float = TOL) -> float:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape:
        raise AssertionError(f"{label}: shape {left.shape} != {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise AssertionError(f"{label}: nonfinite metric")
    difference = float(np.max(np.abs(left - right))) if left.size else 0.0
    if not np.allclose(left, right, atol=atol, rtol=rtol):
        raise AssertionError(f"{label}: max absolute difference {difference:.17g} exceeds tolerance")
    return difference


def calculate_metrics(predictions: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predictions.ndim != 4 or predictions.shape[1:] != target.shape:
        raise AssertionError(f"prediction/target shapes disagree: {predictions.shape} vs {target.shape}")
    if predictions.shape[1] != 4 or target.shape[0] != 4:
        raise AssertionError("RGBA metric inputs must have exactly four channels")
    if not np.isfinite(predictions).all() or not np.isfinite(target).all():
        raise AssertionError("prediction or target contains nonfinite values")
    mse = np.square(predictions - target).mean(axis=(1, 2, 3))
    blank_mse = float(np.square(target).mean())
    if not math.isfinite(blank_mse) or blank_mse <= 0:
        raise AssertionError("blank-image MSE must be finite and positive")
    predicted_alpha = predictions[:, 3] > 0.5
    target_alpha = target[3] > 0.5
    intersection = np.logical_and(predicted_alpha, target_alpha).sum(axis=(1, 2))
    union = np.logical_or(predicted_alpha, target_alpha).sum(axis=(1, 2))
    return {
        "mse": mse,
        "nmse": mse / blank_mse,
        "alpha_iou": intersection / np.maximum(union, 1),
    }


def compare_saved_metrics(calculated: dict[str, np.ndarray], saved: dict[str, Any], label: str) -> float:
    largest = 0.0
    for metric in ("mse", "nmse", "alpha_iou"):
        summary = saved[metric]
        largest = max(largest, assert_close(calculated[metric], summary["per_episode"], f"{label}.{metric}.per_episode"))
        largest = max(largest, assert_close(float(calculated[metric].mean()), summary["mean"], f"{label}.{metric}.mean"))
        std = float(calculated[metric].std(ddof=1)) if len(calculated[metric]) > 1 else 0.0
        largest = max(largest, assert_close(std, summary["std"], f"{label}.{metric}.std"))
    return largest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path.name}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise AssertionError(f"{path.name}:{line_number}: expected an object")
            rows.append(row)
    return rows


def verify_source_hashes(source_hashes: dict[str, str], captured_root: Path | None = None) -> None:
    for rel, expected in source_hashes.items():
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise AssertionError(f"unsafe source binding: {rel}")
        current = ROOT / rel_path
        if not current.is_file() or sha256_file(current) != expected:
            raise AssertionError(f"current source hash mismatch: {rel}")
        if captured_root is not None:
            captured = captured_root / rel_path
            if not captured.is_file() or sha256_file(captured) != expected:
                raise AssertionError(f"frozen source snapshot hash mismatch: {rel}")


def privacy_check(path: Path) -> None:
    if path.suffix.lower() not in (".json", ".jsonl", ".md", ".csv"):
        return
    content = path.read_text(encoding="utf-8")
    match = PRIVATE_TEXT.search(content)
    if match:
        raise AssertionError(f"private host/account identifier in public artifact {path.name}: {match.group(0)!r}")


def validate_private_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_manifest = read_json(RUN / "manifest.json")
    qualification = read_json(RUN / "qualification.json")
    aggregate = read_json(RUN / "aggregate.json")
    status = read_json(RUN / "status.json")
    config = run_manifest["config"]
    source_hashes = run_manifest["source_hashes"]
    if config != qualification["config"] or config != aggregate["config"]:
        raise AssertionError("run, qualification, and aggregate configurations differ")
    if source_hashes != qualification["source_hashes"]:
        raise AssertionError("qualification source bindings differ from the frozen run manifest")
    if qualification.get("status") != "PASS" or aggregate.get("status") != "COMPLETE" or status.get("status") != "COMPLETE":
        raise AssertionError("run is not both qualified and complete")
    if status.get("completed_units") != 9 or status.get("planned_units") != 9:
        raise AssertionError("run completion count is not 9/9")
    if len(aggregate.get("arms", [])) != 9 or aggregate.get("completed_units") != 9 or aggregate.get("planned_units") != 9:
        raise AssertionError("aggregate does not contain all nine arms")
    if aggregate["decision"]["verdict"] != "AU_BENEFIT_DEVELOPMENTAL":
        raise AssertionError("unexpected frozen run verdict")
    verify_source_hashes(source_hashes, RUN / "source")
    for rel, digest in run_manifest["data_hashes"].items():
        path = RUN / rel
        if not path.is_file() or sha256_file(path) != digest:
            raise AssertionError(f"frozen input hash mismatch: {rel}")
    qual_hash = sha256_file(RUN / "qualification.json")
    if run_manifest.get("qualification_sha256") != qual_hash:
        raise AssertionError("qualification hash differs from frozen run manifest")
    privacy_check(RUN / "qualification.json")
    return run_manifest, qualification, aggregate, status


def parameter_hash(state: dict[str, Any], torch: Any) -> tuple[str, int]:
    if set(PARAMETER_KEYS) - set(state):
        raise AssertionError("checkpoint is missing a canonical trainable parameter")
    unexpected = set(state) - set(PARAMETER_KEYS) - {"perception.sobel_pair"}
    if unexpected:
        raise AssertionError(f"unexpected checkpoint model entries: {sorted(unexpected)}")
    digest = hashlib.sha256(b"au-nca-canonical-trainable-parameters-v1\0")
    count = 0
    for name in PARAMETER_KEYS:
        value = state[name]
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise AssertionError(f"checkpoint parameter {name} is not a finite tensor")
        tensor = value.detach().to(device="cpu").contiguous()
        count += tensor.numel()
        descriptor = {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    if count != 8336:
        raise AssertionError(f"expected 8336 trainable scalars, got {count}")
    return digest.hexdigest(), count


def inspect_checkpoints(run_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PyTorch is required only for export-time CPU checkpoint bindings") from exc
    rows: list[dict[str, Any]] = []
    for block in range(3):
        for arm in ARMS:
            folder = RUN / f"block{block:02d}" / arm / "checkpoints"
            labels = [f"u{update:04d}.pt" for update in CHECKPOINT_UPDATES] + ["latest.pt"]
            arm_hashes: dict[int, str] = {}
            for label in labels:
                path = folder / label
                if not path.is_file():
                    raise AssertionError(f"missing saved checkpoint: block{block:02d}/{arm}/{label}")
                # CPU-only loading; inspect metadata and canonical parameters, then release this payload.
                value = torch.load(path, map_location="cpu", weights_only=False)
                expected_update = int(label[1:5]) if label.startswith("u") else 3000
                if value.get("config") != run_manifest["config"]:
                    raise AssertionError(f"checkpoint config mismatch: block{block:02d}/{arm}/{label}")
                if value.get("source_hashes") != run_manifest["source_hashes"]:
                    raise AssertionError(f"checkpoint source binding mismatch: block{block:02d}/{arm}/{label}")
                if value.get("update") != expected_update:
                    raise AssertionError(f"checkpoint update mismatch: block{block:02d}/{arm}/{label}")
                digest, parameter_count = parameter_hash(value["model"], torch)
                file_sha = sha256_file(path)
                rows.append({
                    "block": block,
                    "arm": arm,
                    "checkpoint": label,
                    "update": expected_update,
                    "checkpoint_sha256": file_sha,
                    "size_bytes": path.stat().st_size,
                    "parameter_sha256": digest,
                    "parameter_count": parameter_count,
                    "weights_published": False,
                })
                arm_hashes[expected_update] = digest
                del value
                gc.collect()
            latest_hash = next(row["parameter_sha256"] for row in rows if row["block"] == block and row["arm"] == arm and row["checkpoint"] == "latest.pt")
            named_hash = next(row["parameter_sha256"] for row in rows if row["block"] == block and row["arm"] == arm and row["checkpoint"] == "u3000.pt")
            if latest_hash != named_hash:
                raise AssertionError(f"latest and u3000 model parameters differ: block{block:02d}/{arm}")
    if len(rows) != 36:
        raise AssertionError(f"expected 36 checkpoint bindings, got {len(rows)}")
    return rows


def write_checkpoint_bindings(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("block", "arm", "checkpoint", "update", "checkpoint_sha256", "size_bytes", "parameter_sha256", "parameter_count", "weights_published")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_allowlist(run_manifest: dict[str, Any], qualification: dict[str, Any], aggregate: dict[str, Any], checkpoint_rows: list[dict[str, Any]]) -> None:
    if PUBLIC.exists() or PUBLICATION.exists():
        raise FileExistsError("AU-NCA public output already exists; preserve it and inspect before replacing")
    PUBLIC.mkdir(parents=True)
    for name in ("target_rgba.npy", "block00_inputs.npz", "block01_inputs.npz", "block02_inputs.npz", "qualification.json", "aggregate.json"):
        copy_file(RUN / name, PUBLIC / name)
    public_manifest = {
        "schema": "au-nca-public-evidence-v1",
        "run_id": "au_nca_20261009_01",
        "status": "COMPLETE",
        "completed_units": 9,
        "planned_units": 9,
        "verdict": aggregate["decision"]["verdict"],
        "config": run_manifest["config"],
        "source_hashes": run_manifest["source_hashes"],
        "data_hashes": run_manifest["data_hashes"],
        "qualification_sha256": sha256_file(RUN / "qualification.json"),
        "private_artifacts_excluded": [
            "checkpoint tensor contents", "optimizer and pool state", "local dispatch and launch receipts",
            "status and lock files", "worker logs and machine identifiers",
        ],
        "checkpoint_bindings_note": "Local checkpoint payloads were checked on CPU during export; bindings retain hashes, sizes, update numbers, and canonical parameter hashes, while weights are excluded.",
    }
    write_json(PUBLIC / "manifest.json", public_manifest)

    results_by_block_arm = {(int(result["block"]), result["arm"]): result for result in aggregate["arms"]}
    for block in range(3):
        for arm in ARMS:
            private_arm = RUN / f"block{block:02d}" / arm
            public_arm = PUBLIC / f"block{block:02d}" / arm
            public_arm.mkdir(parents=True, exist_ok=True)
            for name in ("result.json", "training.jsonl", "predictions.npz", "continuation_curves.json", "gradient_diagnostics.json"):
                copy_file(private_arm / name, public_arm / name)
            for update in (1000, 2000):
                for suffix in ("json", "npz"):
                    name = f"diagnostic_u{update:04d}.{suffix}"
                    copy_file(private_arm / name, public_arm / name)
            partials = sorted(private_arm.glob("training_before_resume_*.jsonl"))
            if partials:
                if (block, arm) != (0, "au_k8") or len(partials) != 1:
                    raise AssertionError(f"unexpected historical partial training logs for block{block:02d}/{arm}")
                copy_file(partials[0], public_arm / "training_before_resume_historical_partial.jsonl")
            if (block, arm) not in results_by_block_arm:
                raise AssertionError(f"missing aggregate result for block{block:02d}/{arm}")
    write_checkpoint_bindings(PUBLIC / "checkpoint_bindings.csv", checkpoint_rows)

    # Include only a normalized, low-detail execution note. No private receipt is copied.
    resume_receipt = read_json(RUN.parent / "au_nca_20261009_resume01_guard_aa8bd81c0e6c" / "launch_receipt.json")
    recovery_events = read_jsonl(RUN / "recovery_events.jsonl")
    if resume_receipt.get("status") != "COMPLETE" or resume_receipt.get("worker_exit_code") != 0 or not resume_receipt.get("idle_sleep_request_cleared"):
        raise AssertionError("resumed worker receipt does not confirm clean completion")
    if not any(event.get("event") == "explicit_resume" for event in recovery_events):
        raise AssertionError("explicit resume event is missing")
    historical_path = PUBLIC / "block00/au_k8/training_before_resume_historical_partial.jsonl"
    historical_rows = read_jsonl(historical_path)
    if [row.get("update") for row in historical_rows] != list(range(1, 1201)):
        raise AssertionError("historical pre-resume log should preserve updates 1..1200")
    provenance = {
        "execution": "COMPLETE",
        "planned_arms": 9,
        "completed_arms": 9,
        "completed_arms_skipped_on_resume": 2,
        "resumed_arm": "block00/au_k8",
        "committed_checkpoint_update_before_resume": 1175,
        "resume_started_at_update": 1176,
        "pre_resume_log_records": len(historical_rows),
        "pre_resume_log_recorded_through_update": 1200,
        "historical_tail_after_committed_checkpoint": "Updates 1176..1200 are retained from the interrupted attempt as historical partial evidence; they are excluded from the canonical 1..3000 training log.",
        "initial_interruption_cause": "Unknown; no stderr evidence was retained in the public package.",
        "resumed_worker_exit_code": 0,
        "idle_sleep_request_cleared": True,
        "aggregate_elapsed_seconds_resumed_process_only": float(aggregate["elapsed_seconds"]),
        "initial_interrupted_attempt_runtime_included": False,
        "inter_attempt_gap_included": False,
        "aggregate_is_total_wall_clock_across_attempts": False,
        "private_launch_receipts_published": False,
    }
    write_json(PUBLIC / "execution_provenance.json", provenance)

    (PUBLIC / "REPRODUCTION.md").write_text(
        "# AU-NCA v0 evidence reproduction\n\n"
        "Start with RESULTS.md, final_metrics.csv, per_episode_metrics.csv, aggregate.json, and validation.json. "
        "The public package includes the fixed target, three packed input banks, all nine arm results and training logs, "
        "18 intermediate T64 diagnostics, all final saved RGBA predictions, continuation curves, gradient diagnostics, "
        "and 36 metadata-only checkpoint bindings. Checkpoint weights, optimizer/pool state, local receipts, and worker logs are omitted.\n\n"
        "From the project repository root, run the independent public-package check:\n\n"
        "```powershell\npython -X utf8 -B tools/export_au_nca.py --verify-only\n```\n\n"
        "Verification uses public saved arrays and JSON, hashes, and current source files. It needs NumPy and does not read "
        "the private run or checkpoints, initialize CUDA, train, or run inference. The RGBA metrics are recomputed in float64 "
        "from saved predictions and target and checked against the frozen result JSON at absolute and relative tolerance 1e-12.\n\n"
        "Training logs contain exactly updates 1..3000 for each arm (27,000 canonical update records). The separate "
        "training_before_resume_historical_partial.jsonl file preserves an interrupted prefix through update 1200; the "
        "committed checkpoint was at update 1175, so its 1176..1200 tail is historical partial evidence and is excluded "
        "from canonical counts. The aggregate 1,254.7 seconds covers the resumed execution only; it excludes the earlier "
        "interrupted attempt and the gap, so it is not total wall-clock time across attempts.\n\n"
        "No prediction or training operation is needed to reproduce this verification. The three independent units are "
        "paired initialization/schedule blocks; the 32 firing-mask episodes are repeated measurements within each block.\n\n"
        "## Fresh run commands\n\n"
        "For a new experiment, use Windows PowerShell in the registered CUDA environment and choose fresh output names. "
        "The qualification command includes a small CUDA smoke; the training command launches a new run and can take substantially longer.\n\n"
        "```powershell\npython -X utf8 -B new/au_nca/run.py --check --out analyses/au_nca_qualification_20261010_01\n"
        "python -X utf8 -B new/au_nca/run.py --out runs/au_nca_20261010_01 --qualification analyses/au_nca_qualification_20261010_01/qualification.json\n```\n",
        encoding="utf-8",
    )

    (PUBLIC / "RESULTS.md").write_text(build_results_markdown(aggregate), encoding="utf-8")


def build_results_markdown(aggregate: dict[str, Any]) -> str:
    arms = {(int(result["block"]), result["arm"]): result for result in aggregate["arms"]}
    rows = [
        "# AU-NCA v0: accumulated-update learned feedback",
        "",
        "Execution **COMPLETE** (9/9 arms); verdict **AU_BENEFIT_DEVELOPMENTAL**.",
        "",
        "| Block | Arm | T64 NMSE | T64 alpha IoU | T128 NMSE | T256 NMSE | Damaged T256 NMSE |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for block in range(3):
        for arm in ARMS:
            result = arms[(block, arm)]
            evaluation = result["evaluation"]
            values = [
                evaluation["intact_t64"]["nmse"]["mean"],
                evaluation["intact_t64"]["alpha_iou"]["mean"],
                evaluation["intact_t128"]["nmse"]["mean"],
                evaluation["intact_t256"]["nmse"]["mean"],
                evaluation["damaged_t256"]["nmse"]["mean"],
            ]
            rows.append(f"| {block} | {arm} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    primary = aggregate["decision"]["paired_primary"]
    gains = [float(pair["original_short_minus_au"]) for pair in primary]
    mean_gain = float(np.mean(gains))
    strong_count = sum(
        pair["nmse"]["original_k8"] >= 0.20 and pair["nmse"]["au_k8"] <= 0.10
        for pair in primary
    )
    controls = [arms[(block, "original_k64")]["evaluation"]["intact_t64"] for block in range(3)]
    rows.extend([
        "",
        "All three original K64 controls meet the frozen T64 criterion (NMSE <= 0.10 and alpha IoU >= 0.80). AU-K8 beats original K8 by at least 0.05 NMSE in two of three blocks; mean paired gain is "
        f"{mean_gain:.6f} (positive favors AU-K8). The third block favors original K8. The stronger per-block recovery criterion is met in {strong_count}/3 blocks, so this result does not establish strong recovery.",
        "",
        "The independent unit is a paired initialization/schedule block (n=3). The 32 mask episodes per arm are repeated measurements, not independent blocks. T128/T256 reuse the T64 state with no parameter updates; damage recovery is untrained and secondary. Long-rollout and damage outcomes vary by block.",
        "",
        "This is a fixed-step, fixed-target developmental screen of an accumulated-update NCA adaptation, not an exact reproduction of a published Growing/Persistent NCA training recipe. It does not establish general temporal-credit recovery or reliability.",
        "",
        "The AU state uses 145 scalars per cell versus 16 for the original cell. All arms share the same 8,336 trainable feature/projection parameters. See aggregate.json and the per-episode table for complete saved metrics.",
        "",
        "Execution resumed at update 1176 from a committed u1175 checkpoint. The historical partial log records through u1200; its 1176..1200 tail is excluded from the canonical 27,000 update records. The reported aggregate runtime belongs to the resumed process only and excludes the earlier interrupted attempt and the gap.",
        "",
    ])
    if len(controls) != 3:
        raise AssertionError("expected one control per block")
    return "\n".join(rows)


def compute_decision(results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    controls = [result for result in results if result["arm"] == "original_k64"]
    qualified = [
        result["evaluation"]["intact_t64"]["nmse"]["mean"] <= config["control_nmse_max"]
        and result["evaluation"]["intact_t64"]["alpha_iou"]["mean"] >= config["control_alpha_iou_min"]
        for result in controls
    ]
    controls_ok = len(controls) == config["blocks"] and all(qualified)
    pairs: list[dict[str, Any]] = []
    for block in range(config["blocks"]):
        by_arm = {result["arm"]: result for result in results if int(result["block"]) == block}
        if len(by_arm) != 3:
            continue
        values = {arm: float(by_arm[arm]["evaluation"]["intact_t64"]["nmse"]["mean"]) for arm in ARMS}
        pairs.append({
            "block": block,
            "nmse": values,
            "original_short_minus_au": values["original_k8"] - values["au_k8"],
            "full_minus_au": values["original_k64"] - values["au_k8"],
        })
    verdict = "POSITIVE_CONTROL_UNQUALIFIED"
    if controls_ok:
        if len(pairs) < config["blocks"]:
            verdict = "NUMERICAL_FAILURE_DEVELOPMENTAL"
        else:
            gains = [pair["original_short_minus_au"] for pair in pairs]
            strong = sum(pair["nmse"]["original_k8"] >= config["strong_gap_original_nmse_min"] and pair["nmse"]["au_k8"] <= config["strong_recovery_au_nmse_max"] for pair in pairs) >= 2
            benefit = sum(gain >= config["paired_nmse_gain_min"] for gain in gains) >= 2 and float(np.mean(gains)) >= -config["mean_worsening_tolerance"]
            verdict = "SHORT_CREDIT_RECOVERY_DEVELOPMENTAL" if strong and benefit else (
                "AU_BENEFIT_DEVELOPMENTAL" if benefit else "NO_QUALIFIED_AU_BENEFIT_DEVELOPMENTAL"
            )
    return {"verdict": verdict, "all_controls_qualified": controls_ok, "control_passes": sum(qualified), "paired_primary": pairs}


def check_input_banks(folder: Path, manifest: dict[str, Any]) -> int:
    expected_intact_shape = (256, 32, 1, 32, 32)
    expected_damage_shape = (192, 32, 1, 32, 32)
    for block in range(3):
        filename = f"block{block:02d}_inputs.npz"
        if sha256_file(folder / filename) != manifest["data_hashes"][filename]:
            raise AssertionError(f"public bank does not match its frozen data hash: {filename}")
        with np.load(folder / filename, allow_pickle=False) as packed:
            if set(packed.files) != {"pool_indices", "intact_bits", "intact_shape", "damage_bits", "damage_shape"}:
                raise AssertionError(f"unexpected packed bank members in {filename}")
            indices = packed["pool_indices"]
            intact_shape = tuple(int(x) for x in packed["intact_shape"])
            damage_shape = tuple(int(x) for x in packed["damage_shape"])
            if indices.shape != (3000, 8) or indices.dtype != np.int32 or indices.min() < 0 or indices.max() >= 512:
                raise AssertionError(f"invalid pool-index bank: {filename}")
            if intact_shape != expected_intact_shape or damage_shape != expected_damage_shape:
                raise AssertionError(f"invalid packed firing-mask shapes: {filename}")
            intact = np.unpackbits(packed["intact_bits"], count=int(np.prod(intact_shape))).reshape(intact_shape)
            damaged = np.unpackbits(packed["damage_bits"], count=int(np.prod(damage_shape))).reshape(damage_shape)
            if not np.array_equal(damaged, intact[64:]):
                raise AssertionError(f"damaged masks do not match the post-64-step intact suffix: {filename}")
    return 3


def verify_csv_metrics(folder: Path, final_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]]) -> None:
    with (folder / "final_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 45:
        raise AssertionError(f"expected 45 final metric rows, got {len(rows)}")
    expected = {(row["block"], row["arm"], row["branch"]): row for row in final_rows}
    for row in rows:
        key = (int(row["block"]), row["arm"], row["branch"])
        saved = expected.pop(key)
        for metric in ("mse_mean", "mse_std", "nmse_mean", "nmse_std", "alpha_iou_mean", "alpha_iou_std"):
            assert_close(float(row[metric]), saved[metric], f"final_metrics.csv {key} {metric}")
    if expected:
        raise AssertionError(f"missing final metric CSV rows: {list(expected)[:3]}")
    with (folder / "per_episode_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1440:
        raise AssertionError(f"expected 1440 per-episode metric rows, got {len(rows)}")
    expected_eps = {(row["block"], row["arm"], row["branch"], row["episode"]): row for row in episode_rows}
    for row in rows:
        key = (int(row["block"]), row["arm"], row["branch"], int(row["episode"]))
        saved = expected_eps.pop(key)
        for metric in ("mse", "nmse", "alpha_iou"):
            assert_close(float(row[metric]), saved[metric], f"per_episode_metrics.csv {key} {metric}")
    if expected_eps:
        raise AssertionError(f"missing per-episode metric rows: {list(expected_eps)[:3]}")


def validate_public_payload(folder: Path, *, check_current_sources: bool) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_json(folder / "manifest.json")
    aggregate = read_json(folder / "aggregate.json")
    qualification = read_json(folder / "qualification.json")
    if manifest.get("schema") != "au-nca-public-evidence-v1" or manifest.get("run_id") != "au_nca_20261009_01":
        raise AssertionError("unexpected public manifest schema or run id")
    if manifest.get("status") != "COMPLETE" or manifest.get("completed_units") != 9 or manifest.get("planned_units") != 9:
        raise AssertionError("public manifest does not record 9/9 completion")
    if aggregate.get("status") != "COMPLETE" or len(aggregate.get("arms", [])) != 9:
        raise AssertionError("public aggregate is incomplete")
    if aggregate.get("config") != manifest.get("config") or qualification.get("config") != manifest.get("config"):
        raise AssertionError("public configuration bindings disagree")
    if aggregate["decision"] != compute_decision(aggregate["arms"], manifest["config"]):
        raise AssertionError("public decision does not recompute from the nine saved results")
    if manifest.get("verdict") != aggregate["decision"]["verdict"] or manifest["verdict"] != "AU_BENEFIT_DEVELOPMENTAL":
        raise AssertionError("public verdict mismatch")
    if qualification.get("status") != "PASS" or qualification.get("source_hashes") != manifest["source_hashes"]:
        raise AssertionError("public qualification/source binding mismatch")
    if sha256_file(folder / "qualification.json") != manifest.get("qualification_sha256"):
        raise AssertionError("public qualification hash mismatch")
    if check_current_sources:
        verify_source_hashes(manifest["source_hashes"])
    for rel, digest in manifest["data_hashes"].items():
        if not (folder / rel).is_file() or sha256_file(folder / rel) != digest:
            raise AssertionError(f"public data hash mismatch: {rel}")
    bank_count = check_input_banks(folder, manifest)
    target = np.load(folder / "target_rgba.npy", allow_pickle=False)
    if target.shape != (1, 4, 32, 32) or not np.isfinite(target).all():
        raise AssertionError("invalid public target RGBA array")
    target_for_metrics = target[0]

    result_map = {(int(row["block"]), row["arm"]): row for row in aggregate["arms"]}
    if set(result_map) != {(block, arm) for block in range(3) for arm in ARMS}:
        raise AssertionError("aggregate arm identities are incomplete or duplicated")
    final_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    max_error = 0.0
    for block in range(3):
        for arm in ARMS:
            base = folder / f"block{block:02d}" / arm
            result = read_json(base / "result.json")
            aggregate_result = result_map[(block, arm)]
            if result != aggregate_result or result.get("status") != "COMPLETE":
                raise AssertionError(f"per-arm result does not match aggregate: block{block:02d}/{arm}")
            if result["evaluation"].get("episodes") != 32:
                raise AssertionError(f"wrong evaluation episode count: block{block:02d}/{arm}")
            with np.load(base / "predictions.npz", allow_pickle=False) as archive:
                if set(archive.files) != set(BRANCHES):
                    raise AssertionError(f"final prediction bank names differ: block{block:02d}/{arm}")
                predictions = {key: archive[key] for key in BRANCHES}
            for branch in BRANCHES:
                array = predictions[branch]
                if array.shape != (32, 4, 32, 32) or not np.isfinite(array).all():
                    raise AssertionError(f"invalid final prediction bank: block{block:02d}/{arm}/{branch}")
                values = calculate_metrics(array, target_for_metrics)
                max_error = max(max_error, compare_saved_metrics(values, result["evaluation"][branch], f"block{block:02d}/{arm}/{branch}"))
                row = {"block": block, "arm": arm, "branch": branch, "episodes": 32}
                for metric in ("mse", "nmse", "alpha_iou"):
                    row[f"{metric}_mean"] = float(values[metric].mean())
                    row[f"{metric}_std"] = float(values[metric].std(ddof=1))
                final_rows.append(row)
                for episode in range(32):
                    episode_rows.append({
                        "block": block,
                        "arm": arm,
                        "branch": branch,
                        "episode": episode,
                        "mse": float(values["mse"][episode]),
                        "nmse": float(values["nmse"][episode]),
                        "alpha_iou": float(values["alpha_iou"][episode]),
                    })

            # The saved continuation curve must agree with the same-state final arrays at T64/T128/T256.
            curve = read_json(base / "continuation_curves.json")
            if len(curve) != 32 * 32:
                raise AssertionError(f"expected 1024 continuation records: block{block:02d}/{arm}")
            curve_map: dict[tuple[int, int], dict[str, Any]] = {}
            for entry in curve:
                if not all(math.isfinite(float(entry[key])) for key in ("episode", "step", "mse", "nmse", "alpha_iou", "state_rms", "alive_fraction")):
                    raise AssertionError(f"nonfinite continuation record: block{block:02d}/{arm}")
                key = (int(entry["episode"]), int(entry["step"]))
                if key in curve_map:
                    raise AssertionError(f"duplicate continuation record {key}: block{block:02d}/{arm}")
                curve_map[key] = entry
            expected_curve_keys = {(episode, step) for episode in range(32) for step in range(8, 257, 8)}
            if set(curve_map) != expected_curve_keys:
                raise AssertionError(f"continuation curve episode/step grid mismatch: block{block:02d}/{arm}")
            for step, branch in ((64, "intact_t64"), (128, "intact_t128"), (256, "intact_t256")):
                curve_values = [curve_map[(episode, step)] for episode in range(32)]
                for metric in ("mse", "nmse", "alpha_iou"):
                    max_error = max(max_error, assert_close(
                        [entry[metric] for entry in curve_values],
                        [float(x) for x in result["evaluation"][branch][metric]["per_episode"]],
                        f"continuation curve {branch}.{metric} block{block:02d}/{arm}",
                    ))

            training = read_jsonl(base / "training.jsonl")
            if [row.get("update") for row in training] != list(range(1, 3001)):
                raise AssertionError(f"canonical training log is not updates 1..3000: block{block:02d}/{arm}")
            for index, row in enumerate(training, 1):
                if set(row) != set(TRAINING_FIELDS) or not all(math.isfinite(float(row[key])) for key in TRAINING_FIELDS[1:]):
                    raise AssertionError(f"invalid canonical training row at update {index}: block{block:02d}/{arm}")

            for update in (1000, 2000):
                diagnostic = read_json(base / f"diagnostic_u{update:04d}.json")
                if diagnostic.get("episodes") != 8 or diagnostic.get("same_state_continuation") is not False:
                    raise AssertionError(f"invalid intermediate diagnostic metadata: block{block:02d}/{arm}/u{update}")
                with np.load(base / f"diagnostic_u{update:04d}.npz", allow_pickle=False) as diagnostic_npz:
                    if set(diagnostic_npz.files) != {"intact_t64"}:
                        raise AssertionError(f"invalid intermediate prediction bank: block{block:02d}/{arm}/u{update}")
                    diagnostic_prediction = diagnostic_npz["intact_t64"]
                if diagnostic_prediction.shape != (8, 4, 32, 32):
                    raise AssertionError(f"invalid intermediate T64 shape: block{block:02d}/{arm}/u{update}")
                values = calculate_metrics(diagnostic_prediction, target_for_metrics)
                max_error = max(max_error, compare_saved_metrics(values, diagnostic["intact_t64"], f"diagnostic block{block:02d}/{arm}/u{update}"))

            gradient_diagnostics = read_json(base / "gradient_diagnostics.json")
            if not isinstance(gradient_diagnostics, dict) or not gradient_diagnostics.get("same_parameters"):
                raise AssertionError(f"invalid gradient diagnostic summary: block{block:02d}/{arm}")

    if (folder / "final_metrics.csv").exists() and (folder / "per_episode_metrics.csv").exists():
        verify_csv_metrics(folder, final_rows, episode_rows)
    with (folder / "checkpoint_bindings.csv").open(encoding="utf-8", newline="") as handle:
        bindings = list(csv.DictReader(handle))
    if len(bindings) != 36:
        raise AssertionError(f"expected 36 checkpoint bindings, got {len(bindings)}")
    expected_binding_keys = {(block, arm, label, update) for block in range(3) for arm in ARMS for label, update in (("u1000.pt", 1000), ("u2000.pt", 2000), ("u3000.pt", 3000), ("latest.pt", 3000))}
    actual_binding_keys = set()
    latest_hashes: dict[tuple[int, str], str] = {}
    named_hashes: dict[tuple[int, str], str] = {}
    for row in bindings:
        key = (int(row["block"]), row["arm"], row["checkpoint"], int(row["update"]))
        actual_binding_keys.add(key)
        if not re.fullmatch(r"[0-9a-f]{64}", row["checkpoint_sha256"]) or not re.fullmatch(r"[0-9a-f]{64}", row["parameter_sha256"]):
            raise AssertionError("malformed checkpoint binding hash")
        if int(row["size_bytes"]) <= 0 or int(row["parameter_count"]) != 8336 or row["weights_published"] != "False":
            raise AssertionError("invalid or unsafe checkpoint binding metadata")
        arm_key = (int(row["block"]), row["arm"])
        if row["checkpoint"] == "latest.pt":
            latest_hashes[arm_key] = row["parameter_sha256"]
        if row["checkpoint"] == "u3000.pt":
            named_hashes[arm_key] = row["parameter_sha256"]
    if actual_binding_keys != expected_binding_keys or latest_hashes != named_hashes:
        raise AssertionError("checkpoint binding rows or latest/u3000 parameter hashes are inconsistent")

    historical = folder / "block00/au_k8/training_before_resume_historical_partial.jsonl"
    historical_rows = read_jsonl(historical)
    if [row.get("update") for row in historical_rows] != list(range(1, 1201)):
        raise AssertionError("historical partial log must preserve updates 1..1200")
    provenance = read_json(folder / "execution_provenance.json")
    if provenance.get("committed_checkpoint_update_before_resume") != 1175 or provenance.get("resume_started_at_update") != 1176:
        raise AssertionError("recovery provenance has an unexpected checkpoint/resume boundary")
    if provenance.get("pre_resume_log_recorded_through_update") != 1200 or provenance.get("pre_resume_log_records") != 1200:
        raise AssertionError("recovery provenance does not preserve the historical partial tail")
    if provenance.get("aggregate_elapsed_seconds_resumed_process_only") != float(aggregate["elapsed_seconds"]):
        raise AssertionError("runtime scope/value differs from aggregate")
    if provenance.get("aggregate_is_total_wall_clock_across_attempts") is not False:
        raise AssertionError("runtime summary must keep resumed-process and total wall-clock scopes separate")

    return {
        "status": "PASS",
        "scope": "Offline public evidence integrity; no training, model inference, CUDA, private run, or checkpoint payload is used.",
        "completed_arms": 9,
        "input_banks": bank_count,
        "canonical_training_updates": 27000,
        "historical_partial_records_excluded": 1200,
        "final_metric_rows": len(final_rows),
        "per_episode_metric_rows": len(episode_rows),
        "intermediate_diagnostics": 18,
        "intermediate_prediction_episodes": 144,
        "continuation_curve_records": 9216,
        "checkpoint_bindings": len(bindings),
        "checkpoint_weights_published": False,
        "max_metric_recomputation_abs_error": max_error,
        "decision_recomputed": aggregate["decision"],
        "recovery_boundary": {"committed_update": 1175, "resume_update": 1176, "historical_log_through": 1200},
    }, final_rows, episode_rows


def write_metric_tables(folder: Path, final_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]]) -> None:
    final_fields = ("block", "arm", "branch", "episodes", "mse_mean", "mse_std", "nmse_mean", "nmse_std", "alpha_iou_mean", "alpha_iou_std")
    with (folder / "final_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        for row in final_rows:
            writer.writerow({key: format(row[key], ".17g") if isinstance(row[key], float) else row[key] for key in final_fields})
    episode_fields = ("block", "arm", "branch", "episode", "mse", "nmse", "alpha_iou")
    with (folder / "per_episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=episode_fields)
        writer.writeheader()
        for row in episode_rows:
            writer.writerow({key: format(row[key], ".17g") if isinstance(row[key], float) else row[key] for key in episode_fields})


def package_file_map(folder: Path) -> tuple[dict[str, dict[str, Any]], int]:
    artifacts: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for path in sorted(item for item in folder.rglob("*") if item.is_file()):
        if path.suffix.lower() in (".pt", ".pth", ".ckpt"):
            raise AssertionError(f"checkpoint weight file entered public package: {path.name}")
        rel = path.relative_to(ROOT).as_posix()
        size = path.stat().st_size
        artifacts[rel] = {"sha256": sha256_file(path), "size_bytes": size}
        total_bytes += size
        privacy_check(path)
    return artifacts, total_bytes


def export() -> dict[str, Any]:
    run_manifest, qualification, aggregate, _status = validate_private_inputs()
    checkpoint_rows = inspect_checkpoints(run_manifest)
    export_allowlist(run_manifest, qualification, aggregate, checkpoint_rows)
    validation, final_rows, episode_rows = validate_public_payload(PUBLIC, check_current_sources=True)
    write_metric_tables(PUBLIC, final_rows, episode_rows)
    validation, final_rows, episode_rows = validate_public_payload(PUBLIC, check_current_sources=True)
    write_json(PUBLIC / "validation.json", validation)
    artifacts, package_bytes = package_file_map(PUBLIC)
    publication = {
        "schema": "au-nca-publication-v1",
        "run_id": "au_nca_20261009_01",
        "published_artifacts": artifacts,
        "evidence_package_bytes": package_bytes,
        "validation": validation,
        "checkpoint_contents_published": False,
        "private_run_preserved": True,
    }
    write_json(PUBLICATION, publication)
    return verify_public()


def verify_public() -> dict[str, Any]:
    publication = read_json(PUBLICATION)
    if publication.get("schema") != "au-nca-publication-v1" or publication.get("run_id") != "au_nca_20261009_01":
        raise AssertionError("unexpected publication manifest")
    artifacts, package_bytes = package_file_map(PUBLIC)
    if artifacts != publication.get("published_artifacts"):
        raise AssertionError("public package file hashes/sizes differ from publication manifest")
    if package_bytes != publication.get("evidence_package_bytes"):
        raise AssertionError("public package byte count differs from publication manifest")
    validation, _final_rows, _episode_rows = validate_public_payload(PUBLIC, check_current_sources=True)
    if validation != read_json(PUBLIC / "validation.json") or validation != publication.get("validation"):
        raise AssertionError("independent public verification differs from stored validation")
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
        "verdict": validation["decision_recomputed"]["verdict"],
        "method": "public saved arrays, JSON, CSV, and hashes only; no private run/checkpoint files, training, inference, or CUDA",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="verify the public evidence package without opening the private run or checkpoints")
    args = parser.parse_args(argv)
    try:
        summary = verify_public() if args.verify_only else export()
    except Exception as exc:
        print(f"AU-NCA evidence export/verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
