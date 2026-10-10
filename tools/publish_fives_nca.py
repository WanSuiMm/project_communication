"""Export and offline-verify the compact public FIVES NCA evidence package.

The export reads saved JSON/JSONL evidence only. ``--verify`` reads the public
package and the bound source files; neither mode runs training or inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INSPECTION = ROOT / "runs/fives_nca_20261010_01/inspections/20261010_1538"
PUBLIC = ROOT / "evidence/fives_nca_20261010_01"
SOURCE_ROOT = ROOT / "new/real_task_fives"
HORIZONS = (8, 16, 32, 64, 128, 256)
DIAGNOSTIC_UPDATES = (250, 500, 1000)
METRICS = ("dice", "cldice", "precision", "recall", "betti0_error", "betti1_error")
GATE_THRESHOLDS = {
    "coarse_dice_floor": ("coarse", "dice", 0.60, "ge"),
    "coarse_cldice_floor": ("coarse", "cldice", 0.50, "ge"),
    "k64_cldice_gain": ("k64_vs_coarse", "cldice", 0.01, "ge"),
    "k64_dice_guard": ("k64_vs_coarse", "dice", -0.005, "ge"),
    "within_model_depth": ("t64_vs_t8", "cldice", 0.01, "ge"),
}
SENSITIVE_KEYS = {
    "argv", "child_pid", "command", "command_line", "cwd", "dispatch",
    "dispatch_id", "gpu_uuid", "host", "hostname", "ip", "launch_receipt",
    "machine", "parent_pid", "pid", "ppid", "receipt", "run_dir", "stderr",
    "stdout", "user", "username", "worker_pid",
}
PRIVATE_PATH = re.compile(
    r"(?i)(?:\b[A-Z]:[\\/][^\r\n\"']+|"
    r"/(?:home|data/users)/[^\s\"']+|/mnt/[a-z]/[^\s\"']+)"
)
PRIVATE_IP = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
)
REDACTIONS: dict[str, int] = defaultdict(int)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(value: Any) -> Any:
    """Remove private receipt fields and redact machine paths/IPs in-place-safe form."""
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEYS:
                REDACTIONS[f"removed_field:{str(key).lower()}"] += 1
                continue
            clean[key] = sanitize(item)
        return clean
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        if PRIVATE_PATH.search(value):
            REDACTIONS["redacted_machine_path"] += 1
            return "[redacted machine-specific path]"
        if PRIVATE_IP.search(value):
            REDACTIONS["redacted_private_ip"] += 1
            return "[redacted private IP]"
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    # Keep large evidence files below the repository's 1 MB publication limit.
    if len(body.encode("utf-8")) >= 950_000:
        body = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    path.write_text(body, encoding="utf-8", newline="\n")


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise AssertionError("cannot compute a metric mean from zero rows")
    result: dict[str, float] = {}
    for metric in METRICS:
        vals = [float(row["metrics"][metric]) for row in rows]
        if not all(math.isfinite(value) for value in vals):
            raise AssertionError(f"nonfinite metric value: {metric}")
        result[metric] = sum(vals) / len(vals)
    return result


def summarize(aggregate: dict[str, Any], manifest: dict[str, Any], pipeline: dict[str, Any], diagnostics_root: Path) -> dict[str, Any]:
    baseline_rows = aggregate["baseline"]["rows"]
    final_rows = aggregate["arms"]["standard_k64"]["final"]["rows"]
    coarse = mean_metrics(baseline_rows)
    by_horizon: dict[str, list[dict[str, Any]]] = {
        str(t): [row for row in final_rows if int(row["t"]) == t] for t in HORIZONS
    }
    means = {key: mean_metrics(rows) for key, rows in by_horizon.items()}
    t64 = means["64"]
    t8 = means["8"]
    differences = {
        "k64_cldice_gain": t64["cldice"] - coarse["cldice"],
        "k64_dice_guard": t64["dice"] - coarse["dice"],
        "within_model_depth": t64["cldice"] - t8["cldice"],
    }
    gates = {
        "coarse_dice_floor": coarse["dice"] >= 0.60,
        "coarse_cldice_floor": coarse["cldice"] >= 0.50,
        "k64_cldice_gain": differences["k64_cldice_gain"] >= 0.01,
        "k64_dice_guard": differences["k64_dice_guard"] >= -0.005,
        "within_model_depth": differences["within_model_depth"] >= 0.01,
    }
    if gates != aggregate["qualification"]["gates"]:
        raise AssertionError(f"recomputed qualification gates differ: {gates}")
    if sum(not passed for passed in gates.values()) != 2:
        raise AssertionError(f"expected exactly two failed gates, got {gates}")

    diagnostic: dict[str, Any] = {}
    for update in DIAGNOSTIC_UPDATES:
        data = read_json(diagnostics_root / f"eval_u{update}.json")
        rows = data["rows"]
        if not rows or any(int(row["t"]) != 64 for row in rows):
            raise AssertionError(f"u{update} diagnostic is not a T64-only evaluation")
        diagnostic[str(update)] = {
            "n_images": len({row["id"] for row in rows}),
            "n_plans": len(rows),
            "mean": mean_metrics(rows),
        }

    return {
        "schema": "fives_real_task_nca_public_summary_v1",
        "status": aggregate["status"],
        "verdict": aggregate["verdict"],
        "dataset": {
            "name": manifest["dataset"],
            "training_images": 480,
            "validation_images": 120,
            "resolution": "512x512 development resolution",
            "split_unit": manifest["split_unit"]["unit"],
            "patient_disjoint": manifest["split_unit"]["patient_disjoint"],
            "test_labels_loaded": False,
        },
        "training": {
            "trained_arm": "standard_k64",
            "updates": int(aggregate["arms"]["standard_k64"]["update"]),
            "training_rollout_steps": 64,
            "evaluation_repeats_per_image": 2,
            "unrun_arms": ["standard_t8", "standard_k8", "au_k8"],
            "elapsed_seconds": float(pipeline["elapsed_seconds"]),
            "elapsed_scope": "recovered pipeline process only; excludes earlier acquisition/download/transfer work; not an end-to-end speed measurement",
        },
        "coarse_baseline_mean": coarse,
        "final_horizon_means": {
            key: {
                "n_images": len({row["id"] for row in by_horizon[key]}),
                "n_plans": len(by_horizon[key]),
                **means[key],
            }
            for key in means
        },
        "intermediate_t64_diagnostics": diagnostic,
        "gate_differences": differences,
        "gates": gates,
    }


def write_per_image_csv(path: Path, aggregate: dict[str, Any], image_order: list[str]) -> int:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate["arms"]["standard_k64"]["final"]["rows"]:
        grouped[(str(row["id"]), int(row["t"]))].append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "t", "plans", *METRICS]
    row_count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for t in HORIZONS:
            for image_id in image_order:
                rows = grouped[(image_id, t)]
                if len(rows) != 2:
                    raise AssertionError(f"expected two evaluation plans for {image_id} at T{t}")
                means = mean_metrics(rows)
                writer.writerow({
                    "id": image_id,
                    "t": t,
                    "plans": len(rows),
                    **{name: format(means[name], ".17g") for name in METRICS},
                })
                row_count += 1
    if row_count != len(image_order) * len(HORIZONS):
        raise AssertionError(f"unexpected per-image CSV row count: {row_count}")
    return row_count


def load_training_samples() -> list[dict[str, Any]]:
    source = INSPECTION / "training_attempt_fd5f8b054ff5.jsonl"
    samples: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row_type") == "attempt_start":
                REDACTIONS["omitted_dispatch_attempt_marker"] += 1
                continue
            row.pop("attempt", None)
            clean = sanitize(row)
            if "update" not in clean:
                raise AssertionError(f"training sample line {line_number} has no update")
            samples.append(clean)
    if not samples or int(samples[0]["update"]) != 1 or int(samples[-1]["update"]) != 1500:
        raise AssertionError("sampled training records do not span updates 1 through 1500")
    return samples


def privacy_check(folder: Path) -> None:
    for path in folder.iterdir():
        if path.suffix.lower() not in {".json", ".jsonl", ".md", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8")
        if PRIVATE_PATH.search(text) or PRIVATE_IP.search(text):
            raise AssertionError(f"private path or IP remains in {path.name}")
        data: Any = None
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        elif path.suffix.lower() == ".jsonl":
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        if data is not None:
            inspect_private_keys(data, path.name)


def inspect_private_keys(value: Any, filename: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEYS:
                raise AssertionError(f"private receipt key remains in {filename}: {key}")
            inspect_private_keys(item, filename)
    elif isinstance(value, list):
        for item in value:
            inspect_private_keys(item, filename)


def export() -> None:
    aggregate_raw = read_json(INSPECTION / "aggregate.json")
    manifest_raw = read_json(INSPECTION / "manifest.json")
    pipeline_raw = read_json(INSPECTION / "pipeline_status.json")
    qualification = read_json(INSPECTION / "data_qualification.json")
    software = read_json(INSPECTION / "software_check.json")
    if aggregate_raw.get("status") != "COMPLETE" or aggregate_raw.get("verdict") != "TASK_UNQUALIFIED":
        raise AssertionError("unexpected frozen endpoint status/verdict")
    sources = aggregate_raw["arms"]["standard_k64"]["binding"]["sources"]
    if sources != manifest_raw["provenance"]["code"] or sources != qualification["source_hashes"]:
        raise AssertionError("source hash dictionaries disagree across saved evidence")
    if sources != software["sources"]:
        raise AssertionError("saved software-check source binding differs from execution source hashes")
    for name, expected in sources.items():
        source_path = SOURCE_ROOT / name
        if not source_path.is_file() or sha256_file(source_path) != expected:
            raise AssertionError(f"executed source hash does not match current source: {name}")
    if len(aggregate_raw["baseline"]["rows"]) != 120:
        raise AssertionError("coarse baseline does not contain 120 validation image rows")
    if len(aggregate_raw["arms"]["standard_k64"]["final"]["rows"]) != 120 * 2 * len(HORIZONS):
        raise AssertionError("final metrics do not contain 120 images x 2 plans x 6 horizons")
    if pipeline_raw.get("status") != "COMPLETE" or pipeline_raw.get("verdict") != "TASK_UNQUALIFIED":
        raise AssertionError("pipeline receipt does not agree with aggregate status")

    PUBLIC.mkdir(parents=True, exist_ok=True)
    write_json(PUBLIC / "aggregate.json", sanitize(aggregate_raw))
    write_json(PUBLIC / "manifest.json", sanitize(manifest_raw))
    write_json(PUBLIC / "data_qualification.json", sanitize(qualification))
    write_json(PUBLIC / "software_check.json", sanitize(software))
    write_json(PUBLIC / "pipeline_status.json", sanitize(pipeline_raw))

    for update in DIAGNOSTIC_UPDATES:
        write_json(PUBLIC / f"eval_u{update}.json", sanitize(read_json(INSPECTION / f"eval_u{update}.json")))

    samples = load_training_samples()
    with (PUBLIC / "training_samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in samples:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")

    image_order = list(manifest_raw["image_ids"]["outer_validation"])
    summary = summarize(aggregate_raw, manifest_raw, pipeline_raw, INSPECTION)
    write_json(PUBLIC / "summary.json", summary)
    final_rows = write_per_image_csv(PUBLIC / "final_per_image_metrics.csv", aggregate_raw, image_order)

    results = render_results(summary, samples=len(samples), final_rows=final_rows)
    (PUBLIC / "RESULTS.md").write_text(results, encoding="utf-8", newline="\n")

    raw_inputs = {
        "aggregate_sha256": sha256_file(INSPECTION / "aggregate.json"),
        "data_qualification_sha256": sha256_file(INSPECTION / "data_qualification.json"),
        "manifest_sha256": sha256_file(INSPECTION / "manifest.json"),
        "pipeline_status_sha256": sha256_file(INSPECTION / "pipeline_status.json"),
        "sampled_training_log_sha256": sha256_file(INSPECTION / "training_attempt_fd5f8b054ff5.jsonl"),
        "software_check_sha256": sha256_file(INSPECTION / "software_check.json"),
        **{f"eval_u{update}_sha256": sha256_file(INSPECTION / f"eval_u{update}.json") for update in DIAGNOSTIC_UPDATES},
    }
    source_binding_digest = hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    public_hashes = {
        path.name: sha256_file(path)
        for path in sorted(PUBLIC.iterdir())
        if path.is_file() and path.name not in {"provenance.json", "sanitization.json"}
    }
    write_json(PUBLIC / "sanitization.json", {
        "scope": "allowlisted text and JSON evidence only",
        "excluded": [
            "model weights and checkpoints",
            "source images, masks, prepared NPZ data, and caches",
            "full runtime logs and dispatch/launch receipts",
            "disposable GPU software-smoke output",
            "attempt marker from sampled training JSONL",
        ],
        "redactions": dict(sorted(REDACTIONS.items())),
        "notes": [
            "Metric rows, aggregate gates, configuration, split identifiers and hashes are retained.",
            "The complete final per-plan rows remain in aggregate.json; CSV values average the two plans per image.",
            "Only sparse training records saved at update 1 and subsequent sampled updates are included.",
        ],
    })
    write_json(PUBLIC / "provenance.json", {
        "schema": "fives_nca_public_provenance_v1",
        "run_id": "fives_nca_20261010_01",
        "status": aggregate_raw["status"],
        "verdict": aggregate_raw["verdict"],
        "source_root": "new/real_task_fives",
        "executed_source_sha256": sources,
        "source_binding_sha256": source_binding_digest,
        "source_hashes_verified_against_current_tree": True,
        "inspection_input_sha256": raw_inputs,
        "public_artifact_sha256": public_hashes,
        "training_sample_rows": len(samples),
        "final_per_image_horizon_rows": final_rows,
    })
    privacy_check(PUBLIC)
    verify_public()


def render_results(summary: dict[str, Any], *, samples: int, final_rows: int) -> str:
    def metric(value: float) -> str:
        return f"{value:.6f}"

    coarse = summary["coarse_baseline_mean"]
    lines = [
        "# FIVES real-task NCA v0 results",
        "",
        f"Status: **{summary['status']}**. Scientific verdict: **{summary['verdict']}**.",
        "",
        "One developmental block trained `standard_k64` for 1,500 updates with a 64-step rollout and full 64-step credit. All final metrics use the same saved checkpoint at 512 × 512 evaluation horizons; T8 is an evaluation horizon for this K64-trained model, not a separately trained T8 reference.",
        "",
        "| Evaluation | Mean Dice | Mean clDice |",
        "|---|---:|---:|",
        f"| Coarse teacher | {metric(coarse['dice'])} | {metric(coarse['cldice'])} |",
    ]
    for t in HORIZONS:
        row = summary["final_horizon_means"][str(t)]
        lines.append(f"| T{t} | {metric(row['dice'])} | {metric(row['cldice'])} |")
    lines += [
        "",
        "Final horizon means use 120 validation images and two matched stochastic plans per image. The repeated plans are measurements, not independent samples. `final_per_image_metrics.csv` contains 720 image-by-horizon rows averaged over those plans; `aggregate.json` preserves all 1,440 plan-specific final rows and raw gate inputs.",
        "",
        "Exactly two frozen gates failed: `k64_cldice_gain` (T64 clDice improvement over the coarse baseline was "
        f"{summary['gate_differences']['k64_cldice_gain']:+.6f}, below +0.010) and `within_model_depth` (T64 minus T8 clDice was "
        f"{summary['gate_differences']['within_model_depth']:+.6f}, below +0.010). The coarse floors and T64 Dice guard passed. The overall run completed, but the task remained unqualified.",
        "",
        "The official training split supplied 480 development images and 120 validation images, with image ID as the split unit. No patient mapping was available, so this is not a patient-disjoint result. The official archive was unpacked; test labels were excluded from discovery, decoding, training, and evaluation. No test labels were loaded.",
        "",
        "The separate `standard_t8`, `standard_k8`, and `au_k8` arms were not run. This evidence supports no AU conclusion and does not establish that the task generally lacks long-computation benefit.",
        "",
        "The process-level pipeline elapsed time was "
        f"{summary['training']['elapsed_seconds']:.3f} seconds. It covers the recovered pipeline process, excludes earlier acquisition/download/transfer work, and is not an end-to-end speed measurement.",
        "",
        "Intermediate T64 checkpoints are formation diagnostics on the fixed 16-image cohort:",
        "",
        "| Update | Mean Dice | Mean clDice |",
        "|---:|---:|---:|",
    ]
    for update in DIAGNOSTIC_UPDATES:
        row = summary["intermediate_t64_diagnostics"][str(update)]["mean"]
        lines.append(f"| {update} | {metric(row['dice'])} | {metric(row['cldice'])} |")
    lines += [
        "",
        f"Saved software-check results are in `software_check.json`. `training_samples.jsonl` contains {samples} sampled update records, not a complete per-update log. `manifest.json` preserves the split, OOF/full-teacher provenance, and checkpoint/data hashes. The source hashes in `provenance.json` bind the run to `new/real_task_fives`.",
        "",
        "Offline verification: `python tools/publish_fives_nca.py --verify`.",
        "",
    ]
    return "\n".join(lines)


def verify_public() -> dict[str, Any]:
    aggregate = read_json(PUBLIC / "aggregate.json")
    manifest = read_json(PUBLIC / "manifest.json")
    pipeline = read_json(PUBLIC / "pipeline_status.json")
    summary = read_json(PUBLIC / "summary.json")
    provenance = read_json(PUBLIC / "provenance.json")
    if aggregate.get("status") != "COMPLETE" or aggregate.get("verdict") != "TASK_UNQUALIFIED":
        raise AssertionError("public aggregate status/verdict mismatch")
    sources = aggregate["arms"]["standard_k64"]["binding"]["sources"]
    if sources != provenance["executed_source_sha256"]:
        raise AssertionError("public provenance source hashes differ from aggregate binding")
    if sources != manifest["provenance"]["code"]:
        raise AssertionError("public manifest source hashes differ from aggregate binding")
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if digest != provenance["source_binding_sha256"]:
        raise AssertionError("source binding digest mismatch")
    for name, expected in sources.items():
        source_path = SOURCE_ROOT / name
        if not source_path.is_file() or sha256_file(source_path) != expected:
            raise AssertionError(f"current source does not match executed source hash: {name}")
    for name, expected in provenance["public_artifact_sha256"].items():
        path = PUBLIC / name
        if not path.is_file() or sha256_file(path) != expected:
            raise AssertionError(f"public evidence hash mismatch: {name}")

    final_rows = aggregate["arms"]["standard_k64"]["final"]["rows"]
    baseline_rows = aggregate["baseline"]["rows"]
    if len(baseline_rows) != 120 or len(final_rows) != 120 * 2 * len(HORIZONS):
        raise AssertionError("public aggregate row counts do not match the frozen design")
    ids = {str(row["id"]) for row in baseline_rows}
    if len(ids) != 120 or any({str(row["id"]) for row in final_rows if int(row["t"]) == t} != ids for t in HORIZONS):
        raise AssertionError("final horizon image sets differ from the 120-image baseline")
    for t in HORIZONS:
        rows = [row for row in final_rows if int(row["t"]) == t]
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[str(row["id"])] += 1
        if len(rows) != 240 or set(counts.values()) != {2}:
            raise AssertionError(f"T{t} does not have two plans for every image")

    recomputed = summarize(aggregate, manifest, pipeline, PUBLIC)
    compare_summary(recomputed, summary)
    if aggregate["qualification"]["gates"] != recomputed["gates"]:
        raise AssertionError("public gate values do not recompute from raw metric rows")
    if sum(not x for x in recomputed["gates"].values()) != 2:
        raise AssertionError("expected exactly two failed gates")

    csv_path = PUBLIC / "final_per_image_metrics.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 720:
        raise AssertionError(f"expected 720 plan-averaged CSV rows, found {len(csv_rows)}")
    expected_grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        expected_grouped[(str(row["id"]), int(row["t"]))].append(row)
    for row in csv_rows:
        key = (row["id"], int(row["t"]))
        raw = expected_grouped[key]
        for metric in METRICS:
            expected = sum(float(r["metrics"][metric]) for r in raw) / len(raw)
            if not math.isclose(float(row[metric]), expected, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError(f"per-image CSV mismatch at {key} for {metric}")

    privacy_check(PUBLIC)
    return {
        "status": "PASS",
        "checks": [
            "source hashes match the executed binding and current source tree",
            "public evidence hashes match provenance.json",
            "baseline and final per-plan row counts and image coverage match",
            "aggregate means and all five gates recompute from saved rows",
            "per-image CSV matches the two-plan averages in aggregate.json",
            "private paths, IPs, and receipt metadata are absent",
        ],
        "images": len(ids),
        "horizons": list(HORIZONS),
        "final_raw_plan_rows": len(final_rows),
        "final_per_image_horizon_rows": len(csv_rows),
        "failed_gates": [key for key, passed in recomputed["gates"].items() if not passed],
    }


def compare_summary(expected: Any, actual: Any, label: str = "summary") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise AssertionError(f"{label}: JSON object keys differ")
        for key, value in expected.items():
            compare_summary(value, actual[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if expected != actual:
            raise AssertionError(f"{label}: list differs")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(expected, float(actual), rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{label}: numeric value differs")
    elif expected != actual:
        raise AssertionError(f"{label}: value differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify the existing public evidence package offline")
    args = parser.parse_args()
    try:
        if args.verify:
            print(json.dumps(verify_public(), indent=2, sort_keys=True))
        else:
            export()
            print(json.dumps(verify_public(), indent=2, sort_keys=True))
    except (AssertionError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
