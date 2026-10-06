"""Publish and verify the frozen continuous-coverage run using saved data only."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/continuous_coverage_20261006_02"
QUALIFICATION = ROOT / "analyses/continuous_coverage_check_20261006_03.json"
PUBLIC = ROOT / "evidence/continuous_coverage_20261006"
MANIFEST = ROOT / "CONTINUOUS_COVERAGE_PUBLICATION_MANIFEST.json"
PROTOCOL = "continuous_execution_coverage_v1"
ARMS = ("reset64x4", "continuous256")
BLOCKS = 8
CHECKPOINTS = tuple(range(0, 301, 25))
SIZES = (32, 64)
MAX_FILE_BYTES = 90 * 1024 * 1024
PRIVATE_KEYS = {"pid", "host", "hostname", "username", "gpu_uuid", "executable",
                "command", "launch_command", "cwd"}
PRIVATE_TEXT = re.compile(
    r"\b[A-Za-z]:[\\/]|\\\\[^\\\s]+[\\/]|(?<![\w.])/(?:[^/\s]+/)+[^/\s]+|"
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b|BEGIN [^\n]*PRIVATE KEY|"
    r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|"
    r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}"
)


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def safe(value):
    if isinstance(value, dict):
        assert not ({str(key).lower() for key in value} & PRIVATE_KEYS), "private metadata key"
        for item in value.values():
            safe(item)
    elif isinstance(value, list):
        for item in value:
            safe(item)
    elif isinstance(value, str):
        assert not PRIVATE_TEXT.search(value), "private text detected"


def json_bytes(path: Path) -> bytes:
    return path.read_bytes()


def copy_bound(source: Path, destination: Path, bindings: dict, run_relative: str,
               encoding: str = "byte-exact copy"):
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest = sha(source)
    if encoding == "gzip lossless":
        destination.write_bytes(gzip.compress(source.read_bytes(), compresslevel=9, mtime=0))
    elif encoding == "byte-exact copy":
        shutil.copyfile(source, destination)
    else:
        raise ValueError(encoding)
    assert destination.stat().st_size < MAX_FILE_BYTES, destination
    bindings[run_relative] = {
        "sha256": source_digest,
        "public_path": destination.relative_to(ROOT).as_posix(),
        "encoding": encoding,
    }


def gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        return json.load(handle)


def bank_tensor_hash(bank: dict[str, np.ndarray]) -> str:
    """Match the frozen torch tensor_hash over CPU NumPy bank tensors."""
    digest = hashlib.sha256()
    for key, value in sorted(bank.items()):
        array = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_bank(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def decode_trace(path: Path, size: int) -> dict[str, np.ndarray]:
    expected_shape = (257, 32, size, size)
    count = math.prod(expected_shape)
    traces = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in ("correct", "original_correct", "flipped_correct"):
            shape = tuple(int(value) for value in archive[f"{name}_shape"])
            packed = archive[f"{name}_packed"]
            assert shape == expected_shape, (path, name, shape)
            assert packed.dtype == np.uint8 and packed.size == (count + 7) // 8
            unpacked = np.unpackbits(packed, bitorder="little", count=count)
            roundtrip = np.packbits(unpacked, bitorder="little")
            assert np.array_equal(roundtrip, packed), (path, name, "bitpack roundtrip")
            traces[name] = unpacked.reshape(expected_shape).astype(bool, copy=False)
    return traces


def trace_metrics(correct: np.ndarray, bank: dict[str, np.ndarray]) -> dict:
    changed = bank["changed"][:, 0].astype(bool)
    distance = bank["distance"][:, 0]
    strict = changed & (distance > 16) & (distance < 32)
    map_pixels = strict.sum(axis=(1, 2), dtype=np.int64)
    map_correct = (correct[64] & strict).sum(axis=(1, 2), dtype=np.int64)
    eligible = map_pixels > 0
    strict_pixels = int(map_pixels.sum())
    strict_correct = int(map_correct.sum())
    reference = correct[64] & changed
    retained = int((reference & correct[256]).sum())
    reference_pixels = int(reference.sum())
    reference_maps = int(reference.sum(axis=(1, 2), dtype=np.int64).astype(bool).sum())
    continuously_correct = int((reference & np.all(correct[64:], axis=0)).sum())
    return {
        "R_strict_pooled_T64": strict_correct / strict_pixels if strict_pixels else None,
        "R_strict_mean_T64": float((map_correct[eligible] / map_pixels[eligible]).mean()) if eligible.any() else None,
        "R_strict_numerator": strict_correct,
        "R_strict_denominator": strict_pixels,
        "S_retention64_to256": retained / reference_pixels if reference_pixels else None,
        "S_retention_numerator": retained,
        "S_continuous_survival64_to256": continuously_correct / reference_pixels if reference_pixels else None,
        "S_continuous_survival_numerator": continuously_correct,
        "retention_reference_pixels": reference_pixels,
        "retention_reference_maps": reference_maps,
    }


def close(actual, expected, name: str):
    if actual is None or expected is None:
        assert actual is expected, (name, actual, expected)
    else:
        assert math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12), (
            name, actual, expected)


def compare_trace_metrics(actual: dict, row: dict, label: str):
    for key in ("R_strict_pooled_T64", "R_strict_mean_T64", "S_retention64_to256",
                "S_continuous_survival64_to256", "retention_reference_pixels",
                "retention_reference_maps"):
        close(actual[key], row[key], f"{label}.{key}")
    # Dense and final compact rows preserve the exact count behind each rate.
    if "S_retention_numerator" in row:
        assert actual["S_retention_numerator"] == row["S_retention_numerator"]


def joint_pass(metrics: dict) -> bool:
    return bool(
        metrics["R_strict_mean_T64"] is not None and metrics["R_strict_mean_T64"] >= 0.80
        and metrics["R_strict_pooled_T64"] is not None and metrics["R_strict_pooled_T64"] >= 0.80
        and metrics["S_retention64_to256"] is not None and metrics["S_retention64_to256"] >= 0.95
        and metrics["retention_reference_maps"] >= 16
        and metrics["retention_reference_pixels"] >= 100
    )


def exact_two_sided_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant)


def check_paired_aggregate(summary: dict, perarm: list[dict], trace_rows: dict) -> dict:
    by_pair = {}
    for row in perarm:
        assert row["update"] == 300
        by_pair.setdefault(row["block"], {})[row["arm"]] = row
    assert set(by_pair) == set(range(BLOCKS))
    wins = losses = 0
    final_joint = {arm: 0 for arm in ARMS}
    final_full = {arm: 0 for arm in ARMS}
    delta_values = {"R_strict_mean_T64": [], "R_strict_pooled_T64": [],
                    "S_retention64_to256": [], "S_continuous_survival64_to256": []}
    positive_s = 0
    for block, pair in by_pair.items():
        assert set(pair) == set(ARMS)
        recomputed = {}
        for arm in ARMS:
            row = pair[arm]
            actual = trace_rows[(block, arm, 300, 32)]
            frozen = row["metrics"]["sizes"]["32"]
            compare_trace_metrics(actual, frozen, f"final block{block:02d}/{arm}")
            readiness = joint_pass(actual)
            assert readiness == bool(row["joint"]["pass"])
            recomputed[arm] = {**actual, "joint_pass": readiness}
            final_joint[arm] += int(readiness)
            final_full[arm] += int(bool(row["metrics"]["full_pass"]))
        cpass, rpass = recomputed["continuous256"]["joint_pass"], recomputed["reset64x4"]["joint_pass"]
        if cpass and not rpass:
            wins += 1
        elif rpass and not cpass:
            losses += 1
        for key in delta_values:
            left, right = recomputed["reset64x4"][key], recomputed["continuous256"][key]
            if key.startswith("S_"):
                support = all(
                    recomputed[arm]["retention_reference_maps"] >= 16
                    and recomputed[arm]["retention_reference_pixels"] >= 100 for arm in ARMS
                )
                delta = right - left if left is not None and right is not None and support else None
            else:
                delta = right - left if left is not None and right is not None else None
            if delta is not None:
                delta_values[key].append(delta)
            if key == "S_retention64_to256" and delta is not None and delta > 0:
                positive_s += 1

    p_value = exact_two_sided_p(wins, losses)
    primary = summary["primary"]
    assert primary["wins"] == wins and primary["losses"] == losses
    assert primary["exact_two_sided_p"] == p_value
    assert primary["reset_passes"] == final_joint["reset64x4"]
    assert primary["continuous_passes"] == final_joint["continuous256"]
    assert summary["primary_verdict"] == "NO_CONTINUOUS_COVERAGE_RELIABILITY_QUALIFICATION"
    assert final_joint == {"reset64x4": 0, "continuous256": 0}
    assert final_full == {"reset64x4": 0, "continuous256": 0}
    for key, values in delta_values.items():
        expected = summary["paired_metrics"][key]
        mean = sum(values) / len(values) if values else None
        close(mean, expected["mean_continuous_minus_reset"], f"paired.{key}.mean")
        assert expected["n_defined"] == len(values)
    assert positive_s == summary["paired_metrics"]["S_retention64_to256"]["wins"] == 4
    return {
        "independent_unit": "paired training block",
        "paired_blocks": BLOCKS,
        "joint_readiness_passes_at_u300": final_joint,
        "saved_old_full_passes_at_u300": final_full,
        "primary_wins": wins,
        "primary_losses": losses,
        "primary_exact_two_sided_p": p_value,
        "R_strict_pooled_T64_mean_delta": summary["paired_metrics"]["R_strict_pooled_T64"]["mean_continuous_minus_reset"],
        "R_strict_pooled_T64_positive_pairs": summary["paired_metrics"]["R_strict_pooled_T64"]["wins"],
        "S_retention64_to256_mean_delta": summary["paired_metrics"]["S_retention64_to256"]["mean_continuous_minus_reset"],
        "S_retention64_to256_positive_pairs": positive_s,
        "primary_verdict": summary["primary_verdict"],
    }


def sanitize_qualification(raw: dict) -> dict:
    sections = raw["sections"]
    cuda = sections["cuda_graph"]
    cuda_fields = ("status", "updates", "batch_seed_by_update", "forward_steps_per_update",
                   "backward_windows_per_update", "capture_warmup_updates_with_optimizer",
                   "loss_gradients_parameters_adam_bitwise_equal")
    sanitized = {
        "status": raw["status"],
        "protocol": raw["protocol"],
        "seed": raw["seed"],
        "bank_seed": raw["bank_seed"],
        "updates_per_mode": raw["updates_per_mode"],
        "modes": raw["modes"],
        "sections": {
            "cpu": sections["cpu"],
            "cuda_graph": {
                "status": cuda["status"],
                "gpu_model": cuda["clock"].get("device_name"),
                "modes": {
                    mode: {key: value for key, value in item.items() if key in cuda_fields}
                    for mode, item in cuda["modes"].items()
                },
            },
            "trace_replay": {key: value for key, value in sections["trace_replay"].items()
                             if key != "elapsed_seconds"},
        },
        "source_sha256": raw["source_sha256"],
        "runtime_limit_enforced": raw["runtime_limit_enforced"],
        "sanitization": "Pass criteria and scientific checks retained; timestamps, device index/memory/clock, timing, and allocation telemetry removed.",
    }
    assert sanitized["status"] == "PASS" and sanitized["protocol"] == PROTOCOL
    safe(sanitized)
    return sanitized


def current_source_bindings(run_manifest: dict) -> list[dict]:
    results, mismatches = [], []
    for relative, expected in run_manifest["source_sha256"].items():
        snapshot = RUN / "source" / relative
        current = ROOT / relative
        snapshot_hash = sha(snapshot) if snapshot.is_file() else None
        current_hash = sha(current) if current.is_file() else None
        matched = snapshot_hash == current_hash == expected
        results.append({"path": relative, "sha256": expected, "snapshot_matches": snapshot_hash == expected,
                        "current_matches": current_hash == expected})
        if not matched:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError("source snapshot/current hash mismatch: " + ", ".join(mismatches))
    return results


def raw_artifacts() -> list[tuple[Path, str, str]]:
    items = []
    for name in ("config.json", "dense.json", "metrics.csv", "perarm.json", "summary.json",
                 "RESULTS.md", "plans.json", "reach_retention_trajectory.png"):
        encoding = "gzip lossless" if name == "plans.json" else "byte-exact copy"
        public_name = "plans.json.gz" if name == "plans.json" else ("frozen_RESULTS.md" if name == "RESULTS.md" else name)
        items.append((RUN / name, public_name, encoding))
    for name in ("train.npz", "evaluation32.npz", "evaluation64.npz"):
        items.append((RUN / "banks" / name, f"banks/{name}", "byte-exact copy"))
    for block in range(BLOCKS):
        for arm in ARMS:
            folder = RUN / f"block{block:02d}" / arm
            items.append((folder / "training_curve.json", f"block{block:02d}/{arm}/training_curve.json.gz", "gzip lossless"))
            for update in CHECKPOINTS:
                source = folder / "evaluation" / f"u{update:03d}"
                public_folder = f"block{block:02d}/{arm}/evaluation/u{update:03d}"
                items.append((source / "summary.json", f"{public_folder}/summary.json.gz", "gzip lossless"))
                for size in SIZES:
                    name = f"size{size}_traces.npz"
                    items.append((source / name, f"{public_folder}/{name}", "byte-exact copy"))
                if update == 300:
                    items.append((source / "matched_frontier_strata.csv",
                                  f"{public_folder}/matched_frontier_strata.csv.gz", "gzip lossless"))
    return items


def build():
    assert not PUBLIC.exists() and not MANIFEST.exists(), "Refusing to overwrite published evidence"
    run_manifest = read(RUN / "manifest.json")
    status = read(RUN / "status.json")
    summary = read(RUN / "summary.json")
    dense = read(RUN / "dense.json")
    perarm = read(RUN / "perarm.json")
    assert run_manifest["protocol"] == PROTOCOL
    assert summary["status"] == status["status"] == "COMPLETE"
    assert summary["completed_arms"] == summary["expected_arms"] == status["completed_arms"] == 16
    assert summary["dense_records"] == len(dense) == 208
    assert len(perarm) == 16 and status["dense_records"] == 208
    assert len(list(path for path in RUN.rglob("*") if path.is_file())) == 929
    source_bindings = current_source_bindings(run_manifest)
    assert sha(RUN / "plans.json") == run_manifest["schedule_plan_sha256"]
    qualification = read(QUALIFICATION)
    assert sha(QUALIFICATION) == run_manifest["qualification_sha256"]
    assert qualification["status"] == "PASS" and qualification["source_sha256"] == run_manifest["source_sha256"]

    checkpoint_bindings = {}
    for row in dense:
        relative = row["checkpoint"]
        assert relative not in checkpoint_bindings
        path = RUN / relative
        digest = sha(path)
        assert digest == row["checkpoint_sha256"], (relative, digest, row["checkpoint_sha256"])
        checkpoint_bindings[relative.replace("\\", "/")] = {
            "file_sha256": digest, "parameter_sha256": row["parameter_sha256"],
        }
    assert len(checkpoint_bindings) == 208

    PUBLIC.mkdir(parents=True)
    bindings = {}
    for source, public_relative, encoding in raw_artifacts():
        assert source.is_file(), source
        rel = source.relative_to(RUN).as_posix()
        destination = PUBLIC / public_relative
        if source.suffix == ".json":
            safe(read(source))
        if source.suffix in (".csv", ".md"):
            assert not PRIVATE_TEXT.search(source.read_text(encoding="utf-8-sig")), source
        copy_bound(source, destination, bindings, rel, encoding)

    sanitized = sanitize_qualification(qualification)
    write(PUBLIC / "runtime_qualification.json", sanitized)
    banks = {}
    for name, filename in (("train", "train.npz"), ("evaluation32", "evaluation32.npz"),
                           ("evaluation64", "evaluation64.npz")):
        bank = load_bank(PUBLIC / "banks" / filename)
        actual_hash = bank_tensor_hash(bank)
        assert actual_hash == run_manifest["data_sha256"][name], (name, actual_hash)
        if name.startswith("evaluation"):
            banks[int(name.replace("evaluation", ""))] = bank

    plans = gzip_json(PUBLIC / "plans.json.gz")
    for block in range(BLOCKS):
        plan = plans[str(block)]
        assert plan["initialization_seed"] == 96001 + block
        assert plan["schedule_seed"] == 97001 + block
        batches = np.asarray(plan["batch_indices"])
        assert batches.shape == (300, 8) and batches.min() >= 0 and batches.max() < 512

    trace_rows = {}
    for row in dense:
        block, arm, update = row["block"], row["arm"], row["update"]
        assert arm in ARMS and update in CHECKPOINTS
        for size in SIZES:
            path = PUBLIC / f"block{block:02d}/{arm}/evaluation/u{update:03d}/size{size}_traces.npz"
            traces = decode_trace(path, size)
            measured = trace_metrics(traces["correct"], banks[size])
            expected = row["metrics"]["sizes"][str(size)]
            compare_trace_metrics(measured, expected, f"block{block:02d}/{arm}/u{update:03d}/size{size}")
            trace_rows[(block, arm, update, size)] = measured
    final = check_paired_aggregate(summary, perarm, trace_rows)

    for row in perarm:
        eval_summary_path = PUBLIC / f"block{row['block']:02d}/{row['arm']}/evaluation/u300/summary.json.gz"
        eval_summary = gzip_json(eval_summary_path)
        assert eval_summary["full_evaluated"] is True
        assert eval_summary["phenotype_gate"]["pass"] == row["metrics"]["full_pass"]
        compare_trace_metrics(trace_rows[(row["block"], row["arm"], 300, 32)],
                              row["metrics"]["sizes"]["32"], "perarm u300")

    curve_count = 0
    for block in range(BLOCKS):
        for arm in ARMS:
            curve = gzip_json(PUBLIC / f"block{block:02d}/{arm}/training_curve.json.gz")
            assert len(curve) == 300 and curve[0]["update"] == 1 and curve[-1]["update"] == 300
            curve_count += 1
    assert curve_count == 16
    # metrics.csv is already retained byte-exact; ensure its frozen row count is complete.
    with (PUBLIC / "metrics.csv").open("r", newline="", encoding="utf-8-sig") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 416

    file_counts = {
        "training_curves": 16,
        "evaluation_summaries": 208,
        "packed_trace_banks": 416,
        "final_frontier_csv": 16,
        "data_banks": 3,
        "checkpoint_files_published": 0,
    }
    validation = {
        "status": "PASS",
        "protocol": PROTOCOL,
        "scope": "Saved-data CPU arithmetic, provenance/file hashes, and record counts only; no model inference, training, optimizer update, or frontier regeneration.",
        "source_bindings": {"status": "PASS", "files": len(source_bindings), "mismatches": []},
        "run_input_files": 929,
        "public_scientific_file_counts": file_counts,
        "data_bank_tensor_hashes_recomputed": 3,
        "schedule_plan_sha256": run_manifest["schedule_plan_sha256"],
        "checkpoint_sha256_verified_locally": 208,
        "checkpoint_parameter_hashes_retained": 208,
        "checkpoint_contents_published": False,
        "packed_trace_shapes": {"size32": [257, 32, 32, 32], "size64": [257, 32, 64, 64]},
        "packed_trace_boolean_arrays": ["correct", "original_correct", "flipped_correct"],
        "packed_trace_roundtrip_equal": 416,
        "R_S_and_continuous_survival_recomputed": 208,
        "recomputed_saved_metric_rows": 208,
        "final_aggregate_recomputed_from_u300_records": final,
        "frontier_and_old_full_scope": "Saved final summaries and perarm records were reused for frontier and old Full; neither frontier rows nor Full were regenerated from traces.",
        "runtime_qualification_sanitized": True,
        "frozen_evidence_changed": False,
        "execution_elapsed_seconds": summary["elapsed_seconds"],
        "publication_fit_or_inference": False,
    }
    safe(validation)
    write(PUBLIC / "validation.json", validation)

    results = f"""# Continuous execution-state coverage

The frozen experiment completed 16/16 trajectories and 208/208 dense checkpoint evaluations. The formal endpoint is u300; all earlier checkpoints are formation diagnostics.

Both arms had joint-readiness 0/8 and old Full 0/8. The frozen paired reliability verdict is **{summary['primary_verdict']}** (joint-success wins {summary['primary']['wins']}, losses {summary['primary']['losses']}, exact two-sided p={summary['primary']['exact_two_sided_p']}). The independent unit is the paired training block (n=8).

At u300, continuous-minus-reset mean pooled strict reach R at T64 was {summary['paired_metrics']['R_strict_pooled_T64']['mean_continuous_minus_reset']:.10f}. Mean S retention from T64 to T256 changed by {summary['paired_metrics']['S_retention64_to256']['mean_continuous_minus_reset']:+.10f}; S was higher in {summary['paired_metrics']['S_retention64_to256']['wins']}/8 blocks. The S gain does not meet the frozen joint reliability criterion because neither arm passed readiness in any block.

Execution took {summary['elapsed_seconds']:.2f} seconds. Publication used no model fitting, training, or inference. The all-checkpoint R/S/survival checks use saved Boolean traces and evaluation banks. Final frontier and old Full values remain the saved u300 evaluator outputs; this export did not regenerate frontier strata.

See [REPRODUCTION.md](REPRODUCTION.md), [summary.json](summary.json), [dense.json](dense.json), [perarm.json](perarm.json), [metrics.csv](metrics.csv), and [validation.json](validation.json). Packed traces, checkpoint hashes, schedules, curves and full per-checkpoint summaries are retained as secondary data.
"""
    (PUBLIC / "RESULTS.md").write_text(results, encoding="utf-8")
    reproduction = """# Verification and reproduction

From the repository root, verify this publication using CPU-only saved-data checks:

```powershell
python -X utf8 -B tools/export_continuous_coverage.py --verify-only
```

This command does not require the local `runs/` tree, analysis qualification file, or checkpoint files. It validates the package file hashes, all three data-bank tensor hashes, the eight frozen schedules, every packed Boolean trace shape and lossless bitpack roundtrip, R/S/continuous-survival arithmetic for all 208 checkpoints, and the paired u300 aggregate. `validation.json` records the scope. The final saved frontier and old Full summaries are reused and not regenerated by this check.

The frozen protocol and runnable implementation are in `new/continuous_coverage/PROTOCOL.md` and `new/continuous_coverage/`. The formal run used 8 paired blocks, 300 super-updates per arm, and a fixed u300 endpoint. Re-running training requires the qualified PyTorch/CUDA environment and a GPU; this archive does not include checkpoint or optimizer-state contents. The public dense/perarm records retain every checkpoint file hash and parameter hash. All 208 checkpoint file hashes were checked against the local files while building the publication.

The publication manifest binds the protocol ID, source snapshot hashes, source/current equality, data-bank tensor hashes, schedule-plan hash, original qualification hash, and all checkpoint hashes. The run's source snapshot was checked against the current checkout before export; each recorded source hash matched both copies. The runtime qualification was sanitized to retain its pass evidence while removing time, device-index, memory and timing details.

`RESULTS.md` gives the frozen result and claim boundary. `config.json` records the configuration; `summary.json` and `perarm.json` hold the formal endpoint and paired blocks; `dense.json` and `metrics.csv` retain all checkpoint metrics. Each block/arm folder contains its full lossless training curve, all 13 per-checkpoint summaries and both-size Boolean trace banks. The three source banks are in `banks/`; final matched-frontier rows are losslessly gzip-compressed under each u300 evaluation folder. NPZ files remain separate and unchanged.
"""
    (PUBLIC / "REPRODUCTION.md").write_text(reproduction, encoding="utf-8")
    safe(read(PUBLIC / "config.json"))
    safe(read(PUBLIC / "summary.json"))
    safe(read(PUBLIC / "dense.json"))
    safe(read(PUBLIC / "perarm.json"))
    safe(read(PUBLIC / "runtime_qualification.json"))
    assert not PRIVATE_TEXT.search(results + reproduction)

    published = {path.relative_to(ROOT).as_posix(): sha(path)
                 for path in PUBLIC.rglob("*") if path.is_file()}
    assert len(published) == 671, len(published)
    public_bytes = sum((ROOT / relative).stat().st_size for relative in published)
    exporter_hash = sha(Path(__file__))
    publication = {
        "protocol": "continuous_coverage_saved_data_publication_v1",
        "run_id": RUN.name,
        "review_base": run_manifest["git_review_base"],
        "source_sha256": run_manifest["source_sha256"],
        "source_binding_status": "PASS",
        "source_binding_file_count": len(source_bindings),
        "data_sha256": run_manifest["data_sha256"],
        "schedule_plan_sha256": run_manifest["schedule_plan_sha256"],
        "qualification_sha256": run_manifest["qualification_sha256"],
        "sanitized_qualification_sha256": sha(PUBLIC / "runtime_qualification.json"),
        "checkpoint_bindings": checkpoint_bindings,
        "exporter_sha256": exporter_hash,
        "raw_artifact_bindings": bindings,
        "published_sha256": published,
        "published_file_count": len(published),
        "published_bytes": public_bytes,
        "excluded": ["all checkpoint and optimizer-state contents", "run manifest and status/PID records",
                     "source snapshot copies (hash-bound to run snapshots and current source files)",
                     "launch receipts, logs, host/session/absolute-path metadata"],
    }
    safe(publication)
    write(MANIFEST, publication)
    verify()
    print(json.dumps({"status": "PASS", "files": len(published), "MiB": round(public_bytes / 1024**2, 2),
                      "trace_banks": 416, "recomputed_checkpoints": 208,
                      "checkpoint_hashes_verified_locally": 208, **final}, allow_nan=False), flush=True)


def verify():
    """Verify only the public package; deliberately independent of runs/checkpoints."""
    publication = read(MANIFEST)
    safe(publication)
    assert sha(Path(__file__)) == publication["exporter_sha256"]
    actual = {path.relative_to(ROOT).as_posix(): sha(path)
              for path in PUBLIC.rglob("*") if path.is_file()}
    assert actual == publication["published_sha256"]
    assert len(actual) == publication["published_file_count"] == 671
    assert len(publication["checkpoint_bindings"]) == 208
    assert not any((ROOT / relative).suffix == ".pt" for relative in actual)
    assert sum((ROOT / relative).stat().st_size for relative in actual) == publication["published_bytes"]
    assert all((ROOT / relative).stat().st_size < MAX_FILE_BYTES for relative in actual)

    for run_relative, binding in publication["raw_artifact_bindings"].items():
        public_path = ROOT / binding["public_path"]
        if binding["encoding"] == "gzip lossless":
            with gzip.open(public_path, "rb") as handle:
                raw_digest = hashlib.sha256(handle.read()).hexdigest()
            assert raw_digest == binding["sha256"], run_relative
        else:
            assert binding["encoding"] == "byte-exact copy"
            assert sha(public_path) == binding["sha256"], run_relative

    for relative in actual:
        path = ROOT / relative
        if path.suffix == ".json":
            safe(read(path))
        elif path.suffix in (".md", ".csv"):
            assert not PRIVATE_TEXT.search(path.read_text(encoding="utf-8-sig")), relative
    sanitized = read(PUBLIC / "runtime_qualification.json")
    assert sanitized["status"] == "PASS" and sanitized["protocol"] == PROTOCOL
    assert sanitized["source_sha256"] == publication["source_sha256"]
    assert "clock" not in sanitized["sections"]["cuda_graph"]

    for name, filename in (("train", "train.npz"), ("evaluation32", "evaluation32.npz"),
                           ("evaluation64", "evaluation64.npz")):
        assert bank_tensor_hash(load_bank(PUBLIC / "banks" / filename)) == publication["data_sha256"][name]
    plans = gzip_json(PUBLIC / "plans.json.gz")
    assert sha_bytes(gzip.decompress((PUBLIC / "plans.json.gz").read_bytes())) == publication["schedule_plan_sha256"]
    for block in range(BLOCKS):
        plan = plans[str(block)]
        assert plan["initialization_seed"] == 96001 + block
        assert plan["schedule_seed"] == 97001 + block
        batches = np.asarray(plan["batch_indices"])
        assert batches.shape == (300, 8) and batches.min() >= 0 and batches.max() < 512

    summary, dense, perarm = read(PUBLIC / "summary.json"), read(PUBLIC / "dense.json"), read(PUBLIC / "perarm.json")
    assert summary["status"] == "COMPLETE" and len(dense) == 208 and len(perarm) == 16
    banks = {size: load_bank(PUBLIC / f"banks/evaluation{size}.npz") for size in SIZES}
    trace_rows = {}
    for row in dense:
        block, arm, update = row["block"], row["arm"], row["update"]
        assert arm in ARMS and update in CHECKPOINTS
        for size in SIZES:
            path = PUBLIC / f"block{block:02d}/{arm}/evaluation/u{update:03d}/size{size}_traces.npz"
            traces = decode_trace(path, size)
            measured = trace_metrics(traces["correct"], banks[size])
            compare_trace_metrics(measured, row["metrics"]["sizes"][str(size)],
                                  f"block{block:02d}/{arm}/u{update:03d}/size{size}")
            trace_rows[(block, arm, update, size)] = measured
    final = check_paired_aggregate(summary, perarm, trace_rows)
    for row in perarm:
        saved = gzip_json(PUBLIC / f"block{row['block']:02d}/{row['arm']}/evaluation/u300/summary.json.gz")
        assert saved["full_evaluated"] is True
        assert saved["phenotype_gate"]["pass"] == row["metrics"]["full_pass"]

    assert len(list(PUBLIC.rglob("*.npz"))) == 419
    assert len(list(PUBLIC.rglob("training_curve.json.gz"))) == 16
    assert len(list(PUBLIC.rglob("summary.json.gz"))) == 208
    assert len(list(PUBLIC.rglob("size*_traces.npz"))) == 416
    assert len(list(PUBLIC.rglob("matched_frontier_strata.csv.gz"))) == 16
    validation = read(PUBLIC / "validation.json")
    assert validation["status"] == "PASS" and validation["R_S_and_continuous_survival_recomputed"] == 208
    assert validation["final_aggregate_recomputed_from_u300_records"] == final
    print(json.dumps({"status": "PASS", "files": len(actual), "trace_banks": 416,
                      "recomputed_checkpoints": 208, "bank_hashes": 3, **final}, allow_nan=False), flush=True)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify the published package without accessing run, qualification, or checkpoint files.")
    args = parser.parse_args()
    verify() if args.verify_only else build()
