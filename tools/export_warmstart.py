"""Export allowlisted detached warm-start evidence and verify it without GPU work."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/warmstart_20261003_paired01"
PREFLIGHT_OLD = ROOT / "runs/warmstart_20261003_preflight01"
PREFLIGHT = ROOT / "runs/warmstart_20261003_preflight02"
CPU_OLD = ROOT / "analyses/warmstart_20261003_cpu01.json"
CPU = ROOT / "analyses/warmstart_20261003_cpu02.json"
LOCAL_VALIDATION = ROOT / "analyses/warmstart_20261003_publication_validation/validation.json"
OUT = ROOT / "evidence/warmstart_20261003"
PUBLICATION = ROOT / "WARMSTART_PUBLICATION_MANIFEST.json"
VALIDATOR = ROOT / "tools/validate_warmstart.py"
PROTOCOL_ID = "detached_warmstart_v1_nocap"
SEEDS = (2, 3, 4, 5)
VARIANTS = ("baseline", "warmstart")
ARMS = tuple(f"{variant}_seed{seed}" for seed in SEEDS for variant in VARIANTS)
STAGES = (0, 100, 200, 300)
TRACE_STEPS = 128

PRIVATE_KEYS = {
    "pid", "host", "hostname", "username", "gpu", "gpu_uuid", "device",
    "command", "argv", "cwd", "launch_command", "working_directory",
    "run_directory", "preflight_directory", "executable", "run", "out",
    "user", "account", "account_name", "machine", "machine_name", "computer",
    "computer_name", "gpu_name", "gpu_model", "hardware", "serial", "ip",
    "ip_address", "network_interface",
}
PRIVATE_TEXT = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|mnt|Users|root|tmp|var|opt)/)|"
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
)
FRONTIER_COLUMNS = (
    "size", "map_index", "start_step", "end_step", "distance",
    "frontier_opportunities", "frontier_acquired", "nonfrontier_opportunities",
    "nonfrontier_acquired", "matched_weight", "frontier_rate",
    "nonfrontier_rate", "rate_difference",
)
TURNOVER_COLUMNS = (
    "arm", "seed", "variant", "training_update", "t", "correct",
    "changed_pixels", "q_equal_map_mean", "q_eligible_maps",
    "q_pooled_correct", "q_pooled_pixels", "q_pooled_rate", "acquired",
    "wrong_old", "g_equal_map_mean", "g_eligible_maps", "g_pooled_numerator",
    "g_pooled_denominator", "g_pooled_rate", "destroyed", "correct_old",
    "d_equal_map_mean", "d_eligible_maps", "d_pooled_numerator",
    "d_pooled_denominator", "d_pooled_rate", "delta_correct", "accounting_error",
)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def public(value: Any) -> Any:
    """Recursively remove execution identity fields from metadata copies."""
    if isinstance(value, dict):
        return {key: public(item) for key, item in value.items() if key.lower() not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [public(item) for item in value]
    return value


def close(a: float | None, b: float | None, *, tol: float = 1e-11) -> None:
    if a is None or b is None:
        assert a is b, (a, b)
    else:
        assert math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol), (a, b)


def components(summary: dict[str, Any]) -> dict[str, bool]:
    checks = summary["phenotype_gate"]["checks"]
    return {
        "reach": all(row["pass"] for name, row in checks.items() if name.startswith("reach_")),
        "hold": all(row["pass"] for name, row in checks.items() if name.startswith("hold_")),
        "dynamics": all(
            row["pass"] for name, row in checks.items()
            if name.startswith(("size32_", "size64_")) and "frontier" not in name
        ),
        "frontier": all(row["pass"] for name, row in checks.items() if "frontier" in name),
        "full_phenotype": bool(summary["phenotype_gate"]["pass"]),
    }


def coverage(size_summary: dict[str, Any], horizon: int) -> dict[str, Any]:
    return size_summary["endpoints"][str(horizon)]["strict_16_32"]["pooled_coverage"]


def arm_row(name: str, arm: dict[str, Any], saved: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "seed": int(arm["seed"]),
        "variant": arm["variant"],
        "historical_replay": arm.get("historical_replay", {}).get("status"),
        **components(saved),
        "failed_checks": [
            name for name, check in saved["phenotype_gate"]["checks"].items()
            if not check["pass"]
        ],
        "checks": saved["phenotype_gate"]["checks"],
        "size32_T64_original_ba": saved["sizes"]["32"]["endpoints"]["64"]["original_ba"],
        "size32_T64_flipped_ba": saved["sizes"]["32"]["endpoints"]["64"]["flipped_ba"],
    }
    for horizon in (64, 128, 256):
        c = coverage(saved["sizes"]["32"], horizon)
        row[f"size32_strict_pooled_T{horizon}"] = c["value"]
        row[f"size32_strict_pooled_T{horizon}_numerator"] = c["numerator"]
        row[f"size32_strict_pooled_T{horizon}_denominator"] = c["denominator"]
    for size in ("32", "64"):
        data = saved["sizes"][size]
        transition = data["transitions"]["all_changed"]["64_to_256"]
        retention = transition["pooled"]["retention"]
        ever = data["ever_regressed_over_ever_correct"]
        start = data["endpoints"]["64"]["all_changed"]["pooled_coverage"]["value"]
        end = data["endpoints"]["256"]["all_changed"]["pooled_coverage"]["value"]
        frontier = data["frontier"]
        row.update({
            f"size{size}_retention64_to256": retention["value"],
            f"size{size}_retained_numerator": retention["numerator"],
            f"size{size}_retained_denominator": retention["denominator"],
            f"size{size}_coverage_gain64_to256": end - start,
            f"size{size}_coverage_T64": start,
            f"size{size}_coverage_T256": end,
            f"size{size}_ever_regressed_ratio": ever["value"],
            f"size{size}_ever_regressed_numerator": ever["numerator"],
            f"size{size}_ever_regressed_denominator": ever["denominator"],
            f"size{size}_frontier_effect": frontier["mean_map_weighted_difference"],
            f"size{size}_frontier_eligible_maps": frontier["eligible_maps"],
            f"size{size}_frontier_common_strata": frontier["common_strata"],
        })
    return row


def report(summary: dict[str, Any], validation_included: bool) -> str:
    lines = [
        "# Detached warm-start development screen",
        "",
        f"Decision: **{summary['decision']}**. The run completed {summary['completed_arms']} arms × {summary['updates_per_arm']} updates in {summary['elapsed_seconds']:.3f} seconds.",
        "",
        "The independent unit is the paired initialization (four pairs, one fixed bank and batch schedule). All four baseline arms replayed their historical six-endpoint controls; the known seed-4 baseline passed the fresh full phenotype. Baseline passed 1/4 and warm-start 0/4, so the prespecified positive development criterion was not met.",
        "",
        "## Primary gates",
        "",
        "| Seed | Baseline replay | Baseline reach | Hold | Dynamics | Frontier | Full | Warm-start reach | Hold | Dynamics | Frontier | Full |",
        "|---:|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    arms = {row["name"]: row for row in summary["arms"]}
    for seed in SEEDS:
        base = arms[f"baseline_seed{seed}"]
        warm = arms[f"warmstart_seed{seed}"]
        lines.append(
            f"| {seed} | {base['historical_replay']} | {base['reach']} | {base['hold']} | {base['dynamics']} | {base['frontier']} | {base['full_phenotype']} | "
            f"{warm['reach']} | {warm['hold']} | {warm['dynamics']} | {warm['frontier']} | {warm['full_phenotype']} |"
        )
    lines += [
        "",
        "## Partial metrics",
        "",
        "These secondary values retain the fixed per-arm measurements even when the conjunction fails. Strict pooled coverage uses changed pixels with `16 < BFS distance < 32`. Retention, growth, and regression use all changed pixels. Rates are pooled counts; the complete map-level profiles and every frozen gate value are in the per-arm summaries.",
        "",
        "| Arm | Strict pooled T64 | T128 | T256 | Size32 retention 64→256 | gain 64→256 | regressed / ever correct | Size64 retention 64→256 | gain 64→256 | regressed / ever correct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["arms"]:
        lines.append(
            f"| {row['name']} | {row['size32_strict_pooled_T64']:.4f} | {row['size32_strict_pooled_T128']:.4f} | {row['size32_strict_pooled_T256']:.4f} | "
            f"{row['size32_retention64_to256']:.4f} ({row['size32_retained_numerator']}/{row['size32_retained_denominator']}) | {row['size32_coverage_gain64_to256']:+.4f} | "
            f"{row['size32_ever_regressed_ratio']:.4f} ({row['size32_ever_regressed_numerator']}/{row['size32_ever_regressed_denominator']}) | "
            f"{row['size64_retention64_to256']:.4f} ({row['size64_retained_numerator']}/{row['size64_retained_denominator']}) | {row['size64_coverage_gain64_to256']:+.4f} | "
            f"{row['size64_ever_regressed_ratio']:.4f} ({row['size64_ever_regressed_numerator']}/{row['size64_ever_regressed_denominator']}) |"
        )
    lines += [
        "",
        "Partial metrics vary across paired seeds. Seed-2 warm-start improves T64 size32 open-grid BA (0.569/0.589 to 0.900/0.862), size32 retention (0.000 to 0.888), and size32 coverage gain (-0.040 to +0.083), while still missing the full gate. Seed-4 warm-start moves in the other direction: strict pooled T64 is 0.851 baseline versus 0.470 warm-start; retention is 1.000/0.969 versus 0.041/0.051 at sizes 32/64; regression is 0.030/0.068 versus 0.958/0.955. This screen does not support a uniformly worse or uniformly better claim. It tests one detached prefix-age exposure recipe on one fixed bank and schedule; it does not establish a phase transition, continuation closure, causal handoff, or general rejection of warm-start training.",
        "",
        "## Frozen gate definitions",
        "",
        "Reach: size32/T64 strict paired map mean and pooled coverage ≥0.80, with original and source-flipped open-grid balanced accuracy ≥0.85. Hold: size32/T128 and T256 drops from T64 ≤0.03 balanced accuracy and ≤0.05 strict mean and pooled coverage. Dynamics: at each size, all-changed T64→T256 retention ≥0.95, pooled coverage gain ≥0.05, and ever-regressed / ever-correct ≤0.15. Frontier: at each size, exact-map/time/BFS-distance matched effect ≥0.05, at least 16 eligible maps and 100 common strata. The complete gate values and individual checks are included in each arm summary.",
        "",
        "All stage diagnostics are exploratory observations at updates 0, 100, 200, and 300 on 16 held-out size32 maps. Integer acquisition/destruction rows preserve their denominators and the identity `correct_new - correct_old = acquired - destroyed`; stage snapshots are not additional independent runs.",
        "",
        "## Reading order",
        "",
        "1. This report and [compact aggregate](summary.json).",
        "2. [Frozen protocol](../../new/warmstart/PROTOCOL.md), [configuration](config.json), and [source bindings](source_bindings.json).",
        "3. Per-arm records under `raw/`, complete gate summaries under `training/arms/`, and matched strata under `frontier/`.",
        "4. [Stage snapshots](stage/stage_summaries.json) and [integer turnover rows](stage/turnover.csv).",
        "5. [Provenance and excluded-artifact hashes](provenance.json), [public arithmetic validation](publication_validation.json)" + (", and [full local validation](validation/local_validation.json)." if validation_included else "."),
        "",
        "Checkpoints, NPZ traces, source snapshots, full diagnostic JSON traces, and private execution receipts remain excluded. Their available hashes are recorded in provenance.json.",
    ]
    return "\n".join(lines) + "\n"


def reproduction(validation_included: bool) -> str:
    tail = " The full sanitized local validation receipt is available at [validation/local_validation.json](validation/local_validation.json)." if validation_included else " A full sanitized local validation receipt will be added when available."
    return f"""# Reproduction and public verification

From the repository root, run the public CPU arithmetic and hash checks:

```powershell
python tools/export_warmstart.py --verify-only
python new/warmstart/check.py --out analyses/warmstart_cpu01.json
```

The first command uses only the Python standard library. It checks the publication hashes, all eight complete phenotype gate summaries, the matched-frontier count arithmetic, the integer turnover identity, and the sanitized public tree. It performs no training, model inference, CUDA work, or GPU detection.

The second command is the frozen focused CPU qualification. It checks the unchanged historical training path, detached-prefix gradient behavior, sticky prefix finiteness, and the diagnostic arithmetic fixtures. Use a new output path that does not already exist.

## Fresh-clone references and GPU rerun

The required historical references are already public in `evidence/streaming_carry_init2345/`: its manifest, shared schedule, and `stream_K8_seed2.json` through `stream_K8_seed5.json`. The frozen warm-start runner reads these public files directly and verifies them through `STREAMING_CARRY_PUBLICATION_MANIFEST.json`; no private run archive or checkpoint restoration is required.

After the CPU qualification, a new formal screen can be run with fresh output names:

```powershell
python new/warmstart/run.py --preflight --qualification analyses/warmstart_cpu01.json --out runs/warmstart_preflight_new
python new/warmstart/run.py --out runs/warmstart_formal_new --preflight-dir runs/warmstart_preflight_new
```

The preflight estimate is informational under `{PROTOCOL_ID}`; the amended protocol has no runtime qualification limit or watchdog. Its gradient, replay, memory, source, and finite-state gates remain fixed. The formal run uses the frozen four initialization pairs and the same fixed training bank, batch schedule, and sampled prefix ages. The in-repo [protocol](../../new/warmstart/PROTOCOL.md) defines the full recipe and stop criteria.

## What this public package reproduces

The public tool verifies saved arithmetic and binding hashes. It does not reconstruct training or inference from excluded model weights and NPZ traces. Full local validator outcomes, when included, are CPU verification of saved records; they do not repeat the historical GPU replay or the 8-arm training.{tail}
"""


def protocol_amendment() -> str:
    return """# Protocol amendment and execution record

Protocol ID: `detached_warmstart_v1_nocap`.

The initial CPU01 and preflight01 records remain immutable. Preflight01 failed its former 2,400-second launch estimate gate: its estimate was 2,451.034 seconds, while its gradient-entry and memory gates passed. Before the formal run, the user removed that runtime cap. The amended CPU02 and preflight02 qualification passed; the time estimate remained informational and neither a runtime deadline nor a termination watchdog was enabled.

The runtime qualification rule changed before the formal run. The frozen implementation also contains a synchronization-only prefix-finiteness fix: device failure flags stay sticky across each eight-step window and host checks occur at window boundaries and at the final prefix step. CPU02 checks exact historical equivalence and a NaN/recovery fixture; the change preserves numerical states, gradients, and transient-error detection. The eight-arm recipe, training-state intervention, fixed data and batch schedules, four paired initializations, endpoint, full-phenotype thresholds, and stop conditions remained fixed. The completed formal run used 300 updates per arm. All four historical baseline replays passed. The result is `DEVELOPMENT_NOT_QUALIFIED` (baseline 1/4, warm-start 0/4); the failed old preflight is not a scientific treatment result.

See the exact frozen [protocol](../../new/warmstart/PROTOCOL.md), `config.json`, `source_bindings.json`, and hash-bound [provenance](provenance.json). This result is conditional on one bank and one batch schedule. It does not identify a phase transition, continuation closure, unique handoff mechanism, or general warm-start failure.
"""


def stage_snapshot(name: str, arm: dict[str, Any], update: int, diagnostic: dict[str, Any], source_sha: str) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    for step in (32, 64, 128):
        metrics = diagnostic["endpoint_metrics"][str(step)]
        strict = metrics["paired_strict_16_32"]
        endpoints[str(step)] = {
            "paired_strict_mean": strict["mean_across_maps"],
            "paired_strict_eligible_maps": strict["eligible_maps"],
            "paired_strict_correct": strict["pooled_correct"],
            "paired_strict_pixels": strict["pooled_pixels"],
            "paired_strict_pooled": strict["pooled_accuracy"],
            "original_open_balanced_accuracy": metrics["whole_open_balanced_accuracy_original"]["mean_across_maps"],
            "flipped_open_balanced_accuracy": metrics["whole_open_balanced_accuracy_flipped"]["mean_across_maps"],
            "original_open_eligible_maps": metrics["whole_open_balanced_accuracy_original"]["eligible_maps"],
            "flipped_open_eligible_maps": metrics["whole_open_balanced_accuracy_flipped"]["eligible_maps"],
            "state_rms": diagnostic["state_rms"][str(step)],
        }
    return {
        "arm": name,
        "seed": int(arm["seed"]),
        "variant": arm["variant"],
        "training_update": update,
        "checkpoint_parameter_sha256": arm["checkpoints"][str(update)]["parameter_sha256"],
        "diagnostic_sha256": source_sha,
        "num_maps": diagnostic["num_maps"],
        "map_shape": diagnostic["map_shape"],
        "changed_pixels_total": sum(diagnostic["changed_pixels_per_map"]),
        "endpoints": endpoints,
        "survival_64_to_128": {
            "multi_step_equal_map_mean": diagnostic["survival"]["multi_step_survival"]["equal_map_mean"],
            "multi_step_eligible_maps": diagnostic["survival"]["multi_step_survival"]["equal_map_eligible_maps"],
            "multi_step_pooled_correct": diagnostic["survival"]["multi_step_survival"]["pooled_correct_at_start_and_every_step"],
            "multi_step_pooled_start": diagnostic["survival"]["multi_step_survival"]["pooled_correct_at_start"],
            "multi_step_pooled_rate": diagnostic["survival"]["multi_step_survival"]["pooled_rate"],
            "endpoint_retention_equal_map_mean": diagnostic["survival"]["endpoint_retention"]["equal_map_mean"],
            "endpoint_retention_eligible_maps": diagnostic["survival"]["endpoint_retention"]["equal_map_eligible_maps"],
            "endpoint_retained": diagnostic["survival"]["endpoint_retention"]["pooled_correct_at_start_and_endpoint"],
            "endpoint_start": diagnostic["survival"]["endpoint_retention"]["pooled_correct_at_start"],
            "endpoint_pooled_rate": diagnostic["survival"]["endpoint_retention"]["pooled_rate"],
        },
        "coverage_gain_64_to_128": diagnostic["coverage_gain"],
        "relapse": {
            "ever_correct": diagnostic["relapse"]["ever_correct"],
            "relapsed": diagnostic["relapse"]["relapsed"],
            "pooled_rate": diagnostic["relapse"]["equal_map_rate"]["pooled_rate"],
            "equal_map_mean": diagnostic["relapse"]["equal_map_rate"]["equal_map_mean"],
            "eligible_maps": diagnostic["relapse"]["equal_map_rate"]["equal_map_eligible_maps"],
        },
    }


def turnover_row(name: str, arm: dict[str, Any], update: int, step: dict[str, Any]) -> dict[str, Any]:
    q = step["q"]
    g, d = step["g_acquisition_rate"], step["d_destruction_rate"]
    return {
        "arm": name, "seed": int(arm["seed"]), "variant": arm["variant"],
        "training_update": update, "t": int(step["t"]),
        "correct": int(step["correct"]), "changed_pixels": int(step["changed_pixels"]),
        "q_equal_map_mean": q["equal_map_mean"], "q_eligible_maps": q["equal_map_eligible_maps"],
        "q_pooled_correct": q["pooled_correct"], "q_pooled_pixels": q["pooled_pixels"],
        "q_pooled_rate": q["pooled_accuracy"],
        "acquired": step["acquired"], "wrong_old": step["wrong_old"],
        "g_equal_map_mean": None if g is None else g["equal_map_mean"],
        "g_eligible_maps": None if g is None else g["equal_map_eligible_maps"],
        "g_pooled_numerator": None if g is None else g["pooled_numerator"],
        "g_pooled_denominator": None if g is None else g["pooled_denominator"],
        "g_pooled_rate": None if g is None else g["pooled_rate"],
        "destroyed": step["destroyed"], "correct_old": step["correct_old"],
        "d_equal_map_mean": None if d is None else d["equal_map_mean"],
        "d_eligible_maps": None if d is None else d["equal_map_eligible_maps"],
        "d_pooled_numerator": None if d is None else d["pooled_numerator"],
        "d_pooled_denominator": None if d is None else d["pooled_denominator"],
        "d_pooled_rate": None if d is None else d["pooled_rate"],
        "delta_correct": step["delta_correct"], "accounting_error": step["accounting_error"],
    }


def write_turnover(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TURNOVER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def export_frontier(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(source) == sha(target)


def check_frontier(path: Path, saved: dict[str, Any]) -> int:
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = {}
    seen: set[tuple[int, int, int, int, int]] = set()
    rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == FRONTIER_COLUMNS
        for row in reader:
            size = int(row["size"])
            map_index = int(row["map_index"])
            start = int(row["start_step"])
            end = int(row["end_step"])
            distance = int(row["distance"])
            key = (size, map_index, start, end, distance)
            assert key not in seen and size in (32, 64) and map_index in range(32)
            assert start in range(0, 249, 8) and end == start + 1 and distance >= 0
            seen.add(key)
            nf, af = int(row["frontier_opportunities"]), int(row["frontier_acquired"])
            nn, an = int(row["nonfrontier_opportunities"]), int(row["nonfrontier_acquired"])
            assert nf > 0 and nn > 0 and 0 <= af <= nf and 0 <= an <= nn
            weight, fr, nr = nf * nn / (nf + nn), af / nf, an / nn
            difference = fr - nr
            close(float(row["matched_weight"]), weight)
            close(float(row["frontier_rate"]), fr)
            close(float(row["nonfrontier_rate"]), nr)
            close(float(row["rate_difference"]), difference)
            grouped.setdefault((str(size), map_index), []).append((weight, difference))
            rows += 1
    for size in ("32", "64"):
        frontier = saved["sizes"][size]["frontier"]
        effects = []
        for map_row in frontier["per_map"]:
            values = grouped.get((size, int(map_row["map_index"])), [])
            assert len(values) == int(map_row["common_strata"])
            effect = sum(weight * delta for weight, delta in values) / sum(weight for weight, _ in values) if values else None
            close(effect, map_row["weighted_difference"])
            if effect is not None:
                effects.append(effect)
        assert len(effects) == frontier["eligible_maps"]
        close(sum(effects) / len(effects) if effects else None, frontier["mean_map_weighted_difference"])
        assert sum(len(values) for (observed_size, _), values in grouped.items() if observed_size == size) == frontier["common_strata"]
    return rows


def check_turnover(path: Path) -> int:
    count = 0
    seen: set[tuple[str, int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == TURNOVER_COLUMNS
        for row in reader:
            t = int(row["t"])
            update = int(row["training_update"])
            assert update in STAGES and t in range(TRACE_STEPS + 1)
            key = (row["arm"], update, t)
            assert key not in seen
            seen.add(key)
            correct, changed = int(row["correct"]), int(row["changed_pixels"])
            assert 0 <= correct <= changed and int(row["q_pooled_correct"]) == correct
            assert int(row["q_pooled_pixels"]) == changed
            close(float(row["q_pooled_rate"]), correct / changed if changed else None)
            if t == 0:
                assert row["acquired"] == row["destroyed"] == row["delta_correct"] == ""
                assert row["accounting_error"] == ""
            else:
                acquired, destroyed, delta = int(row["acquired"]), int(row["destroyed"]), int(row["delta_correct"])
                assert int(row["accounting_error"]) == 0
                assert delta == acquired - destroyed
                assert int(row["wrong_old"]) >= acquired and int(row["correct_old"]) >= destroyed
                assert int(row["wrong_old"]) + int(row["correct_old"]) == changed
                assert correct - int(row["correct_old"]) == delta
                for prefix, numerator, denominator in (
                    ("g", int(row["g_pooled_numerator"]), int(row["g_pooled_denominator"])),
                    ("d", int(row["d_pooled_numerator"]), int(row["d_pooled_denominator"])),
                ):
                    assert numerator >= 0 and numerator <= denominator
                    if denominator:
                        close(float(row[f"{prefix}_pooled_rate"]), numerator / denominator)
                    else:
                        assert numerator == 0 and row[f"{prefix}_pooled_rate"] == ""
            count += 1
    assert count == len(ARMS) * len(STAGES) * (TRACE_STEPS + 1), count
    return count


def check_summary(out: Path) -> dict[str, Any]:
    summary = read(out / "summary.json")
    aggregate = read(out / "training/aggregate.json")
    assert summary["protocol"] == PROTOCOL_ID
    assert aggregate["complete"] is True and aggregate["completed_arms"] == 8
    assert aggregate["updates_per_arm"] == 300 and aggregate["decision"] == "DEVELOPMENT_NOT_QUALIFIED"
    assert len(summary["arms"]) == 8
    assert summary["baseline_passes"] == 1 and summary["warmstart_passes"] == 0
    assert summary["paired_full_pass_change"] == -1
    source_rows = {row["name"]: row for row in summary["arms"]}
    assert set(source_rows) == set(ARMS)
    total_replays = 0
    expected_check_names = {
        *(f"hold_size32_T{h}_{metric}" for h in (128, 256) for metric in (
            "flipped_ba_drop", "original_ba_drop", "strict_mean_map_coverage_drop", "strict_pooled_coverage_drop")),
        *(f"reach_size32_T64_{metric}" for metric in (
            "flipped_ba", "original_ba", "strict_map_mean", "strict_pooled")),
        *(f"size{size}_{metric}" for size in (32, 64) for metric in (
            "all_changed_coverage_gain64_to256", "all_changed_retention64_to256",
            "common_frontier_strata", "eligible_frontier_maps", "ever_regressed_over_ever_correct",
            "matched_frontier_effect")),
    }
    for name in ARMS:
        saved = read(out / f"training/arms/{name}_summary.json")
        row = source_rows[name]
        checks = saved["phenotype_gate"]["checks"]
        assert set(checks) == expected_check_names
        for check_name, check in checks.items():
            threshold = str(check["threshold"])
            match = re.fullmatch(r"(>=|<=)\s*([0-9]+(?:\.[0-9]+)?)", threshold)
            assert match is not None, (check_name, threshold)
            value = check["value"]
            expected_pass = False if value is None else (
                float(value) >= float(match.group(2)) if match.group(1) == ">="
                else float(value) <= float(match.group(2))
            )
            assert bool(check["pass"]) is expected_pass, (name, check_name, value, threshold)
        assert bool(saved["phenotype_gate"]["pass"]) is all(bool(check["pass"]) for check in checks.values())
        assert components(saved) == {key: row[key] for key in components(saved)}
        assert row["full_phenotype"] == saved["phenotype_gate"]["pass"]
        assert row["failed_checks"] == [key for key, check in saved["phenotype_gate"]["checks"].items() if not check["pass"]]
        assert row["checks"] == saved["phenotype_gate"]["checks"]
        for size in ("32", "64"):
            # Make sure the published strict pooled endpoints reproduce the full summary.
            for horizon in (64, 128, 256):
                c = coverage(saved["sizes"]["32"], horizon)
                close(row[f"size32_strict_pooled_T{horizon}"], c["value"])
                assert row[f"size32_strict_pooled_T{horizon}_numerator"] == c["numerator"]
                assert row[f"size32_strict_pooled_T{horizon}_denominator"] == c["denominator"]
            data = saved["sizes"][size]
            transition = data["transitions"]["all_changed"]["64_to_256"]["pooled"]["retention"]
            close(row[f"size{size}_retention64_to256"], transition["value"])
            assert row[f"size{size}_retained_numerator"] == transition["numerator"]
            assert row[f"size{size}_retained_denominator"] == transition["denominator"]
            close(row[f"size{size}_ever_regressed_ratio"], data["ever_regressed_over_ever_correct"]["value"])
            close(row[f"size{size}_frontier_effect"], data["frontier"]["mean_map_weighted_difference"])
        total_replays += row["historical_replay"] == "PASS"
        csv_path = out / f"frontier/{name}.csv"
        assert check_frontier(csv_path, saved) > 0
    assert total_replays == 4
    assert sum(source_rows[f"baseline_seed{seed}"]["full_phenotype"] for seed in SEEDS) == 1
    assert sum(source_rows[f"warmstart_seed{seed}"]["full_phenotype"] for seed in SEEDS) == 0
    for seed in SEEDS:
        pair = summary["pairs"][str(seed)]
        assert pair["baseline"] == source_rows[f"baseline_seed{seed}"]["full_phenotype"]
        assert pair["warmstart"] == source_rows[f"warmstart_seed{seed}"]["full_phenotype"]
    return summary


def assert_public_tree(out: Path) -> None:
    for path in out.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise AssertionError(f"Unexpected binary public artifact: {path.relative_to(out).as_posix()}")
        assert not PRIVATE_TEXT.search(data), f"Private machine/account/path text in {path.relative_to(out).as_posix()}"
        if path.suffix == ".json":
            assert public(json.loads(data)) == json.loads(data), f"Unredacted private field in {path.relative_to(out).as_posix()}"


def verify(out: Path, bind: bool = True) -> dict[str, Any]:
    summary = check_summary(out)
    turnover_rows = check_turnover(out / "stage/turnover.csv")
    stage_summary = read(out / "stage/stage_summaries.json")
    assert len(stage_summary["stages"]) == len(ARMS) * len(STAGES)
    source_bindings = read(out / "source_bindings.json")
    assert len(source_bindings["source_sha256"]) == 70
    for name, expected in source_bindings["source_sha256"].items():
        assert sha(ROOT / name) == expected, f"Source binding drift: {name}"
    assert_public_tree(out)
    if bind:
        publication = read(PUBLICATION)
        assert publication["protocol"] == PROTOCOL_ID
        assert publication["source_sha256"] == source_bindings["source_sha256"]
        for name, expected in publication["publication_tool_sha256"].items():
            assert sha(ROOT / name) == expected, f"Publication tool drift: {name}"
        for name, expected in publication["published_evidence_sha256"].items():
            assert sha(ROOT / name) == expected, f"Published evidence drift: {name}"
    return {
        "status": "PASS",
        "training": False,
        "gpu_execution": False,
        "arms": len(summary["arms"]),
        "historical_replays": 4,
        "frontier_csvs_checked": 8,
        "stage_snapshots": len(stage_summary["stages"]),
        "integer_turnover_rows": turnover_rows,
        "source_bindings": len(source_bindings["source_sha256"]),
        "decision": summary["decision"],
        "scope": "Public arithmetic, source hashes, and staged evidence bindings; no model inference or training.",
    }


def copy_sanitized(source: Path, target: Path, original_hashes: dict[str, str], output_map: dict[str, Any]) -> None:
    relative_source = source.relative_to(ROOT).as_posix()
    relative_target = target.relative_to(OUT).as_posix()
    original_hashes[relative_source] = sha(source)
    doc = read(source)
    cleaned = public(doc)
    if cleaned == doc:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        output_map[relative_target] = {"kind": "byte_identical_copy", "source": relative_source}
    else:
        write_json(target, cleaned)
        output_map[relative_target] = {"kind": "sanitized_copy", "source": relative_source}


def export() -> dict[str, Any]:
    assert not OUT.exists(), f"Publication output already exists: {OUT.name}"
    validation_included = LOCAL_VALIDATION.exists()
    if validation_included:
        assert read(LOCAL_VALIDATION).get("status") == "PASS", "Local full validation is present but did not pass"
    run_manifest = read(RUN / "manifest.json")
    run_status = read(RUN / "status.json")
    aggregate = read(RUN / "aggregate.json")
    assert run_manifest["protocol"] == PROTOCOL_ID
    assert run_status["status"] == "COMPLETE" and run_status["completed_arms"] == 8
    assert aggregate["complete"] and aggregate["completed_arms"] == 8
    assert aggregate["decision"] == "DEVELOPMENT_NOT_QUALIFIED"
    assert aggregate["passes"] == {"baseline": 1, "warmstart": 0}
    assert aggregate["historical_controls_qualified"] and aggregate["known_seed4_fresh_qualified"]
    assert len(run_manifest["source_sha256"]) == 70
    for name, expected in run_manifest["source_sha256"].items():
        assert sha(ROOT / name) == expected, f"Frozen source changed since run: {name}"
        snapshot = RUN / "source" / name
        assert sha(snapshot) == expected, f"Run source snapshot hash mismatch: {name}"

    original_hashes: dict[str, str] = {}
    output_map: dict[str, Any] = {}
    source_binding = {
        "protocol": PROTOCOL_ID,
        "review_base": run_manifest["git_base"],
        "source_sha256": run_manifest["source_sha256"],
        "source_snapshot_policy": "The 70 executed source snapshots are excluded from this public package; their hashes bind to the repository files above and the local validation report.",
    }
    write_json(OUT / "source_bindings.json", source_binding)

    config_keys = (
        "protocol", "expected_formal_arms", "updates_per_arm", "maximum_seconds",
        "runtime_limit_enforced", "watchdog_enabled", "gradient_horizon",
        "trained_suffix_steps", "loss_times_suffix", "warm_examples", "fresh_examples",
        "prefix_age_choices", "age_schedule_sha256", "age_unit", "stage_updates",
        "stage_diagnostic_seed", "stage_diagnostic_maps", "stage_diagnostic_data_sha256",
        "stage_diagnostic_steps", "train_data_seed", "train_data_sha256", "schedule_seed",
        "schedule_sha256", "new_evaluation_seeds", "maps_per_size",
        "new_evaluation_data_sha256", "historical_evaluation_data_sha256",
        "optimizer", "backend", "torch", "numpy", "threads",
    )
    config = {key: run_manifest[key] for key in config_keys}
    config.update({
        "independent_unit": "paired initialization",
        "n": 4,
        "arms": [{"seed": seed, "variants": list(VARIANTS)} for seed in SEEDS],
        "stage_updates": list(STAGES),
        "protocol_note": "The runtime estimate is informational; the formal run used no deadline or watchdog.",
    })
    write_json(OUT / "config.json", config)

    training_out = OUT / "training"
    training_out.mkdir(parents=True, exist_ok=True)
    copy_sanitized(RUN / "aggregate.json", training_out / "aggregate.json", original_hashes, output_map)
    copy_sanitized(RUN / "status.json", training_out / "completion.json", original_hashes, output_map)
    copy_sanitized(RUN / "manifest.json", training_out / "execution_manifest.json", original_hashes, output_map)
    for filename in ("schedule.json", "ages.json"):
        src = RUN / filename
        dst = training_out / filename
        original_hashes[src.relative_to(ROOT).as_posix()] = sha(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        output_map[dst.relative_to(OUT).as_posix()] = {"kind": "byte_identical_copy", "source": src.relative_to(ROOT).as_posix()}

    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    (training_out / "arms").mkdir(parents=True, exist_ok=True)
    (OUT / "frontier").mkdir(parents=True, exist_ok=True)
    arm_rows = []
    stage_rows = []
    turnover_rows = []
    arm_records: dict[str, dict[str, Any]] = {}
    for name in ARMS:
        arm = read(RUN / f"{name}.json")
        saved = read(RUN / f"{name}_summary.json")
        assert arm["status"] == "COMPLETE" and arm["completed_updates"] == 300
        assert saved["model_name"] == name and saved["phenotype_gate"]["pass"] == aggregate["phenotype_pass"][name]
        arm_records[name] = arm
        arm_rows.append(arm_row(name, arm, saved))
        copy_sanitized(RUN / f"{name}.json", OUT / f"raw/{name}.json", original_hashes, output_map)
        summary_source = RUN / f"{name}_summary.json"
        summary_target = training_out / f"arms/{name}_summary.json"
        copy_sanitized(summary_source, summary_target, original_hashes, output_map)
        export_frontier(RUN / f"{name}_matched_frontier_strata.csv", OUT / f"frontier/{name}.csv")
        original_hashes[(RUN / f"{name}_matched_frontier_strata.csv").relative_to(ROOT).as_posix()] = sha(RUN / f"{name}_matched_frontier_strata.csv")
        output_map[f"frontier/{name}.csv"] = {"kind": "byte_identical_copy", "source": (RUN / f"{name}_matched_frontier_strata.csv").relative_to(ROOT).as_posix()}
        for update in STAGES:
            filename = f"{name}_u{update:03}_diagnostic.json"
            diagnostic_path = RUN / filename
            diagnostic = read(diagnostic_path)
            assert diagnostic["arm"] == name and diagnostic["training_update"] == update
            assert diagnostic["scientific_endpoint"] is False and diagnostic["exploratory"] is True
            diagnostic_hash = sha(diagnostic_path)
            original_hashes[diagnostic_path.relative_to(ROOT).as_posix()] = diagnostic_hash
            stage_rows.append(stage_snapshot(name, arm, update, diagnostic, diagnostic_hash))
            turnover_rows.extend(turnover_row(name, arm, update, step) for step in diagnostic["steps"])

    write_json(OUT / "summary.json", {
        "protocol": PROTOCOL_ID,
        "review_base": run_manifest["git_base"],
        "execution_status": "COMPLETE",
        "elapsed_seconds": run_status["elapsed_seconds"],
        "expected_arms": 8,
        "completed_arms": aggregate["completed_arms"],
        "updates_per_arm": aggregate["updates_per_arm"],
        "independent_unit": aggregate["independent_unit"],
        "n": aggregate["n"],
        "decision": aggregate["decision"],
        "historical_controls_qualified": aggregate["historical_controls_qualified"],
        "known_seed4_fresh_qualified": aggregate["known_seed4_fresh_qualified"],
        "baseline_passes": aggregate["passes"]["baseline"],
        "warmstart_passes": aggregate["passes"]["warmstart"],
        "paired_full_pass_change": aggregate["paired_full_pass_change"],
        "pairs": aggregate["pairs"],
        "arms": arm_rows,
        "scope": aggregate["scope"],
    })
    write_json(OUT / "stage/stage_summaries.json", {
        "schema_version": "warmstart-stage-summary-v1",
        "scope": "Exploratory diagnostic snapshots; training_update is a stage, and maps/time steps are descriptive rather than independent units.",
        "training_updates": list(STAGES),
        "stages": stage_rows,
    })
    write_turnover(OUT / "stage/turnover.csv", turnover_rows)

    # Preserve compact hashes and status-only records for both generations of preflight.
    old_preflight_aggregate = read(PREFLIGHT_OLD / "aggregate.json")
    assert old_preflight_aggregate["decision"] == "PREFLIGHT_FAILED"
    assert old_preflight_aggregate["estimated_formal_seconds"] > 2400
    old_preflight = {
        "decision": old_preflight_aggregate["decision"],
        "estimated_formal_seconds": old_preflight_aggregate["estimated_formal_seconds"],
        "former_maximum_seconds": 2400,
        "time_estimate_was_launch_gate": True,
        "gradient_entry_gate": old_preflight_aggregate["gradient_entry_gate"],
        "memory_gate": old_preflight_aggregate["memory_gate"],
    }
    new_preflight_aggregate = read(PREFLIGHT / "aggregate.json")
    assert new_preflight_aggregate["decision"] == "PREFLIGHT_PASSED"
    new_preflight = {
        "decision": new_preflight_aggregate["decision"],
        "estimated_formal_seconds": new_preflight_aggregate["estimated_formal_seconds"],
        "maximum_seconds": None,
        "time_estimate_is_launch_gate": False,
        "runtime_limit_enforced": False,
        "gradient_entry_gate": new_preflight_aggregate["gradient_entry_gate"],
        "memory_gate": new_preflight_aggregate["memory_gate"],
    }
    write_json(OUT / "validation/preflight_history.json", {
        "preflight01": old_preflight,
        "cpu02_status": read(CPU)["status"],
        "cpu02_sha256": sha(CPU),
        "preflight02": new_preflight,
        "preflight02_input_sha256": {
            name: sha(PREFLIGHT / name) for name in ("status.json", "manifest.json", "aggregate.json")
        },
    })
    copy_sanitized(CPU, OUT / "validation/cpu02.json", original_hashes, output_map)
    for path in (PREFLIGHT_OLD / "status.json", PREFLIGHT_OLD / "manifest.json", PREFLIGHT_OLD / "aggregate.json",
                 PREFLIGHT / "status.json", PREFLIGHT / "manifest.json", PREFLIGHT / "aggregate.json", CPU_OLD):
        if path.exists():
            original_hashes[path.relative_to(ROOT).as_posix()] = sha(path)

    if validation_included:
        local_doc = read(LOCAL_VALIDATION)
        copy_sanitized(LOCAL_VALIDATION, OUT / "validation/local_validation.json", original_hashes, output_map)

    (OUT / "PROTOCOL_AMENDMENT.md").write_text(protocol_amendment(), encoding="utf-8")
    (OUT / "RESULTS.md").write_text(report(read(OUT / "summary.json"), validation_included), encoding="utf-8")
    (OUT / "REPRODUCTION.md").write_text(reproduction(validation_included), encoding="utf-8")

    # Bind every original run file by relative path, including excluded checkpoints,
    # NPZ traces, source snapshots, and full diagnostic JSON records.
    all_run_hashes = {
        path.relative_to(ROOT).as_posix(): sha(path)
        for path in RUN.rglob("*") if path.is_file()
    }
    checkpoint_hashes = {
        relative: digest for relative, digest in all_run_hashes.items()
        if relative.lower().endswith((".pt", ".npz")) or "/source/" in relative
    }
    provenance = {
        "review_base": run_manifest["git_base"],
        "protocol": PROTOCOL_ID,
        "formal_run": {"status": "COMPLETE", "completed_arms": 8, "updates_per_arm": 300,
                       "elapsed_seconds": run_status["elapsed_seconds"]},
        "original_run_manifest_sha256": all_run_hashes["runs/warmstart_20261003_paired01/manifest.json"],
        "original_run_status_sha256": all_run_hashes["runs/warmstart_20261003_paired01/status.json"],
        "original_run_artifact_sha256": all_run_hashes,
        "excluded_checkpoint_npz_and_source_snapshot_sha256": checkpoint_hashes,
        "preflight_history_sha256": {
            "preflight01": {name: sha(PREFLIGHT_OLD / name) for name in ("status.json", "manifest.json", "aggregate.json")},
            "cpu01": sha(CPU_OLD) if CPU_OLD.exists() else None,
            "cpu02": sha(CPU),
            "preflight02": {name: sha(PREFLIGHT / name) for name in ("status.json", "manifest.json", "aggregate.json")},
        },
        "source_bindings": run_manifest["source_sha256"],
        "publication_tools_sha256": {
            "tools/export_warmstart.py": sha(ROOT / "tools/export_warmstart.py"),
            "tools/validate_warmstart.py": sha(VALIDATOR),
        },
        "public_output_map": output_map,
        "sanitization": "Private manifest fields, process/account/machine identity, absolute paths, and private receipts are excluded. Raw arm records are recursively sanitized. Full diagnostics are reduced to stage snapshots and integer turnover rows. Checkpoint, NPZ, source snapshot, original run manifest/status, and full diagnostic source hashes are preserved here.",
        "excluded_classes": ["model checkpoints", "NPZ evaluation traces", "70 frozen source snapshots", "full per-step diagnostic JSON", "host/process/GPU execution receipt"],
        "validation_scope": "No new training or inference for publication. The public verifier checks saved summaries, frontier arithmetic, turnover identities, source hashes and output bindings.",
    }
    write_json(OUT / "provenance.json", provenance)

    # Check the newly assembled tree before writing the self-describing validation record.
    result = verify(OUT, bind=False)
    write_json(OUT / "publication_validation.json", result)
    assert_public_tree(OUT)

    published = {
        path.relative_to(ROOT).as_posix(): sha(path)
        for path in sorted(OUT.rglob("*")) if path.is_file()
    }
    write_json(PUBLICATION, {
        "protocol": PROTOCOL_ID,
        "review_base": run_manifest["git_base"],
        "execution_status": "COMPLETE",
        "completed_arms": 8,
        "updates_per_arm": 300,
        "baseline_passes": 1,
        "warmstart_passes": 0,
        "historical_replays_passed": 4,
        "decision": "DEVELOPMENT_NOT_QUALIFIED",
        "source_sha256": run_manifest["source_sha256"],
        "publication_tool_sha256": {
            "tools/export_warmstart.py": sha(ROOT / "tools/export_warmstart.py"),
            "tools/validate_warmstart.py": sha(VALIDATOR),
        },
        "published_evidence_sha256": published,
        "local_validation_included": validation_included,
        "excluded_original_hash_count": len(all_run_hashes),
        "notes": [
            "Conditional on one fixed training bank and batch schedule; four paired initializations.",
            "The only treatment is detached on-policy prefix-state exposure; the runtime amendment changed no scientific condition.",
            "No phase transition, continuation closure, causal handoff law, or general warm-start rejection is established.",
            "No GPU training or inference was run for publication.",
        ],
    })
    return {"status": "EXPORTED", "files": len(published), "bytes": sum((ROOT / name).stat().st_size for name in published)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify published arithmetic and hashes without GPU work.")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(OUT, bind=True), indent=2))
    else:
        print(json.dumps(export(), indent=2))


if __name__ == "__main__":
    main()
