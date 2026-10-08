"""Compact paired endpoint and checkpoint reports for addressed-delta screen."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any


ARMS = ("current", "additive", "delta")
BLOCKS = tuple(range(4))
CHECKPOINTS = tuple(range(0, 301, 25))
FORMAL_UPDATE = 300
PRIMARY_CANDIDATE = "delta"
PRIMARY_REFERENCE = "additive"
SIZE_KEYS = ("32", "64")

_SCALAR_FIELDS = (
    "R_strict_pooled_T64",
    "R_strict_mean_T64",
    "S_retention64_to256",
    "S_continuous_survival64_to256",
    "retention_reference_maps",
    "retention_reference_pixels",
)
_COVERAGE_FIELDS = tuple(
    f"coverage_T{time}_{stat}"
    for time in (128, 256)
    for stat in ("pooled", "mean_map")
)
_SCORE_FIELDS = tuple(
    f"{branch}_{metric}_T{time}"
    for time in (64, 128, 256)
    for branch in ("original", "flipped")
    for metric in ("balanced_accuracy", "bce")
)
_CSV_FIELDS = _SCALAR_FIELDS + _COVERAGE_FIELDS + _SCORE_FIELDS


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    result = _number(value)
    if result is None or result != math.floor(result):
        return None
    return int(result)


def _record_update(row: dict[str, Any]) -> int | None:
    update = _integer(row.get("update"))
    return update if update is not None else _integer(row.get("checkpoint"))


def _readiness(row: Any) -> bool | None:
    if not isinstance(row, dict):
        return None
    value = row.get("joint")
    if isinstance(value, bool):
        return value
    if isinstance(value, dict) and isinstance(value.get("pass"), bool):
        return value["pass"]
    return None


def exact_two_sided_binomial_p(wins: int, losses: int) -> float:
    """Exact two-sided sign-test p-value over discordant paired blocks."""
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses cannot be negative")
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    return min(
        1.0,
        2.0 * sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant,
    )


def _index_records(
    rows: list[dict[str, Any]], *, final: bool
) -> tuple[dict[tuple[int, str] | tuple[int, str, int], dict[str, Any]], dict[str, int]]:
    indexed: dict[Any, dict[str, Any]] = {}
    seen: set[Any] = set()
    ambiguous: set[Any] = set()
    invalid = duplicates = 0
    allowed_updates = {FORMAL_UPDATE} if final else set(CHECKPOINTS)
    for row in rows:
        if not isinstance(row, dict):
            invalid += 1
            continue
        arm = row.get("arm")
        block = _integer(row.get("block"))
        update = _record_update(row)
        if (
            arm not in ARMS
            or block not in BLOCKS
            or update not in allowed_updates
            or _readiness(row) is None
        ):
            invalid += 1
            continue
        key = (block, arm) if final else (block, arm, update)
        if key in seen:
            duplicates += 1
            ambiguous.add(key)
            indexed.pop(key, None)
            continue
        seen.add(key)
        if key not in ambiguous:
            indexed[key] = row
    return indexed, {
        "input_records": len(rows),
        "observed_unique_records": len(indexed),
        "invalid_records": invalid,
        "duplicate_records": duplicates,
        "ambiguous_keys": len(ambiguous),
    }


def _global_error(execution_status: Any) -> bool:
    if not isinstance(execution_status, dict):
        return False
    error_states = {"error", "failed", "interrupted"}
    for key in ("status", "global_status", "training_status", "engineering_status"):
        value = execution_status.get(key)
        if isinstance(value, str) and value.strip().lower() in error_states:
            return True
    for key in ("global_error", "training_error", "engineering_error"):
        value = execution_status.get(key)
        if value is True or (isinstance(value, str) and value.strip()):
            return True
    return execution_status.get("ok") is False or execution_status.get("success") is False


def _size_point(row: dict[str, Any] | None, size: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    metrics = row.get("metrics")
    sizes = metrics.get("sizes") if isinstance(metrics, dict) else None
    if not isinstance(sizes, dict):
        return {}
    point = sizes.get(size)
    if point is None:
        point = sizes.get(int(size))
    return point if isinstance(point, dict) else {}


def _coverage(point: dict[str, Any], time: int, stat: str) -> float | None:
    coverage = point.get("coverage")
    if not isinstance(coverage, dict):
        return None
    time_point = coverage.get(str(time), coverage.get(time))
    if not isinstance(time_point, dict):
        return None
    return _number(time_point.get(stat))


def _endpoint_score(
    row: dict[str, Any] | None, size: str, time: int, branch: str, metric: str
) -> float | None:
    if not isinstance(row, dict):
        return None
    metrics = row.get("metrics")
    scores = metrics.get("endpoint_scores") if isinstance(metrics, dict) else None
    if not isinstance(scores, dict):
        return None
    size_scores = scores.get(size, scores.get(int(size)))
    if not isinstance(size_scores, dict):
        return None
    time_scores = size_scores.get(str(time), size_scores.get(time))
    if not isinstance(time_scores, dict):
        return None
    branch_scores = time_scores.get(branch)
    if not isinstance(branch_scores, dict):
        return None
    return _number(branch_scores.get(metric))


def _flat_metrics(row: dict[str, Any] | None, size: str) -> dict[str, float | int | None]:
    point = _size_point(row, size)
    result: dict[str, float | int | None] = {
        field: _number(point.get(field)) for field in _SCALAR_FIELDS
    }
    for field in _COVERAGE_FIELDS:
        time_key, stat = field.removeprefix("coverage_").split("_", 1)
        result[field] = _coverage(point, int(time_key[1:]), stat)
    for time in (64, 128, 256):
        for branch in ("original", "flipped"):
            for metric in ("balanced_accuracy", "bce"):
                field = f"{branch}_{metric}_T{time}"
                result[field] = _endpoint_score(row, size, time, branch, metric)
    return result


def _metric_means(
    rows: dict[tuple[int, str], dict[str, Any]], arm: str
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for size in SIZE_KEYS:
        size_result: dict[str, dict[str, float | int | None]] = {}
        for field in _CSV_FIELDS:
            values = [
                value
                for block in BLOCKS
                if (row := rows.get((block, arm))) is not None
                if (value := _number(_flat_metrics(row, size).get(field))) is not None
            ]
            size_result[field] = {
                "mean": sum(values) / len(values) if values else None,
                "observed": len(values),
                "expected": len(BLOCKS),
            }
        size_result["support_record_coverage"] = {
            "observed_blocks": sum(
                (rows.get((block, arm)) is not None)
                and _number(_flat_metrics(rows.get((block, arm)), size).get("retention_reference_maps")) is not None
                for block in BLOCKS
            ),
            "expected_blocks": len(BLOCKS),
        }
        result[size] = size_result
    return result


def _stage_diagnostics(
    stage_rows: dict[tuple[int, str, int], dict[str, Any]],
    counts: dict[str, int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        observed = sum((block, arm, update) in stage_rows for block in BLOCKS for update in CHECKPOINTS)
        complete_trajectories = 0
        ever_ready = 0
        known_nonready = 0
        indeterminate_without_pass = 0
        regressed_after_pass = 0
        for block in BLOCKS:
            path = [stage_rows.get((block, arm, update)) for update in CHECKPOINTS]
            readiness = [_readiness(row) for row in path]
            has_pass = any(value is True for value in readiness)
            has_unknown = any(value is None for value in readiness)
            if has_pass:
                ever_ready += 1
            elif has_unknown:
                indeterminate_without_pass += 1
            else:
                known_nonready += 1
            if not has_unknown:
                complete_trajectories += 1
                if has_pass:
                    first = readiness.index(True)
                    if any(value is False for value in readiness[first + 1 :]):
                        regressed_after_pass += 1
        first_pass_counts = {
            str(update): sum(
                _readiness(stage_rows.get((block, arm, update))) is True
                and not any(
                    _readiness(stage_rows.get((block, arm, earlier))) is True
                    for earlier in CHECKPOINTS
                    if earlier < update
                )
                for block in BLOCKS
            )
            for update in CHECKPOINTS
        }
        result[arm] = {
            "expected_stage_records": len(BLOCKS) * len(CHECKPOINTS),
            "observed_stage_records": observed,
            "complete_trajectories": complete_trajectories,
            "incomplete_trajectories": len(BLOCKS) - complete_trajectories,
            "ever_ready_blocks": ever_ready,
            "known_nonready_blocks": known_nonready,
            "indeterminate_without_any_pass_blocks": indeterminate_without_pass,
            "observed_first_pass_checkpoint_counts": first_pass_counts,
            "complete_trajectories_with_a_later_readiness_failure": regressed_after_pass,
        }
    return {
        "expected_stage_records": len(BLOCKS) * len(ARMS) * len(CHECKPOINTS),
        "observed_unique_stage_records": counts["observed_unique_records"],
        "record_counts": counts,
        "by_arm": result,
        "checkpoint_selection": "none; dense checkpoints are diagnostics only",
    }


def aggregate(
    final_rows: list[dict[str, Any]], dense_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate the frozen u300 endpoint and checkpoint-grid records.

    `final_rows` must contain the 12 formal u300 records. `dense_records` must
    independently contain all 156 unique records at updates 0,25,...,300.
    A final row never fills a missing checkpoint-grid row.
    """
    final_index, final_counts = _index_records(final_rows, final=True)
    stage_index, stage_counts = _index_records(dense_records, final=False)

    by_arm: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        values = [_readiness(final_index.get((block, arm))) for block in BLOCKS]
        observed = sum(value is not None for value in values)
        passes = sum(value is True for value in values)
        by_arm[arm] = {
            "expected_blocks": len(BLOCKS),
            "observed_endpoints": observed,
            "ready": passes,
            "not_ready": observed - passes,
            "missing_or_invalid": len(BLOCKS) - observed,
        }

    wins = losses = ties = paired = 0
    for block in BLOCKS:
        delta = _readiness(final_index.get((block, PRIMARY_CANDIDATE)))
        additive = _readiness(final_index.get((block, PRIMARY_REFERENCE)))
        if delta is None or additive is None:
            continue
        paired += 1
        if delta and not additive:
            wins += 1
        elif additive and not delta:
            losses += 1
        else:
            ties += 1
    net_wins = wins - losses
    p_value = exact_two_sided_binomial_p(wins, losses)

    observed_delta_ready = by_arm["delta"]["ready"]
    observed_rule_satisfied = observed_delta_ready >= 2 and net_wins >= 1
    final_expected = len(BLOCKS) * len(ARMS)
    stage_expected = len(BLOCKS) * len(ARMS) * len(CHECKPOINTS)
    final_complete = (
        len(final_index) == final_expected
        and final_counts["input_records"] == final_expected
        and final_counts["invalid_records"] == 0
        and final_counts["duplicate_records"] == 0
    )
    stage_complete = (
        len(stage_index) == stage_expected
        and stage_counts["input_records"] == stage_expected
        and stage_counts["invalid_records"] == 0
        and stage_counts["duplicate_records"] == 0
    )
    # Execution status is supplied by report(); aggregate remains deterministic
    # and usable on its own for CPU-only review and self-tests.
    complete = final_complete and stage_complete
    developmental_signal = complete and observed_rule_satisfied
    status = "COMPLETE" if complete else "INCOMPLETE"
    verdict = (
        "DEVELOPMENTAL_SIGNAL" if developmental_signal else "NO_DEVELOPMENTAL_SIGNAL"
    ) if complete else "INCOMPLETE"
    return {
        "protocol": "addressed_delta_native_k8_development_v1",
        "status": status,
        "verdict": verdict,
        "completeness": {
            "expected_final_records": final_expected,
            "observed_unique_final_records": final_counts["observed_unique_records"],
            "final_records_complete": final_complete,
            "expected_dense_stage_records": stage_expected,
            "observed_unique_dense_stage_records": stage_counts["observed_unique_records"],
            "dense_stage_records_complete": stage_complete,
            "complete_requires_both_sets": True,
        },
        "record_counts": {"final": final_counts, "dense_stages": stage_counts},
        "final_readiness": {"by_arm": by_arm},
        "primary_delta_minus_additive": {
            "observed_pairs": paired,
            "expected_pairs": len(BLOCKS),
            "delta_only_wins": wins,
            "additive_only_wins": losses,
            "ties": ties,
            "net_paired_wins": net_wins,
            "exact_two_sided_p_descriptive_only": p_value,
            "developmental_signal_rule": {
                "delta_ready_at_least": 2,
                "net_paired_wins_at_least": 1,
                "criterion_satisfied_on_available_records": observed_rule_satisfied,
                "passed": developmental_signal,
            },
            "inference_boundary": (
                "Four paired blocks are developmental evidence only; even 4 wins "
                "and 0 losses have exact two-sided p=0.125."
            ),
        },
        "dense_stage_diagnostics": _stage_diagnostics(stage_index, stage_counts),
        "final_metric_means_by_arm": {
            arm: _metric_means(final_index, arm) for arm in ARMS
        },
        "claim_boundary": (
            "Current is contextual. Delta versus additive is the sole primary "
            "contrast. Full phenotype evaluation is not run. No checkpoint selection."
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _atomic_write(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    fd: int | None = None
    for attempt in range(10):
        candidate = path.with_name(
            f".{path.name}.{os.getpid()}.{time.time_ns()}.{attempt}.tmp"
        )
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            temp_path = candidate
            break
        except FileExistsError:
            continue
    if fd is None or temp_path is None:
        raise FileExistsError(f"Could not allocate an atomic temporary file for {path}")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            write_fn(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def _json_ready_summary(
    final_rows: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
    execution_status: Any,
) -> dict[str, Any]:
    summary = aggregate(final_rows, dense_records)
    error = _global_error(execution_status)
    if isinstance(execution_status, dict):
        execution = execution_status
    else:
        execution = {"provided_status": execution_status} if execution_status is not None else {}
    summary["execution_status"] = execution
    if error:
        summary["status"] = "ERROR"
        summary["verdict"] = "INCOMPLETE"
        summary["completeness"]["global_training_or_engineering_error"] = True
    else:
        summary["completeness"]["global_training_or_engineering_error"] = False
    return summary


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "block": row.get("block"),
        "arm": row.get("arm"),
        "update": _record_update(row),
        "joint_pass": _readiness(row),
        "evaluation_status": row.get("evaluation_status"),
        "checkpoint": _json_cell(row.get("checkpoint")),
        "checkpoint_hash": row.get("checkpoint_hash", row.get("hash")),
    }
    for size in SIZE_KEYS:
        flat = _flat_metrics(row, size)
        for field in _CSV_FIELDS:
            result[f"size{size}_{field}"] = flat.get(field)
    return result


def _csv_header() -> list[str]:
    return [
        "block",
        "arm",
        "update",
        "joint_pass",
        "evaluation_status",
        "checkpoint",
        "checkpoint_hash",
        *(f"size{size}_{field}" for size in SIZE_KEYS for field in _CSV_FIELDS),
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    header = _csv_header()

    def write(handle) -> None:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))

    _atomic_write(path, write)


def _fmt(value: Any, digits: int = 4) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _markdown(summary: dict[str, Any]) -> str:
    completeness = summary["completeness"]
    primary = summary["primary_delta_minus_additive"]
    lines = [
        "# Addressed delta development screen",
        "",
        f"Status: **{summary['status']}**. Verdict: **{summary['verdict']}**.",
        "",
        "The primary comparison is delta versus additive on size-32 joint readiness at u300. "
        "The four paired blocks are developmental evidence only.",
        "",
        f"Records: final {completeness['observed_unique_final_records']}/"
        f"{completeness['expected_final_records']}; dense stages "
        f"{completeness['observed_unique_dense_stage_records']}/"
        f"{completeness['expected_dense_stage_records']}.",
        "",
        "## u300 readiness",
        "",
        "| Arm | Ready | Observed |", "|---|---:|---:|",
    ]
    for arm in ARMS:
        row = summary["final_readiness"]["by_arm"][arm]
        lines.append(f"| {arm} | {row['ready']} | {row['observed_endpoints']} |")
    lines += [
        "",
        f"Delta-only wins={primary['delta_only_wins']}, additive-only wins="
        f"{primary['additive_only_wins']}, ties={primary['ties']}, net="
        f"{primary['net_paired_wins']}; exact two-sided p="
        f"{_fmt(primary['exact_two_sided_p_descriptive_only'], 3)} (descriptive only).",
        "",
        "## Final-block metric means",
        "",
        "Means below average available block values. Retention and support remain "
        "undefined when the frozen evaluator does not provide them.",
        "",
        "| Arm | Size | R pooled T64 | R mean T64 | Retention 64→256 | Continuous survival | Support maps | Support pixels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for size in SIZE_KEYS:
            metrics = summary["final_metric_means_by_arm"][arm][size]
            values = [metrics[field]["mean"] for field in _SCALAR_FIELDS]
            lines.append(
                f"| {arm} | {size} | {_fmt(values[0])} | {_fmt(values[1])} | "
                f"{_fmt(values[2])} | {_fmt(values[3])} | {_fmt(values[4], 1)} | {_fmt(values[5], 1)} |"
            )
    lines += [
        "",
        "## Checkpoint diagnostics",
        "",
        "| Arm | Stage records | Complete trajectories | Ever ready | Known nonready | Indeterminate without pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = summary["dense_stage_diagnostics"]["by_arm"][arm]
        lines.append(
            f"| {arm} | {row['observed_stage_records']}/52 | {row['complete_trajectories']}/4 | "
            f"{row['ever_ready_blocks']} | {row['known_nonready_blocks']} | "
            f"{row['indeterminate_without_any_pass_blocks']} |"
        )
    lines += [
        "",
        "Per-stage metrics, including both evaluation sizes and endpoint scores where available, "
        "are in `metrics.csv`; the 12 formal endpoints are in `final_metrics.csv`. "
        "No full Boolean traces or Full phenotype results are persisted.",
        "",
        "A developmental signal is not a reliability or significance result. "
        "Even 4 delta-only wins and 0 losses give exact two-sided p=0.125.",
        "",
    ]
    return "\n".join(lines)


def report(
    out: str | Path,
    final_rows: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
    execution_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write atomic JSON/Markdown/CSV summaries and return the aggregate.

    `final_rows` and `dense_records` are intentionally independent evidence
    streams. Endpoint rows never fill a missing stage record.
    """
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    summary = _json_ready_summary(final_rows, dense_records, execution_status)
    final_index, _ = _index_records(final_rows, final=True)
    stage_index, _ = _index_records(dense_records, final=False)
    stage_rows = [
        stage_index[key]
        for key in sorted(stage_index, key=lambda item: (item[0], item[2], item[1]))
    ]
    endpoint_rows = [
        final_index[key]
        for key in sorted(final_index, key=lambda item: (item[0], item[1]))
    ]

    _atomic_write(
        output / "summary.json",
        lambda handle: json.dump(
            summary, handle, indent=2, sort_keys=True, allow_nan=False, default=_json_default
        ),
    )
    _atomic_write(output / "RESULTS.md", lambda handle: handle.write(_markdown(summary)))
    _write_csv(output / "metrics.csv", stage_rows)
    _write_csv(output / "final_metrics.csv", endpoint_rows)
    return summary


def _self_test() -> None:
    def fixtures(delta_ready: set[int], additive_ready: set[int]):
        finals: list[dict[str, Any]] = []
        stages: list[dict[str, Any]] = []
        readiness = {
            "current": set(),
            "additive": additive_ready,
            "delta": delta_ready,
        }
        for block in BLOCKS:
            for arm in ARMS:
                passed = block in readiness[arm]
                final = {
                    "block": block,
                    "arm": arm,
                    "update": FORMAL_UPDATE,
                    "joint": {"pass": passed},
                    "evaluation_status": "ok",
                    "metrics": {},
                }
                finals.append(final)
                for update in CHECKPOINTS:
                    stages.append({**final, "update": update})
        return finals, stages

    all_fail_final, all_fail_stage = fixtures(set(), set())
    all_fail = aggregate(all_fail_final, all_fail_stage)
    assert all_fail["status"] == "COMPLETE"
    assert all_fail["verdict"] == "NO_DEVELOPMENTAL_SIGNAL"
    assert all_fail["final_readiness"]["by_arm"]["delta"]["ready"] == 0

    positive_final, positive_stage = fixtures({0, 1}, set())
    positive = aggregate(positive_final, positive_stage)
    assert positive["status"] == "COMPLETE"
    assert positive["verdict"] == "DEVELOPMENTAL_SIGNAL"
    assert positive["primary_delta_minus_additive"]["net_paired_wins"] == 2

    all_wins_final, all_wins_stage = fixtures(set(BLOCKS), set())
    all_wins = aggregate(all_wins_final, all_wins_stage)
    assert all_wins["primary_delta_minus_additive"]["exact_two_sided_p_descriptive_only"] == 0.125

    missing = aggregate(all_fail_final, all_fail_stage[:-1])
    assert missing["status"] == "INCOMPLETE" and missing["verdict"] == "INCOMPLETE"

    finals_only = aggregate(all_fail_final, [row for row in all_fail_stage if row["update"] == 300])
    assert finals_only["completeness"]["final_records_complete"] is True
    assert finals_only["completeness"]["dense_stage_records_complete"] is False
    assert finals_only["verdict"] == "INCOMPLETE"

    errored = _json_ready_summary(all_fail_final, all_fail_stage, {"status": "ERROR"})
    assert errored["status"] == "ERROR" and errored["verdict"] == "INCOMPLETE"

    with tempfile.TemporaryDirectory(
        prefix=".addressed_delta_report_", dir=Path(__file__).resolve().parent
    ) as directory:
        written = report(directory, positive_final, positive_stage, execution_status={})
        expected_files = {"summary.json", "RESULTS.md", "metrics.csv", "final_metrics.csv"}
        assert written["verdict"] == "DEVELOPMENTAL_SIGNAL"
        assert expected_files <= {path.name for path in Path(directory).iterdir()}
        with (Path(directory) / "metrics.csv").open(encoding="utf-8", newline="") as handle:
            assert sum(1 for _ in csv.DictReader(handle)) == len(BLOCKS) * len(ARMS) * len(CHECKPOINTS)
        with (Path(directory) / "final_metrics.csv").open(encoding="utf-8", newline="") as handle:
            assert sum(1 for _ in csv.DictReader(handle)) == len(BLOCKS) * len(ARMS)

    assert exact_two_sided_binomial_p(4, 0) == 0.125
    assert exact_two_sided_binomial_p(0, 0) == 1.0
    print("addressed_delta_reporting_self_test=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run CPU-only synthetic reporting checks")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        parser.error("provide --self-test; this module is imported by the experiment runner")


if __name__ == "__main__":
    main()
