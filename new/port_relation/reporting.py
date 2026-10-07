"""Paired endpoint and dense-stage reports for the port-relation screen."""
from __future__ import annotations

import argparse
import csv
import math
import tempfile
from pathlib import Path
from typing import Any


ARMS = ("current", "constant", "conditioned")
PRIMARY_CANDIDATE = "conditioned"
PRIMARY_REFERENCE = "current"
CHECKPOINTS = tuple(range(0, 301, 25))
FORMAL_UPDATE = 300
EXPECTED_BLOCKS = 8
NET_GAIN_THRESHOLD = 6
PRIMARY_ALPHA = 0.05
SIZE_COLUMNS = ("32", "64")

_METRIC_COLUMNS = (
    "R_strict_pooled_T64",
    "R_strict_mean_T64",
    "S_retention64_to256",
    "S_retention_numerator",
    "S_continuous_survival64_to256",
    "retention_reference_pixels",
    "retention_reference_maps",
    "ever_regressed_fraction",
    "ever_regressed_numerator",
    "ever_regressed_denominator",
    "coverage_gain64_to256",
    "frontier_status",
    "frontier_effect",
)


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
    if update is not None:
        return update
    return _integer(row.get("checkpoint"))


def _readiness(row: Any) -> bool | None:
    """Read the runner's frozen joint gate, accepting the old key alias too."""
    if not isinstance(row, dict):
        return None
    for key in ("joint", "readiness"):
        value = row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, dict) and isinstance(value.get("pass"), bool):
            return value["pass"]
    return None


def exact_two_sided_binomial_p(wins: int, losses: int) -> float:
    """Exact two-sided sign-test p value over discordant paired blocks."""
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


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, float | int]:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Require 0 <= successes <= trials")
    if trials == 0:
        return {"successes": successes, "trials": trials, "lower": 0.0, "upper": 1.0}
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denominator
    return {
        "successes": successes,
        "trials": trials,
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
    }


def _formal_index(
    rows: list[dict[str, Any]], expected_blocks: int
) -> tuple[dict[tuple[int, str], dict[str, Any]], int, int]:
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    seen: set[tuple[int, str]] = set()
    ambiguous: set[tuple[int, str]] = set()
    invalid = duplicates = 0
    for row in rows:
        if not isinstance(row, dict):
            invalid += 1
            continue
        arm = row.get("arm")
        block = _integer(row.get("block"))
        raw_update = row.get("update")
        update = FORMAL_UPDATE if raw_update is None else _integer(raw_update)
        valid = (
            arm in ARMS
            and block is not None
            and block in range(expected_blocks)
            and update == FORMAL_UPDATE
            and _readiness(row) is not None
            and ("formal_endpoint" not in row or row.get("formal_endpoint") is True)
        )
        if not valid:
            invalid += 1
            continue
        key = (block, arm)
        if key in seen:
            duplicates += 1
            ambiguous.add(key)
            indexed.pop(key, None)
            continue
        seen.add(key)
        if key not in ambiguous:
            indexed[key] = row
    return indexed, invalid, duplicates


def _stage_index(
    rows: list[dict[str, Any]], expected_blocks: int
) -> tuple[dict[tuple[int, str, int], dict[str, Any]], dict[str, int]]:
    indexed: dict[tuple[int, str, int], dict[str, Any]] = {}
    seen: set[tuple[int, str, int]] = set()
    ambiguous: set[tuple[int, str, int]] = set()
    counts = {
        "ignored_non_grid_records": 0,
        "ignored_out_of_range_records": 0,
        "invalid_records": 0,
        "duplicate_predeclared_records": 0,
    }
    grid = set(CHECKPOINTS)
    for row in rows:
        if not isinstance(row, dict):
            counts["invalid_records"] += 1
            continue
        arm = row.get("arm")
        block = _integer(row.get("block"))
        if arm not in ARMS or block is None or block not in range(expected_blocks):
            counts["ignored_out_of_range_records"] += 1
            continue
        update = _record_update(row)
        if update is None:
            counts["invalid_records"] += 1
            continue
        if update not in grid:
            counts["ignored_non_grid_records"] += 1
            continue
        key = (block, arm, update)
        if key in seen:
            counts["duplicate_predeclared_records"] += 1
            ambiguous.add(key)
            indexed.pop(key, None)
            continue
        seen.add(key)
        if _readiness(row) is None:
            counts["invalid_records"] += 1
            continue
        if key not in ambiguous:
            indexed[key] = row
    counts["observed_predeclared_records"] = len(indexed)
    return indexed, counts


def _stage_summary(
    indexed: dict[tuple[int, str, int], dict[str, Any]], expected_blocks: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    expected_per_arm = expected_blocks * len(CHECKPOINTS)
    for arm in ARMS:
        ever_ready = known_nonready = indeterminate = 0
        first_ready_paths = readiness_losses = complete_paths = observed = 0
        for block in range(expected_blocks):
            path = [indexed.get((block, arm, update)) for update in CHECKPOINTS]
            observed += sum(row is not None for row in path)
            readiness = [_readiness(row) for row in path]
            has_unknown = any(value is None for value in readiness)
            if any(value is True for value in readiness):
                ever_ready += 1
            elif has_unknown:
                indeterminate += 1
            else:
                known_nonready += 1
            if has_unknown:
                continue
            complete_paths += 1
            if any(readiness):
                first = readiness.index(True)
                if first < len(CHECKPOINTS) - 1:
                    first_ready_paths += 1
                    if not all(readiness[first + 1 :]):
                        readiness_losses += 1
        denominator = ever_ready + known_nonready
        result[arm] = {
            "expected_checkpoint_records": expected_per_arm,
            "observed_checkpoint_records": observed,
            "complete_trajectories": complete_paths,
            "incomplete_trajectories": expected_blocks - complete_paths,
            "everready": {
                "successes": ever_ready,
                "known_nonready": known_nonready,
                "indeterminate_no_pass_trajectories": indeterminate,
                "denominator": denominator,
                "expected_trajectories": expected_blocks,
                "value": ever_ready / denominator if denominator else None,
            },
            "readiness_loss_after_first": {
                "losses": readiness_losses,
                "denominator": first_ready_paths,
                "value": readiness_losses / first_ready_paths if first_ready_paths else None,
            },
        }
    return result


def _size_metrics(record: dict[str, Any], size: str) -> dict[str, Any]:
    metrics = record.get("metrics")
    sizes = metrics.get("sizes") if isinstance(metrics, dict) else None
    point = sizes.get(size) if isinstance(sizes, dict) else None
    if point is None and isinstance(sizes, dict):
        point = sizes.get(int(size))
    return point if isinstance(point, dict) else {}


def _final_metric_summary(
    formal: dict[tuple[int, str], dict[str, Any]], expected_blocks: int
) -> dict[str, Any]:
    fields = ("R_strict_pooled_T64", "R_strict_mean_T64", "S_retention64_to256")
    result: dict[str, Any] = {}
    for arm in ARMS:
        result[arm] = {}
        for size in SIZE_COLUMNS:
            collected: dict[str, list[float]] = {field: [] for field in fields}
            for block in range(expected_blocks):
                row = formal.get((block, arm))
                point = _size_metrics(row, size) if row is not None else {}
                for field in fields:
                    value = _number(point.get(field))
                    if value is not None:
                        collected[field].append(value)
            result[arm][size] = {
                field: {
                    "mean": sum(values) / len(values) if values else None,
                    "observed": len(values),
                    "expected": expected_blocks,
                }
                for field, values in collected.items()
            }
    return result


def _old_full_secondary(
    formal: dict[tuple[int, str], dict[str, Any]], expected_blocks: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        values = []
        for block in range(expected_blocks):
            row = formal.get((block, arm))
            metrics = row.get("metrics") if isinstance(row, dict) else None
            value = metrics.get("full_pass") if isinstance(metrics, dict) else None
            values.append(value if isinstance(value, bool) else None)
        observed = [value for value in values if value is not None]
        successes = sum(observed)
        result[arm] = {
            "passes": successes,
            "observed": len(observed),
            "unknown": expected_blocks - len(observed),
            "rate": successes / len(observed) if observed else None,
        }
    return result


def _contrast(
    formal: dict[tuple[int, str], dict[str, Any]],
    candidate: str,
    reference: str,
    expected_blocks: int,
) -> dict[str, Any]:
    wins = losses = both_ready = neither_ready = complete = 0
    per_block = []
    for block in range(expected_blocks):
        candidate_ready = _readiness(formal.get((block, candidate)))
        reference_ready = _readiness(formal.get((block, reference)))
        if candidate_ready is None or reference_ready is None:
            comparison = "missing_or_invalid_formal_pair"
        else:
            complete += 1
            if candidate_ready and not reference_ready:
                wins += 1
                comparison = "candidate_only"
            elif reference_ready and not candidate_ready:
                losses += 1
                comparison = "reference_only"
            elif candidate_ready:
                both_ready += 1
                comparison = "both_ready"
            else:
                neither_ready += 1
                comparison = "neither_ready"
        per_block.append(
            {
                "block": block,
                "status": "paired" if comparison != "missing_or_invalid_formal_pair" else comparison,
                "candidate_ready": candidate_ready,
                "reference_ready": reference_ready,
                "comparison": comparison,
            }
        )
    return {
        "candidate": candidate,
        "reference": reference,
        "complete_pairs": complete,
        "expected_pairs": expected_blocks,
        "wins": wins,
        "losses": losses,
        "both_ready": both_ready,
        "neither_ready": neither_ready,
        "discordant_pairs": wins + losses,
        "net_gain": wins - losses,
        "per_block": per_block,
    }


def aggregate(
    final_rows: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
    expected_blocks: int = EXPECTED_BLOCKS,
) -> dict[str, Any]:
    """Summarize u300 paired readiness and the complete predeclared stage grid."""
    if expected_blocks != EXPECTED_BLOCKS:
        raise ValueError("port-relation screen is frozen at exactly 8 paired blocks")

    formal, invalid_formal, duplicate_formal = _formal_index(final_rows, expected_blocks)
    complete_by_arm = {
        arm: sum((block, arm) in formal for block in range(expected_blocks)) for arm in ARMS
    }
    formal_complete = all(complete_by_arm[arm] == expected_blocks for arm in ARMS)
    primary = _contrast(formal, PRIMARY_CANDIDATE, PRIMARY_REFERENCE, expected_blocks)
    primary["endpoint"] = "joint readiness at update 300, size 32"
    primary["exact_two_sided_p"] = (
        exact_two_sided_binomial_p(primary["wins"], primary["losses"])
        if primary["complete_pairs"] == expected_blocks
        else None
    )
    primary["net_gain_threshold"] = NET_GAIN_THRESHOLD
    primary["net_gain_threshold_met"] = (
        primary["net_gain"] >= NET_GAIN_THRESHOLD if formal_complete else None
    )
    primary["alpha"] = PRIMARY_ALPHA

    indexed, stage_counts = _stage_index(dense_records, expected_blocks)
    stage_by_arm = _stage_summary(indexed, expected_blocks)
    expected_records = expected_blocks * len(ARMS) * len(CHECKPOINTS)
    expected_trajectories = expected_blocks * len(ARMS)
    complete_trajectories = sum(
        stage_by_arm[arm]["complete_trajectories"] for arm in ARMS
    )
    stage_complete = (
        stage_counts["observed_predeclared_records"] == expected_records
        and stage_counts["duplicate_predeclared_records"] == 0
        and stage_counts["invalid_records"] == 0
        and complete_trajectories == expected_trajectories
    )
    content_complete = formal_complete and stage_complete
    qualified = (
        bool(
            primary["net_gain"] >= NET_GAIN_THRESHOLD
            and primary["exact_two_sided_p"] <= PRIMARY_ALPHA
        )
        if content_complete and primary["exact_two_sided_p"] is not None
        else None
    )
    primary["reliability_qualified"] = qualified
    verdict = (
        "INCOMPLETE"
        if not content_complete
        else "CONDITIONED_RELIABILITY_QUALIFIED"
        if qualified
        else "NO_CONDITIONED_RELIABILITY_QUALIFICATION"
    )
    primary["verdict"] = verdict

    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        rows = [formal[(block, arm)] for block in range(expected_blocks) if (block, arm) in formal]
        ready = sum(_readiness(row) is True for row in rows)
        arm_complete = len(rows) == expected_blocks
        per_arm[arm] = {
            "valid_formal_blocks": len(rows),
            "expected_formal_blocks": expected_blocks,
            "formal_readiness_passes": ready,
            "formal_readiness_rate": ready / len(rows) if rows else None,
            "formal_readiness_wilson95": wilson_interval(ready, len(rows)) if arm_complete else None,
            "formal_readiness_ci_status": "COMPLETE_BLOCKS" if arm_complete else "INCOMPLETE_NO_CI",
        }

    secondary = {
        "conditioned_minus_constant": _contrast(formal, "conditioned", "constant", expected_blocks),
        "constant_minus_current": _contrast(formal, "constant", "current", expected_blocks),
        "interpretation": "Descriptive paired counts only; these contrasts do not qualify the primary result and receive no multiplicity-adjusted claims.",
    }
    return {
        "schema": "port-relation-paired-report-v1",
        "status": "COMPLETE" if content_complete else "INCOMPLETE",
        "expected_blocks": expected_blocks,
        "arms": list(ARMS),
        "formal_update": FORMAL_UPDATE,
        "predeclared_checkpoints": list(CHECKPOINTS),
        "formal_rows": {
            "valid_rows": len(formal),
            "invalid_rows": invalid_formal,
            "duplicate_rows": duplicate_formal,
            "complete_blocks_by_arm": complete_by_arm,
            "complete": formal_complete,
        },
        "checkpoint_grid": {
            **stage_counts,
            "expected_records": expected_records,
            "expected_size_rows": expected_records * len(SIZE_COLUMNS),
            "exported_size_rows": len(indexed) * len(SIZE_COLUMNS),
            "complete_trajectories": complete_trajectories,
            "expected_trajectories": expected_trajectories,
            "complete": stage_complete,
            "by_arm": stage_by_arm,
        },
        "overall_verdict": verdict,
        "primary_comparison": primary,
        "secondary_comparisons": secondary,
        "per_arm": per_arm,
        "old_full_secondary": _old_full_secondary(formal, expected_blocks),
        "final_metrics": _final_metric_summary(formal, expected_blocks),
        "qualification_rule": (
            "The sole primary screen compares conditioned with current on size-32 joint readiness at u300. "
            "Qualification requires conditioned-only minus current-only blocks >=6/8 and an exact two-sided "
            "sign-test p<=0.05 over discordant paired blocks."
        ),
        "claim_boundary": (
            "This is finite evidence for this three-arm candidate and eight paired initialization/schedule blocks. "
            "A failure to form the endpoint rejects reliable formation for this candidate under this protocol; "
            "it does not close the broader port-relation route or prove a relation-algebra causal mechanism. "
            "A scheduled nonfinite evaluation is recorded as non-ready with trace_complete=false and "
            "numerical_failure=true; its attempt record is not a completed Boolean bank trace. Training or "
            "engineering errors stop the run and leave the experiment incomplete."
        ),
    }


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _bool_cell(value: bool | None) -> str:
    return "ready" if value is True else "not ready" if value is False else "unknown"


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    update = _record_update(record)
    if update is None and record.get("formal_endpoint") is True:
        update = FORMAL_UPDATE
    return {
        "block": record.get("block"),
        "init_seed": record.get("init_seed", record.get("initialization_seed")),
        "initialization_seed": record.get("initialization_seed", record.get("init_seed")),
        "schedule_seed": record.get("schedule_seed"),
        "arm": record.get("arm"),
        "update": update,
        "checkpoint": record.get("checkpoint", record.get("checkpoint_path")),
        "model_sha": record.get("model_sha", record.get("model_sha256", record.get("parameter_sha256"))),
        "checkpoint_sha256": record.get("checkpoint_sha256"),
        "last_loss": record.get("last_loss"),
        "seconds": record.get("seconds"),
        "evaluation_seconds": record.get("evaluation_seconds"),
        "formal_endpoint": record.get("formal_endpoint"),
        "trace_complete": record.get("trace_complete"),
        "numerical_failure": record.get("numerical_failure"),
    }


def _write_metrics_csv(path: Path, records: list[dict[str, Any]], expected_blocks: int) -> None:
    fields = (
        "block", "init_seed", "initialization_seed", "schedule_seed", "arm", "update",
        "checkpoint", "model_sha", "checkpoint_sha256", "last_loss", "seconds",
        "evaluation_seconds", "size", *_METRIC_COLUMNS, "joint_readiness", "full_pass",
        "formal_endpoint", "trace_complete", "numerical_failure",
    )
    indexed, _ = _stage_index(records, expected_blocks)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for block, arm, update in sorted(indexed, key=lambda key: (key[0], ARMS.index(key[1]), key[2])):
            record = indexed[(block, arm, update)]
            metrics = record.get("metrics")
            metrics = metrics if isinstance(metrics, dict) else {}
            for size in SIZE_COLUMNS:
                point = _size_metrics(record, size)
                writer.writerow({
                    **_metadata(record), "block": block, "arm": arm, "update": update,
                    "size": size, **{key: point.get(key) for key in _METRIC_COLUMNS},
                    "joint_readiness": _readiness(record) if size == "32" else None,
                    "full_pass": metrics.get("full_pass"),
                })


def _write_final_metrics_csv(path: Path, final_rows: list[dict[str, Any]], expected_blocks: int) -> None:
    fields = (
        "block", "init_seed", "initialization_seed", "schedule_seed", "arm", "update",
        "checkpoint", "model_sha", "checkpoint_sha256", "last_loss", "seconds",
        "evaluation_seconds", "size", *_METRIC_COLUMNS, "joint_readiness", "full_pass",
        "formal_endpoint", "trace_complete", "numerical_failure",
    )
    formal, _, _ = _formal_index(final_rows, expected_blocks)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for block, arm in sorted(formal, key=lambda key: (key[0], ARMS.index(key[1]))):
            record = formal[(block, arm)]
            metrics = record.get("metrics")
            metrics = metrics if isinstance(metrics, dict) else {}
            for size in SIZE_COLUMNS:
                point = _size_metrics(record, size)
                writer.writerow({
                    **_metadata(record), "block": block, "arm": arm, "update": FORMAL_UPDATE,
                    "size": size, **{key: point.get(key) for key in _METRIC_COLUMNS},
                    "joint_readiness": _readiness(record) if size == "32" else None,
                    "full_pass": metrics.get("full_pass"),
                })


def _status_fields(status: Any) -> tuple[str, bool | None, list[str]]:
    if status is None or status == {}:
        return "UNSPECIFIED", None, []
    if isinstance(status, dict):
        label = status.get("status", status.get("run_status", "UNKNOWN"))
        error = status.get("error")
        errors = status.get("errors")
        messages = [str(error)] if error else []
        if isinstance(errors, list):
            messages.extend(str(item) for item in errors if item)
        elif errors:
            messages.append(str(errors))
    else:
        label, messages = status, []
    normalized = str(label).upper()
    complete = normalized in {"COMPLETE", "COMPLETED", "SUCCESS", "SUCCEEDED"}
    return str(label), complete, messages


def report(
    folder: str | Path,
    final_records: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
    status: Any = None,
) -> dict[str, Any]:
    """Write RESULTS.md and paired metric CSVs, then return the aggregate."""
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    summary = aggregate(final_records, dense_records)
    formal_lookup, _, _ = _formal_index(final_records, EXPECTED_BLOCKS)
    execution_label, execution_complete, execution_errors = _status_fields(status)
    summary["execution_status"] = execution_label
    summary["execution_errors"] = execution_errors
    if execution_complete is False:
        summary["status"] = "INCOMPLETE"
        summary["overall_verdict"] = "INCOMPLETE"
        summary["primary_comparison"]["reliability_qualified"] = None
        summary["primary_comparison"]["verdict"] = "INCOMPLETE"

    primary = summary["primary_comparison"]
    formal = summary["formal_rows"]
    grid = summary["checkpoint_grid"]
    lines = [
        "# Port-relation paired formation screen",
        "",
        f"Execution status: {execution_label}",
        f"Aggregation status: {summary['status']}",
        f"Overall verdict: {summary['overall_verdict']}",
    ]
    if execution_errors:
        lines += ["", "Execution errors:", *(f"- {item}" for item in execution_errors)]
    lines += [
        "",
        "The sole primary endpoint is size-32 joint readiness at update 300. The only qualifying contrast is conditioned versus current; constant comparisons, old Full, size-64, and intermediate stages are descriptive.",
        "",
        "| Arm | Valid u300 blocks | Expected | Joint-ready | Wilson 95% interval | Old Full passes | Full observed |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for arm in ARMS:
        row = summary["per_arm"][arm]
        interval = row["formal_readiness_wilson95"]
        interval_text = f"[{_cell(interval['lower'])}, {_cell(interval['upper'])}]" if interval else "— (incomplete; no CI)"
        full = summary["old_full_secondary"][arm]
        lines.append(
            f"| {arm} | {row['valid_formal_blocks']} | {row['expected_formal_blocks']} | "
            f"{row['formal_readiness_passes']} | {interval_text} | {full['passes']} | "
            f"{full['observed']}/{row['expected_formal_blocks']} |"
        )
    primary_result = (
        "INCOMPLETE" if primary["reliability_qualified"] is None
        else "QUALIFIED" if primary["reliability_qualified"] else "NOT QUALIFIED"
    )
    lines += [
        "",
        "## Sole primary contrast",
        "",
        "| Contrast | Complete pairs | Conditioned-only wins | Current-only losses | Net gain | Exact two-sided p | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| conditioned − current at u300 | {primary['complete_pairs']}/{primary['expected_pairs']} | "
        f"{primary['wins']} | {primary['losses']} | {_cell(primary['net_gain'])} | "
        f"{_cell(primary['exact_two_sided_p'])} | {primary_result} |",
        "",
        summary["qualification_rule"],
        "",
        "A missing endpoint pair or incomplete dense grid leaves the primary verdict INCOMPLETE. A complete negative result rejects reliable formation for this candidate under this protocol only.",
        "",
        "## Paired block outcomes",
        "",
        "| Block | Current | Constant | Conditioned | Primary comparison |",
        "|---:|---|---|---|---|",
    ]
    for row in primary["per_block"]:
        block = row["block"]
        lines.append(
            f"| {block} | {_bool_cell(_readiness(formal_lookup.get((block, 'current'))))} | "
            f"{_bool_cell(_readiness(formal_lookup.get((block, 'constant'))))} | "
            f"{_bool_cell(_readiness(formal_lookup.get((block, 'conditioned'))))} | {row['comparison']} |"
        )
    lines += [
        "",
        "## Secondary paired counts",
        "",
        "These are descriptive block counts only; they have no qualification threshold and do not change the primary conclusion.",
        "",
        "| Contrast | Complete pairs | Candidate-only | Reference-only | Both ready | Neither ready |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, title in (("conditioned_minus_constant", "conditioned − constant"), ("constant_minus_current", "constant − current")):
        item = summary["secondary_comparisons"][key]
        lines.append(
            f"| {title} | {item['complete_pairs']}/{item['expected_pairs']} | {item['wins']} | "
            f"{item['losses']} | {item['both_ready']} | {item['neither_ready']} |"
        )
    lines += [
        "",
        "## Final u300 reach and retention metrics",
        "",
        "Values are arithmetic means across blocks with defined values; n reports the number of defined blocks.",
        "",
        "| Arm | Size | R strict pooled T64 | R strict map mean T64 | S retention T64→T256 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for size in SIZE_COLUMNS:
            metrics = summary["final_metrics"][arm][size]
            cells = [
                f"{_cell(metrics[key]['mean'])} (n={metrics[key]['observed']}/{metrics[key]['expected']})"
                for key in ("R_strict_pooled_T64", "R_strict_mean_T64", "S_retention64_to256")
            ]
            lines.append(f"| {arm} | {size} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += [
        "",
        "## Dense stage diagnostics",
        "",
        "Intermediate stages and ever-ready counts are formation diagnostics; they never replace the fixed u300 endpoint or select a checkpoint.",
        "",
        "| Arm | Complete stage-record paths | Ever-ready trajectories | Ever-ready fraction | Readiness losses after first ready |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = grid["by_arm"][arm]
        ever = row["everready"]
        loss = row["readiness_loss_after_first"]
        lines.append(
            f"| {arm} | {row['complete_trajectories']}/{summary['expected_blocks']} | "
            f"{ever['successes']}/{ever['denominator']} | {_cell(ever['value'])} | "
            f"{loss['losses']}/{loss['denominator']} |"
        )
    lines += [
        "",
        f"Valid u300 rows: {formal['valid_rows']}/24. Predeclared stage records: "
        f"{grid['observed_predeclared_records']}/{grid['expected_records']}; complete stage-record paths: "
        f"{grid['complete_trajectories']}/{grid['expected_trajectories']}.",
        "",
        summary["claim_boundary"],
        "",
        f"final_metrics.csv contains u300 rows by block, arm, and size. metrics.csv contains predeclared stage rows by block, arm, update, and size (expected {grid['expected_size_rows']} rows when complete).",
    ]
    (destination / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_final_metrics_csv(destination / "final_metrics.csv", final_records, EXPECTED_BLOCKS)
    _write_metrics_csv(destination / "metrics.csv", dense_records, EXPECTED_BLOCKS)
    return summary


def _synthetic(
    conditioned_ready_blocks: set[int],
    current_ready_blocks: set[int] | None = None,
    constant_ready_blocks: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current_ready_blocks = current_ready_blocks or set()
    constant_ready_blocks = constant_ready_blocks or set()
    ready_by_arm = {
        "current": current_ready_blocks,
        "constant": constant_ready_blocks,
        "conditioned": conditioned_ready_blocks,
    }
    finals: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    for block in range(EXPECTED_BLOCKS):
        for arm in ARMS:
            ready = block in ready_by_arm[arm]
            metrics = {
                "full_pass": False,
                "sizes": {
                    size: {
                        "R_strict_pooled_T64": 0.82 if ready else 0.5,
                        "R_strict_mean_T64": 0.81 if ready else 0.5,
                        "S_retention64_to256": 0.96 if ready else 0.5,
                        "S_retention_numerator": 960 if ready else 50,
                        "S_continuous_survival64_to256": 0.94 if ready else 0.4,
                        "retention_reference_pixels": 1000 if ready else 50,
                        "retention_reference_maps": 16 if ready else 8,
                        "ever_regressed_fraction": 0.01,
                        "ever_regressed_numerator": 1,
                        "ever_regressed_denominator": 100,
                        "coverage_gain64_to256": 0.02,
                        "frontier_status": "not_evaluated",
                        "frontier_effect": None,
                    }
                    for size in SIZE_COLUMNS
                },
            }
            final = {
                "block": block,
                "arm": arm,
                "update": FORMAL_UPDATE,
                "init_seed": 130001 + block,
                "schedule_seed": 131001 + block,
                "metrics": metrics,
                "joint": {"pass": ready},
                "last_loss": 0.5,
                "formal_endpoint": True,
            }
            finals.append(final)
            for update in CHECKPOINTS:
                stage_ready = ready
                # One transient pass checks that u300 remains the formal endpoint.
                if arm == "conditioned" and block == 7 and update == 275:
                    stage_ready = True
                if arm == "conditioned" and block == 7 and update == FORMAL_UPDATE:
                    stage_ready = False
                stages.append({
                    **final,
                    "update": update,
                    "joint": {"pass": stage_ready},
                    "formal_endpoint": update == FORMAL_UPDATE,
                    "checkpoint": f"u{update:03d}.pt",
                })
    return finals, stages


def _self_test() -> None:
    passed_finals, passed_stages = _synthetic(set(range(6)))
    passed = aggregate(passed_finals, passed_stages)
    assert passed["status"] == "COMPLETE"
    assert passed["overall_verdict"] == "CONDITIONED_RELIABILITY_QUALIFIED"
    assert passed["primary_comparison"]["wins"] == 6
    assert passed["primary_comparison"]["losses"] == 0
    assert passed["primary_comparison"]["exact_two_sided_p"] == 0.03125
    assert passed["checkpoint_grid"]["expected_records"] == 312
    assert passed["checkpoint_grid"]["expected_size_rows"] == 624

    failed_finals, failed_stages = _synthetic(set(range(5)))
    failed = aggregate(failed_finals, failed_stages)
    assert failed["status"] == "COMPLETE"
    assert failed["overall_verdict"] == "NO_CONDITIONED_RELIABILITY_QUALIFICATION"
    assert failed["primary_comparison"]["net_gain"] == 5

    negative_finals, negative_stages = _synthetic(set())
    negative = aggregate(negative_finals, negative_stages)
    assert negative["status"] == "COMPLETE"
    assert negative["per_arm"]["conditioned"]["formal_readiness_passes"] == 0
    assert negative["overall_verdict"] == "NO_CONDITIONED_RELIABILITY_QUALIFICATION"

    partial = aggregate(passed_finals, passed_stages[:-1])
    assert partial["status"] == "INCOMPLETE"
    assert partial["primary_comparison"]["reliability_qualified"] is None

    with tempfile.TemporaryDirectory(
        prefix=".port-relation-report-", dir=Path(__file__).resolve().parent
    ) as temp:
        written = report(temp, passed_finals, passed_stages, status={})
        with (Path(temp) / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            stage_rows = sum(1 for _ in csv.DictReader(handle))
        with (Path(temp) / "final_metrics.csv").open(newline="", encoding="utf-8") as handle:
            final_rows = sum(1 for _ in csv.DictReader(handle))
        assert stage_rows == 624 and final_rows == 48
        assert written["checkpoint_grid"]["exported_size_rows"] == 624
        assert written["overall_verdict"] == "CONDITIONED_RELIABILITY_QUALIFIED"
    print("port-relation reporting CPU self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    parser.error("this module exposes aggregate()/report(); use --self-test for CPU checks")


if __name__ == "__main__":
    main()
