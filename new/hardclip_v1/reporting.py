"""Paired fixed-endpoint and checkpoint-grid summaries for HardClip v1."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

try:
    from hybrid_writer.reporting import exact_two_sided_binomial_p, wilson_interval
except ModuleNotFoundError as exc:
    # Direct execution of this file starts with hardclip_v1/ on sys.path;
    # runner imports normally already expose the sibling packages under new/.
    if exc.name != "hybrid_writer":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_writer.reporting import exact_two_sided_binomial_p, wilson_interval


ARMS = ("neural", "hardclip")
CHECKPOINTS = tuple(range(0, 301, 25))
FORMAL_UPDATE = 300
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
    number = _number(value)
    if number is None or number != math.floor(number):
        return None
    return int(number)


def _readiness(row: Any) -> bool | None:
    if not isinstance(row, dict):
        return None
    joint = row.get("joint")
    if not isinstance(joint, dict):
        return None
    value = joint.get("pass")
    return value if isinstance(value, bool) else None


def _formal_rows(
    final_rows: list[dict[str, Any]], expected_blocks: int
) -> tuple[dict[tuple[int, str], dict[str, Any]], int, int]:
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    seen: set[tuple[int, str]] = set()
    invalid = duplicates = 0
    for row in final_rows:
        if not isinstance(row, dict):
            invalid += 1
            continue
        arm = row.get("arm")
        block = _integer(row.get("block"))
        update = _integer(row.get("update"))
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
            indexed.pop(key, None)
            continue
        seen.add(key)
        indexed[key] = row
    return indexed, invalid, duplicates


def _checkpoint_index(
    records: list[dict[str, Any]], expected_blocks: int | None
) -> tuple[dict[tuple[int, str, int], dict[str, Any]], dict[str, int]]:
    indexed: dict[tuple[int, str, int], dict[str, Any]] = {}
    seen: set[tuple[int, str, int]] = set()
    counts = {
        "observed_predeclared_records": 0,
        "ignored_non_grid_records": 0,
        "ignored_out_of_range_records": 0,
        "invalid_records": 0,
        "duplicate_predeclared_records": 0,
    }
    grid = set(CHECKPOINTS)
    for row in records:
        if not isinstance(row, dict):
            counts["invalid_records"] += 1
            continue
        arm = row.get("arm")
        block = _integer(row.get("block"))
        update = _integer(row.get("update"))
        if (
            arm not in ARMS
            or block is None
            or block < 0
            or (expected_blocks is not None and block not in range(expected_blocks))
        ):
            counts["ignored_out_of_range_records"] += 1
            continue
        if update is None:
            counts["invalid_records"] += 1
            continue
        if update not in grid:
            counts["ignored_non_grid_records"] += 1
            continue
        key = (block, arm, update)
        if key in seen:
            counts["duplicate_predeclared_records"] += 1
            indexed.pop(key, None)
            continue
        seen.add(key)
        if _readiness(row) is None:
            counts["invalid_records"] += 1
            continue
        indexed[key] = row
        counts["observed_predeclared_records"] += 1
    return indexed, counts


def _checkpoint_summary(
    indexed: dict[tuple[int, str, int], dict[str, Any]], expected_blocks: int
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
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

        ever_denominator = ever_ready + known_nonready
        summaries[arm] = {
            "expected_checkpoint_records": expected_per_arm,
            "observed_checkpoint_records": observed,
            "complete_trajectories": complete_paths,
            "incomplete_trajectories": expected_blocks - complete_paths,
            "everready": {
                "successes": ever_ready,
                "known_nonready": known_nonready,
                "indeterminate_no_pass_trajectories": indeterminate,
                "denominator": ever_denominator,
                "expected_trajectories": expected_blocks,
                "value": ever_ready / ever_denominator if ever_denominator else None,
            },
            "readiness_loss_after_first": {
                "losses": readiness_losses,
                "denominator": first_ready_paths,
                "value": (
                    readiness_losses / first_ready_paths if first_ready_paths else None
                ),
                "definition": (
                    "Among complete trajectories first joint-ready before u300, "
                    "fraction with any later predeclared checkpoint not joint-ready."
                ),
            },
        }
    return summaries


def _full_secondary(
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


def aggregate(
    final_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    expected_blocks: int = 8,
) -> dict[str, Any]:
    """Aggregate the u300 paired endpoint and all predeclared checkpoints.

    Overall completeness requires every paired u300 row and all expected
    block/arm/checkpoint records. The u300 test is reported when all formal
    pairs exist; missing secondary trajectories still keep the overall result
    INCOMPLETE.
    """
    if expected_blocks <= 0:
        raise ValueError("expected_blocks must be positive")

    formal, invalid_formal_rows, duplicate_formal_rows = _formal_rows(
        final_rows, expected_blocks
    )
    complete_by_arm = {
        arm: sum((block, arm) in formal for block in range(expected_blocks))
        for arm in ARMS
    }
    complete_pairs = 0
    wins = losses = both_pass = neither_pass = 0
    per_block: list[dict[str, Any]] = []
    for block in range(expected_blocks):
        candidate = formal.get((block, "hardclip"))
        reference = formal.get((block, "neural"))
        candidate_pass = _readiness(candidate)
        reference_pass = _readiness(reference)
        if candidate_pass is None or reference_pass is None:
            per_block.append({"block": block, "status": "missing_or_invalid_formal_pair"})
            continue
        complete_pairs += 1
        if candidate_pass and not reference_pass:
            wins += 1
            comparison = "hardclip_only"
        elif reference_pass and not candidate_pass:
            losses += 1
            comparison = "neural_only"
        elif candidate_pass:
            both_pass += 1
            comparison = "both_ready"
        else:
            neither_pass += 1
            comparison = "neither_ready"
        per_block.append(
            {
                "block": block,
                "status": "paired",
                "hardclip_ready": candidate_pass,
                "neural_ready": reference_pass,
                "comparison": comparison,
            }
        )

    formal_complete = complete_pairs == expected_blocks
    p_value = exact_two_sided_binomial_p(wins, losses) if formal_complete else None
    net_gain = wins - losses
    primary_qualified = (
        bool(net_gain >= NET_GAIN_THRESHOLD and p_value <= PRIMARY_ALPHA)
        if p_value is not None
        else None
    )
    indexed, record_counts = _checkpoint_index(records, expected_blocks)
    checkpoint_by_arm = _checkpoint_summary(indexed, expected_blocks)
    expected_records = expected_blocks * len(ARMS) * len(CHECKPOINTS)
    expected_trajectories = expected_blocks * len(ARMS)
    complete_trajectories = sum(
        checkpoint_by_arm[arm]["complete_trajectories"] for arm in ARMS
    )
    grid_complete = (
        record_counts["observed_predeclared_records"] == expected_records
        and record_counts["duplicate_predeclared_records"] == 0
        and record_counts["invalid_records"] == 0
        and complete_trajectories == expected_trajectories
    )
    content_complete = formal_complete and grid_complete

    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        rows = [
            formal[(block, arm)]
            for block in range(expected_blocks)
            if (block, arm) in formal
        ]
        ready = sum(_readiness(row) is True for row in rows)
        arm_complete = len(rows) == expected_blocks
        per_arm[arm] = {
            "valid_formal_blocks": len(rows),
            "expected_formal_blocks": expected_blocks,
            "formal_readiness_passes": ready,
            "formal_readiness_rate": ready / len(rows) if rows else None,
            "formal_readiness_wilson95": (
                wilson_interval(ready, len(rows)) if arm_complete else None
            ),
            "formal_readiness_ci_status": (
                "COMPLETE_BLOCKS" if arm_complete else "INCOMPLETE_NO_CI"
            ),
        }

    if not content_complete:
        verdict = "INCOMPLETE"
    elif primary_qualified:
        verdict = "HARDCLIP_RELIABILITY_QUALIFIED"
    else:
        verdict = "NO_HARDCLIP_RELIABILITY_QUALIFICATION"

    return {
        "schema": "hardclip-v1-paired-report-v1",
        "status": "COMPLETE" if content_complete else "INCOMPLETE",
        "expected_blocks": expected_blocks,
        "arms": list(ARMS),
        "predeclared_checkpoints": list(CHECKPOINTS),
        "formal_update": FORMAL_UPDATE,
        "formal_rows": {
            "valid_rows": len(formal),
            "invalid_rows": invalid_formal_rows,
            "duplicate_rows": duplicate_formal_rows,
            "complete_pairs": complete_pairs,
            "expected_pairs": expected_blocks,
            "complete_blocks_by_arm": complete_by_arm,
            "complete": formal_complete,
        },
        "checkpoint_grid": {
            **record_counts,
            "expected_records": expected_records,
            "complete_trajectories": complete_trajectories,
            "expected_trajectories": expected_trajectories,
            "complete": grid_complete,
            "by_arm": checkpoint_by_arm,
        },
        "overall_verdict": verdict,
        "primary_comparison": {
            "candidate": "hardclip",
            "reference": "neural",
            "complete_pairs": complete_pairs,
            "expected_pairs": expected_blocks,
            "wins": wins,
            "losses": losses,
            "both_ready": both_pass,
            "neither_ready": neither_pass,
            "discordant_pairs": wins + losses,
            "net_gain": net_gain,
            "exact_two_sided_p": p_value,
            "net_gain_threshold": NET_GAIN_THRESHOLD,
            "net_gain_threshold_met": (
                net_gain >= NET_GAIN_THRESHOLD if formal_complete else None
            ),
            "reliability_qualified": primary_qualified,
            "per_block": per_block,
        },
        "per_arm": per_arm,
        "old_full_secondary": _full_secondary(formal, expected_blocks),
        "qualification_rule": (
            "HardClip qualifies only when hardclip-only minus neural-only wins "
            "are at least 6 and the exact two-sided discordant-pair binomial "
            "p value is at most 0.05. This is the sole primary contrast; no Holm adjustment."
        ),
        "claim_boundary": (
            "Finite evidence for this fixed K8 recipe and intervention. A negative "
            "result does not exclude all small benefits or all large-write mechanisms."
        ),
    }


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_metrics_csv(
    path: Path, records: list[dict[str, Any]], expected_blocks: int
) -> None:
    fields = (
        "block",
        "arm",
        "update",
        "size",
        *_METRIC_COLUMNS,
        "joint_readiness",
        "full_pass",
        "formal_endpoint",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        indexed, _ = _checkpoint_index(records, expected_blocks)
        for block, arm, update in sorted(
            indexed, key=lambda key: (key[0], ARMS.index(key[1]), key[2])
        ):
            record = indexed[(block, arm, update)]
            metrics = record.get("metrics")
            metrics = metrics if isinstance(metrics, dict) else {}
            sizes = metrics.get("sizes")
            sizes = sizes if isinstance(sizes, dict) else {}
            joint_value = _readiness(record)
            for size in SIZE_COLUMNS:
                size_metrics = sizes.get(size)
                size_metrics = size_metrics if isinstance(size_metrics, dict) else {}
                row = {
                    "block": block,
                    "arm": arm,
                    "update": update,
                    "size": size,
                    **{key: size_metrics.get(key) for key in _METRIC_COLUMNS},
                    "joint_readiness": joint_value if size == "32" else None,
                    "full_pass": metrics.get("full_pass"),
                    "formal_endpoint": record.get("formal_endpoint"),
                }
                writer.writerow(row)


def _run_status_fields(run_status: Any) -> tuple[str, bool, list[str]]:
    if isinstance(run_status, dict):
        label = run_status.get("status", run_status.get("run_status", "UNKNOWN"))
        error = run_status.get("error")
        errors = run_status.get("errors")
        messages = []
        if error:
            messages.append(str(error))
        if isinstance(errors, list):
            messages.extend(str(item) for item in errors if item)
        elif errors:
            messages.append(str(errors))
    else:
        label = run_status if run_status is not None else "UNKNOWN"
        messages = []
    normalized = str(label).upper()
    execution_complete = normalized in {"COMPLETE", "COMPLETED", "SUCCESS", "SUCCEEDED"}
    return str(label), execution_complete, messages


def report(
    out: str | Path,
    run_status: Any,
    records: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the concise result report and size-level checkpoint metrics CSV."""
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    summary = aggregate(final_rows, records)
    execution_label, execution_complete, execution_errors = _run_status_fields(run_status)
    summary["execution_status"] = execution_label
    summary["execution_errors"] = execution_errors
    if not execution_complete or summary["status"] != "COMPLETE":
        summary["status"] = "INCOMPLETE"
        summary["overall_verdict"] = "INCOMPLETE"

    primary = summary["primary_comparison"]
    lines = [
        "# HardClip paired reliability",
        "",
        f"Execution status: {execution_label}",
        f"Aggregation status: {summary['status']}",
        f"Overall verdict: {summary['overall_verdict']}",
    ]
    if execution_errors:
        lines += ["", "Execution errors:", *(f"- {item}" for item in execution_errors)]
    lines += [
        "",
        "The formal endpoint is joint readiness at u300. Checkpoint diagnostics use only u0, u25, ..., u300; no peak checkpoint is selected.",
        "",
        "| Arm | Valid u300 blocks | Expected | Joint-ready | Wilson 95% readiness interval |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        row = summary["per_arm"][arm]
        interval = row["formal_readiness_wilson95"]
        interval_text = (
            f"[{_cell(interval['lower'])}, {_cell(interval['upper'])}]"
            if interval is not None
            else "— (incomplete; no CI)"
        )
        lines.append(
            f"| {arm} | {row['valid_formal_blocks']} | {row['expected_formal_blocks']} "
            f"| {row['formal_readiness_passes']} | {interval_text} |"
        )
    pvalue = primary["exact_two_sided_p"]
    primary_result = (
        "INCOMPLETE"
        if primary["reliability_qualified"] is None
        else "QUALIFIED" if primary["reliability_qualified"] else "NOT QUALIFIED"
    )
    lines += [
        "",
        "## Sole primary contrast",
        "",
        "| Contrast | Complete pairs | HardClip-only wins | Neural-only wins | Net gain | Exact two-sided p | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| hardclip − neural | {primary['complete_pairs']} | {primary['wins']} | "
        f"{primary['losses']} | {_cell(primary['net_gain'])} | {_cell(pvalue)} | {primary_result} |",
        "",
        summary["qualification_rule"],
        "",
        "## Old Full gate (secondary)",
        "",
        "| Arm | Full passes | Observed | Unknown | Pass fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = summary["old_full_secondary"][arm]
        lines.append(
            f"| {arm} | {row['passes']} | {row['observed']} | {row['unknown']} | {_cell(row['rate'])} |"
        )
    lines += [
        "",
        "## Predeclared checkpoint diagnostics",
        "",
        "| Arm | Complete trajectories | Ever-ready | Ever-ready fraction | First-ready before u300 | Lost readiness later | Loss fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    grid = summary["checkpoint_grid"]
    for arm in ARMS:
        row = grid["by_arm"][arm]
        ever = row["everready"]
        loss = row["readiness_loss_after_first"]
        lines.append(
            f"| {arm} | {row['complete_trajectories']}/{summary['expected_blocks']} | "
            f"{ever['successes']}/{ever['denominator']} | {_cell(ever['value'])} | "
            f"{loss['denominator']} | {loss['losses']} | {_cell(loss['value'])} |"
        )
    lines += [
        "",
        f"Predeclared checkpoint records: {grid['observed_predeclared_records']}/{grid['expected_records']}; "
        f"complete trajectories: {grid['complete_trajectories']}/{grid['expected_trajectories']}. "
        "Ever-ready excludes unresolved no-pass trajectories; a zero denominator is undefined. "
        "Readiness loss counts complete trajectories that first pass before u300 and fail at a later checkpoint.",
        "",
        summary["claim_boundary"],
        "",
        "metrics.csv contains one row per checkpoint and evaluated size (416 rows when all 208 expected records are present).",
    ]
    (destination / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_metrics_csv(destination / "metrics.csv", records, summary["expected_blocks"])
    return summary


def _synthetic(
    readiness_by_block: dict[int, tuple[bool, bool]] | None = None,
    missing_checkpoint: tuple[int, str, int] | None = None,
    unknown_neural_checkpoints: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    readiness_by_block = readiness_by_block or {
        block: (False, False) for block in range(8)
    }
    final_rows = []
    records = []
    for block in range(8):
        neural_ready, hardclip_ready = readiness_by_block.get(block, (False, False))
        for arm, ready in (("neural", neural_ready), ("hardclip", hardclip_ready)):
            joint = {"pass": ready}
            metrics = {"full_pass": False, "sizes": {"32": {}, "64": {}}}
            final_rows.append(
                {
                    "block": block,
                    "arm": arm,
                    "update": FORMAL_UPDATE,
                    "formal_endpoint": True,
                    "joint": joint,
                    "metrics": metrics,
                }
            )
            for update in CHECKPOINTS:
                if missing_checkpoint == (block, arm, update):
                    continue
                value: bool | None = ready
                if unknown_neural_checkpoints and arm == "neural":
                    value = None
                records.append(
                    {
                        "block": block,
                        "arm": arm,
                        "update": update,
                        "formal_endpoint": update == FORMAL_UPDATE,
                        "joint": {"pass": value},
                        "metrics": metrics,
                    }
                )
    return final_rows, records


def _self_test() -> None:
    all_both = {block: (True, True) for block in range(8)}
    finals, records = _synthetic(all_both)
    summary = aggregate(finals, records)
    assert summary["status"] == "COMPLETE"
    assert summary["checkpoint_grid"]["expected_trajectories"] == 16
    assert summary["checkpoint_grid"]["observed_predeclared_records"] == 208
    assert summary["primary_comparison"]["exact_two_sided_p"] == 1.0
    assert summary["overall_verdict"] == "NO_HARDCLIP_RELIABILITY_QUALIFICATION"

    all_neither = {block: (False, False) for block in range(8)}
    finals, records = _synthetic(all_neither)
    assert aggregate(finals, records)["primary_comparison"]["neither_ready"] == 8

    six_wins = {
        block: (False, block < 6) for block in range(8)
    }
    finals, records = _synthetic(six_wins)
    summary = aggregate(finals, records)
    assert summary["primary_comparison"]["wins"] == 6
    assert summary["primary_comparison"]["losses"] == 0
    assert summary["primary_comparison"]["exact_two_sided_p"] == 0.03125
    assert summary["overall_verdict"] == "HARDCLIP_RELIABILITY_QUALIFIED"

    one_win = {block: (False, block == 0) for block in range(8)}
    finals, records = _synthetic(one_win)
    summary = aggregate(finals, records)
    assert summary["primary_comparison"]["exact_two_sided_p"] == 1.0
    assert summary["overall_verdict"] == "NO_HARDCLIP_RELIABILITY_QUALIFICATION"

    finals, records = _synthetic()
    summary = aggregate(finals[:-1], records)
    assert summary["status"] == "INCOMPLETE"
    assert summary["overall_verdict"] == "INCOMPLETE"
    finals, records = _synthetic(missing_checkpoint=(0, "neural", 0))
    assert aggregate(finals, records)["status"] == "INCOMPLETE"

    finals, records = _synthetic(unknown_neural_checkpoints=True)
    summary = aggregate(finals, records)
    assert summary["checkpoint_grid"]["by_arm"]["neural"]["everready"]["value"] is None
    assert summary["checkpoint_grid"]["by_arm"]["neural"]["readiness_loss_after_first"]["value"] is None


if __name__ == "__main__":
    _self_test()
    print("hardclip_v1 reporting self-test: PASS")
