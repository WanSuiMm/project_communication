"""Three-arm endpoint aggregation for the delayed-credit screen."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ARMS = ("dense_k8", "terminal_k8", "terminal_k64")
ENDPOINT = 300


def _json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return repr(value)


def _number(value: Any) -> float | None:
    try:
        result = float(value) if value is not None and not isinstance(value, bool) else None
        return result if result is not None and math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _gate(evaluation: Any, name: str) -> bool | None:
    gate = evaluation.get(name) if isinstance(evaluation, Mapping) else None
    if isinstance(gate, bool):
        return gate
    if isinstance(gate, Mapping):
        return next((gate[k] for k in ("pass", "passed", "qualified") if isinstance(gate.get(k), bool)), None)
    return None


def _primary(evaluation: Any) -> dict[str, Any]:
    result = {"strict_T64_mean_map": None, "strict_T64_pooled": None, "observed": {}}
    if not isinstance(evaluation, Mapping):
        return result
    sizes = evaluation.get("sizes", {})
    size32 = sizes.get("32", sizes.get(32, {})) if isinstance(sizes, Mapping) else {}
    endpoints = size32.get("endpoints", {}) if isinstance(size32, Mapping) else {}
    endpoint = endpoints.get("64") if isinstance(endpoints, Mapping) else None
    strict = endpoint.get("strict_16_32") if isinstance(endpoint, Mapping) else None
    pooled = strict.get("pooled_coverage") if isinstance(strict, Mapping) else None
    result["strict_T64_mean_map"] = _number(
        strict.get("mean_map_coverage") if isinstance(strict, Mapping) else None
    )
    result["strict_T64_pooled"] = _number(
        pooled.get("value") if isinstance(pooled, Mapping) else None
    )
    gate = evaluation.get("reach_gate", {})
    observed = gate.get("observed", {}) if isinstance(gate, Mapping) else {}
    if isinstance(observed, Mapping):
        result["observed"] = _json(observed)
        if result["strict_T64_mean_map"] is None:
            result["strict_T64_mean_map"] = _number(observed.get("strict_T64_mean_map"))
        if result["strict_T64_pooled"] is None:
            result["strict_T64_pooled"] = _number(observed.get("strict_T64_pooled"))
    return result


def _complete(row: Mapping[str, Any] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    update = row.get("update") if row.get("update") is not None else row.get("completed_updates")
    done = row.get("completed_updates")
    bad = {"failed", "error", "interrupted", "incomplete"}
    state = str(row.get("status", row.get("training_status", ""))).strip().lower()
    return (
        _number(update) == ENDPOINT
        and (done is None or (_number(done) is not None and _number(done) >= ENDPOINT))
        and row.get("training_complete") is not False
        and state not in bad
    )


def _paired(blocks: list[dict[str, Any]], qualified_only: bool) -> dict[str, Any]:
    scores = dict(wins=0, losses=0, ties=0, unscored=0)
    gates = dict(C_pass_B_fail=0, B_pass_C_fail=0, both_pass=0, both_fail=0, unknown=0)
    considered = 0
    for block in blocks:
        c, b = block["arms"]["terminal_k64"], block["arms"]["terminal_k8"]
        if qualified_only and c["reach"] != "pass":
            continue
        considered += 1
        pair = (c["reach"], b["reach"])
        key = {
            ("pass", "fail"): "C_pass_B_fail", ("fail", "pass"): "B_pass_C_fail",
            ("pass", "pass"): "both_pass", ("fail", "fail"): "both_fail",
        }.get(pair, "unknown")
        gates[key] += 1
        cs, bs = c["primary"]["strict_T64_mean_map"], b["primary"]["strict_T64_mean_map"]
        if not c["endpoint_complete"] or not b["endpoint_complete"] or cs is None or bs is None:
            scores["unscored"] += 1
        elif math.isclose(cs, bs, rel_tol=0.0, abs_tol=1e-12):
            scores["ties"] += 1
        else:
            scores["wins" if cs > bs else "losses"] += 1
    return {"blocks_considered": considered, "map_mean_score": scores, "reach_gate": gates}


def aggregate(rows: list[dict[str, Any]], status: Any) -> dict[str, Any]:
    """Aggregate u300 rows and retain every supplied failure or incomplete record."""
    blocks = list(status.get("expected_blocks", range(4))) if isinstance(status, Mapping) else list(range(4))
    index: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    invalid = 0
    for row in rows:
        if not isinstance(row, Mapping) or row.get("block") is None or row.get("arm") not in ARMS:
            invalid += 1
        else:
            index.setdefault((str(row["block"]), str(row["arm"])), []).append(row)

    counts = {arm: {"pass": 0, "fail": 0, "unknown": 0} for arm in ARMS}
    per_block, complete = [], True
    for block_id in blocks:
        arms = {}
        for arm in ARMS:
            candidates = index.get((str(block_id), arm), [])
            row = candidates[0] if len(candidates) == 1 else None
            trained = len(candidates) == 1 and _complete(row)
            evaluation = row.get("evaluation") if row else None
            gate = _gate(evaluation, "reach_gate")
            reach = "unknown" if not trained or gate is None else ("pass" if gate else "fail")
            hold_gate = _gate(evaluation, "hold_gate")
            hold = "unknown" if not trained or hold_gate is None else ("pass" if hold_gate else "fail")
            counts[arm][reach] += 1
            endpoint_complete = trained and reach != "unknown"
            complete &= endpoint_complete
            arms[arm] = {
                "record_status": row.get("status") if row else ("duplicate" if len(candidates) > 1 else "missing"),
                "update": row.get("update", row.get("completed_updates")) if row else None,
                "training_complete": trained, "endpoint_complete": endpoint_complete,
                "reach": reach, "hold": hold, "primary": _primary(evaluation),
            }
        per_block.append({
            "block": block_id, "arms": arms,
            "classification_ABC": {
                "A": arms["dense_k8"]["reach"], "B": arms["terminal_k8"]["reach"],
                "C": arms["terminal_k64"]["reach"],
            },
        })

    if isinstance(status, Mapping) and status.get("primary_complete") is False:
        complete = False
    execution_status = status if isinstance(status, str) else status.get("status", "INCOMPLETE")
    if execution_status != "COMPLETE":
        complete = False
    cpass, bpass = counts["terminal_k64"]["pass"], counts["terminal_k8"]["pass"]
    all_pairs, c_pairs = _paired(per_block, False), _paired(per_block, True)
    if not complete:
        verdict = "INCOMPLETE"
    elif cpass < 3:
        verdict = "BASELINE_UNQUALIFIED"
    elif all_pairs["reach_gate"]["C_pass_B_fail"] >= 3:
        verdict = "DEVELOPMENTAL_CREDIT_GAP"
    elif bpass >= 3:
        verdict = "NO_K8_GAP_ON_THIS_SCREEN"
    else:
        verdict = "MIXED_DEVELOPMENTAL_RESULT"
    return {
        "schema_version": 1, "protocol": "delayed_credit_cue_once_streaming_v0",
        "status": _json(status), "complete": complete, "expected_blocks": blocks,
        "expected_arms": list(ARMS), "input_rows": len(rows), "invalid_rows": invalid,
        "reach_counts": counts, "C_qualified_blocks": cpass, "B_qualified_blocks": bpass,
        "per_block": per_block, "B_vs_C_all_blocks": all_pairs,
        "B_vs_C_among_C_qualified": c_pairs, "verdict": verdict,
        "dense_A_note": "A failure does not invalidate a qualified B/C contrast; it leaves any dense-rescue claim unqualified.",
        "rows": [_json(row) for row in rows],
    }


def report(out: str | Path, rows: list[dict[str, Any]], status: Any) -> dict[str, Any]:
    """Write summary.json and RESULTS.md and return the same summary."""
    summary = aggregate(rows, status)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    temporary_summary = output / "summary.json.tmp"
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(output / "summary.json")
    n = len(summary["expected_blocks"])
    lines = [
        "# Delayed-credit cue-once screen", "", f"Execution status: {summary['status']}.",
        f"Recorded trajectories: {len(rows)}/12.", f"Verdict: {summary['verdict']}.",
        f"Complete: {summary['complete']}.", "", "| Arm | Reach pass | Reach fail | Unknown |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm, c in summary["reach_counts"].items():
        lines.append(f"| {arm} | {c['pass']}/{n} | {c['fail']}/{n} | {c['unknown']}/{n} |")
    lines += ["", "| Block | A dense K8 | B terminal K8 | C terminal K64 |", "| ---: | --- | --- | --- |"]
    for block in summary["per_block"]:
        c = block["classification_ABC"]
        lines.append(f"| {block['block']} | {c['A']} | {c['B']} | {c['C']} |")
    lines += [
        "", f"C-qualified: {summary['C_qualified_blocks']}/{n}; B-qualified: {summary['B_qualified_blocks']}/{n}.",
        f"All-block B/C reach contrasts: {summary['B_vs_C_all_blocks']['reach_gate']}.",
        f"C-qualified B/C strict-band map-mean scores: {summary['B_vs_C_among_C_qualified']['map_mean_score']}.",
        "T128/T256 are secondary; the primary gate is size 32 at T64.",
        "Independent unit: paired model block; maps and pixels are within-block measurements.",
        summary["dense_A_note"], "Full statuses, errors, and incomplete records are in summary.json.",
    ]
    temporary_report = output / "RESULTS.md.tmp"
    temporary_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_report.replace(output / "RESULTS.md")
    return summary
