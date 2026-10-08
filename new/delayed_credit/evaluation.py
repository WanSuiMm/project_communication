"""Paired delayed-credit evaluation with source cues available only at t=0."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluation helper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = _load(
    "_delayed_credit_continuous_evaluation",
    ROOT / "new/continuous_coverage/evaluation.py",
)
C = E.C
from run_revision import score


class _StaticGeometryProxy:
    """Forward model state/readout calls while hiding step-time cue channels."""

    def __init__(self, model):
        self._model = model
        self._step_inputs: dict[int, torch.Tensor] = {}

    def eval(self):
        self._model.eval()
        return self

    def state_dict(self, *args, **kwargs):
        return self._model.state_dict(*args, **kwargs)

    def initial(self, x: torch.Tensor):
        # The two source-cue channels remain available to the initial encoder.
        return self._model.initial(x)

    def logits(self, state):
        return self._model.logits(state)

    @staticmethod
    def _geometry_only(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Delayed-credit inputs must have mask plus two cue channels")
        return torch.cat((x[:, :1], torch.zeros_like(x[:, 1:])), dim=1)

    def bind_step_inputs(self, *inputs: torch.Tensor) -> None:
        self._step_inputs = {
            int(x.data_ptr()): self._geometry_only(x) for x in inputs
        }

    def step(self, state, x: torch.Tensor):
        # Both paired branches use the same static geometry (mask, 0, 0) at
        # every step. The cue-bearing x is still passed unchanged to initial().
        step_x = self._step_inputs.get(int(x.data_ptr()))
        if step_x is None:
            step_x = self._geometry_only(x)
        return self._model.step(state, step_x)


@torch.no_grad()
def _trace_valid_prefix(model, data, size: int):
    """Retain only the prefix before the first nonfinite paired state/logit."""
    shape = (257, len(data["x"]), size, size)
    device = data["x"].device
    arrays = {
        key: torch.empty(shape, dtype=torch.bool, device=device)
        for key in ("correct", "original_correct", "flipped_correct")
    }
    a, b = model.initial(data["x"]), model.initial(data["x_flip"])
    finite_flags = []
    violations = []
    outside_deltas = []
    logits_at = {}
    for time in range(257):
        if time:
            a, b = model.step(a, data["x"]), model.step(b, data["x_flip"])
        logits, flipped = model.logits(a), model.logits(b)
        finite_leaves = [torch.isfinite(value).all() for value in (*a, *b)]
        finite_leaves.extend((torch.isfinite(logits).all(), torch.isfinite(flipped).all()))
        finite_flags.append(torch.stack(finite_leaves).all())

        good_a = (logits >= 0) == (data["y"] >= 0.5)
        good_b = (flipped >= 0) == (data["y_flip"] >= 0.5)
        both = good_a & good_b
        arrays["correct"][time].copy_(both[:, 0])
        arrays["original_correct"][time].copy_(good_a[:, 0])
        arrays["flipped_correct"][time].copy_(good_b[:, 0])

        outside = data["changed"].bool() & (data["distance"] > 2 * time)
        violations.append((both & outside).any())
        outside_deltas.append(((logits - flipped).abs() * outside).max())
        if time in (64, 128, 256):
            logits_at[str(time)] = (logits, flipped)

    finite_by_time = torch.stack(finite_flags).cpu().numpy().astype(bool, copy=False)
    bad_times = np.flatnonzero(~finite_by_time)
    failure_step = int(bad_times[0]) if bad_times.size else None
    valid_through = failure_step - 1 if failure_step is not None else 256
    valid_count = valid_through + 1

    if valid_count:
        violation = torch.stack(violations[:valid_count]).any()
        outside_delta = torch.stack(outside_deltas[:valid_count]).max()
        assert not bool(violation), "Paired correctness outside two-hop causal cone"
        outside_delta_value = float(outside_delta)
        assert outside_delta_value <= 1e-6, "Source difference outside causal cone"
    else:
        outside_delta_value = None

    records = {}
    for time in (64, 128, 256):
        if valid_through < time:
            continue
        logits, flipped = logits_at[str(time)]
        records[str(time)] = {
            "original": score(logits, data["y"], data["mask"]),
            "flipped": score(flipped, data["y_flip"], data["mask"]),
        }

    values = {
        key: tensor[:valid_count].cpu().numpy().copy()
        for key, tensor in arrays.items()
    }
    status = {
        "finite": failure_step is None,
        "valid_through_step": int(valid_through),
        "numerical_failure_at_step": failure_step,
        "outside_source_cone_max_logit_delta": outside_delta_value,
    }
    return values, records, status


def _pack_valid_prefix(path: Path, values: dict, status: dict) -> None:
    """Pack each recorded Boolean prefix losslessly and state its valid times."""
    items = {}
    for key, value in values.items():
        array = np.asarray(value, dtype=bool)
        items[key + "_shape"] = np.asarray(array.shape, dtype=np.int32)
        items[key + "_packed"] = np.packbits(array.reshape(-1), bitorder="little")
    valid_through = status["valid_through_step"]
    items["valid_through_step"] = np.asarray(valid_through, dtype=np.int32)
    items["numerical_failure_at_step"] = np.asarray(
        -1 if status["numerical_failure_at_step"] is None else status["numerical_failure_at_step"],
        dtype=np.int32,
    )
    items["valid_times"] = np.arange(valid_through + 1, dtype=np.int16)
    np.savez_compressed(path, **items)


def _per_map_metrics(correct: np.ndarray, bank, size: int, valid_through: int) -> list[dict]:
    changed = E._array(bank["changed"])[:, 0].astype(bool, copy=False)
    distance = E._array(bank["distance"])[:, 0]
    strict = changed & (distance > 16) & (distance < 32)
    rows = []
    for index in range(len(changed)):
        selected = {"changed": changed[index], "strict": strict[index]}
        has_t64 = valid_through >= 64
        has_t256 = valid_through >= 256
        reference = correct[64, index] & changed[index] if has_t64 else None
        reference_pixels = int(reference.sum()) if reference is not None else None
        retained = (
            int((reference & correct[256, index]).sum())
            if reference is not None and has_t256
            else None
        )
        row = {
            "size": int(size),
            "map_index": int(index),
            "strict_pixels": int(strict[index].sum()),
            "changed_pixels": int(changed[index].sum()),
            "retention_reference_pixels": reference_pixels,
            "retained_pixels_T256": retained,
            "continuously_retained_pixels_T64_to256": (
                int((reference & correct[64:, index].all(axis=0)).sum())
                if reference is not None and has_t256
                else None
            ),
            "retention64_to256": (
                retained / reference_pixels
                if retained is not None and reference_pixels
                else None
            ),
        }
        for time in (64, 128, 256):
            for label, mask in selected.items():
                pixels = int(mask.sum())
                if valid_through >= time:
                    good = int((correct[time, index] & mask).sum())
                    row[f"{label}_correct_T{time}"] = good
                    row[f"{label}_coverage_T{time}"] = good / pixels if pixels else None
                else:
                    row[f"{label}_correct_T{time}"] = None
                    row[f"{label}_coverage_T{time}"] = None
        rows.append(row)
    return rows


def _prefix_dense_summary(traces, records, banks, trace_status):
    """Keep dense-summary fields while leaving every unavailable endpoint null."""
    summary = {
        "schema": "continuous-coverage-dense-v1",
        "sizes": {},
        "phenotype_gate": None,
        "full_evaluated": False,
        "claim_boundary": "Delayed-credit prefix diagnostic; no checkpoint selection.",
    }
    for size in (32, 64):
        key = str(size)
        values, bank = traces[key], banks[size]
        correct = values["correct"]
        valid_through = trace_status[key]["valid_through_step"]
        changed = E._array(bank["changed"])[:, 0].astype(bool, copy=False)
        distance = E._array(bank["distance"])[:, 0]
        strict = changed & (distance > 16) & (distance < 32)
        row = {
            "maps": len(changed),
            "changed_pixels": int(changed.sum()),
            "endpoints": {},
            "transitions": {"all_changed": {}, "strict_16_32": {}},
            "continuous_survival64_to256": (
                E.survival(values, bank) if valid_through >= 256 else None
            ),
            "valid_through_step": int(valid_through),
        }
        for time in (64, 128, 256):
            if valid_through >= time:
                rec = records[key][str(time)]
                row["endpoints"][str(time)] = {
                    "original_ba": rec["original"]["balanced_accuracy"],
                    "flipped_ba": rec["flipped"]["balanced_accuracy"],
                    "all_changed": E._coverage(correct[time], changed),
                    "strict_16_32": E._coverage(correct[time], strict),
                }
            else:
                row["endpoints"][str(time)] = None
        for label, selection in (("all_changed", changed), ("strict_16_32", strict)):
            for start, end in ((64, 128), (64, 256), (128, 256)):
                row["transitions"][label][f"{start}_to_{end}"] = (
                    E._transition(correct, selection, start, end)
                    if valid_through >= end
                    else None
                )
        if correct.shape[0]:
            first, _, relapse = E._first_and_stable(correct)
            ever = changed & (first >= 0)
            row["ever_regressed_over_ever_correct"] = E._ratio(
                int((changed & relapse).sum()), int(ever.sum())
            )
            row["ever_regressed_over_ever_correct_valid_through_step"] = int(valid_through)
        else:
            row["ever_regressed_over_ever_correct"] = None
            row["ever_regressed_over_ever_correct_valid_through_step"] = -1
        summary["sizes"][key] = row
    return summary


def _reach_gate(summary: dict, trace_status: dict) -> dict:
    valid_t64 = trace_status["valid_through_step"] >= 64
    endpoint = summary["sizes"]["32"]["endpoints"]["64"]
    strict = endpoint["strict_16_32"] if endpoint is not None else None
    pooled = strict["pooled_coverage"]["value"] if strict is not None else None
    mean_map = strict["mean_map_coverage"] if strict is not None else None
    original_ba = endpoint["original_ba"] if endpoint is not None else None
    flipped_ba = endpoint["flipped_ba"] if endpoint is not None else None
    eligible_maps = strict["eligible_maps"] if strict is not None else None
    pixels = strict["pixels"] if strict is not None else None
    checks = {
        "finite_trace_through_T64": valid_t64,
        "strict_T64_pooled_ge_0_80": pooled is not None and pooled >= 0.80,
        "strict_T64_mean_map_ge_0_80": mean_map is not None and mean_map >= 0.80,
        "original_open_BA_T64_ge_0_85": original_ba is not None and original_ba >= 0.85,
        "flipped_open_BA_T64_ge_0_85": flipped_ba is not None and flipped_ba >= 0.85,
        "strict_eligible_maps_at_least_16": eligible_maps is not None and eligible_maps >= 16,
        "strict_pixels_at_least_100": pixels is not None and pixels >= 100,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    support_ok = checks["strict_eligible_maps_at_least_16"] and checks["strict_pixels_at_least_100"]
    return {
        "pass": all(checks.values()),
        "reasons": reasons,
        "checks": checks,
        "observed": {
            "strict_T64_pooled": pooled,
            "strict_T64_mean_map": mean_map,
            "original_open_BA_T64": original_ba,
            "flipped_open_BA_T64": flipped_ba,
        },
        "support": {"eligible_maps": eligible_maps, "pixels": pixels},
        "failure_class": (
            "NUMERICAL_FAILURE"
            if not valid_t64
            else ("UNQUALIFIED_SUPPORT" if not support_ok else ("REACH_CRITERION_FAILED" if reasons else None))
        ),
        "criterion_boundary": (
            "Primary reach uses size32 T64 strict-band coverage, paired open balanced "
            "accuracy, and minimum strict-band support; it has no retention or gain requirement."
        ),
    }


def _hold_gate(summary: dict, trace_status: dict) -> dict:
    endpoints = summary["sizes"]["32"]["endpoints"]
    base = endpoints["64"]
    base_strict = base["strict_16_32"] if base is not None else None
    base_pooled = (
        base_strict["pooled_coverage"]["value"] if base_strict is not None else None
    )
    base_mean = base_strict["mean_map_coverage"] if base_strict is not None else None
    checks = {"finite_trace_through_T256": trace_status["valid_through_step"] >= 256}
    observed = {}

    for time in (128, 256):
        endpoint = endpoints[str(time)]
        for branch in ("original", "flipped"):
            baseline_ba = base[f"{branch}_ba"] if base is not None else None
            current_ba = endpoint[f"{branch}_ba"] if endpoint is not None else None
            drop = (
                baseline_ba - current_ba
                if baseline_ba is not None and current_ba is not None
                else None
            )
            name = f"{branch}_BA_drop_T{time}_le_0_03"
            observed[name] = drop
            checks[name] = drop is not None and drop <= 0.03

        current_strict = endpoint["strict_16_32"] if endpoint is not None else None
        current_pooled = (
            current_strict["pooled_coverage"]["value"] if current_strict is not None else None
        )
        pooled_drop = (
            base_pooled - current_pooled
            if base_pooled is not None and current_pooled is not None
            else None
        )
        name = f"strict_pooled_drop_T{time}_le_0_05"
        observed[name] = pooled_drop
        checks[name] = pooled_drop is not None and pooled_drop <= 0.05

        current_mean = current_strict["mean_map_coverage"] if current_strict is not None else None
        mean_drop = (
            base_mean - current_mean
            if base_mean is not None and current_mean is not None
            else None
        )
        name = f"strict_mean_map_drop_T{time}_le_0_05"
        observed[name] = mean_drop
        checks[name] = mean_drop is not None and mean_drop <= 0.05

    numerical_failure = trace_status["valid_through_step"] < 256
    reasons = [name for name, passed in checks.items() if not passed]
    if numerical_failure:
        reasons.append("NUMERICAL_FAILURE_before_T256")
    return {
        "pass": all(checks.values()),
        "reasons": reasons,
        "checks": checks,
        "observed_drops_from_T64": observed,
        "failure_class": "NUMERICAL_FAILURE" if numerical_failure else ("HOLD_CRITERION_FAILED" if reasons else None),
        "criterion_boundary": (
            "Secondary size32 hold compares each later endpoint with its own T64; "
            "it is reported separately from the primary reach gate."
        ),
    }


@torch.no_grad()
def evaluate(model, banks, folder) -> dict:
    """Record paired traces at sizes32/64 and evaluate delayed-credit gates.

    ``initial`` receives each original cue-bearing input. Every subsequent
    ``step`` receives only static geometry ``(mask, 0, 0)`` on both branches.
    The full Boolean trace is stored losslessly in the compact packed NPZ form
    used by continuous coverage evaluation.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before = C.tensor_hash(model.state_dict())
    proxy = _StaticGeometryProxy(model).eval()

    traces, records, per_map, trace_status = {}, {}, {}, {}
    for size in (32, 64):
        data = {key: value.cuda() for key, value in banks[size].items()}
        proxy.bind_step_inputs(data["x"], data["x_flip"])
        values, endpoint_records, status = _trace_valid_prefix(proxy, data, size)
        traces[str(size)] = values
        records[str(size)] = endpoint_records
        trace_status[str(size)] = status
        per_map[str(size)] = _per_map_metrics(
            values["correct"], banks[size], size, status["valid_through_step"]
        )
        _pack_valid_prefix(folder / f"size{size}_traces.npz", values, status)
        del data

    if all(status["finite"] for status in trace_status.values()):
        summary = E.dense_summary(traces, records, banks)
    else:
        summary = _prefix_dense_summary(traces, records, banks, trace_status)
    summary["evaluation_variant"] = "delayed_credit_v0_sourcecue_initialonly"
    summary["claim_boundary"] = (
        "Source cues enter only through the initial state. Primary reach is a size32 T64 "
        "diagnostic; the later hold is secondary. Dataset qualification and task-learning "
        "claims are outside this evaluation."
    )
    summary["per_map_metrics"] = per_map
    summary["trace_status"] = {"sizes": trace_status}
    size32_status = trace_status["32"]
    summary["numerical_status"] = {
        "finite_all_steps_both_sizes": all(status["finite"] for status in trace_status.values()),
        "primary_T64_valid": size32_status["valid_through_step"] >= 64,
        "secondary_T256_valid": size32_status["valid_through_step"] >= 256,
    }
    summary["reach_gate"] = _reach_gate(summary, size32_status)
    summary["hold_gate"] = _hold_gate(summary, size32_status)
    if size32_status["valid_through_step"] >= 64:
        if all(status["finite"] for status in trace_status.values()):
            summary["evaluation_status"] = "COMPLETE"
        elif size32_status["valid_through_step"] < 256:
            summary["evaluation_status"] = "T64_REACH_COMPLETE_WITH_LATE_NUMERICAL_FAILURE"
        else:
            summary["evaluation_status"] = "T64_REACH_COMPLETE_WITH_OTHER_NUMERICAL_FAILURE"
    else:
        summary["evaluation_status"] = "NUMERICAL_FAILURE_BEFORE_T64"
    summary = E._json_ready(summary)

    C.write(
        folder / "per_map_metrics.json",
        {"schema": "delayed-credit-per-map-v0", "sizes": per_map},
    )
    C.write(folder / "summary.json", summary)
    if C.tensor_hash(model.state_dict()) != before:
        raise RuntimeError("Delayed-credit evaluation mutated model parameters")
    return summary
