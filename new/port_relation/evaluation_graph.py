"""CUDA-graph paired evaluator for fixed held-out port-relation banks.

The rollout, Boolean predicates, causal guards, endpoint scoring, and summary
formats mirror ``new/continuous_coverage/evaluation.py``.  CUDA graphs remove
the repeated Python dispatch from the 256-step paired rollouts; scoring and
host conversions remain outside capture.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "new/trajectory_qualification"))
import common as C  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D = _load("_port_relation_graph_activity", HERE / "activity.py")
E = _load("_port_relation_frozen_evaluation", ROOT / "new/continuous_coverage/evaluation.py")


@dataclass
class _CapturedSize:
    size: int
    bank_hash: str
    data: dict[str, torch.Tensor]
    graph: torch.cuda.CUDAGraph
    outputs: dict[str, Any]


@dataclass
class _ModelCache:
    model: torch.nn.Module
    parameter_binding: tuple
    activity: Any
    bank_hashes: dict[int, str] | None = None
    graphs: dict[int, _CapturedSize] = field(default_factory=dict)
    capture_seconds: float = 0.0
    graph_captures: int = 0
    graph_replays: int = 0
    evaluations: int = 0


_CACHES: dict[int, _ModelCache] = {}


def _parameter_binding(model: torch.nn.Module) -> tuple:
    """Identify parameter objects and addresses, while allowing in-place updates."""
    return (
        type(model).__module__,
        type(model).__qualname__,
        getattr(model, "arm", None),
        tuple(
            (
                name,
                id(parameter),
                int(parameter.data_ptr()),
                int(parameter.untyped_storage().data_ptr()),
                int(parameter.storage_offset()),
                tuple(parameter.shape),
                tuple(parameter.stride()),
                str(parameter.dtype),
                str(parameter.device),
            )
            for name, parameter in model.named_parameters()
        ),
    )


def _binding_digest(binding: tuple) -> str:
    return hashlib.sha256(repr(binding).encode("utf-8")).hexdigest()


def _entry(model: torch.nn.Module) -> _ModelCache:
    key = id(model)
    cached = _CACHES.get(key)
    if cached is not None:
        if cached.model is not model:
            raise RuntimeError("Evaluation cache identity collision")
        if cached.parameter_binding != _parameter_binding(model):
            raise RuntimeError(
                "Captured evaluation parameter storage changed; clear the model cache"
            )
        return cached
    cached = _ModelCache(
        model=model,
        parameter_binding=_parameter_binding(model),
        activity=D.EvaluationActivity(model),
    )
    _CACHES[key] = cached
    return cached


def get_activity(model: torch.nn.Module):
    """Return the persistent evaluation activity object bound to ``model``."""
    return _entry(model).activity


def _reset_activity(activity) -> None:
    for stats in activity.stats.values():
        stats.zero_()
    activity.calls = {32: 0, 64: 0}


def _trace_cuda(model, data, size: int) -> dict[str, Any]:
    """Capture-safe counterpart of the frozen ``E.trace`` tensor loop."""
    shape = (257, len(data["x"]), size, size)
    arrays = {
        key: torch.empty(shape, dtype=torch.bool, device=data["x"].device)
        for key in ("correct", "original_correct", "flipped_correct")
    }
    a, b = model.initial(data["x"]), model.initial(data["x_flip"])
    finite = torch.ones((), dtype=torch.bool, device=data["x"].device)
    violation = torch.zeros((), dtype=torch.bool, device=data["x"].device)
    outside_delta = torch.zeros((), device=data["x"].device)
    logits_at = {}

    for t in range(257):
        if t:
            a, b = model.step(a, data["x"]), model.step(b, data["x_flip"])
        for value in (*a, *b):
            finite = finite & torch.isfinite(value).all()
        logits, flipped = model.logits(a), model.logits(b)
        finite = finite & torch.isfinite(logits).all() & torch.isfinite(flipped).all()
        good_a = (logits >= 0) == (data["y"] >= .5)
        good_b = (flipped >= 0) == (data["y_flip"] >= .5)
        both = good_a & good_b
        arrays["correct"][t].copy_(both[:, 0])
        arrays["original_correct"][t].copy_(good_a[:, 0])
        arrays["flipped_correct"][t].copy_(good_b[:, 0])
        outside = data["changed"].bool() & (data["distance"] > 2 * t)
        violation = violation | (both & outside).any()
        outside_delta = torch.maximum(
            outside_delta, ((logits - flipped).abs() * outside).max()
        )
        if t in (64, 128, 256):
            logits_at[str(t)] = (logits, flipped)

    return {
        "arrays": arrays,
        "logits_at": logits_at,
        "finite": finite,
        "violation": violation,
        "outside_delta": outside_delta,
    }


def _capture_size(model, bank, size: int, bank_hash: str, activity) -> tuple[_CapturedSize, float]:
    device = next(model.parameters()).device
    data = {key: value.to(device=device) for key, value in bank.items()}
    graph = torch.cuda.CUDAGraph()
    result: dict[str, Any] = {}
    saved_observer = getattr(model, "relation_observer", None)
    capture_start = time.monotonic()

    try:
        # One eager pass on a side stream initializes kernels and allocator
        # state.  Instrumentation is disabled here because replayed capture
        # operations, rather than host calls during warmup, populate activity.
        model.relation_observer = None
        current = torch.cuda.current_stream(device=device)
        warmup = torch.cuda.Stream(device=device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup), torch.no_grad():
            _trace_cuda(model, data, size)
        current.wait_stream(warmup)
        torch.cuda.synchronize(device)

        _reset_activity(activity)
        model.relation_observer = activity
        # Use PyTorch's non-default capture stream; the legacy default stream
        # cannot be captured. All warmup work completed above.
        with torch.no_grad(), torch.cuda.graph(graph):
            result.update(_trace_cuda(model, data, size))
        torch.cuda.synchronize(device)
        if activity.calls[size] != 512:
            raise RuntimeError(f"Unexpected capture activity cadence for size {size}: {activity.calls}")
        return _CapturedSize(size, bank_hash, data, graph, result), time.monotonic() - capture_start
    finally:
        model.relation_observer = saved_observer
        _reset_activity(activity)


def _to_numpy_traces(outputs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: value.cpu().numpy() for key, value in outputs["arrays"].items()}


def _check_and_score(outputs, data, size: int):
    # These scalar conversions intentionally occur after graph replay.
    if not bool(outputs["finite"]):
        raise FloatingPointError("Nonfinite paired rollout")
    assert not bool(outputs["violation"]), "Paired correctness outside two-hop causal cone"
    outside_delta = float(outputs["outside_delta"])
    assert outside_delta <= 1e-6, "Source difference outside causal cone"

    score = C._load_audit().score
    records = {}
    for t in (64, 128, 256):
        logits, flipped = outputs["logits_at"][str(t)]
        records[str(t)] = {
            "original": score(logits, data["y"], data["mask"]),
            "flipped": score(flipped, data["y_flip"], data["mask"]),
        }
    return records


def _dense_summary(traces, records, banks):
    """Same dense report without unused first/stable passage-time argmaxes."""
    summary = {'schema': 'continuous-coverage-dense-v1', 'sizes': {},
               'phenotype_gate': None, 'full_evaluated': False,
               'claim_boundary': 'Predeclared formation diagnostic; no checkpoint selection.'}
    for size in (32, 64):
        values, bank = traces[str(size)], banks[size]
        correct = values['correct']
        changed = E._array(bank['changed'])[:, 0].astype(bool, copy=False)
        distance = E._array(bank['distance'])[:, 0]
        strict = changed & (distance > 16) & (distance < 32)
        # Frozen dense_summary uses first>=0 only; terminal stable time is
        # discarded. Streaming Boolean reductions preserve those exact counts.
        ever = changed & np.any(correct, axis=0)
        relapse = np.any(correct[:-1] & ~correct[1:], axis=0)
        row = {'maps': len(changed), 'changed_pixels': int(changed.sum()),
               'endpoints': {}, 'transitions': {'all_changed': {}, 'strict_16_32': {}},
               'continuous_survival64_to256': E.survival(values, bank)}
        for t in (64, 128, 256):
            rec = records[str(size)][str(t)]
            row['endpoints'][str(t)] = {
                'original_ba': rec['original']['balanced_accuracy'],
                'flipped_ba': rec['flipped']['balanced_accuracy'],
                'all_changed': E._coverage(correct[t], changed),
                'strict_16_32': E._coverage(correct[t], strict)}
        for label, selection in (('all_changed', changed), ('strict_16_32', strict)):
            for start, end in ((64, 128), (64, 256), (128, 256)):
                row['transitions'][label][f'{start}_to_{end}'] = E._transition(
                    correct, selection, start, end)
        row['ever_regressed_over_ever_correct'] = E._ratio(
            int((changed & relapse).sum()), int(ever.sum()))
        summary['sizes'][str(size)] = row
    return summary


def evaluate(model, banks, folder, full=False):
    """Evaluate with one CUDA graph per model and bank size.

    The returned measured dictionary and files have the same schema as the
    frozen evaluator.  Repeated calls reuse graphs only when the bank contents
    and model parameter storage still match their captured bindings.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    model.eval()
    if not torch.cuda.is_available() or not next(model.parameters()).is_cuda:
        raise RuntimeError("CUDA graph evaluation requires a CUDA model")

    cached = _entry(model)
    activity = cached.activity
    saved_observer = getattr(model, "relation_observer", None)
    model.relation_observer = activity
    before = C.tensor_hash(model.state_dict())
    bank_hashes = {size: C.tensor_hash(banks[size]) for size in (32, 64)}
    if cached.bank_hashes is None:
        cached.bank_hashes = bank_hashes
    elif cached.bank_hashes != bank_hashes:
        model.relation_observer = saved_observer
        raise RuntimeError("Evaluation banks changed for a captured model cache")

    try:
        for size in (32, 64):
            current = cached.graphs.get(size)
            if current is not None:
                if current.bank_hash != bank_hashes[size]:
                    raise RuntimeError(f"Evaluation bank changed for size {size}")
                continue
            captured, elapsed = _capture_size(
                model, banks[size], size, bank_hashes[size], activity
            )
            cached.graphs[size] = captured
            cached.capture_seconds += elapsed
            cached.graph_captures += 1

        _reset_activity(activity)
        traces, records = {}, {}
        for size in (32, 64):
            captured = cached.graphs[size]
            captured.graph.replay()
            cached.graph_replays += 1
        torch.cuda.synchronize(next(model.parameters()).device)

        # The Python callback itself ran only during capture.  Replayed CUDA
        # graph additions update the persistent stats tensors; publish the
        # predeclared call cadence explicitly for EvaluationActivity.result().
        activity.calls = {32: 512, 64: 512}

        for size in (32, 64):
            captured = cached.graphs[size]
            outputs = captured.outputs
            values = _to_numpy_traces(outputs)
            records[str(size)] = _check_and_score(outputs, captured.data, size)
            traces[str(size)] = values
            E.pack(folder / f"size{size}_traces.npz", values)

        if full:
            summary = E.summarize_from_traces(traces, records, banks)
            frontier = summary.pop("_matched_frontier_rows")
            summary["phenotype_gate"] = E.predicate(summary)
            summary["full_evaluated"] = True
            for size in (32, 64):
                summary["sizes"][str(size)]["continuous_survival64_to256"] = E.survival(
                    traces[str(size)], banks[size]
                )
            E._write_frontier_csv(folder / "matched_frontier_strata.csv", frontier)
        else:
            summary = _dense_summary(traces, records, banks)
        summary = E._json_ready(summary)
        C.write(folder / "summary.json", summary)
        if C.tensor_hash(model.state_dict()) != before:
            raise AssertionError("Evaluation mutated parameters")
        cached.evaluations += 1
        return summary
    finally:
        model.relation_observer = saved_observer


def cache_stats(model) -> dict[str, Any]:
    """Return non-scientific cache telemetry without changing evaluator output."""
    cached = _CACHES.get(id(model))
    if cached is None or cached.model is not model:
        return {"captured_sizes": [], "graph_captures": 0, "graph_replays": 0,
                "evaluations": 0, "capture_seconds": 0.0}
    return {
        "captured_sizes": sorted(cached.graphs),
        "graph_captures": cached.graph_captures,
        "graph_replays": cached.graph_replays,
        "evaluations": cached.evaluations,
        "capture_seconds": cached.capture_seconds,
        "parameter_binding_sha256": _binding_digest(cached.parameter_binding),
    }


def clear_cache(model) -> None:
    """Release this model's captured graphs, static banks, and activity tensors."""
    cached = _CACHES.get(id(model))
    if cached is None or cached.model is not model:
        return
    if torch.cuda.is_available() and any(parameter.is_cuda for parameter in model.parameters()):
        torch.cuda.synchronize(next(model.parameters()).device)
    del _CACHES[id(model)]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

