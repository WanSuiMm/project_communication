"""Zero-training K8/full64 gradients for frozen continuation models."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {"encoder": ("encoder.",), "f_in/f_out": ("f_in.", "f_out."),
          "q_in/q_out": ("q_in.", "q_out."), "readout": ("readout.",)}
_HISTORICAL = None


def _historical():
    """Import the exact original helper and its balanced-loss module safely."""
    global _HISTORICAL
    if _HISTORICAL is not None:
        return _HISTORICAL
    task_path = ROOT / "new/nca_inertial_wind_tunnel/tasks.py"
    helper_path = ROOT / "new/short_bptt/training.py"
    spec = importlib.util.spec_from_file_location("_continuation_tasks", task_path)
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)
    names = ("tasks", "revision_cells", "masked_cells")
    previous, paths = {n: sys.modules.get(n) for n in names}, sys.path[:]
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules["tasks"] = tasks
        spec = importlib.util.spec_from_file_location("_continuation_training", helper_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = paths
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]
    _HISTORICAL = module, tasks
    return _HISTORICAL


def _params(model):
    return sorted(model.named_parameters(), key=lambda pair: pair[0])


def _hash(model):
    digest = hashlib.sha256()
    for name, parameter in _params(model):
        value = parameter.detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite frozen parameter")
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _vector(model):
    named = _params(model)
    names = [name for name, _ in named]
    shapes = [list(p.shape) for _, p in named]
    pieces = [torch.zeros_like(p, device="cpu", dtype=torch.float64).reshape(-1)
              if p.grad is None else p.grad.detach().cpu().double().reshape(-1) for _, p in named]
    vector = torch.cat(pieces).numpy() if pieces else np.zeros(0, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise FloatingPointError("Nonfinite parameter gradient")
    return vector, names, shapes


def _norm(v):
    value = float(np.linalg.norm(v))
    return value if value > 0 else None


def _cosine(a, b):
    na, nb = _norm(a), _norm(b)
    return None if na is None or nb is None else float(np.dot(a, b) / (na * nb))


def _group_norms(vector, names, shapes):
    offsets, start = {}, 0
    for name, shape in zip(names, shapes):
        end = start + int(np.prod(shape, dtype=np.int64))
        offsets[name] = start, end
        start = end
    result = {}
    for group, prefixes in GROUPS.items():
        parts = [vector[a:b] for name, (a, b) in offsets.items() if name.startswith(prefixes)]
        result[group] = _norm(np.concatenate(parts)) if parts else None
    return result


def _direction(models, pair, names, shapes):
    """pair=(source_id,target_id); v is theta[target]-theta[source]."""
    source, target = pair
    if source not in models or target not in models:
        raise KeyError("direction_pair must name existing source and target models")
    vectors = []
    for model_id in pair:
        params = dict(_params(models[model_id]))
        if list(params) != names or [list(params[n].shape) for n in names] != shapes:
            raise ValueError("direction-pair models must have matching parameters")
        vectors.append(torch.cat([params[n].detach().cpu().double().reshape(-1)
                                  for n in names]).numpy())
    vector = vectors[1] - vectors[0]
    norm = float(np.linalg.norm(vector))
    return (vector / norm if norm > 0 else None), norm


def _metrics(vector, names, shapes, direction):
    norm = _norm(vector)
    return {"norm": norm, "group_norms": _group_norms(vector, names, shapes),
            "dot_with_unit_direction": (None if norm is None or direction is None
                                        else float(np.dot(vector, direction)))}


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")[:72] or "model"


def audit_gradients(models: dict[str, torch.nn.Module], data: dict[str, torch.Tensor],
                    batches: list[list[int]], direction_pair: tuple[str, str],
                    outdir: Path, device: str = "cuda") -> dict[str, Any]:
    """Audit four fixed batches; direction_pair is (S1/source, F1/target).

    The unit vector is theta[F1]-theta[S1]. Interpret it only for a same-init
    pair; its dot product is local gradient alignment, not causal/global descent.
    """
    if len(batches) != 4 or any(len(row) != 8 for row in batches):
        raise ValueError("Expected exactly four fixed batches of eight maps")
    if not models or any(k not in data for k in ("x", "y", "mask")):
        raise ValueError("models and data['x','y','mask'] are required")
    if any(not torch.is_tensor(data[k]) or data[k].device.type != "cpu" for k in ("x", "y", "mask")):
        raise ValueError("x, y, and mask must be CPU tensors")
    if any(min(row) < 0 or max(row) >= len(data["x"]) for row in batches):
        raise IndexError("batch index outside the supplied data bank")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    helper, _ = _historical()
    ids = list(models)
    slugs = [_slug(i) for i in ids]
    if len(set(slugs)) != len(slugs):
        raise ValueError("model ids collide after filename sanitization")
    schemas = [([n for n, _ in _params(m)], [list(p.shape) for _, p in _params(m)])
               for m in models.values()]
    names, shapes = schemas[0]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("all models must share sorted parameter names and shapes")
    direction, direction_norm = _direction(models, direction_pair, names, shapes)
    before = {i: _hash(m) for i, m in models.items()}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = {"schema": "continuation_gradient_audit_v1", "training_updates": 0,
        "forward_steps": 64, "loss_every": 8, "gradient_horizons": {"k8": 8, "full64": 64},
        "forward_loss_tolerance": 1e-6, "device_type": device.type,
        "batch_indices": [[int(i) for i in row] for row in batches],
        "parameter_names": names, "parameter_shapes": shapes,
        "direction_pair": {"source_id": direction_pair[0], "target_id": direction_pair[1],
            "parameter_distance": direction_norm if direction_norm > 0 else None,
            "unit_vector_defined": direction is not None, "same_init_required": True},
        "models": {}}

    for model_id, slug in zip(ids, slugs):
        model = models[model_id]
        was_training = model.training
        model.to(device).eval()
        rows, grads = [], {"k8": [], "full64": []}
        try:
            for bi, indices in enumerate(batches):
                idx = torch.as_tensor(indices, dtype=torch.long)
                batch = {k: data[k].index_select(0, idx).to(device) for k in ("x", "y", "mask")}
                vectors, losses, calls = {}, {}, {}
                for key, horizon in (("k8", 8), ("full64", 64)):
                    model.zero_grad(set_to_none=True)
                    loss, state, trace = helper.backward_trajectory(
                        model, batch, gradient_horizon=horizon, steps=64, loss_every=8)
                    losses[key] = float(loss.detach().cpu())
                    vectors[key], got_names, got_shapes = _vector(model)
                    if got_names != names or got_shapes != shapes:
                        raise RuntimeError("parameter order changed")
                    grads[key].append(vectors[key]); calls[key] = trace["backward_calls"]
                    model.zero_grad(set_to_none=True)
                    del loss, state, trace
                difference = abs(losses["k8"] - losses["full64"])
                if difference > 1e-6:
                    raise AssertionError(f"forward loss mismatch on batch {bi}: {difference}")
                file = f"{slug}_batch{bi:02d}.npz"
                np.savez_compressed(outdir / file, gradient_k8=vectors["k8"],
                    gradient_full64=vectors["full64"], batch_indices=np.asarray(indices, dtype=np.int64),
                    parameter_names=np.asarray(names))
                n8, nf = np.linalg.norm(vectors["k8"]), np.linalg.norm(vectors["full64"])
                rows.append({"batch_index": bi,
                    "forward_loss": {"k8": losses["k8"], "full64": losses["full64"],
                                     "absolute_difference": difference},
                    "k8": _metrics(vectors["k8"], names, shapes, direction),
                    "full64": _metrics(vectors["full64"], names, shapes, direction),
                    "cosine_k8_full64": _cosine(vectors["k8"], vectors["full64"]),
                    "norm_ratio_k8_over_full64": float(n8 / nf) if nf > 0 else None,
                    "backward_calls": calls})
                del batch, vectors
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model.to("cpu"); model.zero_grad(set_to_none=True); model.train(was_training)
        after = _hash(model)
        if after != before[model_id]:
            raise AssertionError(f"frozen parameters changed for {model_id}")
        means = {k: np.mean(np.stack(v), axis=0, dtype=np.float64) for k, v in grads.items()}
        mean_file = f"{slug}_mean_gradients.npz"
        np.savez_compressed(outdir / mean_file, mean_gradient_k8=means["k8"],
                            mean_gradient_full64=means["full64"], parameter_names=np.asarray(names))
        n8, nf = np.linalg.norm(means["k8"]), np.linalg.norm(means["full64"])
        result["models"][str(model_id)] = {"parameter_sha256_before": before[model_id],
            "parameter_sha256_after": after, "batch_gradients": rows,
            "mean_gradient": {k: _metrics(v, names, shapes, direction) for k, v in means.items()},
            "mean_gradient_cosine_k8_full64": _cosine(means["k8"], means["full64"]),
            "mean_gradient_norm_ratio_k8_over_full64": float(n8 / nf) if nf > 0 else None,
            "gradient_files": [f"{slug}_batch{i:02d}.npz" for i in range(4)] + [mean_file]}
    result["parameters_unchanged"] = all(
        m["parameter_sha256_before"] == m["parameter_sha256_after"] for m in result["models"].values())
    (outdir / "metadata.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def sanity_check() -> dict[str, Any]:
    """Run a tiny n=8 CPU K8/full16 helper check with active output layers."""
    helper, tasks = _historical()
    rng = torch.random.get_rng_state()
    try:
        torch.manual_seed(91032)
        data = tasks.bank(8, 2, 91032, "cpu")
        model = helper.make_cell("ws_additive").cpu().eval()
        with torch.no_grad():
            for layer in (model.f_out, model.q_out, model.readout):
                torch.nn.init.normal_(layer.weight, std=0.03)
                torch.nn.init.normal_(layer.bias, std=0.01)
        before, losses, traces, vectors = _hash(model), {}, {}, {}
        for key, horizon in (("k8", 8), ("full16", 16)):
            assert all(p.grad is None for p in model.parameters())
            loss, _, traces[key] = helper.backward_trajectory(
                model, data, gradient_horizon=horizon, steps=16, loss_every=8)
            losses[key] = float(loss)
            vectors[key], _, _ = _vector(model)
            model.zero_grad(set_to_none=True)
            assert all(p.grad is None for p in model.parameters())
        if abs(losses["k8"] - losses["full16"]) > 1e-6:
            raise AssertionError("K8/full16 forward losses differ")
        if not np.any(vectors["k8"]) or not np.any(vectors["full16"]):
            raise AssertionError("active fixture produced zero gradients")
        if _hash(model) != before:
            raise AssertionError("helper modified frozen parameters")
        if traces["k8"]["backward_calls"] != 2 or traces["full16"]["backward_calls"] != 1:
            raise AssertionError("unexpected backward cadence")
        return {"status": "pass", "loss_difference": abs(losses["k8"] - losses["full16"]),
                "parameters_unchanged": True, "zero_grads_between_modes": True,
                "traces": traces}
    finally:
        torch.random.set_rng_state(rng)
