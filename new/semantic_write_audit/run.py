"""Frozen-checkpoint semantic-write interventions. No training or fitting."""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "new/trajectory_qualification"))
import common as C


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load_module("_semantic_factorial_cells", ROOT / "new/latent_factorial/cells.py")
PROTOCOL = "semantic_write_source_causal_v1"
CONDITIONS = ("natural", "source_parallel_t8", "source_parallel_t64",
              "source_orthogonal_t8", "oracle_solved_t8", "nullspace_zero_t8")
ACTIVATE = {name: (64 if name.endswith("t64") else 8) for name in CONDITIONS}
OPERATORS = {"natural": "natural", "source_parallel_t8": "source_negative_parallel",
             "source_parallel_t64": "source_negative_parallel",
             "source_orthogonal_t8": "source_orthogonal_matched",
             "oracle_solved_t8": "oracle_solved_negative_parallel",
             "nullspace_zero_t8": "nullspace_zero"}
STEPS = 256
TIMES = (8, 16, 32, 64, 128, 256)
FACTORIAL = ROOT / "runs/latent_factorial_20261005_01"
EVIDENCE = ROOT / "evidence/latent_factorial_20261005"
SEED4 = ROOT / "runs/streaming_carry_20261002_init2345/stream_K8_seed4.pt"
GROUPS = ("source", "paired_solved", "frontier_wrong", "nonfrontier_wrong")
DIAGNOSTIC_COLUMNS = ("count", "sum_margin", "sum_proposed_delta", "negative_count",
                      "proposed_crossing_count", "negative_write_mass", "positive_write_mass",
                      "sum_parallel_energy", "sum_nullspace_energy")
WRITE_RETRIES = 0


def safe_write(path, value):
    """Atomic JSON replacement with bounded retries for Windows reader locks."""
    global WRITE_RETRIES
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    deadline = time.monotonic() + 10
    delay = .05
    while True:
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
            return
        except PermissionError:
            WRITE_RETRIES += 1
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(.25, delay * 1.5)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bindings():
    names = {"new/semantic_write_audit/run.py", "new/semantic_write_audit/checks.py",
             "new/semantic_write_audit/check_io.py", "new/semantic_write_audit/EXECUTION_FIX.md",
             "new/semantic_write_audit/PROTOCOL.md", "tools/launch_semantic_write_audit.ps1",
             "new/latent_factorial/cells.py", "new/streaming_carry/stream_cells.py",
             "new/workspace_revision/revision_cells.py", "new/masked_medium/masked_cells.py",
             "new/nca_inertial_wind_tunnel/cells.py", "new/nca_inertial_wind_tunnel/tasks.py",
             "new/trajectory_qualification/common.py"}
    names.update(read(FACTORIAL / "manifest.json")["source_sha256"])
    return {name: C.sha(ROOT / name) for name in sorted(names)}


def model_plans():
    indexed = {(row["block"], row["arm"]): row for row in read(FACTORIAL / "perarm.json")}
    plans = []
    for block in range(8):
        for arm in M.ARMS:
            row = indexed[block, arm]
            plans.append({"name": f"block{block:02d}_{arm}", "block": block, "arm": arm,
                          "seed": row["initialization_seed"], "checkpoint": row["checkpoint"],
                          "checkpoint_sha256": row["checkpoint_sha256"],
                          "parameter_sha256": row["final_parameter_sha256"]})
    historical = read(SEED4.with_suffix(".json"))
    published = read(ROOT / "STREAMING_CARRY_PUBLICATION_MANIFEST.json")
    plans.append({"name": "historical_seed4", "block": None, "arm": "historical_seed4",
                  "seed": 4, "checkpoint": SEED4.relative_to(ROOT).as_posix(),
                  "checkpoint_sha256": published["checkpoint_sha256"][SEED4.name],
                  "parameter_sha256": historical["final_parameter_sha256"]})
    return plans


def checkpoint_path(plan):
    return ROOT / plan["checkpoint"] if plan["block"] is None else FACTORIAL / plan["checkpoint"]


def load_model(plan, device="cuda"):
    path = checkpoint_path(plan)
    assert C.sha(path) == plan["checkpoint_sha256"], "Checkpoint binding mismatch"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["completed_updates"] == 300
    model = C.StreamingCell() if plan["block"] is None else M.make_model(plan["arm"], plan["seed"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert C.tensor_hash(model.state_dict()) == plan["parameter_sha256"]
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def data_banks():
    expected = read(FACTORIAL / "manifest.json")["data_sha256"]
    banks = {}
    for size in (32, 64):
        with np.load(EVIDENCE / f"banks/evaluation{size}.npz", allow_pickle=False) as values:
            bank = {key: torch.from_numpy(values[key].copy()) for key in values.files}
        assert C.tensor_hash(bank) == expected[f"evaluation{size}"], "Data binding mismatch"
        banks[size] = bank
    return banks


def parallel(q, weight):
    squared = weight.square().sum()
    assert weight.numel() > 1
    projection = F.conv2d(q, weight)
    return projection * weight.reshape(1, -1, 1, 1) / squared


def intervention(q, z, readout_weight, alpha, x, y, condition, bias=None):
    """Modify only Q; source operators do not use y in their gates."""
    if condition == "natural":
        zero = q[:, :1].new_zeros(q[:, :1].shape)
        return q, {"active": zero.bool(), "removed_energy": zero}
    w = readout_weight.reshape(1, q.shape[1], 1, 1)
    q_parallel = parallel(q, w)
    q_perp = q - q_parallel
    logits = F.conv2d(z, w, bias)
    proposed = F.conv2d(q, w)
    source_sign = x[:, 1:2] - x[:, 2:3]
    source = source_sign != 0
    source_gate = source & ((logits >= 0) == (source_sign > 0)) & (source_sign * proposed < 0)
    zero = torch.zeros_like(proposed, dtype=torch.bool)
    if condition == "source_negative_parallel":
        active = source_gate
        modified = q - active * q_parallel
    elif condition == "oracle_solved_negative_parallel":
        sign = 2 * y - 1
        active = x[:, :1].bool() & ((logits >= 0) == (y >= .5)) & (sign * proposed < 0)
        modified = q - active * q_parallel
    elif condition == "nullspace_zero":
        modified, active = q_parallel, torch.ones_like(zero)
    elif condition == "source_orthogonal_matched":
        active = source_gate
        norm = q_perp.square().sum(1, keepdim=True).sqrt()
        basis = F.one_hot(w.flatten().abs().argmin(), num_classes=w.numel()).to(q.dtype)
        vector = w.flatten()
        basis = basis - vector * (basis @ vector) / vector.square().sum()
        basis = (basis / basis.norm()).reshape(1, -1, 1, 1)
        unit = torch.where(norm > 1e-8, q_perp / norm.clamp_min(1e-8), basis)
        amount = q_parallel.square().sum(1, keepdim=True).sqrt()
        modified = q - active * amount * unit
    else:
        raise ValueError(condition)
    return modified, {"active": active, "removed_energy": (modified-q).square().sum(1, keepdim=True),
                      "parallel_energy": q_parallel.square().sum(1, keepdim=True),
                      "nullspace_energy": q_perp.square().sum(1, keepdim=True)}


@torch.no_grad()
def step_with_intervention(model, state, x, y, condition):
    captured = {}

    def hook(module, args, q):
        changed, stats = intervention(q, state[1], model.readout.weight, model.alpha,
                                      x, y, condition, model.readout.bias)
        captured.update(q=q, stats=stats)
        return changed

    handle = model.q_out.register_forward_hook(hook)
    try:
        next_state = model.step(state, x)
    finally:
        handle.remove()
    return next_state, captured["q"], captured["stats"]


def neighbor_solved(good):
    padded = F.pad(good.float(), (1, 1, 1, 1))
    return ((padded[..., :-2, 1:-1] + padded[..., 2:, 1:-1]
             + padded[..., 1:-1, :-2] + padded[..., 1:-1, 2:]) > 0)


def group_masks(good, bank):
    changed = bank["changed"].bool()
    frontier = ~good & changed & neighbor_solved(good & bank["mask"].bool())
    return (changed & (bank["distance"] == 0), changed & good,
            frontier, changed & ~good & ~frontier)


def write_diagnostics(q, logits, y, model, masks):
    sign = 2*y-1
    margin = sign*logits
    delta = model.alpha * sign * F.conv2d(q, model.readout.weight)
    correct_now = (logits >= 0) == (y >= .5)
    proposed_wrong = ((logits + model.alpha * F.conv2d(q, model.readout.weight)) >= 0) != (y >= .5)
    qp = parallel(q, model.readout.weight)
    columns = (torch.ones_like(margin), margin, delta, (delta < 0).float(),
               (correct_now & proposed_wrong).float(),
               (-delta).clamp_min(0), delta.clamp_min(0),
               qp.square().sum(1, keepdim=True), (q-qp).square().sum(1, keepdim=True))
    return torch.stack([torch.stack([(value * mask).sum((1, 2, 3))
                                     for value in columns], dim=-1) for mask in masks])


@torch.no_grad()
def trace(model, data_cpu, condition, steps=256, diagnostics=True):
    bank = {key: value.cuda() for key, value in data_cpu.items()}
    x, xf, y, yf = (bank[key] for key in ("x", "x_flip", "y", "y_flip"))
    a, b = model.initial(x), model.initial(xf)
    n, _, size, _ = x.shape
    correct = torch.empty((steps+1, n, size, size), dtype=torch.bool, device=x.device)
    originals = torch.empty_like(correct) if condition == "natural" else None
    flips = torch.empty_like(correct) if condition == "natural" else None
    diag = torch.empty((steps, 4, n, len(DIAGNOSTIC_COLUMNS)), device=x.device) if diagnostics else None
    intervention_stats = torch.zeros((steps, n, 2), device=x.device)
    sampled_margins = []
    changed_mask = bank["changed"][:, 0].bool()
    source_mask = changed_mask & (bank["distance"][:, 0] == 0)
    source_margins, source_proposals = [], []
    assert int(source_mask.sum()) == n, "One changed-component source per map required"
    finite = torch.ones((), dtype=torch.bool, device=x.device)
    accounting_max = torch.zeros((), device=x.device)
    for t in range(steps+1):
        la, lb = model.logits(a), model.logits(b)
        ga, gb = (la >= 0) == (y >= .5), (lb >= 0) == (yf >= .5)
        good = ga & gb
        correct[t] = good[:, 0]
        if originals is not None:
            originals[t], flips[t] = ga[:, 0], gb[:, 0]
        if t in TIMES:
            sampled_margins.append(torch.minimum((2*y-1)*la, (2*yf-1)*lb)[:, 0][changed_mask])
        if condition == "natural":
            source_margins.append(torch.stack(((2*y-1)*la, (2*yf-1)*lb))[:, :, 0][:, source_mask])
        if t == steps:
            break
        operator = OPERATORS[condition] if t >= ACTIVATE[condition] else "natural"
        an, qa, sa = step_with_intervention(model, a, x, y, operator)
        bn, qb, sb = step_with_intervention(model, b, xf, yf, operator)
        if diagnostics:
            masks = group_masks(good, bank)
            diag[t] = write_diagnostics(qa, la, y, model, masks) + write_diagnostics(qb, lb, yf, model, masks)
        if condition == "natural":
            increments = torch.stack((model.alpha*(2*y-1)*F.conv2d(qa, model.readout.weight),
                                      model.alpha*(2*yf-1)*F.conv2d(qb, model.readout.weight)))
            source_proposals.append(increments[:, :, 0][:, source_mask])
        intervention_stats[t, :, 0] = sa["active"].sum((1, 2, 3)) + sb["active"].sum((1, 2, 3))
        intervention_stats[t, :, 1] = sa["removed_energy"].sum((1, 2, 3)) + sb["removed_energy"].sum((1, 2, 3))
        if condition == "natural":
            ea = (model.logits(an) - la - model.alpha * F.conv2d(qa, model.readout.weight)).abs().amax()
            eb = (model.logits(bn) - lb - model.alpha * F.conv2d(qb, model.readout.weight)).abs().amax()
            accounting_max = torch.maximum(accounting_max, torch.maximum(ea, eb))
        finite = finite & torch.stack([torch.isfinite(v).all() for v in (*an, *bn)]).all()
        a, b = an, bn
    assert bool(finite), "Nonfinite forward state"
    result = {"correct": correct.cpu().numpy(),
              "sampled_margin_times": np.asarray([t for t in TIMES if t <= steps]),
              "sampled_margins": torch.stack(sampled_margins).cpu().numpy(),
              "intervention_stats": intervention_stats.cpu().numpy(),
              "logit_accounting_max_absolute_error": float(accounting_max)}
    if originals is not None:
        result.update(original_correct=originals.cpu().numpy(), flipped_correct=flips.cpu().numpy())
        result.update(source_margins=torch.stack(source_margins).cpu().numpy(),
                      source_proposed_increments=torch.stack(source_proposals).cpu().numpy())
    if diag is not None:
        result["write_diagnostics"] = diag.cpu().numpy()
    return result


def summarize(result, bank, reference):
    good = result["correct"]
    changed = bank["changed"][:, 0].numpy().astype(bool)
    distance = bank["distance"][:, 0].numpy()
    selections = {"strict_far": changed & (distance > 16) & (distance < 32),
                  "off_source": changed & (distance > 0), "source": changed & (distance == 0),
                  "all_changed": changed}
    endpoints, per_map = {}, []
    for name, mask in selections.items():
        base = good[64] & mask
        natural_base = reference[64] & mask
        once = good.any(0) & mask
        ever_wrong_after_correct = ((np.maximum.accumulate(good, axis=0) & ~good).any(0) & mask)
        ratio = lambda num, den: float(num/den) if den else None
        row = {"pixels": int(mask.sum()),
               "coverage": {str(t): ratio((good[t] & mask).sum(), mask.sum()) for t in (64, 128, 256)},
               "retention_own64_to256": ratio((good[256] & base).sum(), base.sum()),
               "retention_natural64_to256": ratio((good[256] & natural_base).sum(), natural_base.sum()),
               "gained_vs_natural64": int((good[256] & mask & ~reference[64]).sum()),
               "lost_vs_natural64": int((~good[256] & natural_base).sum()),
               "ever_regressed_fraction": ratio(ever_wrong_after_correct.sum(), once.sum())}
        endpoints[name] = row
        for i in range(len(mask)):
            for t in (64, 128, 256):
                per_map.append({"selection": name, "map": i, "time": t,
                                "correct": int((good[t, i] & mask[i]).sum()),
                                "pixels": int(mask[i].sum()),
                                "coverage": ratio((good[t, i] & mask[i]).sum(), mask[i].sum())})
    return endpoints, per_map


def replay(plan, size, result):
    if plan["block"] is None:
        return {"status": "NEW_COMMON_COHORT_REFERENCE", "historical_bank_replay": False}
    path = EVIDENCE / f"block{plan['block']:02d}" / plan["arm"] / "evaluation" / f"{plan['arm']}_size{size}.npz"
    counts = {}
    with np.load(path, allow_pickle=False) as old:
        for key in ("correct", "original_correct", "flipped_correct"):
            counts[key] = int(np.count_nonzero(old[key] != result[key]))
    assert sum(counts.values()) == 0, f"Natural replay failed: {plan['name']}/{size}/{counts}"
    return {"status": "PASS", "mismatched_bits": counts, "saved_trace_sha256": C.sha(path)}


def save_unit(out, plan, size, condition, result, endpoints, per_map):
    folder = out / plan["name"] / f"size{size}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{condition}.npz"
    arrays = {"correct_packed": np.packbits(result["correct"].reshape(-1), bitorder="little"),
              "correct_shape": np.asarray(result["correct"].shape),
              "sampled_margin_times": result["sampled_margin_times"],
              "sampled_margins": result["sampled_margins"],
              "intervention_stats": result["intervention_stats"]}
    if "write_diagnostics" in result:
        arrays["write_diagnostics"] = result["write_diagnostics"]
    if "source_margins" in result:
        arrays["source_margins"] = result["source_margins"]
        arrays["source_proposed_increments"] = result["source_proposed_increments"]
    np.savez_compressed(path, **arrays)
    safe_write(folder / f"{condition}.json", {"protocol": PROTOCOL, "model": plan["name"],
        "size": size, "condition": condition, "endpoints": endpoints,
        "logit_accounting_max_absolute_error": result["logit_accounting_max_absolute_error"],
        "arrays_sha256": C.sha(path), "diagnostic_groups": GROUPS,
        "sampled_margin_layout": "time x changed pixels in flattened map,row,column order",
        "source_array_layout": "time x world(original,flip) x map",
        "diagnostic_columns": DIAGNOSTIC_COLUMNS,
        "intervention_columns": ["active_branch_cells", "removed_Q_squared_norm"]})
    with (folder / f"{condition}_per_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_map[0]))
        writer.writeheader()
        writer.writerows(per_map)


def aggregate(rows):
    primary_rows = {(r["block"], r["condition"]): r for r in rows
                    if r["arm"] == "native" and r["size"] == 32}
    diffs = []
    for block in range(8):
        get = lambda condition: primary_rows[block, condition]["endpoints"]["strict_far"]["coverage"]["256"]
        diffs.append({"block": block, "source_minus_natural": get("source_parallel_t8")-get("natural"),
                      "source_minus_orthogonal": get("source_parallel_t8")-get("source_orthogonal_t8")})
    natural_delta = float(np.mean([r["source_minus_natural"] for r in diffs]))
    control_delta = float(np.mean([r["source_minus_orthogonal"] for r in diffs]))
    positives = sum(r["source_minus_natural"] > 0 for r in diffs)
    signal = natural_delta >= .05 and control_delta >= .05 and positives >= 6
    control = [r for r in rows if r["arm"] == "historical_seed4" and r["size"] == 32 and r["condition"] == "natural"][0]
    retention = control["endpoints"]["all_changed"]["retention_own64_to256"]
    reference_ok = (control["endpoints"]["strict_far"]["coverage"]["256"] >= .8
                    and retention is not None and retention >= .95)
    grouped = []
    for arm in (*M.ARMS, "historical_seed4"):
        for size in (32, 64):
            for condition in CONDITIONS:
                entries = [r for r in rows if (r["arm"], r["size"], r["condition"]) == (arm, size, condition)]
                fields = {"strict256": [r["endpoints"]["strict_far"]["coverage"]["256"] for r in entries],
                          "offsource256": [r["endpoints"]["off_source"]["coverage"]["256"] for r in entries],
                          "source256": [r["endpoints"]["source"]["coverage"]["256"] for r in entries],
                          "offsource_retention_natural64": [r["endpoints"]["off_source"]["retention_natural64_to256"] for r in entries]}
                grouped.append({"arm": arm, "size": size, "condition": condition, "models": len(entries),
                    **{k: {"mean": float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else None,
                           "defined_models": sum(v is not None for v in values)} for k, values in fields.items()}})
    return {"protocol": PROTOCOL, "status": "COMPLETE", "completed_units": len(rows),
            "training": False, "optimizer_updates": 0, "primary": {
                "native_blocks": 8, "size": 32, "metric": "strict16<d<32 paired coverage atT256",
                "source_minus_natural_mean": natural_delta, "source_minus_orthogonal_mean": control_delta,
                "positive_blocks": positives, "per_block": diffs,
                "verdict": "EXPLORATORY_SOURCE_DRIVER_SIGNAL" if signal else "NO_PRIMARY_SOURCE_DRIVER_SIGNAL"},
            "historical_reference_qualified_on_common_cohort": reference_ok,
            "grouped": grouped, "claim_boundary": "Exploratory source perturbation, not training reliability or upstream Q root-cause proof; protected/oracle retention is not evidence."}


def report(out, summary):
    lines = ["# Semantic-write source audit", "", "Status: "+summary["status"], "",
             "Zero training; protected-source and oracle retention are manipulation checks."]
    if summary["status"] == "COMPLETE":
        p = summary["primary"]
        lines += ["", "Primary: "+p["verdict"],
                  f"Source-natural mean={p['source_minus_natural_mean']:.4f}; source-orthogonal mean={p['source_minus_orthogonal_mean']:.4f}; positive blocks={p['positive_blocks']}/8.",
                  "", "|Arm|Size|Condition|Strict T256|Off-source T256|Source T256|", "|---|---:|---|---:|---:|---:|"]
        for row in summary["grouped"]:
            lines.append(f"|{row['arm']}|{row['size']}|{row['condition']}|{row['strict256']['mean']:.4f}|{row['offsource256']['mean']:.4f}|{row['source256']['mean']:.4f}|")
    (out / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def check(out):
    assert not out.exists()
    torch.set_num_threads(2)
    checks = load_module("_semantic_checks", Path(__file__).with_name("checks.py"))
    cpu = checks.run_checks(intervention, step_with_intervention)
    io_tests = load_module("_semantic_io_checks", Path(__file__).with_name("check_io.py"))
    io = io_tests.check(safe_write, out.with_suffix(".io_fixture"))
    C.setup_backend()
    banks = data_banks()
    plan = model_plans()[0]
    model = load_model(plan)
    model_hash = C.tensor_hash(model.state_dict())
    subset = {key: value[:4] for key, value in banks[32].items()}
    measured = {}
    for condition in ("natural", "source_parallel_t8", "source_orthogonal_t8", "oracle_solved_t8", "nullspace_zero_t8"):
        torch.cuda.synchronize()
        tick = time.monotonic()
        result = trace(model, subset, condition, steps=16, diagnostics=condition == "natural")
        torch.cuda.synchronize()
        measured[condition] = time.monotonic()-tick
        assert result["correct"].shape == (17, 4, 32, 32)
        assert np.isfinite(result["sampled_margins"]).all()
        if condition == "natural":
            natural_prefix = result["correct"][:9].copy()
            assert result["logit_accounting_max_absolute_error"] < 1e-3
        else:
            assert np.array_equal(result["correct"][:9], natural_prefix)
    assert C.tensor_hash(model.state_dict()) == model_hash
    actual_shape = {}
    for size in (32, 64):
        torch.cuda.synchronize()
        tick = time.monotonic()
        result = trace(model, banks[size], "natural", steps=256, diagnostics=True)
        torch.cuda.synchronize()
        actual_shape[str(size)] = {"maps": 32, "steps": 256,
                                  "seconds": time.monotonic()-tick,
                                  "logit_accounting_error": result["logit_accounting_max_absolute_error"],
                                  "natural_replay": replay(plan, size, result)}
    out.parent.mkdir(parents=True, exist_ok=True)
    safe_write(out, {"protocol": PROTOCOL, "status": "PASS", "checked_utc": C.now(),
                  "source_sha256": bindings(), "cpu": cpu, "cuda_smoke_seconds": measured,
                  "io": io,
                  "actual_shape_natural_smoke": actual_shape,
                  "training": False, "optimizer_updates": 0,
                  "runtime_limit_enforced": False, "scope": "CPU projection/wiring plus short intervention smoke and actual32-map256-step natural replay; not intervention efficacy."})
    print(json.dumps({"status": "PASS", "qualification": out.relative_to(ROOT).as_posix(),
                      "smoke_seconds": measured, "actual_shape": actual_shape}), flush=True)


def unpack_correct(path):
    with np.load(path, allow_pickle=False) as values:
        shape = tuple(int(v) for v in values["correct_shape"])
        bits = np.unpackbits(values["correct_packed"], bitorder="little", count=int(np.prod(shape)))
    return bits.reshape(shape).astype(bool)


def import_completed(parent, out, plans, banks):
    """Copy only completed, hash-bound units; preserve the interrupted run."""
    assert parent.is_relative_to(ROOT / "runs") and parent != out
    previous = read(parent / "manifest.json")
    assert previous["protocol"] == PROTOCOL and previous["plans"] == plans
    assert previous["data_sha256"] == {str(k): C.tensor_hash(v) for k, v in banks.items()}
    old_tree = ast.parse((parent / "source/new/semantic_write_audit/run.py").read_text(encoding="utf-8"))
    new_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    scientific_functions = ("parallel", "intervention", "step_with_intervention", "neighbor_solved",
                            "group_masks", "write_diagnostics", "trace", "summarize", "aggregate")
    get_functions = lambda tree: {node.name: ast.dump(node, include_attributes=False)
                                  for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    old_functions, new_functions = get_functions(old_tree), get_functions(new_tree)
    assert all(old_functions[name] == new_functions[name] for name in scientific_functions), "Scientific function changed"
    for name, digest in previous["source_sha256"].items():
        if name != "new/semantic_write_audit/run.py":
            assert C.sha(ROOT / name) == digest, f"Scientific dependency changed: {name}"
    rows = read(parent / "perunit.json")
    expected = [(p["name"], size, c) for p in plans for size in banks for c in CONDITIONS]
    keys = [(r["model"], r["size"], r["condition"]) for r in rows]
    assert keys == expected[:len(rows)] and len(set(keys)) == len(rows), "Non-prefix or duplicate imports"
    file_hashes = {}
    for row in rows:
        relative = Path(row["model"]) / f"size{row['size']}"
        source_folder, destination = parent / relative, out / relative
        destination.mkdir(parents=True, exist_ok=True)
        meta = read(source_folder / f"{row['condition']}.json")
        assert (meta["model"], meta["size"], meta["condition"]) == (row["model"], row["size"], row["condition"])
        assert meta["endpoints"] == row["endpoints"]
        assert C.sha(source_folder / f"{row['condition']}.npz") == meta["arrays_sha256"]
        for suffix in (".npz", ".json", "_per_map.csv"):
            source = source_folder / f"{row['condition']}{suffix}"
            target = destination / source.name
            shutil.copyfile(source, target)
            digest = C.sha(source)
            assert C.sha(target) == digest
            file_hashes[target.relative_to(out).as_posix()] = digest
    replays = read(parent / "replay.json")
    assert all(record["status"] == "PASS" for record in replays)
    assert {(r["model"], r["size"]) for r in replays} == {(r["model"], r["size"]) for r in rows if r["condition"] == "natural"}
    note = {"status": "PASS", "imported_units": len(rows),
            "source_run": parent.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": C.sha(parent / "manifest.json"),
            "source_perunit_sha256": C.sha(parent / "perunit.json"),
            "source_replay_sha256": C.sha(parent / "replay.json"),
            "scientific_function_AST_equivalence": list(scientific_functions),
            "file_sha256": file_hashes, "prior_error": read(parent / "status.json")["error"]}
    safe_write(out / "import_validation.json", note)
    return rows, replays


def run(out, qualification, resume_from=None):
    assert not out.exists()
    qual = read(qualification)
    assert qual["status"] == "PASS" and qual["protocol"] == PROTOCOL
    hashes = bindings()
    assert qual["source_sha256"] == hashes
    C.setup_backend()
    plans, banks = model_plans(), data_banks()
    out.mkdir(parents=True)
    started = time.monotonic()
    manifest = {"protocol": PROTOCOL, "started_utc": C.now(), "pid": os.getpid(),
                "host": os.environ.get("COMPUTERNAME"), "gpu": torch.cuda.get_device_name(),
                "command": [sys.executable, *sys.argv], "torch": str(torch.__version__),
                "training": False, "optimizer_updates": 0, "expected_units": 396,
                "continuous_monitoring": False, "runtime_limit_enforced": False,
                "source_sha256": hashes, "qualification_sha256": C.sha(qualification),
                "data_sha256": {str(k): C.tensor_hash(v) for k, v in banks.items()},
                "plans": plans, "conditions": CONDITIONS,
                "git_review_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    manifest["resume_from"] = resume_from.relative_to(ROOT).as_posix() if resume_from else None
    safe_write(out / "manifest.json", manifest)
    safe_write(out / "config.json", {"protocol": PROTOCOL, "steps": STEPS, "conditions": CONDITIONS,
        "activation_after_state": ACTIVATE, "maps_each_size": 32, "sizes": [32, 64],
        "native_primary_blocks": list(range(8)), "independent_unit": "paired training block",
        "diagnostic_groups": GROUPS, "diagnostic_columns": DIAGNOSTIC_COLUMNS})
    (out / "banks").mkdir()
    for size in banks:
        shutil.copyfile(EVIDENCE / f"banks/evaluation{size}.npz", out / f"banks/evaluation{size}.npz")
    for name in hashes:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    rows, replays = [], []
    if resume_from is not None:
        rows, replays = import_completed(resume_from, out, plans, banks)
    imported_units = len(rows)
    completed = {(row["model"], row["size"], row["condition"]) for row in rows}
    safe_write(out / "perunit.json", rows)
    safe_write(out / "replay.json", replays)

    def progress(phase, **extra):
        value = {"status": "RUNNING", "protocol": PROTOCOL, "phase": phase,
                 "pid": os.getpid(), "expected_units": 396, "completed_units": len(rows),
                 "elapsed_seconds": time.monotonic()-started, "updated_utc": C.now(), **extra}
        safe_write(out / "status.json", value)
        print(json.dumps(value), flush=True)

    progress("starting")
    try:
        for plan in plans:
            if all((plan["name"], size, condition) in completed for size in banks for condition in CONDITIONS):
                continue
            model = load_model(plan)
            for size, bank in banks.items():
                reference = None
                for condition in CONDITIONS:
                    if (plan["name"], size, condition) in completed:
                        if condition == "natural":
                            reference = unpack_correct(out / plan["name"] / f"size{size}/natural.npz")
                        continue
                    progress("inference", model=plan["name"], size=size, condition=condition)
                    tick = time.monotonic()
                    result = trace(model, bank, condition, diagnostics=condition == "natural")
                    if condition == "natural":
                        reference = result["correct"].copy()
                        replays.append({"model": plan["name"], "size": size, **replay(plan, size, result)})
                        safe_write(out / "replay.json", replays)
                    assert reference is not None
                    assert np.array_equal(result["correct"][:ACTIVATE[condition]+1], reference[:ACTIVATE[condition]+1]) if condition != "natural" else True
                    endpoints, per_map = summarize(result, bank, reference)
                    save_unit(out, plan, size, condition, result, endpoints, per_map)
                    rows.append({"model": plan["name"], "block": plan["block"], "arm": plan["arm"],
                                 "size": size, "condition": condition, "endpoints": endpoints,
                                 "seconds": time.monotonic()-tick})
                    safe_write(out / "perunit.json", rows)
                    progress("unit_complete", model=plan["name"], size=size, condition=condition)
                    del result
                del reference
            assert C.tensor_hash(model.state_dict()) == plan["parameter_sha256"]
            del model
            gc.collect()
            torch.cuda.empty_cache()
        assert len(rows) == 396 and bindings() == hashes
        summary = aggregate(rows)
        summary["elapsed_seconds"] = time.monotonic()-started
        summary["timing_scope"] = "Current continuation worker only; imported units were computed in interrupted source run"
        summary["imported_units"] = imported_units
        summary["newly_computed_units"] = len(rows)-imported_units
        summary["json_replacement_retry_count"] = WRITE_RETRIES
        safe_write(out / "summary.json", summary)
        report(out, summary)
        safe_write(out / "status.json", {"status": "COMPLETE", "protocol": PROTOCOL,
                "completed_units": len(rows), "expected_units": 396,
                "elapsed_seconds": time.monotonic()-started, "finished_utc": C.now(),
                "training": False, "optimizer_updates": 0})
    except Exception as error:
        safe_write(out / "status.json", {"status": "ERROR", "protocol": PROTOCOL,
                "completed_units": len(rows), "expected_units": 396,
                "error": str(error), "traceback": traceback.format_exc(), "updated_utc": C.now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    assert out.is_relative_to(ROOT / ("analyses" if args.check else "runs"))
    if args.check:
        check(out)
    else:
        assert args.qualification
        run(out, ROOT / args.qualification, (ROOT / args.resume_from).resolve() if args.resume_from else None)


if __name__ == "__main__":
    main()
