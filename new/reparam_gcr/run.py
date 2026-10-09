"""One new K8 arm: exact raw-state lift with learned endpoint projection."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "new/learnable_gcr"
REFERENCE = ROOT / "evidence/learnable_gcr_20261009_01"
REFERENCE_CHECKPOINT = ROOT / "runs/learnable_gcr_20261009_01/gcr_k64/checkpoints/u150.pt"
sys.path.insert(0, str(OLD))
import run as base  # noqa: E402

spec = importlib.util.spec_from_file_location("_reparam_gcr_cells", HERE / "cells.py")
cell_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cell_module
spec.loader.exec_module(cell_module)
ReparamGCR = cell_module.ReparamGCR
OrderedCell = cell_module.OrderedCell

CONFIG = copy.deepcopy(base.CONFIG)
CONFIG.update(protocol="reparam_gcr_v0", arms=[["reparam_gcr_k8", "reparam_gcr", 8]],
              reference_evidence=REFERENCE.relative_to(ROOT).as_posix(),
              reference_commit="58aa2720c39c57de565d4086d7fca3eeb80cb22e",
              state_scalars_per_node=73, learned_parameters=5921,
              primary_tolerance=.02, minimum_descriptive_improvement=.10)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def source_hashes():
    paths = [HERE / n for n in ("run.py", "cells.py", "PROTOCOL.md")]
    paths += [OLD / n for n in ("run.py", "cells.py", "tasks.py", "PROTOCOL.md")]
    paths += [ROOT / "tools" / n for n in ("start_protected_job.ps1", "protected_job_worker.ps1")]
    return {p.relative_to(ROOT).as_posix(): base.sha(p) for p in paths}


def reference_inputs():
    manifest, aggregate = read(REFERENCE / "manifest.json"), read(REFERENCE / "aggregate.json")
    assert aggregate["status"] == "COMPLETE" and manifest["config"] == base.CONFIG
    for rel, digest in manifest["data_hashes"].items():
        assert base.sha(REFERENCE / rel) == digest, rel
    for rel, digest in manifest["source_hashes"].items():
        assert base.sha(ROOT / rel) == digest, rel
    banks = {n: dict(np.load(REFERENCE / "banks" / f"{n}.npz", allow_pickle=False)) for n in CONFIG["banks"]}
    schedule = np.load(REFERENCE / "banks/schedule.npy", allow_pickle=False)
    r2 = {n: aggregate["results"][n]["evaluations"]["heldout"]["horizons"]["T64"]["r2"] for n in ("gcr_k8", "gcr_k64")}
    assert r2["gcr_k64"] >= .5
    return manifest, banks, schedule, r2


def forward(model, batch, credit):
    state = model.initialize(batch)
    with torch.no_grad():
        for _ in range(64 - credit):
            state = model.step(state, batch)
    state = model.detach(state)
    for _ in range(credit):
        state = model.step(state, batch)
    return model.predict(state)


def gradient(model, batch, credit, old=False):
    model.zero_grad(set_to_none=True)
    prediction = base.forward_credit(model, batch, 64, credit) if old else forward(model, batch, credit)
    loss = (prediction - batch["y"]).square().mean()
    loss.backward()
    groups = {"phi": [], "interpreter": [], "all": []}
    for name, param in model.named_parameters():
        assert param.grad is not None and bool(torch.isfinite(param.grad).all()), name
        value = param.grad.detach().flatten().clone()
        groups[name.split(".")[0]].append(value)
        groups["all"].append(value)
    return float(loss.detach()), {k: torch.cat(v) for k, v in groups.items()}


def agreement(actual, reference, scale=1.):
    a, b = actual.double(), reference.double()
    norm = float(b.norm())
    assert norm > 1e-14 and float(a.norm()) > 1e-14
    return {"cosine": float(torch.dot(a, b) / (a.norm() * b.norm())),
            "norm_ratio": float(a.norm() / b.norm()),
            "relative_l2_error": float((a - scale * b).norm() / b.norm()), "expected_scale": scale}


def pair_check(batch, parameters, dtype, device):
    old = OrderedCell("gcr", CONFIG["init_seed"], .5).to(device=device, dtype=dtype)
    new = ReparamGCR(CONFIG["init_seed"], .5).to(device=device, dtype=dtype)
    if parameters is not None:
        old.load_state_dict(parameters)
        new.load_state_dict(parameters)
    assert new.param_counts()["total"] == 5921
    assert list(old.state_dict()) == list(new.state_dict())
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    tol = {"atol": 1e-9, "rtol": 1e-9} if dtype == torch.float64 else {"atol": 1e-5, "rtol": 1e-4}
    max_state, max_prediction = 0., 0.
    with torch.no_grad():
        s, lifted = old.initialize(batch), new.initialize(batch)
        p = old.feature(batch["x"])
        for _ in range(64):
            s, lifted = old.step(s, batch, p=p), new.step(lifted, batch)
            projected = new.project_workspace(lifted.workspace)
            torch.testing.assert_close(projected, s.workspace, **tol)
            assert torch.equal(s.count, lifted.count)
            torch.testing.assert_close(s.z, lifted.z, **tol)
            torch.testing.assert_close(old.predict(s), new.predict(lifted), **tol)
            max_state = max(max_state, float((projected - s.workspace).abs().max()))
            max_prediction = max(max_prediction, float((old.predict(s) - new.predict(lifted)).abs().max()))
    old_loss, old_full = gradient(old, batch, 64, old=True)
    new_loss, new_full = gradient(new, batch, 64)
    _, new_short = gradient(new, batch, 8)
    assert np.isclose(old_loss, new_loss, **tol)
    full = {k: agreement(new_full[k], old_full[k]) for k in old_full}
    short = {k: agreement(new_short[k], old_full[k], 255 / 256) for k in old_full}
    full_tol, short_tol, cosine = (1e-8, 1e-7, .999999) if dtype == torch.float64 else (1e-4, 1e-4, .99999)
    assert all(v["relative_l2_error"] <= full_tol for v in full.values()), full
    assert all(v["relative_l2_error"] <= short_tol and v["cosine"] >= cosine for v in short.values()), short
    return {"dtype": str(dtype), "loss": old_loss, "steps_checked": 64,
            "max_projected_state_absolute_error": max_state,
            "max_prediction_absolute_error": max_prediction,
            "full_gradient_equivalence": full, "new_k8_vs_old_full_gradient": short}


def train_update(model, opt, batch):
    opt.zero_grad(set_to_none=True)
    loss = (forward(model, batch, 8) - batch["y"]).square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite endpoint loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite updated parameter")
    return float(loss.detach()), float(norm.detach())


def check(out, device):
    assert not out.exists(), "Use a fresh qualification directory"
    out.mkdir(parents=True)
    manifest, banks, schedule, r2 = reference_inputs()
    binding = read(REFERENCE / "gcr_k64/checkpoints/u150.binding.json")
    assert base.sha(REFERENCE_CHECKPOINT) == binding["checkpoint_sha256"]
    saved = torch.load(REFERENCE_CHECKPOINT, map_location="cpu", weights_only=False)
    assert saved["next_update"] == 151 and saved["source_hashes"] == manifest["source_hashes"]
    cpu = base.sub(base.bank_to_torch(banks["train"], "cpu"), slice(0, 4))
    cpu = {k: v.double() if v.is_floating_point() else v for k, v in cpu.items()}
    checks = {"initial_cpu_float64": pair_check(cpu, None, torch.float64, "cpu"),
              "trained_reference_cpu_float64": pair_check(cpu, saved["model"], torch.float64, "cpu")}
    b = base.sub(base.bank_to_torch(banks["train"], device), torch.as_tensor(schedule[0], device=device))
    checks["initial_cuda_float32"] = pair_check(b, None, torch.float32, device)
    model = ReparamGCR(CONFIG["init_seed"], .5).to(device)
    one = base.sub(base.bank_to_torch(banks["train"], device), slice(0, 1))
    one["x"] = one["x"].detach().clone().requires_grad_(True)
    forward(model, one, 8).sum().backward()
    n = int(banks["train"]["lengths"][0])
    xy = banks["train"]["path"][0, :n]
    ids = torch.as_tensor(xy[:, 0] * 8 + xy[:, 1], device=device, dtype=torch.long)
    assert torch.count_nonzero(one["x"].grad[0, ids[:-8]]) == 0
    assert float(one["x"].grad[0, ids[-8:]].norm()) > 0
    assert float(model.phi.weight.grad.norm()) > 0
    smoke = ReparamGCR(CONFIG["init_seed"], .5).to(device)
    opt = base.optimizer(smoke)
    torch.cuda.reset_peak_memory_stats()
    base.sync()
    start = time.perf_counter()
    loss, norm = train_update(smoke, opt, b)
    base.sync()
    seconds = time.perf_counter() - start
    result = {"status": "PASS", "config": CONFIG, "source_hashes": source_hashes(),
              "data_hashes": manifest["data_hashes"], "locked_reference_r2": r2,
              "trained_reference_checkpoint_sha256": binding["checkpoint_sha256"], "checks": checks,
              "credit_cut": {"early_input_gradient_exactly_zero": True, "suffix_input_and_phi_gradient_nonzero": True},
              "smoke": {"update_seconds": seconds, "estimated_training_seconds": 150 * seconds,
                        "loss": loss, "gradient_norm": norm,
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20},
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "gpu": torch.cuda.get_device_name()}
    base.atomic_json(out / "qualification.json", result)
    print(json.dumps({"status": "PASS", "smoke": result["smoke"],
                      "cuda_short_gradient": checks["initial_cuda_float32"]["new_k8_vs_old_full_gradient"]}), flush=True)


@torch.no_grad()
def evaluate(model, batch, horizons):
    predictions, counts = {t: [] for t in horizons}, {t: [] for t in horizons}
    base.sync()
    start = time.perf_counter()
    for first in range(0, len(batch["y"]), 16):
        b = base.sub(batch, slice(first, first + 16))
        state = model.initialize(b)
        for t in range(1, max(horizons) + 1):
            state = model.step(state, b)
            if t in predictions:
                predictions[t].append(model.predict(state).cpu().numpy())
                counts[t].append(state.count[torch.arange(len(b["y"]), device=b["x"].device), b["endpoint"], 0].cpu().numpy())
    base.sync()
    arrays = {"y": batch["y"].cpu().numpy(), "lengths": batch["lengths"].cpu().numpy()}
    result = {"seconds": time.perf_counter() - start, "horizons": {}}
    for t in horizons:
        pred, count = np.concatenate(predictions[t]), np.concatenate(counts[t])
        if not np.isfinite(pred).all():
            raise FloatingPointError(f"Nonfinite T{t} prediction")
        arrays[f"pred_T{t}"], arrays[f"count_T{t}"] = pred, count
        met = base.metrics(arrays["y"], pred)
        met["count_matches_length"] = bool(np.array_equal(count, arrays["lengths"]))
        result["horizons"][f"T{t}"] = met
    return result, arrays


def save_checkpoint(path, model, opt, next_update):
    temp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "next_update": next_update,
                "config": CONFIG, "source_hashes": source_hashes()}, temp)
    os.replace(temp, path)


def decision(r, reference):
    short, full = reference["gcr_k8"], reference["gcr_k64"]
    if r >= .5 and abs(r - full) <= .02:
        word = "NEAR_FULL_CREDIT_DEVELOPMENTAL"
    elif r > full + .02:
        word = "ABOVE_FULL_CREDIT_REFERENCE_DEVELOPMENTAL"
    elif r - short >= .10:
        word = "PARTIAL_DEVELOPMENTAL_SIGNAL"
    else:
        word = "NO_DEVELOPMENTAL_SIGNAL"
    return {"verdict": word, "new_k8_r2": r, "old_k8_r2": short, "old_k64_r2": full,
            "new_minus_old_k8": r - short, "new_minus_old_k64": r - full,
            "recovered_credit_gap_fraction": (r - short) / (full - short), "reference_qualified": full >= .5}


def run(out, qualification, device, resume):
    start = time.perf_counter()
    qual = read(qualification)
    assert qual["status"] == "PASS" and qual["config"] == CONFIG and qual["source_hashes"] == source_hashes()
    manifest, banks, schedule, reference_r2 = reference_inputs()
    assert qual["data_hashes"] == manifest["data_hashes"] and qual["locked_reference_r2"] == reference_r2
    if resume:
        local = read(out / "manifest.json")
        assert local["config"] == CONFIG and local["source_hashes"] == source_hashes()
        for rel, digest in local["data_hashes"].items():
            assert base.sha(out / rel) == digest
        base.append_json(out / "recovery_events.jsonl", {"event": "explicit_resume", "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    else:
        assert not out.exists(), "Use a fresh scientific output"
        out.mkdir(parents=True)
        shutil.copytree(REFERENCE / "banks", out / "banks")
        shutil.copyfile(qualification, out / "qualification.json")
        for rel in source_hashes():
            dest = out / "source" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, dest)
        base.atomic_json(out / "manifest.json", {"config": CONFIG, "source_hashes": source_hashes(),
                        "data_hashes": manifest["data_hashes"], "locked_reference_r2": reference_r2,
                        "reference_aggregate_sha256": base.sha(REFERENCE / "aggregate.json"),
                        "independent_unit": "one previously paired initialization/schedule block"})
    base.atomic_json(out / "local_dispatch_receipt.json", {"pid": os.getpid(), "host": socket.gethostname(),
                     "start_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gpu": torch.cuda.get_device_name(),
                     "runtime_cap": None, "resume": resume})
    batches = {k: base.bank_to_torch(v, device) for k, v in banks.items()}
    model = ReparamGCR(CONFIG["init_seed"], .5).to(device)
    opt = base.optimizer(model)
    arm = out / "reparam_gcr_k8"
    arm.mkdir(exist_ok=True)
    (arm / "checkpoints").mkdir(exist_ok=True)
    latest = arm / "checkpoints/latest.pt"
    first_update = 1
    if resume and latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        assert saved["config"] == CONFIG and saved["source_hashes"] == source_hashes()
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        first_update = saved["next_update"]
    torch.cuda.reset_peak_memory_stats()
    base.sync()
    train_start = time.perf_counter()
    for u in range(first_update, 151):
        ids = torch.as_tensor(schedule[u - 1], device=device)
        loss, norm = train_update(model, opt, base.sub(batches["train"], ids))
        base.append_json(arm / "training.jsonl", {"update": u, "loss": loss, "gradient_norm_before_clip": norm})
        save_checkpoint(latest, model, opt, u + 1)
        if u in (50, 100, 150):
            named = arm / "checkpoints" / f"u{u:03d}.pt"
            if not named.exists():
                shutil.copyfile(latest, named)
            ev, arrays = evaluate(model, batches["heldout"], [64])
            base.atomic_json(arm / f"heldout_u{u:03d}.json", ev)
            np.savez_compressed(arm / f"heldout_u{u:03d}.npz", **arrays)
        if u == 1 or u % 10 == 0:
            base.sync()
            status = {"status": "RUNNING", "arm": "reparam_gcr_k8", "update": u, "total_updates": 150,
                      "loss": loss, "elapsed_seconds": time.perf_counter() - start}
            base.atomic_json(out / "status.json", status)
            print(json.dumps(status), flush=True)
    base.sync()
    systems = {"training_and_intermediate_eval_seconds": time.perf_counter() - train_start,
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
               "parameters": model.param_counts(), "state_scalars_per_node": 73, "resumed_from_update": first_update}
    evaluations = {}
    for bank in ("heldout", "length64", "length96"):
        ev, arrays = evaluate(model, batches[bank], [64, 128, 256] if bank == "heldout" else [128, 256])
        evaluations[bank] = ev
        np.savez_compressed(arm / f"{bank}_final_predictions.npz", **arrays)
    r = evaluations["heldout"]["horizons"]["T64"]["r2"]
    result = {"status": "COMPLETE", "config": CONFIG, "decision": decision(r, reference_r2),
              "evaluations": evaluations, "systems": systems, "elapsed_seconds": time.perf_counter() - start,
              "claim_scope": "one-block exact raw-state lift and late projection; no generic NCA or reliability claim"}
    base.atomic_json(arm / "result.json", result)
    base.atomic_json(out / "aggregate.json", result)
    d = result["decision"]
    rows = ["# Reparameterized GCR K8: one-block developmental result", "",
            f"Execution: **COMPLETE**. Frozen verdict: **{d['verdict']}**.", "",
            "| Primary held-out T64 R2 | Value |", "|---|---:|",
            f"| Old GCR K8 (locked) | {d['old_k8_r2']:.6f} |",
            f"| Old GCR K64 (locked) | {d['old_k64_r2']:.6f} |",
            f"| Reparam GCR K8 | {r:.6f} |", "",
            f"New-minus-old K8: {d['new_minus_old_k8']:.6f}; new-minus-old K64: {d['new_minus_old_k64']:.6f}.",
            f"Elapsed seconds: {result['elapsed_seconds']:.3f}. Exactly one new150-update arm was trained.", "",
            "Forward/full-gradient equivalence and short-credit checks are bound in qualification.json.",
            "Raw sufficient state is locally produced, with73 instead of21 per-node scalars and the same5921 learned parameters.",
            "Known held-out cohorts are reused; u150 is primary and intermediate checkpoints do not select the model.",
            "This tests late parameter placement for a linear degree-two task, not generic learned recurrence or autonomous NCA stability.",
            "The earlier Full Writer comparison remains control-unqualified. All predictions and metrics are in the saved NPZ and aggregate.json.", ""]
    (out / "RESULTS.md").write_text("\n".join(rows), encoding="utf-8")
    base.atomic_json(out / "status.json", {"status": "COMPLETE", "completed_units": 1, "total_units": 1,
                     "elapsed_seconds": result["elapsed_seconds"], "decision": d})
    print(json.dumps({"status": "COMPLETE", "decision": d}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    device = base.setup()
    if args.check:
        check(out, device)
    else:
        if args.qualification is None:
            parser.error("--qualification required")
        fresh = not out.exists()
        if not fresh and not args.resume:
            raise FileExistsError("Use a fresh run name")
        try:
            run(out, args.qualification.resolve(), device, args.resume)
        except Exception as exc:
            if fresh and out.exists():
                base.atomic_json(out / "status.json", {"status": "ERROR", "verdict": "INCOMPLETE",
                                 "error_type": type(exc).__name__, "error": str(exc)})
                (out / "fatal_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
