"""Frozen cyclic-2D quadratic-write screen; qualification then protected launch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from cells import OriginalState, SpatialPolynomialCell
from tasks import TEACHER_PARAMETERS, make_bank, make_schedule

CONFIG = {
    "protocol": "spatial_nonlinear_lift_v0", "updates": 300,
    "batch_size": 16, "train_steps": 64, "blocks": 3,
    "init_seeds": [180101, 180102, 180103],
    "schedule_seeds": [180201, 180202, 180203],
    "lr": .01, "weight_decay": .0001, "betas": [.9, .999],
    "eps": 1e-8, "clip_norm": 1., "checkpoints": [100, 200, 300],
    "banks": {"train": [256, 8, 64, 180301],
              "heldout": [64, 8, 64, 180401],
              "long128": [32, 8, 128, 180501],
              "long256": [32, 8, 256, 180502],
              "spatial16": [32, 16, 64, 180601]},
    "arms": [["original_k64", "original", 64],
             ["original_k8", "original", 8], ["lifted_k8", "lifted", 8]],
    "teacher": TEACHER_PARAMETERS, "primary": "heldout_T64_u300",
    "positive_control_r2": .5, "gap_threshold": .10,
    "equivalent_tolerance_r2": .02, "device": "cuda:0", "dtype": "float32",
}


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_json(path, obj):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [HERE / n for n in ("run.py", "cells.py", "tasks.py", "THEORY.md", "PROTOCOL.md")]
    paths += [ROOT / "tools" / n for n in ("start_protected_job.ps1", "protected_job_worker.ps1")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent device fallback")
    return torch.device(CONFIG["device"])


def sync():
    torch.cuda.synchronize()


def teacher(device, dtype=torch.float32, quadratic=True):
    model = SpatialPolynomialCell("original", 0, dtype=dtype).to(device)
    with torch.no_grad():
        for name, values in TEACHER_PARAMETERS.items():
            getattr(model, name).copy_(torch.as_tensor(values, device=device, dtype=dtype))
        if not quadratic:
            model.gamma[2].zero_()
    return model


def forward(model, e, credit, return_state=False):
    steps = e.shape[1]
    if not 1 <= credit <= steps:
        raise ValueError("credit must be between one and the execution horizon")
    state = model.initialize(e[:, 0])
    prefix = steps - credit
    if prefix:
        with torch.no_grad():
            for t in range(prefix):
                state = model.step(state, e[:, t])
        state = model.detach(state)
    for t in range(prefix, steps):
        state = model.step(state, e[:, t])
    return state if return_state else model.predict(state)


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            weight_decay=CONFIG["weight_decay"],
                            betas=tuple(CONFIG["betas"]), eps=CONFIG["eps"])


def update(model, opt, e, y, credit):
    opt.zero_grad(set_to_none=True)
    loss = (forward(model, e, credit) - y).square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite endpoint loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_norm"], error_if_nonfinite=True)
    opt.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite parameter after update")
    return float(loss.detach()), float(norm.detach())


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise FloatingPointError("Nonfinite saved target or prediction")
    mse = float(np.square(y - pred).mean())
    variance = float(np.square(y - y.mean()).mean())
    if variance <= 1e-12:
        raise ValueError("R2 target variance is too small")
    return {"mse": mse, "r2": 1 - mse / variance, "target_variance": variance,
            "per_map_mse": np.square(y - pred).mean(axis=(1, 2)).tolist()}


def grad(model, e, credit):
    y = forward(model, e, credit)
    loss = (y - .17).square().mean()
    values = torch.autograd.grad(loss, tuple(model.parameters()))
    return torch.cat([g.reshape(-1) for g in values]), {
        n: float(g.norm()) for (n, _), g in zip(model.named_parameters(), values)}


def agreement(actual, reference):
    denominator = reference.norm().clamp_min(1e-30)
    return {"relative_l2": float((actual - reference).norm() / denominator),
            "cosine": float(torch.dot(actual, reference) / (actual.norm() * denominator))}


def pair_check(e, perturb=False):
    dtype, device = e.dtype, e.device
    old = SpatialPolynomialCell("original", CONFIG["init_seeds"][0], dtype=dtype).to(device)
    new = SpatialPolynomialCell("lifted", CONFIG["init_seeds"][0], dtype=dtype).to(device)
    if perturb:
        with torch.no_grad():
            for i, p in enumerate(old.parameters()):
                p.add_(.13 * torch.sin(torch.arange(p.numel(), device=device, dtype=dtype) + i + 1))
    new.load_state_dict(old.state_dict())
    maximum, output_error = 0., 0.
    tol = 1e-10 if dtype == torch.float64 else 1e-4
    with torch.no_grad():
        a, b = old.initialize(e[:, 0]), new.initialize(e[:, 0])
        for t in range(e.shape[1]):
            a, b = old.step(a, e[:, t]), new.step(b, e[:, t])
            for x, y in zip(old.project(a), new.project(b)):
                torch.testing.assert_close(x, y, atol=tol, rtol=tol)
                maximum = max(maximum, float((x - y).abs().max()))
            x, y = old.predict(a), new.predict(b)
            torch.testing.assert_close(x, y, atol=tol, rtol=tol)
            output_error = max(output_error, float((x - y).abs().max()))
    gold, _ = grad(old, e, e.shape[1])
    gfull, _ = grad(new, e, e.shape[1])
    gshort, groups = grad(new, e, 8)
    _, old_groups = grad(old, e, 8)
    result = {"max_projected_state_error": maximum, "max_output_error": output_error,
              "full_gradient": agreement(gfull, gold), "lifted_k8_vs_original_full": agreement(gshort, gold),
              "lifted_k8_group_norms": groups, "original_k8_group_norms": old_groups}
    gtol = 1e-9 if dtype == torch.float64 else 1e-4
    assert result["full_gradient"]["relative_l2"] <= gtol, result
    assert result["lifted_k8_vs_original_full"]["relative_l2"] <= gtol, result
    assert all(v > 0 for v in groups.values()) and all(v > 0 for v in old_groups.values())
    return result


def check(out, device):
    if out.exists():
        raise FileExistsError("Use a fresh qualification directory")
    out.mkdir(parents=True)
    probe = make_bank(16, 8, 64, 180701)
    e = torch.as_tensor(probe["e"], device=device)
    cpu = e[:2, :, :4, :4].cpu().double()
    checks = {"initial_cpu64": pair_check(cpu), "perturbed_cpu64": pair_check(cpu, True),
              "initial_actual_cuda32": pair_check(e)}
    original = SpatialPolynomialCell("original", CONFIG["init_seeds"][0], dtype=torch.float64)
    u = torch.full((2, 4, 4), .4, dtype=torch.float64, requires_grad=True)
    state = OriginalState(u, torch.zeros_like(u), 3)
    z = original.step(state, cpu[:, 0]).z
    first = torch.autograd.grad(z.sum(), u, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), u)[0]
    curvature = float(second.abs().max())
    assert curvature > 1e-8
    new = SpatialPolynomialCell("lifted", CONFIG["init_seeds"][0]).to(device)
    cut_input = e[:1].clone().requires_grad_(True)
    forward(new, cut_input, 8).sum().backward()
    early_zero = bool(torch.count_nonzero(cut_input.grad[:, :-8]) == 0)
    suffix_norm = float(cut_input.grad[:, -8:].norm())
    assert early_zero and suffix_norm > 0
    with torch.no_grad():
        y = forward(teacher(device), e, 64)
        y_no_square = forward(teacher(device, quadratic=False), e, 64)
        variance = float(y.var(unbiased=False))
        quadratic_fraction = float((y - y_no_square).square().mean().sqrt() / y.square().mean().sqrt())
    assert variance > 1e-4 and quadratic_fraction >= .10
    timings = {}
    for name, kind, credit in CONFIG["arms"]:
        model = SpatialPolynomialCell(kind, CONFIG["init_seeds"][0]).to(device)
        opt = optimizer(model)
        sync()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loss, norm = update(model, opt, e, y, credit)
        sync()
        timings[name] = {"update_seconds": time.perf_counter() - started,
                         "loss": loss, "gradient_norm": norm,
                         "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                         "incremental_peak_mib": (torch.cuda.max_memory_allocated() - baseline) / 2**20}
        del model, opt
    result = {"status": "PASS", "config": CONFIG, "source_hashes": source_hashes(),
              "checks": checks, "state_hessian_max_absolute": curvature,
              "early_input_gradient_zero": early_zero, "suffix_input_gradient_norm": suffix_norm,
              "teacher_probe": {"target_variance": variance, "quadratic_output_rms_fraction": quadratic_fraction},
              "smoke": timings, "estimated_training_seconds": 900 * sum(x["update_seconds"] for x in timings.values()),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "gpu": torch.cuda.get_device_name()}
    atomic_json(out / "qualification.json", result)
    print(json.dumps({k: result[k] for k in ("status", "state_hessian_max_absolute", "teacher_probe", "smoke", "estimated_training_seconds")}), flush=True)


@torch.no_grad()
def prepare_banks(out, device):
    directory = out / "banks"
    directory.mkdir()
    model = teacher(device)
    hashes, support = {}, {}
    for name, spec in CONFIG["banks"].items():
        generated = make_bank(*spec)
        e = torch.as_tensor(generated["e"], device=device)
        y = forward(model, e, e.shape[1]).cpu().numpy()
        arrays = {k: v for k, v in generated.items() if isinstance(v, np.ndarray)}
        arrays["y"] = y
        dest = directory / f"{name}.npz"
        np.savez_compressed(dest, **arrays)
        hashes[dest.relative_to(out).as_posix()] = sha(dest)
        support[name] = {"metadata": generated["metadata"], "target_variance": float(y.var()),
                         "target_min": float(y.min()), "target_max": float(y.max())}
        assert support[name]["target_variance"] > 1e-4
    for block, seed in enumerate(CONFIG["schedule_seeds"]):
        dest = directory / f"schedule_block{block:02d}.npy"
        np.save(dest, make_schedule(CONFIG["updates"], CONFIG["batch_size"], CONFIG["banks"]["train"][0], seed))
        hashes[dest.relative_to(out).as_posix()] = sha(dest)
    atomic_json(directory / "teacher.json", TEACHER_PARAMETERS)
    hashes["banks/teacher.json"] = sha(directory / "teacher.json")
    return hashes, support


def load_banks(out, device):
    banks = {}
    for name in CONFIG["banks"]:
        with np.load(out / "banks" / f"{name}.npz", allow_pickle=False) as saved:
            banks[name] = {k: torch.as_tensor(saved[k], device=device) for k in ("e", "y")}
    return banks


@torch.no_grad()
def evaluate(model, bank):
    sync()
    started = time.perf_counter()
    pred = forward(model, bank["e"], bank["e"].shape[1]).cpu().numpy()
    sync()
    y = bank["y"].cpu().numpy()
    return {"metrics": metrics(y, pred), "seconds": time.perf_counter() - started}, {"y": y, "prediction": pred}


def save_checkpoint(path, model, opt, next_update, block, arm):
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "next_update": next_update, "block": block, "arm": arm,
                "config": CONFIG, "source_hashes": source_hashes()}, temporary)
    os.replace(temporary, path)


def train_arm(out, block, spec, banks, completed, start, resume):
    name, kind, credit = spec
    arm = out / f"block{block:02d}" / name
    if (arm / "result.json").exists():
        return read(arm / "result.json")
    arm.mkdir(parents=True, exist_ok=True)
    checkpoints = arm / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    model = SpatialPolynomialCell(kind, CONFIG["init_seeds"][block]).to(CONFIG["device"])
    opt = optimizer(model)
    latest = checkpoints / "latest.pt"
    first_update = 1
    if resume and (arm / "training.jsonl").exists() and not latest.exists():
        raise RuntimeError("Existing training log has no recovery checkpoint; refusing a silent restart")
    if resume and latest.exists():
        saved = torch.load(latest, map_location=CONFIG["device"], weights_only=False)
        assert saved["config"] == CONFIG and saved["source_hashes"] == source_hashes()
        assert saved["block"] == block and saved["arm"] == name
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        first_update = saved["next_update"]
        if (arm / "training.jsonl").exists():
            lines = [x for x in (arm / "training.jsonl").read_text().splitlines() if read_line_update(x) < first_update]
            (arm / "training.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    elif not latest.exists():
        save_checkpoint(checkpoints / "u000.pt", model, opt, 1, block, name)
    schedule = np.load(out / "banks" / f"schedule_block{block:02d}.npy", allow_pickle=False)
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    sync()
    train_start = time.perf_counter()
    for u in range(first_update, CONFIG["updates"] + 1):
        ids = torch.as_tensor(schedule[u - 1], device=CONFIG["device"])
        loss, norm = update(model, opt, banks["train"]["e"][ids], banks["train"]["y"][ids], credit)
        append_json(arm / "training.jsonl", {"update": u, "loss": loss, "gradient_norm_before_clip": norm})
        save_checkpoint(latest, model, opt, u + 1, block, name)
        if u in CONFIG["checkpoints"]:
            named = checkpoints / f"u{u:03d}.pt"
            if not named.exists():
                shutil.copyfile(latest, named)
            ev, arrays = evaluate(model, banks["heldout"])
            atomic_json(arm / f"heldout_u{u:03d}.json", ev)
            np.savez_compressed(arm / f"heldout_u{u:03d}.npz", **arrays)
        if u == 1 or u % 10 == 0:
            status = {"status": "RUNNING", "completed_units": completed, "total_units": 9,
                      "block": block, "arm": name, "update": u, "updates": CONFIG["updates"],
                      "loss": loss, "elapsed_seconds": time.perf_counter() - start}
            atomic_json(out / "status.json", status)
            print(json.dumps(status), flush=True)
    sync()
    systems = {"training_and_intermediate_eval_seconds": time.perf_counter() - train_start,
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
               "incremental_peak_mib": (torch.cuda.max_memory_allocated() - baseline) / 2**20,
               "parameters": sum(p.numel() for p in model.parameters()),
               "state_scalars_per_cell": 2 if kind == "original" else 42,
               "resumed_from_update": first_update}
    evaluations = {}
    for bank_name in CONFIG["banks"]:
        if bank_name == "train":
            continue
        ev, arrays = evaluate(model, banks[bank_name])
        evaluations[bank_name] = ev
        np.savez_compressed(arm / f"{bank_name}_predictions.npz", **arrays)
    final = {n: p.detach().cpu().tolist() for n, p in model.named_parameters()}
    result = {"status": "COMPLETE", "block": block, "arm": name, "credit": credit,
              "init_seed": CONFIG["init_seeds"][block], "schedule_seed": CONFIG["schedule_seeds"][block],
              "evaluations": evaluations, "systems": systems, "final_parameters": final}
    atomic_json(arm / "result.json", result)
    return result


def read_line_update(line):
    return json.loads(line)["update"]


def decision(results):
    controls = [r for r in results if r["arm"] == "original_k64"]
    qualified = len(controls) == 3 and all(r["evaluations"]["heldout"]["metrics"]["r2"] >= .5 for r in controls)
    pairs = []
    for block in range(3):
        row = {r["arm"]: r["evaluations"]["heldout"]["metrics"]["r2"] for r in results if r["block"] == block}
        if len(row) == 3:
            pairs.append({"block": block, **row,
                          "full_minus_original_short": row["original_k64"] - row["original_k8"],
                          "lifted_short_minus_full": row["lifted_k8"] - row["original_k64"]})
    verdict = "POSITIVE_CONTROL_UNQUALIFIED"
    if qualified:
        delta = [p["full_minus_original_short"] for p in pairs]
        if max(abs(p["lifted_short_minus_full"]) for p in pairs) > .02:
            verdict = "EQUIVALENT_TRAINING_DIVERGENCE_DEVELOPMENTAL"
        elif np.mean(delta) >= .10 and sum(x > 0 for x in delta) >= 2:
            verdict = "SHORT_CREDIT_RECOVERY_DEVELOPMENTAL"
        else:
            verdict = "NO_MATERIAL_SHORT_CREDIT_GAP_DEVELOPMENTAL"
    return {"verdict": verdict, "controls_qualified": qualified, "paired_primary": pairs}


def report(out, results, start, stopped=False):
    d = decision(results)
    aggregate = {"status": "COMPLETE", "config": CONFIG, "decision": d,
                 "completed_units": len(results), "planned_units": 9,
                 "stopped_at_predeclared_control": stopped,
                 "elapsed_seconds": time.perf_counter() - start, "arms": results,
                 "claim_scope": "three-block matched polynomial-teacher screen; not generic NCA or reliability evidence"}
    atomic_json(out / "aggregate.json", aggregate)
    rows = ["# Spatial nonlinear lift: cyclic 2D quadratic-write screen", "",
            f"Execution: **COMPLETE**. Verdict: **{d['verdict']}**.",
            f"Completed {len(results)}/9 planned arms; predeclared control stop: {stopped}.", "",
            "| Block | Arm | Heldout T64 R2 | Long128 R2 | Long256 R2 | Side16 T64 R2 |", "|---|---|---:|---:|---:|---:|"]
    for r in results:
        values = [r["evaluations"][n]["metrics"]["r2"] for n in ("heldout", "long128", "long256", "spatial16")]
        rows.append(f"| {r['block']} | {r['arm']} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    rows += ["", "All predictions and per-map MSE are retained. Primary model selection is fixed at u300.",
             "Original and lifted have nine identical parameter coordinates and exact real-arithmetic forward functions.",
             "Lifted execution uses42 rather than2 scalars per cell, plus a shared time counter.",
             "Full-credit/lifted-short equality is predicted algebraically; empirical interest is the original K8 gap and systems cost.",
             "Long tests use separate proportionally phased teacher-labeled episodes, not autonomous continuation or stability tests.",
             "This is a matched finite-degree teacher screen with fixed input features and transport; no generic nonlinear-feedback claim.", ""]
    (out / "RESULTS.md").write_text("\n".join(rows), encoding="utf-8")
    atomic_json(out / "status.json", {"status": "COMPLETE", "completed_units": len(results), "total_units": 9,
                "stopped_at_predeclared_control": stopped, "elapsed_seconds": aggregate["elapsed_seconds"], "decision": d})
    print(json.dumps({"status": "COMPLETE", "completed_units": len(results), "decision": d}), flush=True)


def run(out, qualification, device, resume):
    started = time.perf_counter()
    qual = read(qualification)
    assert qual["status"] == "PASS" and qual["config"] == json.loads(json.dumps(CONFIG))
    assert qual["source_hashes"] == source_hashes()
    if resume:
        manifest = read(out / "manifest.json")
        assert manifest["source_hashes"] == source_hashes() and manifest["config"] == json.loads(json.dumps(CONFIG))
        for name, digest in manifest["data_hashes"].items():
            assert sha(out / name) == digest
        append_json(out / "recovery_events.jsonl", {"event": "explicit_resume", "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    else:
        if out.exists():
            raise FileExistsError("Use a fresh run directory")
        out.mkdir(parents=True)
        shutil.copyfile(qualification, out / "qualification.json")
        for name in source_hashes():
            destination = out / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        hashes, support = prepare_banks(out, device)
        atomic_json(out / "manifest.json", {"config": CONFIG, "source_hashes": source_hashes(),
                    "data_hashes": hashes, "bank_support": support, "qualification_sha256": sha(qualification),
                    "independent_unit": "paired initialization/minibatch-schedule block"})
    atomic_json(out / "local_dispatch_receipt.json", {"host": socket.gethostname(), "pid": os.getpid(),
                "gpu": torch.cuda.get_device_name(), "device": str(device),
                "launched_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runtime_cap": None, "resume": resume})
    banks = load_banks(out, device)
    results = []
    for block in range(3):
        for spec in CONFIG["arms"]:
            result = train_arm(out, block, spec, banks, len(results), started, resume)
            results.append(result)
            if block == 0 and spec[0] == "original_k64" and result["evaluations"]["heldout"]["metrics"]["r2"] < .5:
                report(out, results, started, stopped=True)
                return
    report(out, results, started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    # Reject non-owned existing outputs before entering an error-writing scope.
    existed_on_entry = out.exists()
    if existed_on_entry:
        if args.check or not args.resume:
            raise FileExistsError("Use a fresh output directory")
        existing_manifest = read(out / "manifest.json")
        assert existing_manifest["source_hashes"] == source_hashes()
        assert existing_manifest["config"] == json.loads(json.dumps(CONFIG))
        for name, digest in existing_manifest["data_hashes"].items():
            assert sha(out / name) == digest
    elif args.resume:
        raise FileNotFoundError("Resume requires an existing bound run directory")
    device = setup()
    try:
        if args.check:
            check(out, device)
        else:
            if args.qualification is None:
                parser.error("--qualification is required")
            run(out, args.qualification.resolve(), device, args.resume)
    except Exception as exc:
        if out.exists():
            atomic_json(out / "status.json", {"status": "ERROR", "verdict": "INCOMPLETE",
                        "error_type": type(exc).__name__, "error": str(exc)})
            (out / "fatal_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
