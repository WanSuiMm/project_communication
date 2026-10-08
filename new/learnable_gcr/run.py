"""Frozen one-block learnable ordered-composition screen; no auto monitoring."""
from __future__ import annotations

import argparse
import csv
import hashlib
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
sys.path.insert(0, str(HERE))
from cells import FixedEvidenceMLP, OrderedCell  # noqa: E402
from tasks import bank_to_torch, direct_second_order, make_bank, teacher  # noqa: E402

CONFIG = {
    "protocol": "learnable_gcr_v0", "teacher_seed": 170001,
    "init_seed": 170101, "schedule_seed": 170201,
    "updates": 150, "batch_size": 16, "train_steps": 64,
    "lr": .001, "weight_decay": .0001, "betas": [.9, .999],
    "eps": 1e-8, "clip_norm": 1., "rho": .5,
    "banks": {
        "train": [512, 8, [16, 32], 170301],
        "heldout": [128, 8, [16, 32], 170401],
        "length64": [128, 16, [64, 64], 170501],
        "length96": [128, 16, [96, 96], 170502],
    },
    "arms": [["full_writer_k8", "full_writer", 8],
             ["full_writer_k64", "full_writer", 64],
             ["gcr_k8", "gcr", 8], ["gcr_k64", "gcr", 64]],
    "primary": "heldout_T64_u150", "checkpoints": [50, 100, 150],
    "device": "cuda:0", "dtype": "float32", "tf32": False,
}


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_json(path, obj):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [HERE / n for n in ("run.py", "cells.py", "tasks.py", "PROTOCOL.md")]
    paths += [ROOT / "tools" / n for n in ("start_protected_job.ps1", "protected_job_worker.ps1")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def sync():
    torch.cuda.synchronize()


def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("This frozen local screen requires CUDA; no silent device fallback")
    return torch.device(CONFIG["device"])


def make_banks():
    spec = teacher(CONFIG["teacher_seed"])
    banks = {}
    for name, (count, side, lengths, seed) in CONFIG["banks"].items():
        banks[name] = make_bank(count, side, tuple(lengths), seed, spec)
    return spec, banks


def sub(batch, indices):
    return {k: v[indices] for k, v in batch.items()}


def forward_credit(model, batch, steps, credit, *, return_state=False):
    state = model.initialize(batch)
    prefix = steps - credit
    if prefix:
        with torch.no_grad():
            p = model.feature(batch["x"])
            for _ in range(prefix):
                state = model.step(state, batch, p=p)
    state = model.detach(state)
    # Raw inputs are constant through time. Reusing this graph is exact, not
    # a detached encoder cache: every suffix use contributes to phi's gradient.
    p = model.feature(batch["x"])
    for _ in range(credit):
        state = model.step(state, batch, p=p)
    prediction = model.predict(state)
    return (prediction, state) if return_state else prediction


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            weight_decay=CONFIG["weight_decay"],
                            betas=tuple(CONFIG["betas"]), eps=CONFIG["eps"])


def train_step(model, opt, batch, credit=None):
    opt.zero_grad(set_to_none=True)
    pred = model(batch["features"]) if credit is None else forward_credit(model, batch, 64, credit)
    loss = (pred - batch["y"]).square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite terminal MSE before backward")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_norm"], error_if_nonfinite=True)
    opt.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite parameter after optimizer step")
    return float(loss.detach()), float(norm.detach())


def metrics(y, pred):
    y, pred = np.asarray(y, np.float64), np.asarray(pred, np.float64)
    if not np.isfinite(pred).all():
        return {"finite": False, "mse": None, "r2": None, "n": len(y)}
    mse = float(np.mean((y - pred) ** 2))
    variance = float(np.mean((y - y.mean()) ** 2))
    return {"finite": True, "mse": mse, "r2": 1. - mse / variance if variance > 0 else None,
            "target_variance": variance, "n": len(y)}


@torch.no_grad()
def evaluate(model, batch, horizons):
    predictions, counts = {t: [] for t in horizons}, {t: [] for t in horizons}
    sync()
    start = time.perf_counter()
    for first in range(0, len(batch["y"]), CONFIG["batch_size"]):
        b = sub(batch, slice(first, first + CONFIG["batch_size"]))
        state = model.initialize(b)
        p = model.feature(b["x"])
        for t in range(1, max(horizons) + 1):
            state = model.step(state, b, p=p)
            if t in predictions:
                predictions[t].append(model.predict(state).cpu().numpy())
                counts[t].append(state.count[torch.arange(len(b["y"]), device=p.device), b["endpoint"], 0].cpu().numpy())
    sync()
    seconds = time.perf_counter() - start
    y = batch["y"].cpu().numpy()
    arrays = {"y": y, "lengths": batch["lengths"].cpu().numpy()}
    result = {"seconds": seconds, "horizons": {}}
    for t in horizons:
        pred, count = np.concatenate(predictions[t]), np.concatenate(counts[t])
        arrays[f"pred_T{t}"] = pred
        arrays[f"count_T{t}"] = count
        result["horizons"][f"T{t}"] = metrics(y, pred)
        result["horizons"][f"T{t}"]["count_matches_length"] = bool(np.array_equal(count, arrays["lengths"]))
    return result, arrays


def fixed_features(bank):
    n_examples = len(bank["y"])
    side = int(round(np.sqrt(bank["x"].shape[1])))
    out = np.empty((n_examples, 73), np.float32)
    for i, length in enumerate(bank["lengths"]):
        n = int(length)
        xy = bank["path"][i, :n]
        tokens = bank["x"][i, xy[:, 0] * side + xy[:, 1]].astype(np.float64)
        s1, s2 = np.zeros(8), np.zeros((8, 8))
        for x in tokens:
            s2 += np.outer(s1, x)
            s1 += x
        out[i] = np.concatenate(([np.log1p(n) / np.log(97.)], s1 / n, (s2 / n).ravel()))
    return out


def check(out, device):
    if out.exists():
        raise FileExistsError(f"Qualification output already exists: {out}")
    out.mkdir(parents=True)
    assertions = []
    spec, banks = make_banks()
    for name, bank in banks.items():
        side = int(round(np.sqrt(bank["x"].shape[1])))
        for i in range(len(bank["y"])):
            n = int(bank["lengths"][i])
            xy = bank["path"][i, :n]
            ids = xy[:, 0] * side + xy[:, 1]
            assert len(np.unique(ids)) == n
            assert np.all(np.abs(np.diff(xy, axis=0)).sum(axis=1) == 1)
            assert bank["mask"][i].sum() == n
            assert bank["pred"][i, ids[0]] == -1
            assert np.array_equal(bank["pred"][i, ids[1:]], ids[:-1])
            if i < 4:
                tokens = bank["x"][i, ids].astype(np.float64)
                projection = np.stack((tokens @ spec["a"].astype(np.float64), tokens @ spec["b"].astype(np.float64)), axis=1)
                _, exact = direct_second_order(projection)
                assert np.isclose(np.tanh(exact / n), bank["y"][i], atol=1e-7)
        assertions.append(f"{name}: simple Manhattan path, predecessor/mask, independent bank and teacher target")
    b = sub(bank_to_torch(banks["train"], device), slice(0, 16))
    gcr, fw = [OrderedCell(k, CONFIG["init_seed"]).to(device) for k in ("gcr", "full_writer")]
    for name, tensor in gcr.state_dict().items():
        assert torch.equal(tensor, fw.state_dict()[name]), name
    assertions.append("Shared encoder and interpreter initial weights are bit-identical")
    with torch.no_grad():
        state = gcr.initialize(b)
        encoded = gcr.feature(b["x"])
        for _ in range(64):
            state = gcr.step(state, b, p=encoded)
        for i, n in enumerate(b["lengths"].tolist()):
            xy = banks["train"]["path"][i, :n]
            ids = torch.tensor(xy[:, 0] * 8 + xy[:, 1], device=device, dtype=torch.long)
            s1, s2 = torch.zeros(4, device=device), torch.zeros(4, 4, device=device)
            for token in encoded[i, ids]:
                s2 = s2 + torch.outer(s1, token)
                s1 = s1 + token
            expected = torch.cat((s1, s2.flatten()))
            assert torch.allclose(state.workspace[i, b["endpoint"][i]], expected, atol=5e-5, rtol=5e-5)
        assert torch.equal(state.count[torch.arange(16, device=device), b["endpoint"], 0], b["lengths"].float())
    assertions.append("Literal GCR overwrite reaches exact stationary first/second prefix sums and n")
    reversals = []
    for i in range(16):
        n = int(banks["train"]["lengths"][i])
        xy = banks["train"]["path"][i, :n]
        tokens = banks["train"]["x"][i, xy[:, 0] * 8 + xy[:, 1]].astype(np.float64)
        projections = np.stack((tokens @ spec["a"], tokens @ spec["b"]), axis=1)
        _, rev = direct_second_order(projections[::-1])
        reversals.append(abs(np.tanh(rev / n) - banks["train"]["y"][i]))
    assert max(reversals) > .1
    assertions.append("Teacher is order-sensitive, not reducible to the unordered token mean")
    feature_reference = fixed_features(banks["train"])
    known_pair = feature_reference[:, 9:].reshape(-1, 8, 8)
    known_target = np.tanh(np.einsum("i,bij,j->b", spec["a"], known_pair, spec["b"]))
    assert np.allclose(known_target, banks["train"]["y"], atol=2e-7)
    assertions.append("Raw fixed evidence contains teacher with S2/n scaling")
    timings = {}
    for kind in ("gcr", "full_writer"):
        model = OrderedCell(kind, CONFIG["init_seed"]).to(device)
        p8, state8 = forward_credit(model, b, 64, 8, return_state=True)
        p64, state64 = forward_credit(model, b, 64, 64, return_state=True)
        p8, p64 = p8.detach(), p64.detach()
        assert torch.equal(p8, p64), f"{kind}: credit changed forward"
        assert all(torch.equal(a, c) for a, c in zip(state8, state64)), f"{kind}: credit changed final state"
        assert torch.equal((p8 - b["y"]).square().mean(), (p64 - b["y"]).square().mean())
        del state8, state64
        one = sub(b, slice(0, 1))
        one["x"] = one["x"].detach().clone().requires_grad_(True)
        forward_credit(model, one, 64, 8).sum().backward()
        assert model.phi.weight.grad is not None and float(model.phi.weight.grad.norm()) > 0
        xy = banks["train"]["path"][0, :int(one["lengths"][0])]
        ids = torch.as_tensor(xy[:, 0] * 8 + xy[:, 1], device=device, dtype=torch.long)
        assert torch.count_nonzero(one["x"].grad[0, ids[:-8]]) == 0
        assert float(one["x"].grad[0, ids[-8:]].norm()) > 0
        model.zero_grad(set_to_none=True)
        one["x"].grad = None
        forward_credit(model, one, 64, 64).sum().backward()
        assert float(one["x"].grad[0, ids[:-8]].norm()) > 0
        assertions.append(f"{kind}: same states/predictions/loss K8/K64; phi learns in suffix; early input credit is actually truncated")
        for credit in (8, 64):
            smoke = OrderedCell(kind, CONFIG["init_seed"]).to(device)
            opt = optimizer(smoke)
            torch.cuda.reset_peak_memory_stats()
            sync()
            start = time.perf_counter()
            loss, norm = train_step(smoke, opt, b, credit)
            sync()
            timings[f"{kind}_k{credit}"] = {
                "update_seconds": time.perf_counter() - start, "loss": loss, "gradient_norm": norm,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                "parameters": smoke.param_counts(),
            }
            del smoke, opt
    diagnostic = FixedEvidenceMLP(CONFIG["init_seed"]).to(device)
    db = {"features": torch.as_tensor(feature_reference[:16], device=device), "y": b["y"]}
    diag_loss, diag_norm = train_step(diagnostic, optimizer(diagnostic), db)
    result = {"status": "PASS", "config": CONFIG, "source_hashes": source_hashes(),
              "assertions": assertions, "timings": timings,
              "diagnostic": {"loss": diag_loss, "gradient_norm": diag_norm},
              "estimated_primary_training_seconds": 150 * sum(v["update_seconds"] for v in timings.values()),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(),
              "target_distributions": {k: {"variance": float(np.var(v["y"])), "saturation_abs_gt_095": float(np.mean(abs(v["y"]) > .95))} for k, v in banks.items()}}
    atomic_json(out / "qualification.json", result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


def save_checkpoint(path, model, opt, next_update):
    temp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "next_update": next_update, "config": CONFIG, "source_hashes": source_hashes()}, temp)
    os.replace(temp, path)


def initialize_run(out, qualification, resume):
    qual = json.loads(qualification.read_text(encoding="utf-8"))
    if qual["status"] != "PASS" or qual["config"] != CONFIG or qual["source_hashes"] != source_hashes():
        raise RuntimeError("Qualification does not match current frozen source/config")
    if resume:
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config"] != CONFIG or manifest["source_hashes"] != source_hashes():
            raise RuntimeError("Resume source/config differs from original")
        for rel, digest in manifest["data_hashes"].items():
            if sha(out / rel) != digest:
                raise RuntimeError(f"Resume data changed: {rel}")
        banks = {name: dict(np.load(out / "banks" / f"{name}.npz")) for name in CONFIG["banks"]}
        schedule = np.load(out / "banks" / "schedule.npy")
        append_json(out / "recovery_events.jsonl", {"event": "explicit_resume", "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        return banks, schedule
    if out.exists():
        raise FileExistsError(f"Run output already exists; use a new name: {out}")
    out.mkdir(parents=True)
    (out / "banks").mkdir()
    spec, banks = make_banks()
    schedule = np.random.default_rng(CONFIG["schedule_seed"]).integers(0, len(banks["train"]["y"]), (150, 16), dtype=np.int64)
    for name, bank in banks.items():
        np.savez_compressed(out / "banks" / f"{name}.npz", **bank)
    np.save(out / "banks" / "schedule.npy", schedule)
    atomic_json(out / "banks" / "teacher.json", {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in spec.items()})
    for rel in source_hashes():
        target = out / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    shutil.copyfile(qualification, out / "qualification.json")
    data_hashes = {p.relative_to(out).as_posix(): sha(p) for p in sorted((out / "banks").iterdir())}
    atomic_json(out / "manifest.json", {"config": CONFIG, "source_hashes": source_hashes(), "data_hashes": data_hashes,
                                        "independent_unit": "one paired initialization and schedule block",
                                        "supervision": "endpoint MSE only at T64"})
    return banks, schedule


def train_arm(out, name, model, bank, schedule, credit, heldout, resume):
    arm = out / name
    arm.mkdir(exist_ok=True)
    (arm / "checkpoints").mkdir(exist_ok=True)
    opt, next_update = optimizer(model), 1
    latest = arm / "checkpoints" / "latest.pt"
    if resume and latest.exists():
        saved = torch.load(latest, map_location=bank["y"].device, weights_only=False)
        if saved["config"] != CONFIG or saved["source_hashes"] != source_hashes():
            raise RuntimeError("Arm checkpoint source/config mismatch")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        next_update = saved["next_update"]
    torch.cuda.reset_peak_memory_stats()
    sync()
    start = time.perf_counter()
    for u in range(next_update, CONFIG["updates"] + 1):
        batch = sub(bank, torch.as_tensor(schedule[u - 1], device=bank["y"].device))
        loss, norm = train_step(model, opt, batch, credit)
        append_json(arm / "training.jsonl", {"update": u, "loss": loss, "gradient_norm_before_clip": norm})
        # A small atomic latest file protects each completed optimizer update;
        # immutable named checkpoints retain the prespecified diagnostic stages.
        save_checkpoint(latest, model, opt, u + 1)
        if u in CONFIG["checkpoints"]:
            named = arm / "checkpoints" / f"u{u:03d}.pt"
            if not named.exists():
                shutil.copyfile(latest, named)
            if credit is not None:
                evaluation, arrays = evaluate(model, heldout, [64])
            else:
                with torch.no_grad():
                    pred = model(heldout["features"]).cpu().numpy()
                evaluation = {"horizons": {"T64": metrics(heldout["y"].cpu().numpy(), pred)}}
                arrays = {"y": heldout["y"].cpu().numpy(), "pred": pred}
            atomic_json(arm / f"heldout_u{u:03d}.json", evaluation)
            np.savez_compressed(arm / f"heldout_u{u:03d}.npz", **arrays)
        if u % 10 == 0 or u == 1:
            sync()
            progress = {"status": "RUNNING", "arm": name, "update": u, "total_updates": 150,
                        "loss": loss, "elapsed_arm_seconds": time.perf_counter() - start}
            atomic_json(out / "status.json", progress)
            print(json.dumps(progress), flush=True)
    sync()
    result = {"training_seconds_this_dispatch": time.perf_counter() - start,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
              "resumed_from_update": next_update,
              "parameters": model.param_counts() if credit is not None else {"total": sum(p.numel() for p in model.parameters())}}
    atomic_json(arm / "systems.json", result)
    return result


def verdict(results):
    names = [row[0] for row in CONFIG["arms"]]
    if any(results.get(n, {}).get("status") != "COMPLETE" for n in names):
        return {"verdict": "INCOMPLETE_OR_NUMERICAL_FAILURE", "controls_qualified": False}
    r = {n: results[n]["evaluations"]["heldout"]["horizons"]["T64"]["r2"] for n in names}
    if any(v is None for v in r.values()):
        return {"verdict": "METRIC_UNQUALIFIED", "controls_qualified": False}
    fw_gap = r["full_writer_k64"] - r["full_writer_k8"]
    gcr_gap = r["gcr_k64"] - r["gcr_k8"]
    qualified = r["full_writer_k64"] >= .5 and r["gcr_k64"] >= .5
    if not qualified:
        word = "POSITIVE_CONTROLS_UNQUALIFIED"
    elif fw_gap < .10:
        word = "NO_IDENTIFIED_CREDIT_GAP"
    elif r["gcr_k8"] >= .5 and r["gcr_k8"] - r["full_writer_k8"] >= .10 and gcr_gap <= .05:
        word = "DEVELOPMENTAL_SIGNAL"
    else:
        word = "NO_DEVELOPMENTAL_SIGNAL"
    return {"verdict": word, "controls_qualified": qualified, "primary_r2": r,
            "full_writer_credit_gap": fw_gap, "gcr_credit_gap": gcr_gap,
            "architecture_by_credit_gap": fw_gap - gcr_gap}


def report(out, aggregate):
    rows = ["# Learnable GCR: one-block developmental screen", "",
            f"Execution: **{aggregate['status']}**. Frozen verdict: **{aggregate['decision']['verdict']}**.", "",
            "One paired initialization/schedule block, 150 updates, terminal T64 MSE only. This does not estimate architecture reliability.", "",
            "| Arm | Held-out T64 MSE | Held-out T64 R2 | Length64 T128 R2 | Length96 T128 R2 |", "|---|---:|---:|---:|---:|"]
    for name, result in aggregate["results"].items():
        if result["status"] != "COMPLETE":
            rows.append(f"| {name} | {result['status']} | — | — | — |")
            continue
        ev = result["evaluations"]
        first = ev["heldout"]["horizons"]["T64"]
        long64 = ev["length64"]["horizons"]["T128"]["r2"]
        long96 = ev["length96"]["horizons"]["T128"]["r2"]
        rows.append(f"| {name} | {first['mse']:.6f} | {first['r2']:.4f} | {long64:.4f} | {long96:.4f} |")
    rows += ["", "The fixed-evidence MLP is an offline readout diagnostic, not a matched NCA.", "",
             "The continuous architecture-by-credit contrast and all systems/evaluation metrics are in [aggregate.json](aggregate.json).",
             "Per-example predictions and exact data banks are compressed NPZ. Models and optimizer checkpoints remain local.", "",
             "Qualification establishes path/algebra/gradient plumbing only. K64 learning qualification is decided by the scientific endpoint.",
             "A GCR-K8 success would reflect shared local parameter learning from suffix uses; it would not recover gradients of individual distant early writes.",
             "The task matches second-order evidence. Routing, cyclic duplicates, general learned write, other tasks and multiple seeds remain untested.", ""]
    (out / "RESULTS.md").write_text("\n".join(rows), encoding="utf-8")


def run(out, qualification, device, resume):
    started = time.perf_counter()
    banks, schedule = initialize_run(out, qualification, resume)
    atomic_json(out / "local_dispatch_receipt.json", {"pid": os.getpid(), "host": socket.gethostname(),
        "start_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gpu": torch.cuda.get_device_name(),
        "device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda,
        "out_absolute": str(out), "resume": resume, "runtime_cap": None})
    batches = {name: bank_to_torch(bank, device) for name, bank in banks.items()}
    fixed_batches = {name: {"features": torch.as_tensor(fixed_features(bank), device=device), "y": batches[name]["y"]} for name, bank in banks.items()}
    results = {}
    for name, kind, credit in CONFIG["arms"] + [["fixed_raw_evidence_mlp", "diagnostic", None]]:
        result_path = out / name / "result.json"
        if resume and result_path.exists():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if prior["status"] == "COMPLETE":
                results[name] = prior
                continue
        model = FixedEvidenceMLP(CONFIG["init_seed"]).to(device) if credit is None else OrderedCell(kind, CONFIG["init_seed"], CONFIG["rho"]).to(device)
        used = fixed_batches if credit is None else batches
        try:
            systems = train_arm(out, name, model, used["train"], schedule, credit, used["heldout"], resume)
            evaluations = {}
            for bank_name in ("heldout", "length64", "length96"):
                if credit is None:
                    with torch.no_grad():
                        pred = model(used[bank_name]["features"]).cpu().numpy()
                    met = metrics(used[bank_name]["y"].cpu().numpy(), pred)
                    evaluation = {"horizons": {f"T{t}": met for t in ([64, 128, 256] if bank_name == "heldout" else [128, 256])}}
                    arrays = {"y": used[bank_name]["y"].cpu().numpy(), "pred": pred}
                else:
                    evaluation, arrays = evaluate(model, used[bank_name], [64, 128, 256] if bank_name == "heldout" else [128, 256])
                if not all(v["finite"] for v in evaluation["horizons"].values()):
                    raise FloatingPointError(f"Nonfinite evaluation: {bank_name}")
                evaluations[bank_name] = evaluation
                np.savez_compressed(out / name / f"{bank_name}_final_predictions.npz", **arrays)
            result = {"status": "COMPLETE", "systems": systems, "evaluations": evaluations}
        except (FloatingPointError, RuntimeError) as exc:
            # Keep failed arms visible, then finish the other prespecified arms.
            result = {"status": "NUMERICAL_OR_EXECUTION_FAILURE", "error_type": type(exc).__name__, "error": str(exc)}
            (out / name).mkdir(exist_ok=True)
            (out / name / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        atomic_json(result_path, result)
        results[name] = result
        del model
        torch.cuda.empty_cache()
    complete = all(r["status"] == "COMPLETE" for r in results.values())
    aggregate = {"status": "COMPLETE" if complete else "COMPLETED_WITH_FAILURES", "config": CONFIG,
                 "elapsed_seconds": time.perf_counter() - started, "decision": verdict(results), "results": results,
                 "claim_scope": "one-block task-matched developmental screen; no generic short-credit guarantee"}
    atomic_json(out / "aggregate.json", aggregate)
    report(out, aggregate)
    atomic_json(out / "status.json", {"status": aggregate["status"], "elapsed_seconds": aggregate["elapsed_seconds"],
                                     "completed_units": len(results), "total_units": 5, "decision": aggregate["decision"]})
    print(json.dumps({"status": aggregate["status"], "decision": aggregate["decision"]}, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--resume", action="store_true", help="explicitly resume same frozen run from atomic checkpoints")
    args = parser.parse_args()
    out = args.out.resolve()
    device = setup()
    if args.check:
        check(out, device)
    else:
        if args.qualification is None:
            parser.error("--qualification is required for scientific training")
        was_fresh = not out.exists()
        if not was_fresh and not args.resume:
            raise FileExistsError(f"Run output already exists; use a new name: {out}")
        try:
            run(out, args.qualification.resolve(), device, args.resume)
        except Exception as exc:
            if was_fresh and out.exists():
                atomic_json(out / "status.json", {"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)})
                (out / "fatal_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise


if __name__ == "__main__":
    main()
