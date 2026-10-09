"""AU-NCA developmental screen: fixed-target growth with learned feedback."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from cells import NCACell
from tasks import make_seed, make_target

CONFIG = {
    "protocol": "accumulated_update_nca_v0", "blocks": 3,
    "init_seeds": [190101, 190102, 190103],
    "schedule_seeds": [190201, 190202, 190203],
    "evaluation_seeds": [190401, 190402, 190403],
    "arms": [["original_k64", "original", 64],
             ["original_k8", "original", 8], ["au_k8", "au", 8]],
    "updates": 3000, "batch_size": 8, "size": 32,
    "channels": 16, "hidden": 128, "pool_size": 512,
    "train_steps": 64, "fire_probability": .5,
    "lr": .002, "weight_decay": 0., "betas": [.9, .999], "eps": 1e-8,
    "per_parameter_gradient_normalization": True,
    "checkpoints": [1000, 2000, 3000], "recovery_interval": 25,
    "eval_episodes": 32, "eval_batch": 8, "eval_steps": [64, 128, 256],
    "control_nmse_max": .10, "control_alpha_iou_min": .80,
    "paired_nmse_gain_min": .05, "mean_worsening_tolerance": .02,
    "strong_gap_original_nmse_min": .20, "strong_recovery_au_nmse_max": .10,
    "dtype": "float32", "device": "cuda:0",
    "training_damage": False, "runtime_cap": None,
}


class ActiveRunError(RuntimeError):
    """A second process must not mutate a currently owned run."""


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    files = [HERE / name for name in ("run.py", "cells.py", "checks.py", "tasks.py", "PROTOCOL.md", "THEORY.md")]
    files += [ROOT / "tools" / name for name in ("start_protected_job.ps1", "protected_job_worker.ps1")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent fallback")
    return torch.device(CONFIG["device"])


def new_model(block, device):
    torch.manual_seed(CONFIG["init_seeds"][block])
    return NCACell(channels=CONFIG["channels"], hidden=CONFIG["hidden"]).to(device)


def new_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                             betas=tuple(CONFIG["betas"]), eps=CONFIG["eps"],
                             weight_decay=CONFIG["weight_decay"])


def train_masks(block, update, device):
    seed = CONFIG["schedule_seeds"][block] + 104729 * update + 10000000
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (CONFIG["train_steps"], CONFIG["batch_size"], 1, CONFIG["size"], CONFIG["size"])
    return (torch.rand(shape, generator=generator, device=device) < CONFIG["fire_probability"]).float()


def rollout(model, initial, masks, mode, credit):
    state = model.initialize(initial, mode)
    prefix = len(masks) - credit
    if prefix:
        with torch.no_grad():
            for fire in masks[:prefix]:
                state = model.step(state, fire, mode)
        state = model.detach(state, mode)
    for fire in masks[prefix:]:
        state = model.step(state, fire, mode)
    return state


def group_norm(model, prefix):
    terms = [p.grad.detach().square().sum() for name, p in model.named_parameters()
             if name.startswith(prefix) and p.grad is not None]
    return torch.stack(terms).sum().sqrt() if terms else torch.tensor(0., device=next(model.parameters()).device)


def update(model, opt, initial, masks, mode, credit, target):
    opt.zero_grad(set_to_none=True)
    state = rollout(model, initial, masks, mode, credit)
    visible = model.visible(state, mode)
    loss = (visible[:, :4] - target).square().mean()
    loss.backward()
    w_norm = group_norm(model, "projection")
    eta_norm = group_norm(model, "feature")
    finite = torch.isfinite(loss) & torch.isfinite(w_norm) & torch.isfinite(eta_norm) & torch.isfinite(visible).all()
    if not bool(finite):
        raise FloatingPointError("Nonfinite loss or gradient; arm terminated without JSON inf")
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(p.grad.norm() + 1e-8)
    # The stored pool state is produced entirely by the pre-step parameters.
    # AU re-encodes this visible state at the next optimizer-update boundary.
    pooled = visible.detach()
    opt.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite parameter after optimizer update")
    return pooled, float(loss.detach()), float(w_norm), float(eta_norm)


def metric_arrays(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite evaluation state")
    mse = np.square(prediction - target).mean(axis=(1, 2, 3))
    blank = float(np.square(target).mean())
    foreground = prediction[:, 3] > .5
    truth = target[0, 3] > .5
    intersection = np.logical_and(foreground, truth).sum(axis=(1, 2))
    union = np.logical_or(foreground, truth).sum(axis=(1, 2))
    return {"mse": mse, "nmse": mse / blank,
            "alpha_iou": intersection / np.maximum(union, 1)}


def summarize_metrics(values):
    return {key: {"mean": float(value.mean()), "std": float(value.std(ddof=1)) if len(value) > 1 else 0.,
                  "per_episode": value.tolist()} for key, value in values.items()}


def build_inputs(out):
    target = make_target(CONFIG["size"]).numpy().astype(np.float32)
    np.save(out / "target_rgba.npy", target)
    banks = {}
    for block in range(CONFIG["blocks"]):
        rng = np.random.default_rng(CONFIG["schedule_seeds"][block])
        ids = np.stack([rng.choice(CONFIG["pool_size"], CONFIG["batch_size"], replace=False)
                        for _ in range(CONFIG["updates"])]).astype(np.int32)
        erng = np.random.default_rng(CONFIG["evaluation_seeds"][block])
        shape = (256, CONFIG["eval_episodes"], 1, CONFIG["size"], CONFIG["size"])
        intact = erng.random(shape, dtype=np.float32) < CONFIG["fire_probability"]
        damaged = intact[64:].copy()
        path = out / f"block{block:02d}_inputs.npz"
        np.savez_compressed(path, pool_indices=ids, intact_bits=np.packbits(intact.reshape(-1)),
                            intact_shape=np.asarray(intact.shape), damage_bits=np.packbits(damaged.reshape(-1)),
                            damage_shape=np.asarray(damaged.shape))
        banks[path.name] = sha(path)
    banks["target_rgba.npy"] = sha(out / "target_rgba.npy")
    return banks


def load_inputs(out, block):
    with np.load(out / f"block{block:02d}_inputs.npz") as bank:
        intact_shape = tuple(bank["intact_shape"].tolist())
        damage_shape = tuple(bank["damage_shape"].tolist())
        return (bank["pool_indices"].copy(),
                np.unpackbits(bank["intact_bits"], count=int(np.prod(intact_shape))).reshape(intact_shape),
                np.unpackbits(bank["damage_bits"], count=int(np.prod(damage_shape))).reshape(damage_shape))


@torch.no_grad()
def evaluate(model, mode, target, intact, damaged=None, diagnostic=False):
    device = next(model.parameters()).device
    count = min(8, intact.shape[1]) if diagnostic else intact.shape[1]
    snapshots = {t: [] for t in ([64] if diagnostic else CONFIG["eval_steps"])}
    repairs = {t: [] for t in (128, 256)} if not diagnostic and damaged is not None else {}
    curves = []
    for lo in range(0, count, CONFIG["eval_batch"]):
        hi = min(count, lo + CONFIG["eval_batch"])
        plan = torch.as_tensor(intact[:, lo:hi], device=device, dtype=torch.float32)
        state = model.initialize(make_seed(hi-lo, CONFIG["size"], CONFIG["channels"], device=device), mode)
        damaged_state = None
        damage_plan = torch.as_tensor(damaged[:, lo:hi], device=device, dtype=torch.float32) if repairs else None
        for t in range(1, 65 if diagnostic else 257):
            state = model.step(state, plan[t-1], mode)
            visible = model.visible(state, mode)
            if t % 8 == 0:
                rgba = visible[:, :4].cpu().numpy()
                mets = metric_arrays(rgba, target)
                for j in range(hi-lo):
                    curves.append({"episode": lo+j, "step": t,
                                   **{k: float(v[j]) for k, v in mets.items()},
                                   "state_rms": float(visible[j].square().mean().sqrt()),
                                   "alive_fraction": float((visible[j, 3] > .1).float().mean())})
            if t in snapshots:
                snapshots[t].append(visible[:, :4].cpu().numpy())
            if t == 64 and not diagnostic:
                # Keep the actual intact state for autonomous continuation.
                # Canonicalize only the damaged branch; fixed-W forward
                # dynamics depend on the visible state, not its AU gauge.
                boundary = visible.detach().clone()
                if repairs:
                    erase = torch.ones((1, 1, CONFIG["size"], CONFIG["size"]), device=device)
                    erase[:, :, :, CONFIG["size"] // 2:] = 0
                    damaged_state = model.initialize(boundary * erase, mode)
            elif t > 64 and damaged_state is not None:
                damaged_state = model.step(damaged_state, damage_plan[t-65], mode)
                if t in repairs:
                    repairs[t].append(model.visible(damaged_state, mode)[:, :4].cpu().numpy())
    arrays, summary = {}, {}
    for label, collection in (("intact", snapshots), ("damaged", repairs)):
        for t, chunks in collection.items():
            prediction = np.concatenate(chunks)
            arrays[f"{label}_t{t}"] = prediction
            summary[f"{label}_t{t}"] = summarize_metrics(metric_arrays(prediction, target))
    summary["episodes"] = count
    summary["same_state_continuation"] = not diagnostic
    summary["damage_is_secondary_untrained"] = True
    return summary, arrays, curves


def flat_grad(model, mode, credit, initial, masks, target):
    state = rollout(model, initial, masks, mode, credit)
    loss = (model.visible(state, mode)[:, :4] - target).square().mean()
    grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    groups = {"W": [], "eta": []}
    for (name, p), grad in zip(model.named_parameters(), grads):
        groups["W" if name.startswith("projection") else "eta"].append(
            (torch.zeros_like(p) if grad is None else grad).detach().double().flatten())
    return {key: torch.cat(value) for key, value in groups.items()}, float(loss.detach())


def gradient_diagnostics(model, target_tensor, masks):
    initial = make_seed(CONFIG["batch_size"], CONFIG["size"], CONFIG["channels"],
                        device=next(model.parameters()).device)
    full, loss_full = flat_grad(model, "original", 64, initial, masks, target_tensor)
    old, loss_old = flat_grad(model, "original", 8, initial, masks, target_tensor)
    au, loss_au = flat_grad(model, "au", 8, initial, masks, target_tensor)
    result = {"loss_full": loss_full, "loss_original_k8": loss_old, "loss_au_k8": loss_au,
              "same_parameters": True, "groups": {}}
    for key in ("W", "eta"):
        gf = full[key]
        norm = float(gf.norm())
        row = {"full_norm": norm, "original_k8_norm": float(old[key].norm()),
               "au_k8_norm": float(au[key].norm()),
               "au_minus_original_norm": float((au[key] - old[key]).norm())}
        for label, gs in (("original_k8", old[key]), ("au_k8", au[key])):
            snorm = float(gs.norm())
            row[label + "_cosine_full"] = float(torch.dot(gf, gs) / (gf.norm()*gs.norm())) if norm > 1e-20 and snorm > 1e-20 else None
            row[label + "_relative_residual"] = float((gf-gs).norm()) / norm if norm > 1e-20 else None
        result["groups"][key] = row
    return result


def qualify(out, device):
    from checks import run_checks
    out.mkdir(parents=True)
    checked = run_checks(device=str(device))
    if not checked["passed"]:
        atomic_json(out / "qualification.json", {"status": "FAIL", "checks": checked})
        raise AssertionError("AU-NCA equivalence checks failed")
    target = make_target(CONFIG["size"]).to(device)
    initial = make_seed(CONFIG["batch_size"], CONFIG["size"], CONFIG["channels"], device=device)
    timings = {}
    for name, mode, credit in CONFIG["arms"]:
        model = new_model(0, device)
        opt = new_optimizer(model)
        masks = train_masks(0, 1, device)
        update(model, opt, initial, masks, mode, credit, target)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        losses = []
        for u in (2, 3, 4):
            _, loss, _, _ = update(model, opt, initial, train_masks(0, u, device), mode, credit, target)
            losses.append(loss)
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - started) / 3
        timings[name] = {"seconds_per_update": seconds, "smoke_losses": losses,
                         "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                         "parameter_count": sum(p.numel() for p in model.parameters()),
                         "state_scalars_per_cell": 16 if mode == "original" else 145}
        del opt, model
    record = {"status": "PASS", "config": CONFIG, "source_hashes": sources(), "checks": checked,
              "actual_shape_smoke": timings,
              "training_estimate_seconds": sum(v["seconds_per_update"] for v in timings.values()) * CONFIG["updates"] * CONFIG["blocks"],
              "estimate_excludes_evaluation_and_io": True}
    atomic_json(out / "qualification.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("checks", "source_hashes", "config")}), flush=True)


def checkpoint(model, opt, update_number, pool=None):
    value = {"model": model.state_dict(), "optimizer": opt.state_dict(), "update": update_number,
             "config": CONFIG, "source_hashes": sources(),
             "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all()}
    if pool is not None:
        value["pool"] = pool.detach().cpu()
    return value


def qualified(result):
    if result["status"] != "COMPLETE":
        return False
    primary = result["evaluation"]["intact_t64"]
    return primary["nmse"]["mean"] <= CONFIG["control_nmse_max"] and primary["alpha_iou"]["mean"] >= CONFIG["control_alpha_iou_min"]


def train_arm(out, block, spec, target, data, completed, started, resume):
    name, mode, credit = spec
    folder = out / f"block{block:02d}" / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        if not resume:
            raise FileExistsError("Existing arm result without explicit resume")
        return read(folder / "result.json")
    device = torch.device(CONFIG["device"])
    model = new_model(block, device)
    opt = new_optimizer(model)
    indices, intact, damaged = data
    target_tensor = torch.as_tensor(target, device=device)
    pool = make_seed(CONFIG["pool_size"], CONFIG["size"], CONFIG["channels"], device=device)
    latest = folder / "checkpoints" / "latest.pt"
    first = 1
    if resume and latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        assert saved["config"] == CONFIG and saved["source_hashes"] == sources()
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng_state"]])
        pool.copy_(saved["pool"])
        first = saved["update"] + 1
        # Preserve every pre-recovery curve, including an interrupted tail.
        if (folder / "training.jsonl").exists():
            old = (folder / "training.jsonl").read_text(encoding="utf-8").splitlines()
            shutil.copyfile(folder / "training.jsonl", folder / f"training_before_resume_{time.time_ns()}.jsonl")
            kept = []
            for line in old:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["update"] < first:
                    kept.append(line)
            (folder / "training.jsonl").write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    failure = None
    for u in range(first, CONFIG["updates"]+1):
        ids = torch.as_tensor(indices[u-1], device=device, dtype=torch.long)
        initial = pool[ids].clone()
        with torch.no_grad():
            seed_losses = (initial[:, :4] - target_tensor).square().mean(dim=(1, 2, 3))
            worst = seed_losses.argmax()
            initial[worst] = make_seed(1, CONFIG["size"], CONFIG["channels"], device=device)[0]
        try:
            produced, loss, w_norm, eta_norm = update(model, opt, initial, train_masks(block, u, device), mode, credit, target_tensor)
        except FloatingPointError as exc:
            failure = {"update": u, "error": str(exc)}
            atomic_torch(folder / "checkpoints" / "numerical_failure.pt", checkpoint(model, opt, u, pool))
            break
        pool[ids] = produced
        jsonl(folder / "training.jsonl", {"update": u, "loss": loss,
              "W_gradient_norm_before_normalization": w_norm, "eta_gradient_norm_before_normalization": eta_norm})
        if u % CONFIG["recovery_interval"] == 0 or u == 1:
            atomic_torch(latest, checkpoint(model, opt, u, pool))
            status = {"status": "RUNNING", "completed_units": completed, "total_units": 9,
                      "block": block, "arm": name, "update": u, "updates": CONFIG["updates"],
                      "loss": loss, "elapsed_seconds": time.perf_counter()-started}
            atomic_json(out / "status.json", status)
            print(json.dumps(status), flush=True)
        if u in CONFIG["checkpoints"]:
            atomic_torch(folder / "checkpoints" / f"u{u:04d}.pt", checkpoint(model, opt, u))
            if u != CONFIG["updates"]:
                ev, arrays, _ = evaluate(model, mode, target, intact, diagnostic=True)
                atomic_json(folder / f"diagnostic_u{u:04d}.json", ev)
                np.savez_compressed(folder / f"diagnostic_u{u:04d}.npz", **arrays)
    torch.cuda.synchronize()
    systems = {"seconds_this_execution": time.perf_counter()-train_started,
               "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "state_scalars_per_cell": 16 if mode == "original" else 145,
               "resumed_first_update": first}
    result = {"status": "NUMERICAL_FAILURE" if failure else "COMPLETE", "block": block,
              "arm": name, "mode": mode, "credit": credit,
              "init_seed": CONFIG["init_seeds"][block], "schedule_seed": CONFIG["schedule_seeds"][block],
              "systems": systems, "failure": failure}
    if not failure:
        ev, arrays, curves = evaluate(model, mode, target, intact, damaged)
        result["evaluation"] = ev
        np.savez_compressed(folder / "predictions.npz", **arrays)
        atomic_json(folder / "continuation_curves.json", curves)
        # Diagnostics are secondary and cannot overwrite a completed primary evaluation.
        try:
            diagnostics = gradient_diagnostics(model, target_tensor, train_masks(block, 9000, device))
            atomic_json(folder / "gradient_diagnostics.json", diagnostics)
        except FloatingPointError as exc:
            atomic_json(folder / "gradient_diagnostics.json", {"status": "NONFINITE_DIAGNOSTIC", "error": str(exc)})
    atomic_json(folder / "result.json", result)
    return result


def decision(results):
    controls = [r for r in results if r["arm"] == "original_k64"]
    controls_ok = len(controls) == CONFIG["blocks"] and all(qualified(r) for r in controls)
    pairs = []
    for block in range(CONFIG["blocks"]):
        arms = {r["arm"]: r for r in results if r["block"] == block}
        if len(arms) != 3:
            continue
        values = {name: r.get("evaluation", {}).get("intact_t64", {}).get("nmse", {}).get("mean") for name, r in arms.items()}
        row = {"block": block, "nmse": values}
        if all(v is not None for v in values.values()):
            row["original_short_minus_au"] = values["original_k8"] - values["au_k8"]
            row["full_minus_au"] = values["original_k64"] - values["au_k8"]
        pairs.append(row)
    verdict = "POSITIVE_CONTROL_UNQUALIFIED"
    if controls_ok:
        finite_pairs = [p for p in pairs if "original_short_minus_au" in p]
        if len(finite_pairs) < CONFIG["blocks"]:
            verdict = "NUMERICAL_FAILURE_DEVELOPMENTAL"
        else:
            gains = [p["original_short_minus_au"] for p in finite_pairs]
            strong = sum(p["nmse"]["original_k8"] >= .20 and p["nmse"]["au_k8"] <= .10 for p in finite_pairs) >= 2
            benefit = sum(g >= .05 for g in gains) >= 2 and float(np.mean(gains)) >= -.02
            verdict = "SHORT_CREDIT_RECOVERY_DEVELOPMENTAL" if strong and benefit else (
                "AU_BENEFIT_DEVELOPMENTAL" if benefit else "NO_QUALIFIED_AU_BENEFIT_DEVELOPMENTAL")
    return {"verdict": verdict, "all_controls_qualified": controls_ok,
            "control_passes": sum(qualified(r) for r in controls), "paired_primary": pairs}


def report(out, results, started, stopped):
    d = decision(results)
    aggregate = {"status": "COMPLETE", "config": CONFIG, "decision": d, "arms": results,
                 "completed_units": len(results), "planned_units": 9,
                 "predeclared_control_stop": stopped, "elapsed_seconds": time.perf_counter()-started,
                 "independent_unit": "paired initialization and schedule block; not firing-mask episode",
                 "scope": "three-block single procedural target learned-feedback NCA screen; no universal BPTT claim"}
    atomic_json(out / "aggregate.json", aggregate)
    rows = ["# AU-NCA v0: learned-feedback growth", "",
            f"Execution **COMPLETE**; verdict **{d['verdict']}**.",
            f"Finished {len(results)}/9 planned arms; predeclared first-control stop: {stopped}.", "",
            "| Block | Arm | Status | T64 NMSE | T64 alpha IoU | T128 NMSE | T256 NMSE | Damaged T256 NMSE |",
            "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in results:
        values = []
        for key, metric in (("intact_t64", "nmse"), ("intact_t64", "alpha_iou"),
                            ("intact_t128", "nmse"), ("intact_t256", "nmse"), ("damaged_t256", "nmse")):
            val = r.get("evaluation", {}).get(key, {}).get(metric, {}).get("mean")
            values.append("NA" if val is None else f"{val:.6f}")
        rows.append(f"| {r['block']} | {r['arm']} | {r['status']} | " + " | ".join(values) + " |")
    rows += ["", "Primary model selection is fixed at u3000 and cold-seed T64. Intermediate checkpoints are diagnostic only.",
             "NMSE divides RGBA MSE by the blank-image MSE; lower is better. The fixed target is preserved as target_rgba.npy.",
             "T128/T256 continue the same T64 state with no parameter update. Damage erases all 16 channels over the right half; regeneration is untrained and secondary.",
             "The AU accumulator is rebased at pool/optimizer boundaries. It is retained across K8 cuts within a rollout.",
             "State cost is16 versus145 scalars/cell; both arms use the same learned feature and projection parameters.",
             "Pool indices and firing plans match across arms. Learned visible pool values and the highest-loss reset identity can differ after training.",
             "This is a fixed-step, fixed-target adaptation of Growing/Persistent NCA, not an exact reproduction of its published training recipe.",
             "An unqualified K64 control prevents a temporal-credit recovery claim. Three blocks cannot establish general reliability.", ""]
    (out / "RESULTS.md").write_text("\n".join(rows), encoding="utf-8")
    atomic_json(out / "status.json", {k: aggregate[k] for k in ("status", "completed_units", "planned_units", "predeclared_control_stop", "elapsed_seconds", "decision")})
    print(json.dumps({"status": "COMPLETE", "decision": d}), flush=True)


@contextmanager
def run_lock(out):
    import msvcrt
    handle = (out / "run.lock").open("a+b")
    try:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise ActiveRunError("This run is already owned by another process") from exc
        yield
    finally:
        handle.close()


def validate_resume(out):
    manifest = read(out / "manifest.json")
    assert manifest["config"] == CONFIG and manifest["source_hashes"] == sources()
    for name, digest in manifest["data_hashes"].items():
        assert sha(out / name) == digest


def run(out, qualification, device, resume):
    qual = read(qualification)
    assert qual["status"] == "PASS" and qual["config"] == CONFIG and qual["source_hashes"] == sources()
    if resume:
        validate_resume(out)
    else:
        out.mkdir(parents=True)
        shutil.copyfile(qualification, out / "qualification.json")
        for name in sources():
            dest = out / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        hashes = build_inputs(out)
        atomic_json(out / "manifest.json", {"config": CONFIG, "source_hashes": sources(),
                    "data_hashes": hashes, "qualification_sha256": sha(qualification)})
    with run_lock(out):
        started = time.perf_counter()
        if resume:
            jsonl(out / "recovery_events.jsonl", {"event": "explicit_resume", "local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        atomic_json(out / "local_dispatch_receipt.json", {"host": socket.gethostname(), "pid": os.getpid(),
                    "gpu": torch.cuda.get_device_name(), "device": str(device),
                    "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runtime_cap": None,
                    "resume": resume, "command": sys.argv})
        target = np.load(out / "target_rgba.npy")
        results = []
        for block in range(CONFIG["blocks"]):
            data = load_inputs(out, block)
            for spec in CONFIG["arms"]:
                result = train_arm(out, block, spec, target, data, len(results), started, resume)
                results.append(result)
                if block == 0 and spec[0] == "original_k64" and not qualified(result):
                    report(out, results, started, stopped=True)
                    return
        report(out, results, started, stopped=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Outputs must stay in the owning project")
    if out.exists():
        if args.check or not args.resume:
            raise FileExistsError("Use a fresh output directory")
        validate_resume(out)
        if read(out / "status.json").get("status") == "COMPLETE":
            raise RuntimeError("Completed evidence is read-only")
    elif args.resume:
        raise FileNotFoundError("Resume requires a bound existing run")
    if not args.check and args.qualification is None:
        parser.error("--qualification is required for scientific training")
    device = setup()
    try:
        if args.check:
            qualify(out, device)
        else:
            run(out, args.qualification.resolve(), device, args.resume)
    except ActiveRunError:
        raise
    except Exception as exc:
        if out.exists():
            atomic_json(out / "status.json", {"status": "ERROR", "verdict": "INCOMPLETE",
                        "error_type": type(exc).__name__, "error": str(exc)})
            (out / "fatal_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
