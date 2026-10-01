"""Paired workspace/task-state additive versus revision screen; see PROTOCOL.md."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import traceback
import numpy as np
import torch
from revision_cells import ARMS, make_cell
from tasks import bank, subset, balanced_loss, metrics, per_example_balanced_accuracy

ROOT = Path(__file__).resolve().parents[2]
DEADLINE = float("inf")


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def check_budget():
    if time.monotonic() >= DEADLINE:
        raise TimeoutError("Frozen whole-run time budget exhausted")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def finite(state):
    if not all(bool(torch.isfinite(a).all()) for a in state):
        raise FloatingPointError("Nonfinite state")


def detached(state):
    return tuple(a.detach() for a in state)


def accuracy_vector(logits, target, mask):
    good = ((logits >= 0) == (target >= .5)).float()
    return (good * mask).sum((1, 2, 3)) / mask.sum((1, 2, 3)).clamp_min(1)


@torch.no_grad()
def score(logits, y, mask, distance=None):
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Nonfinite output")
    result = metrics(logits, y, mask, distance)
    result["per_map_ba"] = per_example_balanced_accuracy(logits, y, mask).cpu().tolist()
    return result


def open_rms(value, mask):
    denom = mask.sum((1, 2, 3)).clamp_min(1) * value.shape[1]
    return ((value.square() * mask).sum((1, 2, 3)) / denom).sqrt()


def erase_patch(state, positions, side, both=False):
    w, z = state
    keep = torch.ones_like(z[:, :1])
    for i, (y, x) in enumerate(positions):
        keep[i, :, y:y+side, x:x+side] = 0
    return (w * keep if both else w), z * keep


def schedule(seed, steps, batch, examples, preflight):
    rng = np.random.default_rng(20000 + seed)
    rows = []
    for j in range(steps):
        rows.append({
            "indices": rng.integers(0, examples, batch).tolist(),
            "T": 64 if preflight else int(rng.choice([32, 48, 64])),
            "warm": 64 if preflight else int(rng.choice([0, 32, 64])),
            "K": 32 if preflight else int(rng.choice([16, 32])),
            "mode": ("hold", "switch", "repair")[j % 3],
            "both": bool(rng.integers(0, 2)),
            "positions": rng.integers(0, 25, (batch, 2)).tolist(),
        })
    return rows


def step_loop(model, state, x, steps):
    for t in range(steps):
        if t % 16 == 0:
            check_budget()
        state = model.step(state, x)
    finite(state)
    return state


def branch_loss(model, state, x, y, mask, steps):
    values = []
    for t in range(1, steps + 1):
        if t % 16 == 0:
            check_budget()
        state = model.step(state, x)
        if t in {max(1, steps - 4), steps}:
            values.append(balanced_loss(model.logits(state), y, mask))
    return state, torch.stack(values).mean()


@torch.no_grad()
def evaluate(model, data, preflight):
    x, y, mask = data["x"], data["y"], data["mask"]
    xf, yf, change = data["x_flip"], data["y_flip"], data["changed"]
    horizons = [16, 64] if preflight else [16, 32, 64, 96, 128, 192, 256]
    recovery = [8, 64] if preflight else [8, 16, 32, 64, 128]
    state, flip = model.initial(x), model.initial(xf)
    curves, saved = {}, None
    for t in range(1, max(horizons) + 1):
        check_budget()
        state, flip = model.step(state, x), model.step(flip, xf)
        if t in horizons:
            finite(state); finite(flip)
            logits, flip_logits = model.logits(state), model.logits(flip)
            row = score(logits, y, mask, data["distance"])
            row["flip"] = score(flip_logits, yf, mask)
            both = (((logits >= 0) == (y >= .5)) &
                    ((flip_logits >= 0) == (yf >= .5))).float()
            vec = (both * change).sum((1, 2, 3)) / change.sum((1, 2, 3)).clamp_min(1)
            row["paired"] = float(vec.mean())
            row["paired_per_map"] = vec.cpu().tolist()
            row["paired_pooled_pixels"] = float((both * change).sum() / change.sum().clamp_min(1))
            row["paired_distance"] = {}
            for lo, hi in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 100000)]:
                selected = change * ((data["distance"] >= lo) & (data["distance"] < hi))
                count = int(selected.sum())
                row["paired_distance"][f"{lo}_{hi}"] = {
                    "pixels": count,
                    "pooled_accuracy": float((both * selected).sum() / selected.sum()) if count else None}
            next_state = model.step(state, x)
            finite(next_state)
            for name, value in [
                ("w_rms", state[0]), ("z_rms", state[1]),
                ("dw_rms", next_state[0] - state[0]), ("dz_rms", next_state[1] - state[1]),
                ("output_change_rms", model.logits(next_state) - logits)]:
                per_map = open_rms(value, mask)
                row[name] = float(per_map.mean())
                row[name + "_per_map"] = per_map.cpu().tolist()
            curves[str(t)] = row
        if t == 64:
            saved = tuple(a.clone() for a in state)
    assert saved is not None
    eligible = per_example_balanced_accuracy(model.logits(saved), y, mask) >= .95
    size = x.shape[-1]
    side = max(1, round(size / 4))
    positions = np.random.default_rng(50000 + size).integers(0, size - side + 1, (len(x), 2)).tolist()
    branches = {
        "clean": saved, "cold": model.initial(x),
        "switch": saved, "switch_cold": model.initial(xf),
        "z_damage": erase_patch(saved, positions, side),
        "wz_damage": erase_patch(saved, positions, side, True)}
    interventions = {}

    def capture():
        row = {}
        for name, s in branches.items():
            finite(s)
            is_switch = name.startswith("switch")
            target = yf if is_switch else y
            logits = model.logits(s)
            r = score(logits, target, mask)
            if is_switch:
                for label, selected in [("changed", change), ("unchanged", mask - change)]:
                    values = accuracy_vector(logits, target, selected)
                    r[label + "_accuracy"] = float(values.mean())
                    r[label + "_per_map"] = values.cpu().tolist()
            if name in {"clean", "cold", "z_damage", "wz_damage"}:
                r["conditional_clean95_ba"] = (
                    float(per_example_balanced_accuracy(logits, target, mask)[eligible].mean())
                    if bool(eligible.any()) else None)
            row[name] = r
        return row

    interventions["0"] = capture()
    for k in range(1, max(recovery) + 1):
        check_budget()
        for name in branches:
            branches[name] = model.step(branches[name], xf if name.startswith("switch") else x)
        if k in recovery:
            interventions[str(k)] = capture()
    return {
        "status": "EVALUATED", "curve": curves,
        "interventions_from_T64": interventions,
        "repair": {"eligible_clean95": int(eligible.sum()), "total_maps": len(x),
                   "side": side, "positions": positions,
                   "eligible_map_indices": eligible.nonzero().flatten().cpu().tolist()},
        "hold_min_ba": None if preflight else min(curves[str(t)]["balanced_accuracy"] for t in [128, 192, 256]),
        "hold_definition": "Minimum aggregate BA at sampled T128,192,256; not continuous-time stability."}


@torch.no_grad()
def benchmark(model, x):
    for _ in range(2):
        step_loop(model, model.initial(x), x, 64)
    samples = []
    if x.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(x.device)
    for _ in range(3):
        check_budget(); sync(x.device)
        start = time.perf_counter()
        state = model.initial(x)
        for _t in range(64):
            state = model.step(state, x)
        logits = model.logits(state)
        sync(x.device)
        samples.append(time.perf_counter() - start)
        finite(state)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite benchmark output")
    return {"batch": len(x), "macro_steps": 64, "communication_phases": 128,
            "samples_seconds": samples, "median_seconds": float(np.median(samples)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(x.device) if x.device.type == "cuda" else None,
            "includes": "initialization, recurrent inference, readout; CUDA synchronized"}


def train_one(args, arm, seed, train, rows, tests, out):
    torch.manual_seed(seed)
    model = make_cell(arm).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    path = out / f"{arm}_seed{seed}.json"
    result = {"arm": arm, "seed": seed, "status": "TRAINING",
              "model": model.metadata(), "completed_updates": 0,
              "initial_parameter_sha256": tensor_hash(model.state_dict()),
              "train_data_sha256": tensor_hash(train),
              "schedule_sha256": sha(out / f"schedule_seed{seed}.json"),
              "training_curve": [], "evaluation": {}}
    write(path, result)
    clipped, start = 0, time.perf_counter()
    try:
        model.train()
        for iteration, row in enumerate(rows, 1):
            check_budget()
            batch = subset(train, row["indices"])
            optimizer.zero_grad(set_to_none=True)
            state, reach = branch_loss(model, model.initial(batch["x"]), batch["x"],
                                       batch["y"], batch["mask"], row["T"])
            if not bool(torch.isfinite(reach)):
                raise FloatingPointError("Nonfinite reach loss")
            reach.backward()
            with torch.no_grad():
                state = step_loop(model, detached(state), batch["x"], row["warm"])
            state = detached(state)
            bx, by = batch["x"], batch["y"]
            if row["mode"] == "switch":
                bx, by = batch["x_flip"], batch["y_flip"]
            elif row["mode"] == "repair":
                state = erase_patch(state, row["positions"], 8, row["both"])
            _, auxiliary = branch_loss(model, state, bx, by, batch["mask"], row["K"])
            if not bool(torch.isfinite(auxiliary)):
                raise FloatingPointError("Nonfinite auxiliary loss")
            auxiliary.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite parameter gradient")
            clipped += int(norm > 1.)
            optimizer.step()
            result["completed_updates"] = iteration
            if iteration == 1 or iteration % 100 == 0 or iteration == len(rows) or args.preflight:
                log = {"iteration": iteration, "reach_loss": float(reach.detach()),
                       "auxiliary_loss": float(auxiliary.detach()), "gradient_norm": float(norm),
                       **{k: row[k] for k in ["T", "warm", "K", "mode", "both"]}}
                result["training_curve"].append(log)
                write(out / "status.json", {"status": "RUNNING", "phase": "training",
                      "arm": arm, "seed": seed, "completed_updates": iteration, "pid": os.getpid(),
                      "updated_utc": now()})
                with (out / "training.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"arm": arm, "seed": seed, **log}) + "\n")
                print(json.dumps({"arm": arm, "seed": seed, **log}), flush=True)
                write(path, result)
        sync(torch.device(args.device))
        result["training_seconds"] = time.perf_counter() - start
        result["gradient_clip_fraction"] = clipped / len(rows)
        result["status"] = "TRAINED"
        result["final_parameter_sha256"] = tensor_hash(model.state_dict())
        torch.save({"arm": arm, "seed": seed, "completed_updates": result["completed_updates"],
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                   out / f"{arm}_seed{seed}.pt")
        write(path, result)
        model.eval()
        for size, test in tests.items():
            write(out / "status.json", {"status": "RUNNING", "phase": "evaluation", "arm": arm,
                  "seed": seed, "size": size, "pid": os.getpid(), "updated_utc": now()})
            result["evaluation"][str(size)] = evaluate(model, test, args.preflight)
            write(path, result)
        result["latency"] = benchmark(model, tests[32]["x"][:1])
        result["status"] = "COMPLETE"
    except TimeoutError as error:
        result["status"], result["error"] = "TIME_BUDGET", str(error)
    except FloatingPointError as error:
        result["status"], result["error"] = "NONFINITE", str(error)
    except Exception as error:
        result["status"], result["error"] = "ERROR", repr(error)
        result["traceback"] = traceback.format_exc()
    finally:
        result["elapsed_seconds"] = time.perf_counter() - start
        if result["status"] != "COMPLETE" and not (out / f"{arm}_seed{seed}.pt").exists():
            torch.save({"arm": arm, "seed": seed, "completed_updates": result["completed_updates"],
                        "partial": True, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                       out / f"{arm}_seed{seed}.pt")
        write(path, result)
        del optimizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def summary(out, results, preflight):
    seeds = [0] if preflight else [0, 1]
    expected_updates = 3 if preflight else 600
    expected_sizes = {"32"} if preflight else {"32", "64"}
    complete = len(results) == 2 * len(seeds) and all(
        r["status"] == "COMPLETE" and r["completed_updates"] == expected_updates
        and set(r["evaluation"]) == expected_sizes for r in results)
    by_key = {(r["arm"], r["seed"]): r for r in results}
    pairs = []
    for seed in seeds:
        a, r = [by_key.get((arm, seed)) for arm in ARMS]
        if a is None or r is None:
            continue
        identical = {field: a[field] == r[field] for field in
                     ["initial_parameter_sha256", "train_data_sha256", "schedule_sha256"]}
        if not all(identical.values()):
            complete = False
        pair = {"seed": seed, "paired_identity": identical}
        if not preflight and a["status"] == r["status"] == "COMPLETE":
            ev_a, ev_r = a["evaluation"]["32"], r["evaluation"]["32"]
            ca, cr = ev_a["curve"]["64"], ev_r["curve"]["64"]
            ia, ir = ev_a["interventions_from_T64"]["64"], ev_r["interventions_from_T64"]["64"]
            delta = ev_r["hold_min_ba"] - ev_a["hold_min_ba"]
            pair.update({
                "additive_hold": ev_a["hold_min_ba"], "revision_hold": ev_r["hold_min_ba"],
                "hold_effect_pp": 100 * delta,
                "reach_effect_pp": 100 * (cr["balanced_accuracy"] - ca["balanced_accuracy"]),
                "checks": {
                    "positive_hold_effect": delta > 0,
                    "revision_reach_ba85": cr["balanced_accuracy"] >= .85,
                    "revision_reach_paired50": cr["paired"] >= .50,
                    "reach_noninferior3pp": cr["balanced_accuracy"] >= ca["balanced_accuracy"] - .03,
                    "hold_within3pp_of_reach": ev_r["hold_min_ba"] >= cr["balanced_accuracy"] - .03,
                    "switch_changed80": ir["switch"]["changed_accuracy"] >= .80,
                    "switch_unchanged85": ir["switch"]["unchanged_accuracy"] >= .85,
                    "switch_changed_noninferior3pp": ir["switch"]["changed_accuracy"] >= ia["switch"]["changed_accuracy"] - .03,
                    "switch_unchanged_noninferior3pp": ir["switch"]["unchanged_accuracy"] >= ia["switch"]["unchanged_accuracy"] - .03,
                    "repair_ba85": ir["z_damage"]["balanced_accuracy"] >= .85,
                    "repair_noninferior3pp": ir["z_damage"]["balanced_accuracy"] >= ia["z_damage"]["balanced_accuracy"] - .03,
                }})
        pairs.append(pair)
    mean_effect, decision = None, "INCOMPLETE_OR_INVALID"
    if preflight:
        decision = "PREFLIGHT_PASSED" if complete else "PREFLIGHT_FAILED"
    elif complete:
        mean_effect = float(np.mean([p["hold_effect_pp"] for p in pairs]))
        passed = mean_effect >= 5 and all(all(p["checks"].values()) for p in pairs)
        decision = "REVISION_JOINT_SCREEN_PASS" if passed else "NO_JOINT_SCREEN_PASS"
    report = {"decision": decision, "complete": complete, "mean_hold_effect_pp": mean_effect,
              "pairs": pairs, "statuses": [
                  {"arm": arm, "seed": seed,
                   "status": by_key.get((arm, seed), {}).get("status", "NOT_STARTED"),
                   "updates": by_key.get((arm, seed), {}).get("completed_updates", 0)}
                  for seed in seeds for arm in ARMS],
              "limits": "Two paired model seeds, shared evaluation maps; descriptive only. No idempotence/stability guarantee. Two communication phases per macro-step."}
    write(out / "aggregate.json", report)
    lines = ["# Workspace + revision paired screen", "", f"Decision: {decision}.",
             "Engineering preflight only." if preflight else "Two-seed exploratory screen; not a population-level superiority claim.",
             "", "| Arm | Seed | Status | Updates | Size | BA64 | Hold min BA | Paired64 | Switch changed K64 | Z-repair BA K64 |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "null" if value is None else f"{100*value:.2f}%"
    for r in results:
        for size, ev in r["evaluation"].items():
            c, i = ev["curve"]["64"], ev["interventions_from_T64"]["64"]
            lines.append(f"| {r['arm']} | {r['seed']} | {r['status']} | {r['completed_updates']} | {size} | "
                         f"{fmt(c['balanced_accuracy'])} | {fmt(ev['hold_min_ba'])} | {fmt(c['paired'])} | "
                         f"{fmt(i['switch']['changed_accuracy'])} | {fmt(i['z_damage']['balanced_accuracy'])} |")
    lines += ["", f"Mean paired hold effect (revision - additive): {mean_effect} pp.",
              "See aggregate.json for every frozen predicate; raw arm JSON retains all endpoints/controls.",
              "Hold = minimum aggregate BA at T128/192/256. Null is unavailable, never success.",
              "Repair uses specified damage only; source switch retains W and Z.",
              "Prior checkpoint scores use different training and are not matched controls.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def source_hashes():
    paths = list((ROOT / "new/workspace_revision").glob("*.py"))
    paths += [ROOT / "new/workspace_revision/PROTOCOL.md",
              ROOT / "new/masked_medium/masked_cells.py",
              ROOT / "new/nca_inertial_wind_tunnel/cells.py",
              ROOT / "new/nca_inertial_wind_tunnel/tasks.py",
              ROOT / "tools/launch_revision.ps1"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def main():
    global DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device != "cuda" or not torch.cuda.is_available():
        parser.error("Frozen screen requires CUDA; CPU model checks use check.py")
    torch.backends.cudnn.benchmark = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cap = 300 if args.preflight else 1500
    DEADLINE = started + cap
    hashes = source_hashes()
    for name in hashes:
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    manifest = {"protocol": "workspace_revision_v1", "started_utc": now(), "pid": os.getpid(),
                "preflight": args.preflight, "training": True, "config": vars(args),
                "seeds": [0] if args.preflight else [0, 1], "updates_per_arm": 3 if args.preflight else 600,
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "numpy": np.__version__,
                "maximum_minutes": cap / 60, "source_sha256": hashes,
                "backend": {"cudnn_benchmark": torch.backends.cudnn.benchmark,
                            "cudnn_deterministic": torch.backends.cudnn.deterministic,
                            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn_tf32": torch.backends.cudnn.allow_tf32}}
    write(out / "manifest.json", manifest)
    write(out / "status.json", {"status": "STARTED", "pid": os.getpid(), "updated_utc": now()})
    def forced_timeout():
        write(out / "watchdog_timeout.json", {"status": "HARD_TIME_BUDGET", "utc": now(), "pid": os.getpid()})
        os._exit(124)
    watchdog = threading.Timer(cap + 60, forced_timeout)
    watchdog.daemon = True
    watchdog.start()
    results = []
    try:
        tests = {n: bank(n, 2 if args.preflight else 16, 30000 + n, args.device)
                 for n in ([32] if args.preflight else [32, 64])}
        manifest["evaluation_data_sha256"] = {str(n): tensor_hash(v) for n, v in tests.items()}
        write(out / "manifest.json", manifest)
        for seed in manifest["seeds"]:
            check_budget()
            train = bank(32, 512, 10000 + seed, args.device)
            rows = schedule(seed, manifest["updates_per_arm"], 8, 512, args.preflight)
            write(out / f"schedule_seed{seed}.json", rows)
            order = list(ARMS) if seed == 0 else list(reversed(ARMS))
            for arm in order:
                check_budget()
                result = train_one(args, arm, seed, train, rows, tests, out)
                results.append(result)
                summary(out, results, args.preflight)
            del train
        report = summary(out, results, args.preflight)
        status = report["decision"] if args.preflight else ("COMPLETE" if report["complete"] else "FAILED_OR_INCOMPLETE")
    except TimeoutError:
        summary(out, results, args.preflight)
        status = "TIME_BUDGET"
    except Exception as error:
        write(out / "error.json", {"error": repr(error), "traceback": traceback.format_exc()})
        summary(out, results, args.preflight)
        status = "ERROR"
    finally:
        watchdog.cancel()
    write(out / "status.json", {"status": status, "pid": os.getpid(), "finished_utc": now(),
          "elapsed_seconds": time.monotonic() - started, "completed_arms": len(results)})
    print(json.dumps({"status": status, "elapsed_seconds": time.monotonic() - started}), flush=True)
    expected_status = "PREFLIGHT_PASSED" if args.preflight else "COMPLETE"
    if status != expected_status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
