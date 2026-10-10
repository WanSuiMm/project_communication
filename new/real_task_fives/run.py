"""Terminal-supervised conditional NCA screen on prepared real FIVES images."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from cells import ConditionalNCACell
from common import (CONFIG, atomic_json, atomic_torch, read_json, run_lock, setup,
                    source_hashes, masked_loss, finite_model, sample_batch, training_fires, runtime_binding)
from data import sha256, split_ids, id_sort_key, inner_folds
from metrics import segmentation_metrics


def new_model(device):
    torch.manual_seed(CONFIG["init_seed"])
    return ConditionalNCACell(CONFIG["channels"], CONFIG["hidden"], CONFIG["perception_width"]).to(device)


def rollout_train(model, batch, fires, mode, credit, use_checkpoint=True):
    """No parameter updates inside execution; terminal loss only."""
    state = model.initialize(batch["coarse"], batch["fov"], mode)
    features = model.encode_image(batch["rgb"])
    prefix = len(fires) - credit
    if prefix:
        with torch.no_grad():
            for fire in fires[:prefix]:
                state = model.step(state, features.detach(), fire, batch["fov"], mode)
        state = model.detach(state, mode)

    def segment(current, image_features, selected_fires, fov):
        for fire in selected_fires:
            current = model.step(current, image_features, fire, fov, mode)
        return current

    for start in range(prefix, len(fires), CONFIG["checkpoint_chunk"]):
        selected = fires[start:start + CONFIG["checkpoint_chunk"]]
        if use_checkpoint and mode == "original":
            state = checkpoint(segment, state, features, selected, batch["fov"],
                               use_reentrant=False, preserve_rng_state=False)
        else:
            state = segment(state, features, selected, batch["fov"])
    return model.visible(state, mode)


def train_update(model, optimizer, images, update, mode, credit, steps, device):
    batch = sample_batch(images, update, device)
    fires = training_fires(update, steps, batch["coarse"].shape, device)
    optimizer.zero_grad(set_to_none=True)
    visible = rollout_train(model, batch, fires, mode, credit)
    a, b = CONFIG["halo"], CONFIG["halo"] + CONFIG["scored_center"]
    loss = masked_loss(visible[:, :1, a:b, a:b], batch["target"][:, :, a:b, a:b],
                       batch["fov"][:, :, a:b, a:b])
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(visible).all()):
        raise FloatingPointError("Nonfinite loss/state before backward")
    loss.backward()
    norms = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"Nonfinite gradient: {name}")
            norm = parameter.grad.norm()
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError(f"Nonfinite gradient norm: {name}")
            norms[name] = float(norm)
            parameter.grad.div_(norm + 1e-8)
    optimizer.step()
    if not finite_model(model):
        raise FloatingPointError("Nonfinite model after optimizer update")
    return {"update": update, "loss": float(loss.detach()), "gradient_norms": norms}


def load_prepared(prepared):
    prepared = Path(prepared)
    manifest = read_json(prepared / "manifest.json")
    train_ids, val_ids = split_ids(manifest["image_ids"]["official_training"])
    if (manifest["image_ids"]["outer_train"] != train_ids or
            manifest["image_ids"]["outer_validation"] != val_ids or
            manifest["excluded_test_labels"]["loaded"]):
        raise ValueError("Prepared data split/test-label policy mismatch")
    folds = inner_folds(train_ids)
    if manifest["inner_oof_folds"] != folds:
        raise ValueError("Prepared OOF assignment mismatch")
    teachers = {row["role"]: row for row in manifest["model_checkpoints"]}
    plans = [(f"fold{f['fold']}", f["train_ids"], f["heldout_ids"], CONFIG["oof_teacher_steps"])
             for f in folds] + [("full", train_ids, val_ids, CONFIG["full_teacher_steps"])]
    if set(teachers) != {plan[0] for plan in plans}:
        raise ValueError("Prepared teacher roles mismatch")
    for role, fit_ids, predict_ids, steps in plans:
        row = teachers[role]
        teacher_path = prepared / "teachers" / role / row["file"]
        if (Path(row["file"]).name != row["file"] or
                row["train_ids"] != sorted(fit_ids, key=id_sort_key) or
                row["provenance"] != {"train_ids": fit_ids, "predict_ids": predict_ids, "steps": steps} or
                sha256(teacher_path) != row["sha256"]):
            raise ValueError(f"Prepared teacher provenance mismatch: {role}")
    expected = {row["id"]: row for row in manifest["prepared_files"]}
    if set(expected) != set(train_ids + val_ids):
        raise ValueError("Prepared image IDs mismatch")
    images = {}
    for image_id in train_ids + val_ids:
        row = expected[image_id]
        if Path(row["file"]).name != row["file"]:
            raise ValueError("Prepared filename must be a basename")
        path = prepared / row["file"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Prepared image changed: {image_id}")
        with np.load(path, allow_pickle=False) as arrays:
            image = {key: arrays[key].astype(np.float32) for key in ("rgb", "target", "fov", "coarse")}
            if str(arrays["id"]) != image_id:
                raise ValueError("NPZ image ID mismatch")
        spatial = image["rgb"].shape[-2:]
        for key, count in (("rgb", 3), ("target", 1), ("fov", 1), ("coarse", 1)):
            value = image[key]
            if value.shape != (count, *spatial) or not np.isfinite(value).all():
                raise ValueError(f"Invalid prepared {key} shape or value")
            if np.any(value < 0) or np.any(value > 1):
                raise ValueError(f"Out-of-range prepared {key}")
            if key in ("target", "fov") and np.any((value != 0) & (value != 1)):
                raise ValueError(f"Prepared {key} must be binary")
        image["id"] = image_id
        images[image_id] = image
    return [images[i] for i in train_ids], [images[i] for i in val_ids]


def summarize(rows):
    """Average repetitions within image, then give each original image one vote."""
    result = {}
    for horizon in sorted({row["t"] for row in rows}):
        selected = [row for row in rows if row["t"] == horizon]
        ids = sorted({row["id"] for row in selected})
        keys = selected[0]["metrics"].keys()
        per_image = {i: {key: float(np.mean([row["metrics"][key] for row in selected if row["id"] == i]))
                          for key in keys} for i in ids}
        result[str(horizon)] = {"per_image": per_image,
            "mean": {key: float(np.mean([per_image[i][key] for i in ids])) for key in keys}}
    return result


def coarse_baseline(images):
    rows = [{"id": image["id"], "repeat": 0, "t": 0,
             "metrics": segmentation_metrics(image["coarse"] * image["fov"],
                                             image["target"] * image["fov"], image["fov"])}
            for image in images]
    return {"rows": rows, "summary": summarize(rows)}


def diagnostic_images(images):
    selected = []
    for category in ("A", "D", "G", "N"):
        group = sorted([image for image in images if image["id"].endswith("_" + category)],
                       key=lambda image: id_sort_key(image["id"]))
        selected.extend(group[:CONFIG["diagnostic_per_category"]])
    return selected


@torch.no_grad()
def evaluate(model, images, device, hidden_reset=False, horizons=None):
    """Common materialized transition for all arms at fixed inference parameters."""
    rows = []
    horizons = CONFIG["eval_steps"] if horizons is None else horizons
    model.eval()
    for image in images:
        fov = torch.from_numpy(image["fov"][None]).to(device)
        rgb = torch.from_numpy(image["rgb"][None]).to(device)
        coarse = torch.from_numpy(image["coarse"][None]).to(device)
        image_features = model.encode_image(rgb)
        for repeat in range(CONFIG["evaluation_repeats"]):
            generator = torch.Generator(device=device).manual_seed(
                CONFIG["eval_seed"] + int(hashlib.sha256(image["id"].encode()).hexdigest()[:8], 16) + 104729 * repeat)
            state = model.initialize(coarse, fov, "original")
            for t in range(1, max(horizons) + 1):
                fire = (torch.rand(fov.shape, generator=generator, device=device) < .5).float()
                state = model.step(state, image_features, fire, fov, "original")
                if t in horizons:
                    if not bool(torch.isfinite(state).all()):
                        raise FloatingPointError(f"Nonfinite inference state at T{t}, image {image['id']}")
                    probability = (state[:, :1].sigmoid() * fov)[0].cpu().numpy()
                    rows.append({"id": image["id"], "repeat": repeat, "t": t,
                                 "metrics": segmentation_metrics(probability, image["target"] * image["fov"], image["fov"])})
                if hidden_reset and t == 32:
                    # Preserves the current output; this is an evaluation-only intervention.
                    state = state.clone()
                    state[:, 1:] = 0
    return {"hidden_reset_at_32": hidden_reset, "inference_representation": "materialized_original",
            "rows": rows, "summary": summarize(rows)}


def train_arm(name, training, validation, out, device, binding):
    mode, credit, steps = CONFIG["arms"][name]
    auxiliary = name == "standard_t8"
    horizons = [8] if auxiliary else CONFIG["eval_steps"]
    diagnostics = diagnostic_images(validation)
    arm = out / name
    arm.mkdir(exist_ok=True)
    result_path = arm / "result.json"
    if result_path.exists():
        saved = read_json(result_path)
        if saved["binding"] != binding:
            raise ValueError("Completed arm binding changed")
        return saved
    model = new_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    latest = arm / "checkpoints" / "latest.pt"
    start = 0
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["binding"] != binding or saved["arm"] != name:
            raise ValueError("Checkpoint source/data/config binding mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["update"]
        # Resume an interrupted named-checkpoint evaluation before advancing.
        if start in CONFIG["checkpoints"] and not (arm / f"eval_u{start}.json").exists():
            selected = validation if start == CONFIG["updates"] else diagnostics
            selected_horizons = horizons if start == CONFIG["updates"] else ([8] if auxiliary else [64])
            atomic_json(arm / f"eval_u{start}.json", evaluate(model, selected, device, horizons=selected_horizons))
    # Preserve interrupted attempts as evidence without silently merging duplicate updates.
    attempt = uuid.uuid4().hex[:12]
    log = arm / f"training_attempt_{attempt}.jsonl"
    with log.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"row_type": "attempt_start", "attempt": attempt,
                                "restored_update": start}) + "\n")
    for update in range(start + 1, CONFIG["updates"] + 1):
        model.train()
        row = train_update(model, optimizer, training, update, mode, credit, steps, device)
        if update == 1 or update % 10 == 0:
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            atomic_json(out / "progress.json", {"status": "RUNNING", "arm": name, **row})
        if update % CONFIG["recovery_interval"] == 0 or update in CONFIG["checkpoints"]:
            saved = {"binding": binding, "arm": name, "update": update,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     "schedule": "stateless per-update named seeds"}
            atomic_torch(latest, saved)
        if update in CONFIG["checkpoints"]:
            atomic_torch(arm / "checkpoints" / f"u{update}.pt", saved)
            selected = validation if update == CONFIG["updates"] else diagnostics
            selected_horizons = horizons if update == CONFIG["updates"] else ([8] if auxiliary else [64])
            atomic_json(arm / f"eval_u{update}.json", evaluate(model, selected, device, horizons=selected_horizons))
    # If interrupted during an evaluation, the final weights can be recovered from latest.
    final_eval_path = arm / f"eval_u{CONFIG['updates']}.json"
    if not final_eval_path.exists():
        atomic_json(final_eval_path, evaluate(model, validation, device, horizons=horizons))
    intervention = None if auxiliary else evaluate(model, diagnostics, device, hidden_reset=True)
    result = {"binding": binding, "arm": name, "update": CONFIG["updates"],
              "parameters": model.trainable_parameter_count, "state_channels": model.state_channels,
              "final": read_json(final_eval_path), "hidden_reset": intervention,
              "training_attempts": [p.name for p in sorted(arm.glob("training_attempt_*.jsonl"))]}
    atomic_json(result_path, result)
    return result


def mean_metric(result, t, key):
    return result["final"]["summary"][str(t)]["mean"][key]


def qualification(baseline, k64, short=None):
    coarse = baseline["summary"]["0"]["mean"]
    gates = {"coarse_dice_floor": coarse["dice"] >= .60,
             "coarse_cldice_floor": coarse["cldice"] >= .50,
             "k64_cldice_gain": mean_metric(k64, 64, "cldice") - coarse["cldice"] >= .01,
             "k64_dice_guard": mean_metric(k64, 64, "dice") - coarse["dice"] >= -.005,
             "within_model_depth": mean_metric(k64, 64, "cldice") - mean_metric(k64, 8, "cldice") >= .01}
    if short is not None:
        gates["trained_short_reference"] = mean_metric(k64, 64, "cldice") - mean_metric(short, 8, "cldice") >= .005
    return {"gates": gates, "qualified": all(gates.values()) and short is not None,
            "preliminary_pass": all(gates.values())}


def write_report(out, aggregate):
    atomic_json(out / "aggregate.json", aggregate)
    lines = ["# FIVES real-task NCA v0", "", f"Status: **{aggregate['status']}**.", "",
             "One developmental paired block; original retinal image is the measurement unit.", "",
             "| Arm | T8 clDice | T64 clDice | T64 Dice | T256 clDice |",
             "|---|---:|---:|---:|---:|"]
    for name, result in aggregate.get("arms", {}).items():
        values = [f"{mean_metric(result, t, key):.6f}" if str(t) in result["final"]["summary"] else "n/a"
                  for t, key in ((8, "cldice"), (64, "cldice"), (64, "dice"), (256, "cldice"))]
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    lines += ["", f"Verdict: **{aggregate.get('verdict', 'PENDING')}**.", "",
              "Qualification gates and image-level paired measurements: `aggregate.json`.",
              "Per-plan trajectories of metrics and evaluation-only hidden resets: arm result JSON.",
              "FIVES512 development resolution;120 validation images. No test labels or clinical claims."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def experiment(prepared, out, device, resume=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    binding = {"sources": source_hashes(), "config": CONFIG,
               "prepared_manifest_sha256": sha256(Path(prepared) / "manifest.json"),
               "runtime": runtime_binding(device)}
    with run_lock(out):
        provenance = out / "binding.json"
        if provenance.exists():
            if not resume:
                raise FileExistsError("Run exists; use --resume explicitly")
            if read_json(provenance) != binding:
                raise ValueError("Run binding changed; create a fresh run")
        else:
            atomic_json(provenance, binding)
        training, validation = load_prepared(prepared)
        aggregate = {"status": "RUNNING", "verdict": "PENDING", "arms": {}, "binding": binding}
        started = time.time()
        try:
            baseline = coarse_baseline(validation)
            aggregate["baseline"] = baseline
            values = baseline["summary"]["0"]["mean"]
            if values["dice"] < .60 or values["cldice"] < .50:
                aggregate.update(status="COMPLETE", verdict="COARSE_BASELINE_UNQUALIFIED")
                write_report(out, aggregate)
                atomic_json(out / "progress.json", {"status": "COMPLETE", "verdict": aggregate["verdict"]})
                return aggregate
            for name in ("standard_k64", "standard_t8", "standard_k8", "au_k8"):
                aggregate["arms"][name] = train_arm(name, training, validation, out, device, binding)
                if name in ("standard_k64", "standard_t8"):
                    q = qualification(baseline, aggregate["arms"]["standard_k64"],
                                      aggregate["arms"].get("standard_t8"))
                    aggregate["qualification"] = q
                    if not q["preliminary_pass"]:
                        aggregate.update(status="COMPLETE", verdict="TASK_UNQUALIFIED")
                        write_report(out, aggregate)
                        atomic_json(out / "progress.json", {"status": "COMPLETE", "verdict": aggregate["verdict"]})
                        return aggregate
                write_report(out, aggregate)
            k64, k8, au = [aggregate["arms"][i] for i in ("standard_k64", "standard_k8", "au_k8")]
            gap = mean_metric(k64, 64, "cldice") - mean_metric(k8, 64, "cldice")
            gain = mean_metric(au, 64, "cldice") - mean_metric(k8, 64, "cldice")
            gates = {"credit_gap": gap >= .01, "au_gain": gain >= .005,
                     "gap_closure": gap > 0 and gain >= .25 * gap,
                     "dice_guard": mean_metric(au, 64, "dice") >= mean_metric(k8, 64, "dice") - .005}
            verdict = ("NO_QUALIFIED_CREDIT_GAP" if not gates["credit_gap"] else
                       "AU_BENEFIT_DEVELOPMENTAL" if all(gates.values()) else "NO_QUALIFIED_AU_BENEFIT")
            aggregate.update(status="COMPLETE", verdict=verdict, comparison={"gap": gap, "gain": gain, "gates": gates})
            aggregate["elapsed_seconds_this_process"] = time.time() - started
            write_report(out, aggregate)
            atomic_json(out / "progress.json", {"status": "COMPLETE", "verdict": verdict})
        except Exception:
            aggregate.update(status="ERROR", verdict="INCOMPLETE", error=traceback.format_exc())
            write_report(out, aggregate)
            atomic_json(out / "progress.json", {"status": "ERROR", "error": traceback.format_exc()})
            raise
    return aggregate


def software_check(out):
    """A small checkpointing check with arrays, never a synthetic research task."""
    from checks import run_checks
    report = run_checks()
    device = setup("cpu")
    model = new_model(device).double()
    with torch.no_grad():
        model.projection.weight.normal_(0, .01)
    other = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(77)
    batch = {"rgb": torch.rand(1, 3, 9, 9, generator=generator, dtype=torch.float64),
             "coarse": torch.rand(1, 1, 9, 9, generator=generator, dtype=torch.float64),
             "fov": torch.ones(1, 1, 9, 9, dtype=torch.float64)}
    fires = (torch.rand(16, 1, 1, 9, 9, generator=generator) < .5).double()
    a = rollout_train(model, batch, fires, "original", 16, True)
    b = rollout_train(other, batch, fires, "original", 16, False)
    a.square().mean().backward()
    b.square().mean().backward()
    errors = [float((p.grad - q.grad).abs().max()) for p, q in zip(model.parameters(), other.parameters())]
    report["activation_checkpoint_max_forward_error"] = float((a - b).abs().max().detach())
    report["activation_checkpoint_max_gradient_error"] = max(errors)
    report["passed"] = report["passed"] and max(errors) < 1e-10
    from coarse import SmallUNet
    from data import inner_folds
    teacher = SmallUNet().eval()
    with torch.no_grad():
        logits = teacher(torch.zeros(1, 3, 32, 32))
    report["coarse_forward_shape"] = list(logits.shape)
    fixture_ids = [f"{i}_{c}" for c in ("A", "D", "G", "N") for i in range(1, 151)]
    train_ids, val_ids = split_ids(fixture_ids)
    folds = inner_folds(train_ids)
    report["oof_split_disjoint"] = all(not set(f["train_ids"]) & set(f["heldout_ids"]) for f in folds)
    report["outer_split_disjoint"] = not set(train_ids) & set(val_ids)
    line = np.zeros((9, 9), np.float32)
    line[4] = 1
    metrics = segmentation_metrics(line, line, np.ones_like(line))
    report["perfect_line_metrics"] = metrics
    report["passed"] = (report["passed"] and report["oof_split_disjoint"] and report["outer_split_disjoint"]
                        and tuple(logits.shape) == (1, 1, 32, 32) and bool(torch.isfinite(logits).all())
                        and metrics["dice"] == metrics["cldice"] == 1)
    report["sources"] = source_hashes()
    atomic_json(out, report)
    if not report["passed"]:
        raise AssertionError("Software check failed")
    print(json.dumps({"passed": True, "training_launched": False, "report": str(out)}))


def gpu_smoke(out, device, example):
    """One disposable update at each memory-critical shape, using real pixels."""
    if device.type != "cuda":
        raise RuntimeError("Deployment smoke requires the selected CUDA GPU")
    smoke_example = {**example, "coarse": np.full_like(example["target"], .1)}
    rows = []
    for name in ("standard_k64", "au_k8"):
        mode, credit, steps = CONFIG["arms"][name]
        model = new_model(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        value = train_update(model, optimizer, [smoke_example], 1, mode, credit, steps, device)
        torch.cuda.synchronize(device)
        rows.append({"arm": name, "seconds": time.perf_counter() - started,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                     "loss": value["loss"], "finite": finite_model(model)})
        del model, optimizer
        torch.cuda.empty_cache()
    report = {"scope": "two disposable software-smoke updates, not scientific training",
              "real_image_id": example["id"], "coarse_input": "constant0.1 for software shape check only", "rows": rows,
              "passed": all(row["finite"] for row in rows), "sources": source_hashes()}
    atomic_json(out, report)
    if not report["passed"]:
        raise AssertionError("GPU smoke failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        software_check(Path(args.out))
    else:
        if not args.prepared:
            parser.error("--prepared is required for a real-data experiment")
        experiment(args.prepared, Path(args.out), setup(args.device), args.resume)


if __name__ == "__main__":
    main()
