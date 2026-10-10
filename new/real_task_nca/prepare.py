"""Train leakage-controlled frozen coarse segmenters and prepare DRIVE NPZs."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

from common import CONFIG, atomic_json, read_json, run_lock, setup, source_hashes, runtime_binding
from coarse import train_teacher, load_teacher, predict_image
from data import (discover_drive, split_ids, inner_folds, load_image,
                  save_prepared_npz, build_manifest, write_manifest)


def prepare(drive, out, device, resume=False):
    dataset = discover_drive(drive)
    train_ids, val_ids = split_ids()
    source = build_manifest(dataset)
    binding = {"sources": source["source_files"], "code": source_hashes(), "config": CONFIG,
               "runtime": runtime_binding(device)}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        binding_file = out / "binding.json"
        if binding_file.exists():
            if not resume:
                raise FileExistsError("Preparation exists; use --resume explicitly")
            if read_json(binding_file) != binding:
                raise ValueError("Preparation source/config/data changed; use a fresh directory")
            if (out / "manifest.json").exists():
                return out
        else:
            atomic_json(binding_file, binding)
        records = {record.id: record for record in dataset.train}
        files, teachers = {}, {}
        folds = inner_folds(train_ids)
        plans = [(f"fold{f['fold']}", f["train_ids"], f["heldout_ids"], CONFIG["oof_teacher_steps"])
                 for f in folds]
        plans.append(("full", train_ids, val_ids, CONFIG["full_teacher_steps"]))
        started = time.time()
        try:
            for index, (role, fit_ids, prediction_ids, steps) in enumerate(plans):
                atomic_json(out / "progress.json", {"status": "RUNNING", "phase": role,
                            "fit_ids": fit_ids, "prediction_ids": prediction_ids})
                path = train_teacher(dataset.train, fit_ids, out / "teachers" / role,
                                     CONFIG["teacher_seeds"][index], steps, device)
                teachers[role] = {"path": path, "provenance": {"train_ids": fit_ids,
                                  "predict_ids": prediction_ids, "steps": steps}}
                teacher = load_teacher(path, device)
                for image_id in prediction_ids:
                    image = load_image(records[image_id])
                    coarse = predict_image(teacher, image["rgb"], device)
                    files[image_id] = save_prepared_npz(out / f"{image_id}.npz", image_id,
                                          image["rgb"], image["target"], image["fov"], coarse)
                del teacher
            manifest = build_manifest(dataset, files, teachers, {
                "protocol": CONFIG["protocol"], "code": source_hashes(),
                "prediction_policy": "OOF on outer-training; full teacher on outer-validation",
                "no_label_derived_corruption": True})
            write_manifest(out / "manifest.json", manifest)
            atomic_json(out / "progress.json", {"status": "COMPLETE", "elapsed_seconds": time.time() - started})
        except Exception:
            atomic_json(out / "progress.json", {"status": "ERROR", "phase": role,
                        "error": traceback.format_exc(), "elapsed_seconds": time.time() - started})
            raise
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", required=True, help="Official extracted DRIVE directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--then-run", help="Run NCA qualification in this fresh run directory after preparation")
    args = parser.parse_args()
    device = setup(args.device)
    prepared = prepare(args.drive, args.out, device, args.resume)
    if args.then_run:
        from run import experiment
        experiment(prepared, Path(args.then_run), device, args.resume)


if __name__ == "__main__":
    main()
