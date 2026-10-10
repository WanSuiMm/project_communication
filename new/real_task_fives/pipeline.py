"""Official data acquisition -> one GPU smoke -> real-data qualification."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

from common import atomic_json, CONFIG, setup, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.time()
    status_path = root / "pipeline_status.json"

    def status(phase, **extra):
        atomic_json(status_path, {"status": "RUNNING", "phase": phase,
                    "elapsed_seconds": time.time() - started, **extra})
        print(phase, flush=True)

    try:
        from acquire import acquire_official
        from data import discover_fives, load_image, split_ids
        from prepare import prepare
        from run import experiment, software_check, gpu_smoke
        status("ACQUIRING_OFFICIAL_FIVES", scientific_training_started=False)
        extracted = acquire_official(root / "data/downloads", root / "data/fives")
        dataset = discover_fives(extracted)
        train_ids, val_ids = split_ids(dataset)
        example = load_image(dataset.train[0])
        atomic_json(root / "data_qualification.json", {
            "source": "official Figshare19688169/file34969398", "dataset": "FIVES",
            "train_count": len(train_ids), "validation_count": len(val_ids),
            "example_id": example["id"], "rgb_shape": list(example["rgb"].shape),
            "target_shape": list(example["target"].shape), "test_labels_loaded": False,
            "config": CONFIG, "source_hashes": source_hashes()})
        status("SOFTWARE_AND_GPU_SMOKE", scientific_training_started=False)
        software_check(root / "software_check.json")
        device = setup(args.device)
        gpu_smoke(root / "gpu_smoke.json", device, example)
        status("TRAINING_OOF_COARSE_MODELS", scientific_training_started=True)
        prepared = prepare(extracted, root / "data/prepared", device, args.resume)
        status("NCA_K64_QUALIFICATION", scientific_training_started=True)
        result = experiment(prepared, root / "runs/fives_nca_20261010_01", device, args.resume)
        atomic_json(status_path, {"status": "COMPLETE", "phase": "FINISHED", "verdict": result["verdict"],
                                 "elapsed_seconds": time.time() - started})
    except Exception:
        atomic_json(status_path, {"status": "ERROR", "phase": "INCOMPLETE", "error": traceback.format_exc(),
                                 "elapsed_seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
