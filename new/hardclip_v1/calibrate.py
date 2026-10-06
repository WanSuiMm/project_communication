"""Replay all eight old neural checkpoints without task outcomes to freeze caps."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "new"))
from hybrid_writer.cells import make_model
from primitive import calibrate_write_caps, _self_test

PROTOCOL = "hardclip_v1_tail_calibration_v1"
PRIOR = ROOT / "runs/hybrid_writer_20261006_01"
PUBLIC_BINDING = ROOT / "HYBRID_WRITER_PUBLICATION_MANIFEST.json"
BLOCKS = tuple(range(8))
INDICES = tuple(range(16))
STEPS = 64
TARGET = .01
MAX_TRIGGER = .20


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def tensor_hash(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        x = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str((x.shape, x.dtype)).encode()); h.update(x.tobytes())
    return h.hexdigest()


def forbidden_readout(*args):
    raise AssertionError("Calibration must never call the task readout")


def run(out):
    assert not out.exists(), "Refusing to overwrite calibration"
    assert torch.cuda.is_available() and str(torch.__version__).startswith("2.5.1")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    publication = json.loads(PUBLIC_BINDING.read_text())
    source = dict(publication["source_sha256"])
    source.update({p.relative_to(ROOT).as_posix(): sha(p) for p in HERE.iterdir()
                   if p.suffix in (".py", ".md")})
    for name, expected in source.items():
        assert sha(ROOT / name) == expected, name
    bank_path = PRIOR / "banks/train.npz"
    assert sha(bank_path) == publication["raw_artifact_bindings"]["banks/train.npz"]["sha256"]
    # NPZ is lazy: no label, flipped-input, distance or performance array is read.
    with np.load(bank_path, allow_pickle=False) as bank:
        selected = np.ascontiguousarray(bank["x"][list(INDICES)])
    assert selected.shape == (16, 3, 32, 32) and selected.dtype == np.float32
    x = torch.from_numpy(selected).cuda()
    open_mask = selected[:, 0].astype(bool)
    open_cells = int(open_mask.sum())
    assert open_cells > 0
    out.mkdir(parents=True)
    start = time.monotonic()
    config = {"protocol": PROTOCOL, "prior_run": PRIOR.name, "blocks": list(BLOCKS),
              "prior_arm": "neural", "prior_update": 300, "training_bank_seed": 10002,
              "map_indices": list(INDICES), "world": "original", "steps": STEPS,
              "target_removed_fraction": TARGET, "maximum_trigger_fraction": MAX_TRIGGER,
              "dose_tolerance": 1e-7, "sample_unit": "open cell at each step, per lane",
              "pooling": "equal block/time/open-cell events", "labels_used": False,
              "task_readout_called": False, "optimizer_constructed": False,
              "training_updates": 0, "caps_are_actual_write_rms": True}
    write(out / "config.json", config)
    write(out / "manifest.json", {"protocol": PROTOCOL, "pid": os.getpid(),
          "host": os.environ.get("COMPUTERNAME"), "started_utc": now(),
          "command": [sys.executable, *sys.argv], "gpu": torch.cuda.get_device_name(),
          "torch": str(torch.__version__), "numpy": str(np.__version__),
          "source_sha256": source, "training_bank_file_sha256": sha(bank_path),
          "publication_binding_sha256": sha(PUBLIC_BINDING), "runtime_limit_enforced": False})
    for name in source:
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dest)
    np.savez_compressed(out / "calibration_inputs.npz", x=selected, open_mask=open_mask,
                        map_indices=np.asarray(INDICES, dtype=np.int32))
    try:
        _self_test()
        records, samples = [], []
        for block in BLOCKS:
            relative = f"block{block:02d}/neural/checkpoints/u300.pt"
            binding = publication["checkpoint_bindings"][relative]
            path = PRIOR / relative
            assert sha(path) == binding["file_sha256"]
            payload = torch.load(path, map_location="cpu", weights_only=False)
            assert payload["completed_updates"] == 300 and payload["arm"] == "neural"
            model = make_model("neural", payload["initialization_seed"], "cuda")
            model.load_state_dict(payload["state_dict"], strict=True); model.eval()
            before = tensor_hash(model.state_dict())
            assert before == binding["parameter_sha256"]
            hook = model.readout.register_forward_pre_hook(forbidden_readout)
            values = []
            def observe(detail, incoming_x):
                raw = detail["delta"].reshape(16, 4, 6, 32, 32)
                values.append(raw.square().mean(dim=2).sqrt().detach())
            model.writer_observer = observe
            with torch.no_grad():
                state = model.initial(x)
                for _ in range(STEPS):
                    state = model.step(state, x)
                assert all(bool(torch.isfinite(v).all()) for v in state), "Nonfinite calibration state"
                values = torch.stack(values).cpu().numpy()
            model.writer_observer = None; hook.remove()
            assert tensor_hash(model.state_dict()) == before
            assert np.isfinite(values).all() and values.shape == (64, 16, 4, 32, 32)
            # [time,batch,y,x,lane] -> [time,open-cell,lane]. No sample subsampling.
            block_samples = values.transpose(0, 1, 3, 4, 2)[:, open_mask]
            assert block_samples.shape == (64, open_cells, 4)
            samples.append(block_samples.copy())
            records.append({"block": block, "checkpoint": relative, **binding,
                            "parameter_hash_unchanged": True,
                            "samples_per_lane": int(64 * open_cells)})
            write(out / "status.json", {"status": "RUNNING", "pid": os.getpid(),
                  "completed_blocks": len(records), "expected_blocks": 8, "phase": "sampling"})
            print(json.dumps({"sampled_block": block, "samples_per_lane": 64 * open_cells}), flush=True)
            del payload, model, state, values, block_samples
            gc.collect(); torch.cuda.empty_cache()
        values = np.stack(samples).astype(np.float32, copy=False)
        assert values.shape == (8, 64, open_cells, 4)
        np.savez_compressed(out / "raw_write_rms.npz", raw_write_rms=values,
                            blocks=np.asarray(BLOCKS, dtype=np.int32),
                            times=np.arange(1, 65, dtype=np.int32),
                            open_cell_indices=np.argwhere(open_mask).astype(np.int32))
        pooled = values.reshape(-1, 4)
        calibration = calibrate_write_caps(pooled, removed_fraction=TARGET,
                                           max_trigger_fraction=MAX_TRIGGER)
        per_block = []
        for row in calibration["lanes"]:
            lane = row["lane"]; cap = float(np.float32(row["max_write_rms"]))
            r = pooled[:, lane].astype(np.float64)
            dose = float(np.maximum(r-cap, 0).sum()/r.sum())
            activation = float(np.mean(r > cap))
            row.update(frozen_fp32_write_cap=cap, fp32_removed_fraction=dose,
                       fp32_trigger_fraction=activation)
            assert abs(dose - TARGET) <= 1e-7, (lane, dose)
            for block in BLOCKS:
                q = values[block, :, :, lane].astype(np.float64).reshape(-1)
                per_block.append({"block": block, "lane": lane, "trigger_fraction": float(np.mean(q>cap)),
                                  "removed_fraction": float(np.maximum(q-cap,0).sum()/q.sum()),
                                  "mean_raw_write_rms": float(q.mean()), "max_raw_write_rms": float(q.max())})
        passes = all(row["sparse_enough"] and row["fp32_trigger_fraction"] <= MAX_TRIGGER
                     for row in calibration["lanes"])
        assert passes == (calibration["status"] == "CALIBRATED")
        for name, expected in source.items():
            assert sha(ROOT / name) == expected, name
        calibration.update(protocol=PROTOCOL, completed_blocks=8, expected_blocks=8,
                           sampling_shape=list(values.shape), checkpoints=records,
                           training_authorized_by_calibration=passes,
                           task_outcomes_used=False, training_updates=0,
                           elapsed_seconds=time.monotonic()-start, finished_utc=now(),
                           per_block_dose=per_block, raw_samples_sha256=sha(out / "raw_write_rms.npz"),
                           source_binding_status="PASS", parameter_mutations=0,
                           frozen_caps=[row["frozen_fp32_write_cap"] for row in calibration["lanes"]])
        write(out / "summary.json", calibration)
        write(out / "status.json", {"status": "COMPLETE", "calibration_verdict": calibration["status"],
                                  "pid": os.getpid(), "completed_blocks": 8, "training_started": False})
        lines = ["# HardClip-v1: outcome-blind tail calibration", "",
                 f"Status: COMPLETE8/8. Verdict: {calibration['status']}.", "",
                 "Fixed target:1% removed raw-write magnitude per lane; maximum20% trigger rate.",
                 "", "|Lane|Frozen actual-write cap|Removed fraction|Trigger fraction|Sparse enough|",
                 "|---|---:|---:|---:|---|"]
        for row in calibration["lanes"]:
            lines.append(f"|{('N','E','S','W')[row['lane']]}|{row['frozen_fp32_write_cap']:.9g}|"
                         f"{row['fp32_removed_fraction']:.9g}|{row['fp32_trigger_fraction']:.9g}|{row['sparse_enough']}|")
        lines += ["", "All8 old neural u300 checkpoints, SAME16 fixed training maps, cold64 steps.",
                  "No labels, readout, optimizer, losses or task outcomes were used. Zero training updates.",
                  "", "Calibration PASS permits the frozen two-arm experiment; it is not efficacy evidence."
                  if passes else "At least one lane lacks the prescribed sparse tail. Efficacy training is NOT launched; no cutoff rescue.",
                  "", "Full indexed FP32 samples and checkpoint/source hashes are retained locally.",
                  f"Elapsed {calibration['elapsed_seconds']:.2f} seconds."]
        (out / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        print(json.dumps({"status": "COMPLETE", "verdict": calibration["status"],
                          "frozen_caps": calibration["frozen_caps"],
                          "trigger_rates": [r["fp32_trigger_fraction"] for r in calibration["lanes"]],
                          "elapsed_seconds": calibration["elapsed_seconds"]}), flush=True)
    except Exception as exc:
        write(out / "status.json", {"status": "ERROR", "error": repr(exc), "pid": os.getpid(),
                                  "training_started": False, "elapsed_seconds": time.monotonic()-start})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out if args.out.is_absolute() else ROOT / args.out)
