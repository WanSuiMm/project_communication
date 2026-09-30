"""Frozen A0 calibration and 2x2 transport experiment; no automatic retries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import torch
from torch.nn import functional as F

from .a0_models import A0Model, VARIANTS
from .data import make_batch
from .runner import dump, evaluate, latency, sync


ROOT = Path(__file__).resolve().parent
CONDITIONS = (("train_shape_d16", 16, 28), ("large_shape_d16", 16, 140),
              ("d32", 32, 140), ("d64", 64, 140), ("d128", 128, 140))


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.glob("*.py"))}


def config():
    return dict(protocol="rt_a0_2d3d_v2_1", dimensions=[2, 3],
                seeds=[1729, 2718, 31415, 57721], width=64, message_width=32,
                groups=4, recurrent_steps=8, train_steps=480, batch_size=8,
                eval_batch_size=8, eval_per_axis=64, train_distances=[4, 8, 12, 16],
                lr=.003, weight_decay=.0001, gradient_clip=1., epsilon=1e-6,
                budget_minutes=25., device="cuda", variants=list(VARIANTS),
                fit_threshold=.99, per_axis_threshold=.98,
                distance_threshold=.99, axis_order="unchanged_v1_palindrome",
                confidence="sigmoid_per_group_shared_by_all_arms_initial_0.5",
                raw_definition="T(c*v)", normalized_definition="T(c*v)/(T(c)+epsilon)")


def fits(row, condition, cfg):
    ev = row.get("eval", {}).get(condition)
    return (row.get("status") == "completed" and ev is not None
            and ev["accuracy"] >= cfg["fit_threshold" if condition == "train_shape_d16" else "distance_threshold"]
            and all(axis["accuracy"] >= cfg["per_axis_threshold"] for axis in ev["by_axis"]))


def decisions(cfg, rows):
    answer = {}
    for dim in cfg["dimensions"]:
        controls = [r for r in rows if r["dim"] == dim and r["variant"] == "attention"]
        calibrated = (len(controls) == len(cfg["seeds"])
                      and all(fits(r, "train_shape_d16", cfg) for r in controls))
        entry = {"calibration": "PASS" if calibrated else
                 "PENDING" if len(controls) < len(cfg["seeds"]) else "INCONCLUSIVE_POSITIVE_CONTROL",
                 "arms": {}}
        for variant in VARIANTS[1:]:
            arm = [r for r in rows if r["dim"] == dim and r["variant"] == variant]
            if not calibrated:
                verdict = "NOT_RUN_CALIBRATION_NOT_PASSED"
            elif len(arm) < len(cfg["seeds"]):
                verdict = "INCOMPLETE"
            elif not all(r["status"] == "completed" for r in arm):
                verdict = "INCOMPLETE_BUDGET_OR_NUMERICS"
            elif not all(fits(r, "train_shape_d16", cfg) for r in arm):
                verdict = "NOT_QUALIFIED_FIT"
            elif not all(fits(r, c, cfg) for r in arm for c, _, _ in CONDITIONS[1:]):
                verdict = "FIT_PASS_GENERALIZATION_NOT_QUALIFIED"
            else:
                verdict = "A0_SINGLE_SOURCE_PASS"
            entry["arms"][variant] = verdict
        answer[str(dim)] = entry
    return answer


def paired_effects(cfg, rows):
    effects = []
    for dim in cfg["dimensions"]:
        for medium in ("constant", "learned"):
            for seed in cfg["seeds"]:
                pair = {r["variant"]: r for r in rows
                        if r["dim"] == dim and r["seed"] == seed and r["status"] == "completed"}
                raw, norm = pair.get(medium + "_raw"), pair.get(medium + "_normalized")
                if raw is None or norm is None:
                    continue
                effects.append(dict(dim=dim, medium=medium, seed=seed,
                                    normalized_minus_raw={c: norm["eval"][c]["accuracy"] - raw["eval"][c]["accuracy"]
                                                          for c, _, _ in CONDITIONS}))
    return effects


def write_summary(cfg, rows, out, status):
    verdict = decisions(cfg, rows)
    summary = dict(protocol=cfg["protocol"], status=status, decisions=verdict,
                   results=rows, paired_normalization_effects=paired_effects(cfg, rows),
                   independent_unit="training_seed", expected_trials_if_calibrated=40,
                   completed_trials=sum(r["status"] == "completed" for r in rows))
    dump(out / "aggregate.json", summary)
    lines = ["# A0: 2D / 3D transport qualification", "", f"Status: {status}", "",
             "Four training seeds are independent replicates. Cells, test examples and timing repetitions are not extra training replicates.", "",
             "| Dim | Attention calibration | constant raw | constant normalized | learned raw | learned normalized |",
             "|---|---|---|---|---|---|"]
    for dim, item in verdict.items():
        lines.append(f"| {dim}D | {item['calibration']} | " + " | ".join(item["arms"][v] for v in VARIANTS[1:]) + " |")
    lines += ["", "| Dim | Arm | Seed | train d16 | large d16 | d32 | d64 | d128 | ms/image | Status |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        values = [f"{row['eval'][c]['accuracy']:.4f}" if c in row.get("eval", {}) else "-" for c, _, _ in CONDITIONS]
        ms = f"{row['latency']['median_ms']:.2f}" if "latency" in row else "-"
        lines.append(f"| {row['dim']} | {row['variant']} | {row['seed']} | " + " | ".join(values) + f" | {ms} | {row['status']} |")
    lines += ["", "Interpretation: a single-source pass establishes qualification only, not learned routing, local-detail preservation, or an architecture advantage.",
              "Both raw and normalized RT arms use q=c*v, identical initial parameters within each medium/seed, and identical packed transport work. They differ only by receiver normalization. They are not the old v1 raw arm.",
              "Axis order is held fixed, with axis-level evaluations retained. No axis-equivariance claim or B/C escalation is made.",
              "The numerical/oracle report is checks.json. Diagnostics record learned confidence, denominator and source-mass contribution; oracle confidence is not given to trained models.",
              "No extra steps, replacement seeds, checkpoint selection or rescue sweeps are scheduled."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def trial(cfg, out, dim, variant, seed, deadline):
    key = f"A0_{dim}d_{variant}_seed{seed}"
    folder = out / key
    folder.mkdir()
    torch.manual_seed(seed)
    random.seed(seed)
    model = A0Model(dim, variant, cfg["width"], cfg["message_width"], cfg["groups"],
                    cfg["recurrent_steps"], cfg["epsilon"]).to(cfg["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    row = dict(key=key, dim=dim, variant=variant, seed=seed, status="running", history=[],
               parameters=sum(p.numel() for p in model.parameters()), steps_done=0)
    dump(folder / "result.json", row)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(cfg["train_steps"]):
        if time.monotonic() >= deadline:
            row["status"] = "time_budget_exhausted"
            break
        batch = make_batch(dim, "A", cfg["batch_size"], cfg["train_distances"][step % 4],
                           seed * 100000 + step, device=cfg["device"], long_size=28,
                           axis=(step // 4) % dim)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["x"], batch["target_index"])
        loss = F.cross_entropy(logits, batch["labels"])
        if not torch.isfinite(loss):
            row["status"] = "nonfinite_loss"
            break
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
        if not torch.isfinite(norm):
            row["status"] = "nonfinite_gradient"
            break
        optimizer.step()
        row["steps_done"] = step + 1
        if step == 0 or (step + 1) % 120 == 0:
            item = dict(step=step + 1, loss=float(loss.detach()),
                        batch_accuracy=float((logits.argmax(-1) == batch["labels"]).float().mean()),
                        unclipped_grad_norm=float(norm), elapsed_s=time.perf_counter() - started)
            row["history"].append(item)
            print(json.dumps(dict(trial=key, **item)), flush=True)
    sync(cfg["device"])
    row["training_seconds"] = time.perf_counter() - started
    row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    if row["status"] == "running":
        row["status"] = "completed"
    if row["status"] == "completed":
        model.eval()
        row["eval"] = {name: evaluate(model, dim, "A", distance, length, cfg["eval_per_axis"],
                                     cfg["eval_batch_size"], cfg["device"])
                       for name, distance, length in CONDITIONS}
        row["latency"] = latency(model, dim, "A", cfg["device"])
        row["diagnostics"] = []
        with torch.no_grad():
            for name, distance, length in (CONDITIONS[0], CONDITIONS[-1]):
                for axis in range(dim):
                    batch = make_batch(dim, "A", 8, distance, 812345 + axis,
                                       device=cfg["device"], long_size=length, axis=axis)
                    _, stats = model(batch["x"], batch["target_index"], diagnostics=True)
                    row["diagnostics"].append(dict(condition=name, axis=axis, **stats))
        torch.save(model.state_dict(), folder / "model.pt")
    dump(folder / "result.json", row)
    print(json.dumps(dict(finished=key, status=row["status"],
                          fit=row.get("eval", {}).get("train_shape_d16", {}).get("accuracy"))), flush=True)
    del model, optimizer
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "qualify"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--checks", help="Successful check report from the exact current Python sources")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = config()
    if not torch.cuda.is_available():
        raise SystemExit("This frozen local A0 requires CUDA; CPU is not substituted.")
    hashes = source_hashes()
    checked = None
    if args.mode == "qualify":
        if not args.checks:
            raise SystemExit("Run check first and supply its checks.json with --checks.")
        checked = json.loads(Path(args.checks).read_text(encoding="utf-8"))
        if checked.get("status") != "PASS" or checked.get("source_sha256") != hashes:
            raise SystemExit("Checks did not pass or Python sources changed since checks.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "config.json", dict(cfg, mode=args.mode))
    receipt = dict(started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                   host=platform.node(), python=platform.python_version(), torch=torch.__version__,
                   device=torch.cuda.get_device_name(0), cwd=str(Path.cwd()), argv=sys.argv,
                   source_sha256=hashes,
                   protocol_sha256=hashlib.sha256((ROOT / "A0_PROTOCOL.md").read_bytes()).hexdigest())
    dump(out / "launch_receipt.json", receipt)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for source in list(ROOT.glob("*.py")) + [ROOT / "A0_PROTOCOL.md"]:
        (snapshot / source.name).write_bytes(source.read_bytes())
    if args.mode == "check":
        from .a0_checks import oracle_diagnostics, run_checks
        try:
            numerical = run_checks(device=cfg["device"])
            oracle = oracle_diagnostics(device=cfg["device"])
            # The checks module owns assertions of the actual numerical/oracle contract.
            report = dict(status="PASS", source_sha256=hashes, numerical=numerical, oracle=oracle)
            dump(out / "checks.json", report)
            print(json.dumps(dict(status="PASS", out=str(out))), flush=True)
        except Exception as error:
            dump(out / "checks.json", dict(status="FAIL", source_sha256=hashes, error=repr(error)))
            raise
        return
    dump(out / "checks.json", checked)
    rows = []
    deadline = time.monotonic() + cfg["budget_minutes"] * 60
    write_summary(cfg, rows, out, "RUNNING_CALIBRATION")
    print(json.dumps(dict(dispatched=True, protocol=cfg["protocol"], pid=os.getpid(),
                          out=str(out), maximum_minutes=cfg["budget_minutes"])), flush=True)

    def execute(dim, variant, seed):
        if time.monotonic() >= deadline:
            return False
        row = trial(cfg, out, dim, variant, seed, deadline)
        rows.append(row)
        write_summary(cfg, rows, out, "RUNNING")
        return row["status"] == "completed"

    try:
        for seed in cfg["seeds"]:
            for dim in cfg["dimensions"]:
                if not execute(dim, "attention", seed):
                    write_summary(cfg, rows, out, "STOPPED_BUDGET_OR_NUMERICS")
                    return
        qualified = [dim for dim in cfg["dimensions"]
                     if decisions(cfg, rows)[str(dim)]["calibration"] == "PASS"]
        # Interleave matched raw/normalized pairs and dimensions in a frozen order.
        for seed in cfg["seeds"]:
            for medium in ("constant", "learned"):
                for dim in qualified:
                    for mode in ("raw", "normalized"):
                        if not execute(dim, medium + "_" + mode, seed):
                            write_summary(cfg, rows, out, "STOPPED_BUDGET_OR_NUMERICS")
                            return
        write_summary(cfg, rows, out, "COMPLETED_FROZEN_SCHEDULE")
    except Exception as error:
        dump(out / "failure.json", dict(error=repr(error), completed_trials=len(rows)))
        write_summary(cfg, rows, out, "FAILED_EXCEPTION")
        raise


if __name__ == "__main__":
    main()
