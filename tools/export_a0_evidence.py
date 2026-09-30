"""Export one immutable A0 snapshot; never edit the running/local evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("train_shape_d16", "large_shape_d16", "d32", "d64", "d128")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def summarize(aggregate, cfg, checks):
    rows = [r for r in aggregate["results"] if r["status"] == "completed"]
    calibrated = [int(dim) for dim, item in aggregate["decisions"].items()
                  if item["calibration"] == "PASS"]
    controls = len(cfg["dimensions"]) * len(cfg["seeds"])
    scheduled = controls + len(calibrated) * len(cfg["seeds"]) * 4
    groups = []
    for dim in cfg["dimensions"]:
        for arm in cfg["variants"]:
            arm_rows = [r for r in rows if r["dim"] == dim and r["variant"] == arm]
            if not arm_rows:
                continue
            entry = dict(dim=dim, arm=arm, seeds=[r["seed"] for r in arm_rows],
                         parameters=arm_rows[0]["parameters"], conditions={})
            for condition in CONDITIONS:
                values = [r["eval"][condition]["accuracy"] for r in arm_rows]
                entry["conditions"][condition] = dict(per_seed_accuracy=values,
                                                      mean_accuracy=statistics.mean(values))
            entry["mean_of_seed_median_latency_ms"] = statistics.mean(r["latency"]["median_ms"] for r in arm_rows)
            groups.append(entry)
    effects = []
    for dim in cfg["dimensions"]:
        for medium in ("constant", "learned"):
            paired = []
            for seed in cfg["seeds"]:
                candidates = {r["variant"]: r for r in rows if r["dim"] == dim and r["seed"] == seed}
                raw, norm = candidates.get(medium + "_raw"), candidates.get(medium + "_normalized")
                if raw is None or norm is None:
                    continue
                paired.append(dict(seed=seed, delta_percentage_points={
                    c: 100 * (norm["eval"][c]["accuracy"] - raw["eval"][c]["accuracy"])
                    for c in CONDITIONS}))
            if paired:
                effects.append(dict(dim=dim, medium=medium, paired=paired,
                                    mean_delta_percentage_points={c: statistics.mean(p["delta_percentage_points"][c] for p in paired)
                                                                  for c in CONDITIONS}))
    oracle = checks["oracle"]
    big = [r for r in oracle if r["tau"] == 4096]
    far_big = [r for r in big if r["distance"] == 128]
    far_small = [r for r in oracle if r["tau"] == 1 and r["distance"] == 128]
    diagnostic_rows = []
    for row in rows:
        if row["variant"] == "attention":
            continue
        ds = [d for d in row["diagnostics"] if d["condition"] == "d128"]
        diagnostic_rows.append(dict(dim=row["dim"], arm=row["variant"], seed=row["seed"],
                                    mean_across_axes={key: statistics.mean(d[key] for d in ds) for key in (
                                        "source_confidence_mean", "background_confidence_mean",
                                        "target_denominator_mean", "target_source_mass_fraction_mean",
                                        "target_message_rms", "target_below_epsilon_fraction")}))
    return dict(protocol=cfg["protocol"], status=aggregate["status"], decisions=aggregate["decisions"],
                completed_trials=len(rows), maximum_if_both_dimensions_calibrated=40,
                scheduled_trials_after_calibration=scheduled,
                skipped_trials_due_to_calibration=40 - scheduled,
                independent_unit="training_seed", seed_order=cfg["seeds"],
                metrics=groups, paired_normalization_effects=effects,
                oracle=dict(conditions=len(oracle), large_tau_min_raw_sign_accuracy=min(r["raw_sign_accuracy"] for r in big),
                            d128_large_tau_max_normalized_value_error=max(r["normalized_abs_error_max"] for r in far_big),
                            d128_small_tau_undecoded=sum(r["raw_ties_not_decoded"] for r in far_small),
                            d128_small_tau_total=sum(r["batch_size"] for r in far_small)),
                d128_diagnostics=diagnostic_rows)


def report(summary):
    lines = ["# A0 exported evidence", "", f"Run status: {summary['status']}", "",
             f"Completed trials: {summary['completed_trials']}/{summary['scheduled_trials_after_calibration']} in the conditional schedule. "
             f"The initial maximum was 40; {summary['skipped_trials_due_to_calibration']} were excluded by failed calibration.", "",
             "Seeds are 1729 / 2718 / 31415 / 57721. Accuracy values below are percentages; seed count is the replicate count.", "",
             "| Dim | Arm | Train d16, per seed | d128, per seed | Mean seed-median latency, ms |",
             "|---|---|---|---|---|"]
    for entry in summary["metrics"]:
        fit = " / ".join(f"{100*x:.2f}" for x in entry["conditions"]["train_shape_d16"]["per_seed_accuracy"])
        far = " / ".join(f"{100*x:.2f}" for x in entry["conditions"]["d128"]["per_seed_accuracy"])
        lines.append(f"| {entry['dim']} | {entry['arm']} | {fit} | {far} | {entry['mean_of_seed_median_latency_ms']:.2f} |")
    lines += ["", "## Frozen decisions", "", "```json", json.dumps(summary["decisions"], indent=2), "```", "",
              "## Paired normalization effect", "", "Normalized minus raw, percentage points. These are descriptive paired outcomes, not significance claims.", "",
              "| Dim | Medium | train d16 mean | large d16 mean | d32 mean | d64 mean | d128 mean | d128 per seed |",
              "|---|---|---|---|---|---|---|---|"]
    for entry in summary["paired_normalization_effects"]:
        delta = " | ".join(f"{entry['mean_delta_percentage_points'][c]:+.3f}" for c in CONDITIONS)
        per_seed = " / ".join(f"{r['delta_percentage_points']['d128']:+.3f}" for r in entry["paired"])
        lines.append(f"| {entry['dim']} | {entry['medium']} | {delta} | {per_seed} |")
    lines += ["", "## Interpretation boundaries", "",
              "Oracle signal recoverability and passed numerical checks do not establish learned-model qualification. Inspect training-scale fitting before interpreting distant accuracy.",
              "A dimension that failed attention calibration has no RT treatment estimate. Missing 3D arms are protocol skips, not failed RT runs.",
              "A0 raw and normalized arms both emit q=c*v and use matched initial parameters/medium scale and packed solver work. A0 raw is not the historical v1 raw model.",
              "Fixed axis splitting, narrow straight geometry and portable PyTorch kernels remain limitations. B/C and width sweeps were not run.",
              "Read summary.json first; aggregate.json has the complete frozen per-axis metrics and histories, and checks.json has the oracle and software records."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out.exists():
        raise SystemExit("Export directory already exists; use a fresh destination.")
    if (ROOT / "A0_PUBLICATION_MANIFEST.json").exists():
        raise SystemExit("An A0 publication manifest already exists; do not overwrite frozen evidence.")
    # aggregate.json is atomically replaced by the runner. Read it ONCE; do not
    # mix a mutable RESULTS.md or per-trial files into this captured snapshot.
    aggregate_bytes = (run / "aggregate.json").read_bytes()
    aggregate = json.loads(aggregate_bytes)
    config_bytes, checks_bytes = [(run / name).read_bytes() for name in ("config.json", "checks.json")]
    cfg, checks = json.loads(config_bytes), json.loads(checks_bytes)
    receipt = json.loads((run / "launch_receipt.json").read_bytes())
    if checks["status"] != "PASS" or checks["source_sha256"] != receipt["source_sha256"]:
        raise SystemExit("Run/check source identity mismatch.")
    for name, expected in receipt["source_sha256"].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise SystemExit(f"Current training source differs from run: {name}")
    if sha((ROOT / "A0_PROTOCOL.md").read_bytes()) != receipt["protocol_sha256"]:
        raise SystemExit("A0 protocol changed since launch.")
    summary = summarize(aggregate, cfg, checks)
    if aggregate["status"] == "COMPLETED_FROZEN_SCHEDULE" and summary["completed_trials"] != summary["scheduled_trials_after_calibration"]:
        raise SystemExit("Completed status disagrees with conditional schedule count.")
    out.mkdir(parents=True)
    blobs = {"aggregate.json": aggregate_bytes, "config.json": config_bytes,
             "checks.json": checks_bytes, "summary.json": encode(summary),
             "RESULTS.md": report(summary).encode("utf-8")}
    for name, data in blobs.items():
        (out / name).write_bytes(data)
    manifest = dict(canonical_run=run.name, protocol=cfg["protocol"],
                    captured_utc=datetime.now(timezone.utc).isoformat(),
                    publication_kind="completed_run" if aggregate["status"] == "COMPLETED_FROZEN_SCHEDULE" else "interim_or_stopped_snapshot",
                    run_status=aggregate["status"], decisions=aggregate["decisions"],
                    started_utc=receipt["started_utc"],
                    runtime={k: receipt[k] for k in ("python", "torch", "device")},
                    source_sha256=receipt["source_sha256"], protocol_sha256=receipt["protocol_sha256"],
                    published_evidence_sha256={str((out / name).relative_to(ROOT)).replace("\\", "/"): sha(data)
                                               for name, data in blobs.items()},
                    excluded=["checkpoints", "launch receipts", "machine identities", "private paths",
                              "transient logs", "duplicate source snapshots"])
    (ROOT / "A0_PUBLICATION_MANIFEST.json").write_bytes(encode(manifest))
    print(json.dumps({k: summary[k] for k in ("status", "completed_trials", "scheduled_trials_after_calibration", "decisions")}, indent=2))


if __name__ == "__main__":
    main()
