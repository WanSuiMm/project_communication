"""Verify and publish saved workspace/revision evidence; no model rollout."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "new/workspace_revision"))
from revision_cells import make_cell
from tasks import bank

ARMS = ("ws_additive", "ws_revision")
HORIZONS = (16, 32, 64, 96, 128, 192, 256)
RECOVERY = (0, 8, 16, 32, 64, 128)
BASE = "b992e2eac68d1bcfc81b70044bb89100bc664c2d"


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((a.shape, a.dtype)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def close(a, b, tolerance=1e-6):
    assert abs(a - b) <= tolerance, (a, b)


def all_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            all_finite(item)
    elif isinstance(value, list):
        for item in value:
            all_finite(item)


def checked_score(row):
    assert len(row["per_map_ba"]) == 16
    close(row["balanced_accuracy"], float(np.mean(row["per_map_ba"])))


def extract(arm, seed, size, ev):
    c, i = ev["curve"], ev["interventions_from_T64"]
    return {
        "arm": arm, "seed": seed, "size": size,
        "reach_ba64": c["64"]["balanced_accuracy"], "reach_paired64": c["64"]["paired"],
        "hold_min_ba": min(c[str(t)]["balanced_accuracy"] for t in (128, 192, 256)),
        "ba256": c["256"]["balanced_accuracy"], "bce64": c["64"]["bce"], "bce256": c["256"]["bce"],
        "warm_switch_changed64": i["64"]["switch"]["changed_accuracy"],
        "cold_switch_changed64": i["64"]["switch_cold"]["changed_accuracy"],
        "warm_switch_changed128": i["128"]["switch"]["changed_accuracy"],
        "cold_switch_changed128": i["128"]["switch_cold"]["changed_accuracy"],
        "warm_switch_unchanged64": i["64"]["switch"]["unchanged_accuracy"],
        "clean_continue64_ba": i["64"]["clean"]["balanced_accuracy"],
        "damage_immediate_ba": i["0"]["z_damage"]["balanced_accuracy"],
        "z_repair64_ba": i["64"]["z_damage"]["balanced_accuracy"],
        "wz_repair64_ba": i["64"]["wz_damage"]["balanced_accuracy"],
        "z_repair64_conditional_ba": i["64"]["z_damage"]["conditional_clean95_ba"],
        "wz_repair64_conditional_ba": i["64"]["wz_damage"]["conditional_clean95_ba"],
        "eligible_clean95_maps": ev["repair"]["eligible_clean95"],
        "w_rms64": c["64"]["w_rms"], "w_rms256": c["256"]["w_rms"],
        "z_rms64": c["64"]["z_rms"], "z_rms256": c["256"]["z_rms"],
        "dw_rms256": c["256"]["dw_rms"], "dz_rms256": c["256"]["dz_rms"],
    }


def plot(records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8), constrained_layout=True)
    colors = {"ws_additive": "#D55E00", "ws_revision": "#0072B2"}
    names = {"ws_additive": "Additive", "ws_revision": "Revision"}
    for seed in (0, 1):
        for col, size in enumerate((32, 64)):
            ax = axes[seed, col]
            for arm in ARMS:
                ev = records[(arm, seed)]["evaluation"][str(size)]
                ax.plot(HORIZONS, [100*ev["curve"][str(t)]["balanced_accuracy"] for t in HORIZONS],
                        "o-", color=colors[arm], label=names[arm], markersize=4)
            ax.axvline(64, color="gray", linewidth=.7, linestyle=":")
            ax.set(title=f"Seed {seed}: BA, size {size}", xlabel="Macro-steps", ylabel="Balanced accuracy (%)")
            ax.set_ylim(40, 103); ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax = axes[seed, 2]
        for arm in ARMS:
            ev = records[(arm, seed)]["evaluation"]["32"]["interventions_from_T64"]
            for branch, style in (("switch", "-"), ("switch_cold", "--")):
                label = f"{names[arm]}: " + ("warm" if branch == "switch" else "cold")
                ax.plot(RECOVERY, [100*ev[str(k)][branch]["changed_accuracy"] for k in RECOVERY],
                        style, color=colors[arm], label=label)
        ax.set(title=f"Seed {seed}: source switch, size 32",
               xlabel="Steps after switch / cold initialization", ylabel="Changed-component accuracy (%)")
        ax.set_ylim(-3, 103); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle("Workspace revision: reach/hold can improve while reopening fails\n"
                 "Two paired model seeds; each line averages 16 shared maps; no confidence intervals", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    publication = ROOT / "REVISION_PUBLICATION_MANIFEST.json"
    assert not publication.exists(), "Do not overwrite published evidence"
    assert not out.exists() or list(out.iterdir()) == [out / "INTERPRETATION.md"], "Only a prepared interpretation may precede export"
    status, manifest, aggregate = [read(run / f) for f in ("status.json", "manifest.json", "aggregate.json")]
    assert status["status"] == "COMPLETE" and status["completed_arms"] == 4
    assert manifest["seeds"] == [0, 1] and manifest["updates_per_arm"] == 600 and not manifest["preflight"]
    for name, expected in manifest["source_sha256"].items():
        assert sha(ROOT / name) == sha(run / "source" / name) == expected, name
    assert not (run / "watchdog_timeout.json").exists()
    torch.set_num_threads(2)
    check = subprocess.run([sys.executable, str(ROOT / "new/workspace_revision/check.py")],
                           capture_output=True, check=True)
    cpu_check = json.loads(check.stdout.decode("utf-8"))
    assert cpu_check["status"] == "PASS"
    test_hashes = {str(n): tensor_hash(bank(n, 16, 30000+n)) for n in (32, 64)}
    assert test_hashes == manifest["evaluation_data_sha256"]
    records, checkpoints, rows, pairs = {}, {}, [], []
    checked_points = 0
    for seed in (0, 1):
        train_hash = tensor_hash(bank(32, 512, 10000+seed))
        scheduled = read(run / f"schedule_seed{seed}.json")
        assert len(scheduled) == 600
        assert {m: sum(r["mode"] == m for r in scheduled) for m in ("hold", "switch", "repair")} == {
            "hold": 200, "switch": 200, "repair": 200}
        for j, s in enumerate(scheduled):
            assert s["mode"] == ("hold", "switch", "repair")[j % 3]
            assert s["T"] in (32, 48, 64) and s["warm"] in (0, 32, 64) and s["K"] in (16, 32)
            assert len(s["indices"]) == len(s["positions"]) == 8
            assert all(0 <= v < 512 for v in s["indices"])
            assert all(0 <= v <= 24 for pos in s["positions"] for v in pos)
        for arm in ARMS:
            key = f"{arm}_seed{seed}"
            record = records[(arm, seed)] = read(run / f"{key}.json")
            all_finite(record)
            assert record["status"] == "COMPLETE" and record["completed_updates"] == 600
            assert set(record["evaluation"]) == {"32", "64"}
            assert record["model"]["parameter_count"] == 5033
            assert record["model"]["communication_phases_per_macro_step"] == 2
            assert record["train_data_sha256"] == train_hash
            assert record["schedule_sha256"] == sha(run / f"schedule_seed{seed}.json")
            torch.manual_seed(seed)
            initial = make_cell(arm)
            assert tensor_hash(initial.state_dict()) == record["initial_parameter_sha256"]
            checkpoint = torch.load(run / f"{key}.pt", map_location="cpu", weights_only=True)
            assert checkpoint["completed_updates"] == 600
            assert tensor_hash(checkpoint["state_dict"]) == record["final_parameter_sha256"]
            checkpoints[key] = sha(run / f"{key}.pt")
            for size in (32, 64):
                ev = record["evaluation"][str(size)]
                assert ev["status"] == "EVALUATED"
                assert set(ev["curve"]) == set(map(str, HORIZONS))
                assert set(ev["interventions_from_T64"]) == set(map(str, RECOVERY))
                assert ev["repair"]["total_maps"] == 16
                eligibility = [i for i, v in enumerate(ev["curve"]["64"]["per_map_ba"]) if v >= .95]
                assert ev["repair"]["eligible_map_indices"] == eligibility
                assert ev["repair"]["eligible_clean95"] == len(eligibility)
                for c in ev["curve"].values():
                    checked_score(c); checked_score(c["flip"]); checked_points += 2
                    close(c["paired"], float(np.mean(c["paired_per_map"])))
                    for name in ("w_rms", "z_rms", "dw_rms", "dz_rms", "output_change_rms"):
                        # Stored means use GPU float32 reduction; NumPy uses float64.
                        assert math.isclose(c[name], float(np.mean(c[name+"_per_map"])), rel_tol=1e-6, abs_tol=1e-6)
                close(ev["hold_min_ba"], min(ev["curve"][str(t)]["balanced_accuracy"] for t in (128, 192, 256)))
                for intervention in ev["interventions_from_T64"].values():
                    assert set(intervention) == {"clean", "cold", "switch", "switch_cold", "z_damage", "wz_damage"}
                    for name, item in intervention.items():
                        checked_score(item); checked_points += 1
                        if name.startswith("switch"):
                            for label in ("changed", "unchanged"):
                                close(item[label+"_accuracy"], float(np.mean(item[label+"_per_map"])))
                        else:
                            if not eligibility:
                                assert item["conditional_clean95_ba"] is None
                            else:
                                close(item["conditional_clean95_ba"], float(np.mean([item["per_map_ba"][i] for i in eligibility])))
                rows.append(extract(arm, seed, size, ev))
        a, r = [next(row for row in rows if row["arm"] == arm and row["seed"] == seed and row["size"] == 32) for arm in ARMS]
        effect = 100*(r["hold_min_ba"]-a["hold_min_ba"])
        checks = {
            "positive_hold_effect": effect > 0,
            "revision_reach_ba85": r["reach_ba64"] >= .85,
            "revision_reach_paired50": r["reach_paired64"] >= .50,
            "reach_noninferior3pp": r["reach_ba64"] >= a["reach_ba64"]-.03,
            "hold_within3pp_of_reach": r["hold_min_ba"] >= r["reach_ba64"]-.03,
            "switch_changed80": r["warm_switch_changed64"] >= .80,
            "switch_unchanged85": r["warm_switch_unchanged64"] >= .85,
            "switch_changed_noninferior3pp": r["warm_switch_changed64"] >= a["warm_switch_changed64"]-.03,
            "switch_unchanged_noninferior3pp": r["warm_switch_unchanged64"] >= a["warm_switch_unchanged64"]-.03,
            "repair_ba85": r["z_repair64_ba"] >= .85,
            "repair_noninferior3pp": r["z_repair64_ba"] >= a["z_repair64_ba"]-.03,
        }
        raw_pair = next(p for p in aggregate["pairs"] if p["seed"] == seed)
        assert raw_pair["checks"] == checks
        close(raw_pair["hold_effect_pp"], effect)
        close(raw_pair["reach_effect_pp"], 100*(r["reach_ba64"]-a["reach_ba64"]))
        for field in ("initial_parameter_sha256", "train_data_sha256", "schedule_sha256"):
            assert records[(ARMS[0], seed)][field] == records[(ARMS[1], seed)][field]
            assert raw_pair["paired_identity"][field] is True
        pairs.append({"seed": seed, "hold_effect_pp": effect,
                      "failed_predicates": [k for k, v in checks.items() if not v]})
    mean_effect = float(np.mean([p["hold_effect_pp"] for p in pairs]))
    close(mean_effect, aggregate["mean_hold_effect_pp"])
    decision = "REVISION_JOINT_SCREEN_PASS" if mean_effect >= 5 and all(not p["failed_predicates"] for p in pairs) else "NO_JOINT_SCREEN_PASS"
    assert aggregate["complete"] is True and aggregate["decision"] == decision
    preflight_dir = ROOT / "runs/workspace_revision_20261002_dispatch_check"
    preflight = read(preflight_dir / "status.json")
    assert preflight["status"] == "PREFLIGHT_PASSED"
    assert read(preflight_dir / "manifest.json")["source_sha256"] == manifest["source_sha256"]
    out.mkdir(parents=True, exist_ok=True); (out / "raw").mkdir(); (out / "schedules").mkdir()
    for arm, seed in records:
        shutil.copyfile(run / f"{arm}_seed{seed}.json", out / "raw" / f"{arm}_seed{seed}.json")
    for seed in (0, 1):
        shutil.copyfile(run / f"schedule_seed{seed}.json", out / "schedules" / f"schedule_seed{seed}.json")
    for name in ("aggregate.json", "RESULTS.md"):
        shutil.copyfile(run / name, out / name)
    write(out / "completion.json", {k: v for k, v in status.items() if k != "pid"})
    clean_manifest = {k: v for k, v in manifest.items() if k != "pid"}
    clean_manifest["config"] = {k: v for k, v in manifest["config"].items() if k != "out"}
    write(out / "manifest.json", clean_manifest)
    write(out / "analysis.json", {"decision": decision, "primary_mean_hold_effect_pp": mean_effect,
                                "pairs": pairs, "rows": rows, "independent_unit": "model seed; n=2",
                                "evaluation_maps_per_size": 16})
    plot(records, out / "overview.png")
    write(out / "validation.json", {
        "cpu_model_check": cpu_check, "executed_source_snapshot_matches": True,
        "initial_and_final_parameter_hashes_verified": True,
        "training_and_evaluation_banks_recreated_on_cpu": True,
        "source_snapshot_files": len(manifest["source_sha256"]),
        "schedule_rows_verified": 1200, "mode_counts_per_seed": {"hold": 200, "switch": 200, "repair": 200},
        "ba_aggregates_recomputed_from_per_map": checked_points,
        "paired_rms_switch_and_conditional_metrics_recomputed": True,
        "rms_reduction_tolerance": {"relative": 1e-6, "absolute": 1e-6, "reason": "GPU float32 versus CPU float64 mean"},
        "primary_predicates_independently_recomputed": 22,
        "mean_hold_effect_pp": mean_effect, "decision": decision,
        "raw_and_schedules_byte_identical": True,
        "new_training_or_gpu_rollout_for_publication": False,
        "preflight": {k: v for k, v in preflight.items() if k != "pid"}})
    publication_doc = {
        "review_base": BASE, "protocol": manifest["protocol"],
        "execution_status": "COMPLETE", "scientific_status": decision,
        "model_seeds": [0, 1], "updates_per_arm": 600,
        "source_sha256": manifest["source_sha256"],
        "publication_tool_sha256": {"tools/export_revision_evidence.py": sha(__file__)},
        "checkpoint_sha256": checkpoints,
        "original_private_manifest_sha256": sha(run / "manifest.json"),
        "original_private_completion_sha256": sha(run / "status.json"),
        "published_evidence_sha256": {
            f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(out.rglob("*")) if f.is_file()},
        "excluded": ["checkpoints", "host and PID", "local launch receipts", "transient logs", "smoke outputs"],
        "notes": ["Original raw results, schedules and source snapshots remain unchanged.",
                  "Only manifest/completion publication copies omit local metadata.",
                  "No post-hoc rescue run or hyperparameter sweep.",
                  "Two seeds; size64 secondary; previous training recipes are not matched controls."]}
    write(publication, publication_doc)
    print(json.dumps({"status": "EXPORTED", "decision": decision, "mean_hold_effect_pp": mean_effect,
                      "ba_aggregates_verified": checked_points, "files": len(publication_doc["published_evidence_sha256"])}, indent=2))


if __name__ == "__main__":
    main()
