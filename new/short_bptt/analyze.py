"""Verify and summarize a completed matched-BPTT screen on CPU only."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from training import ROOT, make_cell
from run_revision import sha, tensor_hash, write
from tasks import bank


def read(path):return json.loads(path.read_text(encoding="utf-8"))
def close(a,b):assert math.isclose(a,b,abs_tol=1e-6,rel_tol=1e-6),(a,b)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True);parser.add_argument("--out",required=True)
    args=parser.parse_args();run=Path(args.run).resolve();out=Path(args.out).resolve()
    assert not out.exists();torch.set_num_threads(2)
    manifest,status,agg=[read(run/name) for name in ("manifest.json","status.json","aggregate.json")]
    assert status["status"]=="COMPLETE" and status["completed_arms"]==8 and agg["complete"]
    assert manifest["updates_per_arm"]==300 and not manifest["preflight"]
    assert not (run/"watchdog_timeout.json").exists()
    for name,h in manifest["source_sha256"].items():assert sha(ROOT/name)==sha(run/"source"/name)==h,name
    data={n:bank(n,16,30000+n) for n in (32,64)}
    for n,d in data.items():assert tensor_hash(d)==manifest["evaluation_data_sha256"][str(n)]
    trainhash={s:tensor_hash(bank(32,512,10000+s)) for s in (0,1)}
    for s in (0,1):
        scheduled=read(run/f"schedule_seed{s}.json")
        assert scheduled==np.random.default_rng(20000+s).integers(0,512,(300,8)).tolist()
    raw={};rows=[];system=[];ba_count=paired_count=0
    for path in sorted(run.glob("ws_*.json")):
        r=read(path);arch,k,s=r["architecture"],r["gradient_horizon"],r["seed"]
        assert r["status"]=="COMPLETE" and r["completed_updates"]==300
        assert r["train_data_sha256"]==trainhash[s] and r["schedule_sha256"]==sha(run/f"schedule_seed{s}.json")
        torch.manual_seed(s);model=make_cell(arch)
        assert tensor_hash(model.state_dict())==r["initial_parameter_sha256"]
        cp=torch.load(path.with_suffix(".pt"),map_location="cpu",weights_only=True)
        assert cp["completed_updates"]==300 and tensor_hash(cp["state_dict"])==r["final_parameter_sha256"]
        assert len(r["update_seconds"])==300
        for log in r["training_curve"]:
            assert log["forward_steps"]==64 and log["loss_count"]==8
            assert log["backward_calls"]==(8 if k==8 else 1)
            assert log["interior_detach_boundaries"]==(7 if k==8 else 0)
        raw[(arch,k,s)]=r
        system.append({"architecture":arch,"K":k,"seed":s,"training_seconds":r["training_seconds"],
            "peak_allocated_MiB":r["training_peak_allocated_bytes"]/2**20,
            "peak_increment_MiB":r["training_peak_increment_bytes"]/2**20,
            "clip_fraction":r["gradient_clip_fraction"],"last_logged_loss":r["training_curve"][-1]["mean_trajectory_loss"]})
        for n,curves in r["evaluation"].items():
            assert set(curves)=={"64","128","256"}
            for t,v in curves.items():
                for branch in ("original","flipped"):
                    assert len(v[branch]["per_map_ba"])==16
                    close(v[branch]["balanced_accuracy"],float(np.mean(v[branch]["per_map_ba"])));ba_count+=1
                for m in (v["paired"],v["far_paired"],*v["distance_bins"].values()):
                    valid=[x for x in m["per_map"] if x is not None]
                    assert len(valid)==m["eligible_maps"]
                    if valid:close(m["mean"],float(np.mean(valid)))
                    else:assert m["mean"] is None
                    count=sum(m["per_map_pixels"]);assert count==m["pooled_pixels"]
                    if count:
                        weighted=sum((x if x is not None else 0)*c for x,c in zip(m["per_map"],m["per_map_pixels"]))/count
                        close(m["pooled_accuracy"],weighted)
                    paired_count+=1
                assert v["outside_forward_lightcone_max_logit_difference"]<=1e-6
                rows.append({"architecture":arch,"K":k,"seed":s,"size":int(n),"T":int(t),
                    "BA":v["original"]["balanced_accuracy"],"flip_BA":v["flipped"]["balanced_accuracy"],
                    "all_paired":v["paired"]["mean"],"far_paired":v["far_paired"]["mean"],
                    "far_pooled":v["far_paired"]["pooled_accuracy"]})
    assert len(raw)==8 and len(rows)==48
    pairs=[]
    for pair in agg["pairs"]:
        arch,s=pair["architecture"],pair["seed"]
        f,q=raw[(arch,64,s)],raw[(arch,8,s)]
        assert all(f[key]==q[key] for key in ("initial_parameter_sha256","train_data_sha256","schedule_sha256"))
        a,b=f["evaluation"]["32"]["64"],q["evaluation"]["32"]["64"]
        fb,sb=a["original"]["balanced_accuracy"],b["original"]["balanced_accuracy"]
        fp,sp=a["far_paired"]["mean"],b["far_paired"]["mean"]
        qualified=fb>=.85 and fp>=.8
        reach=sp>=.8 and sp>=fp-.05 and sb>=.85 and sb>=fb-.03
        hold=all(v["original"]["balanced_accuracy"]>=sb-.03 and v["far_paired"]["mean"]>=sp-.05
                 for t,v in q["evaluation"]["32"].items() if t in ("128","256"))
        assert pair["full_qualified"]==qualified and pair["short_reach_predicates_pass"]==reach and pair["short_hold_predicates_pass"]==hold
        close(pair["far_effect_pp"],100*(sp-fp));assert pair["decision"]=="BASELINE_UNQUALIFIED"
        pairs.append(pair)
    full_peak=np.mean([r["peak_allocated_MiB"] for r in system if r["K"]==64])
    short_peak=np.mean([r["peak_allocated_MiB"] for r in system if r["K"]==8])
    validation={"source_snapshot_files_verified":len(manifest["source_sha256"]),"checkpoint_hashes_verified":8,
        "data_and_schedule_hashes_verified":True,"BA_aggregates_recomputed":ba_count,
        "paired_aggregates_recomputed":paired_count,"gate_pairs_recomputed":4,
        "new_training_or_inference":False,"training_trace_matches_K":True}
    out.mkdir(parents=True)
    write(out/"analysis.json",{"execution":"COMPLETE","elapsed_seconds":status["elapsed_seconds"],
        "scientific_status":"BASELINE_UNQUALIFIED_ALL_FOUR_PAIRS_WITH_ONE_SHORT_WINDOW_POSITIVE_CASE",
        "independent_unit":"model seed, n=2 per architecture", "pairs":pairs,"rows":rows,"systems":system,
        "memory":{"full_MiB":float(full_peak),"short_MiB":float(short_peak),"reduction_fraction":float(1-short_peak/full_peak)},
        "validation":validation})
    with (out/"curves.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=["# Matched short-BPTT screen: completed result review", "",
        f"All eight arms completed300 updates in {status['elapsed_seconds']:.3f} seconds (12.18 minutes).",
        "**All four full-BPTT controls failed the frozen far-paired qualification.**",
        "The formal comparison therefore remains BASELINE_UNQUALIFIED for both",
        "architectures. This does not erase a positive short-window case, and it",
        "does not establish either general truncation failure or superiority.", "",
        "## Primary size32/T64 endpoint", "",
        "Percentages. Far means graph distance>16; the statistic averages the",
        "16 per-map paired fractions. BOTH fresh source alternatives must be correct.", "",
        "| Architecture | Seed | K64 BA | K8 BA | K64 far paired | K8 far paired | Effect pp | K8 reach / hold predicates |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for p in pairs:
        lines.append(f"| {p['architecture']} | {p['seed']} | {100*p['full_ba64']:.2f} | {100*p['short_ba64']:.2f} | {100*p['full_far_paired64']:.2f} | {100*p['short_far_paired64']:.2f} | {p['far_effect_pp']:+.2f} | {p['short_reach_predicates_pass']} / {p['short_hold_predicates_pass']} |")
    lines += ["", "Full controls require BA>=85% and far paired>=80%; none met the latter.",
        "The additive K8 seed1 satisfies its reach and hold predicates, but the",
        "predeclared architecture screen also requires qualified full controls and",
        "both seeds to pass. Additive seed0 and both revision short arms fail.", "",
        "## Positive case and its limits", "",
        "Additive K8 seed1, size32: BA98.03% and far paired80.87% atT64.",
        "Far paired remains79.20% atT128 and78.24% atT256; BA256 is96.56%.",
        "This is a descriptive example of source-dependent computation beyond the",
        "K8 single-window dependency radius, under this trained shared rule.", "",
        "However,80.87% is a per-map average. The corresponding pooled-pixel fraction",
        "is59.53%; maps have between2 and345 far pixels. The frozen averaging stays",
        "unchanged, and both views must remain visible. Long-distance performance",
        "is much weaker than the aggregate headline:", "",
        "| Size | T | Distance bin | Paired accuracy % | Eligible maps | Pixels |",
        "|---|---:|---|---:|---:|---:|"]
    best=raw[("ws_additive",8,1)]
    for n in ("32","64"):
        for t in ("64","256"):
            for label in ("16_32","32_64","64_100000"):
                b=best["evaluation"][n][t]["distance_bins"][label]
                value="null" if b["mean"] is None else f"{100*b['mean']:.2f}"
                lines.append(f"| {n} | {t} | {label} | {value} | {b['eligible_maps']} | {b['pooled_pixels']} |")
    lines += ["", "The[16,32) bin includes distance16, whereas the primary far metric uses",
        "strictly>16. The size32 distance>=64 bin has only one eligible map; it",
        "cannot support a population conclusion. At size64, this same checkpoint's",
        "far paired is28.40% atT64 and41.51% atT256. Robust distance/scale extrapolation",
        "is not established. Two communication phases per step mean K8 reaches at",
        "most16 graph edges per gradient window; distance is not the window count.", "",
        "## Optimization and systems", "",
        "| Architecture | Seed | K | Training seconds | Peak allocated MiB | Gradient clipping % | Last logged loss |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in system:
        lines.append(f"| {r['architecture']} | {r['seed']} | {r['K']} | {r['training_seconds']:.2f} | {r['peak_allocated_MiB']:.2f} | {100*r['clip_fraction']:.2f} | {r['last_logged_loss']:.4f} |")
    lines += ["", f"Peak allocated CUDA memory falls from{full_peak:.2f} to{short_peak:.2f} MiB",
        f"({100*(1-short_peak/full_peak):.1f}% lower). These peaks include data/optimizer/gradient",
        "storage, not just activations. Training durations are broadly similar;",
        "the screen does not establish a meaningful speedup or accuracy-matched",
        "systems advantage. Timings are sequential observations on one GPU.", "",
        "Additive fullK64 clips88.67%/90.67% of updates, versus6.67%/6.67% forK8.",
        "This is a useful optimization observation, not proof that clipping caused",
        "the full-control failure. Detaching changes the accumulated gradient and",
        "therefore how the identical clipping rule acts. ShortK8 revision BA falls",
        "from83.90/64.38% atT64 to59.64/46.18% atT256 across seeds0/1.", "",
        "## Validity and claim boundary", "",
        "- All sources match their executed snapshots. Initial and final parameter",
        "  hashes, data banks, schedules and all four matched identities verify.",
        f"- {ba_count} BA and {paired_count} paired aggregates were independently recomputed",
        "  from per-map values; all four frozen pair decisions agree with the runner.",
        "- Forward length64, eight equally weighted losses, one optimizer step per",
        "  trajectory and300 updates are shared. Only the graph cuts differ within",
        "  an architecture/seed pair. Logged backward calls/cut counts agree.",
        "- The300-update budget was selected from preflight timing before efficacy",
        "  training. This changed objective and budget cannot borrow qualification",
        "  from the earlier600-update reach/auxiliary experiment.",
        "- Training uses fresh64-step trajectories; inference extends to256. The",
        "  result does not test256-step training, state-pool handoffs, reopen or",
        "  arbitrary delayed credit assignment, and does not establish novelty.",
        "- No extra training, checkpoint replay, retuning or rescue run was performed",
        "  for this inspection. Two seeds are descriptive, not significance evidence.", "",
        "[All48 curve rows](curves.csv), [structured analysis](analysis.json),",
        "[figure](overview.png), [provenance](provenance.json)."]
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for i,n in enumerate((32,64)):
        for j,arch in enumerate(("ws_additive","ws_revision")):
            ax=axes[i,j]
            for k,color in ((64,"#D55E00"),(8,"#0072B2")):
                for s,style in ((0,"--"),(1,"-")):
                    yy=[100*raw[(arch,k,s)]["evaluation"][str(n)][str(t)]["far_paired"]["mean"] for t in (64,128,256)]
                    ax.plot((64,128,256),yy,style,marker="o",color=color,label=f"K{k}, seed{s}")
            ax.set(title=f"{arch.removeprefix('ws_')}, size{n}",xlabel="Forward steps",ylabel="Far paired accuracy (%)",ylim=(-3,103))
            if n==32:ax.axhline(80,color="gray",ls=":",lw=.8)
            ax.legend(fontsize=7);ax.grid(alpha=.2)
        ax=axes[i,2];labels=("0_8","8_16","16_32","32_64","64_100000")
        for t,color in ((64,"#0072B2"),(128,"#009E73"),(256,"#D55E00")):
            yy=[100*best["evaluation"][str(n)][str(t)]["distance_bins"][b]["mean"] for b in labels]
            ax.plot(range(5),yy,"o-",color=color,label=f"T{t}")
        ax.set(title=f"Additive K8 seed1, size{n}",xticks=range(5),xticklabels=["0-7","8-15","16-31","32-63","64+"],xlabel="Graph-distance bin",ylabel="Paired accuracy (%)",ylim=(-3,103))
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle("Matched short BPTT: all full controls unqualified; one short-window positive case\nPer-map averages; two model seeds; size32 64+ bin contains one eligible map",fontsize=12)
    fig.savefig(out/"overview.png",dpi=160);plt.close(fig)
    write(out/"provenance.json",{"source_run":run.relative_to(ROOT).as_posix(),"training":False,"inference":False,
        "input_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(run.glob("*.json")) if p.name!="launch_receipt.json"},
        "checkpoint_sha256":{p.name:sha(p) for p in sorted(run.glob("*.pt"))},
        "analysis_source_sha256":{Path(__file__).relative_to(ROOT).as_posix():sha(Path(__file__))},
        "output_sha256":{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({"status":"ANALYZED","validation":validation,"memory_reduction_percent":100*(1-short_peak/full_peak)},indent=2))


if __name__=="__main__":main()
