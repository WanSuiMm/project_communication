"""CPU-only summaries of the frozen source-switch audit; never run the model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit import sha, tensor_hash, bank, write, KPOINTS, BRANCHES


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def positive_fraction(accuracy, bits):
    # Each changed component has a constant binary target; recover the exact
    # predicted-positive fraction from accuracy without a new model rollout.
    return [float(a if bit==1 else 1-a) for a,bit in zip(accuracy,bits)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True); parser.add_argument("--out",required=True)
    args=parser.parse_args(); run=Path(args.run).resolve(); out=Path(args.out).resolve()
    assert not out.exists()
    status=read(run/"status.json"); manifest=read(run/"manifest.json")
    assert status["status"]=="COMPLETE" and status["sizes"]==[32,64]
    for name,expected in manifest["source_sha256"].items():
        assert sha(ROOT/name)==sha(run/"source"/name)==expected,name
    expected_data=read(ROOT/"evidence/workspace_revision_paired01/manifest.json")["evaluation_data_sha256"]
    raws={n:read(run/f"size{n}.json") for n in (32,64)}
    rows=[]; factors=[]; contrasts=[]; evidence={}
    for n,d in raws.items():
        data=bank(n,16,30000+n); assert tensor_hash(data)==expected_data[str(n)]
        bits=((data["y_flip"]*data["changed"]).sum((1,2,3))/data["changed"].sum((1,2,3))).tolist()
        assert all(b in (0,1) for b in bits)
        evidence[str(n)]={"new_target_bits_per_map":bits,"negative_label_fraction":float(np.mean(np.array(bits)==0))}
        for k in KPOINTS:
            a=d["anchors"][str(k)]
            for branch,v in a["branches"].items():
                o,q=v["output"],v["candidate"]
                pos=positive_fraction(o["changed"]["per_map"],bits)
                rows.append({"size":n,"K":k,"branch":branch,
                    "output_changed":o["changed"]["mean"],"candidate_changed":q["changed"]["mean"],
                    "output_unchanged":o["unchanged"]["mean"],"output_ba":o["balanced_accuracy"],
                    "candidate_new_margin":q["changed_margin"]["mean"],
                    "output_changed_per_map":o["changed"]["per_map"],
                    "predicted_positive_fraction":float(np.mean(pos)),"predicted_positive_fraction_per_map":pos})
            if k in (0,64,128):
                f=a["factorial"]
                for key,v in f.items():
                    factors.append({"size":n,"K":k,"query":key,"changed":v["changed"]["mean"],
                        "margin":v["changed_margin"]["mean"],"per_map":v["changed"]["per_map"]})
                def values(w,z,metric):
                    return np.array(f[f"W_{w}__Z_{z}__X_new"][metric]["per_map"])
                for metric in ("changed","changed_margin"):
                    ww=values("warm","warm",metric); nw=values("new_aged","warm",metric)
                    wn=values("warm","new_aged",metric); nn=values("new_aged","new_aged",metric)
                    for name,arr in (("replace_W_hold_Z",nw-ww),("replace_Z_hold_W",wn-ww),
                        ("joint_minus_additive_contrast",nn-nw-wn+ww)):
                        contrasts.append({"size":n,"K":k,"metric":metric,"contrast":name,
                            "mean":float(arr.mean()),"per_map":arr.tolist()})
        warm64=d["anchors"]["64"]["branches"]["warm"]["candidate"]["changed"]["mean"]
        warm128=d["anchors"]["128"]["branches"]["warm"]["candidate"]["changed"]["mean"]
        evidence[str(n)]["old_aligned_candidate_at_both_endpoints"]=warm64<=.05 and warm128<=.05
        evidence[str(n)]["intervention_recovery128"]={}
        base=d["anchors"]["128"]["branches"]["warm"]["output"]["changed"]["mean"]
        for row in [r for r in rows if r["size"]==n and r["K"]==128]:
            evidence[str(n)]["intervention_recovery128"][row["branch"]]=(
                row["output_changed"]>=.8 and row["output_unchanged"]>=.85 and row["output_changed"]-base>=.2)
    validation={"replay_comparisons":sum(len(v["validation"]["replay"]) for v in raws.values()),
        "replay_max_error":max(r["max_error"] for v in raws.values() for r in v["validation"]["replay"]),
        "candidate_step_max_error":max(v["validation"]["step_max_error"] for v in raws.values()),
        "affine_readout_max_error":max(v["validation"]["affine_readout_max_error"] for v in raws.values()),
        "source_snapshot_and_data_hashes_match":True,"training":False,"analysis_inference":False,
        "elapsed_seconds":status["elapsed_seconds"]}
    out.mkdir(parents=True)
    write(out/"analysis.json",{"status":"COMPLETE","independent_models":1,"model_seed":0,
        "interpretation":"OLD_ALIGNED_CANDIDATE_AND_W_Z_INTERACTION_WITHOUT_SINGLE_BLOCK_RECOVERY",
        "validation":validation,"evidence":evidence,"rows":rows,"factorial":factors,"contrasts":contrasts,
        "posthoc_note":"Predicted-label bias derived from saved per-map accuracy and fixed binary component labels; no additional inference."})
    def pct(v):return f"{100*v:.2f}"
    lines=["# Candidate/workspace switch audit: completed", "",
        "Zero training, revision seed0 only, 16 fixed maps per size. All 232 saved",
        "replay comparisons have zero error. Candidate reconstruction also has zero",
        "error; affine readout identity maximum error is below 7.2e-7. Inference",
        f"completed in {status['elapsed_seconds']:.2f} seconds.", "",
        "**The candidate retains the old answer, but the audit does not isolate W",
        "as the sole cause. Single-block resets or mature-state transplants fail to",
        "recover reliable switching; cold reset of both blocks succeeds.**", "",
        "## Current output versus candidate", "",
        "Percent correct on the changed component under the new target:", "",
        "| Size | K | Warm output | Warm candidate | Cold output | Cold candidate |",
        "|---|---:|---:|---:|---:|---:|"]
    for n in (32,64):
        for k in KPOINTS:
            r=raws[n]["anchors"][str(k)]["branches"]
            vals=[r[b][m]["changed"]["mean"] for b,m in (("warm","output"),("warm","candidate"),("cold","output"),("cold","candidate"))]
            lines.append(f"| {n} | {k} | "+" | ".join(map(pct,vals))+" |")
    lines += ["", "Q_K is computed after the W update and drives Z_(K+1). O(Q_K) is",
        "the existing affine readout applied to the candidate. Persistent wrong",
        "candidate readout is directly observed; slow exponential averaging alone",
        "does not explain warm failure. This says nothing about a unique origin",
        "inside the coupled W/Z recurrence.", "", "## Full-rollout interventions", "",
        "All branches use the new input. Accuracy columns are percentages at K128.", "",
        "| Size | Intervention | Changed | Unchanged | BA | Predicted positive on changed | Recovery criterion |",
        "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        if r["K"]!=128: continue
        passed=evidence[str(r["size"])]["intervention_recovery128"][r["branch"]]
        lines.append(f"| {r['size']} | {r['branch']} | "+" | ".join(pct(r[key]) for key in
            ("output_changed","output_unchanged","output_ba","predicted_positive_fraction"))+f" | {passed} |")
    lines += ["", "`reset_W` uses encoder(new input); `reset_Z` uses zero. `cold` resets",
        "both. `transplant_W/Z` uses one block from a separate 64-step new-input",
        "rollout, retaining the other old block. Donors contain extra computation.", "",
        "**At size32 the 37.5% result is not partial retrieval of the new source:**",
        "all four single-block interventions predict negative on every pixel of",
        "every changed component at K128. Six of the 16 new component labels are",
        "negative. This label-bias observation is a post-hoc derivation from saved",
        "per-map accuracy and the fixed labels, requiring no extra model rollout.", "",
        "In the Z-donor transplant, size32 output starts 100% correct on the changed",
        "component and falls to 37.5%; injecting a correct task-state block does not",
        "sustain the new answer in the old workspace. Conversely a new W donor with",
        "old Z also fails. Resetting Z alone also damages unchanged accuracy.", "",
        "## Candidate-only W/Z factorial", "",
        "All queries below use the same new X at K64. W donors are post-F, Z donors",
        "are pre-update. New donors come from an intact fresh-new rollout at T128.", "",
        "| Size | W donor | Z donor | Candidate changed accuracy (%) | New-target signed margin |",
        "|---|---|---|---:|---:|"]
    for n in (32,64):
        for w in ("warm","new_aged"):
            for z in ("warm","new_aged"):
                q=raws[n]["anchors"]["64"]["factorial"][f"W_{w}__Z_{z}__X_new"]
                lines.append(f"| {n} | {w} | {z} | {pct(q['changed']['mean'])} | {q['changed_margin']['mean']:.4f} |")
    lines += ["", "Both W and Z substitutions change the fixed candidate function; their",
        "joint effect includes an interaction. Mixing blocks from incompatible",
        "trajectories can be off-distribution. Neither these contrasts nor reset",
        "failures establish a unique natural causal mechanism or prove that both",
        "blocks must always be reset. The all-new donor combination is an intact",
        "computed solution, not an equal-cost practical remedy.", "",
        "Direct old/new X swaps inside Qnet barely change the shown aggregate",
        "responses. This local query excludes X's paths through F, initialization",
        "and repeated updates, and changes a single source location. It does not",
        "show that the network generally ignores external evidence.", "",
        "## What the feedback gets right and what remains unproven", "",
        "- Confirmed: cold source solving versus warm source revision are distinct;",
        "  the candidate itself remains old-aligned at K64/K128 in both sizes.",
        "- Refined: the W-only stale-workspace explanation is insufficiently",
        "  isolated. Z feedback and W/Z compatibility matter in the tested interventions.",
        "- Not established: useful internal computation merely from continued W",
        "  motion, a unique stale-state mechanism, general persistent-memory success,",
        "  or conditional invalidation as the necessary next architectural primitive.",
        "- The earlier repair contrast retains more state under Z-only damage; it",
        "  did not by itself prove that repair and reopen failure share one cause.",
        "- At the original zero-output-layer initialization the state Jacobian is",
        "  block diagonal (I,0.5I). That is a code-level fact, not a diagnosis of",
        "  seed1 failure or a measurement of parameter credit and later dynamics.",
        "- No short-versus-long BPTT experiment, new training, seed1 audit or",
        "  generalization across independently trained models was performed.", "",
        "The prior paired screen's NO_JOINT_SCREEN_PASS and -9.02 pp mean hold effect",
        "remain unchanged. This is a diagnostic result for one existing checkpoint.", "",
        "See [compact analysis](analysis.json), [figure](overview.png),",
        "[provenance](provenance.json) and the frozen protocol in the source tree."]
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    colors={"warm":"#D55E00","reset_W":"#009E73","reset_Z":"#CC79A7","cold":"#0072B2","transplant_W":"#E69F00","transplant_Z":"#555555"}
    for i,n in enumerate((32,64)):
        ax=axes[i,0]
        for branch in ("warm","cold"):
            for typ,style in (("output","-"),("candidate","--")):
                ys=[100*raws[n]["anchors"][str(k)]["branches"][branch][typ]["changed"]["mean"] for k in KPOINTS]
                ax.plot(KPOINTS,ys,style,color=colors[branch],label=f"{branch} {typ}")
        ax.set(title=f"Size {n}: current / next candidate",xlabel="K after source switch",ylabel="New-target accuracy (%)",ylim=(-3,103));ax.legend(fontsize=8);ax.grid(alpha=.2)
        ax=axes[i,1]
        for branch in BRANCHES:
            ys=[100*raws[n]["anchors"][str(k)]["branches"][branch]["output"]["changed"]["mean"] for k in KPOINTS]
            ax.plot(KPOINTS,ys,color=colors[branch],label=branch)
        ax.set(title=f"Size {n}: interventions",xlabel="K after intervention",ylabel="Changed accuracy (%)",ylim=(-3,103));ax.legend(fontsize=7);ax.grid(alpha=.2)
        ax=axes[i,2]
        matrix=np.array([[100*raws[n]["anchors"]["64"]["factorial"][f"W_{w}__Z_{z}__X_new"]["changed"]["mean"] for z in ("warm","new_aged")] for w in ("warm","new_aged")])
        ax.imshow(matrix,vmin=0,vmax=100,cmap="Blues")
        for (r,c),value in np.ndenumerate(matrix):ax.text(c,r,f"{value:.2f}%",ha="center",va="center",color="white" if value>65 else "black",fontsize=12)
        ax.set(title=f"Size {n}: candidate donors at K64",xticks=[0,1],xticklabels=["Warm Z","New-aged Z"],yticks=[0,1],yticklabels=["Warm W+","New-aged W+"])
    fig.suptitle("One frozen revision checkpoint: candidate persistence, coupled state interventions",fontsize=13)
    fig.savefig(out/"overview.png",dpi=160);plt.close(fig)
    write(out/"provenance.json",{"source_run":run.relative_to(ROOT).as_posix(),"training":False,
        "input_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in (run/"manifest.json",run/"status.json",run/"size32.json",run/"size64.json")},
        "checkpoint_sha256":manifest["checkpoint_sha256"],"audit_source_sha256":manifest["source_sha256"],
        "analysis_source_sha256":{Path(__file__).relative_to(ROOT).as_posix():sha(__file__)},
        "output_sha256":{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({"status":"COMPLETE","validation":validation,"evidence":evidence},indent=2))


if __name__=="__main__":main()
