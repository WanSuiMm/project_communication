"""Bounded matched K8/K64 training; see frozen PROTOCOL.md."""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
import traceback
import numpy as np
import torch
from training import ROOT, backward_trajectory, make_cell
from run_revision import now, sha, write, tensor_hash, score
from tasks import bank, subset

ARCHS=("ws_additive","ws_revision")
DEADLINE=float("inf")


def budget():
    if time.monotonic()>DEADLINE:raise TimeoutError("25-minute whole-run cap")


def sources():
    base=json.loads((ROOT/"REVISION_PUBLICATION_MANIFEST.json").read_text())
    for name,h in base["source_sha256"].items():assert sha(ROOT/name)==h,name
    files=["new/short_bptt/PROTOCOL.md","new/short_bptt/training.py","new/short_bptt/check.py",
           "new/short_bptt/run.py","tools/launch_short_bptt.ps1"]
    return {**base["source_sha256"],**{f:sha(ROOT/f) for f in files}}


def paired(logits, flipped, data, selected):
    correct=(((logits>=0)==(data["y"]>=.5))&((flipped>=0)==(data["y_flip"]>=.5))).float()
    counts=selected.sum((1,2,3));eligible=counts>0
    values=(correct*selected).sum((1,2,3))/counts.clamp_min(1)
    return {"mean":float(values[eligible].mean()) if bool(eligible.any()) else None,
        "per_map":[v if e else None for v,e in zip(values.cpu().tolist(),eligible.cpu().tolist())],
        "eligible_maps":int(eligible.sum()),"per_map_pixels":counts.cpu().tolist(),
        "pooled_pixels":int(counts.sum()),
        "pooled_accuracy":float((correct*selected).sum()/counts.sum()) if bool(eligible.any()) else None}


@torch.no_grad()
def evaluate(model,data,preflight):
    a,b=model.initial(data["x"]),model.initial(data["x_flip"])
    horizons=(64,) if preflight else (64,128,256)
    curves={}
    for t in range(1,max(horizons)+1):
        if t%8==1:budget()
        a=model.step(a,data["x"]);b=model.step(b,data["x_flip"])
        if t in horizons:
            assert all(bool(torch.isfinite(v).all()) for v in (*a,*b)),"Nonfinite evaluation state"
            x,z=model.logits(a),model.logits(b)
            far=data["changed"]*(data["distance"]>16)
            row={"original":score(x,data["y"],data["mask"]),
                "flipped":score(z,data["y_flip"],data["mask"]),
                "paired":paired(x,z,data,data["changed"]),"far_paired":paired(x,z,data,far),"distance_bins":{}}
            for lo,hi in ((0,8),(8,16),(16,32),(32,64),(64,100000)):
                select=data["changed"]*((data["distance"]>=lo)&(data["distance"]<hi))
                row["distance_bins"][f"{lo}_{hi}"]=paired(x,z,data,select)
            outside=data["changed"]*(data["distance"]>2*t)
            row["outside_forward_lightcone_pixels"]=int(outside.sum())
            row["outside_forward_lightcone_max_logit_difference"]=float(((x-z).abs()*outside).max())
            assert row["outside_forward_lightcone_max_logit_difference"]<=1e-6
            curves[str(t)]=row
    return curves


def train_one(out,arch,k,seed,rows,train,tests,preflight):
    budget();torch.manual_seed(seed)
    model=make_cell(arch).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    name=f"{arch}_K{k}_seed{seed}"
    result={"architecture":arch,"gradient_horizon":k,"seed":seed,"status":"TRAINING",
        "initial_parameter_sha256":tensor_hash(model.state_dict()),"train_data_sha256":tensor_hash(train),
        "schedule_sha256":sha(out/f"schedule_seed{seed}.json"),"parameter_count":5033,
        "completed_updates":0,"training_curve":[],"update_seconds":[],"evaluation":{}}
    write(out/f"{name}.json",result)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    result["allocated_before_training_bytes"]=torch.cuda.memory_allocated()
    start=time.monotonic();clipped=0
    try:
        for iteration,row in enumerate(rows,1):
            budget();batch=subset(train,row)
            torch.cuda.synchronize();tick=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss,state,trace=backward_trajectory(model,batch,k,budget=budget)
            if not all(bool(torch.isfinite(v).all()) for v in state):raise FloatingPointError("Nonfinite training state")
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(norm)):raise FloatingPointError("Nonfinite gradient")
            clipped+=int(norm>1.)
            optimizer.step();torch.cuda.synchronize()
            result["update_seconds"].append(time.perf_counter()-tick)
            result["completed_updates"]=iteration
            if iteration==1 or iteration%100==0 or iteration==len(rows) or preflight:
                log={"update":iteration,"mean_trajectory_loss":float(loss),"gradient_norm_before_clip":float(norm),
                     "update_seconds":result["update_seconds"][-1],**trace}
                result["training_curve"].append(log)
                write(out/"status.json",{"status":"RUNNING","phase":"training","arm":name,
                      "completed_updates":iteration,"pid":os.getpid(),"updated_utc":now()})
                write(out/f"{name}.json",result)
                print(json.dumps({"arm":name,**log}),flush=True)
        result["training_seconds"]=time.monotonic()-start
        result["training_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        result["training_peak_increment_bytes"]=result["training_peak_allocated_bytes"]-result["allocated_before_training_bytes"]
        result["gradient_clip_fraction"]=clipped/len(rows)
        result["final_parameter_sha256"]=tensor_hash(model.state_dict())
        torch.save({"architecture":arch,"gradient_horizon":k,"seed":seed,"completed_updates":len(rows),
            "state_dict":{n:v.detach().cpu() for n,v in model.state_dict().items()}},out/f"{name}.pt")
        result["status"]="TRAINED";write(out/f"{name}.json",result)
        model.eval()
        for size,data in tests.items():
            write(out/"status.json",{"status":"RUNNING","phase":"evaluation","arm":name,"size":size,
                  "pid":os.getpid(),"updated_utc":now()})
            result["evaluation"][str(size)]=evaluate(model,data,preflight)
        result["status"]="COMPLETE"
    except Exception as error:
        result["status"]="TIME_BUDGET" if isinstance(error,TimeoutError) else "ERROR"
        result["error"]=repr(error);result["traceback"]=traceback.format_exc()
        if not (out/f"{name}.pt").exists():
            torch.save({"partial":True,"completed_updates":result["completed_updates"],
                "state_dict":{n:v.detach().cpu() for n,v in model.state_dict().items()}},out/f"{name}.pt")
    finally:
        result["elapsed_seconds"]=time.monotonic()-start;write(out/f"{name}.json",result)
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return result


def aggregate(out,results,preflight,updates):
    complete=len(results)==(4 if preflight else 8) and all(r["status"]=="COMPLETE" and r["completed_updates"]==updates for r in results)
    if preflight:
        recommendation=None;cost=None
        if complete:
            cost=max(float(np.median(r["update_seconds"][1:])) for r in results)
            recommendation=min(600,50*math.floor(1140/(8*1.25*cost)/50))
        doc={"complete":complete,"decision":"PREFLIGHT_PASSED" if complete and recommendation>=200 else "PREFLIGHT_FAILED",
             "budget_worst_update_seconds":cost,"recommended_updates":recommendation}
    else:
        pairs=[];architectures={}
        for arch in ARCHS:
            for seed in (0,1):
                rs={r["gradient_horizon"]:r for r in results if r["architecture"]==arch and r["seed"]==seed and r["status"]=="COMPLETE"}
                if len(rs)!=2:continue
                f=rs[64]["evaluation"]["32"]["64"];s=rs[8]["evaluation"]["32"]["64"]
                fb,sb=f["original"]["balanced_accuracy"],s["original"]["balanced_accuracy"]
                fp,sp=f["far_paired"]["mean"],s["far_paired"]["mean"]
                identity={key:rs[64][key]==rs[8][key] for key in ("initial_parameter_sha256","train_data_sha256","schedule_sha256")}
                assert all(identity.values())
                qualified=fb>=.85 and fp>=.80
                reach=sp>=.80 and sp>=fp-.05 and sb>=.85 and sb>=fb-.03
                hold=all(rs[8]["evaluation"]["32"][str(t)]["original"]["balanced_accuracy"]>=sb-.03 and
                         rs[8]["evaluation"]["32"][str(t)]["far_paired"]["mean"]>=sp-.05 for t in (128,256))
                pairs.append({"architecture":arch,"seed":seed,"paired_identity":identity,
                    "full_ba64":fb,"short_ba64":sb,"full_far_paired64":fp,"short_far_paired64":sp,
                    "far_effect_pp":100*(sp-fp),"full_qualified":qualified,"short_reach_predicates_pass":reach,
                    "short_hold_predicates_pass":hold,"decision":"BASELINE_UNQUALIFIED" if not qualified else
                       ("SHORT_REACH_PASS" if reach else "SHORT_REACH_FAIL")})
            pp=[p for p in pairs if p["architecture"]==arch]
            architectures[arch]={"completed_pairs":len(pp),"mean_far_effect_pp":float(np.mean([p["far_effect_pp"] for p in pp])) if pp else None,
                "reach_status":"INCOMPLETE" if len(pp)!=2 else ("BASELINE_UNQUALIFIED" if not all(p["full_qualified"] for p in pp) else
                    ("SHORT_REACH_SCREEN_PASS" if all(p["short_reach_predicates_pass"] for p in pp) else "SHORT_REACH_SCREEN_FAIL")),
                "reach_and_hold_pass":len(pp)==2 and all(p["full_qualified"] and p["short_reach_predicates_pass"] and p["short_hold_predicates_pass"] for p in pp)}
        doc={"complete":complete,"decision":"COMPLETE" if complete else "INCOMPLETE","pairs":pairs,"architectures":architectures}
        lines=["# Matched short-BPTT screen", "",f"Execution complete: {complete}. Frozen protocol: new/short_bptt/PROTOCOL.md.","",
            "Size32/T64; far means graph distance >16. Percentages; model seed is independent unit.","",
            "| Cell | Seed | Full BA | Short BA | Full far paired | Short far paired | Far effect pp | Decision | Hold predicates |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|"]
        for p in pairs:
            lines.append(f"| {p['architecture']} | {p['seed']} | {100*p['full_ba64']:.2f} | {100*p['short_ba64']:.2f} | {100*p['full_far_paired64']:.2f} | {100*p['short_far_paired64']:.2f} | {p['far_effect_pp']:+.2f} | {p['decision']} | {p['short_hold_predicates_pass']} |")
        lines += ["","Full controls that fail qualification prevent interpreting a truncation failure.",
                  "Hold alone cannot rescue failed reach. Size64 and longer rollout are secondary.",
                  "See per-arm JSON for all horizons, per-map scores, distance bins, timing and memory."]
        (out/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    doc["statuses"]=[{k:r[k] for k in ("architecture","gradient_horizon","seed","status","completed_updates")} for r in results]
    write(out/"aggregate.json",doc);return doc


def main():
    global DEADLINE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True);parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--preflight-dir")
    args=parser.parse_args();source_hashes=sources()
    if args.preflight:updates=3
    else:
        if not args.preflight_dir:parser.error("Formal run requires --preflight-dir")
        pf=Path(args.preflight_dir).resolve()
        assert json.loads((pf/"status.json").read_text())["status"]=="PREFLIGHT_PASSED"
        assert json.loads((pf/"manifest.json").read_text())["source_sha256"]==source_hashes
        updates=json.loads((pf/"aggregate.json").read_text())["recommended_updates"]
        assert 200<=updates<=600 and updates%50==0
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();DEADLINE=started+(180 if args.preflight else 1500)
    torch.set_num_threads(2);assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    tests={n:bank(n,2 if args.preflight else 16,30000+n,'cuda') for n in ((32,) if args.preflight else (32,64))}
    manifest={"protocol":"short_bptt_v1","started_utc":now(),"pid":os.getpid(),"training":True,
        "preflight":args.preflight,"updates_per_arm":updates,"seeds":[0] if args.preflight else [0,1],
        "architectures":ARCHS,"gradient_horizons":[8,64],"forward_steps":64,"loss_times":list(range(8,65,8)),
        "source_sha256":source_hashes,"evaluation_data_sha256":{str(n):tensor_hash(d) for n,d in tests.items()},
        "torch":torch.__version__,"numpy":np.__version__,"gpu":torch.cuda.get_device_name(),
        "maximum_seconds":180 if args.preflight else 1500,
        "backend":{"cudnn_benchmark":False,"cudnn_deterministic":False,"cudnn_tf32":True,"matmul_tf32":False}}
    write(out/"manifest.json",manifest)
    for name in source_hashes:
        dest=out/"source"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    def timeout():
        write(out/"watchdog_timeout.json",{"status":"TIME_BUDGET","time":now()});os._exit(124)
    watchdog=threading.Timer(240 if args.preflight else 1560,timeout);watchdog.daemon=True;watchdog.start()
    results=[];status="ERROR"
    try:
        for seed in manifest["seeds"]:
            train=bank(32,512,10000+seed,'cuda')
            rows=np.random.default_rng(20000+seed).integers(0,512,(updates,8)).tolist()
            write(out/f"schedule_seed{seed}.json",rows)
            order=[("ws_additive",64),("ws_additive",8),("ws_revision",8),("ws_revision",64)] if seed==0 else [
                   ("ws_revision",64),("ws_revision",8),("ws_additive",8),("ws_additive",64)]
            for arch,k in order:
                budget();r=train_one(out,arch,k,seed,rows,train,tests,args.preflight);results.append(r)
                aggregate(out,results,args.preflight,updates)
                if r["status"]=="TIME_BUDGET":raise TimeoutError("Arm reached cap")
            del train
        report=aggregate(out,results,args.preflight,updates)
        status=report["decision"] if args.preflight else ("COMPLETE" if report["complete"] else "FAILED_OR_INCOMPLETE")
    except Exception as error:
        status="TIME_BUDGET" if isinstance(error,TimeoutError) else "ERROR"
        write(out/"error.json",{"error":repr(error),"traceback":traceback.format_exc()})
        aggregate(out,results,args.preflight,updates)
    finally:watchdog.cancel()
    write(out/"status.json",{"status":status,"pid":os.getpid(),"completed_arms":len(results),"finished_utc":now(),"elapsed_seconds":time.monotonic()-started})
    print(json.dumps({"status":status,"elapsed_seconds":time.monotonic()-started}),flush=True)
    if status!=("PREFLIGHT_PASSED" if args.preflight else "COMPLETE"):raise SystemExit(1)


if __name__=="__main__":main()
