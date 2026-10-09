"""Three new ReLU-input-lift K8 arms against frozen AU-NCA controls."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from new.au_nca import run as prior
from new.relu_input_lift.cells import ReLUInputLiftCell as NCACell

CONFIG = dict(prior.CONFIG)
CONFIG.update({
    "protocol": "relu_activation_conditioned_input_lift_v0",
    "arms": [["relu_lift_k8", "relu_lift", 8]],
    "historical_control_arms": ["original_k64", "original_k8", "au_k8"],
    "state_scalars_per_cell": 6289,
    "benefit_mean_gain_min": .03, "benefit_improving_blocks_min": 2,
    "benefit_worsening_over_02_blocks_max": 1,
    "strong_mean_nmse_max": .10, "strong_half_remaining_gap_blocks_min": 2,
})
atomic_json, atomic_torch, read, sha = prior.atomic_json, prior.atomic_torch, prior.read, prior.sha


def sources():
    names = [f"new/relu_input_lift/{n}" for n in ("run.py", "cells.py", "checks.py", "PROTOCOL.md", "THEORY.md")]
    names += [f"new/au_nca/{n}" for n in ("run.py", "cells.py", "tasks.py", "checks.py", "PROTOCOL.md", "THEORY.md")]
    names += ["tools/start_protected_job.ps1", "tools/protected_job_worker.ps1"]
    return {name: sha(ROOT / name) for name in names}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reference_binding(reference):
    reference = reference.resolve()
    if not reference.is_relative_to(ROOT):
        raise ValueError("Reference must be in this project")
    manifest = read(reference / "manifest.json")
    aggregate = read(reference / "aggregate.json")
    if manifest["config"] != prior.CONFIG or aggregate["status"] != "COMPLETE":
        raise ValueError("Frozen reference configuration or completion mismatch")
    if not aggregate["decision"]["all_controls_qualified"]:
        raise ValueError("Frozen full-credit positive controls are not qualified")
    if (sha(reference / "qualification.json") != manifest["qualification_sha256"]
            or read(reference / "qualification.json")["status"] != "PASS"):
        raise ValueError("Historical qualification binding changed")
    for name, expected in manifest["source_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Historical implementation changed: {name}")
    for name, expected in manifest["data_hashes"].items():
        if sha(reference / name) != expected:
            raise ValueError(f"Historical input changed: {name}")
    artifacts = {name: sha(reference / name) for name in ("aggregate.json", "manifest.json", "qualification.json")}
    controls = []
    for block in range(CONFIG["blocks"]):
        for arm in CONFIG["historical_control_arms"]:
            name = f"block{block:02d}/{arm}/result.json"
            result = read(reference / name)
            if result["status"] != "COMPLETE" or result["block"] != block or result["arm"] != arm:
                raise ValueError(f"Incomplete/mismatched frozen control: {name}")
            if result["init_seed"] != CONFIG["init_seeds"][block] or result["schedule_seed"] != CONFIG["schedule_seeds"][block]:
                raise ValueError("Historical seed binding changed")
            if arm == "original_k64" and not prior.qualified(result):
                raise ValueError("Historical K64 qualification failed")
            artifacts[name] = sha(reference / name)
            controls.append(result)
    return {"reference": reference.relative_to(ROOT).as_posix(),
            "data_hashes": manifest["data_hashes"], "source_hashes": manifest["source_hashes"],
            "artifact_hashes": artifacts, "historical_not_rerun": True}, controls


def new_model(block, device):
    torch.manual_seed(CONFIG["init_seeds"][block])
    return NCACell(channels=CONFIG["channels"], hidden=CONFIG["hidden"]).to(device)


def gradient_diagnostics(model, target, masks):
    initial = prior.make_seed(CONFIG["batch_size"], CONFIG["size"], CONFIG["channels"], device=next(model.parameters()).device)
    gradients, losses = {}, {}
    for label, mode, credit in (("full", "original", 64), ("original_k8", "original", 8),
                               ("au_k8", "au", 8), ("relu_lift_k8", "relu_lift", 8)):
        gradients[label], losses[label] = prior.flat_grad(model, mode, credit, initial, masks, target)
    result = {"same_parameters_inputs_loss": True, "losses": losses, "groups": {}}
    for group in ("W", "eta"):
        gf = gradients["full"][group]
        fnorm = float(gf.norm())
        row = {"full_norm": fnorm}
        for name in ("original_k8", "au_k8", "relu_lift_k8"):
            gs = gradients[name][group]
            snorm = float(gs.norm())
            row[name] = {"norm": snorm,
                "cosine_full": float(torch.dot(gs, gf)/(gs.norm()*gf.norm())) if min(snorm, fnorm) > 1e-20 else None,
                "relative_residual": float((gf-gs).norm())/fnorm if fnorm > 1e-20 else None,
                "captured_parallel_component": float(torch.dot(gs,gf))/(fnorm*fnorm) if fnorm > 1e-20 else None}
        row["new_minus_au_norm"] = float((gradients["relu_lift_k8"][group]-gradients["au_k8"][group]).norm())
        result["groups"][group] = row
    return result


def qualify(out, reference, device):
    from new.relu_input_lift.checks import run_checks
    out.mkdir(parents=True)
    binding, _ = reference_binding(reference)
    checked = run_checks(device=str(device))
    if not checked["passed"]:
        atomic_json(out / "qualification.json", {"status": "FAIL", "checks": checked})
        raise AssertionError("ReLU input lift qualification failed")
    gc.collect()
    torch.cuda.empty_cache()
    target = prior.make_target(CONFIG["size"]).to(device)
    initial = prior.make_seed(CONFIG["batch_size"], CONFIG["size"], CONFIG["channels"], device=device)
    model = new_model(0, device)
    opt = prior.new_optimizer(model)
    prior.update(model, opt, initial, prior.train_masks(0,1,device), "relu_lift", 8, target)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    for u in (2,3,4):
        _, loss, _, _ = prior.update(model, opt, initial, prior.train_masks(0,u,device), "relu_lift", 8, target)
        losses.append(loss)
    torch.cuda.synchronize()
    seconds = (time.perf_counter()-started)/3
    smoke = {"seconds_per_update": seconds, "smoke_losses": losses,
             "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
             "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
             "parameter_count": sum(p.numel() for p in model.parameters()),
             "state_scalars_per_cell": CONFIG["state_scalars_per_cell"]}
    record = {"status": "PASS", "config": CONFIG, "source_hashes": sources(), "checks": checked,
              "reference_binding": binding, "actual_shape_smoke": smoke,
              "training_estimate_seconds": seconds*CONFIG["updates"]*CONFIG["blocks"],
              "estimate_excludes_evaluation_and_io": True}
    atomic_json(out / "qualification.json", record)
    print(json.dumps({"status":"PASS", "actual_shape_smoke":smoke,
                      "training_estimate_seconds":record["training_estimate_seconds"]}), flush=True)


def checkpoint(model, opt, u, pool=None):
    value = {"model": model.state_dict(), "optimizer": opt.state_dict(), "update": u,
             "config": CONFIG, "source_hashes": sources(), "torch_rng_state": torch.get_rng_state(),
             "cuda_rng_state": torch.cuda.get_rng_state_all()}
    if pool is not None:
        value["pool"] = pool.detach().cpu()
    return value


def train_arm(out, block, target, data, completed, started, resume):
    folder = out / f"block{block:02d}" / "relu_lift_k8"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists():
        if not resume:
            raise FileExistsError("Arm exists without explicit resume")
        return read(result_path)
    device = torch.device(CONFIG["device"])
    model = new_model(block, device)
    opt = prior.new_optimizer(model)
    indices, intact, damaged = data
    target_tensor = torch.as_tensor(target, device=device)
    pool = prior.make_seed(CONFIG["pool_size"], CONFIG["size"], CONFIG["channels"], device=device)
    latest = folder / "checkpoints" / "latest.pt"
    first = 1
    if resume and latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["config"] != CONFIG or saved["source_hashes"] != sources():
            raise ValueError("Checkpoint provenance changed")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        pool.copy_(saved["pool"])
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng_state"]])
        first = saved["update"]+1
        log = folder / "training.jsonl"
        if log.exists():
            shutil.copyfile(log, folder / f"training_before_resume_{time.time_ns()}.jsonl")
            kept = []
            for line in log.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["update"] < first:
                    kept.append(line)
            log.write_text("\n".join(kept)+("\n" if kept else ""), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    arm_started = time.perf_counter()
    failure = None
    for u in range(first, CONFIG["updates"]+1):
        ids = torch.as_tensor(indices[u-1], device=device, dtype=torch.long)
        initial = pool[ids].clone()
        with torch.no_grad():
            worst = (initial[:,:4]-target_tensor).square().mean(dim=(1,2,3)).argmax()
            initial[worst] = prior.make_seed(1, CONFIG["size"], CONFIG["channels"], device=device)[0]
        try:
            produced, loss, wn, an = prior.update(model, opt, initial, prior.train_masks(block,u,device), "relu_lift", 8, target_tensor)
        except FloatingPointError as exc:
            failure = {"update":u,"error":str(exc)}
            atomic_torch(folder / "checkpoints" / "numerical_failure.pt", checkpoint(model,opt,u,pool))
            break
        pool[ids] = produced
        prior.jsonl(folder / "training.jsonl", {"update":u,"loss":loss,
                    "W_gradient_norm_before_normalization":wn,"eta_gradient_norm_before_normalization":an})
        if u == 1 or u % CONFIG["recovery_interval"] == 0:
            atomic_torch(latest, checkpoint(model,opt,u,pool))
            status = {"status":"RUNNING","completed_units":completed,"total_units":CONFIG["blocks"],
                      "block":block,"arm":"relu_lift_k8","update":u,"updates":CONFIG["updates"],
                      "loss":loss,"elapsed_seconds":time.perf_counter()-started}
            atomic_json(out / "status.json", status)
            print(json.dumps(status), flush=True)
        if u in CONFIG["checkpoints"]:
            atomic_torch(folder / "checkpoints" / f"u{u:04d}.pt", checkpoint(model,opt,u))
            if u != CONFIG["updates"]:
                ev, arrays, _ = prior.evaluate(model,"relu_lift",target,intact,diagnostic=True)
                atomic_json(folder / f"diagnostic_u{u:04d}.json",ev)
                np.savez_compressed(folder / f"diagnostic_u{u:04d}.npz",**arrays)
    torch.cuda.synchronize()
    result = {"status":"NUMERICAL_FAILURE" if failure else "COMPLETE","block":block,
              "arm":"relu_lift_k8","mode":"relu_lift","credit":8,
              "init_seed":CONFIG["init_seeds"][block],"schedule_seed":CONFIG["schedule_seeds"][block],
              "failure":failure,"systems":{"seconds_this_execution":time.perf_counter()-arm_started,
              "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
              "parameter_count":sum(p.numel() for p in model.parameters()),
              "state_scalars_per_cell":CONFIG["state_scalars_per_cell"],"resumed_first_update":first}}
    if not failure:
        ev, arrays, curves = prior.evaluate(model,"relu_lift",target,intact,damaged)
        result["evaluation"] = ev
        np.savez_compressed(folder / "predictions.npz",**arrays)
        atomic_json(folder / "continuation_curves.json",curves)
    # Save the completed primary before optional gradient diagnostics.
    atomic_json(result_path,result)
    if not failure:
        try:
            diag = gradient_diagnostics(model,target_tensor,prior.train_masks(block,9000,device))
            atomic_json(folder / "gradient_diagnostics.json",diag)
        except (FloatingPointError, torch.OutOfMemoryError) as exc:
            atomic_json(folder / "gradient_diagnostics.json",{"status":"DIAGNOSTIC_UNAVAILABLE","error_type":type(exc).__name__,"error":str(exc)})
            torch.cuda.empty_cache()
    return result


def decision(results, controls):
    rows = []
    for block in range(CONFIG["blocks"]):
        values = {r["arm"]:r.get("evaluation",{}).get("intact_t64",{}).get("nmse",{}).get("mean")
                  for r in controls+results if r["block"] == block}
        if all(values.get(a) is not None for a in ["relu_lift_k8","au_k8","original_k64","original_k8"]):
            rows.append({"block":block,"nmse":values,"au_minus_new":values["au_k8"]-values["relu_lift_k8"],
                         "half_remaining_gap_recovered":values["relu_lift_k8"] <= .5*(values["au_k8"]+values["original_k64"])})
    if len(results)<CONFIG["blocks"]:
        return {"verdict":"INCOMPLETE","paired_primary":rows}
    if len(rows)<CONFIG["blocks"]:
        return {"verdict":"NUMERICAL_FAILURE_DEVELOPMENTAL","paired_primary":rows}
    gains = [r["au_minus_new"] for r in rows]
    mean = float(np.mean(gains))
    benefit = (mean >= CONFIG["benefit_mean_gain_min"]
               and sum(g > 0 for g in gains) >= CONFIG["benefit_improving_blocks_min"]
               and sum(g < -.02 for g in gains) <= CONFIG["benefit_worsening_over_02_blocks_max"])
    strong = (benefit and float(np.mean([r["nmse"]["relu_lift_k8"] for r in rows])) <= CONFIG["strong_mean_nmse_max"]
              and sum(r["half_remaining_gap_recovered"] for r in rows) >= CONFIG["strong_half_remaining_gap_blocks_min"])
    verdict = "RELU_LIFT_RECOVERY_DEVELOPMENTAL" if strong else (
        "RELU_LIFT_BENEFIT_DEVELOPMENTAL" if benefit else "NO_QUALIFIED_RELU_LIFT_BENEFIT")
    return {"verdict":verdict,"mean_au_minus_new":mean,"improving_blocks":sum(g>0 for g in gains),"paired_primary":rows}


def report(out, results, controls, started):
    d = decision(results,controls)
    complete = len(results) == CONFIG["blocks"] and all(r["status"] == "COMPLETE" for r in results)
    failed = any(r["status"] != "COMPLETE" for r in results)
    aggregate = {"status":"COMPLETE" if complete else ("NUMERICAL_FAILURE" if failed else "INCOMPLETE"),
                 "config":CONFIG,"decision":d,"arms":results,"historical_controls":controls,
                 "completed_units":len(results),"planned_units":CONFIG["blocks"],
                 "elapsed_seconds_this_execution":time.perf_counter()-started,
                 "independent_unit":"paired initialization and schedule block; not evaluation episode",
                 "historical_controls_not_rerun":True}
    atomic_json(out / "aggregate.json",aggregate)
    lines = ["# ReLU activation-conditioned input lift v0","",
             f"Execution **{aggregate['status']}**; verdict **{d['verdict']}**.",
             f"New arms completed: {len(results)}/3. Nine historical arms are frozen controls, not new training.","",
             "|Block|Arm|T64 NMSE|T64 alpha IoU|T128 NMSE|T256 NMSE|Damaged T256 NMSE|",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in sorted(controls+results,key=lambda r:(r["block"],r["arm"])):
        vals = [r.get("evaluation",{}).get(k,{}).get(m,{}).get("mean") for k,m in
                (("intact_t64","nmse"),("intact_t64","alpha_iou"),("intact_t128","nmse"),("intact_t256","nmse"),("damaged_t256","nmse"))]
        lines.append(f"|{r['block']}|{r['arm']}|"+"|".join("NA" if v is None else f"{v:.6f}" for v in vals)+"|")
    lines += ["","Primary is cold-seed T64 NMSE at u3000; intermediate checkpoints are diagnostic only.",
              "Same-state T128/T256 and untrained damage are secondary. Gradient cosine is not a qualification endpoint.",
              "Direct historical feature-layer credit is restored; feedback through pre-cut perceived states remains truncated.",
              "State is6289 floats/cell versus145 for AU and16 for Original. Parameters remain8336.",
              "This three-block single-target screen cannot establish general NCA reliability or a universal BPTT solution.",""]
    (out / "RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    atomic_json(out / "status.json",{k:aggregate[k] for k in ("status","decision","completed_units","planned_units","elapsed_seconds_this_execution")})
    print(json.dumps({"status":aggregate["status"],"decision":d}),flush=True)


def validate_resume(out):
    manifest = read(out / "manifest.json")
    if manifest["config"] != CONFIG or manifest["source_hashes"] != sources():
        raise ValueError("Run provenance changed")
    for name, expected in manifest["data_hashes"].items():
        if sha(out/name)!=expected:
            raise ValueError("Run input changed")


def run(out, qualification, reference, device, resume):
    qual = read(qualification)
    binding, controls = reference_binding(reference)
    if qual["status"]!="PASS" or qual["config"]!=CONFIG or qual["source_hashes"]!=sources() or qual["reference_binding"]!=binding:
        raise ValueError("Qualification/provenance mismatch")
    if resume:
        validate_resume(out)
        if read(out/"manifest.json")["reference_binding"]!=binding:
            raise ValueError("Historical reference changed")
    else:
        out.mkdir(parents=True)
        shutil.copyfile(qualification,out/"qualification.json")
        for name in sources():
            dest=out/"source"/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,dest)
        for name in binding["data_hashes"]:
            shutil.copyfile(reference/name,out/name)
        for name in binding["artifact_hashes"]:
            dest=out/"reference"/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(reference/name,dest)
        atomic_json(out/"manifest.json",{"config":CONFIG,"source_hashes":sources(),
                    "data_hashes":binding["data_hashes"],"reference_binding":binding,"qualification_sha256":sha(qualification)})
    with prior.run_lock(out):
        started=time.perf_counter()
        if resume:
            prior.jsonl(out/"recovery_events.jsonl",{"event":"explicit_resume","local_time":time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        atomic_json(out/"local_dispatch_receipt.json",{"host":socket.gethostname(),"pid":os.getpid(),
                    "gpu":torch.cuda.get_device_name(),"device":str(device),"command":sys.argv,
                    "started_local":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"runtime_cap":None,"resume":resume})
        target=np.load(out/"target_rgba.npy")
        results=[]
        for block in range(CONFIG["blocks"]):
            result=train_arm(out,block,target,prior.load_inputs(out,block),len(results),started,resume)
            results.append(result)
            if result["status"] != "COMPLETE":
                break
        report(out,results,controls,started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--qualification",type=Path)
    parser.add_argument("--reference",type=Path,default=ROOT/"runs/au_nca_20261009_01")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    out=args.out.resolve();reference=args.reference.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Output outside owning project")
    if out.exists():
        if args.check or not args.resume:
            raise FileExistsError("Use a fresh output directory")
        validate_resume(out)
        if read(out/"status.json").get("status")=="COMPLETE":
            raise RuntimeError("Completed evidence is read-only")
    elif args.resume:
        raise FileNotFoundError("Resume requires existing bound run")
    if not args.check and args.qualification is None:
        parser.error("--qualification is required")
    device=prior.setup()
    try:
        if args.check:
            qualify(out,reference,device)
        else:
            run(out,args.qualification.resolve(),reference,device,args.resume)
    except prior.ActiveRunError:
        raise
    except Exception as exc:
        if out.exists():
            atomic_json(out/"status.json",{"status":"ERROR","verdict":"INCOMPLETE","error_type":type(exc).__name__,"error":str(exc)})
            (out/"fatal_error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()
