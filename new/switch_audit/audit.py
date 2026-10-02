"""Zero-training candidate and state-intervention audit; see PROTOCOL.md."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "new/workspace_revision"))
from revision_cells import make_cell
from run_revision import sha, tensor_hash, write, score, accuracy_vector, open_rms
from tasks import bank

KPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
HORIZONS = (16, 32, 64, 96, 128, 192, 256)
BRANCHES = ("warm", "reset_W", "reset_Z", "cold", "transplant_W", "transplant_Z")
DEADLINE = float("inf")


def budget():
    if time.monotonic() > DEADLINE:
        raise TimeoutError("Five-minute audit budget exhausted")


def now():
    return datetime.now(timezone.utc).isoformat()


def vec(value):
    return {"mean": float(value.mean()), "per_map": value.cpu().tolist()}


def maximum_error(a, b):
    return float((a-b).abs().max())


@torch.no_grad()
def probe(model, state, x):
    w, z = state
    wplus = w + model.eta * model.f_out(torch.tanh(model.f_in(model._features(w, z, x))))
    q = model._candidate(wplus, z, x)
    nxt = (wplus, z + model.alpha*(q-z))
    real = model.step(state, x)
    step_error = max(maximum_error(a, b) for a, b in zip(nxt, real))
    affine_error = maximum_error(model.logits(nxt), .5*model.readout(z)+.5*model.readout(q))
    assert step_error <= 1e-6
    torch.testing.assert_close(model.logits(nxt), .5*model.readout(z)+.5*model.readout(q), rtol=1e-5, atol=2e-6)
    assert all(bool(torch.isfinite(t).all()) for t in (*nxt, q))
    return wplus, q, nxt, step_error, affine_error


def check():
    torch.manual_seed(123)
    model = make_cell("ws_revision")
    with torch.no_grad():
        model.f_out.weight.normal_(std=.03)
        model.q_out.weight.normal_(std=.03)
        model.readout.bias.fill_(.37)
    data = bank(8, 2, 901)
    state = model.initial(data["x"])
    state = (state[0], torch.randn_like(state[1]))
    saved = tuple(t.clone() for t in state)
    wp, q, nxt, se, ae = probe(model, state, data["x_flip"])
    assert all(torch.equal(a,b) for a,b in zip(saved,state))
    assert torch.equal(q, model._candidate(wp, state[1], data["x_flip"]))
    assert all(torch.equal(a,b) for a,b in zip(model.initial(data["x_flip"]),
                   (model.encoder(data["x_flip"]), torch.zeros_like(state[1]))))
    return {"status":"PASS", "step_max_error":se, "affine_readout_max_error":ae,
            "checks":["post_F_candidate_clock", "affine_readout_with_nonzero_bias", "no_state_mutation", "reset_both_equals_cold"]}


@torch.no_grad()
def describe(logits, data):
    out = score(logits, data["y_flip"], data["mask"])
    sign = 2*data["y_flip"]-1
    for name, mask in (("changed", data["changed"]), ("unchanged", data["mask"]-data["changed"])):
        out[name] = vec(accuracy_vector(logits, data["y_flip"], mask))
        margin = (logits*sign*mask).sum((1,2,3))/mask.sum((1,2,3)).clamp_min(1)
        out[name+"_margin"] = vec(margin)
    out["changed_distance"] = {}
    for lo, hi in ((0,1),(1,4),(4,8),(8,16),(16,100000)):
        mask = data["changed"]*((data["distance"]>=lo)&(data["distance"]<hi))
        counts = mask.sum((1,2,3)); eligible = counts>0
        acc = accuracy_vector(logits, data["y_flip"], mask)
        out["changed_distance"][f"{lo}_{hi}"] = {
            "mean": float(acc[eligible].mean()) if bool(eligible.any()) else None,
            "eligible_maps": int(eligible.sum()), "pixels":int(counts.sum())}
    return out


def compare(recorded, observed, label, comparisons, accuracy=False):
    a,b = np.asarray(recorded),np.asarray(observed)
    error = float(np.max(np.abs(a-b)))
    assert np.allclose(a,b,atol=1e-6,rtol=0 if accuracy else 1e-5), (label,error)
    comparisons.append({"field":label,"max_error":error})


@torch.no_grad()
def natural(model, data, reference, comparisons):
    x, xf = data["x"], data["x_flip"]
    old, new = model.initial(x), model.initial(xf)
    save_times = set(KPOINTS)|{64+k for k in KPOINTS}
    old_saved, new_saved = {0:old}, {0:new}
    for t in range(1,257):
        budget()
        old, new = model.step(old,x), model.step(new,xf)
        if t in save_times:
            old_saved[t],new_saved[t] = old,new
        if t in HORIZONS:
            prev = reference["curve"][str(t)]
            for name,state,target,r in (("old",old,data["y"],prev),("new",new,data["y_flip"],prev["flip"])):
                measured = score(model.logits(state),target,data["mask"])
                for field in ("balanced_accuracy","bce","per_map_ba"):
                    compare(r[field],measured[field],f"{name}/T{t}/{field}",comparisons,field!="bce")
            for j, field in enumerate(("w_rms","z_rms")):
                compare(prev[field+"_per_map"], open_rms(old[j],data["mask"]).cpu().tolist(),f"T{t}/{field}",comparisons)
    return old_saved,new_saved


@torch.no_grad()
def audit_size(model, data, reference):
    comparisons=[]
    olds, news = natural(model,data,reference,comparisons)
    w,z = olds[64]; init = model.initial(data["x_flip"])
    branches={"warm":(w,z),"reset_W":(init[0],z),"reset_Z":(w,init[1]),
              "cold":init,"transplant_W":(news[64][0],z),"transplant_Z":(w,news[64][1])}
    records={}; max_step=0.; max_affine=0.
    for k in range(129):
        budget()
        if k in KPOINTS:
            row={"branches":{},"factorial":{},"references":{}}
            probes={}
            for name, state in branches.items():
                wp,q,nxt,se,ae = probe(model,state,data["x_flip"])
                max_step=max(max_step,se); max_affine=max(max_affine,ae)
                probes[name]=(wp,q)
                row["branches"][name]={"output":describe(model.logits(state),data),
                    "candidate":describe(model.readout(q),data),
                    "rms":{label:vec(open_rms(v,data["mask"])) for label,v in
                        (("W",state[0]),("Z",state[1]),("Q",q),("dW",nxt[0]-state[0]),("dZ",nxt[1]-state[1]))}}
            aged = news[64+k]
            wp_new, q_new, _,se,ae = probe(model,aged,data["x_flip"])
            max_step=max(max_step,se); max_affine=max(max_affine,ae)
            row["references"]["new_age64plusK"]={"output":describe(model.logits(aged),data),"candidate":describe(model.readout(q_new),data)}
            row["references"]["old_age64plusK"]={"output":describe(model.logits(olds[64+k]),data)}
            for wn,wd in (("warm",probes["warm"][0]),("new_aged",wp_new)):
                for zn,zd in (("warm",branches["warm"][1]),("new_aged",aged[1])):
                    for xn,xd in (("old",data["x"]),("new",data["x_flip"])):
                        q=model._candidate(wd,zd,xd)
                        row["factorial"][f"W_{wn}__Z_{zn}__X_{xn}"]=describe(model.readout(q),data)
            row["encoded_W_candidate"]={}
            for xn,xd in (("old",data["x"]),("new",data["x_flip"])):
                q=model._candidate(init[0],branches["warm"][1],xd)
                row["encoded_W_candidate"][xn]=describe(model.readout(q),data)
            for branch,key in (("warm","switch"),("cold","switch_cold")):
                if str(k) in reference["interventions_from_T64"]:
                    prev=reference["interventions_from_T64"][str(k)][key]
                    curr=row["branches"][branch]["output"]
                    for field in ("balanced_accuracy","bce","per_map_ba"):
                        compare(prev[field],curr[field],f"{branch}/K{k}/{field}",comparisons,field!="bce")
                    for field in ("changed","unchanged"):
                        compare(prev[field+"_per_map"],curr[field]["per_map"],f"{branch}/K{k}/{field}",comparisons,True)
            records[str(k)]=row
        if k<128:
            branches={name:model.step(state,data["x_flip"]) for name,state in branches.items()}
    return {"status":"COMPLETE","map_count":len(data["x"]),"anchors":records,
            "validation":{"replay":comparisons,"step_max_error":max_step,"affine_readout_max_error":max_affine}}


def main():
    global DEADLINE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--out")
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.check:
        print(json.dumps(check(),indent=2)); return
    if not args.out: parser.error("--out required")
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); DEADLINE=started+300
    manifest=json.loads((ROOT/"REVISION_PUBLICATION_MANIFEST.json").read_text())
    path=ROOT/"runs/workspace_revision_20261002_paired01/ws_revision_seed0.pt"
    assert sha(path)==manifest["checkpoint_sha256"]["ws_revision_seed0"]
    for name,expected in manifest["source_sha256"].items(): assert sha(ROOT/name)==expected,name
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.allow_tf32=True; torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(0)
    model=make_cell("ws_revision").cuda().eval()
    cp=torch.load(path,map_location="cpu",weights_only=True)
    model.load_state_dict(cp["state_dict"])
    model.requires_grad_(False)
    raw=json.loads((ROOT/"evidence/workspace_revision_paired01/raw/ws_revision_seed0.json").read_text())
    assert tensor_hash(model.state_dict())==raw["final_parameter_sha256"]
    sources={**manifest["source_sha256"],**{p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__),Path(__file__).with_name("PROTOCOL.md"))}}
    for name in sources:
        dest=out/"source"/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,dest)
    write(out/"manifest.json",{"protocol":"switch_audit_v1","started_utc":now(),"training":False,
        "checkpoint_sha256":sha(path),"source_sha256":sources,"torch":torch.__version__,
        "gpu":torch.cuda.get_device_name(),"maximum_seconds":300,"anchors":KPOINTS,
        "sizes":[32,64],"model_seeds":[0],"maps_per_size":16})
    write(out/"launch_receipt.json",{"host":os.environ.get("COMPUTERNAME"),"pid":os.getpid(),
        "started_utc":now(),"command":sys.argv,"output":str(out),"gpu":torch.cuda.get_device_name()})
    def expire():
        write(out/"watchdog_timeout.json",{"status":"TIME_BUDGET","time":now()}); os._exit(124)
    watchdog=threading.Timer(320,expire); watchdog.daemon=True; watchdog.start()
    sizes=[]
    try:
        eval_manifest=json.loads((ROOT/"evidence/workspace_revision_paired01/manifest.json").read_text())
        for n in (32,64):
            write(out/"status.json",{"status":"RUNNING","size":n,"pid":os.getpid(),"updated_utc":now()})
            data=bank(n,16,30000+n)
            assert tensor_hash(data)==eval_manifest["evaluation_data_sha256"][str(n)]
            data={k:v.cuda() for k,v in data.items()}
            result=audit_size(model,data,raw["evaluation"][str(n)])
            write(out/f"size{n}.json",result); sizes.append(n)
            print(json.dumps({"size":n,"status":"COMPLETE","elapsed_seconds":time.monotonic()-started}),flush=True)
        assert tensor_hash(model.state_dict())==raw["final_parameter_sha256"]
        write(out/"status.json",{"status":"COMPLETE","sizes":sizes,"finished_utc":now(),"elapsed_seconds":time.monotonic()-started})
    except Exception as error:
        write(out/"status.json",{"status":"ERROR","sizes":sizes,"error":repr(error),"traceback":traceback.format_exc(),"finished_utc":now()})
        raise
    finally:
        watchdog.cancel()


if __name__=="__main__":
    main()
