"""Publish completed switch audit with lossless compact raw JSON; CPU only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding="utf-8"))
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True);parser.add_argument("--analysis",required=True);parser.add_argument("--out",required=True)
    args=parser.parse_args();run=Path(args.run).resolve();analysis=Path(args.analysis).resolve();out=Path(args.out).resolve()
    pub=ROOT/"SWITCH_PUBLICATION_MANIFEST.json"
    assert not out.exists() and not pub.exists()
    manifest=read(run/"manifest.json");status=read(run/"status.json");summary=read(analysis/"analysis.json")
    assert status["status"]=="COMPLETE" and manifest["training"] is False
    for name,h in manifest["source_sha256"].items():assert sha(ROOT/name)==sha(run/"source"/name)==h,name
    old=read(analysis/"provenance.json")
    for name,h in old["input_sha256"].items():assert sha(ROOT/name)==h,name
    for name,h in old["analysis_source_sha256"].items():assert sha(ROOT/name)==h,name
    for name,h in old["output_sha256"].items():assert sha(analysis/name)==h,name
    raws={n:read(run/f"size{n}.json") for n in (32,64)}
    for row in summary["rows"]:
        b=raws[row["size"]]["anchors"][str(row["K"])]["branches"][row["branch"]]
        assert row["output_changed"]==b["output"]["changed"]["mean"]
        assert row["candidate_changed"]==b["candidate"]["changed"]["mean"]
        assert row["output_unchanged"]==b["output"]["unchanged"]["mean"]
        bits=summary["evidence"][str(row["size"])]["new_target_bits_per_map"]
        pos=[a if bit==1 else 1-a for a,bit in zip(b["output"]["changed"]["per_map"],bits)]
        assert pos==row["predicted_positive_fraction_per_map"]
        assert float(np.mean(pos))==row["predicted_positive_fraction"]
    for row in summary["contrasts"]:
        f=raws[row["size"]]["anchors"][str(row["K"])]["factorial"]
        def v(w,z):return np.array(f[f"W_{w}__Z_{z}__X_new"][row["metric"]]["per_map"])
        ww=v("warm","warm");nw=v("new_aged","warm");wn=v("warm","new_aged");nn=v("new_aged","new_aged")
        value={"replace_W_hold_Z":nw-ww,"replace_Z_hold_W":wn-ww,"joint_minus_additive_contrast":nn-nw-wn+ww}[row["contrast"]]
        assert value.tolist()==row["per_map"] and float(value.mean())==row["mean"]
    replay=[r for d in raws.values() for r in d["validation"]["replay"]]
    assert len(replay)==232 and max(r["max_error"] for r in replay)==0
    out.mkdir(parents=True);(out/"raw").mkdir()
    rawhashes={}
    for n,value in raws.items():
        dest=out/"raw"/f"size{n}.json"
        dest.write_text(json.dumps(value,separators=(",",":"),allow_nan=False)+"\n",encoding="utf-8")
        assert read(dest)==value and dest.stat().st_size<1_000_000
        rawhashes[f"size{n}"]={"original_sha256":sha(run/f"size{n}.json"),"compact_sha256":sha(dest),"decoded_json_equal":True}
    for name in ("analysis.json","overview.png"):shutil.copyfile(analysis/name,out/name)
    report=(analysis/"RESULTS.md").read_text(encoding="utf-8")
    title,body=report.split("\n",1)
    report=title+"\n\nRead the [post-execution review notes](../../new/switch_audit/REVIEW_NOTES.md)\nfor exact replay scope and the quantified factorial contrasts.\n"+body
    (out/"RESULTS.md").write_text(report,encoding="utf-8")
    shutil.copyfile(run/"manifest.json",out/"manifest.json");shutil.copyfile(run/"status.json",out/"completion.json")
    write(out/"validation.json",{**summary["validation"],"summary_rows_checked":len(summary["rows"]),
        "per_map_contrasts_checked":len(summary["contrasts"]),"raw_compaction":rawhashes,
        "replay_scope":"Both natural trajectories BA/BCE/per-map BA; original W/Z RMS only; warm/cold switch metrics.",
        "original_reference_has_no_fresh_flip_norms":True,"no_new_inference_for_publication":True})
    write(out/"provenance.json",{"protocol":"switch_audit_v1","source_run":"switch_audit_20261002_seed0",
        "checkpoint_sha256":manifest["checkpoint_sha256"],"original_analysis_provenance_sha256":sha(analysis/"provenance.json"),
        "raw":rawhashes,"analysis_sha256":sha(out/"analysis.json"),"source_sha256":manifest["source_sha256"],
        "notes":["Compact raw JSON has identical decoded values; original raw bytes remain local.",
                 "Published RESULTS adds a link to post-execution review notes; original report remains unchanged.",
                 "Launch receipt and checkpoints excluded; no new training or inference during publication."]})
    write(pub,{"review_base":"7cb1ca975c13104b3abb157cf5324ea977ad1051","execution_status":"COMPLETE",
        "scientific_status":summary["interpretation"],"training":False,"model_seed":0,
        "source_sha256":manifest["source_sha256"],"analysis_source_sha256":old["analysis_source_sha256"],
        "publication_tool_sha256":{Path(__file__).relative_to(ROOT).as_posix():sha(Path(__file__))},
        "review_notes_sha256":sha(ROOT/"new/switch_audit/REVIEW_NOTES.md"),
        "checkpoint_sha256":manifest["checkpoint_sha256"],
        "published_evidence_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "excluded":["checkpoint","launch receipt","local PID and host","transient logs"],
        "notes":["Raw JSON losslessly compacted; not byte-identical to original.",
                 "No independent-model replication; prior two-seed joint-gate failure unchanged."]})
    print(json.dumps({"status":"EXPORTED","replay_fields":len(replay),"summary_rows":len(summary["rows"]),
        "contrast_records":len(summary["contrasts"]),"bytes":sum(p.stat().st_size for p in out.rglob("*") if p.is_file())},indent=2))


if __name__=="__main__":main()
