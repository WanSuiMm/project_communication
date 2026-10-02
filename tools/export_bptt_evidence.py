"""Publish saved matched-BPTT evidence without training or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True);p.add_argument("--analysis",required=True);p.add_argument("--out",required=True)
    args=p.parse_args();run=Path(args.run).resolve();analysis=Path(args.analysis).resolve();out=Path(args.out).resolve()
    publication=ROOT/"BPTT_PUBLICATION_MANIFEST.json"
    assert not out.exists() and not publication.exists()
    provenance=read(analysis/"provenance.json");manifest=read(run/"manifest.json");status=read(run/"status.json")
    summary=read(analysis/"analysis.json")
    assert status["status"]=="COMPLETE" and status["completed_arms"]==8
    for name,h in provenance["input_sha256"].items():assert sha(ROOT/name)==h,name
    for name,h in provenance["checkpoint_sha256"].items():assert sha(run/name)==h,name
    for name,h in provenance["analysis_source_sha256"].items():assert sha(ROOT/name)==h,name
    for name,h in provenance["output_sha256"].items():assert sha(analysis/name)==h,name
    for name,h in manifest["source_sha256"].items():assert sha(ROOT/name)==sha(run/"source"/name)==h,name
    preflight=ROOT/"runs/short_bptt_20261002_preflight"
    assert read(preflight/"status.json")["status"]=="PREFLIGHT_PASSED"
    assert read(preflight/"manifest.json")["source_sha256"]==manifest["source_sha256"]
    assert read(preflight/"aggregate.json")["recommended_updates"]==manifest["updates_per_arm"]
    out.mkdir(parents=True);(out/"raw").mkdir();(out/"schedules").mkdir();(out/"preflight").mkdir()
    for file in sorted(run.glob("ws_*.json")):shutil.copyfile(file,out/"raw"/file.name)
    for file in sorted(run.glob("schedule_*.json")):shutil.copyfile(file,out/"schedules"/file.name)
    for name in ("RESULTS.md","analysis.json","curves.csv","overview.png"):shutil.copyfile(analysis/name,out/name)
    shutil.copyfile(run/"aggregate.json",out/"aggregate.json")
    shutil.copyfile(run/"RESULTS.md",out/"runner_RESULTS.md")
    write(out/"manifest.json",{k:v for k,v in manifest.items() if k!="pid"})
    write(out/"completion.json",{k:v for k,v in status.items() if k!="pid"})
    for name in ("manifest.json","status.json","aggregate.json"):
        write(out/"preflight"/name,{k:v for k,v in read(preflight/name).items() if k!="pid"})
    write(out/"validation.json",{**summary["validation"],"raw_and_schedules_byte_identical":True,
        "preflight_hashes_match":True,"budget_selection_precedes_efficacy":True})
    write(out/"provenance.json",{"source_run":"short_bptt_20261002_paired01",
        "original_analysis_provenance_sha256":sha(analysis/"provenance.json"),
        "checkpoint_sha256":provenance["checkpoint_sha256"],
        "original_private_manifest_sha256":sha(run/"manifest.json"),
        "original_private_completion_sha256":sha(run/"status.json"),
        "notes":["Raw arm results, schedules and analysis outputs copied byte-identically.",
                 "PID removed only from public manifest/completion/preflight metadata.",
                 "Checkpoints and machine launch receipts remain local; no new inference or training."]})
    write(publication,{"review_base":"c2d705fcb56fdc2b31c68a33ecc7a02610b3fdee",
        "execution_status":"COMPLETE","scientific_status":summary["scientific_status"],
        "source_sha256":manifest["source_sha256"],"analysis_source_sha256":provenance["analysis_source_sha256"],
        "publication_tool_sha256":{Path(__file__).relative_to(ROOT).as_posix():sha(Path(__file__))},
        "checkpoint_sha256":provenance["checkpoint_sha256"],
        "published_evidence_sha256":{f.relative_to(ROOT).as_posix():sha(f) for f in sorted(out.rglob("*")) if f.is_file()},
        "excluded":["checkpoints","local launch receipt","PID and host","transient logs"],
        "notes":["All four full controls unqualified at the frozen far-paired endpoint.",
                 "One K8 additive seed meets reach/hold predicates; not an architecture-level pass.",
                 "Per-map far accuracy80.87% corresponds to pooled59.53%; distance/size limits retained."]})
    print(json.dumps({"status":"EXPORTED","files":sum(f.is_file() for f in out.rglob('*')),
        "bytes":sum(f.stat().st_size for f in out.rglob('*') if f.is_file())},indent=2))


if __name__=="__main__":main()
