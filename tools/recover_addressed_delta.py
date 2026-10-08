"""Resume frozen addressed-write code with JSON-canonical config comparison.

The original runner, its qualification and parent evidence remain unchanged.
This compatibility entrypoint changes only tuple/list config representation;
it records and snapshots its own source in the recovered run manifest.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
ENTRY = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location(
    "_addressed_recovery_runner", ROOT / "new/addressed_delta/run.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--resume-parent", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    parent = RUNNER.area_path(args.resume_parent, "runs")
    qualification = RUNNER.area_path(args.qualification, "analyses")
    checked = RUNNER.read(qualification)
    source = RUNNER.hashes()
    if (checked.get("status") != "PASS"
            or checked.get("protocol") != RUNNER.PROTOCOL
            or checked["source_sha256"] != source):
        raise RuntimeError("Frozen source qualification no longer matches")

    # JSON serializes tuples as lists. The canonical form must still agree
    # exactly with every parent configuration value, not only optimizer betas.
    config_value = json.loads(json.dumps(RUNNER.config(), allow_nan=False))
    if RUNNER.read(parent / "config.json") != config_value:
        raise RuntimeError("Parent scientific configuration differs")
    RUNNER.config = lambda: config_value
    records = RUNNER.committed(parent, source)
    prefix = {}
    for row in records:
        key = f"block{row['block']:02d}/{row['arm']}"
        prefix[key] = max(prefix.get(key, 0), row["update"])
    provenance = {
        "entrypoint": ENTRY.relative_to(ROOT).as_posix(),
        "entrypoint_sha256": RUNNER.C.sha(ENTRY),
        "qualification_sha256": RUNNER.C.sha(qualification),
        "parent_manifest_sha256": RUNNER.C.sha(parent / "manifest.json"),
        "fix": "JSON-canonical config equality for tuple/list serialization",
        "numerical_training_code_changed": False,
        "inherited_committed_stages": len(records),
        "usable_prefix": prefix,
    }
    if args.check:
        target = RUNNER.area_path(args.out, "analyses")
        if target.exists():
            raise FileExistsError("New recovery-check output required")
        RUNNER.write(target, {"status": "PASS", **provenance})
        print(json.dumps({"status": "PASS", **provenance}), flush=True)
        return

    out = RUNNER.area_path(args.out, "runs")
    original_write = RUNNER.write

    def write_with_provenance(path, value):
        if Path(path) == out / "manifest.json":
            snapshot = out / "source" / ENTRY.relative_to(ROOT)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ENTRY, snapshot)
            value = {**value, "recovery_compatibility": provenance}
        original_write(path, value)

    RUNNER.write = write_with_provenance
    RUNNER.run(out, qualification, parent)


if __name__ == "__main__":
    main()
