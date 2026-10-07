"""CPU-audited continuation entrypoint for the frozen RRC-v0 run.

Plan-only mode validates the original qualification, source snapshot, banks,
schedule, committed dense stages, and checkpoint payloads without initializing
CUDA. Execution mode builds a new child run and continues only from the latest
fully recorded stage for each unfinished arm.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
PROTOCOL = "rrc_v0_prestream_relation_factorization_v1"
ARMS = ("current", "factorized", "rrc")
BLOCKS, UPDATES = 8, 300
CHECKPOINTS = tuple(range(0, UPDATES + 1, 25))
INIT_SEEDS = tuple(range(120001, 120009))
SCHEDULE_SEEDS = tuple(range(121001, 121009))
DEFAULT_PARENT = "runs/rrc_v0_20261007_01"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen runtime module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Importing the frozen runner only defines its model/runtime functions. This
# recovery entrypoint never calls its run() function, which refuses existing
# run directories and always starts from update zero.
FROZEN = _load("_rrc_recovery_frozen_run", ROOT / "new/rrc_v0/run.py")
C, R, E, M = FROZEN.C, FROZEN.R, FROZEN.E, FROZEN.M
D, P = FROZEN.D, FROZEN.P


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _inside(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _project_path(value: str | Path, area: str) -> Path:
    path = (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    base = (ROOT / area).resolve()
    if path == base or not _inside(path, base):
        raise ValueError(f"Path must be a child of {area}: {value}")
    return path


def _relative_name(name: str) -> PurePosixPath:
    value = PurePosixPath(name)
    if value.is_absolute() or not value.parts or any(part in ("", ".", "..") for part in value.parts):
        raise ValueError(f"Unsafe project-relative path: {name!r}")
    if "\\" in name:
        raise ValueError(f"Expected slash-separated project-relative path: {name!r}")
    return value


def _safe_child(base: Path, relative: str) -> Path:
    parts = _relative_name(relative)
    result = base.joinpath(*parts.parts)
    if not _inside(result, base):
        raise ValueError(f"Path escapes run directory: {relative!r}")
    return result


def _load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _check_payload(row: dict, checkpoint: Path, block_plan: dict) -> dict:
    payload = _load_checkpoint(checkpoint)
    expected = {
        "protocol": PROTOCOL,
        "arm": row["arm"],
        "initialization_seed": block_plan["initialization_seed"],
        "schedule_seed": block_plan["schedule_seed"],
        "completed_updates": row["update"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Checkpoint metadata mismatch for {key}: {checkpoint}")
    if C.tensor_hash(payload["state_dict"]) != row["parameter_sha256"]:
        raise ValueError(f"Checkpoint parameter hash mismatch: {checkpoint}")

    model = FROZEN.model_for(row["arm"], block_plan["initialization_seed"], "cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer = C.optimizer_for(model)
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state_dict"]))
    if C.tensor_hash(model.state_dict()) != row["parameter_sha256"]:
        raise ValueError(f"Restored model differs from committed parameter hash: {checkpoint}")
    states = list(optimizer.state.values())
    if row["update"] == 0:
        if states:
            raise ValueError(f"u000 checkpoint unexpectedly has Adam state: {checkpoint}")
    else:
        if len(states) != len(list(model.parameters())):
            raise ValueError(f"Adam state does not cover all parameters: {checkpoint}")
        if any(int(state["step"].item()) != row["update"] for state in states):
            raise ValueError(f"Adam update count differs from checkpoint: {checkpoint}")
    group = optimizer.param_groups[0]
    for key, value in C.OPTIMIZER.items():
        if group[key] != value:
            raise ValueError(f"AdamW setting {key} differs from frozen protocol")
    return payload


def _bank_tensor_hash(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        tensors = {key: torch.from_numpy(np.ascontiguousarray(data[key])) for key in data.files}
    return C.tensor_hash(tensors)


def _verify_source_bundle(parent: Path, manifest: dict, qualification_arg: str | None):
    expected_source = manifest.get("source_sha256")
    if not isinstance(expected_source, dict) or not expected_source:
        raise ValueError("Parent manifest has no frozen source_sha256 map")
    current = FROZEN.hashes()
    if current != expected_source:
        changed = sorted(set(current) | set(expected_source))
        changed = [name for name in changed if current.get(name) != expected_source.get(name)]
        raise ValueError("Current qualified source differs: " + ", ".join(changed[:12]))

    snapshot = parent / "source"
    expected_names = set(expected_source)
    actual_names = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file()}
    if actual_names != expected_names:
        missing, extra = sorted(expected_names - actual_names), sorted(actual_names - expected_names)
        raise ValueError(f"Parent source snapshot file set differs (missing={missing[:5]}, extra={extra[:5]})")
    for name, expected_hash in expected_source.items():
        path = _safe_child(snapshot, name)
        if _sha(path) != expected_hash:
            raise ValueError(f"Parent source snapshot hash mismatch: {name}")

    if qualification_arg:
        qualification = _project_path(qualification_arg, "analyses")
        candidates = [qualification]
    else:
        candidates = list((ROOT / "analyses").glob("rrc_v0_qualification_*.json"))
        candidates = [path for path in candidates if _sha(path) == manifest.get("qualification_sha256")]
    matches = [p for p in candidates if p.is_file() and _sha(p) == manifest.get("qualification_sha256")]
    if len(matches) != 1:
        raise ValueError("Could not identify exactly one unchanged original RRC-v0 qualification")
    qualification = matches[0]
    qualification_record = _read_json(qualification)
    if qualification_record.get("status") != "PASS" or qualification_record.get("protocol") != PROTOCOL:
        raise ValueError("The parent qualification is not a PASS for the frozen RRC-v0 protocol")
    if qualification_record.get("source_sha256") != expected_source:
        raise ValueError("The original qualification is not bound to the parent's source hashes")
    return qualification, expected_source


def _verify_protocol(parent: Path, manifest: dict):
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Parent run protocol does not match RRC-v0")
    if manifest.get("expected_arms") != BLOCKS * len(ARMS):
        raise ValueError("Parent manifest expected-arm count differs from the frozen design")
    if manifest.get("expected_dense_records") != BLOCKS * len(ARMS) * len(CHECKPOINTS):
        raise ValueError("Parent manifest expected-stage count differs from the frozen design")
    config = _read_json(parent / "config.json")
    expected_config = {
        "protocol": FROZEN.PROTOCOL,
        "arms": list(FROZEN.ARMS),
        "blocks": FROZEN.BLOCKS,
        "super_updates": FROZEN.UPDATES,
        "parameter_counts": dict(FROZEN.COUNTS),
        "carrier_channels": 24,
        "latent_channels": 8,
        "initialization_seeds": list(FROZEN.INIT_SEEDS),
        "schedule_seeds": list(FROZEN.SCHEDULE_SEEDS),
        "initial_relation": {"rho": 0.1, "pi": [0.25] * 4},
        "content_spatial_feature": "prestream_L(C_t)",
        "primary_contrast": ["rrc", "factorized"],
        "mode": FROZEN.MODE,
        "batch": 8,
        "forward_steps_per_update": 256,
        "loss_windows": 32,
        "credit_horizon": 8,
        "optimizer_steps_per_update": 1,
        "parameters_fixed_within_super_update": True,
        "train_seed": 10002,
        "eval_seeds": {str(size): seed for size, seed in FROZEN.EVAL_SEEDS.items()},
        "checkpoints": list(FROZEN.CHECKPOINTS),
        "formal_checkpoint": UPDATES,
        "optimizer": {**C.OPTIMIZER, "betas": list(C.OPTIMIZER["betas"])},
        "clip": 1.0,
        "runtime_limit_enforced": False,
    }
    if config != expected_config:
        mismatched = [key for key in sorted(set(config) | set(expected_config))
                      if config.get(key) != expected_config.get(key)]
        raise ValueError(f"Full frozen protocol config mismatch: {mismatched}")
    plans = _read_json(parent / "plans.json")
    if set(plans) != {str(i) for i in range(BLOCKS)}:
        raise ValueError("The parent schedule must contain exactly eight blocks")
    for block in range(BLOCKS):
        item = plans[str(block)]
        if item.get("initialization_seed") != INIT_SEEDS[block] or item.get("schedule_seed") != SCHEDULE_SEEDS[block]:
            raise ValueError(f"Frozen seed mismatch in block {block}")
        expected_order = list(ARMS[block % 3 :] + ARMS[: block % 3])
        if item.get("arm_order") != expected_order:
            raise ValueError(f"Frozen arm order mismatch in block {block}")
        batches = item.get("batch_indices")
        if not isinstance(batches, list) or len(batches) != UPDATES:
            raise ValueError(f"Block {block} does not have 300 fixed minibatches")
        if any(not isinstance(batch, list) or len(batch) != 8 or
               any(not isinstance(i, int) or i < 0 or i >= 512 for i in batch)
               for batch in batches):
            raise ValueError(f"Block {block} has invalid fixed minibatch indices")
    if C.sha(parent / "plans.json") != manifest.get("schedule_plan_sha256"):
        raise ValueError("Parent plan hash differs from its launch manifest")
    data_hashes = manifest.get("data_sha256")
    if not isinstance(data_hashes, dict):
        raise ValueError("Parent manifest has no bank hashes")
    if not isinstance(manifest.get("gpu"), str) or not manifest["gpu"]:
        raise ValueError("Parent manifest has no recorded GPU model name")
    for label, key in (("train", "train"), ("evaluation32", "evaluation32"), ("evaluation64", "evaluation64")):
        path = parent / "banks" / f"{label}.npz"
        if not path.is_file() or _bank_tensor_hash(path) != data_hashes.get(key):
            raise ValueError(f"Parent {label} bank is missing or has a different tensor hash")
    return config, plans, data_hashes


def _verify_stage_files(parent: Path, row: dict) -> tuple[Path, Path]:
    block, arm, update = row["block"], row["arm"], row["update"]
    expected_checkpoint = f"block{block:02d}/{arm}/checkpoints/u{update:03d}.pt"
    expected_stage = f"block{block:02d}/{arm}/evaluation/u{update:03d}"
    if row.get("checkpoint") != expected_checkpoint:
        raise ValueError(f"Unexpected checkpoint reference in dense row: {row}")
    if row.get("evaluation_summary") != f"{expected_stage}/summary.json":
        raise ValueError(f"Unexpected evaluation summary reference in dense row: {row}")
    if row.get("relation_activity") != f"{expected_stage}/relation_activity.json":
        raise ValueError(f"Unexpected relation activity reference in dense row: {row}")
    checkpoint = _safe_child(parent, row["checkpoint"])
    if not checkpoint.is_file() or _sha(checkpoint) != row.get("checkpoint_sha256"):
        raise ValueError(f"Missing or changed committed checkpoint: {expected_checkpoint}")
    stage = _safe_child(parent, expected_stage)
    required = {"summary.json", "relation_activity.json", "size32_traces.npz", "size64_traces.npz"}
    if update == UPDATES:
        required.add("matched_frontier_strata.csv")
    for name in required:
        path = stage / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Committed evaluation stage is incomplete: {expected_stage}/{name}")
        if name.endswith(".npz"):
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"Corrupt trace archive: {expected_stage}/{name}")
    summary = _read_json(stage / "summary.json")
    activity = _read_json(stage / "relation_activity.json")
    if not isinstance(summary, dict) or not isinstance(activity.get("rows"), list):
        raise ValueError(f"Malformed committed evaluation records: {expected_stage}")
    if M.compact_pair_metrics(summary) != row.get("metrics") or M.joint_readiness(summary) != row.get("joint"):
        raise ValueError(f"Dense row does not match its saved evaluation summary: {expected_stage}")
    return checkpoint, stage


def _validate_latest_checkpoint(rows_by_arm: dict, plans: dict):
    latest = {}
    for block in range(BLOCKS):
        for arm in ARMS:
            key = (block, arm)
            rows = rows_by_arm.get(key, [])
            if not rows:
                latest[key] = None
                continue
            row = rows[-1]
            payload = _check_payload(row, _safe_child(_current_parent, row["checkpoint"]), plans[str(block)])
            latest[key] = {"update": row["update"], "checkpoint": row["checkpoint"],
                           "parameter_sha256": row["parameter_sha256"]}
            del payload
    return latest


_current_parent: Path


def _tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha(path)))
    return digest.hexdigest()


def build_resume_plan(parent_value: str | Path, out_value: str | Path,
                      qualification_value: str | None = None) -> dict:
    """Validate the immutable parent and return a CPU-only continuation plan."""
    global _current_parent
    parent = _project_path(parent_value, "runs")
    out = _project_path(out_value, "runs")
    if not parent.is_dir():
        raise FileNotFoundError(f"Parent run does not exist: {parent}")
    if parent == out or _inside(out, parent) or _inside(parent, out):
        raise ValueError("Parent and child run directories must be distinct and non-nested")
    if out.exists():
        raise FileExistsError(f"Child output must be a new unique run directory: {out}")
    _current_parent = parent

    manifest_path = parent / "manifest.json"
    manifest = _read_json(manifest_path)
    qualification, old_source = _verify_source_bundle(parent, manifest, qualification_value)
    config, plans, data_hashes = _verify_protocol(parent, manifest)

    dense = _read_json(parent / "dense.json")
    if not isinstance(dense, list):
        raise ValueError("Parent dense.json is not a list")
    rows_by_arm: dict[tuple[int, str], list[dict]] = {}
    seen = set()
    for row in dense:
        if not isinstance(row, dict):
            raise ValueError("Parent dense.json contains a non-record entry")
        block, arm, update = row.get("block"), row.get("arm"), row.get("update")
        if block not in range(BLOCKS) or arm not in ARMS or update not in CHECKPOINTS:
            raise ValueError(f"Parent dense record is outside the frozen design: {row}")
        key = (block, arm, update)
        if key in seen:
            raise ValueError(f"Parent dense.json contains duplicate stage {key}")
        seen.add(key)
        plan = plans[str(block)]
        if row.get("initialization_seed") != plan["initialization_seed"] or row.get("schedule_seed") != plan["schedule_seed"]:
            raise ValueError(f"Dense record seeds differ from schedule plan: {key}")
        if row.get("formal_endpoint") is not (update == UPDATES):
            raise ValueError(f"Dense record endpoint flag is invalid: {key}")
        _verify_stage_files(parent, row)
        rows_by_arm.setdefault((block, arm), []).append(row)
    if len(dense) != 137:
        raise ValueError(f"Expected 137 committed parent stages, found {len(dense)}")
    for key, rows in rows_by_arm.items():
        rows.sort(key=lambda row: row["update"])
        actual = [row["update"] for row in rows]
        if actual != list(CHECKPOINTS[: len(actual)]):
            raise ValueError(f"Committed stages are not a prefix for {key}: {actual}")

    grouped_latest = _validate_latest_checkpoint(rows_by_arm, plans)
    completed, partial, unstarted = [], [], []
    remaining_stage_records = 0
    for block in range(BLOCKS):
        for arm in ARMS:
            item = grouped_latest[(block, arm)]
            label = f"block{block:02d}/{arm}"
            if item is None:
                unstarted.append(label)
                remaining_stage_records += len(CHECKPOINTS)
            elif item["update"] == UPDATES:
                completed.append(label)
            else:
                partial.append({"arm": label, **item, "next_update": item["update"] + 1,
                                "first_batch_indices": plans[str(block)]["batch_indices"][item["update"]]})
                remaining_stage_records += sum(update > item["update"] for update in CHECKPOINTS)
    if len(completed) != 10 or len(partial) != 1 or len(unstarted) != 13:
        raise ValueError("Parent committed-stage layout differs from the expected 10/1/13 recovery plan")
    if partial[0]["arm"] != "block03/factorized" or partial[0]["update"] != 150:
        raise ValueError("Expected to resume block03/factorized from the fully recorded u150 stage")
    if remaining_stage_records != 175:
        raise ValueError(f"Expected 175 remaining checkpoint stages, found {remaining_stage_records}")

    qualification_rel = qualification.relative_to(ROOT).as_posix()
    wrapper_paths = (ROOT / "new/rrc_recovery/run.py", ROOT / "new/rrc_recovery/check_recovery.py")
    wrapper_hashes = {path.relative_to(ROOT).as_posix(): _sha(path) for path in wrapper_paths}
    readme_path = ROOT / "new/rrc_recovery/README.md"
    return {
        "parent": parent.relative_to(ROOT).as_posix(),
        "out": out.relative_to(ROOT).as_posix(),
        "protocol": PROTOCOL,
        "qualification": qualification_rel,
        "qualification_sha256": manifest["qualification_sha256"],
        "parent_manifest_sha256": _sha(manifest_path),
        "parent_dense_sha256": _sha(parent / "dense.json"),
        "parent_source_sha256": old_source,
        "wrapper_execution_sha256": wrapper_hashes,
        "wrapper_readme_sha256": _sha(readme_path),
        "parent_data_sha256": data_hashes,
        "parent_schedule_plan_sha256": manifest["schedule_plan_sha256"],
        "parent_gpu_model": manifest["gpu"],
        "config": config,
        "plans": plans,
        "dense_records": dense,
        "rows_by_arm": rows_by_arm,
        "latest": grouped_latest,
        "completed_arms": completed,
        "partial_arms": partial,
        "unstarted_arms": unstarted,
        "arms_to_run": len(partial) + len(unstarted),
        "inherited_stage_records": len(dense),
        "remaining_stage_records": remaining_stage_records,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "parent_read_only": True,
    }


def public_plan(plan: dict) -> dict:
    keys = ("parent", "out", "protocol", "qualification", "qualification_sha256",
            "parent_manifest_sha256", "parent_dense_sha256", "parent_data_sha256",
            "parent_schedule_plan_sha256", "parent_gpu_model", "completed_arms", "partial_arms",
            "unstarted_arms", "arms_to_run", "inherited_stage_records",
            "remaining_stage_records", "cuda_initialized", "parent_read_only")
    return {key: plan[key] for key in keys}


def _copy_parent_committed(plan: dict, parent: Path, out: Path, runtime: dict) -> None:
    out.mkdir(parents=True, exist_ok=False)
    for name in ("config.json", "plans.json"):
        shutil.copyfile(parent / name, out / name)
    shutil.copytree(parent / "banks", out / "banks")
    shutil.copytree(parent / "source", out / "source")

    for relative, expected_hash in plan["wrapper_execution_sha256"].items():
        source = _safe_child(ROOT, relative)
        if _sha(source) != expected_hash:
            raise ValueError(f"Recovery wrapper changed during setup: {relative}")
        target = _safe_child(out / "source", relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    readme_relative = "new/rrc_recovery/README.md"
    readme_source = ROOT / readme_relative
    if _sha(readme_source) != plan["wrapper_readme_sha256"]:
        raise ValueError("Recovery README changed during setup")
    readme_target = _safe_child(out / "source", readme_relative)
    readme_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(readme_source, readme_target)

    copied_evaluations = set()
    for row in plan["dense_records"]:
        cp_rel = row["checkpoint"]
        cp_src = _safe_child(parent, cp_rel)
        cp_dst = _safe_child(out, cp_rel)
        cp_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cp_src, cp_dst)
        stage_rel = f"block{row['block']:02d}/{row['arm']}/evaluation/u{row['update']:03d}"
        if stage_rel not in copied_evaluations:
            stage_src, stage_dst = _safe_child(parent, stage_rel), _safe_child(out, stage_rel)
            shutil.copytree(stage_src, stage_dst)
            copied_evaluations.add(stage_rel)
        marker = cp_dst.with_name(cp_dst.name + ".complete.json")
        _write_json_atomic(marker, {
            "stage_complete": True,
            "stage_origin": "parent_dense_record",
            "record_sha256": _sha_json(row),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "evaluation_tree_sha256": _tree_hash(_safe_child(out, stage_rel)),
            "completed_update": row["update"],
        })

    for (block, arm), rows in plan["rows_by_arm"].items():
        max_update = rows[-1]["update"]
        source_curve = _safe_child(parent, f"block{block:02d}/{arm}/training_curve.json")
        if not source_curve.is_file():
            raise ValueError(f"Missing training curve for committed arm block{block:02d}/{arm}")
        curve = _read_json(source_curve)
        if not isinstance(curve, list):
            raise ValueError(f"Malformed training curve for block{block:02d}/{arm}")
        trimmed = [row for row in curve if isinstance(row, dict) and row.get("update", 0) <= max_update]
        updates = [row.get("update") for row in trimmed]
        if updates != list(range(1, max_update + 1)):
            raise ValueError(f"Training curve is not complete through committed u{max_update}: block{block:02d}/{arm}")
        _write_json_atomic(_safe_child(out, f"block{block:02d}/{arm}/training_curve.json"), trimmed)

    final = [row for row in plan["dense_records"] if row["update"] == UPDATES]
    _write_json_atomic(out / "dense.json", plan["dense_records"])
    _write_json_atomic(out / "perarm.json", final)
    _write_json_atomic(out / "resume_plan.json", public_plan(plan))
    manifest = {
        "protocol": PROTOCOL,
        "status": "RESUME_READY",
        "parent_run": plan["parent"],
        "parent_manifest_sha256": plan["parent_manifest_sha256"],
        "parent_dense_sha256": plan["parent_dense_sha256"],
        "parent_qualification": plan["qualification"],
        "qualification_sha256": plan["qualification_sha256"],
        "old_source_sha256": plan["parent_source_sha256"],
        "wrapper_execution_sha256": plan["wrapper_execution_sha256"],
        "wrapper_readme_sha256": plan["wrapper_readme_sha256"],
        "data_sha256": plan["parent_data_sha256"],
        "schedule_plan_sha256": plan["parent_schedule_plan_sha256"],
        "parent_gpu_model": plan["parent_gpu_model"],
        "runtime": runtime,
        "gpu_comparison_scope": "GPU model name only; device identity is not compared",
        "expected_arms": BLOCKS * len(ARMS),
        "expected_dense_records": BLOCKS * len(ARMS) * len(CHECKPOINTS),
        "inherited_dense_records": plan["inherited_stage_records"],
        "inherited_completed_arms": len(plan["completed_arms"]),
        "arms_to_run": plan["arms_to_run"],
        "remaining_stage_records": plan["remaining_stage_records"],
        "runtime_limit_enforced": False,
        "watchdog_enabled": False,
        "continuous_monitoring": False,
        "checkpoint_resume": "model_state_dict_and_AdamW_state_dict",
        "bitwise_uninterrupted_equivalence_guaranteed": False,
        "rng_state_saved_in_original_checkpoint": False,
    }
    _write_json_atomic(out / "manifest.json", manifest)
    _write_json_atomic(out / "status.json", {
        "protocol": PROTOCOL, "status": "RUNNING", "phase": "resume_setup",
        "inherited_dense_records": len(plan["dense_records"]),
        "completed_arms": len(final), "expected_arms": BLOCKS * len(ARMS),
        "updated_utc": C.now(),
    })
    P.report(out, final, plan["dense_records"], status={"status": "RUNNING", "phase": "resume_setup"})


def _optimizer_step_count(optimizer) -> int:
    steps = [int(state["step"].item()) for state in optimizer.state.values()]
    if not steps:
        return 0
    if len(set(steps)) != 1:
        raise ValueError("AdamW parameter step counters diverged")
    return steps[0]


def _same_tree(a, b) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, type(a)):
        return len(a) == len(b) and all(_same_tree(x, y) for x, y in zip(a, b))
    return a == b


def _load_training_banks(run_dir: Path):
    def load(label):
        with np.load(run_dir / "banks" / f"{label}.npz", allow_pickle=False) as data:
            return {key: torch.from_numpy(np.array(data[key], copy=True)) for key in data.files}
    train_cpu = load("train")
    eval_banks = {32: load("evaluation32"), 64: load("evaluation64")}
    return train_cpu, eval_banks


def _record_stage(out: Path, block: int, arm: str, update: int, model, optimizer,
                  plan: dict, curve: list, evaluation_seconds: float, graph,
                  records: list, final: list, started: float,
                  capture_setup_scope: str = "full_child_arm_session") -> float:
    folder = out / f"block{block:02d}" / arm
    checkpoint = folder / "checkpoints" / f"u{update:03d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_tmp = checkpoint.with_name(checkpoint.name + ".tmp")
    payload = C.payload(model, optimizer, plan["initialization_seed"], update)
    payload.update(protocol=PROTOCOL, arm=arm, schedule_seed=plan["schedule_seed"])
    torch.save(payload, checkpoint_tmp)
    os.replace(checkpoint_tmp, checkpoint)
    parameter_hash = C.tensor_hash(model.state_dict())

    evaluation = folder / "evaluation" / f"u{update:03d}"
    evaluation_tmp = folder / "evaluation" / f".u{update:03d}.inprogress"
    evaluation.parent.mkdir(parents=True, exist_ok=True)
    if evaluation.exists() or evaluation_tmp.exists():
        raise FileExistsError(f"Output stage already exists: {evaluation}")
    tick = time.monotonic()
    measured = FROZEN.observed_evaluation(model, _evaluation_banks, evaluation_tmp, full=update == UPDATES)
    elapsed = time.monotonic() - tick
    evaluation_seconds += elapsed
    os.replace(evaluation_tmp, evaluation)
    row = {
        "block": block, "arm": arm, "update": update,
        "initialization_seed": plan["initialization_seed"], "schedule_seed": plan["schedule_seed"],
        "checkpoint": checkpoint.relative_to(out).as_posix(),
        "checkpoint_sha256": _sha(checkpoint), "parameter_sha256": parameter_hash,
        "evaluation_seconds": elapsed,
        "evaluation_summary": (evaluation / "summary.json").relative_to(out).as_posix(),
        "relation_activity": (evaluation / "relation_activity.json").relative_to(out).as_posix(),
        "metrics": M.compact_pair_metrics(measured), "joint": M.joint_readiness(measured),
        "formal_endpoint": update == UPDATES, "checkpoint_selection": False,
    }
    if update == UPDATES:
        row.update(last_loss=curve[-1]["mean_super_update_loss"],
                   training_seconds=sum(item["seconds"] for item in curve),
                   all_checkpoint_evaluation_seconds=evaluation_seconds,
                   capture_setup_seconds=graph.setup_seconds,
                   capture_setup_scope=capture_setup_scope)
        final.append(row)
    records.append(row)
    records.sort(key=lambda item: (item["block"], item["arm"], item["update"]))
    final.sort(key=lambda item: (item["block"], item["arm"]))
    _write_json_atomic(out / "dense.json", records)
    _write_json_atomic(folder / "training_curve.json", curve)
    _write_json_atomic(out / "perarm.json", final)
    P.report(out, final, records, status={"status": "RUNNING", "phase": "stage_evaluation"})
    _write_json_atomic(out / "status.json", {
        "protocol": PROTOCOL, "status": "RUNNING", "phase": "stage_committing",
        "block": block, "arm": arm, "completed_updates": update,
        "completed_arms": len(final), "expected_arms": BLOCKS * len(ARMS),
        "completed_dense_records": len(records), "expected_dense_records": BLOCKS * len(ARMS) * len(CHECKPOINTS),
        "elapsed_seconds": time.monotonic() - started, "updated_utc": C.now(),
    })
    marker = checkpoint.with_name(checkpoint.name + ".complete.json")
    _write_json_atomic(marker, {
        "stage_complete": True, "stage_origin": "recovery_execution",
        "record_sha256": _sha_json(row), "checkpoint_sha256": row["checkpoint_sha256"],
        "evaluation_tree_sha256": _tree_hash(evaluation), "completed_update": update,
    })
    print(json.dumps({"phase": "stage_complete", "block": block, "arm": arm,
                      "completed_updates": update, "dense_records": len(records)}), flush=True)
    model.train()
    return evaluation_seconds


_evaluation_banks = None


def execute_resume(plan: dict) -> dict:
    """Create a new output run and continue the frozen training protocol."""
    global _evaluation_banks
    parent, out = ROOT / plan["parent"], ROOT / plan["out"]
    C.setup_backend()
    gpu_model = torch.cuda.get_device_name()
    if gpu_model != plan["parent_gpu_model"]:
        raise ValueError(f"GPU model mismatch: parent={plan['parent_gpu_model']!r}, runtime={gpu_model!r}")
    runtime = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "cuda": str(torch.version.cuda),
        "gpu_model": gpu_model,
    }
    _copy_parent_committed(plan, parent, out, runtime)
    source_before = FROZEN.hashes()
    plan_hash = C.sha(out / "plans.json")
    train_cpu, banks = _load_training_banks(out)
    train = {key: value.cuda() for key, value in train_cpu.items()}
    _evaluation_banks = {size: {key: value.cuda() for key, value in data.items()}
                         for size, data in banks.items()}
    records = list(plan["dense_records"])
    final = [row for row in records if row["update"] == UPDATES]
    all_rows = plan["rows_by_arm"]
    eval_elapsed = {key: sum(row["evaluation_seconds"] for row in rows)
                    for key, rows in all_rows.items()}
    started = time.monotonic()
    last_detail = {"phase": "starting"}
    try:
        for block in range(BLOCKS):
            block_plan = plan["plans"][str(block)]
            for arm in block_plan["arm_order"]:
                key = (block, arm)
                latest = plan["latest"][key]
                if latest is not None and latest["update"] == UPDATES:
                    continue
                if latest is None:
                    capture_setup_scope = "full_child_arm_session"
                    start_update = 1
                    model = FROZEN.model_for(arm, block_plan["initialization_seed"], "cuda")
                    if C.tensor_hash(model.state_dict()) != block_plan["initial_parameter_sha256"][arm]:
                        raise ValueError(f"Fresh initialization differs from frozen plan: block{block:02d}/{arm}")
                    optimizer = C.optimizer_for(model)
                    curve = []
                    # The zero-update checkpoint and evaluation are the first
                    # committed child stages for an unstarted arm.
                    eval_elapsed[key] = _record_stage(out, block, arm, 0, model, optimizer, block_plan, curve,
                                                      eval_elapsed.get(key, 0.0), None, records, final, started)
                else:
                    capture_setup_scope = "resumed_session_only"
                    start_update = latest["update"] + 1
                    checkpoint = _safe_child(out, latest["checkpoint"])
                    payload = _load_checkpoint(checkpoint)
                    model = FROZEN.model_for(arm, block_plan["initialization_seed"], "cuda")
                    model.load_state_dict(payload["state_dict"], strict=True)
                    optimizer = C.optimizer_for(model)
                    optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state_dict"]))
                    if C.tensor_hash(model.state_dict()) != latest["parameter_sha256"]:
                        raise ValueError(f"Restored model state mismatch: block{block:02d}/{arm}")
                    if _optimizer_step_count(optimizer) != latest["update"]:
                        raise ValueError(f"Restored AdamW step mismatch: block{block:02d}/{arm}")
                    curve_path = _safe_child(out, f"block{block:02d}/{arm}/training_curve.json")
                    curve = _read_json(curve_path)
                    if len(curve) != latest["update"]:
                        raise ValueError(f"Restored curve is not trimmed to committed update: block{block:02d}/{arm}")

                model.train()
                activity = D.TrainingActivity(model)
                model.relation_observer = activity
                batches = block_plan["batch_indices"]
                first = C.subset(train, batches[start_update - 1])
                model_before_capture = C.tensor_hash(model.state_dict())
                optimizer_before_capture = C.cpu_tree(optimizer.state_dict())
                graph = R.CapturedSuperK8(model, first, FROZEN.MODE, FROZEN.LOSS_FN)
                if C.tensor_hash(model.state_dict()) != model_before_capture:
                    raise ValueError("CUDA Graph setup changed restored model parameters")
                if not _same_tree(optimizer_before_capture, C.cpu_tree(optimizer.state_dict())):
                    raise ValueError("CUDA Graph setup changed restored AdamW state")
                for update in range(start_update, UPDATES + 1):
                    tick = time.monotonic()
                    data = first if update == start_update else C.subset(train, batches[update - 1])
                    activity.reset()
                    loss, state, finite = graph.run(data)
                    relation_grad = D.gradient_norm(model)
                    norm = R.finish_update(model, optimizer, finite)
                    torch.cuda.synchronize()
                    curve.append({
                        "update": update, "mean_super_update_loss": float(loss),
                        "gradient_norm_before_clip": float(norm), "seconds": time.monotonic() - tick,
                        "cold_initializations": 4, **FROZEN.CV.CADENCE,
                        "relation_gradient_norm_before_clip": relation_grad,
                        "relation_parameters_after_update": D.parameters(model),
                        "relation_activity": activity.result(),
                    })
                    if _optimizer_step_count(optimizer) != update:
                        raise ValueError(f"AdamW step count differs after update {update}")
                    last_detail = {"phase": "training", "block": block, "arm": arm,
                                   "completed_updates": update, "loss": float(loss)}
                    if update == start_update or update % 25 == 0:
                        print(json.dumps(last_detail), flush=True)
                    if update in CHECKPOINTS:
                        eval_elapsed[key] = _record_stage(
                            out, block, arm, update, model, optimizer, block_plan,
                            curve, eval_elapsed.get(key, 0.0), graph, records, final, started,
                            capture_setup_scope=capture_setup_scope,
                        )
                    if update != start_update:
                        del data
                model.relation_observer = None
                del model, optimizer, graph, state, first, curve, activity
                import gc
                gc.collect()
                torch.cuda.empty_cache()

        if len(final) != BLOCKS * len(ARMS) or len(records) != BLOCKS * len(ARMS) * len(CHECKPOINTS):
            raise ValueError(f"Final record counts are incomplete: {len(final)} arms, {len(records)} stages")
        if FROZEN.hashes() != source_before:
            raise ValueError("Qualified source changed during continuation")
        if C.sha(out / "plans.json") != plan_hash:
            raise ValueError("Copied fixed schedule changed during continuation")
        result = P.aggregate(final, records, expected_blocks=BLOCKS)
        if result["status"] != "COMPLETE":
            raise ValueError("RRC aggregate did not reach its complete record state")
        result.update(protocol=PROTOCOL, status="COMPLETE", completed_arms=len(final),
                      expected_arms=BLOCKS * len(ARMS), dense_records=len(records),
                      inherited_dense_records=plan["inherited_stage_records"],
                      resumed_from_parent=plan["parent"])
        result.update(finished_utc=C.now(), elapsed_seconds=time.monotonic() - started,
                      runtime_limit_enforced=False)
        _write_json_atomic(out / "summary.json", result)
        _write_json_atomic(out / "status.json", {**result, "phase": "complete"})
        P.report(out, final, records, status=result)
        print(json.dumps(result), flush=True)
        return result
    except Exception as error:
        failure = {"protocol": PROTOCOL, "status": "ERROR", "phase": last_detail.get("phase"),
                   "completed_arms": len(final), "expected_arms": BLOCKS * len(ARMS),
                   "dense_records": len(records), "expected_dense_records": BLOCKS * len(ARMS) * len(CHECKPOINTS),
                   "error": repr(error), "traceback": traceback.format_exc(),
                   "overall_verdict": "INCOMPLETE", "finished_utc": C.now(),
                   "elapsed_seconds": time.monotonic() - started, "runtime_limit_enforced": False}
        _write_json_atomic(out / "error.json", failure)
        _write_json_atomic(out / "summary.json", failure)
        _write_json_atomic(out / "status.json", failure)
        P.report(out, final, records, status=failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default=DEFAULT_PARENT)
    parser.add_argument("--out", required=True, help="New, unique child run directory under runs/")
    parser.add_argument("--qualification", help="Optional explicit original qualification JSON under analyses/")
    parser.add_argument("--plan-only", action="store_true", help="Run the CPU-only recovery audit and print the plan")
    args = parser.parse_args()
    plan = build_resume_plan(args.parent, args.out, args.qualification)
    if args.plan_only:
        print(json.dumps(public_plan(plan), indent=2, allow_nan=False))
        return
    if plan["cuda_initialized"]:
        raise RuntimeError("CUDA was initialized before execution setup")
    execute_resume(plan)


if __name__ == "__main__":
    main()
