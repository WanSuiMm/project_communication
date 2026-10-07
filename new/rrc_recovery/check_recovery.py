"""CPU-only recovery audit and model/AdamW restore parity check."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run as RECOVERY  # noqa: E402


def _assert_same_tree(left, right, path="root"):
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise AssertionError(f"Tensor differs at {path}")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"Mapping keys differ at {path}")
        for key in left:
            _assert_same_tree(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        if len(left) != len(right):
            raise AssertionError(f"Sequence lengths differ at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_same_tree(a, b, f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"Values differ at {path}: {left!r} != {right!r}")


def _tiny_batch(seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand((1, 3, 4, 4), generator=generator, dtype=torch.float32)
    mask = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    x[:, :1] = mask
    y = torch.zeros_like(mask)
    y[:, :, ::2, ::2] = 1.0
    return {"x": x, "y": y, "mask": mask}


def _one_super_update(model, optimizer, data, expected_step):
    model.zero_grad(set_to_none=True)
    loss, state, finite = RECOVERY.R.math_backward(
        model, data, RECOVERY.FROZEN.MODE, RECOVERY.FROZEN.LOSS_FN
    )
    norm = RECOVERY.R.finish_update(model, optimizer, finite)
    if any(parameter.grad is None for parameter in model.parameters()):
        raise AssertionError("At least one parameter received no gradient")
    if RECOVERY._optimizer_step_count(optimizer) != expected_step:
        raise AssertionError(f"Tiny update did not reach AdamW step {expected_step}")
    return float(loss), float(norm), state


def tiny_restore_parity() -> dict:
    """Save after one tiny CPU update, restore, then compare the next update."""
    torch.set_num_threads(1)
    batch1, batch2 = _tiny_batch(17001), _tiny_batch(17002)
    rows = []
    for index, arm in enumerate(RECOVERY.ARMS):
        seed = 17010 + index
        original = RECOVERY.FROZEN.model_for(arm, seed, "cpu")
        original_optimizer = RECOVERY.C.optimizer_for(original)
        first_loss, first_norm, _ = _one_super_update(original, original_optimizer, batch1, 1)
        payload = RECOVERY.C.payload(original, original_optimizer, seed, 1)
        payload.update(protocol=RECOVERY.PROTOCOL, arm=arm, schedule_seed=18000 + index)

        with tempfile.TemporaryDirectory(prefix="rrc-recovery-cpu-", dir=HERE) as temp:
            checkpoint = Path(temp) / "tiny.pt"
            torch.save(payload, checkpoint)
            saved = RECOVERY._load_checkpoint(checkpoint)
            restored = RECOVERY.FROZEN.model_for(arm, seed, "cpu")
            restored.load_state_dict(saved["state_dict"], strict=True)
            restored_optimizer = RECOVERY.C.optimizer_for(restored)
            restored_optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state_dict"]))
            _assert_same_tree(RECOVERY.C.cpu_tree(original.state_dict()),
                              RECOVERY.C.cpu_tree(restored.state_dict()), f"{arm}.checkpoint_model")
            _assert_same_tree(RECOVERY.C.cpu_tree(original_optimizer.state_dict()),
                              RECOVERY.C.cpu_tree(restored_optimizer.state_dict()), f"{arm}.checkpoint_adamw")

            next_loss_a, next_norm_a, state_a = _one_super_update(original, original_optimizer, batch2, 2)
            next_loss_b, next_norm_b, state_b = _one_super_update(restored, restored_optimizer, batch2, 2)
            _assert_same_tree(RECOVERY.C.cpu_tree(original.state_dict()),
                              RECOVERY.C.cpu_tree(restored.state_dict()), f"{arm}.next_model")
            _assert_same_tree(RECOVERY.C.cpu_tree(original_optimizer.state_dict()),
                              RECOVERY.C.cpu_tree(restored_optimizer.state_dict()), f"{arm}.next_adamw")
            _assert_same_tree(RECOVERY.C.cpu_tree(state_a), RECOVERY.C.cpu_tree(state_b), f"{arm}.next_state")
            if next_loss_a != next_loss_b or next_norm_a != next_norm_b:
                raise AssertionError(f"{arm} loss or clipping norm differs after restore")
        rows.append({
            "arm": arm,
            "status": "PASS",
            "tiny_spatial_shape": [4, 4],
            "super_update_forward_steps": 256,
            "credit_horizon": 8,
            "first_loss": first_loss,
            "first_gradient_norm": first_norm,
            "next_loss_exact": True,
            "next_adamw_and_model_state_exact": True,
        })
    return {"status": "PASS", "device": "cpu", "arms": rows}


def verify_plan_counts(plan: dict) -> None:
    if plan["inherited_stage_records"] != 137:
        raise AssertionError("Expected 137 inherited committed stages")
    if len(plan["completed_arms"]) != 10:
        raise AssertionError("Expected 10 complete u300 trajectories")
    if len(plan["partial_arms"]) != 1:
        raise AssertionError("Expected exactly one partially recorded trajectory")
    partial = plan["partial_arms"][0]
    if partial["arm"] != "block03/factorized" or partial["update"] != 150:
        raise AssertionError("The partial trajectory must resume from block03/factorized u150")
    if partial["next_update"] != 151:
        raise AssertionError("The next update after u150 must be u151")
    if partial["first_batch_indices"] != plan["plans"]["3"]["batch_indices"][150]:
        raise AssertionError("The resumed first minibatch must use plan[150]")
    if len(plan["unstarted_arms"]) != 13 or plan["arms_to_run"] != 14:
        raise AssertionError("Expected 13 unstarted trajectories plus one partial trajectory")
    if plan["remaining_stage_records"] != 175:
        raise AssertionError("Expected 175 future checkpoint/evaluation stages")
    if plan["cuda_initialized"]:
        raise AssertionError("CPU plan/check must not initialize CUDA")


def verify_print_flush_usage() -> dict:
    """Catch passing print-only flush arguments into json.dumps."""
    source_path = HERE / "run.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    bad_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "dumps" and any(keyword.arg == "flush" for keyword in node.keywords):
            bad_calls.append(node.lineno)
    if bad_calls:
        raise AssertionError(f"json.dumps has print-only flush keyword at lines {bad_calls}")
    return {"status": "PASS", "guard": "no_flush_keyword_on_json_dumps"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default=RECOVERY.DEFAULT_PARENT)
    parser.add_argument("--out", required=True, help="Unused unique child path checked by the plan audit")
    parser.add_argument("--qualification", help="Optional explicit original qualification JSON")
    parser.add_argument("--skip-tiny-parity", action="store_true")
    parser.add_argument("--report", help="Optional new JSON audit receipt path under analyses/")
    args = parser.parse_args()
    plan = RECOVERY.build_resume_plan(args.parent, args.out, args.qualification)
    verify_plan_counts(plan)
    print_guard = verify_print_flush_usage()
    parity = None if args.skip_tiny_parity else tiny_restore_parity()
    if parity is not None and parity["status"] != "PASS":
        raise SystemExit(1)
    report = {
        "status": "PASS",
        "plan": RECOVERY.public_plan(plan),
        "source_binding": {
            "original_qualification_sha256": plan["qualification_sha256"],
            "parent_manifest_sha256": plan["parent_manifest_sha256"],
            "parent_dense_sha256": plan["parent_dense_sha256"],
            "parent_source_sha256": plan["parent_source_sha256"],
            "wrapper_execution_sha256": plan["wrapper_execution_sha256"],
        },
        "execution_print_ast_guard": print_guard,
        "tiny_restore_optimizer_parity": parity,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "parent_modified": False,
    }
    if args.report:
        report_path = RECOVERY._project_path(args.report, "analyses")
        if report_path.exists():
            raise FileExistsError(f"Audit report must be a new file: {report_path}")
        report["report_path"] = report_path.relative_to(RECOVERY.ROOT).as_posix()
        RECOVERY._write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
