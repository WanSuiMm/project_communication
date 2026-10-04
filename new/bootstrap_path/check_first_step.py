"""One cheap CPU fixture for the one-update bootstrap-path audit."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import torch

_FIRST_STEP_SPEC = importlib.util.spec_from_file_location(
    "_bootstrap_path_check_impl", Path(__file__).with_name("first_step.py"))
if _FIRST_STEP_SPEC is None or _FIRST_STEP_SPEC.loader is None:
    raise ImportError("Cannot load adjacent first_step.py")
_FIRST_STEP_MODULE = importlib.util.module_from_spec(_FIRST_STEP_SPEC)
sys.modules[_FIRST_STEP_SPEC.name] = _FIRST_STEP_MODULE
_FIRST_STEP_SPEC.loader.exec_module(_FIRST_STEP_MODULE)
ROOT = _FIRST_STEP_MODULE.ROOT
_load_exact_dependencies = _FIRST_STEP_MODULE._load_exact_dependencies
audit_first_step = _FIRST_STEP_MODULE.audit_first_step


def _load_tasks():
    path = ROOT / "new/nca_inertial_wind_tunnel/tasks.py"
    spec = importlib.util.spec_from_file_location("_bootstrap_path_check_tasks", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_check() -> dict:
    started = time.monotonic()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(min(old_threads, 2))
    sentinel = object()
    prior_tasks = sys.modules.get("tasks", sentinel)
    try:
        # Deliberately seed a conflicting cached name. The audit must load its
        # exact task file for backward_trajectory, then restore this entry.
        sys.modules["tasks"] = types.ModuleType("tasks")
        tasks = _load_tasks()
        data = tasks.subset(tasks.bank(8, 2, 48271, device="cpu"), [0, 1])
        assert all(torch.isfinite(data[k]).all() for k in ("x", "y", "mask"))

        cell_type, _ = _load_exact_dependencies()
        torch.manual_seed(48272)
        model = cell_type()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        report = audit_first_step(model, data)
        after = dict(model.named_parameters())

        assert report["trajectory"]["forward_steps"] == 64
        assert report["trajectory"]["loss_count"] == 8
        assert report["trajectory"]["backward_calls"] == 8
        assert report["optimizer"]["steps"] == 1
        assert not torch.equal(before["q_out.weight"], after["q_out.weight"].detach())
        assert report["q_out_weight_geometry"]["applied_gradient"]["singular_values"]
        assert report["post_update_forward_step"]["z_write"]["finite"]
        assert report["interpretation_boundary"]["claims_general_mechanism"] is False
        assert "rank-one gradient alone" in report["interpretation_boundary"]["claim_limit"]
        formula_errors = [v["first_step_formula_relative_error"]
                          for v in report["parameters"].values()]
        max_formula_error = max(formula_errors, default=0.0)
        assert max_formula_error < 2e-5, f"AdamW first-step formula error {max_formula_error}"

        # This parameter is nonzero, has a zero task gradient at initialization,
        # and still receives AdamW's decoupled weight-decay update.
        readout_weight = report["parameters"]["readout.weight"]
        assert readout_weight["parameter_l2_before"] > 0
        assert readout_weight["gradient_l2_applied"] == 0
        assert readout_weight["delta_l2_actual"] > 0
        assert readout_weight["decoupled_weight_decay_delta_l2"] > 0
        json.dumps(report, allow_nan=False)
        assert sys.modules["tasks"].__name__ == "tasks" and not hasattr(sys.modules["tasks"], "balanced_loss")
        return {
            "status": "PASS",
            "fixture": {"size": 8, "map_count": 2, "device": "cpu"},
            "forward_steps": report["trajectory"]["forward_steps"],
            "loss_count": report["trajectory"]["loss_count"],
            "optimizer_steps": report["optimizer"]["steps"],
            "q_out_weight_delta_l2": report["parameters"]["q_out.weight"]["delta_l2_actual"],
            "readout_weight_decay_delta_l2": readout_weight["decoupled_weight_decay_delta_l2"],
            "max_firststep_formula_relative_error": max_formula_error,
            "semantic_projection_fraction": report["post_update_forward_step"]["z_write"]["semantic_projection_fraction"],
            "elapsed_seconds": time.monotonic() - started,
            "general_mechanism_claim": False,
        }
    finally:
        if prior_tasks is sentinel:
            sys.modules.pop("tasks", None)
        else:
            sys.modules["tasks"] = prior_tasks
        torch.set_num_threads(old_threads)


if __name__ == "__main__":
    print(json.dumps(run_check(), indent=2, allow_nan=False))
