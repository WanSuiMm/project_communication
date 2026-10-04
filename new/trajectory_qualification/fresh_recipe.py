"""Stage-two fresh-pair qualification for a selected u3 state-transfer recipe.

This module deliberately evaluates the root runner's already-frozen recipe
selection.  It does not select a recipe, alter the model, or estimate a basin
volume.  Each pair compares two runs from the same initialization and the same
300-update batch schedule; only the selected state components differ after
update three.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

try:
    from . import common as C
except ImportError:  # Direct import when the runner adds this folder to sys.path.
    import common as C


RECIPE_COMPONENTS = {
    "shadow_moments": (False, True),
    "shadow_parameters": (True, False),
    "shadow_joint": (True, True),
}
INITIALIZATION_SEEDS = tuple(range(31001, 31017))
SCHEDULE_SEEDS = tuple(range(32001, 32017))
TRAIN_BANK_SEED = 10002
EVAL_SEEDS = {32: 83032, 64: 83064}
UPDATES = 300
PREFIX_UPDATES = 3
BATCH_SIZE = 8
WILSON_Z_95 = 1.959963984540054


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wilson_interval(successes: int, trials: int, z: float = WILSON_Z_95) -> dict[str, float | int]:
    """Return the two-sided Wilson score interval without external statistics packages."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Require 0 <= successes <= trials")
    if trials == 0:
        return {"successes": successes, "trials": trials, "lower": 0.0, "upper": 1.0}
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denom
    return {
        "successes": successes,
        "trials": trials,
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
    }


def exact_two_sided_binomial_p(wins: int, losses: int) -> float:
    """Exact two-sided McNemar/binomial p value for paired discordances."""
    if wins < 0 or losses < 0:
        raise ValueError("Discordance counts cannot be negative")
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    lower_tail = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def check_statistics_fixtures() -> dict[str, Any]:
    """Cheap CPU-only fixture check for the primary paired statistics."""
    p_all_wins = exact_two_sided_binomial_p(16, 0)
    p_no_discordance = exact_two_sided_binomial_p(0, 0)
    interval = wilson_interval(8, 16)
    assert p_all_wins == 2 / (2**16)
    assert p_no_discordance == 1.0
    assert 0.0 < interval["lower"] < 0.5 < interval["upper"] < 1.0
    return {
        "sixteen_wins_zero_losses_p": p_all_wins,
        "zero_discordances_p": p_no_discordance,
        "wilson_8_of_16": interval,
    }


def check_statistics() -> dict[str, Any]:
    """Entry point used by the root runner's CPU-only smoke check."""
    return check_statistics_fixtures()


def _hash_value(digest: Any, value: Any) -> None:
    """Hash nested checkpoint payloads canonically, including optimizer metadata."""
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"torch:")
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"numpy:")
        digest.update(str((array.shape, str(array.dtype))).encode("utf-8"))
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict{")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (tuple, list)):
        digest.update(b"tuple[" if isinstance(value, tuple) else b"list[")
        for item in value:
            _hash_value(digest, item)
        digest.update(b"]")
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(b"scalar:")
        digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"))
        digest.update(b";")
    else:
        raise TypeError(f"Unsupported checkpoint hash value: {type(value).__name__}")


def payload_hash(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, payload)
    return digest.hexdigest()


def _tensor_map_hash(values: dict[str, torch.Tensor]) -> str:
    return C.tensor_hash(values)


def _theta_hash(model: torch.nn.Module) -> str:
    return _tensor_map_hash(model.state_dict())


def _moment_map(optimizer_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for parameter_id, state in optimizer_state["state"].items():
        for slot in ("exp_avg", "exp_avg_sq"):
            if slot not in state:
                raise AssertionError(f"AdamW state lacks {slot} for parameter {parameter_id}")
            result[f"{parameter_id}:{slot}"] = state[slot]
    if not result:
        raise AssertionError("AdamW moments are unexpectedly empty")
    return result


def _moment_hash(optimizer: torch.optim.Optimizer) -> str:
    return _tensor_map_hash(_moment_map(optimizer.state_dict()))


def _step_values(optimizer: torch.optim.Optimizer) -> list[int]:
    state = optimizer.state_dict()["state"]
    return sorted({int(value["step"].item()) for value in state.values()})


def _step_hash(optimizer: torch.optim.Optimizer) -> str:
    state = optimizer.state_dict()["state"]
    return _tensor_map_hash({f"{parameter_id}:step": value["step"]
                             for parameter_id, value in state.items()})


def _parameter_group_hash(optimizer: torch.optim.Optimizer) -> str:
    return payload_hash(optimizer.state_dict()["param_groups"])


def _parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters()]


def _copy_shadow_moments(
    target: torch.optim.Optimizer,
    shadow: torch.optim.Optimizer,
) -> None:
    target_state = target.state_dict()
    shadow_state = shadow.state_dict()
    target_ids = target_state["param_groups"][0]["params"]
    shadow_ids = shadow_state["param_groups"][0]["params"]
    if len(target_ids) != len(shadow_ids):
        raise AssertionError("Shadow and treatment optimizer parameter counts differ")
    for target_id, shadow_id in zip(target_ids, shadow_ids):
        target_slots = target_state["state"][target_id]
        shadow_slots = shadow_state["state"][shadow_id]
        if set(target_slots) != set(shadow_slots):
            raise AssertionError("Shadow and treatment optimizer state schemas differ")
        for name in ("exp_avg", "exp_avg_sq"):
            target_slots[name] = shadow_slots[name].detach().clone()
        # Deliberately retain target_slots['step'] at three.
    target.load_state_dict(target_state)


def _check_arm_checkpoint(
    payload: dict[str, Any], seed: int, completed_updates: int,
) -> None:
    if payload["initialization_seed"] != seed:
        raise AssertionError("Checkpoint initialization seed mismatch")
    if payload["completed_updates"] != completed_updates:
        raise AssertionError("Checkpoint update count mismatch")
    state = payload["optimizer_state_dict"]["state"]
    if len(state) != len(payload["optimizer_state_dict"]["param_groups"][0]["params"]):
        raise AssertionError("Optimizer state is incomplete")
    steps = sorted({int(value["step"].item()) for value in state.values()})
    if steps != [completed_updates]:
        raise AssertionError(f"Expected Adam step {completed_updates}, found {steps}")


def _schedule_records() -> tuple[list[dict[str, Any]], np.ndarray]:
    fixed = np.random.default_rng(20002).integers(0, 512, (300, 8))[:3]
    records = []
    for init_seed, schedule_seed in zip(INITIALIZATION_SEEDS, SCHEDULE_SEEDS):
        schedule = np.random.default_rng(schedule_seed).integers(0, 512, (UPDATES, BATCH_SIZE))
        records.append({
            "pair_id": f"init{init_seed}_schedule{schedule_seed}",
            "initialization_seed": init_seed,
            "schedule_seed": schedule_seed,
            "batch_indices": schedule.tolist(),
        })
    return records, fixed


def _write_json(path: Path, value: Any) -> None:
    C.write(path, value)


def _report(
    progress: Callable[[dict[str, Any]], Any],
    phase: str,
    pair_id: str | None = None,
    arm: str | None = None,
    completed_updates: int | None = None,
    **extra: Any,
) -> None:
    event: dict[str, Any] = {"phase": phase}
    if pair_id is not None:
        event["pair_id"] = pair_id
    if arm is not None:
        event["arm"] = arm
    if completed_updates is not None:
        event["completed_updates"] = completed_updates
    event.update(extra)
    progress(event)


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _aggregate_continuous(arms: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {
        name: [row["evaluation_compact"] for row in arms if row["arm"] == name]
        for name in ("baseline", "treatment")
    }
    metric_paths = {
        "retention64_to256": ("retention64_to256",),
        "ever_regressed_fraction": ("ever_regressed_fraction",),
        "frontier_effect": ("frontier_effect",),
        "coverage_t64": ("coverage", "64"),
        "coverage_t128": ("coverage", "128"),
        "coverage_t256": ("coverage", "256"),
        "strict_mean_t64": ("strict_mean", "64"),
        "strict_mean_t128": ("strict_mean", "128"),
        "strict_mean_t256": ("strict_mean", "256"),
    }
    result: dict[str, Any] = {}
    sizes = ("32", "64")
    for size in sizes:
        result[size] = {}
        for metric, path in metric_paths.items():
            arm_means: dict[str, float | None] = {}
            for arm_name, rows in by_arm.items():
                numbers = []
                for compact in rows:
                    value: Any = compact["sizes"][size]
                    for part in path:
                        value = value[part]
                    if isinstance(value, dict):
                        raise AssertionError(f"Expected scalar for continuous metric {metric}")
                    if value is not None:
                        numbers.append(float(value))
                arm_means[arm_name] = _mean(numbers)
            baseline = arm_means["baseline"]
            treatment = arm_means["treatment"]
            result[size][metric] = {
                "baseline_mean": baseline,
                "treatment_mean": treatment,
                "baseline_valid_pairs": sum(
                    1 for compact in by_arm["baseline"]
                    if _metric_value(compact, size, path) is not None
                ),
                "treatment_valid_pairs": sum(
                    1 for compact in by_arm["treatment"]
                    if _metric_value(compact, size, path) is not None
                ),
                "treatment_minus_baseline": (
                    treatment - baseline if baseline is not None and treatment is not None else None
                ),
            }
    return result


def _metric_value(compact: dict[str, Any], size: str, path: tuple[str, ...]) -> Any:
    value: Any = compact["sizes"][size]
    for part in path:
        value = value[part]
    return value


def _paired_summary(arms: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = {row["pair_id"]: row for row in arms if row["arm"] == "baseline"}
    treatment = {row["pair_id"]: row for row in arms if row["arm"] == "treatment"}
    if set(baseline) != set(treatment) or len(baseline) != 16:
        raise AssertionError("Stage two must finish all sixteen matched pairs")
    baseline_passes = sum(bool(row["phenotype_pass"]) for row in baseline.values())
    treatment_passes = sum(bool(row["phenotype_pass"]) for row in treatment.values())
    treatment_wins = sum(
        not bool(baseline[pair]["phenotype_pass"]) and bool(treatment[pair]["phenotype_pass"])
        for pair in baseline
    )
    baseline_wins = sum(
        bool(baseline[pair]["phenotype_pass"]) and not bool(treatment[pair]["phenotype_pass"])
        for pair in baseline
    )
    ties_both_pass = sum(
        bool(baseline[pair]["phenotype_pass"]) and bool(treatment[pair]["phenotype_pass"])
        for pair in baseline
    )
    ties_both_fail = len(baseline) - treatment_wins - baseline_wins - ties_both_pass
    delta = (treatment_passes - baseline_passes) / len(baseline)
    p_value = exact_two_sided_binomial_p(treatment_wins, baseline_wins)
    net_wins = treatment_wins - baseline_wins
    qualified = net_wins >= 4 and delta >= 0.25 and p_value <= 0.05
    return {
        "n_pairs": len(baseline),
        "baseline_passes": baseline_passes,
        "treatment_passes": treatment_passes,
        "baseline_pass_rate": baseline_passes / len(baseline),
        "treatment_pass_rate": treatment_passes / len(baseline),
        "baseline_pass_rate_wilson_95": wilson_interval(baseline_passes, len(baseline)),
        "treatment_pass_rate_wilson_95": wilson_interval(treatment_passes, len(baseline)),
        "delta_treatment_minus_baseline": delta,
        "minimum_net_pairs_for_delta_threshold": 4,
        "discordant_pairs": treatment_wins + baseline_wins,
        "treatment_only_passes": treatment_wins,
        "baseline_only_passes": baseline_wins,
        "net_discordant_wins": net_wins,
        "ties_both_pass": ties_both_pass,
        "ties_both_fail": ties_both_fail,
        "exact_two_sided_mcnemar_binomial_p": p_value,
        "qualification_thresholds": {"delta_at_least": 0.25, "p_at_most": 0.05},
        "verdict": "RELIABILITY_QUALIFICATION" if qualified else "NO_RELIABILITY_QUALIFICATION",
    }


def _results_markdown(summary: dict[str, Any]) -> str:
    paired = summary["primary_paired_result"]
    lines = [
        "# Fresh Recipe Stage-Two Qualification",
        "",
        f"Selected recipe: `{summary['selection']['recipe']}`.",
        "",
        "Each primary unit is one fresh initialization and one independent 300-update batch schedule. Baseline and treatment share that pair's initialization and schedule; treatment imports the selected fixed-prefix u3 state components from a separate shadow trajectory trained on the same fixed three batches.",
        "",
        f"Verdict: **{paired['verdict']}**.",
        "",
        "| Arm | Passes / 16 | Pass rate | Wilson 95% interval |",
        "|---|---:|---:|---:|",
        f"| Baseline | {paired['baseline_passes']} / 16 | {paired['baseline_pass_rate']:.4f} | [{paired['baseline_pass_rate_wilson_95']['lower']:.4f}, {paired['baseline_pass_rate_wilson_95']['upper']:.4f}] |",
        f"| Treatment | {paired['treatment_passes']} / 16 | {paired['treatment_pass_rate']:.4f} | [{paired['treatment_pass_rate_wilson_95']['lower']:.4f}, {paired['treatment_pass_rate_wilson_95']['upper']:.4f}] |",
        "",
        f"Paired pass-rate delta (treatment minus baseline): **{paired['delta_treatment_minus_baseline']:+.4f}** ({paired['treatment_only_passes']} treatment-only passes, {paired['baseline_only_passes']} baseline-only passes; {paired['ties_both_pass']} both-pass ties and {paired['ties_both_fail']} both-fail ties). Exact two-sided McNemar/binomial p = **{paired['exact_two_sided_mcnemar_binomial_p']:.8g}**.",
        "",
        "Qualification requires at least four net treatment-only passes (delta at least 0.25) and exact two-sided p at most 0.05. All sixteen pairs are run regardless of interim outcomes.",
        "",
        "## Secondary continuous summaries",
        "",
        "Treatment-minus-baseline means are reported by evaluation size in `summary.json`; per-pair compact metrics and full original evaluator summaries are retained with the traces.",
        "",
        "## Claim boundary",
        "",
        "This result concerns reliability under the selected recipe and this frozen 16-pair protocol. It does not estimate geometric basin volume and does not support a universal architecture claim.",
        "",
    ]
    return "\n".join(lines)


def run_stage(
    out: str | Path,
    selection: dict[str, Any],
    progress: Callable[[dict[str, Any]], Any] = lambda _: None,
) -> dict[str, Any]:
    """Run all sixteen fixed-protocol pairs for the root-selected recipe.

    ``selection`` must contain the frozen ``recipe`` chosen by the root stage.
    This function never changes that selection or starts an additional efficacy
    sweep.  Exceptions are recorded in the run manifest and re-raised.
    """
    if not isinstance(selection, dict):
        raise TypeError("selection must be a dictionary from the root selector")
    recipe = selection.get("recipe")
    if recipe not in RECIPE_COMPONENTS:
        raise ValueError(f"Unsupported frozen recipe: {recipe!r}")
    if not callable(progress):
        raise TypeError("progress must be callable")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    schedule_records, fixed_prefix = _schedule_records()
    schedule_path = out / "schedules.json"
    fixed_ids = fixed_prefix.tolist()
    _write_json(schedule_path, {
        "protocol": "fresh_recipe_stage2_v1",
        "fixed_shadow_prefix_rng": "np.random.default_rng(20002).integers(0,512,(300,8))[:3]",
        "fixed_shadow_prefix_indices": fixed_ids,
        "schedule_generation": "independent np.random.default_rng(schedule_seed).integers(0,512,(300,8))",
        "pairs": schedule_records,
    })

    manifest: dict[str, Any] = {
        "protocol": "fresh_recipe_stage2_v1",
        "status": "RUNNING",
        "started_utc": _utc_now(),
        "selection": copy.deepcopy(selection),
        "recipe_components_imported": {
            "parameters": RECIPE_COMPONENTS[recipe][0],
            "adam_moments": RECIPE_COMPONENTS[recipe][1],
        },
        "pair_count": 16,
        "initialization_seeds": list(INITIALIZATION_SEEDS),
        "schedule_seeds": list(SCHEDULE_SEEDS),
        "training_bank": {"size": 32, "count": 512, "seed": TRAIN_BANK_SEED, "device": "cuda"},
        "evaluation_banks": [
            {"size": size, "count": 32, "seed": seed, "device": "cpu"}
            for size, seed in EVAL_SEEDS.items()
        ],
        "fixed_shadow_prefix_indices": fixed_ids,
        "schedule_file": "schedules.json",
        "completed_pairs": [],
        "completed_arms": [],
        "artifacts": {},
        "claim_boundary": "No geometric basin-volume estimate or universal architecture claim.",
    }
    manifest_path = out / "manifest.json"
    _write_json(manifest_path, manifest)
    arms: list[dict[str, Any]] = []

    try:
        C.setup_backend()
        train = C.region_bank(32, 512, TRAIN_BANK_SEED, device="cuda")
        eval_data = {
            size: C.region_bank(size, 32, EVAL_SEEDS[size], device="cpu")
            for size in (32, 64)
        }
        manifest["data_sha256"] = {
            "training_bank": C.tensor_hash(train),
            "evaluation_banks": {str(size): C.tensor_hash(bank) for size, bank in eval_data.items()},
        }
        manifest["backend"] = {
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "device_index": 0,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        }
        manifest["common_sha256"] = C.sha(Path(C.__file__))
        manifest["runner_sha256"] = C.sha(Path(__file__))
        _write_json(manifest_path, manifest)

        for record in schedule_records:
            pair_id = record["pair_id"]
            init_seed = int(record["initialization_seed"])
            schedule_seed = int(record["schedule_seed"])
            schedule = np.asarray(record["batch_indices"], dtype=np.int64)
            if schedule.shape != (UPDATES, BATCH_SIZE) or np.any(schedule < 0) or np.any(schedule >= 512):
                raise AssertionError(f"Invalid fresh schedule for {pair_id}")
            pair_folder = out / pair_id
            pair_folder.mkdir(exist_ok=True)
            _report(progress, "pair_start", pair_id, completed_updates=0,
                    initialization_seed=init_seed, schedule_seed=schedule_seed)

            # All three trajectories begin from independently constructed but
            # bit-identical model and optimizer initial states.
            shadow = C.initial(init_seed)
            shadow_opt = C.optimizer_for(shadow)
            baseline = C.initial(init_seed)
            baseline_opt = C.optimizer_for(baseline)
            treatment = C.initial(init_seed)
            treatment_opt = C.optimizer_for(treatment)
            if not (_parameter_names(shadow) == _parameter_names(baseline) == _parameter_names(treatment)):
                raise AssertionError("Model parameter order differs across matched trajectories")
            models = (("shadow", shadow, shadow_opt), ("baseline", baseline, baseline_opt),
                      ("treatment", treatment, treatment_opt))
            initial_hashes = {}
            for name, model, optimizer in models:
                initial_payload = C.payload(model, optimizer, init_seed, 0)
                initial_hashes[name] = payload_hash(initial_payload)
            if len(set(initial_hashes.values())) != 1:
                raise AssertionError(f"Fresh initializations differ for {pair_id}")
            initial_hash = initial_hashes["baseline"]

            shadow_curve = C.train_updates(
                shadow, shadow_opt, train, fixed_prefix, start=1,
                progress=lambda event: _report(
                    progress, "shadow_prefix_training", pair_id, "shadow",
                    int(event["completed_updates"]), initialization_seed=init_seed,
                ),
            )
            shadow_payload = C.payload(shadow, shadow_opt, init_seed, PREFIX_UPDATES)
            _check_arm_checkpoint(shadow_payload, init_seed, PREFIX_UPDATES)
            shadow_path = C.save_checkpoint(
                pair_folder, "shadow", shadow, shadow_opt, init_seed, PREFIX_UPDATES,
            )
            shadow_theta_hash = _theta_hash(shadow)
            shadow_moment_hash = _moment_hash(shadow_opt)
            shadow_payload_digest = payload_hash(shadow_payload)

            curves: dict[str, list[dict[str, Any]]] = {}
            checkpoint_records: dict[str, dict[str, Any]] = {
                "shadow_u003": {
                    **shadow_path,
                    "path": (Path(pair_id) / shadow_path["path"]).as_posix(),
                }
            }
            pre_intervention: dict[str, dict[str, Any]] = {}
            post_intervention: dict[str, dict[str, Any]] = {}
            u3_payload_hashes: dict[str, dict[str, str]] = {}
            pair_arm_record_start = len(arms)

            for arm, model, optimizer in (("baseline", baseline, baseline_opt),
                                          ("treatment", treatment, treatment_opt)):
                prefix_curve = C.train_updates(
                    model, optimizer, train, schedule[:PREFIX_UPDATES], start=1,
                    progress=lambda event, selected_arm=arm: _report(
                        progress, "training", pair_id, selected_arm,
                        int(event["completed_updates"]), initialization_seed=init_seed,
                        schedule_seed=schedule_seed,
                    ),
                )
                before_payload = C.payload(model, optimizer, init_seed, PREFIX_UPDATES)
                _check_arm_checkpoint(before_payload, init_seed, PREFIX_UPDATES)
                pre_intervention[arm] = {
                    "payload_sha256": payload_hash(before_payload),
                    "theta_sha256": _theta_hash(model),
                    "adam_moments_sha256": _moment_hash(optimizer),
                    "adam_step_values": _step_values(optimizer),
                    "adam_step_sha256": _step_hash(optimizer),
                    "optimizer_param_groups_sha256": _parameter_group_hash(optimizer),
                }

                if arm == "treatment":
                    if RECIPE_COMPONENTS[recipe][0]:
                        model.load_state_dict(copy.deepcopy(shadow.state_dict()))
                    if RECIPE_COMPONENTS[recipe][1]:
                        _copy_shadow_moments(optimizer, shadow_opt)

                after_payload = C.payload(model, optimizer, init_seed, PREFIX_UPDATES)
                _check_arm_checkpoint(after_payload, init_seed, PREFIX_UPDATES)
                post_intervention[arm] = {
                    "payload_sha256": payload_hash(after_payload),
                    "theta_sha256": _theta_hash(model),
                    "adam_moments_sha256": _moment_hash(optimizer),
                    "adam_step_values": _step_values(optimizer),
                    "adam_step_sha256": _step_hash(optimizer),
                    "optimizer_param_groups_sha256": _parameter_group_hash(optimizer),
                }
                u3_payload_hashes[arm] = {
                    "before_intervention": pre_intervention[arm]["payload_sha256"],
                    "after_intervention": post_intervention[arm]["payload_sha256"],
                }

                if arm == "baseline":
                    if pre_intervention[arm] != post_intervention[arm]:
                        raise AssertionError("Baseline changed at the intervention boundary")
                else:
                    for field in ("adam_step_sha256", "optimizer_param_groups_sha256"):
                        if post_intervention[arm][field] != pre_intervention[arm][field]:
                            raise AssertionError(f"Treatment import changed {field}")
                    expected_theta_hash = (
                        shadow_theta_hash if RECIPE_COMPONENTS[recipe][0]
                        else pre_intervention[arm]["theta_sha256"]
                    )
                    expected_moment_hash = (
                        shadow_moment_hash if RECIPE_COMPONENTS[recipe][1]
                        else pre_intervention[arm]["adam_moments_sha256"]
                    )
                    if post_intervention[arm]["theta_sha256"] != expected_theta_hash:
                        raise AssertionError("Treatment parameter import does not match selected recipe")
                    if post_intervention[arm]["adam_moments_sha256"] != expected_moment_hash:
                        raise AssertionError("Treatment Adam moment import does not match selected recipe")
                    if post_intervention[arm]["adam_step_values"] != [PREFIX_UPDATES]:
                        raise AssertionError("Treatment import changed Adam step three")
                    if RECIPE_COMPONENTS[recipe][0] and post_intervention[arm]["theta_sha256"] != shadow_theta_hash:
                        raise AssertionError("Treatment theta differs from shadow theta")
                    if RECIPE_COMPONENTS[recipe][1] and post_intervention[arm]["adam_moments_sha256"] != shadow_moment_hash:
                        raise AssertionError("Treatment moments differ from shadow moments")

                arm_u3_checkpoint = C.save_checkpoint(
                    pair_folder, arm, model, optimizer, init_seed, PREFIX_UPDATES,
                )
                checkpoint_records[f"{arm}_u003"] = {
                    **arm_u3_checkpoint,
                    "path": (Path(pair_id) / arm_u3_checkpoint["path"]).as_posix(),
                }
                suffix_curve = C.train_updates(
                    model, optimizer, train, schedule[PREFIX_UPDATES:], start=PREFIX_UPDATES + 1,
                    progress=lambda event, selected_arm=arm: _report(
                        progress, "training", pair_id, selected_arm,
                        int(event["completed_updates"]), initialization_seed=init_seed,
                        schedule_seed=schedule_seed,
                    ),
                )
                curves[arm] = prefix_curve + suffix_curve
                if len(curves[arm]) != UPDATES or [row["update"] for row in curves[arm]] != list(range(1, UPDATES + 1)):
                    raise AssertionError(f"Expected a complete 300-update {arm} curve")
                final_payload = C.payload(model, optimizer, init_seed, UPDATES)
                _check_arm_checkpoint(final_payload, init_seed, UPDATES)
                arm_u300_checkpoint = C.save_checkpoint(
                    pair_folder, arm, model, optimizer, init_seed, UPDATES,
                )
                checkpoint_records[f"{arm}_u300"] = {
                    **arm_u300_checkpoint,
                    "path": (Path(pair_id) / arm_u300_checkpoint["path"]).as_posix(),
                }

                eval_folder = pair_folder / "evaluation" / arm
                eval_folder.mkdir(parents=True, exist_ok=True)
                _report(progress, "evaluation_started", pair_id, arm, UPDATES,
                        initialization_seed=init_seed)
                eval_summary = C.evaluate(
                    model, eval_data, eval_folder, f"{pair_id}_{arm}", lambda: None,
                )
                _report(progress, "evaluation_completed", pair_id, arm, UPDATES,
                        phenotype_pass=bool(eval_summary["phenotype_gate"]["pass"]))
                compact = C.compact(eval_summary)
                summary_rel = (eval_folder / f"{pair_id}_{arm}_summary.json").relative_to(out).as_posix()
                traces_rel = [
                    (eval_folder / f"{pair_id}_{arm}_size{size}.npz").relative_to(out).as_posix()
                    for size in (32, 64)
                ]
                frontier_rel = (
                    eval_folder / f"{pair_id}_{arm}_matched_frontier_strata.csv"
                ).relative_to(out).as_posix()
                arm_record = {
                    "pair_id": pair_id,
                    "initialization_seed": init_seed,
                    "schedule_seed": schedule_seed,
                    "arm": arm,
                    "recipe": recipe,
                    "phenotype_pass": bool(eval_summary["phenotype_gate"]["pass"]),
                    "training_curve": curves[arm],
                    "shadow_prefix_curve": shadow_curve if arm == "treatment" else None,
                    "initial_payload_sha256": initial_hash,
                    "u3_payload_hashes": u3_payload_hashes,
                    "u3_components": None,
                    "checkpoints": {
                        "shadow_u003": checkpoint_records["shadow_u003"],
                        f"{arm}_u003": checkpoint_records[f"{arm}_u003"],
                        f"{arm}_u300": checkpoint_records[f"{arm}_u300"],
                    },
                    "evaluation_summary": summary_rel,
                    "evaluation_traces": traces_rel,
                    "frontier_strata": frontier_rel,
                    "evaluation_compact": compact,
                }
                arms.append(arm_record)
                manifest["completed_arms"].append(f"{pair_id}:{arm}")
                manifest["artifacts"][f"{pair_id}:{arm}"] = {
                    "evaluation_summary": summary_rel,
                    "evaluation_traces": traces_rel,
                    "frontier_strata": frontier_rel,
                    "checkpoints": copy.deepcopy(arm_record["checkpoints"]),
                }
                _write_json(out / "perarm.json", arms)
                _write_json(manifest_path, manifest)
                _report(progress, "arm_complete", pair_id, arm, UPDATES,
                        phenotype_pass=arm_record["phenotype_pass"])

            # Both arm records carry the complete pair-level audit, including
            # the treatment's before/after hashes and the imported shadow state.
            for pair_arm in arms[pair_arm_record_start:]:
                pair_arm["u3_payload_hashes"] = copy.deepcopy(u3_payload_hashes)
                pair_arm["u3_components"] = {
                    "baseline_before": copy.deepcopy(pre_intervention["baseline"]),
                    "baseline_after": copy.deepcopy(post_intervention["baseline"]),
                    "treatment_before": copy.deepcopy(pre_intervention["treatment"]),
                    "treatment_after": copy.deepcopy(post_intervention["treatment"]),
                    "shadow": {
                        "payload_sha256": shadow_payload_digest,
                        "theta_sha256": shadow_theta_hash,
                        "adam_moments_sha256": shadow_moment_hash,
                        "adam_step_values": _step_values(shadow_opt),
                        "adam_step_sha256": _step_hash(shadow_opt),
                        "optimizer_param_groups_sha256": _parameter_group_hash(shadow_opt),
                    },
                    "selected_recipe": recipe,
                }
            _write_json(out / "perarm.json", arms)
            _write_json(manifest_path, manifest)

            manifest["completed_pairs"].append(pair_id)
            _write_json(manifest_path, manifest)
            _report(progress, "pair_complete", pair_id, completed_updates=UPDATES,
                    arms_completed=["baseline", "treatment"])

        primary = _paired_summary(arms)
        outcomes_by_pair: dict[str, dict[str, Any]] = {}
        for arm_record in arms:
            outcomes_by_pair.setdefault(arm_record["pair_id"], {})[arm_record["arm"]] = arm_record
        summary = {
            "protocol": "fresh_recipe_stage2_v1",
            "status": "COMPLETE",
            "selection": copy.deepcopy(selection),
            "recipe_components_imported": {
                "parameters": RECIPE_COMPONENTS[recipe][0],
                "adam_moments": RECIPE_COMPONENTS[recipe][1],
            },
            "primary_unit": "fresh initialization plus independently sampled 300-update batch schedule",
            "primary_paired_result": primary,
            "secondary_continuous_means": _aggregate_continuous(arms),
            "per_pair_passes": [
                {
                    "pair_id": pair_id,
                    "baseline_pass": bool(pair["baseline"]["phenotype_pass"]),
                    "treatment_pass": bool(pair["treatment"]["phenotype_pass"]),
                    "treatment_minus_baseline": int(bool(pair["treatment"]["phenotype_pass"])) - int(bool(pair["baseline"]["phenotype_pass"])),
                }
                for pair_id, pair in sorted(outcomes_by_pair.items())
            ],
            "artifact_routes": {
                "schedules": "schedules.json",
                "per_arm_training_curves_and_pair_audits": "perarm.json",
                "full_evaluation_summaries_traces_and_frontiers": "<pair_id>/evaluation/<arm>/",
                "checkpoints": "<pair_id>/checkpoints/",
            },
            "claim_boundary": "Reliability qualification for the selected recipe and frozen protocol only; no geometric basin-volume estimate or universal architecture claim.",
        }
        _write_json(out / "summary.json", summary)
        (out / "RESULTS.md").write_text(_results_markdown(summary), encoding="utf-8")
        manifest["status"] = "COMPLETE"
        manifest["completed_utc"] = _utc_now()
        manifest["summary_file"] = "summary.json"
        manifest["results_file"] = "RESULTS.md"
        _write_json(manifest_path, manifest)
        _report(progress, "stage_complete", completed_updates=UPDATES,
                verdict=primary["verdict"], delta=primary["delta_treatment_minus_baseline"],
                p_value=primary["exact_two_sided_mcnemar_binomial_p"])
        return summary
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["failed_utc"] = _utc_now()
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            _write_json(manifest_path, manifest)
        finally:
            raise
