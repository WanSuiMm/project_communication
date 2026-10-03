"""Zero-training, event-matched local rollback follow-up for Streaming seed4."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
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
for _relative in (
    "new/stream_path_audit",
    "new/streaming_carry",
    "new/workspace_revision",
    "new/short_bptt_phase2",
    "new/nca_inertial_wind_tunnel",
):
    _path = str(ROOT / _relative)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stream_cells import StreamingCell  # noqa: E402
from run_revision import score, sha, tensor_hash, write  # noqa: E402
from tasks import bank  # noqa: E402

_audit_spec = importlib.util.spec_from_file_location(
    "causal_stream_path_audit", ROOT / "new/stream_path_audit/audit.py"
)
_audit = importlib.util.module_from_spec(_audit_spec)
assert _audit_spec and _audit_spec.loader
_audit_spec.loader.exec_module(_audit)

TIMES = (16, 32, 64, 128)
SIZES = (32, 64)
MAPS = 32
MAX_EVENTS_PER_MAP_TIME = 4
MIN_EVENTS_PER_SIZE = 64
MIN_MAPS_PER_SIZE = 16
CHUNK_EVENTS = 16
HARD_CAP_SECONDS = 300
WATCHDOG_SECONDS = 330
DEADLINE = float("inf")
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
CONDITIONS = ("native", "noop_reference", "sender_rollback", "wrong_neighbor_sham", "offcone")


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def budget():
    if time.monotonic() >= DEADLINE:
        raise TimeoutError("Frozen 300-second causal follow-up cap")


def state_copy(state):
    return tuple(value.clone() for value in state)


def gather_site(value, positions):
    """Return one [channels] vector per batch row at its (y,x) site."""
    return torch.stack([value[i, :, y, x] for i, (y, x) in enumerate(positions)])


def site_delta(previous, current, positions):
    return gather_site(previous, positions) - gather_site(current, positions)


def replace_site(current, previous, positions):
    """Exact local temporal restore; every non-selected cell is bit-identical."""
    result = current.clone()
    for i, (y, x) in enumerate(positions):
        result[i, :, y, x] = previous[i, :, y, x]
    return result


def apply_site_delta(current, deltas, positions):
    """Add one independent channel-vector perturbation per batch row."""
    result = current.clone()
    for i, (y, x) in enumerate(positions):
        result[i, :, y, x] = current[i, :, y, x] + deltas[i]
    return result


def scaled_rollback_sham(q_previous, q_current, r_previous, r_current, q_positions, r_positions):
    """Use r's own rollback direction, scaled to q's norm, for each map row."""
    q_delta = site_delta(q_previous, q_current, q_positions)
    r_delta = site_delta(r_previous, r_current, r_positions)
    q_norm = torch.linalg.vector_norm(q_delta, dim=1, keepdim=True)
    r_norm = torch.linalg.vector_norm(r_delta, dim=1, keepdim=True)
    active = q_norm[:, 0] > 0
    if bool((active & (r_norm[:, 0] == 0)).any()):
        raise ValueError("Wrong-neighbor rollback direction is zero while q rollback is nonzero")
    scale = torch.where(active[:, None], q_norm / r_norm.clamp_min(torch.finfo(r_norm.dtype).tiny), 0.0)
    return r_delta * scale, q_delta, r_delta


def _neighbors(y, x, n):
    for dy, dx in DIRECTIONS:
        yy, xx = y + dy, x + dx
        if 0 <= yy < n and 0 <= xx < n:
            yield yy, xx


def _world_logits(model, states):
    return model.logits(states).detach().cpu()


@torch.no_grad()
def capture_states(model, data):
    """Capture states immediately before and after each frozen selection time."""
    x, xf = data["x"], data["x_flip"]
    original, flipped = model.initial(x), model.initial(xf)
    snapshots = {}
    for t in range(1, max(TIMES) + 1):
        budget()
        previous_a, previous_b = original, flipped
        current_a, current_b = model.step(previous_a, x), model.step(previous_b, xf)
        if t in TIMES:
            snapshots[t] = {
                "previous": (state_copy(previous_a), state_copy(previous_b)),
                "current": (state_copy(current_a), state_copy(current_b)),
            }
        original, flipped = current_a, current_b
    return snapshots


def _rollback_direction_valid(snapshot, map_index, q, r):
    for world in (0, 1):
        prev, curr = snapshot["previous"][world], snapshot["current"][world]
        for block in (0, 1):
            qd = prev[block][map_index, :, q[0], q[1]] - curr[block][map_index, :, q[0], q[1]]
            rd = prev[block][map_index, :, r[0], r[1]] - curr[block][map_index, :, r[0], r[1]]
            if bool((qd != 0).any()) and not bool((rd != 0).any()):
                return False
    return True


def _offcone_site(data_cpu, map_index, p, excluded, sorted_sites):
    py, px = p
    for y, x in sorted_sites:
        if abs(y - py) + abs(x - px) > 2 and (y, x) not in excluded:
            return (y, x)
    return None


def select_events(model, data_cpu, snapshots, size):
    """Freeze a lexical event list using only t−1/t states and pair labels."""
    events = []
    n = size
    opened = data_cpu["mask"][:, 0].bool()
    changed = data_cpu["changed"][:, 0].bool()
    distance = data_cpu["distance"][:, 0]
    source = (data_cpu["x"][:, 1] != 0) | (data_cpu["x"][:, 2] != 0)
    offcone_sites = {}
    for map_index in range(MAPS):
        offcone_sites[map_index] = [
            (y, x) for y in range(n) for x in range(n)
            if opened[map_index, y, x] and changed[map_index, y, x] and not source[map_index, y, x]
        ]

    for t in TIMES:
        snap = snapshots[t]
        prev_a = _world_logits(model, snap["previous"][0])
        prev_b = _world_logits(model, snap["previous"][1])
        curr_a = _world_logits(model, snap["current"][0])
        curr_b = _world_logits(model, snap["current"][1])
        good_previous = (((prev_a >= 0) == (data_cpu["y"] >= .5))
                         & ((prev_b >= 0) == (data_cpu["y_flip"] >= .5)))[:, 0] & changed
        good_current = (((curr_a >= 0) == (data_cpu["y"] >= .5))
                        & ((curr_b >= 0) == (data_cpu["y_flip"] >= .5)))[:, 0] & changed

        for map_index in range(MAPS):
            chosen = []
            for py in range(n):
                for px in range(n):
                    p = (py, px)
                    if not opened[map_index, py, px] or not changed[map_index, py, px]:
                        continue
                    if int(distance[map_index, py, px]) <= 8 or bool(good_current[map_index, py, px]):
                        continue
                    q_options = []
                    r_options = []
                    p_dist = int(distance[map_index, py, px])
                    for yy, xx in _neighbors(py, px, n):
                        if not opened[map_index, yy, xx] or not changed[map_index, yy, xx]:
                            continue
                        if int(distance[map_index, yy, xx]) != p_dist - 1:
                            continue
                        if not bool(good_previous[map_index, yy, xx]) and bool(good_current[map_index, yy, xx]) and not source[map_index, yy, xx]:
                            q_options.append((yy, xx))
                        if not bool(good_previous[map_index, yy, xx]) and not bool(good_current[map_index, yy, xx]):
                            r_options.append((yy, xx))
                    q_options.sort()
                    r_options.sort()
                    if not q_options or not r_options:
                        continue
                    for q in q_options:
                        for r in sorted(r_options):
                            if r == q or not _rollback_direction_valid(snap, map_index, q, r):
                                continue
                            s = _offcone_site(data_cpu, map_index, p, {p, q, r}, offcone_sites[map_index])
                            if s is None:
                                continue
                            chosen.append((p, q, r, s))
                            # p, q and r are visited in lexical order, so these are the first four tuples.
                            if len(chosen) >= MAX_EVENTS_PER_MAP_TIME:
                                break
                        if len(chosen) >= MAX_EVENTS_PER_MAP_TIME:
                            break
                    if len(chosen) >= MAX_EVENTS_PER_MAP_TIME:
                        break
                if len(chosen) >= MAX_EVENTS_PER_MAP_TIME:
                    break
            for p, q, r, s in chosen:
                events.append({
                    "event_id": f"s{size}_t{t}_m{map_index:02d}_p{p[0]:02d}-{p[1]:02d}_q{q[0]:02d}-{q[1]:02d}_r{r[0]:02d}-{r[1]:02d}",
                    "size": size, "time": t, "map_index": map_index,
                    "p": list(p), "q": list(q), "r": list(r),
                    "s": None if s is None else list(s),
                    "p_distance": int(distance[map_index, p[0], p[1]]),
                    "q_distance": int(distance[map_index, q[0], q[1]]),
                    "event_selection": "lexical t-1/t paired-correctness and frozen input geometry",
                    "outcomes": {},
                })
    return events


def _make_condition_batch(model, data, snap, events):
    """Create independent full-map clones; each row is one event/condition/world."""
    flat_states = []
    base_states = []
    x_rows = []
    locations = []
    for ei, event in enumerate(events):
        map_index = event["map_index"]
        q, r = tuple(event["q"]), tuple(event["r"])
        s = None if event["s"] is None else tuple(event["s"])
        for ci, condition in enumerate(CONDITIONS):
            for world in (0, 1):
                previous = snap["previous"][world]
                current = snap["current"][world]
                base_state = tuple(block[map_index].clone() for block in current)
                row_state = state_copy(base_state)
                if condition == "sender_rollback":
                    row_state = tuple(
                        _replace_row_site(row_state[block], previous[block][map_index], q)
                        for block in (0, 1)
                    )
                elif condition == "wrong_neighbor_sham":
                    row_state = _sham_row(row_state, previous, current, map_index, q, r)
                elif condition == "offcone" and s is not None:
                    row_state = _offcone_row(row_state, previous, current, map_index, q, s)
                flat_states.append(row_state)
                base_states.append(base_state)
                x_rows.append(data["x" if world == 0 else "x_flip"][map_index])
                locations.append({"event": ei, "condition": ci, "world": world, "p": tuple(event["p"]), "q": q, "r": r, "s": s})
    w = torch.stack([row[0] for row in flat_states])
    z = torch.stack([row[1] for row in flat_states])
    x = torch.stack(x_rows).to(data["x"].device)
    base_w = torch.stack([row[0] for row in base_states])
    base_z = torch.stack([row[1] for row in base_states])
    return (w, z), x, locations, (base_w, base_z)


def _replace_row_site(current, previous, site):
    y, x = site
    result = current.clone()
    result[:, y, x] = previous[:, y, x]
    return result


def _sham_row(state, previous, current, map_index, q, r):
    blocks = []
    for b in (0, 1):
        qd = previous[b][map_index, :, q[0], q[1]] - current[b][map_index, :, q[0], q[1]]
        rd = previous[b][map_index, :, r[0], r[1]] - current[b][map_index, :, r[0], r[1]]
        qn = torch.linalg.vector_norm(qd)
        rn = torch.linalg.vector_norm(rd)
        if bool(qn > 0) and not bool(rn > 0):
            raise FloatingPointError("Matched sham rollback direction became zero")
        delta = rd * (qn / rn) if bool(qn > 0) else torch.zeros_like(rd)
        row = state[b].clone()
        row[:, r[0], r[1]] = state[b][:, r[0], r[1]] + delta
        blocks.append(row)
    return tuple(blocks)


def _offcone_row(state, previous, current, map_index, q, s):
    blocks = []
    for b in (0, 1):
        delta = previous[b][map_index, :, q[0], q[1]] - current[b][map_index, :, q[0], q[1]]
        row = state[b].clone()
        row[:, s[0], s[1]] = state[b][:, s[0], s[1]] + delta
        blocks.append(row)
    return tuple(blocks)


def event_batches(events):
    """Pipeline batching primitive: never mixes snapshot times in a GPU chunk."""
    for t in TIMES:
        time_events = [event for event in events if event["time"] == t]
        for start in range(0, len(time_events), CHUNK_EVENTS):
            chunk = time_events[start:start + CHUNK_EVENTS]
            if any(event["time"] != t for event in chunk):
                raise AssertionError("Mixed-time event batch would use the wrong snapshot")
            yield t, chunk


@torch.no_grad()
def run_event_outcomes(model, data, snapshots, events):
    for t, batch_events in event_batches(events):
            budget()
            snap = snapshots[t]
            states, x, locations, base_states = _make_condition_batch(model, data, snap, batch_events)
            input_before = x.clone()
            pre_logits = model.logits(states)

            # q's restored Z must reproduce its own t−1 pointwise readout in both worlds.
            for row_index, loc in enumerate(locations):
                if loc["condition"] == CONDITIONS.index("sender_rollback"):
                    prev_z = snap["previous"][loc["world"]][1][batch_events[loc["event"]]["map_index"]]
                    qy, qx = loc["q"]
                    expected = model.readout(prev_z[None])[0, 0, qy, qx]
                    torch.testing.assert_close(pre_logits[row_index, 0, qy, qx], expected, atol=1e-6, rtol=0)

            # Compare each clone with the untouched t state and verify only its assigned site changed.
            for ei, event in enumerate(batch_events):
                p = tuple(event["p"])
                q, r, s = tuple(event["q"]), tuple(event["r"]), tuple(event["s"])
                for ci, condition in enumerate(CONDITIONS):
                    for world in (0, 1):
                        row = (ei * len(CONDITIONS) + ci) * 2 + world
                        for block in (0, 1):
                            actual, base = states[block][row], base_states[block][row]
                            if condition in ("native", "noop_reference"):
                                if not torch.equal(actual, base):
                                    raise AssertionError("Native/no-op branch was not an identical state copy")
                                continue
                            site = q if condition == "sender_rollback" else r if condition == "wrong_neighbor_sham" else s
                            outside_check = base.clone()
                            outside_check[:, site[0], site[1]] = actual[:, site[0], site[1]]
                            if not torch.equal(actual, outside_check):
                                raise AssertionError(f"{condition} changed cells outside its chosen site")
                            if not torch.equal(actual[:, p[0], p[1]], base[:, p[0], p[1]]):
                                raise AssertionError("Intervention changed target p before its next step")
                            if condition == "sender_rollback":
                                prev_block = snap["previous"][world][block][event["map_index"]]
                                if not torch.equal(actual[:, site[0], site[1]], prev_block[:, site[0], site[1]]):
                                    raise AssertionError("Sender rollback did not exactly restore t−1 W/Z")
                            elif condition == "wrong_neighbor_sham":
                                qd = snap["previous"][world][block][event["map_index"], :, q[0], q[1]] - snap["current"][world][block][event["map_index"], :, q[0], q[1]]
                                rd = snap["previous"][world][block][event["map_index"], :, r[0], r[1]] - snap["current"][world][block][event["map_index"], :, r[0], r[1]]
                                qnorm, rnorm = torch.linalg.vector_norm(qd), torch.linalg.vector_norm(rd)
                                applied_norm = torch.linalg.vector_norm(actual[:, r[0], r[1]] - base[:, r[0], r[1]])
                                if bool(qnorm > 0) and not bool(rnorm > 0):
                                    raise FloatingPointError("Nonzero q rollback has zero r direction")
                                norm_error = float((applied_norm - qnorm).abs())
                                tolerance = max(1e-6, float(qnorm) * 1e-5)
                                if norm_error > tolerance:
                                    raise FloatingPointError(f"Sham norm mismatch {norm_error} > {tolerance}")
                                event.setdefault("rollback_norm_match", {}).setdefault(str(world), {})["W" if block == 0 else "Z"] = {
                                    "q_rollback_norm": float(qnorm), "applied_sham_norm": float(applied_norm),
                                    "absolute_error": norm_error, "tolerance": tolerance,
                                }
                            else:
                                qd = snap["previous"][world][block][event["map_index"], :, q[0], q[1]] - snap["current"][world][block][event["map_index"], :, q[0], q[1]]
                                expected = base[:, s[0], s[1]] + qd
                                torch.testing.assert_close(actual[:, s[0], s[1]], expected, atol=1e-6, rtol=0)

            results = {event["event_id"]: {"outcomes": {}} for event in batch_events}
            for k in range(1, 5):
                budget()
                states = model.step(states, x)
                if not all(bool(torch.isfinite(v).all()) for v in states):
                    raise FloatingPointError("Nonfinite intervention state")
                if k not in (1, 4):
                    continue
                logits = model.logits(states).detach()
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Nonfinite intervention logits")
                for ei, event in enumerate(batch_events):
                    for world in (0, 1):
                        native = (ei * len(CONDITIONS) + CONDITIONS.index("native")) * 2 + world
                        noop = (ei * len(CONDITIONS) + CONDITIONS.index("noop_reference")) * 2 + world
                        if not torch.equal(states[0][native], states[0][noop]) or not torch.equal(states[1][native], states[1][noop]):
                            raise AssertionError("Native and no-op trajectories diverged")
                        if not torch.equal(logits[native], logits[noop]):
                            raise AssertionError("Native and no-op logits diverged")
                if k == 1:
                    for ei, event in enumerate(batch_events):
                        py, px = event["p"]
                        for world in (0, 1):
                            native = (ei * len(CONDITIONS) + CONDITIONS.index("native")) * 2 + world
                            offcone = (ei * len(CONDITIONS) + CONDITIONS.index("offcone")) * 2 + world
                            error = float((logits[native, 0, py, px] - logits[offcone, 0, py, px]).abs())
                            if error > 1e-6:
                                raise AssertionError(f"Off-cone step-1 target changed by {error}")
                for ei, event in enumerate(batch_events):
                    row = results[event["event_id"]]["outcomes"].setdefault(str(k), {})
                    py, px = event["p"]
                    orig_truth = data["y"][event["map_index"], 0, py, px]
                    flip_truth = data["y_flip"][event["map_index"], 0, py, px]
                    for ci, condition in enumerate(CONDITIONS):
                        if condition == "offcone" and k == 4:
                            continue
                        i0 = (ei * len(CONDITIONS) + ci) * 2
                        i1 = i0 + 1
                        la, lb = logits[i0, 0, py, px], logits[i1, 0, py, px]
                        good = bool(((la >= 0) == (orig_truth >= .5)) & ((lb >= 0) == (flip_truth >= .5)))
                        row[condition] = {
                            "original_logit": float(la), "flipped_logit": float(lb),
                            "paired_correct": good, "paired_acquisition": int(good),
                        }
            for event in batch_events:
                event.update(results[event["event_id"]])
            if not torch.equal(x, input_before):
                raise AssertionError("Intervention changed model input")


def aggregate(events, size):
    selected = [event for event in events if event["size"] == size]
    by_map = {}
    for event in selected:
        by_map.setdefault(event["map_index"], []).append(event)
    maps = []
    contrasts = {"native_minus_sender_rollback": [], "sham_minus_sender_rollback": []}
    for map_index, rows in sorted(by_map.items()):
        map_row = {"map_index": map_index, "events": len(rows), "horizons": {}}
        for horizon in (1, 4):
            native = np.mean([row["outcomes"][str(horizon)]["native"]["paired_acquisition"] for row in rows])
            noop = np.mean([row["outcomes"][str(horizon)]["noop_reference"]["paired_acquisition"] for row in rows])
            sender = np.mean([row["outcomes"][str(horizon)]["sender_rollback"]["paired_acquisition"] for row in rows])
            sham = np.mean([row["outcomes"][str(horizon)]["wrong_neighbor_sham"]["paired_acquisition"] for row in rows])
            map_row["horizons"][str(horizon)] = {
                "native_acquisition": float(native), "noop_acquisition": float(noop),
                "sender_rollback_acquisition": float(sender), "wrong_neighbor_sham_acquisition": float(sham),
                "native_minus_sender": float(native - sender),
                "sham_minus_sender": float(sham - sender),
            }
            contrasts["native_minus_sender_rollback"].append(float(native - sender))
            contrasts["sham_minus_sender_rollback"].append(float(sham - sender))
        maps.append(map_row)
    result = {"size": size, "event_count": len(selected), "eligible_map_count": len(by_map), "per_map": maps, "horizons": {}}
    for horizon in (1, 4):
        result["horizons"][str(horizon)] = {
            key: float(np.mean([row["horizons"][str(horizon)][field] for row in maps])) if maps else None
            for key, field in (
                ("native_acquisition", "native_acquisition"),
                ("noop_acquisition", "noop_acquisition"),
                ("sender_rollback_acquisition", "sender_rollback_acquisition"),
                ("wrong_neighbor_sham_acquisition", "wrong_neighbor_sham_acquisition"),
                ("native_minus_sender_rollback", "native_minus_sender"),
                ("sham_minus_sender_rollback", "sham_minus_sender"),
            )
        }
    return result


def _historical_bindings(out):
    pub_path = ROOT / "STREAM_PATH_AUDIT_PUBLICATION_MANIFEST.json"
    pub = read(pub_path)
    historical_run = ROOT / "runs/stream_path_audit_20261002_seed4_01"
    old_manifest = read(historical_run / "manifest.json")
    old = read(historical_run / "raw_conditions.json")
    frontier_pub_path = ROOT / "FRONTIER_AUDIT_PUBLICATION_MANIFEST.json"
    frontier_pub = read(frontier_pub_path)
    frontier_run = ROOT / "runs/frontier_audit_20261003_seed4_01"
    carry_pub = read(ROOT / "STREAMING_CARRY_PUBLICATION_MANIFEST.json")
    checkpoint_path = ROOT / "runs/streaming_carry_20261002_init2345/stream_K8_seed4.pt"
    carry_run = checkpoint_path.parent
    carry_record_path = carry_run / "stream_K8_seed4.json"
    evidence_record_path = ROOT / "evidence/streaming_carry_init2345/raw/stream_K8_seed4.json"
    checkpoint_file_sha = sha(checkpoint_path)
    checkpoint_expected_sha = carry_pub["checkpoint_sha256"][checkpoint_path.name]
    if checkpoint_file_sha != checkpoint_expected_sha:
        raise AssertionError("Seed4 checkpoint differs from publication binding")
    carry_record_sha = sha(carry_record_path)
    if carry_record_sha != sha(evidence_record_path):
        raise AssertionError("Historical trained record differs from its published copy")

    # Bind the 45-source frontier head against both the publication hashes and its executed snapshot.
    frontier_sources = frontier_pub["source_sha256"]
    if len(frontier_sources) != 45:
        raise AssertionError(f"Expected 45 frozen frontier sources, found {len(frontier_sources)}")
    sources = dict(frontier_sources)
    for name, expected in frontier_sources.items():
        saved_path = frontier_run / "source" / name
        if sha(ROOT / name) != expected or sha(saved_path) != expected:
            raise AssertionError(f"Frontier source/snapshot drift: {name}")
    # Preserve the 39-file stream-path execution binding against its own run snapshot only.
    if len(old_manifest["source_sha256"]) != 39:
        raise AssertionError(f"Expected 39 stream-path executed sources, found {len(old_manifest['source_sha256'])}")
    historical_stream_sources = {}
    for name, expected in old_manifest["source_sha256"].items():
        saved_path = historical_run / "source" / name
        saved_sha = sha(saved_path)
        if saved_sha != expected:
            raise AssertionError(f"Historical executed snapshot mismatch: {name}")
        historical_stream_sources[name] = expected
    for name, expected in pub["source_sha256"].items():
        if historical_stream_sources.get(name) != expected:
            raise AssertionError(f"Stream-path publication and executed snapshot differ: {name}")
    package = ROOT / "new/seed4_followup"
    package_names = sorted([p.relative_to(ROOT).as_posix() for p in package.glob("*.py")])
    package_names += ["new/seed4_followup/PROTOCOL.md"]
    package_names += ["tools/launch_seed4_followup.ps1"]
    for name in package_names:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Expected frozen follow-up source missing: {name}")
        sources[name] = sha(path)
    for name in package_names:
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    for name in sources:
        if name in package_names:
            continue
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    for name in historical_stream_sources:
        dest = out / "source/historical_stream_path" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(historical_run / "source" / name, dest)
    # Keep the manifests whose source lists define the two frozen historical heads.
    shutil.copyfile(pub_path, out / "source/STREAM_PATH_AUDIT_PUBLICATION_MANIFEST.json")
    shutil.copyfile(frontier_pub_path, out / "source/FRONTIER_AUDIT_PUBLICATION_MANIFEST.json")
    shutil.copyfile(ROOT / "STREAMING_CARRY_PUBLICATION_MANIFEST.json", out / "source/STREAMING_CARRY_PUBLICATION_MANIFEST.json")
    return {
        "historical_run": historical_run, "old_manifest": old_manifest, "old": old,
        "source_sha256": sources, "checkpoint_path": checkpoint_path,
        "historical_stream_path_source_sha256": historical_stream_sources,
        "carry_record": read(carry_record_path),
        "checkpoint_file_sha256": checkpoint_file_sha,
        "checkpoint_expected_sha256": checkpoint_expected_sha,
        "carry_record_sha256": carry_record_sha,
        "carry_publication_manifest_sha256": sha(ROOT / "STREAMING_CARRY_PUBLICATION_MANIFEST.json"),
        "path_audit_publication_manifest_sha256": sha(pub_path),
        "frontier_publication_manifest_sha256": sha(frontier_pub_path),
        "historical_source_count": len(frontier_sources),
        "executed_source_count": len(old_manifest["source_sha256"]),
        "historical_raw_conditions_sha256": sha(historical_run / "raw_conditions.json"),
        "followup_source_files": package_names,
        "carry_evaluation_data_sha256": read(carry_run / "manifest.json")["evaluation_data_sha256"],
    }


def _capture_manifest(out, bindings, data_hashes, model, parameter_hash, start):
    import socket
    manifest = {
        "protocol": "seed4_local_temporal_rollback_causal_followup_v1",
        "training": False, "model_seed": 4, "model_replications": 1,
        "started_utc": now(), "pid": os.getpid(), "host": socket.gethostname(),
        "device": "cuda:0", "gpu": torch.cuda.get_device_name(0),
        "command": [sys.executable, *sys.argv],
        "maximum_seconds": HARD_CAP_SECONDS, "watchdog_seconds": WATCHDOG_SECONDS,
        "checkpoint_file_sha256": bindings["checkpoint_file_sha256"],
        "checkpoint_parameter_sha256": parameter_hash,
        "carry_publication_manifest_sha256": bindings["carry_publication_manifest_sha256"],
        "path_audit_publication_manifest_sha256": bindings["path_audit_publication_manifest_sha256"],
        "frontier_publication_manifest_sha256": bindings["frontier_publication_manifest_sha256"],
        "historical_training_record_sha256": bindings["carry_record_sha256"],
        "historical_source_count": bindings["historical_source_count"],
        "executed_source_count": bindings["executed_source_count"],
        "source_sha256": bindings["source_sha256"],
        "historical_stream_path_source_sha256": bindings["historical_stream_path_source_sha256"],
        "historical_raw_conditions_sha256": bindings["historical_raw_conditions_sha256"],
        "evaluation_maps_per_size": MAPS,
        "evaluation_data_sha256": data_hashes,
        "selection_times": list(TIMES), "max_events_per_map_time": MAX_EVENTS_PER_MAP_TIME,
        "event_minima": {"events_per_size": MIN_EVENTS_PER_SIZE, "eligible_maps_per_size": MIN_MAPS_PER_SIZE},
        "conditions": list(CONDITIONS),
        "primary": "mean across eligible maps of native paired acquisition minus sender rollback paired acquisition; report sham minus sender as matched local control",
        "primary_signal_thresholds": {"native_minus_sender_each_size": 0.10, "sham_minus_sender_each_size": 0.05},
        "backend": {"cudnn_benchmark": False, "cudnn_deterministic": False,
                    "cudnn_tf32": True, "matmul_tf32": False, "threads": 2},
        "torch": torch.__version__, "numpy": np.__version__,
        "parameter_hash_before": parameter_hash,
    }
    write(out / "manifest.json", manifest)
    return manifest


def _report(summary):
    lines = [
        "# Seed4 local rollback follow-up", "",
        f"Run status: {summary['status']}. Training: none. Checkpoint: seed4 K8, 300 updates.", "",
        "The intervention restores one neighboring cell’s W/Z to its own preceding macro-step. The sham applies the wrong neighbor’s own rollback direction, scaled separately for W and Z to match the selected neighbor’s rollback norm in each source world.", "",
        "| Size | Events | Eligible maps | Horizon | Native − sender rollback | Sham − sender rollback | Native acquisition | Sender acquisition | Sham acquisition |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in SIZES:
        row = summary["sizes"].get(str(size), {})
        for horizon in (1, 4):
            h = row.get("horizons", {}).get(str(horizon), {})
            values = [h.get(k) for k in ("native_minus_sender_rollback", "sham_minus_sender_rollback", "native_acquisition", "sender_rollback_acquisition", "wrong_neighbor_sham_acquisition")]
            fmt = lambda v: "NA" if v is None else f"{100*v:.2f}%"
            lines.append(f"| {size} | {row.get('event_count', 0)} | {row.get('eligible_map_count', 0)} | {horizon} | " + " | ".join(fmt(v) for v in values) + " |")
    lines += [
        "", "Primary outcomes are paired correctness at p after one step; four steps are secondary. Effects average events within each eligible map, then average maps. Reused maps, times, and events are correlated; model replication count is one.",
        "", "This is a selected local temporal-state sensitivity contrast. Whole-cell W/Z rollback is not an edge-message knockout, and equal-norm sham does not match activation context or direction. The result does not establish causal flood fill, unique edge necessity, or neuron semantics. Off-cone invariance and the duplicated no-op trajectory are implementation qualifications only.",
        "", "Read compact_summary.json and replaychecks.json first; events.json is the event-level record.",
    ]
    return "\n".join(lines) + "\n"


def run(out_path):
    global DEADLINE
    out = Path(out_path).resolve()
    if not out.is_relative_to(ROOT / "runs"):
        raise ValueError("Output must be a new directory under project runs/")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    DEADLINE = started + HARD_CAP_SECONDS
    timer = threading.Timer(WATCHDOG_SECONDS, lambda: (write(out / "status.json", {"status": "WATCHDOG_TIMEOUT", "finished_utc": now()}), os._exit(124)))
    timer.daemon = True
    timer.start()
    try:
        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError("Frozen local CUDA requirement is unavailable")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        bindings = _historical_bindings(out)
        checkpoint = torch.load(bindings["checkpoint_path"], map_location="cpu", weights_only=True)
        if checkpoint["variant"] != "stream" or checkpoint["seed"] != 4 or checkpoint["completed_updates"] != 300:
            raise AssertionError("Wrong frozen seed4 checkpoint record")
        model = StreamingCell().eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        parameter_before = tensor_hash(model.state_dict())
        old_record = bindings["carry_record"]
        if parameter_before != old_record["final_parameter_sha256"]:
            raise AssertionError("Checkpoint parameter hash differs from historical record")
        model = model.cuda()

        data_cpu = {size: bank(size, MAPS, 40000 + size) for size in SIZES}
        data_hashes = {str(size): tensor_hash(data) for size, data in data_cpu.items()}
        if (data_hashes != bindings["old_manifest"]["evaluation_data_sha256"]
                or data_hashes != bindings["carry_evaluation_data_sha256"]):
            raise AssertionError("Evaluation bank does not match frozen historical tensors")
        _capture_manifest(out, bindings, data_hashes, model, parameter_before, started)
        data = {size: {key: value.cuda() for key, value in bank_data.items()} for size, bank_data in data_cpu.items()}

        # Untouched baseline, evaluated by the frozen stream-path evaluator/phase-II scorer.
        write(out / "status.json", {"status": "BASELINE_REPLAY", "updated_utc": now(), "pid": os.getpid()})
        _audit.DEADLINE = DEADLINE
        replay = {"status": "PENDING", "size_horizon_records": 0, "integer_leaves": 0,
                  "float_leaves": 0, "maximum_absolute_error": 0.0,
                  "comparison": "stream_path_audit.compare_tree with frozen stream_path_audit.evaluate/phase2 score"}
        for size in SIZES:
            measured = _audit.evaluate(model, data[size], "full")
            for t in (64, 128, 256):
                _audit.compare_tree(bindings["old"]["full"][str(size)][str(t)]["evaluation"],
                                    measured[str(t)]["evaluation"], f"{size}/{t}", replay)
                replay["size_horizon_records"] += 1
        if replay["size_horizon_records"] != 6:
            raise AssertionError("Historical baseline replay did not cover all six size/horizon records")
        replay["status"] = "PASS"
        write(out / "replaychecks.json", replay)

        # Snapshot pass is separate from the baseline evaluator and never changes the reference.
        snapshots = {}
        for size in SIZES:
            write(out / "status.json", {"status": "SNAPSHOTTING", "size": size, "updated_utc": now(), "pid": os.getpid()})
            snapshots[size] = capture_states(model, data[size])
        event_sets = {size: select_events(model, data_cpu[size], snapshots[size], size) for size in SIZES}
        events = [event for size in SIZES for event in event_sets[size]]
        counts = {str(size): {
            "events": len(event_sets[size]),
            "eligible_maps": len({e["map_index"] for e in event_sets[size]}),
            "events_by_time": {str(t): sum(e["time"] == t for e in event_sets[size]) for t in TIMES},
        } for size in SIZES}
        for event in events:
            event["outcomes"] = {}
        write(out / "events.json", {"selection_frozen_before_outcomes": True, "counts": counts, "events": events})

        insufficient = [size for size in SIZES if counts[str(size)]["events"] < MIN_EVENTS_PER_SIZE
                        or counts[str(size)]["eligible_maps"] < MIN_MAPS_PER_SIZE]
        if insufficient:
            summary = {
                "status": "INSUFFICIENT_MATCHED_EVENTS", "training": False,
                "reason": "Frozen event minima failed; no widening or extra sweep was performed.",
                "insufficient_sizes": insufficient, "event_minima": {"events": MIN_EVENTS_PER_SIZE, "eligible_maps": MIN_MAPS_PER_SIZE},
                "sizes": {str(size): {"event_count": counts[str(size)]["events"], "eligible_map_count": counts[str(size)]["eligible_maps"], "horizons": {}} for size in SIZES},
                "replay": replay, "parameters_unchanged": tensor_hash(model.state_dict()) == parameter_before,
                "elapsed_seconds": time.monotonic() - started,
                "interpretation": "No causal contrast estimated because the frozen matched-event population was below its minimum.",
            }
        else:
            write(out / "status.json", {"status": "INTERVENTIONS", "event_counts": counts, "updated_utc": now(), "pid": os.getpid()})
            for size in SIZES:
                run_event_outcomes(model, data[size], snapshots[size], event_sets[size])
                write(out / "events.json", {"selection_frozen_before_outcomes": True, "counts": counts, "events": events})
            sizes_summary = {str(size): aggregate(events, size) for size in SIZES}
            signal = all(
                sizes_summary[str(size)]["horizons"]["1"]["native_minus_sender_rollback"] >= .10
                and sizes_summary[str(size)]["horizons"]["1"]["sham_minus_sender_rollback"] >= .05
                for size in SIZES
            )
            summary = {
                "status": "COMPLETE", "training": False, "event_counts": counts,
                "sizes": sizes_summary, "primary_signal_thresholds_met": signal,
                "primary_signal_definition": "Both sizes: native−sender >=0.10 and sham−sender >=0.05 at step 1 after qualification.",
                "scientific_status": "DESCRIPTIVE_LOCAL_ROLLBACK_SIGNAL" if signal else "NO_PRIMARY_THRESHOLD_SIGNAL",
                "replay": replay,
                "parameters_unchanged": tensor_hash(model.state_dict()) == parameter_before,
                "checkpoint_unchanged": sha(bindings["checkpoint_path"]) == bindings["checkpoint_file_sha256"],
                "interpretation": "Selected-checkpoint local temporal W/Z intervention; not direct edge-message knockout, causal flood fill, unique-edge proof or semantic attribution.",
            }
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["model_replications"] = 1
        write(out / "compact_summary.json", summary)
        (out / "RESULTS.md").write_text(_report(summary), encoding="utf-8")
        if not summary["parameters_unchanged"]:
            raise AssertionError("Model parameters changed during no-training audit")
        if "checkpoint_unchanged" in summary and not summary["checkpoint_unchanged"]:
            raise AssertionError("Frozen checkpoint file changed")
        write(out / "status.json", {"status": summary["status"], "finished_utc": now(), "pid": os.getpid(),
                                     "elapsed_seconds": summary["elapsed_seconds"], "event_counts": counts,
                                     "replay": "PASS"})
        return summary
    except Exception as error:
        write(out / "error.json", {"error": repr(error), "traceback": traceback.format_exc()})
        write(out / "status.json", {"status": "TIME_BUDGET" if isinstance(error, TimeoutError) else "ERROR",
                                     "finished_utc": now(), "pid": os.getpid(),
                                     "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        timer.cancel()


def cpu_check():
    """One small, non-training semantic qualification for intervention helpers."""
    torch.set_num_threads(2)
    torch.manual_seed(49021)
    model = StreamingCell(workspace_channels=8, latent_channels=4, workspace_hidden=8, candidate_hidden=8).eval()
    with torch.no_grad():
        model.f_out.weight.normal_(std=.04); model.f_out.bias.normal_(std=.02)
        model.q_out.weight.normal_(std=.04); model.q_out.bias.normal_(std=.02)
    x = bank(8, 2, 39123)["x"]
    w = torch.randn(2, 8, 8, 8)
    z = torch.randn(2, 4, 8, 8)
    state = (w, z)
    saved = state_copy(state)
    inputs = x.clone()
    q = [(2, 2), (2, 3)]
    r = [(2, 3), (2, 4)]
    s = [(5, 5), (5, 6)]
    previous = (w.clone(), z.clone())
    previous[0][0, :, 2, 2] += torch.arange(1, 9, dtype=w.dtype) * .03
    previous[0][1, :, 2, 3] += torch.arange(1, 9, dtype=w.dtype) * .02
    previous[1][0, :, 2, 2] += torch.arange(1, 5, dtype=z.dtype) * .05
    previous[1][1, :, 2, 3] += torch.arange(1, 5, dtype=z.dtype) * .04
    previous[0][0, :, 2, 3] -= torch.arange(1, 9, dtype=w.dtype) * .02
    previous[0][1, :, 2, 4] -= torch.arange(1, 9, dtype=w.dtype) * .01
    previous[1][0, :, 2, 3] -= torch.arange(1, 5, dtype=z.dtype) * .02
    previous[1][1, :, 2, 4] -= torch.arange(1, 5, dtype=z.dtype) * .03

    qd_w = site_delta(previous[0], w, q)
    rd_w = site_delta(previous[0], w, r)
    assert bool((qd_w != 0).any()) and bool((rd_w != 0).any())
    for block in (0, 1):
        sham, q_delta, _ = scaled_rollback_sham(previous[block], state[block], previous[block], state[block], q, r)
        torch.testing.assert_close(torch.linalg.vector_norm(sham, dim=1), torch.linalg.vector_norm(q_delta, dim=1), atol=1e-6, rtol=1e-6)

    rollback = (replace_site(w, previous[0], q), replace_site(z, previous[1], q))
    expected_rollback = state_copy(state)
    for i, (y, xx) in enumerate(q):
        expected_rollback[0][i, :, y, xx] = previous[0][i, :, y, xx]
        expected_rollback[1][i, :, y, xx] = previous[1][i, :, y, xx]
    assert all(torch.equal(rollback[b], expected_rollback[b]) for b in (0, 1))
    assert torch.equal(x, inputs)
    # Restored pointwise logits equal those from t−1 exactly at each q.
    for i, (y, xx) in enumerate(q):
        torch.testing.assert_close(model.logits(rollback)[i, 0, y, xx], model.logits(previous)[i, 0, y, xx], atol=1e-6, rtol=0)

    # No-op copies and batched versus singleton forward paths are equivalent.
    noop = state_copy(state)
    assert all(torch.equal(a, b) for a, b in zip(state, noop))
    batch_out = model.step(rollback, x)
    singles = [model.step((rollback[0][i:i+1], rollback[1][i:i+1]), x[i:i+1]) for i in range(2)]
    for block in (0, 1):
        expected = torch.cat([single[block] for single in singles], dim=0)
        torch.testing.assert_close(batch_out[block], expected, atol=1e-6, rtol=1e-6)
    noop_out = model.step(noop, x)
    native_out = model.step(state, x)
    assert all(torch.equal(noop_out[b], native_out[b]) for b in (0, 1))

    # An off-cone local state change at Manhattan distance 4 leaves p's next-step logits fixed.
    q_one = [(2, 2), (2, 3)]
    s_one = [(5, 5), (5, 6)]
    qdw, qdz = site_delta(previous[0], w, q_one), site_delta(previous[1], z, q_one)
    off = (apply_site_delta(w, qdw, s_one), apply_site_delta(z, qdz, s_one))
    out_native = model.step(state, x)
    out_off = model.step(off, x)
    for i in range(2):
        py, px = (2, 2)
        torch.testing.assert_close(model.logits(out_native)[i, 0, py, px], model.logits(out_off)[i, 0, py, px], atol=1e-6, rtol=0)
    assert all(torch.equal(a, b) for a, b in zip(state, saved)) and torch.equal(x, inputs)
    mixed = ([{"time": 16, "event_id": f"a{i}"} for i in range(CHUNK_EVENTS + 1)]
             + [{"time": 32, "event_id": "b0"}])
    batches = list(event_batches(mixed))
    assert [(t, len(chunk)) for t, chunk in batches] == [(16, CHUNK_EVENTS), (16, 1), (32, 1)]
    assert all(all(row["time"] == t for row in chunk) for t, chunk in batches)
    return {
        "status": "PASS", "training": False, "nonzero_state": True,
        "blockwise_sham_norm_match": True, "rollback_only_selected_sites": True,
        "q_rollback_logits_match_previous": True, "input_unchanged": True,
        "batch_independent": True, "offcone_step1_invariant": True,
        "mixed_time_pipeline_batching": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = run(args.out)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
