"""Frozen, zero-training gauge-aware producer/consumer continuation audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
for rel in ('new/nca_inertial_wind_tunnel', 'new/workspace_revision',
            'new/short_bptt', 'new/streaming_carry', 'new/seed4_followup'):
    sys.path.append(str(ROOT / rel))
from stream_cells import StreamingCell
from tasks import bank, subset, per_example_balanced_accuracy
from run_revision import tensor_hash
from phenotype import summarize_from_traces, predicate, _write_frontier_csv
from alignment import (fit_alignment, apply_alignment, identity_alignment,
                       alignment_diagnostics, cycle_diagnostics, exact_gauge_clone)
from metrics import summarize_handoff, decompose
from gradients import audit_gradients

PROTOCOL = 'continuation_interface_v1'
OLD = ROOT / 'runs/bootstrap_20261004_seed4_01'
FRESH = ROOT / 'runs/trajectory_20261004_serial01/stage2_fresh_recipe'
BATCH = 8
SHORT_TIMES = tuple(range(8, 65, 8))
FIT_TIMES = tuple(range(8, 49, 8))
VALID_TIMES = (56, 64)
PERM = np.random.default_rng(95008).permutation(8)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ready(value):
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return ready(value.tolist())
    if isinstance(value, np.generic):
        return ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(ready(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    core = ['new/nca_inertial_wind_tunnel/tasks.py', 'new/workspace_revision/revision_cells.py',
            'new/workspace_revision/run_revision.py', 'new/masked_medium/masked_cells.py',
            'new/short_bptt/training.py', 'new/streaming_carry/stream_cells.py',
            'new/seed4_followup/phenotype.py', 'new/frontier_audit/audit.py',
            'new/frontier_audit/metrics.py']
    old = read(ROOT / 'TRAJECTORY_QUALIFICATION_PUBLICATION_MANIFEST.json')['source_sha256']
    core += [rel for rel in old if rel.startswith('new/') and rel.endswith('.py')]
    for rel in core:
        if rel in old:
            assert sha(ROOT / rel) == old[rel], f'Frozen scientific source drift: {rel}'
    names = core + [p.relative_to(ROOT).as_posix() for p in
                    (ROOT / 'new/continuation_interface').glob('*') if p.suffix in ('.py', '.md')]
    launcher = ROOT / 'tools/launch_continuation_interface.ps1'
    if launcher.exists():
        names.append(launcher.relative_to(ROOT).as_posix())
    return {rel: sha(ROOT / rel) for rel in sorted(set(names))}


def catalog():
    rows = {}
    for mid, name in (('S1', 'H'), ('F1', 'H_swap_early8')):
        record = read(OLD / f'{name}.json')
        cp = record['checkpoints']['300']
        rows[mid] = {'id': mid, 'name': name, 'path': (OLD / cp['path']).relative_to(ROOT).as_posix(),
                     'sha256': cp['sha256'], 'parameter_sha256': cp['parameter_sha256'],
                     'prior_class': 'success' if mid == 'S1' else 'failure'}
    for record in read(FRESH / 'perarm.json'):
        if record['arm'] != 'baseline':
            continue
        cp = record['checkpoints']['baseline_u300']
        mid = 'S2' if record['initialization_seed'] == 31006 else f"B{record['initialization_seed']}"
        rows[mid] = {'id': mid, 'name': record['pair_id'] + '_baseline',
                     'initialization_seed': record['initialization_seed'],
                     'path': (FRESH / cp['path']).relative_to(ROOT).as_posix(),
                     'sha256': cp['sha256'], 'parameter_sha256': cp['parameter_sha256'],
                     'prior_class': 'success' if record['phenotype_pass'] else 'failure'}
    assert rows['S2']['prior_class'] == 'success' and len(rows) == 18
    for row in rows.values():
        assert sha(ROOT / row['path']) == row['sha256'], row['id']
    return rows


def load_model(row):
    payload = torch.load(ROOT / row['path'], map_location='cpu', weights_only=False)
    assert payload['completed_updates'] == 300
    model = StreamingCell().eval()
    model.load_state_dict(payload['state_dict'])
    assert tensor_hash(model.state_dict()) == row['parameter_sha256'], row['id']
    return model


def setup():
    assert torch.cuda.is_available() and str(torch.__version__).startswith('2.5.1')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


def per_map_loss(logits, y, mask):
    raw = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    pos, neg = mask * y, mask * (1-y)
    dims = (1, 2, 3)
    np_, nn_ = pos.sum(dims), neg.sum(dims)
    lp = (raw * pos).sum(dims) / np_.clamp_min(1)
    ln = (raw * neg).sum(dims) / nn_.clamp_min(1)
    return (lp + ln) / ((np_ > 0).float() + (nn_ > 0).float()).clamp_min(1)


def inputs(data, ids, device='cuda'):
    b = subset(data, ids)
    return (torch.cat((b['x'], b['x_flip'])).to(device),
            torch.cat((b['y'], b['y_flip'])).to(device),
            torch.cat((b['mask'], b['mask'])).to(device))


@torch.no_grad()
def short_evaluate(model, data, flipped, save_logits):
    count, n = data['x'].shape[0], data['x'].shape[-1]
    cues = 2 if flipped else 1
    losses = np.empty((cues, len(SHORT_TIMES), count), np.float64)
    bas = np.empty_like(losses)
    logits_all = np.empty((cues, len(SHORT_TIMES), count, n, n), np.float32) if save_logits else None
    model.cuda()
    for start in range(0, count, BATCH):
        ids = list(range(start, min(start+BATCH, count)))
        if flipped:
            x, y, mask = inputs(data, ids)
        else:
            b = subset(data, ids)
            x, y, mask = (b[k].cuda() for k in ('x', 'y', 'mask'))
        state = model.initial(x)
        for t in range(1, 65):
            state = model.step(state, x)
            if t not in SHORT_TIMES:
                continue
            logits = model.logits(state)
            assert bool(torch.isfinite(logits).all()) and all(bool(torch.isfinite(s).all()) for s in state)
            k, num = SHORT_TIMES.index(t), len(ids)
            losses[:, k, start:start+num] = per_map_loss(logits, y, mask).cpu().numpy().reshape(cues, num)
            bas[:, k, start:start+num] = per_example_balanced_accuracy(logits, y, mask).cpu().numpy().reshape(cues, num)
            if logits_all is not None:
                logits_all[:, k, start:start+num] = logits[:, 0].cpu().numpy().reshape(cues, num, n, n)
    model.cpu()
    return {'loss_per_map': losses, 'ba_per_map': bas, 'logits': logits_all,
            'loss_mean': float(losses.mean()), 'loss_by_time': losses.mean((0, 2)).tolist(),
            'ba_by_time': bas.mean((0, 2)).tolist(), 'times': SHORT_TIMES}


def disagreement(a, b, data):
    mask = data['mask'][:, 0].numpy()
    denom = mask.sum((1, 2))
    pa, pb = torch.sigmoid(torch.from_numpy(a)).numpy(), torch.sigmoid(torch.from_numpy(b)).numpy()
    mse = (((pa-pb)**2 * mask[None, None]).sum((-1, -2)) / denom[None, None]).mean((0, 1))
    flip = (((a >= 0) != (b >= 0)) * mask[None, None]).sum((-1, -2)) / denom[None, None]
    return {'probability_mse': float(mse.mean()), 'probability_rms': float(np.sqrt(mse.mean())),
            'decision_disagreement': float(flip.mean()), 'per_map_probability_mse': mse.tolist(),
            'per_map_decision_disagreement': flip.mean((0, 1)).tolist()}


@torch.no_grad()
def capture(model, data, times):
    count = data['x'].shape[0]
    pieces = {t: [[], []] for t in times}
    model.cuda()
    for start in range(0, count, BATCH):
        ids = list(range(start, min(start+BATCH, count)))
        x, _, _ = inputs(data, ids)
        state = model.initial(x)
        for t in range(1, max(times)+1):
            state = model.step(state, x)
            if t in times:
                for j in (0, 1):
                    pieces[t][j].append(state[j].cpu().numpy())
    model.cpu()
    # Each batch is [original batch, flipped batch]; restore global cue ordering.
    result = []
    for j in (0, 1):
        time_arrays = []
        for t in times:
            originals, flips = [], []
            for v in pieces[t][j]:
                a, b = np.split(v, 2)
                originals.append(a); flips.append(b)
            time_arrays.append(np.concatenate((np.concatenate(originals), np.concatenate(flips))))
        result.append(np.stack(time_arrays))
    assert all(np.isfinite(v).all() for v in result)
    return tuple(result)


def save_states(path, states, times):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, W=states[0], Z=states[1], times=np.asarray(times))


@torch.no_grad()
def paired_trace(model, data, start_state=None, adapter=None, start_time=0):
    count, n = data['x'].shape[0], data['x'].shape[-1]
    length = 257-start_time
    original = np.empty((length, count, n, n), bool)
    flipped = np.empty_like(original)
    records = {t: {'loss': [], 'ba': []} for t in (64, 128, 256)}
    state64 = [[], []]
    endpoint_logits = {t: [] for t in (64, 128, 256)}
    model.cuda()
    for start in range(0, count, BATCH):
        ids = list(range(start, min(start+BATCH, count)))
        x, y, mask = inputs(data, ids)
        if start_state is None:
            state = model.initial(x)
        else:
            indexes = ids + [i+count for i in ids]
            state = tuple(torch.from_numpy(v[indexes]).cuda() for v in start_state)
            if adapter is not None:
                state = apply_alignment(state, adapter)
        for t in range(start_time, 257):
            if t > start_time:
                state = model.step(state, x)
            logits = model.logits(state)
            if not bool(torch.isfinite(logits).all()) or (t % 8 == 0 and not all(bool(torch.isfinite(v).all()) for v in state)):
                model.cpu()
                raise FloatingPointError(f'Nonfinite continuation at absolute step {t}')
            good = ((logits >= 0) == (y >= .5))[:, 0].cpu().numpy()
            original[t-start_time, ids] = good[:len(ids)]
            flipped[t-start_time, ids] = good[len(ids):]
            if t == 64 and start_state is None:
                for j in (0, 1):
                    state64[j].append(state[j].cpu().numpy())
            if t in records:
                records[t]['loss'].append(per_map_loss(logits, y, mask).cpu().numpy().reshape(2, len(ids)))
                records[t]['ba'].append(per_example_balanced_accuracy(logits, y, mask).cpu().numpy().reshape(2, len(ids)))
                endpoint_logits[t].append(logits[:, 0].cpu().numpy().reshape(2, len(ids), n, n))
    model.cpu()
    summarized = {}
    for t, vals in records.items():
        loss, ba = (np.concatenate(vals[k], axis=1) for k in ('loss', 'ba'))
        summarized[str(t)] = {'original': {'balanced_accuracy': float(ba[0].mean()), 'bce': float(loss[0].mean())},
                               'flipped': {'balanced_accuracy': float(ba[1].mean()), 'bce': float(loss[1].mean())}}
    saved_state = []
    if start_state is None:
        for j in (0, 1):
            a, b = zip(*(np.split(v, 2) for v in state64[j]))
            saved_state.append(np.concatenate((np.concatenate(a), np.concatenate(b))))
    ends = {str(t): np.concatenate(v, axis=1) for t, v in endpoint_logits.items()}
    return {'correct': original & flipped, 'original_correct': original, 'flipped_correct': flipped}, summarized, tuple(saved_state), ends


@torch.no_grad()
def gauge_control(model, data, steps=256):
    model.cuda()
    clone = exact_gauge_clone(model, PERM).cuda()
    x, y, mask = inputs(data, list(range(data['x'].shape[0])))
    a, b = model.initial(x), clone.initial(x)
    known = {'W': np.eye(6), 'Z': np.eye(8)[PERM]}
    maxima = {'native_W': 0., 'native_Z': 0., 'native_logits': 0., 'aligned_W': 0., 'aligned_Z': 0., 'aligned_logits': 0.}
    agreements = {'native': [0, 0], 'aligned': [0, 0], 'raw': [0, 0]}
    aligned = raw = None
    for t in range(steps+1):
        if t:
            a, b = model.step(a, x), clone.step(b, x)
            if t > 64:
                aligned, raw = clone.step(aligned, x), clone.step(raw, x)
        expected = apply_alignment(a, known)
        logits = model.logits(a)
        if t == 64:
            aligned = tuple(v.clone() for v in expected)
            raw = tuple(v.clone() for v in a)
        for mode, state in [('native', b)] + ([('aligned', aligned), ('raw', raw)] if t >= 64 else []):
            output = clone.logits(state)
            assert bool(torch.isfinite(output).all())
            correct_a = ((logits >= 0) == (y >= .5))
            correct_b = ((output >= 0) == (y >= .5))
            # Paired correctness, not branchwise agreement.
            num = data['x'].shape[0]
            pair_a = correct_a[:num] & correct_a[num:]
            pair_b = correct_b[:num] & correct_b[num:]
            pick = data['changed'].cuda().bool()
            agreements[mode][0] += int(((pair_a == pair_b) & pick).sum())
            agreements[mode][1] += int(pick.sum())
            if mode != 'raw':
                for key, s, e in [('W', state[0], expected[0]), ('Z', state[1], expected[1]), ('logits', output, logits)]:
                    error = float((s-e).abs().max() / (1 + e.abs().max()))
                    maxima[f'{mode}_{key}'] = max(maxima[f'{mode}_{key}'], error)
    model.cpu(); clone.cpu()
    ratios = {k: {'numerator': v[0], 'denominator': v[1], 'value': v[0]/v[1] if v[1] else None} for k, v in agreements.items()}
    passed = max(maxima.values()) <= 1e-3 and all(ratios[k]['value'] is not None and ratios[k]['value'] >= .999 for k in ('native', 'aligned'))
    return {'status': 'PASS' if passed else 'GAUGE_CONTROL_UNQUALIFIED', 'permutation': PERM,
            'max_normalized_discrepancy': maxima, 'paired_agreement': ratios, 'steps': steps,
            'raw_failure_required': False}


def make_banks(out):
    specs = {'train': (32, 512, 10002), 'short': (32, 32, 91032), 'fit': (32, 16, 92032),
             'validation': (32, 16, 93032), 'test32': (32, 32, 94032), 'test64': (64, 32, 94064)}
    data, records, seen = {}, {}, set()
    for name, (n, count, seed) in specs.items():
        data[name] = bank(n, count, seed)
        collisions = 0
        for x in data[name]['x']:
            key = hashlib.sha256(x.numpy().tobytes()).hexdigest()
            if key in seen:
                collisions += 1
            seen.add(key)
        assert collisions == 0, f'Sampled bank collision: {name}'
        records[name] = {'size': n, 'count': count, 'seed': seed, 'data_sha256': tensor_hash(data[name]), 'cross_role_collisions': collisions}
        (out/'banks').mkdir(exist_ok=True)
        np.savez_compressed(out/'banks'/f'{name}.npz', **{k: v.numpy() for k, v in data[name].items()})
    return data, records


def transfer_gate(summary, native):
    checks = {}
    for size in ('32', '64'):
        s, ref = summary[size]['bands']['all_changed'], native[size]['bands']['all_changed']
        p, g = s['preservation']['pooled']['value'], s['sustained_progress']['pooled']['value']
        pr, gr = ref['preservation']['pooled']['value'], ref['sustained_progress']['pooled']['value']
        checks[f'{size}_preservation'] = p is not None and p >= .95
        checks[f'{size}_sustained_progress'] = g is not None and g >= .10
        checks[f'{size}_progress_eligible'] = s['sustained_progress']['valid_maps'] >= 16 and s['sustained_progress']['pooled']['denominator'] >= 100
        checks[f'{size}_preservation_vs_native'] = p is not None and pr is not None and p >= pr-.05
        checks[f'{size}_progress_vs_native'] = g is not None and gr is not None and g >= gr-.05
    return {'pass': all(checks.values()), 'checks': checks}


def run(out, qualification):
    setup()
    q = read(qualification)
    assert q['status'] == 'PASS' and q['protocol'] == PROTOCOL
    sources, rows = source_hashes(), catalog()
    assert sources == q['source_sha256']
    assert {k: v['sha256'] for k, v in rows.items()} == q['reference_checkpoint_sha256']
    out = out.resolve()
    assert out.parent == ROOT/'runs' and not out.exists(), 'Use a new direct run directory'
    out.mkdir(parents=True)
    begun = time.monotonic()
    def status(phase, **extra):
        write(out/'status.json', {'status': 'RUNNING', 'protocol': PROTOCOL, 'phase': phase,
                                 'pid': os.getpid(), 'updated_utc': now(), **extra})
        print(json.dumps({'phase': phase, **extra}), flush=True)
    manifest = {'protocol': PROTOCOL, 'status': 'RUNNING', 'pid': os.getpid(), 'started_utc': now(),
                'source_sha256': sources, 'models': rows, 'training': False,
                'runtime_limit_enforced': False, 'torch_version': str(torch.__version__),
                'numpy_version': np.__version__, 'gpu': torch.cuda.get_device_name(0),
                'backend': {'cudnn_benchmark': False, 'cudnn_deterministic': False, 'cudnn_tf32': True, 'matmul_tf32': False},
                'qualification_sha256': sha(qualification)}
    write(out/'manifest.json', manifest)
    for rel in sources:
        dest = out/'source'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, dest)
    try:
        status('banks')
        data, bank_records = make_banks(out)
        manifest['banks'] = bank_records; write(out/'manifest.json', manifest)
        models = {k: load_model(v) for k, v in rows.items()}
        objective, short_outputs = {}, {}
        for index, (mid, model) in enumerate(models.items()):
            status('phase0', model=mid, completed_models=index, total_models=18)
            train = short_evaluate(model, data['train'], False, False)
            short = short_evaluate(model, data['short'], True, True)
            short_outputs[mid] = short.pop('logits'); train.pop('logits')
            objective[mid] = {'train': train, 'short': short}
            write(out/'phase0'/f'{mid}.json', objective[mid])
            np.savez_compressed(out/'phase0'/f'{mid}_short_logits.npz', logits=short_outputs[mid], times=SHORT_TIMES)
        comparisons = {}
        ids = list(models)
        for p in ids:
            for c in ids:
                comparisons[f'{p}__{c}'] = disagreement(short_outputs[p], short_outputs[c], data['short'])
        candidates = [mid for mid in ids if mid.startswith('B')]
        f2 = min(candidates, key=lambda mid: (comparisons[f'{mid}__S1']['probability_mse'], rows[mid]['initialization_seed']))
        f3 = min((mid for mid in candidates if mid != f2), key=lambda mid: (comparisons[f'{mid}__S2']['probability_mse'], rows[mid]['initialization_seed']))
        selection = {'models': ['S1','S2','F1','F2','F3'], 'F2_source': f2, 'F3_source': f3,
                     'selection_times': SHORT_TIMES, 'new_long_outcomes_used': False,
                     'rule': 'F2 nearest S1 probability MSE; F3 nearest S2 excluding F2; seed tie-break'}
        write(out/'phase0'/'comparisons.json', comparisons)
        write(out/'selection.json', selection)
        selected = {'S1': models['S1'], 'S2': models['S2'], 'F1': models['F1'], 'F2': models[f2], 'F3': models[f3]}
        selected_sources = {'S1': 'S1', 'S2': 'S2', 'F1': 'F1', 'F2': f2, 'F3': f3}
        del models, short_outputs
        gc.collect()
        status('gauge_control')
        gauge = gauge_control(selected['S1'], subset(data['fit'], list(range(4))))
        write(out/'gauge_control.json', gauge)
        assert gauge['status'] == 'PASS', 'GAUGE_CONTROL_UNQUALIFIED'
        status('alignment_capture')
        calibration, validation = {}, {}
        for mid, model in selected.items():
            calibration[mid] = capture(model, data['fit'], FIT_TIMES)
            validation[mid] = capture(model, data['validation'], VALID_TIMES)
            save_states(out/'states'/f'{mid}_calibration.npz', calibration[mid], FIT_TIMES)
            save_states(out/'states'/f'{mid}_validation.npz', validation[mid], VALID_TIMES)
        mask_fit = np.concatenate((data['fit']['mask'].numpy(), data['fit']['mask'].numpy()))
        mask_val = np.concatenate((data['validation']['mask'].numpy(), data['validation']['mask'].numpy()))
        adapters, alignment_records = {}, {}
        for p in selected:
            for c in selected:
                key = f'{p}__{c}'
                adapters[key] = identity_alignment() if p == c else fit_alignment(calibration[p], calibration[c], mask_fit)
                alignment_records[key] = {'fit': adapters[key].get('fit'),
                                         'validation': alignment_diagnostics(validation[p], validation[c], mask_val, adapters[key])}
        (out/'alignment').mkdir(exist_ok=True)
        for key, adapter in adapters.items():
            np.savez_compressed(out/'alignment'/f'{key}.npz', W=adapter['W'], Z=adapter['Z'])
            p,c = key.split('__')
            record = alignment_records[key]
            record['cycle'] = cycle_diagnostics(adapter, adapters[f'{c}__{p}'])
            v = record['validation']
            record['restricted_map_qualified'] = all(v['normalized_rms_residual'][b] <= .10 and
                v['matrix_diagnostics'][b]['condition_number'] <= 100 and
                record['cycle'][b]['relative_frobenius_error'] <= .10 for b in ('W','Z'))
        write(out/'alignment'/'validation.json', alignment_records)
        del calibration, validation
        gc.collect()
        native, states, native_phenotypes, cohorts = {}, {}, {}, {}
        banks = {32: data['test32'], 64: data['test64']}
        for mid, model in selected.items():
            traces, endpoint_records = {}, {}
            native[mid], states[mid], cohorts[mid] = {}, {}, {}
            for size, d in banks.items():
                status('native', model=mid, size=size)
                trace, endpoints, snapshot, logits = paired_trace(model, d)
                traces[str(size)], endpoint_records[str(size)] = trace, endpoints
                native[mid][str(size)] = summarize_handoff(trace['correct'][64:], trace['correct'][64], d['changed'].numpy(), d['distance'].numpy())
                cohorts[mid][size] = trace['correct'][64].copy()
                states[mid][size] = snapshot
                folder = out/'native'; folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'{mid}_size{size}_trace.npz', **trace)
                save_states(out/'states'/f'{mid}_test{size}_T64.npz', tuple(v[None] for v in snapshot), (64,))
                np.savez_compressed(folder/f'{mid}_size{size}_logits.npz', **{'T'+t: v for t,v in logits.items()})
            ph = summarize_from_traces(traces, endpoint_records, banks)
            frontier = ph.pop('_matched_frontier_rows')
            ph['phenotype_gate'] = predicate(ph)
            native_phenotypes[mid] = ph
            _write_frontier_csv(out/'native'/f'{mid}_frontier.csv', frontier)
            write(out/'native'/f'{mid}_phenotype.json', ph)
            write(out/'native'/f'{mid}_handoff.json', native[mid])
        common_qualified = all(native_phenotypes[s]['phenotype_gate']['pass'] for s in ('S1','S2'))
        records = {}
        for mode in ('raw','aligned'):
            for p in selected:
                for c in selected:
                    key = f'{mode}__{p}__{c}'
                    record = {'producer': p, 'consumer': c, 'mode': mode, 'status': 'COMPLETE', 'sizes': {}}
                    for size,d in banks.items():
                        status('cross_consumer', mode=mode, producer=p, consumer=c, size=size, completed_cells=len(records), total_cells=50)
                        if p == c:
                            record['sizes'][str(size)] = native[p][str(size)]
                            continue
                        try:
                            trace, endpoints, _, logits = paired_trace(selected[c], d, states[p][size],
                                adapters[f'{p}__{c}'] if mode == 'aligned' else None, 64)
                            record['sizes'][str(size)] = summarize_handoff(trace['correct'], cohorts[p][size], d['changed'].numpy(), d['distance'].numpy())
                            folder = out/'cross'; folder.mkdir(exist_ok=True)
                            np.savez_compressed(folder/f'{key}_size{size}_trace.npz', correct=trace['correct'], times=np.arange(64,257))
                            np.savez_compressed(folder/f'{key}_size{size}_logits.npz', **{'T'+t:v for t,v in logits.items()})
                        except FloatingPointError as exc:
                            record['status'] = 'NONFINITE_CONTINUATION'
                            record['sizes'][str(size)] = {'status': 'NONFINITE_CONTINUATION', 'reason': str(exc)}
                    record['transfer_gate'] = transfer_gate(record['sizes'], native[p]) if record['status'] == 'COMPLETE' else {'pass': False, 'reason': 'nonfinite continuation'}
                    records[key] = record
                    write(out/'cross'/f'{key}.json', record)
        decompositions = {}
        for mode in ('raw','aligned'):
            for size in ('32','64'):
                for metric in ('preservation','sustained_progress','coverage256'):
                    matrix = [[records[f'{mode}__{p}__{c}']['sizes'][size].get('bands',{}).get('all_changed',{}).get(metric,{}).get('pooled',{}).get('value') for c in selected] for p in selected]
                    decompositions[f'{mode}_size{size}_{metric}'] = {'model_order': list(selected), 'matrix': matrix, 'decomposition': decompose(matrix)}
        write(out/'cross'/'decomposition.json', decompositions)
        status('frozen_gradients')
        gradients = audit_gradients(selected, data['short'], [list(range(i,i+8)) for i in range(0,32,8)], ('S1','F1'), out/'gradients')
        for mid, model in selected.items():
            assert tensor_hash(model.state_dict()) == rows[selected_sources[mid]]['parameter_sha256'], f'Parameters changed: {mid}'
        short_flags = {}
        for success in ('S1','S2'):
            for failure in ('F1','F2','F3'):
                ss, fs = selected_sources[success], selected_sources[failure]
                cmp = comparisons[f'{ss}__{fs}']
                train_gap = abs(objective[ss]['train']['loss_mean']-objective[fs]['train']['loss_mean'])
                short_gap = abs(objective[ss]['short']['loss_mean']-objective[fs]['short']['loss_mean'])
                short_flags[f'{success}__{failure}'] = {'train_loss_gap': train_gap, 'short_loss_gap': short_gap,
                    'probability_rms': cmp['probability_rms'], 'decision_disagreement': cmp['decision_disagreement'],
                    'matched_observables_flag': train_gap <= .02 and short_gap <= .02 and cmp['probability_rms'] <= .05 and cmp['decision_disagreement'] <= .05}
                gaps = {size: abs(native[success][size]['bands']['all_changed']['coverage256']['pooled']['value']-
                                  native[failure][size]['bands']['all_changed']['coverage256']['pooled']['value']) for size in ('32','64')}
                short_flags[f'{success}__{failure}']['long_coverage_gaps'] = gaps
                short_flags[f'{success}__{failure}']['observational_underidentification_pattern'] = (
                    short_flags[f'{success}__{failure}']['matched_observables_flag'] and any(v >= .25 for v in gaps.values()))
        summary = {'protocol': PROTOCOL, 'status': 'COMPLETE', 'training': False, 'selection': selection,
                   'common_success_full_qualified': common_qualified,
                   'common_native_full_passes': {k:v['phenotype_gate']['pass'] for k,v in native_phenotypes.items()},
                   'gauge_control_status': gauge['status'], 'alignment_validation': alignment_records,
                   'cross_transfer_passes': {k:v['transfer_gate']['pass'] for k,v in records.items()},
                   'short_observable_comparisons': short_flags,
                   'gradient_diagnostics': gradients, 'elapsed_seconds': time.monotonic()-begun,
                   'claim_boundary': 'Selected checkpoint interface audit; no causal separation of objective, credit, and optimizer mechanisms, no general shared-algorithm or reliability claim.'}
        write(out/'summary.json', summary)
        table = ['# Continuation interface audit', '', f"Status: COMPLETE. Common native Full qualification: {common_qualified}.",
                 f"Gauge control: {gauge['status']}. No model training or optimizer steps.", '',
                 '| Mode | Producer | Consumer | Transfer proxy |', '|---|---|---|---:|']
        for r in records.values():
            table.append(f"| {r['mode']} | {r['producer']} | {r['consumer']} | {r['transfer_gate']['pass']} |")
        table += ['', 'Read summary.json, selection.json, phase0/comparisons.json, alignment/validation.json,',
                  'native/*_phenotype.json, cross/decomposition.json and per-cell JSON before the larger arrays.',
                  'Negative alignment qualifies only the tested restricted linear interface family.']
        (out/'RESULTS.md').write_text('\n'.join(table)+'\n', encoding='utf-8')
        manifest.update(status='COMPLETE', finished_utc=now(), elapsed_seconds=summary['elapsed_seconds'])
        write(out/'manifest.json', manifest)
        write(out/'status.json', {'status':'COMPLETE','protocol':PROTOCOL,'phase':'complete','pid':os.getpid(),'updated_utc':now(), 'completed_cells':50,'elapsed_seconds':summary['elapsed_seconds']})
        print(json.dumps({'status':'COMPLETE','elapsed_seconds':summary['elapsed_seconds']}),flush=True)
    except Exception as exc:
        write(out/'status.json', {'status':'ERROR','protocol':PROTOCOL,'pid':os.getpid(),'updated_utc':now(),'reason':str(exc),'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--qualification', required=True)
    args = parser.parse_args()
    run(ROOT / args.out, ROOT / args.qualification)
