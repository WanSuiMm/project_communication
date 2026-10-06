"""Boolean paired recorder and dense metrics; no changes to frozen evaluators."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'new/trajectory_qualification'))
import common as C
from phenotype import (_array, _coverage, _transition, _first_and_stable, _ratio,
                       summarize_from_traces, predicate, _write_frontier_csv, _json_ready)


@torch.no_grad()
def trace(model, data, size):
    """Same per-world batch shape/tie rule as the old trace; one CPU transfer."""
    shape = (257, len(data['x']), size, size)
    arrays = {key: torch.empty(shape, dtype=torch.bool, device='cuda')
              for key in ('correct', 'original_correct', 'flipped_correct')}
    a, b = model.initial(data['x']), model.initial(data['x_flip'])
    finite = torch.ones((), dtype=torch.bool, device='cuda')
    violation = torch.zeros((), dtype=torch.bool, device='cuda')
    outside_delta = torch.zeros((), device='cuda')
    records = {}
    score = C._load_audit().score
    for t in range(257):
        if t:
            a, b = model.step(a, data['x']), model.step(b, data['x_flip'])
        for value in (*a, *b):
            finite = finite & torch.isfinite(value).all()
        logits, flipped = model.logits(a), model.logits(b)
        finite = finite & torch.isfinite(logits).all() & torch.isfinite(flipped).all()
        good_a = (logits >= 0) == (data['y'] >= .5)
        good_b = (flipped >= 0) == (data['y_flip'] >= .5)
        both = good_a & good_b
        arrays['correct'][t].copy_(both[:, 0])
        arrays['original_correct'][t].copy_(good_a[:, 0])
        arrays['flipped_correct'][t].copy_(good_b[:, 0])
        outside = data['changed'].bool() & (data['distance'] > 2*t)
        violation = violation | (both & outside).any()
        outside_delta = torch.maximum(outside_delta, ((logits-flipped).abs()*outside).max())
        if t in (64, 128, 256):
            records[str(t)] = {'original': score(logits, data['y'], data['mask']),
                               'flipped': score(flipped, data['y_flip'], data['mask'])}
    if not bool(finite):
        raise FloatingPointError('Nonfinite paired rollout')
    assert not bool(violation), 'Paired correctness outside two-hop causal cone'
    assert float(outside_delta) <= 1e-6, 'Source difference outside causal cone'
    return {key: value.cpu().numpy() for key, value in arrays.items()}, records


def pack(path, values):
    """Retain all integer times and both branch traces, exactly and compactly."""
    items = {}
    for key, value in values.items():
        array = np.asarray(value, dtype=bool)
        items[key+'_shape'] = np.asarray(array.shape, dtype=np.int32)
        items[key+'_packed'] = np.packbits(array.reshape(-1), bitorder='little')
    np.savez_compressed(path, **items)


def survival(values, bank):
    correct = values['correct']
    selected = _array(bank['changed'])[:, 0].astype(bool)
    before = correct[64] & selected
    stayed = before & np.all(correct[64:], axis=0)
    return _ratio(int(stayed.sum()), int(before.sum()))


def dense_summary(traces, records, banks):
    summary = {'schema': 'continuous-coverage-dense-v1', 'sizes': {},
               'phenotype_gate': None, 'full_evaluated': False,
               'claim_boundary': 'Predeclared formation diagnostic; no checkpoint selection.'}
    for size in (32, 64):
        values, bank = traces[str(size)], banks[size]
        correct = values['correct']
        changed = _array(bank['changed'])[:, 0].astype(bool)
        distance = _array(bank['distance'])[:, 0]
        strict = changed & (distance > 16) & (distance < 32)
        first, _, relapse = _first_and_stable(correct)
        row = {'maps': len(changed), 'changed_pixels': int(changed.sum()),
               'endpoints': {}, 'transitions': {'all_changed': {}, 'strict_16_32': {}},
               'continuous_survival64_to256': survival(values, bank)}
        for t in (64, 128, 256):
            rec = records[str(size)][str(t)]
            row['endpoints'][str(t)] = {
                'original_ba': rec['original']['balanced_accuracy'],
                'flipped_ba': rec['flipped']['balanced_accuracy'],
                'all_changed': _coverage(correct[t], changed),
                'strict_16_32': _coverage(correct[t], strict)}
        for label, selection in (('all_changed', changed), ('strict_16_32', strict)):
            for start, end in ((64, 128), (64, 256), (128, 256)):
                row['transitions'][label][f'{start}_to_{end}'] = _transition(correct, selection, start, end)
        ever = changed & (first >= 0)
        row['ever_regressed_over_ever_correct'] = _ratio(int((changed & relapse).sum()), int(ever.sum()))
        summary['sizes'][str(size)] = row
    return summary


def evaluate(model, banks, folder, full=False):
    folder.mkdir(parents=True, exist_ok=False)
    model.eval()
    before = C.tensor_hash(model.state_dict())
    traces, records = {}, {}
    for size in (32, 64):
        data = {key: value.cuda() for key, value in banks[size].items()}
        values, rec = trace(model, data, size)
        traces[str(size)], records[str(size)] = values, rec
        pack(folder / f'size{size}_traces.npz', values)
        del data
    if full:
        summary = summarize_from_traces(traces, records, banks)
        frontier = summary.pop('_matched_frontier_rows')
        summary['phenotype_gate'] = predicate(summary)
        summary['full_evaluated'] = True
        for size in (32, 64):
            summary['sizes'][str(size)]['continuous_survival64_to256'] = survival(traces[str(size)], banks[size])
        _write_frontier_csv(folder / 'matched_frontier_strata.csv', frontier)
    else:
        summary = dense_summary(traces, records, banks)
    summary = _json_ready(summary)
    C.write(folder / 'summary.json', summary)
    assert C.tensor_hash(model.state_dict()) == before, 'Evaluation mutated parameters'
    return summary
