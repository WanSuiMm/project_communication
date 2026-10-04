"""Frozen 23-arm schedule interventions; no model or optimizer changes."""
from collections import Counter
import numpy as np

UPDATES = 300
BATCH = 8
LENGTHS = (1, 3, 8)
ALTERNATES = (20012, 20022)


def base_schedules():
    return {seed: np.random.default_rng(seed).integers(0, 512, (UPDATES, BATCH))
            for seed in (20002, *ALTERNATES)}


def suite():
    bases = base_schedules()
    h = bases[20002]
    arms = []

    def add(name, kind, rows, alternate=None, k=None, edits=None):
        arms.append({'name': name, 'kind': kind, 'alternate_seed': alternate,
                     'k': k, 'edited_updates': edits or [],
                     'rows': np.array(rows, copy=True).tolist()})

    add('H', 'historical_control', h)
    for seed in ALTERNATES:
        s = bases[seed]
        add(f'S{seed}', 'alternate_control', s, seed)
        for k in LENGTHS:
            rows = h.copy(); rows[:k] = s[:k]
            add(f'S{seed}_replace_early{k}', 'replace_early', rows, seed, k,
                list(range(1, k+1)))
            rows = s.copy(); rows[:k] = h[:k]
            add(f'S{seed}_preserve_early{k}', 'preserve_early', rows, seed, k,
                list(range(1, k+1)))
            rows = h.copy(); rows[180-k:180] = s[:k]
            add(f'S{seed}_replace_late{k}', 'replace_late', rows, seed, k,
                list(range(181-k, 181)))
    for name, first in (('H_swap_early8', 0), ('H_swap_late8', 164)):
        rows = h.copy()
        rows[first:first+8] = h[172:180]
        rows[172:180] = h[first:first+8]
        add(name, 'multiset_order_swap', rows, k=8,
            edits=list(range(first+1, first+9))+list(range(173, 181)))
    assert len(arms) == 23
    return arms


def check_suite():
    bases = base_schedules()
    specs = suite()
    assert len({s['name'] for s in specs}) == 23
    for spec in specs:
        rows = np.array(spec['rows'])
        assert rows.shape == (300, 8) and rows.min() >= 0 and rows.max() < 512
        kind = spec['kind']
        ref = bases[spec['alternate_seed']] if kind in ('alternate_control', 'preserve_early') else bases[20002]
        changed = (np.flatnonzero(np.any(rows != ref, axis=1))+1).tolist()
        assert changed == spec['edited_updates'], (spec['name'], changed)
        if kind == 'multiset_order_swap':
            assert Counter(map(tuple, rows)) == Counter(map(tuple, bases[20002]))
            assert Counter(rows.ravel()) == Counter(bases[20002].ravel())
        elif kind in ('replace_early', 'preserve_early', 'replace_late'):
            donor = bases[20002] if kind == 'preserve_early' else bases[spec['alternate_seed']]
            ids = np.array(spec['edited_updates'])-1
            expected = donor[:spec['k']] if kind == 'replace_late' else donor[ids]
            assert np.array_equal(rows[ids], expected)
        else:
            assert np.array_equal(rows, ref)
    return {'status': 'PASS', 'arms': len(specs), 'multiset_preserving_controls': 2,
            'lengths': list(LENGTHS), 'alternate_schedule_units': list(ALTERNATES)}
