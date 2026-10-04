"""Eight fixed u3 parameter/moment cross continuations, with exact native controls."""
from __future__ import annotations

import gc
import json
import time

import numpy as np
import torch

from common import (PRIOR, OPTIMIZER, region_bank, tensor_hash, sha, write,
                    load_state, train_updates, evaluate, compact,
                    save_checkpoint, compare_replay)


class ReplayError(RuntimeError):
    pass


NATIVE = {
    ('H', 'H', 'H'): 'H',
    ('H', 'H', 'S'): 'S20022_preserve_early3',
    ('S', 'S', 'H'): 'S20022_replace_early3',
    ('S', 'S', 'S'): 'S20022',
}


def specs():
    # Native replay controls precede all hybrids, so efficacy has qualified controls.
    triples = list(NATIVE) + [('H', 'S', s) for s in ('H', 'S')] + [('S', 'H', s) for s in ('H', 'S')]
    return [{'name': f'theta{p}_m{m}_suffix{s}', 'theta': p, 'moments': m,
             'suffix': s, 'native_reference': NATIVE.get((p, m, s))}
            for p, m, s in triples]


def select_recipe(records):
    lookup = {(r['theta'], r['moments'], r['suffix']): r['endpoint']['pass'] for r in records}
    assert len(lookup) == 8
    assert lookup[('S', 'S', 'H')] and not lookup[('S', 'S', 'S')]
    assert lookup[('H', 'H', 'H')] and lookup[('H', 'H', 'S')]
    moment_ok = all(lookup[('S', 'H', s)] for s in ('H', 'S'))
    parameter_ok = all(lookup[('H', 'S', s)] for s in ('H', 'S'))
    recipe = 'shadow_moments' if moment_ok else ('shadow_parameters' if parameter_ok else 'shadow_joint')
    return {'recipe': recipe, 'moment_candidate_eligible': moment_ok,
            'parameter_candidate_eligible': parameter_ok,
            'priority': ['shadow_moments', 'shadow_parameters', 'shadow_joint'],
            'criterion': 'Rescue native-S failure on S suffix and preserve native-S pass on H suffix.',
            'claim_boundary': 'Conditional donor-state recipe selection; no suffix-independent component mechanism.',
            'lookup': {f'theta{p}_m{m}_suffix{s}': v for (p, m, s), v in lookup.items()}}


def run_stage(out, progress=lambda _: None):
    out.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    states = {key: torch.load(PRIOR / 'checkpoints' / f'{name}_u003.pt',
                              map_location='cpu', weights_only=False)
              for key, name in (('H', 'H'), ('S', 'S20022'))}
    rows = {key: np.random.default_rng(seed).integers(0, 512, (300, 8)).tolist()
            for key, seed in (('H', 20002), ('S', 20022))}
    train = region_bank(32, 512, 10002, 'cuda')
    fixed = {n: region_bank(n, 32, 50000 + n) for n in (32, 64)}
    write(out / 'manifest.json', {'protocol': 'u3_state_cross_v1', 'expected_arms': 8,
          'start_update': 3, 'end_update': 300, 'initialization_seed': 4,
          'optimizer': OPTIMIZER, 'step_counter_preserved': True, 'arms': specs(),
          'reference_checkpoint_sha256': {key: sha(PRIOR / 'checkpoints' / f'{name}_u003.pt')
                 for key, name in (('H', 'H'), ('S', 'S20022'))},
          'train_data_sha256': tensor_hash(train),
          'evaluation_data_sha256': {str(n): tensor_hash(d) for n, d in fixed.items()},
          'evaluation_banks_reused': True, 'runtime_limit_enforced': False})
    for key, schedule in rows.items():
        write(out / f'suffix{key}_schedule.json', schedule)
    records = []
    for spec in specs():
        name = spec['name']
        emit = lambda value: progress({'stage': 1, 'arm': name, **value})
        model, optimizer = load_state(states[spec['theta']], states[spec['moments']])
        record = {**spec, 'status': 'TRAINING', 'start_parameter_sha256': tensor_hash(model.state_dict())}
        started = time.monotonic()
        record['training_curve'] = train_updates(model, optimizer, train, rows[spec['suffix']][3:], 4, emit)
        record['training_seconds'] = time.monotonic() - started
        record['final_parameter_sha256'] = tensor_hash(model.state_dict())
        record['checkpoint'] = save_checkpoint(out, name, model, optimizer, 4)
        if spec['native_reference']:
            previous = json.loads((PRIOR / f'{spec["native_reference"]}.json').read_text(encoding='utf-8'))
            if record['final_parameter_sha256'] != previous['final_parameter_sha256']:
                record['status'] = 'REPLAY_ERROR'
                write(out / f'{name}.json', record)
                raise ReplayError(f'Native final parameter mismatch: {name}')
        emit({'phase': 'evaluation', 'completed_updates': 300})
        summary = evaluate(model, fixed, out, name, lambda: None)
        record['endpoint'] = compact(summary)
        if spec['native_reference']:
            previous_summary = json.loads((PRIOR / f'{spec["native_reference"]}_summary.json').read_text(encoding='utf-8'))
            try:
                record['native_replay'] = compare_replay(previous_summary['sizes'], summary['sizes'])
                assert previous_summary['phenotype_gate']['pass'] == summary['phenotype_gate']['pass']
            except Exception as error:
                record['status'] = 'REPLAY_ERROR'
                write(out / f'{name}.json', record)
                raise ReplayError(f'Native phenotype mismatch: {name}') from error
        record['status'] = 'COMPLETE'
        write(out / f'{name}.json', record)
        records.append(record)
        write(out / 'summary.json', {'status': 'RUNNING', 'completed_arms': len(records),
                                     'expected_arms': 8, 'arms': records})
        emit({'phase': 'arm_complete', 'completed_arms': len(records), 'completed_updates': 300})
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    selection = select_recipe(records)
    result = {'status': 'COMPLETE', 'completed_arms': 8, 'expected_arms': 8,
              'elapsed_seconds': time.monotonic() - begun, 'selection': selection, 'arms': records}
    write(out / 'summary.json', result)
    write(out / 'selection.json', selection)
    lines = ['# Update3 parameter / Adam-state cross', '', 'All four native controls exactly reproduce final parameters and saved phenotype metrics.', '',
             '| Parameter source | Adam source | Suffix | Full | size32 strict T256 |', '|---|---|---|---|---:|']
    for row in records:
        lines.append(f"| {row['theta']} | {row['moments']} | {row['suffix']} | {row['endpoint']['pass']} | {row['endpoint']['sizes']['32']['strict_mean']['256']:.6f} |")
    lines += ['', 'Selected stage2 recipe: `' + selection['recipe'] + '`.',
              'Selection tests rescue on the S suffix and preservation on the H suffix separately.',
              'One selected init and two selected suffixes; no reliability or universal component-mechanism claim.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result
