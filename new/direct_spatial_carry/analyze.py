"""CPU arithmetic/provenance audit of saved carry results; no inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from carry_cells import ROOT, make_variant

sys.path.insert(0, str(ROOT / 'new/short_bptt_phase2'))
from run_revision import tensor_hash
from tasks import bank


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    if a is None or b is None:
        assert a is None and b is None, (a, b)
    else:
        assert math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-6), (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    assert not out.exists()
    torch.set_num_threads(2)
    manifest, status, aggregate = (read(run / n) for n in ('manifest.json', 'status.json', 'aggregate.json'))
    assert status['status'] == 'COMPLETE' and status['completed_arms'] == 8 and aggregate['complete']
    assert status['elapsed_seconds'] < manifest['maximum_seconds'] == 1500
    assert not (run / 'watchdog_timeout.json').exists() and not (run / 'error.json').exists()
    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT / name) == sha(run / 'source' / name) == expected, name
    assert read(run / 'schedule.json') == np.random.default_rng(20002).integers(0, 512, (300, 8)).tolist()
    assert manifest['updates_per_arm'] == 300 and manifest['gradient_horizon'] == 8
    train_hash = tensor_hash(bank(32, 512, 10002))
    assert train_hash == manifest['train_data_sha256']
    data = {n: bank(n, 32, 40000+n) for n in (32, 64)}
    for n, d in data.items():
        assert tensor_hash(d) == manifest['evaluation_data_sha256'][str(n)] == manifest['executed_eval_data_sha256'][str(n)]
    initial = {}
    for seed in (2, 3, 4, 5):
        for variant in ('baseline', 'carry'):
            torch.manual_seed(seed)
            initial[seed, variant] = tensor_hash(make_variant(variant).state_dict())
        assert initial[seed, 'baseline'] == initial[seed, 'carry']
    records, verdicts, curves, historical = [], [], [], []
    ba_count = paired_count = 0
    paths = sorted(run.glob('*_K8_seed*.json'))
    assert len(paths) == 8
    for path in paths:
        r = read(path)
        seed, variant = r['seed'], r['variant']
        assert r['status'] == 'COMPLETE' and r['completed_updates'] == 300 and r['gradient_horizon'] == 8
        assert r['parameter_count'] == 5033 and r['architecture'] == make_variant(variant).metadata()
        assert r['initial_parameter_sha256'] == initial[seed, variant]
        assert r['train_data_sha256'] == train_hash and r['schedule_sha256'] == sha(run / 'schedule.json')
        checkpoint = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
        assert checkpoint['completed_updates'] == 300 and checkpoint['variant'] == variant
        assert checkpoint['architecture'] == r['architecture'] and checkpoint['seed'] == seed
        assert tensor_hash(checkpoint['state_dict']) == r['final_parameter_sha256']
        assert len(r['update_seconds']) == 300
        for log in r['training_curve']:
            assert all(log[k] == value for k, value in {'forward_steps': 64, 'loss_count': 8,
                      'backward_calls': 8, 'interior_detach_boundaries': 7}.items())
        if variant == 'baseline':
            old = read(ROOT / f'evidence/short_bptt_phase2_init2345/raw/additive_K8_seed{seed}.json')
            matched = {key: r[key] == old[key] for key in ('initial_parameter_sha256', 'final_parameter_sha256',
                       'train_data_sha256', 'schedule_sha256', 'evaluation', 'gradient_clip_fraction')}
            assert all(matched.values()), (seed, matched)
            historical.append({'seed': seed, **matched})
        assert set(r['evaluation']) == {'32', '64'}
        for size, sequence in r['evaluation'].items():
            assert set(sequence) == {'64', '128', '256'}
            d = data[int(size)]['distance'].numpy()
            changed = data[int(size)]['changed'].numpy().astype(bool)
            for t, e in sequence.items():
                for branch in ('original', 'flipped'):
                    assert len(e[branch]['per_map_ba']) == 32
                    close(float(np.mean(e[branch]['per_map_ba'])), e[branch]['balanced_accuracy'])
                    ba_count += 1
                masks = {'paired': changed, '0_8': (d>=0)&(d<8), '8_16': (d>=8)&(d<16),
                         'equal_16': d==16, 'strict_16_32': (d>16)&(d<32),
                         '32_64': (d>=32)&(d<64), '64_128': (d>=64)&(d<128),
                         '128_inf': d>=128, 'far_gt16': d>16, 'far_gt32': d>32,
                         'r_le1': (d>=0)&(d<=16), 'r_1_2': (d>16)&(d<=32),
                         'r_gt2': d>32, 'outside_forward_lightcone': d>2*int(t)}
                metrics = {'paired': e['paired'], **e['bands']}
                assert set(masks) == set(metrics)
                for name, m in metrics.items():
                    counts = (changed & masks[name]).sum((1, 2, 3))
                    hits = np.array(m['per_map_correct'])
                    assert counts.tolist() == m['per_map_pixels']
                    assert len(hits) == len(m['per_map']) == 32 and np.all((0<=hits)&(hits<=counts))
                    valid = []
                    for n, h, value in zip(counts, hits, m['per_map']):
                        close(h/n if n else None, value)
                        if n:
                            valid.append(h/n)
                    assert len(valid) == m['eligible_maps']
                    assert int(counts.sum()) == m['pooled_pixels'] and int(hits.sum()) == m['pooled_correct']
                    close(float(np.mean(valid)) if valid else None, m['mean'])
                    close(float(hits.sum()/counts.sum()) if valid else None, m['pooled_accuracy'])
                    paired_count += 1
                partition = ('0_8', '8_16', 'equal_16', 'strict_16_32', '32_64', '64_128', '128_inf')
                assert np.array_equal(sum(np.array(metrics[b]['per_map_correct']) for b in partition), metrics['paired']['per_map_correct'])
                assert e['outside_forward_lightcone_max_logit_difference'] <= 1e-6
                for band, m in e['bands'].items():
                    curves.append({'seed': seed, 'variant': variant, 'size': int(size), 'T': int(t), 'band': band,
                                   'mean': m['mean'], 'pooled': m['pooled_accuracy'], 'pixels': m['pooled_pixels'],
                                   'correct': m['pooled_correct'], 'eligible_maps': m['eligible_maps'],
                                   'ba_original': e['original']['balanced_accuracy'], 'ba_flipped': e['flipped']['balanced_accuracy']})
        primary = r['evaluation']['32']['64']
        p = primary['bands']['strict_16_32']
        reach = all(primary[b]['balanced_accuracy'] >= .85 for b in ('original', 'flipped')) and p['mean'] >= .8 and p['pooled_accuracy'] >= .8
        hold = all(all(r['evaluation']['32'][t][b]['balanced_accuracy'] >= primary[b]['balanced_accuracy']-.03 for b in ('original', 'flipped'))
                   and all(r['evaluation']['32'][t]['bands']['strict_16_32'][f] >= p[f]-.05 for f in ('mean', 'pooled_accuracy')) for t in ('128', '256'))
        row = {'seed': seed, 'variant': variant, 'reach': reach, 'hold': hold, 'reach_and_hold': reach and hold,
               'primary_mean': p['mean'], 'primary_pooled': p['pooled_accuracy'],
               'ba_original': primary['original']['balanced_accuracy'], 'ba_flipped': primary['flipped']['balanced_accuracy']}
        oldrow = next(x for x in aggregate['rows'] if x['seed'] == seed and x['variant'] == variant)
        for key, value in row.items():
            if isinstance(value, float):
                close(value, oldrow[key])
            else:
                assert value == oldrow[key]
        verdicts.append(row)
        records.append({'seed': seed, 'variant': variant, 'training_seconds': r['training_seconds'],
                        'peak_MiB': r['training_peak_allocated_bytes']/2**20, 'clip_fraction': r['gradient_clip_fraction']})
    assert {(r['seed'], r['variant']) for r in verdicts} == {(s,v) for s in (2,3,4,5) for v in ('baseline','carry')}
    counts = {v: {key: sum(r['variant'] == v and r[key] for r in verdicts) for key in ('reach','reach_and_hold')} for v in ('baseline','carry')}
    kept = {v: all(next(r for r in verdicts if r['variant']==v and r['seed']==s)['reach_and_hold'] for s in (2,5)) for v in counts}
    verdict = ('BASELINE_REPRODUCTION_DRIFT' if not kept['baseline'] else
               'DEVELOPMENT_GO' if counts['carry']['reach_and_hold'] >= 3 and kept['carry'] and counts['carry']['reach_and_hold'] > counts['baseline']['reach_and_hold'] else
               'DEVELOPMENT_NO_GO')
    assert counts == aggregate['counts'] and kept == aggregate['historical_positive_seeds_preserved']
    assert verdict == aggregate['decision']
    lookup = {(c['seed'],c['variant'],c['size'],c['T'],c['band']): c for c in curves}
    assert len(lookup) == len(curves) == 624
    effects = []
    for c in curves:
        if c['variant'] != 'carry':
            continue
        b = lookup[c['seed'],'baseline',c['size'],c['T'],c['band']]
        effects.append({**{k: c[k] for k in ('seed','size','T','band','pixels','eligible_maps')},
                        **{k+'_effect_pp': 100*(c[k]-b[k]) if c[k] is not None else None for k in ('mean','pooled','ba_original','ba_flipped')}})
    assert len(effects) == len(aggregate['paired_effects']) == 312
    for e in effects:
        old = next(x for x in aggregate['paired_effects'] if all(x[k] == e[k] for k in ('seed','size','T','band')))
        for k in ('mean','pooled','ba_original','ba_flipped'):
            close(e[k+'_effect_pp'], old[k+'_effect_pp'])
    # Check the published-by-runner CSVs, not only the JSON source values.
    for filename, expected in (('curves.csv', curves), ('paired_effects.csv', effects)):
        with (run / filename).open(newline='', encoding='utf-8') as f:
            actual = list(csv.DictReader(f))
        keynames = ('seed','variant','size','T','band') if filename == 'curves.csv' else ('seed','size','T','band')
        keyed = {tuple(str(r[k]) for k in keynames): r for r in expected}
        assert len(actual) == len(keyed)
        for a in actual:
            e = keyed[tuple(a[k] for k in keynames)]
            for k, value in e.items():
                if isinstance(value, (int,float)):
                    close(float(a[k]), value)
                else:
                    assert a[k] == ('' if value is None else value)
    validation = {'status': 'PASS', 'source_files': len(manifest['source_sha256']), 'checkpoints': 8,
                  'data_schedule_initial_parameters': True, 'BA_aggregates': ba_count,
                  'paired_aggregates_and_denominators': paired_count, 'arm_decisions': 8,
                  'paired_effects': len(effects), 'CSV_rows': 936,
                  'historical_baseline_exact_evaluation_and_final_parameters': 4,
                  'new_training_or_inference': False}
    primary_effects = [e for e in effects if e['size']==32 and e['T']==64 and e['band']=='strict_16_32']
    out.mkdir(parents=True)
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write('analysis.json', {'execution': {k:v for k,v in status.items() if k != 'pid'}, 'decision': verdict,
                            'counts': counts, 'verdicts': verdicts, 'primary_paired_effects': primary_effects,
                            'systems': records, 'baseline_reproduction': historical, 'validation': validation})
    lines = ['# Direct Spatial Carry: completed negative development screen', '',
             f"All8 arms completed300 updates in{status['elapsed_seconds']:.3f}s ({status['elapsed_seconds']/60:.2f} minutes). **{verdict}**.",
             'Baseline reaches and holds in2/4; carry in0/4. All four baseline final parameter hashes and complete evaluation records exactly reproduce Phase II.', '',
             'The sole change is W -> W-0.5*D_M^dagger*L_M(W) in the W skip path. This is fixed lazy diffusion over the correct masked medium.',
             'F/Q, additive Z,5033 parameters, K8 training,300 updates, initialization and data/schedule are matched.', '',
             '## Primary: size32/T64, strict16<d<32', '',
             '| Seed | Baseline mean / pooled % | Carry mean / pooled % | Pooled change pp |', '|---|---:|---:|---:|']
    for seed in (2,3,4,5):
        b,c = (next(v for v in verdicts if v['seed']==seed and v['variant']==arm) for arm in ('baseline','carry'))
        lines.append(f"| {seed} | {100*b['primary_mean']:.2f} / {100*b['primary_pooled']:.2f} | {100*c['primary_mean']:.2f} / {100*c['primary_pooled']:.2f} | {100*(c['primary_pooled']-b['primary_pooled']):.2f} |")
    lines += ['', 'Every seed loses both primary statistics. Carry does not rescue seeds3/4 and loses the previous successes2/5.',
              'Carry seed2 has hold=True only because its already-zero primary score does not decline; it never reaches. Hold alone is not task success.',
              'Whole-grid BA can remain relatively high while paired correctness on far changed-component pixels fails; these are different populations and requirements.', '',
              '## Farther propagation on size64', '',
              '| Seed | T | Baseline d>32 mean / pooled % | Carry d>32 mean / pooled % |', '|---|---:|---:|---:|']
    for seed in (2,3,4,5):
        for t in (64,128,256):
            b,c = (lookup[seed,v,64,t,'far_gt32'] for v in ('baseline','carry'))
            lines.append(f"| {seed} | {t} | {100*b['mean']:.2f} / {100*b['pooled']:.2f} | {100*c['mean']:.2f} / {100*c['pooled']:.2f} |")
    lines += ['', 'All distance/size/horizon results, including isolated positive contrasts, are retained in [curves](curves.csv) and [paired effects](paired_effects.csv).',
              'Empty bands remain null in JSON (blank in CSV); size32 has no changed-component pixels at d>=64.', '',
              '## Interpretation and scope', '',
              '- This rejects this rho=0.5 normalized-average W-carry recipe under the frozen K8 development protocol; do not advance it to confirmation.',
              '- It does not reject all spatial carry, directional transport, cellular computation or short BPTT.',
              '- The experiment replaces part of local identity retention with neighbor mixing. It does not isolate smoothing, signal dilution, cancellation-learning burden or another failure mechanism.',
              '- Channel coordinates are preserved by the fixed operator, but task information need not be preserved. Bounded carry amplitude is not a guarantee for the whole recurrent cell.',
              '- Seeds and evaluation maps were already inspected. n=4 is conditional on one training bank/schedule; no significance or cross-distribution claim.',
              '- No new training, checkpoint inference, sweep, confirmation or monitoring was performed for this publication.', '',
              '## Systems and verification', '', '| Seed | Variant | Training seconds | Peak allocated MiB | Clipped updates % |', '|---|---|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['training_seconds']:.2f} | {r['peak_MiB']:.2f} | {100*r['clip_fraction']:.2f} |")
    lines += ['', f"Verified{validation['source_files']} source snapshots,8 checkpoints, all initialization/data/schedule bindings,{ba_count} BA and{paired_count} paired aggregates/denominators,8 arm decisions,312 paired effects and936 CSV rows.",
              'This checks saved-result arithmetic and provenance; it does not regenerate logits or independently rescore pixels from weights.', '',
              '[Compact analysis](analysis.json), [validation](validation.json), [raw arm records](raw/).']
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write('validation.json', validation)
    inputs = [f for f in run.iterdir() if f.is_file() and f.suffix in ('.json','.csv','.md')]
    inputs += [ROOT/f'evidence/short_bptt_phase2_init2345/raw/additive_K8_seed{s}.json' for s in (2,3,4,5)]
    write('provenance.json', {'source_run': run.relative_to(ROOT).as_posix(),
          'input_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(inputs)},
          'checkpoint_sha256': {f.name: sha(f) for f in sorted(run.glob('*.pt'))},
          'analysis_source_sha256': {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))},
          'output_sha256': {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file()}})
    print(json.dumps({'decision': verdict, 'counts': counts, 'validation': validation, 'systems': records}, indent=2))


if __name__ == '__main__':
    main()
