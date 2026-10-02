"""CPU arithmetic/provenance audit of saved streaming-carry results; no inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from stream_cells import ROOT, make_variant

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
    assert manifest['protocol'] == 'streaming_carry_development_v1'
    assert status['status'] == 'COMPLETE' and status['completed_arms'] == 8 and aggregate['complete']
    assert len(manifest['source_sha256']) == 35
    assert status['elapsed_seconds'] < manifest['maximum_seconds'] == 1500
    assert not (run / 'watchdog_timeout.json').exists() and not (run / 'error.json').exists()
    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT / name) == sha(run / 'source' / name) == expected, name
    assert read(run / 'schedule.json') == np.random.default_rng(20002).integers(0, 512, (300, 8)).tolist()
    assert manifest['updates_per_arm'] == 300 and manifest['gradient_horizon'] == 8
    assert manifest['variants'] == ['baseline', 'stream']
    train_hash = tensor_hash(bank(32, 512, 10002))
    assert train_hash == manifest['train_data_sha256']
    data = {n: bank(n, 32, 40000+n) for n in (32, 64)}
    for n, d in data.items():
        assert tensor_hash(d) == manifest['evaluation_data_sha256'][str(n)] == manifest['executed_eval_data_sha256'][str(n)]
    initial = {}
    for seed in (2, 3, 4, 5):
        for variant in ('baseline', 'stream'):
            torch.manual_seed(seed)
            initial[seed, variant] = tensor_hash(make_variant(variant).state_dict())
        assert initial[seed, 'baseline'] == initial[seed, 'stream']
    records, verdicts, curves, historical = [], [], [], []
    ba_count = paired_count = 0
    paths = sorted(run.glob('*_K8_seed*.json'))
    assert len(list(run.glob('*.pt'))) == 8
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
    assert {(r['seed'], r['variant']) for r in verdicts} == {(s,v) for s in (2,3,4,5) for v in ('baseline','stream')}
    counts = {v: {key: sum(r['variant'] == v and r[key] for r in verdicts) for key in ('reach','reach_and_hold')} for v in ('baseline','stream')}
    kept = {v: all(next(r for r in verdicts if r['variant']==v and r['seed']==s)['reach_and_hold'] for s in (2,5)) for v in counts}
    verdict = ('BASELINE_REPRODUCTION_DRIFT' if not kept['baseline'] else
               'DEVELOPMENT_GO' if counts['stream']['reach_and_hold'] >= 3 and kept['stream'] and counts['stream']['reach_and_hold'] > counts['baseline']['reach_and_hold'] else
               'DEVELOPMENT_NO_GO')
    assert counts == aggregate['counts'] and kept == aggregate['historical_positive_seeds_preserved']
    assert verdict == aggregate['decision']
    lookup = {(c['seed'],c['variant'],c['size'],c['T'],c['band']): c for c in curves}
    assert len(lookup) == len(curves) == 624
    effects = []
    for c in curves:
        if c['variant'] != 'stream':
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
    csv_rows = 0
    for filename, expected in (('curves.csv', curves), ('paired_effects.csv', effects)):
        with (run / filename).open(newline='', encoding='utf-8') as f:
            actual = list(csv.DictReader(f))
        keynames = ('seed','variant','size','T','band') if filename == 'curves.csv' else ('seed','size','T','band')
        keyed = {tuple(str(r[k]) for k in keynames): r for r in expected}
        actual_keys = [tuple(a[k] for k in keynames) for a in actual]
        assert len(actual) == len(keyed) and len(set(actual_keys)) == len(actual)
        assert set(actual_keys) == set(keyed)
        csv_rows += len(actual)
        for a in actual:
            e = keyed[tuple(a[k] for k in keynames)]
            for k, value in e.items():
                if isinstance(value, (int,float)):
                    close(float(a[k]), value)
                else:
                    assert a[k] == ('' if value is None else value)
    assert (ba_count, paired_count, len(curves), len(effects), csv_rows) == (96, 672, 624, 312, 936)
    validation = {'status': 'PASS', 'source_files': len(manifest['source_sha256']), 'checkpoints': 8,
                  'data_schedule_initial_parameters': True, 'BA_aggregates': ba_count,
                  'paired_aggregates_and_denominators': paired_count, 'arm_decisions': 8,
                  'paired_effects': len(effects), 'CSV_rows': csv_rows,
                  'historical_baseline_exact_evaluation_and_final_parameters': 4,
                  'new_training_or_model_inference': False,
                  'checkpoint_check': 'CPU weights_only state_dict hash against saved final parameters'}
    primary_effects = [e for e in effects if e['size']==32 and e['T']==64 and e['band']=='strict_16_32']
    out.mkdir(parents=True)
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write('analysis.json', {'execution': {k:v for k,v in status.items() if k != 'pid'}, 'decision': verdict,
                            'counts': counts, 'verdicts': verdicts, 'primary_paired_effects': primary_effects,
                            'systems': records, 'baseline_reproduction': historical, 'validation': validation})
    seeds = sorted({r['seed'] for r in verdicts})
    variants = manifest['variants']
    n_seeds = len(seeds)
    by_arm = {(r['seed'], r['variant']): r for r in verdicts}
    stream_success = [r for r in verdicts if r['variant'] == 'stream' and r['reach_and_hold']]
    stream_failed = [r for r in verdicts if r['variant'] == 'stream' and not r['reach_and_hold']]
    stream_hold_only = [r for r in verdicts if r['variant'] == 'stream' and r['hold'] and not r['reach']]

    def pct(value):
        return 'n/a' if value is None else f'{100*value:.2f}'

    def seed_names(rows):
        return ', '.join(str(r['seed']) for r in sorted(rows, key=lambda r: r['seed'])) or 'none'

    lines = ['# Streaming Carry: completed development screen', '',
             f"**{verdict}.** All {len(verdicts)} arms completed {manifest['updates_per_arm']} updates in {status['elapsed_seconds']:.1f}s ({status['elapsed_seconds']/60:.1f} minutes).",
             f"Reach-and-hold counts under the frozen gate: baseline {counts['baseline']['reach_and_hold']}/{n_seeds}; stream {counts['stream']['reach_and_hold']}/{n_seeds}.",
             f"All {len(historical)} baseline records reproduce the Phase II final-parameter hashes and complete saved evaluations; matched training data and schedule are also identical.", '',
             '## Primary gate: size32, T64, strict16<d<32', '',
             '| Seed | Variant | BA original / flipped % | Paired mean / pooled % | Reach | Hold | Reach + hold |',
             '|---:|---|---:|---:|---|---|---|']
    for seed in seeds:
        for variant in variants:
            r = by_arm[seed, variant]
            lines.append(f"| {seed} | {variant} | {pct(r['ba_original'])} / {pct(r['ba_flipped'])} | {pct(r['primary_mean'])} / {pct(r['primary_pooled'])} | {r['reach']} | {r['hold']} | {r['reach_and_hold']} |")
    lines += ['', f"Stream met both predicates for seed(s): {seed_names(stream_success)}. It did not meet both for seed(s): {seed_names(stream_failed)}."]
    for r in stream_success:
        lines.append(f"The passing stream record at seed {r['seed']} has original/flipped BA {pct(r['ba_original'])}%/{pct(r['ba_flipped'])}% and primary paired mean/pooled accuracy {pct(r['primary_mean'])}%/{pct(r['primary_pooled'])}%; this single record remains within the overall {verdict} decision.")
    if stream_hold_only:
        lines.append(f"Hold passes without reach at seed(s) {seed_names(stream_hold_only)}; hold alone does not satisfy the task gate.")
    lines += ['', 'The fixed operator is specified as a permutation of four 6-channel directional ports in W (24 channels) alongside stationary local Z (8 channels). Each open-cell port has one predecessor; blocked links reverse the lane, wall ports stay fixed, and the inverse is B T B. Thus T is bijective and preserves Euclidean norm for each payload coordinate. This property applies to T alone.',
              'In the learned step, F receives incoming T(W), stationary Z, old L(W), old L(Z), and X before the local residual update; Q then uses W\'. The macro-step has at most two graph hops. The candidate also changes F\'s first feature to incoming T(W), so this screen cannot isolate a causal effect of T alone. Neither T\'s losslessness nor norm preservation implies stability or information retention for the learned recurrence.',
              'Reach uses both whole-grid BAs >=85% and both strict-band paired statistics >=80%. Hold requires the same BAs to drop by at most 3 percentage points and the paired statistics by at most 5 points at both T128 and T256 relative to T64.', '',
              '## Size32 primary statistics across horizons', '',
              '| Seed | T | Baseline BA O/F % | Baseline paired mean / pooled % | Stream BA O/F % | Stream paired mean / pooled % |',
              '|---:|---:|---:|---:|---:|---:|']
    for seed in seeds:
        for t in (64, 128, 256):
            b = lookup[seed, 'baseline', 32, t, 'strict_16_32']
            s = lookup[seed, 'stream', 32, t, 'strict_16_32']
            lines.append(f"| {seed} | {t} | {pct(b['ba_original'])} / {pct(b['ba_flipped'])} | {pct(b['mean'])} / {pct(b['pooled'])} | {pct(s['ba_original'])} / {pct(s['ba_flipped'])} | {pct(s['mean'])} / {pct(s['pooled'])} |")
    lines += ['', 'T64 determines reach; T128 and T256 determine hold against the corresponding T64 values.', '',
              '## Size64 far changed-component band (d>32)', '',
              '| Seed | T | Baseline mean / pooled % | Stream mean / pooled % |', '|---:|---:|---:|---:|']
    for seed in seeds:
        for t in (64, 128, 256):
            b, s = (lookup[seed, variant, 64, t, 'far_gt32'] for variant in variants)
            lines.append(f"| {seed} | {t} | {pct(b['mean'])} / {pct(b['pooled'])} | {pct(s['mean'])} / {pct(s['pooled'])} |")
    lines += ['', 'These farther-band values are descriptive and do not replace the primary gate. All evaluated bands, denominators, horizons, sizes and paired effects are retained in [curves](curves.csv) and [paired effects](paired_effects.csv). Empty bands remain null in JSON and blank in CSV.', '',
              '## Scope and verification', '',
              '- This is a development result on previously inspected seeds and maps, conditional on one training bank and schedule (n=4). It supports no significance, cross-distribution reliability or general robustness claim.',
              '- The fixed transport is lossless; that does not imply lossless learned updates, useful information retention, or stability of the full recurrence.',
              '- The saved-result audit does not attribute outcomes to a causal dynamics or failure mechanism.',
              '- No new training or model inference, parameter sweep, confirmation run or monitoring was performed for this publication.', '',
              '## Systems and verification', '', '| Seed | Variant | Training seconds | Peak allocated MiB | Clipped updates % |', '|---:|---|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['training_seconds']:.2f} | {r['peak_MiB']:.2f} | {100*r['clip_fraction']:.2f} |")
    lines += ['', f"Verified {validation['source_files']} bound source files and {validation['checkpoints']} checkpoint state hashes; all initialization/data/schedule bindings; {ba_count} BA aggregates and {paired_count} paired aggregates/denominators; {len(verdicts)} arm decisions; {len(effects)} paired effects; and {validation['CSV_rows']} CSV rows.",
              'This checks saved-result arithmetic and provenance. Checkpoints were loaded on CPU only to verify saved parameter hashes; no model outputs were regenerated or rescored.', '',
              '[Compact analysis](analysis.json), [validation](validation.json), [raw arm records](raw/).']
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write('validation.json', validation)
    inputs = [f for f in run.iterdir() if f.is_file() and f.suffix in ('.json','.csv','.md')]
    inputs += [ROOT/f'evidence/short_bptt_phase2_init2345/raw/additive_K8_seed{s}.json' for s in (2,3,4,5)]
    write('provenance.json', {'source_run': run.relative_to(ROOT).as_posix(),
          'input_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(inputs)},
          'checkpoint_sha256': {f.name: sha(f) for f in sorted(run.glob('*.pt'))},
          'analysis_source_sha256': {Path(__file__).resolve().relative_to(ROOT).as_posix(): sha(Path(__file__).resolve())},
          'output_sha256': {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file()}})
    print(json.dumps({'decision': verdict, 'counts': counts, 'validation': validation, 'systems': records}, indent=2))


if __name__ == '__main__':
    main()
