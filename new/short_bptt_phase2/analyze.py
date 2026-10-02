"""CPU-only verification of saved Phase-II results; no model inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from training import ROOT, make_cell
from run_revision import tensor_hash
from tasks import bank


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-6), (a, b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    assert not out.exists()
    torch.set_num_threads(2)
    manifest, status, agg = (read(run / n) for n in ('manifest.json', 'status.json', 'aggregate.json'))
    assert status['status'] == 'COMPLETE' and status['completed_arms'] == 12 and agg['complete']
    assert not (run / 'watchdog_timeout.json').exists()
    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT / name) == sha(run / 'source' / name) == expected, name
    schedule = read(run / 'schedule.json')
    assert schedule == np.random.default_rng(20002).integers(0, 512, (300, 8)).tolist()
    close(manifest['updates_per_arm'], 300)
    trainhash = tensor_hash(bank(32, 512, 10002))
    assert trainhash == manifest['train_data_sha256']
    data = {n: bank(n, 32, 40000+n) for n in (32, 64)}
    for n, d in data.items():
        assert tensor_hash(d) == manifest['evaluation_data_sha256'][str(n)]
        assert tensor_hash(d) == manifest['executed_eval_data_sha256'][str(n)]
    initial = {}
    for seed in (2, 3, 4, 5):
        torch.manual_seed(seed)
        initial[seed] = tensor_hash(make_cell('ws_additive').state_dict())
    records, curves, verdicts = [], [], []
    ba_count = paired_count = 0
    for file in sorted(run.glob('additive_*.json')):
        r = read(file)
        k, seed = r['gradient_horizon'], r['seed']
        assert r['status'] == 'COMPLETE' and r['completed_updates'] == 300
        assert r['initial_parameter_sha256'] == initial[seed]
        assert r['train_data_sha256'] == trainhash and r['schedule_sha256'] == sha(run / 'schedule.json')
        cp = torch.load(file.with_suffix('.pt'), map_location='cpu', weights_only=True)
        assert cp['completed_updates'] == 300 and tensor_hash(cp['state_dict']) == r['final_parameter_sha256']
        assert len(r['update_seconds']) == 300
        for log in r['training_curve']:
            assert log['forward_steps'] == 64 and log['loss_count'] == 8
            assert log['backward_calls'] == 64//k and log['interior_detach_boundaries'] == 64//k-1
        for size, sequence in r['evaluation'].items():
            assert set(sequence) == {'64', '128', '256'}
            d = data[int(size)]['distance'].numpy()
            changed = data[int(size)]['changed'].numpy().astype(bool)
            for t, v in sequence.items():
                for branch in ('original', 'flipped'):
                    assert len(v[branch]['per_map_ba']) == 32
                    close(float(np.mean(v[branch]['per_map_ba'])), v[branch]['balanced_accuracy'])
                    ba_count += 1
                # Independently reconstruct interval membership from graph distances.
                masks = {'paired': changed, '0_8': (d>=0)&(d<8), '8_16': (d>=8)&(d<16),
                         'equal_16': d==16, 'strict_16_32': (d>16)&(d<32),
                         '32_64': (d>=32)&(d<64), '64_128': (d>=64)&(d<128),
                         '128_inf': d>=128, 'far_gt16': d>16, 'far_gt32': d>32,
                         'r_le1': (d>=0)&(d<=2*k), 'r_1_2': (d>2*k)&(d<=4*k),
                         'r_gt2': d>4*k, 'outside_forward_lightcone': d>2*int(t)}
                metrics = {'paired': v['paired'], **v['bands']}
                assert set(masks) == set(metrics)
                for name, m in metrics.items():
                    counts = (changed & masks[name]).sum((1, 2, 3))
                    assert counts.tolist() == m['per_map_pixels']
                    hits = np.array(m['per_map_correct'])
                    assert len(hits) == 32 and np.all((0<=hits)&(hits<=counts))
                    valid = []
                    for n, hit, fraction in zip(counts, hits, m['per_map']):
                        if n:
                            close(hit/n, fraction)
                            valid.append(hit/n)
                        else:
                            assert fraction is None
                    assert len(valid) == m['eligible_maps']
                    assert int(counts.sum()) == m['pooled_pixels'] and int(hits.sum()) == m['pooled_correct']
                    if valid:
                        close(float(np.mean(valid)), m['mean'])
                        close(hits.sum()/counts.sum(), m['pooled_accuracy'])
                    else:
                        assert m['mean'] is None and m['pooled_accuracy'] is None
                    paired_count += 1
                partition = ['0_8', '8_16', 'equal_16', 'strict_16_32', '32_64', '64_128', '128_inf']
                assert np.array_equal(sum(np.array(metrics[b]['per_map_correct']) for b in partition),
                                      metrics['paired']['per_map_correct'])
                assert v['outside_forward_lightcone_max_logit_difference'] <= 1e-6
                row = {'seed': seed, 'K': k, 'size': int(size), 'T': int(t),
                       'BA': v['original']['balanced_accuracy'], 'flipped_BA': v['flipped']['balanced_accuracy']}
                for name in ('strict_16_32', 'far_gt16', 'far_gt32', '32_64', '64_128', '128_inf'):
                    for field in ('mean', 'pooled_accuracy', 'eligible_maps', 'pooled_pixels'):
                        row[name+'_'+field] = metrics[name][field]
                curves.append(row)
        primary = r['evaluation']['32']['64']
        m = primary['bands']['strict_16_32']
        reach = all(primary[b]['balanced_accuracy'] >= .85 for b in ('original', 'flipped')) and m['mean'] >= .8 and m['pooled_accuracy'] >= .8
        hold = all(all(r['evaluation']['32'][t][b]['balanced_accuracy'] >= primary[b]['balanced_accuracy']-.03 for b in ('original', 'flipped'))
                   and all(r['evaluation']['32'][t]['bands']['strict_16_32'][f] >= m[f]-.05 for f in ('mean', 'pooled_accuracy')) for t in ('128', '256'))
        old = next(x for x in agg['rows'] if x['seed'] == seed and x['K'] == k)
        assert old['reach'] == reach and old['hold'] == hold
        close(old['primary_mean'], m['mean'])
        close(old['primary_pooled'], m['pooled_accuracy'])
        verdicts.append({'seed': seed, 'K': k, 'reach': reach, 'hold': hold,
                         'primary_mean': m['mean'], 'primary_pooled': m['pooled_accuracy']})
        records.append({'seed': seed, 'K': k, 'training_seconds': r['training_seconds'],
                        'peak_MiB': r['training_peak_allocated_bytes']/2**20,
                        'clip_fraction': r['gradient_clip_fraction']})
    assert len(records) == 12 and len(curves) == 72
    counts = {str(k): {'reach': sum(v['K']==k and v['reach'] for v in verdicts),
                      'reach_and_hold': sum(v['K']==k and v['reach'] and v['hold'] for v in verdicts)} for k in (8,16,64)}
    for k, c in counts.items():
        assert all(agg['counts'][k][name] == value for name, value in c.items())
    assert agg['K8_narrow_reach'] == ('REPLICATED' if counts['8']['reach'] >= 3 else 'NOT_REPLICATED')
    assert agg['K8_narrow_sustained'] == ('REPLICATED' if counts['8']['reach_and_hold'] >= 3 else 'NOT_REPLICATED')
    assert agg['comparison_status'] == ('FULL_CONTROL_QUALIFIED' if counts['64']['reach'] >= 3 else 'BASELINE_UNQUALIFIED')
    for effect in agg['paired_effects']:
        short = next(x for x in agg['rows'] if x['seed']==effect['seed'] and x['K']==effect['K'])
        full = next(x for x in agg['rows'] if x['seed']==effect['seed'] and x['K']==64)
        for key in ('ba_original', 'ba_flipped', 'primary_mean', 'primary_pooled'):
            close(effect[key+'_effect_pp'], 100*(short[key]-full[key]))
    validation = {'status': 'PASS', 'source_files': len(manifest['source_sha256']), 'checkpoints': 12,
                  'data_schedule_initial_parameters': True, 'BA_aggregates': ba_count,
                  'paired_aggregates_and_denominators': paired_count, 'arm_decisions': 12,
                  'new_training_or_inference': False}
    out.mkdir(parents=True)
    with (out/'curves.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(sorted(curves, key=lambda x: (x['seed'], x['K'], x['size'], x['T'])))
    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write('analysis.json', {'execution': status, 'counts': counts, 'verdicts': verdicts,
                            'systems': records, 'validation': validation})
    lines = ['# Phase-II result inspection', '',
             f"All12 arms completed300 updates in{status['elapsed_seconds']:.3f}s. Frozen K8 reach and sustained replication: {agg['K8_narrow_reach']} / {agg['K8_narrow_sustained']}.",
             f"Comparison status: {agg['comparison_status']}. Fixed train bank/schedule, four initialization seeds; conditional descriptive result.", '',
             '| K | Reach count | Reach and hold count |', '|---|---:|---:|']
    lines += [f"| {k} | {c['reach']}/4 | {c['reach_and_hold']}/4 |" for k,c in counts.items()]
    lines += ['', 'Primary size32/T64 STRICT16<d<32, mean and pooled>=80%, both BAs>=85%. K8 requires>=3/4 seeds.', '',
              '| Seed | K | Primary mean % | Primary pooled % | Reach | Hold |', '|---|---:|---:|---:|---|---|']
    lines += [f"| {v['seed']} | {v['K']} | {100*v['primary_mean']:.2f} | {100*v['primary_pooled']:.2f} | {v['reach']} | {v['hold']} |" for v in sorted(verdicts,key=lambda x:(x['seed'],x['K']))]
    lines += ['', '## Interpretation', '',
              '- K8 has two new positive cases (seeds2/5), both retained throughT256. The predefined>=3/4 initialization replication failed; do not call the whole screen passed.',
              '- K16 reaches the narrow endpoint in3/4 seeds, but only1/4 meets reach+hold. This band fits within its32-edge window; it does not show cross-window composition forK16.',
              '- All four K64 controls remain unqualified. K8 is also worse than K64 on seed3. No generally optimal truncation horizon or superiority claim.',
              '- Low clipping is not sufficient: all K8 clip<=6%, including the two reach failures. High clipping is not invariably failure: K16 seed2 clips60% and passes reach+hold. These observations do not isolate a mechanism.',
              '- Size32 evaluation has no changed-component pixels at distance>=64. Those entries are null, not zero. Use size64 for this distance range.', '',
              '## Successful K8 cases: limits of distance/size transfer', '',
              '| Seed | Size | T | Primary mean / pooled % | d>32 mean / pooled % | d>16 mean / pooled % |', '|---|---:|---:|---:|---:|---:|']
    for c in sorted(curves,key=lambda x:(x['seed'],x['size'],x['T'])):
        if c['K'] == 8 and c['seed'] in (2,5):
            pair = lambda band: f"{100*c[band+'_mean']:.2f} / {100*c[band+'_pooled_accuracy']:.2f}"
            lines.append(f"| {c['seed']} | {c['size']} | {c['T']} | {pair('strict_16_32')} | {pair('far_gt32')} | {pair('far_gt16')} |")
    lines += ['', 'Both successful K8 cases retain high narrow-band scores at size64, but broad far accuracy remains much weaker. Extra rollout helps farther propagation for seed2; seed5 improves less. This is not robust global propagation.', '',
              '## Systems and verification', '', '| Seed | K | Train seconds | Peak MiB | Clipping % |', '|---|---:|---:|---:|---:|']
    lines += [f"| {r['seed']} | {r['K']} | {r['training_seconds']:.2f} | {r['peak_MiB']:.2f} | {100*r['clip_fraction']:.2f} |" for r in records]
    lines += ['', f"Verified{validation['source_files']} source snapshots,12 checkpoints, shared data/schedule and initial parameters,{ba_count} BA aggregates,{paired_count} paired aggregates with reconstructed denominators, all12 decisions and paired effects. No new training or inference.", '',
              'Narrow band and architecture were selected from Phase I; new evaluation maps were frozen prospectively. The previous gate remains unchanged. n=4 is initialization replication conditional on one bank/schedule, not significance or training-distribution robustness.', '',
              '[All72 curve rows](curves.csv), [structured analysis](analysis.json), [provenance](provenance.json).']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write('provenance.json', {'source_run': run.relative_to(ROOT).as_posix(),
                             'input_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(run.glob('*.json'))},
                             'checkpoint_sha256': {f.name: sha(f) for f in sorted(run.glob('*.pt'))},
                             'analysis_source_sha256': {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))},
                             'output_sha256': {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file()}})
    print(json.dumps({'counts': counts, 'validation': validation}, indent=2))


if __name__ == '__main__':
    main()
