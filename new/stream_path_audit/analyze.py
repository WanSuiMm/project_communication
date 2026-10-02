"""Independent CPU arithmetic/provenance checks of the saved operator audit."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'new/nca_inertial_wind_tunnel'))
from tasks import bank


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    if a is None or b is None:
        assert a is None and b is None
    else:
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6), (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    assert not out.exists()
    manifest, status, summary, raw = [read(run/name) for name in
        ('manifest.json', 'status.json', 'summary.json', 'raw_conditions.json')]
    assert status['status'] == summary['status'] == 'COMPLETE'
    assert status['completed_conditions'] == 4 and status['elapsed_seconds'] <= 300
    assert summary['training'] is False and summary['model_seed'] == 4
    assert summary['replay']['maximum_absolute_error'] == 0
    assert summary['replay']['size_horizon_records'] == 6
    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT/name) == sha(run/'source'/name) == expected, name
    checkpoint = ROOT/'runs/streaming_carry_20261002_init2345/stream_K8_seed4.pt'
    assert sha(checkpoint) == manifest['checkpoint_file_sha256']
    assert set(raw) == {'full', 'no_transport', 'no_perception', 'neither'}
    metrics = {}; checked = ba_checked = 0
    for size in (32, 64):
        data = bank(size, 32, 40000+size)
        d, changed = data['distance'].numpy(), data['changed'].numpy().astype(bool)
        for condition, sizes in raw.items():
            assert set(sizes) == {'32', '64'}
            assert set(sizes[str(size)]) == {'8', '16', '32', '64', '128', '256'}
            for t, row in sizes[str(size)].items():
                bands = {'all_changed': changed, '0_8': (d>=0)&(d<8), '8_16': (d>=8)&(d<16),
                         'equal_16': d==16, 'strict_16_32': (d>16)&(d<32),
                         '32_64': (d>=32)&(d<64), '64_128': (d>=64)&(d<128),
                         '128_inf': d>=128, 'far_gt16': d>16, 'far_gt32': d>32,
                         'r_le1': (d>=0)&(d<=16), 'r_1_2': (d>16)&(d<=32),
                         'r_gt2': d>32, 'outside_forward_lightcone': d>2*int(t)}
                e = row['evaluation']
                observed = {'all_changed': e['paired'], **e['bands']}
                assert set(bands) == set(observed)
                for name, metric in observed.items():
                    counts = (changed & bands[name]).sum((1, 2, 3))
                    hits = np.asarray(metric['per_map_correct'])
                    assert metric['per_map_pixels'] == counts.tolist()
                    assert len(hits) == len(metric['per_map']) == 32
                    assert np.all((0 <= hits) & (hits <= counts))
                    ratios = []
                    for n, h, ratio in zip(counts, hits, metric['per_map']):
                        close(h/n if n else None, ratio)
                        if n:
                            ratios.append(h/n)
                    assert metric['pooled_pixels'] == int(counts.sum())
                    assert metric['pooled_correct'] == int(hits.sum())
                    assert metric['eligible_maps'] == len(ratios)
                    close(float(np.mean(ratios)) if ratios else None, metric['mean'])
                    close(int(hits.sum())/int(counts.sum()) if ratios else None, metric['pooled_accuracy'])
                    metrics[condition, size, int(t), name] = metric
                    checked += 1
                partition = ('0_8', '8_16', 'equal_16', 'strict_16_32', '32_64', '64_128', '128_inf')
                assert np.array_equal(sum(np.array(observed[n]['per_map_correct']) for n in partition), e['paired']['per_map_correct'])
                for branch in ('original', 'flipped'):
                    close(float(np.mean(e[branch]['per_map_ba'])), e[branch]['balanced_accuracy'])
                    ba_checked += 1
                radius = {'full': 2, 'no_transport': 2, 'no_perception': 1, 'neither': 0}[condition]*int(t)
                diag = row['diagnostics']
                assert diag['actual_hop_radius_upper_bound'] == radius
                assert diag['actual_outside_lightcone_pixels'] == int((changed & (d>radius)).sum())
                assert diag['actual_outside_lightcone_max_logit_difference'] <= 1e-6
                if condition == 'neither':
                    assert all(observed[b]['pooled_correct'] == 0 for b in ('8_16', 'equal_16', 'strict_16_32', 'far_gt32'))
    with (run/'curves.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == checked
    seen = set()
    for row in rows:
        key = (row['condition'], int(row['size']), int(row['T']), row['band'])
        assert key not in seen
        seen.add(key)
        p = metrics[key]
        for field, metric in (('correct', 'pooled_correct'), ('pixels', 'pooled_pixels'), ('eligible_maps', 'eligible_maps')):
            assert int(row[field]) == p[metric]
        for field, metric in (('mean', 'mean'), ('pooled', 'pooled_accuracy')):
            close(float(row[field]) if row[field] else None, p[metric])
    assert seen == set(metrics)
    contrasts = read(run/'contrasts.json')
    seen = set()
    for row in contrasts:
        key = (int(row['size']), int(row['T']), row['band'])
        identity = (row['condition'], *key)
        assert identity not in seen
        seen.add(identity)
        if row['condition'] == 'factorial_interaction':
            for field, metric in (('pooled_effect_pp', 'pooled_accuracy'), ('mean_effect_pp', 'mean')):
                value = sum(sign*metrics[name, *key][metric] for name, sign in
                            (('full', 1), ('no_transport', -1), ('no_perception', -1), ('neither', 1)))
                close(100*value, row[field])
        else:
            a, b = metrics['full', *key], metrics[row['condition'], *key]
            close(100*(b['mean']-a['mean']), row['mean_effect_pp'])
            close(100*(b['pooled_accuracy']-a['pooled_accuracy']), row['pooled_effect_pp'])
            for aa, bb, value in zip(a['per_map'], b['per_map'], row['per_map_effect_pp']):
                close(None if aa is None else 100*(bb-aa), value)
    expected = {(condition, size, t, band) for condition in ('no_transport', 'no_perception', 'neither', 'factorial_interaction')
                for name, size, t, band in metrics if name == 'full' and metrics[name, size, t, band]['mean'] is not None}
    assert seen == expected
    result = {'status': 'PASS', 'source_bindings': len(manifest['source_sha256']),
              'paired_aggregates_and_denominators': checked, 'ba_means': ba_checked,
              'curve_rows': len(rows), 'contrasts': len(contrasts),
              'replay_maximum_error': 0.0,
              'analysis_source_sha256': sha(Path(__file__)),
              'input_sha256': {n: sha(run/n) for n in ('manifest.json', 'status.json', 'summary.json', 'raw_conditions.json', 'curves.csv', 'contrasts.json')}}
    out.mkdir(parents=True, exist_ok=False)
    (out/'validation.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
