"""Publish the completed bootstrap-path screen; verification repeats no training."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/bootstrap_20261004_seed4_01'
PUBLIC = ROOT/'evidence/bootstrap_path_20261004'
MANIFEST = ROOT/'BOOTSTRAP_PATH_PUBLICATION_MANIFEST.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(a, b, tolerance=1e-10):
    if a is None or b is None:
        assert a is b
    else:
        assert math.isclose(a, b, abs_tol=tolerance, rel_tol=tolerance), (a, b)


def ratios(value):
    if isinstance(value, dict):
        if set(('numerator', 'denominator', 'value')) <= value.keys():
            n, d = value['numerator'], value['denominator']
            equal(value['value'], n/d if d else None)
        for v in value.values():
            ratios(v)
    elif isinstance(value, list):
        for v in value:
            ratios(v)


def gates(summary):
    g = summary['phenotype_gate']
    for row in g['checks'].values():
        v, threshold = row['value'], row['threshold']
        expected = v is not None and (v >= float(threshold[2:]) if threshold.startswith('>=')
                                     else v <= float(threshold[2:]))
        assert row['pass'] == expected
    assert g['pass'] == all(v['pass'] for v in g['checks'].values())


def validate_local(manifest, aggregate):
    """Check saved counts against Boolean traces and frontier integer counts."""
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT/'new/nca_inertial_wind_tunnel'))
    from tasks import bank
    sys.path.insert(0, str(ROOT/'new/workspace_revision'))
    from run_revision import tensor_hash
    banks = {n: bank(n, 32, 50000+n) for n in (32, 64)}
    assert {str(n): tensor_hash(d) for n, d in banks.items()} == manifest['fixed_evaluation_data_sha256']
    validated = 0
    for arm in aggregate['contrasts']:
        name = arm['name']
        raw = read(RUN/f'{name}.json')
        s = read(RUN/f'{name}_summary.json')
        assert raw['status'] == 'COMPLETE' and raw['completed_updates'] == 300
        assert raw['initial_parameter_sha256'] == read(RUN/'H.json')['initial_parameter_sha256']
        assert raw['parameter_count'] == 5033 and not raw['optimizer_reset_at_splice']
        assert sha(RUN/f'{name}_summary.json') == raw['summary_sha256']
        assert raw['endpoint'] == arm['endpoint'] and raw['phenotype_gate'] == s['phenotype_gate']
        gates(s); ratios(s)
        for size, data in banks.items():
            row = s['sizes'][str(size)]
            with np.load(RUN/f'{name}_size{size}.npz') as traces:
                c = traces['correct']
                assert c.shape == (257, 32, size, size)
                assert np.array_equal(c, traces['original_correct'] & traces['flipped_correct'])
                changed = data['changed'][:, 0].numpy().astype(bool)
                distance = data['distance'][:, 0].numpy()
                for selection, mask in (('all_changed', changed), ('strict_16_32', changed & (distance > 16) & (distance < 32))):
                    denominators = mask.sum(axis=(1, 2)); eligible = denominators > 0
                    for t in (64, 128, 256):
                        e = row['endpoints'][str(t)][selection]
                        counts = (c[t] & mask).sum(axis=(1, 2))
                        assert e['correct_pixels'] == int(counts.sum()) and e['pixels'] == int(denominators.sum())
                        equal(e['mean_map_coverage'], float((counts[eligible]/denominators[eligible]).mean()))
                    for start, end in ((64, 128), (64, 256), (128, 256)):
                        tr = row['transitions'][selection][f'{start}_to_{end}']['pooled']
                        before, after = c[start] & mask, c[end] & mask
                        assert tr['from_correct'] == int(before.sum())
                        assert tr['retained'] == int((before & after).sum())
                        assert tr['lost'] == int((before & ~after).sum())
                        assert tr['gained'] == int((~before & after).sum())
                ever = np.any(c, axis=0) & changed
                regressed = np.any(c[:-1] & ~c[1:], axis=0) & changed
                assert row['ever_regressed_over_ever_correct']['numerator'] == int(regressed.sum())
                assert row['ever_regressed_over_ever_correct']['denominator'] == int(ever.sum())
                for branch, label in (('original', 'y'), ('flipped', 'y_flip')):
                    truth = data[label][:, 0].numpy() >= .5
                    mask = data['mask'][:, 0].numpy().astype(bool)
                    for t in (64, 128, 256):
                        good = traces[branch+'_correct'][t]
                        pos, neg = mask & truth, mask & ~truth
                        pn, nn = pos.sum((1, 2)), neg.sum((1, 2))
                        ba = ((good & pos).sum((1, 2))/np.maximum(pn, 1)+(good & neg).sum((1, 2))/np.maximum(nn, 1))/np.maximum((pn>0).astype(int)+(nn>0), 1)
                        equal(row['endpoints'][str(t)][branch+'_ba'], float(ba.mean()), 1e-7)
        frontier = {32: {}, 64: {}}
        with (RUN/f'{name}_matched_frontier_strata.csv').open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            n, m = int(r['size']), int(r['map_index'])
            nf, nn = int(r['frontier_opportunities']), int(r['nonfrontier_opportunities'])
            gf, gn = int(r['frontier_acquired']), int(r['nonfrontier_acquired'])
            weight, diff = nf*nn/(nf+nn), gf/nf-gn/nn
            equal(float(r['matched_weight']), weight); equal(float(r['rate_difference']), diff)
            frontier[n].setdefault(m, []).append((weight, diff))
        for n, maps in frontier.items():
            f = s['sizes'][str(n)]['frontier']
            effects = [sum(w*d for w, d in terms)/sum(w for w, _ in terms) for terms in maps.values()]
            assert f['eligible_maps'] == len(effects)
            assert f['common_strata'] == sum(len(terms) for terms in maps.values())
            equal(f['mean_map_weighted_difference'], sum(effects)/len(effects))
        validated += 1
    return {'status': 'PASS', 'validated_arms': validated, 'trace_banks': validated*2,
            'checks': ['source/snapshot hashes', 'control replays', '300-update identities',
                       'endpoint coverage and paired conjunction', 'retention counts',
                       'every-step regression', 'open-grid BA', 'frontier integer arithmetic', 'all frozen gate decisions'],
            'scope': 'Saved evidence arithmetic only; no new model inference or training.'}


def build():
    assert not PUBLIC.exists() and not MANIFEST.exists(), 'Never overwrite a published evidence snapshot'
    status, m, a = read(RUN/'status.json'), read(RUN/'manifest.json'), read(RUN/'aggregate.json')
    assert status['status'] == 'COMPLETE' and status['completed_arms'] == 23
    assert a['complete'] and a['completed_arms'] == 23 and a['historical_control_qualified']
    for name, digest in m['source_sha256'].items():
        assert sha(ROOT/name) == digest and sha(RUN/'source'/name) == digest
    h = read(RUN/'H.json')
    assert h['historical_replay']['status'] == 'PASS' and h['historical_replay']['maximum_absolute_error'] == 0
    for seed in (20012, 20022):
        r = read(RUN/f'S{seed}.json')
        assert r['prior_final_parameter_replay'] == 'PASS' and r['prior_phenotype_replay']['status'] == 'PASS'
    validation = validate_local(m, a)
    PUBLIC.mkdir(parents=True)
    (PUBLIC/'arms').mkdir()
    (PUBLIC/'training').mkdir()
    (PUBLIC/'frontier').mkdir()
    (PUBLIC/'traces').mkdir()
    (PUBLIC/'first_step_arrays').mkdir()
    raw_hashes = {'status.json': sha(RUN/'status.json'), 'aggregate.json': sha(RUN/'aggregate.json'),
                  'manifest.json': sha(RUN/'manifest.json'), 'first_step/summary.json': sha(RUN/'first_step/summary.json')}
    for arm in a['contrasts']:
        name = arm['name']
        shutil.copyfile(RUN/f'{name}_summary.json', PUBLIC/'arms'/f'{name}.json')
        raw = read(RUN/f'{name}.json')
        keys = ('name', 'kind', 'alternate_seed', 'k', 'edited_updates', 'status', 'completed_updates',
                'parameter_count', 'initial_parameter_sha256', 'final_parameter_sha256', 'schedule_sha256',
                'optimizer_reset_at_splice', 'training_curve', 'actual_first_step_replay', 'historical_replay',
                'prior_final_parameter_replay', 'prior_phenotype_replay')
        write(PUBLIC/'training'/f'{name}.json', {k: raw[k] for k in keys if k in raw})
        shutil.copyfile(RUN/f'{name}_matched_frontier_strata.csv', PUBLIC/'frontier'/f'{name}.csv')
        for size in (32, 64):
            shutil.copyfile(RUN/f'{name}_size{size}.npz', PUBLIC/'traces'/f'{name}_size{size}.npz')
        for suffix in ('.json', '_summary.json', '_size32.npz', '_size64.npz', '_matched_frontier_strata.csv'):
            raw_hashes[name+suffix] = sha(RUN/(name+suffix))
    write(PUBLIC/'summary.json', {**a, 'execution': {'status': 'COMPLETE', 'updates_per_arm': 300,
          'elapsed_seconds': status['elapsed_seconds'], 'runtime_limit_enforced': False},
          'control_replay': h['historical_replay']})
    shutil.copyfile(RUN/'first_step/summary.json', PUBLIC/'first_step_summary.json')
    for arrays in (RUN/'first_step').glob('*.npz'):
        shutil.copyfile(arrays, PUBLIC/'first_step_arrays'/arrays.name)
        raw_hashes['first_step/'+arrays.name] = sha(arrays)
    config_keys = ('protocol', 'seed', 'expected_formal_arms', 'updates_per_arm', 'arms', 'torch', 'numpy',
                   'threads', 'backend', 'optimizer', 'gradient_horizon', 'forward_steps', 'loss_times',
                   'optimizer_reset_at_splice', 'test_maps_reused', 'runtime_limit_enforced')
    write(PUBLIC/'config.json', {k: m[k] for k in config_keys})
    write(PUBLIC/'validation.json', validation)
    first = read(PUBLIC/'first_step_summary.json')
    successes = [r['name'] for r in a['contrasts'] if r['endpoint']['pass']]
    assert successes == ['H', 'S20022_replace_early3', 'S20022_preserve_early3']
    text = ['# Seed4 bootstrap-path screen: completed', '',
        f"23/23 arms x300 updates; {status['elapsed_seconds']/60:.2f} minutes. Historical H reproduces its parameters and six complete endpoints exactly. Both S controls reproduce prior results.", '',
        'Data-only publication for independent analysis. One init4 trajectory family, two selected alternate schedules and reused 2D size32/64 evaluation maps. `Full` is the unchanged conjunction of reach/hold/dynamics/frontier gates, not a size32-only endpoint.', '',
        '| Arm | Full | size32 strict mean T64 % | T128 % | T256 % |', '|---|---|---:|---:|---:|']
    for r in a['contrasts']:
        e = r['endpoint']; s = e['sizes']['32']['strict_mean']
        text.append(f"| {r['name']} | {e['pass']} | {100*s['64']:.2f} | {100*s['128']:.2f} | {100*s['256']:.2f} |")
    text += ['', 'Read [aggregate](summary.json), [first-step measurements](first_step_summary.json), [configuration](config.json), [frozen protocol](../../new/bootstrap_path/PROTOCOL.md), [validation](validation.json) and [reproduction](REPRODUCTION.md).', '',
        'Full per-arm `arms/` JSON retains denominators and gate failures. `training/` contains all300 update records. `frontier/` contains exact-distance integer counts. `traces/` contains46 compressed Boolean trajectory banks; `first_step_arrays/` contains six gradient/delta/moment/Z-write NPZ files. Large arrays are secondary. Checkpoints, PIDs, private machine manifests and logs remain local.']
    (PUBLIC/'RESULTS.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    (PUBLIC/'REPRODUCTION.md').write_text('''# Reproduction boundaries

From the repository root, `python -X utf8 -B tools/export_bootstrap_path.py --verify-only`
checks public hashes, ratios and gate decisions using the standard library.
The original CPU sanity check and CUDA preflight/formal commands are in the
[frozen protocol](../../new/bootstrap_path/PROTOCOL.md). They require Python,
NumPy and the historical PyTorch2.5.1 CUDA environment; no new dependency.

Exact retraining also expects the excluded reference archives explicitly named
in `new/bootstrap_path/run.py`: initial checkpoints2/3/4/5, historical schedule
and prior control records/manifests. A fresh clone can verify this publication
but cannot run the frozen runner without those archives. Training/evaluation
maps and schedule interventions are procedurally generated from recorded seeds.
Initial/model/data/schedule hashes and all negative endpoints are published.
No excluded archive is silently regenerated or overwritten. The validation
recomputed endpoint, retention, regression, BA and frontier arithmetic from
all46 published Boolean-trace banks; it did not repeat model execution.
''', encoding='utf-8')
    published = {p.relative_to(ROOT).as_posix(): sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
    write(MANIFEST, {'protocol': m['protocol'], 'source_sha256': m['source_sha256'],
          'exporter_sha256': sha(Path(__file__)), 'published_evidence_sha256': published,
          'local_raw_evidence_sha256': raw_hashes, 'data_sha256': {
              k: m[k] for k in ('train_data_sha256', 'fixed_evaluation_data_sha256', 'historical_evaluation_data_sha256')},
          'schedule_sha256': m['schedule_sha256'], 'reference_sha256': m['reference_sha256'],
          'excluded': ['checkpoints', 'launch receipt', 'machine manifest', 'logs'],
          'summary_exports': 'Arm/first-step summaries byte-identical; training records allowlisted; manifest/status sanitized.'})
    verify()


def verify():
    m = read(MANIFEST)
    for path, digest in {**m['source_sha256'], **m['published_evidence_sha256']}.items():
        assert sha(ROOT/path) == digest, path
    assert sha(Path(__file__)) == m['exporter_sha256']
    a = read(PUBLIC/'summary.json')
    assert a['complete'] and len(a['contrasts']) == 23
    for r in a['contrasts']:
        s = read(PUBLIC/'arms'/f'{r["name"]}.json')
        ratios(s); gates(s)
        assert r['endpoint']['pass'] == s['phenotype_gate']['pass']
    assert [r['name'] for r in a['contrasts'] if r['endpoint']['pass']] == ['H', 'S20022_replace_early3', 'S20022_preserve_early3']
    assert all(not v['strong_conditional_prefix_pattern'] for v in a['prefix_signatures'].values())
    print(json.dumps({'status': 'PASS', 'public_arms': 23, 'published_files': len(m['published_evidence_sha256'])}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    verify() if args.verify_only else build()
