"""Publish saved seed4 frontier evidence; no training or model inference."""
import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT/'FRONTIER_AUDIT_PUBLICATION_MANIFEST.json'
PRIVATE = {'pid', 'host', 'hostname', 'username', 'gpu', 'gpu_uuid', 'device',
           'device_uuid', 'command', 'argv', 'cwd', 'launch_command', 'working_directory'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def public(value):
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k.lower() not in PRIVATE}
    if isinstance(value, list):
        return [public(v) for v in value]
    return value


def equal(a, b):
    if a is None or b is None:
        assert a is b, (a, b)
    else:
        assert np.isclose(a, b, atol=1e-12, rtol=1e-12), (a, b)


def verify(out):
    """Recompute the matched effects from public integer CSV counts."""
    summary = read(out/'summary.json')
    assert summary['model_seed'] == 4 and summary['model_replications'] == 1 and not summary['training']
    replay = read(out/'replay_validation.json')
    assert replay['status'] == 'PASS' and replay['size_horizon_records'] == 6
    assert replay['maximum_absolute_error'] == 0
    assert read(out/'validation.json')['status'] == 'PASS'
    stats = {'status': 'PASS', 'training': False, 'model_inference': False,
             'matched_strata': 0, 'maps': 0, 'intervals': 0, 'transition_accounting': 0,
             'paired_endpoint_aggregates': 0}
    for size in (32, 64):
        row = summary['sizes'][str(size)]
        matched = defaultdict(lambda: [0., 0., 0])
        with (out/f'frontier_matches_size{size}.csv').open(encoding='utf-8', newline='') as handle:
            for saved in csv.DictReader(handle):
                i, t, end, d, nf, af, nn, an = [int(saved[k]) for k in
                    ('map_index', 'from_step', 'to_step', 'distance', 'frontier_opportunities',
                     'frontier_acquired', 'nonfrontier_opportunities', 'nonfrontier_acquired')]
                assert 0 <= i < 32 and t in range(0, 256, 8) and end == t+1 and d >= 0
                assert nf > 0 and nn > 0 and 0 <= af <= nf and 0 <= an <= nn
                weight, effect = nf*nn/(nf+nn), af/nf-an/nn
                equal(float(saved['weight']), weight)
                equal(float(saved['acquisition_difference']), effect)
                group = matched[(i, t)]
                group[0] += weight
                group[1] += weight*effect
                group[2] += 1
                stats['matched_strata'] += 1
        effects = []
        for i, m in enumerate(row['frontier']['per_map']):
            assert m['map_index'] == i
            groups = [v for (j, _), v in matched.items() if j == i]
            weight = sum(g[0] for g in groups)
            term = sum(g[1] for g in groups)
            effect = term/weight if weight else None
            equal(m['weight_sum'], weight)
            equal(m['weighted_difference'], effect)
            assert m['strata_used'] == sum(g[2] for g in groups)
            if effect is not None:
                effects.append(effect)
            stats['maps'] += 1
        equal(row['frontier']['mean_map_weighted_difference'], np.mean(effects) if effects else None)
        equal(row['frontier']['median_map_weighted_difference'], np.median(effects) if effects else None)
        assert row['frontier']['eligible_maps'] == len(effects)
        assert row['frontier']['strata_used'] == sum(g[2] for g in matched.values())
        for interval in row['frontier']['by_start']:
            values = [v[1]/v[0] for (_, t), v in matched.items() if t == interval['from_step']]
            assert interval['maps_with_matched_strata'] == len(values)
            equal(interval['mean_across_maps'], np.mean(values) if values else None)
            equal(interval['median_across_maps'], np.median(values) if values else None)
            stats['intervals'] += 1
        for rows in row['transitions'].values():
            for tr in rows:
                p = tr['pooled']
                assert p['from_correct'] == p['retained']+p['lost']
                assert p['to_correct'] == p['retained']+p['gained']
                assert 0 <= p['from_correct'] <= p['pixels'] and 0 <= p['to_correct'] <= p['pixels']
                equal(p['retention']['value'], p['retained']/p['from_correct'] if p['from_correct'] else None)
                stats['transition_accounting'] += 1
        for endpoint in row['endpoints'].values():
            for name in ('paired', 'strict_16_32', 'far_gt32'):
                p = endpoint[name]
                assert sum(p['per_map_pixels']) == p['pooled_pixels']
                assert sum(p['per_map_correct']) == p['pooled_correct']
                equal(p['pooled_accuracy'], p['pooled_correct']/p['pooled_pixels'] if p['pooled_pixels'] else None)
                stats['paired_endpoint_aggregates'] += 1
    assert stats['matched_strata'] == 6882
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run')
    parser.add_argument('--analysis')
    parser.add_argument('--out', default='evidence/frontier_audit_seed4')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    assert out.is_relative_to(ROOT/'evidence')
    if args.verify_only:
        publication = read(PUBLICATION)
        for group in ('source_sha256', 'publication_tool_sha256', 'published_evidence_sha256'):
            for name, expected in publication[group].items():
                assert sha(ROOT/name) == expected, name
        print(json.dumps(verify(out)))
        return
    assert args.run and args.analysis and not out.exists() and not PUBLICATION.exists()
    run, analysis = Path(args.run).resolve(), Path(args.analysis).resolve()
    manifest, status, original, validation = [read(p) for p in
        (run/'manifest.json', run/'status.json', run/'summary.json', analysis/'validation.json')]
    assert status['status'] == 'COMPLETE' and validation['status'] == 'PASS'
    assert validation['run_manifest_sha256'] == sha(run/'manifest.json')
    assert validation['summary_sha256'] == sha(run/'summary.json')
    assert manifest['git_review_base'] == '2e550abfeb1bf53e1348dae7f5653fa6ccd8e47b'
    assert manifest['model_seed'] == original['model_seed'] == 4
    assert manifest['model_replications'] == original['model_replications'] == 1
    assert manifest['training'] is original['training'] is False
    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT/name) == sha(run/'source'/name) == expected, name
    output = read(run/'output_manifest.json')
    assert output['parameters_and_checkpoint_unchanged']
    for name, expected in output['sha256'].items():
        assert sha(run/name) == expected, name
    assert sha(ROOT/'runs/streaming_carry_20261002_init2345/stream_K8_seed4.pt') == manifest['checkpoint_file_sha256']
    out.mkdir(parents=True, exist_ok=False)
    copied = {}
    for source, name in [(run/'replay_validation.json', 'replay_validation.json'),
        (run/'behavior_curves.png', 'behavior_curves.png'), (run/'acquisition_maps.png', 'acquisition_maps.png'),
        (analysis/'validation.json', 'validation.json'), (analysis/'within_interval.json', 'within_interval.json'),
        (ROOT/'analyses/frontier_audit_20261003_seed4_preflight01.json', 'cpu_validation.json')]:
        shutil.copyfile(source, out/name)
        assert sha(source) == sha(out/name)
        copied[name] = sha(source)
    assert read(out/'cpu_validation.json') == {'checks': 5, 'status': 'PASS'}
    compact = copy.deepcopy(original)
    for size in compact['sizes'].values():
        for rows in size['transitions'].values():
            for row in rows:
                del row['per_map']
    write(out/'summary.json', compact)
    write(out/'manifest.json', public(manifest))
    write(out/'completion.json', public(status))
    columns = ('map_index', 'from_step', 'to_step', 'distance', 'frontier_opportunities',
               'frontier_acquired', 'nonfrontier_opportunities', 'nonfrontier_acquired',
               'weight', 'acquisition_difference')
    for size in (32, 64):
        behavior = read(run/f'behavior_size{size}.json')
        with (out/f'frontier_matches_size{size}.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, columns)
            writer.writeheader()
            for m in behavior['maps']:
                for interval in m['frontier']:
                    for d in interval['exact_distance']:
                        if d['matched_weight'] is None:
                            continue
                        saved = {k: d[k] for k in columns[3:8]}
                        saved.update(map_index=m['map_index'], from_step=interval['from_step'],
                            to_step=interval['to_step'], weight=d['matched_weight'],
                            acquisition_difference=d['matched_rate_difference'])
                        writer.writerow(saved)
    report = (run/'RESULTS.md').read_text(encoding='utf-8')
    report = report.replace('behavior_size32.json / behavior_size64.json contain per-map/time/exact-BFS-distance '
        'frontier and nonfrontier opportunities and acquisitions.',
        'The full local behavior files contain every map/time/exact-BFS-distance stratum. '
        'Public frontier_matches_size32.csv / frontier_matches_size64.csv contain the common strata '
        'used in the matched comparison, with integer opportunities/acquisitions.')
    report = report.replace('2. replay_validation.json and the separately saved CPU validation.',
        '2. [Replay](replay_validation.json), [full saved-trace validation](validation.json), '
        '[publication arithmetic](publication_validation.json), and [diagnosis](DIAGNOSIS.md).')
    report = report.replace('3. behavior_size*.json for denominators and per-map detail; trace_size*.npz is secondary.',
        '3. [Size32 matched strata](frontier_matches_size32.csv) and [size64 matched strata](frontier_matches_size64.csv). '
        'Full behavior JSON, evaluation payloads and compressed traces remain local; '
        '[provenance](provenance.json) binds their hashes. No raw arrays need to be opened first.')
    report += '\n## Reproduction from this repository\n\n'
    report += 'Public matched effects and hash bindings can be checked without checkpoints:\n\n'
    report += '```\npython tools/export_frontier_audit.py --verify-only\npython new/frontier_audit/check_metrics.py\n```\n\n'
    report += ('Full trajectory replay requires the original local checkpoint and run/source bindings '
               'named by the frozen protocol; checkpoints and full traces are excluded from GitHub. '
               'The public checker verifies exported arithmetic, not a new model rollout.\n')
    (out/'RESULTS.md').write_text(report, encoding='utf-8')
    diagnosis = (analysis/'DIAGNOSIS.md').read_text(encoding='utf-8').replace(
        '../../runs/frontier_audit_20261003_seed4_01/', '')
    (out/'DIAGNOSIS.md').write_text(diagnosis, encoding='utf-8')
    write(out/'publication_validation.json', verify(out))
    write(out/'provenance.json', {
        'review_base': manifest['git_review_base'], 'training': False, 'model_replications': 1,
        'original_private_manifest_sha256': sha(run/'manifest.json'),
        'original_private_completion_sha256': sha(run/'status.json'),
        'original_summary_sha256': sha(run/'summary.json'),
        'original_report_sha256': sha(run/'RESULTS.md'),
        'original_diagnosis_sha256': sha(analysis/'DIAGNOSIS.md'),
        'local_artifacts_sha256': output['sha256'], 'byte_identical_copies_sha256': copied,
        'sanitized_copies': ['manifest.json', 'completion.json'],
        'summary_transformation': 'Only redundant per-map transition rows removed; all other fields unchanged.',
        'report_transformation': 'Route local validation/artifact references to public copies and add reproduction commands; metrics unchanged.',
        'diagnosis_transformation': 'Relative local run links changed to public siblings; text and metrics unchanged.',
        'matched_csv_scope': 'Only strata with both groups; exact integer counts and weights copied from local behavior JSON.',
        'validation_scope': 'Full saved-trace validation refers to original local run hashes; publication checker verifies public counts and bindings.',
        'excluded': ['checkpoints', 'source snapshots', 'full traces', 'full behavioral JSON', 'full evaluation payloads',
                     'private manifest/completion originals', 'launch metadata', 'logs'],
    })
    for path in out.iterdir():
        assert path.stat().st_size < 1_000_000, path.name
        if path.suffix == '.json':
            assert read(path) == public(read(path)), path.name
    write(PUBLICATION, {
        'review_base': manifest['git_review_base'], 'execution_status': 'COMPLETE', 'training': False,
        'model_seed': 4, 'model_replications': 1,
        'scientific_scope': 'Selected-checkpoint descriptive output behavior through T256; no new architecture gate.',
        'source_sha256': manifest['source_sha256'],
        'publication_tool_sha256': {'tools/export_frontier_audit.py': sha(Path(__file__))},
        'checkpoint_file_sha256': manifest['checkpoint_file_sha256'],
        'checkpoint_parameter_sha256': manifest['checkpoint_parameter_sha256'],
        'published_evidence_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(out.iterdir())},
        'notes': ['Earlier architecture no-go decisions unchanged; maps reused, selected model n=1.',
                  'High endpoint retention coexists with every-step relapse; terminal stability ends at256.',
                  'Matched frontier acquisition is association, not causal handoff or flood-fill identification.',
                  'No training or trained-checkpoint inference performed for publication.'],
    })
    print(json.dumps({'status': 'EXPORTED', 'files': len(list(out.iterdir())),
                      'total_bytes': sum(p.stat().st_size for p in out.iterdir()), 'source_bindings': len(manifest['source_sha256'])}))


if __name__ == '__main__':
    main()
