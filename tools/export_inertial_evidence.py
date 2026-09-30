"""Publish the completed inertial screen without modifying frozen local evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'new/nca_inertial_wind_tunnel'
ARMS = ('nca_state_matched', 'momentum_nca', 'rd_nca', 'inertial_rd')


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')


def read(path):
    return json.loads(path.read_bytes())


def summarize(reports, manifest, status):
    summary = {
        'protocol': 'inertial_seeded_region_identity_seed0',
        'execution_status': 'COMPLETED_FROZEN_SCREEN',
        'scientific_status': 'NEGATIVE_EXPLORATORY_SCREEN',
        'scientific_status_scope': 'Descriptive interpretation after the frozen screen; not a preregistered binary efficacy gate.',
        'independent_training_seeds': [0], 'dimension': 2,
        'evaluation_examples_per_size': 16,
        'total_wall_seconds': (datetime.fromisoformat(status['finished_utc'])-
                               datetime.fromisoformat(manifest['started_utc'])).total_seconds(),
        'all_sustained_95_thresholds_missing': True,
        'candidate_conditional_repair_status': 'UNEVALUABLE_ZERO_ELIGIBLE',
        'not_established': ['benefit over generic momentum', 'matched-quality speedup',
                            'conditional repair advantage', 'multi-seed superiority',
                            '3D effectiveness', 'PDE/NCA-family impossibility'],
        'arms': [],
    }
    for arm in ARMS:
        r = reports[arm]
        item = {key: r[key] for key in ('status', 'seed', 'completed_updates',
                'training_seconds_including_python_and_first_compile',
                'train_peak_allocated_bytes', 'gradient_clip_fraction', 'gradient_probes')}
        item.update(arm=arm, initial_model=r['initial_model'], final_model=r['final_model'], evaluation={})
        for size, e in r['evaluation'].items():
            paired = e['paired_source_information']
            curves = {}
            for t, point in e['curve'].items():
                curves[t] = {key: point[key] for key in
                             ('balanced_accuracy', 'bce', 'content_rms', 'velocity_rms')}
                curves[t]['paired_correct'] = paired[t]['both_counterfactuals_correct_on_changed_component']
                curves[t]['median_seconds_per_query'] = e['latency_by_horizon'][t]['median_seconds_per_query']
            item['evaluation'][size] = {
                'curve': curves,
                'first_sustained_95_steps': e['first_sustained_95_steps'],
                'seconds_per_query_at_sustained_95': e['seconds_per_query_at_sustained_95'],
                'repair_eligible_at_T64': e['conditional_state_recovery']['0.0625']['clean_correct_eligible_count'],
            }
            if e['first_sustained_95_steps'] is not None:
                summary['all_sustained_95_thresholds_missing'] = False
        summary['arms'].append(item)
    return summary


def report(summary):
    lines = ['# Inertial NCA: completed seed-0 screen', '',
             'Four arms completed 800 updates each and all evaluations. No arm hit the wall-clock cap or a nonfinite status.',
             'This is a negative exploratory result for this recipe. It is not a rejection of the architecture family.', '',
             'Balanced accuracy (%), mean across 16 held-out maps at each size. One training seed is the replicate count.', '',
             '| Arm | 32x32, T64 | 32x32, T256 | 64x64, T256 | 128x128, T256 |',
             '|---|---:|---:|---:|---:|']
    for r in summary['arms']:
        e = r['evaluation']
        scores = [e[s]['curve'][t]['balanced_accuracy'] for s,t in
                  [('32','64'),('32','256'),('64','256'),('128','256')]]
        lines.append('| '+r['arm']+' | '+' | '.join(f'{100*v:.2f}' for v in scores)+' |')
    lines += ['', '## Paired-source correctness', '',
              'Both original and flipped-source answers must be correct at each pixel of the largest component.',
              'These percentages are pooled over changed-component pixels; they are not independent replicates.', '',
              '| Arm | 32x32, T64 | 32x32, T256 | 64x64, T256 | 128x128, T256 |',
              '|---|---:|---:|---:|---:|']
    for r in summary['arms']:
        e = r['evaluation']
        scores = [e[s]['curve'][t]['paired_correct'] for s,t in
                  [('32','64'),('32','256'),('64','256'),('128','256')]]
        lines.append('| '+r['arm']+' | '+' | '.join(f'{100*v:.2f}' for v in scores)+' |')
    lines += ['', '## Decision and limits', '',
              '- Inertial RD underperforms generic momentum already at the training size and T64. Its T256 degradation is not the only source of the negative result.',
              '- Inertial RD reaches only 72.89% BA at T32 on 32x32, then drops to 69.94% at T64 and 53.50% at T256. At 64x64 and 128x128, T256 BA is 50% and paired-source correctness is zero.',
              '- All sustained aggregate 95% thresholds are null. No matched-quality time-to-solution claim is available.',
              '- Inertial RD has zero pre-damage eligible maps at every size. Its main conditional-repair outcome is unevaluable, not a measured zero repair success rate.',
              '- Other models learn partial structure but do not solve the frozen 95% criterion. This task/recipe has not established a robust high-quality reference solution.',
              '- Single seed, small held-out banks, fixed evaluation seeds and limited rollout checkpoints prevent general superiority or impossibility conclusions. No follow-up training was launched.', '',
              '## Evidence route', '',
              '1. [summary.json](summary.json) includes every accuracy horizon and paired score, coefficients, gradients, timings and repair eligibility.',
              '2. [config.json](config.json) records the actual configuration and task/metric conventions.',
              '3. [validation.json](validation.json) and [linear_checks.json](linear_checks.json) concern implementation validity only.',
              '4. Read an individual arm only for full repair/revision controls or distance bins:', '']
    lines += [f'- [{arm}](arms/{arm}_seed0.json)' for arm in ARMS]
    lines += ['', 'The per-arm JSON files are exact copies of frozen output, verified by SHA256. Machine receipts, PIDs, checkpoints and transient logs remain local. Source and evidence hashes are in [the publication manifest](../../INERTIAL_PUBLICATION_MANIFEST.json).', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checks', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    publication = ROOT / 'INERTIAL_PUBLICATION_MANIFEST.json'
    if out.exists() or publication.exists():
        raise SystemExit('Use a new export directory; existing publication evidence is immutable.')
    out.relative_to(ROOT)
    manifest, status = read(run/'manifest.json'), read(run/'status.json')
    if status['status'] != 'FINISHED' or status['pid'] != manifest['pid']:
        raise SystemExit('Run is incomplete or its process identity does not match.')
    source_hashes = {}
    for name, expected in manifest['code_sha256'].items():
        current, executed = SOURCE/name, run/'source'/name
        if sha(current.read_bytes()) != expected or sha(executed.read_bytes()) != expected:
            raise SystemExit(f'Executed/current source mismatch: {name}')
        source_hashes[current.relative_to(ROOT).as_posix()] = expected
    protocol_hashes = {}
    for name in ('THEORY.md', 'WIND_TUNNEL.md'):
        if (SOURCE/name).read_bytes() != (run/'source'/name).read_bytes():
            raise SystemExit(f'Executed/current protocol mismatch: {name}')
        protocol_hashes[(SOURCE/name).relative_to(ROOT).as_posix()] = sha((SOURCE/name).read_bytes())
    raw = {arm: (run/f'{arm}_seed0.json').read_bytes() for arm in ARMS}
    reports = {arm: json.loads(blob) for arm, blob in raw.items()}
    for arm, r in reports.items():
        if r['status'] != 'TRAINED' or r['completed_updates'] != 800 or r['seed'] != 0:
            raise SystemExit(f'Incomplete or unexpected arm: {arm}')
        if r['config'] != manifest['config'] or set(r['evaluation']) != {'32','64','128'}:
            raise SystemExit(f'Config/evaluation mismatch: {arm}')
        for e in r['evaluation'].values():
            if e['status'] != 'EVALUATED' or set(e['curve']) != {'16','32','64','128','256'}:
                raise SystemExit(f'Incomplete evaluation: {arm}')
    linear_blob = (args.checks/'linear_checks.json').read_bytes()
    linear = json.loads(linear_blob)
    if linear['real_stability_tests']['mismatches'] or linear['complex_stability_tests']['mismatches']:
        raise SystemExit('Linear validation did not pass.')
    # CPU correctness only; this never launches optimization or reruns the screen.
    test_code = """import io,json,torch,unittest
torch.set_num_threads(2)
suite=unittest.defaultTestLoader.loadTestsFromName('test_core')
r=unittest.TextTestRunner(stream=io.StringIO()).run(suite)
print(json.dumps(dict(tests_run=r.testsRun,failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful())))
raise SystemExit(not r.wasSuccessful())
"""
    checked = subprocess.run([sys.executable, '-c', test_code], cwd=SOURCE,
                             capture_output=True, text=True, check=True)
    unit_tests = json.loads(checked.stdout)
    preflight = {arm: read(args.preflight/f'{arm}_seed0.json') for arm in ARMS}
    if any(r['status'] != 'TRAINED' or r['completed_updates'] != 3 for r in preflight.values()):
        raise SystemExit('Expected preflight evidence is missing.')
    validation = {
        'unit_tests': unit_tests, 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'executed_source_sha256_matches_published': True,
        'linear_checks_scope': 'Pure-transport and frozen linear systems only; not full learned stability or task efficacy.',
        'preflight_scope': 'Three full-size CUDA optimizer updates per arm plus abbreviated evaluation; no efficacy claim.',
        'preflight': [{ 'arm': arm, 'updates': r['completed_updates'],
                       'parameters': r['initial_model']['parameters'],
                       'train_peak_allocated_bytes': r['train_peak_allocated_bytes']}
                      for arm,r in preflight.items()],
    }
    config = dict(manifest['config'])
    config.pop('out')
    config['conventions'] = {
        'train_data_seed': 10000, 'training_schedule_seed': 20000,
        'evaluation_data_seeds': {'32':30032,'64':30064,'128':30128},
        'balanced_accuracy': 'Mean across maps of mean recall over classes present in each map.',
        'paired_correct': 'Pooled changed-component pixels with both original and flipped answers correct.',
        'sustained_95': 'First tested horizon with aggregate BA>=.95 at it and all later tested horizons; missing=null.',
        'damage_eligibility': 'Per-map pre-damage BA>=.95 at T64; zero eligible gives null conditional outcome.',
        'labels': 'Random position-independent component identities, conditioned on both labels appearing per original map.',
        'source_flip': 'Largest connected component; geometry unchanged.',
        'geometry': 'Rooms grow with canvas size; walls and doors stay one pixel wide.',
        'spatial_operator': 'Whole-grid replicated-boundary five-point positive Laplacian, not component-masked.',
        'timing': 'Three warmups then median of three synchronized batch-eight full inference times divided by eight; throughput-normalized per-query timing, not isolated batch-one latency.',
    }
    summary = summarize(reports, manifest, status)
    blobs = {'summary.json': encode(summary), 'config.json': encode(config),
             'RESULTS.md': report(summary).encode('utf-8'), 'validation.json': encode(validation),
             'linear_checks.json': linear_blob}
    blobs.update({f'arms/{arm}_seed0.json': blob for arm,blob in raw.items()})
    out.mkdir(parents=True)
    for name, blob in blobs.items():
        dest=out/name; dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(blob)
    public = {
        'canonical_run': run.name, 'protocol': summary['protocol'],
        'publication_kind': 'completed_exploratory_screen',
        'execution_status': summary['execution_status'], 'scientific_status': summary['scientific_status'],
        'source_sha256': source_hashes, 'protocol_files_sha256': protocol_hashes,
        'original_archive_sha256': manifest['original_zip_sha256'],
        'started_utc': manifest['started_utc'], 'finished_utc': status['finished_utc'],
        'runtime': {key: manifest[key] for key in ('torch','numpy','device','gpu','backend')},
        'published_evidence_sha256': {(out/name).relative_to(ROOT).as_posix():sha(blob) for name,blob in blobs.items()},
        'per_arm_outputs_are_byte_identical_to_local_run': True,
        'excluded': ['checkpoints','launch receipts','machine identifiers','PIDs','transient logs',
                     'local output directories','duplicate source snapshots','original ZIP','external CPU smoke outputs'],
    }
    publication.write_bytes(encode(public))
    print(json.dumps({'status': summary['execution_status'], 'scientific_status':summary['scientific_status'],
                      'exported_files':len(blobs), 'unit_tests':unit_tests}, indent=2))


if __name__ == '__main__':
    main()
