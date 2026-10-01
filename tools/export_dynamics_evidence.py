"""Publish the completed zero-training audit and its bounded interpretation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIRS = {'nca_state_matched': 'inertial_seed0', 'momentum_nca': 'inertial_seed0',
                  'masked_state_nca': 'masked_state_seed0', 'masked_momentum_nca': 'masked_momentum_seed0'}


def read(path): return json.loads(path.read_bytes())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    run, analysis, out = args.run.resolve(), args.analysis.resolve(), args.out.resolve()
    out.relative_to(ROOT)
    publication = ROOT/'DYNAMICS_PUBLICATION_MANIFEST.json'
    if out.exists() or publication.exists(): raise RuntimeError('Publication outputs must be new')
    manifest, status, summary = read(run/'manifest.json'), read(run/'status.json'), read(run/'summary.json')
    derived, provenance = read(analysis/'analysis.json'), read(analysis/'provenance.json')
    assert status['status'] == summary['status'] == 'COMPLETE'
    assert not manifest['training'] and not manifest['preflight']
    assert sha(run/'summary.json') == provenance['run_summary_sha256']
    assert sha(run/'manifest.json') == provenance['manifest_sha256']
    assert sha(ROOT/'new/dynamics_audit/analyze.py') == provenance['analysis_source_sha256']
    for rel, expected in manifest['source_sha256'].items():
        assert sha(ROOT/rel) == sha(run/'source'/rel) == expected, rel
    reference_hashes = {}
    errors, products, open16 = [], [], []
    raw_files = []
    assert len(summary['records']) == 8
    for record in summary['records'].values():
        arm, size = record['arm'], record['size']
        ref = ROOT/'evidence'/REFERENCE_DIRS[arm]/'arms'/(arm+'_seed0.json')
        reference_hashes[ref.relative_to(ROOT).as_posix()] = sha(ref)
        old = read(ref)['evaluation'][str(size)]
        curve_path = run/f'{arm}_size{size}_curves.json'
        dynamics_path = run/f'{arm}_size{size}_dynamics.json'
        assert read(curve_path)['curves'] == record['curves']
        assert read(dynamics_path) == record['dynamics']
        assert len(record['curves']) == len(record['dynamics']) == 8
        raw_files.extend((curve_path, dynamics_path))
        for t, point in record['curves'].items():
            if t in old['curve']:
                err = {k: abs(point[k]-old['curve'][t][k]) for k in ('balanced_accuracy', 'bce', 'content_rms')}
                if point['velocity_rms'] is not None: err['velocity_rms'] = abs(point['velocity_rms']-old['curve'][t]['velocity_rms'])
                err['paired'] = abs(point['paired']-old['paired_source_information'][t]['both_counterfactuals_correct_on_changed_component'])
                errors.append(err)
            diag = record['dynamics'][t]
            for values in diag['products'].values():
                for value in values:
                    assert value['converged'] == (value['relative_singular_residual'] <= .01 and value['start_relative_spread'] <= .02)
                products.extend(values)
            open16.extend(diag['products']['open_input_output_K16'])
    maximum = max(v for e in errors for v in e.values())
    assert len(errors) == 40 and maximum == 0
    assert (len(products), sum(p['converged'] for p in products)) == (1536, 562)
    assert (len(open16), sum(p['converged'] for p in open16)) == (256, 148)
    assert all(p['finite_time_log_gain'] > 0 for p in open16)
    assert derived['maximum_replay_error'] == maximum
    assert derived['all_products'] == {'count': 1536, 'converged': 562}
    # Only the tiny CPU derivative gate is rerun for publication; no GPU audit.
    check = subprocess.run([sys.executable, str(ROOT/'new/dynamics_audit/check.py')],
                           capture_output=True, check=True)
    numerical = json.loads(check.stdout.decode('utf-8'))
    assert numerical['status'] == 'PASS'
    preflight = read(ROOT/'runs/dynamics_audit_20261001_preflight/status.json')
    assert preflight['status'] == 'PREFLIGHT_PASSED'
    out.mkdir(parents=True); (out/'raw').mkdir()
    for path in raw_files: shutil.copyfile(path, out/'raw'/path.name)
    for name in ('analysis.json', 'provenance.json', 'dynamics_overview.png'):
        shutil.copyfile(analysis/name, out/name)
    shutil.copyfile(run/'RESULTS.md', out/'RESULTS.md')
    text = (analysis/'INTERPRETATION.md').read_text(encoding='utf-8')
    text = text.replace('The original completed run remains\nunchanged under `runs/dynamics_audit_20261001_seed0/` from the repository root.',
                        'All16 detailed [curve and dynamics files](raw/) are byte-identical to the original run.\n'
                        'The redundant combined local summary is excluded; its hash is retained in provenance.json.\n'
                        'Read this interpretation and analysis.json before opening the detailed files.')
    (out/'INTERPRETATION.md').write_text(text, encoding='utf-8')
    write(out/'manifest.json', {k: v for k, v in manifest.items() if k != 'pid'})
    write(out/'completion.json', {k: v for k, v in status.items() if k != 'pid'})
    validation = {'cpu_numerical_gate': numerical, 'historical_replay_count': len(errors),
                  'historical_replay_max_error': maximum,
                  'published_raw_files_byte_identical': True, 'executed_source_snapshot_matches': True,
                  'preflight': {k: v for k, v in preflight.items() if k != 'pid'},
                  'new_training_or_gpu_audit_for_publication': False,
                  'product_estimates_converged': 562, 'product_estimates_total': 1536,
                  'open_K16_converged': 148, 'open_K16_total': 256}
    write(out/'validation.json', validation)
    public = {
        'review_base': 'aa2d50a91e4ff02f74144dc699124bd886b5aa85',
        'protocol': manifest['protocol'], 'execution_status': 'COMPLETE', 'training': False,
        'scientific_status': 'NO_OBSERVED_MAXIMUM_GAIN_ZERO_CROSSING_WITH_DIAGNOSTIC_LIMITS',
        'source_sha256': manifest['source_sha256'], 'reference_sha256': reference_hashes,
        'analysis_source_sha256': {'new/dynamics_audit/analyze.py': provenance['analysis_source_sha256']},
        'publication_tool_sha256': {'tools/export_dynamics_evidence.py': sha(Path(__file__))},
        'checkpoint_sha256': manifest['checkpoint_sha256'],
        'published_evidence_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(out.rglob('*')) if f.is_file()},
        'excluded': ['checkpoints', 'machine receipts', 'PID', 'transient logs', 'redundant combined raw summary', 'local PDF copy'],
        'notes': ['All prior frozen sources and evidence unchanged.',
                  'All16 raw curve/dynamics files and compact derived JSON copied byte-for-byte.',
                  'Interpretation differs from local copy only in public evidence routing.',
                  'Manifest and completion records omit PID; original records retained locally.',
                  'Power convergence and finite-perturbation limitations are material.',
                  'No asymptotic Lyapunov, causal mechanism or new architecture claim.'],
    }
    write(publication, public)
    print(json.dumps({'status': 'EXPORTED', 'files': len(public['published_evidence_sha256']),
                      'historical_replay_max_error': maximum, 'cpu_gate': numerical['status']}, indent=2))


if __name__ == '__main__': main()
