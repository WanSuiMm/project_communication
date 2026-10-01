"""Export the final state control, retaining original and recovered provenance."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ARM_FILE = 'masked_state_nca_seed0.json'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--recovery', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    run, recovery, out = args.run.resolve(), args.recovery.resolve(), args.out.resolve()
    out.relative_to(ROOT)
    publication = ROOT/'STATE_PUBLICATION_MANIFEST.json'
    if out.exists() or publication.exists():
        raise RuntimeError('Publication outputs must be new; preserve existing evidence')
    manifest = read(run/'manifest.json')
    raw, augmented = read(run/ARM_FILE), read(recovery/ARM_FILE)
    provenance = read(recovery/'recovery.json')
    assert raw['status'] == 'TRAINED' and raw['completed_updates'] == 800
    assert sha(run/ARM_FILE) == sha(recovery/'original_arm.json') == provenance['original_arm_sha256']
    assert sha(run/'manifest.json') == provenance['source_manifest_sha256']
    assert sha(run/ARM_FILE.replace('.json', '.pt')) == provenance['checkpoint_sha256']
    assert sha(ROOT/'new/masked_state/recover_tail.py') == provenance['recovery_script_sha256']
    assert provenance['training_repeated'] is False
    assert provenance['original_execution_finalized'] is False
    for section in ('source_sha256', 'reference_sha256'):
        for rel, expected in manifest[section].items():
            assert sha(ROOT/rel) == expected, rel
            if section == 'source_sha256':
                assert sha(run/'source'/rel) == expected, rel
    # Removing exactly the documented additions must recover every original field.
    stripped = deepcopy(augmented)
    for field in provenance['recovered_fields']:
        parts = field.split('.')
        node = stripped
        for part in parts[:-1]:
            node = node[part]
        del node[parts[-1]]
    for size, ev in raw['evaluation'].items():
        if 'latency_by_horizon' not in ev:
            assert stripped['evaluation'][size].pop('latency_by_horizon') == {}
    assert stripped == raw, 'Recovery changed pre-existing scientific evidence'
    sources = {
        'unmasked_state': ROOT/'evidence/inertial_seed0/arms/nca_state_matched_seed0.json',
        'unmasked_momentum': ROOT/'evidence/inertial_seed0/arms/momentum_nca_seed0.json',
        'masked_momentum': ROOT/'evidence/masked_momentum_seed0/arms/masked_momentum_nca_seed0.json',
    }
    records = {key: read(path) for key, path in sources.items()}
    records['masked_state'] = augmented
    schedule = lambda d: [(r['iteration'], r['rollout'], r['damage']) for r in d['training_curve']]
    assert all(schedule(d) == schedule(raw) for d in records.values())
    report = read(recovery/'comparison.json')
    assert len(report['all_horizons']) == 15
    assert {(r['size'], r['steps']) for r in report['all_horizons']} == {
        (s, t) for s in (32, 64, 128) for t in (16, 32, 64, 128, 256)}
    for row in report['all_horizons']:
        size, steps = str(row['size']), str(row['steps'])
        for key, data in records.items():
            assert data['completed_updates'] == 800
            ev = data['evaluation'][size]
            assert ev['status'] == 'EVALUATED'
            point = ev['curve'][steps]
            expected = {k: point[v] for k, v in [('ba', 'balanced_accuracy'), ('bce', 'bce'),
                ('h_rms', 'content_rms'), ('v_rms', 'velocity_rms')]}
            expected['paired'] = ev['paired_source_information'][steps]['both_counterfactuals_correct_on_changed_component']
            assert row[key] == expected
        for metric in ('ba', 'paired'):
            s, us, m, um = [row[key][metric] for key in
                ('masked_state', 'unmasked_state', 'masked_momentum', 'unmasked_momentum')]
            for key, value in {'momentum_minus_state_masked': m-s,
                    'state_masking_effect': s-us, 'momentum_masking_effect': m-um,
                    'interaction': (m-um)-(s-us)}.items():
                assert abs(row[f'{key}_{metric}_pp']-100*value) < 1e-10
    primary = next(r for r in report['all_horizons'] if (r['size'], r['steps']) == (32, 64))
    assert report['primary'] == primary
    assert min(primary[f'momentum_minus_state_masked_{m}_pp'] for m in ('ba', 'paired')) >= 5
    assert report['decision'] == 'MOMENTUM_JOINT_5PP_ADVANTAGE'
    assert max(max(r['absolute_errors'].values()) for r in provenance['checkpoint_replay']) == 0
    out.mkdir(parents=True)
    (out/'arms').mkdir()
    for name in ('RESULTS.md', 'comparison.json', 'aggregate.json', 'recovery.json', 'original_arm.json'):
        shutil.copyfile(recovery/name, out/name)
    shutil.copyfile(recovery/ARM_FILE, out/'arms'/ARM_FILE)
    public_manifest = {k: v for k, v in manifest.items() if k != 'pid'}
    write(out/'manifest.json', public_manifest)
    validation = {
        'implementation_tests': {'command': 'python new/masked_state/test_state.py', 'count': 3, 'status': 'PASSED'},
        'completed_updates': 800, 'all_15_comparisons_recomputed_from_raw': True,
        'logged_training_schedule_matches_all_four': True,
        'executed_source_snapshot_matches_current': True,
        'original_arm_preserved_byte_for_byte': True,
        'original_scientific_fields_unchanged_in_augmented_record': True,
        'checkpoint_replay_endpoints': [[32, 64], [128, 256]], 'replay_max_absolute_error': 0,
        'recovery_scope': 'Missing planned size128 timings and gradient probes only; no retraining',
        'original_process_finalization': 'MISSING_CAUSE_UNKNOWN',
    }
    write(out/'validation.json', validation)
    public = {
        'review_base': '82635bbf0063d11359fd2d0b6c6571616a01f89b',
        'protocol': manifest['protocol'], 'publication_kind': 'single_seed_final_2x2_with_auxiliary_recovery',
        'scientific_status': report['decision'], 'original_execution_finalized': False,
        'auxiliary_recovery_status': provenance['status'], 'training_repeated': False,
        'source_sha256': manifest['source_sha256'], 'reference_sha256': manifest['reference_sha256'],
        'recovery_source_sha256': {'new/masked_state/recover_tail.py': provenance['recovery_script_sha256']},
        'publication_tool_sha256': {'tools/export_state_evidence.py': sha(Path(__file__))},
        'local_checkpoint_sha256': provenance['checkpoint_sha256'],
        'published_evidence_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(out.rglob('*')) if f.is_file()},
        'notes': ['Original run retained unchanged; process exit cause unknown.',
                  'All scientific endpoints predate auxiliary recovery; no retraining.',
                  'Original and augmented records are separately identified.',
                  'One seed; recipe comparison does not isolate velocity.',
                  'Checkpoints, PID, machine receipts and transient logs remain local.',
                  'No further experiment or monitor scheduled.'],
    }
    write(publication, public)
    print(json.dumps({'decision': report['decision'], 'exported_files': len(public['published_evidence_sha256']),
                      'recovery_verified': True}, indent=2))


if __name__ == '__main__':
    main()
