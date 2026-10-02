"""Publish the audited persistent-role results without private run metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
REVIEW_BASE = '891f968d85e84e1bc47e0fd3bcbe83210c55f2f6'
PRIVATE_KEYS = {
    'pid', 'gpu', 'gpu_uuid', 'gpu_id', 'device_uuid', 'device_id',
    'hostname', 'host_name', 'host', 'username', 'user', 'cwd',
    'working_directory', 'command', 'launch_command',
}
PRIVATE_IPV4 = (
    r'(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|'
    r'172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})'
)
PRIVATE_TEXT = re.compile(
    r'(?i)(\bpid\b|gpu[_ -]?uuid|\bhostname\b|\busername\b|'
    r'[A-Z]:\\|' + PRIVATE_IPV4 + r'\b|[A-Za-z0-9._-]+@' + PRIVATE_IPV4 + r'\b)')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def repo_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def public(value):
    if isinstance(value, dict):
        return {key: public(item) for key, item in value.items()
                if str(key).lower() not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [public(item) for item in value]
    return value


def private_fields(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in PRIVATE_KEYS:
                found.add(str(key))
            found.update(private_fields(item))
    elif isinstance(value, list):
        for item in value:
            found.update(private_fields(item))
    return found


def assert_safe_json(path):
    value = read(path)
    assert not private_fields(value), (path, sorted(private_fields(value)))


def assert_safe_text(path):
    text = path.read_text(encoding='utf-8')
    assert not PRIVATE_TEXT.search(text), path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'analysis', 'preflight', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run, analysis, preflight, out = map(
        repo_path, (args.run, args.analysis, args.preflight, args.out))
    publication = ROOT / 'PERSISTENT_ROLES_PUBLICATION_MANIFEST.json'
    assert not out.exists(), f'Public evidence already exists: {out}'
    assert not publication.exists(), f'Publication manifest already exists: {publication}'

    provenance = read(analysis / 'provenance.json')
    summary = read(analysis / 'analysis.json')
    validation = read(analysis / 'validation.json')
    manifest, status, aggregate = (read(run / name) for name in
                                   ('manifest.json', 'status.json', 'aggregate.json'))
    pf_manifest, pf_status, pf_aggregate = (read(preflight / name) for name in
                                            ('manifest.json', 'status.json', 'aggregate.json'))
    assert validation['status'] == 'PASS'
    assert summary['decision'] == aggregate['decision'] == 'DEVELOPMENT_NO_GO'
    assert summary['counts'] == aggregate['counts']
    assert summary['counts']['roles'] == {'reach': 0, 'reach_and_hold': 0}
    assert status['status'] == 'COMPLETE' and status['completed_arms'] == 12
    assert pf_status['status'] == 'PREFLIGHT_PASSED'
    assert len(manifest['source_sha256']) == 51
    assert manifest['source_sha256'] == pf_manifest['source_sha256']
    assert manifest['preflight_sha256'] == {
        name: sha(preflight / name) for name in ('status.json', 'manifest.json', 'aggregate.json')}

    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT / name) == sha(run / 'source' / name) == sha(preflight / 'source' / name) == expected, name
    for name, expected in provenance['input_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in provenance['checkpoint_file_sha256'].items():
        assert sha(run / name) == expected, name
    for name, expected in provenance['analysis_source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in provenance['output_sha256'].items():
        assert sha(analysis / name) == expected, name
    assert provenance['original_private_manifest_sha256'] == sha(run / 'manifest.json')
    assert provenance['original_private_completion_sha256'] == sha(run / 'status.json')
    assert provenance['original_preflight_manifest_sha256'] == sha(preflight / 'manifest.json')
    assert provenance['original_preflight_completion_sha256'] == sha(preflight / 'status.json')

    raw = sorted(run.glob('*_K8_seed*.json'))
    checkpoints = sorted(run.glob('*.pt'))
    assert len(raw) == 12 and len(checkpoints) == 12
    assert len(provenance['parameter_sha256']) == 12
    assert len(summary['primary_rows']) == 12
    assert validation['paired_aggregates_and_independent_denominators'] == 1008
    assert validation['curve_csv_rows'] == 936 and validation['paired_effect_csv_rows'] == 624

    out.mkdir(parents=True)
    raw_dir = out / 'raw'
    raw_dir.mkdir()
    for source in raw:
        assert_safe_json(source)
        destination = raw_dir / source.name
        shutil.copyfile(source, destination)
        assert sha(source) == sha(destination)

    for name in ('schedule.json', 'aggregate.json', 'curves.csv', 'paired_effects.csv'):
        source, destination = run / name, out / name
        if source.suffix == '.json':
            assert_safe_json(source)
        else:
            assert_safe_text(source)
        shutil.copyfile(source, destination)
        assert sha(source) == sha(destination), name
    assert_safe_text(run / 'RESULTS.md')
    shutil.copyfile(run / 'RESULTS.md', out / 'runner_RESULTS.md')

    removed = set()
    for value in (manifest, status, pf_manifest, pf_status, pf_aggregate):
        removed.update(private_fields(value))
    write(out / 'manifest.json', public(manifest))
    write(out / 'completion.json', public(status))
    pf_dir = out / 'preflight'
    pf_dir.mkdir()
    write(pf_dir / 'manifest.json', public(pf_manifest))
    write(pf_dir / 'status.json', public(pf_status))
    write(pf_dir / 'aggregate.json', public(pf_aggregate))
    write(out / 'analysis.json', public(summary))
    shutil.copyfile(analysis / 'RESULTS.md', out / 'RESULTS.md')
    public_validation = {
        **validation,
        'raw_arm_schedule_aggregate_and_CSVs_byte_identical': True,
        'formal_run_bound_to_passed_preflight': True,
        'private_fields_removed_from_manifest_and_completion_copies': sorted(removed),
        'checkpoint_files_published': False,
        'new_training_or_model_inference': False,
    }
    write(out / 'validation.json', public_validation)
    write(out / 'provenance.json', {
        'source_run': run.name,
        'source_preflight': preflight.name,
        'original_analysis_provenance_sha256': sha(analysis / 'provenance.json'),
        'original_private_manifest_sha256': provenance['original_private_manifest_sha256'],
        'original_private_completion_sha256': provenance['original_private_completion_sha256'],
        'original_preflight_manifest_sha256': provenance['original_preflight_manifest_sha256'],
        'original_preflight_completion_sha256': provenance['original_preflight_completion_sha256'],
        'original_preflight_aggregate_sha256': sha(preflight / 'aggregate.json'),
        'source_sha256': manifest['source_sha256'],
        'checkpoint_file_sha256': provenance['checkpoint_file_sha256'],
        'parameter_sha256': provenance['parameter_sha256'],
        'analysis_source_sha256': provenance['analysis_source_sha256'],
        'sanitized_fields_removed': sorted(removed),
        'notes': [
            'The twelve raw arm records, schedule, aggregate, curves, and paired effects are byte-identical copies.',
            'PID and machine-identifying fields were removed only from manifest/completion/preflight copies.',
            'Original private-file SHA-256 values are retained for provenance; checkpoint files remain local.',
            'Checkpoint state hashes were verified on CPU; no model inference or new training was performed.',
        ],
    })

    for path in out.rglob('*.json'):
        assert_safe_json(path)
    for path in out.rglob('*.md'):
        assert_safe_text(path)
    published_hashes = {
        path.relative_to(ROOT).as_posix(): sha(path)
        for path in sorted(out.rglob('*')) if path.is_file()
    }
    publication_doc = {
        'review_base': REVIEW_BASE,
        'execution_status': 'COMPLETE',
        'decision': summary['decision'],
        'counts': summary['counts'],
        'source_sha256': manifest['source_sha256'],
        'analysis_source_sha256': provenance['analysis_source_sha256'],
        'publication_tool_sha256': {
            Path(__file__).resolve().relative_to(ROOT).as_posix(): sha(Path(__file__).resolve())
        },
        'checkpoint_file_sha256': provenance['checkpoint_file_sha256'],
        'checkpoint_parameter_sha256': provenance['parameter_sha256'],
        'historical_control_reproductions': summary['historical_control_reproduction'],
        'published_evidence_sha256': published_hashes,
        'notes': [
            'Complete matched result is DEVELOPMENT_NO_GO under the frozen endpoint and hold gates.',
            'The roles parameterization changes H/C allocation, carrier width, perception, shared local rule, and readout clock together; it does not isolate H causality.',
            'The K8 carrier and task-readout radius bounds are 16 and 15; the frozen strict d>16 primary lies beyond that readout radius.',
            'The result is conditional on previously inspected seeds/maps and one shared training bank/schedule.',
            'Checkpoints and machine launch receipts are excluded; raw saved results are retained byte-identically.',
        ],
        'excluded': ['checkpoint files', 'machine launch receipts', 'PID fields', 'GPU fields', 'transient logs'],
    }
    write(publication, publication_doc)
    print(json.dumps({
        'status': 'EXPORTED',
        'public_files': len(published_hashes),
        'public_bytes': sum(path.stat().st_size for path in out.rglob('*') if path.is_file()),
        'removed_fields': sorted(removed),
        'publication_manifest': publication.relative_to(ROOT).as_posix(),
    }, indent=2))


if __name__ == '__main__':
    main()
