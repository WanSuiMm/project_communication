"""Publish saved seed4 audit evidence; do not perform training or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_KEYS = {'pid', 'host', 'hostname', 'username', 'gpu', 'gpu_uuid', 'device',
                'device_uuid', 'command', 'launch_command', 'cwd', 'working_directory'}


def read(path):
    return Path(path).read_text(encoding='utf-8')


def doc(path):
    return json.loads(read(path))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public(value):
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k.lower() not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [public(v) for v in value]
    return value


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'analysis', 'out'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    run, analysis, out = [Path(v).resolve() for v in (args.run, args.analysis, args.out)]
    assert out.is_relative_to(ROOT/'evidence')
    publication_path = ROOT/'STREAM_PATH_AUDIT_PUBLICATION_MANIFEST.json'
    assert not out.exists() and not publication_path.exists()
    manifest, status, summary, validation = [doc(p) for p in
        (run/'manifest.json', run/'status.json', run/'summary.json', analysis/'validation.json')]
    assert status['status'] == summary['status'] == 'COMPLETE'
    assert validation['status'] == 'PASS' and summary['replay']['maximum_absolute_error'] == 0
    assert status['completed_conditions'] == 4 and summary['training'] is False
    assert manifest['model_seed'] == 4 and summary['model_replications'] == 1
    assert manifest['git_review_base'] == '87763540c7e342a8fe278b8c198930f4decc2c1a'
    for name, expected in validation['input_sha256'].items():
        assert sha(run/name) == expected, name
    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT/name) == sha(run/'source'/name) == expected, name
    analysis_source = 'new/stream_path_audit/analyze.py'
    assert sha(ROOT/analysis_source) == validation['analysis_source_sha256']
    checkpoint = ROOT/'runs/streaming_carry_20261002_init2345/stream_K8_seed4.pt'
    assert sha(checkpoint) == manifest['checkpoint_file_sha256']
    out.mkdir(parents=True, exist_ok=False)
    copied = ('summary.json', 'raw_conditions.json', 'curves.csv', 'contrasts.json',
              'replay_validation.json', 'RESULTS.md')
    for name in copied:
        shutil.copyfile(run/name, out/name)
        assert sha(run/name) == sha(out/name)
    shutil.copyfile(analysis/'validation.json', out/'validation.json')
    interpretation = read(run/'INTERPRETATION.md').replace(
        'The validation record is under analyses/stream_path_audit_20261002_seed4_01/.',
        'See [validation](validation.json). Its input_sha256 binds original run files;\n'
        'private manifest/completion bytes are excluded, while the public copies\n'
        'are bound separately in STREAM_PATH_AUDIT_PUBLICATION_MANIFEST.json.')
    (out/'INTERPRETATION.md').write_text(interpretation, encoding='utf-8')
    write(out/'manifest.json', public(manifest))
    write(out/'completion.json', public(status))
    provenance = {
        'review_base': manifest['git_review_base'], 'training': False,
        'model_replications': 1,
        'original_private_manifest_sha256': sha(run/'manifest.json'),
        'original_private_completion_sha256': sha(run/'status.json'),
        'original_interpretation_sha256': sha(run/'INTERPRETATION.md'),
        'byte_identical_copies': list(copied)+['validation.json'],
        'sanitized_copies': ['manifest.json', 'completion.json'],
        'interpretation_edit': 'Replace local analysis-directory reference with public validation link and hash-scope explanation only.',
        'validation_input_scope': 'input_sha256 names refer to original run inputs, not sanitized public manifest/completion.',
        'excluded': ['checkpoints', 'source snapshots', 'machine launch metadata', 'private manifest/completion originals', 'logs'],
    }
    write(out/'provenance.json', provenance)
    for path in out.iterdir():
        content = read(path)
        assert not re.search(r'[A-Za-z]:[\\/]|(?<![\d.])(?:192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|10\.)\d+\.\d+', content), path
        if path.suffix == '.json':
            value = doc(path)
            assert value == public(value), ('Private keys', path)
    publication = {
        'review_base': manifest['git_review_base'], 'execution_status': 'COMPLETE',
        'scientific_status': 'SELECTED_CHECKPOINT_OPERATOR_SENSITIVITY', 'training': False,
        'model_seed': 4, 'model_replications': 1,
        'source_sha256': manifest['source_sha256'],
        'analysis_source_sha256': {analysis_source: sha(ROOT/analysis_source)},
        'publication_tool_sha256': {'tools/export_stream_path_audit.py': sha(Path(__file__))},
        'checkpoint_file_sha256': manifest['checkpoint_file_sha256'],
        'checkpoint_parameter_sha256': manifest['checkpoint_parameter_sha256'],
        'published_evidence_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(out.iterdir())},
        'notes': [
            'Cold-rollout knockouts alter learned input/state distributions and hop depth; this is not training-cause identification.',
            'Both F and Q Laplacian input slots are disabled together; transport removal changes the carry and incoming F feature.',
            'Original six historical evaluations replay exactly; prior multi-seed no-go and seed4 positive result are unchanged.',
            'Validation input_sha256 refers to original run bytes; published_evidence_sha256 binds these public files.',
            'Metrics, curves, contrasts, replay validation and arithmetic validation are byte-identical copies; private metadata are excluded.',
        ],
    }
    write(publication_path, publication)
    print(json.dumps({'status': 'EXPORTED', 'files': len(publication['published_evidence_sha256']),
                      'total_bytes': sum(p.stat().st_size for p in out.iterdir()),
                      'source_bindings': len(manifest['source_sha256'])}))


if __name__ == '__main__':
    main()
