"""Publish completed carry evidence, retaining raw bytes and excluding PID metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REVIEW_BASE = '0decc2926b52967496d57a90f8f803383e383178'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def public(value):
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k != 'pid'}
    if isinstance(value, list):
        return [public(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'analysis', 'preflight', 'out'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    run, analysis, pf, out = (Path(getattr(args,n)).resolve() for n in ('run','analysis','preflight','out'))
    publication = ROOT / 'DIRECT_CARRY_PUBLICATION_MANIFEST.json'
    assert not out.exists() and not publication.exists()
    provenance = read(analysis / 'provenance.json')
    manifest, status, aggregate = (read(run / n) for n in ('manifest.json','status.json','aggregate.json'))
    summary = read(analysis / 'analysis.json')
    assert status['status'] == 'COMPLETE' and status['completed_arms'] == 8
    assert summary['validation']['status'] == 'PASS' and summary['decision'] == aggregate['decision']
    for name, expected in provenance['input_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in provenance['checkpoint_sha256'].items():
        assert sha(run / name) == expected, name
    for name, expected in provenance['analysis_source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in provenance['output_sha256'].items():
        assert sha(analysis / name) == expected, name
    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT / name) == sha(run / 'source' / name) == expected, name
    for name, expected in manifest['preflight_sha256'].items():
        assert sha(pf / name) == expected, name
    assert read(pf / 'status.json')['status'] == 'PREFLIGHT_PASSED'
    assert read(pf / 'manifest.json')['source_sha256'] == manifest['source_sha256']
    assert read(pf / 'aggregate.json')['updates_per_arm'] == manifest['updates_per_arm'] == 300
    raw = sorted(run.glob('*_K8_seed*.json'))
    assert len(raw) == 8
    (out / 'raw').mkdir(parents=True)
    (out / 'preflight').mkdir()
    for file in raw:
        shutil.copyfile(file, out / 'raw' / file.name)
        assert sha(file) == sha(out / 'raw' / file.name)
    for name in ('schedule.json','aggregate.json','curves.csv','paired_effects.csv'):
        shutil.copyfile(run / name, out / name)
    shutil.copyfile(run / 'RESULTS.md', out / 'runner_RESULTS.md')
    shutil.copyfile(analysis / 'RESULTS.md', out / 'RESULTS.md')
    write(out / 'analysis.json', public(summary))
    write(out / 'manifest.json', public(manifest))
    write(out / 'completion.json', public(status))
    for name in ('manifest.json','status.json','aggregate.json'):
        write(out / 'preflight' / name, public(read(pf / name)))
    write(out / 'validation.json', {**summary['validation'], 'raw_schedule_aggregate_CSV_byte_identical': True,
                                  'passed_preflight_bound_by_formal_manifest': True})
    write(out / 'provenance.json', {
        'source_run': run.name, 'original_analysis_provenance_sha256': sha(analysis / 'provenance.json'),
        'original_private_manifest_sha256': sha(run / 'manifest.json'),
        'original_private_completion_sha256': sha(run / 'status.json'),
        'checkpoint_sha256': provenance['checkpoint_sha256'],
        'notes': ['Raw arm records, schedule, aggregate, CSV and Markdown reports copied byte-identically.',
                  'Only PID metadata removed from manifest/completion/preflight JSON.',
                  'Frozen originals remain local and unchanged. No training or checkpoint inference.']})
    write(publication, {
        'review_base': REVIEW_BASE, 'execution_status': 'COMPLETE', 'decision': aggregate['decision'],
        'counts': aggregate['counts'], 'source_sha256': manifest['source_sha256'],
        'analysis_source_sha256': provenance['analysis_source_sha256'],
        'publication_tool_sha256': {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))},
        'checkpoint_sha256': provenance['checkpoint_sha256'],
        'published_evidence_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(out.rglob('*')) if f.is_file()},
        'notes': ['Carry reaches and holds in0/4, concurrent baseline in2/4.',
                  'Baseline final parameters and complete evaluations exactly reproduce Phase II in4/4.',
                  'All primary paired means and pooled values decline with carry.',
                  'Previously inspected seeds/maps: development only. The result rejects this fixed-average recipe, not all spatial carry.'],
        'excluded': ['checkpoints','machine launch receipts','PID','transient logs','local smoke artifacts']})
    print(json.dumps({'status': 'EXPORTED', 'files': sum(f.is_file() for f in out.rglob('*')),
                      'bytes': sum(f.stat().st_size for f in out.rglob('*') if f.is_file())}, indent=2))


if __name__ == '__main__':
    main()
