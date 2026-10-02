"""Publish frozen Phase-II evidence with process metadata excluded."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REVIEW_BASE = '7d1e32697144add1c063d4157bbd1ca62338a45f'


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
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'analysis', 'preflight', 'out'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    run, analysis, pf, out = (Path(getattr(args, n)).resolve() for n in ('run','analysis','preflight','out'))
    publication = ROOT/'BPTT_PHASE2_PUBLICATION_MANIFEST.json'
    assert not out.exists() and not publication.exists()
    provenance = read(analysis/'provenance.json')
    manifest, status, aggregate = (read(run/name) for name in ('manifest.json','status.json','aggregate.json'))
    summary = read(analysis/'analysis.json')
    assert status['status'] == 'COMPLETE' and status['completed_arms'] == 12
    assert summary['validation']['status'] == 'PASS'
    for name, expected in provenance['input_sha256'].items():
        assert sha(ROOT/name) == expected, name
    for name, expected in provenance['checkpoint_sha256'].items():
        assert sha(run/name) == expected, name
    for name, expected in provenance['analysis_source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    for name, expected in provenance['output_sha256'].items():
        assert sha(analysis/name) == expected, name
    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT/name) == sha(run/'source'/name) == expected, name
    for name, expected in manifest['preflight_sha256'].items():
        assert sha(pf/name) == expected, name
    assert read(pf/'status.json')['status'] == 'PREFLIGHT_PASSED'
    assert read(pf/'manifest.json')['source_sha256'] == manifest['source_sha256']
    assert read(pf/'aggregate.json')['updates_per_arm'] == manifest['updates_per_arm'] == 300
    raw = sorted(run.glob('additive_*.json'))
    assert len(raw) == 12
    out.mkdir(parents=True)
    (out/'raw').mkdir()
    (out/'preflight').mkdir()
    for file in raw:
        shutil.copyfile(file, out/'raw'/file.name)
        assert sha(file) == sha(out/'raw'/file.name)
    for name in ('schedule.json', 'aggregate.json'):
        shutil.copyfile(run/name, out/name)
    shutil.copyfile(run/'RESULTS.md', out/'runner_RESULTS.md')
    for name in ('RESULTS.md','curves.csv'):
        shutil.copyfile(analysis/name, out/name)
    write(out/'analysis.json', public(summary))
    write(out/'manifest.json', public(manifest))
    write(out/'completion.json', public(status))
    for name in ('manifest.json','status.json','aggregate.json'):
        write(out/'preflight'/name, public(read(pf/name)))
    write(out/'validation.json', {**summary['validation'], 'raw_and_schedule_byte_identical': True,
                                  'passed_preflight_bound_by_formal_manifest': True})
    write(out/'provenance.json', {
        'source_run': run.name,
        'original_analysis_provenance_sha256': sha(analysis/'provenance.json'),
        'original_analysis_json_sha256': sha(analysis/'analysis.json'),
        'original_private_manifest_sha256': sha(run/'manifest.json'),
        'original_private_completion_sha256': sha(run/'status.json'),
        'checkpoint_sha256': provenance['checkpoint_sha256'],
        'notes': ['Raw arm results, schedule, aggregate, Markdown reports and CSV copied byte-identically.',
                  'Only PID metadata removed from public analysis/completion/manifest/preflight JSON.',
                  'Original local artifacts are unchanged. No training or inference during publication.']})
    write(publication, {
        'review_base': REVIEW_BASE, 'execution_status': 'COMPLETE',
        'K8_narrow_reach': aggregate['K8_narrow_reach'],
        'K8_narrow_sustained': aggregate['K8_narrow_sustained'],
        'comparison_status': aggregate['comparison_status'],
        'source_sha256': manifest['source_sha256'],
        'analysis_source_sha256': provenance['analysis_source_sha256'],
        'publication_tool_sha256': {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))},
        'checkpoint_sha256': provenance['checkpoint_sha256'],
        'published_evidence_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(out.rglob('*')) if f.is_file()},
        'notes': ['K8 reaches and holds in2/4 seeds, below the frozen3/4 criterion.',
                  'K16 reaches3/4, reaches and holds1/4; K64 qualifies0/4.',
                  'New narrow endpoint was selected from Phase I and frozen before this run.',
                  'Fixed bank and schedule; n=4 initialization replication only; old gate unchanged.'],
        'excluded': ['checkpoints','machine launch receipts','PID','transient logs','local smoke artifacts']})
    print(json.dumps({'status':'EXPORTED', 'files':sum(f.is_file() for f in out.rglob('*')),
                      'bytes':sum(f.stat().st_size for f in out.rglob('*') if f.is_file())}, indent=2))


if __name__ == '__main__':
    main()
