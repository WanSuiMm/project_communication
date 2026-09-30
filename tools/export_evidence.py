"""Publish an allowlisted snapshot without changing frozen local evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    content = (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f'Refusing to replace an existing export: {path.name}')
    else:
        path.write_bytes(content)


def copy_new(source, destination):
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f'Existing export differs: {destination.name}')
    else:
        shutil.copyfile(source, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--smoke', required=True)
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--checks', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run, smoke, analysis, out = [Path(p).resolve() for p in (args.run, args.smoke, args.analysis, args.out)]
    receipt = read(run / 'launch_receipt.json')
    aggregate = read(run / 'aggregate.json')
    assert aggregate['status'] == 'COMPLETED_FROZEN_SCHEDULE'
    assert len(aggregate['results']) == 20
    hashes = receipt['source_sha256']
    for name, expected in hashes.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Training source changed since qualification: {name}')
    out.mkdir(parents=True, exist_ok=True)
    for name in ('aggregate.json', 'config.json', 'RESULTS.md'):
        copy_new(run / name, out / name)
    for name in ('summary.json', 'gate_a.png'):
        copy_new(analysis / name, out / name)
    validation = out.parent / 'validation'
    validation.mkdir(exist_ok=True)
    copy_new(smoke / 'smoke.json', validation / 'smoke.json')
    checks = read(Path(args.checks))
    assert checks['implicit_edge_and_rhs_gradcheck'] is True
    write(validation / 'transport_checks.json', checks)
    manifest = {
        'canonical_run': run.name,
        'protocol': read(run / 'config.json')['protocol'],
        'started_utc': receipt['started_utc'],
        'scientific_status': aggregate['decisions'],
        'runtime': {'python': receipt['python'], 'torch': receipt['torch'],
                    'gpu_model': receipt['device']},
        'original_training_source_sha256': hashes,
        'training_source_matches_published_files': True,
        'packaging_only_additions': ['run.py', 'tools/export_evidence.py', 'review documentation'],
        'excluded': ['checkpoints', 'launch receipts', 'machine identities', 'caches',
                     'interrupted orientation-confounded v1 output', 'duplicate source snapshots'],
        'excluded_pilot_reason': 'axis=step mod dim confounded distance and orientation in 2D; '
                                 'canonical v1_1 crosses every distance with every axis',
        'published_evidence_sha256': {
            str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.parent.rglob('*') if p.is_file()
        }
    }
    write(root / 'PUBLICATION_MANIFEST.json', manifest)
    print(json.dumps({'exported_run': run.name, 'training_source_hashes_verified': True,
                      'evidence_files': len(manifest['published_evidence_sha256'])}))


if __name__ == '__main__':
    main()
