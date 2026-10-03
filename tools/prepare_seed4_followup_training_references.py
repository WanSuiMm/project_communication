"""Restore public reference inputs needed by the frozen A/B runner on a fresh clone."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    publication = json.loads((ROOT/'STREAMING_CARRY_PUBLICATION_MANIFEST.json').read_text())
    directory = ROOT/'runs/streaming_carry_20261002_init2345'
    if directory.exists():
        raise SystemExit('Reference directory already exists; leave the existing archive unchanged.')
    inputs = {
        'manifest.json': 'evidence/streaming_carry_init2345/manifest.json',
        'schedule.json': 'evidence/streaming_carry_init2345/schedule.json',
        'stream_K8_seed4.json': 'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json',
    }
    for source in inputs.values():
        expected = publication['published_evidence_sha256'][source]
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == expected
    directory.mkdir(parents=True, exist_ok=False)
    for name, source in inputs.items():
        shutil.copyfile(ROOT/source, directory/name)
    print('Restored three public reference inputs for A/B. No training/checkpoint was created.')


if __name__ == '__main__':
    main()
