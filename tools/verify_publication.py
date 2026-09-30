"""Verify local Markdown routing, provenance, and an exact staged allowlist."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    errors = []
    for path in ROOT.glob('*.md'):
        for target in re.findall(r'\]\(([^\s)]+)\)', path.read_text(encoding='utf-8')):
            if target.startswith(('http://', 'https://', '#')):
                continue
            local = target.split('#', 1)[0]
            if not (path.parent / local).exists():
                errors.append(f'Broken link: {path.name}: {local}')
    manifest = json.loads((ROOT / 'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['original_training_source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            errors.append(f'Training-source hash mismatch: {name}')
    for name, expected in manifest['published_evidence_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            errors.append(f'Evidence hash mismatch: {name}')
    patterns = {
        'private_key': r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
        'github_token': r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})',
        'api_key': r'(?:sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})',
        'absolute_windows_path': r'(?<![A-Za-z])[A-Za-z]:[\\/]',
        'private_machine_path': r'/(?:home|data/users|Users)/[^\s"\']+',
        'private_ip': r'\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b',
        'gpu_uuid': r'\bGPU-[a-fA-F0-9]{8}-',
    }
    files = git('diff', '--cached', '--name-only', '-z').decode().split('\0')
    files = [p for p in files if p]
    total = 0
    for name in files:
        path = Path(name)
        if name.startswith(('runs/', 'analyses/')) or '__pycache__' in path.parts:
            errors.append(f'Local output staged: {name}')
        if path.suffix in {'.pt', '.pth', '.ckpt', '.pyc', '.log', '.pid'} or path.name == 'launch_receipt.json':
            errors.append(f'Excluded artifact staged: {name}')
        blob = git('show', f':{name}')
        total += len(blob)
        if len(blob) > 1_000_000:
            errors.append(f'Oversized staged file: {name}')
        if path.suffix == '.png':
            continue
        try:
            content = blob.decode('utf-8-sig')
        except UnicodeDecodeError:
            errors.append(f'Unexpected binary file: {name}')
            continue
        for rule, pattern in patterns.items():
            if re.search(pattern, content):
                # Do not emit potentially sensitive matching text.
                errors.append(f'Content scan: {name}: {rule}')
    if not files:
        errors.append('No staged files to verify')
    print(json.dumps({'staged_files': len(files), 'total_bytes': total, 'errors': errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
