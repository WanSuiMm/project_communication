"""Verify local Markdown routing, provenance, and an exact staged allowlist."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
import posixpath


ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    errors = []
    published = set(filter(None, git('ls-files', '-z').decode().split('\0')))
    for name in sorted(published):
        path = Path(name)
        if path.suffix.lower() != '.md':
            continue
        for target in re.findall(r'\]\(([^\s)]+)\)', git('show', f':{name}').decode('utf-8-sig')):
            if target.startswith(('http://', 'https://', '#')):
                continue
            local = unquote(target.split('#', 1)[0])
            resolved = posixpath.normpath(posixpath.join(path.parent.as_posix(), local))
            if resolved not in published and not any(p.startswith(resolved.rstrip('/')+'/') for p in published):
                errors.append(f'Unpublished/broken link: {name}: {local}')
    for manifest_path in ROOT.glob('*PUBLICATION_MANIFEST.json'):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        sources = manifest.get('original_training_source_sha256', manifest.get('source_sha256', {}))
        for name, expected in sources.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                errors.append(f'Training-source hash mismatch: {name}')
        evidence = {**manifest.get('published_evidence_sha256', {}),
                    **manifest.get('evidence_sha256', {})}
        for name, expected in evidence.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                errors.append(f'Evidence hash mismatch: {name}')
        for name, expected in manifest.get('protocol_files_sha256', {}).items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                errors.append(f'Frozen protocol hash mismatch: {name}')
        bindings = {**sources, **evidence, **manifest.get('protocol_files_sha256', {}),
                    **manifest.get('reference_sha256', {}),
                    **manifest.get('recovery_source_sha256', {}),
                    **manifest.get('publication_tool_sha256', {})}
        for name, expected in bindings.items():
            if name not in published:
                errors.append(f'Manifest references unpublished file: {name}')
            elif hashlib.sha256(git('show', f':{name}')).hexdigest() != expected:
                errors.append(f'Staged source/evidence hash mismatch: {name}')
        if 'protocol_sha256' in manifest:
            if hashlib.sha256((ROOT / 'A0_PROTOCOL.md').read_bytes()).hexdigest() != manifest['protocol_sha256']:
                errors.append('A0 frozen protocol hash mismatch')
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
        if path.suffix in {'.pt', '.pth', '.ckpt', '.pyc', '.log', '.pid', '.zip'} or path.name == 'launch_receipt.json':
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
        if name.startswith('evidence/') and path.suffix == '.json':
            data = json.loads(content)

            def inspect_keys(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in {'host', 'pid', 'cwd', 'argv', 'stdout', 'stderr', 'gpu_uuid'}:
                            errors.append(f'Private receipt metadata staged: {name}: {key}')
                        inspect_keys(item)
                elif isinstance(value, list):
                    for item in value:
                        inspect_keys(item)

            inspect_keys(data)
    if not files:
        errors.append('No staged files to verify')
    print(json.dumps({'staged_files': len(files), 'total_bytes': total, 'errors': errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
