"""Zero-training selected StreamingCell reevaluation with the unchanged Full rule."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / 'runs/addressed_delta_20261008_02'
HISTORY = ROOT / 'runs/seed4_followup_20261003_training01/manifest.json'
PROTOCOL = 'streaming_selected_full_reevaluation_v1'
SPECIFICATIONS = ((0, 300, True), (2, 300, True), (0, 275, False))
COHORTS = ('historical_full', 'addressed_selection')

spec = importlib.util.spec_from_file_location('_streaming_full_existing_evaluation',
    ROOT / 'new/continuous_coverage/evaluation.py')
E = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = E
spec.loader.exec_module(E)
C = E.C


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    C.write(path, value)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def load_banks():
    historical = {s: C.region_bank(s, 32, 50000+s) for s in (32, 64)}
    expected_history = read(HISTORY)['fresh_evaluation_data_sha256']
    selection = {}
    expected_selection = read(PARENT / 'manifest.json')['data_sha256']
    for size in (32, 64):
        assert C.tensor_hash(historical[size]) == expected_history[str(size)], 'Historical bank drift'
        with np.load(PARENT / f'banks/evaluation{size}.npz', allow_pickle=False) as arrays:
            selection[size] = {k: torch.from_numpy(np.ascontiguousarray(arrays[k])) for k in arrays.files}
        assert C.tensor_hash(selection[size]) == expected_selection[f'evaluation{size}'], 'Selection bank drift'
        assert len(selection[size]['x']) == len(historical[size]['x']) == 32
    banks = dict(zip(COHORTS, (historical, selection)))
    hashes = {name: {str(s): C.tensor_hash(data[s]) for s in (32, 64)}
              for name, data in banks.items()}
    return banks, hashes


def load_checkpoint(block, update, primary, device='cpu'):
    folder = PARENT / f'block{block:02d}/current'
    path = folder / f'checkpoints/u{update:03d}.pt'
    marker = read(Path(str(path)+'.complete.json'))
    row = read(folder / f'evaluation/u{update:03d}/record.json')
    file_hash = C.sha(path)
    assert marker['stage_complete'] and marker['completed_update'] == update
    assert digest(row) == marker['record_sha256'], 'Stage record binding drift'
    assert file_hash == marker['checkpoint_sha256'] == row['checkpoint_sha256']
    assert row['checkpoint'] == path.relative_to(PARENT).as_posix()
    assert row['arm'] == 'current' and row['block'] == block and row['update'] == update
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['protocol'] == 'addressed_delta_native_k8_development_v1'
    assert payload['arm'] == 'current' and payload['completed_updates'] == update
    assert payload['initialization_seed'] == row['initialization_seed'] == 140001+block
    assert payload['schedule_seed'] == row['schedule_seed'] == 141001+block
    model = C.StreamingCell(streaming=True).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 5033
    assert C.tensor_hash(model.state_dict()) == row['parameter_sha256']
    model.eval().requires_grad_(False)
    binding = {'model': f'block{block:02d}_u{update:03d}', 'block': block,
        'update': update, 'primary': primary, 'checkpoint': relative(path),
        'checkpoint_sha256': file_hash, 'parameter_sha256': row['parameter_sha256'],
        'initialization_seed': row['initialization_seed'], 'schedule_seed': row['schedule_seed'],
        'stage_record_sha256': C.sha(folder / f'evaluation/u{update:03d}/record.json')}
    return model, binding


def sources():
    C._load_audit()
    # The audit loads these modules outside sys.modules (including temporary
    # `metrics` binding); include them explicitly in the frozen source receipt.
    files = {HERE / 'run.py', HERE / 'PROTOCOL.md',
        ROOT / 'new/frontier_audit/metrics.py',
        ROOT / 'new/short_bptt_phase2/run.py', ROOT / 'new/stream_path_audit/audit.py'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix == '.py' and path.is_file():
                files.add(path)
    hashes = {relative(p): C.sha(p) for p in sorted(files)}
    parent_sources = read(PARENT / 'manifest.json')['source_sha256']
    for name, actual in hashes.items():
        if not name.startswith('new/streaming_full_reeval/'):
            assert actual == parent_sources[name], f'Frozen parent source drift: {name}'
    return hashes


def verify_sources(expected):
    assert all(C.sha(ROOT / name) == h for name, h in expected.items()), 'Source changed during evaluation'


def inputs():
    banks, hashes = load_banks()
    bindings = []
    for args in SPECIFICATIONS:
        model, binding = load_checkpoint(*args)
        bindings.append(binding)
        del model
    return banks, {'protocol': PROTOCOL, 'bank_sha256': hashes, 'checkpoints': bindings,
        'source_sha256': sources(), 'historical_manifest_sha256': C.sha(HISTORY),
        'parent_manifest_sha256': C.sha(PARENT / 'manifest.json')}


def short_check(out):
    C.setup_backend()
    banks, binding = inputs()
    model, _ = load_checkpoint(*SPECIFICATIONS[0], device='cuda')
    before = C.tensor_hash(model.state_dict())
    shapes = []
    started = time.monotonic()
    with torch.inference_mode():
        for size in (32, 64):
            data = {k: v.cuda() for k, v in banks['historical_full'][size].items()}
            a, b = model.initial(data['x']), model.initial(data['x_flip'])
            for _ in range(4):
                a, b = model.step(a, data['x']), model.step(b, data['x_flip'])
            assert all(bool(torch.isfinite(v).all()) for v in (*a, *b, model.logits(a), model.logits(b)))
            shapes.append(list(model.logits(a).shape))
            del a, b, data
    assert C.tensor_hash(model.state_dict()) == before
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    # Deterministic bit packing round trip, independent of task performance.
    example = (np.arange(257*3*5).reshape(257, 3, 5) % 3) == 0
    restored = np.unpackbits(np.packbits(example.reshape(-1), bitorder='little'),
        bitorder='little', count=example.size).reshape(example.shape).astype(bool)
    assert np.array_equal(example, restored)
    torch.cuda.synchronize()
    result = {'status': 'PASS', 'scientific_endpoint_evaluated': False,
        'training': False, 'optimizer_updates': 0, 'actual_shape_four_steps': shapes,
        'check_seconds': time.monotonic()-started, 'input_binding': binding,
        'created_utc': C.now()}
    write(out, result)
    print(json.dumps({'status': 'PASS', 'qualification': relative(out),
        'checkpoint_count': 3, 'cohorts': 2, 'shape_checks': shapes}), flush=True)


def report(out, rows, status):
    lines = ['# StreamingCell Full reevaluation', '',
        f'Status: **{status}**, {len(rows)}/6 model/cohort units. No training or optimizer updates.', '',
        'Selected-checkpoint descriptive evaluation. Reused unchanged historical Full predicate.',
        'The addressed cohort selected the primary models; it is not independent validation.', '',
        '| Checkpoint | Role | Cohort | Full | Failed gates |',
        '|---|---|---|---|---|']
    for row in rows:
        lines.append(f"| {row['model']} | {'primary' if row['primary'] else 'diagnostic'} | "
            f"{row['cohort']} | {'PASS' if row['full_pass'] else 'FAIL'} | {len(row['reasons'])} |")
    lines += ['', '## Complete gate values and evidence', '']
    for row in rows:
        folder = row['unit']
        lines += [f"- {row['model']} / {row['cohort']}: [full summary]({folder}/summary.json), "
            f"[matched frontier]({folder}/matched_frontier_strata.csv)."]
        lines += ['  '+reason for reason in row['reasons']]
    lines += ['', 'Each unit retains losslessly packed size32/size64 original, flipped and paired',
        'correctness arrays at every integer time0..256, plus per-map summaries and full frontier strata.',
        'No intermediate diagnostic checkpoint replaces u300. Pass counts are not a formation rate.', '']
    path = out / 'RESULTS.md'
    temporary = path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(path)
    write(out / 'summary.json', {'protocol': PROTOCOL, 'status': status,
        'expected_units': 6, 'completed_units': len(rows), 'training': False,
        'optimizer_updates': 0, 'units': rows,
        'primary_passes_by_cohort': {name: sum(r['primary'] and r['cohort'] == name and r['full_pass']
                                             for r in rows) for name in COHORTS},
        'claim_boundary': 'Selected-checkpoint descriptive Full evaluation; no formation-rate or mechanism claim.'})


def run(out, qualification):
    if out.exists():
        raise FileExistsError('New run directory required; existing evidence is read-only')
    C.setup_backend()
    banks, binding = inputs()
    check = read(qualification)
    assert check['status'] == 'PASS' and check['input_binding'] == binding, 'Qualification/input drift'
    out.mkdir(parents=True)
    started, rows = time.monotonic(), []
    manifest = {**binding, 'started_utc': C.now(), 'training': False, 'optimizer_updates': 0,
        'expected_units': 6, 'qualification': relative(qualification),
        'qualification_sha256': C.sha(qualification), 'runtime_limit_enforced': False,
        'recurring_monitor': False, 'pid': os.getpid(), 'gpu': torch.cuda.get_device_name(),
        'torch': str(torch.__version__), 'backend': {'cudnn_benchmark': False,
            'cudnn_deterministic': False, 'cudnn_tf32': True, 'matmul_tf32': False}}
    write(out / 'manifest.json', manifest)
    for name, data in banks.items():
        for size, bank in data.items():
            path = out / f'banks/{name}/size{size}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **{k: v.numpy() for k, v in bank.items()})
    for name in binding['source_sha256']:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    report(out, rows, 'RUNNING')
    try:
        for block, update, primary in SPECIFICATIONS:
            model, model_binding = load_checkpoint(block, update, primary, device='cuda')
            for cohort in COHORTS:
                unit = f"{model_binding['model']}/{cohort}"
                write(out / 'status.json', {'status': 'RUNNING', 'current_unit': unit,
                    'completed_units': len(rows), 'expected_units': 6, 'updated_utc': C.now()})
                tick = time.monotonic()
                summary = E.evaluate(model, banks[cohort], out / unit, full=True)
                assert C.tensor_hash(model.state_dict()) == model_binding['parameter_sha256']
                assert all(p.grad is None for p in model.parameters())
                row = {**model_binding, 'cohort': cohort, 'unit': unit,
                    'full_pass': summary['phenotype_gate']['pass'],
                    'reasons': summary['phenotype_gate']['reasons'],
                    'seconds': time.monotonic()-tick, 'completed_utc': C.now()}
                artifacts = {p.name: C.sha(p) for p in (out / unit).iterdir() if p.is_file()}
                write(out / unit / 'complete.json', {'complete': True, 'binding': row,
                    'artifacts_sha256': artifacts})
                rows.append(row)
                report(out, rows, 'RUNNING')
                print(json.dumps({'completed_units': len(rows), 'unit': unit,
                    'full_pass': row['full_pass'], 'failed_gates': len(row['reasons']),
                    'seconds': row['seconds']}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        verify_sources(binding['source_sha256'])
        for cohort in COHORTS:
            assert {str(s): C.tensor_hash(v) for s, v in banks[cohort].items()} == binding['bank_sha256'][cohort]
        report(out, rows, 'COMPLETE')
        write(out / 'status.json', {'status': 'COMPLETE', 'completed_units': len(rows),
            'expected_units': 6, 'seconds': time.monotonic()-started, 'finished_utc': C.now(),
            'training': False, 'optimizer_updates': 0})
    except BaseException as error:
        report(out, rows, 'INCOMPLETE')
        write(out / 'status.json', {'status': 'ERROR', 'completed_units': len(rows),
            'expected_units': 6, 'error': repr(error), 'traceback': traceback.format_exc(),
            'updated_utc': C.now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--out', required=True)
    parser.add_argument('--qualification')
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    area = ROOT / ('analyses' if args.check else 'runs')
    assert out.is_relative_to(area) and out != area, 'Output must be inside the project evidence area'
    if args.check:
        assert not out.exists(), 'New qualification output required'
        short_check(out)
    else:
        assert args.qualification, 'Passed qualification required'
        run(out, (ROOT / args.qualification).resolve())


if __name__ == '__main__':
    main()
