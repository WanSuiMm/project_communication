"""Frozen block7 checkpoint continuation and same-state one-step interventions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'new/streaming_carry'))
from stream_cells import StreamingCell
_spec = importlib.util.spec_from_file_location('_block7_collapse_metrics', Path(__file__).with_name('metrics.py'))
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
summarize_suffix, summarize_step = _metrics.summarize_suffix, _metrics.summarize_step

PROTOCOL = 'block7_collapse_continuation_v1'
UPDATES = (225, 250, 275, 300)
SIZES = (32, 64)
TIMES = (64, 128, 192)
SOURCE = ROOT / 'runs/continuous_coverage_20261006_02'
PUBLIC = ROOT / 'evidence/continuous_coverage_20261006'
MODEL_SOURCES = ('new/streaming_carry/stream_cells.py',
                 'new/workspace_revision/revision_cells.py',
                 'new/masked_medium/masked_cells.py',
                 'new/nca_inertial_wind_tunnel/cells.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def tensor_hash(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def setup_backend():
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'Local CUDA backend required'
    assert str(torch.__version__).startswith('2.5.1'), 'Historical Torch2.5.1 required'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


def load_material():
    manifest = read(ROOT / 'CONTINUOUS_COVERAGE_PUBLICATION_MANIFEST.json')
    bindings = {}
    for name in MODEL_SOURCES:
        assert sha(ROOT / name) == manifest['source_sha256'][name], f'Model source drift: {name}'
        bindings[name] = sha(ROOT / name)
    models, checkpoints = {}, {}
    for update in UPDATES:
        relative = f'block07/reset64x4/checkpoints/u{update:03}.pt'
        bound = manifest['checkpoint_bindings'][relative]
        path = SOURCE / relative
        assert sha(path) == bound['file_sha256'], f'Checkpoint binding drift u{update}'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['completed_updates'] == update
        assert payload['initialization_seed'] == 96008 and payload['schedule_seed'] == 97008
        assert payload['arm'] == 'reset64x4'
        assert payload['protocol'] == 'continuous_execution_coverage_v1'
        model = StreamingCell().cuda().eval()
        model.load_state_dict(payload['state_dict'], strict=True)
        model.requires_grad_(False)
        assert sum(p.numel() for p in model.parameters()) == 5033
        assert tensor_hash(model.state_dict()) == bound['parameter_sha256']
        models[update] = model
        checkpoints[str(update)] = {'path': f'runs/continuous_coverage_20261006_02/{relative}',
                                    'sha256': sha(path), 'parameter_sha256': bound['parameter_sha256']}
        del payload
    banks = {}
    for size in SIZES:
        relative = f'evidence/continuous_coverage_20261006/banks/evaluation{size}.npz'
        assert sha(ROOT / relative) == manifest['published_sha256'][relative]
        with np.load(ROOT / relative, allow_pickle=False) as archive:
            bank = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
        assert tensor_hash(bank) == manifest['data_sha256'][f'evaluation{size}']
        banks[size] = bank
        bindings[relative] = sha(ROOT / relative)
    natives = {}
    for update in UPDATES:
        for size in SIZES:
            relative = f'evidence/continuous_coverage_20261006/block07/reset64x4/evaluation/u{update:03}/size{size}_traces.npz'
            assert sha(ROOT / relative) == manifest['published_sha256'][relative]
            with np.load(ROOT / relative, allow_pickle=False) as archive:
                arrays = {}
                for key in ('correct', 'original_correct', 'flipped_correct'):
                    shape = tuple(archive[key+'_shape'])
                    arrays[key] = np.unpackbits(archive[key+'_packed'], bitorder='little',
                                               count=int(np.prod(shape))).reshape(shape).astype(bool)
            assert np.array_equal(arrays['correct'], arrays['original_correct'] & arrays['flipped_correct'])
            natives[update, size] = arrays
            bindings[relative] = sha(ROOT / relative)
    return models, banks, natives, {'input_sha256': bindings, 'checkpoints': checkpoints}


def pack(path, arrays):
    values = {}
    for key, value in arrays.items():
        array = np.asarray(value, dtype=bool)
        values[key+'_shape'] = np.asarray(array.shape, dtype=np.int32)
        values[key+'_packed'] = np.packbits(array.reshape(-1), bitorder='little')
    np.savez_compressed(path, **values)


def state_copy(state):
    return tuple(value.clone() for value in state)


def finite_state(flag, *states):
    for state in states:
        for value in state:
            flag = flag & torch.isfinite(value).all()
    return flag


def observe(reader, a, b, bank):
    first, second = reader.logits(a), reader.logits(b)
    good_a = ((first >= 0) == (bank['y'] >= .5))[:, 0]
    good_b = ((second >= 0) == (bank['y_flip'] >= .5))[:, 0]
    return {'original_correct': good_a, 'flipped_correct': good_b,
            'correct': good_a & good_b}, torch.isfinite(first).all() & torch.isfinite(second).all()


def save_states(path, snapshots):
    arrays = {}
    for t, (a, b) in snapshots.items():
        for world, state in (('original', a), ('flipped', b)):
            arrays[f't{t}_{world}_C'] = state[0].cpu().numpy()
            arrays[f't{t}_{world}_Z'] = state[1].cpu().numpy()
    np.savez_compressed(path, **arrays)


@torch.no_grad()
def produce(model, bank, stop=64):
    a, b = model.initial(bank['x']), model.initial(bank['x_flip'])
    shape = (stop+1, len(bank['x']), *bank['x'].shape[-2:])
    traces = {key: torch.empty(shape, device='cuda', dtype=torch.bool)
              for key in ('correct', 'original_correct', 'flipped_correct')}
    finite = torch.ones((), dtype=torch.bool, device='cuda')
    for t in range(stop+1):
        if t:
            a, b = model.step(a, bank['x']), model.step(b, bank['x_flip'])
        values, good = observe(model, a, b, bank)
        finite = finite_state(finite & good, a, b)
        for key in traces:
            traces[key][t].copy_(values[key])
    if not bool(finite):
        raise FloatingPointError('Nonfinite producer')
    return (a, b), {key: value.cpu().numpy() for key, value in traces.items()}


@torch.no_grad()
def continue_state(consumer, readers, pair, bank, stop=192):
    a, b = state_copy(pair[0]), state_copy(pair[1])
    shape = (stop+1, len(bank['x']), *bank['x'].shape[-2:])
    traces = {view: {key: torch.empty(shape, device='cuda', dtype=torch.bool)
                     for key in ('correct', 'original_correct', 'flipped_correct')}
              for view in readers}
    snapshots = {64: (state_copy(a), state_copy(b))}
    finite = torch.ones((), dtype=torch.bool, device='cuda')
    for offset in range(stop+1):
        if offset:
            a, b = consumer.step(a, bank['x']), consumer.step(b, bank['x_flip'])
        finite = finite_state(finite, a, b)
        for view, reader in readers.items():
            values, good = observe(reader, a, b, bank)
            finite = finite & good
            for key in values:
                traces[view][key][offset].copy_(values[key])
        if offset+64 in (128, 192):
            snapshots[offset+64] = (state_copy(a), state_copy(b))
    if not bool(finite):
        raise FloatingPointError('Nonfinite consumer')
    return {view: {key: value.cpu().numpy() for key, value in arrays.items()}
            for view, arrays in traces.items()}, snapshots


def compare_bits(measured, expected):
    return {key: int(np.count_nonzero(measured[key] != expected[key])) for key in measured}


def report(out, matrix, steps, validation, elapsed, complete=False):
    summary = {'protocol': PROTOCOL, 'status': 'COMPLETE' if complete else 'RUNNING',
               'training_updates': 0, 'optimizer_updates': 0, 'parameter_changes': False,
               'independent_training_trajectories': 1, 'completed_matrix_units': len(matrix),
               'expected_matrix_units': 32, 'completed_single_step_units': len(steps),
               'expected_single_step_units': 12, 'elapsed_seconds': elapsed,
               'native_replay': validation,
               'claim_boundary': 'Selected same-trajectory interventions; compatibility and readout controls, not population reliability or unique mechanism identification.'}
    write(out/'summary.json', summary)
    write(out/'matrix.json', matrix)
    write(out/'single_step.json', steps)
    lines = ['# Block7 formation/collapse continuation audit', '',
             f"Status: **{summary['status']}**, matrix {len(matrix)}/32, single-step {len(steps)}/12.", '',
             'No training, optimizer update, alignment fit or parameter change. One selected training trajectory.', '',
             'Tables use the fixed u275 readout. R_C (native consumer) and R_P views, all counts and per-map rows are in matrix.json.', '',
             'Preservation is continuous on producer-native T64-correct cells; progress is sustained for times241..256 on producer-native T64-wrong cells.', '']
    for size in SIZES:
        lines.extend([f'## size{size}', '', '|Producer|Consumer|Preservation|Sustained progress|Terminal coverage|Handoff destruction|',
                      '|---:|---:|---:|---:|---:|---:|'])
        for record in matrix:
            if record['size'] != size:
                continue
            row = record['views']['fixed275']['all_changed']
            def fmt(value):
                return 'null' if value is None else f'{value:.6f}'
            keys = ('continuous_preservation', 'sustained_progress_last16')
            values = [fmt(row[key]['pooled']['value']) for key in keys]
            values.extend((fmt(row['endpoint_coverage']['T256']['pooled']['value']), fmt(row['handoff_destruction']['pooled']['value'])))
            lines.append(f"|{record['producer']}|{record['consumer']}|"+'|'.join(values)+'|')
        lines.append('')
    lines.extend(['## Same-state single-step (fixed R275)', '',
                  '|Size|Native state time|Consumer|Destruction|Acquisition|Coverage change|',
                  '|---:|---:|---:|---:|---:|---:|'])
    for record in steps:
        row = record['metrics']['all_changed']
        values = [fmt(row[k]['pooled']['value']) for k in ('destruction', 'acquisition')]
        values.append(fmt(row['net_coverage_change']['pooled']['value']))
        lines.append(f"|{record['size']}|{record['state_time']}|{record['consumer']}|"+'|'.join(values)+'|')
    lines.extend(['', 'Raw producer/consumer matrices and readout-switch changes are descriptive. A failed handoff can reflect interface mismatch; a failed rescue is not irreversibility. Earlier frozen u300 qualification remains negative.', ''])
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


@torch.no_grad()
def smoke(out, models, banks, binding):
    started = time.monotonic()
    bank = {key: value[:2].cuda() for key, value in banks[32].items()}
    states, prefix = produce(models[275], bank)
    traces, _ = continue_state(models[275], {'native': models[275]}, states, bank, stop=8)
    # Independent uninterrupted short path, exercising whole-state handoff.
    _, full = produce(models[275], bank, stop=72)
    diffs = compare_bits(traces['native'], {k: v[64:] for k, v in full.items()})
    assert not any(diffs.values()), diffs
    before_hash = tensor_hash(models[275].state_dict())
    assert all(p.grad is None and not p.requires_grad for model in models.values() for p in model.parameters())
    assert before_hash == binding['checkpoints']['275']['parameter_sha256']
    result = {'protocol': PROTOCOL, 'status': 'PASS', 'smoke_maps': 2, 'size': 32,
              'handoff_mismatch_bits': diffs, 'training_updates': 0,
              'elapsed_seconds': time.monotonic()-started,
              'estimated_formal_seconds': None, 'runtime_limit_enforced': False}
    write(out/'smoke.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = (ROOT/args.out).resolve()
    assert out.is_relative_to(ROOT), 'Output must be inside project'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    matrix, steps, validation = [], [], []
    try:
        setup_backend()
        models, banks, natives, binding = load_material()
        sources = {name: sha(ROOT/name) for name in MODEL_SOURCES}
        for path in Path(__file__).parent.iterdir():
            if path.suffix in ('.py', '.md'):
                sources[path.relative_to(ROOT).as_posix()] = sha(path)
        for name in sources:
            destination = out/'source'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, destination)
        manifest = {'protocol': PROTOCOL, 'started_utc': now(), 'pid': os.getpid(),
                    'host': os.environ.get('COMPUTERNAME'), 'gpu': torch.cuda.get_device_name(),
                    'command': [sys.executable, *sys.argv], 'source_sha256': sources,
                    **binding, 'backend': {'torch': str(torch.__version__), 'cudnn_benchmark': False,
                    'cudnn_deterministic': False, 'cudnn_tf32': True, 'matmul_tf32': False},
                    'review_base': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    'runtime_limit_enforced': False, 'watchdog_enabled': False,
                    'continuous_monitoring': False, 'training_updates': 0}
        write(out/'manifest.json', manifest)
        if args.smoke:
            smoke(out, models, banks, binding)
            return
        (out/'states').mkdir()
        (out/'matrix_traces').mkdir()
        (out/'single_step_traces').mkdir()

        def progress(phase, **kwargs):
            value = {'status': 'RUNNING', 'pid': os.getpid(), 'updated_utc': now(),
                     'elapsed_seconds': time.monotonic()-started, 'phase': phase,
                     'completed_matrix_units': len(matrix), 'completed_single_step_units': len(steps),
                     'expected_measurement_units': 44, **kwargs}
            write(out/'status.json', value)
            print(json.dumps(value), flush=True)

        progress('inputs_bound')
        for size in SIZES:
            bank = {key: value.cuda() for key, value in banks[size].items()}
            changed = banks[size]['changed'][:,0].numpy().astype(bool)
            distance = banks[size]['distance'][:,0].numpy()
            for producer in UPDATES:
                progress('producer', size=size, producer=producer)
                pair, prefix = produce(models[producer], bank)
                prefix_diffs = compare_bits(prefix, {k:v[:65] for k,v in natives[producer,size].items()})
                assert not any(prefix_diffs.values()), f'Native prefix mismatch: {prefix_diffs}'
                save_states(out/'states'/f'size{size}_p{producer}_T64.npz', {64: pair})
                baseline = prefix['correct'][64]
                for consumer in UPDATES:
                    progress('consumer', size=size, producer=producer, consumer=consumer)
                    tick = time.monotonic()
                    readers = {'producer': models[producer], 'consumer': models[consumer], 'fixed275': models[275]}
                    values, snapshots = continue_state(models[consumer], readers, pair, bank)
                    suffix_diffs = None
                    if consumer == producer:
                        suffix_diffs = compare_bits(values['consumer'], {k:v[64:] for k,v in natives[producer,size].items()})
                        assert not any(suffix_diffs.values()), f'Native suffix mismatch: {suffix_diffs}'
                        validation.append({'size': size, 'checkpoint': producer,
                                           'prefix_mismatch_bits': prefix_diffs, 'suffix_mismatch_bits': suffix_diffs})
                    if producer == 275 and consumer == 275:
                        save_states(out/'states'/f'size{size}_p275_T64_T128_T192.npz', snapshots)
                        native_snapshots = snapshots
                    pack(out/'matrix_traces'/f'size{size}_p{producer}_c{consumer}.npz',
                         {f'{view}_{key}':value for view, arrays in values.items() for key,value in arrays.items()})
                    record = {'size': size, 'producer': producer, 'consumer': consumer,
                              'views': {view:summarize_suffix(arrays['correct'], baseline, changed, distance)
                                        for view,arrays in values.items()},
                              'native_replay': suffix_diffs, 'elapsed_seconds': time.monotonic()-tick}
                    matrix.append(record)
                    report(out, matrix, steps, validation, time.monotonic()-started)
                    del values, snapshots
                del pair, prefix
            for t in TIMES:
                pair = native_snapshots[t]
                before, ok = observe(models[275], *pair, bank)
                assert bool(ok)
                for consumer in (275,300):
                    progress('single_step', size=size, state_time=t, consumer=consumer)
                    a = models[consumer].step(state_copy(pair[0]), bank['x'])
                    b = models[consumer].step(state_copy(pair[1]), bank['x_flip'])
                    after, ok = observe(models[275], a, b, bank)
                    assert bool(finite_state(ok,a,b)), 'Nonfinite single-step'
                    before_cpu = {k:v.cpu().numpy() for k,v in before.items()}
                    after_cpu = {k:v.cpu().numpy() for k,v in after.items()}
                    if consumer == 275:
                        diffs = compare_bits(after_cpu, {k:v[t+1] for k,v in natives[275,size].items()})
                        assert not any(diffs.values()), f'Native single-step mismatch: {diffs}'
                    record = {'size':size,'state_time':t,'producer':275,'consumer':consumer,
                              'readout':275, 'metrics':summarize_step(before_cpu['correct'],after_cpu['correct'],changed,distance)}
                    steps.append(record)
                    pack(out/'single_step_traces'/f'size{size}_t{t}_c{consumer}.npz',
                         {**{f'before_{k}':v for k,v in before_cpu.items()},
                          **{f'after_{k}':v for k,v in after_cpu.items()}})
                    report(out,matrix,steps,validation,time.monotonic()-started)
                    del a,b,after
            del bank,native_snapshots
        assert len(matrix)==32 and len(steps)==12 and len(validation)==8
        for update, model in models.items():
            assert tensor_hash(model.state_dict()) == binding['checkpoints'][str(update)]['parameter_sha256']
            assert all(p.grad is None for p in model.parameters())
        for name,expected in {**sources,**binding['input_sha256']}.items():
            assert sha(ROOT/name)==expected, f'Binding changed during audit: {name}'
        report(out,matrix,steps,validation,time.monotonic()-started,complete=True)
        write(out/'status.json', {'status':'COMPLETE','pid':os.getpid(),'finished_utc':now(),
                                 'completed_matrix_units':32,'completed_single_step_units':12,
                                 'elapsed_seconds':time.monotonic()-started,'training_updates':0})
        print(json.dumps(read(out/'status.json')),flush=True)
    except Exception as error:
        write(out/'status.json', {'status':'ERROR','pid':os.getpid(),'updated_utc':now(),
                                 'error':str(error),'completed_matrix_units':len(matrix),
                                 'completed_single_step_units':len(steps)})
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
