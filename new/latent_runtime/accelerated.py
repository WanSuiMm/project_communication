"""Resume the frozen scientific suite in a separate qualified local-graph run."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = load('_accelerated_width_original', ROOT / 'new/latent_width/run.py')
C = R.C
G = load('_accelerated_width_runtime', Path(__file__).with_name('runtime.py'))
AMENDMENT = 'local_cuda_graph_k8_v1'


def runtime_hashes():
    names = [p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.iterdir()
             if p.suffix in ('.py', '.md')]
    names.append('tools/launch_latent_accelerated.ps1')
    return {name: C.sha(ROOT / name) for name in sorted(names)}


def verify_import(original, row):
    checkpoint = original / row['checkpoint']
    assert C.sha(checkpoint) == row['checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['completed_updates'] == 300 and saved['arm'] == row['arm']
    assert saved['initialization_seed'] == row['initialization_seed']
    assert C.tensor_hash(saved['state_dict']) == row['final_parameter_sha256']
    assert all(int(s['step'].item()) == 300 for s in saved['optimizer_state_dict']['state'].values())
    curve = R.read(original / row['training_curve'])
    assert len(curve) == 300 and [v['update'] for v in curve] == list(range(1, 301))
    summary = R.read(original / row['evaluation_summary'])
    assert bool(summary['phenotype_gate']['pass']) == row['phenotype_pass']
    assert C.compact(summary) == row['evaluation_compact']
    for size in (32, 64):
        assert (original / row['evaluation_summary']).with_name(f"{row['arm']}_size{size}.npz").is_file()
    return saved


def run(original, out, qualification, handover):
    assert not out.exists() and original.is_relative_to(ROOT / 'runs')
    q = R.read(qualification)
    assert q['status'] == 'PASS' and q['protocol'] == 'native_latent_graph_exact300_v1'
    for name, digest in q['source_sha256'].items():
        assert C.sha(ROOT / name) == digest, f'Qualification source drift: {name}'
    for name, digest in q['references_sha256'].items():
        assert C.sha(ROOT / name) == digest, f'Qualification reference drift: {name}'
    previous = R.read(original / 'manifest.json')
    frozen = R.source_hashes()
    assert previous['source_sha256'] == frozen
    assert C.sha(original / 'plans.json') == previous['schedule_plan_sha256'] == q['original_plan_sha256']
    stop = R.read(handover)
    assert stop['original_pid'] == previous['pid'] and stop['original_worker_stopped']
    plans = R.read(original / 'plans.json')
    imported = R.read(original / 'perarm.json')
    assert stop['completed_rows_sha256'] == C.sha(original / 'perarm.json')
    assert len(imported) == stop['imported_completed_arms']
    assert len({(r['block'], r['arm']) for r in imported}) == len(imported)
    runtime_sources = runtime_hashes()
    C.setup_backend()
    out.mkdir(parents=True)
    started = time.monotonic()
    for name in ('plans.json', 'config.json'):
        shutil.copyfile(original / name, out / name)
    shutil.copytree(original / 'source', out / 'source')
    for name, digest in frozen.items():
        assert C.sha(out / 'source' / name) == digest
    for name in runtime_sources:
        destination = out / 'runtime_source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    rows = []
    migration = {'original_run': original.relative_to(ROOT).as_posix(), 'execution_amendment': AMENDMENT,
                 'qualification_sha256': C.sha(qualification), 'handover_sha256': C.sha(handover),
                 'imported_arms': [], 'partial_attempts_excluded': stop['partial_attempts_excluded']}
    for row in imported:
        verify_import(original, row)
        block, arm = row['block'], row['arm']
        expected = plans[str(block)]
        assert arm in expected['arm_order'] and row['initialization_seed'] == expected['initialization_seed']
        assert row['schedule_seed'] == expected['schedule_seed']
        relative = Path(f'block{block:02d}') / arm
        original_dir = original / relative
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        hashes = {p.relative_to(original_dir).as_posix(): C.sha(p) for p in original_dir.rglob('*') if p.is_file()}
        shutil.copytree(original_dir, destination)
        assert all(C.sha(destination / name) == digest for name, digest in hashes.items())
        copied = dict(row)
        copied['execution_backend'] = 'eager_imported'
        copied['provenance'] = {'original_run': original.relative_to(ROOT).as_posix(),
                                'original_arm_directory': relative.as_posix(), 'imported_artifacts_sha256': hashes}
        rows.append(copied)
        migration['imported_arms'].append({'block': block, 'arm': arm, 'artifacts_sha256': hashes})
    C.write(out / 'migration.json', migration)
    C.write(out / 'perarm.json', rows)
    R.report(out, {'status': 'RUNNING'}, rows)
    train = C.region_bank(32, 512, 10002, 'cuda')
    evaluation = {size: C.region_bank(size, 32, seed) for size, seed in R.EVAL_SEEDS.items()}
    data_hashes = {'train': C.tensor_hash(train), **{f'evaluation{size}': C.tensor_hash(data) for size, data in evaluation.items()}}
    assert data_hashes == previous['data_sha256']
    assert data_hashes['train'] == q['training_data_sha256']
    manifest = {**previous, 'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'),
                'started_utc': C.now(), 'command': [sys.executable, *sys.argv],
                'execution_amendment': AMENDMENT, 'scientific_protocol': R.PROTOCOL,
                'original_run': original.relative_to(ROOT).as_posix(),
                'runtime_source_sha256': runtime_sources, 'qualification_sha256': C.sha(qualification),
                'handover_sha256': C.sha(handover), 'imported_completed_arms': len(rows),
                'numerical_qualification': 'all6 block00 300 updates parameters+Adam bitwise exact',
                'qualification_scope': 'finite block00 replay, not universal guarantee',
                'inference_evaluation_backend': 'unchanged eager'}
    C.write(out / 'manifest.json', manifest)

    def progress(value):
        C.write(out / 'status.json', {'status': 'RUNNING', 'pid': os.getpid(),
                                    'elapsed_seconds': time.monotonic()-started, 'updated_utc': C.now(),
                                    'execution_amendment': AMENDMENT, 'expected_arms': 96,
                                    'completed_arms': len(rows), 'imported_completed_arms': len(imported), **value})
        print(json.dumps(value), flush=True)

    result = None
    try:
        completed = {(r['block'], r['arm']) for r in rows}
        for block, plan in plans.items():
            block_id = int(block)
            for arm in plan['arm_order']:
                if (block_id, arm) in completed:
                    continue
                folder = out / f'block{block_id:02d}' / arm
                folder.mkdir(parents=True)
                model = R.model_for(arm, plan['initialization_seed'])
                optimizer = C.optimizer_for(model)
                initial_hash = C.tensor_hash(model.state_dict())
                torch.save(R.checkpoint_payload(arm, model, optimizer, plan['initialization_seed'], 0), folder / 'initial.pt')
                progress({'block': block_id, 'arm': arm, 'phase': 'capture', 'completed_updates': 0})
                trained = G.train_graph_updates(model, optimizer, train, plan['batch_indices'],
                    progress=lambda event: progress({'block': block_id, 'arm': arm, **event}))
                curve = trained['curve']
                assert len(curve) == 300 and curve[-1]['update'] == 300
                assert all(int(s['step'].item()) == 300 for s in optimizer.state.values())
                C.write(folder / 'training_curve.json', curve)
                C.write(folder / 'runtime.json', {k: v for k, v in trained.items() if k != 'curve'})
                checkpoint = folder / 'final_u300.pt'
                torch.save(R.checkpoint_payload(arm, model, optimizer, plan['initialization_seed'], 300), checkpoint)
                final_hash = C.tensor_hash(model.state_dict())
                progress({'block': block_id, 'arm': arm, 'phase': 'evaluation', 'completed_updates': 300})
                measured = C.evaluate(model, evaluation, folder / 'evaluation', arm, lambda: None)
                timing = R.systems(model, evaluation)
                assert C.tensor_hash(model.state_dict()) == final_hash
                C.write(folder / 'systems.json', timing)
                rows.append({'block': block_id, 'initialization_seed': plan['initialization_seed'],
                             'schedule_seed': plan['schedule_seed'], 'arm': arm, 'metadata': model.metadata(),
                             'initial_parameter_sha256': initial_hash, 'final_parameter_sha256': final_hash,
                             'checkpoint_sha256': C.sha(checkpoint), 'checkpoint': checkpoint.relative_to(out).as_posix(),
                             'training_curve': (folder / 'training_curve.json').relative_to(out).as_posix(),
                             'training_seconds': sum(r['seconds'] for r in curve),
                             'capture_setup_seconds': trained['capture_setup_seconds'], 'systems': timing,
                             'evaluation_summary': (folder / 'evaluation' / f'{arm}_summary.json').relative_to(out).as_posix(),
                             'phenotype_pass': bool(measured['phenotype_gate']['pass']), 'evaluation_compact': C.compact(measured),
                             'execution_backend': 'cuda_graph_k8'})
                completed.add((block_id, arm))
                C.write(out / 'perarm.json', rows)
                R.report(out, {'status': 'RUNNING'}, rows)
                progress({'block': block_id, 'arm': arm, 'phase': 'arm_complete', 'completed_updates': 300})
                del model, optimizer, measured, curve, trained
                gc.collect()
                torch.cuda.empty_cache()
        assert len(rows) == len(completed) == 96
        assert R.source_hashes() == frozen and runtime_hashes() == runtime_sources
        assert C.sha(out / 'plans.json') == previous['schedule_plan_sha256']
        assert C.tensor_hash(train) == data_hashes['train']
        assert all(C.tensor_hash(data) == data_hashes[f'evaluation{size}'] for size, data in evaluation.items())
        result = R.aggregate(rows)
    except Exception as error:
        C.write(out / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        result = {'protocol': R.PROTOCOL, 'status': 'ERROR', 'completed_arms': len(rows),
                  'expected_arms': 96, 'error': repr(error), 'scientific_verdict': 'INCOMPLETE'}
    result.update({'execution_amendment': AMENDMENT, 'original_run': original.relative_to(ROOT).as_posix(),
                   'imported_completed_arms': len(imported), 'new_completed_arms': len(rows)-len(imported),
                   'finished_utc': C.now(), 'elapsed_seconds': time.monotonic()-started,
                   'runtime_limit_enforced': False, 'qualification_scope': manifest['qualification_scope']})
    C.write(out / 'summary.json', result)
    C.write(out / 'status.json', {**result, 'pid': os.getpid()})
    R.report(out, result, rows)
    print(json.dumps(result), flush=True)
    if result['status'] != 'COMPLETE':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--qualification', required=True)
    parser.add_argument('--handover', required=True)
    args = parser.parse_args()
    paths = {key: (ROOT / getattr(args, key)).resolve() for key in ('original', 'out', 'qualification', 'handover')}
    assert all(paths[k].is_relative_to(ROOT / ('analyses' if k == 'qualification' else 'runs')) for k in paths)
    run(**paths)


if __name__ == '__main__':
    main()
