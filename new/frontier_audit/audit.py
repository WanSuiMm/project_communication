"""One fixed-checkpoint, zero-training Streaming seed4 behavioral audit."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for relative in ('new/nca_inertial_wind_tunnel', 'new/workspace_revision',
                 'new/short_bptt_phase2', 'new/streaming_carry'):
    sys.path.insert(0, str(ROOT / relative))
from stream_cells import StreamingCell
from run_revision import now, score, sha, tensor_hash, write
from tasks import bank
from metrics import analyze_trace
from report import build_summary, save_report

spec = importlib.util.spec_from_file_location('frontier_phase2', ROOT / 'new/short_bptt_phase2/run.py')
phase2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase2)
spec = importlib.util.spec_from_file_location('frontier_replay', ROOT / 'new/stream_path_audit/audit.py')
replay_module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT / 'new/stream_path_audit'))
spec.loader.exec_module(replay_module)

TIMES = tuple(range(8, 257, 8))
CAP = 300
DEADLINE = float('inf')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def budget():
    if time.monotonic() > DEADLINE:
        raise TimeoutError('Frozen300-second behavioral audit cap')


@torch.no_grad()
def trace(model, data, size):
    shape = (257, data['x'].shape[0], size, size)
    correct = np.empty(shape, dtype=bool)
    original = np.empty(shape, dtype=bool)
    flipped_correct = np.empty(shape, dtype=bool)
    margins = np.empty(shape, dtype=np.float32)
    a, b = model.initial(data['x']), model.initial(data['x_flip'])
    records = {}
    for t in range(257):
        budget()
        if t:
            a, b = model.step(a, data['x']), model.step(b, data['x_flip'])
        assert all(bool(torch.isfinite(v).all()) for v in (*a, *b)), 'Nonfinite state'
        logits, flipped = model.logits(a), model.logits(b)
        assert bool(torch.isfinite(logits).all()) and bool(torch.isfinite(flipped).all())
        good_a = (logits >= 0) == (data['y'] >= .5)
        good_b = (flipped >= 0) == (data['y_flip'] >= .5)
        both = good_a & good_b
        margin = torch.minimum((2*data['y']-1)*logits, (2*data['y_flip']-1)*flipped)
        assert bool(torch.isfinite(margin).all())
        original[t] = good_a[:, 0].cpu().numpy()
        flipped_correct[t] = good_b[:, 0].cpu().numpy()
        correct[t] = both[:, 0].cpu().numpy()
        margins[t] = margin[:, 0].cpu().numpy()
        outside = data['changed'].bool() & (data['distance'] > 2*t)
        assert not bool((both & outside).any()), ('Paired correctness outside light cone', size, t)
        delta = float(((logits-flipped).abs()*outside).max())
        assert delta <= 1e-6, ('Source-flip difference outside light cone', size, t)
        if t not in TIMES:
            continue
        masks = phase2.selections(data, 8, t)
        records[str(t)] = {
            'original': score(logits, data['y'], data['mask']),
            'flipped': score(flipped, data['y_flip'], data['mask']),
            'paired': phase2.paired(logits, flipped, data, data['changed']),
            'bands': {name: phase2.paired(logits, flipped, data, mask)
                      for name, mask in masks.items()},
            'outside_forward_lightcone_max_logit_difference': delta,
        }
    return {'correct': correct, 'original_correct': original,
            'flipped_correct': flipped_correct, 'margin': margins}, records


def main():
    global DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    assert out.is_relative_to(ROOT / 'runs'), 'New outputs belong under project runs/'
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    DEADLINE = start + CAP
    timer = threading.Timer(CAP+30, lambda: (
        write(out/'status.json', {'status': 'WATCHDOG_TIMEOUT', 'finished_utc': now()}), os._exit(124)))
    timer.daemon = True
    timer.start()
    try:
        torch.set_num_threads(2)
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        historical = ROOT / 'runs/streaming_carry_20261002_init2345'
        checkpoint_path = historical / 'stream_K8_seed4.pt'
        file_hash = sha(checkpoint_path)
        old = read(historical/'stream_K8_seed4.json')
        publication = read(ROOT/'STREAMING_CARRY_PUBLICATION_MANIFEST.json')
        old_manifest = read(historical/'manifest.json')
        assert file_hash == publication['checkpoint_sha256'][checkpoint_path.name]
        assert sha(historical/'stream_K8_seed4.json') == sha(ROOT/'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json')
        hashes = dict(publication['source_sha256'])
        assert len(hashes) == 35
        for name, expected in hashes.items():
            assert sha(ROOT/name) == sha(historical/'source'/name) == expected, name
        for name in ('PROTOCOL.md', 'audit.py', 'metrics.py', 'check_metrics.py', 'report.py', 'validate.py'):
            relative = 'new/frontier_audit/'+name
            hashes[relative] = sha(ROOT/relative)
        hashes['STREAMING_CARRY_PUBLICATION_MANIFEST.json'] = sha(ROOT/'STREAMING_CARRY_PUBLICATION_MANIFEST.json')
        hashes['evidence/streaming_carry_init2345/raw/stream_K8_seed4.json'] = sha(ROOT/'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json')
        # compare_tree is reused solely as the frozen replay checker.
        hashes['new/stream_path_audit/audit.py'] = sha(ROOT/'new/stream_path_audit/audit.py')
        hashes['new/stream_path_audit/operators.py'] = sha(ROOT/'new/stream_path_audit/operators.py')
        data_cpu = {size: bank(size, 32, 40000+size) for size in (32, 64)}
        data_hashes = {str(size): tensor_hash(value) for size, value in data_cpu.items()}
        assert data_hashes == old_manifest['evaluation_data_sha256'] == old_manifest['executed_eval_data_sha256']
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert checkpoint['variant'] == 'stream' and checkpoint['seed'] == 4 and checkpoint['completed_updates'] == 300
        model = StreamingCell().eval()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        parameter_hash = tensor_hash(model.state_dict())
        assert parameter_hash == old['final_parameter_sha256']
        model.cuda()
        manifest = {'protocol': 'original_seed4_frontier_behavior_v1',
                    'training': False, 'model_seed': 4, 'model_replications': 1,
                    'started_utc': now(), 'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'),
                    'gpu': torch.cuda.get_device_name(0), 'command': [sys.executable, *sys.argv],
                    'maximum_seconds': CAP, 'steps': 256, 'trace_times': 'every integer0..256',
                    'sample_times': list(TIMES), 'maps_per_size': 32, 'sizes': [32, 64],
                    'checkpoint_file_sha256': file_hash, 'checkpoint_parameter_sha256': parameter_hash,
                    'historical_reference_sha256': sha(historical/'stream_K8_seed4.json'),
                    'evaluation_data_sha256': data_hashes, 'source_sha256': hashes,
                    'git_review_base': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    'torch': torch.__version__, 'numpy': np.__version__, 'backend': old_manifest['backend']}
        write(out/'manifest.json', manifest)
        for name in hashes:
            target = out/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        replay = {'status': 'PENDING', 'size_horizon_records': 0,
                  'integer_leaves': 0, 'float_leaves': 0, 'maximum_absolute_error': 0.}
        traces, records, behaviors = {}, {}, {}
        for size in (32, 64):
            write(out/'status.json', {'status': 'RUNNING', 'phase': 'trace', 'size': size,
                                     'pid': os.getpid(), 'updated_utc': now()})
            data = {key: value.cuda() for key, value in data_cpu[size].items()}
            traces[str(size)], records[str(size)] = trace(model, data, size)
            for t in ('64', '128', '256'):
                replay_module.compare_tree(old['evaluation'][str(size)][t], records[str(size)][t], str(size)+'/'+t, replay)
                replay['size_horizon_records'] += 1
            print(json.dumps({'size': size, 'trace': 'COMPLETE', 'replay_records':3}), flush=True)
        assert replay['size_horizon_records'] == 6
        replay['status'] = 'PASS'
        write(out/'replay_validation.json', replay)
        write(out/'evaluations.json', records)
        assert tensor_hash(model.state_dict()) == parameter_hash and sha(checkpoint_path) == file_hash
        for size in (32, 64):
            budget()
            write(out/'status.json', {'status': 'RUNNING', 'phase': 'behavior_analysis',
                                     'size': size, 'pid': os.getpid(), 'updated_utc': now()})
            values = traces[str(size)]
            np.savez_compressed(out/f'trace_size{size}.npz', **values)
            cpu = {key: value.numpy() for key, value in data_cpu[size].items()}
            behaviors[str(size)] = analyze_trace(values['correct'], values['margin'], cpu, TIMES)
            write(out/f'behavior_size{size}.json', behaviors[str(size)])
        budget()
        summary = build_summary(traces, records, behaviors, replay, time.monotonic()-start, data_cpu)
        write(out/'summary.json', summary)
        save_report(out, summary, traces, data_cpu)
        budget()
        output_hashes = {path.name: sha(path) for path in sorted(out.iterdir())
                         if path.is_file() and path.name not in ('status.json','output_manifest.json')}
        write(out/'output_manifest.json', {'sha256': output_hashes,
                                          'parameters_and_checkpoint_unchanged': True})
        write(out/'status.json', {'status': 'COMPLETE', 'training': False, 'model_seed': 4,
                                 'finished_utc': now(), 'elapsed_seconds': time.monotonic()-start})
        print(json.dumps({'status':'COMPLETE','elapsed_seconds':time.monotonic()-start,
                          'replay':replay,'summary':out.name}), flush=True)
    except Exception as error:
        write(out/'error.json', {'error':repr(error), 'traceback':traceback.format_exc()})
        write(out/'status.json', {'status':'TIME_BUDGET' if isinstance(error,TimeoutError) else 'ERROR',
                                 'pid':os.getpid(),'finished_utc':now(),'elapsed_seconds':time.monotonic()-start})
        raise
    finally:
        timer.cancel()


if __name__ == '__main__':
    main()
