"""Zero-training historical seed4 credit-locality formation audit."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
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
sys.path.insert(0, str(ROOT/'new/continuation_interface'))
spec = importlib.util.spec_from_file_location('credit_formation_base', ROOT/'new/continuation_interface/run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
import gradients as original_gradients
from credit_metrics import summarize_pair, formation_verdict

PROTOCOL = 'credit_locality_formation_v1'
UPDATES = (140, 145, 175, 180, 190, 195, 200)
OLD = ROOT/'runs/transition_20261003_seed4_dense01'
BANK_ROOT = ROOT/'evidence/continuation_interface_20261004'
BATCHES = [list(range(i, i+8)) for i in range(0, 32, 8)]
sha, read, write = base.sha, base.read, base.write


def sources():
    bindings = read(ROOT/'CONTINUATION_INTERFACE_PUBLICATION_MANIFEST.json')['source_sha256']
    for rel, digest in bindings.items():
        assert sha(ROOT/rel) == digest, f'Frozen source drift: {rel}'
    for p in (ROOT/'new/credit_formation').glob('*'):
        if p.suffix in ('.py', '.md'):
            bindings[p.relative_to(ROOT).as_posix()] = sha(p)
    launch = ROOT/'tools/launch_credit_formation.ps1'
    bindings[launch.relative_to(ROOT).as_posix()] = sha(launch)
    return dict(sorted(bindings.items()))


def catalog():
    publication = read(ROOT/'TRANSITION_PUBLICATION_MANIFEST.json')
    expected = publication['source_bindings']['run_input_artifact_sha256']['run/training.json']
    assert sha(OLD/'training.json') == expected, 'Historical training record drift'
    assert read(OLD/'historical_replay.json')['status'] == 'PASS'
    assert read(OLD/'summary.json')['execution_complete'] is True
    rows = {}
    for update in UPDATES:
        row = read(OLD/'training.json')['checkpoints'][str(update)].copy()
        row.update(update=update, path=(OLD/row['path']).relative_to(ROOT).as_posix())
        assert sha(ROOT/row['path']) == row['sha256'], update
        rows[update] = row
    return rows


def load_model(row):
    payload = torch.load(ROOT/row['path'], map_location='cpu', weights_only=True)
    assert payload['completed_updates'] == row['update'] and payload['seed'] == 4
    model = base.StreamingCell().eval()
    model.load_state_dict(payload['state_dict'])
    assert base.tensor_hash(model.state_dict()) == row['parameter_sha256']
    return model


def banks():
    published = read(ROOT/'CONTINUATION_INTERFACE_PUBLICATION_MANIFEST.json')['published_sha256']
    configuration = read(BANK_ROOT/'config.json')
    data, records = {}, {}
    for size in (32, 64):
        p = BANK_ROOT/'banks'/f'test{size}.npz'
        rel = p.relative_to(ROOT).as_posix()
        assert sha(p) == published[rel], rel
        with np.load(p, allow_pickle=False) as arrays:
            data[size] = {k: torch.from_numpy(arrays[k].copy()) for k in arrays.files}
        assert len(data[size]['x']) == 32
        expected = configuration['banks'][f'test{size}']
        assert base.tensor_hash(data[size]) == expected['data_sha256']
        records[size] = {**expected, 'path': rel, 'file_sha256': sha(p)}
    return data, records


def state_digest(state):
    h = hashlib.sha256()
    for v in state:
        a = v.detach().cpu().contiguous().numpy()
        if not np.isfinite(a).all():
            raise FloatingPointError('Nonfinite forward state')
        h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def backward_record(model, batch, horizon, steps=64):
    helper, _ = original_gradients._historical()
    original_initial, original_step = model.initial, model.step
    digests = []
    def initial(x):
        state = original_initial(x); digests.append(state_digest(state)); return state
    def step(state, x):
        result = original_step(state, x); digests.append(state_digest(result)); return result
    model.initial, model.step = initial, step
    try:
        loss, state, cadence = helper.backward_trajectory(model, batch, horizon, steps=steps, loss_every=8)
        vector, names, shapes = original_gradients._vector(model)
        assert len(digests) == steps+1
        return vector, names, shapes, float(loss.cpu()), digests, cadence
    finally:
        model.initial, model.step = original_initial, original_step
        model.zero_grad(set_to_none=True)


def gradients_at(model, data, out, notify):
    before = original_gradients._hash(model)
    model.cuda().eval()
    records, vectors = [], []
    try:
        for bi, ids in enumerate(BATCHES):
            notify(bi)
            batch = {k: data[k][ids].cuda() for k in ('x', 'y', 'mask')}
            a, names, shapes, la, da, ca = backward_record(model, batch, 8)
            b, nb, sb, lb, db, cb = backward_record(model, batch, 64)
            assert names == nb and shapes == sb
            assert da == db, f'Forward trajectory mismatch on batch{bi}'
            assert abs(la-lb) <= 1e-6
            assert ca['backward_calls'] == 8 and cb['backward_calls'] == 1
            metrics = summarize_pair(a, b, names, shapes)
            assert metrics['overall']['finite']
            np.savez_compressed(out/f'batch{bi:02}.npz', gradient_k8=a, gradient_full64=b,
                                indices=np.asarray(ids), parameter_names=np.asarray(names))
            records.append({'batch': bi, 'indices': ids, 'loss8': la, 'loss64': lb,
                            'trajectory_sha256_by_step': da, 'exact_trajectory_match': True,
                            'metrics': metrics})
            vectors.append((a, b))
            del batch
    finally:
        model.cpu(); model.zero_grad(set_to_none=True)
    after = original_gradients._hash(model)
    assert before == after, 'Frozen parameters changed'
    a, b = (np.mean(np.stack([pair[i] for pair in vectors]), axis=0) for i in (0, 1))
    np.savez_compressed(out/'mean_gradients.npz', mean_gradient_k8=a,
                        mean_gradient_full64=b, parameter_names=np.asarray(names))
    result = {'training_updates': 0, 'parameters_unchanged': True,
              'parameter_sha256_before': before, 'parameter_sha256_after': after,
              'parameter_names': names, 'parameter_shapes': shapes,
              'batches': records, 'mean_metrics': summarize_pair(a, b, names, shapes)}
    write(out/'metadata.json', result)
    return result


def save_figures(out, credits, phenotypes):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    updates = list(UPDATES)
    fig = plt.figure(figsize=(11, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax = fig.add_subplot(gs[0, :]); right = ax.twinx()
    for metric, style in (('preservation', '-o'), ('sustained_progress', '-s')):
        ax.plot(updates, [phenotypes[u]['32'][metric] for u in updates], style, label=metric)
    right.plot(updates, [credits[u]['overall']['cosine'] for u in updates], '--^', color='tab:red', label='overall cosine')
    ax.set_ylim(-.03, 1.03); ax.set_ylabel('Size32 continuation rate'); right.set_ylabel('Gradient cosine')
    ax.set_title('Historical seed4: continuation and credit synchrony')
    lines = ax.get_lines()+right.get_lines(); ax.legend(lines, [v.get_label() for v in lines], loc='best')
    for row, scope in enumerate(('E', 'F', 'Q', 'R')):
        p = fig.add_subplot(gs[1+row//2, row%2])
        for metric, style in (('cosine', '-o'), ('C_parallel', '--s'), ('C_miss', ':^')):
            p.plot(updates, [credits[u]['groups'][scope][metric] for u in updates], style, label=metric)
        p.set_title(scope); p.legend(fontsize=8); p.set_ylabel('Metric (projection is unbounded)')
    for p in fig.axes:
        p.set_xticks(updates); p.set_xlabel('Training update'); p.grid(alpha=.2)
    fig.suptitle('One selected trajectory; no training updates; association only', fontsize=11)
    fig.savefig(out/'credit_formation.png', dpi=160); fig.savefig(out/'credit_formation.pdf'); plt.close(fig)


def run(out, qualification):
    base.setup()
    binding = sources(); rows = catalog(); data, bank_records = banks()
    q = read(qualification)
    assert q['status'] == 'PASS' and q['protocol'] == PROTOCOL
    assert q['source_sha256'] == binding and q['checkpoints'] == base.ready(rows)
    assert q['banks'] == base.ready(bank_records)
    assert out.parent == ROOT/'runs' and not out.exists(), 'Fresh run directory required'
    out.mkdir(parents=True)
    started = time.monotonic()
    manifest = {'protocol': PROTOCOL, 'status': 'RUNNING', 'training': False,
                'training_updates': 0, 'runtime_limit_enforced': False,
                'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'),
                'command': sys.argv, 'started_utc': base.now(),
                'source_sha256': binding, 'checkpoints': rows, 'banks': bank_records,
                'batches': BATCHES, 'torch_version': str(torch.__version__),
                'numpy_version': np.__version__, 'gpu': torch.cuda.get_device_name(0),
                'backend': {'cudnn_benchmark': False, 'cudnn_deterministic': False,
                            'cudnn_tf32': True, 'matmul_tf32': False}}
    write(out/'manifest.json', manifest)
    for rel in binding:
        p = out/'source'/rel; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/rel, p)
    (out/'banks').mkdir()
    for row in bank_records.values(): shutil.copyfile(ROOT/row['path'], out/'banks'/Path(row['path']).name)
    completed = 0
    def status(phase, **extra):
        write(out/'status.json', {'protocol': PROTOCOL, 'status': 'RUNNING', 'phase': phase,
              'pid': os.getpid(), 'completed_checkpoints': completed, 'expected_checkpoints': len(UPDATES),
              'updated_utc': base.now(), **extra})
    credit_rows, phenotype_rows, full_flags, batch_credits = {}, {}, {}, {}
    try:
        for update in UPDATES:
            model = load_model(rows[update]); before = original_gradients._hash(model)
            folder = out/f'u{update:03}'; folder.mkdir()
            grad_folder = folder/'gradients'; grad_folder.mkdir()
            result = gradients_at(model, data[32], grad_folder,
                                  lambda bi: status('gradients', update=update, batch=bi))
            credit_rows[update] = result['mean_metrics']
            batch_credits[update] = [v['metrics'] for v in result['batches']]
            traces, endpoints, handoffs = {}, {}, {}
            for size in (32, 64):
                status('continuation', update=update, size=size)
                trace, scores, snapshot, logits = base.paired_trace(model, data[size])
                traces[size], endpoints[size] = trace, scores
                handoffs[str(size)] = base.summarize_handoff(trace['correct'][64:], trace['correct'][64],
                                      data[size]['changed'].numpy(), data[size]['distance'].numpy())
                np.savez_compressed(folder/f'size{size}_trace.npz', **trace)
                np.savez_compressed(folder/f'size{size}_logits.npz', **{'T'+t: v for t, v in logits.items()})
                del snapshot
            ph = base.summarize_from_traces(traces, endpoints, data)
            frontier = ph.pop('_matched_frontier_rows'); ph['phenotype_gate'] = base.predicate(ph)
            base._write_frontier_csv(folder/'frontier.csv', frontier)
            write(folder/'phenotype.json', ph); write(folder/'continuation.json', handoffs)
            phenotype_rows[update] = {size: {metric: h['bands']['all_changed'][metric]['pooled']['value']
                                    for metric in ('preservation', 'sustained_progress')}
                                    for size, h in handoffs.items()}
            full_flags[update] = ph['phenotype_gate']['pass']
            assert original_gradients._hash(model) == before
            assert base.tensor_hash(model.state_dict()) == rows[update]['parameter_sha256']
            completed += 1; status('checkpoint_complete', update=update)
            del model, traces, endpoints
        verdict = formation_verdict(credit_rows, phenotype_rows)
        summary = {'protocol': PROTOCOL, 'status': 'COMPLETE', 'training': False,
                   'training_updates': 0, 'completed_checkpoints': completed,
                   'updates': list(UPDATES), 'credit_metrics': credit_rows,
                   'batch_credit_metrics': batch_credits, 'continuation': phenotype_rows,
                   'native_full_flags': full_flags, 'formation_screen': verdict,
                   'primary_size': 32, 'secondary_size': 64,
                   'claim_boundary': 'Within-trajectory synchrony only; no causal or population conclusion.'}
        with (out/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
            fields = ['update', 'scope', 'aggregation', 'cosine', 'C_parallel', 'C_miss', 'norm_ratio', 'norm8', 'norm64']
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for update in UPDATES:
                for tag, metrics in [('mean_vector', credit_rows[update])]+[(f'batch{i}', v) for i, v in enumerate(batch_credits[update])]:
                    for scope, values in [('overall', metrics['overall'])]+list(metrics['groups'].items()):
                        w.writerow({'update': update, 'scope': scope, 'aggregation': tag,
                                    **{k: values[k] for k in fields[3:]}})
        save_figures(out, credit_rows, phenotype_rows)
        summary['elapsed_seconds'] = round(time.monotonic()-started, 3)
        write(out/'summary.json', summary)
        table = ['# Credit-locality formation audit', '', 'Status: COMPLETE. No model training or optimizer updates.',
                 f"Formation screen: {verdict['verdict']} (synchrony only).", '',
                 '| Update | P32 | G32 | P64 | G64 | Cosine | C_parallel | C_miss | Full |',
                 '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        def fmt(x): return 'null' if x is None else f'{x:.4f}'
        for update in UPDATES:
            p = phenotype_rows[update]; c = credit_rows[update]['overall']
            v = [update, fmt(p['32']['preservation']), fmt(p['32']['sustained_progress']),
                 fmt(p['64']['preservation']), fmt(p['64']['sustained_progress']),
                 fmt(c['cosine']), fmt(c['C_parallel']), fmt(c['C_miss']), full_flags[update]]
            table.append('| '+' | '.join(map(str, v))+' |')
        table += ['', 'Read summary.json and metrics.csv first. Per-batch gradients, traces, cohort denominators,',
                  'existing Full flags and figure are retained. An absent behavior contrast is unqualified,',
                  'not a credit null. Synchrony is not proof of state-to-credit causality.']
        (out/'RESULTS.md').write_text('\n'.join(table)+'\n', encoding='utf-8')
        manifest.update(status='COMPLETE', finished_utc=base.now(), elapsed_seconds=summary['elapsed_seconds'])
        write(out/'manifest.json', manifest)
        write(out/'status.json', {'status': 'COMPLETE', 'protocol': PROTOCOL, 'pid': os.getpid(),
              'completed_checkpoints': completed, 'elapsed_seconds': summary['elapsed_seconds'], 'updated_utc': base.now()})
    except BaseException as error:
        write(out/'status.json', {'status': 'ERROR', 'protocol': PROTOCOL, 'pid': os.getpid(),
              'completed_checkpoints': completed, 'error': repr(error), 'updated_utc': base.now()})
        traceback.print_exc(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True); p.add_argument('--qualification', required=True)
    a = p.parse_args()
    run((ROOT/a.out).resolve(), (ROOT/a.qualification).resolve())
