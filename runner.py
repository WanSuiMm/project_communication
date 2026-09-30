"""Bounded local qualification. Invoke as python -m project_ReactionTransport.runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

from .data import make_batch
from .models import GateModel


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def forward_batch(model, batch):
    return model(batch['x'], batch['target_index'],
                 oracle_edges=batch['oracle_edges'] if model.variant == 'oracle' else None)


@torch.no_grad()
def evaluate(model, dim, gate, distance, long_size, count, batch_size, device):
    correct = source_correct = detail_correct = total = 0
    by_axis = []
    for axis in range(dim):
        ac = sc = dc = n = 0
        for start in range(0, count, batch_size):
            size = min(batch_size, count - start)
            batch = make_batch(dim, gate, size, distance, 900000 + axis * 10000 + start,
                               device=device, long_size=long_size, axis=axis)
            logits = forward_batch(model, batch)
            pred = logits.argmax(-1)
            ac += (pred == batch['labels']).sum().item()
            if gate == 'C':
                # Marginal probabilities, not bits of the joint argmax.
                p = logits.softmax(-1).reshape(size, 2, 2)
                sp, dp = p.sum(2).argmax(-1), p.sum(1).argmax(-1)
            else:
                sp, dp = pred, torch.zeros_like(pred)
            sc += (sp == batch['source_labels']).sum().item()
            dc += (dp == batch['detail_labels']).sum().item() if gate == 'C' else 0
            n += size
        by_axis.append({'axis': axis, 'n': n, 'accuracy': ac / n,
                        'source_accuracy': sc / n, 'detail_accuracy': dc / n if gate == 'C' else None})
        correct += ac
        source_correct += sc
        detail_correct += dc
        total += n
    return {'n': total, 'accuracy': correct / total, 'source_accuracy': source_correct / total,
            'detail_accuracy': detail_correct / total if gate == 'C' else None, 'by_axis': by_axis}


@torch.no_grad()
def latency(model, dim, gate, device, iterations=7):
    batch = make_batch(dim, gate, 1, 128, 8181, device=device, long_size=140, axis=dim - 1)
    for _ in range(2):
        forward_batch(model, batch)
    sync(device)
    values = []
    for _ in range(iterations):
        start = time.perf_counter()
        forward_batch(model, batch)
        sync(device)
        values.append(1000 * (time.perf_counter() - start))
    return {'median_ms': statistics.median(values), 'repetitions': iterations,
            'batch_size': 1, 'shape': list(batch['x'].shape), 'includes_python_and_device_sync': True}


def trial(cfg, out, dim, gate, variant, seed, message_width, deadline):
    key = f'{gate}_{dim}d_{variant}_r{message_width}_seed{seed}'
    trial_dir = out / key
    trial_dir.mkdir()
    torch.manual_seed(seed)
    random.seed(seed)
    model = GateModel(dim, variant, width=cfg['width'], message_width=message_width,
                      groups=4, steps=cfg['recurrent_steps'], classes=4 if gate == 'C' else 2).to(cfg['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    result = {'key': key, 'gate': gate, 'dim': dim, 'variant': variant, 'seed': seed,
              'message_width': message_width, 'parameters': sum(p.numel() for p in model.parameters()),
              'status': 'running', 'history': []}
    dump(trial_dir / 'result.json', result)
    if cfg['device'] == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    steps_done = 0
    for step in range(cfg['train_steps']):
        if time.monotonic() >= deadline:
            break
        distance = cfg['train_distances'][step % len(cfg['train_distances'])]
        batch = make_batch(dim, gate, cfg['batch_size'], distance, seed * 100000 + step,
                           device=cfg['device'], long_size=28,
                           axis=(step // len(cfg['train_distances'])) % dim)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_batch(model, batch)
        loss = F.cross_entropy(logits, batch['labels'])
        if not torch.isfinite(loss):
            result['status'] = 'nonfinite_loss'
            break
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
        if not torch.isfinite(grad_norm):
            result['status'] = 'nonfinite_gradient'
            break
        optimizer.step()
        steps_done = step + 1
        if step == 0 or (step + 1) % 40 == 0 or step + 1 == cfg['train_steps']:
            item = {'step': steps_done, 'loss': loss.item(),
                    'batch_accuracy': (logits.argmax(-1) == batch['labels']).float().mean().item(),
                    'unclipped_grad_norm': grad_norm.item(), 'elapsed_s': time.perf_counter() - start_time}
            result['history'].append(item)
            print(json.dumps({'trial': key, **item}), flush=True)
    sync(cfg['device'])
    result['training_seconds'] = time.perf_counter() - start_time
    result['steps_done'] = steps_done
    result['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20 if cfg['device'] == 'cuda' else None
    if result['status'] == 'running':
        result['status'] = 'completed' if steps_done == cfg['train_steps'] else 'time_budget_exhausted'
    if result['status'] == 'completed':
        model.eval()
        result['eval'] = {}
        for name, distance, length in [('train_shape_d16', 16, 28), ('large_shape_d16', 16, 140),
                                        ('d32', 32, 140), ('d64', 64, 140), ('d128', 128, 140)]:
            result['eval'][name] = evaluate(model, dim, gate, distance, length, cfg['eval_per_axis'],
                                            cfg['eval_batch_size'], cfg['device'])
        result['latency'] = latency(model, dim, gate, cfg['device'])
        if gate == 'A' and variant == 'nca':
            original = model.steps
            model.steps = 128
            result['latency_128_updates'] = latency(model, dim, gate, cfg['device'], iterations=3)
            model.steps = original
        torch.save(model.state_dict(), trial_dir / 'model.pt')
    dump(trial_dir / 'result.json', result)
    print(json.dumps({'finished': key, 'status': result['status'],
                      'accuracy_d128': result.get('eval', {}).get('d128', {}).get('accuracy')}), flush=True)
    del model, optimizer
    if cfg['device'] == 'cuda':
        torch.cuda.empty_cache()
    return result


def per_seed(rows, dim, gate, variant, width=None):
    return {r['seed']: r for r in rows if r['dim'] == dim and r['gate'] == gate
            and r['variant'] == variant and r['status'] == 'completed'
            and (width is None or r['message_width'] == width)}


def verdicts(cfg, rows):
    decisions = {}
    for dim in cfg['dimensions']:
        ds = {}
        rt = per_seed(rows, dim, 'A', 'learned', cfg['message_width'])
        nca, vit = per_seed(rows, dim, 'A', 'nca'), per_seed(rows, dim, 'A', 'vit')
        seeds = cfg['seeds']
        if not all(s in rt and s in nca and s in vit for s in seeds):
            ds['A'] = 'PENDING_OR_INCOMPLETE'
        elif not all(vit[s]['eval']['train_shape_d16']['accuracy'] >= .90 for s in seeds):
            ds['A'] = 'INCONCLUSIVE_POSITIVE_CONTROL'
        elif not all(rt[s]['eval']['train_shape_d16']['accuracy'] >= .90 for s in seeds):
            ds['A'] = 'NOT_QUALIFIED_FIT_WITHIN_BUDGET'
        else:
            passed = all(
                rt[s]['eval'][d]['accuracy'] >= .85
                and rt[s]['eval'][d]['accuracy'] - nca[s]['eval'][d]['accuracy'] >= .15
                for s in seeds for d in ('d32', 'd64', 'd128'))
            speed = all(rt[s]['latency']['median_ms'] < nca[s]['latency_128_updates']['median_ms'] for s in seeds)
            ds['A'] = 'PASS' if passed and speed else 'NOT_QUALIFIED_DISTANCE_OR_LATENCY'
        learned = per_seed(rows, dim, 'B', 'learned', cfg['message_width'])
        constant, oracle = per_seed(rows, dim, 'B', 'constant'), per_seed(rows, dim, 'B', 'oracle')
        if ds['A'] != 'PASS':
            ds['B'] = 'NOT_RUN_A_DID_NOT_PASS'
        elif not all(s in learned and s in constant and s in oracle for s in seeds):
            ds['B'] = 'PENDING_OR_INCOMPLETE'
        elif not all(oracle[s]['eval']['d64']['accuracy'] >= .85 for s in seeds):
            ds['B'] = 'INCONCLUSIVE_ORACLE_CONTROL'
        else:
            passed = all(learned[s]['eval'][d]['accuracy'] >= .80
                         and learned[s]['eval'][d]['accuracy'] - constant[s]['eval'][d]['accuracy'] >= .10
                         for s in seeds for d in ('d32', 'd64'))
            ds['B'] = 'PASS' if passed else 'NOT_QUALIFIED_SELECTIVITY'
        message, state = per_seed(rows, dim, 'C', 'message_full'), per_seed(rows, dim, 'C', 'state_full')
        if ds['B'] != 'PASS':
            ds['C'] = 'NOT_RUN_B_DID_NOT_PASS'
        elif not all(s in message and s in state for s in seeds):
            ds['C'] = 'PENDING_OR_INCOMPLETE'
        else:
            passed = all(message[s]['eval']['d64']['accuracy'] >= .75
                         and message[s]['eval']['d64']['detail_accuracy'] - state[s]['eval']['d64']['detail_accuracy'] >= .10
                         and message[s]['eval']['d64']['source_accuracy'] >= state[s]['eval']['d64']['source_accuracy'] - .05
                         for s in seeds)
            ds['C'] = 'PASS' if passed else 'NOT_QUALIFIED_DETAIL_ADVANTAGE'
        decisions[str(dim)] = ds
    return decisions


def write_summary(cfg, rows, out, status):
    decisions = verdicts(cfg, rows)
    summary = {'status': status, 'decisions': decisions, 'results': rows}
    dump(out / 'aggregate.json', summary)
    lines = ['# Reaction-Transport local qualification', '', f'Status: {status}', '',
             'Two training seeds are paired replicates; voxel counts and timing repetitions are not independent training runs.', '',
             '| Dimension | Gate A | Gate B | Gate C |', '|---|---|---|---|']
    for dim, d in decisions.items():
        lines.append(f"| {dim}D | {d['A']} | {d['B']} | {d['C']} |")
    lines += ['', '| Gate | Dim | Variant | r | Seed | Fit d16 | d32 | d64 | d128 | ms/image |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        ev = r.get('eval', {})
        values = [f"{ev[k]['accuracy']:.3f}" if k in ev else '-' for k in ('train_shape_d16', 'd32', 'd64', 'd128')]
        ms = f"{r['latency']['median_ms']:.2f}" if 'latency' in r else '-'
        lines.append(f"| {r['gate']} | {r['dim']} | {r['variant']} | {r['message_width']} | {r['seed']} | {' | '.join(values)} | {ms} |")
    lines += ['', 'Scope: thin 2D grids and narrow 3D volumes, axes rotated; this does not qualify arbitrary curved 3D topology, natural vision, or a fused-kernel performance claim.',
              'Failure within this frozen budget is a qualification failure, not an impossibility theorem. No threshold or hyperparameter rescue is performed.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return decisions


def smoke(cfg, out):
    rows = []
    for dim in (2, 3):
        for variant in ('nca', 'vit', 'constant', 'learned', 'message_full', 'state_full', 'oracle'):
            gate = 'C' if variant in ('message_full', 'state_full') else 'B'
            model = GateModel(dim, variant, width=cfg['width'], message_width=cfg['message_width'],
                              groups=4, steps=cfg['recurrent_steps'], classes=4 if gate == 'C' else 2).to(cfg['device'])
            batch = make_batch(dim, gate, 2, 8, 47, device=cfg['device'], long_size=20)
            start = time.perf_counter()
            logits = forward_batch(model, batch)
            loss = F.cross_entropy(logits, batch['labels'])
            loss.backward()
            sync(cfg['device'])
            assert torch.isfinite(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            edge_grad = sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
                            if 'edge' in n and p.grad is not None)
            if variant in ('learned', 'message_full', 'state_full'):
                assert edge_grad > 0, (dim, variant, 'missing edge gradient')
            row = {'dim': dim, 'variant': variant, 'shape': list(batch['x'].shape),
                   'loss': loss.item(), 'forward_backward_seconds': time.perf_counter() - start,
                   'edge_gradient_l1': edge_grad}
            rows.append(row)
            print(json.dumps(row), flush=True)
    dump(out / 'smoke.json', rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'qualify'], default='qualify')
    parser.add_argument('--out', required=True)
    parser.add_argument('--minutes', type=float, default=25)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cfg = dict(protocol='rt_2d3d_qualification_v1_1', dimensions=[2, 3], seeds=[1729, 2718],
               width=64, message_width=32, groups=4, recurrent_steps=8, train_steps=240,
               batch_size=8, eval_batch_size=8, eval_per_axis=64, train_distances=[4, 8, 12, 16],
               lr=0.003, weight_decay=0.0001, gradient_clip=1.0, budget_minutes=args.minutes,
               device='cuda' if torch.cuda.is_available() else 'cpu', mode=args.mode,
               rank_sweep=[4, 8, 16, 32, 64], rank_sweep_gate='B_after_A_B_C_pass')
    dump(out / 'config.json', cfg)
    source_dir = Path(__file__).resolve().parent
    provenance = {'started_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                  'host': platform.node(), 'python': platform.python_version(), 'torch': torch.__version__,
                  'device': torch.cuda.get_device_name(0) if cfg['device'] == 'cuda' else 'cpu',
                  'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in source_dir.glob('*.py')}}
    dump(out / 'launch_receipt.json', provenance)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    for pattern in ('*.py', '*.md'):
        for source in source_dir.glob(pattern):
            (snapshot / source.name).write_bytes(source.read_bytes())
    if args.mode == 'smoke':
        smoke(cfg, out)
        return
    deadline = time.monotonic() + 60 * args.minutes
    rows = []

    def execute(dim, gate, variant, seed, width=32):
        if time.monotonic() >= deadline:
            return False
        try:
            result = trial(cfg, out, dim, gate, variant, seed, width, deadline)
        except Exception as error:
            dump(out / 'failure.json', {'dim': dim, 'gate': gate, 'variant': variant, 'seed': seed,
                                        'error': repr(error)})
            raise
        rows.append(result)
        write_summary(cfg, rows, out, 'RUNNING')
        return result['status'] == 'completed'

    # Both dimensionalities complete Gate A before any B/C qualification.
    for variant in ('cnn', 'nca', 'vit', 'constant', 'learned'):
        for seed in cfg['seeds']:
            for dim in cfg['dimensions']:
                if not execute(dim, 'A', variant, seed):
                    write_summary(cfg, rows, out, 'STOPPED_BUDGET_OR_NUMERICS')
                    return
    for dim in cfg['dimensions']:
        if verdicts(cfg, rows)[str(dim)]['A'] != 'PASS':
            continue
        for variant in ('constant', 'oracle', 'learned'):
            for seed in cfg['seeds']:
                if not execute(dim, 'B', variant, seed):
                    write_summary(cfg, rows, out, 'STOPPED_BUDGET_OR_NUMERICS')
                    return
        if verdicts(cfg, rows)[str(dim)]['B'] != 'PASS':
            continue
        for variant in ('message_full', 'state_full'):
            for seed in cfg['seeds']:
                if not execute(dim, 'C', variant, seed, width=cfg['width']):
                    write_summary(cfg, rows, out, 'STOPPED_BUDGET_OR_NUMERICS')
                    return
        if verdicts(cfg, rows)[str(dim)]['C'] != 'PASS':
            continue
        for width in (4, 8, 16, 64):
            for seed in cfg['seeds']:
                if not execute(dim, 'B', 'learned', seed, width):
                    write_summary(cfg, rows, out, 'STOPPED_BUDGET_OR_NUMERICS')
                    return
    write_summary(cfg, rows, out, 'COMPLETED_FROZEN_SCHEDULE')


if __name__ == '__main__':
    main()
