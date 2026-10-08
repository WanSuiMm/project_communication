"""Lightweight cached paired evaluation; full Boolean arrays remain transient."""
from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


G = load('_addressed_frozen_pair_graph', ROOT / 'new/port_relation/evaluation_graph.py')
M = load('_addressed_frozen_metrics', ROOT / 'new/continuous_coverage/metrics.py')
_CACHE = {}


def binding(model):
    return tuple((n, p.data_ptr(), tuple(p.shape), str(p.device)) for n, p in model.named_parameters())


@torch.no_grad()
def captured(model, bank, size):
    key = (id(model), size)
    bank_hash = G.C.tensor_hash(bank)
    if key in _CACHE:
        result = _CACHE[key]
        if result['model'] is not model or result['binding'] != binding(model) or result['bank_hash'] != bank_hash:
            raise RuntimeError('Evaluation parameter storage or frozen bank changed')
        return result
    started = time.monotonic()
    data = {k: v.to(next(model.parameters()).device) for k, v in bank.items()}
    current, side = torch.cuda.current_stream(), torch.cuda.Stream()
    side.wait_stream(current)
    with torch.cuda.stream(side):
        G._trace_cuda(model, data, size)
    current.wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = G._trace_cuda(model, data, size)
    torch.cuda.synchronize()
    result = {'model': model, 'binding': binding(model), 'bank_hash': bank_hash,
              'data': data, 'outputs': outputs, 'graph': graph,
              'capture_seconds': time.monotonic() - started}
    _CACHE[key] = result
    return result


def clear(model):
    for key in list(_CACHE):
        if key[0] == id(model):
            del _CACHE[key]
    gc.collect()
    torch.cuda.empty_cache()


def map_metrics(correct, bank, size):
    changed = bank['changed'].numpy()[:, 0].astype(bool)
    distance = bank['distance'].numpy()[:, 0]
    strict = changed & (distance > 16) & (distance < 32)
    result = []
    for i in range(len(changed)):
        reference = correct[64, i] & changed[i]
        n = int(reference.sum())
        row = {'size': size, 'map_index': i, 'strict_pixels': int(strict[i].sum()),
               'changed_pixels': int(changed[i].sum()), 'retention_reference_pixels': n,
               'retained_pixels_T256': int((reference & correct[256, i]).sum()),
               'continuously_retained_pixels_T64_to256': int((reference & correct[64:, i].all(axis=0)).sum())}
        row['retention64_to256'] = row['retained_pixels_T256'] / n if n else None
        for t in (64, 128, 256):
            for label, mask in (('strict', strict[i]), ('changed', changed[i])):
                count = int(mask.sum())
                good = int((correct[t, i] & mask).sum())
                row[f'{label}_correct_T{t}'] = good
                row[f'{label}_coverage_T{t}'] = good / count if count else None
        result.append(row)
    return result


@torch.no_grad()
def evaluate(model, banks):
    model.eval()
    before = G.C.tensor_hash(model.state_dict())
    traces, scores, maps, diagnostics = {}, {}, [], {}
    for size in (32, 64):
        entry = captured(model, banks[size], size)
        entry['graph'].replay()
        torch.cuda.synchronize()
        outputs = entry['outputs']
        scores[str(size)] = G._check_and_score(outputs, entry['data'], size)
        traces[str(size)] = G._to_numpy_traces(outputs)
        maps.extend(map_metrics(traces[str(size)]['correct'], banks[size], size))
        diagnostics[str(size)] = {'trace_complete': True, 'finite': True,
                                 'outside_source_cone_max_logit_delta': float(outputs['outside_delta'])}
    # Frozen compact-summary arithmetic, without per-time files or old Full.
    summary = G.E._json_ready(G._dense_summary(traces, scores, banks))
    compact = M.compact_pair_metrics(summary)
    compact['endpoint_scores'] = scores
    if G.C.tensor_hash(model.state_dict()) != before:
        raise RuntimeError('Evaluation mutated model parameters')
    return {'metrics': compact, 'joint': M.joint_readiness(summary),
            'evaluation_status': 'COMPLETE', 'numerical_failure': False,
            'trace_complete': True, 'full_evaluated': False,
            'map_metrics': maps, 'causal_finite_guards': diagnostics,
            'trace_persistence': 'none; arrays used transiently for exact frozen metrics'}
