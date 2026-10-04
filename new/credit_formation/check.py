"""One focused metric and frozen-checkpoint CPU/CUDA qualification."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

import run
from credit_metrics import summarize_pair, formation_verdict


def vector_checks():
    names = ['encoder.weight', 'f_in.weight', 'q_in.weight', 'readout.weight']
    shapes = [[1]]*4
    full = np.array([1., 0., 0., 0.])
    row = summarize_pair(np.array([1., 10., 0., 0.]), full, names, shapes)
    assert np.isclose(row['overall']['C_parallel'], 1.)
    assert np.isclose(row['overall']['C_miss'], 10.)
    assert np.isclose(row['overall']['cosine'], 1/np.sqrt(101))
    zero = summarize_pair(np.zeros(4), np.zeros(4), names, shapes)
    assert all(zero['overall'][k] is None for k in ('cosine', 'C_parallel', 'C_miss'))
    credits = {u: summarize_pair(np.ones(4)*scale, np.ones(4), names, shapes)
               for u, scale in ((175, .8), (180, -.2), (195, .8), (200, .9))}
    phenotypes = {u: {'32': {'preservation': p, 'sustained_progress': g},
                     '64': {'preservation': p, 'sustained_progress': g}}
                  for u, p, g in ((175, .9, .7), (180, .4, .2), (195, .9, .7), (200, .91, .75))}
    assert formation_verdict(credits, phenotypes)['verdict'] == 'STRONG_SYNCHRONY'
    partial = copy.deepcopy(credits)
    for row in partial.values():
        row['overall']['cosine'] = .8; row['overall']['C_parallel'] = .8
    assert formation_verdict(partial, phenotypes)['verdict'] == 'MODULE_SYNCHRONY'
    failure = copy.deepcopy(partial)
    for row in failure.values():
        for scope in ('F', 'Q'):
            row['groups'][scope]['cosine'] = .8; row['groups'][scope]['C_parallel'] = .8
    assert formation_verdict(failure, phenotypes)['verdict'] == 'CREDIT_SYNCHRONY_NOT_SUPPORTED'
    absent = copy.deepcopy(phenotypes); absent[180]['32'] = dict(absent[175]['32'])
    assert formation_verdict(credits, absent)['verdict'] == 'PHENOTYPE_UNQUALIFIED'
    return {'status': 'PASS', 'orthogonal_error_at_projection_one': True,
            'zero_norms_undefined': True, 'four_screen_branches': True}


def check(out):
    assert out.parent == run.ROOT/'analyses' and not out.exists()
    vectors = vector_checks()
    run.base.setup()
    sources, rows = run.sources(), run.catalog()
    data, bank_rows = run.banks()
    for row in rows.values(): run.load_model(row)
    free, total = torch.cuda.mem_get_info()
    assert free >= 3500*1024**2, 'Insufficient GPU headroom'
    model = run.load_model(rows[140]).cuda().eval()
    before = run.original_gradients._hash(model)
    batch = {k: data[32][k][:2].cuda() for k in ('x', 'y', 'mask')}
    a, names, shapes, la, da, ca = run.backward_record(model, batch, 8, steps=16)
    b, nb, sb, lb, db, cb = run.backward_record(model, batch, 16, steps=16)
    assert names == nb and shapes == sb and da == db and abs(la-lb) <= 1e-6
    assert ca['backward_calls'] == 2 and cb['backward_calls'] == 1
    assert np.any(a) and np.any(b) and np.isfinite(a).all() and np.isfinite(b).all()
    assert before == run.original_gradients._hash(model)
    model.cpu(); del batch
    for size in (32, 64):
        small = {k: v[:2] for k, v in data[size].items()}
        trace, scores, _, _ = run.base.paired_trace(model, small)
        assert trace['correct'].shape == (257, 2, size, size)
        run.base.summarize_handoff(trace['correct'][64:], trace['correct'][64],
                                  small['changed'].numpy(), small['distance'].numpy())
    assert before == run.original_gradients._hash(model)
    result = {'protocol': run.PROTOCOL, 'status': 'PASS', 'training_updates': 0,
              'source_sha256': sources, 'checkpoints': rows, 'banks': bank_rows,
              'cpu': vectors, 'gpu': {'exact_forward_match_steps': 16,
                  'forward_loss_difference': abs(la-lb), 'paired_256_sizes': [32, 64],
                  'parameters_unchanged': True, 'gradient_nonzero_finite': True}}
    run.write(out, result)
    print({'status': 'PASS', 'checkpoints': len(rows), 'cpu': 'PASS', 'cuda': 'PASS'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    a = p.parse_args(); check((run.ROOT/a.out).resolve())
