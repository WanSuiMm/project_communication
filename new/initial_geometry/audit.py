"""Bounded CPU audit of exact historical update-zero StreamingCell geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as TF

ROOT = Path(__file__).resolve().parents[2]
for rel in ('new/nca_inertial_wind_tunnel', 'new/workspace_revision',
            'new/short_bptt', 'new/streaming_carry'):
    sys.path.insert(0, str(ROOT / rel))
from stream_cells import StreamingCell, stream
from training import backward_trajectory
from tasks import bank, subset

SEEDS = (2, 3, 4, 5)
BANKROOT = ROOT / 'runs/audit195_200_20261003_joint01'
CKPTROOT = ROOT / 'runs/warmstart_20261003_paired01'
BANDS = ((0, 8), (9, 16), (17, 32), (33, 100000))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def thash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((a.shape, a.dtype)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def selected(data):
    maps, flats = [], []
    for m in range(len(data['x'])):
        d = data['distance'][m, 0].numpy().reshape(-1)
        c = data['changed'][m, 0].bool().numpy().reshape(-1)
        ids = []
        for lo, hi in BANDS:
            pool = np.flatnonzero(c & (d >= lo) & (d <= hi))
            if len(pool):
                ids.extend(pool[np.linspace(0, len(pool)-1, min(32, len(pool))).astype(int)].tolist())
        maps.extend([m] * len(ids)); flats.extend(ids)
    return np.array(maps), np.array(flats)


def take(a, maps, flats):
    return a.flatten(2).permute(0, 2, 1)[maps, flats].double().numpy()


def spectrum(a, center=False):
    a = np.asarray(a, dtype=np.float64)
    if center:
        a = a-a.mean(0, keepdims=True)
    eig = np.linalg.eigvalsh(a.T@a/max(1, len(a)))[::-1].clip(0)
    total = float(eig.sum())
    p = eig/total if total else np.zeros_like(eig)
    active = eig > (eig[0]*1e-8 if len(eig) and eig[0] else 0)
    positive = eig[active]
    return {'eigenvalues': eig.tolist(), 'trace': total,
            'rank_relative_1e8': int(active.sum()), 'columns': int(a.shape[1]),
            'effective_rank': float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())) if total else 0.,
            'nonzero_condition': float(positive[0]/positive[-1]) if len(positive) else None,
            'top_fraction': float(p[0]) if len(p) else 0.}


def local_support(w, source, mask):
    v = w.any(1, keepdim=True).float()
    padded = TF.pad(v, (1, 1, 1, 1))
    neighbor = (padded[:, :, 1:-1, :-2] + padded[:, :, 1:-1, 2:]
                + padded[:, :, :-2, 1:-1] + padded[:, :, 2:, 1:-1]) > 0
    return (v.bool() | (neighbor & mask.bool()) | source)


def feature(model, w, x):
    z = w.new_zeros((len(w), 8, *w.shape[-2:]))
    return torch.tanh(model.q_in(model._features(w, z, x)))


def lane_stats(w, mask):
    values = w.permute(0, 2, 3, 1)[mask[:, 0].bool()].double().reshape(-1, 4, 6)
    rms = values.square().mean((0, 2)).sqrt()
    centered = values-values.mean(0, keepdim=True)
    crms = centered.square().mean((0, 2)).sqrt()
    pair = []
    for i in range(4):
        for j in range(i+1, 4):
            a, b = centered[:, i].reshape(-1), centered[:, j].reshape(-1)
            den = a.norm()*b.norm()
            pair.append(float(a@b/den) if den else 0.)
    return {'raw_rms': rms.tolist(), 'centered_rms': crms.tolist(),
            'raw_max_min': float(rms.max()/rms.min()),
            'centered_max_min': float(crms.max()/crms.min()),
            'centered_cosines': pair, 'mean_abs_cosine': float(np.abs(pair).mean())}


def relational(a, b, available, sign, maps, distance, rnorm2):
    diff = a-b
    norms = np.linalg.norm(diff, axis=1)
    scale = (np.linalg.norm(a, axis=1)+np.linalg.norm(b, axis=1))/2
    rel = norms/(scale+1e-12)
    oriented = diff*sign[:, None]
    sp = spectrum(diff*np.sqrt(rnorm2))
    # Frobenius alignment K with yy^T, evaluated without forming a sample Gram.
    numerator = float(np.square(oriented.sum(0)).sum())
    denominator = float(np.linalg.norm(diff.T@diff, 'fro')*len(diff))
    sp['label_alignment'] = numerator/denominator if denominator else None
    rows = []
    for m in np.unique(maps):
        for lo, hi in BANDS:
            take_ = (maps == m) & (distance >= lo) & (distance <= hi)
            active = take_ & available
            if take_.any():
                rows.append({'map': int(m), 'band': f'{lo}:{hi}', 'n': int(take_.sum()),
                    'available': int(active.sum()), 'nonzero': int((norms[take_]>1e-8).sum()),
                    'mean_pair_norm': float(norms[take_].mean()),
                    'mean_relative_separation': float(rel[take_].mean()),
                    'available_mean_relative': float(rel[active].mean()) if active.any() else None})
    return {'spectrum': sp, 'available_count': int(available.sum()), 'target_count': len(norms),
            'unavailable_nonzero_count': int(((~available) & (norms>1e-8)).sum()),
            'map_mean_relative': float(np.mean([rel[maps==m].mean() for m in np.unique(maps)])),
            'available_map_mean_relative': float(np.mean([rel[(maps==m)&available].mean()
                for m in np.unique(maps) if ((maps==m)&available).any()])) if available.any() else None,
            'per_map_bands': rows}


@torch.no_grad()
def collect(model, data, size, name, out):
    maps, flats = selected(data)
    distance = data['distance'][:, 0].flatten(1)[maps, flats].numpy()
    sign = (2*data['y'][:, 0].flatten(1)[maps, flats]-1).numpy()
    x = torch.cat((data['x'], data['x_flip']))
    n = len(data['x']); w, z = model.initial(x)
    src = (data['x'] != data['x_flip']).any(1, keepdim=True)
    support = src.expand(-1, 24, -1, -1).clone()
    mask = data['mask']
    rnorm2 = float(model.readout.weight.double().square().sum())
    result = {'bank': name, 'size': size, 'map_count': n, 'target_count': len(maps),
              'lane': lane_stats(w[:n], mask), 'rnorm2': rnorm2, 'features': {}, 'windows': []}
    raw = {'map': maps, 'flat': flats, 'distance': distance, 'sign': sign}
    total = np.zeros((2*len(maps), 16)); window = total.copy()
    ever = np.zeros(len(maps), dtype=bool); access_window = ever.copy()
    for t in range(65):
        if t:
            w = stream(w, x[:, :1])
            support = stream(support.float(), mask).bool()
        h = feature(model, w, x)
        hs = np.concatenate((take(h[:n], maps, flats), take(h[n:], maps, flats)))
        access = take(local_support(support, src, mask), maps, flats)[:, 0].astype(bool)
        if t in (0, 8, 64):
            item = relational(hs[:len(maps)], hs[len(maps):], access, sign, maps, distance, rnorm2)
            item['feature_covariance'] = spectrum(hs, center=True)
            result['features'][str(t)] = item
            raw[f'h{t}'] = hs
        if t:
            window += .5*hs; total += .5*hs
            access_window |= access; ever |= access
        if t and t % 8 == 0:
            k = len(maps)
            block = relational(window[:k], window[k:], access_window, sign, maps, distance, rnorm2)
            full = relational(total[:k], total[k:], ever, sign, maps, distance, rnorm2)
            block['feature_covariance'] = spectrum(window, center=True)
            full['feature_covariance'] = spectrum(total, center=True)
            result['windows'].append({'t': t, 'K8': block, 'full_history': full})
            raw[f'H_K8_{t}'] = window.copy(); raw[f'H_full_{t}'] = total.copy()
            raw[f'available_K8_{t}'] = access_window.copy(); raw[f'available_full_{t}'] = ever.copy()
            window.fill(0); access_window.fill(False)
    # The sampled analytic no-head path is exactly the numerical model path.
    full_state = model.rollout(data['x'][:1], 16)
    assert float(full_state[1].abs().max()) == 0
    assert float(model.logits(full_state).abs().max()) == 0
    assert all(v['unavailable_nonzero_count'] == 0 for v in result['features'].values())
    assert all(v[q]['unavailable_nonzero_count'] == 0 for v in result['windows'] for q in ('K8', 'full_history'))
    np.savez_compressed(out/f'{name}_seed{model.audit_seed}_features.npz', **raw)
    return result


def tangent_sanity(model, x):
    torch.manual_seed(90813)
    results = []
    for steps, detached in ((8, False), (16, False), (16, True)):
        state = model.initial(x[:1])
        accum = torch.zeros((16,))
        for t in range(1, steps+1):
            if detached and t == 9:
                state = tuple(v.detach() for v in state); accum.zero_()
            state = model.step(state, x[:1])
            with torch.no_grad():
                h = feature(model, state[0], x[:1])
                accum += .5*h[0, :, 3, 3]
        value = model.logits(state)[0, 0, 3, 3]
        grads = torch.autograd.grad(value, tuple(model.parameters()), allow_unused=True)
        actual = dict(zip((n for n, _ in model.named_parameters()), grads))
        r = model.readout.weight.detach().reshape(8)
        expected = r[:, None]*accum[None, :]
        err = float((actual['q_out.weight'].reshape(8,16)-expected).abs().max())
        bias_steps = 8 if detached else steps
        err = max(err, float((actual['q_out.bias']-.5*bias_steps*r).abs().max()))
        closed = {n: float(g.abs().max()) for n, g in actual.items()
                  if g is not None and n not in ('q_out.weight','q_out.bias','readout.bias')}
        assert err <= 2e-6 and max(closed.values(), default=0.) == 0
        results.append({'steps': steps, 'detach_at8': detached, 'max_error': err,
                        'closed_task_derivatives': closed})
    return results


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True)
    args = parser.parse_args(); out = Path(args.out)
    assert not out.exists(), 'New output directory required'
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    start = time.monotonic()
    sources = ('new/initial_geometry/audit.py', 'new/initial_geometry/PROTOCOL.md',
               'new/streaming_carry/stream_cells.py', 'new/workspace_revision/revision_cells.py',
               'new/masked_medium/masked_cells.py', 'new/nca_inertial_wind_tunnel/tasks.py',
               'new/short_bptt/training.py')
    frozen = json.loads((BANKROOT/'manifest.json').read_text())['bindings']['source_sha256']
    for p in sources[2:]:
        assert sha(ROOT/p) == frozen[p], f'Source drift: {p}'
    manifest = {'protocol': 'initial_geometry_v1', 'source_sha256': {p:sha(ROOT/p) for p in sources},
                'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu',
                'checkpoints': {}, 'banks': {}, 'no_training': True}
    # Exact original first batch, used only for one backward with no optimizer.
    historical = json.loads((ROOT/'evidence/streaming_carry_init2345/schedule.json').read_text())
    rows = historical if isinstance(historical,list) else historical['rows']
    train_bank = bank(32,512,10002)
    reference4 = json.loads((ROOT/'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json').read_text())
    assert thash(train_bank) == reference4['train_data_sha256'], 'Historical training bank drift'
    assert sha(ROOT/'evidence/streaming_carry_init2345/schedule.json') == reference4['schedule_sha256']
    manifest['historical_first_batch'] = {'indices':rows[0], 'train_data_sha256':thash(train_bank),
        'schedule_sha256':sha(ROOT/'evidence/streaming_carry_init2345/schedule.json')}
    first = subset(train_bank, rows[0])
    data_banks = {}
    for kind in ('primary', 'confirmation'):
        for size in (32,64):
            name = f'bank_{kind}{size}'
            p = BANKROOT/f'{name}.npz'; a = np.load(p)
            data_banks[name] = ({k:torch.from_numpy(a[k]) for k in a.files}, size)
            manifest['banks'][name] = {'file':p.relative_to(ROOT).as_posix(),'sha256':sha(p)}
    summary = {'protocol':'initial_geometry_v1', 'status':'RUNNING', 'seeds':{},
               'independent_seed_count':4, 'training_updates':0, 'limitations':[
                   'Post hoc diagnostic; four previously inspected seeds, one historical success.',
                   'Existing inspected banks; no fresh confirmation or initialization probability.',
                   'Feature and parameter geometry depends on coordinates and sample weights.',
                   'Initial feature distinction is not a learned continuation quotient.']}
    for seed in SEEDS:
        checkpoint_path = CKPTROOT/f'baseline_seed{seed}_u000.pt'
        checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        assert checkpoint['completed_updates'] == 0 and checkpoint['seed'] == seed
        model = StreamingCell(); model.load_state_dict(checkpoint['state_dict']); model.eval(); model.audit_seed = seed
        initial_hash = thash(model.state_dict())
        ref = json.loads((ROOT/f'evidence/streaming_carry_init2345/raw/stream_K8_seed{seed}.json').read_text())
        assert initial_hash == ref['initial_parameter_sha256']
        manifest['checkpoints'][str(seed)] = {'file':checkpoint_path.relative_to(ROOT).as_posix(),
            'file_sha256':sha(checkpoint_path), 'initial_parameter_sha256':initial_hash}
        seed_row = {'initial_parameter_sha256':initial_hash, 'sanity':tangent_sanity(model,first['x']), 'banks':{}}
        model.zero_grad(set_to_none=True)
        loss, _, trace = backward_trajectory(model, first, 8)
        seed_row['first_batch_K8'] = {'loss':float(loss), 'trace':trace,
            'gradient_l2':{n:float(p.grad.double().norm()) if p.grad is not None else None
                           for n,p in model.named_parameters()}}
        g = model.q_out.weight.grad.detach().reshape(8,16).double()
        seed_row['first_batch_K8']['qout_gradient_singular_values'] = torch.linalg.svdvals(g).tolist()
        model.zero_grad(set_to_none=True)
        for name, (data,size) in data_banks.items():
            seed_row['banks'][name] = collect(model,data,size,name,out)
            print(json.dumps({'seed':seed,'bank':name,'elapsed':round(time.monotonic()-start,2)}),flush=True)
        assert thash(model.state_dict()) == initial_hash, 'Parameters changed'
        summary['seeds'][str(seed)] = seed_row
        save(out/'summary.json',summary); save(out/'manifest.json',manifest)
    assert all(sha(ROOT/p)==v for p,v in manifest['source_sha256'].items())
    summary['status'] = 'COMPLETE'; summary['elapsed_seconds'] = time.monotonic()-start
    save(out/'summary.json',summary); save(out/'manifest.json',manifest)
    save(out/'status.json', {'status':'COMPLETE','seeds':4,'banks_per_seed':4,'updates':0,
                             'elapsed_seconds':summary['elapsed_seconds']})
    print(json.dumps({'status':'COMPLETE','elapsed':summary['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
