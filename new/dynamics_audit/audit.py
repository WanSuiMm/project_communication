"""Frozen-checkpoint audit of gain, drift and task information; never trains."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import time
import traceback
import numpy as np
import torch

from operators import factory, base, pack, unpack, LinearStep, product, estimate, norm, unit, mul, dot, conv
from tasks import bank, metrics, components

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = {
    'nca_state_matched': ('inertial_20261001_seed0', 'inertial_seed0', 'ebc3860e920ebe717de6b62095581ae3793b4103dbcc076680f66f221e2d2045'),
    'momentum_nca': ('inertial_20261001_seed0', 'inertial_seed0', 'fe2f4a2a2f1d7170e44618ffbb9b2b2245f115ca75c63c85d789d1e6d0a72c98'),
    'masked_state_nca': ('masked_state_20261001_seed0', 'masked_state_seed0', '813799a66c552efa33edb6f31100bfa7189b59781621fd8bab88c6d937590c90'),
    'masked_momentum_nca': ('masked_momentum_20261001_seed0', 'masked_momentum_seed0', 'e08afce75406afe5c5f44831e982c0814f39fdf9801d07e613c06257c456c5a8'),
}
ANCHORS = (16, 32, 48, 64, 96, 128, 192, 256)
DEADLINE = float('inf')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_bytes())
def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def check_time():
    if time.monotonic() > DEADLINE: raise TimeoutError('Frozen audit time cap reached; no automatic retry')


def component_fraction(h, labels, mask):
    rows = []
    for i, ids in enumerate(labels):
        z = h[i].double().flatten(1)
        valid = ids >= 0
        groups = int(ids.max())+1
        sums = torch.zeros(z.shape[0], groups, dtype=torch.float64, device=h.device)
        sums.index_add_(1, ids[valid], z[:, valid])
        counts = torch.bincount(ids[valid], minlength=groups).double()
        projected = (sums.square()/counts).sum()
        total = z[:, valid].square().sum()
        rows.append(float(projected/total.clamp_min(1e-300)))
    return rows


def rms_open(h, mask):
    return norm(h*mask)/(mask.flatten(1).sum(1)*h.shape[1]).sqrt()


def summarize_tangent(model, vector, logscale, mask, dx_logits, sign):
    z = conv(vector[:, :base(model).channels], base(model).readout.weight)*mask
    n = norm(z)
    # All magnitudes remain in log form, including exponentially large sensitivities.
    target_mean = (z.double()*sign*mask).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1)
    finite_delta = dx_logits*mask
    cosine = dot(z, finite_delta)/(norm(z)*norm(finite_delta)).clamp_min(1e-300)
    rows = []
    for i in range(len(vector)):
        rows.append({'log_state_l2': float(logscale[i]+norm(vector)[i].clamp_min(1e-300).log()),
                     'log_changed_logit_l2': float(logscale[i]+n[i].clamp_min(1e-300).log()),
                     'signed_target_derivative_sign': int(target_mean[i].sign()),
                     'log_abs_mean_signed_target_derivative': float(logscale[i]+target_mean[i].abs().clamp_min(1e-300).log()),
                     'cosine_with_finite_binary_source_flip': float(cosine[i])})
    return rows


def scaled_update(step, vector, scale, forcing):
    advanced = step.apply(vector)+mul(forcing, (-scale).exp())
    divisor = norm(advanced).clamp_min(1.)
    return mul(advanced, 1/divisor), scale+divisor.log()


@torch.no_grad()
def trajectories(model, data, anchors, diagnostic_maps):
    x, mask = data['x'], data['mask']
    state, flip = model.initial(x), model.initial(data['x_flip'])
    labels = [torch.tensor(components(m[0].cpu().numpy() > .5)[0].ravel(), device=x.device, dtype=torch.long) for m in mask]
    dx = data['x_flip'][:diagnostic_maps]-x[:diagnostic_maps]
    assert not bool(dx[:, 0].any())
    initial_h = conv(dx, base(model).encoder.weight)
    initial = torch.cat((initial_h, torch.zeros_like(initial_h)), 1) if base(model).inertial else initial_h
    tangents = {key: (initial.clone() if key != 'drive_only' else torch.zeros_like(initial),
                      torch.zeros(diagnostic_maps, device=x.device, dtype=torch.float64))
                for key in ('total', 'initial_only', 'drive_only')}
    curves, saved, slopes = {}, {}, {}
    save_times = set(anchors) | {t+16 for t in anchors}
    window_times = {t+k for t in anchors for k in range(16)}
    for t in range(max(anchors)+17):
        check_time()
        if not bool(torch.isfinite(pack(state)).all() and torch.isfinite(pack(flip)).all()):
            raise FloatingPointError('Nonfinite nonlinear trajectory')
        if t in save_times: saved[t] = pack(state)[:diagnostic_maps].clone()
        if t in anchors:
            z, zf = model.logits(state), model.logits(flip)
            row = metrics(z, data['y'], mask)
            both = ((z >= 0) == (data['y'] >= .5)) & ((zf >= 0) == (data['y_flip'] >= .5))
            change = data['changed']
            row['paired'] = float((both*change).sum()/change.sum())
            row['content_rms'] = float(state[0].square().mean().sqrt())
            row['velocity_rms'] = None if state[1] is None else float(state[1].square().mean().sqrt())
            row['h_component_constant_fraction_per_map'] = component_fraction(state[0], labels, mask)
            row['h_open_rms_per_map'] = rms_open(state[0], mask).cpu().tolist()
            row['v_open_rms_per_map'] = None if state[1] is None else rms_open(state[1], mask).cpu().tolist()
            row['v_component_constant_fraction_per_map'] = None if state[1] is None else component_fraction(state[1], labels, mask)
            next_state = model.step(state, x)
            update = next_state[0]-state[0]
            row['h_update_open_rms_per_map'] = rms_open(update, mask).cpu().tolist()
            row['update_component_constant_fraction_per_map'] = component_fraction(update, labels, mask)
            row['per_map'] = []
            for i in range(len(x)):
                item = metrics(z[i:i+1], data['y'][i:i+1], mask[i:i+1])
                margins = ((2*data['y'][i]-1)*z[i])[mask[i] > .5].double()
                wrong = margins[margins < 0]
                item.update(paired=float((both[i]*change[i]).sum()/change[i].sum()),
                            margin_mean=float(margins.mean()), margin_q10=float(margins.quantile(.1)),
                            wrong_margin_median=None if not len(wrong) else float(wrong.median()),
                            finite_source_flip_logit_rms=float(rms_open((zf-z)[i:i+1], change[i:i+1])[0]))
                row['per_map'].append(item)
            row['source_tangents'] = {key: summarize_tangent(model, vec, scale, change[:diagnostic_maps],
                (zf-z)[:diagnostic_maps], (2*data['y_flip']-1)[:diagnostic_maps]) for key, (vec, scale) in tangents.items()}
            curves[str(t)] = row
        if t == max(anchors)+16: break
        step = LinearStep(model, state[0][:diagnostic_maps], x[:diagnostic_maps])
        if t in window_times: slopes[t] = step
        forcing = step.input_apply(dx)
        for key, (vec, scale) in tangents.items():
            tangents[key] = scaled_update(step, vec, scale, torch.zeros_like(forcing) if key == 'initial_only' else forcing)
        state = model.step(state, x)
        flip = model.step(flip, data['x_flip'])
    return curves, saved, slopes


def free_baseline(model):
    if not base(model).inertial: return {str(k): {'sigma': 1., 'finite_time_log_gain': 0.} for k in (1, 8, 16)}
    beta = base(model).coefficients()[0].double().flatten().cpu()
    rows = {}
    for k in (1, 8, 16):
        mat = torch.zeros(len(beta), 2, 2, dtype=torch.float64)
        mat[:, 0, 0] = 1; mat[:, 0, 1] = beta*(1-beta**k)/(1-beta); mat[:, 1, 1] = beta**k
        sigma = float(torch.linalg.svdvals(mat)[:, 0].max())
        rows[str(k)] = {'sigma': sigma, 'finite_time_log_gain': math.log(sigma)/k}
    return rows


@torch.no_grad()
def perturbation(model, start, end, x, y, mask, direction, forward):
    # Open-state perturbations; metric units are fixed, not whitened along the trajectory.
    count = mask.flatten(1).sum(1)*start.shape[1]
    magnitude = rms_open(start, mask).clamp_min(1.)*count.sqrt()
    expected = forward(direction)
    reference = unpack(start, model)
    for _ in range(16): reference = model.step(reference, x)
    reference_end = pack(reference)
    baseline = metrics(model.logits(reference), y, mask)
    batch_replay_error = norm((reference_end-end)*mask)/norm(end*mask).clamp_min(1e-300)
    records = []
    for epsilon in (1e-4, 1e-3):
        factor = magnitude*epsilon
        state = unpack(start+mul(direction, factor), model)
        for _ in range(16):
            check_time(); state = model.step(state, x)
        diff = (pack(state)-reference_end)*mask
        linear = mul(expected, factor)
        error = norm(diff-linear)/norm(linear).clamp_min(1e-300)
        m = metrics(model.logits(state), y, mask)
        records.append({'relative_rms_epsilon': epsilon, 'relative_linearization_error_per_map': error.cpu().tolist(),
                        'nonlinear_gain_per_map': (norm(diff)/factor).cpu().tolist(),
                        'linear_gain_per_map': norm(expected).cpu().tolist(),
                        'unperturbed_window_batch_replay_relative_error': batch_replay_error.cpu().tolist(),
                        'baseline_subset': baseline, 'perturbed_subset': m})
    return records


@torch.no_grad()
def diagnose(model, data, saved, slopes, anchors, iterations, starts, maps, out, arm, size):
    x, y, mask = data['x'][:maps], data['y'][:maps], data['mask'][:maps]
    results = {}
    for t in anchors:
        check_time()
        report = {'products': {}, 'momentum_blocks': {}, 'perturbations': {}}
        shape = saved[t].shape
        for scope, projection in (('full', torch.ones_like(mask)), ('open_input_output', mask)):
            for k in (1, 8, 16):
                forward, backward = product([slopes[j] for j in range(t, t+k)], projection)
                rows, vectors = estimate(forward, backward, shape, x.device, x.dtype, projection,
                                          iterations, starts, 61000+size+t+k, check_time)
                out_vector = forward(vectors)
                logit = conv(out_vector[:, :base(model).channels], base(model).readout.weight)*mask
                for i, r in enumerate(rows):
                    r.update(map=i, finite_time_log_gain=math.log(max(r['sigma_estimate'], 1e-300))/k,
                             output_open_energy_fraction=float(norm(out_vector*mask)[i].square()/norm(out_vector)[i].square().clamp_min(1e-300)),
                             readout_open_logit_l2_gain=float(norm(logit)[i]))
                report['products'][f'{scope}_K{k}'] = rows
                if scope == 'open_input_output' and k == 16:
                    report['perturbations']['estimated_worst'] = perturbation(model, saved[t], saved[t+16], x, y, mask, vectors, forward)
                    generator = torch.Generator(device=x.device).manual_seed(71000+size+t)
                    random, _ = unit(torch.randn(shape, device=x.device, generator=generator)*mask)
                    report['perturbations']['random'] = perturbation(model, saved[t], saved[t+16], x, y, mask, random, forward)
        if base(model).inertial:
            step = slopes[t]
            hshape = saved[t][:, :base(model).channels].shape
            for name, f, adj in (('dVnext_dH', step.force, step.force_adj),
                                 ('dHnext_dH', lambda q: q+step.force(q), lambda q: q+step.force_adj(q))):
                rows, _ = estimate(f, adj, hshape, x.device, x.dtype, torch.ones_like(mask),
                                    iterations, starts, 81000+size+t, check_time)
                report['momentum_blocks'][name] = rows
            b = base(model).coefficients()[0]
            report['momentum_blocks']['dHnext_dV_and_dVnext_dV_exact_norm'] = float(b.max())
        results[str(t)] = report
        write(out/f'{arm}_size{size}_dynamics.json', results)
        write(out/'status.json', {'status': 'RUNNING', 'phase': 'jacobian_diagnostics', 'arm': arm, 'size': size,
                                  'last_completed_anchor': t, 'pid': os.getpid()})
        print(json.dumps({'arm': arm, 'size': size, 'completed_anchor': t}), flush=True)
    return results


def report(out, records, status, elapsed):
    lines = ['# Generic NCA dynamics audit', '', f'Execution status: `{status}`. No training or model modification.',
             f'Elapsed seconds: {elapsed:.1f}. One training seed; maps are diagnostic examples, not training replicates.', '',
             'Read manifest.json for frozen settings. Raw curves and per-map dynamics are separate files.', '',
             '| Arm | Size | T | BA % | BCE | Paired % | Open K16 median log gain | Converged estimates / maps | H mean-mode energy |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, r in records.items():
        for t, curve in r['curves'].items():
            diag = r.get('dynamics', {}).get(t, {}).get('products', {}).get('open_input_output_K16', [])
            gain = 'pending' if not diag else f"{np.median([p['finite_time_log_gain'] for p in diag]):.5f}"
            convs = 'pending' if not diag else f"{sum(p['converged'] for p in diag)}/{len(diag)}"
            lines.append(f"| {r['arm']} | {r['size']} | {t} | {100*curve['balanced_accuracy']:.2f} | {curve['bce']:.4f} | {100*curve['paired']:.2f} | {gain} | {convs} | {np.mean(curve['h_component_constant_fraction_per_map']):.4f} |")
    lines += ['', 'The performance columns use all 16 frozen maps; Jacobian diagnostics use fixed maps 0..3.',
              'Positive finite-window log gain is not asymptotic instability. Compare each arm to its free-momentum baseline.',
              'Unconverged power estimates remain lower estimates; do not claim an upper bound or near-criticality from them.',
              'Open projection is at product input/output only; whole-grid paths may traverse walls between endpoints.',
              'Source tangents include encoder initialization and repeated source drive, separately and jointly.',
              'Binary source flips are finite interventions; their agreement with infinitesimal tangents is not assumed.',
              'No causal velocity isolation, asymptotic convergence, trained dynamics fix, or systems speedup is established.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    global DEADLINE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); DEADLINE = start+(180 if args.preflight else 1500)
    records = {}; status = 'RUNNING'
    try:
        publication = read(ROOT/'STATE_PUBLICATION_MANIFEST.json')
        hashes = {**publication['source_sha256'], **publication['reference_sha256'], **publication['published_evidence_sha256']}
        for rel, expected in hashes.items():
            assert sha(ROOT/rel) == expected, f'Frozen evidence changed: {rel}'
        setting = read(ROOT/'evidence/masked_state_seed0/manifest.json')['backend']
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = setting['cudnn_benchmark']
        torch.backends.cudnn.deterministic = setting['cudnn_deterministic']
        torch.backends.cudnn.allow_tf32 = setting['cudnn_allow_tf32']
        torch.backends.cuda.matmul.allow_tf32 = setting['matmul_allow_tf32']
        assert torch.cuda.is_available(), 'No silent CPU substitution'
        anchors = (32,) if args.preflight else ANCHORS
        sizes = (32,) if args.preflight else (32, 64)
        maps, iterations, starts = (1, 4, 2) if args.preflight else (4, 16, 2)
        sources = {**publication['source_sha256'], **{f.relative_to(ROOT).as_posix(): sha(f)
                   for f in HERE.iterdir() if f.suffix in ('.py', '.md')}}
        for rel in sources:
            target = out/'source'/rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/rel, target)
        manifest = {'protocol': 'generic_nca_dynamics_audit_seed0', 'pid': os.getpid(), 'training': False,
                    'preflight': args.preflight, 'started_utc': datetime.now(timezone.utc).isoformat(),
                    'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'backend': setting,
                    'sizes': sizes, 'anchors': anchors, 'test_maps': 16, 'jacobian_map_ids': list(range(maps)),
                    'test_seeds': {str(s): 30000+s for s in sizes}, 'windows': [1, 8, 16],
                    'power_iterations': iterations, 'power_starts': starts, 'cap_seconds': 180 if args.preflight else 1500,
                    'state_metric': 'raw Euclidean H,V with fixed units; no per-time whitening',
                    'source_sha256': sources, 'reference_sha256': publication['reference_sha256'],
                    'checkpoint_sha256': {a: v[2] for a, v in ARMS.items()}}
        write(out/'manifest.json', manifest)
        write(out/'local_receipt.json', {'host': socket.gethostname(), 'pid': os.getpid(), 'output': str(out.resolve())})
        for arm, (run, evidence, expected) in ARMS.items():
            path = ROOT/'runs'/run/(arm+'_seed0.pt')
            assert sha(path) == expected, f'Checkpoint hash mismatch: {arm}'
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            assert checkpoint['arm'] == arm and checkpoint['completed_updates'] == 800
            model = factory(arm).cuda().eval()
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.requires_grad_(False)
            reference = read(ROOT/'evidence'/evidence/'arms'/(arm+'_seed0.json'))
            for size in sizes:
                check_time()
                write(out/'status.json', {'status': 'RUNNING', 'phase': 'trajectory_replay', 'arm': arm, 'size': size, 'pid': os.getpid()})
                data = bank(size, 16, 30000+size, 'cuda')
                curves, saved, slopes = trajectories(model, data, anchors, maps)
                errors = {}
                for t, row in curves.items():
                    old = reference['evaluation'][str(size)]['curve'].get(t)
                    if old is None: continue
                    errors[t] = {k: abs(row[k]-old[k]) for k in ('balanced_accuracy', 'bce', 'content_rms')}
                    if row['velocity_rms'] is not None: errors[t]['velocity_rms'] = abs(row['velocity_rms']-old['velocity_rms'])
                    errors[t]['paired'] = abs(row['paired']-reference['evaluation'][str(size)]['paired_source_information'][t]['both_counterfactuals_correct_on_changed_component'])
                maximum = max(v for row in errors.values() for v in row.values())
                write(out/f'{arm}_size{size}_curves.json', {'curves': curves, 'reconstruction': errors, 'max_error': maximum})
                assert maximum <= 5e-5, f'Replay gate failed: {arm}/{size}/{maximum}'
                item = {'arm': arm, 'size': size, 'curves': curves, 'free_momentum_baseline': free_baseline(model), 'reconstruction_max_error': maximum}
                records[f'{arm}_{size}'] = item
                item['dynamics'] = diagnose(model, data, saved, slopes, anchors, iterations, starts, maps, out, arm, size)
                del data, saved, slopes
                write(out/'summary.json', {'status': status, 'records': records})
                report(out, records, status, time.monotonic()-start)
            del model
        status = 'PREFLIGHT_PASSED' if args.preflight else 'COMPLETE'
    except TimeoutError as error:
        status = 'BUDGET_LIMIT'; write(out/'error.json', {'error': str(error)})
    except Exception as error:
        status = 'FAILED'; write(out/'error.json', {'error': str(error), 'traceback': traceback.format_exc()})
    elapsed = time.monotonic()-start
    write(out/'summary.json', {'status': status, 'training': False, 'records': records, 'elapsed_seconds': elapsed})
    report(out, records, status, elapsed)
    write(out/'status.json', {'status': status, 'pid': os.getpid(), 'finished_utc': datetime.now(timezone.utc).isoformat(),
                             'elapsed_seconds': elapsed, 'training': False})
    print(json.dumps({'status': status, 'elapsed_seconds': elapsed}), flush=True)
    if status not in ('COMPLETE', 'PREFLIGHT_PASSED'): raise SystemExit(1)


if __name__ == '__main__': main()
