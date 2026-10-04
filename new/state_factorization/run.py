"""Zero-training output-preserving S1/F1 state localization."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(ROOT/'new/continuation_interface'))
base = load_module('state_factor_base', ROOT/'new/continuation_interface/run.py')
metrics = load_module('state_factor_metrics', Path(__file__).with_name('metrics.py'))
PROTOCOL = 'output_preserving_state_factorization_v1'
ARMS = ('FF', 'F-span', 'F-null', 'FS', 'SF', 'S-span', 'S-null', 'SS')
BANK_SPECS = {32: (32, 96032), 64: (32, 96064)}
PUBLIC = ROOT/'evidence/continuation_interface_20261004'
sha, read, write = base.sha, base.read, base.write


def sources():
    bindings = read(ROOT/'CONTINUATION_INTERFACE_PUBLICATION_MANIFEST.json')['source_sha256'].copy()
    for rel, digest in bindings.items():
        assert sha(ROOT/rel) == digest, f'Frozen scientific source drift: {rel}'
    for p in Path(__file__).parent.glob('*'):
        if p.suffix in ('.py', '.md'):
            bindings[p.relative_to(ROOT).as_posix()] = sha(p)
    p = ROOT/'tools/launch_state_factorization.ps1'
    bindings[p.relative_to(ROOT).as_posix()] = sha(p)
    return dict(sorted(bindings.items()))


def catalog():
    publication = read(ROOT/'CONTINUATION_INTERFACE_PUBLICATION_MANIFEST.json')
    cfg = PUBLIC/'config.json'
    assert sha(cfg) == publication['published_sha256'][cfg.relative_to(ROOT).as_posix()]
    rows = {k: read(cfg)['models'][k] for k in ('S1', 'F1')}
    for k, row in rows.items():
        assert sha(ROOT/row['path']) == row['sha256'], f'Checkpoint drift: {k}'
    return rows


def input_hashes(data):
    return {hashlib.sha256(v.numpy().tobytes()).hexdigest()
            for key in ('x', 'x_flip') for v in data[key]}


def make_banks(out):
    publication = read(ROOT/'CONTINUATION_INTERFACE_PUBLICATION_MANIFEST.json')
    seen = set()
    exclusion = {}
    for p in sorted((PUBLIC/'banks').glob('*.npz')):
        rel = p.relative_to(ROOT).as_posix()
        assert sha(p) == publication['published_sha256'][rel]
        with np.load(p, allow_pickle=False) as arr:
            old = {k: torch.from_numpy(arr[k].copy()) for k in ('x', 'x_flip')}
        seen.update(input_hashes(old)); exclusion[rel] = sha(p)
    assert len(exclusion) == 6, 'Missing historical bank exclusion'
    for size, seed in ((32,96932),(64,96964)):
        seen.update(input_hashes(base.bank(size,2,seed)))
    data, records = {}, {}
    (out/'banks').mkdir()
    for size, (count, seed) in BANK_SPECS.items():
        d = base.bank(size, count, seed)
        fresh = input_hashes(d)
        assert len(fresh) == 2*count and not (fresh & seen), 'Input bank collision'
        seen.update(fresh)
        path = out/'banks'/f'test{size}.npz'
        np.savez_compressed(path, **{k: v.numpy() for k, v in d.items()})
        data[size] = d
        records[size] = {'size': size, 'count': count, 'seed': seed,
                        'data_sha256': base.tensor_hash(d), 'file_sha256': sha(path),
                        'cross_role_collisions': 0}
    return data, records, exclusion


def projections(zf, zs, weight):
    r = np.asarray(weight, dtype=np.float64).reshape(-1)
    norm2 = float(r @ r)
    if not np.isfinite(r).all() or norm2 <= 0:
        raise ValueError('Nonzero finite readout required')
    if zf.shape != zs.shape or zf.ndim != 4 or zf.shape[1] != len(r):
        raise ValueError('Mismatched Z/readout shape')
    delta = zs.astype(np.float64) - zf.astype(np.float64)
    parallel = r[None, :, None, None] * np.einsum('c,bchw->bhw', r, delta)[:, None] / norm2
    span = (zf.astype(np.float64) + parallel).astype(np.float32)
    null = (zf.astype(np.float64) + delta - parallel).astype(np.float32)
    return span, null, {'weight': r, 'projector': np.outer(r, r)/norm2,
                        'readout_norm': np.sqrt(norm2),
                        'float64_null_dot_max_abs': float(np.abs(np.einsum('c,bchw->bhw', r, delta-parallel)).max())}


def arm_states(s, f, weight):
    span, null, record = projections(f[1], s[1], weight)
    z = (f[1], span, null, s[1])
    return {arm: (w, zz) for arm, (w, zz) in zip(ARMS,
            [(ww, zz) for ww in (f[0], s[0]) for zz in z])}, record


@torch.no_grad()
def handoff_diagnostics(model, data, state, reference, donor, arm):
    model.cuda().eval()
    count = len(data['x'])
    raw, norm, disagreements = [], [], []
    descriptor = []
    stream = sys.modules['stream_cells'].stream
    for start in range(0, count, base.BATCH):
        ids = list(range(start, min(start+base.BATCH, count)))
        indices = ids + [i+count for i in ids]
        x, _, _ = base.inputs(data, ids)
        w, z = (torch.from_numpy(v[indices]).cuda() for v in state)
        expected_z = reference[1] if arm in ('FF', 'SF', 'F-null', 'S-null') else donor[1]
        ze = torch.from_numpy(expected_z[indices]).cuda()
        logits, expected = model.logits((w, z)), model.logits((w, ze))
        error = (logits-expected).abs()
        raw.extend(error.flatten(1).max(1).values.cpu().tolist())
        norm.extend((error.flatten(1).max(1).values/(1+expected.abs().flatten(1).max(1).values)).cpu().tolist())
        pick = x[:, :1].bool()
        disagreements.extend((((logits >= 0) != (expected >= 0)) & pick).flatten(1).sum(1).cpu().tolist())
        old = model._features(w, z, x)
        incoming = stream(w, x[:, :1])
        features = torch.cat((incoming, old[:, model.workspace_channels:]), dim=1)
        force = model.f_out(torch.tanh(model.f_in(features)))
        wn = incoming + model.eta*force
        q = model._candidate(wn, z, x)
        actual = model.step((w, z), x)
        assert torch.equal(actual[0], wn) and torch.equal(actual[1], z+model.alpha*q), 'Instrumented step drift'
        def rms(v):
            m = x[:, :1].double()
            return ((v.double().square()*m).sum((1, 2, 3))/(m.sum((1, 2, 3))*v.shape[1]).clamp_min(1)).sqrt()
        rw, rz, rf, rq = (rms(v) for v in (w, z, force, q))
        dw = w - torch.from_numpy(reference[0][indices]).cuda()
        dz = z - torch.from_numpy(reference[1][indices]).cuda()
        rdw, rdz = rms(dw), rms(dz)
        for j, index in enumerate(indices):
            descriptor.append({'cue': 'original' if index < count else 'flipped', 'map_index': index % count,
                'W_rms': float(rw[j]), 'Z_rms': float(rz[j]), 'F_rms': float(rf[j]), 'Q_rms': float(rq[j]),
                'delta_W_vs_F_rms': float(rdw[j]), 'delta_Z_vs_F_rms': float(rdz[j]),
                'Z_over_W': float(rz[j]/rw[j]) if rw[j] > 0 else None,
                'F_over_W': float(rf[j]/rw[j]) if rw[j] > 0 else None,
                'Q_over_Z': float(rq[j]/rz[j]) if rz[j] > 0 else None})
    model.cpu()
    neutral = arm in ('FF', 'SF', 'F-null', 'S-null')
    numerical = max(norm) <= 1e-6
    exact_decisions = sum(disagreements) == 0
    passed = numerical and exact_decisions
    if arm in ('FF', 'SF'):
        assert max(raw) == 0 and passed, 'W substitution changed Z-only output'
    return {'reference_output': 'FF' if neutral else 'F1_readout_Z_S', 'readout_neutral_arm': neutral,
            'max_absolute_logit_error': max(raw), 'max_normalized_logit_error': max(norm),
            'open_decision_disagreements': int(sum(disagreements)),
            'max_absolute_by_batch_cue_map': raw, 'max_normalized_by_batch_cue_map': norm,
            'decision_disagreements_by_batch_cue_map': disagreements,
            'readout_property_qualified': passed, 'instrumented_step_exact': True, 'scale_per_map': descriptor}


@torch.no_grad()
def suffix(model, data, state):
    """Reuse the frozen suffix while recording norms and checking all states."""
    count = len(data['x']); norms = np.empty((193, 2, count, 2), np.float64)
    original_logits = model.logits
    calls = 0
    def logits(current):
        nonlocal calls
        block, t = divmod(calls, 193)
        start = block*base.BATCH; num = min(base.BATCH, count-start)
        if num <= 0:
            raise AssertionError('Suffix observation order drift')
        m = data['mask'][start:start+num].cuda()
        m = torch.cat((m, m))
        for j, value in enumerate(current):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f'Nonfinite state at absolute step {64+t}')
            values = ((value.double().square()*m).sum((1,2,3))/(m.sum((1,2,3))*value.shape[1]).clamp_min(1)).sqrt()
            norms[t, :, start:start+num, j] = values.cpu().numpy().reshape(2,num)
        calls += 1
        return original_logits(current)
    model.logits = logits
    try:
        result = base.paired_trace(model, data, state, start_time=64)
        assert calls == 193*((count+base.BATCH-1)//base.BATCH)
        return (*result, norms)
    finally:
        model.logits = original_logits
        model.cpu()


def export_table(out, records, qualification):
    fields = ['size', 'arm', 'cohort', 'metric', 'numerator', 'denominator', 'pooled', 'map_mean', 'valid_maps']
    with (out/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for arm, record in records.items():
            for size, result in record['sizes'].items():
                if result.get('status') == 'NONFINITE_CONTINUATION':
                    continue
                for cohort, values in result['cohorts'].items():
                    for metric, values in values.items():
                        writer.writerow({'size': size, 'arm': arm, 'cohort': cohort, 'metric': metric,
                            **{k: values['pooled'][v] for k, v in (('numerator','numerator'),('denominator','denominator'),('pooled','value'))},
                            'map_mean': values['map_mean'], 'valid_maps': values['valid_maps']})
    lines = ['# Output-preserving state factorization', '', 'Status: COMPLETE. Zero training/optimizer updates.',
             f"Primary episode: {qualification['primary_status']}. Confirmation: {qualification['confirmation_status']}.", '',
             '| Size | Arm | Keep continuous | Prog immediate | Prog sustained | Prog delayed sustained | Rescue proxy | Neutral output |',
             '|---:|---|---:|---:|---:|---:|---|---|']
    def fmt(x): return 'null' if x is None else f'{x:.4f}'
    for size in ('32', '64'):
        for arm in ARMS:
            result = records[arm]['sizes'][size]
            if result.get('status') == 'NONFINITE_CONTINUATION':
                lines.append(f'| {size} | {arm} | NONFINITE | | | | False | |'); continue
            c = result['cohorts']; diag = records[arm]['handoff'][size]
            vals = [size, arm, fmt(c['keep']['continuous64_256']['pooled']['value']),
                    *[fmt(c['prog'][k]['pooled']['value']) for k in ('immediate64','sustained241_256','delayed_sustained')],
                    records[arm]['rescue'][size]['pass'], diag['readout_property_qualified'] if diag['readout_neutral_arm'] else 'visible']
            lines.append('| '+' | '.join(map(str, vals))+' |')
    lines += ['', 'Read summary.json and metrics.csv first. Native Full, transfer/localization proxy and',
              'output neutrality are separate qualifications. Cohorts/denominators are fixed before intervention.',
              'Hybrid failures do not establish a W/Z consistency relation; successes are conditional state effects.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def figure(out, records):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for row, size in enumerate(('32','64')):
        for col, (cohort, key, title) in enumerate((('keep','continuous64_256','Continuous preservation'),
                 ('prog','sustained241_256','Sustained progress'),('prog','delayed_sustained','Delayed sustained progress'))):
            values = [records[a]['sizes'][size].get('cohorts',{}).get(cohort,{}).get(key,{}).get('pooled',{}).get('value') for a in ARMS]
            axes[row,col].bar(ARMS, [np.nan if v is None else v for v in values])
            axes[row,col].set_ylim(0,1.05); axes[row,col].set_title(f'Size{size}: {title}')
            axes[row,col].tick_params(axis='x', rotation=40)
    fig.suptitle('Fixed F1 consumer; fixed native T64 cohorts; no training')
    fig.savefig(out/'state_factorization.png', dpi=160); fig.savefig(out/'state_factorization.pdf'); plt.close(fig)


def run(out, qualification):
    base.setup()
    bindings, rows = sources(), catalog()
    q = read(qualification)
    assert q['status'] == 'PASS' and q['protocol'] == PROTOCOL
    assert q['source_sha256'] == bindings and q['checkpoints'] == rows
    assert out.parent == ROOT/'runs' and not out.exists(), 'New direct run directory required'
    out.mkdir(); begun = time.monotonic(); completed = 0
    manifest = {'protocol': PROTOCOL, 'status': 'RUNNING', 'training': False, 'training_updates': 0,
                'pid': os.getpid(), 'host': os.environ.get('COMPUTERNAME'), 'command': sys.argv,
                'started_utc': base.now(), 'source_sha256': bindings, 'checkpoints': rows,
                'torch_version': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0),
                'runtime_limit_enforced': False, 'watchdog_enabled': False, 'continuous_monitoring': False}
    write(out/'manifest.json', manifest)
    def status(phase, **extra):
        record = {'protocol': PROTOCOL, 'status': 'RUNNING', 'phase': phase, 'pid': os.getpid(),
                  'updated_utc': base.now(), 'completed_cells': completed, 'expected_cells': 16, **extra}
        write(out/'status.json', record); print(record, flush=True)
    try:
        for rel in bindings:
            p = out/'source'/rel; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/rel, p)
        status('banks')
        data, banks, exclusion = make_banks(out)
        manifest.update(banks=banks, excluded_bank_sha256=exclusion); write(out/'manifest.json', manifest)
        config = {'protocol': PROTOCOL, 'arms': ARMS, 'banks': banks, 'models': rows,
                  'consumer': 'F1', 'handoff': 64, 'last_step': 256, 'primary_size': 32,
                  'confirmation_size': 64, 'batch_maps': 8, 'cues': ['original','flipped'],
                  'training_updates': 0, 'runtime_limit_enforced': False,
                  'backend': {'cudnn_benchmark': False, 'cudnn_deterministic': False,
                              'cudnn_tf32': True, 'matmul_tf32': False}}
        write(out/'config.json', config)
        models = {k: base.load_model(row) for k, row in rows.items()}
        states, native, native_full = {}, {}, {}
        for mid, model in models.items():
            states[mid] = {}; traces = {}; scores = {}
            for size, d in data.items():
                status('native', model=mid, size=size)
                trace, endpoints, snapshot, logits = base.paired_trace(model, d)
                traces[size], scores[size], states[mid][size] = trace, endpoints, snapshot
                folder = out/'native'; folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'{mid}_size{size}_trace.npz', **trace)
                np.savez_compressed(folder/f'{mid}_size{size}_logits.npz', **{'T'+t:v for t,v in logits.items()})
                folder = out/'states'; folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'{mid}_size{size}_T64.npz', W=snapshot[0], Z=snapshot[1])
            phenotype = base.summarize_from_traces(traces, scores, data)
            frontier = phenotype.pop('_matched_frontier_rows'); phenotype['phenotype_gate'] = base.predicate(phenotype)
            base._write_frontier_csv(out/'native'/f'{mid}_frontier.csv', frontier)
            write(out/'native'/f'{mid}_phenotype.json', phenotype)
            native[mid] = traces; native_full[mid] = phenotype['phenotype_gate']['pass']
        cohorts, all_states = {}, {}
        weight = models['F1'].readout.weight.detach().cpu().numpy().reshape(-1)
        for size, d in data.items():
            cohorts[size] = metrics.cohorts(native['S1'][size]['correct'][64], native['F1'][size]['correct'][64], d['changed'][:,0].numpy().astype(bool))
            (out/'cohorts').mkdir(exist_ok=True)
            np.savez_compressed(out/'cohorts'/f'size{size}.npz', **cohorts[size])
            all_states[size], projection = arm_states(states['S1'][size], states['F1'][size], weight)
            np.savez_compressed(out/f'projection_size{size}.npz', **projection)
        snapshot_hashes = {mid: {size: base.tensor_hash({'W':torch.from_numpy(s[0]),'Z':torch.from_numpy(s[1])})
                           for size,s in snapshots.items()} for mid,snapshots in states.items()}
        manifest['snapshot_sha256'] = snapshot_hashes
        manifest['snapshot_cue_order'] = 'all_original_maps_then_all_flipped_maps'
        write(out/'manifest.json', manifest)
        records = {arm: {'arm': arm, 'sizes': {}, 'handoff': {}, 'rescue': {}} for arm in ARMS}
        for arm in ARMS:
            for size, d in data.items():
                status('intervention', arm=arm, size=size)
                diag = handoff_diagnostics(models['F1'], d, all_states[size][arm], states['F1'][size], states['S1'][size], arm)
                records[arm]['handoff'][str(size)] = diag
                folder = out/'arms'; folder.mkdir(exist_ok=True)
                try:
                    trace, endpoints, _, logits, norms = suffix(models['F1'], d, all_states[size][arm])
                    if arm == 'FF':
                        for key in trace:
                            assert np.array_equal(trace[key], native['F1'][size][key][64:]), 'FF/native suffix mismatch'
                    result = metrics.summarize(trace['correct'], cohorts[size])
                    result['endpoint_scores'] = endpoints
                    records[arm]['sizes'][str(size)] = result
                    np.savez_compressed(folder/f'{arm}_size{size}_trace.npz', **trace, times=np.arange(64,257))
                    np.savez_compressed(folder/f'{arm}_size{size}_logits.npz', **{'T'+t:v for t,v in logits.items()})
                    np.savez_compressed(folder/f'{arm}_size{size}_norms.npz', norms=norms, times=np.arange(64,257),
                                        cues=np.asarray(['original','flipped']), fields=np.asarray(['W_rms','Z_rms']))
                except FloatingPointError as exc:
                    records[arm]['sizes'][str(size)] = {'status':'NONFINITE_CONTINUATION','reason': str(exc)}
                completed += 1
                write(folder/f'{arm}.json', records[arm])
                status('cell_complete', arm=arm, size=size)
        for arm in ARMS:
            for size in ('32','64'):
                result = records[arm]['sizes'][size]
                if result.get('status') == 'NONFINITE_CONTINUATION':
                    records[arm]['rescue'][size] = {'pass':False,'reason':'nonfinite'}
                else:
                    gate = metrics.rescue_gate(result, records['FF']['sizes'][size])
                    if records[arm]['handoff'][size]['readout_neutral_arm']:
                        gate['output_neutrality_qualified'] = records[arm]['handoff'][size]['readout_property_qualified']
                        gate['pass'] = gate['pass'] and gate['output_neutrality_qualified']
                    records[arm]['rescue'][size] = gate
            write(out/'arms'/f'{arm}.json', records[arm])
        for k, model in models.items():
            assert base.tensor_hash(model.state_dict()) == rows[k]['parameter_sha256'], f'Frozen parameters changed: {k}'
            for size, state in states[k].items():
                assert base.tensor_hash({'W':torch.from_numpy(state[0]),'Z':torch.from_numpy(state[1])}) == snapshot_hashes[k][size]
        qualified = {}
        for size in ('32','64'):
            qualified[size] = native_full['S1'] and not native_full['F1'] and records['SS']['rescue'][size]['pass']
        qualification_record = {'native_full_flags': native_full,
            'primary_status': 'QUALIFIED_LOCALIZATION_EPISODE' if qualified['32'] else 'LOCALIZATION_EPISODE_UNQUALIFIED',
            'confirmation_status': 'QUALIFIED' if qualified['64'] else 'UNQUALIFIED',
            'anchor_requires': ['native_S1_Full','native_F1_not_Full','SS_rescue_proxy_including_gain_over_FF']}
        summary = {'protocol': PROTOCOL, 'status':'COMPLETE','training_updates':0,'completed_cells':completed,
                   'qualification': qualification_record, 'records': records, 'parameters_unchanged':True,
                   'claim_boundary':'Selected-checkpoint finite state-localization interventions; no encoded-information, consistency-relation, or population reliability proof.'}
        export_table(out, records, qualification_record); figure(out, records)
        summary['elapsed_seconds'] = round(time.monotonic()-begun, 3); write(out/'summary.json', summary)
        manifest.update(status='COMPLETE',finished_utc=base.now(),elapsed_seconds=summary['elapsed_seconds'])
        write(out/'manifest.json',manifest)
        write(out/'status.json',{'protocol':PROTOCOL,'status':'COMPLETE','pid':os.getpid(),'phase':'complete',
             'completed_cells':completed,'expected_cells':16,'elapsed_seconds':summary['elapsed_seconds'],'updated_utc':base.now()})
    except BaseException as exc:
        write(out/'status.json',{'protocol':PROTOCOL,'status':'ERROR','pid':os.getpid(),'completed_cells':completed,
              'updated_utc':base.now(),'error':repr(exc),'traceback':traceback.format_exc()})
        traceback.print_exc(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True); p.add_argument('--qualification', required=True)
    args = p.parse_args()
    run((ROOT/args.out).resolve(), (ROOT/args.qualification).resolve())
