"""Independent CPU accounting of saved seed4 trajectories; no model inference."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for name in ('new/nca_inertial_wind_tunnel', 'new/workspace_revision'):
    sys.path.insert(0, str(ROOT/name))
from tasks import bank
from run_revision import tensor_hash


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eq(a, b):
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        assert np.isclose(a, b, atol=1e-12, rtol=1e-12), (a, b)
    else:
        assert a == b, (a, b)


def times(c, margin):
    shape = c.shape[1:]
    first = np.full(shape, -1, dtype=int)
    confident = first.copy()
    suffix_start = np.zeros(shape, dtype=int)
    seen = np.zeros(shape, dtype=bool)
    relapse = seen.copy()
    for t, current in enumerate(c):
        first[(first < 0) & current] = t
        confident[(confident < 0) & current & (margin[t] > .1)] = t
        relapse |= seen & ~current
        seen |= current
        suffix_start[~current] = t+1
    stable = np.where(c[-1], suffix_start, -1)
    return first, stable, confident, relapse


def check_times(stats, selected, values):
    acquired = values[selected & (values >= 0)]
    eq(stats['n'], int(acquired.size))
    eq(stats['sum'], int(acquired.sum()) if acquired.size else None)
    eq(stats['median'], float(np.median(acquired)) if acquired.size else None)


def check_root_times(stats, selected, values):
    acquired = values[selected & (values >= 0)]
    eq(stats['pixels'], int(selected.sum()))
    eq(stats['observed'], int(acquired.size))
    eq(stats['censored'], int(selected.sum()-acquired.size))
    eq(stats['sum_observed_steps'], int(acquired.sum()))
    eq(stats['median_observed_step'], float(np.median(acquired)) if acquired.size else None)
    eq(stats['min_observed_step'], int(acquired.min()) if acquired.size else None)
    eq(stats['max_observed_step'], int(acquired.max()) if acquired.size else None)


def check_ratio(stats, n, d):
    eq(stats['numerator'], int(n))
    eq(stats['denominator'], int(d))
    eq(stats['value'], float(n/d) if d else None)


def bfs(mask, expected, source):
    distance = np.full(mask.shape, -1, dtype=int)
    y, x = map(int, source)
    distance[y, x] = 0
    queue = deque([(y, x)])
    while queue:
        y, x = queue.popleft()
        for a, b in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= a < len(mask) and 0 <= b < len(mask) and mask[a, b] and distance[a, b] < 0:
                distance[a, b] = distance[y, x]+1
                queue.append((a, b))
    assert np.array_equal(distance[mask], expected[mask])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    assert out.is_relative_to(ROOT/'analyses') and not out.exists()
    manifest, summary = read(run/'manifest.json'), read(run/'summary.json')
    assert read(run/'status.json')['status'] == 'COMPLETE'
    assert not manifest['training'] and manifest['model_seed'] == 4
    counts = {'source_bindings': 0, 'output_hashes': 0, 'maps': 0, 'exact_distance_rows': 0,
              'agent_transitions': 0, 'summary_transitions': 0, 'frontier_strata': 0, 'paired_endpoints': 0}
    for relative, expected in manifest['source_sha256'].items():
        assert sha(ROOT/relative) == sha(run/'source'/relative) == expected, relative
        counts['source_bindings'] += 1
    output_manifest = read(run/'output_manifest.json')
    for name, expected in output_manifest['sha256'].items():
        assert sha(run/name) == expected, name
        counts['output_hashes'] += 1
    assert output_manifest['parameters_and_checkpoint_unchanged']
    checkpoint_path = ROOT/'runs/streaming_carry_20261002_init2345/stream_K8_seed4.pt'
    assert sha(checkpoint_path) == manifest['checkpoint_file_sha256']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert checkpoint['seed'] == 4 and checkpoint['variant'] == 'stream' and checkpoint['completed_updates'] == 300
    assert tensor_hash(checkpoint['state_dict']) == manifest['checkpoint_parameter_sha256']
    replay = read(run/'replay_validation.json')
    assert replay['status'] == 'PASS' and replay['size_horizon_records'] == 6
    evaluations = read(run/'evaluations.json')
    for size in (32, 64):
        key = str(size)
        data = bank(size, 32, 40000+size)
        assert tensor_hash(data) == manifest['evaluation_data_sha256'][key]
        mask = data['changed'][:, 0].numpy().astype(bool)
        opened = data['mask'][:, 0].numpy().astype(bool)
        distance = data['distance'][:, 0].numpy()
        with np.load(run/f'trace_size{size}.npz') as saved:
            c, a, b, margin = [saved[k] for k in ('correct', 'original_correct', 'flipped_correct', 'margin')]
        assert c.shape == (257, 32, size, size) and c.dtype == bool
        assert a.dtype == b.dtype == bool and margin.dtype == np.float32
        assert a.shape == b.shape == margin.shape == c.shape and np.isfinite(margin).all()
        assert np.array_equal(c, a & b)
        assert not ((margin > 0) & ~c).any() and not ((margin < 0) & c).any()
        for t in range(257):
            assert not (c[t] & mask & (distance > 2*t)).any()
        first, stable, confident, relapse = times(c, margin)
        behavior = read(run/f'behavior_size{size}.json')
        root = summary['sizes'][key]
        check_root_times(root['first_correct'], mask, first)
        check_root_times(root['terminal_stable_through256'], mask, stable)
        check_root_times(root['first_margin_gt0_1'], mask, confident)
        check_ratio(root['ever_regressed_after_being_correct'], (relapse & mask).sum(), ((first >= 0) & mask).sum())
        frontier_map_effects = []
        total_strata = 0
        for i, row in enumerate(behavior['maps']):
            eq(row['map_index'], i)
            selected, d = mask[i], distance[i]
            assert selected.any() and not (selected & ~opened[i]).any()
            source = np.argwhere(selected & (d == 0))
            assert len(source) == 1
            cue = data['x'][i, 1:3].sum(axis=0).numpy() > 0
            assert (cue & selected).sum() == 1 and cue[tuple(source[0])]
            bfs(selected, d, source[0])
            eq(row['changed_pixels'], int(selected.sum()))
            eq(row['ever_relapse_pixels'], int((relapse[i] & selected).sum()))
            eq(row['loss_events'], int((c[:-1, i] & ~c[1:, i] & selected).sum()))
            local = row['correctness']['all_changed']
            root_local = root['per_map'][i]
            for field, value in [('first_correct', first[i]), ('terminal_stable_through256', stable[i]),
                                 ('first_margin_gt0_1', confident[i])]:
                check_root_times(root_local[field], selected, value)
            check_ratio(root_local['ever_regressed'], (relapse[i] & selected).sum(), ((first[i] >= 0) & selected).sum())
            eq(local['ever_correct_pixels'], int(((first[i] >= 0) & selected).sum()))
            eq(local['never_correct_pixels'], int(((first[i] < 0) & selected).sum()))
            eq(local['terminal_correct_pixels'], int((c[-1, i] & selected).sum()))
            for field, value in [('first_correct_time', first[i]), ('terminal_stable_time', stable[i]),
                                 ('confident_first_correct', confident[i])]:
                check_times(local[field], selected, value)
            for dr in row['correctness']['exact_distance']:
                at = selected & (d == dr['distance'])
                eq(dr['pixels'], int(at.sum()))
                eq(dr['ever_correct_pixels'], int((at & (first[i] >= 0)).sum()))
                eq(dr['terminal_correct_pixels'], int((at & c[-1, i]).sum()))
                for field, value in [('first_correct_time', first[i]), ('terminal_stable_time', stable[i]),
                                     ('confident_first_correct', confident[i])]:
                    check_times(dr[field], at, value)
                counts['exact_distance_rows'] += 1
            for tr in row['transitions']:
                before, after = c[tr['from_step'], i], c[tr['to_step'], i]
                fc, fw = (before & selected).sum(), (~before & selected).sum()
                retain = (before & after & selected).sum()
                loss = (before & ~after & selected).sum()
                gain = (~before & after & selected).sum()
                for k, v in [('fromcorrect', fc), ('fromwrong', fw), ('retained', retain), ('lost', loss), ('gained', gain),
                             ('net_change', gain-loss)]:
                    eq(tr[k], int(v))
                check_ratio(tr['retention'], retain, fc)
                check_ratio(tr['relapse'], loss, fc)
                counts['agent_transitions'] += 1
            map_weight, map_term, map_strata = 0., 0., 0
            for fr in row['frontier']:
                t = fr['from_step']
                eq(fr['to_step'], t+1)
                good = c[t, i] & selected
                neighbor = np.zeros_like(selected)
                neighbor[1:, :] |= good[:-1, :]
                neighbor[:-1, :] |= good[1:, :]
                neighbor[:, 1:] |= good[:, :-1]
                neighbor[:, :-1] |= good[:, 1:]
                wrong = selected & ~c[t, i]
                frontier, nonfrontier = wrong & neighbor, wrong & ~neighbor
                weight_sum, weighted_term, strata = 0., 0., 0
                for dr in fr['exact_distance']:
                    at = selected & (d == dr['distance'])
                    nf, nn = (frontier & at).sum(), (nonfrontier & at).sum()
                    af, an = (frontier & at & c[t+1, i]).sum(), (nonfrontier & at & c[t+1, i]).sum()
                    for k, v in [('frontier_opportunities', nf), ('nonfrontier_opportunities', nn),
                                 ('frontier_acquired', af), ('nonfrontier_acquired', an)]:
                        eq(dr[k], int(v))
                    check_ratio(dr['frontier_acquisition_rate'], af, nf)
                    check_ratio(dr['nonfrontier_acquisition_rate'], an, nn)
                    if nf and nn:
                        w, effect = float(nf*nn/(nf+nn)), float(af/nf-an/nn)
                        eq(dr['matched_weight'], w)
                        eq(dr['matched_rate_difference'], effect)
                        weight_sum += w
                        weighted_term += w*effect
                        strata += 1
                    else:
                        assert dr['matched_weight'] is dr['matched_rate_difference'] is None
                    counts['frontier_strata'] += 1
                fm = fr['distance_matched_weighted_difference']
                eq(fm['strata_used'], strata)
                eq(fm['weight_sum'], weight_sum if weight_sum else None)
                eq(fm['value'], weighted_term/weight_sum if weight_sum else None)
                map_weight += weight_sum
                map_term += weighted_term
                map_strata += strata
            effect = map_term/map_weight if map_weight else None
            rm = root['frontier']['per_map'][i]
            eq(rm['weight_sum'], map_weight)
            eq(rm['strata_used'], map_strata)
            eq(rm['weighted_difference'], effect)
            if effect is not None:
                frontier_map_effects.append(effect)
            total_strata += map_strata
            counts['maps'] += 1
        eq(root['frontier']['mean_map_weighted_difference'],
           float(np.mean(frontier_map_effects)) if frontier_map_effects else None)
        eq(root['frontier']['eligible_maps'], len(frontier_map_effects))
        eq(root['frontier']['strata_used'], total_strata)
        for dr in root['distance_profile']:
            selected = mask & (distance == dr['distance'])
            for field, value in [('first_correct', first), ('terminal_stable_through256', stable),
                                 ('first_margin_gt0_1', confident)]:
                check_root_times(dr[field], selected, value)
            check_ratio(dr['ever_regressed'], (relapse & selected).sum(), ((first >= 0) & selected).sum())
        for scope, rows in root['transitions'].items():
            selected = mask if scope == 'all_changed' else mask & (distance > 16) & (distance < 32)
            for tr in rows:
                for i, p in enumerate(tr['per_map']):
                    before, after = c[tr['from_step'], i], c[tr['to_step'], i]
                    sel = selected[i]
                    for k, value in [('pixels', sel.sum()), ('from_correct', (before & sel).sum()),
                                     ('to_correct', (after & sel).sum()), ('retained', (before & after & sel).sum()),
                                     ('lost', (before & ~after & sel).sum()), ('gained', (~before & after & sel).sum())]:
                        eq(p[k], int(value))
                    check_ratio(p['retention'], p['retained'], p['from_correct'])
                pooled = tr['pooled']
                for field in ('pixels', 'from_correct', 'to_correct', 'retained', 'lost', 'gained'):
                    eq(pooled[field], sum(p[field] for p in tr['per_map']))
                eq(pooled['to_correct']-pooled['from_correct'], pooled['gained']-pooled['lost'])
                check_ratio(pooled['retention'], pooled['retained'], pooled['from_correct'])
                counts['summary_transitions'] += 1
        for t in range(8, 257, 8):
            rec = evaluations[key][str(t)]
            selections = {'paired': mask, 'strict_16_32': mask & (distance > 16) & (distance < 32),
                          'far_gt32': mask & (distance > 32)}
            for name, selected in selections.items():
                stored = rec['paired'] if name == 'paired' else rec['bands'][name]
                den = selected.sum(axis=(1, 2))
                hits = (c[t] & selected).sum(axis=(1, 2))
                eq(stored['per_map_pixels'], den.tolist())
                eq(stored['per_map_correct'], hits.tolist())
                eq(stored['pooled_pixels'], int(den.sum()))
                eq(stored['pooled_correct'], int(hits.sum()))
                eq(stored['pooled_accuracy'], float(hits.sum()/den.sum()) if den.sum() else None)
                counts['paired_endpoints'] += 1
    out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'PASS', 'training': False, 'model_inference': False, 'checks': counts,
              'run_manifest_sha256': sha(run/'manifest.json'), 'summary_sha256': sha(run/'summary.json'),
              'scope': 'Independent saved-trace accounting, bindings, margins, BFS and local frontier counts.'}
    (out/'validation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
