"""Compact descriptive reports; no architecture gate or causal inference."""
import json
from pathlib import Path

import numpy as np


def passage_times(correct, margin):
    ever = correct.any(axis=0)
    first = np.where(ever, correct.argmax(axis=0), -1)
    wrong = ~correct
    last_wrong = np.where(wrong.any(axis=0), len(correct)-1-wrong[::-1].argmax(axis=0), -1)
    stable = np.where(correct[-1], last_wrong+1, -1)
    confident = correct & (margin > .1)
    first_confident = np.where(confident.any(axis=0), confident.argmax(axis=0), -1)
    relapse = (correct[:-1] & ~correct[1:]).any(axis=0)
    return first, stable, first_confident, relapse


def timing(values):
    observed = values[values >= 0]
    return {'pixels': int(values.size), 'observed': int(observed.size),
            'censored': int(values.size-observed.size),
            'sum_observed_steps': int(observed.sum()),
            'median_observed_step': float(np.median(observed)) if observed.size else None,
            'min_observed_step': int(observed.min()) if observed.size else None,
            'max_observed_step': int(observed.max()) if observed.size else None}


def ratio(n, d):
    return {'numerator': int(n), 'denominator': int(d), 'value': float(n/d) if d else None}


def transition(correct, mask, a, b):
    per_map = []
    for i in range(len(mask)):
        before, after = correct[a, i][mask[i]], correct[b, i][mask[i]]
        retained = int((before & after).sum())
        lost = int((before & ~after).sum())
        gained = int((~before & after).sum())
        row = {'map_index': i, 'pixels': int(before.size), 'from_correct': int(before.sum()),
               'to_correct': int(after.sum()), 'retained': retained, 'lost': lost, 'gained': gained}
        row['retention'] = ratio(retained, retained+lost)
        row['relapse'] = ratio(lost, retained+lost)
        row['acquisition'] = ratio(gained, int((~before).sum()))
        per_map.append(row)
    totals = {k: sum(row[k] for row in per_map)
              for k in ('pixels', 'from_correct', 'to_correct', 'retained', 'lost', 'gained')}
    totals['retention'] = ratio(totals['retained'], totals['from_correct'])
    totals['relapse'] = ratio(totals['lost'], totals['from_correct'])
    totals['acquisition'] = ratio(totals['gained'], totals['pixels']-totals['from_correct'])
    nonempty = [row['retention']['value'] for row in per_map if row['retention']['value'] is not None]
    return {'from_step': a, 'to_step': b, 'pooled': totals, 'per_map': per_map,
            'mean_map_retention': float(np.mean(nonempty)) if nonempty else None}


def build_summary(traces, records, behaviors, replay, elapsed, data_cpu):
    result = {'protocol': 'original_seed4_frontier_behavior_v1', 'training': False,
              'model_seed': 4, 'model_replications': 1, 'maps_per_size': 32,
              'trace_every_step': True, 'last_step': 256, 'replay': replay,
              'elapsed_at_summary_seconds': elapsed, 'sizes': {},
              'claim_boundary': 'Selected-checkpoint output behavior through T256; no causal algorithm, '
                                'latent closure, training basin or architecture reliability claim.'}
    for size in (32, 64):
        key = str(size)
        values, data = traces[key], data_cpu[size]
        mask = data['changed'][:, 0].numpy().astype(bool)
        distance = data['distance'][:, 0].numpy()
        c = values['correct']
        first, stable, confident, relapse = passage_times(c, values['margin'])
        strict = mask & (distance > 16) & (distance < 32)
        pairs = [(t-8, t) for t in range(8, 257, 8)] + [(64, 128), (64, 256), (128, 256)]
        row = {'changed_pixels': int(mask.sum()), 'first_correct': timing(first[mask]),
               'terminal_stable_through256': timing(stable[mask]),
               'first_margin_gt0_1': timing(confident[mask]),
               'ever_regressed_after_being_correct': ratio((relapse & mask).sum(), ((first >= 0) & mask).sum()),
               'per_map': [], 'distance_profile': [], 'transitions': {}, 'endpoints': {}}
        for i in range(len(mask)):
            row['per_map'].append({'map_index': i, 'changed_pixels': int(mask[i].sum()),
                'first_correct': timing(first[i][mask[i]]),
                'terminal_stable_through256': timing(stable[i][mask[i]]),
                'first_margin_gt0_1': timing(confident[i][mask[i]]),
                'ever_regressed': ratio((relapse[i] & mask[i]).sum(), ((first[i] >= 0) & mask[i]).sum())})
        for d in sorted(np.unique(distance[mask]).tolist()):
            selected = mask & (distance == d)
            row['distance_profile'].append({'distance': int(d), 'pixels': int(selected.sum()),
                'first_correct': timing(first[selected]), 'terminal_stable_through256': timing(stable[selected]),
                'first_margin_gt0_1': timing(confident[selected]),
                'ever_regressed': ratio((relapse & selected).sum(), ((first >= 0) & selected).sum())})
        for name, selected in (('all_changed', mask), ('strict_16_32', strict)):
            row['transitions'][name] = [transition(c, selected, a, b) for a, b in pairs]
        for t in (64, 128, 256):
            rec = records[key][str(t)]
            row['endpoints'][str(t)] = {'paired': rec['paired'],
                'strict_16_32': rec['bands']['strict_16_32'], 'far_gt32': rec['bands']['far_gt32'],
                'original_ba': rec['original']['balanced_accuracy'],
                'flipped_ba': rec['flipped']['balanced_accuracy']}
        # The detailed analyzer is an independently owned implementation.
        frontier_maps = []
        for m in behaviors[key]['maps']:
            matched = [f['distance_matched_weighted_difference'] for f in m['frontier']]
            weight = sum(f['weight_sum'] or 0 for f in matched)
            effect = sum((f['weight_sum'] or 0)*(f['value'] or 0) for f in matched)/weight if weight else None
            frontier_maps.append({'map_index': m['map_index'], 'weight_sum': weight,
                'strata_used': sum(f['strata_used'] for f in matched), 'weighted_difference': effect})
        eligible = [m['weighted_difference'] for m in frontier_maps if m['weighted_difference'] is not None]
        row['frontier'] = {'acquisition_horizon': 't to t+1 at t=0,8,...,248', 'per_map': frontier_maps,
            'eligible_maps': len(eligible), 'mean_map_weighted_difference': float(np.mean(eligible)) if eligible else None,
            'median_map_weighted_difference': float(np.median(eligible)) if eligible else None,
            'strata_used': sum(m['strata_used'] for m in frontier_maps),
            'by_start': [{k: v for k, v in f.items() if k != 'map_values'}
                         for f in behaviors[key]['frontier_overall_by_interval']]}
        result['sizes'][key] = row
    return result


def percent(value):
    return 'null' if value is None else f'{100*value:.2f}%'


def save_report(out, summary, traces, data_cpu):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    lines = ['# Original Streaming seed4: behavior audit', '',
        'Zero training, one selected checkpoint, original 32 maps at each size32/64. '
        'Every step0..256 was recorded. All six historical endpoint payloads replayed successfully.', '',
        '## Endpoint replay', '', '|Size|T|Primary strict16<d<32 pooled|All changed pooled|d>32 pooled|',
        '|---|---:|---:|---:|---:|']
    for size, row in summary['sizes'].items():
        for t, rec in row['endpoints'].items():
            lines.append(f"|{size}|{t}|{percent(rec['strict_16_32']['pooled_accuracy'])}|"
                         f"{percent(rec['paired']['pooled_accuracy'])}|{percent(rec['far_gt32']['pooled_accuracy'])}|")
    lines += ['', '## Retention versus turnover', '',
              '|Size|Scope|Steps|Previously correct|Lost|Retained fraction|Gained|',
              '|---|---|---|---:|---:|---:|---:|']
    for size, row in summary['sizes'].items():
        for scope, rows in row['transitions'].items():
            for rec in rows:
                if (rec['from_step'], rec['to_step']) not in ((64, 128), (64, 256), (128, 256)):
                    continue
                p = rec['pooled']
                lines.append(f"|{size}|{scope}|{rec['from_step']}→{rec['to_step']}|{p['from_correct']}|"
                             f"{p['lost']}|{percent(p['retention']['value'])}|{p['gained']}|")
    lines += ['', '## Every-step acquisition', '',
              '|Size|Changed pixels|Ever correct|Never correct through256|Correct at256|Ever regressed / ever correct|',
              '|---|---:|---:|---:|---:|---:|']
    for size, row in summary['sizes'].items():
        r = row['ever_regressed_after_being_correct']
        lines.append(f"|{size}|{row['changed_pixels']}|{row['first_correct']['observed']}|"
                     f"{row['first_correct']['censored']}|{row['terminal_stable_through256']['observed']}|"
                     f"{r['numerator']}/{r['denominator']} ({percent(r['value'])})|")
    lines += ['', 'First passage and terminal stability are separate. Terminal stability means no later error '
              'in the recorded interval ending at256; it does not mean stability forever. '
              'Timing medians in summary.json condition on acquisition, with censor counts alongside.', '',
              '## Frontier association', '',
              'behavior_size32.json / behavior_size64.json contain per-map/time/exact-BFS-distance '
              'frontier and nonfrontier opportunities and acquisitions. Only common strata contribute '
              'to the weighted acquisition difference. This is descriptive association, not a causal '
              'neighbor handoff test; one macro step can use two graph hops.', '',
              '|Size|Eligible maps|Matched map/time/distance strata|Equal-map mean acquisition difference|',
              '|---|---:|---:|---:|']
    for size, row in summary['sizes'].items():
        f = row['frontier']
        effect = f['mean_map_weighted_difference']
        lines.append(f"|{size}|{f['eligible_maps']}|{f['strata_used']}|{100*effect:+.2f} pp|" if effect is not None
                     else f"|{size}|0|0|null|")
    lines += ['', 'Within each map, weight common exact-distance/time strata by '
              'n_frontier*n_nonfrontier/(n_frontier+n_nonfrontier), then give maps equal weight. '
              'Acquisition is t→t+1 at t=0,8,...,248. There is no significance test.', '',
              '## Reading order and limits', '',
              '1. This report, summary.json and the figures below.',
              '2. replay_validation.json and the separately saved CPU validation.',
              '3. behavior_size*.json for denominators and per-map detail; trace_size*.npz is secondary.', '',
              'A positive retention/frontier pattern does not establish flood-fill, an invariant latent '
              'manifold, hidden-state boundedness, an optimization basin, or multi-seed reliability. '
              'All previous architecture gate verdicts remain unchanged.', '',
              '![Distance and rollout profiles](behavior_curves.png)', '',
              '![Preselected map0 acquisition](acquisition_maps.png)', '',
              'Both maps are fixed map0. Gray means outside the changed component or no observed '
              'acquisition in that panel; colors show observed times only.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, size in enumerate((32, 64)):
        row = summary['sizes'][str(size)]
        c, data = traces[str(size)]['correct'], data_cpu[size]
        mask = data['changed'][:, 0].numpy().astype(bool)
        d = data['distance'][:, 0].numpy()
        times = list(range(0, 257, 8))
        for label, selected in [('all changed', mask), ('16<d<32', mask & (d > 16) & (d < 32)),
                                ('d>32', mask & (d > 32))]:
            if selected.any():
                axes[0, col].plot(times, [c[t][selected].mean() for t in times], label=label)
        axes[0, col].set(title=f'Size{size}: paired correctness', xlabel='Macro steps', ylabel='Pooled fraction', ylim=(0, 1.02))
        axes[0, col].legend()
        profile = row['distance_profile']
        for name, label, style in [('first_correct', 'first correct', '-'),
                                   ('terminal_stable_through256', 'stable through256', '--')]:
            axes[1, col].plot([r['distance'] for r in profile],
                             [r[name]['median_observed_step'] if r[name]['observed'] else np.nan for r in profile],
                             style, label=label)
        ax2 = axes[1, col].twinx()
        ax2.plot([r['distance'] for r in profile],
                 [r['terminal_stable_through256']['observed']/r['pixels'] for r in profile],
                 color='gray', alpha=.45, label='correct at256 fraction')
        ax2.set(ylabel='Correct at256 fraction', ylim=(0, 1.02))
        axes[1, col].set(title='Medians among acquired; gray curve shows censoring', xlabel='Source BFS distance', ylabel='Step')
        axes[1, col].legend(loc='upper left')
    fig.savefig(out/'behavior_curves.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for row_index, size in enumerate((32, 64)):
        values = traces[str(size)]
        first, stable, _, relapse = passage_times(values['correct'], values['margin'])
        changed = data_cpu[size]['changed'][0, 0].numpy().astype(bool)
        for j, (array, title) in enumerate(((first[0], 'First paired correct'), (stable[0], 'Stable through256'),
                                           (relapse[0].astype(int), 'Any later regression'))):
            display = np.ma.masked_where(~changed | (array < 0), array)
            cmap = plt.get_cmap('viridis').copy()
            cmap.set_bad('#dddddd')
            image = axes[row_index, j].imshow(display, cmap=cmap, vmin=0, vmax=256 if j < 2 else 1, interpolation='nearest')
            axes[row_index, j].set(title=f'Size{size}, map0: {title}')
            axes[row_index, j].set_axis_off()
            fig.colorbar(image, ax=axes[row_index, j], shrink=.8)
    fig.savefig(out/'acquisition_maps.png', dpi=160)
    plt.close(fig)
