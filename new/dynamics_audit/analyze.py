"""Post-hoc CPU-only summarization of a completed audit; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = {
    'nca_state_matched': ('inertial_seed0', 'inertial_20261001_seed0'),
    'momentum_nca': ('inertial_seed0', 'inertial_20261001_seed0'),
    'masked_state_nca': ('masked_state_seed0', 'masked_state_20261001_seed0'),
    'masked_momentum_nca': ('masked_momentum_seed0', 'masked_momentum_20261001_seed0'),
}


def read(path): return json.loads(path.read_bytes())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(values):
    a = np.array(values, dtype=float)
    return {'count': len(a), 'median': float(np.median(a)), 'minimum': float(a.min()),
            'maximum': float(a.max()), 'p90': float(np.quantile(a, .9))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    run, out = args.run, args.out
    data, manifest, status = read(run/'summary.json'), read(run/'manifest.json'), read(run/'status.json')
    assert status['status'] == data['status'] == 'COMPLETE'
    for rel, expected in manifest['source_sha256'].items():
        assert sha(ROOT/rel) == sha(run/'source'/rel) == expected, rel
    for arm, (_, directory) in REFERENCES.items():
        assert sha(ROOT/'runs'/directory/(arm+'_seed0.pt')) == manifest['checkpoint_sha256'][arm]
    rows, products, open16, replay = [], [], [], []
    perturbation_errors = {kind: {str(e): [] for e in (1e-4, 1e-3)} for kind in ('random', 'estimated_worst')}
    max_batch_error = 0.
    for record in data['records'].values():
        arm, size = record['arm'], record['size']
        old = read(ROOT/'evidence'/REFERENCES[arm][0]/'arms'/(arm+'_seed0.json'))['evaluation'][str(size)]
        assert len(record['curves']) == len(record['dynamics']) == 8
        assert read(run/f'{arm}_size{size}_curves.json')['curves'] == record['curves']
        assert read(run/f'{arm}_size{size}_dynamics.json') == record['dynamics']
        for t, c in record['curves'].items():
            if t in old['curve']:
                error = {key: abs(c[key]-old['curve'][t][key]) for key in
                         ('balanced_accuracy', 'bce', 'content_rms')}
                if c['velocity_rms'] is not None: error['velocity_rms'] = abs(c['velocity_rms']-old['curve'][t]['velocity_rms'])
                error['paired'] = abs(c['paired']-old['paired_source_information'][t]['both_counterfactuals_correct_on_changed_component'])
                replay.append({'arm': arm, 'size': size, 't': int(t), 'errors': error})
            diag = record['dynamics'][t]
            for points in diag['products'].values():
                assert len(points) == 4
                for point in points:
                    assert point['converged'] == (point['relative_singular_residual'] <= .01 and point['start_relative_spread'] <= .02)
                products.extend(points)
            points = diag['products']['open_input_output_K16']; open16.extend(points)
            for kind in perturbation_errors:
                for case in diag['perturbations'][kind]:
                    perturbation_errors[kind][str(case['relative_rms_epsilon'])].extend(case['relative_linearization_error_per_map'])
                    max_batch_error = max(max_batch_error, *case['unperturbed_window_batch_replay_relative_error'])
            rows.append({'arm': arm, 'size': size, 't': int(t), 'ba_all16': c['balanced_accuracy'],
                         'ba_jacobian_maps4': float(np.mean([a['balanced_accuracy'] for a in c['per_map'][:4]])),
                         'bce_all16': c['bce'], 'paired_all16': c['paired'],
                         'finite_time_log_gain_K16': stats([a['finite_time_log_gain'] for a in points]),
                         'sigma_K16': stats([a['sigma_estimate'] for a in points]),
                         'converged_K16': sum(a['converged'] for a in points),
                         'h_open_rms_mean_maps': float(np.mean(c['h_open_rms_per_map'])),
                         'v_open_rms_mean_maps': None if c['v_open_rms_per_map'] is None else float(np.mean(c['v_open_rms_per_map'])),
                         'h_update_open_rms_mean_maps': float(np.mean(c['h_update_open_rms_per_map'])),
                         'h_component_constant_energy_fraction_mean_maps': float(np.mean(c['h_component_constant_fraction_per_map'])),
                         'update_component_constant_energy_fraction_mean_maps': float(np.mean(c['update_component_constant_fraction_per_map'])),
                         'finite_source_flip_logit_rms_median_maps': float(np.median([a['finite_source_flip_logit_rms'] for a in c['per_map']])),
                         'wrong_margin_median_of_maps_with_errors': float(np.median([a['wrong_margin_median'] for a in c['per_map'] if a['wrong_margin_median'] is not None])),
                         'free_momentum_K16': record['free_momentum_baseline']['16']})
    assert len(replay) == 40 and len(rows) == 64 and len(products) == 1536 and len(open16) == 256
    max_error = max(e for r in replay for e in r['errors'].values())
    assert max_error <= 5e-5
    result = {'status': 'VERIFIED_COMPLETED_AUDIT', 'training': False,
              'execution_elapsed_seconds': status['elapsed_seconds'], 'finished_utc': status['finished_utc'],
              'source_and_checkpoint_hashes_match': True, 'independent_replay_comparisons': len(replay),
              'maximum_replay_error': max_error, 'maximum_window_batch_replay_relative_error': max_batch_error,
              'all_products': {'count': len(products), 'converged': sum(a['converged'] for a in products)},
              'open_K16': {'count': len(open16), 'converged': sum(a['converged'] for a in open16),
                           'positive_log_gain_count': sum(a['finite_time_log_gain'] > 0 for a in open16),
                           'log_gain': stats([a['finite_time_log_gain'] for a in open16])},
              'perturbation_relative_errors': {kind: {eps: {**stats(v), 'below_10_percent': sum(x < .1 for x in v)}
                    for eps, v in cases.items()} for kind, cases in perturbation_errors.items()},
              'rows': rows,
              'limits': ['One training seed; Jacobian estimates on fixed four maps.',
                         'Positive finite-window gain is not asymptotic instability.',
                         'Unconverged power iterations are not upper bounds.',
                         'Finite worst-direction perturbations often leave the linear regime.',
                         'Component energies are means of per-map fractions, not pooled-energy fractions.',
                         'No new model execution or experiment performed for this analysis.']}
    out.mkdir(parents=True, exist_ok=False)
    (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps({'run': run.as_posix(),
        'run_summary_sha256': sha(run/'summary.json'), 'manifest_sha256': sha(run/'manifest.json'),
        'analysis_source_sha256': sha(Path(__file__)), 'new_model_execution': False}, indent=2)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'nca_state_matched': 'State, whole grid', 'momentum_nca': 'Momentum, whole grid',
              'masked_state_nca': 'State, masked', 'masked_momentum_nca': 'Momentum, masked'}
    colors = ['#777777', '#0072B2', '#D55E00', '#009E73']
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for col, size in enumerate((32, 64)):
        for (arm, label), color in zip(labels.items(), colors):
            r = [r for r in rows if r['arm'] == arm and r['size'] == size]
            t = [a['t'] for a in r]
            axes[0, col].plot(t, [100*a['ba_all16'] for a in r], color=color, marker='.', label=label)
            gain = [a['finite_time_log_gain_K16']['median'] for a in r]
            axes[1, col].plot(t, gain, color=color)
            axes[1, col].fill_between(t, [a['finite_time_log_gain_K16']['minimum'] for a in r],
                                     [a['finite_time_log_gain_K16']['maximum'] for a in r], color=color, alpha=.07)
            for a, g in zip(r, gain):
                axes[1, col].plot(a['t'], g, 'o', color=color, markerfacecolor=color if a['converged_K16'] == 4 else 'white', markersize=4)
            axes[2, col].plot(t, [a['h_open_rms_mean_maps'] for a in r], color=color, marker='.')
        axes[0, col].set_title(f'{size} x {size} | one training seed')
        axes[0, col].set_ylim(0, 102)
        axes[1, col].axhline(0, color='black', linewidth=.7)
        axes[2, col].set_yscale('log'); axes[2, col].set_xlabel('Rollout step t (window starts here)')
        for row in range(3):
            axes[row, col].axvline(64, color='#aaaaaa', linestyle='--', linewidth=.8)
            axes[row, col].grid(alpha=.15)
    axes[0, 0].set_ylabel('Balanced accuracy %, 16 maps')
    axes[1, 0].set_ylabel('Median log gain / 16, 4 maps')
    axes[2, 0].set_ylabel('Mean open H RMS, 16 maps')
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, .99))
    fig.suptitle('Accuracy deteriorates while local finite-window gain usually declines', y=1.015, fontsize=13)
    fig.text(.5, .015, 'Hollow gain markers: fewer than 4/4 estimates converged. Shading: map min/max, not a confidence interval.\n'
             'Finite-window log gain is not an asymptotic Lyapunov exponent; no criticality crossing is established.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(out/'dynamics_overview.png', dpi=170, bbox_inches='tight')
    fig.savefig(out/'dynamics_overview.pdf', bbox_inches='tight')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'limits')}, indent=2))


if __name__ == '__main__': main()
