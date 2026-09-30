"""Read a frozen aggregate and write new standalone scientific plots."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.out)
    data = json.loads((run / 'aggregate.json').read_text(encoding='utf-8'))
    if data['status'] == 'RUNNING':
        raise RuntimeError('Wait for the frozen schedule to finish before plotting.')
    out.mkdir(parents=True, exist_ok=False)
    variants = ['cnn', 'nca', 'vit', 'constant', 'learned']
    labels = ['CNN', 'NCA', 'Attention', 'Constant RT', 'Learned RT']
    colors = ['#7e8892', '#bd8741', '#2784ae', '#8663b2', '#dd4c4c']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), layout='constrained')
    plt.rcParams.update({'font.size': 10})
    rows = []
    for col, dim in enumerate([2, 3]):
        ax, timing = axes[0, col], axes[1, col]
        latency_values = []
        for variant, label, color in zip(variants, labels, colors):
            trials = [r for r in data['results'] if r['gate'] == 'A' and r['dim'] == dim
                      and r['variant'] == variant and r['status'] == 'completed']
            if not trials:
                latency_values.append(float('nan'))
                continue
            values = np.array([[r['eval'][d]['accuracy'] for d in ['large_shape_d16', 'd32', 'd64', 'd128']]
                               for r in trials])
            x = np.array([16, 32, 64, 128])
            ax.plot(x, values.mean(0), marker='o', linewidth=1.8, color=color, label=label)
            ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=.09)
            ms = statistics.mean(r['latency']['median_ms'] for r in trials)
            latency_values.append(ms)
            rows.append({'dim': dim, 'variant': variant,
                         'fit_d16_mean': statistics.mean(r['eval']['train_shape_d16']['accuracy'] for r in trials),
                         'large_d16_mean': values[:, 0].mean(), 'd32_mean': values[:, 1].mean(),
                         'd64_mean': values[:, 2].mean(), 'd128_mean': values[:, 3].mean(),
                         'd128_per_seed': [r['eval']['d128']['accuracy'] for r in trials],
                         'latency_mean_ms': ms,
                         'train_seconds_sum': sum(r['training_seconds'] for r in trials),
                         'peak_allocated_mib': max(r['peak_allocated_mib'] for r in trials),
                         'parameters': trials[0]['parameters']})
        ax.axhline(.5, linestyle=':', color='black', linewidth=.9, label='Chance' if col == 0 else None)
        ax.axhline(.85, linestyle='--', color='#555555', linewidth=.8)
        ax.set(xscale='log', xticks=[16, 32, 64, 128], xticklabels=['16', '32', '64', '128'],
               ylim=(0.35, 1.02), xlabel='Source-target distance (cells)', ylabel='Accuracy',
               title=f'{dim}D: fixed large grid; mean and range of 2 seeds')
        ax.grid(alpha=.15)
        if col == 0:
            ax.legend(fontsize=8, loc='best', ncol=2)
        bars = timing.bar(range(5), latency_values, color=colors, width=.65)
        for bar, ms in zip(bars, latency_values):
            if np.isfinite(ms):
                timing.text(bar.get_x() + bar.get_width() / 2, ms, f'{ms:.1f}', ha='center', va='bottom', fontsize=8)
        timing.set(xticks=range(5), xticklabels=labels, ylabel='Synchronized latency (ms/image)',
                   title=f'{dim}D: 8 updates, batch 1, d=128')
        timing.tick_params(axis='x', labelrotation=18)
        timing.set_ylim(0, max(v for v in latency_values if np.isfinite(v)) * 1.18)
        timing.grid(axis='y', alpha=.15)
    fig.suptitle('Reaction-Transport: local 2D / 3D qualification\n240 training updates per model; portable PyTorch implementation', fontsize=13)
    fig.savefig(out / 'gate_a.png', dpi=180)
    fig.savefig(out / 'gate_a.pdf')
    (out / 'summary.json').write_text(json.dumps({'decisions': data['decisions'], 'rows': rows}, indent=2), encoding='utf-8')
    print(json.dumps({'decisions': data['decisions'], 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
