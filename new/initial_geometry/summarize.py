"""Render the frozen zero-training diagnostic, without new inference."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True)
    run = Path(ap.parse_args().run)
    s = json.loads((run/'summary.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    assert s['status'] == 'COMPLETE' and s['training_updates'] == 0
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    for item in list(manifest['banks'].values()) + list(manifest['checkpoints'].values()):
        digest = item.get('sha256', item.get('file_sha256'))
        assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest() == digest
    lines = ['# Historical update-zero geometry: diagnostic', '',
        f"COMPLETE: four exact initial checkpoints, four reused banks per seed; zero training; {s['elapsed_seconds']:.2f} CPU seconds.", '',
        'Initial parameters match the original historical hashes. Tangent/autograd checks at full8/full16/detached16 pass;',
        f"maximum absolute error {max(x['max_error'] for r in s['seeds'].values() for x in r['sanity']):.3g}.",
        'No parameter changed. Source/first-training-batch/schedule bindings pass.', '',
        'All initial state Jacobians are the same transport direct-sum identity by construction.',
        'This diagnostic measures random task features, not that state Jacobian or a learned continuation quotient.', '',
        '## Primary size32: fixed definitions', '',
        '| Seed | Raw lane RMS max/min | Centered max/min | Mean absolute lane cosine | Initial Q-feature effective rank | K8@64 available pair separation | K8@64 relation-kernel effective rank | K8@64 label alignment |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed,r in s['seeds'].items():
        b = r['banks']['bank_primary32']; lane = b['lane']; k = b['windows'][-1]['K8']
        lines.append(f"| {seed} | {lane['raw_max_min']:.3f} | {lane['centered_max_min']:.3f} | {lane['mean_abs_cosine']:.3f} | {b['features']['0']['feature_covariance']['effective_rank']:.3f} | {k['available_map_mean_relative']:.4f} | {k['spectrum']['effective_rank']:.3f} | {k['spectrum']['label_alignment']:.5f} |")
    lines += ['', 'Available pair separation is the per-map mean of ||H-H_flip|| divided by the average norm of H/H_flip,',
        'restricted to targets with cue accessibility in that K8 window. It is not accuracy or an invariant information measure.',
        'Bias-heavy features affect this relative normalization. Initial Q-feature covariance is centered over both worlds.',
        'Relation kernel uses paired differences, so bias terms cancel.', '',
        '## Bank consistency: K8 tangent at endpoint64', '',
        '| Bank | Seed | Available pair separation | Effective rank | Nonzero eigenvalue condition | Label alignment |',
        '|---|---:|---:|---:|---:|---:|']
    for name in s['seeds']['4']['banks']:
        for seed,r in s['seeds'].items():
            k = r['banks'][name]['windows'][-1]['K8']; sp = k['spectrum']
            lines.append(f"| {name} | {seed} | {k['available_map_mean_relative']:.4f} | {sp['effective_rank']:.3f} | {sp['nonzero_condition']:.2f} | {sp['label_alignment']:.5f} |")
    lines += ['', '## First historical batch: original K8 loss gradients', '',
        '| Seed | Q-out weight gradient L2 | First singular value | Second singular value |',
        '|---:|---:|---:|---:|']
    for seed,r in s['seeds'].items():
        g = r['first_batch_K8']; sv = g['qout_gradient_singular_values']
        lines.append(f"| {seed} | {g['gradient_l2']['q_out.weight']:.8f} | {sv[0]:.8f} | {sv[1]:.3g} |")
        assert all(v == 0 for n,v in g['gradient_l2'].items() if n not in ('q_out.weight','q_out.bias','readout.bias'))
    lines += ['', 'Encoder, F, Q-in and readout.weight task gradients are exactly zero at initialization.',
        'Q-out weight gradient is rank one up to FP32 roundoff; its output direction is the readout vector.',
        'The balanced-loss bias gradients are near-zero numerical cancellation. This is not an AdamW update analysis.', '',
        '## Interpretation and boundaries', '',
        '- Seed4 has the most balanced RAW lane norms on all four banks, but seed3 or seed5 has better centered balance;',
        '  seed4 does not have the smallest cross-lane correlation. Raw balance partly describes means/biases.',
        '- Seed4 has the largest measured available relative separation at endpoint64, on all four banks,',
        '  while initial Q-feature effective rank is highest for seed5. Favorable spectrum rankings depend on bank/window.',
        '- Seed3 has the highest endpoint64 K8 paired kernel-label alignment on all four banks.',
        '  Larger separation and higher effective rank do not entail easier target learning.',
        '- Full-history tangents and actual K8 tangents differ; both are retained at all eight windows in summary.json.',
        '- Four reused seeds and inspected banks cannot establish statistical anomaly, a successful-initialization criterion,',
        '  causality, or reproducibility. Same seed4 initialization already failed the full phenotype under four new schedules.',
        '- No source-flip feature difference occurs outside the explicitly computed cue-accessibility upper bound.',
        '  Initial inability to distinguish a remote cue before arrival is expected locality, not representation failure.', '',
        'Read [summary.json](summary.json) for every time/window/bank and per-map denominator. Feature NPZs are secondary.',
        'Definitions and reproduction: [protocol](../../new/initial_geometry/PROTOCOL.md),',
        '[audit code](../../new/initial_geometry/audit.py). No new architecture gate was run.']
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'VALIDATED','report':str(run/'RESULTS.md')}))


if __name__ == '__main__':
    main()
