"""Publish allowlisted seed4 A/B/C evidence; verify public arithmetic without GPU."""
import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/seed4_followup_20261003'
PUBLICATION = ROOT/'SEED4_FOLLOWUP_PUBLICATION_MANIFEST.json'
TRAIN = ROOT/'runs/seed4_followup_20261003_training01'
CAUSAL = ROOT/'runs/seed4_followup_20261003_causal01'
REVIEW = ROOT/'analyses/seed4_followup_20261003_training_review01'
PRIVATE = {'pid', 'host', 'hostname', 'username', 'gpu', 'gpu_uuid', 'device',
           'command', 'argv', 'cwd', 'launch_command', 'working_directory',
           'run_directory', 'preflight_directory', 'executable', 'run', 'out'}
sys.path.insert(0, str(ROOT/'new/seed4_followup'))
from phenotype import predicate


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def public(value):
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k.lower() not in PRIVATE}
    if isinstance(value, list):
        return [public(v) for v in value]
    return value


def compact_training_validation(doc):
    result = copy.deepcopy(doc)
    result['checks'] = [{'name': c['name'], 'pass': c['pass']} for c in doc['checks']]
    for arm in result['arms']:
        arm.pop('sizes', None)
        for size in arm['matched_frontier_csv']['sizes'].values():
            size.pop('per_map', None)
    result['detail_policy'] = (
        'Compact copy: check names/verdicts, provenance, checkpoints/traces, gates and CSV counts retained. '
        'Duplicate metric profiles and successful-check detail payloads omitted; published arm summaries '
        'retain the profiles. The complete local validation report SHA is in provenance.json.')
    return result


def equal(a, b):
    if a is None or b is None:
        assert a is b, (a, b)
    else:
        assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11), (a, b)


def components(summary):
    checks = summary['phenotype_gate']['checks']
    return {
        'reach': all(v['pass'] for k, v in checks.items() if k.startswith('reach_')),
        'hold': all(v['pass'] for k, v in checks.items() if k.startswith('hold_')),
        'dynamics': all(v['pass'] for k, v in checks.items() if k.startswith(('size32_', 'size64_')) and 'frontier' not in k),
        'frontier': all(v['pass'] for k, v in checks.items() if 'frontier' in k),
        'full_phenotype': summary['phenotype_gate']['pass'],
    }


def row(name, arm, summary):
    e = summary['sizes']['32']['endpoints']['64']
    result = {'name': name, 'experiment': arm['experiment'], 'schedule_seed': arm['schedule_seed'],
            'direction_seed': arm['direction_seed'], 'epsilon': arm['epsilon'], **components(summary),
            'T64_strict_map_mean': e['strict_16_32']['mean_map_coverage'],
            'T64_strict_pooled': e['strict_16_32']['pooled_coverage']['value'],
            'T64_ba_original': e['original_ba'], 'T64_ba_flipped': e['flipped_ba'],
            'failed_checks': [k for k, v in summary['phenotype_gate']['checks'].items() if not v['pass']]}
    for size in ('32', '64'):
        s = summary['sizes'][size]
        result['size'+size+'_ever_regressed_ratio'] = s['ever_regressed_over_ever_correct']['value']
        result['size'+size+'_retention64_to256'] = s['transitions']['all_changed']['64_to_256']['pooled']['retention']['value']
        result['size'+size+'_coverage_gain64_to256'] = s['endpoints']['256']['all_changed']['pooled_coverage']['value']-s['endpoints']['64']['all_changed']['pooled_coverage']['value']
        result['size'+size+'_frontier_effect'] = s['frontier']['mean_map_weighted_difference']
    return result


def report(summary):
    lookup = {r['name']: r for r in summary['arms']}
    a, b = lookup['A_schedule20032'], lookup['B_direction70003_eps05']
    lines = ['# Original seed4: independent schedule, initialization and rollback screens', '',
             f"A/B execution COMPLETE: 13 arms x300 updates, {summary['training_elapsed_seconds']:.3f}s ({summary['training_elapsed_seconds']/60:.2f}min).",
             'Historical control: identical initial/final parameters and six full saved endpoints; fresh control phenotype QUALIFIED.', '',
             '| Experiment | Independent unit | Full phenotype passes | Frozen threshold | Result |',
             '|---|---|---:|---:|---|',
             '| A: fixed init, changed schedule | 4 schedules | 0/4 | >=3/4 | NOT_QUALIFIED |',
             '| B: epsilon=.01, fixed training | 4 directions | 0/4 | >=3/4 | NOT_QUALIFIED |',
             '| B: epsilon=.05, same paired directions | 4 directions | 0/4 | >=3/4 | NOT_QUALIFIED |', '',
             'All 12 A/B arms pass both matched-frontier gates, but fail the complete phenotype. Frontier association alone is insufficient for retained useful computation.', '',
             '| Arm | T64 strict mean % | T64 strict pooled % | Reach | Hold | Dynamics | Frontier | Full |',
             '|---|---:|---:|---|---|---|---|---|']
    for r in summary['arms']:
        lines.append(f"| {r['name']} | {100*r['T64_strict_map_mean']:.2f} | {100*r['T64_strict_pooled']:.2f} | {r['reach']} | {r['hold']} | {r['dynamics']} | {r['frontier']} | {r['full_phenotype']} |")
    lines += ['', 'Reach: size32/T64 strict16<d<32 paired mean and pooled>=.80; both open-grid BA>=.85.',
              'Hold: size32/T128 AND T256 drops from T64 <=.03 BA and <=.05 strict paired mean/pooled.',
              'Dynamics: at each spatial size, T64->T256 retention>=.95, coverage gain>=.05, every-step regression/ever-correct<=.15.',
              'Frontier: each size matched effect>=.05, eligible maps>=16, common strata>=100. Empty groups fail.', '',
              f"Schedule20032 passes hold, retention and coverage-gain checks, but reaches only {100*a['T64_strict_map_mean']:.1f}% mean/{100*a['T64_strict_pooled']:.1f}% pooled at T64 and regresses {100*a['size32_ever_regressed_ratio']:.1f}%/{100*a['size64_ever_regressed_ratio']:.1f}% of ever-correct pixels.",
              f"Direction70003/epsilon=.05 passes size32 reach+hold, but fails parts of the size64 dynamics and regresses {100*b['size32_ever_regressed_ratio']:.1f}%/{100*b['size64_ever_regressed_ratio']:.1f}%. No radius monotonicity or basin boundary follows.", '',
              '## C: selected checkpoint local temporal-state intervention', '',
              'Zero training; 93/257 events at size32/64, 31 eligible maps each. Historical six-endpoint replay and numerical controls pass.', '',
              '| Size | Horizon | Native minus sender pp | Wrong-sham minus sender pp | Endpoint |',
              '|---:|---:|---:|---:|---|']
    for size in ('32', '64'):
        for h in ('1', '4'):
            values = summary['causal']['sizes'][size]['horizons'][h]
            lines.append(f"| {size} | {h} | {100*values['native_minus_sender_rollback']:.2f} | {100*values['sham_minus_sender_rollback']:.2f} | {'Primary' if h=='1' else 'Secondary'} |")
    lines += ['', '**NO_PRIMARY_THRESHOLD_SIGNAL**: step1 requires native-minus-sender>=.10 AND sham-minus-sender>=.05 at both sizes. Size32 misses the sham threshold; size64 misses both.',
              'Larger step4 contrasts are secondary and do not rescue this endpoint. The threshold miss is not a statistical null test.',
              'This is a whole-cell W/Z temporal rollback on selected targets, not an edge-specific message knockout. Equal-norm sham does not match phase, direction or activation context. Model n=1, reused historical maps.', '',
              '## Reading order and reproduction', '',
              '1. This report, [interpretation](INTERPRETATION.md), and [compact aggregate](summary.json).',
              '2. [Frozen protocol](../../new/seed4_followup/PROTOCOL.md), [compact training validation](training/validation.json), [causal validation](causal/validation.json), and [public arithmetic check](publication_validation.json).',
              '3. Individual training/*_summary.json files retain every gate, denominator, map/censor profile. raw/ retains original arm records. frontier/ CSVs and causal/events.json are secondary; do not open them first.',
              '4. [Bindings](../../SEED4_FOLLOWUP_PUBLICATION_MANIFEST.json) and [provenance](provenance.json) identify byte-identical and sanitized evidence; NPZ/checkpoints remain local.', '',
              'Public verification and focused CPU checks from repository root:', '',
              '```', 'python tools/export_seed4_followup.py --verify-only',
              'python new/seed4_followup/check.py --out analyses/NEW_FOLLOWUP_CPU.json', '```', '',
              'For a fresh clone, `python tools/prepare_seed4_followup_training_references.py` restores three already-public archival inputs without overwriting an existing archive. Then use the frozen train.py preflight/formal commands in [reproduction notes](REPRODUCTION.md). No checkpoint is needed to retrain A/B.',
              'Exact C replay needs the original excluded checkpoint/source archives. CPU publication checks validate saved arithmetic; they do not repeat GPU inference or training.',
              'Earlier architecture no-go decisions remain unchanged. No general NCA rejection, training basin radius, short-BPTT mechanism or unique handoff law follows.']
    return '\n'.join(lines)+'\n'


INTERPRETATION = """# What this follow-up establishes

The original selected initialization, historical schedule and bank reproduce
the old parameters/endpoints and pass the new full phenotype on fresh maps.
Changing only the schedule does not reproduce the full behavior in four
draws. None of four sampled initialization directions reproduces it at either
paired radius. These are conditional results for one bank and training recipe.

The A/B arms preserve matched-frontier acquisition associations even
when they fail the full gate. Retained correctness, continued coverage, and
low per-step regression are separate requirements. Partial reach or hold
cannot substitute for their conjunction; positive frontier association alone
does not explain the successful seed4 algorithm or short-BPTT credit assignment.

B perturbs all trainable parameters, including initially zero output layers.
It does not keep the zero-output initialization manifold, use layerwise noise,
or perturb final weights. Four directions at two radii are paired observations,
not eight independent replicates. No certified basin radius, monotone boundary,
or probability of architecture success is established.

C estimates effects of a whole-cell W/Z rollback on selected wrong targets
with newly correct neighbors. The wrong-neighbor sham matches blockwise norms,
but not direction, phase or activation context. Off-cone/no-op controls qualify
the implementation. A failed one-step threshold is not proof of zero causal
effect. Four-step secondary contrasts are larger and remain secondary. These
results do not identify a unique sender, semantic handoff rule or flood-fill.

All earlier architecture no-go verdicts stand. This upload preserves negative
and partial outcomes and makes no new architecture recommendation.
"""

REPRODUCTION = """# Reproduction and available evidence

From the repository root, install requirements.txt. The frozen runtime was
PyTorch2.5.1 with FP32 CUDA, threads2, cudnn benchmark/deterministic False,
cudnn TF32 True, and matmul TF32 False. The protocol specifies all seeds,
sample sizes, optimizer, state dimensions, clocks, gates and caps.

## Public CPU verification

```
python tools/export_seed4_followup.py --verify-only
python new/seed4_followup/check.py --out analyses/NEW_FOLLOWUP_CPU.json
```

The first command verifies source/evidence/tool hashes, gate arithmetic,
matched integer CSV strata and equal-map C contrasts. The second runs the
focused initialization/metric/intervention fixtures. Neither trains a model
or performs GPU inference. training/validation.json is a compact record of
the full local Boolean-trace and CPU-checkpoint validation. It retains check
names/verdicts, all gate values and provenance; duplicate metric profiles and
successful-check details remain local, with the full report SHA bound in
provenance.json. Public arithmetic is narrower.

## A/B training on a fresh clone

```
python tools/prepare_seed4_followup_training_references.py
python new/seed4_followup/check.py --out analyses/NEW_FOLLOWUP_CPU.json
python new/seed4_followup/train.py --preflight --qualification analyses/NEW_FOLLOWUP_CPU.json --out runs/NEW_FOLLOWUP_PREFLIGHT
python new/seed4_followup/train.py --out runs/NEW_FOLLOWUP_TRAINING --preflight-dir runs/NEW_FOLLOWUP_PREFLIGHT
```

Use new output names. The preparation helper copies only the published
historical manifest, schedule and seed4 record into the expected archival
location; it refuses an existing directory. A/B do not require a checkpoint.
The runner retains a strict historical parameter/endpoint replay gate and
stops on drift. Exact replay across different GPU/backend configurations is
not guaranteed by these local checks. No training was repeated for upload.

## C and full local validation

The frozen C runner requires the original trained checkpoint and historical
source/run archives named by the protocol. They are excluded from GitHub.
For a local archive, use causal.py --out runs/NEW_CAUSAL and then
tools/validate_seed4_followup_causal.py with --run/--out. Full A/B saved-trace
validation uses tools/validate_seed4_followup_training.py; see its entry
arguments. Public C events retain coordinates, two-world logits, controls and
norm matching, and can reproduce reported equal-map contrasts without a model.

Raw arm records, model summaries, matched CSVs, schedules and C events are
preserved as secondary evidence. Checkpoints, full NPZ trajectories, source
snapshots, machine/process metadata, receipts and logs remain local. Their
hashes are bound in provenance.json. Read RESULTS.md and summary.json first.
"""


def verify(out, bind=True):
    if bind:
        publication = read(PUBLICATION)
        for name, expected in publication['source_sha256'].items():
            assert sha(ROOT/name) == expected, name
        for name, expected in publication['causal_source_sha256'].items():
            assert sha(ROOT/name) == expected, name
        for name, expected in publication['publication_tool_sha256'].items():
            assert sha(ROOT/name) == expected, name
        for name, expected in publication['published_evidence_sha256'].items():
            assert sha(ROOT/name) == expected, name
    summary = read(out/'summary.json')
    aggregate = read(out/'training/aggregate.json')
    assert aggregate['complete'] and aggregate['completed_arms'] == 13
    assert len(summary['arms']) == 13 and aggregate['control_fresh_qualification'] == 'QUALIFIED'
    strata, models = 0, 0
    for r in summary['arms']:
        saved = read(out/f'training/{r["name"]}_summary.json')
        assert predicate(saved) == saved['phenotype_gate']
        assert components(saved) == {k: r[k] for k in components(saved)}
        assert r['full_phenotype'] == aggregate['phenotype_pass'][r['name']]
        terms = defaultdict(list)
        unique = set()
        with (out/f'frontier/{r["name"]}.csv').open(newline='', encoding='utf-8') as handle:
            for line in csv.DictReader(handle):
                size, m, t, d = (int(line[k]) for k in ('size', 'map_index', 'start_step', 'distance'))
                key = (size, m, t, d)
                assert key not in unique and size in (32, 64) and t in range(0, 249, 8)
                unique.add(key)
                nf, af, nn, an = (int(line[k]) for k in ('frontier_opportunities', 'frontier_acquired', 'nonfrontier_opportunities', 'nonfrontier_acquired'))
                assert nf > 0 and nn > 0 and 0 <= af <= nf and 0 <= an <= nn
                weight, difference = nf*nn/(nf+nn), af/nf-an/nn
                equal(weight, float(line['matched_weight']))
                equal(difference, float(line['rate_difference']))
                terms[(str(size), m)].append((weight, difference))
                strata += 1
        for size in ('32', '64'):
            f = saved['sizes'][size]['frontier']
            effects = []
            for m in f['per_map']:
                values = terms.get((size, m['map_index']), [])
                effect = sum(w*d for w, d in values)/sum(w for w, _ in values) if values else None
                equal(effect, m['weighted_difference'])
                assert len(values) == m['common_strata']
                if effect is not None:
                    effects.append(effect)
            assert len(effects) == f['eligible_maps']
            equal(sum(effects)/len(effects) if effects else None, f['mean_map_weighted_difference'])
            assert sum(len(v) for (s, _), v in terms.items() if s == size) == f['common_strata']
        models += 1
    assert sum(r['full_phenotype'] for r in summary['arms'] if r['experiment'] == 'A') == aggregate['A']['passes'] == 0
    for eps in (.01, .05):
        selected = [r for r in summary['arms'] if r['experiment']=='B' and r['epsilon']==eps]
        assert len(selected) == 4 and sum(r['full_phenotype'] for r in selected) == aggregate['B'][str(eps)]['passes'] == 0
    causal = read(out/'causal/summary.json')
    events = read(out/'causal/events.json')['events']
    assert len(events) == 350 and len({e['event_id'] for e in events}) == 350
    primary = True
    for size in ('32', '64'):
        groups = defaultdict(list)
        for e in events:
            if str(e['size']) == size:
                groups[e['map_index']].append(e)
        assert len(groups) == causal['sizes'][size]['eligible_map_count'] == 31
        for h in ('1', '4'):
            means = {condition: sum(sum(e['outcomes'][h][condition]['paired_acquisition'] for e in group)/len(group) for group in groups.values())/len(groups)
                     for condition in ('native', 'sender_rollback', 'wrong_neighbor_sham')}
            record = causal['sizes'][size]['horizons'][h]
            equal(means['native']-means['sender_rollback'], record['native_minus_sender_rollback'])
            equal(means['wrong_neighbor_sham']-means['sender_rollback'], record['sham_minus_sender_rollback'])
        primary &= causal['sizes'][size]['horizons']['1']['native_minus_sender_rollback'] >= .10 and causal['sizes'][size]['horizons']['1']['sham_minus_sender_rollback'] >= .05
    assert primary == causal['primary_signal_thresholds_met'] == False
    return {'status': 'PASS', 'gpu_execution': False, 'training': False,
            'model_summaries': models, 'common_frontier_strata': strata, 'causal_events': len(events),
            'scope': 'Public gate/CSV/event arithmetic and hash bindings; full Boolean traces and CPU checkpoint hashes were verified locally.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(OUT), indent=2))
        return
    assert not OUT.exists(), 'Keep existing publication immutable'
    status, manifest = read(TRAIN/'status.json'), read(TRAIN/'manifest.json')
    assert status['status'] == 'COMPLETE' and status['completed_arms'] == 13
    assert read(REVIEW/'validation.json')['status'] == 'PASS'
    assert read(ROOT/'analyses/seed4_followup_20261003_causal_publication01/validation.json')['status'] == 'PASS'
    OUT.mkdir(parents=True, exist_ok=False)
    copied, original_hashes = {}, {}
    def deliver(source, relative, sanitize=False, compact=False):
        target = OUT/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        original_hashes[relative] = sha(source)
        if sanitize:
            doc = public(read(source))
            if compact:
                doc = compact_training_validation(doc)
            if 'cpu_qualification' in doc and doc['cpu_qualification']:
                doc['cpu_qualification'].pop('path', None)
            write(target, doc)
        else:
            shutil.copyfile(source, target)
            assert sha(source) == sha(target)
            copied[relative] = sha(target)
    aggregate = read(TRAIN/'aggregate.json')
    rows, traces = [], {}
    deliver(TRAIN/'aggregate.json', 'training/aggregate.json')
    for name in aggregate['phenotype_pass']:
        arm, saved = read(TRAIN/f'{name}.json'), read(TRAIN/f'{name}_summary.json')
        rows.append(row(name, arm, saved))
        deliver(TRAIN/f'{name}.json', f'raw/{name}.json')
        deliver(TRAIN/f'{name}_summary.json', f'training/{name}_summary.json')
        deliver(TRAIN/f'{name}_matched_frontier_strata.csv', f'frontier/{name}.csv')
        traces[name] = {p.name: sha(p) for p in (TRAIN/f'{name}.pt', TRAIN/f'{name}_size32.npz', TRAIN/f'{name}_size64.npz')}
    for seed in (20002, 20012, 20022, 20032, 20042):
        deliver(TRAIN/f'schedule{seed}.json', f'training/schedule{seed}.json')
    deliver(TRAIN/'manifest.json', 'training/manifest.json', True)
    deliver(TRAIN/'status.json', 'training/completion.json', True)
    deliver(REVIEW/'validation.json', 'training/validation.json', True, True)
    deliver(REVIEW/'arm_metrics.csv', 'training/arm_metrics.csv')
    deliver(ROOT/'analyses/seed4_followup_20261003_publication_cpu01.json', 'cpu_validation.json', True)
    pf = ROOT/'runs/seed4_followup_20261003_preflight01'
    for name in ('aggregate.json', 'manifest.json', 'status.json'):
        deliver(pf/name, f'preflight/{name}', True)
    for name, target in (('compact_summary.json', 'summary.json'), ('events.json', 'events.json'),
                         ('replaychecks.json', 'replaychecks.json'), ('RESULTS.md', 'RESULTS.md')):
        deliver(CAUSAL/name, f'causal/{target}')
    deliver(CAUSAL/'compact_summary.json', 'causal/compact_summary.json')
    deliver(CAUSAL/'manifest.json', 'causal/manifest.json', True)
    deliver(CAUSAL/'status.json', 'causal/completion.json', True)
    deliver(ROOT/'analyses/seed4_followup_20261003_causal_publication01/validation.json', 'causal/validation.json', True)
    summary = {'review_base': manifest['git_base'], 'execution_status': 'COMPLETE',
               'training_elapsed_seconds': status['elapsed_seconds'], 'A': aggregate['A'], 'B': aggregate['B'],
               'control_fresh_qualification': aggregate['control_fresh_qualification'], 'arms': rows,
               'causal': read(CAUSAL/'compact_summary.json'),
               'scope': 'Conditional selected initialization, one bank/recipe; four schedules and four paired directions, one selected C checkpoint.'}
    write(OUT/'summary.json', summary)
    (OUT/'RESULTS.md').write_text(report(summary), encoding='utf-8')
    (OUT/'INTERPRETATION.md').write_text(INTERPRETATION, encoding='utf-8')
    (OUT/'REPRODUCTION.md').write_text(REPRODUCTION, encoding='utf-8')
    write(OUT/'provenance.json', {
        'review_base': manifest['git_base'], 'original_artifacts_sha256': original_hashes,
        'byte_identical_copies_sha256': copied, 'local_checkpoints_traces_sha256': traces,
        'sanitization': 'Machine/process/command/path fields removed from metadata and validation copies. Training validation additionally omits duplicate profiles and successful-check detail payloads; its detail_policy identifies this compacting. All scientific counts/gates remain in arm summaries.',
        'excluded': ['checkpoints', 'NPZ traces', 'source snapshots', 'logs', 'launch receipts', 'private metadata originals'],
        'verification': 'Full local CPU trace/checkpoint validation plus narrower public summary/CSV/event arithmetic. No inference or training for publication.'})
    write(OUT/'publication_validation.json', verify(OUT, False))
    for path in OUT.rglob('*.json'):
        doc = read(path)
        assert public(doc) == doc, path
        assert not re.search(r'[A-Za-z]:[\\/]|/home/|192\.168\.|172\.25\.|Cxxxx|\bchenx\b', path.read_text()), path
    tool_names = ['tools/export_seed4_followup.py', 'tools/validate_seed4_followup_training.py',
                  'tools/validate_seed4_followup_causal.py', 'tools/prepare_seed4_followup_training_references.py']
    write(PUBLICATION, {'review_base': manifest['git_base'], 'execution_status': 'COMPLETE',
        'training_arm_count': 13, 'updates_per_arm': 300, 'A_passes': 0, 'B_passes_by_radius': {'.01': 0, '.05': 0},
        'C_primary_signal': False, 'source_sha256': manifest['source_sha256'],
        'causal_source_sha256': read(CAUSAL/'manifest.json')['source_sha256'],
        'publication_tool_sha256': {name: sha(ROOT/name) for name in tool_names},
        'published_evidence_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
        'notes': ['Earlier architecture no-go decisions unchanged.', 'A n=4 schedules; B n=4 paired directions, no basin-radius theorem.',
                  'C n=1 selected checkpoint, step4 secondary cannot rescue failed step1.', 'No new GPU inference/training for delivery.']})
    print(json.dumps({'status': 'EXPORTED', 'files': len(list(OUT.rglob('*'))),
                      'bytes': sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))


if __name__ == '__main__':
    main()
