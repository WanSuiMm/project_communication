"""Build or verify the immutable public trajectory-qualification evidence bundle."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/trajectory_20261004_serial01'
PUBLIC = ROOT / 'evidence/trajectory_qualification_20261004'
MANIFEST = ROOT / 'TRAJECTORY_QUALIFICATION_PUBLICATION_MANIFEST.json'
ADAPTER = 'tools/launch_trajectory_qualification.ps1'
PRIVATE_SOURCE = 'evidence/short_bptt_phase2_init2345/manifest.json'
PRIVATE_KEYS = {'pid', 'gpu', 'gpu_uuid', 'gpu_id', 'device_uuid', 'hostname',
                'host_name', 'host', 'username', 'user', 'cwd',
                'working_directory', 'command', 'launch_command', 'executable'}
PRIVATE_TEXT = re.compile(
    r'(?i)(\b[A-Z]:[\\/]|\\\\[^\\]+\\[^\\]+|'
    r'(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){2,3}\b)')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(a, b):
    assert (a is None) == (b is None)
    if a is not None:
        assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10), (a, b)


def check_ratios(value):
    if isinstance(value, dict):
        if {'numerator', 'denominator', 'value'} <= value.keys():
            n, d = value['numerator'], value['denominator']
            equal(value['value'], None if d == 0 else n / d)
        for item in value.values():
            check_ratios(item)
    elif isinstance(value, list):
        for item in value:
            check_ratios(item)


def private_fields(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in PRIVATE_KEYS:
                found.add(str(key))
            found.update(private_fields(item))
    elif isinstance(value, list):
        for item in value:
            found.update(private_fields(item))
    elif isinstance(value, str) and PRIVATE_TEXT.search(value):
        found.add('private-looking text')
    return found


def safe_json(path):
    value = read(path)
    assert not private_fields(value), (str(path), sorted(private_fields(value)))
    return value


def safe_text(path):
    assert not PRIVATE_TEXT.search(Path(path).read_text(encoding='utf-8-sig')), str(path)


def phenotype_gate(summary):
    gate = summary['phenotype_gate']
    for row in gate['checks'].values():
        value, threshold = row['value'], row['threshold']
        expected = value is not None and (
            value >= float(threshold[2:]) if threshold.startswith('>=')
            else value <= float(threshold[2:]))
        assert row['pass'] == expected
    assert gate['pass'] == all(row['pass'] for row in gate['checks'].values())


def pair_counts(summary):
    rows = summary['per_pair_passes']
    assert len(rows) == 16
    b = sum(bool(x['baseline_pass']) for x in rows)
    t = sum(bool(x['treatment_pass']) for x in rows)
    wins = sum(x['treatment_pass'] and not x['baseline_pass'] for x in rows)
    losses = sum(x['baseline_pass'] and not x['treatment_pass'] for x in rows)
    p = 1.0 if wins + losses == 0 else min(
        1.0, 2 * sum(math.comb(wins + losses, k) for k in range(min(wins, losses) + 1)) /
        (2 ** (wins + losses)))
    r = summary['primary_paired_result']
    assert r['n_pairs'] == len(rows) and r['baseline_passes'] == b and r['treatment_passes'] == t
    assert r['treatment_only_passes'] == wins and r['baseline_only_passes'] == losses
    assert r['discordant_pairs'] == wins + losses
    equal(r['exact_two_sided_mcnemar_binomial_p'], p)
    assert r['ties_both_pass'] == sum(x['baseline_pass'] and x['treatment_pass'] for x in rows)
    assert r['ties_both_fail'] == sum(not x['baseline_pass'] and not x['treatment_pass'] for x in rows)
    return r


def source_checks(original):
    for rel, digest in original.items():
        archived = RUN / 'source' / rel
        live = ROOT / rel
        assert sha(archived) == digest, rel
        if rel not in (ADAPTER, PRIVATE_SOURCE):
            assert sha(live) == digest, rel
    old = (RUN / 'source' / ADAPTER).read_text(encoding='utf-8-sig').splitlines()
    new = (ROOT / ADAPTER).read_text(encoding='utf-8-sig').splitlines()
    assert len(old) == len(new) and [i for i, (a, b) in enumerate(zip(old, new)) if a != b] == [26, 27]
    assert new[26].strip() == '$pythonPath=(Get-Command python -CommandType Application | Select-Object -First 1).Source'
    assert new[27].strip().startswith('if (-not $pythonPath)')
    safe_text(ROOT / ADAPTER)
    return sha(ROOT / ADAPTER)


def copy(src, dst, raw):
    src, dst = Path(src), Path(dst)
    try:
        key = src.relative_to(RUN).as_posix()
    except ValueError:
        key = 'external/' + src.name
    raw[key] = sha(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def validate_public():
    m = read(MANIFEST)
    assert not private_fields(m)
    assert sha(Path(__file__)) == m['exporter_sha256']
    for rel, digest in m['published_evidence_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    assert sha(ROOT / ADAPTER) == m['source_adaptations'][0]['published_sha256']
    excluded_sources = {x['path'] for x in m['source_snapshot_exclusions']}
    for rel, digest in m['source_sha256'].items():
        if rel not in excluded_sources and rel != ADAPTER:
            assert sha(ROOT / rel) == digest, rel
        snapshot = RUN / 'source' / rel
        if snapshot.is_file():
            assert sha(snapshot) == digest, rel
    for rel, digest in m['reference_sha256'].items():
        if (ROOT / rel).is_file():
            assert sha(ROOT / rel) == digest, rel
    for path in PUBLIC.rglob('*.json'):
        value = safe_json(path)
        check_ratios(value)
        if 'phenotype_gate' in value:
            phenotype_gate(value)
    s1 = read(PUBLIC / 'stage1_state_cross/summary.json')
    sel = read(PUBLIC / 'stage1_state_cross/selection.json')
    assert len(sel['lookup']) == 8
    for arm, passed in sel['lookup'].items():
        assert read(PUBLIC / f'stage1_state_cross/evaluations/{arm}_summary.json')['phenotype_gate']['pass'] == passed
    s2 = read(PUBLIC / 'stage2_fresh_recipe/summary.json')
    p = pair_counts(s2)
    s3 = read(PUBLIC / 'stage3_distance/summary.json')
    assert s1['status'] == s2['status'] == s3['status'] == 'COMPLETE'
    assert len(s1['arms']) == 8 and len(s2['per_pair_passes']) == 16 and len(s3['arms']) == 8
    assert sum(1 for _ in PUBLIC.rglob('*.npz')) == 96
    assert sum(1 for _ in PUBLIC.rglob('*_matched_frontier_strata.csv')) == 40
    for arm in s3['arms']:
        ev = read(PUBLIC / f"stage3_distance/evaluations/{arm['arm_id']}_evaluation.json")
        assert ev['gate']['status'] in ('PASS', 'FAIL')
        assert ev['gate']['status'] == ('PASS' if all(ev['gate']['checks'].values()) else 'FAIL')
    print(json.dumps({'status': 'PASS', 'arms': 48, 'stage2_pairs': p['n_pairs'],
                      'trace_banks': 96, 'frontier_csvs': 40,
                      'published_files': len(m['published_evidence_sha256'])}))


def build(validation_path):
    assert not PUBLIC.exists() and not MANIFEST.exists(), 'Refusing to overwrite published evidence'
    status, root_m = read(RUN / 'status.json'), read(RUN / 'manifest.json')
    assert status['status'] == 'COMPLETE' and all(x['status'] == 'COMPLETE' for x in status['stages'].values())
    validation = safe_json(validation_path)
    assert validation.get('status') == 'PASS'
    adapter_sha = source_checks(root_m['source_sha256'])
    for rel, digest in root_m['reference_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    s1 = RUN / 'stage1_state_cross'; s2 = RUN / 'stage2_fresh_recipe'; s3 = RUN / 'stage3_distance'
    m1, m2, m3 = (read(p / 'manifest.json') for p in (s1, s2, s3))
    a1, a2, a3 = (read(p / 'summary.json') for p in (s1, s2, s3))
    assert [a1['completed_arms'], 2 * len(a2['per_pair_passes']), a3['complete_arms']] == [8, 32, 8]
    pair_counts(a2)
    raw = {}
    for p in (RUN / 'status.json', RUN / 'manifest.json', s1 / 'manifest.json',
              s2 / 'manifest.json', s3 / 'manifest.json'):
        raw[p.relative_to(RUN).as_posix()] = sha(p)
    PUBLIC.mkdir(parents=True)
    for stage in ('stage1_state_cross', 'stage2_fresh_recipe', 'stage3_distance'):
        (PUBLIC / stage).mkdir()
    # Full scientific summaries are copied byte-for-byte after privacy checks.
    for name, summary in ((s1 / 'summary.json', a1), (s2 / 'summary.json', a2), (s3 / 'summary.json', a3)):
        assert not private_fields(summary)
        copy(name, PUBLIC / name.parent.name / 'summary.json', raw)
    sel = safe_json(s1 / 'selection.json')
    copy(s1 / 'selection.json', PUBLIC / 'stage1_state_cross/selection.json', raw)
    copy(validation_path, PUBLIC / 'validation.json', raw)
    # Configurations use field allowlists; runtime receipts and device fields stay local.
    configs = [
        (m1, ('protocol', 'expected_arms', 'start_update', 'end_update', 'initialization_seed', 'optimizer',
              'step_counter_preserved', 'arms', 'reference_checkpoint_sha256', 'train_data_sha256',
              'evaluation_data_sha256', 'evaluation_banks_reused', 'runtime_limit_enforced')),
        (m2, ('protocol', 'selection', 'recipe_components_imported', 'pair_count', 'initialization_seeds',
              'schedule_seeds', 'training_bank', 'evaluation_banks', 'fixed_shadow_prefix_indices',
              'data_sha256', 'common_sha256', 'runner_sha256')),
        (m3, ('task', 'architecture', 'model', 'train', 'optimizer', 'evaluation', 'gate', 'claim_boundary',
              'train_data_sha256', 'evaluation_data_sha256', 'torch_version', 'numpy_version', 'backend'))]
    for folder, (manifest, keys) in zip(('stage1_state_cross', 'stage2_fresh_recipe', 'stage3_distance'), configs):
        src = {'stage1_state_cross': s1, 'stage2_fresh_recipe': s2, 'stage3_distance': s3}[folder] / 'manifest.json'
        raw[src.relative_to(RUN).as_posix()] = sha(src)
        write(PUBLIC / folder / 'config.json', {k: manifest[k] for k in keys if k in manifest})
    # Stage 1: eight full summaries, curves, traces and frontier strata.
    for p in s1.glob('*_summary.json'):
        v = safe_json(p); phenotype_gate(v)
        copy(p, PUBLIC / 'stage1_state_cross/evaluations' / p.name, raw)
        arm = p.name[:-len('_summary.json')]
        training = safe_json(s1 / f'{arm}.json')
        raw[f'stage1_state_cross/{arm}.json'] = sha(s1 / f'{arm}.json')
        write(PUBLIC / 'stage1_state_cross/training' / f'{arm}.json', {
            'name': arm, 'training_curve': training['training_curve'],
            'checkpoint_sha256': training['checkpoint']['sha256'],
            'parameter_sha256': training['checkpoint']['parameter_sha256']})
        for suffix in ('_size32.npz', '_size64.npz', '_matched_frontier_strata.csv'):
            copy(s1 / f'{arm}{suffix}', PUBLIC / 'stage1_state_cross' /
                 ('traces' if suffix.endswith('.npz') else 'frontier') / f'{arm}{suffix}', raw)
    for n in ('suffixH_schedule.json', 'suffixS_schedule.json'):
        copy(s1 / n, PUBLIC / 'stage1_state_cross/schedules' / n, raw)
    # Stage 2: paired records, audit hashes, and original artifact files from the run manifest.
    copy(s2 / 'perarm.json', PUBLIC / 'stage2_fresh_recipe/perarm.json', raw)
    copy(s2 / 'schedules.json', PUBLIC / 'stage2_fresh_recipe/schedules.json', raw)
    perarm = safe_json(s2 / 'perarm.json')
    audit = []
    for row in perarm:
        key = f"{row['pair_id']}_{row['arm']}"
        write(PUBLIC / 'stage2_fresh_recipe/training' / f'{key}.json', {
            'pair_id': row['pair_id'], 'arm': row['arm'], 'training_curve': row['training_curve'],
            'shadow_prefix_curve': row['shadow_prefix_curve']})
        audit.append({'pair_id': row['pair_id'], 'arm': row['arm'], 'u3_payload_hashes': row['u3_payload_hashes'],
                      'u3_components': row['u3_components'], 'checkpoint_hashes': {
                          k: {q: v[q] for q in ('sha256', 'parameter_sha256')}
                          for k, v in row['checkpoints'].items()}})
    write(PUBLIC / 'stage2_fresh_recipe/component_audit.json', audit)
    for key, art in m2['artifacts'].items():
        for rel in [art['evaluation_summary'], *art['evaluation_traces'], art['frontier_strata']]:
            src = s2 / rel
            if rel.endswith('_summary.json'):
                phenotype_gate(safe_json(src))
                dest = PUBLIC / 'stage2_fresh_recipe/evaluations' / Path(rel).name
            elif rel.endswith('.npz'):
                dest = PUBLIC / 'stage2_fresh_recipe/traces' / Path(rel).name
            else:
                dest = PUBLIC / 'stage2_fresh_recipe/frontier' / Path(rel).name
            copy(src, dest, raw)
    # Stage 3: full evaluations, training curves, schedules, checkpoint hashes and traces.
    copy(s3 / 'arms.json', PUBLIC / 'stage3_distance/arms.json', raw)
    for p in s3.glob('*_evaluation.json'):
        safe_json(p); copy(p, PUBLIC / 'stage3_distance/evaluations' / p.name, raw)
    for p in (s3 / 'curves').glob('*.json'):
        safe_json(p); copy(p, PUBLIC / 'stage3_distance/training' / p.name, raw)
    for p in s3.glob('*.npz'):
        copy(p, PUBLIC / 'stage3_distance/traces' / p.name, raw)
    for p in s3.glob('*.npy'):
        copy(p, PUBLIC / 'stage3_distance/schedules' / p.name, raw)
    # Bind raw source and reference hashes; only the portable launcher adapter differs.
    run_summary = {'status': 'COMPLETE', 'stages': {
                       '1': {'status': a1['status'], 'completed_arms': a1['completed_arms']},
                       '2': {'status': a2['status'], 'completed_arms': 2 * len(a2['per_pair_passes'])},
                       '3': {'status': a3['status'], 'completed_arms': a3['complete_arms']}},
                   'total_arms': 48,
                   'trace_banks': 96, 'frontier_csvs': 40}
    write(PUBLIC / 'summary.json', run_summary)
    (PUBLIC / 'RESULTS.md').write_text(results_text(a1, sel, a2, a3), encoding='utf-8')
    (PUBLIC / 'REPRODUCTION.md').write_text(reproduction_text(), encoding='utf-8')
    for folder, content in {
        'stage1_state_cross': f"Status: {a1['status']}; arms: {a1['completed_arms']}. Read `summary.json`, `selection.json`, `config.json`, then per-arm evaluations and training curves.\n",
        'stage2_fresh_recipe': f"Status: {a2['status']}; arms: {2 * len(a2['per_pair_passes'])}; pairs: {a2['primary_paired_result']['n_pairs']}; exact two-sided McNemar p: {a2['primary_paired_result']['exact_two_sided_mcnemar_binomial_p']}. Read `summary.json`, `config.json`, `schedules.json`, then `perarm.json` and evaluations.\n",
        'stage3_distance': f"Status: {a3['status']}; arms: {a3['complete_arms']}; decision: {a3['decision']}. Read `summary.json`, `config.json`, then per-arm evaluations and training curves.\n"}.items():
        (PUBLIC / folder / 'RESULTS.md').write_text(content, encoding='utf-8')
    safe_json(PUBLIC / 'summary.json'); safe_text(PUBLIC / 'RESULTS.md'); safe_text(PUBLIC / 'REPRODUCTION.md')
    files = {p.relative_to(ROOT).as_posix(): sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
    manifest = {
        'protocol': 'trajectory_qualification_publication_v1', 'run_id': RUN.name,
        'exporter_sha256': sha(Path(__file__)), 'published_evidence_sha256': files,
        'raw_evidence_sha256': raw, 'source_sha256': root_m['source_sha256'],
        'source_adaptations': [{'path': ADAPTER, 'reason': 'Portable Python executable lookup only',
                                'executed_sha256': root_m['source_sha256'][ADAPTER],
                                'published_sha256': adapter_sha}],
        'source_snapshot_exclusions': [{'path': PRIVATE_SOURCE,
                                        'sha256': root_m['source_sha256'][PRIVATE_SOURCE],
                                        'reason': 'Archived source manifest contains machine metadata'}],
        'reference_sha256': root_m['reference_sha256'],
        'counts': {'arms': 48, 'stage1_arms': 8, 'stage2_arms': 32, 'stage2_pairs': 16,
                   'stage3_arms': 8, 'trace_banks': 96, 'frontier_csvs': 40},
        'excluded': ['checkpoint binaries', 'PIDs', 'private host fields', 'absolute paths',
                     'logs', 'launch receipts']}
    write(MANIFEST, manifest)
    validate_public()


def results_text(s1, selection, s2, s3):
    out = ['# Trajectory qualification: recorded data', '',
           '| Stage | Status | Arms | Recorded decision |', '|---|---|---:|---|',
           f"| 1: state cross | {s1['status']} | {s1['completed_arms']} | selector in `stage1_state_cross/selection.json` |",
           f"| 2: fresh recipe | {s2['status']} | {2 * len(s2['per_pair_passes'])} | {s2['primary_paired_result']['verdict']} |",
           f"| 3: distance task | {s3['status']} | {s3['complete_arms']} | {s3['decision']} |", '',
           '## Stage 1 lookup', '', '| Arm | Evaluation pass | Lookup |', '|---|---:|---:|']
    for name, value in selection['lookup'].items():
        out.append(f"| {name} | {value} | {value} |")
    p = s2['primary_paired_result']
    out += ['', '## Stage 2 paired counts', '',
            f"16 pairs; baseline passes {p['baseline_passes']}; treatment passes {p['treatment_passes']}; "
            f"treatment-only {p['treatment_only_passes']}; baseline-only {p['baseline_only_passes']}; "
            f"discordant {p['discordant_pairs']}; exact two-sided McNemar p={p['exact_two_sided_mcnemar_binomial_p']}.", '',
            '## Stage 3 per-arm gates and MAE (cells)', '',
            '| Arm | Gate checks | 32 T64 | 32 T256 | 64 T64 | 64 T256 | d>32 MAE T64 | d>32 MAE T256 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in s3['arms']:
        e = arm['evaluation']; c32 = e['sizes']['32']['checkpoints']; c64 = e['sizes']['64']['checkpoints']
        passed = sum(bool(x) for x in e['gate']['checks'].values())
        a, b = c64['64']['d_gt_32']['mean_eligible_map_mae_cells'], c64['256']['d_gt_32']['mean_eligible_map_mae_cells']
        out.append(f"| {arm['arm_id']} | {passed}/8 {e['gate']['status']} | {fmt(c32['64']['mean_map_mae_cells'])} | "
                   f"{fmt(c32['256']['mean_map_mae_cells'])} | {fmt(c64['64']['mean_map_mae_cells'])} | "
                   f"{fmt(c64['256']['mean_map_mae_cells'])} | {fmt(a)} | {fmt(b)} |")
    out += ['', '## Data reading order', '',
            '1. `summary.json` and this table for arm and pair counts.',
            '2. Each stage `summary.json`, then its `config.json` and fixed schedules.',
            '3. Per-arm evaluation JSON and training curves.',
            '4. `component_audit.json` for U3 component/checkpoint hashes.',
            '5. The 96 compressed NPZ trace banks and 40 frontier CSVs are the larger raw evidence files.']
    return '\n'.join(out) + '\n'


def fmt(value):
    return 'NA' if value is None else f'{value:.4f}'


def reproduction_text():
    return '''# Verification and reproduction

From the repository root, run `python -X utf8 -B tools/export_trajectory_qualification.py --verify-only`.
This checks published SHA-256 bindings, source/reference hashes when available,
ratio consistency, saved gate decisions, stage counts, and stage 2 paired counts
and exact p-value with Python's standard library. It does not train or evaluate a model.

To rebuild the publication from the completed local run, pass the sanitized
validation artifact: `python -X utf8 -B tools/export_trajectory_qualification.py
--validation analyses/trajectory_publication_validation_20261004.json`.
The portable launcher resolves `python` from PATH; its runner still requires
the recorded Torch 2.5.1 environment. Exact stage 1 reproduction also needs the
archived initialization/reference checkpoints identified by the published
hashes. Checkpoint binaries and launch receipts are excluded, so a fresh clone
can verify the saved evidence but requires those local archives to rerun stage 1.
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--validation')
    args = parser.parse_args()
    if args.verify_only:
        validate_public()
    elif not args.validation:
        parser.error('--validation is required when building the publication')
    else:
        build(args.validation)
