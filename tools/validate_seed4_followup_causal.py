"""CPU-only validation of saved event geometry, logits and equal-map contrasts."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'new/nca_inertial_wind_tunnel'))
sys.path.insert(0, str(ROOT/'new/workspace_revision'))
from tasks import bank
from run_revision import sha, tensor_hash, write


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    assert run.is_relative_to(ROOT/'runs') and out.is_relative_to(ROOT/'analyses') and not out.exists()
    summary, manifest, status = (read(run/name) for name in ('compact_summary.json', 'manifest.json', 'status.json'))
    assert summary['status'] == status['status'] == 'COMPLETE'
    assert summary['parameters_unchanged'] and summary['checkpoint_unchanged']
    assert summary['elapsed_seconds'] <= manifest['maximum_seconds'] == 300
    assert summary['replay']['status'] == 'PASS' and summary['replay']['size_horizon_records'] == 6
    assert summary['replay']['maximum_absolute_error'] == 0
    # Independently anchor the reference records used by this already-frozen run.
    path_pub = read(ROOT/'STREAM_PATH_AUDIT_PUBLICATION_MANIFEST.json')
    carry_pub = read(ROOT/'STREAMING_CARRY_PUBLICATION_MANIFEST.json')
    frontier_pub = read(ROOT/'FRONTIER_AUDIT_PUBLICATION_MANIFEST.json')
    assert sha(ROOT/'STREAM_PATH_AUDIT_PUBLICATION_MANIFEST.json') == manifest['path_audit_publication_manifest_sha256']
    assert sha(ROOT/'STREAMING_CARRY_PUBLICATION_MANIFEST.json') == manifest['carry_publication_manifest_sha256']
    assert sha(ROOT/'FRONTIER_AUDIT_PUBLICATION_MANIFEST.json') == manifest['frontier_publication_manifest_sha256']
    assert all(manifest['source_sha256'][name] == expected for name, expected in frontier_pub['source_sha256'].items())
    checkpoint = ROOT/'runs/streaming_carry_20261002_init2345/stream_K8_seed4.pt'
    assert sha(checkpoint) == manifest['checkpoint_file_sha256'] == carry_pub['checkpoint_sha256'][checkpoint.name]
    for evidence, local, field, publication in (
        ('evidence/stream_path_audit_seed4/raw_conditions.json',
         'runs/stream_path_audit_20261002_seed4_01/raw_conditions.json',
         'historical_raw_conditions_sha256', path_pub),
        ('evidence/streaming_carry_init2345/raw/stream_K8_seed4.json',
         'runs/streaming_carry_20261002_init2345/stream_K8_seed4.json',
         'historical_training_record_sha256', carry_pub)):
        assert sha(ROOT/evidence) == sha(ROOT/local) == manifest[field] == publication['published_evidence_sha256'][evidence]
    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT/name) == sha(run/'source'/name) == expected, name
    payload = read(run/'events.json')
    assert payload['selection_frozen_before_outcomes']
    events = payload['events']
    assert len({e['event_id'] for e in events}) == len(events)
    result = {'status': 'PASS', 'gpu_execution': False, 'training': False,
              'source_bindings': len(manifest['source_sha256']), 'events': len(events), 'sizes': {}}
    signal = True
    for size in (32, 64):
        data = bank(size, 32, 40000+size)
        assert tensor_hash(data) == manifest['evaluation_data_sha256'][str(size)]
        selected = [e for e in events if e['size'] == size]
        groups = {}
        slot_counts = {}
        for event in selected:
            m, t = event['map_index'], event['time']
            assert t in (16, 32, 64, 128)
            slot_counts[(m, t)] = slot_counts.get((m, t), 0)+1
            assert slot_counts[(m, t)] <= 4
            p, q, r, s = (event[k] for k in ('p', 'q', 'r', 's'))
            assert s is not None and len({tuple(p), tuple(q), tuple(r), tuple(s)}) == 4
            assert all(bool(data['changed'][m, 0, y, x]) for y, x in (p, q, r, s))
            d = lambda pos: int(data['distance'][m, 0, pos[0], pos[1]])
            assert d(p) > 8 and d(q) == d(r) == d(p)-1
            assert sum(abs(a-b) for a, b in zip(p, q)) == sum(abs(a-b) for a, b in zip(p, r)) == 1
            assert sum(abs(a-b) for a, b in zip(p, s)) > 2
            assert all(float(data['x'][m, 1:, y, x].abs().sum()) == 0 for y, x in (q, s))
            for world in ('0', '1'):
                for block in ('W', 'Z'):
                    norms = event['rollback_norm_match'][world][block]
                    assert norms['absolute_error'] <= norms['tolerance']
                    close(abs(norms['q_rollback_norm']-norms['applied_sham_norm']), norms['absolute_error'])
            for horizon in ('1', '4'):
                row = event['outcomes'][horizon]
                for name, outcome in row.items():
                    a, b = outcome['original_logit'], outcome['flipped_logit']
                    assert math.isfinite(a) and math.isfinite(b)
                    good = ((a >= 0) == bool(data['y'][m, 0, p[0], p[1]] >= .5)) and ((b >= 0) == bool(data['y_flip'][m, 0, p[0], p[1]] >= .5))
                    assert good == outcome['paired_correct'] and int(good) == outcome['paired_acquisition']
                assert row['native'] == row['noop_reference']
                if horizon == '1':
                    for key in ('original_logit', 'flipped_logit'):
                        assert abs(row['native'][key]-row['offcone'][key]) <= 1e-6
            groups.setdefault(m, []).append(event)
        assert len(selected) >= 64 and len(groups) >= 16
        assert len(selected) == summary['sizes'][str(size)]['event_count'] == payload['counts'][str(size)]['events']
        assert len(groups) == summary['sizes'][str(size)]['eligible_map_count']
        contrasts = {}
        for h in ('1', '4'):
            means = {name: sum(sum(e['outcomes'][h][name]['paired_acquisition'] for e in group)/len(group) for group in groups.values())/len(groups)
                     for name in ('native', 'sender_rollback', 'wrong_neighbor_sham')}
            native_effect = means['native']-means['sender_rollback']
            sham_effect = means['wrong_neighbor_sham']-means['sender_rollback']
            reported = summary['sizes'][str(size)]['horizons'][h]
            close(native_effect, reported['native_minus_sender_rollback'])
            close(sham_effect, reported['sham_minus_sender_rollback'])
            contrasts[h] = {'native_minus_sender': native_effect, 'sham_minus_sender': sham_effect}
        signal &= contrasts['1']['native_minus_sender'] >= .10 and contrasts['1']['sham_minus_sender'] >= .05
        result['sizes'][str(size)] = {'events': len(selected), 'maps': len(groups), 'contrasts': contrasts}
    assert signal == summary['primary_signal_thresholds_met']
    assert summary['scientific_status'] == ('DESCRIPTIVE_LOCAL_ROLLBACK_SIGNAL' if signal else 'NO_PRIMARY_THRESHOLD_SIGNAL')
    result['primary_signal_thresholds_met'] = signal
    result['saved_artifact_sha256'] = {name: sha(run/name) for name in ('manifest.json', 'status.json', 'compact_summary.json', 'events.json', 'replaychecks.json')}
    result['validation_source_sha256'] = sha(__file__)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
