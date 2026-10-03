"""Focused CPU qualification of initialization identity and paired noise radii."""
import json
from pathlib import Path
import sys

import torch

from initialization import ROOT, initial_model
sys.path.insert(0, str(ROOT/'new/workspace_revision'))
from run_revision import tensor_hash


def run_checks():
    historical = json.loads((ROOT/'evidence/streaming_carry_init2345/raw/stream_K8_seed4.json').read_text())
    base, meta = initial_model()
    assert tensor_hash(base.state_dict()) == historical['initial_parameter_sha256']
    assert meta['actual_delta_l2'] == 0 and meta['parameter_count'] == 5033
    base_state = base.state_dict()
    direction_results = []
    for seed in range(70002, 70006):
        small, a = initial_model(.01, seed)
        large, b = initial_model(.05, seed)
        for data, eps in ((a, .01), (b, .05)):
            assert abs(data['actual_relative_l2']/eps-1) < 1e-5
            assert any(v['originally_all_zero'] and v['actual_delta_l2'] > 0 for v in data['blocks'].values())
        da = torch.cat([(small.state_dict()[k]-base_state[k]).flatten() for k in sorted(base_state)])
        db = torch.cat([(large.state_dict()[k]-base_state[k]).flatten() for k in sorted(base_state)])
        # Adding either ray to the same FP32 base rounds each coordinate.
        ray_error = float((db-5*da).double().norm()/db.double().norm())
        assert ray_error < 1e-5
        direction_results.append({'direction_seed': seed,
            'relative_l2': [a['actual_relative_l2'], b['actual_relative_l2']],
            'relative_ray_roundoff_error': ray_error})
    return {'status': 'PASS', 'historical_initial_parameter_sha256': tensor_hash(base.state_dict()),
            'paired_directions': direction_results, 'training': False}


if __name__ == '__main__':
    print(json.dumps(run_checks()))
