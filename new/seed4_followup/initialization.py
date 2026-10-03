"""Historical seed4 initialization and explicit global-L2 perturbation rays."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'new/streaming_carry'))
from stream_cells import StreamingCell


def initial_model(epsilon=0., direction_seed=None):
    torch.manual_seed(4)
    model = StreamingCell()
    params = sorted(model.named_parameters())
    base = torch.cat([p.detach().flatten().clone() for _, p in params])
    base_norm = float(base.double().norm())
    metadata = {'epsilon': epsilon, 'direction_seed': direction_seed,
                'base_l2': base_norm, 'parameter_count': base.numel(), 'blocks': {}}
    if epsilon:
        assert epsilon in (.01, .05) and direction_seed in range(70002, 70006)
        generator = torch.Generator(device='cpu').manual_seed(direction_seed)
        direction = torch.randn(base.shape, generator=generator)
        direction = direction/direction.double().norm().float()
        delta = direction*(epsilon*base_norm)
        cursor = 0
        with torch.no_grad():
            for name, p in params:
                part = delta[cursor:cursor+p.numel()].reshape(p.shape)
                old = p.clone()
                p.add_(part)
                metadata['blocks'][name] = {'base_l2': float(old.double().norm()),
                    'actual_delta_l2': float((p-old).double().norm()),
                    'originally_all_zero': bool((old == 0).all())}
                cursor += p.numel()
    actual = torch.cat([p.detach().flatten() for _, p in params])-base
    metadata['actual_delta_l2'] = float(actual.double().norm())
    metadata['actual_relative_l2'] = metadata['actual_delta_l2']/base_norm
    return model, metadata
