"""Fixed-weight pathway switches; original frozen cell is not modified."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'new/streaming_carry'))
from stream_cells import StreamingCell, stream
from revision_cells import masked_laplacian

CONDITIONS = {
    'full': (True, True),
    'no_transport': (False, True),
    'no_perception': (True, False),
    'neither': (False, False),
}


def step(model, state, x, transport=True, perception=True):
    """F sees pre-stream Laplacians; Q sees post-F W and pre-update Z."""
    model._validate_x(x)
    w, z = state
    incoming = stream(w, x[:, :1]) if transport else w
    if perception:
        lw = masked_laplacian(w, x[:, :1])
        lz = masked_laplacian(z, x[:, :1])
    else:
        lw, lz = torch.zeros_like(w), torch.zeros_like(z)
    f_features = torch.cat((incoming, z, lw, lz, x), 1)
    wplus = incoming + model.eta * model.f_out(torch.tanh(model.f_in(f_features)))
    lwp = masked_laplacian(wplus, x[:, :1]) if perception else torch.zeros_like(wplus)
    q_features = torch.cat((wplus, z, lwp, lz, x), 1)
    q = model.q_out(torch.tanh(model.q_in(q_features)))
    return wplus, z + model.alpha * q


def hops_per_step(transport, perception):
    return 2 if perception else int(transport)
