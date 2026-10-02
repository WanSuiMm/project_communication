"""One nonzero independent CPU smoke for all fixed-weight switches."""
import json

import torch
from torch.nn import functional as F

from operators import CONDITIONS, StreamingCell, step
from run_revision import tensor_hash
from tasks import bank


def reference_laplacian(h, mask):
    result = torch.zeros_like(h)
    height, width = h.shape[-2:]
    for y in range(height):
        for x in range(width):
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < height and 0 <= xx < width:
                    edge = mask[:, :, y, x] * mask[:, :, yy, xx]
                    result[:, :, y, x] += edge * (h[:, :, y, x]-h[:, :, yy, xx])
    return result


def reference_stream(w, mask):
    result = torch.zeros_like(w)
    height, width = w.shape[-2:]
    lane = w.shape[1] // 4
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    for b in range(w.shape[0]):
        for y in range(height):
            for x in range(width):
                for d, (dy, dx) in enumerate(directions):
                    yy, xx = y+dy, x+dx
                    if not mask[b, 0, y, x]:
                        yy, xx, destination = y, x, d
                    elif 0 <= yy < height and 0 <= xx < width and mask[b, 0, yy, xx]:
                        destination = d
                    else:
                        yy, xx, destination = y, x, (d+2) % 4
                    result[b, destination*lane:(destination+1)*lane, yy, xx] = w[b, d*lane:(d+1)*lane, y, x]
    return result


def reference_step(model, state, x, transport, perception):
    w, z = state
    incoming = reference_stream(w, x[:, :1]) if transport else w
    lap = lambda v: reference_laplacian(v, x[:, :1]) if perception else torch.zeros_like(v)
    conv = lambda layer, v: F.conv2d(v, layer.weight, layer.bias)
    features = torch.cat((incoming, z, lap(w), lap(z), x), 1)
    wplus = incoming + model.eta * conv(model.f_out, torch.tanh(conv(model.f_in, features)))
    features = torch.cat((wplus, z, lap(wplus), lap(z), x), 1)
    zplus = z + model.alpha * conv(model.q_out, torch.tanh(conv(model.q_in, features)))
    return wplus, zplus


@torch.no_grad()
def check():
    torch.set_num_threads(2)
    torch.manual_seed(193)
    model = StreamingCell().eval()
    model.f_out.weight.normal_(std=.03)
    model.q_out.weight.normal_(std=.03)
    model.f_out.bias.normal_(std=.01)
    model.q_out.bias.normal_(std=.01)
    x = bank(8, 2, 911)['x']
    state = (torch.randn(2, 24, 8, 8), torch.randn(2, 8, 8, 8))
    saved = tuple(v.clone() for v in state)
    parameter_hash = tensor_hash(model.state_dict())
    errors = {}
    for name, switches in CONDITIONS.items():
        observed = step(model, state, x, *switches)
        expected = reference_step(model, state, x, *switches)
        for a, b in zip(observed, expected):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        errors[name] = max(float((a-b).abs().max()) for a, b in zip(observed, expected))
    for a, b in zip(step(model, state, x), model.step(state, x)):
        assert torch.equal(a, b), 'All-on must exactly reproduce original cell'
    model.streaming = False
    for a, b in zip(step(model, state, x, False, True), model.step(state, x)):
        assert torch.equal(a, b), 'T->I must equal existing cell flag'
    assert all(torch.equal(a, b) for a, b in zip(state, saved))
    assert tensor_hash(model.state_dict()) == parameter_hash
    # With both paths off, a change at one location cannot change another cell.
    altered = (state[0].clone(), state[1].clone())
    altered[0][:, :, 3, 3] += 4
    a = step(model, state, x, False, False)
    b = step(model, altered, x, False, False)
    for aa, bb in zip(a, b):
        difference = aa-bb
        difference[:, :, 3, 3] = 0
        assert not bool(difference.any())
    return {'status': 'PASS', 'independent_reference_max_errors': errors,
            'all_on_original_exact': True, 'identity_transport_existing_flag_exact': True,
            'parameters_and_inputs_unchanged': True, 'neither_is_pointwise': True}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
