"""CPU checks of the new K16 path and equivalence to frozen K8/K64 training."""
import copy
import importlib.util
import json
import torch
from training import ROOT, backward_trajectory, make_cell, balanced_loss
from tasks import bank


def main():
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('phase1_training', ROOT / 'new/short_bptt/training.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    torch.manual_seed(811)
    data = bank(8, 2, 815)
    model = make_cell('ws_additive')
    with torch.no_grad():
        model.f_out.weight.normal_(std=.03)
        model.q_out.weight.normal_(std=.03)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    outputs = {}
    checks = []
    for k in (8, 16, 64):
        trained = copy.deepcopy(model)
        loss, state, trace = backward_trajectory(trained, data, k)
        assert trace == {'backward_calls': 64 // k, 'interior_detach_boundaries': 64 // k - 1,
                         'forward_steps': 64, 'loss_count': 8}
        assert all(not v.requires_grad and v.grad_fn is None for v in state)
        assert all(torch.equal(initial[n], v) for n, v in trained.state_dict().items())
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trained.parameters())
        outputs[k] = (loss, state, trained)
        if k in (8, 64):
            reference = copy.deepcopy(model)
            rl, rs, _ = old.backward_trajectory(reference, data, k)
            torch.testing.assert_close(loss, rl, rtol=0, atol=0)
            for a, b in zip(state, rs):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for a, b in zip(trained.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-7)
        else:
            # Explicit independent windows with isolated gradient buffers;
            # each window has losses at its eighth and sixteenth steps.
            reference = copy.deepcopy(model)
            rs = reference.initial(data['x'])
            accumulated = [torch.zeros_like(p) for p in reference.parameters()]
            for window in range(4):
                reference.zero_grad(set_to_none=True)
                window_losses = []
                for local_t in range(1, 17):
                    rs = reference.step(rs, data['x'])
                    if local_t in (8, 16):
                        window_losses.append(balanced_loss(reference.logits(rs), data['y'], data['mask']) / 8)
                (window_losses[0] + window_losses[1]).backward()
                if window > 0:
                    assert reference.encoder.weight.grad is None
                for total, p in zip(accumulated, reference.parameters()):
                    if p.grad is not None:
                        total.add_(p.grad)
                rs = tuple(v.detach() for v in rs)
            for a, p in zip(accumulated, trained.parameters()):
                torch.testing.assert_close(a, p.grad, rtol=1e-6, atol=1e-7)
        checks.append({'K': k, 'reference_gradient_matches': True, 'trace': trace})
    for k in (8, 16):
        torch.testing.assert_close(outputs[k][0], outputs[64][0], rtol=0, atol=0)
        for a, b in zip(outputs[k][1], outputs[64][1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert any(float((p.grad-q.grad).abs().max()) > 1e-5 for p, q in
                   zip(outputs[k][2].parameters(), outputs[64][2].parameters()))
    print(json.dumps({'status': 'PASS', 'forward_identical': True,
                      'weights_unchanged': True, 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
