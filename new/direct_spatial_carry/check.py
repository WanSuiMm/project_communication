"""One CPU contract check before the bounded CUDA preflight."""
import copy
import importlib.util
import json
from pathlib import Path

import torch
from carry_cells import DirectCarryCell, RevisionCell, masked_degree, spatial_carry

spec = importlib.util.spec_from_file_location('carry_runner', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def explicit_carry(w, mask):
    out = w.clone()
    for b in range(w.shape[0]):
        for y in range(w.shape[2]):
            for x in range(w.shape[3]):
                if not mask[b, 0, y, x]:
                    continue
                neighbors = [(yy, xx) for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1))
                             if 0 <= yy < w.shape[2] and 0 <= xx < w.shape[3] and mask[b, 0, yy, xx]]
                if neighbors:
                    out[b, :, y, x] = .5 * w[b, :, y, x] + .5 * torch.stack(
                        [w[b, :, yy, xx] for yy, xx in neighbors]).mean(0)
    return out


def main():
    torch.set_num_threads(2)
    torch.manual_seed(713)
    mask = torch.tensor([[[[1, 1, 0, 1], [1, 0, 0, 0], [0, 1, 1, 0]]]], dtype=torch.double)
    w = torch.randn(1, 3, 3, 4, dtype=torch.double, requires_grad=True)
    got, ref = spatial_carry(w, mask), explicit_carry(w, mask)
    torch.testing.assert_close(got, ref, atol=1e-14, rtol=1e-14)
    assert masked_degree(mask)[0, 0, 0, 0] == 2  # Boundary ghosts excluded.
    assert masked_degree(mask)[0, 0, 0, 3] == 0  # Isolated open cell.
    assert float(got.detach().abs().max()) <= float(w.detach().abs().max()) + 1e-14
    for c in range(3):
        torch.testing.assert_close(got[:, c:c+1], spatial_carry(w[:, c:c+1], mask), rtol=0, atol=0)
    constant = torch.tensor([1., -2., 3.], dtype=torch.double).view(1, 3, 1, 1).expand_as(w)
    torch.testing.assert_close(spatial_carry(constant, mask), constant, rtol=0, atol=0)
    perturb = torch.randn_like(w)
    ga = torch.autograd.grad((got * perturb).sum(), w)[0]
    gb = torch.autograd.grad((ref * perturb).sum(), w)[0]
    torch.testing.assert_close(ga, gb, rtol=1e-14, atol=1e-14)
    changed = w.detach().clone()
    changed[:, :, 2, 1:3] += 100
    torch.testing.assert_close(spatial_carry(changed, mask)[:, :, :2], got.detach()[:, :, :2], rtol=0, atol=0)

    # Init construction consumes exactly the old draws and adds no parameters.
    for seed in (2, 3, 4, 5):
        torch.manual_seed(seed)
        old = RevisionCell('ws_additive')
        torch.manual_seed(seed)
        new = DirectCarryCell()
        assert sum(p.numel() for p in new.parameters()) == 5033
        assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
        assert runner.tensor_hash(old.state_dict()) == runner.historical_record(seed)['initial_parameter_sha256']

    # Nonzero F/Q exposes feature-order errors hidden by zero initialization.
    with torch.no_grad():
        old.f_out.weight.normal_(std=.02)
        old.q_out.weight.normal_(std=.02)
    data = runner.bank(8, 2, 714)
    zero = DirectCarryCell(rho=0.)
    zero.load_state_dict(old.state_dict())
    old_loss, old_state, old_trace = runner.backward_trajectory(old, data, 8)
    zero_loss, zero_state, zero_trace = runner.backward_trajectory(zero, data, 8)
    torch.testing.assert_close(old_loss, zero_loss, rtol=0, atol=0)
    for a, b in zip(old_state, zero_state):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(old.parameters(), zero.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-7)
    assert old_trace == zero_trace == {'backward_calls': 8, 'interior_detach_boundaries': 7,
                                      'forward_steps': 64, 'loss_count': 8}

    # Independent full-cell reference uses the explicit neighbor loop, old F
    # and unchanged additive Q. Verify state AND derivatives of both states.
    carry = DirectCarryCell()
    carry.load_state_dict(old.state_dict())
    w0, z0 = carry.initial(data['x'])
    w0 = w0.detach().requires_grad_()
    z0 = (z0.detach() + torch.randn_like(z0) * .1).requires_grad_()
    feature = old._features(w0, z0, data['x'])
    force = old.f_out(torch.tanh(old.f_in(feature)))
    wr = explicit_carry(w0, data['x'][:, :1]) + old.eta * force
    zr = z0 + old.alpha * old._candidate(wr, z0, data['x'])
    wa, za = carry.step((w0, z0), data['x'])
    torch.testing.assert_close(wa, wr, rtol=1e-5, atol=2e-7)
    torch.testing.assert_close(za, zr, rtol=1e-5, atol=2e-7)
    weights = (torch.randn_like(wa), torch.randn_like(za))
    da = torch.autograd.grad((wa * weights[0]).sum() + (za * weights[1]).sum(), (w0, z0))
    dr = torch.autograd.grad((wr * weights[0]).sum() + (zr * weights[1]).sum(), (w0, z0))
    for a, b in zip(da, dr):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)
    carry.zero_grad(set_to_none=True)
    before = copy.deepcopy(carry.state_dict())
    _, state, trace = runner.backward_trajectory(carry, data, 8)
    assert trace == old_trace
    assert all(not v.requires_grad and v.grad_fn is None for v in state)
    assert all(torch.equal(v, carry.state_dict()[k]) for k, v in before.items())
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in carry.parameters())

    # Decision truth table tests preserved positives, count gain and completeness.
    def rows(baseline, candidate):
        return [{'seed': s, 'variant': v, 'reach': s in good, 'hold': True, 'reach_and_hold': s in good}
                for v, good in (('baseline', baseline), ('carry', candidate)) for s in runner.SEEDS]
    assert runner.decision(rows({2, 5}, {2, 3, 5}), True)['decision'] == 'DEVELOPMENT_GO'
    assert runner.decision(rows({2, 5}, {3, 4, 5}), True)['decision'] == 'DEVELOPMENT_NO_GO'
    assert runner.decision(rows({2, 3, 5}, {2, 3, 5}), True)['decision'] == 'DEVELOPMENT_NO_GO'
    assert runner.decision(rows({2}, {2, 3, 4, 5}), True)['decision'] == 'BASELINE_REPRODUCTION_DRIFT'
    assert runner.decision(rows({2, 5}, {2, 3, 4, 5}), False)['decision'] == 'INCOMPLETE'
    print(json.dumps({'status': 'PASS', 'explicit_neighbor_reference': True,
                      'boundary_isolation_components_channels': True, 'carry_derivative_matches': True,
                      'identical_5033_parameters_all_seeds': True, 'zero_rho_forward_gradient_equivalence': True,
                      'nonzero_full_cell_reference': True, 'K8_detach_and_no_inner_update': True,
                      'development_gate_truth_table': True}, indent=2))


if __name__ == '__main__':
    main()
