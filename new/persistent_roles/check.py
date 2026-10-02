"""Focused CPU checks for the new three-state cell and frozen runner."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
for name in ("local_interface", "nca_inertial_wind_tunnel", "short_bptt_phase2"):
    sys.path.insert(0, str(ROOT / "new" / name))
from role_cells import PersistentRoleCell, VARIANTS, make_variant
from tasks import balanced_loss
from training import backward_trajectory


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old_checks = load("frozen_interface_checks", "new/local_interface/check.py")
runner = load("persistent_role_runner", "new/persistent_roles/run.py")


def reference_phase(model, state, x):
    h, c, z = state
    a = F.conv2d(torch.cat((h, c, z, x), 1), model.r_in.weight, model.r_in.bias)
    d = F.conv2d(torch.tanh(a), model.r_out.weight, model.r_out.bias)
    return (h + .1 * d[:, :12],
            old_checks._explicit_stream(c + .1 * d[:, 12:24], x[:, :1]),
            z + .5 * d[:, 24:])


def nonzero(model):
    with torch.no_grad():
        model.r_out.weight.normal_(std=.03)
        model.r_out.bias.normal_(std=.01)


def check_shapes_and_initialization():
    refs = runner.cpu_initialization_reference()
    assert len(refs) == 4
    assert len(runner.sources()) == 51
    x = torch.zeros(1, 3, 5, 6)
    x[:, 0] = 1
    for v in VARIANTS:
        torch.manual_seed(2)
        m = make_variant(v)
        assert sum(p.numel() for p in m.parameters()) == 5033
        state = m.initial(x)
        next_state = m.step(state, x)
        assert sum(a.shape[1] for a in state) == 32
        assert [a.shape for a in state] == [a.shape for a in next_state]
        assert m.logits(next_state).shape == (1, 1, 5, 6)
    assert [a.shape[1] for a in state] == [12, 12, 8]
    torch.manual_seed(2)
    m = PersistentRoleCell()
    encoded = m.encoder(x)
    h, c, z = m.initial(x)
    torch.testing.assert_close(torch.cat((h, c), 1), encoded, rtol=0, atol=0)
    assert not bool(z.any())


def check_independent_forward_gradient_and_base():
    torch.manual_seed(51)
    m = PersistentRoleCell().double()
    x = torch.zeros(1, 3, 4, 5, dtype=torch.double)
    x[:, 0] = 1
    x[:, 0, 1, 2] = 0
    state = tuple(torch.randn(1, c, 4, 5, dtype=torch.double, requires_grad=True)
                  for c in (12, 12, 8))
    pure = m.step(state, x)
    torch.testing.assert_close(pure[0], state[0], rtol=0, atol=0)
    torch.testing.assert_close(pure[2], state[2], rtol=0, atol=0)
    expected_c = old_checks._explicit_stream(old_checks._explicit_stream(state[1], x[:, :1]), x[:, :1])
    torch.testing.assert_close(pure[1], expected_c, rtol=0, atol=0)
    probes = tuple(torch.randn_like(a) for a in pure)
    grads = torch.autograd.grad(sum((a*b).sum() for a, b in zip(pure, probes)), state)
    probe_norm = sum(a.square().sum() for a in probes)
    grad_norm = sum(a.square().sum() for a in grads)
    torch.testing.assert_close(grad_norm, probe_norm, rtol=1e-12, atol=1e-12)
    # Nonzero output weights make every local residual branch participate.
    nonzero(m)
    actual = m.step(state, x)
    expected = reference_phase(m, reference_phase(m, state, x), x)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    variables = (*state, *m.parameters())
    ga = torch.autograd.grad(sum((a*b).sum() for a, b in zip(actual, probes)), variables,
                             allow_unused=True)
    gb = torch.autograd.grad(sum((a*b).sum() for a, b in zip(expected, probes)), variables,
                             allow_unused=True)
    for a, b in zip(ga, gb):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def check_readout_clock_and_isolation():
    m = PersistentRoleCell()
    with torch.no_grad():
        for p in (m.r_in.weight, m.r_in.bias, m.r_out.weight, m.r_out.bias):
            p.zero_()
        m.r_in.weight[0, 15, 0, 0] = 1  # incoming C east, payload0
        m.r_out.weight[24, 0, 0, 0] = 1  # read into Z0; no carrier modification
    x = torch.zeros(1, 3, 1, 40)
    x[:, 0] = 1
    state = (torch.zeros(1, 12, 1, 40), torch.zeros(1, 12, 1, 40),
             torch.zeros(1, 8, 1, 40))
    source = 3
    state[1][0, 3, 0, source] = 1
    for p in range(1, 17):
        state = m.phase(state, x)
        assert state[1][0, 3, 0, source+p] == 1
        assert state[2][0, 0, 0, source+p-1] > 0
        assert not bool(state[2][..., source+p:].any())
    # General nonzero local rule, disconnected components and pointwise X.
    torch.manual_seed(56)
    m = PersistentRoleCell()
    nonzero(m)
    x = torch.zeros(1, 3, 5, 7)
    x[:, 0] = 1
    x[:, 0, :, 3] = 0
    flipped = x.clone()
    flipped[:, 1, 2, 1] = 1
    a, b = m.rollout(x, 6), m.rollout(flipped, 6)
    for u, v in zip(a, b):
        torch.testing.assert_close(u[..., 4:], v[..., 4:], rtol=0, atol=0)


def check_three_state_K8_clock():
    torch.manual_seed(71)
    m = PersistentRoleCell()
    nonzero(m)
    reference = copy.deepcopy(m)
    x = torch.zeros(1, 3, 4, 5)
    x[:, 0] = 1
    x[:, 1, 1, 1] = 1
    y = torch.zeros(1, 1, 4, 5)
    y[..., :2] = 1
    data = {'x': x, 'y': y, 'mask': x[:, :1]}
    before = {k: v.clone() for k, v in m.state_dict().items()}
    loss, final, trace = backward_trajectory(m, data, 8)
    state = reference.initial(x)
    manual_loss = 0.
    for t in range(1, 65):
        state = reference.step(state, x)
        if t % 8 == 0:
            term = balanced_loss(reference.logits(state), y, data['mask']) / 8
            manual_loss += float(term.detach())
            term.backward()
            saved = tuple(a.clone().detach() for a in state)
            state = tuple(a.detach() for a in state)
            for a, b in zip(state, saved):
                assert torch.equal(a, b)
    assert trace == {'backward_calls': 8, 'interior_detach_boundaries': 7,
                     'forward_steps': 64, 'loss_count': 8}
    torch.testing.assert_close(loss, loss.new_tensor(manual_loss), rtol=1e-6, atol=1e-7)
    assert len(final) == 3 and all(not a.requires_grad for a in final)
    for a, b in zip(final, state):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for (key, a), b in zip(m.named_parameters(), reference.parameters()):
        assert torch.equal(a, before[key])  # trainer does not step parameters
        assert a.grad is not None and bool(torch.isfinite(a.grad).all())
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-7)


def check_decision_and_hold_cases():
    rows = [{'seed': s, 'variant': v, 'reach': False, 'hold': True,
             'reach_and_hold': False} for s in (2, 3, 4, 5) for v in VARIANTS]
    for row in rows:
        passed = ((row['variant'] == 'baseline' and row['seed'] in (2, 5)) or
                  (row['variant'] == 'stream' and row['seed'] == 4) or
                  (row['variant'] == 'roles' and row['seed'] in (2, 4, 5)))
        row.update(reach=passed, reach_and_hold=passed)
    assert runner.decision(rows, True, {'all_reproduced': True})['decision'] == 'DEVELOPMENT_GO'
    assert runner.decision(rows, False, {'all_reproduced': True})['decision'] == 'INCOMPLETE'
    assert runner.decision(rows, True, {'all_reproduced': False})['decision'] == 'CONTROL_REPRODUCTION_DRIFT'
    failed = copy.deepcopy(rows)
    next(r for r in failed if r['variant'] == 'roles' and r['seed'] == 4)['reach_and_hold'] = False
    assert runner.decision(failed, True, {'all_reproduced': True})['decision'] == 'DEVELOPMENT_NO_GO'
    omitted = copy.deepcopy(rows)
    for r in omitted:
        if r['variant'] == 'roles':
            r['reach_and_hold'] = r['seed'] in (3, 4, 5)
    assert runner.decision(omitted, True, {'all_reproduced': True})['decision'] == 'DEVELOPMENT_NO_GO'
    for horizon in ('128', '256'):
        for field in ('ba_original', 'ba_flipped', 'primary_mean', 'primary_pooled'):
            record = {'seed': 2, 'variant': 'roles', 'evaluation': {'32': {}}}
            for t in ('64', '128', '256'):
                record['evaluation']['32'][t] = {
                    'original': {'balanced_accuracy': .9},
                    'flipped': {'balanced_accuracy': .9},
                    'bands': {'strict_16_32': {'mean': .9, 'pooled_accuracy': .9}}}
            assert runner.predicates(record)['reach_and_hold']
            e = record['evaluation']['32'][horizon]
            if field.startswith('ba_'):
                e[field.removeprefix('ba_')]['balanced_accuracy'] = .86
            else:
                e['bands']['strict_16_32']['mean' if field == 'primary_mean' else 'pooled_accuracy'] = .84
            assert not runner.predicates(record)['hold']


if __name__ == '__main__':
    torch.set_num_threads(2)
    checks = (check_shapes_and_initialization, check_independent_forward_gradient_and_base,
              check_readout_clock_and_isolation, check_three_state_K8_clock,
              check_decision_and_hold_cases)
    for check in checks:
        check()
        print(json.dumps({'check': check.__name__, 'passed': True}), flush=True)
    print(json.dumps({'status': 'CPU_CHECK_PASSED', 'checks': len(checks)}), flush=True)
