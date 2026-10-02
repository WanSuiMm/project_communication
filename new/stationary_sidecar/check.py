"""Focused CPU qualification of the nested sidecar and frozen K8 protocol."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import torch
from sidecar_cells import (ROOT, VARIANTS, PARAMETERS, CORE_NAMES,
                           make_variant, core_state_dict)

spec = importlib.util.spec_from_file_location('sidecar_runner', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def lap_reference(value, mask):
    # Explicit in-domain graph neighbors; exterior replicated ghosts contribute0.
    out = torch.zeros_like(value)
    for y in range(value.shape[-2]):
        for x in range(value.shape[-1]):
            for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < value.shape[-2] and 0 <= xx < value.shape[-1]:
                    out[:, :, y, x] += (mask[:, :, y, x] * mask[:, :, yy, xx] *
                                      (value[:, :, y, x] - value[:, :, yy, xx]))
    return out


def stream_reference(w, mask):
    out = torch.zeros_like(w)
    written = torch.zeros(w.shape[0], 4, *w.shape[-2:], dtype=torch.int32)
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    opposite = (2, 3, 0, 1)
    c = w.shape[1] // 4
    for b in range(w.shape[0]):
        for y in range(w.shape[-2]):
            for x in range(w.shape[-1]):
                for d, (dy, dx) in enumerate(directions):
                    yy, xx, dd = y, x, d
                    if bool(mask[b, 0, y, x]):
                        ty, tx = y + dy, x + dx
                        if (0 <= ty < w.shape[-2] and 0 <= tx < w.shape[-1]
                                and bool(mask[b, 0, ty, tx])):
                            yy, xx = ty, tx
                        else:
                            dd = opposite[d]
                    assert written[b, dd, yy, xx] == 0
                    out[b, dd*c:(dd+1)*c, yy, xx] = w[b, d*c:(d+1)*c, y, x]
                    written[b, dd, yy, xx] += 1
    assert bool((written == 1).all())
    return out


def features_reference(w, z, x):
    return torch.cat((w, z, lap_reference(w, x[:, :1]),
                      lap_reference(z, x[:, :1]), x), 1)


def randomize(model):
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.startswith(('f_out', 'q_out', 'feedback_')):
                value.normal_(std=.02)


def neutral_check(variant, original, x):
    candidate = make_variant(variant).double()
    candidate.load_state_dict(original.state_dict(), strict=False)
    w = torch.randn(2, 24, 5, 7, dtype=torch.double)
    z = torch.randn(2, 8, 5, 7, dtype=torch.double)
    h = torch.randn(2, 12, 5, 7, dtype=torch.double)
    inputs_a = (w.clone().requires_grad_(), z.clone().requires_grad_())
    inputs_b = (w.clone().requires_grad_(), z.clone().requires_grad_(), h.requires_grad_())
    a = original.step(inputs_a, x)
    b = candidate.step(inputs_b, x)
    assert all(torch.equal(aa, bb) for aa, bb in zip(a, b[:2]))
    assert torch.equal(original.logits(a), candidate.logits(b))
    noises = tuple(torch.randn_like(v) for v in a)
    original.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    sum((v*n).sum() for v, n in zip(a, noises)).backward()
    sum((v*n).sum() for v, n in zip(b[:2], noises)).backward()
    for name in CORE_NAMES:
        aa = dict(original.named_parameters())[name].grad
        bb = dict(candidate.named_parameters())[name].grad
        if aa is None:
            assert bb is None
        else:
            torch.testing.assert_close(aa, bb, rtol=1e-12, atol=1e-12)
    for aa, bb in zip(inputs_a, inputs_b[:2]):
        torch.testing.assert_close(aa.grad, bb.grad, rtol=1e-12, atol=1e-12)
    if inputs_b[2].grad is not None:
        assert int(torch.count_nonzero(inputs_b[2].grad)) == 0
    # Arbitrary old F/Q tails are nonzero, so this is not just an init identity test.
    return candidate


def reference_check(variant, x):
    model = make_variant(variant).double()
    randomize(model)
    w = torch.randn(2, 24, 5, 7, dtype=torch.double)
    z = torch.randn(2, 8, 5, 7, dtype=torch.double)
    h = torch.randn(2, 12, 5, 7, dtype=torch.double)
    f = features_reference(w, z, x)
    u = .1 * model.g_out(torch.tanh(model.g_in(f)))
    active = h + u if variant == 'memory' else u
    incoming = stream_reference(w, x[:, :1])
    force = model.f_out(torch.tanh(model.f_in(torch.cat((incoming, f[:, 24:]), 1))))
    wr = incoming + .1 * (force + model.feedback_f(active))
    q = model.q_out(torch.tanh(model.q_in(features_reference(wr, z, x))))
    zr = z + .5 * (q + model.feedback_q(active))
    hr = active if variant == 'memory' else torch.zeros_like(h)
    actual = model.step((w, z, h), x)
    for aa, bb in zip(actual, (wr, zr, hr)):
        torch.testing.assert_close(aa, bb, rtol=1e-12, atol=1e-12)
    if variant == 'stateless':
        alternate = model.step((w, z, h + 100), x)
        assert all(torch.equal(aa, bb) for aa, bb in zip(actual, alternate))
        assert float(model.last_injection_q.abs().max()) > 0


def clock_and_gradient_entry(variant):
    torch.manual_seed(2)
    model = make_variant(variant)
    data = runner.bank(8, 4, 10002)
    expected = {'backward_calls': 8, 'interior_detach_boundaries': 7,
                'forward_steps': 64, 'loss_count': 8}
    # Entire carried H value survives K8 cuts; only its autograd history is cut.
    reference = model.initial(data['x'])
    with torch.no_grad():
        for _ in range(64):
            reference = model.step(reference, data['x'])
    _, end, trace = runner.p2.backward_trajectory(model, data, 8)
    assert trace == expected and len(end) == 3
    assert all(v.grad_fn is None and not v.requires_grad for v in end)
    assert all(torch.equal(aa, bb) for aa, bb in zip(end, reference))
    if variant == 'memory':
        assert float(end[2].abs().max()) > 0
    else:
        assert int(torch.count_nonzero(end[2])) == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    logs = []
    for iteration in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        loss, state, trace = runner.p2.backward_trajectory(model, data, 8)
        norms = runner.gradient_groups(model)
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        if iteration == 1:
            assert norms['feedback_f.weight'] == 0
            assert all(norms[name] == 0 for name in norms if name.startswith('g_'))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert bool(torch.isfinite(norm)) and bool(torch.isfinite(loss))
        optimizer.step()
        logs.append({'update': iteration, 'side_gradient_norms': norms})
    assert runner.gradient_entry_check({'variant': variant, 'training_curve': logs})
    return logs


def geometry_check(variant):
    model = make_variant(variant)
    randomize(model)
    data = runner.bank(16, 2, 723)
    a, b = model.initial(data['x']), model.initial(data['x_flip'])
    tested = 0
    with torch.no_grad():
        for t in range(1, 6):
            a = model.step(a, data['x'])
            b = model.step(b, data['x_flip'])
            outside = data['changed'].bool() & (data['distance'] > 2*t)
            tested += int(outside.sum())
            assert float(((model.logits(a)-model.logits(b)).abs()*outside).max()) == 0
        # Disconnected components: arbitrary W/Z/H perturbations cannot cross wall.
        x = torch.randn(1, 3, 5, 7)
        x[:, 0] = 1
        x[:, 0, :, 3] = 0
        a = model.initial(x)
        b = tuple(v.clone() for v in a)
        for v in b:
            v[:, :, :, 4:] += 10
        for _ in range(8):
            a, b = model.step(a, x), model.step(b, x)
            assert all(torch.equal(aa[:, :, :, :4], bb[:, :, :, :4]) for aa, bb in zip(a, b))
    assert tested > 0
    return tested


def decision_check():
    def rows(stream, memory, stateless):
        return [{'variant': variant, 'seed': seed, 'reach': seed in good,
                 'hold': True, 'reach_and_hold': seed in good}
                for variant, good in zip(VARIANTS, (stream, memory, stateless))
                for seed in runner.SEEDS]
    reproduced = {'all_reproduced': True}
    assert runner.decision(rows({4}, {2, 3, 4}, {2}), True, reproduced)['decision'] == 'DEVELOPMENT_GO'
    assert runner.decision(rows({4}, {2, 3, 5}, {2}), True, reproduced)['decision'] == 'DEVELOPMENT_NO_GO'
    assert runner.decision(rows({4}, {2, 3, 4}, {2, 3, 4}), True, reproduced)['decision'] == 'DEVELOPMENT_NO_GO'
    assert runner.decision(rows({2}, {2, 3, 4, 5}, set()), True, reproduced)['decision'] == 'CONTROL_REPRODUCTION_DRIFT'
    assert runner.decision(rows({4}, {2, 3, 4, 5}, set()), False, reproduced)['decision'] == 'INCOMPLETE'
    assert runner.decision(rows({4}, {2, 3, 4, 5}, set()), True, {'all_reproduced': False})['decision'] == 'CONTROL_REPRODUCTION_DRIFT'
    # T128 pass must never conceal a T256 failure.
    record = {'seed': 2, 'variant': 'memory', 'evaluation': {'32': {}}}
    metric = {'original': {'balanced_accuracy': .95}, 'flipped': {'balanced_accuracy': .95},
              'bands': {'strict_16_32': {'mean': .9, 'pooled_accuracy': .9}}}
    record['evaluation']['32'] = {str(t): copy.deepcopy(metric) for t in (64, 128, 256)}
    assert runner.predicates(record)['reach_and_hold']
    record['evaluation']['32']['256']['bands']['strict_16_32']['mean'] = .8
    assert not runner.predicates(record)['hold']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    torch.manual_seed(8127)
    identities = runner.cpu_initialization_reference()
    x = torch.randn(2, 3, 5, 7, dtype=torch.double)
    x[:, 0] = (torch.rand(2, 5, 7) > .25).double()
    old = make_variant('stream').double()
    randomize(old)
    gradients, geometry = {}, {}
    for variant in ('memory', 'stateless'):
        neutral_check(variant, old, x)
        reference_check(variant, x)
        gradients[variant] = clock_and_gradient_entry(variant)
        geometry[variant] = geometry_check(variant)
    decision_check()
    result = {'status': 'PASS', 'checked_utc': runner.now(),
              'parameter_count': PARAMETERS, 'identities': identities,
              'neutral_nonzero_core_forward_and_derivatives': True,
              'independent_nonzero_graph_stream_and_sidecar_reference': True,
              'stateless_immediate_consumption_and_no_H_dependence': True,
              'K8_all_state_detach_preserves_H_values': True,
              'geometry_checks_outside_pixels': geometry,
              'gradient_entry': gradients, 'gate_truth_table_and_both_hold_horizons': True,
              'source_sha256': runner.sources()}
    runner.write(out, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('identities', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
