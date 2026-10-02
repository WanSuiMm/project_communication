"""CPU checks of masked port streaming and the matched two-phase cell."""
import copy
import importlib.util
import json
from pathlib import Path

import torch
from stream_cells import DIRECTIONS, OPPOSITE, StreamingCell, RevisionCell, stream, inverse_stream

spec = importlib.util.spec_from_file_location('stream_runner', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def explicit_stream(w, mask):
    """Independent outgoing-port scatter with an exactly-once write check."""
    n, channels, height, width = w.shape
    c = channels // 4
    out = torch.zeros_like(w)
    written = torch.zeros(n, 4, height, width, dtype=torch.int32)
    for b in range(n):
        for y in range(height):
            for x in range(width):
                for d, (dy, dx) in enumerate(DIRECTIONS):
                    yy, xx, dd = y, x, d
                    if mask[b, 0, y, x]:
                        ty, tx = y+dy, x+dx
                        if 0 <= ty < height and 0 <= tx < width and mask[b, 0, ty, tx]:
                            yy, xx = ty, tx
                        else:
                            dd = OPPOSITE[d]
                    assert written[b, dd, yy, xx] == 0
                    out[b, dd*c:(dd+1)*c, yy, xx] = w[b, d*c:(d+1)*c, y, x]
                    written[b, dd, yy, xx] += 1
    assert bool((written == 1).all())
    return out


def main():
    torch.set_num_threads(2)
    torch.manual_seed(721)
    masks = torch.tensor([[(v >> j) & 1 for j in range(6)] for v in range(64)], dtype=torch.double).reshape(64,1,2,3)
    labels = torch.arange(64*8*2*3, dtype=torch.double).reshape(64,8,2,3)
    got = stream(labels, masks)
    assert torch.equal(got, explicit_stream(labels, masks))
    assert torch.equal(got.flatten(1).sort(1).values, labels.flatten(1).sort(1).values)
    assert torch.equal(inverse_stream(got, masks), labels)
    assert torch.equal(stream(inverse_stream(labels, masks), masks), labels)
    w = torch.randn_like(labels, requires_grad=True)
    moved = stream(w, masks)
    torch.testing.assert_close(moved.square().sum((1,2,3)), w.square().sum((1,2,3)), atol=1e-12, rtol=1e-12)
    v = torch.randn_like(w)
    grad = torch.autograd.grad((moved*v).sum(), w)[0]
    assert torch.equal(grad, inverse_stream(v, masks))
    # Every payload coordinate is transported independently, despite lane swaps.
    for payload in (0,1):
        idx = [2*d+payload for d in range(4)]
        assert torch.equal(stream(w.detach()[:,idx], masks), moved.detach()[:,idx])
    wall_values = (1-masks).expand_as(w).bool()
    assert torch.equal(moved.detach()[wall_values], w.detach()[wall_values])
    # Two components separated by a wall; perturbing one never reaches the other.
    mask = torch.ones(1,1,5,7)
    mask[:,:,:,3] = 0
    left = torch.randn(1,8,5,7)
    right = left.clone()
    right[:,:,:,4:] += 10
    for _ in range(12):
        left, right = stream(left, mask), stream(right, mask)
        assert torch.equal(left[:,:,:,:4], right[:,:,:,:4])

    for seed in runner.SEEDS:
        torch.manual_seed(seed)
        old = RevisionCell('ws_additive')
        torch.manual_seed(seed)
        candidate = StreamingCell()
        assert sum(p.numel() for p in candidate.parameters()) == 5033
        assert runner.tensor_hash(old.state_dict()) == runner.tensor_hash(candidate.state_dict())
        assert runner.tensor_hash(old.state_dict()) == runner.historical_record(seed)['initial_parameter_sha256']
    data = runner.bank(8, 2, 722)
    # Zero F/Q produces only a permutation trajectory plus fixed local Z.
    pure = candidate.initial(data['x'])
    state, reference = pure, pure[0]
    with torch.no_grad():
        for _ in range(16):
            state = candidate.step(state, data['x'])
            reference = explicit_stream(reference, data['x'][:,:1])
            assert torch.equal(state[0], reference) and torch.equal(state[1], pure[1])
    # Nonzero tails expose dependency/feature-clock mistakes hidden at init.
    with torch.no_grad():
        old.f_out.weight.normal_(std=.02)
        old.q_out.weight.normal_(std=.02)
    identity = StreamingCell(streaming=False)
    identity.load_state_dict(old.state_dict())
    loss_a, state_a, trace_a = runner.backward_trajectory(old, data, 8)
    loss_b, state_b, trace_b = runner.backward_trajectory(identity, data, 8)
    assert torch.equal(loss_a, loss_b) and all(torch.equal(a,b) for a,b in zip(state_a,state_b))
    for a,b in zip(old.parameters(), identity.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-6,atol=1e-7)
    expected = {'backward_calls':8,'interior_detach_boundaries':7,'forward_steps':64,'loss_count':8}
    assert trace_a == trace_b == expected

    candidate.load_state_dict(old.state_dict())
    w0,z0 = candidate.initial(data['x'])
    w0 = w0.detach().requires_grad_()
    z0 = (z0.detach()+torch.randn_like(z0)*.1).requires_grad_()
    old_features = old._features(w0,z0,data['x'])
    incoming = explicit_stream(w0,data['x'][:,:1])
    reference_features = torch.cat((incoming,old_features[:,24:]),dim=1)
    wr = incoming+.1*old.f_out(torch.tanh(old.f_in(reference_features)))
    zr = z0+.5*old._candidate(wr,z0,data['x'])
    wa,za = candidate.step((w0,z0),data['x'])
    assert torch.equal(wa,wr) and torch.equal(za,zr)
    noise = (torch.randn_like(wa),torch.randn_like(za))
    da = torch.autograd.grad((wa*noise[0]).sum()+(za*noise[1]).sum(),(w0,z0))
    dr = torch.autograd.grad((wr*noise[0]).sum()+(zr*noise[1]).sum(),(w0,z0))
    for a,b in zip(da,dr):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)
    candidate.zero_grad(set_to_none=True)
    before = copy.deepcopy(candidate.state_dict())
    _, end, trace = runner.backward_trajectory(candidate,data,8)
    assert trace == expected and all(v.grad_fn is None and not v.requires_grad for v in end)
    assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in candidate.parameters())
    # Counterfactual source cannot influence logits beyond2t graph edges.
    wide = runner.bank(16,2,723)
    a,b = candidate.initial(wide['x']),candidate.initial(wide['x_flip'])
    tested = 0
    with torch.no_grad():
        for t in range(1,5):
            a,b = candidate.step(a,wide['x']),candidate.step(b,wide['x_flip'])
            outside = wide['changed'].bool() & (wide['distance'] > 2*t)
            tested += int(outside.sum())
            assert float(((candidate.logits(a)-candidate.logits(b)).abs()*outside).max()) == 0
    assert tested > 0
    def rows(base, stream_seeds):
        return [{'seed':s,'variant':v,'reach':s in good,'hold':True,'reach_and_hold':s in good}
                for v,good in (('baseline',base),('stream',stream_seeds)) for s in runner.SEEDS]
    assert runner.decision(rows({2,5},{2,3,5}),True)['decision'] == 'DEVELOPMENT_GO'
    assert runner.decision(rows({2,5},{3,4,5}),True)['decision'] == 'DEVELOPMENT_NO_GO'
    assert runner.decision(rows({2,3,5},{2,3,5}),True)['decision'] == 'DEVELOPMENT_NO_GO'
    assert runner.decision(rows({2},{2,3,4,5}),True)['decision'] == 'BASELINE_REPRODUCTION_DRIFT'
    assert runner.decision(rows({2,5},{2,3,4,5}),False)['decision'] == 'INCOMPLETE'
    print(json.dumps({'status':'PASS','exhaustive_binary_2x3_masks':64,
          'explicit_scatter_bijection_and_inverse':True,'norm_and_adjoint':True,
          'payload_no_mixing_and_no_component_leak':True,'matched_5033_parameters':True,
          'identity_transport_forward_gradient_equivalence':True,'zero_residual_pure_stream':True,
          'nonzero_residual_reference':True,'K8_clock_and_two_hop_lightcone':True,
          'development_decision_truth_table':True},indent=2))


if __name__ == '__main__':
    main()
