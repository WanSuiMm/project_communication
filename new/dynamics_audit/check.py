"""One numerical gate: compare matrix-free derivatives to autograd and exact SVD."""
import json
import torch
from operators import factory, base, pack, unpack, LinearStep, product, estimate, norm


def main():
    torch.set_num_threads(2)
    torch.manual_seed(712)
    records = []
    for arm in ('nca_state_matched', 'momentum_nca', 'masked_state_nca', 'masked_momentum_nca'):
        model = factory(arm).double().eval()
        torch.nn.init.normal_(base(model).program[-1].weight, std=.03)
        if base(model).inertial:
            with torch.no_grad(): base(model).raw_beta.copy_(torch.linspace(-1, 2, base(model).channels).view(1, -1, 1, 1))
        x = torch.randn(1, 3, 4, 4, dtype=torch.float64)
        x[:, 0] = 1
        x[:, 0, :, 2] = 0
        state = pack(model.initial(x)).detach().requires_grad_(True)
        q, u = torch.randn_like(state), torch.randn_like(state)
        fun = lambda s: pack(model.step(unpack(s, model), x))
        step = LinearStep(model, state[:, :base(model).channels], x)
        y, exact_jvp = torch.autograd.functional.jvp(fun, state, q)
        exact_vjp = torch.autograd.grad((fun(state)*u).sum(), state)[0]
        e1 = float(norm(step.apply(q)-exact_jvp)/norm(exact_jvp))
        e2 = float(norm(step.adj(u)-exact_vjp)/norm(exact_vjp))
        dx = torch.randn_like(x); dx[:, 0] = 0
        _, exact_input = torch.autograd.functional.jvp(lambda z: pack(model.step(unpack(state, model), z)), x, dx)
        e3 = float(norm(step.input_apply(dx)-exact_input)/norm(exact_input))
        chain = []; st = state.detach()
        for _ in range(3):
            chain.append(LinearStep(model, st[:, :base(model).channels], x))
            st = pack(model.step(unpack(st, model), x)).detach()
        mask = x[:, :1]
        forward, backward = product(chain, mask)
        error = abs(float((forward(q)*u).sum()-(q*backward(u)).sum()))/max(1, abs(float((forward(q)*u).sum())))
        eps = 1e-5
        def rollout(s):
            for _ in range(3): s = pack(model.step(unpack(s, model), x))
            return s
        fd = (rollout(state+eps*q*mask)-rollout(state-eps*q*mask))/(2*eps)*mask
        ef = float(norm(fd-forward(q))/norm(forward(q)))
        initial = model.initial(x)
        encoder = base(model).encoder
        ih = torch.nn.functional.conv2d(dx, encoder.weight)
        tangent = pack((ih, torch.zeros_like(ih) if base(model).inertial else None))
        st = initial
        for _ in range(3):
            ls = LinearStep(model, st[0], x)
            tangent = ls.apply(tangent)+ls.input_apply(dx)
            st = model.step(st, x)
        _, exact_total = torch.autograd.functional.jvp(lambda z: pack(model.rollout(z, 3)), x, dx)
        et = float(norm(tangent-exact_total)/norm(exact_total))
        assert max(e1, e2, e3, error, et) < 1e-10
        assert ef < 1e-7
        records.append({'arm': arm, 'state_jvp_relative_error': e1, 'state_vjp_relative_error': e2,
                        'source_jvp_relative_error': e3, 'total_driven_rollout_jvp_error': et,
                        'product_adjoint_error': error, 'product_finite_difference_error': ef})
    # A deliberately non-normal free-momentum matrix: finite gain > 1 is not exponential instability.
    beta = .9
    mat = torch.tensor([[1., beta], [0., beta]], dtype=torch.float64)
    power = torch.linalg.matrix_power(mat, 16)
    f = lambda v: (v.flatten(1)@power.T).view_as(v)
    ft = lambda v: (v.flatten(1)@power).view_as(v)
    with torch.no_grad():
        est, _ = estimate(f, ft, (1, 2, 1, 1), 'cpu', torch.float64, 1., 16, 2, 413)
    exact = float(torch.linalg.svdvals(power)[0])
    assert abs(est[0]['sigma_estimate']-exact) < 1e-10
    assert exact > 1 and est[0]['converged']
    print(json.dumps({'status': 'PASS', 'derivative_checks': records,
                      'free_momentum_exact_sigma_K16': exact, 'power_estimation_error': abs(est[0]['sigma_estimate']-exact)}, indent=2))


if __name__ == '__main__': main()
