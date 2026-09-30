"""Small numerical/gradient checks, not scientific qualification evidence."""
import json

import torch
import torch.nn.functional as F

from .transport import solve_lines, prepare_transport, apply_transport


def dense_matrix(edges):
    w = F.pad(edges, (0, 1))
    left = F.pad(edges, (1, 0))
    return (torch.diag_embed(1 + w + left)
            - torch.diag_embed(edges, offset=1)
            - torch.diag_embed(edges, offset=-1))


def run_checks():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    rhs = torch.randn(2, 3, 9, dtype=torch.float64, requires_grad=True)
    edges = torch.rand(2, 8, dtype=torch.float64, requires_grad=True) * 10
    actual = solve_lines(rhs, F.pad(edges, (0, 1)))
    expected = torch.linalg.solve(dense_matrix(edges), rhs.transpose(-1, -2)).transpose(-1, -2)
    err = (actual - expected).abs().max().item()
    assert err < 1e-11, err
    assert torch.autograd.gradcheck(
        lambda q, e: solve_lines(q, F.pad(e, (0, 1))), (rhs, edges),
        eps=1e-6, atol=2e-6, rtol=2e-4, fast_mode=True)
    report = {'dense_max_error': err, 'implicit_edge_and_rhs_gradcheck': True}
    for dim, shape in ((2, (4, 7)), (3, (3, 4, 5))):
        edge_list = []
        for axis in range(dim):
            w = torch.rand(2, 2, *shape, dtype=torch.float64)
            w.select(axis + 2, shape[axis] - 1).zero_()
            edge_list.append(w)
        prepared = prepare_transport(edge_list, torch.tensor([0.7, 3.0], dtype=torch.float64))
        x = torch.randn(2, 4, *shape, dtype=torch.float64)
        y = torch.randn_like(x)
        tx, ty = apply_transport(x, prepared), apply_transport(y, prepared)
        constant_error = (apply_transport(torch.ones_like(x), prepared) - 1).abs().max().item()
        symmetry_error = ((x * ty).sum() - (tx * y).sum()).abs().item()
        assert constant_error < 1e-12
        assert symmetry_error < 1e-11
        assert tx.norm() <= x.norm() + 1e-12
        assert (tx.sum(tuple(range(2, dim + 2))) - x.sum(tuple(range(2, dim + 2)))).abs().max() < 1e-11
        report[f'{dim}d'] = {'constant_error': constant_error, 'symmetry_error': symmetry_error,
                             'norm_ratio': (tx.norm() / x.norm()).item()}
    if torch.cuda.is_available():
        # Large propagation scales must also work in the actual float32 device path.
        device = 'cuda'
        q = torch.randn(2, 3, 145, device=device)
        e = torch.rand(2, 144, device=device) * 4096
        value = solve_lines(q, F.pad(e, (0, 1)))
        reference = torch.linalg.solve(dense_matrix(e.double()), q.double().transpose(-1, -2)).transpose(-1, -2)
        rel = ((value.double() - reference).norm() / reference.norm()).item()
        assert rel < 0.01, rel
        report['cuda_float32_large_scale_relative_error'] = rel
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    run_checks()
