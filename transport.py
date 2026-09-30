"""Symmetric directional implicit transport in 2D/3D.

Parallel cyclic reduction (PCR), with factorizations reused across right-hand
sides and recurrent steps. The custom first-order adjoint differentiates the
linear system, including every edge weight; no unrolled solver backward.
This portable PyTorch implementation is not a fused GPU kernel.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def minus(x, stride):
    return F.pad(x[..., :-stride], (stride, 0))


def plus(x, stride):
    return F.pad(x[..., stride:], (0, stride))


@dataclass
class PCRFactors:
    stages: list
    inverse_diagonal: torch.Tensor


def factorize(weights):
    """weights[S,L]: edge i--i+1, final entry ignored (no-flux boundary)."""
    with torch.no_grad():
        w = F.pad(weights[..., :-1], (0, 1))
        left = minus(w, 1)
        a, b, c = -left, 1 + left + w, -w
        stages = []
        stride = 1
        while stride < w.shape[-1]:
            bm, bp = minus(b, stride), plus(b, stride)
            # Missing neighbors have zero off-diagonal, with a safe divisor.
            alpha = -a / torch.where(bm != 0, bm, torch.ones_like(bm))
            beta = -c / torch.where(bp != 0, bp, torch.ones_like(bp))
            stages.append((stride, alpha, beta))
            b_next = b + alpha * minus(c, stride) + beta * plus(a, stride)
            a, c = alpha * minus(a, stride), beta * plus(c, stride)
            b = b_next
            stride *= 2
        return PCRFactors(stages, b.reciprocal())


def solve_factors(rhs, factors):
    """rhs[S,R,L], R independent channel right-hand sides."""
    value = rhs
    for stride, alpha, beta in factors.stages:
        value = (value + alpha[:, None, :] * minus(value, stride)
                 + beta[:, None, :] * plus(value, stride))
    return value * factors.inverse_diagonal[:, None, :]


class _ImplicitSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rhs, weights, factors):
        result = solve_factors(rhs, factors)
        ctx.factors = factors
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, upstream):
        (result,) = ctx.saved_tensors
        adjoint = solve_factors(upstream, ctx.factors)
        dm = result[..., :-1] - result[..., 1:]
        dv = adjoint[..., :-1] - adjoint[..., 1:]
        grad_weights = F.pad(-(dm * dv).sum(dim=1), (0, 1))
        return adjoint, grad_weights, None


def solve_lines(rhs, weights, factors=None):
    if factors is None:
        factors = factorize(weights)
    return _ImplicitSolve.apply(rhs, weights, factors)


@dataclass
class Direction:
    axis: int
    groups: int
    weights: torch.Tensor
    factors: PCRFactors


def prepare_transport(edges, tau):
    """XYX in 2D; XYZ YX in 3D (longest index axis is named X).

    Edges: list of B,G,*spatial arrays. Tau: G. Axes other than the
    central axis use half-scale on each symmetric occurrence.
    """
    ndim = len(edges)
    if ndim not in (2, 3):
        raise ValueError('Only 2D and 3D are supported.')
    groups = edges[0].shape[1]
    descending = list(reversed(range(ndim)))
    order = descending + descending[-2::-1]
    prepared = {}
    for axis in descending:
        scale = 1.0 if axis == 0 else 0.5
        weights = edges[axis] * tau.reshape(1, groups, *([1] * ndim)) * scale
        lines = weights.movedim(axis + 2, -1).reshape(-1, weights.shape[axis + 2])
        prepared[axis] = Direction(axis, groups, lines, factorize(lines))
    return [prepared[axis] for axis in order]


def apply_direction(q, direction):
    batch, channels, *spatial = q.shape
    groups, axis = direction.groups, direction.axis
    if channels % groups:
        raise ValueError('Message channels must be divisible by groups.')
    per_group = channels // groups
    arranged = q.reshape(batch, groups, per_group, *spatial).movedim(axis + 3, -1)
    other = arranged.shape[3:-1]
    length = arranged.shape[-1]
    lines = arranged.reshape(batch, groups, per_group, -1, length)
    lines = lines.permute(0, 1, 3, 2, 4).reshape(-1, per_group, length)
    result = solve_lines(lines, direction.weights, direction.factors)
    result = result.reshape(batch, groups, -1, per_group, length).permute(0, 1, 3, 2, 4)
    result = result.reshape(batch, groups, per_group, *other, length)
    return result.movedim(-1, axis + 3).reshape(batch, channels, *spatial)


def apply_transport(q, prepared):
    for direction in prepared:
        q = apply_direction(q, direction)
    return q
