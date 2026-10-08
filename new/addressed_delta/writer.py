"""Proposed carrier writer; a reference implementation, not a trained result.

Target integration: C24/Z8 StreamingCell from project_communication at f080e69.
Retain the existing transport, pre-stream perception, encoder, Q and readout.
Replace the carrier's arbitrary additive F write with this writer.

Forward inputs:
  incoming: [B, 24, H, W] = T_M(C)
  features: [B, 67, H, W] = [incoming, Z, L(C), L(Z), X]
Forward output: next carrier [B, 24, H, W]. Do not add a second residual outside.

The delta rule itself is established prior work, e.g. arXiv:2406.06484 and
arXiv:2412.06464. Its use here is an untested spatial-carrier design proposal.
"""
from __future__ import annotations

import argparse
import json
import torch
from torch import Tensor, nn


def normalize_reads(raw: Tensor) -> Tensor:
    """K=A/sqrt(1+||A||_F^2), independently at each cell; ||K||_2<1.

    raw: [B, reads, ports, H, W]. Scaling BEFORE squaring prevents ordinary
    float32 overflow in the normalizer. No batch/spatial statistics are used.
    """
    if raw.ndim != 5 or not raw.is_floating_point():
        raise ValueError('raw must be a floating [B, reads, ports, H, W] tensor')
    scale = raw.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
    scaled = raw / scale
    denominator_sq = scale.reciprocal().square() + scaled.square().sum(
        dim=(1, 2), keepdim=True)
    return scaled * torch.rsqrt(denominator_sq)


def addressed_write(incoming: Tensor, keys: Tensor, values: Tensor,
                    step_size: float = 0.1, *, subtract_read: bool = True) -> Tensor:
    """U + eta K^T(V-KU); additive control removes ONLY the -KU term.

    incoming: [B, ports, payload, H, W]
    keys:     [B, reads, ports, H, W], normalized by normalize_reads
    values:   [B, reads, payload, H, W]

    Energy and frozen-coefficient nonexpansion results require ||K||_2<=1,
    0<eta<=1, and subtract_read=True. They do not certify the full Jacobian
    when keys/values depend on the input, nor task correctness.
    """
    if not 0.0 < step_size <= 1.0:
        raise ValueError('step_size must be in (0, 1]')
    if incoming.ndim != 5 or keys.ndim != 5 or values.ndim != 5:
        raise ValueError('incoming, keys and values must have five dimensions')
    b, ports, payload, h, w = incoming.shape
    reads = keys.shape[1]
    if keys.shape != (b, reads, ports, h, w):
        raise ValueError('incompatible key shape')
    if values.shape != (b, reads, payload, h, w):
        raise ValueError('incompatible value shape')
    residual = values
    if subtract_read:
        residual = values - torch.einsum('brphw,bpdhw->brdhw', keys, incoming)
    correction = torch.einsum('brphw,brdhw->bpdhw', keys, residual)
    return incoming + step_size * correction


class AddressedDeltaWriter(nn.Module):
    """Full-perception, four-read carrier update. Persistent state is unchanged.

    Zero KEYS and nonzero random VALUE head give an exact identity initial
    carrier write. Do NOT zero both heads. The key branch can receive a task
    gradient once the unchanged downstream Q/readout exposes one.
    """
    def __init__(self, input_channels: int = 67, hidden: int = 40,
                 ports: int = 4, payload: int = 6, reads: int = 4,
                 step_size: float = 0.1, *, subtract_read: bool = True) -> None:
        super().__init__()
        if min(input_channels, hidden, ports, payload, reads) <= 0:
            raise ValueError('all channel counts must be positive')
        if not 0.0 < step_size <= 1.0:
            raise ValueError('step_size must be in (0, 1]')
        self.input_channels = input_channels
        self.ports, self.payload, self.reads = ports, payload, reads
        self.step_size, self.subtract_read = step_size, subtract_read
        self.perception = nn.Conv2d(input_channels, hidden, 1)
        # Construct/init the value head normally; the zero key prevents writes.
        self.value_head = nn.Conv2d(hidden, reads * payload, 1)
        self.key_head = nn.Conv2d(hidden, reads * ports, 1)
        nn.init.zeros_(self.key_head.weight)
        nn.init.zeros_(self.key_head.bias)

    def controls(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 4 or features.shape[1] != self.input_channels:
            raise ValueError('features has the wrong shape')
        b, _, h, w = features.shape
        hidden = torch.tanh(self.perception(features))
        raw = self.key_head(hidden).reshape(b, self.reads, self.ports, h, w)
        values = self.value_head(hidden).reshape(b, self.reads, self.payload, h, w)
        return normalize_reads(raw), values

    def forward(self, incoming: Tensor, features: Tensor) -> Tensor:
        if incoming.ndim != 4 or incoming.shape[1] != self.ports * self.payload:
            raise ValueError('incoming has the wrong shape')
        b, _, h, w = incoming.shape
        if features.shape[0] != b or features.shape[2:] != (h, w):
            raise ValueError('incoming/features batch or spatial dimensions differ')
        keys, values = self.controls(features)
        lanes = incoming.reshape(b, self.ports, self.payload, h, w)
        updated = addressed_write(lanes, keys, values, self.step_size,
                                  subtract_read=self.subtract_read)
        return updated.reshape_as(incoming)


def self_test() -> dict:
    """CPU plumbing/algebra checks only; no task training or seed4 inference."""
    torch.set_num_threads(2)
    torch.manual_seed(7281)
    dtype = torch.float64
    eta = 0.1
    u = torch.randn(64, 4, 6, 1, 1, dtype=dtype)
    v = torch.randn(64, 4, 6, 1, 1, dtype=dtype)
    k = normalize_reads(torch.randn(64, 4, 4, 1, 1, dtype=dtype))
    out = addressed_write(u, k, v, eta)
    # Read residual identity: e'=(I-eta KK^T)e.
    e = v - torch.einsum('brphw,bpdhw->brdhw', k, u)
    ep = v - torch.einsum('brphw,bpdhw->brdhw', k, out)
    kt_e = torch.einsum('brphw,brdhw->bpdhw', k, e)
    ep_ref = e - eta * torch.einsum('brphw,bpdhw->brdhw', k, kt_e)
    assert torch.allclose(ep, ep_ref, atol=1e-12, rtol=1e-12)
    assert torch.all(ep.square().sum((1, 2)) <= e.square().sum((1, 2)) + 1e-11)
    # Energy bound, not a constant state bound.
    slack = (u.square().sum((1, 2)) + eta*v.square().sum((1, 2))
             - out.square().sum((1, 2)))
    assert slack.min() >= -1e-11
    # Same controls on two inputs: nonexpansion.
    other = torch.randn_like(u)
    out_other = addressed_write(other, k, v, eta)
    assert torch.all((out-out_other).square().sum((1, 2))
                     <= (u-other).square().sum((1, 2)) + 1e-11)
    # Unread directions in a rank-one example are preserved exactly.
    kr = torch.zeros(1, 1, 4, 1, 1, dtype=dtype)
    kr[:, :, 0] = 0.75
    ur = torch.randn(1, 4, 6, 1, 1, dtype=dtype)
    vr = torch.randn(1, 1, 6, 1, 1, dtype=dtype)
    outr = addressed_write(ur, kr, vr, eta)
    assert torch.equal(outr[:, 1:], ur[:, 1:])
    # Already-satisfied local read produces no write.
    matched = torch.einsum('brphw,bpdhw->brdhw', k, u)
    assert torch.equal(addressed_write(u, k, matched, eta), u)
    # Learnable zero-key initial state.
    model = AddressedDeltaWriter().double()
    xi = torch.randn(2, 67, 3, 4, dtype=dtype)
    incoming = torch.randn(2, 24, 3, 4, dtype=dtype)
    result = model(incoming, xi)
    assert torch.equal(result, incoming)
    (result*torch.randn_like(result)).sum().backward()
    key_grad = float(model.key_head.weight.grad.norm())
    assert key_grad > 0.0
    assert model.value_head.weight.grad.abs().max().item() == 0.0
    # All branches become differentiable once keys are nonzero.
    with torch.no_grad():
        model.key_head.weight.normal_(0.0, 0.01)
    model.zero_grad(set_to_none=True)
    model(incoming, xi).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    assert model.value_head.weight.grad.norm() > 0
    # Huge finite float32 raw keys: normalization does not square them first.
    huge = torch.full((1, 4, 4, 1, 1), 1e30, dtype=torch.float32)
    kh = normalize_reads(huge)
    assert torch.isfinite(kh).all()
    assert kh.square().sum().item() <= 1.000001
    return {'status': 'PASS', 'scope': 'CPU primitive algebra and autograd only',
            'torch': torch.__version__, 'tests': [
                'read_error_identity', 'read_error_nonincrease', 'energy_bound',
                'frozen_controls_nonexpansion', 'unread_subspace_preservation',
                'satisfied_read_no_write', 'initial_identity', 'zero_key_gradient',
                'active_branch_gradients', 'large_raw_key_normalization'],
            'writer_parameters': sum(p.numel() for p in model.parameters()),
            'key_gradient_norm_at_identity': key_grad,
            'max_read_error_identity_difference': float((ep-ep_ref).abs().max()),
            'minimum_sampled_energy_bound_slack': float(slack.min()),
            'task_training_performed': False, 'cuda_qualified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--json-out')
    args = parser.parse_args()
    if not args.self_test:
        parser.error('use --self-test; this file does not launch training')
    report = self_test()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out:
        from pathlib import Path
        Path(args.json_out).write_text(text+'\n', encoding='utf-8')
    print(text)
