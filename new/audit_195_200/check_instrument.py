"""One small CPU check for the zero-training 195/200 audit instrumentation."""
from __future__ import annotations

import torch
from torch import nn

from instrument import parts, preactivation_parts, semantic_decomposition
from stream_cells import StreamingCell
from masked_cells import masked_laplacian


def _assert_close(actual, expected, name: str, atol: float = 1e-6) -> float:
    if not torch.allclose(actual, expected, atol=atol, rtol=1e-6):
        difference = (actual - expected).abs().max().item()
        raise AssertionError(f"{name}: max absolute difference {difference:.3g}")
    return (actual - expected).abs().max().item()


def main() -> None:
    torch.manual_seed(195200)
    model = StreamingCell(
        workspace_channels=8,
        latent_channels=4,
        workspace_hidden=11,
        candidate_hidden=7,
    ).cpu()
    model.eval()
    # The production constructor zero-initializes these layers; use nonzero
    # outputs here so the check exercises both update paths without training.
    with torch.no_grad():
        for layer in (model.f_out, model.q_out, model.readout):
            nn.init.normal_(layer.weight, mean=0.0, std=0.08)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=0.02)

    batch, height, width = 1, 5, 6
    x = torch.rand(batch, 3, height, width)
    mask = torch.ones(batch, 1, height, width)
    mask[0, 0, 2, 2] = 0.0
    mask[0, 0, 1, 4] = 0.0
    x[:, :1] = mask
    state = (
        torch.randn(batch, model.workspace_channels, height, width),
        torch.randn(batch, model.latent_channels, height, width),
    )
    w, z = state

    measured = parts(model, state, x)
    assert tuple(measured) == (
        "incoming", "LW", "LZ", "F", "W_new", "Q", "Z_new", "dW", "dZ"
    )
    if measured["F"].abs().max().item() == 0 or measured["Q"].abs().max().item() == 0:
        raise AssertionError("randomized F and Q outputs must be nonzero")

    # Wall ports stay fixed; an open-to-wall link and an exterior link bounce.
    lanes = w.chunk(4, dim=1)
    incoming_lanes = measured["incoming"].chunk(4, dim=1)
    _assert_close(incoming_lanes[0][:, :, 2, 2], lanes[0][:, :, 2, 2], "wall port")
    _assert_close(incoming_lanes[3][:, :, 2, 1], lanes[1][:, :, 2, 1], "wall bounce")
    _assert_close(incoming_lanes[2][:, :, 0, 0], lanes[0][:, :, 0, 0], "edge bounce")

    direct = model.step(state, x)
    step_error = max(
        _assert_close(direct[0], measured["W_new"], "W_new"),
        _assert_close(direct[1], measured["Z_new"], "Z_new"),
    )

    decomposition = semantic_decomposition(model, state, x, measured)
    actual_logit_delta = model.logits(direct) - model.logits(state)
    logit_error = _assert_close(
        decomposition["dlogits"], actual_logit_delta, "logit delta"
    )
    telescope_error = _assert_close(
        decomposition["dlogits_sum"], decomposition["dlogits"], "logit telescope"
    )

    preactivation = model.q_in(model._features(w, z, x))
    reconstructed = sum(preactivation_parts(model, w, z, x).values())
    _assert_close(reconstructed, preactivation, "q_in block sum", atol=2e-6)

    # The binary masked Laplacian is symmetric, so its vector-Jacobian product
    # equals the same operator on the probe; wall values have zero derivative.
    h = torch.randn(batch, 3, height, width, requires_grad=True)
    probe = torch.randn_like(h)
    loss = (masked_laplacian(h, mask) * probe).sum()
    gradient = torch.autograd.grad(loss, h)[0]
    _assert_close(gradient, masked_laplacian(probe, mask), "masked-L derivative")
    if torch.count_nonzero(gradient[:, :, 2, 2]).item() != 0:
        raise AssertionError("masked wall values must have zero derivative")

    print(
        "audit_195_200_instrument: PASS "
        f"(step={step_error:.2g}, logit={logit_error:.2g}, "
        f"telescope={telescope_error:.2g})"
    )


if __name__ == "__main__":
    main()
