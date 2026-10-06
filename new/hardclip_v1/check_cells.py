"""Cheap CPU invariants for the HardClip-v1 cell implementation."""
from __future__ import annotations

import torch

from cells import ARMS, DEFAULT_CAPS, HardClipCell, make_model
from primitive import FixedLaneHardClip
from stream_cells import StreamingCell


def _same_tensor_mapping(left, right) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _neural_parity_and_initialization() -> tuple[HardClipCell, HardClipCell]:
    seed = 12061
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = StreamingCell()
    neural = make_model("neural", seed)
    hardclip = make_model("hardclip", seed)

    assert ARMS == ("neural", "hardclip")
    assert neural.state_dict().keys() == hardclip.state_dict().keys()
    _same_tensor_mapping(neural.state_dict(), hardclip.state_dict())
    assert neural.state_dict()["fixed_clip.max_write_rms"].tolist() == list(DEFAULT_CAPS)
    assert hardclip.state_dict()["fixed_clip.max_write_rms"].tolist() == list(DEFAULT_CAPS)
    assert "fixed_clip.max_write_rms" not in dict(neural.named_parameters())
    assert "fixed_clip.max_write_rms" not in dict(hardclip.named_parameters())
    assert sum(p.numel() for p in neural.parameters()) == 5033
    assert sum(p.numel() for p in hardclip.parameters()) == 5033
    assert tuple(dict(neural.named_parameters())) == tuple(dict(reference.named_parameters()))

    # Random binary worlds exercise the canonical transport/features/Q path.
    torch.manual_seed(81)
    x = torch.rand(2, 3, 5, 7)
    x[:, :1] = (x[:, :1] > 0.25).float()
    x_reference = x.clone().requires_grad_()
    x_neural = x.clone().requires_grad_()
    state_reference = reference.initial(x_reference)
    state_neural = neural.initial(x_neural)
    out_reference = reference.step(state_reference, x_reference)
    out_neural = neural.step(state_neural, x_neural)
    for expected, actual in zip(out_reference, out_neural):
        assert torch.equal(expected, actual)

    loss_reference = sum(value.square().sum() for value in out_reference)
    loss_neural = sum(value.square().sum() for value in out_neural)
    loss_reference.backward()
    loss_neural.backward()
    assert torch.equal(x_reference.grad, x_neural.grad)
    for (name_ref, parameter_ref), (name_new, parameter_new) in zip(
        reference.named_parameters(), neural.named_parameters()
    ):
        assert name_ref == name_new
        assert (parameter_ref.grad is None) == (parameter_new.grad is None), name_ref
        if parameter_ref.grad is not None:
            assert torch.equal(parameter_ref.grad, parameter_new.grad), name_ref
    return neural, hardclip


def _clip_branch_checks() -> None:
    layer = FixedLaneHardClip(DEFAULT_CAPS)
    caps = layer.max_write_rms

    # Below-cap writes preserve the original arithmetic and its exact gradient.
    small = torch.full((2, 24, 3, 2), 0.001, requires_grad=True)
    below = layer(small)
    assert torch.equal(below, 0.1 * small)
    grad, = torch.autograd.grad(below.sum(), small)
    assert torch.equal(grad, torch.full_like(small, 0.1))

    # Exact threshold ties take the raw branch and keep finite gradients.
    tie_layer = FixedLaneHardClip(DEFAULT_CAPS).double()
    tie_caps = tie_layer.max_write_rms
    tie_proposal = (tie_caps / tie_layer.eta).view(1, 4, 1, 1, 1).expand(1, 4, 6, 1, 1)
    tie_proposal = tie_proposal.reshape(1, 24, 1, 1).clone().requires_grad_()
    tie_raw = tie_layer.eta * tie_proposal
    assert torch.equal(
        tie_raw.reshape(1, 4, 6, 1, 1).square().mean(2).sqrt().reshape(4), tie_caps
    )
    tie_result = tie_layer(tie_proposal)
    assert torch.equal(tie_result, tie_raw)
    tie_grad, = torch.autograd.grad(tie_result.sum(), tie_proposal)
    assert torch.isfinite(tie_grad).all() and torch.equal(tie_grad, torch.full_like(tie_grad, 0.1))

    # A genuinely clipped vector stays on the raw ray and lands at each cap.
    torch.manual_seed(441)
    large = (torch.randn(2, 24, 1, 1) * 20).requires_grad_()
    clipped = layer(large)
    raw = 0.1 * large
    raw_lanes = raw.reshape(2, 4, 6, 1, 1)
    clip_lanes = clipped.reshape(2, 4, 6, 1, 1)
    raw_rms = raw_lanes.square().mean(2).sqrt()
    clipped_rms = clip_lanes.square().mean(2).sqrt()
    assert torch.allclose(
        clipped_rms, caps.view(1, 4, 1, 1).expand_as(clipped_rms), rtol=2e-6, atol=2e-7
    )
    scale = clip_lanes.reshape(2, 4, -1).norm(dim=2) / raw_lanes.reshape(2, 4, -1).norm(dim=2)
    expected_scale = (
        caps.view(1, 4) * 6 ** 0.5 / raw_lanes.reshape(2, 4, -1).norm(dim=2)
    )
    assert torch.allclose(scale, expected_scale, rtol=2e-6, atol=2e-7)
    clip_grad, = torch.autograd.grad(clipped.square().sum(), large)
    assert torch.isfinite(clip_grad).all()


def _observer_checks(neural: HardClipCell, hardclip: HardClipCell) -> None:
    torch.manual_seed(902)
    x = torch.rand(1, 3, 4, 5)
    x[:, :1] = (x[:, :1] > 0.3).float()
    state_n = neural.initial(x)
    no_observer = neural.step(state_n, x)
    assert neural.writer_observer is None

    seen = []
    neural.writer_observer = lambda details, observed_x: seen.append(
        (tuple(details), observed_x.shape, details["clipped_write"].detach().clone())
    )
    with_observer = neural.step(state_n, x)
    assert len(seen) == 1
    assert seen[0][0] == (
        "incoming", "proposal", "raw_write", "delta", "clipped_write", "caps"
    )
    assert all(torch.equal(a, b) for a, b in zip(no_observer, with_observer))
    neural.writer_observer = None

    hardclip_state = hardclip.initial(x)
    hardclip.writer_observer = lambda details, observed_x: seen.append(
        (tuple(details), observed_x.shape, details["delta"].detach().clone())
    )
    before_caps = hardclip.fixed_clip.max_write_rms.clone()
    out = hardclip.step(hardclip_state, x)
    assert len(seen) == 2
    assert seen[-1][0] == seen[0][0]
    assert torch.equal(before_caps, hardclip.fixed_clip.max_write_rms)
    assert all(torch.isfinite(value).all() for value in out)


def main() -> None:
    torch.set_num_threads(1)
    neural, hardclip = _neural_parity_and_initialization()
    _clip_branch_checks()
    _observer_checks(neural, hardclip)
    print("PASS: seeded arm parity/state, 5033 parameters, true clipping branches, observer invariants")


if __name__ == "__main__":
    main()

