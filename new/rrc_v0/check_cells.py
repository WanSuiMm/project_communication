"""Small CPU implementation checks for RRC-v0 cell construction."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
NEW_ROOT = HERE.parent
if str(NEW_ROOT / "streaming_carry") not in sys.path:
    sys.path.insert(0, str(NEW_ROOT / "streaming_carry"))

from cells import ARMS, make_model, relation_mix  # noqa: E402
from stream_cells import StreamingCell, stream  # noqa: E402


def tensor_hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def hash_module_state(model: torch.nn.Module, prefixes: tuple[str, ...]) -> dict[str, str]:
    return {
        key: tensor_hash(value)
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def check_current_equivalence(seed: int = 127) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = StreamingCell(streaming=True)
    current = make_model("current", seed)
    assert reference.state_dict().keys() == current.state_dict().keys()
    assert all(
        torch.equal(reference.state_dict()[key], current.state_dict()[key])
        for key in reference.state_dict()
    )

    generator = torch.Generator(device="cpu").manual_seed(1001)
    x = torch.rand((2, 3, 5, 7), generator=generator)
    x[:, :1] = (x[:, :1] > 0.25).float()
    reference_state = reference.initial(x)
    current_state = current.initial(x)
    for old, new in zip(reference_state, current_state):
        assert torch.equal(old, new)
    reference_next = reference.step(reference_state, x)
    current_next = current.step(current_state, x)
    for old, new in zip(reference_next, current_next):
        assert torch.equal(old, new)


def check_shared_initialization(seed: int = 127) -> dict[str, dict[str, str]]:
    models = {arm: make_model(arm, seed) for arm in ARMS}
    common_prefixes = ("encoder.", "q_in.", "q_out.", "readout.")
    common_hashes = {arm: hash_module_state(model, common_prefixes) for arm, model in models.items()}
    assert common_hashes["current"] == common_hashes["factorized"]
    assert common_hashes["current"] == common_hashes["rrc"]

    factorized_f = hash_module_state(models["factorized"], ("f_in.", "f_out."))
    rrc_f = hash_module_state(models["rrc"], ("f_in.", "f_out."))
    assert factorized_f == rrc_f
    assert not hasattr(models["factorized"], "relation_logits")
    assert not hasattr(models["factorized"], "relation_gate")
    assert models["rrc"].relation_logits.requires_grad
    assert models["rrc"].relation_gate.requires_grad
    return {"common_E_Q_readout": common_hashes["current"], "shared_F": factorized_f}


def check_relation_reference() -> None:
    generator = torch.Generator(device="cpu").manual_seed(222)
    incoming = torch.randn((2, 24, 4, 6), generator=generator)
    model = make_model("rrc", 127)
    actual = relation_mix(incoming, model.relation_logits, model.relation_gate)

    lanes = incoming.chunk(4, dim=1)
    permutations = (
        torch.cat((lanes[0], lanes[1], lanes[2], lanes[3]), dim=1),
        torch.cat((lanes[1], lanes[2], lanes[3], lanes[0]), dim=1),
        torch.cat((lanes[2], lanes[3], lanes[0], lanes[1]), dim=1),
        torch.cat((lanes[3], lanes[0], lanes[1], lanes[2]), dim=1),
    )
    average = sum(permutations) / 4.0
    rho = torch.tensor(0.1)
    expected = incoming + rho * (average - incoming)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)

    # Exercise the differentiable relation path and both trainable coefficients.
    probe = torch.randn(actual.shape, generator=generator)
    loss = (actual * probe).sum()
    grad_logits, grad_gate = torch.autograd.grad(
        loss, (model.relation_logits, model.relation_gate)
    )
    assert torch.isfinite(grad_logits).all() and grad_logits.norm() > 0
    assert torch.isfinite(grad_gate).all() and grad_gate.norm() > 0


def check_stream_isometry() -> None:
    generator = torch.Generator(device="cpu").manual_seed(333)
    carrier = torch.randn((3, 24, 8, 9), generator=generator)
    mask = (torch.rand((3, 1, 8, 9), generator=generator) > 0.3).float()
    moved = stream(carrier, mask)
    torch.testing.assert_close(moved.square().sum(), carrier.square().sum(), rtol=0.0, atol=1e-5)


def check_observers_and_shapes() -> None:
    generator = torch.Generator(device="cpu").manual_seed(444)
    x = torch.rand((2, 3, 5, 7), generator=generator)
    x[:, :1] = (x[:, :1] > 0.2).float()
    seen: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def observe(model, incoming_u, relation_h, observation_x):
        seen[model.arm] = (incoming_u, relation_h, observation_x)

    for arm in ARMS:
        model = make_model(arm, 127)
        model.relation_observer = observe
        state = model.initial(x)
        next_state = model.step(state, x)
        assert len(next_state) == 2
        assert next_state[0].shape == (2, 24, 5, 7)
        assert next_state[1].shape == (2, 8, 5, 7)
        assert all(torch.isfinite(value).all() for value in next_state)
        incoming_u, relation_h, observed_x = seen[arm]
        assert observed_x is x
        assert incoming_u.shape == relation_h.shape == (2, 24, 5, 7)
        if arm in ("current", "factorized"):
            assert incoming_u is relation_h
        else:
            assert not torch.equal(incoming_u, relation_h)

        # The observer is a plain attribute and never enters the model state.
        assert "relation_observer" not in model.state_dict()


def main() -> None:
    torch.set_num_threads(1)
    check_current_equivalence()
    hashes = check_shared_initialization()
    check_relation_reference()
    check_stream_isometry()
    check_observers_and_shapes()

    counts = {
        arm: {
            "parameters": sum(p.numel() for p in make_model(arm, 127).parameters()),
            "trainable": sum(
                p.numel() for p in make_model(arm, 127).parameters() if p.requires_grad
            ),
        }
        for arm in ARMS
    }
    assert counts["current"] == {"parameters": 5033, "trainable": 5033}
    assert counts["factorized"] == {"parameters": 4983, "trainable": 4983}
    assert counts["rrc"] == {"parameters": 4988, "trainable": 4988}
    print({"status": "ok", "parameter_counts": counts, "hashes": hashes})


if __name__ == "__main__":
    main()
