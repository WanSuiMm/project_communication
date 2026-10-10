"""Frozen configuration and local runtime helpers; no execution on import."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from data import sha256

HERE = Path(__file__).resolve().parent
CONFIG = {
    "protocol": "real_task_nca_v0", "updates": 1500, "batch": 2,
    "channels": 16, "hidden": 128, "perception_width": 64,
    "input_size": 192, "scored_center": 64, "halo": 64,
    "train_steps": 64, "fire_probability": .5,
    "lr": .001, "weight_decay": 0., "init_seed": 200101,
    "schedule_seed": 200201, "eval_seed": 200401,
    "evaluation_repeats": 2, "eval_steps": [8, 16, 32, 64, 128, 256],
    "checkpoints": [250, 500, 1000, 1500], "recovery_interval": 25,
    "checkpoint_chunk": 8, "per_tensor_gradient_normalization": True,
    "teacher_seeds": [201101, 201102, 201103, 201104, 201105],
    "oof_teacher_steps": 1200, "full_teacher_steps": 1600,
    "arms": {"standard_k64": ["original", 64, 64],
             "standard_t8": ["original", 8, 8],
             "standard_k8": ["original", 8, 64],
             "au_k8": ["au", 8, 64]},
    "supervision": "terminal_only", "runtime_cap": None,
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    files = sorted(HERE.glob("*.py")) + [HERE / "PROTOCOL.md"]
    return {p.name: sha256(p) for p in files}


def setup(device):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no silent device fallback")
    return result


def runtime_binding(device):
    return {"device": str(device), "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


@contextmanager
def run_lock(out):
    """An OS-held lock survives client exit and releases on worker exit."""
    path = Path(out) / "worker.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def masked_loss(logits, target, mask):
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum()
    bce = bce / mask.sum().clamp_min(1)
    probability = logits.sigmoid()
    axes = (1, 2, 3)
    overlap = (probability * target * mask).sum(axes)
    denom = ((probability + target) * mask).sum(axes)
    dice = (2 * overlap + 1) / (denom + 1)
    return .5 * bce + .5 * (1 - dice.mean())


def finite_model(model):
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def sample_batch(images, update, device):
    """Uniform FOV-centred crops, shared transforms, same schedule in all arms."""
    rng = np.random.default_rng(CONFIG["schedule_seed"] + 104729 * update)
    size = CONFIG["input_size"]
    half = size // 2
    result = {key: [] for key in ("rgb", "target", "fov", "coarse")}
    for _ in range(CONFIG["batch"]):
        image = images[int(rng.integers(len(images)))]
        coords = np.argwhere(image["fov"][0] > .5)
        if not len(coords):
            raise ValueError("Training image has empty FOV")
        y, x = coords[int(rng.integers(len(coords)))]
        turns, flip_x, flip_y = int(rng.integers(4)), bool(rng.integers(2)), bool(rng.integers(2))
        for key in result:
            value = np.pad(image[key], ((0, 0), (half, half), (half, half)), mode="constant")
            value = value[:, y:y + size, x:x + size]
            value = np.rot90(value, turns, axes=(-2, -1))
            if flip_x:
                value = value[..., ::-1]
            if flip_y:
                value = value[..., ::-1, :]
            result[key].append(np.ascontiguousarray(value))
    return {key: torch.from_numpy(np.stack(value)).to(device) for key, value in result.items()}


def training_fires(update, steps, shape, device):
    generator = torch.Generator(device=device).manual_seed(CONFIG["schedule_seed"] + 10000000 + 104729 * update)
    # Always generate the identical T64 plan, even for the separately trained
    # T8 reference; CUDA kernels can consume RNG differently for different sizes.
    plan = (torch.rand((CONFIG["train_steps"], shape[0], 1, *shape[-2:]),
                      generator=generator, device=device) < .5).float()
    return plan[:steps]
