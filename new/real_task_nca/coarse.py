"""Coarse DRIVE teacher: a small U-Net and deterministic training utilities."""

from __future__ import annotations

import os
import math
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .data import load_image
except ImportError:  # supports direct script-style imports from this directory
    from data import load_image


MODEL_CONFIG = {"in_channels": 3, "out_channels": 1, "width": 16, "down_levels": 3}
PATCH_SIZE = 128
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
CHECKPOINT_INTERVAL = 100
PREDICT_TILE = 256
PREDICT_STRIDE = 128


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        value = self.reduce(value)
        return self.block(torch.cat((skip, value), dim=1))


class SmallUNet(nn.Module):
    """Four-level U-Net with three downsamplings and a width-16 stem."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.bottleneck = ConvBlock(width * 4, width * 8)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dec3 = UpBlock(width * 8, width * 4, width * 4)
        self.dec2 = UpBlock(width * 4, width * 2, width * 2)
        self.dec1 = UpBlock(width * 2, width, width)
        self.head = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(value)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        center = self.bottleneck(self.pool(enc3))
        decoded = self.dec3(center, enc3)
        decoded = self.dec2(decoded, enc2)
        decoded = self.dec1(decoded, enc1)
        return self.head(decoded)


def _record_id(record: Any, fallback: Any = None) -> str:
    if isinstance(record, Mapping):
        value = record.get("image_id", record.get("id", fallback))
    else:
        value = getattr(record, "image_id", getattr(record, "id", fallback))
    if value is None:
        raise ValueError("Every training record needs an image_id or id")
    return str(value).zfill(2)


def _record_map(records: Any) -> dict[str, Any]:
    if hasattr(records, "train"):
        source = records.train
    elif isinstance(records, Mapping):
        if "train" in records and isinstance(records["train"], Sequence):
            source = records["train"]
        elif all(str(key).isdigit() for key in records):
            return {str(key).zfill(2): value for key, value in records.items()}
        else:
            source = [records]
    else:
        source = records
    mapped: dict[str, Any] = {}
    for record in source:
        image_id = _record_id(record)
        if image_id in mapped:
            raise ValueError(f"Duplicate record ID {image_id}")
        mapped[image_id] = record
    return mapped


def _load_training_arrays(record: Any, image_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(record, Mapping) and all(name in record for name in ("rgb", "target", "fov")):
        data = record
        rgb = np.asarray(data["rgb"], dtype=np.float32)
        target = np.asarray(data["target"], dtype=np.float32)
        fov = np.asarray(data["fov"], dtype=np.float32)
    else:
        data = load_image(record)
        rgb = np.asarray(data["rgb"], dtype=np.float32)
        target_value = data["target"]
        if target_value is None:
            raise ValueError(f"Training record {image_id} has no target")
        target = np.asarray(target_value, dtype=np.float32)
        fov = np.asarray(data["fov"], dtype=np.float32)

    if rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.shape[0] != 3:
        rgb = rgb.transpose(2, 0, 1)
    if target.ndim == 2:
        target = target[None]
    if fov.ndim == 2:
        fov = fov[None]
    if rgb.ndim != 3 or rgb.shape[0] != 3 or target.ndim != 3 or target.shape[0] != 1:
        raise ValueError(f"Invalid RGB/target shapes for image {image_id}: {rgb.shape}, {target.shape}")
    if fov.shape != target.shape or rgb.shape[-2:] != target.shape[-2:]:
        raise ValueError(f"Spatial shape mismatch for training record {image_id}")
    if not (np.isfinite(rgb).all() and np.isfinite(target).all() and np.isfinite(fov).all()):
        raise ValueError(f"Non-finite values in training record {image_id}")
    if np.any((rgb < 0) | (rgb > 1)):
        raise ValueError(f"RGB values for image {image_id} must be in [0,1]")
    return tuple(np.ascontiguousarray(item, dtype=np.float32) for item in (rgb, target, fov))  # type: ignore[return-value]


def _masked_bce_dice(logits: torch.Tensor, target: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
    binary_cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (binary_cross_entropy * fov).sum() / fov.sum().clamp_min(1.0)
    probability = torch.sigmoid(logits)
    axes = (1, 2, 3)
    intersection = (probability * target * fov).sum(dim=axes)
    denominator = ((probability + target) * fov).sum(dim=axes)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 0.5 * bce + 0.5 * (1.0 - dice.mean())


def _augment(
    rgb: np.ndarray, target: np.ndarray, fov: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    turns = int(rng.integers(0, 4))
    if turns:
        rgb = np.rot90(rgb, turns, axes=(-2, -1))
        target = np.rot90(target, turns, axes=(-2, -1))
        fov = np.rot90(fov, turns, axes=(-2, -1))
    if bool(rng.integers(0, 2)):
        rgb, target, fov = (np.flip(value, axis=-1) for value in (rgb, target, fov))
    if bool(rng.integers(0, 2)):
        rgb, target, fov = (np.flip(value, axis=-2) for value in (rgb, target, fov))
    return tuple(np.ascontiguousarray(value) for value in (rgb, target, fov))  # type: ignore[return-value]


def _sample_patch(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb, target, fov = arrays
    height, width = rgb.shape[-2:]
    if height < PATCH_SIZE or width < PATCH_SIZE:
        pad_h, pad_w = max(0, PATCH_SIZE - height), max(0, PATCH_SIZE - width)
        rgb = np.pad(rgb, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        target = np.pad(target, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
        fov = np.pad(fov, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
        height, width = rgb.shape[-2:]
    top = int(rng.integers(0, height - PATCH_SIZE + 1))
    left = int(rng.integers(0, width - PATCH_SIZE + 1))
    region = (slice(None), slice(top, top + PATCH_SIZE), slice(left, left + PATCH_SIZE))
    return _augment(rgb[region], target[region], fov[region], rng)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as tmp:
        temporary = Path(tmp.name)
        torch.save(dict(payload), tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(temporary, path)


def _assert_finite_model(model: nn.Module, stage: str) -> None:
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(f"Nonfinite model parameter at {stage}: {name}")


def _value_is_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_value_is_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_value_is_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _assert_finite_optimizer(optimizer: torch.optim.Optimizer, stage: str) -> None:
    for state in optimizer.state.values():
        if not _value_is_finite(state):
            raise FloatingPointError(f"Nonfinite optimizer state at {stage}")


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch releases
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"Invalid checkpoint at {path}")
    return value


def train_teacher(
    records: Any,
    trainids: Sequence[str | int],
    out: str | os.PathLike[str],
    seed: int,
    steps: int,
    device: str | torch.device,
) -> Path:
    """Train or resume a coarse teacher and return its checkpoint path.

    Training uses 128-pixel patches, batch size 8, synchronized flips and
    quarter-turn rotations, masked BCE plus soft Dice, and AdamW at 1e-3 with
    zero weight decay. Checkpoints are atomically replaced every 100 updates
    and include optimizer and RNG states for deterministic continuation.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    train_ids = sorted({str(image_id).zfill(2) for image_id in trainids}, key=int)
    if len(train_ids) != len(trainids) or not train_ids:
        raise ValueError("trainids must be nonempty and unique")
    record_map = _record_map(records)
    missing = [image_id for image_id in train_ids if image_id not in record_map]
    if missing:
        raise KeyError(f"Missing training records: {missing}")
    image_arrays = {image_id: _load_training_arrays(record_map[image_id], image_id) for image_id in train_ids}

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    output = Path(out)
    if output.suffix.lower() in {".pt", ".pth"}:
        checkpoint_path = output
    else:
        checkpoint_path = output / "teacher_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model = SmallUNet(width=MODEL_CONFIG["width"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    target_device = torch.device(device)
    model.to(target_device)
    start_step = 0

    if checkpoint_path.is_file():
        saved = _safe_torch_load(checkpoint_path)
        if list(saved.get("train_ids", [])) != train_ids or int(saved.get("seed", -1)) != seed:
            raise ValueError("Existing checkpoint provenance does not match requested train IDs and seed")
        if dict(saved.get("model_config", {})) != MODEL_CONFIG:
            raise ValueError("Existing checkpoint model configuration does not match")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        _assert_finite_model(model, "checkpoint restore")
        _assert_finite_optimizer(optimizer, "checkpoint restore")
        start_step = int(saved.get("step", 0))
        if "numpy_rng_state" in saved:
            rng.bit_generator.state = saved["numpy_rng_state"]
        if "python_rng_state" in saved:
            random.setstate(saved["python_rng_state"])
        if "torch_rng_state" in saved:
            torch.set_rng_state(saved["torch_rng_state"])
        if torch.cuda.is_available() and saved.get("torch_cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(saved["torch_cuda_rng_state"])
        if start_step >= steps:
            return checkpoint_path

    for step in range(start_step + 1, steps + 1):
        batch_rgb: list[np.ndarray] = []
        batch_target: list[np.ndarray] = []
        batch_fov: list[np.ndarray] = []
        for _ in range(BATCH_SIZE):
            image_id = train_ids[int(rng.integers(0, len(train_ids)))]
            rgb_patch, target_patch, fov_patch = _sample_patch(image_arrays[image_id], rng)
            batch_rgb.append(rgb_patch)
            batch_target.append(target_patch)
            batch_fov.append(fov_patch)

        inputs = torch.from_numpy(np.stack(batch_rgb)).to(target_device)
        targets = torch.from_numpy(np.stack(batch_target)).to(target_device)
        masks = torch.from_numpy(np.stack(batch_fov)).to(target_device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError(f"Nonfinite teacher logits at update {step}")
        loss = _masked_bce_dice(logits, targets, masks)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite teacher loss at update {step}")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"Nonfinite teacher gradient at update {step}: {name}")
        optimizer.step()
        _assert_finite_model(model, f"update {step}")
        _assert_finite_optimizer(optimizer, f"update {step}")

        if step % CHECKPOINT_INTERVAL == 0 or step == steps:
            # Check again at the persistence boundary so a bad update can
            # never replace the last finite resumable checkpoint.
            _assert_finite_model(model, f"checkpoint at update {step}")
            _assert_finite_optimizer(optimizer, f"checkpoint at update {step}")
            checkpoint = {
                "format_version": 1,
                "step": step,
                "requested_steps": int(steps),
                "seed": seed,
                "train_ids": train_ids,
                "model_config": dict(MODEL_CONFIG),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "numpy_rng_state": rng.bit_generator.state,
                "python_rng_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "provenance": {
                    "architecture": "SmallUNet",
                    "patch_size": PATCH_SIZE,
                    "batch_size": BATCH_SIZE,
                    "loss": "0.5 masked BCEWithLogits + 0.5 masked soft Dice",
                    "optimizer": "AdamW",
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "augmentation": ["horizontal_flip", "vertical_flip", "rot90"],
                    "checkpoint_interval": CHECKPOINT_INTERVAL,
                    "torch_version": torch.__version__,
                },
            }
            _atomic_torch_save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_teacher(path: str | os.PathLike[str], device: str | torch.device) -> SmallUNet:
    """Load a trained teacher checkpoint, including its saved architecture."""

    checkpoint = _safe_torch_load(Path(path))
    config = dict(checkpoint.get("model_config", MODEL_CONFIG))
    if config != MODEL_CONFIG:
        raise ValueError(f"Unsupported coarse model configuration: {config}")
    model = SmallUNet(width=config["width"])
    model.load_state_dict(checkpoint["model_state"])
    _assert_finite_model(model, "load_teacher")
    model.to(torch.device(device))
    model.eval()
    return model


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


def _gaussian_weight(tile: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, tile, dtype=np.float32)
    weight_1d = np.exp(-0.5 * (axis / 0.5) ** 2).astype(np.float32)
    weight_1d = np.maximum(weight_1d, np.float32(1e-3))
    return np.outer(weight_1d, weight_1d).astype(np.float32)


def _coerce_rgb(rgb: np.ndarray | torch.Tensor) -> np.ndarray:
    value = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3 and value.shape[0] != 3 and value.shape[-1] == 3:
        value = value.transpose(2, 0, 1)
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"rgb must be [3,H,W] or [H,W,3], got {value.shape}")
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError("rgb must contain finite values in [0,1]")
    return np.ascontiguousarray(value, dtype=np.float32)


def predict_image(
    model: nn.Module,
    rgb: np.ndarray | torch.Tensor,
    device: str | torch.device,
) -> np.ndarray:
    """Predict a full-resolution probability map using overlapping 256 tiles.

    Tiles advance by 128 pixels and are combined with Gaussian overlap
    weights. Images smaller than a tile are edge-padded and cropped back to
    their original dimensions. No resizing or pixel-grid interpolation is
    performed. Returns float32 ``[1,H,W]`` probabilities.
    """

    image = _coerce_rgb(rgb)
    _, height, width = image.shape
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    tile = PREDICT_TILE
    starts_y = _tile_starts(height, tile, PREDICT_STRIDE)
    starts_x = _tile_starts(width, tile, PREDICT_STRIDE)
    weight = _gaussian_weight(tile)
    accumulated = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    with torch.inference_mode():
        for top in starts_y:
            for left in starts_x:
                tile_array = image[:, top:min(top + tile, height), left:min(left + tile, width)]
                pad_h = tile - tile_array.shape[-2]
                pad_w = tile - tile_array.shape[-1]
                tensor = torch.from_numpy(tile_array[None]).to(target_device)
                if pad_h or pad_w:
                    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
                logits = model(tensor)
                if logits.ndim != 4 or logits.shape[:2] != (1, 1):
                    raise ValueError(f"Teacher must return [1,1,H,W] logits, got {tuple(logits.shape)}")
                probability = torch.sigmoid(logits[0, 0]).detach().cpu().numpy().astype(np.float32)
                valid_h = min(tile, height - top)
                valid_w = min(tile, width - left)
                local_weight = weight[:valid_h, :valid_w]
                accumulated[top:top + valid_h, left:left + valid_w] += (
                    probability[:valid_h, :valid_w] * local_weight
                )
                weight_sum[top:top + valid_h, left:left + valid_w] += local_weight

    prediction = accumulated / np.maximum(weight_sum, np.float32(1e-12))
    return prediction[None].astype(np.float32, copy=False)

