"""Data utilities for the DRIVE real-task NCA experiment.

Only the official training labels are ever discovered or opened. Test images
and their optional field-of-view masks can be loaded, while test labels are
deliberately absent from :class:`DriveRecord` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


OUTER_VAL_IDS = ("21", "26", "31", "36")
OFFICIAL_TRAIN_IDS = tuple(f"{number:02d}" for number in range(21, 41))


@dataclass(frozen=True)
class DriveRecord:
    """Paths and identity for one source image.

    ``target_path`` is populated only for official training images. For test
    records it is always ``None``; test manual annotations are never probed.
    """

    image_id: str
    image_path: Path
    target_path: Path | None
    fov_path: Path | None
    split: str

    @property
    def id(self) -> str:
        return self.image_id


@dataclass(frozen=True)
class DriveDataset:
    """DRIVE source records rooted at the directory containing train/test."""

    root: Path
    train: tuple[DriveRecord, ...]
    test: tuple[DriveRecord, ...]

    @property
    def records(self) -> tuple[DriveRecord, ...]:
        return self.train + self.test

    def __iter__(self):
        # Iteration defaults to labelled training records for safe callers.
        return iter(self.train)


def _locate_root(root: str | os.PathLike[str]) -> Path:
    candidate = Path(root).expanduser()
    candidates = (candidate, candidate / "DRIVE", candidate / "drive")
    for path in candidates:
        if (path / "training").is_dir():
            return path.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find DRIVE/training under: {searched}")


def _indexed_tiffs(directory: Path, suffix: str) -> dict[str, Path]:
    pattern = re.compile(rf"^(\d+)_({re.escape(suffix)})\.tif$", re.IGNORECASE)
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for path in directory.iterdir():
        match = pattern.match(path.name)
        if match:
            image_id = match.group(1)
            if image_id in found:
                raise ValueError(f"Duplicate DRIVE image ID {image_id} in {directory}")
            found[image_id] = path
    return found


def discover_drive(root: str | os.PathLike[str]) -> DriveDataset:
    """Discover DRIVE records from its standard training/test layout.

    The required training image IDs are 21 through 40. Test images are
    optional. The test ``1st_manual`` directory is never listed, checked, or
    opened; only test RGB images and optional ``*_test_mask.gif`` FOV masks
    are included.
    """

    drive_root = _locate_root(root)
    training_root = drive_root / "training"
    images = _indexed_tiffs(training_root / "images", "training")
    expected = set(OFFICIAL_TRAIN_IDS)
    missing = sorted(expected - set(images), key=int)
    unexpected = sorted(set(images) - expected, key=int)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing official training images: {missing}")
        if unexpected:
            details.append(f"unexpected training image IDs: {unexpected}")
        raise ValueError("DRIVE training image set mismatch (" + "; ".join(details) + ")")

    labels_root = training_root / "1st_manual"
    fov_root = training_root / "mask"
    train_records: list[DriveRecord] = []
    for image_id in OFFICIAL_TRAIN_IDS:
        label = labels_root / f"{image_id}_manual1.gif"
        fov = fov_root / f"{image_id}_training_mask.gif"
        if not label.is_file():
            raise FileNotFoundError(f"Missing official training label: {label.name}")
        if not fov.is_file():
            raise FileNotFoundError(f"Missing training FOV mask: {fov.name}")
        train_records.append(DriveRecord(image_id, images[image_id], label, fov, "train"))

    test_root = drive_root / "test"
    test_images = _indexed_tiffs(test_root / "images", "test")
    test_fov_root = test_root / "mask"
    test_records: list[DriveRecord] = []
    for image_id in sorted(test_images, key=int):
        fov = test_fov_root / f"{image_id}_test_mask.gif"
        test_records.append(
            DriveRecord(image_id, test_images[image_id], None, fov if fov.is_file() else None, "test")
        )

    return DriveDataset(drive_root, tuple(train_records), tuple(test_records))


def load_image(record: DriveRecord | Mapping[str, Any]) -> dict[str, Any]:
    """Load one source image without resizing or fitting preprocessing.

    Returns ``rgb`` as float32 ``[3,H,W]`` in ``[0,1]`` and ``fov`` as
    float32 ``[1,H,W]``. Training records also return a binary ``target`` of
    the same spatial size. For test records, ``target`` is ``None``. A
    missing optional test FOV mask is represented by an all-ones mask.
    """

    image_id = str(record["image_id"] if isinstance(record, Mapping) else record.image_id)
    image_path = Path(record["image_path"] if isinstance(record, Mapping) else record.image_path)
    target_value = record.get("target_path") if isinstance(record, Mapping) else record.target_path
    fov_value = record.get("fov_path") if isinstance(record, Mapping) else record.fov_path
    target_path = Path(target_value) if target_value is not None else None
    fov_path = Path(fov_value) if fov_value is not None else None

    with Image.open(image_path) as image:
        rgb_hwc = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb = np.ascontiguousarray(rgb_hwc.transpose(2, 0, 1), dtype=np.float32) / np.float32(255.0)
    height, width = rgb.shape[-2:]

    if fov_path is None:
        fov = np.ones((1, height, width), dtype=np.float32)
    else:
        with Image.open(fov_path) as mask_image:
            mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
        if mask.shape != (height, width):
            raise ValueError(f"FOV mask size mismatch for image {image_id}: {mask.shape} vs {(height, width)}")
        fov = (mask[None] > 0).astype(np.float32)

    target: np.ndarray | None = None
    if target_path is not None:
        if str(record.get("split", "train") if isinstance(record, Mapping) else record.split) != "train":
            raise ValueError("A target path is permitted only on an official training record")
        with Image.open(target_path) as label_image:
            label = np.asarray(label_image.convert("L"), dtype=np.uint8)
        if label.shape != (height, width):
            raise ValueError(f"Target size mismatch for image {image_id}: {label.shape} vs {(height, width)}")
        target = (label[None] > 0).astype(np.float32)

    return {"id": image_id, "image_id": image_id, "rgb": rgb, "target": target, "fov": fov}


def split_ids() -> tuple[list[str], list[str]]:
    """Return sorted outer-training IDs and the frozen outer-validation IDs."""

    val_ids = list(OUTER_VAL_IDS)
    train_ids = [image_id for image_id in OFFICIAL_TRAIN_IDS if image_id not in set(val_ids)]
    return train_ids, val_ids


def inner_folds(trainids: Sequence[str | int]) -> list[dict[str, Any]]:
    """Build four deterministic round-robin OOF folds from sorted IDs.

    Each returned mapping has ``fold``, ``train_ids`` (12 IDs) and
    ``heldout_ids`` (4 IDs) for the standard 16-image outer-training set.
    """

    ids = sorted({str(image_id).zfill(2) for image_id in trainids}, key=int)
    if len(ids) != len(trainids):
        raise ValueError("trainids must be unique")
    if len(ids) != 16:
        raise ValueError(f"Expected 16 outer-training IDs for four-fold OOF, got {len(ids)}")
    if set(ids) & set(OUTER_VAL_IDS):
        raise ValueError("Frozen outer-validation IDs cannot appear in inner OOF folds")
    folds: list[dict[str, Any]] = []
    for fold_index in range(4):
        heldout = ids[fold_index::4]
        fit = [image_id for image_id in ids if image_id not in set(heldout)]
        folds.append({"fold": fold_index, "train_ids": fit, "heldout_ids": heldout})
    return folds


def sha256(path: str | os.PathLike[str]) -> str:
    """Return a lowercase SHA-256 digest for a file, streaming in 1 MiB blocks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_chw(array: np.ndarray, channels: int, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 2 and channels == 1:
        value = value[None]
    if value.ndim == 3 and value.shape[0] != channels and value.shape[-1] == channels:
        value = value.transpose(2, 0, 1)
    if value.ndim != 3 or value.shape[0] != channels:
        raise ValueError(f"{name} must have shape [{channels},H,W], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(value, dtype=np.float32)


def save_prepared_npz(
    path: str | os.PathLike[str],
    image_id: str | int,
    rgb: np.ndarray,
    target: np.ndarray,
    fov: np.ndarray,
    coarse: np.ndarray,
) -> Path:
    """Atomically save the common train/validation NPZ array format."""

    rgb_chw = _as_chw(rgb, 3, "rgb")
    target_chw = _as_chw(target, 1, "target")
    fov_chw = _as_chw(fov, 1, "fov")
    coarse_chw = _as_chw(coarse, 1, "coarse")
    expected_spatial = rgb_chw.shape[-2:]
    for name, value in (("target", target_chw), ("fov", fov_chw), ("coarse", coarse_chw)):
        if value.shape[-2:] != expected_spatial:
            raise ValueError(f"{name} size mismatch: {value.shape[-2:]} vs {expected_spatial}")
    if np.any((rgb_chw < 0) | (rgb_chw > 1)):
        raise ValueError("rgb must be scaled to [0,1]")
    if np.any((target_chw < 0) | (target_chw > 1)) or np.any((fov_chw < 0) | (fov_chw > 1)):
        raise ValueError("target and fov must be scaled to [0,1]")
    if np.any((coarse_chw < 0) | (coarse_chw > 1)):
        raise ValueError("coarse must contain probabilities in [0,1]")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False) as tmp:
        temporary = Path(tmp.name)
        np.savez_compressed(
            tmp,
            id=np.asarray(str(image_id)),
            rgb=rgb_chw,
            target=target_chw,
            fov=fov_chw,
            coarse=coarse_chw,
        )
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(temporary, output)
    return output


def build_manifest(
    dataset: DriveDataset,
    prepared_files: Mapping[str, str | os.PathLike[str]] | None = None,
    checkpoints: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a compact provenance manifest without exposing absolute paths.

    ``checkpoints`` accepts a mapping from role to path (or to a mapping with
    ``path`` and optional ``provenance``), or a sequence of those mappings.
    """

    train_ids, val_ids = split_ids()
    folds = inner_folds(train_ids)

    def relative_name(path: Path) -> str:
        try:
            return path.resolve().relative_to(dataset.root.resolve()).as_posix()
        except ValueError:
            return path.name

    source_files: list[dict[str, str]] = []
    for record in dataset.train:
        for role, file_path in (("image", record.image_path), ("label", record.target_path), ("fov", record.fov_path)):
            assert file_path is not None
            source_files.append({"id": record.image_id, "split": "train", "role": role,
                                 "file": relative_name(file_path), "sha256": sha256(file_path)})
    for record in dataset.test:
        source_files.append({"id": record.image_id, "split": "test", "role": "image",
                             "file": relative_name(record.image_path), "sha256": sha256(record.image_path)})
        if record.fov_path is not None:
            source_files.append({"id": record.image_id, "split": "test", "role": "fov",
                                 "file": relative_name(record.fov_path), "sha256": sha256(record.fov_path)})

    prepared: list[dict[str, str]] = []
    for image_id, path_value in sorted((prepared_files or {}).items(), key=lambda item: int(item[0])):
        file_path = Path(path_value)
        prepared.append({"id": str(image_id).zfill(2), "file": file_path.name, "sha256": sha256(file_path)})

    checkpoint_items: list[tuple[str, Any]]
    if isinstance(checkpoints, Mapping):
        checkpoint_items = list(checkpoints.items())
    else:
        checkpoint_items = [(str(item.get("role", f"checkpoint_{index}")), item)
                            for index, item in enumerate(checkpoints or [])]
    checkpoint_entries: list[dict[str, Any]] = []
    for role, item in checkpoint_items:
        if isinstance(item, Mapping):
            file_value = item.get("path")
            item_provenance = item.get("provenance", {})
        else:
            file_value = item
            item_provenance = {}
        if file_value is None:
            raise ValueError(f"Missing checkpoint path for role {role}")
        file_path = Path(file_value)
        checkpoint_entries.append({
            "role": role,
            "file": file_path.name,
            "sha256": sha256(file_path),
            "provenance": dict(item_provenance) if isinstance(item_provenance, Mapping) else item_provenance,
        })

    return {
        "format_version": 1,
        "dataset": "DRIVE",
        "image_ids": {
            "official_training": list(OFFICIAL_TRAIN_IDS),
            "outer_train": train_ids,
            "outer_validation": val_ids,
            "test_images": [record.image_id for record in dataset.test],
        },
        "inner_oof_folds": folds,
        "source_files": source_files,
        "prepared_files": prepared,
        "model_checkpoints": checkpoint_entries,
        "provenance": dict(provenance or {}),
        "excluded_test_labels": {
            "loaded": False,
            "accessed": False,
            "explicit_test_image_ids": [record.image_id for record in dataset.test],
            "count": len(dataset.test),
            "policy": "test manual annotations are not discovered, opened, or hashed",
        },
    }


def write_manifest(path: str | os.PathLike[str], manifest: Mapping[str, Any]) -> Path:
    """Atomically write a manifest returned by :func:`build_manifest`."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=output.parent,
        prefix=output.name + ".", suffix=".tmp", delete=False,
    ) as tmp:
        temporary = Path(tmp.name)
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(temporary, output)
    return output

