"""Official FIVES training-data discovery and deterministic split utilities.

Only the 600 official training images and their ground-truth masks are
discovered. The official test tree is not scanned. Images are downsampled to
512 x 512 for development; this module does not support 2048-resolution claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


CATEGORY_ORDER = ("A", "D", "G", "N")
IMAGES_PER_CATEGORY = 150
OFFICIAL_TRAIN_COUNT = len(CATEGORY_ORDER) * IMAGES_PER_CATEGORY
IMAGE_SIZE = (512, 512)
FIGSHARE_ARTICLE_ID = 19688169
FIGSHARE_FILE_ID = 34969398
FIGSHARE_FILE_SIZE = 1764974308
FIGSHARE_FILE_MD5 = "789c80dd5376a82063e27fa49192bac9"

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_TRAIN_DIR_NAMES = {"train", "training"}
_LABEL_DIR_NAMES = {
    "groundtruth", "groundtruths", "gt", "mask", "masks", "label", "labels",
    "vessel", "vessels", "annotation", "annotations", "manual", "manualannotation",
}


@dataclass(frozen=True)
class FivesRecord:
    """One official training image and its paired vessel annotation.

    ``image_id`` is the source filename stem, for example ``1_A``. There is no
    official FOV annotation in this data interface, so ``fov_path`` is always
    ``None`` and :func:`load_image` returns an all-ones FOV.
    """

    image_id: str
    image_path: Path
    target_path: Path
    split: str = "train"
    category: str = ""
    fov_path: Path | None = None

    @property
    def id(self) -> str:
        return self.image_id


@dataclass(frozen=True)
class FivesDataset:
    """FIVES records rooted at a source directory; iteration is train-only."""

    root: Path
    train: tuple[FivesRecord, ...]

    @property
    def records(self) -> tuple[FivesRecord, ...]:
        return self.train

    def __iter__(self):
        return iter(self.train)


def category_from_id(image_id: str | int) -> str:
    """Return the disease category encoded in an original FIVES image stem."""

    match = re.fullmatch(r"([0-9]+)_([ADGN])", str(image_id), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid FIVES image ID {image_id!r}; expected a stem such as '1_A'")
    return match.group(2).upper()


def id_sort_key(image_id: str | int) -> tuple[int, int, str]:
    """Stable category/numeric ordering for IDs such as ``1_A`` and ``12_N``."""

    value = str(image_id)
    match = re.fullmatch(r"([0-9]+)_([ADGN])", value, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid FIVES image ID {image_id!r}; expected a stem such as '1_A'")
    category = match.group(2).upper()
    return CATEGORY_ORDER.index(category), int(match.group(1)), value


def _normalized_dir_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _has_label_ancestor(path: Path, root: Path) -> bool:
    for part in path.parent.relative_to(root).parts:
        normalized = _normalized_dir_name(part)
        if normalized in _LABEL_DIR_NAMES or normalized.startswith("groundtruth"):
            return True
    return False


def _is_test_dir(name: str) -> bool:
    normalized = _normalized_dir_name(name)
    return normalized in {"test", "testing", "testdata", "testset", "testingdata"}


def _training_roots(root: Path) -> list[Path]:
    if root.name.lower() in _TRAIN_DIR_NAMES:
        return [root]
    found: list[Path] = []
    for current, dirs, _ in os.walk(root, followlinks=False):
        current_path = Path(current)
        dirs[:] = [
            name for name in dirs
            if not _is_test_dir(name) and not (current_path / name).is_symlink()
        ]
        if current_path != root and current_path.name.lower() in _TRAIN_DIR_NAMES:
            found.append(current_path)
            # A nested train folder is handled by the shallowest discovered root.
            dirs[:] = []
    if not found:
        return []
    found.sort(key=lambda path: (len(path.parts), str(path).casefold()))
    selected: list[Path] = []
    for path in found:
        if not any(path == parent or parent in path.parents for parent in selected):
            selected.append(path)
    return selected


def _files_under_training(root: Path) -> tuple[list[Path], list[Path]]:
    images: list[Path] = []
    labels: list[Path] = []
    for current, dirs, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        dirs[:] = [
            name for name in dirs
            if not _is_test_dir(name) and not (current_path / name).is_symlink()
        ]
        for filename in filenames:
            path = current_path / filename
            if path.suffix.lower() not in _IMAGE_EXTENSIONS or path.is_symlink():
                continue
            if _has_label_ancestor(path, root):
                labels.append(path)
            else:
                images.append(path)
    return images, labels


def _id_index(paths: Iterable[Path], role: str, allowed_ids: set[str]) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        image_id = path.stem
        if image_id not in allowed_ids:
            continue
        if image_id in indexed:
            raise ValueError(f"Duplicate FIVES {role} for {image_id}: {indexed[image_id]} and {path}")
        indexed[image_id] = path
    return indexed


def _validate_official_training_ids(ids: Sequence[str]) -> list[str]:
    if len(ids) != OFFICIAL_TRAIN_COUNT or len(set(ids)) != OFFICIAL_TRAIN_COUNT:
        raise ValueError(
            f"Expected {OFFICIAL_TRAIN_COUNT} unique official FIVES training IDs, got {len(ids)}"
        )
    counts = {category: 0 for category in CATEGORY_ORDER}
    for image_id in ids:
        counts[category_from_id(image_id)] += 1
    if any(counts[category] != IMAGES_PER_CATEGORY for category in CATEGORY_ORDER):
        raise ValueError(
            "Expected 150 official training images per FIVES category "
            f"A/D/G/N, got {counts}"
        )
    return sorted(ids, key=id_sort_key)


def discover_fives(root: str | os.PathLike[str]) -> FivesDataset:
    """Find all 600 official train image/mask pairs under an extracted source.

    Discovery only walks directories named ``train`` or ``training`` and
    prunes nested test directories. Image files are identified by their
    original stem. Targets must occur under a known annotation directory such
    as ``Ground truth``, ``GroundTruth``, ``mask`` or ``labels``; the function
    fails when a mask cannot be located instead of guessing from image content.
    """

    candidate = Path(root).expanduser()
    if not candidate.is_dir():
        raise FileNotFoundError(f"FIVES root is not a directory: {candidate}")
    source_root = candidate.resolve()
    roots = _training_roots(source_root)
    if not roots:
        raise FileNotFoundError(f"Could not find a train/training directory under: {source_root}")

    images: list[Path] = []
    labels: list[Path] = []
    for train_root in roots:
        train_images, train_labels = _files_under_training(train_root)
        images.extend(train_images)
        labels.extend(train_labels)

    image_ids: set[str] = set()
    label_ids: set[str] = set()
    for path in images:
        try:
            category_from_id(path.stem)
        except ValueError:
            continue
        image_ids.add(path.stem)
    for path in labels:
        try:
            category_from_id(path.stem)
        except ValueError:
            continue
        label_ids.add(path.stem)
    image_index = _id_index(images, "image", image_ids)
    label_index = _id_index(labels, "ground-truth mask", label_ids)
    try:
        official_ids = _validate_official_training_ids(sorted(image_index, key=id_sort_key))
    except ValueError as exc:
        raise ValueError(f"FIVES training image set mismatch: {exc}") from exc
    missing_labels = sorted(set(official_ids) - set(label_index), key=id_sort_key)
    extra_labels = sorted(set(label_index) - set(official_ids), key=id_sort_key)
    if missing_labels or extra_labels:
        details = []
        if missing_labels:
            details.append(
                "missing masks in recognized ground-truth directories "
                f"({len(missing_labels)}): {missing_labels[:12]}"
            )
        if extra_labels:
            details.append(f"unpaired training masks ({len(extra_labels)}): {extra_labels[:12]}")
        raise FileNotFoundError("FIVES training set mismatch (" + "; ".join(details) + ")")

    records = tuple(
        FivesRecord(
            image_id=image_id,
            image_path=image_index[image_id].resolve(),
            target_path=label_index[image_id].resolve(),
            split="train",
            category=category_from_id(image_id),
        )
        for image_id in official_ids
    )
    return FivesDataset(source_root, records)


def _record_value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def load_image(record: FivesRecord | Mapping[str, Any]) -> dict[str, Any]:
    """Load one official training image at development resolution 512 x 512.

    RGB uses Lanczos interpolation and returns float32 ``[3,512,512]`` in
    ``[0,1]``. The mask uses nearest-neighbor interpolation and is thresholded
    to binary float32 ``[1,512,512]``. FIVES provides no official FOV mask, so
    ``fov`` is all ones and is never derived from the ground-truth mask.
    The 512-pixel input is a development setting, not a 2048-resolution result.
    """

    image_id = str(_record_value(record, "image_id", _record_value(record, "id")))
    if category_from_id(image_id) not in CATEGORY_ORDER:
        raise ValueError(f"Invalid FIVES training image ID: {image_id}")
    if str(_record_value(record, "split", "train")) != "train":
        raise ValueError("Only official FIVES training records can be loaded by this module")
    image_path = Path(_record_value(record, "image_path"))
    target_value = _record_value(record, "target_path")
    if target_value is None:
        raise ValueError(f"Training record {image_id} has no official ground-truth mask")
    target_path = Path(target_value)

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        with Image.open(target_path) as target_source:
            target_image = target_source.convert("L")
        if image.size != target_image.size:
            raise ValueError(
                f"Image/mask size mismatch for {image_id}: {image.size} vs {target_image.size}"
            )
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize(IMAGE_SIZE, resampling.LANCZOS)
        target_image = target_image.resize(IMAGE_SIZE, resampling.NEAREST)
        rgb_hwc = np.asarray(image, dtype=np.uint8)
        target_hw = np.asarray(target_image, dtype=np.uint8)

    rgb = np.ascontiguousarray(rgb_hwc.transpose(2, 0, 1), dtype=np.float32) / np.float32(255.0)
    target = (target_hw[None] > 0).astype(np.float32)
    fov = np.ones((1, IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.float32)
    return {"id": image_id, "image_id": image_id, "rgb": rgb, "target": target, "fov": fov}


def _ids_from_records_or_ids(records_or_dataset: Any) -> list[str]:
    if hasattr(records_or_dataset, "train"):
        source = list(records_or_dataset.train)
    elif isinstance(records_or_dataset, Mapping):
        if "train" in records_or_dataset and isinstance(records_or_dataset["train"], Sequence):
            source = list(records_or_dataset["train"])
        else:
            source = [records_or_dataset]
    elif isinstance(records_or_dataset, (str, bytes)):
        raise TypeError("Expected a dataset, record collection, or sequence of IDs")
    else:
        source = list(records_or_dataset)

    ids: list[str] = []
    for item in source:
        if isinstance(item, (str, int)):
            image_id = str(item)
        else:
            image_id = str(_record_value(item, "image_id", _record_value(item, "id")))
        # Validate format here so malformed IDs cannot enter saved splits.
        category_from_id(image_id)
        ids.append(image_id)
    if len(set(ids)) != len(ids):
        raise ValueError("FIVES IDs must be unique")

    return _validate_official_training_ids(ids)


def split_ids(records_or_dataset: Any) -> tuple[list[str], list[str]]:
    """Return stratified outer-train and outer-validation IDs for FIVES.

    Within each of A/D/G/N, IDs are sorted by their numeric source stem and
    positions ``[4::5]`` form validation (30/category, 120 total). The other
    120/category form outer training (480 total). The split unit is an image;
    no patient-level mapping is provided here.
    """

    ids = _ids_from_records_or_ids(records_or_dataset)
    by_category = {
        category: sorted((image_id for image_id in ids if category_from_id(image_id) == category),
                         key=id_sort_key)
        for category in CATEGORY_ORDER
    }
    validation = [image_id for category in CATEGORY_ORDER for image_id in by_category[category][4::5]]
    validation_set = set(validation)
    outer_train = [image_id for image_id in ids if image_id not in validation_set]
    return outer_train, validation


def inner_folds(trainids: Sequence[str | int]) -> list[dict[str, Any]]:
    """Create four deterministic category-stratified OOF folds.

    Every fold holds out 30 images per category (120 total) and trains on the
    remaining 360 outer-training images. Fold assignment is round-robin within
    category after sorting by numeric source stem.
    """

    ids = [str(image_id) for image_id in trainids]
    if len(ids) != len(set(ids)):
        raise ValueError("trainids must be unique")
    if any(category_from_id(image_id) not in CATEGORY_ORDER for image_id in ids):
        raise ValueError("Invalid FIVES training ID")
    if len(ids) != 480:
        raise ValueError(f"inner_folds requires exactly 480 outer-training IDs, got {len(ids)}")
    category_counts = {category: 0 for category in CATEGORY_ORDER}
    for image_id in ids:
        category_counts[category_from_id(image_id)] += 1
    if any(category_counts[category] != 120 for category in CATEGORY_ORDER):
        raise ValueError(
            "inner_folds requires 120 outer-training IDs per category "
            f"A/D/G/N, got {category_counts}"
        )

    folds: list[dict[str, Any]] = []
    for fold_index in range(4):
        heldout: list[str] = []
        for category in CATEGORY_ORDER:
            category_ids = sorted(
                (image_id for image_id in ids if category_from_id(image_id) == category),
                key=id_sort_key,
            )
            heldout.extend(category_ids[fold_index::4])
        heldout = sorted(heldout, key=id_sort_key)
        heldout_set = set(heldout)
        fit = sorted((image_id for image_id in ids if image_id not in heldout_set), key=id_sort_key)
        folds.append({"fold": fold_index, "train_ids": fit, "heldout_ids": heldout})
    return folds


def sha256(path: str | os.PathLike[str]) -> str:
    """Return a lowercase SHA-256 digest for a file in 1 MiB blocks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    payload = "\n".join(ids).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


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
    dataset: FivesDataset,
    prepared_files: Mapping[str, str | os.PathLike[str]] | None = None,
    checkpoints: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create source, split, prepared-array and checkpoint provenance."""

    official_ids = _ids_from_records_or_ids(dataset)
    train_ids, val_ids = split_ids(official_ids)
    folds = inner_folds(train_ids)

    def relative_name(path: Path) -> str:
        try:
            return path.resolve().relative_to(dataset.root.resolve()).as_posix()
        except ValueError:
            return path.name

    source_files: list[dict[str, str]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    records = {record.image_id: record for record in dataset.train}
    for record in dataset.train:
        image_digest = sha256(record.image_path)
        label_digest = sha256(record.target_path)
        source_hashes[record.image_id] = {"image": image_digest, "label": label_digest}
        source_files.extend((
            {"id": record.image_id, "split": "train", "role": "image",
             "file": relative_name(record.image_path), "sha256": image_digest},
            {"id": record.image_id, "split": "train", "role": "label",
             "file": relative_name(record.target_path), "sha256": label_digest},
        ))

    prepared: list[dict[str, str]] = []
    for image_id, path_value in sorted((prepared_files or {}).items(), key=lambda item: id_sort_key(str(item[0]))):
        file_path = Path(path_value)
        prepared.append({"id": str(image_id), "file": file_path.name, "sha256": sha256(file_path)})

    if isinstance(checkpoints, Mapping):
        checkpoint_items = list(checkpoints.items())
    else:
        checkpoint_items = [
            (str(item.get("role", f"checkpoint_{index}")), item)
            for index, item in enumerate(checkpoints or [])
        ]
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
        if not isinstance(item_provenance, Mapping):
            item_provenance = {"details": item_provenance}
        training_ids = [str(value) for value in item_provenance.get("train_ids", [])]
        unknown_ids = sorted(set(training_ids) - set(records), key=id_sort_key)
        if unknown_ids:
            raise ValueError(f"Checkpoint {role} names unknown FIVES training IDs: {unknown_ids}")
        checkpoint_entries.append({
            "role": role,
            "file": file_path.name,
            "sha256": sha256(file_path),
            "source": {
                "dataset": "FIVES",
                "provider": "Figshare",
                "article_id": FIGSHARE_ARTICLE_ID,
                "file_id": FIGSHARE_FILE_ID,
            },
            "train_ids": sorted(set(training_ids), key=id_sort_key),
            "image_hashes": {image_id: source_hashes[image_id]["image"]
                             for image_id in sorted(set(training_ids), key=id_sort_key)},
            "label_hashes": {image_id: source_hashes[image_id]["label"]
                             for image_id in sorted(set(training_ids), key=id_sort_key)},
            "provenance": dict(item_provenance),
        })

    fold_hashes = [
        {
            "fold": fold["fold"],
            "train_ids_sha256": _ids_sha256(fold["train_ids"]),
            "heldout_ids_sha256": _ids_sha256(fold["heldout_ids"]),
        }
        for fold in folds
    ]
    return {
        "format_version": 1,
        "dataset": "FIVES",
        "source": {
            "provider": "Figshare",
            "article_id": FIGSHARE_ARTICLE_ID,
            "file_id": FIGSHARE_FILE_ID,
            "archive_size_bytes": FIGSHARE_FILE_SIZE,
            "archive_md5": FIGSHARE_FILE_MD5,
        },
        "resolution_policy": {
            "model_input_hw": list(IMAGE_SIZE[::-1]),
            "status": "512x512 development resolution",
            "claim_boundary": "No 2048-resolution conclusion is supported by these inputs.",
            "rgb_resize": "Lanczos",
            "mask_resize": "nearest-neighbor",
        },
        "split_unit": {
            "unit": "image",
            "patient_disjoint": False,
            "limitation": "No patient-to-image mapping is available in this data module.",
        },
        "image_ids": {
            "official_training": official_ids,
            "outer_train": train_ids,
            "outer_validation": val_ids,
        },
        "split_hashes": {
            "algorithm": "sha256 of UTF-8 IDs joined by LF with a final LF, in saved order",
            "official_training": _ids_sha256(official_ids),
            "outer_train": _ids_sha256(train_ids),
            "outer_validation": _ids_sha256(val_ids),
            "inner_oof_folds": fold_hashes,
        },
        "inner_oof_folds": folds,
        "source_files": source_files,
        "prepared_files": prepared,
        "model_checkpoints": checkpoint_entries,
        "provenance": dict(provenance or {}),
        "excluded_test_labels": {
            "loaded": False,
            "accessed": False,
            "policy": "The test tree and any test labels are not discovered, opened, or hashed.",
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
