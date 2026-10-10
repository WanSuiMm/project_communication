"""Acquire and safely extract the official FIVES Figshare archive.

The default source is Figshare article 19688169, file 34969398. The archive
is resumed through HTTP Range requests and accepted only after exact size and
MD5 checks. Extraction prefers libarchive-c and writes each member manually
into a fresh staging directory after validating its path and file type.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

try:
    from .data import FIGSHARE_ARTICLE_ID, FIGSHARE_FILE_ID, FIGSHARE_FILE_MD5, FIGSHARE_FILE_SIZE
except ImportError:  # supports direct script-style imports from this directory
    from data import FIGSHARE_ARTICLE_ID, FIGSHARE_FILE_ID, FIGSHARE_FILE_MD5, FIGSHARE_FILE_SIZE


FIGSHARE_FILES_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/files"
FIGSHARE_ARTICLE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"
ARCHIVE_NAME = f"FIVES_figshare_{FIGSHARE_ARTICLE_ID}_{FIGSHARE_FILE_ID}.rar"
EXTRACTED_NAME = f"FIVES_figshare_{FIGSHARE_ARTICLE_ID}"
ACQUISITION_MARKER = ".fives_acquisition.json"
USER_AGENT = "ReactionTransport-FIVES-acquire/1.0"
BLOCK_SIZE = 8 * 1024 * 1024


class UnsafeArchiveError(ValueError):
    """Raised when an archive member could escape the extraction root."""


def _digest_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_official_metadata(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("files")
    if not isinstance(payload, list):
        raise RuntimeError("response did not contain a Figshare files list")
    matches = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        try:
            entry_id = int(entry.get("id", -1))
        except (TypeError, ValueError):
            continue
        if entry_id == FIGSHARE_FILE_ID:
            matches.append(entry)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Figshare file {FIGSHARE_FILE_ID}, found {len(matches)}")
    metadata = matches[0]
    try:
        actual_size = int(metadata.get("size", -1))
    except (TypeError, ValueError):
        actual_size = -1
    actual_md5 = str(metadata.get("computed_md5", metadata.get("md5", ""))).lower()
    if actual_size != FIGSHARE_FILE_SIZE or actual_md5 != FIGSHARE_FILE_MD5:
        raise RuntimeError(
            "Official Figshare metadata does not match the pinned archive "
            f"(size={actual_size}, md5={actual_md5})"
        )
    download_url = metadata.get("download_url")
    try:
        parsed_url = urlsplit(download_url) if isinstance(download_url, str) else None
    except ValueError as exc:
        raise RuntimeError(f"Figshare supplied an invalid download URL: {exc}") from exc
    hostname = (parsed_url.hostname or "").lower() if parsed_url else ""
    if (
        parsed_url is None
        or parsed_url.scheme != "https"
        or not (hostname == "figshare.com" or hostname.endswith(".figshare.com"))
    ):
        raise RuntimeError("Figshare did not provide an HTTPS download URL on an official Figshare host")
    return metadata


def _official_metadata(timeout: float = 30.0) -> dict[str, Any]:
    failures = []
    for endpoint in (FIGSHARE_FILES_API, FIGSHARE_ARTICLE_API):
        try:
            return _read_official_metadata(endpoint, timeout)
        except RuntimeError as exc:
            failures.append(f"{endpoint}: {exc}")
    raise RuntimeError(
        "Could not validate the pinned file from official Figshare metadata endpoints: "
        + "; ".join(failures)
    )


def download_official(archive_dir: str | os.PathLike[str], timeout: float = 60.0) -> Path:
    """Resume and verify the official RAR download, then return its path.

    Interrupted downloads remain as ``.part`` and continue from their current
    byte offset on the next call. Existing complete files are reused only when
    their exact size and MD5 match the pinned Figshare artifact.
    """

    directory = Path(archive_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / ARCHIVE_NAME
    partial = destination.with_name(destination.name + ".part")

    if destination.exists():
        if destination.stat().st_size == FIGSHARE_FILE_SIZE and _digest_md5(destination) == FIGSHARE_FILE_MD5:
            return destination
        raise ValueError(f"Existing archive failed pinned size/MD5 validation: {destination}")

    if partial.exists() and partial.stat().st_size == FIGSHARE_FILE_SIZE:
        if _digest_md5(partial) == FIGSHARE_FILE_MD5:
            os.replace(partial, destination)
            return destination
        partial.unlink()

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > FIGSHARE_FILE_SIZE:
        with partial.open("wb"):
            pass
        offset = 0

    # Reuse a verified local cache without requiring network metadata. This
    # supports moving the pinned Figshare archive to an offline server.
    metadata = _official_metadata(timeout=min(timeout, 30.0))
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(str(metadata["download_url"]), headers=headers)
    try:
        response = urlopen(request, timeout=timeout)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Official Figshare download request failed: {exc}") from exc

    with response:
        status = getattr(response, "status", response.getcode())
        append = offset > 0 and status == 206
        if append:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range.strip())
            if match is None or int(match.group(1)) != offset:
                raise RuntimeError(f"Server returned invalid Content-Range for resume: {content_range!r}")
        elif status == 200:
            offset = 0
        else:
            raise RuntimeError(f"Unexpected HTTP status for Figshare download: {status}")

        mode = "ab" if append else "wb"
        written = offset
        with partial.open(mode) as output:
            while True:
                block = response.read(BLOCK_SIZE)
                if not block:
                    break
                written += len(block)
                if written > FIGSHARE_FILE_SIZE:
                    raise RuntimeError("Figshare response exceeded the pinned archive size")
                output.write(block)
            output.flush()
            os.fsync(output.fileno())

    actual_size = partial.stat().st_size
    if actual_size != FIGSHARE_FILE_SIZE:
        raise RuntimeError(
            f"Download is incomplete: {actual_size} bytes, expected {FIGSHARE_FILE_SIZE}; rerun to resume"
        )
    actual_md5 = _digest_md5(partial)
    if actual_md5 != FIGSHARE_FILE_MD5:
        partial.unlink()
        raise RuntimeError(f"Downloaded Figshare archive MD5 mismatch: {actual_md5}")
    os.replace(partial, destination)
    return destination


def probe_extractors() -> dict[str, Any]:
    """Report available safe-extraction backends and common command-line tools."""

    try:
        libarchive_available = importlib.util.find_spec("libarchive") is not None
    except (ImportError, ValueError):
        libarchive_available = False
    seven_zip = next((shutil.which(name) for name in ("7z", "7za", "7zz") if shutil.which(name)), None)
    unrar = shutil.which("unrar")
    return {
        "libarchive_c": libarchive_available,
        "7z": seven_zip,
        "unrar": unrar,
        "selected_preference": "libarchive-c, then 7-Zip",
        "required_python_distribution_if_needed": "libarchive-c",
    }


def _safe_member_path(staging: Path, raw_name: str, *, allow_trailing_slash: bool = False) -> Path:
    if not raw_name or "\x00" in raw_name:
        raise UnsafeArchiveError(f"Invalid empty or NUL archive member path: {raw_name!r}")
    normalized = raw_name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise UnsafeArchiveError(f"Absolute archive member path is forbidden: {raw_name!r}")
    # Archive directory entries commonly end with one separator. Permit that
    # only when the archive metadata identifies a directory; files stay strict.
    if normalized.endswith("/"):
        if not allow_trailing_slash:
            raise UnsafeArchiveError(f"Trailing separator on non-directory member: {raw_name!r}")
        normalized = normalized.rstrip("/")
    if not normalized:
        raise UnsafeArchiveError(f"Invalid empty archive member path: {raw_name!r}")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise UnsafeArchiveError(f"Traversal or ambiguous archive member path is forbidden: {raw_name!r}")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    for part in parts:
        if ":" in part or part.endswith((".", " ")):
            raise UnsafeArchiveError(f"Non-portable archive member path is forbidden: {raw_name!r}")
        stem = part.split(".", 1)[0].upper()
        if stem in reserved:
            raise UnsafeArchiveError(f"Reserved device name in archive member path: {raw_name!r}")
    destination = staging.joinpath(*parts)
    staging_real = staging.resolve()
    destination_real = destination.resolve(strict=False)
    try:
        destination_real.relative_to(staging_real)
    except ValueError as exc:
        raise UnsafeArchiveError(f"Archive member escapes target directory: {raw_name!r}") from exc
    return destination


def _ensure_directory(directory: Path, staging: Path) -> None:
    staging_real = staging.resolve()
    relative = directory.relative_to(staging)
    cursor = staging
    for part in relative.parts:
        cursor = cursor / part
        if cursor.exists() or cursor.is_symlink():
            if cursor.is_symlink() or not cursor.is_dir():
                raise UnsafeArchiveError(f"Archive path collides with a non-directory: {cursor}")
        else:
            cursor.mkdir()
        try:
            cursor.resolve().relative_to(staging_real)
        except ValueError as exc:
            raise UnsafeArchiveError(f"Created path escaped target directory: {cursor}") from exc


def _verify_extracted_tree(staging: Path) -> None:
    staging_real = staging.resolve()
    for current, dirs, files in os.walk(staging, followlinks=False):
        current_path = Path(current)
        for name in dirs + files:
            path = current_path / name
            if path.is_symlink():
                raise UnsafeArchiveError(f"Archive extraction produced a symbolic link: {path.name}")
            try:
                path.resolve().relative_to(staging_real)
            except ValueError as exc:
                raise UnsafeArchiveError(f"Extracted member escaped target directory: {path.name}") from exc


def _valid_completion_marker(target: Path) -> bool:
    final = target / EXTRACTED_NAME
    marker = target / ACQUISITION_MARKER
    if final.is_symlink() or not final.is_dir() or marker.is_symlink() or not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("completed") is True
        and payload.get("provider") == "Figshare"
        and payload.get("article_id") == FIGSHARE_ARTICLE_ID
        and payload.get("file_id") == FIGSHARE_FILE_ID
        and payload.get("archive_size_bytes") == FIGSHARE_FILE_SIZE
        and payload.get("archive_md5") == FIGSHARE_FILE_MD5
        and payload.get("extracted_dir") == EXTRACTED_NAME
    )


def _write_completion_marker(target: Path, extractor: str) -> Path:
    marker = target / ACQUISITION_MARKER
    payload = {
        "format_version": 1,
        "completed": True,
        "provider": "Figshare",
        "article_id": FIGSHARE_ARTICLE_ID,
        "file_id": FIGSHARE_FILE_ID,
        "archive_size_bytes": FIGSHARE_FILE_SIZE,
        "archive_md5": FIGSHARE_FILE_MD5,
        "extracted_dir": EXTRACTED_NAME,
        "extractor": extractor,
    }
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=target,
        prefix=marker.name + ".", suffix=".tmp", delete=False,
    ) as tmp:
        temporary = Path(tmp.name)
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(temporary, marker)
    return marker


def _libarchive_extract(archive_path: Path, staging: Path) -> None:
    import libarchive  # type: ignore[import-not-found]

    seen: set[str] = set()
    with libarchive.file_reader(str(archive_path)) as entries:
        for entry in entries:
            raw_name = os.fsdecode(entry.pathname)
            is_directory = bool(getattr(entry, "isdir", False))
            is_regular = bool(getattr(entry, "isfile", False))
            destination = _safe_member_path(staging, raw_name, allow_trailing_slash=is_directory)
            normalized = destination.relative_to(staging).as_posix().casefold()
            if normalized in seen:
                raise UnsafeArchiveError(f"Duplicate archive member path: {raw_name!r}")
            seen.add(normalized)

            if bool(getattr(entry, "issym", False)) or bool(getattr(entry, "islnk", False)):
                raise UnsafeArchiveError(f"Link entry is forbidden in FIVES archive: {raw_name!r}")
            if not is_directory and not is_regular:
                raise UnsafeArchiveError(f"Non-file archive member is forbidden: {raw_name!r}")

            if is_directory:
                _ensure_directory(destination, staging)
                continue

            _ensure_directory(destination.parent, staging)
            if destination.exists() or destination.is_symlink():
                raise UnsafeArchiveError(f"Duplicate archive output path: {raw_name!r}")
            written = 0
            expected = int(getattr(entry, "size", 0) or 0)
            with destination.open("xb") as output:
                for block in entry.get_blocks():
                    written += len(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if written != expected:
                raise RuntimeError(
                    f"Archive member size mismatch for {raw_name!r}: {written} vs {expected}"
                )


def _seven_zip_members(archive_path: Path, executable: str) -> list[tuple[str, dict[str, str]]]:
    command = [executable, "l", "-slt", "-sccUTF-8", "-bd", str(archive_path)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        raise RuntimeError(f"7-Zip could not list archive members: {result.stderr[-1000:]}")
    entries: list[tuple[str, dict[str, str]]] = []
    fields: dict[str, str] = {}

    def finish() -> None:
        nonlocal fields
        if fields:
            raw_path = fields.get("Path")
            if raw_path:
                try:
                    same_as_archive = Path(raw_path).resolve() == archive_path.resolve()
                except OSError:
                    same_as_archive = False
                if not same_as_archive:
                    entries.append((raw_path, fields))
        fields = {}

    for line in result.stdout.splitlines():
        if line.strip() == "----------":
            finish()
            continue
        if not line.strip():
            finish()
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            fields[key.strip()] = value.strip()
    finish()
    if not entries:
        raise RuntimeError("7-Zip listed no archive members")
    return entries


def _seven_zip_extract(archive_path: Path, staging: Path, executable: str) -> None:
    members = _seven_zip_members(archive_path, executable)
    seen: set[str] = set()
    for raw_name, fields in members:
        is_directory = fields.get("Folder", "").strip() == "+"
        destination = _safe_member_path(staging, raw_name, allow_trailing_slash=is_directory)
        normalized = destination.relative_to(staging).as_posix().casefold()
        if normalized in seen:
            raise UnsafeArchiveError(f"Duplicate archive member path: {raw_name!r}")
        seen.add(normalized)
        lowered = {key.lower(): value for key, value in fields.items()}
        if any("link" in key for key in lowered):
            raise UnsafeArchiveError(f"Link entry is forbidden in FIVES archive: {raw_name!r}")
        mode = lowered.get("unix mode", "")
        try:
            if mode and int(mode, 8) & 0o170000 == 0o120000:
                raise UnsafeArchiveError(f"Symbolic-link entry is forbidden: {raw_name!r}")
        except ValueError:
            raise UnsafeArchiveError(f"Unrecognized Unix mode in archive listing: {mode!r}")

    command = [executable, "x", "-y", "-aoa", f"-o{staging}", str(archive_path)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        raise RuntimeError(f"7-Zip extraction failed: {result.stderr[-1000:]}")


def safe_extract_rar(
    archive_path: str | os.PathLike[str],
    target_dir: str | os.PathLike[str],
) -> Path:
    """Safely extract the pinned RAR into a new child directory under target.

    Every archive path is checked before writing, links are rejected, member
    files are written into a new staging directory, and the completed tree is
    checked again before it is atomically moved into place. libarchive-c is the
    preferred backend; 7-Zip is used only when libarchive-c is unavailable.
    """

    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"FIVES archive not found: {archive}")
    target = Path(target_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    final = target / EXTRACTED_NAME
    if final.exists():
        if _valid_completion_marker(target):
            return final
        raise FileExistsError(f"Extraction target already exists; preserving it: {final}")

    backends = probe_extractors()
    libarchive_available = bool(backends["libarchive_c"])
    seven_zip = backends["7z"]
    if not libarchive_available and not seven_zip:
        unrar = backends["unrar"]
        suffix = f" unrar is present at {unrar} but its listing cannot certify link safety." if unrar else ""
        raise RuntimeError(
            "Safe RAR extraction requires the Python libarchive-c package or 7-Zip."
            f"{suffix} Install libarchive-c with `python -m pip install libarchive-c`."
        )

    staging = Path(tempfile.mkdtemp(prefix=".fives-extract-", dir=target))
    try:
        if libarchive_available:
            try:
                _libarchive_extract(archive, staging)
            except UnsafeArchiveError:
                raise
            except Exception:
                # libarchive-c can be installed without RAR support; a listed
                # 7-Zip backend is the only fallback after that format failure.
                if not seven_zip:
                    raise
                shutil.rmtree(staging)
                staging = Path(tempfile.mkdtemp(prefix=".fives-extract-", dir=target))
                _seven_zip_extract(archive, staging, str(seven_zip))
        else:
            _seven_zip_extract(archive, staging, str(seven_zip))
        _verify_extracted_tree(staging)
        if final.exists():
            raise FileExistsError(f"Extraction target appeared during extraction: {final}")
        os.replace(staging, final)
        return final
    except Exception:
        # staging was created by this call, and its resolved parent is target.
        if staging.exists() and staging.parent.resolve() == target:
            shutil.rmtree(staging)
        raise


def acquire_official(
    archive_dir: str | os.PathLike[str],
    target_dir: str | os.PathLike[str],
    timeout: float = 60.0,
) -> Path:
    """Download and safely extract the pinned official FIVES source."""

    target = Path(target_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    final = target / EXTRACTED_NAME
    marker = target / ACQUISITION_MARKER
    if final.exists():
        if _valid_completion_marker(target):
            return final
        raise FileExistsError(
            f"Existing extraction has no valid pinned-source completion marker; preserving it: {final}"
        )
    if marker.exists():
        raise RuntimeError(f"FIVES completion marker exists without its extraction directory: {marker}")

    archive = download_official(archive_dir, timeout=timeout)
    extracted = safe_extract_rar(archive, target)
    backends = probe_extractors()
    extractor = "libarchive-c" if backends["libarchive_c"] else "7-Zip"
    _write_completion_marker(target, extractor)
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", help="Directory for the verified RAR and resumable .part file")
    parser.add_argument("--target-dir", help="Parent directory for a new extracted FIVES source tree")
    parser.add_argument("--probe", action="store_true", help="Print available extraction backends and exit")
    parser.add_argument("--download-only", action="store_true", help="Download and verify the archive without extracting it")
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe_extractors(), indent=2))
        return
    if args.download_only:
        if not args.archive_dir:
            parser.error("--download-only requires --archive-dir")
        print(download_official(args.archive_dir))
        return
    if not args.archive_dir or not args.target_dir:
        parser.error("acquisition requires both --archive-dir and --target-dir")
    extracted = acquire_official(args.archive_dir, args.target_dir)
    print(extracted)


if __name__ == "__main__":
    main()
