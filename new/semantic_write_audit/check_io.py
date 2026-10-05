"""CPU-only checks for the semantic-write audit's atomic JSON writer."""

from __future__ import annotations

import ctypes
import json
import os
import threading
import time
import uuid
from ctypes import wintypes
from pathlib import Path
from typing import Any, Callable


def _open_read_write_without_delete_sharing(path: Path) -> int:
    """Open an existing file so readers and writers may share it, but rename may not."""
    if os.name != "nt":
        raise RuntimeError("the transient sharing check requires Windows")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE

    # FILE_SHARE_DELETE is deliberately absent: replacing the destination must
    # fail while this handle is open, while JSON readers remain allowed.
    handle = create_file(
        str(path),
        0x80000000,  # GENERIC_READ
        0x00000001 | 0x00000002,  # FILE_SHARE_READ | FILE_SHARE_WRITE
        None,
        3,  # OPEN_EXISTING
        0x00000080,  # FILE_ATTRIBUTE_NORMAL
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(handle)


def _close_windows_handle(handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    if not close_handle(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def check(
    safe_write: Callable[[Path, Any], None], out_directory: str | os.PathLike[str]
) -> dict[str, Any]:
    """Exercise JSON roundtripping and retrying a temporarily locked destination."""
    if os.name != "nt":
        raise RuntimeError("check_io.check is a Windows-only sharing test")

    directory = Path(out_directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f".safe_write_io_check_{uuid.uuid4().hex}.json"
    temp_target = target.with_suffix(target.suffix + ".tmp")

    before = {"stage": "before", "items": [1, 2, 3]}
    after = {"stage": "after", "items": [3, 2, 1]}
    writer_namespace = getattr(safe_write, "__globals__", {})
    retry_counter_available = "WRITE_RETRIES" in writer_namespace
    retries_before = writer_namespace.get("WRITE_RETRIES", 0)
    safe_write(target, before)
    if _read_json(target) != before:
        raise AssertionError("safe_write JSON roundtrip did not preserve the input")

    locked = threading.Event()
    writer_started = threading.Event()
    writer_done = threading.Event()
    holder_errors: list[BaseException] = []
    writer_errors: list[BaseException] = []

    def hold_destination() -> None:
        handle: int | None = None
        try:
            handle = _open_read_write_without_delete_sharing(target)
            locked.set()
            # Start the 100 ms hold only after the writer is about to run. This
            # gives the retry check a reliable window in which the old JSON
            # must remain readable.
            if not writer_started.wait(timeout=5.0):
                raise TimeoutError("safe_write did not start during the lock check")
            time.sleep(0.1)
        except BaseException as exc:
            holder_errors.append(exc)
            locked.set()
        finally:
            if handle is not None:
                try:
                    _close_windows_handle(handle)
                except BaseException as exc:
                    holder_errors.append(exc)

    def write_after() -> None:
        writer_started.set()
        try:
            safe_write(target, after)
        except BaseException as exc:
            writer_errors.append(exc)
        finally:
            writer_done.set()

    holder = threading.Thread(target=hold_destination, name="safe-write-lock", daemon=True)
    writer = threading.Thread(target=write_after, name="safe-write-retry", daemon=True)
    start = time.monotonic()
    try:
        holder.start()
        if not locked.wait(timeout=5.0):
            raise TimeoutError("Windows could not acquire the destination sharing lock")
        if holder_errors:
            raise RuntimeError("Windows sharing lock setup failed") from holder_errors[0]

        # A reader should still be able to parse the committed file while the
        # no-delete-sharing handle prevents the atomic replacement.
        if _read_json(target) != before:
            raise AssertionError("previous JSON was not readable under the sharing lock")

        writer.start()
        if not writer_started.wait(timeout=5.0):
            raise TimeoutError("safe_write retry worker did not start")
        if writer_done.wait(timeout=0.025):
            if writer_errors:
                raise AssertionError("safe_write failed before the lock was released") from writer_errors[0]
            raise AssertionError("safe_write completed while the destination was locked")
        if _read_json(target) != before:
            raise AssertionError("previous JSON changed before the locked rename could succeed")

        if not writer_done.wait(timeout=12.0):
            raise TimeoutError("safe_write did not finish after the sharing lock was released")
        if writer_errors:
            raise RuntimeError("safe_write failed after the transient lock was released") from writer_errors[0]
        if _read_json(target) != after:
            raise AssertionError("safe_write did not commit the new JSON after retrying")

        retries = writer_namespace.get("WRITE_RETRIES", retries_before) - retries_before
        if retry_counter_available and retries < 1:
            raise AssertionError("safe_write completed without recording a PermissionError retry")

        return {
            "roundtrip": "pass",
            "transient_locked_destination": "pass",
            "previous_json_readable_until_replace": True,
            "permission_retries": retries if retry_counter_available else None,
            "lock_seconds": 0.1,
            "locked_write_elapsed_seconds": round(time.monotonic() - start, 3),
        }
    finally:
        # Unblock and join the lock helper even when an earlier assertion fails.
        writer_started.set()
        if holder.is_alive():
            holder.join(timeout=2.0)
        if writer.is_alive():
            writer.join(timeout=12.0)
        for path in (target, temp_target):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
