"""One-shot isolated server bootstrap/dispatch. Receipts remain private/local."""
from __future__ import annotations

import argparse
import ctypes.util
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def environment(root, gpu=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "deps") + os.pathsep + env.get("PYTHONPATH", "")
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def probe():
    modules = ["torch", "numpy", "PIL", "scipy", "skimage", "requests", "libarchive",
               "imageio", "tifffile", "lazy_loader", "packaging"]
    return {"hostname": socket.gethostname(), "python": sys.executable,
            "modules": {name: importlib.util.find_spec(name) is not None for name in modules},
            "libarchive": ctypes.util.find_library("archive"),
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                                            "--format=csv,noheader"], text=True)}


def verify(root):
    directory = root / "new/real_task_fives"
    files = sorted(directory.glob("*.py")) + [directory / "PROTOCOL.md", directory / "README.md"]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def bootstrap(root):
    root.mkdir(parents=True, exist_ok=True)
    # Install only missing packages in this experiment's directory. Never
    # replace a shared environment's NumPy/PyTorch or touch another job.
    packages = {"PIL": "Pillow", "skimage": "scikit-image==0.24.0", "scipy": "scipy==1.14.1",
                "imageio": "imageio", "tifffile": "tifffile", "lazy_loader": "lazy-loader",
                "packaging": "packaging", "requests": "requests", "libarchive": "libarchive-c"}
    needed = [pkg for module, pkg in packages.items() if importlib.util.find_spec(module) is None]
    if needed:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "--target", str(root / "deps"), "--no-deps", *needed], check=True)
    env = environment(root)
    subprocess.run([sys.executable, "-c", "import torch,numpy,PIL,scipy,skimage,requests,libarchive; print('DEPENDENCIES_OK')"],
                   env=env, check=True)


def launch(root, gpu, min_free_mib, resume=False):
    receipt_path = root / "launch_receipt.json"
    if receipt_path.exists() and not resume:
        raise FileExistsError("Existing launch receipt: use explicit recovery, never duplicate a worker")
    if resume:
        previous = json.loads(receipt_path.read_text())
        try:
            os.kill(previous["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Previous worker still exists; refusing duplicate dispatch")
    free = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    if int(free) < min_free_mib:
        raise RuntimeError(f"Selected GPU has only {free} MiB free; no dispatch")
    command = [sys.executable, "-X", "utf8", "-u", "-B",
               str(root / "new/real_task_fives/pipeline.py"), "--root", str(root), "--device", "cuda:0"]
    if resume:
        history = root / "dispatch_attempts" / str(time.time_ns())
        history.mkdir(parents=True)
        for name in ("launch_receipt.json", "pipeline_status.json"):
            source = root / name
            if source.exists():
                shutil.copy2(source, history / name)
        command.append("--resume")
    log = root / "pipeline.log"
    with log.open("ab") as handle:
        process = subprocess.Popen(command, cwd=root, env=environment(root, gpu),
                                   stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    receipt = {"status": "DISPATCHED", "host": socket.gethostname(), "pid": process.pid,
               "gpu": gpu, "gpu_free_mib_at_dispatch": int(free), "command": command,
               "log": str(log), "root": str(root), "launched_unix": time.time(),
               "runtime_cap": None, "recurring_monitor": False,
               "explicit_recovery": resume, "deployed_sources": verify(root)}
    atomic_json(receipt_path, receipt)
    return receipt


def status(root):
    receipt = json.loads((root / "launch_receipt.json").read_text())
    try:
        os.kill(receipt["pid"], 0)
        receipt["pid_exists"] = True
    except ProcessLookupError:
        receipt["pid_exists"] = False
    for name in ("pipeline_status.json", "runs/fives_nca_20261010_01/progress.json"):
        path = root / name
        if path.exists():
            receipt[name] = json.loads(path.read_text())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["probe", "bootstrap", "verify", "launch", "recover", "status"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=8000)
    args = parser.parse_args()
    if args.action == "probe":
        value = probe()
    elif args.action == "bootstrap":
        bootstrap(args.root)
        value = {"status": "BOOTSTRAPPED"}
    elif args.action in ("launch", "recover"):
        value = launch(args.root, args.gpu, args.min_free_mib, args.action == "recover")
    elif args.action == "verify":
        value = verify(args.root)
    else:
        value = status(args.root)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
