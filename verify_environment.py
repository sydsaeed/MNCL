from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np


REQUIRED_MODULES = (
    "torch",
    "torch_geometric",
    "numpy",
    "tqdm",
    "matplotlib",
)


def check_module(name: str) -> tuple[bool, str]:
    """Check whether a Python module can be imported."""
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    version = getattr(module, "__version__", "unknown")
    return True, str(version)


def check_array(path: Path) -> tuple[bool, str]:
    """Check one NPY file and report its shape."""
    if not path.exists():
        return False, "missing"

    try:
        array = np.load(path, mmap_mode="r")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if array.ndim != 2 or array.shape[1] != 3:
        return False, f"invalid shape {array.shape}"
    return True, f"shape={array.shape}, dtype={array.dtype}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="last-fm")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    print("Python:", sys.version.split()[0])
    print("\nPackages")

    ok = True
    for name in REQUIRED_MODULES:
        passed, details = check_module(name)
        mark = "OK" if passed else "FAIL"
        print(f"  {mark:4} {name:16} {details}")
        ok = ok and passed

    dataset_dir = Path(args.data_root) / args.dataset
    print("\nData")
    for filename in ("ratings_final.npy", "kg_final.npy"):
        passed, details = check_array(dataset_dir / filename)
        mark = "OK" if passed else "FAIL"
        print(f"  {mark:4} {filename:20} {details}")
        ok = ok and passed

    print("\nResult:", "READY" if ok else "NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
