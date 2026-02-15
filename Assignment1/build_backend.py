#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    src = root / "cpp_backend" / "kernels.cpp"
    build_dir = root / "build"
    build_dir.mkdir(exist_ok=True)

    system = platform.system().lower()
    if system == "darwin":
        out = build_dir / "libgnrbackend.dylib"
        cmd = ["c++", "-O3", "-std=c++17", "-shared", "-fPIC", str(src), "-o", str(out)]
    elif system == "windows":
        out = build_dir / "gnrbackend.dll"
        cmd = ["g++", "-O3", "-std=c++17", "-shared", str(src), "-o", str(out)]
    else:
        out = build_dir / "libgnrbackend.so"
        cmd = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(src), "-o", str(out)]

    print("Building backend:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Built:", out)


if __name__ == "__main__":
    main()
