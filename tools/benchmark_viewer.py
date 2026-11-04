#!/usr/bin/env python3
"""Measure viewer startup and first/repeated opening of the eight-branch gallery."""

import argparse
from contextlib import redirect_stdout
from importlib.metadata import version
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NUMPY_REPO = (ROOT / "src" / "srs_viser").is_dir()
sys.path.insert(0, str(ROOT / "src" if NUMPY_REPO else ROOT))

if NUMPY_REPO:
    from srs_viser.app import LabApp
    from srs_viser.backend import Backend
    from srs_viser.model import MODES
else:
    from srs_warp_viser.app import LabApp
    from srs_warp_viser.backend import Backend
    from srs_warp_viser.model import MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("numpy", "warp"),
        default="numpy" if NUMPY_REPO else "warp",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, help="Write JSON to a new file")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.output and args.output.exists():
        parser.error("output file already exists")
    samples = {"startup_ms": [], "first_gallery_ms": [], "repeated_gallery_ms": []}
    # Backend initialization and one complete warmup are outside recorded trials.
    with redirect_stdout(sys.stderr):
        backend = Backend(args.backend, args.device)
        for trial in range(args.repeats + 1):
            started = time.perf_counter()
            app = LabApp(backend, port=0, meshes=True)
            startup_ms = (time.perf_counter() - started) * 1000
            try:
                started = time.perf_counter()
                app.commands.put(("mode", MODES["branches"]))
                app.step(1 / 30)
                first_ms = (time.perf_counter() - started) * 1000
                robots = tuple(app.gallery_robots)
                if len(robots) != 8:
                    raise AssertionError("gallery did not create eight robots")
                app.commands.put(("mode", MODES["arm-angle"]))
                app.step(1 / 30)
                started = time.perf_counter()
                app.commands.put(("mode", MODES["branches"]))
                app.step(1 / 30)
                repeated_ms = (time.perf_counter() - started) * 1000
                if tuple(app.gallery_robots) != robots:
                    raise AssertionError("gallery robots were recreated")
                error = app.model.diagnostics()
                if error["position_m"] > 1e-4 or error["rotation_rad"] > 1e-4:
                    raise AssertionError("gallery exceeded TCP tolerance")
                if trial:
                    for key, value in zip(samples, (startup_ms, first_ms, repeated_ms)):
                        samples[key].append(value)
            finally:
                app.close()
    report = {
        "backend": backend.label,
        "repeats": args.repeats,
        "environment": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in ("numpy", "viser", "yourdfpy", "trimesh")
            },
        },
        "scope": "server scene construction and IK; excludes browser, network and WebGL",
        "timings": {
            key: {"median_ms": float(np.median(values)), "samples_ms": values}
            for key, values in samples.items()
        },
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
