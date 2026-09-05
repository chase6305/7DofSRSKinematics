#!/usr/bin/env python3
"""Measure exact IK on fixed/moving TCPs and batched self-motion sampling."""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NUMPY_REPO = (ROOT / "src" / "srs_viser").is_dir()
sys.path.insert(0, str(ROOT / "src" if NUMPY_REPO else ROOT))

if NUMPY_REPO:
    from srs_viser.backend import Backend
    from srs_viser.model import Experiment, pose_error
    from srs_viser.motions import CartesianPath
else:
    from srs_warp_viser.backend import Backend
    from srs_warp_viser.model import Experiment, pose_error
    from srs_warp_viser.motions import CartesianPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("numpy", "warp"),
        default="numpy" if NUMPY_REPO else "warp",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, help="Write a JSON report to this new file"
    )
    args = parser.parse_args()
    if args.frames < 2 or args.repeats < 1:
        parser.error("frames must be >= 2 and repeats must be positive")
    backend = Backend(args.backend, args.device)
    model = Experiment(backend)
    seed, target = model.q, model.target
    fixed = [
        (model.branch, model.anchor_psi + angle)
        for angle in np.linspace(-0.25, 0.25, args.frames)
    ]
    path = CartesianPath(target, "circle", 0.015)
    moving = [
        path.pose(phase)
        for phase in np.linspace(0, 2 * np.pi, args.frames, endpoint=False)
    ]
    manifold = [
        (branch, psi)
        for branch in product((-1, 1), repeat=3)
        for psi in np.linspace(-np.pi, np.pi, 73)
    ]

    def scalar_fixed():
        return [backend.solve(target, branch, psi, seed) for branch, psi in fixed]

    def scalar_moving():
        return [
            backend.solve(pose, model.branch, model.anchor_psi, seed) for pose in moving
        ]

    def batch_manifold():
        valid, joints = backend.solve_many(target, manifold, seed)
        return list(joints[valid])

    report = {"backend": backend.label, "repeats": args.repeats, "workloads": {}}
    for name, operation, requests in (
        ("fixed_tcp", scalar_fixed, args.frames),
        ("moving_tcp", scalar_moving, args.frames),
        ("manifold_batch", batch_manifold, len(manifold)),
    ):
        operation()  # Warm imports, geometry and device transfers outside timing.
        times = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = operation()
            times.append((time.perf_counter() - started) * 1000)
        if not result or any(q is None for q in result):
            raise AssertionError(f"{name} failed to solve a benchmark pose")
        actual = backend.chains(np.asarray(result))[:, -1]
        expected = moving if name == "moving_tcp" else [target] * len(result)
        errors = np.array(
            [pose_error(pose, wanted) for pose, wanted in zip(actual, expected)]
        )
        if np.max(errors) > 3e-6:
            raise AssertionError(f"{name} exceeded pose tolerance")
        median = float(np.median(times))
        report["workloads"][name] = {
            "requests": requests,
            "valid_solutions": len(result),
            "median_ms": median,
            "microseconds_per_request": median * 1000 / requests,
            "max_position_error_m": float(errors[:, 0].max()),
            "max_rotation_error_rad": float(errors[:, 1].max()),
        }
    rendered = json.dumps(report, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
