#!/usr/bin/env python3
"""Open the shared SRS lab, or run the core-only IK benchmark.

Legacy numerical checks accept --validate-only --validation-samples N and
--search-mode continuous/global. Other options are forwarded to the lab.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuka_iiwa_solver import KUKAiiwaSolver


def pose_error(actual: np.ndarray, target: np.ndarray):
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    cosine = np.clip(
        (np.trace(actual[:3, :3].T @ target[:3, :3]) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return position, float(np.rad2deg(np.arccos(cosine)))


def benchmark(
    solver: KUKAiiwaSolver,
    samples: int,
    num_samples: int,
    search_mode: str = "continuous",
) -> dict:
    if samples < 1:
        raise ValueError("validation-samples must be positive")
    rng = np.random.default_rng(6305)
    lower = np.maximum(solver.lower_position_limits, -1.2)
    upper = np.minimum(solver.upper_position_limits, 1.2)
    durations, position_errors, rotation_errors = [], [], []
    successes = 0
    for joints in rng.uniform(lower, upper, size=(samples, 7)):
        target = solver.get_fk(joints)
        # Use a deterministic perturbed seed so the benchmark measures IK
        # search rather than the stationary-target fast path.
        seed = np.clip(
            joints + rng.normal(0.0, 0.2, size=7),
            solver.lower_position_limits,
            solver.upper_position_limits,
        )
        started = time.perf_counter()
        ok, solution = solver.get_ik(
            target, seed, num_samples=num_samples, search_mode=search_mode
        )
        durations.append((time.perf_counter() - started) * 1e3)
        if not ok:
            continue
        successes += 1
        position, rotation = pose_error(solver.get_fk(solution), target)
        position_errors.append(position)
        rotation_errors.append(rotation)
    return {
        "samples": samples,
        "search_mode": search_mode,
        "successes": successes,
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "max_position": max(position_errors, default=float("nan")),
        "max_rotation": max(rotation_errors, default=float("nan")),
    }


def main() -> None:
    legacy_options = {"--validation-samples", "--num-samples", "--search-mode"}
    if not any(arg.split("=", 1)[0] in legacy_options for arg in sys.argv[1:]):
        from srs_viser.app import main as lab_main

        return lab_main()

    parser = argparse.ArgumentParser(description="Core-only IK benchmark")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--num-samples", type=int, default=73)
    parser.add_argument(
        "--search-mode", choices=("continuous", "global"), default="continuous"
    )
    parser.add_argument("--validation-samples", type=int, default=24)
    parser.add_argument("--validate-only", action="store_true", required=True)
    args = parser.parse_args()
    if args.validation_samples < 1 or args.num_samples < 2:
        parser.error("validation-samples must be >= 1 and num-samples must be >= 2")
    validation = benchmark(
        KUKAiiwaSolver(), args.validation_samples, args.num_samples, args.search_mode
    )
    print(
        f"IK [{validation['search_mode']}] {validation['successes']}/{validation['samples']}, "
        f"median/p95={validation['median_ms']:.3f}/{validation['p95_ms']:.3f} ms, "
        f"max residual={validation['max_position'] * 1e3:.3e} mm/{validation['max_rotation']:.3e} deg"
    )
    raise SystemExit(0 if validation["successes"] == validation["samples"] else 1)


if __name__ == "__main__":
    main()
