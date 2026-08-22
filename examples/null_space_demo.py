#!/usr/bin/env python3
"""Demonstrate fixed-TCP arm-angle motion and Jacobian null-space checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuka_iiwa_solver import KUKAiiwaSolver  # noqa: E402
from srs_analytical_solver import SRSConfiguration  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--range-deg", type=float, default=35.0)
    args = parser.parse_args()
    if args.samples < 2 or args.range_deg <= 0.0:
        raise SystemExit("samples must be >= 2 and range-deg must be positive")

    solver = KUKAiiwaSolver()
    seed = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    target = solver.get_fk(seed)
    configuration = solver.get_configuration(seed)
    tangent = solver.get_null_space_direction(seed)
    jacobian_residual = np.linalg.norm(solver.get_jacobian(seed) @ tangent)

    position_errors, rotation_errors, solved = [], [], 0
    print(
        f"branch S/E/W={configuration.shoulder:+d}/"
        f"{configuration.elbow:+d}/{configuration.wrist:+d}, "
        f"seed psi={np.rad2deg(configuration.redundancy):.2f} deg"
    )
    print(f"||J dq/dpsi|| = {jacobian_residual:.3e}")
    for offset in np.linspace(-args.range_deg, args.range_deg, args.samples):
        selected = SRSConfiguration(
            configuration.shoulder,
            configuration.elbow,
            configuration.wrist,
            configuration.redundancy + np.deg2rad(offset),
        )
        ok, joints = solver.solve_configuration(target, selected, seed)
        if not ok:
            print(f"psi offset {offset:+7.2f} deg: unavailable")
            continue
        solved += 1
        actual = solver.get_fk(joints)
        position = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        cosine = np.clip(
            (np.trace(actual[:3, :3].T @ target[:3, :3]) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
        rotation = np.rad2deg(np.arccos(cosine))
        position_errors.append(position)
        rotation_errors.append(rotation)
        print(
            f"psi offset {offset:+7.2f} deg: "
            f"position={position * 1e3:.3e} mm, rotation={rotation:.3e} deg"
        )
    print(
        f"summary: {solved}/{args.samples} solved, max residual="
        f"{max(position_errors, default=float('nan')) * 1e3:.3e} mm / "
        f"{max(rotation_errors, default=float('nan')):.3e} deg"
    )


if __name__ == "__main__":
    main()
