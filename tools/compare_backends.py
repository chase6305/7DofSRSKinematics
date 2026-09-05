#!/usr/bin/env python3
"""Compare NumPy scalar, NumPy batch, and Warp on identical iiwa targets."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warp-repo", type=Path, default=ROOT.parent / "7DofSRSKinematicsWarp"
    )
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--num-samples", type=int, default=73)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.samples < 1 or args.num_samples < 2 or args.repeats < 1:
        parser.error("samples/repeats must be positive and num-samples must be >= 2")
    if not (args.warp_repo / "srs_solver.py").is_file():
        parser.error("warp-repo must point to the 7DofSRSKinematicsWarp checkout")
    sys.path[:0] = [str(ROOT / "src"), str(args.warp_repo)]
    import torch
    from srs_solver import SRSSolver, SRSSolverCfg

    from kuka_iiwa_solver import KUKAiiwaSolver

    reference = KUKAiiwaSolver()
    rng = np.random.default_rng(6305)
    joints = rng.uniform(-1.2, 1.2, (args.samples, 7))
    seeds = np.clip(
        joints + rng.normal(0, 0.2, joints.shape),
        reference.lower_position_limits,
        reference.upper_position_limits,
    )
    targets = np.array([reference.get_fk(q) for q in joints])
    reference_jacobians = np.array([reference.get_jacobian(q) for q in joints])
    reference_chains = np.array([reference.get_all_fk_mat(q) for q in joints])

    def scalar_ik():
        results = [
            reference.get_ik(
                target, seed, num_samples=args.num_samples, search_mode="global"
            )
            for target, seed in zip(targets, seeds)
        ]
        return np.array([ok for ok, _ in results]), np.array(
            [q if ok else np.zeros(7) for ok, q in results]
        )

    backends = [
        (
            "NumPy scalar",
            lambda: np.array([reference.get_fk(q) for q in joints]),
            lambda: np.array([reference.get_jacobian(q) for q in joints]),
            scalar_ik,
            lambda: np.array([reference.get_all_fk_mat(q) for q in joints]),
        )
    ]
    for use_warp, device, label in (
        (False, "cpu", "NumPy batch"),
        (True, args.device, f"Warp {args.device}"),
    ):
        cfg = SRSSolverCfg(num_samples=args.num_samples)
        cfg.qpos_limits = np.column_stack(
            (reference.lower_position_limits, reference.upper_position_limits)
        )
        solver = SRSSolver(cfg, args.samples, device, use_warp=use_warp)
        q = torch.as_tensor(joints, dtype=torch.float32, device=device)
        t = torch.as_tensor(targets, dtype=torch.float32, device=device)
        seed = torch.as_tensor(seeds, dtype=torch.float32, device=device)
        backends.append(
            (
                label,
                lambda solver=solver, q=q: solver.get_fk(q),
                lambda solver=solver, q=q: solver.get_jacobian(q),
                lambda solver=solver, t=t, seed=seed: solver.get_ik(
                    t, seed, search_mode="global"
                ),
                lambda solver=solver, q=q: solver.get_all_fk_mat(q),
            )
        )

    def synchronize():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)

    def as_numpy(value):
        return (
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        )

    def measure(operation):
        operation()  # Compile and warm up outside timing.
        synchronize()
        elapsed = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = operation()
            synchronize()
            elapsed.append((time.perf_counter() - started) * 1e3)
        return result, float(np.median(elapsed))

    print(
        f"Targets={args.samples}, arm-angle samples={args.num_samples}, repeats={args.repeats}; times are per batch"
    )
    for label, fk, jacobian, ik, chain in backends:
        timings, results = [], []
        for operation in (fk, jacobian, ik, chain):
            result, median_ms = measure(operation)
            timings.append(median_ms)
            results.append(result)
        np.testing.assert_allclose(as_numpy(results[0]), targets, atol=1e-6, rtol=0)
        np.testing.assert_allclose(
            as_numpy(results[1]), reference_jacobians, atol=1e-6, rtol=0
        )
        np.testing.assert_allclose(
            as_numpy(results[3]), reference_chains, atol=1e-6, rtol=0
        )
        successes, solutions = results[2]
        successes = as_numpy(successes)
        solutions = as_numpy(solutions).reshape(-1, 7)
        if not successes.all():
            raise RuntimeError(f"{label} solved only {successes.sum()}/{args.samples}")
        actual = np.array([reference.get_fk(q) for q in solutions])
        position = np.linalg.norm(actual[:, :3, 3] - targets[:, :3, 3], axis=1).max()
        rotation = (
            2
            * np.arcsin(
                np.clip(
                    np.linalg.norm(actual[:, :3, :3] - targets[:, :3, :3], axis=(1, 2))
                    / np.sqrt(8),
                    0,
                    1,
                )
            )
        ).max()
        if position > 2e-4 or rotation > 2e-4:
            raise RuntimeError(f"{label} exceeds the pose residual tolerance")
        print(
            f"{label}: FK={timings[0]:.3f} ms, J={timings[1]:.3f} ms, IK={timings[2]:.3f} ms, chain={timings[3]:.3f} ms; {successes.sum()}/{args.samples}, residual={position:.3e} m/{rotation:.3e} rad"
        )

    chains, median_ms = measure(lambda: reference.get_all_fk_mat(joints))
    np.testing.assert_allclose(chains, reference_chains, atol=1e-12, rtol=0)
    print(
        f"NumPy reference vectorized chain: {median_ms:.3f} ms; all link poses verified"
    )


if __name__ == "__main__":
    main()
