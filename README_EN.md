# 7DofSRSKinematics

[中文](README.md) | English

A NumPy-based geometric analytical kinematics library for 7-DoF spherical-shoulder/revolute-elbow/spherical-wrist (S-R-S) manipulators, with a ready-to-use KUKA LBR iiwa 7 R800 model.

## Features

- Analytical FK and IK for a 7-DoF S-R-S chain.
- All eight shoulder/elbow/wrist configuration branches.
- Continuous arm-angle redundancy and explicit branch solving.
- Periodic joint-limit wrapping and seed-weighted solution selection.
- NumPy batched arm-angle search, analytical geometric Jacobians, and fixed-TCP null-space motion.
- Interactive Viser FK/IK simulation and performance dashboard.
- NumPy-only core; visualization dependencies remain optional.
- Colored orange/grey GLB visuals and source R800 collision geometry.

## Installation

```bash
python -m pip install -e .
python -m pip install -e '.[test,visualization]'  # development and demo
```

## Quick start

```python
import numpy as np
from kuka_iiwa_solver import KUKAiiwaSolver

solver = KUKAiiwaSolver()
seed = np.zeros(7)
target = solver.get_fk(np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]))
success, joints = solver.get_ik(target, seed, num_samples=73)
```

The default `search_mode="continuous"` preserves the seed S/E/W branch and true arm angle first, which is intended for servoing. Exhaustive search remains available:

```python
success, joints = solver.get_ik(
    target, seed, num_samples=73, search_mode="global"
)
```

`get_configuration(q)` returns the shoulder, elbow, and wrist signs together with the continuous arm angle. Use it with `solve_configuration(target, configuration, seed)` for exact branch control.

Repeated queries at the same TCP reuse the most recent target geometry across scalar configurations, configuration batches and NumPy IK search, retaining at most two elbow branches. Keys compare the exact contents of the local target, DH parameters and link lengths. Seeds, limits, weights and feasibility checks remain live on every call. Fixed-TCP self-motion benefits; moving targets rebuild geometry and incur a small cache-check overhead.

## API and numerical conventions

- `get_ik` and `solve_configuration` return `(False, None)` when no solution is found. Poses must be finite rigid homogeneous transforms and joint vectors must have shape `(7,)`; invalid inputs raise exceptions.
- `continuous` returns a nearby feasible solution, preferring the seed branch, then falls back to global sampling. `global` uniformly samples `[-π, π)` across all eight branches and minimizes `||(q - seed) * weight||₂` among the samples. Finite sampling does not guarantee a continuous optimum or discovery of very narrow feasible intervals.
- A per-call `num_samples` no longer changes the solver default. Set the default through `set_iteration_params(num_samples=...)`, with at least two samples. The legacy `num_sample` alias is accepted; other unknown IK arguments raise an error.
- Every returned IK solution, including the seed fast path, respects joint limits and the positive elbow branch requested by `set_elbow_up(True)`. Periodic mapping works independently per joint without enumerating a Cartesian product.
- `set_tcp(T)` sets the flange-to-TCP transform used by FK, IK, and Jacobians. Both `get_fk` and `get_all_fk_mat` return world-frame poses; the latter contains seven DH link frames followed by the TCP.
- `get_fk(q, index=-1)` returns `(4, 4)` for `(7,)` input and `(N, 4, 4)` for `(N, 7)` input. `index=-1` selects the TCP, while `0..6` selects a DH link. Empty and strided batches are supported; outputs own their storage and only the required chain prefix is traversed. Batched configuration IK uses this TCP-only path for residual checks without allocating all link poses.
- `get_all_fk_mat(q)` retains a list of eight matrices for `(7,)` input and returns an `(N, 8, 4, 4)` array for `(N, 7)` input. Batched DH evaluation supports empty and strided arrays; each call owns its output.
- `get_jacobian(q)` returns an analytical `6×7` world-frame Jacobian: TCP linear velocity first, angular velocity last. The `step` argument remains accepted for compatibility but is no longer used for finite differences.
- At shoulder or wrist singularities, both outer joints are solved together to minimize weighted seed distance within their limits. When shoulder and wrist centres coincide, the arm angle is undefined and `get_arm_angle` raises `ValueError`. Importing the library does not configure application logging.

## Warp backend comparison

The two repositories install independently; each wheel includes its URDF and visual/collision resources. When changing solver formulas, run this repository's tests and the Warp repository's `test/test_backend_consistency.py`. Cross-repository checks run automatically for sibling checkouts, or use `SRS_NUMPY_REPO` to specify this checkout.

Measure fixed-TCP sweeps, moving-TCP paths and 584-point manifold batches separately. The core-only benchmark warms up each workload, reports median times and checks every returned TCP residual:

```bash
python tools/benchmark_srs.py --frames 180 --repeats 15 --output benchmark.json
```

`--output` writes a new JSON file. Exact configuration IK also uses CPU geometry in the Warp repository; this benchmark does not measure GPU global batch IK throughput.

```bash
python tools/compare_backends.py --warp-repo ../7DofSRSKinematicsWarp --samples 40
```

This tool requires the Warp repository's dependencies. It compares scalar NumPy, batched NumPy fallback, and Warp using identical targets, limits, and arm-angle samples. It checks FK, analytical Jacobians, global IK and residuals, excluding compilation warmup from timings. The default device is CPU; use `--device cuda` for Warp CUDA measurements.

## Viser demo

The SRS lab provides five experiments, also selectable from the Experiment menu:

```bash
python examples/viser_srs_lab.py --demo arm-angle --port 8081
python examples/viser_srs_lab.py --demo branches --port 8082
python examples/viser_srs_lab.py --demo jacobian --port 8083
python examples/viser_trajectory_demo.py --port 8084
python examples/viser_manifold_demo.py --port 8085
```

After installing `.[visualization]`, the same options are available through `srs-viser-lab`. Both repositories pin their visualization extras to the verified Viser 1.0.24 release.

| Mode | Experiment |
|:--|:--|
| `arm-angle` | Sweep the arm angle at a fixed TCP, with shoulder/elbow/wrist markers, the shoulder–wrist axis and geometric elbow circle. Infeasible angles retain the last feasible pose and display a message. |
| `branches` | Compare eight S/E/W branches, availability and minimum joint-limit margin. Tiles are translated for display; their local target and arm angle are identical. |
| `jacobian` | Inspect the linear velocity ellipsoid and joint-centering preference. Play projects centering velocity into the full Jacobian null space and corrects each step back to the exact TCP target. |
| `trajectory` | Follow a line, circle or figure eight with fixed TCP orientation and S/E/W branch. Gray is the reference and green is the FK trace, with error and maximum joint-step metrics. |
| `manifold` | Sample all eight fixed-TCP self-motion branches and project them onto J1/J3/J5. Infeasible samples and large joint jumps break the curves; a marker tracks the current configuration. |

Trajectory controls select the path, plane, size and joint-speed bound. Path plane offers world XY/XZ/YZ and TCP-local XY, using the TCP orientation at the start. Changing the path, plane or size starts a new reference from the current TCP. Progress slows near constraints and pauses at the last feasible pose when no valid step remains. Frequency is a reference value: actual progress may be slower. Pause to drag the TCP gizmo. The right plot magnifies displacement along the selected plane axes by six.

All automatic playback modes respect Joint speed bound using direct joint differences per frame, without wrapping away full turns. Up to 12 progressively halved steps let fast arm-angle sweeps and slow Cartesian paths reduce progress before pausing with the last feasible pose and phase intact. Self-motion playback stops at failed steps instead of continuing its sweep to a distant feasible angle. Manual joint, branch and arm-angle controls select poses directly; changing sweep amplitude recenters playback at the current arm angle.

Pausing and resuming preserves the sweep phase, center and direction. Pose, branch, arm-angle, amplitude or experiment changes start a new sweep.

The CLI also accepts `--path`, `--plane`, `--path-size`, `--preset`, `--frequency`, `--joint-speed` and `--amplitude`. Export reproducible playback without installing Viser:

```bash
python examples/viser_trajectory_demo.py --plane tool-xy --path circle --path-size 0.015 --autoplay
python examples/viser_trajectory_demo.py --plane world-xz --path circle --path-size 0.015 --frames 300 --fps 30 --export-csv trajectory.csv
python examples/viser_manifold_demo.py --frames 300 --fps 30 --export-csv self_motion.csv
```

CSV includes the initial pose and each accepted step: simulated time, seven joint angles, actual/target world TCP positions and rotation matrices, pose errors, joint-limit margin, peak joint speed, phase, arm angle and S/E/W branch. Positions are in meters, angles in radians, and matrices are flattened by row. `--fps` accepts 20–240; export uses fixed simulation time steps of `1/fps`, independent of computation time. All five modes export the selected robot. Early pauses retain accepted rows and return exit code 2 with a reason in the JSON report. Existing files are preserved; `--validate-only` and `--export-csv` are mutually exclusive.

Projection axes switches the manifold between J1/J3/J5, shoulder J1/J2/J3, wrist J5/J6/J7 and J2/J4/J6. Curves update only when the target, sampling resolution or projection changes; playback updates the current marker.

Manifold resolution is controlled by Samples / branch. Its coordinates are joint angles in radians, rendered at scale 0.22; crossings in the projection need not be the same seven-joint configuration. Finite sampling may miss narrow feasible intervals. The arm-angle slider and Play move the robot and current joint-space marker together.

`solve_configurations(target, configurations, seed)` exposes the batched exact IK used for manifold sampling. It returns a `(N,)` boolean mask and `(N, 7)` joints in input order; invalid rows are zero and must be masked. Geometry is prepared once per elbow branch and arm-angle evaluation and FK checks are batched.

Drag the TCP gizmo to set a new target, or use joint sliders and Pose preset to reset the pose. Play starts arm-angle sweeps or joint centering. `--autoplay` starts playback; `--no-mesh` shows only the geometric skeleton. The server binds to `127.0.0.1` by default. Paused idle scenes do not recompute or redraw.

Consecutive drag samples from one control are coalesced on insertion. Different controls, reset and mode changes preserve their order so coalescing cannot silently change the TCP target. Rendering and diagnostics share link poses; self-motion reuses the previous frame's Jacobian. Branch IK and link poses are cached by target and arm angle. Automatic stops immediately refresh the final status.

The orange circle is geometric: joint limits can exclude parts of it. The velocity ellipsoid represents translational velocity for unit joint-speed norm, rendered at scale 0.25, **with orientation unconstrained**. Actual self-motion uses the full `6×7` Jacobian. Centering is a local descent example on the current target and branch, without a global optimality guarantee.

All five numerical experiments can be checked without visualization dependencies. Validation checks limits, separate position/rotation residuals and a decreasing centering objective, and exits nonzero on failure:

```bash
python examples/viser_srs_lab.py --validate-only --validation-frames 40
```

The original FK/IK performance dashboard is also available:

```bash
python examples/viser_demo.py --port 8080
```

Viser uses the fast continuous mode by default; pass `--search-mode global` to compare exhaustive search.

Open `http://localhost:8080`. Joint sliders drive FK; dragging the orange TCP gizmo runs IK. The Null-space motion panel animates arm-angle self-motion while keeping the TCP fixed. The dashboard displays the active branch, arm angle, residual, solve time, rolling update FPS, and startup benchmark median/p95 latency.

Headless validation requires only NumPy and uses the selected search mode:

```bash
python examples/viser_demo.py --validate-only --validation-samples 100
python examples/viser_demo.py --validate-only --validation-samples 100 --search-mode global
```

It uses a fixed random seed and perturbed joint seeds, reports successes, median/p95 latency and pose residuals, and exits with a nonzero status if any solve fails.

Arm-angle/null-space sweep with a fixed TCP:

```bash
python examples/null_space_demo.py --samples 25 --range-deg 35
```

## Tests

```bash
pytest -q
```

## Algorithm and references

The closed-form decomposition locates the wrist centre, solves the shoulder-elbow-wrist triangle, rotates a reference shoulder plane by arm angle ψ, and decomposes the shoulder and wrist spherical joints. Global search reuses geometric coefficients per elbow branch, evaluates arm-angle candidates in batches, ranks valid periodic representations against the seed, and checks the returned solution's FK residual.

- Shimizu et al., “Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators With Joint Limits and Its Application to Redundancy Resolution,” IEEE T-RO, 2008, [DOI](https://doi.org/10.1109/TRO.2008.2003266).
- Faria et al., “Position-based kinematics for 7-DoF serial manipulators with global configuration control, joint limit, and singularity avoidance,” MMT, 2018, [DOI](https://doi.org/10.1016/j.mechmachtheory.2017.10.025).

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

See [urdf/README.md](urdf/README.md) and [urdf/ASSET_LICENSE](urdf/ASSET_LICENSE) for mesh provenance, regeneration, and the BSD-3-Clause asset license.
