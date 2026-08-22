# 7DofSRSKinematics

[中文](README.md) | English

A NumPy-based geometric analytical kinematics library for 7-DoF spherical-shoulder/revolute-elbow/spherical-wrist (S-R-S) manipulators, with a ready-to-use KUKA LBR iiwa 7 R800 model.

## Features

- Analytical FK and IK for a 7-DoF S-R-S chain.
- All eight shoulder/elbow/wrist configuration branches.
- Continuous arm-angle redundancy and explicit branch solving.
- Periodic joint-limit wrapping and seed-weighted solution selection.
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

## Viser demo

```bash
python examples/viser_demo.py --port 8080
```

Viser uses the fast continuous mode by default; pass `--search-mode global` to compare exhaustive search.

Open `http://localhost:8080`. Joint sliders drive FK; dragging the orange TCP gizmo runs IK. The Null-space motion panel animates arm-angle self-motion while keeping the TCP fixed. The dashboard displays the active branch, arm angle, residual, solve time, rolling update FPS, and startup benchmark median/p95 latency.

```bash
python examples/viser_demo.py --validate-only --validation-samples 100
```

Arm-angle/null-space sweep with a fixed TCP:

```bash
python examples/null_space_demo.py --samples 25 --range-deg 35
```

## Tests

```bash
pytest -q
```

## Algorithm and references

The closed-form decomposition locates the wrist centre, solves the shoulder-elbow-wrist triangle, rotates a reference shoulder plane by arm angle ψ, and decomposes the shoulder and wrist spherical joints. `get_ik` searches all eight branches and ranks valid periodic representations against the seed.

- Shimizu et al., “Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators With Joint Limits and Its Application to Redundancy Resolution,” IEEE T-RO, 2008, [DOI](https://doi.org/10.1109/TRO.2008.2003266).
- Faria et al., “Position-based kinematics for 7-DoF serial manipulators with global configuration control, joint limit, and singularity avoidance,” MMT, 2018, [DOI](https://doi.org/10.1016/j.mechmachtheory.2017.10.025).

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

See [urdf/README.md](urdf/README.md) and [urdf/ASSET_LICENSE](urdf/ASSET_LICENSE) for mesh provenance, regeneration, and the BSD-3-Clause asset license.
