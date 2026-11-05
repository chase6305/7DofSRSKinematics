# API reference

## Quick start

```python
import numpy as np
from kuka_iiwa_solver import KUKAiiwaSolver

solver = KUKAiiwaSolver()
seed = np.zeros(7)
target = solver.get_fk(np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]))
success, joints = solver.get_ik(target, seed, num_samples=73)
```

The default `search_mode="continuous"` tries the seed S/E/W branch and arm angle
first, which is intended for servoing. Other searches are available:

```python
success, joints = solver.get_ik(
    target, seed, num_samples=73, search_mode="global"
)
# For narrow joint-limit ranges, check analytical boundary events instead:
success, joints = solver.get_ik(target, seed, search_mode="interval")
```

| Mode | Search policy |
| --- | --- |
| `continuous` | Seed branch and local offsets, then the uniform grid, then interval search if needed. |
| `global` | Uniform `[-π, π)` samples on all eight branches; weighted nearest sampled solution. |
| `interval` | Analytical joint-limit boundaries and the midpoints between them; weighted nearest candidate. |

`interval` does not depend on the configured sample count. It handles narrow
feasible ranges without a denser uniform grid, but does not minimize the
continuous seed-distance objective or implement the papers' avoidance policy.
Boundary roots allow up to 1e-9 rad of reconstruction roundoff before clamping
to the exact joint limits. Every returned candidate must still meet the
configured position and rotation tolerances after FK. If this search still
fails, it checks the joint-limit box midpoint and projected seed. These FK-only
anchors can recover degenerate poses, including coincident shoulder/wrist
centres, without assigning an undefined arm angle.

`get_configuration(q)` returns the shoulder, elbow, and wrist signs together with the continuous arm
angle. Use it with `solve_configuration(target, configuration, seed)` for exact branch control.

Repeated queries at the same TCP reuse up to two elbow geometries; seeds,
limits and weights are evaluated on every call.

## API and numerical conventions

- `get_ik` and `solve_configuration` return `(False, None)` when no solution is found. Poses must be
  finite rigid homogeneous transforms and joint vectors must have shape `(7,)`; invalid inputs raise
  exceptions.
- `continuous` prefers nearby candidates and may change branches during fallback;
  it does not guarantee continuity. `global` minimizes `||(q - seed) * weight||₂`
  among its uniform samples. Finite sampling can miss narrow feasible ranges.
- A per-call `num_samples` leaves the solver default unchanged. Set the default through
  `set_iteration_params(num_samples=...)`, with at least two samples. The legacy `num_sample` alias
  is accepted; other unknown IK arguments raise an error.
- Every returned IK solution, including the seed fast path, respects joint limits and the positive
  elbow branch requested by `set_elbow_up(True)`. Periodic mapping works independently per joint
  without enumerating a Cartesian product.
- Workspace checks allow float32 rounding at full extension; every candidate must still meet the
  configured FK residual tolerances.
- `set_tcp(T)` sets the flange-to-TCP transform used by FK, IK, and Jacobians. Both `get_fk` and
  `get_all_fk_mat` return world-frame poses; the latter contains seven DH link frames followed by
  the TCP.
- `get_fk(q, index=-1)` returns `(4, 4)` for `(7,)` input and `(N, 4, 4)` for `(N, 7)` input.
  `index=-1` selects the TCP, while `0..6` selects a DH link. Empty and strided batches are
  supported; outputs own their storage and only the required chain prefix is traversed.
- `get_all_fk_mat(q)` retains a list of eight matrices for `(7,)` input and returns an `(N, 8, 4,
  4)` array for `(N, 7)` input. Batched DH evaluation supports empty and strided arrays; each call
  owns its output.
- `get_jacobian(q)` returns an analytical world-frame Jacobian: `(6, 7)` for `(7,)` input and `(N,
  6, 7)` for `(N, 7)` input, with TCP linear velocity first and angular velocity last. Empty/strided
  batches are supported and each output has independent storage. The `step` argument remains
  accepted for compatibility but is no longer used for finite differences.
- At shoulder or wrist singularities, both outer joints are solved together to minimize weighted
  seed distance within their limits. When shoulder and wrist centres coincide, the arm angle is
  undefined and `get_arm_angle` raises `ValueError`. Importing the library does not configure
  application logging.

## Exact configuration batches

`solve_configurations(target, configurations, seed)` solves multiple exact
branch/arm-angle requests for one target and seed. It returns boolean `(N,)`
success and `(N, 7)` joints in input order. Invalid rows are zero; mask them
before use. Geometry is shared by elbow branch, and decomposition and FK
validation are batched.

Joint-distance weights are normalized before ranking: multiplying all weights
by the same positive finite factor does not change the selected solution.
Zero weights retain deterministic input ordering for tied candidates.
