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

The default `search_mode="continuous"` preserves the seed S/E/W branch and true arm angle first, which is intended for servoing. Exhaustive search remains available:

```python
success, joints = solver.get_ik(
    target, seed, num_samples=73, search_mode="global"
)
```

`get_configuration(q)` returns the shoulder, elbow, and wrist signs together with the continuous arm angle. Use it with `solve_configuration(target, configuration, seed)` for exact branch control.

Repeated queries at the same TCP reuse up to two elbow geometries; seeds,
limits and weights are evaluated on every call.

## API and numerical conventions

- `get_ik` and `solve_configuration` return `(False, None)` when no solution is found. Poses must be finite rigid homogeneous transforms and joint vectors must have shape `(7,)`; invalid inputs raise exceptions.
- `continuous` returns a nearby feasible solution, preferring the seed branch, then falls back to global sampling. `global` uniformly samples `[-π, π)` across all eight branches and minimizes `||(q - seed) * weight||₂` among the samples. Finite sampling does not guarantee a continuous optimum or discovery of very narrow feasible intervals.
- A per-call `num_samples` leaves the solver default unchanged. Set the default through `set_iteration_params(num_samples=...)`, with at least two samples. The legacy `num_sample` alias is accepted; other unknown IK arguments raise an error.
- Every returned IK solution, including the seed fast path, respects joint limits and the positive elbow branch requested by `set_elbow_up(True)`. Periodic mapping works independently per joint without enumerating a Cartesian product.
- Workspace checks allow float32 rounding at full extension; every candidate must still meet the configured FK residual tolerances.
- `set_tcp(T)` sets the flange-to-TCP transform used by FK, IK, and Jacobians. Both `get_fk` and `get_all_fk_mat` return world-frame poses; the latter contains seven DH link frames followed by the TCP.
- `get_fk(q, index=-1)` returns `(4, 4)` for `(7,)` input and `(N, 4, 4)` for `(N, 7)` input. `index=-1` selects the TCP, while `0..6` selects a DH link. Empty and strided batches are supported; outputs own their storage and only the required chain prefix is traversed.
- `get_all_fk_mat(q)` retains a list of eight matrices for `(7,)` input and returns an `(N, 8, 4, 4)` array for `(N, 7)` input. Batched DH evaluation supports empty and strided arrays; each call owns its output.
- `get_jacobian(q)` returns an analytical world-frame Jacobian: `(6, 7)` for `(7,)` input and `(N, 6, 7)` for `(N, 7)` input, with TCP linear velocity first and angular velocity last. Empty/strided batches are supported and each output has independent storage. The `step` argument remains accepted for compatibility but is no longer used for finite differences.
- At shoulder or wrist singularities, both outer joints are solved together to minimize weighted seed distance within their limits. When shoulder and wrist centres coincide, the arm angle is undefined and `get_arm_angle` raises `ValueError`. Importing the library does not configure application logging.

## Exact configuration batches

`solve_configurations(target, configurations, seed)` solves multiple exact
branch/arm-angle requests for one target and seed. It returns boolean `(N,)`
success and `(N, 7)` joints in input order. Invalid rows are zero; mask them
before use. Geometry is shared by elbow branch, and decomposition and FK
validation are batched.

Joint-distance weights are normalized before ranking: multiplying all weights
by the same positive finite factor does not change the selected solution.
Zero weights retain deterministic input ordering for tied candidates.

## S-R-S recognition from URDF

```bash
srs-identify robot.urdf --base base_link --tip tool_link
```

The command defaults to the URDF root and the sole leaf below the selected base.
Specify the tip explicitly when that subtree has multiple leaves. The tool accepts
URDF XML with fixed joints, independent revolute joints and continuous joints on
the selected path. It uses joint origins and axes, so mesh files are optional.
Expand Xacro macros before passing an XML file to the tool.
`--position-tolerance` and `--angle-tolerance` default to `1e-6` metres and radians.

Joint origins accept `rpy` for URDF 1.0 through 1.2, or `quat_xyzw` for
URDF 1.1 and 1.2. A joint origin must specify at most one rotation representation.
URDF 1.2 requires explicit lower and upper bounds on revolute joints.
[URDF version reference](https://github.com/ros/urdfdom/tree/rolling#urdf-specification)

The report sets `is_srs` when the seven rotary joints form two groups of three
intersecting, nondegenerate axes with a rotary elbow between them. Shoulder and
wrist errors are the maximum distances from their least-squares centers to the
corresponding axes. The elbow center is the projection of the shoulder onto J4.
Centers are expressed in the selected base frame at URDF joint zero.

`solver_compatible` requires adjacent orthogonal axes in the shoulder and wrist,
J4 perpendicular to J3 and J5, and nonzero upper-arm and forearm segments along
J3 and J5. For compatible models, `parameters` contains standard DH rows
`[d, alpha, a, theta]`, lengths `[d_bs, d_se, d_ew, d_wt]`, and the
`base_to_solver` and `flange_to_tip` homogeneous transforms. The base origin is
projected onto J1 and the tip offset along J7 is included in `d_wt`. Tolerances
bound the inferred geometry and the fit of all seven axes at the extracted
solver reference pose. `joint_axis_position_errors` measures the distance from
each canonical joint center to its corresponding URDF axis in metres.
`joint_axis_angle_errors` measures the unsigned angle between each pair of axes
in radians. Both list values in chain order when a canonical fit is attempted.
They describe the fitted model at that reference pose. Cartesian tip errors can
accumulate across multiple moving joints.

`joint_limits` lists URDF bounds in chain order, with `[null, null]` for
continuous joints. The command exits with status 0 with extracted parameters, 1 when
parameters are unavailable, or 2 for invalid input (diagnostic on stderr).

```python
import numpy as np
from srs_kinematics import analyze_urdf

analysis = analyze_urdf("urdf/iiwa_7.urdf", tip_link="link_ee")
solver = analysis.make_solver()
parameters = analysis.parameters
q_urdf = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
q_solver = parameters.to_solver_joints(q_urdf)
target = solver.get_fk(q_solver)
ok, solution = solver.get_ik(target, q_solver + 0.05)
if ok:
    print(dict(zip(analysis.joint_names, parameters.to_urdf_joints(solution))))
```

The coordinate mapping is `q_solver = joint_signs * (q_urdf - joint_zero)`.
Both mapping methods accept arrays whose last dimension is 7. Use solver
coordinates with FK, IK and branch APIs. FK returns the selected tip pose in the
selected URDF base frame. `make_solver()` raises `ValueError` with the analysis
reason when parameters are unavailable.
`analysis.to_dict()` produces the same report structure as the CLI.

### Compare extracted and URDF forward kinematics

```bash
srs-identify robot.urdf --tip tool_link --check-fk
```

For each compatible chain, `--check-fk` adds an `fk_check` object with the sample
count, maximum position error in metres, and maximum rotation error in radians.
It uses 32 deterministic samples over the URDF position bounds, with one turn
from `-π` to `π` for continuous joints. The errors measure the sampled tip poses
and the exit status continues to report solver compatibility.

For custom positions, `analysis.compare_fk(q_urdf)` reads the source URDF and
returns position and rotation errors with the leading shape of `q_urdf`.

### Discover arms in a larger URDF

```bash
srs-identify robot.urdf --scan
srs-identify robot.urdf --scan --base torso --check-fk
```

Scan mode considers windows of seven consecutive independent rotary joints,
allowing fixed joints between them. Each result starts at the beginning of the
fixed segment that mounts J1. Tips follow fixed attachments down to leaf links
or links that mount another movable joint, such as a gripper. Results are sorted
by base and tip, so multiple fixed tool frames can produce several results for
the same seven joints.

The JSON object has a `chains` array containing all geometric S-R-S matches,
including matches rejected by the solver. Exit status 0 indicates at least one
compatible match. Status 1 covers both an empty array and matches without
extractable parameters. In Python, `scan_urdf(path, base_link=None)` returns
the same matches as a tuple of `SRSAnalysis` objects.
