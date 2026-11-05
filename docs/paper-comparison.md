# Comparison with the reference papers

**This repository implements the arm-angle analytical core, but does not fully
reproduce either paper.** The gaps affect completeness and redundancy selection,
not just execution speed.

Reviewed solver baselines: NumPy `8feb79e` and Warp `aab4b5b`.

## Paper scope and implementation

| Topic | Reference method | Repository behavior |
| --- | --- | --- |
| Parameterized IK | Faria, Sections 3–4: S/E/W configuration and arm angle; shoulder/wrist matrix decomposition. | Implemented for the canonical S-R-S chain. `_reference_geometry`, `_build_ik_geometry` and `_candidate_joint_angles` follow this formulation, with numerically stable `atan2` forms. |
| Feasible arm angles | Shimizu, Section III / Eq. (49): analytically intersect joint-feasible arm-angle sets. Faria, Section 5: map limits and singularities to arm-angle intervals. | Missing. Candidate angles are evaluated and filtered by joint limits and FK residuals. No complete interval representation is constructed. |
| Redundancy selection | Shimizu, Section IV / Eq. (51): optimize within the feasible set; its joint-limit example uses shoulder/wrist orientation objectives. Faria, Eq. (36): update the previous arm angle using interval boundaries and parameters K and α. | Different policy: `global` ranks sampled candidates by weighted joint distance to the seed. `continuous` tries the seed branch/angle, local offsets and then a global grid. |
| Singularity avoidance | Faria, Section 5.1.3: exclude a neighborhood of singular arm angles. | Selected singular configurations are handled numerically by coupled outer-joint solutions and SVD. Handling a singularity is not an avoidance policy; there is no exclusion-width parameter δ. |
| Continuity | Faria's interval update depends on the previous arm angle and a feasible interval. | `continuous` expresses a search preference, not a path-continuity guarantee. It can fall back to other branches. Explicit configuration queries and the viewer retain a requested branch or reject a step. |
| Validation | The papers report specific simulation and performance experiments. | Current FK/IK, limits, gradient and viewer tests check this implementation. They do not reproduce those experimental protocols or establish the papers' completeness claims. |

The default DH topology agrees with Faria's Table 1. General DH support in FK
should not be read as a general analytical IK solution for arbitrary offsets.
The viewer's Jacobian joint-centering, backtracking and speed limits are
additional engineering choices, rather than the papers' interval-based policies.

## A valid target that a sampled search misses

A narrow joint box contains the known solution below. With 73 arm-angle samples,
`global` returns failure while the exact configuration query succeeds:

```python
import numpy as np
from kuka_iiwa_solver import KUKAiiwaSolver

solver = KUKAiiwaSolver()
q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
solver.set_position_limits(q - 1e-4, q + 1e-4)
target = solver.get_fk(q)
configuration = solver.get_configuration(q)
seed = np.zeros(7)

exact_ok, exact_q = solver.solve_configuration(target, configuration, seed)
sampled_ok, _ = solver.get_ik(target, seed, num_samples=73, search_mode="global")
print(exact_ok, sampled_ok)  # True, False
```

The [recorded check](benchmarks/sampling-witness.json) has an exact-solution
position error of 5.6e-17 m. The same narrow-box example
also gives exact success and sampled failure in the Warp repository's NumPy
fallback, Warp CPU and Warp CUDA backends. Increasing the sample count can help,
but does not turn a finite grid into a complete feasible-set construction.
Consequently, sampled failure does not prove that a target is unreachable, and
`global` means the best checked sample rather than a continuous global optimum.

Code: [analytical solver](../src/srs_analytical_solver.py),
[periodic joint-limit mapping](../src/solver.py),
[viewer motion model](../src/srs_viser/model.py).

A complete reproduction would additionally need analytical interval construction
and intersection, the chosen paper's redundancy/avoidance policy, and comparisons
against its published experiments. The comparison here establishes those scope
differences; it is not a formal proof of equation-by-equation equivalence.

## Sources

- Shimizu et al. (2008).
  [IEEE Transactions on Robotics](https://doi.org/10.1109/TRO.2008.2003266), Sections III–IV.
  [Publisher record](https://ieeexplore.ieee.org/document/4631505/) and
  [public full-text copy](https://www.scribd.com/document/1006994742/TRO-2008-2003266).
- Faria et al. (2018).
  [Mechanism and Machine Theory](https://doi.org/10.1016/j.mechmachtheory.2017.10.025), Sections 3–6.
  [Author-uploaded manuscript](https://www.researchgate.net/publication/320895915_Position-based_kinematics_for_7-DoF_serial_manipulators_with_global_configuration_control_joint_limit_and_singularity_avoidance).
