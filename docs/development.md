# Development and benchmarks

## Verification

```bash
python -m pytest -q
srs-viser-lab --validate-only
python -m pip wheel . --no-deps --wheel-dir dist
python tools/check_wheel.py dist/*.whl
```

Keep the Warp repository as a sibling checkout, or set `SRS_NUMPY_REPO` when
running its `test/test_backend_consistency.py`. Those checks compare FK,
Jacobians, all eight branches and global solution selection. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for development conventions.
The Python 3.8 CI job uses NumPy 1.21.6 to check the minimum supported NumPy series.

## Solver design

The analytical solution locates the wrist centre, solves the
shoulder–elbow–wrist triangle, rotates a reference shoulder plane by arm angle ψ,
and decomposes the shoulder and wrist spherical joints. Exact batches and
global search share two elbow geometries across the eight S/E/W branches.
Candidates retain their original order for deterministic seed-distance ties;
returned solutions are checked by FK. Finite arm-angle sampling can miss narrow
feasible intervals.

Batched FK and Jacobians traverse all robots together. See [API contracts](api.md)
for shapes, limits, frames and scalar behavior, and [assets](assets.md) for the
bundled R800 geometry and mesh provenance.

At 256 or more robots, DH evaluation reuses one matrix buffer instead of
materializing all seven joint transforms. Small batches keep the eager path.

## Reproduce benchmarks

```bash
python tools/benchmark_srs.py --frames 180 --repeats 15 --output benchmark.json
python tools/compare_backends.py --warp-repo ../7DofSRSKinematicsWarp --samples 40
python tools/benchmark_viewer.py --repeats 7 --output viewer-benchmark.json
```

The first command measures fixed/moving TCP queries, an eight-branch gallery
and a 584-point manifold. The second requires Warp dependencies and compares
identical targets and limits; add `--device cuda` for CUDA. Exact branch/arm-angle
IK uses CPU geometry in both repositories, so these exact-query timings do not
measure GPU global IK throughput. The viewer benchmark requires the visualization
extra but no browser. All tools warm up, report median times, check residuals
and preserve existing output files.

Recorded local comparisons (Python 3.10, NumPy 2.2.6):

| Workload | Reference | Optimized |
|:--|--:|--:|
| 4096 Jacobians | Scalar loop: 118.003 ms | Batched: 2.602 ms |
| First eight-branch gallery opening | `8174d96`: 545 ms | 254 ms |
| 4096 batched FK, CPU | `7e18cdc`: 1.727 ms / 5.28 MB | 1.272 ms / 2.03 MB |

See [Jacobian samples](benchmarks/jacobian.json) and
[viewer samples](benchmarks/viewer.json) for environments and raw trials.
[Batch-chain data](benchmarks/batch-chain.json) records before/after FK, link-pose
and Jacobian calls, including input validation and peak tracked allocations
(outputs included, decimal MB). It uses one CPU thread and checks equal outputs.
Jacobian timing includes stacking the scalar results. Gallery timing includes
IK and server scene construction, excluding initial app construction, network,
browser and WebGL; it uses Viser 1.0.24, yourdfpy 0.0.60 and trimesh 4.12.2.
Hardware and load affect all timings.

## References

- Shimizu et al., “Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators With Joint Limits and Its Application to Redundancy Resolution,” IEEE T-RO, 2008, [DOI](https://doi.org/10.1109/TRO.2008.2003266).
- Faria et al., “Position-based kinematics for 7-DoF serial manipulators with global configuration control, joint limit, and singularity avoidance,” MMT, 2018, [DOI](https://doi.org/10.1016/j.mechmachtheory.2017.10.025).
