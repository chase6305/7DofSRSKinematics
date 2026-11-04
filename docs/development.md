# Development and benchmarks

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

## Tests and references



```bash
pytest -q
```

## Algorithm and references

The closed-form decomposition locates the wrist centre, solves the shoulder-elbow-wrist triangle, rotates a reference shoulder plane by arm angle ψ, and decomposes the shoulder and wrist spherical joints. Global search reuses geometric coefficients per elbow branch, evaluates arm-angle candidates in batches, ranks valid periodic representations against the seed, and checks the returned solution's FK residual.

- Shimizu et al., “Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators With Joint Limits and Its Application to Redundancy Resolution,” IEEE T-RO, 2008, [DOI](https://doi.org/10.1109/TRO.2008.2003266).
- Faria et al., “Position-based kinematics for 7-DoF serial manipulators with global configuration control, joint limit, and singularity avoidance,” MMT, 2018, [DOI](https://doi.org/10.1016/j.mechmachtheory.2017.10.025).

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

See [urdf/README.md](../urdf/README.md) and [urdf/ASSET_LICENSE](../urdf/ASSET_LICENSE) for mesh provenance, regeneration, and the BSD-3-Clause asset license.
