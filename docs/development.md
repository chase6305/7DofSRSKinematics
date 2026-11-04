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

## Exact-configuration batch optimization

Eight S/E/W branches share only two elbow geometries. Batch requests now
decompose all matching shoulder/wrist rows together, reducing up to eight
decomposition calls to two. Shuffled input order, zero/π shoulder and wrist
singularities, periodic limits, seeds and reversed joints are covered by tests.

A local CPU comparison against the previous implementation reused the same
target, seed and configuration objects. Nine interleaved repeats of 30 warmed
calls produced these medians; output masks and joint arrays were exactly equal:

| Exact requests | Before | After | Speedup |
|:--|--:|--:|--:|
| Eight-branch gallery (8) | 0.353 ms | 0.160 ms | 2.21× |
| Self-motion sampling (584) | 0.734 ms | 0.532 ms | 1.38× |

These timings use Python 3.10 and NumPy 2.2.6, and exclude request construction,
the viewer, and WebGL. Warp's exact-configuration API also uses CPU geometry;
the figures do not measure GPU global IK throughput. Hardware and workload
affect the timings.

Run the companion benchmark from the checkout to measure scalar fixed/moving
TCP queries, the eight-branch batch, and the 584-point manifold batch. The
benchmark includes adapter/request construction and verifies TCP residuals:

```bash
python tools/benchmark_srs.py --frames 180 --repeats 15
```

## CPU global search

Global search now evaluates the branch-major arm-angle grid in the same two
elbow groups as exact configuration batches. Continuous local search keeps its
existing seed/branch priority. Global candidates retain their original order,
so zero-weight ties and all-solution ordering stay deterministic.

A local fixed-target comparison (Python 3.10, NumPy 2.2.6; nine interleaved
repeats of 30 warmed calls, seed offset 0.2 rad) measured:

| Arm-angle samples | Before | Grouped | Speedup |
|--:|--:|--:|--:|
| 73 | 0.717 ms | 0.455 ms | 1.58× |
| 361 | 0.982 ms | 0.811 ms | 1.21× |

Across 240 additional cases shared between the NumPy reference and CPU fallback,
before/after results matched, including unequal/zero weights, shoulder/wrist
singularities and all-solution requests. These are CPU timings, excluding the
viewer; finite sampling still does not guarantee finding every feasible interval.
The two repositories use different default limits, so compare before/after
within each row rather than treating the tables as a backend ranking.

Measure the current implementations with the cross-backend tool:

```bash
python tools/compare_backends.py --warp-repo ../7DofSRSKinematicsWarp --samples 40 --num-samples 73
```

## Viewer startup and branch gallery

The primary robot parses the URDF once. Gallery robots own independent joint
states and transform graphs while sharing source meshes and link-local GLB
bytes. Opening the gallery creates its eight robot instances lazily; subsequent
visits reuse them. Integration checks compare the cached GLBs byte-for-byte
with Viser's normal exports and verify independent joint updates.

A local comparison against `8174d96` reduced the first gallery switch from
489 ms to 230 ms (medians of seven interleaved trials after one warmup). It
includes IK and server scene construction, excluding initial app construction,
browser, network and WebGL. The environment is Python 3.10, Viser 1.0.24,
yourdfpy 0.0.60 and trimesh 4.12.2; hardware and load affect results. See the
[raw measurements](benchmarks/viewer.json).

Measure startup and first/repeated gallery opening separately with the viewer
extra installed:

```bash
python tools/benchmark_viewer.py --repeats 7 --output viewer-benchmark.json
```

The tool needs no browser. It checks all eight robot instances, their reuse and
TCP residuals, and writes a new JSON file with dependency versions and samples.
