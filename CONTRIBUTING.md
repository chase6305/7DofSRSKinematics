# Contributing

Contributions are welcome. Keep changes focused on 7-DoF S-R-S kinematics and
do not commit generated files or robot assets without redistribution rights.

## Development setup

```bash
python -m pip install -e '.[test,visualization]'
pytest -q
srs-viser-lab --validate-only
python examples/viser_demo.py --validate-only
python examples/viser_demo.py --validate-only --search-mode global
```

Please add a regression test for solver changes. FK-to-IK tests should verify
translation and rotation independently and include the relevant configuration
branch or singularity. Run `git diff --check` before submitting a change.
Jacobian changes should be checked against independent finite differences with
non-identity base and TCP transforms. Headless validation must work without
visualization dependencies; the test suite also checks FK against the URDF.

Keep the shared SRS lab behavior aligned with `7DofSRSKinematicsWarp`. With both
checkouts available, run `python tools/compare_backends.py` and the Warp test
suite to compare FK, Jacobians and IK. CI also builds the wheel, checks its URDF
mesh references, and validates the installed lab and CSV export outside the
checkout so source files cannot hide packaging errors.

Use short imperative commit subjects. Pull requests should describe API impact,
commands run, numerical tolerances, and any external robot assets required.
