# Contributing

Contributions are welcome. Keep changes focused on 7-DoF S-R-S kinematics and
do not commit generated files or robot assets without redistribution rights.

## Development setup

```bash
python -m pip install -e '.[test,visualization]'
pytest -q
python examples/viser_demo.py --validate-only
```

Please add a regression test for solver changes. FK-to-IK tests should verify
translation and rotation independently and include the relevant configuration
branch or singularity. Run `git diff --check` before submitting a change.

Use short imperative commit subjects. Pull requests should describe API impact,
commands run, numerical tolerances, and any external robot assets required.
