# KUKA iiwa 7 R800 assets

The robot geometry is derived from the `iiwa7-mujoco` model supplied from
`mujoco_iiwa7_R800-main`. Its source is the ROS `iiwa_stack` description.

- `visual/*.glb`: orange/grey split OBJ meshes combined into colored binary
  glTF files by `tools/build_iiwa7_assets.py`.
- `collision/*.stl`: collision meshes copied without geometric modification.
- `ASSET_LICENSE`: BSD 3-Clause license shipped with the source model.

To reproduce the generated assets:

```bash
python tools/build_iiwa7_assets.py /path/to/mujoco_iiwa7_R800-main
```
