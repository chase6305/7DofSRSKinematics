# Robot assets and visual provenance

Both repositories display **KUKA LBR iiwa 7 R800**. The analytical dimensions
are `[0.34, 0.40, 0.40, 0.126]` m: base–shoulder, upper arm, forearm and
wrist–TCP. The URDF is checked independently against FK, including its fixed
flange offset. Warp retains its public URDF filename, `joint1`–`joint7`
joint names and `base`/`link1`–`link7` link names, and now uses the same R800
geometry and frames as the NumPy reference. The fixed `link_ee` frame marks the TCP.
The old Warp display had approximate link distances and different axis signs;
applications using that display's poses should use the corrected URDF.

## The dual-arm tutorial reference

`EmbodiChain/scripts/tutorials/sim/dexlang_bi_arm_both_pick.py` resolves its
arm through `dexlang_release/assets/robots/kuka_iiwa_arm/robot_config.json`.
That file selects `urdf/iiwa14_primitive_collision.urdf`: **iiwa14**, not R800.

| Property | Tutorial model | Bundled model |
|:--|:--|:--|
| Robot | iiwa14 | iiwa 7 R800 |
| Base–shoulder | 0.360 m | 0.340 m |
| Upper arm | 0.420 m | 0.400 m |
| Forearm | 0.400 m | 0.400 m |
| Wrist–flange distance | 0.126 m | 0.126 m |
| Tool frame | `iiwa_link_ee`, with a rotated flange frame | `link_ee` |

The tutorial instantiates the arm twice. It adds either its procedural parallel
gripper/suction assembly or a Robotiq 2F85 plus adapter. The Robotiq case plans
with a 160 mm TCP including the 10 mm adapter. Those tools and the two arm base
transforms are task configuration, not part of the R800 solver.

Its `style_robot_visuals()` builds orange/silver/graphite GLBs with metallic and
roughness factors, converts display sRGB colors to linear glTF factors, and
refines display surfaces separately from collision geometry. This demo adopts
the orange/silver PBR treatment and reconstructs crease-aware normals on the
existing R800 meshes. **No iiwa14 geometry, gripper, planner or EmbodiChain
runtime is required.** Mesh vertex positions, joint origins, limits, inertia
and collision geometry in the NumPy model remain unchanged.

## Regenerate

Run from the repository root with the visualization extra installed:

```bash
python tools/build_iiwa7_assets.py
```

This restyles bundled GLBs in place. To rebuild from the original
`mujoco_iiwa7_R800-main` checkout, including collision copies and its license:

```bash
python tools/build_iiwa7_assets.py /path/to/mujoco_iiwa7_R800-main
```

Source geometry is the `iiwa7-mujoco` model derived from ROS `iiwa_stack`.
The supplied BSD-3-Clause notice is retained in
[urdf/ASSET_LICENSE](../urdf/ASSET_LICENSE). The asset and material generator
is included in both repositories; neither needs an external asset checkout at
runtime. Both repositories include the URDF's visual and collision resources.
