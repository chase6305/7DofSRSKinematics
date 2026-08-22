#!/usr/bin/env python3
"""Build colored GLB visuals and copy collisions from iiwa7-mujoco assets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh


ORANGE = np.array([255, 108, 10, 255], dtype=np.uint8)
GREY = np.array([102, 102, 102, 255], dtype=np.uint8)
PARTS = {
    0: [("link_0.obj", GREY)],
    1: [("link_1.obj", GREY)],
    2: [("link_2_orange.obj", ORANGE), ("link_2_grey.obj", GREY)],
    3: [("link_3.obj", GREY)],
    4: [("link_4_orange.obj", ORANGE), ("link_4_grey.obj", GREY)],
    5: [("link_5.obj", GREY)],
    6: [("link_6_orange.obj", ORANGE), ("link_6_grey.obj", GREY)],
    7: [("link_7.obj", GREY)],
}


def colored_mesh(path: Path, color: np.ndarray) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    meshes = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
    mesh = trimesh.util.concatenate(meshes)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, face_colors=np.tile(color, (len(mesh.faces), 1))
    )
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("urdf"))
    args = parser.parse_args()
    mesh_root = args.source / "iiwa7_mjcf" / "meshes"
    if not (mesh_root / "visual").is_dir() or not (mesh_root / "collision").is_dir():
        raise SystemExit(f"invalid iiwa7-mujoco source: {args.source}")
    visual_output = args.output / "visual"
    collision_output = args.output / "collision"
    visual_output.mkdir(parents=True, exist_ok=True)
    collision_output.mkdir(parents=True, exist_ok=True)
    for index, parts in PARTS.items():
        meshes = [colored_mesh(mesh_root / "visual" / name, color) for name, color in parts]
        combined = trimesh.util.concatenate(meshes)
        combined.export(visual_output / f"link_{index}.glb", file_type="glb")
        shutil.copy2(
            mesh_root / "collision" / f"link_{index}.stl",
            collision_output / f"link_{index}.stl",
        )
    shutil.copy2(args.source / "LICENSE", args.output / "ASSET_LICENSE")


if __name__ == "__main__":
    main()
