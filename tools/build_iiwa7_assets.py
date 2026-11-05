#!/usr/bin/env python3
"""Build PBR R800 visuals, or restyle the bundled meshes without moving vertices."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh

ORANGE = np.array([255, 108, 10, 255], dtype=np.uint8)
GREY = np.array([102, 102, 102, 255], dtype=np.uint8)
FINISHES = {
    "Orange": ((222, 105, 32), 0.55, 0.30),
    "Silver": ((186, 192, 197), 0.65, 0.34),
}
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


def finish_mesh(mesh: trimesh.Trimesh, name: str) -> trimesh.Trimesh:
    """Reconstruct crease-aware normals and assign a linear-space PBR finish."""
    # Discard imported normals before welding: source OBJ/GLB seams can carry
    # flat or inconsistent normals even where positions are coincident.
    mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False
    )
    mesh.merge_vertices()
    mesh = trimesh.graph.smooth_shade(mesh, angle=np.deg2rad(45), facet_minarea=20)
    rgb, metallic, roughness = FINISHES[name]
    srgb = np.asarray(rgb, dtype=float) / 255
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name=name,
            baseColorFactor=[*linear, 1.0],
            metallicFactor=metallic,
            roughnessFactor=roughness,
        )
    )
    return mesh


def colored_mesh(path: Path, color: np.ndarray) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    meshes = (
        list(loaded.geometry.values())
        if isinstance(loaded, trimesh.Scene)
        else [loaded]
    )
    mesh = trimesh.util.concatenate(meshes)
    return finish_mesh(mesh, "Orange" if np.array_equal(color, ORANGE) else "Silver")


def restyle(path: Path) -> list[trimesh.Trimesh]:
    """Accept original vertex colors and previously generated named materials."""
    scene = trimesh.load(path, force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(transform)
        if mesh.visual.kind == "texture":
            name = mesh.visual.material.name
            if name not in FINISHES:
                raise ValueError(f"unknown bundled material: {name}")
            parts.append(finish_mesh(mesh, name))
        else:
            colors, groups = np.unique(
                mesh.visual.face_colors, axis=0, return_inverse=True
            )
            for group, color in enumerate(colors):
                if not any(
                    np.array_equal(color, expected) for expected in (ORANGE, GREY)
                ):
                    raise ValueError(f"unknown bundled face color: {color}")
                part = mesh.submesh([groups == group], append=True)
                name = "Orange" if np.array_equal(color, ORANGE) else "Silver"
                parts.append(finish_mesh(part, name))
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        help="iiwa7-mujoco source; omit to restyle bundled visuals",
    )
    parser.add_argument("--output", type=Path, default=Path("urdf"))
    args = parser.parse_args()
    mesh_root = args.source / "iiwa7_mjcf" / "meshes" if args.source else None
    if mesh_root is not None and (
        not (mesh_root / "visual").is_dir() or not (mesh_root / "collision").is_dir()
    ):
        raise SystemExit(f"invalid iiwa7-mujoco source: {args.source}")
    visual_output = args.output / "visual"
    collision_output = args.output / "collision"
    visual_output.mkdir(parents=True, exist_ok=True)
    collision_output.mkdir(parents=True, exist_ok=True)
    for index, parts in PARTS.items():
        output = visual_output / f"link_{index}.glb"
        meshes = (
            [colored_mesh(mesh_root / "visual" / name, color) for name, color in parts]
            if mesh_root is not None
            else restyle(output)
        )
        trimesh.Scene(meshes).export(output, file_type="glb", include_normals=True)
        if mesh_root is not None:
            shutil.copy2(
                mesh_root / "collision" / f"link_{index}.stl",
                collision_output / f"link_{index}.stl",
            )
    if args.source is not None:
        shutil.copy2(args.source / "LICENSE", args.output / "ASSET_LICENSE")


if __name__ == "__main__":
    main()
