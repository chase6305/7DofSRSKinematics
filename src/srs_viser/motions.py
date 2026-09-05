"""Cartesian reference paths and safe line segments for joint-space plots."""

import numpy as np

PATHS = {"figure-eight": "Figure eight", "circle": "Circle", "line": "Line"}
PLANES = {
    "world-xy": "World XY",
    "world-xz": "World XZ",
    "world-yz": "World YZ",
    "tool-xy": "TCP XY",
}
PROJECTIONS = {
    "J1 / J3 / J5": (0, 2, 4),
    "J1 / J2 / J3": (0, 1, 2),
    "J5 / J6 / J7": (4, 5, 6),
    "J2 / J4 / J6": (1, 3, 5),
}


def bounded_ik_step(current, dt, speed, solve):
    """Backtrack reference progress until the physical joint step fits the bound.

    ``solve(scale)`` must leave experiment state unchanged. Direct joint differences
    deliberately retain full revolutions: wrapping would hide discontinuities at
    position limits. The finite search can pause at very narrow feasible regions.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("joint speed must be positive and finite")
    for attempt in range(12):
        scale = 0.5**attempt
        joints = solve(scale)
        if joints is None or not np.all(np.isfinite(joints)):
            continue
        if np.max(np.abs(joints - current)) <= speed * dt + 1e-8:
            return scale, joints
    return None


class CartesianPath:
    """Closed planar paths starting at the supplied TCP, with fixed orientation."""

    def __init__(self, start, kind="figure-eight", radius=0.035, plane="world-xy"):
        if kind not in PATHS:
            raise ValueError("unknown Cartesian path")
        if plane not in PLANES:
            raise ValueError("unknown path plane")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("path radius must be positive and finite")
        self.start = np.asarray(start, dtype=float).copy()
        if self.start.shape != (4, 4) or not np.all(np.isfinite(self.start)):
            raise ValueError("path start must be a finite 4x4 pose")
        rotation = self.start[:3, :3]
        if (
            not np.allclose(self.start[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or np.linalg.det(rotation) <= 0
        ):
            raise ValueError("path start must be a rigid pose")
        axes = {
            "world-xy": (0, 1),
            "world-xz": (0, 2),
            "world-yz": (1, 2),
            "tool-xy": (0, 1),
        }[plane]
        self.basis = (rotation if plane == "tool-xy" else np.eye(3))[:, axes].copy()
        self.coordinate_labels = tuple(
            ("TCP " if plane == "tool-xy" else "") + "xyz"[axis] for axis in axes
        )
        self.start.setflags(write=False)
        self.basis.setflags(write=False)
        self.plane = plane
        self.kind, self.radius = kind, float(radius)
        self.phase = 0.0
        self.points = self.positions(np.linspace(0, 2 * np.pi, 129))
        self.points.setflags(write=False)

    def positions(self, phase):
        phase = np.asarray(phase)
        offset = np.zeros(phase.shape + (2,))
        offset[..., 0] = self.radius * np.sin(phase)
        if self.kind == "figure-eight":
            offset[..., 1] = self.radius * np.sin(2 * phase) * 0.5
        elif self.kind == "circle":
            offset[..., 1] = self.radius * (np.cos(phase) - 1)
        return self.start[:3, 3] + offset @ self.basis.T

    def project(self, positions):
        """World positions as displacement along the two reference-plane axes."""
        return (np.asarray(positions) - self.start[:3, 3]) @ self.basis

    def pose(self, phase):
        pose = self.start.copy()
        pose[:3, 3] = self.positions(phase)
        return pose


def manifold_segments(joints, valid, axes=(0, 2, 4), scale=0.22, origin=(1.5, 0, 0.75)):
    """Project adjacent feasible samples; never bridge a missing or jumping row."""
    connected = valid[:-1] & valid[1:]
    connected &= np.max(np.abs(np.diff(joints, axis=0)), axis=1) < np.pi / 2
    points = joints[:, axes] * scale + origin
    return np.stack((points[:-1][connected], points[1:][connected]), axis=1)
