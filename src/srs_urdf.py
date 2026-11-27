"""Recognize a 7-DoF S-R-S chain and extract analytical parameters from URDF."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
from xml.etree import ElementTree

import numpy as np

from srs_analytical_solver import SRSAnalyticalSolver

__all__ = ["SRSAnalysis", "SRSParameters", "analyze_urdf"]


@dataclass
class SRSParameters:
    """Canonical DH model and its mapping to the selected URDF chain."""

    link_lengths: np.ndarray
    dh_params: np.ndarray
    base_to_solver: np.ndarray
    flange_to_tip: np.ndarray
    joint_zero: np.ndarray
    joint_signs: np.ndarray
    joint_limits: Tuple[Tuple[Optional[float], Optional[float]], ...]

    @staticmethod
    def _joint_array(values):
        joints = np.asarray(values, dtype=float)
        if joints.shape[-1:] != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("Expected finite joint coordinates with shape (..., 7)")
        return joints

    def to_solver_joints(self, q_urdf):
        """Convert URDF coordinates to solver coordinates in chain order."""
        return self.joint_signs * (self._joint_array(q_urdf) - self.joint_zero)

    def to_urdf_joints(self, q_solver):
        """Convert solver coordinates back to URDF coordinates in chain order."""
        return self.joint_zero + self.joint_signs * self._joint_array(q_solver)


@dataclass
class SRSAnalysis:
    """Geometry and solver compatibility of the selected URDF chain."""

    urdf_path: str
    base_link: str
    tip_link: str
    joint_names: Tuple[str, ...]
    is_srs: bool = False
    reason: Optional[str] = None
    shoulder_center: Optional[np.ndarray] = None
    elbow_center: Optional[np.ndarray] = None
    wrist_center: Optional[np.ndarray] = None
    shoulder_axis_error: Optional[float] = None
    wrist_axis_error: Optional[float] = None
    parameters: Optional[SRSParameters] = None
    joint_axis_position_errors: Optional[np.ndarray] = None
    joint_axis_angle_errors: Optional[np.ndarray] = None

    @property
    def solver_compatible(self):
        return self.parameters is not None

    def to_dict(self) -> dict:
        """Return a JSON-serializable report with independent data."""
        report = asdict(
            self,
            dict_factory=lambda fields: {
                name: value.tolist() if isinstance(value, np.ndarray) else value
                for name, value in fields
            },
        )
        report["solver_compatible"] = self.solver_compatible
        return report

    def make_solver(self) -> SRSAnalyticalSolver:
        """Build a solver whose poses are relative to the selected URDF base."""
        params = self.parameters
        if params is None:
            raise ValueError(self.reason or "No analytical parameters were extracted")
        solver = SRSAnalyticalSolver(self.urdf_path, self.tip_link)
        solver.dh_params = params.dh_params.copy()
        solver.link_lengths = params.link_lengths.copy()
        solver.world_to_base = params.base_to_solver.copy()
        solver.set_tcp(params.flange_to_tip)
        lower, upper = np.asarray(params.joint_limits, dtype=float).T
        lower = np.where(np.isnan(lower), -np.inf, lower)
        upper = np.where(np.isnan(upper), np.inf, upper)
        first, last = params.joint_signs * (
            np.stack([lower, upper]) - params.joint_zero
        )
        solver.lower_position_limits = np.minimum(first, last)
        solver.upper_position_limits = np.maximum(first, last)
        return solver


def _select_chain(robot, base_link, tip_link):
    links = [link.get("name") for link in robot.findall("link")]
    if not links or None in links or "" in links or len(set(links)) != len(links):
        raise ValueError("URDF must contain uniquely named links")

    children = {link: [] for link in links}
    parents = {}
    names = set()
    for joint in robot.findall("joint"):
        name = joint.get("name")
        parent = joint.find("parent")
        child = joint.find("child")
        parent = parent.get("link") if parent is not None else None
        child = child.get("link") if child is not None else None
        if not name or name in names:
            raise ValueError("URDF must contain uniquely named joints")
        if parent not in children or child not in children or child in parents:
            raise ValueError(f"Invalid parent or child link on joint {name!r}")
        names.add(name)
        parents[child] = (parent, joint)
        children[parent].append(child)

    roots = set(links) - parents.keys()
    if len(roots) != 1:
        raise ValueError("URDF must have one root link")
    root = next(iter(roots))
    if base_link is None:
        base_link = root
    if base_link not in children:
        raise ValueError(f"Unknown base link {base_link!r}")

    # Traverse the URDF tree independently of the order of its XML elements.
    stack = [root]
    visited = set()
    while stack:
        link = stack.pop()
        visited.add(link)
        stack.extend(children[link])
    if len(visited) != len(links):
        raise ValueError("URDF links must form a connected tree")

    if tip_link is None:
        stack = [base_link]
        leaves = []
        while stack:
            link = stack.pop()
            if children[link]:
                stack.extend(children[link])
            else:
                leaves.append(link)
        if len(leaves) != 1:
            raise ValueError("Multiple leaf links below the base; specify a tip link")
        tip_link = leaves[0]
    if tip_link not in children:
        raise ValueError(f"Unknown tip link {tip_link!r}")

    chain = []
    link = tip_link
    while link != base_link:
        if link == root:
            raise ValueError(f"Tip link {tip_link!r} is outside base {base_link!r}")
        link, joint = parents[link]
        chain.append(joint)
    return base_link, tip_link, chain[::-1]


def _vector(element, attribute, default, joint_name):
    value = element.get(attribute, default) if element is not None else default
    try:
        vector = np.array(value.split(), dtype=float)
    except ValueError:
        raise ValueError(f"Invalid {attribute} on joint {joint_name!r}") from None
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Invalid {attribute} on joint {joint_name!r}")
    return vector


def _joint_frames(chain):
    frames = []
    for joint in chain:
        name = joint.get("name")
        origin = joint.find("origin")
        roll, pitch, yaw = _vector(origin, "rpy", "0 0 0", name)
        cr, cp, cy = np.cos([roll, pitch, yaw])
        sr, sp, sy = np.sin([roll, pitch, yaw])
        transform = np.eye(4)
        # URDF fixed-axis roll, pitch, yaw gives Rz(yaw) Ry(pitch) Rx(roll).
        transform[:3, :3] = [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
        transform[:3, 3] = _vector(origin, "xyz", "0 0 0", name)
        axis = None
        if joint.get("type") != "fixed":
            axis = _vector(joint.find("axis"), "xyz", "1 0 0", name)
            scale = np.max(np.abs(axis))
            if scale == 0.0:
                raise ValueError(f"Zero rotation axis on joint {name!r}")
            axis /= scale
            axis /= np.linalg.norm(axis)
        frames.append((transform, axis))
    return frames


def _chain_state(frames, angles):
    """Express the axis lines and tip pose in the selected base frame."""
    pose = np.eye(4)
    points, axes = [], []
    angles = iter(angles)
    for origin, axis in frames:
        pose = pose @ origin
        if axis is None:
            continue
        points.append(pose[:3, 3].copy())
        axes.append(pose[:3, :3] @ axis)
        angle = next(angles)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = (
            cosine * np.eye(3)
            + (1.0 - cosine) * np.outer(axis, axis)
            + sine * SRSAnalyticalSolver.skew(axis)
        )
        pose[:3, :3] = pose[:3, :3] @ rotation
    return np.array(points), np.array(axes), pose


def _axis_center(points, axes):
    """Least-squares point on three axis lines and maximum line distance."""
    projection = np.eye(3) - axes[:, :, None] * axes[:, None, :]
    relative_points = points - points[0]
    system = projection.reshape(-1, 3)
    rhs = (projection @ relative_points[..., None]).ravel()
    offset = np.linalg.lstsq(system, rhs, rcond=None)[0]
    distances = np.linalg.norm((system @ offset - rhs).reshape(-1, 3), axis=1)
    return points[0] + offset, float(distances.max())


def _joint_limits(joints):
    limits = []
    for joint in joints:
        if joint.get("type") == "continuous":
            limits.append((None, None))
            continue
        name = joint.get("name")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"Missing position limits on revolute joint {name!r}")
        try:
            lower, upper = (float(limit.get(key, "0")) for key in ("lower", "upper"))
        except ValueError:
            raise ValueError(f"Invalid position limits on joint {name!r}") from None
        if not np.all(np.isfinite([lower, upper])) or lower > upper:
            raise ValueError(f"Invalid position limits on joint {name!r}")
        limits.append((lower, upper))
    return tuple(limits)


def _angle_about(axis, source, target, allow_opposite=False):
    sine = axis @ np.cross(source, target)
    cosine = source @ target - (axis @ source) * (axis @ target)
    if allow_opposite and cosine < 0.0:
        sine, cosine = -sine, -cosine
    return np.arctan2(sine, cosine)


def _canonical_parameters(frames, axes, shoulder, upper, forearm, limits):
    # At canonical zero the arm is straight and all odd axes point along
    # the arm. The even axes point along +Y, -Y and +Y, respectively.
    # Compute each offset in the URDF home frame. Upstream rotations
    # transform both directions equally and cancel in their relative angle.
    a1, a2, a3, a4, a5, a6, a7 = axes
    upper_axis = a3 if a3 @ upper > 0.0 else -a3
    forearm_axis = a5 if a5 @ forearm > 0.0 else -a5
    zero = np.array(
        [
            0.0,
            _angle_about(a2, upper_axis, a1, allow_opposite=True),
            _angle_about(a3, a4, -a2, allow_opposite=True),
            _angle_about(a4, forearm_axis, upper_axis),
            _angle_about(a5, a6, -a4, allow_opposite=True),
            _angle_about(a6, a7, forearm_axis, allow_opposite=True),
            0.0,
        ]
    )
    points, reference_axes, tip_pose = _chain_state(frames, zero)
    wrist, _ = _axis_center(points[4:], reference_axes[4:])
    z = a1 if a1 @ (wrist - shoulder) > 0.0 else -a1
    x = np.cross(a2, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    canonical_axes = np.stack([z, y, z, -y, z, y, z])
    signs = np.where(np.sum(reference_axes * canonical_axes, axis=1) < 0.0, -1, 1)

    link_lengths = np.array(
        [
            z @ shoulder,
            upper_axis @ upper,
            forearm_axis @ forearm,
            z @ (tip_pose[:3, 3] - wrist),
        ]
    )
    dh = np.zeros((7, 4))
    dh[::2, 0] = link_lengths
    dh[:, 1] = (np.pi / 2.0) * np.array([-1, 1, 1, -1, -1, 1, 0])

    base = np.eye(4)
    base[:3, :3] = np.column_stack((x, y, z))
    base[:3, 3] = points[0] - z * (z @ points[0])
    flange = base.copy()
    flange[:3, 3] += z * link_lengths.sum()
    params = SRSParameters(
        link_lengths, dh, base, np.linalg.solve(flange, tip_pose), zero, signs, limits
    )

    # A rotary joint is determined by its axis line. Compare the reconstructed
    # lines at solver zero while the fixed tip transform matches that pose.
    distances = np.repeat(np.cumsum(link_lengths[:3]), (3, 1, 3))
    centers = base[:3, 3] + distances[:, None] * z
    position_errors = np.linalg.norm(np.cross(centers - points, reference_axes), axis=1)
    sine = np.linalg.norm(np.cross(canonical_axes, reference_axes), axis=1)
    cosine = np.abs(np.sum(canonical_axes * reference_axes, axis=1))
    return params, position_errors, np.arctan2(sine, cosine)


def analyze_urdf(
    urdf_path: Union[str, Path],
    *,
    base_link: Optional[str] = None,
    tip_link: Optional[str] = None,
    position_tolerance: float = 1e-6,
    angle_tolerance: float = 1e-6,
) -> SRSAnalysis:
    """Analyze the root-to-tip path, using the sole leaf when the tip is omitted.

    The tolerances are in metres and radians. Joint axes and S/E/W centers
    are evaluated at URDF joint zero. Invalid input raises an exception;
    geometric rejection is recorded in the returned analysis.
    """
    if not np.isfinite(position_tolerance) or position_tolerance <= 0.0:
        raise ValueError("position_tolerance must be positive and finite")
    if not 0.0 < angle_tolerance < np.pi / 4.0:
        raise ValueError("angle_tolerance must be between 0 and pi/4 radians")

    robot = ElementTree.parse(urdf_path).getroot()
    if robot.tag != "robot":
        raise ValueError("Expected a URDF <robot> document")
    base, tip, chain = _select_chain(robot, base_link, tip_link)
    active = [joint for joint in chain if joint.get("type") != "fixed"]
    result = SRSAnalysis(
        str(urdf_path), base, tip, tuple(joint.get("name") for joint in active)
    )
    if len(active) != 7 or any(
        joint.get("type") not in ("revolute", "continuous")
        or joint.find("mimic") is not None
        for joint in active
    ):
        result.reason = "Expected seven independent rotary joints on the selected path"
        return result

    frames = _joint_frames(chain)
    points, axes, _ = _chain_state(frames, np.zeros(7))
    result.shoulder_center, result.shoulder_axis_error = _axis_center(
        points[:3], axes[:3]
    )
    result.wrist_center, result.wrist_axis_error = _axis_center(points[4:], axes[4:])
    angular_threshold = np.sin(angle_tolerance)
    pairs = np.array([0, 1, 4, 5])
    for name, error, triplet in (
        ("Shoulder", result.shoulder_axis_error, axes[:3]),
        ("Wrist", result.wrist_axis_error, axes[4:]),
    ):
        if error > position_tolerance:
            result.reason = (
                f"{name} axes do not intersect within the position tolerance"
            )
            return result
        if np.any(
            np.linalg.norm(np.cross(triplet[:-1], triplet[1:]), axis=1)
            <= angular_threshold
        ):
            result.reason = f"{name} has parallel adjacent axes"
            return result

    result.is_srs = True
    shoulder, wrist, elbow_axis = result.shoulder_center, result.wrist_center, axes[3]
    elbow = points[3] + ((shoulder - points[3]) @ elbow_axis) * elbow_axis
    result.elbow_center = elbow
    upper, forearm = elbow - shoulder, wrist - elbow
    if min(np.linalg.norm(upper), np.linalg.norm(forearm)) <= position_tolerance:
        result.reason = "The upper arm and forearm must have nonzero lengths"
    elif np.any(
        np.abs(np.sum(axes[pairs] * axes[pairs + 1], axis=1)) > angular_threshold
    ):
        result.reason = (
            "The solver requires orthogonal adjacent shoulder and wrist axes"
        )
    elif np.any(np.abs(axes[[2, 4]] @ elbow_axis) > angular_threshold):
        result.reason = "The solver requires J4 perpendicular to J3 and J5"
    elif (
        max(
            np.linalg.norm(np.cross(upper, axes[2])),
            np.linalg.norm(np.cross(forearm, axes[4])),
        )
        > position_tolerance
    ):
        result.reason = "The solver requires the upper arm and forearm along J3 and J5"
    else:
        params, position_errors, angle_errors = _canonical_parameters(
            frames, axes, shoulder, upper, forearm, _joint_limits(active)
        )
        result.joint_axis_position_errors = position_errors
        result.joint_axis_angle_errors = angle_errors
        worst_position = int(np.argmax(position_errors))
        worst_angle = int(np.argmax(angle_errors))
        if position_errors[worst_position] > position_tolerance:
            name = result.joint_names[worst_position]
            result.reason = f"Canonical axis of {name!r} exceeds the position tolerance"
        elif angle_errors[worst_angle] > angle_tolerance:
            name = result.joint_names[worst_angle]
            result.reason = f"Canonical axis of {name!r} exceeds the angle tolerance"
        else:
            result.parameters = params
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("--base", help="base link (defaults to the URDF root)")
    parser.add_argument(
        "--tip", help="tip link (defaults to the only leaf below the base)"
    )
    parser.add_argument(
        "--position-tolerance", type=float, default=1e-6, metavar="METRES"
    )
    parser.add_argument(
        "--angle-tolerance", type=float, default=1e-6, metavar="RADIANS"
    )
    args = parser.parse_args(argv)
    try:
        result = analyze_urdf(
            args.urdf,
            base_link=args.base,
            tip_link=args.tip,
            position_tolerance=args.position_tolerance,
            angle_tolerance=args.angle_tolerance,
        )
    except (OSError, ElementTree.ParseError, ValueError) as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
    return 0 if result.solver_compatible else 1


if __name__ == "__main__":
    sys.exit(main())
