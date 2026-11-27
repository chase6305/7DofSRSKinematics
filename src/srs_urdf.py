"""Recognize a 7-DoF S-R-S chain and extract analytical parameters from URDF."""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
from xml.etree import ElementTree

import numpy as np

from srs_analytical_solver import SRSAnalyticalSolver

__all__ = ["SRSAnalysis", "SRSParameters", "analyze_urdf", "scan_urdf"]


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
    elbow_axis: Optional[np.ndarray] = None
    upper_arm_length: Optional[float] = None
    forearm_length: Optional[float] = None

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
        report["joint_names"] = list(self.joint_names)
        if self.parameters is not None:
            report["parameters"]["joint_limits"] = [
                list(bounds) for bounds in self.parameters.joint_limits
            ]
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

    def compare_fk(self, q_urdf) -> Tuple[np.ndarray, np.ndarray]:
        """Compare extracted and direct URDF FK at given URDF joint coordinates.

        Return position errors in metres and rotation angles in radians. Each
        array has the leading shape of the input (..., 7). This reads the URDF
        given to analyze_urdf or scan_urdf.
        """
        solver = self.make_solver()
        joints = np.asarray(q_urdf, dtype=float)
        mapped = self.parameters.to_solver_joints(joints).reshape(-1, 7)
        tree = _URDFTree(self.urdf_path)
        chain = tree.chain(tree.base(self.base_link), self.tip_link)
        active = [joint for joint in chain if joint.get("type") != "fixed"]
        names = tuple(joint.get("name") for joint in active)
        if names != self.joint_names or any(
            not _is_independent_rotary(j) for j in active
        ):
            raise ValueError(
                "Selected URDF joint names or types changed since analysis"
            )
        frames = _joint_frames(chain, tree.version)

        _, _, reference = _chain_state(
            frames, joints.reshape(-1, 7), capture_axes=False
        )
        delta = solver.get_fk(mapped) - reference
        position = np.linalg.norm(delta[:, :3, 3], axis=1)
        distance = np.linalg.norm(delta[:, :3, :3], axis=(1, 2))
        angle = 2.0 * np.arcsin(np.clip(distance / np.sqrt(8.0), 0.0, 1.0))
        return position.reshape(joints.shape[:-1]), angle.reshape(joints.shape[:-1])


def _is_independent_rotary(joint):
    return (
        joint.get("type") in ("revolute", "continuous") and joint.find("mimic") is None
    )


class _URDFTree:
    def __init__(self, urdf_path):
        self.path = str(urdf_path)
        robot = ElementTree.parse(urdf_path).getroot()
        if robot.tag != "robot":
            raise ValueError("Expected a URDF <robot> document")
        version = robot.get("version", "1.0")
        if re.fullmatch(r"[0-9]+\.[0-9]+", version) is None:
            raise ValueError(f"Invalid URDF version {version!r}")
        self.version = tuple(map(int, version.split(".")))
        if self.version not in ((1, 0), (1, 1), (1, 2)):
            raise ValueError(f"Unsupported URDF version {version!r}")
        links = [link.get("name") for link in robot.findall("link")]
        if not links or not all(links) or len(set(links)) != len(links):
            raise ValueError("URDF must contain uniquely named links")

        self.children = {link: [] for link in links}
        self.parents = {}
        names = set()
        for joint in robot.findall("joint"):
            name = joint.get("name")
            parent = joint.find("parent")
            child = joint.find("child")
            parent = parent.get("link") if parent is not None else None
            child = child.get("link") if child is not None else None
            if not name or name in names:
                raise ValueError("URDF must contain uniquely named joints")
            if (
                parent not in self.children
                or child not in self.children
                or child in self.parents
            ):
                raise ValueError(f"Invalid parent or child link on joint {name!r}")
            names.add(name)
            self.parents[child] = (parent, joint)
            self.children[parent].append(child)

        roots = set(links) - self.parents.keys()
        if len(roots) != 1:
            raise ValueError("URDF must have one root link")
        self.root = next(iter(roots))

        stack = [self.root]
        visited = 0
        while stack:
            link = stack.pop()
            visited += 1
            stack.extend(self.children[link])
        if visited != len(links):
            raise ValueError("URDF links must form a connected tree")

    def base(self, link):
        if link is None:
            return self.root
        if link not in self.children:
            raise ValueError(f"Unknown base link {link!r}")
        return link

    def sole_leaf(self, base):
        stack = [base]
        leaves = []
        while stack:
            link = stack.pop()
            if self.children[link]:
                stack.extend(self.children[link])
            else:
                leaves.append(link)
        if len(leaves) != 1:
            raise ValueError("Multiple leaf links below the base; use --tip or --scan")
        return leaves[0]

    def chain(self, base, tip):
        if tip not in self.children:
            raise ValueError(f"Unknown tip link {tip!r}")
        chain = []
        link = tip
        while link != base:
            if link == self.root:
                raise ValueError(f"Tip link {tip!r} is outside base {base!r}")
            link, joint = self.parents[link]
            chain.append(joint)
        return chain[::-1]

    def _fixed_tips(self, link):
        stack = [link]
        while stack:
            current = stack.pop()
            children = self.children[current]
            fixed = [
                child
                for child in children
                if self.parents[child][1].get("type") == "fixed"
            ]
            if not children or len(fixed) < len(children):
                yield current
            stack.extend(fixed)

    def rotary_paths(self, base):
        # Track the base of each of the last seven contiguous rotary joints.
        # The start of a fixed segment is immediately below the previous joint.
        stack = [(base, base, ())]
        while stack:
            parent, segment_base, recent = stack.pop()
            for child in self.children[parent]:
                joint = self.parents[child][1]
                if joint.get("type") == "fixed":
                    stack.append((child, segment_base, recent))
                    continue
                rotary_bases = (
                    (*recent[-6:], segment_base)
                    if _is_independent_rotary(joint)
                    else ()
                )
                if len(rotary_bases) == 7:
                    for tip in self._fixed_tips(child):
                        yield rotary_bases[0], tip
                stack.append((child, child, rotary_bases))


def _vector(element, attribute, default, joint_name, size=3):
    value = element.get(attribute, default) if element is not None else default
    try:
        vector = np.array(value.split(), dtype=float)
    except ValueError:
        raise ValueError(f"Invalid {attribute} on joint {joint_name!r}") from None
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Invalid {attribute} on joint {joint_name!r}")
    return vector


def _unit_vector(vector, description):
    scale = np.max(np.abs(vector))
    if scale == 0.0:
        raise ValueError(f"Zero {description}")
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def _origin_rotation(origin, name, version):
    if origin is not None and "quat_xyzw" in origin.attrib:
        if version < (1, 1):
            raise ValueError(
                f"quat_xyzw on joint {name!r} requires URDF version 1.1 or newer"
            )
        if "rpy" in origin.attrib:
            raise ValueError(f"Joint {name!r} has both rpy and quat_xyzw orientations")
        q = _vector(origin, "quat_xyzw", "0 0 0 1", name, size=4)
        q = _unit_vector(q, f"quat_xyzw on joint {name!r}")
        cross = SRSAnalyticalSolver.skew(q[:3])
        return np.eye(3) + 2.0 * (q[3] * cross + cross @ cross)

    roll, pitch, yaw = _vector(origin, "rpy", "0 0 0", name)
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    # URDF fixed-axis roll, pitch, yaw gives Rz(yaw) Ry(pitch) Rx(roll).
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _joint_frames(chain, version):
    frames = []
    for joint in chain:
        name = joint.get("name")
        origin = joint.find("origin")
        transform = np.eye(4)
        transform[:3, :3] = _origin_rotation(origin, name, version)
        transform[:3, 3] = _vector(origin, "xyz", "0 0 0", name)
        axis = None
        if joint.get("type") != "fixed":
            axis_element = joint.find("axis")
            if axis_element is not None and "xyz" not in axis_element.attrib:
                raise ValueError(f"Missing xyz in axis of joint {name!r}")
            axis = _vector(axis_element, "xyz", "1 0 0", name)
            axis = _unit_vector(axis, f"rotation axis on joint {name!r}")
        frames.append((transform, axis))
    return frames


def _chain_state(frames, angles=None, *, capture_axes=True):
    """Express the tip pose and optional axis lines in the selected base frame."""
    pose = np.eye(4)
    if angles is not None:
        pose = np.broadcast_to(pose, angles.shape[:-1] + (4, 4))
    identity = np.eye(3)
    points, axes = [], []
    index = 0
    for origin, axis in frames:
        pose = pose @ origin
        if axis is None:
            continue
        if capture_axes:
            points.append(pose[..., :3, 3].copy())
            axes.append(pose[..., :3, :3] @ axis)
        if angles is not None:
            angle = angles[..., index][..., None, None]
            cosine, sine = np.cos(angle), np.sin(angle)
            rotation = (
                cosine * identity
                + (1.0 - cosine) * np.outer(axis, axis)
                + sine * SRSAnalyticalSolver.skew(axis)
            )
            pose[..., :3, :3] = pose[..., :3, :3] @ rotation
        index += 1
    if capture_axes:
        return np.array(points), np.array(axes), pose
    return None, None, pose


def _axis_center(points, axes):
    """Least-squares point on three axis lines and maximum line distance."""
    projection = np.eye(3) - axes[:, :, None] * axes[:, None, :]
    relative_points = points - points[0]
    system = projection.reshape(-1, 3)
    rhs = (projection @ relative_points[..., None]).ravel()
    offset = np.linalg.lstsq(system, rhs, rcond=None)[0]
    distances = np.linalg.norm((system @ offset - rhs).reshape(-1, 3), axis=1)
    return points[0] + offset, float(distances.max())


def _joint_limits(joints, version):
    limits = []
    for joint in joints:
        if joint.get("type") == "continuous":
            limits.append((None, None))
            continue
        name = joint.get("name")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"Missing position limits on revolute joint {name!r}")
        if version >= (1, 2) and not {"lower", "upper"} <= limit.attrib.keys():
            raise ValueError(f"Missing lower or upper limit on revolute joint {name!r}")
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


def _validate_tolerances(position_tolerance, angle_tolerance):
    if not np.isfinite(position_tolerance) or position_tolerance <= 0.0:
        raise ValueError("position_tolerance must be positive and finite")
    if not 0.0 < angle_tolerance < np.pi / 4.0:
        raise ValueError("angle_tolerance must be between 0 and pi/4 radians")


def _analyze_chain(tree, base, tip, position_tolerance, angle_tolerance):
    chain = tree.chain(base, tip)
    active = [joint for joint in chain if joint.get("type") != "fixed"]
    result = SRSAnalysis(
        tree.path, base, tip, tuple(joint.get("name") for joint in active)
    )
    if len(active) != 7:
        result.reason = f"Expected seven independent rotary joints, found {len(active)} movable joints"
        return result
    for joint in active:
        if not _is_independent_rotary(joint):
            name = joint.get("name")
            kind = "mimic" if joint.find("mimic") is not None else joint.get("type")
            result.reason = (
                f"Expected seven independent rotary joints, found {kind} joint {name!r}"
            )
            return result

    try:
        with np.errstate(over="raise", invalid="raise"):
            frames = _joint_frames(chain, tree.version)
            limits = _joint_limits(active, tree.version)
            _fit_geometry(result, frames, limits, position_tolerance, angle_tolerance)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            f"Could not compute finite joint geometry from {base!r} to {tip!r}"
        ) from exc
    return result


def _fit_geometry(result, frames, limits, position_tolerance, angle_tolerance):
    points, axes, _ = _chain_state(frames)
    angular_threshold = np.sin(angle_tolerance)
    pairs = np.array([0, 1, 4, 5])
    for name, triplet in (
        ("Shoulder", axes[:3]),
        ("Wrist", axes[4:]),
    ):
        if np.any(
            np.linalg.norm(np.cross(triplet[:-1], triplet[1:]), axis=1)
            <= angular_threshold
        ):
            result.reason = f"{name} has parallel adjacent axes"
            return

    result.shoulder_center, result.shoulder_axis_error = _axis_center(
        points[:3], axes[:3]
    )
    result.wrist_center, result.wrist_axis_error = _axis_center(points[4:], axes[4:])
    if result.shoulder_axis_error > position_tolerance:
        result.reason = "Shoulder axes do not intersect within the position tolerance"
        return
    if result.wrist_axis_error > position_tolerance:
        result.reason = "Wrist axes do not intersect within the position tolerance"
        return

    result.is_srs = True
    shoulder, wrist, elbow_axis = result.shoulder_center, result.wrist_center, axes[3]
    elbow = points[3] + ((shoulder - points[3]) @ elbow_axis) * elbow_axis
    result.elbow_center = elbow
    result.elbow_axis = elbow_axis.copy()
    upper, forearm = elbow - shoulder, wrist - elbow
    result.upper_arm_length = float(np.linalg.norm(upper))
    result.forearm_length = float(np.linalg.norm(forearm))
    if min(result.upper_arm_length, result.forearm_length) <= position_tolerance:
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
            frames, axes, shoulder, upper, forearm, limits
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
    _validate_tolerances(position_tolerance, angle_tolerance)
    tree = _URDFTree(urdf_path)
    base = tree.base(base_link)
    tip = tree.sole_leaf(base) if tip_link is None else tip_link
    return _analyze_chain(tree, base, tip, position_tolerance, angle_tolerance)


def scan_urdf(
    urdf_path: Union[str, Path],
    *,
    base_link: Optional[str] = None,
    position_tolerance: float = 1e-6,
    angle_tolerance: float = 1e-6,
) -> Tuple[SRSAnalysis, ...]:
    """Find geometrically S-R-S chains in a URDF subtree, sorted by base and tip."""
    _validate_tolerances(position_tolerance, angle_tolerance)
    tree = _URDFTree(urdf_path)
    matches = []
    for base, tip in sorted(tree.rotary_paths(tree.base(base_link))):
        result = _analyze_chain(tree, base, tip, position_tolerance, angle_tolerance)
        if result.is_srs:
            matches.append(result)
    return tuple(matches)


def _cli_report(analysis, check_fk):
    report = analysis.to_dict()
    if check_fk and analysis.solver_compatible:
        params = analysis.parameters
        bounds = np.asarray(params.joint_limits, dtype=float)
        bounds = np.where(np.isnan(bounds), [-np.pi, np.pi], bounds)
        lower, upper = bounds.T
        fractions = np.random.default_rng(0).random((32, 7))
        q = (1.0 - fractions) * lower + fractions * upper
        position, angle = analysis.compare_fk(q)
        report["fk_check"] = {
            "sample_count": len(q),
            "max_position_error": float(position.max()),
            "max_angle_error": float(angle.max()),
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path, help="path to a URDF XML file")
    parser.add_argument(
        "--base", help="base link or scan root (defaults to the URDF root)"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--tip", help="tip link (defaults to the only leaf below the base)"
    )
    selection.add_argument(
        "--scan", action="store_true", help="find S-R-S chains in a subtree"
    )
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=1e-6,
        metavar="METRES",
        help="distance tolerance for axis intersection and fit (default: %(default)s)",
    )
    parser.add_argument(
        "--angle-tolerance",
        type=float,
        default=1e-6,
        metavar="RADIANS",
        help="angle tolerance for axis geometry and fit (default: %(default)s)",
    )
    parser.add_argument(
        "--check-fk",
        action="store_true",
        help="compare URDF and extracted FK at 32 deterministic joint samples",
    )
    args = parser.parse_args(argv)
    options = {
        "base_link": args.base,
        "position_tolerance": args.position_tolerance,
        "angle_tolerance": args.angle_tolerance,
    }
    try:
        if args.scan:
            matches = scan_urdf(args.urdf, **options)
            report = {
                "chains": [_cli_report(match, args.check_fk) for match in matches]
            }
            compatible = any(match.solver_compatible for match in matches)
        else:
            result = analyze_urdf(args.urdf, tip_link=args.tip, **options)
            report = _cli_report(result, args.check_fk)
            compatible = result.solver_compatible
        output = json.dumps(report, indent=2, allow_nan=False)
    except (OSError, ElementTree.ParseError, ValueError) as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0 if compatible else 1


if __name__ == "__main__":
    sys.exit(main())
