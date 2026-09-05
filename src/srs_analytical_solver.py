import logging
import typing
from abc import ABC
from dataclasses import dataclass

import numpy as np

from solver import Solver

logger = logging.getLogger(__name__)

__all__ = ["SRSConfiguration", "SRSAnalyticalSolver"]


@dataclass(frozen=True)
class SRSConfiguration:
    """One shoulder/elbow/wrist branch and its arm-angle redundancy."""

    shoulder: int
    elbow: int
    wrist: int
    redundancy: float

    def __post_init__(self):
        for name in ("shoulder", "elbow", "wrist"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value not in (-1, 1)
            ):
                raise ValueError(f"{name} must be -1 or 1")
        if not isinstance(
            self.redundancy, (int, float, np.integer, np.floating)
        ) or not np.isfinite(self.redundancy):
            raise ValueError("redundancy must be finite")


class SRSAnalyticalSolver(Solver, ABC):
    def __init__(
        self,
        urdf_path: str,
        end_link_name: str,
        **kwargs,
    ):
        """Initialize the NumPy analytical solver for a canonical S-R-S chain."""
        self.world_to_base = np.eye(4)
        self.flange_to_ee = np.eye(4)

        super().__init__(
            urdf_path=urdf_path,
            end_link_name=end_link_name,
            **kwargs,
        )

        self.dof = 7

    def get_dof(self) -> int:
        r"""Returns the degree of freedom (DOF) of the robot.

        Returns:
            int: The degree of freedom of the robot.
        """
        return self.dof

    def get_urdf_path(self) -> str:
        r"""Returns the file path to the URDF (Unified Robot Description Format) file.

        Returns:
            str: The file path to the URDF file.
        """
        return self.urdf_path

    @staticmethod
    def skew(vector: np.ndarray) -> np.ndarray:
        r"""Compute the skew-symmetric matrix of a vector.

        Args:
            vector (np.ndarray): A 3D vector.

        Returns:
            np.ndarray: The skew-symmetric matrix of the input vector.
        """
        return np.array(
            [
                [0, -vector[2], vector[1]],
                [vector[2], 0, -vector[0]],
                [-vector[1], vector[0], 0],
            ]
        )

    def _dh_calc(self, d, alpha, a, theta) -> np.ndarray:
        """Build standard DH transforms, broadcasting over joint arrays."""
        d, alpha, a, theta = np.broadcast_arrays(d, alpha, a, theta)
        ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        transforms = np.zeros(theta.shape + (4, 4))
        transforms[..., 0, 0] = ct
        transforms[..., 0, 1] = -st * ca
        transforms[..., 0, 2] = st * sa
        transforms[..., 0, 3] = a * ct
        transforms[..., 1, 0] = st
        transforms[..., 1, 1] = ct * ca
        transforms[..., 1, 2] = -ct * sa
        transforms[..., 1, 3] = a * st
        transforms[..., 2, 1] = sa
        transforms[..., 2, 2] = ca
        transforms[..., 2, 3] = d
        transforms[..., 3, 3] = 1.0
        return transforms

    def _joint_transforms(self, qpos, stop=7):
        d, alpha, a, theta = self.dh_params[:stop].T
        return self._dh_calc(d, alpha, a, theta + qpos[..., :stop])

    def _fk_chain(self, qpos, world=True):
        transform = self.world_to_base.copy() if world else np.eye(4)
        poses = []
        for joint_transform in self._joint_transforms(qpos):
            transform = transform @ joint_transform
            poses.append(transform)
        return poses

    def set_tcp(self, xpos: np.ndarray):
        """Set the flange-to-TCP rigid transform used by FK, IK and Jacobians."""
        super().set_tcp(xpos)
        self.flange_to_ee = self.tcp_xpos.copy()

    def get_tcp(self) -> np.ndarray:
        return self.flange_to_ee.copy()

    def _configuration(self, rconf: int) -> tuple:
        r"""Determine the configuration of the arm, elbow, and wrist based on a configuration integer.

        Args:
            rconf (int): The configuration integer.

        Returns:
            tuple: A tuple containing the configurations of the arm, elbow, and wrist.
        """
        arm_config = -1 if rconf & 1 else 1
        elbow_config = -1 if rconf & 2 else 1
        wrist_config = -1 if rconf & 4 else 1
        return arm_config, elbow_config, wrist_config

    @staticmethod
    def _configuration_index(shoulder: int, elbow: int, wrist: int) -> int:
        return (shoulder < 0) | ((elbow < 0) << 1) | ((wrist < 0) << 2)

    def get_configuration(self, qpos: np.ndarray) -> SRSConfiguration:
        """Return the discrete S/E/W branch and continuous arm angle."""
        qpos = self._validate_joints(qpos)
        # Branches describe physical angles, independent of full revolutions.
        angles = (qpos + self.dh_params[:, 3] + np.pi) % (2.0 * np.pi) - np.pi
        signs = np.where(angles[[1, 3, 5]] < 0.0, -1, 1)
        return SRSConfiguration(*signs, self.get_arm_angle(qpos))

    def get_arm_angle(self, qpos: np.ndarray) -> float:
        """Measure the elbow-plane rotation about the shoulder-wrist axis."""
        qpos = self._validate_joints(qpos)
        poses = self._fk_chain(qpos, world=False)
        target = poses[-1]
        elbow_angle = (qpos[3] + self.dh_params[3, 3] + np.pi) % (2 * np.pi) - np.pi
        elbow = -1 if elbow_angle < 0.0 else 1
        reference_rotation, _ = self._reference_geometry(target, elbow)
        if reference_rotation is None:
            raise ValueError("arm angle is undefined at this configuration")
        actual_rotation = poses[2][:3, :3]
        wrist = target[:3, 3] - target[:3, 2] * self.dh_params[-1, 0]
        axis = wrist - np.array([0.0, 0.0, self.link_lengths[0]])
        axis /= np.linalg.norm(axis)
        projections = reference_rotation - np.outer(axis, axis @ reference_rotation)
        norms = np.linalg.norm(projections, axis=0)
        column = int(np.argmax(norms))
        reference = projections[:, column] / norms[column]
        actual = actual_rotation[:, column]
        actual = actual - axis * axis.dot(actual)
        actual /= np.linalg.norm(actual)
        # Scalar triple product avoids np.cross's broadcasting machinery.
        signed_sine = (
            axis[0] * (reference[1] * actual[2] - reference[2] * actual[1])
            + axis[1] * (reference[2] * actual[0] - reference[0] * actual[2])
            + axis[2] * (reference[0] * actual[1] - reference[1] * actual[0])
        )
        return float(np.arctan2(signed_sine, reference.dot(actual)))

    def solve_configuration(
        self,
        target_pose: np.ndarray,
        configuration: SRSConfiguration,
        joint_seed: np.ndarray,
    ) -> typing.Tuple[bool, typing.Optional[np.ndarray]]:
        """Solve one explicitly selected S/E/W branch at an exact arm angle."""
        target_pose = self._validate_pose(target_pose)
        joint_seed = self._validate_joints(joint_seed, "joint_seed")
        if not isinstance(configuration, SRSConfiguration):
            raise TypeError("configuration must be an SRSConfiguration")
        rconf = self._configuration_index(
            configuration.shoulder, configuration.elbow, configuration.wrist
        )
        if self._is_elbow_up and configuration.elbow < 0:
            return False, None
        ok, joints = self._compute_inverse_kinematics(
            target_pose, joint_seed, configuration.redundancy, rconf
        )
        if not ok or not self._matches_target(joints, target_pose):
            return False, None
        return True, joints

    def _matches_target(self, joints, target_pose):
        return bool(self._poses_match(self.get_fk(joints), target_pose))

    def _poses_match(self, actual, target_pose):
        position_error = np.linalg.norm(
            actual[..., :3, 3] - target_pose[:3, 3], axis=-1
        )
        # Frobenius distance is 2*sqrt(2)*sin(angle/2), stable near zero.
        rotation_error = 2.0 * np.arcsin(
            np.clip(
                np.linalg.norm(actual[..., :3, :3] - target_pose[:3, :3], axis=(-2, -1))
                / np.sqrt(8.0),
                0.0,
                1.0,
            )
        )
        return (position_error <= self._pos_eps) & (rotation_error <= self._rot_eps)

    def solve_configurations(self, target_pose, configurations, joint_seed):
        """Solve exact configurations for one target and shared seed, preserving order.

        Returns a boolean (N,) mask and (N, 7) joints. Invalid rows are zero.
        Geometry is shared by elbow branch; arm angles and FK checks are batched.
        An empty iterable returns empty arrays. Scalar solve_configuration is unchanged.
        """
        target = self._validate_pose(target_pose)
        seed = self._validate_joints(joint_seed, "joint_seed")
        configurations = tuple(configurations)
        if any(not isinstance(c, SRSConfiguration) for c in configurations):
            raise TypeError("configurations must contain SRSConfiguration values")
        output = np.zeros((len(configurations), 7))
        success = np.zeros(len(configurations), dtype=bool)
        if not configurations:
            return success, output
        branches = np.array(
            [
                self._configuration_index(c.shoulder, c.elbow, c.wrist)
                for c in configurations
            ]
        )
        angles = np.array([c.redundancy for c in configurations])
        local = self._local_target(target)
        geometry = {}
        for branch in np.unique(branches):
            elbow = self._configuration(branch)[1]
            if self._is_elbow_up and elbow < 0:
                continue
            if elbow not in geometry:
                geometry[elbow] = self._prepare_ik(local, elbow)
            if geometry[elbow] is None:
                continue
            rows = np.flatnonzero(branches == branch)
            candidates = self._candidate_joint_angles(
                geometry[elbow], angles[rows], branch, seed
            )
            candidates, valid = self._map_to_limits(candidates, seed)
            output[rows[valid]] = candidates[valid]
            success[rows] = valid
        rows = np.flatnonzero(success)
        if len(rows):
            success[rows] = self._poses_match(self.get_fk(output[rows]), target)
        output[~success] = 0
        return success, output

    def get_jacobian(self, qpos: np.ndarray, step: float = 1e-7) -> np.ndarray:
        """Return the analytical 6x7 geometric Jacobian in the world frame.

        Rows are linear then angular velocity at the TCP. ``step`` is retained
        for API compatibility and validated, but no finite differences are used.
        """
        qpos = self._validate_joints(qpos)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step must be positive and finite")
        poses = self._fk_chain(qpos)
        tcp = (poses[-1] @ self.flange_to_ee)[:3, 3]
        parents = np.stack([self.world_to_base] + poses[:-1])
        axes = parents[:, :3, 2]
        linear = np.cross(axes, tcp - parents[:, :3, 3])
        return np.concatenate((linear.T, axes.T), axis=0)

    def get_null_space_direction(
        self, qpos: np.ndarray, arm_angle_step: float = 1e-4
    ) -> np.ndarray:
        """Return dq/dpsi on the current branch while keeping the TCP fixed.

        A central arm-angle difference is preferred. Near a joint limit the
        method falls back to a one-sided difference, and at an undefined arm
        plane it returns the SVD Jacobian null direction.
        """
        qpos = self._validate_joints(qpos)
        if not np.isfinite(arm_angle_step) or arm_angle_step <= 0.0:
            raise ValueError("arm_angle_step must be positive and finite")
        target = self.get_fk(qpos)
        try:
            configuration = self.get_configuration(qpos)
        except (ValueError, FloatingPointError):
            return self._svd_null_direction(qpos)

        def solve(offset):
            shifted = SRSConfiguration(
                configuration.shoulder,
                configuration.elbow,
                configuration.wrist,
                configuration.redundancy + offset,
            )
            return self.solve_configuration(target, shifted, qpos)

        plus_ok, plus = solve(arm_angle_step)
        minus_ok, minus = solve(-arm_angle_step)
        if plus_ok and minus_ok:
            difference = self._wrapped_difference(plus, minus)
            direction = difference / (2.0 * arm_angle_step)
        elif plus_ok:
            direction = self._wrapped_difference(plus, qpos) / arm_angle_step
        elif minus_ok:
            direction = self._wrapped_difference(qpos, minus) / arm_angle_step
        else:
            return self._svd_null_direction(qpos)
        if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) < 1e-10:
            return self._svd_null_direction(qpos)
        return direction

    def get_null_space_velocity(
        self, qpos: np.ndarray, preferred_velocity: np.ndarray
    ) -> np.ndarray:
        """Project a preferred joint velocity into the Jacobian null space."""
        preferred_velocity = np.asarray(preferred_velocity, dtype=float)
        if preferred_velocity.shape != (7,) or not np.all(
            np.isfinite(preferred_velocity)
        ):
            raise ValueError("preferred_velocity must be a finite array of shape (7,)")
        jacobian = self.get_jacobian(qpos)
        return preferred_velocity - np.linalg.pinv(jacobian) @ (
            jacobian @ preferred_velocity
        )

    @staticmethod
    def _wrapped_difference(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        difference = np.asarray(lhs) - np.asarray(rhs)
        return np.arctan2(np.sin(difference), np.cos(difference))

    def _svd_null_direction(self, qpos: np.ndarray) -> np.ndarray:
        _, _, vh = np.linalg.svd(self.get_jacobian(qpos), full_matrices=True)
        direction = vh[-1]
        # Fix the otherwise arbitrary SVD sign for deterministic output.
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            direction = -direction
        return direction / np.linalg.norm(direction)

    def _calculate_joint_angles(
        self, P_s_to_w: np.ndarray, elbow_GC4: int
    ) -> typing.Tuple[bool, typing.Optional[np.ndarray]]:
        r"""Calculate joint angles based on the position from shoulder to wrist and elbow configuration.

        Args:
            P_s_to_w (np.ndarray): The position vector from shoulder to wrist.
            elbow_GC4 (int): The elbow configuration parameter.

        Returns:
            np.ndarray: A boolean indicating success and an array of joint angles.
        """
        d_se, d_ew = self.link_lengths[1:3]
        joints = np.zeros(7)

        # Check reachability and calculate elbow joint angle
        norm_P26 = np.linalg.norm(P_s_to_w)
        if norm_P26 < 1e-12:
            return False, None
        if not (
            d_se + d_ew + 1e-10 >= norm_P26 and norm_P26 + 1e-10 >= abs(d_se - d_ew)
        ):
            logger.debug("Specified pose outside reachable workspace.")
            return False, None

        elbow_cos_angle = (norm_P26**2 - d_se**2 - d_ew**2) / (2 * d_se * d_ew)

        joints[3] = elbow_GC4 * np.arccos(np.clip(elbow_cos_angle, -1.0, 1.0))

        # Calculate joint 1
        if np.hypot(P_s_to_w[0], P_s_to_w[1]) > 1e-10:
            joints[0] = np.arctan2(P_s_to_w[1], P_s_to_w[0])
        else:
            joints[0] = 0

        # Calculate joint 2
        euclidean_norm = np.hypot(P_s_to_w[0], P_s_to_w[1])
        angle_phi = np.arccos(
            np.clip(
                (d_se**2 + norm_P26**2 - d_ew**2) / (2 * d_se * norm_P26), -1.0, 1.0
            )
        )
        joints[1] = np.arctan2(euclidean_norm, P_s_to_w[2]) + elbow_GC4 * angle_phi

        return True, joints

    def _reference_geometry(self, pose, elbow):
        """Return the reference shoulder rotation and physical joint angles."""
        wrist = pose[:3, 3] - pose[:3, 2] * self.dh_params[-1, 0]
        shoulder = np.array([0.0, 0.0, self.link_lengths[0]])
        ok, joints = self._calculate_joint_angles(wrist - shoulder, elbow)
        if not ok:
            return None, None
        transforms = self._joint_transforms(joints - self.dh_params[:, 3], stop=3)
        rotation = transforms[0, :3, :3] @ transforms[1, :3, :3] @ transforms[2, :3, :3]
        return rotation, joints

    def _reference_plane(self, pose: np.ndarray, elbow_GC4: int) -> tuple:
        """Return the reference plane normal, shoulder rotation and joint angles."""
        rotation, joints = self._reference_geometry(pose, elbow_GC4)
        if rotation is None:
            return False, None, None, None
        transforms = self._joint_transforms(joints - self.dh_params[:, 3], stop=3)
        elbow = (transforms[0] @ transforms[1] @ transforms[2])[:3, 3]
        shoulder = np.array([0.0, 0.0, self.link_lengths[0]])
        wrist = pose[:3, 3] - pose[:3, 2] * self.dh_params[-1, 0]
        return True, np.cross(elbow - shoulder, wrist - shoulder), rotation, joints

    def _local_target(self, target_pose):
        return np.linalg.solve(self.world_to_base, target_pose) @ np.linalg.inv(
            self.flange_to_ee
        )

    def _prepare_ik(self, local_target, elbow):
        """Reuse geometry for the last local target, with at most two elbow branches.

        Content keys detect in-place target or calibration edits. Joint seeds,
        limits and weights are deliberately excluded: candidate decomposition and
        validation still use their current values on every call.
        """
        arrays = (
            np.asarray(value)
            for value in (local_target, self.dh_params, self.link_lengths)
        )
        key = tuple((value.shape, value.dtype.str, value.tobytes()) for value in arrays)
        cached = getattr(self, "_ik_geometry_cache", None)
        geometry = {} if cached is None or cached[0] != key else cached[1]
        if elbow not in geometry:
            prepared = self._build_ik_geometry(local_target, elbow)
            if prepared is not None:
                for matrix in (*prepared[0], *prepared[1]):
                    matrix.setflags(write=False)
            geometry[elbow] = prepared
        self._ik_geometry_cache = (key, geometry)
        return geometry[elbow]

    def _build_ik_geometry(self, local_target, elbow):
        """Compute seed-independent shoulder and wrist coefficients."""
        reference, joints = self._reference_geometry(local_target, elbow)
        if reference is None:
            return None
        axis = local_target[:3, 3] - local_target[:3, 2] * self.dh_params[-1, 0]
        axis = axis - np.array([0.0, 0.0, self.link_lengths[0]])
        axis /= np.linalg.norm(axis)
        skew = self.skew(axis)
        shoulder = (
            skew @ reference,
            -skew @ skew @ reference,
            np.outer(axis, axis) @ reference,
        )
        d, alpha, a, _ = self.dh_params[3]
        elbow_rotation = self._dh_calc(d, alpha, a, joints[3])[:3, :3]
        wrist = tuple(
            elbow_rotation.T @ matrix.T @ local_target[:3, :3] for matrix in shoulder
        )
        return shoulder, wrist, joints[3]

    def _candidate_joint_angles(self, prepared, nsparams, rconf, joint_seed):
        """Decompose one branch for a scalar or a vector of arm angles."""
        shoulder, wrist, elbow_angle = prepared
        arm_config, _, wrist_config = self._configuration(rconf)
        psi = np.asarray(nsparams)
        sine, cosine = np.sin(psi)[..., None, None], np.cos(psi)[..., None, None]
        r03 = shoulder[0] * sine + shoulder[1] * cosine + shoulder[2]
        r47 = wrist[0] * sine + wrist[1] * cosine + wrist[2]
        joints = np.empty(psi.shape + (7,))
        joints[..., 0] = np.arctan2(
            r03[..., 1, 1] * arm_config, r03[..., 0, 1] * arm_config
        )
        joints[..., 1] = (
            np.arctan2(np.hypot(r03[..., 0, 1], r03[..., 1, 1]), r03[..., 2, 1])
            * arm_config
        )
        joints[..., 2] = np.arctan2(
            -r03[..., 2, 2] * arm_config, -r03[..., 2, 0] * arm_config
        )
        joints[..., 3] = elbow_angle
        joints[..., 4] = np.arctan2(
            r47[..., 1, 2] * wrist_config, r47[..., 0, 2] * wrist_config
        )
        joints[..., 5] = (
            np.arctan2(np.hypot(r47[..., 0, 2], r47[..., 1, 2]), r47[..., 2, 2])
            * wrist_config
        )
        joints[..., 6] = np.arctan2(
            r47[..., 2, 1] * wrist_config, -r47[..., 2, 0] * wrist_config
        )

        # At singularities both outer joints must be solved together: fixing
        # the first to the seed can discard feasible solutions at joint limits.
        for first, middle, last, rotation in ((0, 1, 2, r03), (4, 5, 6, r47)):
            singular = np.abs(np.sin(joints[..., middle])) < 1e-8
            if np.any(singular):
                d, alpha, a, _ = self.dh_params[first]
                r1 = self._dh_calc(d, alpha, a, 0.0)[:3, :3]
                d, alpha, a, _ = self.dh_params[middle]
                r2 = self._dh_calc(d, alpha, a, joints[..., middle])[..., :3, :3]
                residual = np.swapaxes(r1 @ r2, -1, -2) @ rotation
                sign = np.where(np.cos(joints[..., middle]) >= 0.0, 1.0, -1.0)
                coupled = sign * np.arctan2(residual[..., 1, 0], residual[..., 0, 0])
                coupled -= self.dh_params[first, 3] + sign * self.dh_params[last, 3]
                outer1, outer2 = self._solve_coupled_joints(
                    coupled, sign, first, last, joint_seed
                )
                joints[..., first] = np.where(
                    singular, outer1 + self.dh_params[first, 3], joints[..., first]
                )
                joints[..., last] = np.where(
                    singular, outer2 + self.dh_params[last, 3], joints[..., last]
                )
        return joints - self.dh_params[:, 3]

    def _solve_coupled_joints(self, coupled, sign, first, last, seed):
        """Minimize weighted seed distance for x + sign*y = coupled (mod 2pi).

        The distance to the joint-limit rectangle is convex in the coupled
        angle. Only the two lattice points around the box-projected seed sum
        can minimize it, independent of how many revolutions the limits span.
        """
        lower, upper = self.lower_position_limits, self.upper_position_limits
        ylo = np.minimum(sign * lower[last], sign * upper[last])
        yhi = np.maximum(sign * lower[last], sign * upper[last])
        period = 2.0 * np.pi
        kmin = np.ceil((lower[first] + ylo - coupled - 1e-12) / period)
        kmax = np.floor((upper[first] + yhi - coupled + 1e-12) / period)
        nearest_sum = np.clip(seed[first], lower[first], upper[first]) + np.clip(
            sign * seed[last], ylo, yhi
        )
        k = np.floor((nearest_sum - coupled) / period)
        sums = coupled[..., None] + period * np.clip(
            np.stack((k, k + 1), axis=-1), kmin[..., None], kmax[..., None]
        )
        lo = np.maximum(lower[first], sums - yhi[..., None])
        hi = np.minimum(upper[first], sums - ylo[..., None])
        weights = (
            np.ones(2)
            if self.ik_nearst_weight is None
            else self.ik_nearst_weight[[first, last]].copy()
        )
        scale = np.max(weights)
        weights = (weights / scale) ** 2 if scale > 0 else np.ones(2)
        x = (
            weights[0] * seed[first]
            + weights[1] * (sums - sign[..., None] * seed[last])
        ) / weights.sum()
        x = np.clip(x, lo, hi)
        y = sign[..., None] * (sums - x)
        costs = weights[0] * (x - seed[first]) ** 2 + weights[1] * (y - seed[last]) ** 2
        index = np.argmin(costs, axis=-1)[..., None]
        x = np.take_along_axis(x, index, axis=-1)[..., 0]
        y = np.take_along_axis(y, index, axis=-1)[..., 0]
        return np.where(kmin <= kmax, x, np.nan), np.where(kmin <= kmax, y, np.nan)

    def _compute_inverse_kinematics(
        self, target_pose, joint_seed, nsparam, rconf, prepared=None
    ):
        if prepared is None:
            prepared = self._prepare_ik(
                self._local_target(target_pose), self._configuration(rconf)[1]
            )
        if prepared is None:
            return False, None
        joints = self._candidate_joint_angles(prepared, nsparam, rconf, joint_seed)
        joints, valid = self._map_to_limits(joints, joint_seed)
        return (True, joints) if valid else (False, None)

    def get_ik(
        self,
        target_pose: np.ndarray,
        joint_seed: np.ndarray,
        num_samples: typing.Optional[int] = None,
        search_mode: str = "continuous",
        local_step: float = np.deg2rad(5.0),
        local_layers: int = 4,
        **kwargs,
    ) -> typing.Tuple[bool, typing.Optional[np.ndarray]]:
        """Solve a pose with deterministic arm-angle sampling.

        ``continuous`` returns the first valid nearby solution, preferring the
        seed branch. ``global`` ranks all sampled branches by weighted Euclidean
        joint distance; this is a sampled optimum, not a continuous one.
        ``num_samples`` overrides the configured default for this call only.
        Invalid inputs raise ValueError; an unavailable solution returns
        ``(False, None)``. Both modes enforce joint limits and elbow preference.
        """
        if "num_sample" in kwargs:
            if num_samples is not None:
                raise TypeError("Use only one of num_samples and num_sample")
            num_samples = kwargs.pop("num_sample")
        if kwargs:
            raise TypeError(f"Unexpected IK arguments: {', '.join(sorted(kwargs))}")
        target_pose = self._validate_pose(target_pose)
        joint_seed = self._validate_joints(joint_seed, "joint_seed")
        if search_mode not in ("continuous", "global"):
            raise ValueError("search_mode must be 'continuous' or 'global'")
        if not np.isfinite(local_step) or local_step <= 0.0:
            raise ValueError("local_step must be positive and finite")
        if (
            isinstance(local_layers, (bool, np.bool_))
            or not isinstance(local_layers, (int, np.integer))
            or local_layers < 0
        ):
            raise ValueError("local_layers must be a non-negative integer")
        samples = self._num_samples if num_samples is None else num_samples
        if (
            isinstance(samples, (bool, np.bool_))
            or not isinstance(samples, (int, np.integer))
            or samples < 2
        ):
            raise ValueError("num_samples must be an integer greater than one")

        # The seed is often the current controller state. Besides avoiding a
        # needless manifold search for a stationary target, this also handles
        # fully extended SRS singularities where the arm plane is undefined.
        mapped_seed, seed_valid = self._map_to_limits(joint_seed, joint_seed)
        if seed_valid and (
            not self._is_elbow_up
            or np.sin(mapped_seed[3] + self.dh_params[3, 3]) >= -1e-12
        ):
            seed_pose = self.get_fk(mapped_seed)
            if (
                np.linalg.norm(seed_pose[:3, 3] - target_pose[:3, 3]) < 1e-12
                and np.max(np.abs(seed_pose[:3, :3] - target_pose[:3, :3])) < 1e-12
            ):
                return True, mapped_seed.copy()

        local_target = self._local_target(target_pose)
        geometry = {1: self._prepare_ik(local_target, 1)}
        if geometry[1] is None:
            return False, None

        def prepare(rconf):
            elbow = self._configuration(rconf)[1]
            if elbow not in geometry:
                geometry[elbow] = self._prepare_ik(local_target, elbow)
            return geometry[elbow]

        def compute_ik_for_params(nsparam, rconf):
            if self._is_elbow_up and rconf & 2:
                return None
            res, joints = self._compute_inverse_kinematics(
                target_pose, joint_seed, nsparam, rconf, prepared=prepare(rconf)
            )
            if res and self._matches_target(joints, target_pose):
                return joints
            return None

        # For servoing, preserve the seed's geometric arm angle and discrete
        # branch before considering a global manifold search. A small Cartesian
        # target update normally succeeds on the very first analytic candidate.
        if search_mode == "continuous":
            try:
                seed_configuration = self.get_configuration(joint_seed)
            except (ValueError, FloatingPointError):
                seed_configuration = None
            if seed_configuration is not None:
                seed_rconf = self._configuration_index(
                    seed_configuration.shoulder,
                    seed_configuration.elbow,
                    seed_configuration.wrist,
                )
                # The first candidate remains scalar for the servoing fast path.
                result = compute_ik_for_params(
                    seed_configuration.redundancy, seed_rconf
                )
                if result is not None:
                    return True, result
                offsets = [0.0]
                for layer in range(1, int(local_layers) + 1):
                    offsets.extend((-layer * local_step, layer * local_step))
                branch_order = sorted(
                    range(8),
                    key=lambda branch: (
                        ((branch ^ seed_rconf) & 2 != 0),
                        bin(branch ^ seed_rconf).count("1"),
                    ),
                )
                # Batch the remaining layers, retaining the original priority.
                for rconf in branch_order:
                    if self._is_elbow_up and rconf & 2:
                        continue
                    local_offsets = offsets[1:] if rconf == seed_rconf else offsets
                    if not local_offsets:
                        continue
                    psis = seed_configuration.redundancy + np.asarray(local_offsets)
                    candidates = self._candidate_joint_angles(
                        prepare(rconf), psis, rconf, joint_seed
                    )
                    candidates, valid = self._map_to_limits(candidates, joint_seed)
                    for candidate in candidates[valid]:
                        if self._matches_target(candidate, target_pose):
                            return True, candidate.copy()

        # Batch every arm angle for each branch; shared geometry and NumPy
        # arrays avoid hundreds of tiny thread-pool jobs and repeated FK calls.
        joints_list = []
        nsparams = np.linspace(-np.pi, np.pi, num=int(samples), endpoint=False)
        for rconf in range(8):
            if self._is_elbow_up and rconf & 2:
                continue
            candidates = self._candidate_joint_angles(
                prepare(rconf), nsparams, rconf, joint_seed
            )
            candidates, valid = self._map_to_limits(candidates, joint_seed)
            joints_list.append(candidates[valid])
        joints_array = np.concatenate(joints_list)
        weights = 1.0 if self.ik_nearst_weight is None else self.ik_nearst_weight
        distances = np.linalg.norm((joints_array - joint_seed) * weights, axis=1)
        # Stable sorting gives reproducible results even for zero-weight ties.
        for index in np.argsort(distances, kind="stable"):
            if self._matches_target(joints_array[index], target_pose):
                return True, joints_array[index].copy()
        logger.debug("No valid IK solution found in the sampled configurations.")
        return False, None

    def get_all_fk_mat(
        self, qpos: np.ndarray
    ) -> typing.Union[typing.List[np.ndarray], np.ndarray]:
        """Return seven world-frame DH link poses followed by the TCP.

        A (7,) input retains the scalar API's list of eight matrices. A (N, 7)
        input returns an independent (N, 8, 4, 4) array, including empty batches.
        """
        qpos = np.asarray(qpos, dtype=float)
        if qpos.ndim == 1:
            qpos = self._validate_joints(qpos)
            poses = self._fk_chain(qpos)
            return poses + [poses[-1] @ self.flange_to_ee]
        if qpos.ndim != 2 or qpos.shape[1] != 7 or not np.all(np.isfinite(qpos)):
            raise ValueError("qpos must be finite and have shape (7,) or (N, 7)")
        transforms = self._joint_transforms(qpos)
        poses = np.empty((len(qpos), 8, 4, 4))
        pose = self.world_to_base
        for index in range(7):
            pose = pose @ transforms[:, index]
            poses[:, index] = pose
        poses[:, 7] = pose @ self.flange_to_ee
        return poses

    def get_fk(self, qpos: np.ndarray, index: int = -1) -> np.ndarray:
        """Return the TCP (-1) or DH link (0..6) in world coordinates.

        Input (7,) returns (4, 4); input (N, 7) returns (N, 4, 4), including
        empty batches. Results own their storage; no full link history is built.
        """
        qpos = np.asarray(qpos, dtype=float)
        if (
            qpos.ndim not in (1, 2)
            or qpos.shape[-1] != 7
            or not np.all(np.isfinite(qpos))
        ):
            raise ValueError("qpos must be finite and have shape (7,) or (N, 7)")
        if (
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or index < -1
            or index >= 7
        ):
            raise IndexError("index must be -1 or an integer between 0 and 6")
        stop = 7 if index == -1 else index + 1
        transform = self.world_to_base.copy()
        transforms = self._joint_transforms(qpos, stop)
        if qpos.ndim == 2:
            transforms = np.moveaxis(transforms, 1, 0)
        for joint_transform in transforms:
            transform = transform @ joint_transform
        return transform @ self.flange_to_ee if index == -1 else transform
