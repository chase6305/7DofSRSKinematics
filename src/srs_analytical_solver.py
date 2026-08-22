import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from solver import Solver
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
try:
    import coloredlogs

    coloredlogs.install(
        level="INFO", fmt="%(asctime)s,%(msecs)03d %(levelname)s %(message)s"
    )
except ImportError:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)

__all__ = ["SRSConfiguration", "SRSAnalyticalSolver"]


@dataclass(frozen=True)
class SRSConfiguration:
    """One shoulder/elbow/wrist branch and its arm-angle redundancy."""

    shoulder: int
    elbow: int
    wrist: int
    redundancy: float

    def __post_init__(self):
        if self.shoulder not in (-1, 1):
            raise ValueError("shoulder must be -1 or 1")
        if self.elbow not in (-1, 1):
            raise ValueError("elbow must be -1 or 1")
        if self.wrist not in (-1, 1):
            raise ValueError("wrist must be -1 or 1")
        if not np.isfinite(self.redundancy):
            raise ValueError("redundancy must be finite")


class SRSAnalyticalSolver(Solver, ABC):
    def __init__(
        self,
        urdf_path: str,
        end_link_name: str,
        **kwargs,
    ):
        r"""Initializes the Pinocchio kinematics and dynamics solver.

            This class leverages Pinocchio to perform kinematic and dynamic
            computations for the specified robot model.

        Args:
            urdf_path (str, optional): Path to the robot's URDF file.
            end_link_name (str): The name of the end-effector link.
            **kwargs: Additional arguments for the base solver.
        """
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

    @staticmethod
    def _safe_cos(value: float) -> float:
        r"""Safely compute the cosine of a value, ensuring the input is within the valid range for cosine.

        Args:
            value (float): The input value.

        Returns:
            float: The input value clipped to the range [-1, 1].
        """
        if value > 1.0:
            value = 1.0
        elif value < -1.0:
            value = -1.0
        return value

    def _dh_calc(self, d: float, alpha: float, a: float, theta: float) -> np.ndarray:
        r"""Calculate the transformation matrix based on Denavit-Hartenberg (D-H) parameters.

        Args:
            d (float): Offset along the previous z to the common normal.
            alpha (float): Angle around the common normal, from the old z axis to the new z axis.
            a (float): Length of the common normal. Assuming a positive value.
            theta (float): Angle around the previous z, from the old x axis to the new x axis.

        Returns:
            np.ndarray: The 4x4 transformation matrix.
        """
        T = np.array(
            [
                [
                    np.cos(theta),
                    -np.sin(theta) * np.cos(alpha),
                    np.sin(theta) * np.sin(alpha),
                    a * np.cos(theta),
                ],
                [
                    np.sin(theta),
                    np.cos(theta) * np.cos(alpha),
                    -np.cos(theta) * np.sin(alpha),
                    a * np.sin(theta),
                ],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )
        return T

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
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
            raise ValueError("qpos must be a finite array of shape (7,)")
        def sign(value):
            return -1 if value < 0.0 else 1
        return SRSConfiguration(
            sign(qpos[1]), sign(qpos[3]), sign(qpos[5]), self.get_arm_angle(qpos)
        )

    def get_arm_angle(self, qpos: np.ndarray) -> float:
        """Measure the elbow-plane rotation about the shoulder-wrist axis."""
        qpos = np.asarray(qpos, dtype=float)
        target = self.get_fk(qpos)
        elbow = -1 if qpos[3] < 0.0 else 1
        ok, _, reference_rotation, _ = self._reference_plane(
            np.linalg.inv(self.world_to_base)
            @ target
            @ np.linalg.inv(self.flange_to_ee),
            elbow,
        )
        if not ok:
            raise ValueError("arm angle is undefined at this configuration")
        actual_rotation = np.eye(3)
        for i in range(3):
            d, alpha, a, theta = self.dh_params[i]
            actual_rotation = actual_rotation @ self._dh_calc(
                d, alpha, a, theta + qpos[i]
            )[:3, :3]
        local_target = (
            np.linalg.inv(self.world_to_base)
            @ target
            @ np.linalg.inv(self.flange_to_ee)
        )
        wrist = local_target[:3, 3] - local_target[:3, :3] @ np.array(
            [0.0, 0.0, self.dh_params[-1, 0]]
        )
        axis = wrist - np.array([0.0, 0.0, self.link_lengths[0]])
        axis /= np.linalg.norm(axis)
        # Pick the best-conditioned column after projecting it onto the plane.
        projections = [
            reference_rotation[:, i] - axis * axis.dot(reference_rotation[:, i])
            for i in range(3)
        ]
        column = int(np.argmax([np.linalg.norm(v) for v in projections]))
        reference = projections[column] / np.linalg.norm(projections[column])
        actual = actual_rotation[:, column]
        actual = actual - axis * axis.dot(actual)
        actual /= np.linalg.norm(actual)
        return float(
            np.arctan2(
                axis.dot(np.cross(reference, actual)), reference.dot(actual)
            )
        )

    def solve_configuration(
        self,
        target_pose: np.ndarray,
        configuration: SRSConfiguration,
        joint_seed: np.ndarray,
    ) -> typing.Tuple[bool, typing.Optional[np.ndarray]]:
        """Solve one explicitly selected S/E/W branch at an exact arm angle."""
        rconf = self._configuration_index(
            configuration.shoulder, configuration.elbow, configuration.wrist
        )
        ok, joints = self._compute_inverse_kinematics(
            np.asarray(target_pose, dtype=float),
            np.asarray(joint_seed, dtype=float),
            configuration.redundancy,
            rconf,
        )
        if not ok:
            return False, None
        actual = self.get_fk(joints)
        position_error = np.linalg.norm(actual[:3, 3] - target_pose[:3, 3])
        rotation_error = np.arccos(np.clip(
            (np.trace(actual[:3, :3].T @ target_pose[:3, :3]) - 1.0) / 2.0,
            -1.0,
            1.0,
        ))
        if position_error > self._pos_eps or rotation_error > self._rot_eps:
            return False, None
        return True, joints

    def get_jacobian(
        self, qpos: np.ndarray, step: float = 1e-7
    ) -> np.ndarray:
        """Return a 6x7 geometric Jacobian using central FK differences.

        The first three rows are linear velocity and the last three are angular
        velocity in the world frame.
        """
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
            raise ValueError("qpos must be a finite array of shape (7,)")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step must be positive and finite")
        jacobian = np.empty((6, 7))
        for index in range(7):
            plus, minus = qpos.copy(), qpos.copy()
            plus[index] += step
            minus[index] -= step
            pose_plus, pose_minus = self.get_fk(plus), self.get_fk(minus)
            jacobian[:3, index] = (
                pose_plus[:3, 3] - pose_minus[:3, 3]
            ) / (2.0 * step)
            rotation_rate = (
                pose_plus[:3, :3] - pose_minus[:3, :3]
            ) / (2.0 * step)
            angular_skew = rotation_rate @ self.get_fk(qpos)[:3, :3].T
            jacobian[3:, index] = np.array(
                [angular_skew[2, 1], angular_skew[0, 2], angular_skew[1, 0]]
            )
        return jacobian

    def get_null_space_direction(
        self, qpos: np.ndarray, arm_angle_step: float = 1e-4
    ) -> np.ndarray:
        """Return dq/dpsi on the current branch while keeping the TCP fixed.

        A central arm-angle difference is preferred. Near a joint limit the
        method falls back to a one-sided difference, and at an undefined arm
        plane it returns the SVD Jacobian null direction.
        """
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
            raise ValueError("qpos must be a finite array of shape (7,)")
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
            raise ValueError(
                "preferred_velocity must be a finite array of shape (7,)"
            )
        jacobian = self.get_jacobian(qpos)
        projector = np.eye(7) - np.linalg.pinv(jacobian) @ jacobian
        return projector @ preferred_velocity

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
    ) -> np.ndarray:
        r"""Calculate joint angles based on the position from shoulder to wrist and elbow configuration.

        Args:
            P_s_to_w (np.ndarray): The position vector from shoulder to wrist.
            elbow_GC4 (int): The elbow configuration parameter.

        Returns:
            np.ndarray: A boolean indicating success and an array of joint angles.
        """
        d_bs, d_se, d_ew = (
            self.link_lengths[0],
            self.link_lengths[1],
            self.link_lengths[2],
        )
        joints = np.zeros(7)

        # Check reachability and calculate elbow joint angle
        norm_P26 = np.linalg.norm(P_s_to_w)
        if not (
            d_se + d_ew + 1e-10 >= norm_P26
            and norm_P26 + 1e-10 >= abs(d_se - d_ew)
        ):
            logging.debug("Specified pose outside reachable workspace.")
            return False, None

        elbow_cos_angle = (norm_P26**2 - d_se**2 - d_ew**2) / (2 * d_se * d_ew)

        if np.abs(elbow_cos_angle) > 1.0:
            logging.debug("Elbow singularity. End effector at limit.")
            return False, None

        joints[3] = elbow_GC4 * np.arccos(elbow_cos_angle)

        # Calculate joint 1
        if np.hypot(P_s_to_w[0], P_s_to_w[1]) > 1e-10:
            joints[0] = np.arctan2(P_s_to_w[1], P_s_to_w[0])
        else:
            joints[0] = 0

        # Calculate joint 2
        euclidean_norm = np.hypot(P_s_to_w[0], P_s_to_w[1])
        angle_phi = np.arccos(
            (d_se**2 + norm_P26**2 - d_ew**2) / (2 * d_se * norm_P26)
        )
        joints[1] = np.arctan2(euclidean_norm, P_s_to_w[2]) + elbow_GC4 * angle_phi

        return True, joints

    def _reference_plane(self, pose: np.ndarray, elbow_GC4: int) -> tuple:
        r"""Calculate the reference plane vector, rotation matrix from base to elbow, and joint values.

        Args:
            pose (np.ndarray): The target pose of the end-effector.
            elbow_GC4 (int): The elbow configuration parameter.

        Returns:
            tuple: A boolean indicating success, the reference plane vector,
                the rotation matrix from base to elbow, and the joint values.
        """
        P_target = pose[:3, 3]
        P02 = np.array([0, 0, self.link_lengths[0]])  # Base to shoulder
        P67 = np.array([0, 0, self.dh_params[-1, 0]])  # Hand to end-effector

        P06 = P_target - pose[:3, :3] @ P67
        P26 = P06 - P02

        # Calculate joint angles
        joint_v = np.zeros(7)
        res, joint_v = self._calculate_joint_angles(P26, elbow_GC4)
        if not res:
            return False, None, None, None

        # Lower arm transformation
        T34_v = np.eye(4)
        T34_v = self._dh_calc(
            self.dh_params[3, 0], self.dh_params[3, 1], self.dh_params[3, 2], joint_v[3]
        )
        P34_v = T34_v[:3, 3]

        # Calculate reference elbow position and normal vector to the reference plane
        v1 = (P34_v - P02) / np.linalg.norm(P34_v - P02)
        v2 = (P06 - P02) / np.linalg.norm(P06 - P02)
        V_v_to_sew = np.cross(v1, v2)  # The normal vector to the plane

        R03_v = np.eye(3)
        for i in range(3):
            R03_v = (
                R03_v
                @ self._dh_calc(
                    self.dh_params[i, 0],
                    self.dh_params[i, 1],
                    self.dh_params[i, 2],
                    joint_v[i],
                )[:3, :3]
            )

        return True, V_v_to_sew, R03_v, joint_v

    def _compute_inverse_kinematics(
        self,
        target_pose: np.ndarray,
        joint_seed: np.ndarray,
        nsparam: float,
        rconf: int,
    ) -> tuple:
        r"""Computes the inverse kinematics for a given a target pose, normalization parameter, and configuration.

        Args:
            target_pose (np.ndarray): The target pose represented as a 4x4 transformation matrix.
            joint_seed (np.ndarray): The initial joint positions used as a seed for the IK computation.

        Raises:
            ValueError: If `target_pose` is not a 4x4 numpy array or if `joint_seed` is not a numpy array.

        Returns:
            Tuple[bool, np.ndarray]: A tuple containing:
                - A boolean indicating whether convergence to the desired pose was achieved.
                - The computed joint positions that correspond to the target pose,
                  or an empty array if convergence was not achieved.
        """
        """Perform inverse kinematics to calculate joint angles given a target pose, normalization parameter, and configuration."""
        arm_config, elbow_config, wrist_config = self._configuration(rconf)

        target_pose = (
            np.linalg.inv(self.world_to_base)
            @ target_pose
            @ np.linalg.inv(self.flange_to_ee)
        )
        P_target = target_pose[:3, 3]
        P02 = np.array([0, 0, self.link_lengths[0]])  # Base to shoulder
        P67 = np.array([0, 0, self.dh_params[-1, 0]])  # Hand to end-effector

        P06 = P_target - target_pose[:3, :3] @ P67
        P26 = P06 - P02

        joints = np.zeros(7)
        # Calculate joint angles
        res, joints = self._calculate_joint_angles(P26, elbow_config)
        if not res:
            return False, []

        # Calculate transformations
        T34 = self._dh_calc(
            self.dh_params[3, 0], self.dh_params[3, 1], self.dh_params[3, 2], joints[3]
        )
        R34 = T34[:3, :3]

        # Calculate reference plane
        res, V_v_to_sew, R03_o, joint_v = self._reference_plane(
            target_pose, elbow_config
        )
        if not res:
            return False, []

        # Calculate shoulder joint rotation matrices
        usw = P26 / np.linalg.norm(P26)
        skew_usw = self.skew(usw)

        # angle_psi = np.arctan2(pose[1, 0], pose[0, 0])
        angle_psi = nsparam

        # Calculate rotation matrix R03
        A_s = skew_usw @ R03_o
        B_s = -skew_usw @ skew_usw @ R03_o
        C_s = (usw.reshape(-1, 1) @ usw.reshape(1, -1)) @ R03_o
        R03 = A_s * np.sin(angle_psi) + B_s * np.cos(angle_psi) + C_s

        # Calculate shoulder joint angles
        joints[0] = np.arctan2(R03[1, 1] * arm_config, R03[0, 1] * arm_config)
        joints[1] = np.arccos(self._safe_cos(R03[2, 1])) * arm_config
        joints[2] = np.arctan2(-R03[2, 2] * arm_config, -R03[2, 0] * arm_config)

        # Calculate wrist joint angles
        A_w = R34.T @ A_s.T @ target_pose[:3, :3]
        B_w = R34.T @ B_s.T @ target_pose[:3, :3]
        C_w = R34.T @ C_s.T @ target_pose[:3, :3]

        # Calculate wrist rotation matrix R47
        R47 = A_w * np.sin(angle_psi) + B_w * np.cos(angle_psi) + C_w

        # Calculate wrist joint angles
        joints[4] = np.arctan2(R47[1, 2] * wrist_config, R47[0, 2] * wrist_config)
        joints[5] = np.arccos(self._safe_cos(R47[2, 2])) * wrist_config
        joints[6] = np.arctan2(R47[2, 1] * wrist_config, -R47[2, 0] * wrist_config)

        for i in range(len(joints)):
            joints[i] -= self.dh_params[i, -1]

        joints = self.qpos_to_limits(joints, joint_seed)
        if 0 == len(joints):
            return False, []

        return True, joints

    def get_ik(
        self,
        target_pose: np.ndarray,
        joint_seed: np.ndarray,
        num_samples: int = None,
        search_mode: str = "continuous",
        local_step: float = np.deg2rad(5.0),
        local_layers: int = 4,
        **kwargs,
    ) -> typing.Tuple[bool, np.ndarray]:
        r"""Computes the inverse kinematics for a given target pose.

        This function generates random joint configurations within the specified limits,
        including the provided joint_seed, and attempts to find valid inverse kinematics solutions.
        It then identifies the joint position that is closest to the joint_seed.

        Args:
            target_pose (np.ndarray): The target pose represented as a 4x4 transformation matrix.
            joint_seed (np.ndarray): The initial joint positions used as a seed, providing a reference for the solution.
            num_samples (int): Number of samples, must be positive.
            search_mode: ``continuous`` tries the seed branch and nearby arm
                angles first; ``global`` always performs exhaustive sampling.
            local_step: Arm-angle spacing for continuous local search.
            local_layers: Number of positive/negative local search layers.

        Returns:
            Tuple[bool, np.ndarray]: A tuple containing:
                - A boolean indicating whether a valid solution was found (True) or not (False).
                - The closest joint position to the joint_seed as a numpy array,
                  or an empty array if no valid solutions were found.

        Notes:
            - The function samples multiple random joint configurations and evaluates them to find a suitable solution.
            - If no valid configurations are found, warnings are logged to provide feedback on the failure.
            - The closest joint configuration to the provided joint_seed is returned if a solution exists.
        """

        target_pose = np.asarray(target_pose, dtype=float)
        joint_seed = np.asarray(joint_seed, dtype=float)
        if target_pose.shape != (4, 4) or not np.all(np.isfinite(target_pose)):
            raise ValueError("target_pose must be a finite 4x4 matrix")
        if joint_seed.shape != (7,) or not np.all(np.isfinite(joint_seed)):
            raise ValueError("joint_seed must be a finite array of shape (7,)")
        if search_mode not in ("continuous", "global"):
            raise ValueError("search_mode must be 'continuous' or 'global'")
        if not np.isfinite(local_step) or local_step <= 0.0:
            raise ValueError("local_step must be positive and finite")
        if not isinstance(local_layers, (int, np.integer)) or local_layers < 0:
            raise ValueError("local_layers must be a non-negative integer")
        if num_samples is not None:
            if not isinstance(num_samples, (int, np.integer)) or num_samples < 2:
                raise ValueError("num_samples must be an integer greater than one")
            self._num_samples = int(num_samples)

        # The seed is often the current controller state. Besides avoiding a
        # needless manifold search for a stationary target, this also handles
        # fully extended SRS singularities where the arm plane is undefined.
        seed_pose = self.get_fk(joint_seed)
        seed_position_error = np.linalg.norm(
            seed_pose[:3, 3] - target_pose[:3, 3]
        )
        seed_rotation_error = np.max(
            np.abs(seed_pose[:3, :3] - target_pose[:3, :3])
        )
        if (
            seed_position_error < 1e-12
            and seed_rotation_error < 1e-12
        ):
            return True, joint_seed.copy()

        def compute_ik_for_params(nsparam, rconf):
            res, joints = self._compute_inverse_kinematics(
                target_pose, joint_seed=joint_seed, nsparam=nsparam, rconf=rconf
            )

            if res:
                new_pose = self.get_fk(joints)
                position_error = np.linalg.norm(new_pose[:3, 3] - target_pose[:3, 3])
                rotation_error = np.arccos(np.clip(
                    (np.trace(new_pose[:3, :3].T @ target_pose[:3, :3]) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                ))
                if position_error < self._pos_eps and rotation_error < self._rot_eps:
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
                # Current branch gets the complete local sweep first. Other
                # branches try the seed arm angle before their nearby samples.
                ordered_pairs = [(seed_rconf, offset) for offset in offsets]
                ordered_pairs.extend(
                    (branch, offset)
                    for branch in branch_order
                    if branch != seed_rconf
                    for offset in offsets
                )
                for rconf, offset in ordered_pairs:
                    psi = np.arctan2(
                        np.sin(seed_configuration.redundancy + offset),
                        np.cos(seed_configuration.redundancy + offset),
                    )
                    result = compute_ik_for_params(psi, rconf)
                    if result is not None:
                        return True, result

        joints_list = []
        nsparams = np.linspace(-np.pi, np.pi, num=self._num_samples)

        with ThreadPoolExecutor() as executor:
            futures = []
            for rconf in range(8):
                for nsparam in nsparams:
                    futures.append(
                        executor.submit(compute_ik_for_params, nsparam, rconf)
                    )

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    joints_list.append(result)

        if len(joints_list) == 0:
            logging.warning("Solve IK failed.")
            return False, None

        joints_array = np.asarray(joints_list)
        weighted_distances = np.linalg.norm(
            (joints_array - joint_seed) * self.ik_nearst_weight, axis=1
        )

        # Find the index of the closest solution
        closest_index = np.argmin(weighted_distances)

        # Return the closest joint position
        closest_qpos = joints_array[closest_index]

        return True, closest_qpos

    def get_all_fk_mat(self, qpos: np.ndarray) -> np.ndarray:
        r"""Computes the forward kinematics for the end-effector given the joint states.

        Args:
            qpos (np.ndarray): A 1D array of shape [dof,] representing the joint state.

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the pose of the end-effector.
        """
        T_total = np.eye(4)
        T_total_list = []
        for i, params in enumerate(self.dh_params):
            d, alpha, a, theta = params
            if i < len(qpos):
                theta += qpos[i]

            T = self._dh_calc(d, alpha, a, theta)
            T_total = T_total @ T
            T_total_list.append(T_total.copy())
        T_total = T_total @ self.flange_to_ee
        T_total_list.append(T_total.copy())

        return T_total_list

    def get_fk(self, qpos: np.ndarray, index: int = -1) -> np.ndarray:
        r"""Get the forward kinematics for a given joint state.

        Args:
            qpos (np.ndarray): A 1D array of shape [dof,] representing the joint state.
            index (int, optional): The index of the link for which to retrieve the pose.
                                Defaults to -1, which typically corresponds to the end-effector.

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the pose of the specified link.
        """
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
            raise ValueError("qpos must be a finite array of shape (7,)")
        if index < -1 or index >= 7:
            raise IndexError("index must be -1 or between 0 and 6")
        stop = 7 if index == -1 else index + 1
        T_total = np.eye(4)
        for i, params in enumerate(self.dh_params[:stop]):
            d, alpha, a, theta = params
            if i < len(qpos):
                theta += qpos[i]

            T = self._dh_calc(d, alpha, a, theta)
            T_total = T_total @ T

        T_total = self.world_to_base @ T_total
        if index == -1:
            T_total = T_total @ self.flange_to_ee
        return T_total
