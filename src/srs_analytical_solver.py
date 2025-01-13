import typing
from abc import ABC, abstractmethod
import numpy as np
from solver import Solver
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import coloredlogs

coloredlogs.install(
    level="INFO", fmt="%(asctime)s,%(msecs)03d %(levelname)s %(message)s"
)

__all__ = ["SRSAnalyticalSolver"]


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
        if not (np.abs(d_bs + d_ew) > norm_P26 and norm_P26 > np.abs(d_bs - d_ew)):
            logging.debug("Specified pose outside reachable workspace.")

        elbow_cos_angle = (norm_P26**2 - d_se**2 - d_ew**2) / (2 * d_se * d_ew)

        if np.abs(elbow_cos_angle) > 1.0:
            logging.debug("Elbow singularity. End effector at limit.")
            return False, None

        joints[3] = elbow_GC4 * np.arccos(elbow_cos_angle)

        # Calculate joint 1
        if np.linalg.norm(P_s_to_w[2]) > 1e-6:
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
            **kwargs: Additional arguments for future extensions.

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

        if num_samples is not None:
            self._num_samples = num_samples

        joints_list = []
        nsparams = np.linspace(
            self.lower_position_limits[3],
            self.upper_position_limits[3],
            num=self._num_samples,
        )  # 10 个不同角度

        def compute_ik_for_params(nsparam, rconf):
            res, joints = self._compute_inverse_kinematics(
                target_pose, joint_seed=joint_seed, nsparam=nsparam, rconf=rconf
            )

            if res:
                new_pose = self.get_fk(joints)
                dis = np.linalg.norm(new_pose - target_pose)
                if dis < self._pos_eps:
                    return joints
            return None

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

        weighted_distances = np.linalg.norm(
            (joints_list - joint_seed) * self.ik_nearst_weight, axis=1
        )

        # Find the index of the closest solution
        closest_index = np.argmin(weighted_distances)

        # Return the closest joint position
        closest_qpos = joints_list[closest_index]

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

    def get_fk(self, qpos: np.ndarray) -> np.ndarray:
        r"""Get the forward kinematics for a given joint state.

        Args:
            qpos (np.ndarray): A 1D array of shape [dof,] representing the joint state.
            index (int, optional): The index of the link for which to retrieve the pose.
                                Defaults to -1, which typically corresponds to the end-effector.

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the pose of the specified link.
        """
        T_total = np.eye(4)
        for i, params in enumerate(self.dh_params):
            d, alpha, a, theta = params
            if i < len(qpos):
                theta += qpos[i]

            T = self._dh_calc(d, alpha, a, theta)
            T_total = T_total @ T

        T_total = self.world_to_base @ T_total @ self.flange_to_ee
        return T_total
