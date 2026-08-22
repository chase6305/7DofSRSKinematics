import typing
from abc import ABCMeta, abstractmethod
from itertools import product
import logging
try:  # Pretty logging is optional; the solver itself only depends on NumPy.
    import coloredlogs

    coloredlogs.install(
        level="INFO", fmt="%(asctime)s,%(msecs)03d %(levelname)s %(message)s"
    )
except ImportError:  # pragma: no cover - depends on the user's environment
    logging.basicConfig(level=logging.INFO)

import numpy as np

__all__ = ["Solver"]


class ISolver(metaclass=ABCMeta):
    @abstractmethod
    def get_ik(
        self,
        target_pose: np.ndarray,
        joint_seed: np.ndarray,
        num_sample: int = None,
    ):
        r"""Computes the inverse kinematics for a given target pose.

        This function generates random joint configurations within the specified limits,
        including the provided joint_seed, and attempts to find valid inverse kinematics solutions.
        It then identifies the joint position that is closest to the joint_seed.

        Args:
            target_pose (np.ndarray): The target pose represented as a 4x4 transformation matrix.
            joint_seed (np.ndarray): The initial joint positions used as a seed.
            num_sample (int, optional): The number of random joint seed to generate.

        Returns:
            Tuple[bool, np.ndarray]: A tuple containing:
                - A boolean indicating whether a valid solution was found.
                - The closest joint position to the joint_seed, or an empty list if no valid solutions were found.
        """
        pass

    @abstractmethod
    def get_fk(self, qpos: np.ndarray, index: int = -1) -> np.ndarray:
        r"""Get the forward kinematics for a given joint state.

        Args:
            qpos (np.ndarray): A 1D array of shape [dof,] representing the joint state.
            index (int, optional): The index of the link for which to retrieve the pose.
                                Defaults to -1, which typically corresponds to the end-effector.

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the pose of the specified link.
        """
        pass


class Solver(ISolver):
    def __init__(self, urdf_path: str, end_link_name: str, **kwargs):
        r"""Initializes the kinematics solver with a robot model.

        Args:
            urdf_path (str): The file path to the robot's URDF file.
            end_link_name (str): The name of the end-effector link.
            **kwargs: Additional keyword arguments for customization.
        """
        self.urdf_path = urdf_path

        self.end_link_name = end_link_name

        # Degrees of freedom of robot joints
        self.dof = 0

        # Determine if the robot's elbow
        self._is_elbow_up = False

        # Initialize solver parameters
        self._pos_eps = 5e-4  # Tolerance for convergence for position
        self._rot_eps = 5e-4  # Tolerance for convergence for rotation
        self._max_iterations = 500  # Maximum number of iterations for the solver
        self._dt = 0.1  # Time step for numerical integration
        self._damp = 1e-6  # Damping factor to prevent numerical instability

        # Flag to indicate whether the solver should only consider position constraints.
        # If True, the solver will ignore rotation constraints during the optimization process.
        # If False, both position and rotation constraints will be taken into account.
        self._is_only_position_constraint = False

        # Number of samples to generate different joint seeds for IK iterations
        self._num_samples = 30

        # Weight for nearest neighbor search in IK (Inverse Kinematics) algorithms
        self.ik_nearst_weight = None

        self.tcp_xpos = np.eye(4)

    def set_iteration_params(
        self,
        pos_eps: float = 5e-4,
        rot_eps: float = 5e-4,
        max_iterations: int = 1000,
        dt: float = 0.1,
        damp: float = 1e-6,
        num_samples: int = 30,
        is_only_position_constraint: bool = False,
    ) -> bool:
        r"""Sets the iteration parameters for the kinematics solver.

        Args:
            pos_eps (float): Pos convergence threshold, must be positive.
            rot_eps (float): Rot convergence threshold, must be positive.
            max_iterations (int): Maximum number of iterations, must be positive.
            dt (float): Time step size, must be positive.
            damp (float): Damping factor, must be non-negative.
            num_samples (int): Number of samples, must be positive.
            is_only_position_constraint (bool): Flag to indicate whether the solver should only consider position constraints.

        Returns:
            bool: True if all parameters are valid and set, False otherwise.
        """
        # Validate parameters
        if pos_eps <= 0:
            logging.warning("Pos epsilon must be positive.")
            return False
        if rot_eps <= 0:
            logging.warning("Rot epsilon must be positive.")
            return False
        if max_iterations <= 0:
            logging.warning("Max iterations must be positive.")
            return False
        if dt <= 0:
            logging.warning("Time step must be positive.")
            return False
        if damp < 0:
            logging.warning("Damping factor must be non-negative.")
            return False
        if num_samples <= 0:
            logging.warning("Number of samples must be positive.")
            return False

        # Set parameters if all are valid
        self._pos_eps = pos_eps
        self._rot_eps = rot_eps
        self._max_iterations = max_iterations
        self._dt = dt
        self._damp = damp
        self._num_samples = num_samples
        self._is_only_position_constraint = is_only_position_constraint

        return True

    def get_iteration_params(self) -> dict:
        r"""Returns the current iteration parameters.

        Returns:
            dict: A dictionary containing the current values of:
                - pos_eps (float): Pos convergence threshold
                - rot_eps (float): Rot convergence threshold
                - max_iterations (int): Maximum number of iterations.
                - dt (float): Time step size.
                - damp (float): Damping factor.
                - num_samples (int): Number of samples.
                - is_only_position_constraint (bool): Flag to indicate whether the solver should only consider position constraints.
        """
        return {
            "pos_eps": self._pos_eps,
            "rot_eps": self._rot_eps,
            "max_iterations": self._max_iterations,
            "dt": self._dt,
            "damp": self._damp,
            "num_samples": self._num_samples,
            "is_only_position_constraint": self._is_only_position_constraint,
        }

    def set_ik_nearst_weight(
        self, ik_weight: np.ndarray, joint_ids: np.ndarray = None
    ) -> bool:
        r"""Sets the inverse kinematics nearest weight.

        Args:
            ik_weight (np.ndarray): A numpy array representing the nearest weights for inverse kinematics.
            joint_ids (np.ndarray, optional): A numpy array representing the indices of the joints to which the weights apply.
                                            If None, defaults to all joint indices.

        Returns:
            bool: True if the weights are set successfully, False otherwise.
        """
        ik_weight = np.asarray(ik_weight, dtype=float)
        if ik_weight.ndim != 1 or not np.all(np.isfinite(ik_weight)):
            logging.warning("ik_weight must be a finite one-dimensional array.")
            return False
        if np.any(ik_weight < 0.0):
            logging.warning("ik_weight values must be non-negative.")
            return False

        # Set joint_ids to all joint indices if it is None
        if joint_ids is None:
            joint_ids = np.arange(self.dof)

        joint_ids = np.array(joint_ids)

        # Check if joint_ids has valid indices
        if np.any(joint_ids >= self.dof) or np.any(joint_ids < 0):
            logging.warning(
                "joint_ids must contain valid indices between 0 and {}.".format(
                    self.dof - 1
                )
            )
            return False

        # Check if ik_weight and joint_ids have the same length
        if ik_weight.shape[0] != joint_ids.shape[0]:
            logging.warning("ik_weight and joint_ids must have the same length.")
            return False

        # Initialize the weights
        if self.ik_nearst_weight is None:
            # If ik_nearst_weight is None, set all weights to 1
            self.ik_nearst_weight = np.ones(self.dof)

            # Set specific weights for joint_ids to the provided ik_weight
            for i, joint_id in enumerate(joint_ids):
                self.ik_nearst_weight[joint_id] = ik_weight[i]
        else:
            # If ik_nearst_weight is not None, only fill joint_ids
            for i, joint_id in enumerate(joint_ids):
                self.ik_nearst_weight[joint_id] = ik_weight[i]

        return True

    def get_ik_nearst_weight(self):
        r"""Gets the inverse kinematics nearest weight.

        Returns:
            np.ndarray: A numpy array representing the nearest weights for inverse kinematics.
        """
        return self.ik_nearst_weight

    def set_position_limits(
        self,
        lower_position_limits: typing.List[float],
        upper_position_limits: typing.List[float],
    ) -> bool:
        r"""Sets the upper and lower joint position limits.

        Parameters:
            lower_position_limits (List[float]): A list of lower limits for each joint.
            upper_position_limits (List[float]): A list of upper limits for each joint.

        Returns:
            bool: True if limits are successfully set, False if the input is invalid.
        """
        if (
            len(lower_position_limits) != self.dof
            or len(upper_position_limits) != self.dof
        ):
            logging.warning("Length of limits must match the number of joints.")
            return False

        if not (
            np.all(np.isfinite(lower_position_limits))
            and np.all(np.isfinite(upper_position_limits))
        ):
            logging.warning("Joint limits must be finite.")
            return False

        if any(
            lower > upper
            for lower, upper in zip(lower_position_limits, upper_position_limits)
        ):
            logging.warning(
                "Each lower limit must be less than or equal to the corresponding upper limit."
            )
            return False

        self.lower_position_limits = np.array(lower_position_limits)
        self.upper_position_limits = np.array(upper_position_limits)
        return True

    def get_position_limits(self) -> dict:
        r"""Returns the current joint position limits.

        Returns:
            dict: A dictionary containing:
                - lower_position_limits (List[float]): The current lower limits for each joint.
                - upper_position_limits (List[float]): The current upper limits for each joint.
        """
        return {
            "lower_position_limits": self.lower_position_limits.tolist(),
            "upper_position_limits": self.upper_position_limits.tolist(),
        }

    def set_elbow_up(self, enable: bool = False):
        r"""Set the elbow position state.

        Args:
            enable (bool): Whether to enable the elbow-up position.
        """
        self._is_elbow_up = enable

    def set_tcp(self, xpos: np.ndarray):
        r"""Sets the TCP position with the given 4x4 homogeneous matrix.

        Args:
            xpos (np.ndarray): The 4x4 homogeneous matrix to be set as the TCP position.

        Raises:
            ValueError: If the input is not a 4x4 numpy array.
        """
        xpos = np.array(xpos)
        if xpos.shape != (4, 4):
            raise ValueError("Input must be a 4x4 homogeneous matrix")
        self.tcp_xpos = xpos

    def get_tcp(self) -> np.ndarray:
        r"""Returns the current TCP position.

        Returns:
            np.ndarray: The current TCP position.

        Raises:
            ValueError: If the TCP position has not been set.
        """
        return self.tcp_xpos

    def qpos_to_limits(
        self,
        q: np.ndarray,
        joint_seed: np.ndarray,
        active_qmask: np.ndarray = None,
    ):
        """Adjusts the joint positions (q) to be within specified limits and as close as possible to the joint seed,
        while minimizing the total weighted difference.

        Args:
            q (np.ndarray): The original joint positions.
            joint_seed (np.ndarray): The desired (seed) joint positions.
            active_qmask (np.ndarray): A mask indicating which joints are active.

        Returns:
            np.ndarray: The adjusted joint positions within the specified limits.
        """
        best_qpos_limit = np.copy(q)
        best_total_q_diff = float("inf")

        if active_qmask is None:
            active_qmask = np.ones_like(q)

        # Initialize a list for possible values for each joint
        possible_arrays = []

        if self.ik_nearst_weight is None:
            self.ik_nearst_weight = np.ones_like(best_qpos_limit)

        # Generate possible values for each joint
        dof_num = len(q)
        for i in range(dof_num):
            if active_qmask[i]:  # Only process active joints
                current_possible_values = []

                # Calculate how many 2π fits into the adjustment to the limits
                lower_adjustment = (q[i] - self.lower_position_limits[i]) // (2 * np.pi)
                upper_adjustment = (self.upper_position_limits[i] - q[i]) // (2 * np.pi)

                # Consider the current value and its periodic adjustments
                for offset in range(
                    int(lower_adjustment) - 1, int(upper_adjustment) + 2
                ):  # Adjust by calculated limits
                    adjusted_value = q[i] + offset * (2 * np.pi)

                    # Check if the adjusted value is within limits
                    if (
                        self.lower_position_limits[i]
                        <= adjusted_value
                        <= self.upper_position_limits[i]
                    ):
                        current_possible_values.append(adjusted_value)

                # Also check the original value
                if (
                    self.lower_position_limits[i]
                    <= q[i]
                    <= self.upper_position_limits[i]
                ):
                    current_possible_values.append(q[i])

                if not current_possible_values:
                    return []  # If no possible values for an active joint
                possible_arrays.append(current_possible_values)
            else:
                # If not active, just append the original value
                possible_arrays.append([q[i]])

        # Generate all possible combinations
        all_possible_combinations = product(*possible_arrays)

        # Check each combination and calculate the absolute difference sum
        for combination in all_possible_combinations:
            total_q_diff = np.sum(
                np.abs(np.array(combination) - joint_seed) * self.ik_nearst_weight
            )

            # If a smaller difference sum is found, update the best solution
            if total_q_diff < best_total_q_diff:
                best_total_q_diff = total_q_diff
                best_qpos_limit = np.array(combination)

        return best_qpos_limit

    @abstractmethod
    def get_ik(
        self,
        target_pose: np.ndarray,
        joint_seed: np.ndarray,
        num_sample: int = None,
    ):
        r"""Computes the inverse kinematics for a given target pose.

        This function generates random joint configurations within the specified limits,
        including the provided joint_seed, and attempts to find valid inverse kinematics solutions.
        It then identifies the joint position that is closest to the joint_seed.

        Args:
            target_pose (np.ndarray): The target pose represented as a 4x4 transformation matrix.
            joint_seed (np.ndarray): The initial joint positions used as a seed.
            num_sample (int, optional): The number of random joint seed to generate.

        Returns:
            Tuple[bool, np.ndarray]: A tuple containing:
                - A boolean indicating whether a valid solution was found.
                - The closest joint position to the joint_seed, or an empty list if no valid solutions were found.
        """
        pass

    @abstractmethod
    def get_fk(self, qpos: np.ndarray, index: int = -1) -> np.ndarray:
        r"""Get the forward kinematics for a given joint state.

        Args:
            qpos (np.ndarray): A 1D array of shape [dof,] representing the joint state.
            index (int, optional): The index of the link for which to retrieve the pose.
                                Defaults to -1, which typically corresponds to the end-effector.

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the pose of the specified link.
        """
        pass

    def limit_robot_config(self, qpos_list: np.ndarray) -> np.ndarray:
        r"""Limit the robot configuration based on the elbow position.

        If the elbow is in the up position, it checks the positions of specific
        links to determine if the configuration is valid.

        Args:
            qpos_list (np.ndarray): The list of joint positions to be limited.

        Returns:
            np.ndarray: The limited list of joint positions if the elbow is up,
                        otherwise returns the original list.
        """
        qpos = np.asarray(qpos_list, dtype=float)
        if not self._is_elbow_up:
            return qpos
        if qpos.ndim == 1:
            qpos = qpos.reshape(1, -1)
        # Keep configurations whose elbow (the fourth revolute joint in an
        # S-R-S chain) is on the requested positive branch.
        limited = qpos[qpos[:, 3] >= 0.0]
        return limited
