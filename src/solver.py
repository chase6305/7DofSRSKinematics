import logging
import typing
from abc import ABCMeta, abstractmethod

import numpy as np

__all__ = ["ISolver", "Solver"]

logger = logging.getLogger(__name__)


class ISolver(metaclass=ABCMeta):
    @abstractmethod
    def get_ik(
        self,
        target_pose: np.ndarray,
        joint_seed: np.ndarray,
        num_samples: typing.Optional[int] = None,
    ):
        """Return (success, joints) for a target pose and joint seed.

        A failed solve returns (False, None). ``num_samples`` optionally
        overrides the configured search sample count.
        """

    @abstractmethod
    def get_fk(self, qpos: np.ndarray, index: int = -1) -> np.ndarray:
        """Return a world-frame TCP or DH link pose; subclasses define batch shapes."""


class Solver(ISolver):
    def __init__(self, urdf_path: str, end_link_name: str, **kwargs):
        """Initialize joint metadata, limits and IK search defaults."""
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
        """Validate and store search tolerances, sampling and iteration settings."""
        positive = (pos_eps, rot_eps, dt)
        try:
            valid = all(
                np.ndim(value) == 0 and np.isfinite(value) and value > 0
                for value in positive
            )
            valid = valid and np.ndim(damp) == 0 and np.isfinite(damp) and damp >= 0
            valid = valid and all(
                isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
                and value >= minimum
                for value, minimum in ((max_iterations, 1), (num_samples, 2))
            )
            valid = valid and isinstance(is_only_position_constraint, (bool, np.bool_))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            logger.warning("Invalid iteration parameters; existing settings retained.")
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
        """Return the current IK settings."""
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
        self, ik_weight: np.ndarray, joint_ids: typing.Optional[np.ndarray] = None
    ) -> bool:
        """Set finite, non-negative weights for all joints or selected unique indices."""
        try:
            ik_weight = np.asarray(ik_weight, dtype=float)
            joint_ids = np.asarray(
                np.arange(self.dof) if joint_ids is None else joint_ids
            )
        except (TypeError, ValueError):
            return False
        if (
            ik_weight.ndim != 1
            or not np.all(np.isfinite(ik_weight))
            or np.any(ik_weight < 0.0)
            or joint_ids.ndim != 1
            or not np.issubdtype(joint_ids.dtype, np.integer)
            or np.any(joint_ids < 0)
            or np.any(joint_ids >= self.dof)
            or ik_weight.shape != joint_ids.shape
            or len(np.unique(joint_ids)) != len(joint_ids)
        ):
            logger.warning(
                "Expected non-negative finite weights and unique integer joint indices."
            )
            return False
        weights = (
            np.ones(self.dof)
            if self.ik_nearst_weight is None
            else self.ik_nearst_weight.copy()
        )
        weights[joint_ids] = ik_weight
        self.ik_nearst_weight = weights
        return True

    def get_ik_nearst_weight(self):
        """Return an independent weight array, or None when unset."""
        return None if self.ik_nearst_weight is None else self.ik_nearst_weight.copy()

    def set_position_limits(
        self,
        lower_position_limits: typing.List[float],
        upper_position_limits: typing.List[float],
    ) -> bool:
        """Set finite lower/upper joint vectors; return False for invalid bounds."""
        try:
            lower = np.asarray(lower_position_limits, dtype=float)
            upper = np.asarray(upper_position_limits, dtype=float)
        except (TypeError, ValueError):
            return False
        if (
            lower.shape != (self.dof,)
            or upper.shape != (self.dof,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
        ):
            logger.warning("Expected finite joint-limit vectors with lower <= upper.")
            return False
        self.lower_position_limits = lower.copy()
        self.upper_position_limits = upper.copy()
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
        """Restrict IK to the positive elbow branch when enabled."""
        self._is_elbow_up = enable

    def set_tcp(self, xpos: np.ndarray):
        """Set the finite, rigid flange-to-TCP transform."""
        self.tcp_xpos = self._validate_pose(xpos, "xpos").copy()

    def get_tcp(self) -> np.ndarray:
        """Return an independent flange-to-TCP transform."""
        return self.tcp_xpos.copy()

    def qpos_to_limits(
        self,
        q: np.ndarray,
        joint_seed: np.ndarray,
        active_qmask: typing.Optional[np.ndarray] = None,
    ):
        """Return the periodic representation nearest the seed, or [] when infeasible."""
        q = self._validate_joints(q, "q")
        joint_seed = self._validate_joints(joint_seed, "joint_seed")
        if active_qmask is not None:
            active_qmask = np.asarray(active_qmask)
            if active_qmask.shape != (self.dof,) or not np.all(
                (active_qmask == 0) | (active_qmask == 1)
            ):
                raise ValueError(
                    "active_qmask must be a boolean vector matching the joints"
                )
        mapped, valid = self._map_to_limits(q, joint_seed, active_qmask)
        return mapped if valid else []

    def _map_to_limits(self, q, joint_seed, active_qmask=None, tolerance=1e-12):
        """Map (..., dof) candidates independently in O(candidates * dof).

        For non-negative weights, each periodic joint can minimize its own
        seed distance; a Cartesian product of periodic values is unnecessary.
        """
        period = 2.0 * np.pi
        lower, upper = self.lower_position_limits, self.upper_position_limits
        minimum = np.ceil((lower - q - tolerance) / period)
        maximum = np.floor((upper - q + tolerance) / period)
        nearest = np.floor((joint_seed - q) / period + 0.5)
        turns = np.clip(nearest, minimum, maximum)
        mapped = np.clip(q + period * turns, lower, upper)
        active = (
            np.ones(self.dof, dtype=bool)
            if active_qmask is None
            else active_qmask.astype(bool)
        )
        valid = np.all((minimum <= maximum) | ~active, axis=-1)
        return np.where(active, mapped, q), valid

    def _validate_joints(self, values, name="qpos"):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.dof,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite array of shape ({self.dof},)")
        return values

    @staticmethod
    def _validate_pose(pose, name="target_pose"):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError(f"{name} must be a finite 4x4 matrix")
        rotation = pose[:3, :3]
        if (
            not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0.0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0.0)
        ):
            raise ValueError(f"{name} must be a rigid homogeneous transform")
        return pose

    def limit_robot_config(self, qpos_list: np.ndarray) -> np.ndarray:
        """Filter joint rows to q4 >= 0 when the positive elbow branch is requested."""
        qpos = np.asarray(qpos_list, dtype=float)
        if not self._is_elbow_up:
            return qpos
        if qpos.ndim == 1:
            qpos = qpos.reshape(1, -1)
        # Keep configurations whose elbow (the fourth revolute joint in an
        # S-R-S chain) is on the requested positive branch.
        limited = qpos[qpos[:, 3] >= 0.0]
        return limited
