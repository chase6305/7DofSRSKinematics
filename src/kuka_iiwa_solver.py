import numpy as np
from srs_analytical_solver import SRSAnalyticalSolver
import logging
import coloredlogs

coloredlogs.install(
    level="INFO", fmt="%(asctime)s,%(msecs)03d %(levelname)s %(message)s"
)

__all__ = ["KUKAiiwaSolver"]


class KUKAiiwaSolver(SRSAnalyticalSolver):
    def __init__(
        self,
        **kwargs,
    ):
        half_pi = np.pi / 2
        self.link_lengths = np.array([0.34, 0.4, 0.4, 0.126])

        self.dh_params = np.array(
            [
                [self.link_lengths[0], -half_pi, 0, 0],  # Joint 1
                [0, half_pi, 0, 0],  # Joint 2
                [self.link_lengths[1], half_pi, 0, 0],  # Joint 3
                [0, -half_pi, 0, 0],  # Joint 4
                [self.link_lengths[2], -half_pi, 0, 0],  # Joint 5
                [0, half_pi, 0, 0],  # Joint 6
                [self.link_lengths[3], 0, 0, 0],  # Joint 7
            ]
        )

        self.d_bs = self.link_lengths[0]
        self.d_se = self.link_lengths[1]
        self.d_ew = self.link_lengths[2]
        self.d_wt = self.link_lengths[3]

        self.lower_position_limits = np.radians(
            [-180, -180, -180, -180, -180, -180, -180]
        )
        self.upper_position_limits = np.radians([180, 180, 180, 180, 180, 180, 180])

        super().__init__(
            urdf_path=None,
            end_link_name="link_ee",
            **kwargs,
        )

        self.flange_to_ee = np.eye(4)
        self.ik_nearst_weight = np.ones(7)


if __name__ == "__main__":
    solver = KUKAiiwaSolver()

    qpos = np.array([np.pi / 4, np.pi / 4, 0.0, np.pi / 4, 0.0, 0, -np.pi / 4])
    xpos = solver.get_fk(qpos=qpos)

    xpos[2, 3] -= 0.1

    res, qpos_ik = solver.get_ik(target_pose=xpos, joint_seed=np.zeros(7))

    from IPython import embed

    embed()
