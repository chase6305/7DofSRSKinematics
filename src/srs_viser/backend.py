"""NumPy adapter used by the shared SRS experiment model."""

from kuka_iiwa_solver import KUKAiiwaSolver
from srs_analytical_solver import SRSConfiguration


class Backend:
    def __init__(self, kind="numpy", device="cpu"):
        if kind != "numpy" or device != "cpu":
            raise ValueError("The NumPy lab uses --backend numpy --device cpu")
        self.solver = KUKAiiwaSolver()
        self.lower = self.solver.lower_position_limits.copy()
        self.upper = self.solver.upper_position_limits.copy()
        self.urdf_path = self.solver.get_urdf_path()
        self.label = "NumPy"

    def fk(self, q):
        return self.solver.get_fk(q)

    def chains(self, joints):
        return self.solver.get_all_fk_mat(joints)

    def jacobian(self, q):
        return self.solver.get_jacobian(q)

    def configuration(self, q):
        config = self.solver.get_configuration(q)
        return (config.shoulder, config.elbow, config.wrist), config.redundancy

    def solve(self, target, branch, psi, seed):
        ok, joints = self.solver.solve_configuration(
            target, SRSConfiguration(*branch, float(psi)), seed
        )
        return joints if ok else None

    def ik(self, target, seed):
        ok, joints = self.solver.get_ik(target, seed)
        return joints if ok else None

    def solve_many(self, target, configurations, seed):
        return self.solver.solve_configurations(
            target,
            [SRSConfiguration(*branch, float(psi)) for branch, psi in configurations],
            seed,
        )
