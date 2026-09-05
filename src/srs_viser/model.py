"""SRS experiment state and geometry, independent of Viser and robot meshes."""

from itertools import product

import numpy as np

from .motions import CartesianPath, bounded_ik_step

BRANCHES = tuple(product((-1, 1), repeat=3))
PRESETS = {
    "Bent arm": np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]),
    "Elbow down": np.array([-0.8, 0.4, -0.3, -1.0, 0.5, -0.7, 0.6]),
    "Near straight": np.array([0.1, 0.1, 0.2, 0.15, -0.2, 0.2, 0.0]),
}
MODES = {
    "arm-angle": "Arm angle",
    "branches": "Eight branches",
    "jacobian": "Jacobian",
    "trajectory": "TCP trajectory",
    "manifold": "Self-motion manifold",
}


def pose_error(actual, target):
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    rotation = float(
        2
        * np.arcsin(
            np.clip(np.linalg.norm(actual[:3, :3] - target[:3, :3]) / np.sqrt(8), 0, 1)
        )
    )
    return position, rotation


def branch_label(branch):
    return " / ".join(f"{name}{sign:+d}" for name, sign in zip("SEW", branch))


def elbow_circle(chain, samples=97):
    """The geometric elbow circle; joint limits can exclude parts of it."""
    shoulder, elbow, wrist = chain[[0, 2, 4], :3, 3]
    axis = wrist - shoulder
    length = np.linalg.norm(axis)
    if length < 1e-9:
        return np.empty((0, 3))
    axis /= length
    center = shoulder + axis * np.dot(elbow - shoulder, axis)
    radius = elbow - center
    tangent = np.cross(axis, radius)
    angles = np.linspace(-np.pi, np.pi, samples)
    return center + np.cos(angles)[:, None] * radius + np.sin(angles)[:, None] * tangent


class Experiment:
    def __init__(self, backend, mode="arm-angle"):
        if mode not in MODES:
            raise ValueError("unknown experiment")
        self.backend = backend
        self.mode = mode
        self.play = False
        self.amplitude = 35.0
        self.frequency = 0.2
        self.phase = 0.0
        self.play_center = 0.0
        self.generation = 0
        self._gallery_key = None
        self._gallery = None
        self._gallery_chains = None
        self._chain_key = None
        self._chain = None
        self._velocity_key = None
        self._velocities = None
        self.path_kind = "figure-eight"
        self.path_plane = "world-xy"
        self.path_radius = 0.035
        self.max_joint_speed = 1.0
        self._manifold_key = None
        self._manifold = None
        self.reset()

    def reset(self, preset="Bent arm"):
        self.set_joints(PRESETS[preset])
        self.preset = preset
        self.play = False
        self.phase = 0.0
        self.play_center = 0.0

    def set_joints(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("joints must be a finite vector of length seven")
        if np.any(joints < self.backend.lower) or np.any(joints > self.backend.upper):
            raise ValueError("joints must be within their position limits")
        # Prepare first: failed geometry extraction must not partially reset state.
        branch, psi = self.backend.configuration(joints)
        chain = self.backend.chains(joints[None])[0]
        chain.setflags(write=False)
        self.q = joints.copy()
        self.anchor_q = joints.copy()
        self._chain, self._chain_key = chain, self.q.tobytes()
        self.target = chain[-1].copy()
        self.branch, self.anchor_psi = branch, psi
        self.offset_deg = 0.0
        self.restart_sweep()
        self.solved = True
        self.message = "Ready"
        self.null_residual = 0.0
        self._velocity_key = None
        self.generation += 1
        self.reset_trajectory()

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError("unknown experiment")
        if self.mode == "trajectory" and mode != "trajectory":
            self.set_joints(self.q)
        self.mode, self.play = mode, False
        self.restart_sweep()
        if mode == "trajectory":
            self.reset_trajectory()

    def reset_trajectory(self):
        self.trajectory = CartesianPath(
            self.target, self.path_kind, self.path_radius, self.path_plane
        )
        self.trail = [self.chain()[-1, :3, 3].copy()]
        self.trajectory_steps = 0
        self.peak_joint_step = 0.0

    def restart_sweep(self):
        """Anchor a new sweep after a pose or amplitude edit; pauses keep phase."""
        self.play_center, self.phase = self.offset_deg, 0.0

    def follow_trajectory(self, dt):
        """Advance on the same IK branch, reducing progress to bound joint speed."""
        start = self.trajectory.phase

        def solve(scale):
            phase = start + 2 * np.pi * self.frequency * dt * scale
            target = self.trajectory.pose(phase)
            return self.backend.solve(
                target,
                self.branch,
                self.anchor_psi + np.deg2rad(self.offset_deg),
                self.q,
            )

        accepted = bounded_ik_step(self.q, dt, self.max_joint_speed, solve)
        if accepted is not None:
            scale, result = accepted
            phase = start + 2 * np.pi * self.frequency * dt * scale
            step = float(np.max(np.abs(result - self.q)))
            self.q, self.target = result, self.trajectory.pose(phase)
            self.trajectory.phase = phase % (2 * np.pi)
            self.generation += 1
            self.solved = True
            self.trajectory_steps += 1
            self.peak_joint_step = max(self.peak_joint_step, step)
            self.trail.append(self.chain()[-1, :3, 3].copy())
            self.trail = self.trail[-512:]
            self.message = "Tracking / fixed orientation / same S/E/W branch"
            return True
        self.message = "No feasible bounded path step; paused at last feasible pose"
        self.play = False
        return False

    def follow_sweep(self, dt):
        """Play fixed-TCP self-motion without advancing through failed IK samples."""
        if not self.solved:
            self.play = False
            self.message = "Select a feasible arm angle and branch before playing"
            return False

        def offset(scale):
            phase = self.phase + 2 * np.pi * self.frequency * dt * scale
            return self.play_center + self.amplitude * np.sin(phase)

        def solve(scale):
            return self.backend.solve(
                self.target,
                self.branch,
                self.anchor_psi + np.deg2rad(offset(scale)),
                self.q,
            )

        accepted = bounded_ik_step(self.q, dt, self.max_joint_speed, solve)
        if accepted is None:
            self.play = False
            self.message = (
                "No feasible bounded self-motion step; paused at last feasible pose"
            )
            return False
        scale, self.q = accepted
        angle = np.deg2rad(offset(scale))
        self.offset_deg = float(np.rad2deg(np.arctan2(np.sin(angle), np.cos(angle))))
        self.phase = (self.phase + 2 * np.pi * self.frequency * dt * scale) % (
            2 * np.pi
        )
        self.message = "Fixed TCP / bounded joint speed / same S/E/W branch"
        return True

    def set_target(self, target):
        result = self.backend.ik(target, self.q)
        if result is None:
            self.message = "No feasible IK solution; target unchanged"
            return False
        self.set_joints(result)
        self.target = np.asarray(target, dtype=float).copy()
        self.reset_trajectory()
        return True

    def set_arm_angle(self, offset_deg, branch=None):
        if not np.isfinite(offset_deg):
            raise ValueError("arm-angle offset must be finite")
        if branch is not None:
            if tuple(branch) not in BRANCHES:
                raise ValueError("invalid S/E/W branch")
            self.branch = tuple(branch)
        angle = np.deg2rad(offset_deg)
        self.offset_deg = float(np.rad2deg(np.arctan2(np.sin(angle), np.cos(angle))))
        self.restart_sweep()
        result = self.backend.solve(
            self.target,
            self.branch,
            self.anchor_psi + np.deg2rad(self.offset_deg),
            self.q,
        )
        self.solved = result is not None
        if self.solved:
            self.q = result
            self.message = "Fixed TCP / valid joint limits"
        else:
            self.message = (
                "Branch unavailable at this angle; retaining last feasible pose"
            )
        return self.solved

    def chain(self):
        """Reuse current link/TCP poses across rendering and diagnostics."""
        key = self.q.tobytes()
        if key != self._chain_key:
            self._chain = self.backend.chains(self.q[None])[0]
            self._chain.setflags(write=False)
            self._chain_key = key
        return self._chain

    def gallery(self):
        """Solve a branch table only when the target, anchor pose, or angle changes."""
        key = (self.generation, self.anchor_psi, self.offset_deg)
        if key != self._gallery_key:
            psi = self.anchor_psi + np.deg2rad(self.offset_deg)
            valid, joints = self.backend.solve_many(
                self.target, [(branch, psi) for branch in BRANCHES], self.anchor_q
            )
            self._gallery = tuple(q if ok else None for ok, q in zip(valid, joints))
            self._gallery_key = key
            self._gallery_chains = None
        return self._gallery

    def manifold(self, samples=73):
        """Sample eight one-dimensional self-motion branches at the fixed TCP."""
        if (
            isinstance(samples, bool)
            or not isinstance(samples, (int, np.integer))
            or samples < 3
        ):
            raise ValueError("manifold samples must be an integer >= 3")
        key = (self.generation, samples)
        if key != self._manifold_key:
            angles = np.linspace(-np.pi, np.pi, samples)
            configs = [(branch, psi) for branch in BRANCHES for psi in angles]
            valid, joints = self.backend.solve_many(self.target, configs, self.anchor_q)
            valid, joints = valid.reshape(8, samples), joints.reshape(8, samples, 7)
            angles.setflags(write=False)
            valid.setflags(write=False)
            joints.setflags(write=False)
            self._manifold = angles, valid, joints
            self._manifold_key = key
        return self._manifold

    def gallery_chains(self):
        """Batch link poses once per gallery update, keeping unavailable slots."""
        solutions = self.gallery()
        if self._gallery_chains is None:
            valid = [q for q in solutions if q is not None]
            chains = self.backend.chains(np.asarray(valid).reshape(-1, 7))
            chains.setflags(write=False)
            rows = iter(chains)
            self._gallery_chains = tuple(
                next(rows) if q is not None else None for q in solutions
            )
        return self._gallery_chains

    def centering_cost(self, q=None):
        q = self.q if q is None else q
        width = np.maximum(self.backend.upper - self.backend.lower, 1e-8)
        middle = (self.backend.upper + self.backend.lower) * 0.5
        return float(np.sum(((q - middle) / width) ** 2))

    def velocities(self):
        key = (
            self.q.tobytes(),
            self.backend.lower.tobytes(),
            self.backend.upper.tobytes(),
        )
        if key == self._velocity_key:
            return self._velocities
        width = np.maximum(self.backend.upper - self.backend.lower, 1e-8)
        middle = (self.backend.upper + self.backend.lower) * 0.5
        preferred = (middle - self.q) / width**2
        jacobian = self.backend.jacobian(self.q)
        projected = preferred - np.linalg.pinv(jacobian, rcond=1e-6) @ (
            jacobian @ preferred
        )
        self.null_residual = float(np.linalg.norm(jacobian @ projected))
        self._velocities = jacobian, preferred, projected
        for value in self._velocities:
            value.setflags(write=False)
        self._velocity_key = key
        return self._velocities

    def center_step(self, dt):
        """Project a descent step and retract to the exact fixed-TCP manifold."""
        _, _, velocity = self.velocities()
        size = np.max(np.abs(velocity))
        if size < 1e-8:
            self.message = "Joint-centering stationary point"
            return False
        velocity = velocity * min(1.0, 0.4 / size)
        old_cost = self.centering_cost()

        def solve(scale):
            trial = np.clip(
                self.q + dt * scale * velocity, self.backend.lower, self.backend.upper
            )
            try:
                _, psi = self.backend.configuration(trial)
            except ValueError:
                return None
            result = self.backend.solve(self.target, self.branch, psi, self.q)
            if result is not None and self.centering_cost(result) < old_cost - 1e-12:
                return result
            return None

        accepted = bounded_ik_step(self.q, dt, self.max_joint_speed, solve)
        if accepted is not None:
            _, result = accepted
            _, psi = self.backend.configuration(result)
            self.q = result
            self.offset_deg = float(
                np.rad2deg(
                    np.arctan2(
                        np.sin(psi - self.anchor_psi), np.cos(psi - self.anchor_psi)
                    )
                )
            )
            self.solved = True
            self.message = "Joint centering / TCP reprojected / bounded joint speed"
            return True
        self.message = "No further feasible descent step"
        return False

    def advance(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if not self.play:
            return False
        # Bound integration when a tab or renderer stalls.
        dt = min(float(dt), 0.05)
        if self.mode == "jacobian":
            changed = self.center_step(dt)
            if not changed:
                self.play = False
            return changed
        if self.mode == "trajectory":
            return self.follow_trajectory(dt)
        return self.follow_sweep(dt)

    def diagnostics(self):
        position, rotation = pose_error(self.chain()[-1], self.target)
        if self.mode == "jacobian":
            self.velocities()
        margin = np.min(
            np.minimum(self.q - self.backend.lower, self.backend.upper - self.q)
        )
        return {
            "position_m": position,
            "rotation_rad": rotation,
            "margin_rad": float(margin),
            "centering_cost": self.centering_cost(),
            "null_residual": self.null_residual,
        }


def validate(backend, frames=40):
    if frames < 2:
        raise ValueError("frames must be >= 2")
    report = {
        "arm_angle_solutions": 0,
        "branch_solutions": 0,
        "centering_steps": 0,
        "max_position_m": 0.0,
        "max_rotation_rad": 0.0,
    }
    experiment = Experiment(backend)
    target = experiment.target.copy()

    def check(q, expected=target):
        if np.any(q < backend.lower - 1e-7) or np.any(q > backend.upper + 1e-7):
            raise AssertionError("demo exceeded a joint limit")
        position, rotation = pose_error(backend.fk(q), expected)
        report["max_position_m"] = max(report["max_position_m"], position)
        report["max_rotation_rad"] = max(report["max_rotation_rad"], rotation)
        if position > 2e-4 or rotation > 2e-4:
            raise AssertionError("fixed-TCP demo exceeded pose residual tolerance")

    for offset in np.linspace(-35, 35, frames):
        if experiment.set_arm_angle(offset):
            report["arm_angle_solutions"] += 1
            check(experiment.q)
    experiment.set_arm_angle(0)
    for q in experiment.gallery():
        if q is not None:
            report["branch_solutions"] += 1
            check(q)
    if not report["arm_angle_solutions"] or not report["branch_solutions"]:
        raise AssertionError("no feasible demo configurations")
    initial_cost = experiment.centering_cost()
    for _ in range(frames):
        report["centering_steps"] += experiment.center_step(1 / 30)
        check(experiment.q)
    if experiment.centering_cost() > initial_cost + 1e-9:
        raise AssertionError("centering increased its objective")
    if report["centering_steps"] == 0 or experiment.null_residual > 1e-5:
        raise AssertionError("null-space demo did not produce a valid descent step")
    report["centering_cost_before"] = initial_cost
    report["centering_cost_after"] = experiment.centering_cost()
    experiment.reset()
    _, valid, joints = experiment.manifold(max(9, frames))
    report["manifold_solutions"] = int(valid.sum())
    for q in joints[valid]:
        check(q, experiment.target)
    if not report["manifold_solutions"]:
        raise AssertionError("no feasible manifold samples")
    experiment.set_mode("trajectory")
    for _ in range(frames):
        if not experiment.follow_trajectory(1 / 30):
            raise AssertionError("default trajectory stopped unexpectedly")
        check(experiment.q, experiment.target)
    report["trajectory_steps"] = experiment.trajectory_steps
    report["peak_trajectory_joint_step_rad"] = experiment.peak_joint_step
    experiment.reset()
    experiment.set_mode("manifold")
    experiment.amplitude, experiment.frequency, experiment.play = 90, 0.5, True
    report["self_motion_steps"] = 0
    report["peak_self_motion_speed_rad_s"] = 0.0
    for _ in range(frames):
        before = experiment.q.copy()
        if not experiment.advance(1 / 30):
            raise AssertionError("default self-motion playback stopped unexpectedly")
        speed = float(np.max(np.abs(experiment.q - before)) * 30)
        if speed > experiment.max_joint_speed + 3e-7:
            raise AssertionError("self-motion playback exceeded the joint-speed bound")
        report["self_motion_steps"] += 1
        report["peak_self_motion_speed_rad_s"] = max(
            report["peak_self_motion_speed_rad_s"], speed
        )
        check(experiment.q, experiment.target)
    return report
