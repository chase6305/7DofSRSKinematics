"""Interactive SRS trajectories, self-motion manifolds and Jacobian experiments."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np

from .backend import Backend
from .commands import CommandQueue
from .model import (
    BRANCHES,
    MODES,
    PRESETS,
    Experiment,
    branch_label,
    elbow_circle,
    validate,
)
from .motions import PATHS, PLANES, PROJECTIONS, manifold_segments

DESCRIPTIONS = {
    "arm-angle": "**Fixed TCP, moving elbow.** Drag the arm-angle slider or press Play. Playback respects the joint-speed bound and pauses at infeasible steps. Orange shows the geometric elbow circle; joint limits can exclude parts of it. Drag the TCP gizmo to choose another target.",
    "branches": "**One local target, eight S/E/W branches.** Tiles are translated only for display. Choose a branch and sweep the shared arm angle; unavailable branches are marked explicitly. Change the pose using the preset or joint controls.",
    "jacobian": "**Motion that preserves the TCP.** Play projects joint-centering velocity into the full Jacobian null space, then corrects each step back to the fixed target. The wire ellipsoid shows translational velocity for unit joint-speed norm, with orientation unconstrained.",
    "trajectory": "**Follow a Cartesian path.** Choose a shape and a world or TCP plane, then Play. Gray is the reference; green is the FK trace. TCP orientation and the S/E/W branch stay fixed. Progress slows to respect the joint-speed bound and pauses when no feasible step remains.",
    "manifold": "**The fixed-TCP self-motion manifold.** Choose three joint coordinates for the colored branch curves, in radians. Move the arm angle or Play to follow the selected branch with bounded joint speed. Playback pauses when no feasible step remains. Gaps mark infeasible samples or discontinuities; crossings in a projection need not be the same seven-joint configuration.",
}


class LabApp:
    """Callbacks enqueue changes; a single render loop owns solver and scene state."""

    def __init__(
        self,
        backend,
        mode="arm-angle",
        host="127.0.0.1",
        port=8081,
        meshes=True,
        *,
        experiment=None,
    ):
        import trimesh
        import viser

        self.trimesh = trimesh
        self.model = Experiment(backend, mode) if experiment is None else experiment
        self.commands = CommandQueue()
        self.server = viser.ViserServer(
            host=host, port=port, label="SRS Kinematics Lab"
        )
        self.closed = False
        self.last_time = time.perf_counter()
        self.last_metrics = 0.0
        self.render_count = 0
        self._manifold_render_key = None
        self.meshes_requested = meshes
        self.robot = None
        try:
            self._build_scene()
            self._build_gui()
            self.server.on_client_connect(self._camera)
            self.render(force_metrics=True)
        except Exception:
            self.server.stop()
            raise

    def _line(self, name, points, color, width=3.0):
        # Viser 1.1 renamed line_width; support the viewer extra's 1.0 API too.
        parameters = inspect.signature(self.server.scene.add_line_segments).parameters
        width_args = (
            {"thickness": width, "thickness_units": "screen"}
            if "thickness" in parameters
            else {"line_width": width}
        )
        return self.server.scene.add_line_segments(
            name, np.asarray(points, dtype=np.float32), colors=color, **width_args
        )

    def _build_scene(self):
        scene = self.server.scene
        scene.add_grid(
            "/grid",
            width=8,
            height=6,
            cell_size=0.2,
            section_size=1.0,
            plane_color=(27, 32, 41),
            plane_opacity=1.0,
            cell_color=(55, 64, 78),
            section_color=(87, 97, 112),
        )
        scene.add_frame("/origin", axes_length=0.2, axes_radius=0.004)
        zeros = np.zeros((4, 2, 3))
        self.arm = self._line("/primary/links", zeros, (111, 206, 244), 6)
        self.joints = scene.add_point_cloud(
            "/primary/sew",
            np.zeros((3, 3)),
            np.array(
                [[240, 163, 74], [220, 105, 167], [103, 207, 195]], dtype=np.uint8
            ),
            point_size=0.024,
            point_shape="circle",
        )
        self.circle = self._line("/primary/elbow_circle", zeros, (244, 166, 74), 2)
        self.axis = self._line(
            "/primary/sw_axis", np.zeros((1, 2, 3)), (155, 149, 198), 2
        )
        self.target = scene.add_transform_controls("/primary/target", scale=0.18)
        self.tcp = scene.add_frame("/primary/tcp", axes_length=0.12, axes_radius=0.004)
        self.target_label = scene.add_label(
            "/primary/label", "Target TCP", position=(0, 0, 1)
        )
        self.sew_labels = [
            scene.add_label(f"/primary/label_{name}", name)
            for name in ("Shoulder", "Elbow", "Wrist")
        ]
        if self.meshes_requested:
            from viser.extras import ViserUrdf

            self.robot = ViserUrdf(
                self.server,
                Path(self.model.backend.urdf_path),
                root_node_name="/primary/robot",
            )
        sphere = self.trimesh.creation.icosphere(subdivisions=2, radius=1)
        self.sphere_vertices, self.sphere_faces = (
            np.asarray(sphere.vertices),
            np.asarray(sphere.faces),
        )
        self.ellipsoid = scene.add_mesh_simple(
            "/jacobian/ellipsoid",
            self.sphere_vertices,
            self.sphere_faces,
            color=(93, 213, 196),
            wireframe=True,
            opacity=0.42,
            cast_shadow=False,
        )
        self.velocity = self._line(
            "/jacobian/preferred_velocity", np.zeros((1, 2, 3)), (246, 112, 116), 5
        )
        self.ellipsoid_axes = self._line(
            "/jacobian/principal_axes", np.zeros((3, 2, 3)), (104, 224, 202), 2
        )
        self.gallery = []
        for index, branch in enumerate(BRANCHES):
            offset = np.array([(index % 4 - 1.5) * 1.45, (index // 4 - 0.5) * 1.65, 0])
            prefix = f"/gallery/branch{index}"
            bones = self._line(prefix + "/links", zeros, (111, 206, 244), 5)
            points = scene.add_point_cloud(
                prefix + "/joints",
                np.zeros((3, 3)),
                (111, 206, 244),
                point_size=0.025,
                point_shape="circle",
            )
            label = scene.add_label(
                prefix + "/label", branch_label(branch), position=offset + [0, 0, 1.38]
            )
            tcp = scene.add_frame(prefix + "/tcp", axes_length=0.095, axes_radius=0.003)
            self.gallery.append((offset, bones, points, label, tcp))
        self.path_root = scene.add_frame("/trajectory", show_axes=False)
        self.path_reference = self._line(
            "/trajectory/reference", zeros, (170, 178, 193), 2
        )
        self.path_trace = self._line("/trajectory/trace", zeros, (96, 226, 160), 5)
        self.path_plot_reference = self._line(
            "/trajectory/xy_reference", zeros, (170, 178, 193), 2
        )
        self.path_plot_trace = self._line(
            "/trajectory/xy_trace", zeros, (96, 226, 160), 5
        )
        self.path_plot_current = scene.add_point_cloud(
            "/trajectory/xy_current",
            np.zeros((1, 3)),
            (255, 214, 93),
            point_size=0.025,
            point_shape="circle",
        )
        self.path_plot_title = scene.add_label(
            "/trajectory/xy_title", "TCP XY displacement ×6", position=(1.3, 0, 1.4)
        )
        self._line(
            "/trajectory/xy_axes",
            np.array(
                [[[1.5, 0, 0.75], [1.9, 0, 0.75]], [[1.5, 0, 0.75], [1.5, 0, 1.15]]]
            ),
            (120, 130, 150),
            1,
        )
        self.path_plot_labels = (
            scene.add_label("/trajectory/x", "Δx", position=(1.93, 0, 0.75)),
            scene.add_label("/trajectory/y", "Δy", position=(1.5, 0, 1.18)),
        )
        self.manifold_root = scene.add_frame("/manifold", show_axes=False)
        self.manifold_origin = np.array([1.5, 0, 0.75])
        scene.add_frame(
            "/manifold/axes",
            position=self.manifold_origin,
            axes_length=0.72,
            axes_radius=0.004,
        )
        self.manifold_axis_labels = []
        for name, direction in zip(("J1", "J3", "J5"), np.eye(3)):
            self.manifold_axis_labels.append(
                scene.add_label(
                    f"/manifold/{name}",
                    name + " [rad]",
                    position=self.manifold_origin + 0.76 * direction,
                )
            )
        scene.add_label(
            "/manifold/title", "Joint-space projection", position=(1.3, 0, 1.65)
        )
        colors = (
            (95, 195, 240),
            (230, 128, 188),
            (141, 195, 93),
            (244, 176, 80),
            (164, 140, 233),
            (92, 215, 187),
            (242, 111, 107),
            (181, 185, 206),
        )
        self.manifold_curves = [
            self._line(f"/manifold/branch{i}", zeros, color, 3)
            for i, color in enumerate(colors)
        ]
        self.manifold_current = scene.add_point_cloud(
            "/manifold/current",
            np.zeros((1, 3)),
            (255, 214, 93),
            point_size=0.04,
            point_shape="circle",
        )

    def _build_gui(self):
        gui = self.server.gui
        gui.configure_theme(
            dark_mode=True,
            control_width="medium",
            brand_color=(238, 155, 63),
            show_share_button=False,
        )
        gui.add_markdown("## SRS Lab\nTrajectories · self-motion · branches")
        self.mode = gui.add_dropdown(
            "Experiment", tuple(MODES.values()), initial_value=MODES[self.model.mode]
        )
        self.description = gui.add_markdown(DESCRIPTIONS[self.model.mode])
        self.preset = gui.add_dropdown(
            "Pose preset", tuple(PRESETS), initial_value=self.model.preset
        )
        self.branch = gui.add_dropdown(
            "S/E/W branch",
            tuple(map(branch_label, BRANCHES)),
            initial_value=branch_label(self.model.branch),
        )
        self.angle = gui.add_slider("Arm angle offset [deg]", -180, 180, 0.5, 0.0)
        self.play = gui.add_checkbox("Play", False)
        self.amplitude = gui.add_slider(
            "Sweep amplitude [deg]", 5, 90, 1, self.model.amplitude
        )
        self.frequency = gui.add_slider(
            "Frequency [Hz]", 0.05, 0.5, 0.05, self.model.frequency
        )
        self.path_kind = gui.add_dropdown(
            "Cartesian path",
            tuple(PATHS.values()),
            initial_value=PATHS[self.model.path_kind],
        )
        self.path_plane = gui.add_dropdown(
            "Path plane",
            tuple(PLANES.values()),
            initial_value=PLANES[self.model.path_plane],
        )
        self.path_radius = gui.add_slider(
            "Path size [m]", 0.005, 0.1, 0.005, self.model.path_radius
        )
        self.max_joint_speed = gui.add_slider(
            "Joint speed bound [rad/s]", 0.1, 2.0, 0.1, self.model.max_joint_speed
        )
        self.manifold_samples = gui.add_slider("Samples / branch", 17, 145, 8, 73)
        self.manifold_projection = gui.add_dropdown(
            "Projection axes", tuple(PROJECTIONS), initial_value="J1 / J3 / J5"
        )
        self.mesh_toggle = gui.add_checkbox(
            "Show robot mesh", self.robot is not None, disabled=self.robot is None
        )
        self.reset_button = gui.add_button("Reset pose")
        self.metrics = gui.add_markdown("")
        self.branch_table = gui.add_markdown("", visible=self.model.mode == "branches")
        self.joint_sliders = []
        with gui.add_folder("Joint controls / new target", expand_by_default=False):
            for index, q in enumerate(self.model.q):
                handle = gui.add_slider(
                    f"J{index + 1} [deg]",
                    float(np.rad2deg(self.model.backend.lower[index])),
                    float(np.rad2deg(self.model.backend.upper[index])),
                    0.1,
                    float(np.rad2deg(q)),
                )
                self.joint_sliders.append(handle)
                handle.on_update(
                    self._callback(
                        "joint",
                        lambda index=index, handle=handle: (
                            index,
                            np.deg2rad(handle.value),
                        ),
                    )
                )
        for name in (
            "mode",
            "preset",
            "branch",
            "angle",
            "play",
            "amplitude",
            "frequency",
            "mesh_toggle",
            "path_kind",
            "path_plane",
            "path_radius",
            "max_joint_speed",
            "manifold_samples",
            "manifold_projection",
        ):
            handle = getattr(self, name)
            handle.on_update(self._callback(name, lambda handle=handle: handle.value))
        self.reset_button.on_click(self._callback("reset", lambda: None))
        self.target.on_update(self._callback("target", self._target_pose))

    def _callback(self, name, read):
        def callback(event):
            # Server-side synchronization also produces GUI callbacks.
            if event.client is not None:
                self.commands.put((name, read()))

        return callback

    def _target_pose(self):
        pose = self.trimesh.transformations.quaternion_matrix(self.target.wxyz)
        pose[:3, 3] = self.target.position
        return pose

    def _camera(self, client):
        if self.model.mode == "branches":
            client.camera.position = (0.8, -6.0, 5.3)
            client.camera.look_at = (0.8, 0, 0.6)
        elif self.model.mode in ("manifold", "trajectory"):
            client.camera.position = (2.7, -3.3, 2.2)
            client.camera.look_at = (0.7, 0, 0.75)
        else:
            client.camera.position = (1.1, -1.45, 1.25)
            client.camera.look_at = (0, 0, 0.65)
        client.camera.up_direction = (0, 0, 1)

    def apply(self, name, value):
        m = self.model
        if name == "mode":
            m.set_mode(next(key for key, label in MODES.items() if label == value))
            self.description.content = DESCRIPTIONS[m.mode]
            for client in self.server.get_clients().values():
                self._camera(client)
        elif name in ("preset", "reset"):
            m.reset(value if name == "preset" else m.preset)
        elif name == "joint":
            m.play = False
            q = m.q.copy()
            q[value[0]] = value[1]
            m.set_joints(q)
        elif name == "target":
            m.play = False
            m.set_target(value)
        elif name == "branch":
            m.play = False
            m.set_arm_angle(
                m.offset_deg, BRANCHES[tuple(map(branch_label, BRANCHES)).index(value)]
            )
        elif name == "angle":
            m.play = False
            m.set_arm_angle(value)
        elif name == "play":
            m.play = value
        elif name in ("amplitude", "frequency"):
            setattr(m, name, value)
            if name == "amplitude":
                m.restart_sweep()
        elif name in ("path_kind", "path_plane", "path_radius"):
            m.play = False
            choices = {"path_kind": PATHS, "path_plane": PLANES}
            if name in choices:
                value = next(
                    key for key, label in choices[name].items() if label == value
                )
            setattr(m, name, value)
            m.reset_trajectory()
        elif name == "max_joint_speed":
            m.max_joint_speed = value
        elif name == "manifold_samples":
            self.manifold_samples.value = value
        elif name == "manifold_projection":
            self.manifold_projection.value = value
        elif name != "mesh_toggle":
            raise ValueError(f"unknown command {name}")

    def render(self, force_metrics=False):
        m = self.model
        gallery_mode = m.mode == "branches"
        with self.server.atomic():
            self.mode.value = MODES[m.mode]
            self.preset.value = m.preset
            self.path_kind.value = PATHS[m.path_kind]
            self.path_plane.value = PLANES[m.path_plane]
            self.path_radius.value = m.path_radius
            self.amplitude.value = m.amplitude
            self.frequency.value = m.frequency
            self.max_joint_speed.value = m.max_joint_speed
            self.play.value = m.play
            self.branch.value = branch_label(m.branch)
            self.angle.value = float(np.clip(m.offset_deg, -180, 180))
            self.angle.disabled = m.mode in ("jacobian", "trajectory")
            self.amplitude.visible = m.mode not in ("jacobian", "trajectory")
            self.frequency.visible = m.mode != "jacobian"
            self.branch_table.visible = m.mode in ("branches", "manifold")
            self.path_kind.visible = self.path_plane.visible = (
                self.path_radius.visible
            ) = m.mode == "trajectory"
            self.manifold_samples.visible = self.manifold_projection.visible = (
                m.mode == "manifold"
            )
            self.path_root.visible = m.mode == "trajectory"
            self.manifold_root.visible = m.mode == "manifold"
            self.mesh_toggle.visible = not gallery_mode
            for slider, value in zip(self.joint_sliders, m.q):
                slider.value = float(np.rad2deg(value))
            for handle in (
                self.arm,
                self.joints,
                self.circle,
                self.axis,
                self.target,
                self.tcp,
                self.target_label,
                *self.sew_labels,
            ):
                handle.visible = not gallery_mode
            if self.robot is not None:
                self.robot.show_visual = not gallery_mode and self.mesh_toggle.value
                if not gallery_mode:
                    self.robot.update_cfg(m.q)
            self.ellipsoid.visible = self.velocity.visible = (
                self.ellipsoid_axes.visible
            ) = m.mode == "jacobian"
            if gallery_mode:
                solutions = m.gallery()
                chains = m.gallery_chains()
                table = ["| Branch | State | Min margin |", "|:--|:--|--:|"]
                for branch, q, chain, (offset, bones, points, label, tcp) in zip(
                    BRANCHES, solutions, chains, self.gallery
                ):
                    label.visible = tcp.visible = True
                    bones.visible = points.visible = q is not None
                    tcp.wxyz = self.trimesh.transformations.quaternion_from_matrix(
                        m.target
                    )
                    tcp.position = m.target[:3, 3] + offset
                    compact_branch = " ".join(
                        f"{name}{'+' if sign > 0 else '-'}"
                        for name, sign in zip("SEW", branch)
                    )
                    label.text = compact_branch + (
                        " · OK" if q is not None else " · no IK"
                    )
                    if q is not None:
                        locations = (
                            np.vstack((np.zeros(3), chain[[0, 2, 4, 7], :3, 3]))
                            + offset
                        )
                        bones.points = np.stack((locations[:-1], locations[1:]), axis=1)
                        color = np.array(
                            (245, 171, 80) if branch == m.branch else (111, 206, 244),
                            dtype=np.uint8,
                        )
                        bones.colors = np.broadcast_to(color, bones.points.shape).copy()
                        points.points = locations[1:4]
                        margin = np.rad2deg(
                            np.min(np.minimum(q - m.backend.lower, m.backend.upper - q))
                        )
                        table.append(
                            f"| {branch_label(branch)} | available | {margin:.1f}° |"
                        )
                    else:
                        table.append(f"| {branch_label(branch)} | unavailable | — |")
                self.branch_table.content = "\n".join(table)
            else:
                for _, bones, points, label, tcp in self.gallery:
                    bones.visible = points.visible = label.visible = tcp.visible = False
                chain = m.chain()
                locations = np.vstack((np.zeros(3), chain[[0, 2, 4, 7], :3, 3]))
                self.arm.points = np.stack((locations[:-1], locations[1:]), axis=1)
                self.joints.points = locations[1:4]
                for handle, point in zip(self.sew_labels, locations[1:4]):
                    handle.position = point + [0.04, 0, 0.02]
                circle = elbow_circle(chain)
                self.circle.visible = len(circle) > 1
                if len(circle) > 1:
                    self.circle.points = np.stack((circle[:-1], circle[1:]), axis=1)
                self.axis.points = np.array([[locations[1], locations[3]]])
                self.target.wxyz = self.trimesh.transformations.quaternion_from_matrix(
                    m.target
                )
                self.target.position = m.target[:3, 3]
                self.tcp.wxyz = self.trimesh.transformations.quaternion_from_matrix(
                    chain[-1]
                )
                self.tcp.position = chain[-1, :3, 3]
                self.target_label.position = m.target[:3, 3] + [0, 0, 0.12]
                if m.mode == "jacobian":
                    jacobian, preferred, _ = m.velocities()
                    axes, radii, _ = np.linalg.svd(jacobian[:3], full_matrices=False)
                    shape = axes * (radii * 0.25)
                    origin = chain[-1, :3, 3]
                    self.ellipsoid.vertices = (
                        self.sphere_vertices @ shape.T + origin
                    ).astype(np.float32)
                    self.ellipsoid_axes.points = np.stack(
                        (origin - shape.T, origin + shape.T), axis=1
                    )
                    self.velocity.points = np.array(
                        [[origin, origin + 0.25 * (jacobian[:3] @ preferred)]]
                    )
                elif m.mode == "trajectory":
                    points = m.trajectory.points
                    self.path_reference.points = np.stack(
                        (points[:-1], points[1:]), axis=1
                    )
                    trail = np.asarray(m.trail)
                    self.path_trace.visible = len(trail) > 1
                    if len(trail) > 1:
                        self.path_trace.points = np.stack(
                            (trail[:-1], trail[1:]), axis=1
                        )

                    def project(values):
                        delta = m.trajectory.project(values)
                        return np.column_stack(
                            (delta[:, 0], np.zeros(len(delta)), delta[:, 1])
                        ) * 6 + [1.5, 0, 0.75]

                    self.path_plot_title.text = (
                        f"TCP displacement ×6 · {PLANES[m.path_plane]}"
                    )
                    for label, coordinate in zip(
                        self.path_plot_labels, m.trajectory.coordinate_labels
                    ):
                        label.text = "Δ" + coordinate
                    reference, actual = project(points), project(trail)
                    self.path_plot_reference.points = np.stack(
                        (reference[:-1], reference[1:]), axis=1
                    )
                    self.path_plot_trace.visible = len(actual) > 1
                    if len(actual) > 1:
                        self.path_plot_trace.points = np.stack(
                            (actual[:-1], actual[1:]), axis=1
                        )
                    self.path_plot_current.points = actual[-1:]
                    self.target.visible = not m.play
                    self.circle.visible = self.axis.visible = False
                elif m.mode == "manifold":
                    axes = PROJECTIONS[self.manifold_projection.value]
                    samples = int(self.manifold_samples.value)
                    key = (m.generation, samples, axes)
                    if key != self._manifold_render_key:
                        _, valid, joints = m.manifold(samples)
                        table = ["| Branch | Feasible samples |", "|:--|--:|"]
                        for branch, mask, q, handle in zip(
                            BRANCHES, valid, joints, self.manifold_curves
                        ):
                            segments = manifold_segments(q, mask, axes=axes)
                            handle.visible = len(segments) > 0
                            if len(segments):
                                handle.points = segments
                            table.append(
                                f"| {branch_label(branch)} | {mask.sum()} / {len(mask)} |"
                            )
                        self._manifold_table = "\n".join(table)
                        for label, axis in zip(self.manifold_axis_labels, axes):
                            label.text = f"J{axis + 1} [rad]"
                        self._manifold_render_key = key
                    self.branch_table.content = self._manifold_table
                    self.manifold_current.points = (
                        m.q[None, list(axes)] * 0.22 + self.manifold_origin
                    )
            now = time.perf_counter()
            if force_metrics or now - self.last_metrics >= 0.1:
                d = m.diagnostics()
                self.metrics.content = (
                    f"**{m.message}**\n\n"
                    "| Metric | Value |\n|:--|--:|\n"
                    f"| Backend | {m.backend.label} |\n"
                    f"| TCP position error | {d['position_m'] * 1000:.5f} mm |\n"
                    f"| TCP orientation error | {np.rad2deg(d['rotation_rad']):.5f}° |\n"
                    f"| Joint-limit margin | {np.rad2deg(d['margin_rad']):.2f}° |\n"
                    f"| Centering objective | {d['centering_cost']:.6f} |\n"
                    f"| ‖J q̇_null‖ | {d['null_residual']:.2e} |"
                )
                if m.mode == "trajectory":
                    self.metrics.content += f"\n| Accepted steps | {m.trajectory_steps} |\n| Peak joint step | {m.peak_joint_step:.5f} rad |"
                self.last_metrics = now
        self.render_count += 1

    def step(self, dt=None):
        now = time.perf_counter()
        dt = max(1e-6, now - self.last_time) if dt is None else dt
        self.last_time = now
        dirty = False
        pending = self.commands.drain()
        for name, value in pending:
            self.apply(name, value)
            dirty = True
        was_playing = self.model.play
        dirty = self.model.advance(dt) or dirty or was_playing
        if dirty:
            stopped = was_playing and not self.model.play
            self.render(force_metrics=bool(pending) or stopped)
        return dirty

    def close(self):
        if not self.closed:
            self.server.stop()
            self.closed = True


def main(default_demo="arm-angle"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=tuple(MODES), default=default_demo)
    parser.add_argument(
        "--backend",
        choices=("numpy", "warp"),
        default="warp" if "warp" in __package__ else "numpy",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--no-mesh", action="store_true")
    parser.add_argument("--autoplay", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--validate-only", action="store_true")
    action.add_argument(
        "--export-csv",
        type=Path,
        help="Record playback without Viser to a new CSV file",
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="Bent arm")
    parser.add_argument("--path", choices=tuple(PATHS), default="figure-eight")
    parser.add_argument("--plane", choices=tuple(PLANES), default="world-xy")
    parser.add_argument(
        "--path-size",
        type=float,
        default=0.035,
        help="Path size in meters [0.005, 0.1]",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=0.2,
        help="Reference frequency in Hz [0.05, 0.5]",
    )
    parser.add_argument(
        "--joint-speed",
        type=float,
        default=1.0,
        help="Playback joint-speed bound in rad/s [0.1, 2]",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=35.0,
        help="Arm-angle sweep amplitude in degrees [5, 90]",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Requested playback steps for CSV export",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="GUI frame rate or CSV simulation rate [20, 240]",
    )
    parser.add_argument("--validation-frames", type=int, default=40)
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Stop after this many seconds; 0 runs until Ctrl+C",
    )
    args = parser.parse_args()
    for name, lower, upper in (
        ("path_size", 0.005, 0.1),
        ("frequency", 0.05, 0.5),
        ("joint_speed", 0.1, 2.0),
        ("amplitude", 5, 90),
        ("fps", 20, 240),
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or not lower <= value <= upper:
            parser.error(
                f"{name.replace('_', '-')} must be finite and between {lower} and {upper}"
            )
    if (
        args.validation_frames < 2
        or args.frames < 1
        or not np.isfinite(args.duration)
        or args.duration < 0
    ):
        parser.error(
            "validation-frames must be >= 2, frames must be positive and duration must be finite and non-negative"
        )
    backend = Backend(args.backend, args.device)
    if args.validate_only:
        print(json.dumps(validate(backend, args.validation_frames), indent=2))
        return
    experiment = Experiment(backend, args.demo)
    experiment.path_kind, experiment.path_plane = args.path, args.plane
    experiment.path_radius, experiment.frequency = args.path_size, args.frequency
    experiment.max_joint_speed, experiment.amplitude = args.joint_speed, args.amplitude
    experiment.reset(args.preset)
    if args.export_csv is not None:
        from .recording import write_playback

        try:
            with args.export_csv.open("x", newline="", encoding="utf-8") as output:
                report = write_playback(experiment, output, args.frames, 1 / args.fps)
        except OSError as error:
            parser.exit(2, f"Cannot write recording: {error}\n")
        report["csv"] = str(args.export_csv)
        print(json.dumps(report, indent=2))
        if not report["completed"]:
            raise SystemExit(2)
        return
    app = LabApp(
        backend,
        args.demo,
        args.host,
        args.port,
        not args.no_mesh,
        experiment=experiment,
    )
    app.model.play = args.autoplay
    started = time.perf_counter()
    print(f"SRS lab ({args.demo}): http://{args.host}:{app.server.get_port()}")
    try:
        while args.duration == 0 or time.perf_counter() - started < args.duration:
            frame_start = time.perf_counter()
            app.step()
            time.sleep(max(0.0, 1 / args.fps - (time.perf_counter() - frame_start)))
    except KeyboardInterrupt:
        pass
    finally:
        app.close()


if __name__ == "__main__":
    main()
