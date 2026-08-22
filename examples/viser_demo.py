#!/usr/bin/env python3
"""Interactive Viser FK/IK demo for the bundled KUKA iiwa model."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import trimesh
import viser
from yourdfpy import URDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuka_iiwa_solver import KUKAiiwaSolver  # noqa: E402


def pose_components(transform: np.ndarray):
    return (
        trimesh.transformations.quaternion_from_matrix(transform),
        transform[:3, 3],
    )


def pose_from_control(control) -> np.ndarray:
    pose = trimesh.transformations.quaternion_matrix(control.wxyz)
    pose[:3, 3] = control.position
    return pose


def pose_error(actual: np.ndarray, target: np.ndarray):
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    cosine = np.clip(
        (np.trace(actual[:3, :3].T @ target[:3, :3]) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return position, float(np.rad2deg(np.arccos(cosine)))


def benchmark(solver: KUKAiiwaSolver, samples: int, num_samples: int) -> dict:
    if samples < 1:
        raise ValueError("validation-samples must be positive")
    rng = np.random.default_rng(6305)
    lower = np.maximum(solver.lower_position_limits, -1.2)
    upper = np.minimum(solver.upper_position_limits, 1.2)
    durations, position_errors, rotation_errors = [], [], []
    successes = 0
    for joints in rng.uniform(lower, upper, size=(samples, 7)):
        target = solver.get_fk(joints)
        # Use a deterministic perturbed seed so the benchmark measures IK
        # search rather than the stationary-target fast path.
        seed = np.clip(
            joints + rng.normal(0.0, 0.2, size=7),
            solver.lower_position_limits,
            solver.upper_position_limits,
        )
        started = time.perf_counter()
        ok, solution = solver.get_ik(target, seed, num_samples=num_samples)
        durations.append((time.perf_counter() - started) * 1e3)
        if not ok:
            continue
        successes += 1
        position, rotation = pose_error(solver.get_fk(solution), target)
        position_errors.append(position)
        rotation_errors.append(rotation)
    return {
        "samples": samples,
        "successes": successes,
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "max_position": max(position_errors, default=float("nan")),
        "max_rotation": max(rotation_errors, default=float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--num-samples", type=int, default=73)
    parser.add_argument(
        "--search-mode", choices=("continuous", "global"),
        default="continuous",
    )
    parser.add_argument("--validation-samples", type=int, default=24)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    solver = KUKAiiwaSolver()
    validation = benchmark(solver, args.validation_samples, args.num_samples)
    if args.validate_only:
        print(
            f"IK {validation['successes']}/{validation['samples']}, "
            f"median/p95={validation['median_ms']:.3f}/"
            f"{validation['p95_ms']:.3f} ms, max residual="
            f"{validation['max_position'] * 1e3:.3e} mm/"
            f"{validation['max_rotation']:.3e} deg"
        )
        return

    urdf_path = ROOT / "urdf" / "iiwa_7.urdf"
    # yourdfpy derives the mesh directory only for string paths (not Path).
    robot = URDF.load(str(urdf_path), load_meshes=True, build_scene_graph=True)
    joints = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    robot.update_cfg(dict(zip(robot.actuated_joint_names, joints)))

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid(
        "/world/grid", width=3.0, height=3.0, cell_size=0.1,
        section_size=0.5, plane_opacity=0.08, shadow_opacity=0.15,
    )
    server.scene.add_frame("/world/base", axes_length=0.16, axes_radius=0.006)

    mesh_handles = {}
    for node_name in robot.scene.graph.nodes_geometry:
        transform, geometry_name = robot.scene.graph[node_name]
        mesh = robot.scene.geometry[geometry_name]
        wxyz, position = pose_components(transform)
        mesh_handles[node_name] = server.scene.add_mesh_trimesh(
            f"/robot/{node_name}", mesh, wxyz=wxyz, position=position
        )

    initial_pose = solver.get_fk(joints)
    wxyz, position = pose_components(initial_pose)
    target = server.scene.add_transform_controls(
        "/target", scale=0.18, wxyz=wxyz, position=position
    )
    actual_frame = server.scene.add_frame(
        "/actual_tcp", axes_length=0.13, axes_radius=0.005,
        wxyz=wxyz, position=position,
    )
    target_marker = server.scene.add_icosphere(
        "/target_marker", radius=0.014, color=(255, 155, 45),
        position=position,
    )
    actual_marker = server.scene.add_icosphere(
        "/actual_marker", radius=0.012, color=(45, 175, 255),
        position=position,
    )
    error_line = server.scene.add_line_segments(
        "/position_error", np.array([[position, position]]),
        colors=(235, 70, 70), line_width=3.0,
    )

    sliders = []
    with server.gui.add_folder("Joint control"):
        for index, value in enumerate(joints):
            sliders.append(
                server.gui.add_slider(
                    f"J{index + 1} [deg]",
                    min=float(np.rad2deg(solver.lower_position_limits[index])),
                    max=float(np.rad2deg(solver.upper_position_limits[index])),
                    step=0.1,
                    initial_value=float(np.rad2deg(value)),
                )
            )
        reset = server.gui.add_button("Reset")
    with server.gui.add_folder("Null-space motion"):
        animate_null_space = server.gui.add_checkbox(
            "Play fixed-TCP arm-angle motion", initial_value=False
        )
        null_range = server.gui.add_slider(
            "Arm-angle range [deg]", min=1.0, max=60.0, step=1.0,
            initial_value=25.0,
        )
        null_speed = server.gui.add_slider(
            "Frequency [Hz]", min=0.05, max=1.0, step=0.05,
            initial_value=0.2,
        )
    status = server.gui.add_markdown("")
    performance = server.gui.add_markdown("")
    frame_times = deque(maxlen=120)
    compute_times = deque(maxlen=120)
    state = {
        "target": initial_pose.copy(),
        "solve_ms": 0.0,
        "solved": True,
        "busy": False,
        "animation_started": time.perf_counter(),
        "animation_target": initial_pose.copy(),
        "animation_configuration": solver.get_configuration(joints),
    }

    def update_scene(compute_started=None) -> None:
        robot.update_cfg(dict(zip(robot.actuated_joint_names, joints)))
        for node_name, handle in mesh_handles.items():
            transform, _ = robot.scene.graph[node_name]
            handle.wxyz, handle.position = pose_components(transform)
        actual = solver.get_fk(joints)
        actual_wxyz, actual_position = pose_components(actual)
        target_position = state["target"][:3, 3]
        actual_frame.wxyz, actual_frame.position = actual_wxyz, actual_position
        actual_marker.position = actual_position
        target_marker.position = target_position
        error_line.points = np.array([[actual_position, target_position]])
        position_error, rotation_error = pose_error(actual, state["target"])
        configuration = solver.get_configuration(joints)
        result = "solved" if state["solved"] else "unreachable"
        status.content = (
            f"### SRS analytical IK — {result}\n"
            "🟠 Target · 🔵 Actual · 🔴 Position residual\n\n"
            "| Metric | Value |\n|:--|--:|\n"
            f"| Branch S/E/W | **{configuration.shoulder:+d} / "
            f"{configuration.elbow:+d} / {configuration.wrist:+d}** |\n"
            f"| Arm angle | **{np.rad2deg(configuration.redundancy):.2f}°** |\n"
            f"| Residual | **{position_error * 1e3:.4f} mm / "
            f"{rotation_error:.5f}°** |\n"
            f"| Last IK solve | **{state['solve_ms']:.3f} ms** |\n\n"
            f"Startup validation: **{validation['successes']}/"
            f"{validation['samples']}** solved, median/p95 **"
            f"{validation['median_ms']:.3f}/{validation['p95_ms']:.3f} ms**, "
            f"max residual **{validation['max_position'] * 1e3:.2e} mm / "
            f"{validation['max_rotation']:.2e}°**"
        )
        completed = time.perf_counter()
        if compute_started is not None:
            compute_times.append((completed - compute_started) * 1e3)
        frame_times.append(completed)
        intervals = np.diff(frame_times)
        fps = 1.0 / np.mean(intervals) if len(intervals) else 0.0
        performance.content = (
            "### Visualization performance\n"
            f"- Update FPS: **{fps:.1f}**\n"
            f"- Scene update avg/max: **{np.mean(compute_times) if compute_times else 0:.2f} / "
            f"{np.max(compute_times) if compute_times else 0:.2f} ms**"
        )

    def sync_sliders() -> None:
        for slider, value in zip(sliders, joints):
            slider.value = float(np.rad2deg(value))

    for index, slider in enumerate(sliders):
        @slider.on_update
        def _on_joint(_event, joint_index=index, handle=slider):
            if state["busy"]:
                return
            started = time.perf_counter()
            joints[joint_index] = np.deg2rad(handle.value)
            state["target"] = solver.get_fk(joints)
            state["solved"] = True
            target.wxyz, target.position = pose_components(state["target"])
            update_scene(started)

    @target.on_update
    def _on_target(_event):
        if state["busy"]:
            return
        state["busy"] = True
        started = time.perf_counter()
        try:
            state["target"] = pose_from_control(target)
            solve_started = time.perf_counter()
            ok, solution = solver.get_ik(
                state["target"], joints.copy(), num_samples=args.num_samples,
                search_mode=args.search_mode,
            )
            state["solve_ms"] = (time.perf_counter() - solve_started) * 1e3
            state["solved"] = ok
            if ok:
                joints[:] = solution
                sync_sliders()
            update_scene(started)
        finally:
            state["busy"] = False

    @reset.on_click
    def _on_reset(_event):
        state["busy"] = True
        try:
            joints[:] = [0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]
            state["target"] = solver.get_fk(joints)
            state["solved"] = True
            sync_sliders()
            target.wxyz, target.position = pose_components(state["target"])
            update_scene(time.perf_counter())
        finally:
            state["busy"] = False

    @animate_null_space.on_update
    def _on_animation_toggle(_event):
        if animate_null_space.value:
            state["animation_started"] = time.perf_counter()
            state["animation_target"] = solver.get_fk(joints)
            state["animation_configuration"] = solver.get_configuration(joints)

    update_scene()
    print(f"Viser demo: http://localhost:{args.port} (Ctrl+C to stop)")
    try:
        while True:
            if animate_null_space.value and not state["busy"]:
                state["busy"] = True
                frame_started = time.perf_counter()
                try:
                    elapsed = frame_started - state["animation_started"]
                    base = state["animation_configuration"]
                    psi = base.redundancy + np.deg2rad(null_range.value) * np.sin(
                        2.0 * np.pi * null_speed.value * elapsed
                    )
                    selected = type(base)(
                        base.shoulder, base.elbow, base.wrist, psi
                    )
                    ok, solution = solver.solve_configuration(
                        state["animation_target"], selected, joints
                    )
                    state["solved"] = ok
                    state["target"] = state["animation_target"]
                    if ok:
                        joints[:] = solution
                        sync_sliders()
                    update_scene(frame_started)
                finally:
                    state["busy"] = False
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
