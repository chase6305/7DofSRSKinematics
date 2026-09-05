"""Stream reproducible fixed-step playback to CSV without viewer dependencies."""

import csv

import numpy as np

from .model import pose_error

CSV_COLUMNS = (
    ("time_s",)
    + tuple(f"q{i + 1}_rad" for i in range(7))
    + tuple(f"{pose}_{axis}_m" for pose in ("tcp", "target") for axis in "xyz")
    + tuple(
        f"{pose}_r{row}{col}"
        for pose in ("tcp", "target")
        for row in range(3)
        for col in range(3)
    )
    + (
        "position_error_m",
        "rotation_error_rad",
        "joint_limit_margin_rad",
        "joint_speed_peak_rad_s",
        "phase_rad",
        "arm_angle_rad",
        "shoulder",
        "elbow",
        "wrist",
    )
)


def write_playback(model, output, frames=300, dt=1 / 30):
    """Write the initial pose and accepted frames; return completion diagnostics.

    Time is simulated time, not wall time. ``dt`` must fit the experiment's 50 ms
    integration cap so recorded time and velocity use the actual integration step.
    A blocked step ends the recording without adding a duplicate, unexecuted row.
    Once recording starts, the experiment is paused on return, including when
    writing or numerical validation fails.
    """
    if (
        isinstance(frames, bool)
        or not isinstance(frames, (int, np.integer))
        or frames < 1
    ):
        raise ValueError("frames must be a positive integer")
    if not np.isfinite(dt) or not 0 < dt <= 0.05:
        raise ValueError("recording dt must be positive and at most 0.05 s")
    report = {
        "backend": model.backend.label,
        "mode": model.mode,
        "preset": model.preset,
        "path": model.path_kind,
        "plane": model.path_plane,
        "path_size_m": model.path_radius,
        "frequency_hz": model.frequency,
        "amplitude_deg": model.amplitude,
        "joint_speed_bound_rad_s": model.max_joint_speed,
        "requested_frames": int(frames),
        "recorded_frames": 0,
        "dt_s": float(dt),
        "simulated_seconds": 0.0,
        "max_position_error_m": 0.0,
        "max_rotation_error_rad": 0.0,
        "peak_joint_speed_rad_s": 0.0,
        "completed": False,
        "stop_reason": None,
    }
    previous = model.q.copy()
    try:
        writer = csv.writer(output)
        writer.writerow(CSV_COLUMNS)
        model.play = True
        for frame in range(frames + 1):
            if frame and not model.advance(dt):
                report["stop_reason"] = model.message
                break
            actual = model.backend.fk(model.q)
            position, rotation = pose_error(actual, model.target)
            margin = float(
                np.min(
                    np.minimum(
                        model.q - model.backend.lower, model.backend.upper - model.q
                    )
                )
            )
            speed = float(np.max(np.abs(model.q - previous)) / dt) if frame else 0.0
            phase = (
                model.trajectory.phase if model.mode == "trajectory" else model.phase
            )
            row = (
                [frame * dt]
                + model.q.tolist()
                + actual[:3, 3].tolist()
                + model.target[:3, 3].tolist()
                + actual[:3, :3].ravel().tolist()
                + model.target[:3, :3].ravel().tolist()
                + [
                    position,
                    rotation,
                    margin,
                    speed,
                    phase,
                    model.anchor_psi + np.deg2rad(model.offset_deg),
                ]
                + list(model.branch)
            )
            if (
                not np.all(np.isfinite(row))
                or position > 2e-4
                or rotation > 2e-4
                or margin < -1e-7
                or speed > model.max_joint_speed + 1e-8 / dt
            ):
                raise AssertionError(
                    "recording exceeded pose, joint-limit or speed tolerance"
                )
            writer.writerow(row)
            previous = model.q.copy()
            report["recorded_frames"] = frame
            report["simulated_seconds"] = frame * dt
            report["max_position_error_m"] = max(
                report["max_position_error_m"], position
            )
            report["max_rotation_error_rad"] = max(
                report["max_rotation_error_rad"], rotation
            )
            report["peak_joint_speed_rad_s"] = max(
                report["peak_joint_speed_rad_s"], speed
            )
        report["completed"] = report["recorded_frames"] == frames
        return report
    finally:
        model.play = False
