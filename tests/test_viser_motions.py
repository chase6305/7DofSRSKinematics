"""Trajectory continuity and the fixed-TCP manifold without visualization extras."""

import numpy as np
import pytest

from srs_viser.backend import Backend
from srs_viser.model import PRESETS, Experiment, pose_error
from srs_viser.motions import (
    PATHS,
    PLANES,
    CartesianPath,
    bounded_ik_step,
    manifold_segments,
)


@pytest.fixture(params=["numpy"])
def backend(request):
    return Backend(request.param)


@pytest.mark.parametrize("kind", PATHS)
def test_full_trajectory_cycles_hold_orientation_limits_and_joint_speed(backend, kind):
    model = Experiment(backend, "trajectory")
    model.path_kind = kind
    model.reset_trajectory()
    start = model.target.copy()
    branch = model.branch
    dt = 1 / 30
    for _ in range(240):
        before = model.q.copy()
        assert model.follow_trajectory(dt)
        assert np.max(np.abs(model.q - before)) <= model.max_joint_speed * dt + 1e-8
        assert model.branch == branch
        assert np.all(model.q >= backend.lower) and np.all(model.q <= backend.upper)
        np.testing.assert_array_equal(model.target[:3, :3], start[:3, :3])
        position, rotation = pose_error(backend.fk(model.q), model.target)
        assert position < 2e-6 and rotation < 3e-6
    assert model.trajectory_steps == 240
    assert len(model.trail) == 241
    np.testing.assert_allclose(
        model.trajectory.points[0], model.trajectory.points[-1], atol=1e-14
    )


def test_blocked_trajectory_pauses_without_jumping_and_mode_switch_reanchors(
    backend, monkeypatch
):
    model = Experiment(backend, "trajectory")
    assert model.follow_trajectory(1 / 30)
    q, target, phase = model.q.copy(), model.target.copy(), model.trajectory.phase
    monkeypatch.setattr(backend, "solve", lambda *args: None)
    model.play = True
    assert not model.advance(1 / 30)
    assert not model.play
    np.testing.assert_array_equal(model.q, q)
    np.testing.assert_array_equal(model.target, target)
    assert model.trajectory.phase == phase
    assert "paused" in model.message
    model.set_mode("manifold")
    np.testing.assert_array_equal(model.anchor_q, q)
    assert max(pose_error(model.target, target)) < 3e-6


def test_manifold_keeps_tcp_and_cache_and_excludes_invalid_segments(backend):
    model = Experiment(backend, "manifold")
    data = model.manifold(41)
    angles, valid, joints = data
    assert valid.shape == (8, 41) and joints.shape == (8, 41, 7)
    assert 0 < valid.sum() < valid.size
    for q in joints[valid]:
        position, rotation = pose_error(backend.fk(q), model.target)
        assert position < 2e-6 and rotation < 4e-6
    model.set_arm_angle(10)
    assert model.manifold(41) is data
    for mask, q in zip(valid, joints):
        segments = manifold_segments(q, mask)
        assert segments.shape[1:] == (2, 3)
    q = np.zeros((5, 7))
    q[3:, 0] = 2 * np.pi
    # A missing sample and a wrap jump must both break the plot.
    assert len(manifold_segments(q, np.array([True, False, True, True, True]))) == 1
    with pytest.raises(ValueError):
        model.manifold(2)
    for array in data:
        assert not array.flags.writeable


def test_tight_speed_bound_reduces_path_progress(backend):
    model = Experiment(backend, "trajectory")
    model.max_joint_speed = 0.1
    before = model.q.copy()
    dt = 1 / 30
    assert model.follow_trajectory(dt)
    assert 0 < model.trajectory.phase < 2 * np.pi * model.frequency * dt
    assert np.max(np.abs(model.q - before)) <= model.max_joint_speed * dt + 1e-8


def test_large_path_with_low_speed_keeps_making_feasible_progress(backend):
    model = Experiment(backend, "trajectory")
    model.path_radius, model.frequency, model.max_joint_speed = 0.1, 0.5, 0.1
    model.reset_trajectory()
    dt = 1 / 30
    for _ in range(40):
        before, phase = model.q.copy(), model.trajectory.phase
        assert model.follow_trajectory(dt)
        assert model.trajectory.phase > phase
        assert np.max(np.abs(model.q - before)) <= model.max_joint_speed * dt + 1e-8
        assert max(pose_error(backend.fk(model.q), model.target)) < 3e-6


@pytest.mark.parametrize("mode", ["arm-angle", "branches", "manifold"])
@pytest.mark.parametrize("preset", PRESETS)
def test_fast_self_motion_sweep_preserves_tcp_branch_and_speed(backend, mode, preset):
    model = Experiment(backend, mode)
    model.reset(preset)
    model.amplitude, model.frequency, model.play = 90, 0.5, True
    target, branch = model.target.copy(), model.branch
    for _ in range(150):
        before = model.q.copy()
        assert model.advance(1 / 30)
        assert np.max(np.abs(model.q - before)) <= model.max_joint_speed / 30 + 1e-8
        assert max(pose_error(backend.fk(model.q), target)) < 3e-6
        assert model.branch == branch
        assert np.all(model.q >= backend.lower) and np.all(model.q <= backend.upper)


def test_self_motion_stops_before_real_infeasible_interval(backend):
    model = Experiment(backend, "manifold")
    assert model.set_arm_angle(134)
    model.play_center = model.offset_deg
    model.amplitude, model.frequency, model.play = 90, 0.5, True
    target = model.target.copy()
    for _ in range(50):
        before, phase, offset = model.q.copy(), model.phase, model.offset_deg
        if not model.advance(1 / 30):
            np.testing.assert_array_equal(model.q, before)
            np.testing.assert_array_equal(model.target, target)
            assert model.phase == phase and model.offset_deg == offset
            assert model.solved and not model.play
            assert "paused" in model.message
            break
        assert np.max(np.abs(model.q - before)) <= model.max_joint_speed / 30 + 1e-8
    else:
        pytest.fail("playback crossed an infeasible arm-angle interval")
    assert 134 < model.offset_deg < 135
    assert max(pose_error(backend.fk(model.q), target)) < 3e-6


def test_play_does_not_resume_from_an_unavailable_manual_selection(backend):
    model = Experiment(backend, "manifold")
    before = model.q.copy()
    assert not model.set_arm_angle(136)
    model.play, model.play_center = True, model.offset_deg
    assert not model.advance(1 / 30)
    assert not model.play
    np.testing.assert_array_equal(model.q, before)
    assert model.offset_deg == 136


def test_full_joint_turn_and_nonfinite_ik_results_are_not_accepted():
    q = np.zeros(7)
    for result in (q + 2 * np.pi, np.full(7, np.nan), np.full(7, np.inf)):
        assert bounded_ik_step(q, 1 / 30, 1, lambda _: result) is None


def test_centering_honors_a_tight_joint_speed_bound(backend):
    model = Experiment(backend, "jacobian")
    model.max_joint_speed, model.play = 0.005, True
    for _ in range(25):
        before, cost = model.q.copy(), model.centering_cost()
        assert model.advance(1 / 30)
        assert np.max(np.abs(model.q - before)) <= model.max_joint_speed / 30 + 1e-8
        assert model.centering_cost() < cost
        assert max(pose_error(backend.fk(model.q), model.target)) < 3e-6


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("kind", PATHS)
def test_trajectory_planes_preserve_orientation_and_reference_coordinates(
    backend, plane, kind
):
    model = Experiment(backend, "trajectory")
    model.path_plane, model.path_kind, model.path_radius = plane, kind, 0.015
    model.reset_trajectory()
    start = model.target.copy()
    path = model.trajectory
    axes = {
        "world-xy": (0, 1),
        "world-xz": (0, 2),
        "world-yz": (1, 2),
        "tool-xy": (0, 1),
    }[plane]
    basis = (start[:3, :3] if plane == "tool-xy" else np.eye(3))[:, axes]
    phase = np.linspace(0, 2 * np.pi, 21)
    expected = np.column_stack((0.015 * np.sin(phase), np.zeros(len(phase))))
    if kind == "circle":
        expected[:, 1] = 0.015 * (np.cos(phase) - 1)
    elif kind == "figure-eight":
        expected[:, 1] = 0.0075 * np.sin(2 * phase)
    np.testing.assert_allclose(
        path.positions(phase), start[:3, 3] + expected @ basis.T, atol=1e-14
    )
    np.testing.assert_allclose(path.project(path.positions(phase)), expected, atol=1e-8)
    assert not path.start.flags.writeable and not path.basis.flags.writeable
    for _ in range(180):
        before = model.q.copy()
        assert model.follow_trajectory(1 / 30)
        np.testing.assert_array_equal(model.target[:3, :3], start[:3, :3])
        assert np.max(np.abs(model.q - before)) <= model.max_joint_speed / 30 + 1e-8
        assert max(pose_error(backend.fk(model.q), model.target)) < 3e-6


@pytest.mark.parametrize(
    "start",
    [
        np.zeros((3, 3)),
        np.full((4, 4), np.nan),
        np.diag([1, 1, -1, 1]),
        np.diag([2, 1, 1, 1]),
    ],
)
def test_path_rejects_invalid_reference_poses(start):
    with pytest.raises(ValueError, match="pose"):
        CartesianPath(start)
