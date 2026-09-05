"""Numerical guarantees of the interactive experiments, without GUI extras."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from srs_viser.backend import Backend
from srs_viser.model import (
    BRANCHES,
    PRESETS,
    Experiment,
    elbow_circle,
    pose_error,
    validate,
)


@pytest.fixture(params=["numpy"])
def backend(request):
    return Backend(request.param)


def test_all_experiments_keep_tcp_and_limits(backend):
    report = validate(backend, frames=16)
    assert report["arm_angle_solutions"] == 16
    assert report["branch_solutions"] == 8
    assert report["centering_steps"] == 16
    assert report["self_motion_steps"] == 16
    assert 0 < report["peak_self_motion_speed_rad_s"] <= 1 + 3e-7
    assert report["centering_cost_after"] < report["centering_cost_before"]
    assert report["max_position_m"] < 2e-6
    assert report["max_rotation_rad"] < 2e-6


@pytest.mark.parametrize("preset", PRESETS)
def test_elbow_circle_preserves_both_link_lengths(backend, preset):
    chain = backend.chains(PRESETS[preset][None])[0]
    shoulder, elbow, wrist = chain[[0, 2, 4], :3, 3]
    circle = elbow_circle(chain)
    for center in (shoulder, wrist):
        np.testing.assert_allclose(
            np.linalg.norm(circle - center, axis=1),
            np.linalg.norm(elbow - center),
            atol=2e-7,
            rtol=0,
        )
    np.testing.assert_allclose(circle[len(circle) // 2], elbow, atol=2e-7)
    np.testing.assert_allclose(circle[0], circle[-1], atol=2e-7)


def test_gallery_reuses_solutions_until_target_or_angle_changes(backend, monkeypatch):
    model = Experiment(backend)
    solve = backend.solve_many
    calls = []

    def counted(*args):
        calls.append(args[1])
        return solve(*args)

    monkeypatch.setattr(backend, "solve_many", counted)
    gallery = model.gallery()
    assert len(calls) == 1
    assert model.gallery() is gallery
    model.branch = BRANCHES[0]
    assert model.gallery() is gallery
    assert len(calls) == 1
    model.set_arm_angle(10)
    model.gallery()
    assert len(calls) == 2
    model.reset("Elbow down")
    model.gallery()
    assert len(calls) == 3


def test_failed_requests_retain_last_feasible_pose(backend, monkeypatch):
    model = Experiment(backend)
    q, target = model.q.copy(), model.target.copy()
    unreachable = target.copy()
    unreachable[0, 3] = 10
    assert not model.set_target(unreachable)
    np.testing.assert_array_equal(model.target, target)
    monkeypatch.setattr(backend, "solve", lambda *args: None)
    assert not model.set_arm_angle(30)
    np.testing.assert_array_equal(model.q, q)
    np.testing.assert_array_equal(model.target, target)
    with pytest.raises(ValueError):
        model.set_arm_angle(0, (0, 1, 1))


def test_centering_is_monotone_and_preserves_target(backend):
    model = Experiment(backend, "jacobian")
    model.play = True
    target = model.target.copy()
    for _ in range(25):
        cost = model.centering_cost()
        model.advance(1 / 30)
        assert model.centering_cost() <= cost
        position, rotation = pose_error(backend.fk(model.q), target)
        assert position < 2e-6
        assert rotation < 2e-6
    q = model.q.copy()
    model.play = False
    assert not model.advance(1 / 30)
    np.testing.assert_array_equal(model.q, q)


def test_playback_caps_stalled_frame_and_starts_from_current_angle(backend):
    model = Experiment(backend)
    model.set_arm_angle(20)
    model.play = True
    model.play_center = model.offset_deg
    model.advance(100)
    expected = 20 + model.amplitude * np.sin(2 * np.pi * model.frequency * 0.05)
    assert model.offset_deg == pytest.approx(expected)


def test_lab_import_does_not_load_viewer_dependencies():
    code = "import sys; import srs_viser.app; assert not {'viser', 'trimesh', 'yourdfpy'} & set(sys.modules)"
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1] / "src",
        check=True,
    )


def test_render_and_integration_share_kinematics_without_stale_results(
    backend, monkeypatch
):
    model = Experiment(backend, "jacobian")
    calls = {"fk": 0, "chains": 0, "jacobian": 0}
    for name in calls:
        original = getattr(backend, name)

        def counted(*args, name=name, original=original):
            calls[name] += 1
            return original(*args)

        monkeypatch.setattr(backend, name, counted)
    # Reset precomputes the chain: rendering and metrics can share it.
    chain = model.chain()
    model.diagnostics()
    velocity = model.velocities()
    assert calls == {"fk": 0, "chains": 0, "jacobian": 1}
    assert model.chain() is chain
    assert model.velocities() is velocity
    with pytest.raises(ValueError):
        chain[0, 0, 0] = 10
    with pytest.raises(ValueError):
        velocity[2][0] = 10
    assert model.center_step(1 / 30)
    new_chain = model.chain()
    model.diagnostics()
    assert calls == {"fk": 0, "chains": 1, "jacobian": 2}
    assert new_chain is not chain
    np.testing.assert_allclose(new_chain[-1], model.target, atol=2e-6)
    projected = model.velocities()[2].copy()
    backend.upper[0] -= 0.1
    assert not np.array_equal(model.velocities()[2], projected)
    assert calls["jacobian"] == 3


def test_gallery_caches_link_poses_and_preserves_unavailable_slots(
    backend, monkeypatch
):
    model = Experiment(backend)
    solve = backend.solve_many

    def omit_one(*args):
        valid, joints = solve(*args)
        valid[2], joints[2] = False, 0
        return valid, joints

    monkeypatch.setattr(backend, "solve_many", omit_one)
    calls = []
    chains = backend.chains

    def counted(joints):
        calls.append(len(joints))
        return chains(joints)

    monkeypatch.setattr(backend, "chains", counted)
    first = model.gallery_chains()
    assert first[2] is None
    assert calls == [7]
    assert model.gallery_chains() is first
    model.branch = BRANCHES[0]
    assert model.gallery_chains() is first
    model.set_arm_angle(10)
    assert model.gallery_chains() is not first
    assert len(calls) == 2
    for q, chain in zip(model.gallery(), model.gallery_chains()):
        if q is not None:
            np.testing.assert_allclose(chain[-1], backend.fk(q), atol=2e-6)


def test_failed_geometry_reset_preserves_complete_state(backend, monkeypatch):
    model = Experiment(backend)
    before = (
        model.q.copy(),
        model.target.copy(),
        model.branch,
        model.generation,
        model.preset,
    )

    def invalid_configuration(q):
        raise ValueError("arm angle undefined")

    monkeypatch.setattr(backend, "configuration", invalid_configuration)
    with pytest.raises(ValueError, match="arm angle undefined"):
        model.reset("Elbow down")
    np.testing.assert_array_equal(model.q, before[0])
    np.testing.assert_array_equal(model.target, before[1])
    assert (model.branch, model.generation, model.preset) == before[2:]
