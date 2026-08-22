import numpy as np
import pytest

from kuka_iiwa_solver import KUKAiiwaSolver
from srs_kinematics import SRSConfiguration


def pose_error(lhs, rhs):
    position = np.linalg.norm(lhs[:3, 3] - rhs[:3, 3])
    rotation = np.arccos(
        np.clip((np.trace(lhs[:3, :3].T @ rhs[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    )
    return position, rotation


@pytest.mark.parametrize(
    "joints",
    [
        np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]),
        np.array([-0.8, 0.4, -0.3, -1.0, 0.5, -0.7, 0.6]),
    ],
)
def test_fk_ik_round_trip(joints):
    solver = KUKAiiwaSolver()
    target = solver.get_fk(joints)
    success, solution = solver.get_ik(target, joints, num_samples=73)
    assert success
    assert max(pose_error(solver.get_fk(solution), target)) < 1e-6


def test_configuration_round_trip():
    solver = KUKAiiwaSolver()
    joints = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    target = solver.get_fk(joints)
    configuration = solver.get_configuration(joints)
    success, solution = solver.solve_configuration(target, configuration, joints)
    assert success
    assert max(pose_error(solver.get_fk(solution), target)) < 1e-6


def test_validation_and_limits():
    solver = KUKAiiwaSolver()
    assert solver.set_position_limits([-1.0] * 7, [1.0] * 7)
    assert not solver.set_position_limits([-1.0] * 6, [1.0] * 6)
    with pytest.raises(ValueError):
        solver.get_fk(np.zeros(6))
    with pytest.raises(ValueError):
        solver.get_ik(np.eye(3), np.zeros(7))


def test_singular_seed_fast_path():
    solver = KUKAiiwaSolver()
    seed = np.zeros(7)
    success, solution = solver.get_ik(solver.get_fk(seed), seed)
    assert success
    np.testing.assert_array_equal(solution, seed)


def test_public_api_and_configuration_validation():
    configuration = SRSConfiguration(1, -1, 1, 0.25)
    assert configuration.elbow == -1
    with pytest.raises(ValueError):
        SRSConfiguration(0, 1, 1, 0.0)


def test_arm_angle_tangent_is_in_jacobian_null_space():
    solver = KUKAiiwaSolver()
    joints = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    direction = solver.get_null_space_direction(joints)
    assert np.linalg.norm(direction) > 0.1
    assert np.linalg.norm(solver.get_jacobian(joints) @ direction) < 1e-6


def test_null_space_velocity_projection():
    solver = KUKAiiwaSolver()
    joints = np.array([-0.4, 0.6, -0.2, -0.9, 0.5, -0.7, 0.3])
    preferred = np.linspace(-1.0, 1.0, 7)
    velocity = solver.get_null_space_velocity(joints, preferred)
    assert np.linalg.norm(solver.get_jacobian(joints) @ velocity) < 1e-8


def test_svd_null_space_fallback_at_straight_arm():
    solver = KUKAiiwaSolver()
    direction = solver.get_null_space_direction(np.zeros(7))
    assert direction.shape == (7,)
    assert np.all(np.isfinite(direction))
    assert np.linalg.norm(solver.get_jacobian(np.zeros(7)) @ direction) < 1e-6


def test_continuous_ik_uses_seed_arm_angle_first(monkeypatch):
    solver = KUKAiiwaSolver()
    target_joints = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    seed = target_joints + np.array([0.03, -0.02, 0.02, 0.01, -0.02, 0.02, -0.01])
    target = solver.get_fk(target_joints)
    calls = 0
    original = solver._compute_inverse_kinematics

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_compute_inverse_kinematics", counted)
    success, solution = solver.get_ik(target, seed, search_mode="continuous")
    assert success
    assert calls == 1
    assert max(pose_error(solver.get_fk(solution), target)) < 1e-6


def test_global_ik_mode_remains_available():
    solver = KUKAiiwaSolver()
    joints = np.array([0.2, -0.4, 0.3, 0.9, -0.2, 0.6, -0.1])
    target = solver.get_fk(joints)
    seed = joints + 0.05
    success, solution = solver.get_ik(
        target, seed, num_samples=9, search_mode="global"
    )
    assert success
    assert max(pose_error(solver.get_fk(solution), target)) < 1e-6
