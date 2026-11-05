"""Limit-boundary queries recover valid targets between uniform samples."""

from itertools import product

import numpy as np
import pytest

from kuka_iiwa_solver import KUKAiiwaSolver
from srs_analytical_solver import SRSConfiguration

Q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])


def assert_solution(solver, target, seed, mode, pose_atol=2e-9):
    with np.errstate(all="raise"):
        ok, joints = solver.get_ik(target, seed, search_mode=mode, num_samples=2)
    assert ok
    assert np.all(joints >= solver.lower_position_limits)
    assert np.all(joints <= solver.upper_position_limits)
    np.testing.assert_allclose(solver.get_fk(joints), target, atol=pose_atol, rtol=0)
    return joints


@pytest.mark.parametrize("mode", ["continuous", "interval"])
def test_narrow_feasible_box_is_recovered_without_increasing_samples(mode):
    solver = KUKAiiwaSolver()
    solver.set_position_limits(Q - 1e-7, Q + 1e-7)
    assert_solution(solver, solver.get_fk(Q), np.zeros(7), mode)


@pytest.mark.parametrize("signs", list(product((-1, 1), repeat=3)))
def test_interval_search_with_dh_offsets_world_tcp_and_shifted_turns(signs):
    solver = KUKAiiwaSolver()
    solver.dh_params[:, 3] = offsets = np.linspace(-0.2, 0.2, 7)
    solver.world_to_base[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    solver.world_to_base[:3, 3] = [0.2, -0.1, 0.3]
    tcp = np.eye(4)
    tcp[:3, 3] = [0.03, -0.02, 0.05]
    solver.set_tcp(tcp)
    q = Q.copy()
    q[[1, 3, 5]] = np.abs(q[[1, 3, 5]]) * signs
    q += 2 * np.pi * np.array([2, -1, 0, 1, -2, 2, 1]) - offsets
    solver.set_position_limits(q - 1e-6, q + 1e-6)
    assert_solution(solver, solver.get_fk(q), np.zeros(7), "interval")


@pytest.mark.parametrize("middle", [None, 1, 5, "both", "straight"])
def test_fixed_joint_limits_and_singular_events(middle):
    solver = KUKAiiwaSolver()
    q = Q.copy()
    if middle in (1, 5):
        q[middle] = 0
    elif middle == "both":
        q[[1, 5]] = 0
    elif middle == "straight":
        q[[1, 3, 5]] = 0
    solver.set_position_limits(q, q)
    assert_solution(solver, solver.get_fk(q), np.ones(7), "interval")


@pytest.mark.parametrize("psi", [-np.pi, np.pi - 1e-10])
def test_periodic_arm_angle_boundary(psi):
    solver = KUKAiiwaSolver()
    solver.set_position_limits(np.full(7, -4), np.full(7, 4))
    target = solver.get_fk(Q)
    branch = solver.get_configuration(Q)
    config = SRSConfiguration(branch.shoulder, branch.elbow, branch.wrist, psi)
    ok, q = solver.solve_configuration(target, config, Q)
    assert ok
    solver.set_position_limits(q - 1e-8, q + 1e-8)
    assert_solution(solver, target, np.zeros(7), "interval")


@pytest.mark.parametrize("middle", [1, 5])
@pytest.mark.parametrize("angle", [1e-4, np.pi - 1e-6])
def test_near_singular_pivot_at_an_exact_limit(middle, angle):
    solver = KUKAiiwaSolver()
    q = Q.copy()
    q[middle] = angle
    lower, upper = q - 0.01, q + 0.003
    lower[middle] = upper[middle] = q[middle]
    solver.set_position_limits(lower, upper)
    assert_solution(solver, solver.get_fk(q), np.zeros(7), "interval")


def test_interval_search_retains_elbow_preference_and_rejects_infeasible_target():
    solver = KUKAiiwaSolver()
    q = Q.copy()
    q[3] *= -1
    target = solver.get_fk(q)
    solver.set_position_limits(q - 1e-5, q + 1e-5)
    assert_solution(solver, target, np.zeros(7), "interval")
    solver.set_elbow_up(True)
    with np.errstate(all="raise"):
        assert solver.get_ik(target, np.zeros(7), search_mode="interval") == (
            False,
            None,
        )
        target[0, 3] = 10
        assert solver.get_ik(target, np.zeros(7), search_mode="interval") == (
            False,
            None,
        )


def test_interval_ranking_is_deterministic_and_invariant_to_weight_scale():
    solver = KUKAiiwaSolver()
    target, seed = solver.get_fk(Q), Q + 0.3
    weights = np.array([1, 3, 0, 2, 1, 0, 4], dtype=float)
    solver.set_ik_nearst_weight(weights)
    expected = assert_solution(solver, target, seed, "interval")
    for scale in (1e-200, 1e200):
        solver.set_ik_nearst_weight(weights * scale)
        actual = assert_solution(solver, target, seed, "interval")
        np.testing.assert_array_equal(actual, expected)
    solver.set_ik_nearst_weight(np.zeros(7))
    expected = assert_solution(solver, target, seed, "interval")
    np.testing.assert_array_equal(
        assert_solution(solver, target, seed, "interval"), expected
    )


def test_boundary_roundoff_clamping_does_not_bypass_pose_tolerances():
    solver = KUKAiiwaSolver()
    solver.set_position_limits(Q, Q)
    nearby = Q.copy()
    nearby[0] += 2e-10
    target = solver.get_fk(nearby)
    ok, joints = solver.get_ik(target, np.zeros(7), search_mode="interval")
    assert ok
    np.testing.assert_array_equal(joints, Q)
    solver.set_iteration_params(pos_eps=1e-12, rot_eps=1e-12)
    assert solver.get_ik(target, np.zeros(7), search_mode="interval") == (False, None)


def test_repeated_target_respects_changed_limits_and_calibration():
    solver = KUKAiiwaSolver()
    target = solver.get_fk(Q)
    config = solver.get_configuration(Q)
    shifted = SRSConfiguration(
        config.shoulder, config.elbow, config.wrist, config.redundancy + 0.2
    )
    ok, other = solver.solve_configuration(target, shifted, Q)
    assert ok
    for q in (Q, other, Q):
        solver.lower_position_limits[:] = q - 1e-6
        solver.upper_position_limits[:] = q + 2e-6
        assert_solution(solver, target, np.zeros(7), "interval")
    solver.dh_params[:, 3] += 0.1
    assert_solution(solver, solver.get_fk(Q), np.zeros(7), "interval")
    tcp = np.eye(4)
    tcp[:3, 3] = [0.01, -0.02, 0.03]
    solver.set_tcp(tcp)
    assert_solution(solver, solver.get_fk(Q), np.zeros(7), "interval")


@pytest.mark.parametrize("angle", [0, 1e-9, 1e-7, np.pi, np.pi - 1e-9, np.pi - 1e-7])
@pytest.mark.parametrize("width", [0, 1e-10, 1e-7])
def test_degenerate_elbow_with_a_narrow_joint_box(angle, width):
    solver = KUKAiiwaSolver()
    solver.set_iteration_params(pos_eps=1e-9, rot_eps=1e-9)
    q = Q.copy()
    q[3] = angle
    solver.set_position_limits(q - width, q + width)
    target = solver.get_fk(q)
    assert_solution(solver, target, np.zeros(7), "interval")
    if angle == np.pi:
        with pytest.raises(ValueError, match="undefined"):
            solver.get_arm_angle(q)


def test_asymmetric_narrow_boxes_across_branches_and_turns():
    solver = KUKAiiwaSolver()
    solver.set_iteration_params(pos_eps=1e-7, rot_eps=1e-7)
    rng = np.random.default_rng(87521)
    for index in range(64):
        q = rng.uniform(-2.8, 2.8, 7)
        if index % 4 == 0:
            q[rng.choice([1, 5])] = rng.choice([0, 1e-9, 1e-6, 1e-4, np.pi])
        solver.dh_params[:, 3] = offsets = rng.uniform(-0.5, 0.5, 7)
        q += 2 * np.pi * rng.integers(-3, 4, 7) - offsets
        width = 10.0 ** rng.uniform(-10, -3, 7)
        if index % 3 == 0:
            width[rng.choice(7, 5, replace=False)] = 4
        elif index % 3 == 1:
            width[rng.integers(7)] = 0
        solver.set_position_limits(
            q - width * rng.uniform(0.05, 1, 7),
            q + width * rng.uniform(0.05, 1, 7),
        )
        assert_solution(
            solver, solver.get_fk(q), np.zeros(7), "interval", pose_atol=1e-7
        )
