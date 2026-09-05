"""Numerical and API regressions for the analytical solver."""

from itertools import product

import numpy as np
import pytest

from kuka_iiwa_solver import KUKAiiwaSolver
from srs_kinematics import SRSConfiguration

JOINTS = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])


def assert_pose(actual, target, atol=1e-7):
    assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) < atol
    assert np.linalg.norm(actual[:3, :3] - target[:3, :3]) < atol


def transform(angle, translation):
    c, s = np.cos(angle), np.sin(angle)
    result = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])
    result[:3, 3] = translation
    return result


@pytest.mark.parametrize("turns", [-10, -2, 2, 10])
def test_periodic_mapping_outside_principal_interval(turns):
    solver = KUKAiiwaSolver()
    q = JOINTS + turns * 2 * np.pi
    np.testing.assert_allclose(solver.qpos_to_limits(q, JOINTS), JOINTS, atol=1e-13)


def test_periodic_mapping_agrees_with_exhaustive_joint_choices():
    solver = KUKAiiwaSolver()
    rng = np.random.default_rng(72)
    for _ in range(40):
        lower = rng.uniform(-12, 2, 7)
        upper = lower + rng.uniform(0.1, 25, 7)
        assert solver.set_position_limits(lower, upper)
        q, seed = rng.uniform(-30, 30, (2, 7))
        choices = [
            [
                value + 2 * np.pi * k
                for k in range(-12, 13)
                if lo <= value + 2 * np.pi * k <= hi
            ]
            for value, lo, hi in zip(q, lower, upper)
        ]
        result = solver.qpos_to_limits(q, seed)
        if any(not values for values in choices):
            assert result == []
        else:
            expected = [
                min(values, key=lambda value: abs(value - desired))
                for values, desired in zip(choices, seed)
            ]
            np.testing.assert_allclose(result, expected, atol=1e-13)


def test_periodic_mapping_handles_many_turns_fixed_limits_and_inactive_joints():
    solver = KUKAiiwaSolver()
    assert solver.set_position_limits([-1e6] * 7, [1e6] * 7)
    seed = JOINTS + np.arange(7) * 200 * np.pi
    np.testing.assert_allclose(solver.qpos_to_limits(JOINTS, seed), seed)
    assert solver.set_position_limits(JOINTS, JOINTS)
    np.testing.assert_allclose(solver.qpos_to_limits(JOINTS + 4 * np.pi, seed), JOINTS)
    q = JOINTS.copy()
    q[2] = 100
    active = np.ones(7, dtype=bool)
    active[2] = False
    np.testing.assert_allclose(solver.qpos_to_limits(q, seed, active), q)


@pytest.mark.parametrize("signs", list(product((-1, 1), repeat=3)))
def test_exact_configuration_round_trip_on_all_eight_branches(signs):
    solver = KUKAiiwaSolver()
    q = JOINTS.copy()
    q[[1, 3, 5]] = np.array(signs) * np.abs(q[[1, 3, 5]])
    target = solver.get_fk(q)
    configuration = solver.get_configuration(q)
    assert (configuration.shoulder, configuration.elbow, configuration.wrist) == signs
    ok, solution = solver.solve_configuration(
        target.tolist(), configuration, (q + 0.03).tolist()
    )
    assert ok
    assert_pose(solver.get_fk(solution), target)
    np.testing.assert_allclose(solution, q, atol=1e-12)


@pytest.mark.parametrize("shoulder,wrist", list(product((0.0, 1e-9, -1e-9), repeat=2)))
def test_spherical_joint_singularities(shoulder, wrist):
    solver = KUKAiiwaSolver()
    q = JOINTS.copy()
    q[1], q[5] = shoulder, wrist
    target = solver.get_fk(q)
    with np.errstate(all="raise"):
        configuration = solver.get_configuration(q)
        ok, solution = solver.solve_configuration(target, configuration, q)
    assert ok
    assert_pose(solver.get_fk(solution), target, atol=1e-8)


@pytest.mark.parametrize("mode", ["continuous", "global"])
def test_round_trip_with_perturbed_seeds_across_joint_range(mode):
    solver = KUKAiiwaSolver()
    rng = np.random.default_rng(702)
    for q in rng.uniform(
        solver.lower_position_limits, solver.upper_position_limits, (40, 7)
    ):
        seed = np.clip(
            q + rng.normal(0, 0.08, 7),
            solver.lower_position_limits,
            solver.upper_position_limits,
        )
        target = solver.get_fk(q)
        ok, solution = solver.get_ik(target, seed, num_samples=73, search_mode=mode)
        assert ok
        assert np.all(solution >= solver.lower_position_limits)
        assert np.all(solution <= solver.upper_position_limits)
        assert_pose(solver.get_fk(solution), target)


def test_world_and_tcp_transforms_are_consistent():
    solver = KUKAiiwaSolver()
    baseline = solver.get_all_fk_mat(JOINTS)
    world = transform(0.4, [0.2, -0.3, 0.1])
    tcp = transform(-0.3, [0.04, 0.03, -0.02])
    solver.world_to_base = world
    solver.set_tcp(tcp)
    poses = solver.get_all_fk_mat(JOINTS)
    for index in range(7):
        np.testing.assert_allclose(poses[index], world @ baseline[index], atol=1e-14)
        np.testing.assert_allclose(
            poses[index], solver.get_fk(JOINTS, index), atol=1e-14
        )
    target = world @ baseline[-1] @ tcp
    assert_pose(poses[-1], target)
    assert_pose(solver.get_fk(JOINTS), target)
    configuration = solver.get_configuration(JOINTS)
    ok, q = solver.solve_configuration(target, configuration, JOINTS)
    assert ok
    assert_pose(solver.get_fk(q), target)
    for mode in ("continuous", "global"):
        ok, q = solver.get_ik(target, JOINTS + 0.03, search_mode=mode)
        assert ok
        assert_pose(solver.get_fk(q), target)
    tcp[0, 3] += 1
    assert solver.get_tcp()[0, 3] != tcp[0, 3]
    exported = solver.get_tcp()
    exported[:] = 0
    assert solver.get_tcp()[3, 3] == 1


@pytest.mark.parametrize("q", [JOINTS, np.zeros(7), -JOINTS])
def test_analytical_jacobian_matches_central_differences(q):
    solver = KUKAiiwaSolver()
    solver.world_to_base = transform(0.6, [0.2, -0.1, 0.3])
    solver.set_tcp(transform(-0.2, [0.08, -0.03, 0.04]))
    jacobian = solver.get_jacobian(q)
    step = 1e-6
    rotation = solver.get_fk(q)[:3, :3]
    for index in range(7):
        delta = np.eye(7)[index] * step
        plus, minus = solver.get_fk(q + delta), solver.get_fk(q - delta)
        linear = (plus[:3, 3] - minus[:3, 3]) / (2 * step)
        omega = ((plus[:3, :3] - minus[:3, :3]) / (2 * step)) @ rotation.T
        angular = np.array([omega[2, 1], omega[0, 2], omega[1, 0]])
        np.testing.assert_allclose(
            jacobian[:, index], np.r_[linear, angular], atol=2e-9, rtol=0
        )


def test_dh_offsets_and_periodic_configuration():
    solver = KUKAiiwaSolver()
    solver.dh_params[:, 3] = np.linspace(-0.2, 0.2, 7)
    configuration = solver.get_configuration(JOINTS)
    periodic = solver.get_configuration(JOINTS + 4 * np.pi)
    assert (configuration.shoulder, configuration.elbow, configuration.wrist) == (
        periodic.shoulder,
        periodic.elbow,
        periodic.wrist,
    )
    assert configuration.redundancy == pytest.approx(periodic.redundancy)
    target = solver.get_fk(JOINTS)
    ok, solution = solver.solve_configuration(target, configuration, JOINTS)
    assert ok
    np.testing.assert_allclose(solution, JOINTS, atol=1e-12)


def test_seed_fast_path_enforces_limits_and_elbow_preference():
    solver = KUKAiiwaSolver()
    target = solver.get_fk(JOINTS)
    ok, q = solver.get_ik(target, JOINTS + 8 * np.pi)
    assert ok
    np.testing.assert_allclose(q, JOINTS, atol=1e-12)
    negative_elbow = JOINTS.copy()
    negative_elbow[3] *= -1
    solver.set_elbow_up(True)
    target = solver.get_fk(negative_elbow)
    for mode in ("continuous", "global"):
        ok, q = solver.get_ik(target, negative_elbow, search_mode=mode, num_samples=73)
        assert ok
        assert q[3] >= 0
        assert_pose(solver.get_fk(q), target)
    configuration = solver.get_configuration(negative_elbow)
    assert solver.solve_configuration(target, configuration, negative_elbow) == (
        False,
        None,
    )
    assert solver.set_position_limits([0.0] * 7, [0.0] * 7)
    assert solver.get_ik(solver.get_fk(JOINTS), JOINTS) == (False, None)


def test_unreachable_and_collapsed_arm_fail_without_numerical_warnings():
    solver = KUKAiiwaSolver()
    far = np.eye(4)
    far[0, 3] = 10
    collapsed = np.eye(4)
    collapsed[2, 3] = solver.d_bs + solver.d_wt
    configuration = SRSConfiguration(1, 1, 1, 0.0)
    with np.errstate(all="raise"):
        for target in (far, collapsed):
            assert solver.get_ik(target, JOINTS) == (False, None)
            assert solver.solve_configuration(target, configuration, JOINTS) == (
                False,
                None,
            )
        straight = solver.get_fk(np.zeros(7))
        ok, q = solver.get_ik(straight, JOINTS)
        assert ok
        assert_pose(solver.get_fk(q), straight)


def test_global_search_selects_weighted_best_sample_and_is_deterministic():
    solver = KUKAiiwaSolver()
    solver.set_position_limits([-2 * np.pi] * 7, [2 * np.pi] * 7)
    seed = JOINTS + 0.2
    target = solver.get_fk(JOINTS)
    samples = 12
    candidates = []
    for signs in product((-1, 1), repeat=3):
        for psi in np.linspace(-np.pi, np.pi, samples, endpoint=False):
            ok, q = solver.solve_configuration(
                target, SRSConfiguration(*signs, psi), seed
            )
            if ok:
                candidates.append(q)
    assert candidates
    for weights in (np.ones(7), np.arange(7), np.zeros(7)):
        assert solver.set_ik_nearst_weight(weights)
        ok, q = solver.get_ik(target, seed, num_samples=samples, search_mode="global")
        assert ok
        expected_distance = min(
            np.linalg.norm((candidate - seed) * weights) for candidate in candidates
        )
        assert np.linalg.norm((q - seed) * weights) == pytest.approx(
            expected_distance, abs=1e-12
        )
        for _ in range(2):
            np.testing.assert_array_equal(
                solver.get_ik(target, seed, num_samples=samples, search_mode="global")[
                    1
                ],
                q,
            )


@pytest.mark.parametrize("kind", ["nan", "reflection", "scale", "bottom_row"])
def test_all_pose_entry_points_reject_non_rigid_transforms(kind):
    solver = KUKAiiwaSolver()
    pose = np.eye(4)
    if kind == "nan":
        pose[0, 3] = np.nan
    elif kind == "reflection":
        pose[0, 0] = -1
    elif kind == "scale":
        pose[0, 0] = 1.1
    else:
        pose[3, 0] = 0.1
    for call in (
        lambda: solver.get_ik(pose, JOINTS),
        lambda: solver.solve_configuration(
            pose, SRSConfiguration(1, 1, 1, 0.0), JOINTS
        ),
        lambda: solver.set_tcp(pose),
    ):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize(
    "settings",
    [
        {"pos_eps": np.nan},
        {"rot_eps": np.inf},
        {"dt": -1},
        {"damp": np.nan},
        {"max_iterations": 1.5},
        {"num_samples": 1},
        {"num_samples": True},
    ],
)
def test_invalid_iteration_settings_are_atomic(settings):
    solver = KUKAiiwaSolver()
    previous = solver.get_iteration_params()
    assert not solver.set_iteration_params(**settings)
    assert solver.get_iteration_params() == previous


@pytest.mark.parametrize("joint_ids", [[0.5], [True], [[0]], [0, 0], 0, [7]])
def test_invalid_weight_indices_are_rejected_without_mutation(joint_ids):
    solver = KUKAiiwaSolver()
    previous = solver.get_ik_nearst_weight()
    assert not solver.set_ik_nearst_weight([2.0], joint_ids)
    np.testing.assert_array_equal(solver.get_ik_nearst_weight(), previous)


def test_input_validation_and_per_call_sampling():
    solver = KUKAiiwaSolver()
    assert not solver.set_position_limits(np.zeros((7, 1)), np.ones((7, 1)))
    assert not solver.set_position_limits(0, 1)
    assert not solver.set_ik_nearst_weight([np.nan] * 7)
    for value in ([0.0] * 6, [np.nan] * 7, np.zeros((7, 1))):
        with pytest.raises(ValueError):
            solver.get_all_fk_mat(value)
        with pytest.raises(ValueError):
            solver.solve_configuration(np.eye(4), SRSConfiguration(1, 1, 1, 0), value)
    with pytest.raises(IndexError):
        solver.get_fk(JOINTS, index=1.5)
    with pytest.raises(ValueError):
        solver.qpos_to_limits(JOINTS, JOINTS, [1, 0])
    target = solver.get_fk(JOINTS)
    assert solver.set_iteration_params(num_samples=19)
    solver.get_ik(target, JOINTS + 0.1, num_samples=7)
    assert solver.get_iteration_params()["num_samples"] == 19
    assert solver.get_ik(target, JOINTS, num_sample=7)[0]
    with pytest.raises(TypeError):
        solver.get_ik(target, JOINTS, num_samples=7, num_sample=9)
    with pytest.raises(TypeError):
        solver.get_ik(target, JOINTS, num_sampels=7)


@pytest.mark.parametrize("first,middle,last", [(0, 1, 2), (4, 5, 6)])
@pytest.mark.parametrize("middle_angle", [0.0, np.pi])
@pytest.mark.parametrize(
    "weights,expected",
    [
        ([1, 1], [0.8, 0.8]),
        ([1, 3], [1.0, 0.6]),
        ([0, 1], [1.0, 0.6]),
        ([0, 0], [0.8, 0.8]),
    ],
)
def test_singular_outer_joints_are_optimized_together(
    first, middle, last, middle_angle, weights, expected
):
    solver = KUKAiiwaSolver()
    q = JOINTS.copy()
    sign = 1 if middle_angle == 0 else -1
    q[[first, middle, last]] = [0.8, middle_angle, sign * 0.8]
    lower, upper = np.full(7, -4.0), np.full(7, 4.0)
    lower[first], upper[first] = 0, 1
    lower[last], upper[last] = (0, 1) if sign == 1 else (-1, 0)
    solver.set_position_limits(lower, upper)
    solver.set_ik_nearst_weight(weights, [first, last])
    seed = q.copy()
    seed[[first, last]] = 0
    target = solver.get_fk(q)
    ok, solution = solver.solve_configuration(target, solver.get_configuration(q), seed)
    assert ok
    assert_pose(solver.get_fk(solution), target)
    np.testing.assert_allclose(
        solution[[first, last]], np.array(expected) * [1, sign], atol=1e-12
    )


def test_singular_joint_limits_with_shifted_revolutions():
    solver = KUKAiiwaSolver()
    q = JOINTS.copy()
    q[[0, 1, 2]] = [2 * np.pi + 0.8, 0, 0.8]
    lower, upper = q - 0.1, q + 0.1
    lower[[0, 2]] = [2 * np.pi, 0]
    upper[[0, 2]] = [2 * np.pi + 1, 1]
    solver.set_position_limits(lower, upper)
    seed = q.copy()
    seed[[0, 2]] = lower[[0, 2]]
    target = solver.get_fk(q)
    ok, solution = solver.solve_configuration(target, solver.get_configuration(q), seed)
    assert ok
    np.testing.assert_allclose(solution, q, atol=1e-12)
