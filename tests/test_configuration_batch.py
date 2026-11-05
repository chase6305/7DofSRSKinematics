"""Exact configuration batches retain scalar branch, seed and limit semantics."""

from itertools import product

import numpy as np
import pytest

from kuka_iiwa_solver import KUKAiiwaSolver
from srs_analytical_solver import SRSConfiguration


@pytest.mark.parametrize(
    "middle,value", [(None, 0), (1, 0), (5, 0), (1, np.pi), (5, np.pi)]
)
def test_batch_matches_scalar_with_frames_singularities_and_shared_geometry(
    middle, value, monkeypatch
):
    solver = KUKAiiwaSolver()
    q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    if middle is not None:
        q[middle] = value
    if value:
        solver.set_position_limits(np.full(7, -4.0), np.full(7, 4.0))
    solver.world_to_base[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    solver.world_to_base[:3, 3] = [0.2, -0.1, 0.3]
    tcp = np.eye(4)
    tcp[:3, 3] = [0.03, -0.02, 0.05]
    solver.set_tcp(tcp)
    solver.set_ik_nearst_weight([1, 2, 3, 1, 3, 2, 1])
    target = solver.get_fk(q)
    psi = solver.get_arm_angle(q)
    configs = [
        SRSConfiguration(*branch, psi + delta)
        for delta in (0.2, 0, -0.2, 0)
        for branch in product((-1, 1), repeat=3)
    ]
    configs = [configs[i] for i in np.random.default_rng(7).permutation(len(configs))]
    expected = [solver.solve_configuration(target, c, q) for c in configs]
    calls = []
    prepare = solver._prepare_ik
    decompose = solver._candidate_joint_angles
    decompositions = []

    def counted(local, elbow):
        calls.append(elbow)
        return prepare(local, elbow)

    monkeypatch.setattr(solver, "_prepare_ik", counted)

    def counted_decomposition(prepared, angles, branches, seed):
        decompositions.append(np.asarray(branches).copy())
        return decompose(prepared, angles, branches, seed)

    monkeypatch.setattr(solver, "_candidate_joint_angles", counted_decomposition)
    ok, joints = solver.solve_configurations(target, iter(configs), q)
    assert sorted(calls) == [-1, 1]
    assert len(decompositions) == 2
    assert all(len(np.unique(branches & 2)) == 1 for branches in decompositions)
    np.testing.assert_array_equal(ok, [item[0] for item in expected])
    assert ok.any()
    for valid, actual, (_, ref) in zip(ok, joints, expected):
        if valid:
            np.testing.assert_allclose(actual, ref, atol=1e-12)
            fk = solver.get_fk(actual)
            assert np.linalg.norm(fk[:3, 3] - target[:3, 3]) < 1e-8
            assert np.linalg.norm(fk[:3, :3] - target[:3, :3]) < 1e-8
        else:
            np.testing.assert_array_equal(actual, np.zeros(7))
    preserved = joints.copy()
    solver.solve_configurations(target, configs[:1], q + 0.02)
    np.testing.assert_array_equal(joints, preserved)


def test_empty_invalid_and_unreachable_configuration_batches():
    solver = KUKAiiwaSolver()
    q = np.zeros(7)
    target = solver.get_fk(q)
    ok, joints = solver.solve_configurations(target, [], q)
    assert ok.shape == (0,) and joints.shape == (0, 7)
    config = SRSConfiguration(1, 1, 1, 0)
    target[0, 3] = 10
    ok, joints = solver.solve_configurations(target, [config], q)
    assert not ok.any()
    assert not joints.any()
    with pytest.raises(TypeError):
        solver.solve_configurations(target, [config, None], q)
    with pytest.raises(ValueError):
        solver.solve_configurations(target, [], np.full(7, np.nan))
    with pytest.raises(ValueError):
        solver.solve_configurations(np.diag([2, 1, 1, 1]), [], q)


def test_periodic_limits_and_elbow_preference_in_batches():
    solver = KUKAiiwaSolver()
    q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]) + 2 * np.pi
    solver.set_position_limits(np.full(7, 2 * np.pi - 2), np.full(7, 2 * np.pi + 2))
    solver.set_elbow_up(True)
    c = solver.get_configuration(q)
    configs = [c, SRSConfiguration(c.shoulder, -1, c.wrist, c.redundancy)]
    ok, result = solver.solve_configurations(solver.get_fk(q), configs, q)
    np.testing.assert_array_equal(ok, [True, False])
    np.testing.assert_allclose(result[0], q, atol=1e-12)
    np.testing.assert_array_equal(result[1], np.zeros(7))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_exact_arm_angle_precision_is_independent_of_scalar_storage(dtype):
    solver = KUKAiiwaSolver()
    solver.set_position_limits(np.full(7, -4.0), np.full(7, 4.0))
    q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    target = solver.get_fk(q)
    branch = solver.get_configuration(q)
    for phase in (dtype(0.1), dtype(-2.7), dtype(65504)):
        requested = SRSConfiguration(branch.shoulder, branch.elbow, branch.wrist, phase)
        canonical = SRSConfiguration(
            branch.shoulder, branch.elbow, branch.wrist, float(phase)
        )
        expected_ok, expected = solver.solve_configuration(target, canonical, q)
        ok, actual = solver.solve_configuration(target, requested, q)
        assert expected_ok and ok
        np.testing.assert_array_equal(actual, expected)
        mask, joints = solver.solve_configurations(target, [requested], q)
        assert mask[0]
        np.testing.assert_array_equal(joints[0], expected)
        np.testing.assert_allclose(solver.get_fk(actual), target, atol=1e-10, rtol=0)


def test_request_cache_detects_replaced_records_and_invalid_list_entries():
    solver = KUKAiiwaSolver()
    solver.set_position_limits(np.full(7, -4.0), np.full(7, 4.0))
    q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    target = solver.get_fk(q)
    branch = solver.get_configuration(q)
    requests = [
        SRSConfiguration(branch.shoulder, branch.elbow, branch.wrist, np.float16(0.1))
    ]
    ok, first = solver.solve_configurations(target, requests, q)
    assert ok[0]
    requests[0] = SRSConfiguration(branch.shoulder, branch.elbow, branch.wrist, 0.1)
    ok, second = solver.solve_configurations(target, requests, q)
    assert ok[0] and np.linalg.norm(first - second) > 1e-6
    expected_ok, expected = solver.solve_configuration(target, requests[0], q)
    assert expected_ok
    np.testing.assert_array_equal(second[0], expected)
    requests[0] = np.zeros(7)
    with pytest.raises(TypeError, match="SRSConfiguration"):
        solver.solve_configurations(target, requests, q)
