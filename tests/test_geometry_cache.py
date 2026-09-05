"""Repeated-target geometry must stay independent of live solver constraints."""

from itertools import product

import numpy as np
import pytest

from srs_viser.backend import Backend
from srs_viser.model import Experiment


@pytest.fixture(params=["numpy"])
def backend(request):
    return Backend(request.param)


def geometry_impl(backend):
    return getattr(backend.solver, "_geometry_impl", backend.solver)


def count_builds(impl, monkeypatch):
    calls = []
    build = impl._build_ik_geometry

    def counted(local, elbow):
        calls.append((local.copy(), elbow))
        return build(local, elbow)

    monkeypatch.setattr(impl, "_build_ik_geometry", counted)
    return calls


def test_scalar_and_batch_queries_share_geometry_but_respect_live_limits(
    backend, monkeypatch
):
    model = Experiment(backend)
    impl = geometry_impl(backend)
    calls = count_builds(impl, monkeypatch)
    requests = [(branch, model.anchor_psi) for branch in product((-1, 1), repeat=3)]
    for seed in (model.q, model.q + 0.03, model.q - 0.03):
        valid, joints = backend.solve_many(model.target, requests, seed)
        assert valid.all()
        for (branch, psi), expected in zip(requests, joints):
            q = backend.solve(model.target, branch, psi, seed)
            np.testing.assert_allclose(q, expected, atol=1e-6)
    assert sorted(elbow for _, elbow in calls) == [-1, 1]
    assert backend.solver.set_position_limits(np.zeros(7), np.zeros(7))
    valid, joints = backend.solve_many(model.target, requests, model.q)
    assert not valid.any() and not joints.any()
    assert backend.solve(model.target, model.branch, model.anchor_psi, model.q) is None
    assert len(calls) == 2


def test_singular_seed_and_weight_changes_are_not_cached(backend, monkeypatch):
    model = Experiment(backend)
    q = model.q.copy()
    q[:3] = [0.8, 0, 0.8]
    impl = geometry_impl(backend)
    # Keep a double-precision target to isolate singular decomposition from GPU FK rounding.
    target = impl._get_fk(q) if hasattr(impl, "_get_fk") else impl.get_fk(q)
    branch, psi = backend.configuration(q)
    calls = count_builds(impl, monkeypatch)
    seed = q.copy()
    seed[[0, 2]] = 0
    first = backend.solve(target, branch, psi, seed)
    np.testing.assert_allclose(first[[0, 2]], [0.8, 0.8], atol=2e-6)
    impl.ik_nearst_weight[0] = 10
    second = backend.solve(target, branch, psi, seed)
    assert second[0] < 0.03 and second[2] > 1.5
    third = backend.solve(target, branch, psi, q)
    np.testing.assert_allclose(third, q, atol=2e-6)
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["target", "dh", "length"])
def test_content_keys_detect_in_place_geometry_changes(backend, monkeypatch, change):
    model = Experiment(backend)
    impl = geometry_impl(backend)
    local = impl._local_target(model.target)
    calls = count_builds(impl, monkeypatch)
    before = impl._prepare_ik(local, 1)
    copied = [[matrix.copy() for matrix in group] for group in before[:2]]
    if change == "target":
        local[0, 3] += 0.01
    elif change == "dh":
        impl.dh_params[-1, 0] += 0.01
    else:
        impl.link_lengths[1] += 0.01
    after = impl._prepare_ik(local, 1)
    assert after is not before and len(calls) == 2
    assert impl._prepare_ik(local.copy(), 1) is after
    assert len(calls) == 2
    assert any(not np.allclose(a, b) for a, b in zip(before[0], after[0]))
    for original, saved in zip(before[:2], copied):
        for matrix, value in zip(original, saved):
            assert not matrix.flags.writeable
            np.testing.assert_array_equal(matrix, value)
    expected = impl._build_ik_geometry(local, 1)
    for cached, computed in zip(after[:2], expected[:2]):
        np.testing.assert_allclose(cached, computed, atol=1e-14)


def test_tcp_update_uses_new_target_geometry(backend):
    model = Experiment(backend)
    assert (
        backend.solve(model.target, model.branch, model.anchor_psi, model.q) is not None
    )
    tcp = np.eye(4)
    tcp[:3, 3] = [0.005, -0.004, 0.003]
    backend.solver.set_tcp(tcp)
    solution = backend.solve(model.target, model.branch, model.anchor_psi, model.q)
    assert solution is not None
    assert not np.allclose(solution, model.q, atol=1e-4)
    np.testing.assert_allclose(backend.fk(solution), model.target, atol=2e-6)


def test_unreachable_results_are_cached_and_only_last_target_is_retained(
    backend, monkeypatch
):
    model = Experiment(backend)
    impl = geometry_impl(backend)
    calls = count_builds(impl, monkeypatch)
    local = impl._local_target(model.target)
    far = local.copy()
    far[0, 3] = 10
    assert impl._prepare_ik(far, 1) is None
    assert impl._prepare_ik(far.copy(), 1) is None
    assert len(calls) == 1
    assert impl._prepare_ik(local, 1) is not None
    assert impl._prepare_ik(local, -1) is not None
    assert len(impl._ik_geometry_cache[1]) == 2
    assert impl._prepare_ik(far, 1) is None
    assert len(calls) == 4 and len(impl._ik_geometry_cache[1]) == 1
