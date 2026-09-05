"""Batched link poses must retain the scalar contract and frame conventions."""

import numpy as np
import pytest

from kuka_iiwa_solver import KUKAiiwaSolver


@pytest.mark.parametrize("count", [0, 1, 8, 64])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batch_chain_against_independent_dh_with_world_tool_and_offsets(count, dtype):
    solver = KUKAiiwaSolver()
    solver.world_to_base[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    solver.world_to_base[:3, 3] = [0.3, -0.1, 0.2]
    tcp = np.array(
        [[1, 0, 0, 0.04], [0, 0, -1, 0.03], [0, 1, 0, -0.02], [0, 0, 0, 1.0]]
    )
    solver.set_tcp(tcp)
    solver.dh_params[:, 3] = np.linspace(-0.1, 0.1, 7)
    joints = (
        np.random.default_rng(813).uniform(-1.5, 1.5, (count * 2, 7)).astype(dtype)[::2]
    )
    actual = solver.get_all_fk_mat(joints)
    assert actual.shape == (count, 8, 4, 4)
    tcp_batch = solver.get_fk(joints)
    assert tcp_batch.shape == (count, 4, 4)
    np.testing.assert_allclose(tcp_batch, actual[:, -1], atol=1e-14, rtol=0)
    for index in range(7):
        partial = solver.get_fk(joints, index)
        assert partial.shape == (count, 4, 4)
        np.testing.assert_allclose(partial, actual[:, index], atol=1e-14, rtol=0)
    for q, chain in zip(joints, actual):
        pose = solver.world_to_base.copy()
        expected = []
        for value, (d, alpha, a, offset) in zip(q.astype(float), solver.dh_params):
            ct, st = np.cos(value + offset), np.sin(value + offset)
            ca, sa = np.cos(alpha), np.sin(alpha)
            pose = pose @ np.array(
                [
                    [ct, -st * ca, st * sa, a * ct],
                    [st, ct * ca, -ct * sa, a * st],
                    [0, sa, ca, d],
                    [0, 0, 0, 1],
                ]
            )
            expected.append(pose)
        expected.append(pose @ tcp)
        np.testing.assert_allclose(chain, expected, atol=1e-14, rtol=0)
        scalar = solver.get_all_fk_mat(q)
        assert isinstance(scalar, list)
        np.testing.assert_allclose(chain, scalar, atol=1e-14, rtol=0)
        np.testing.assert_allclose(chain[-1], solver.get_fk(q), atol=1e-14, rtol=0)
    preserved = actual.copy()
    solver.get_all_fk_mat(np.ones((count + 1, 7)))
    solver.get_fk(np.ones((count + 1, 7)))
    joints[:] = 0
    np.testing.assert_array_equal(actual, preserved)
    np.testing.assert_array_equal(tcp_batch, preserved[:, -1])


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((2, 6)),
        np.zeros((1, 1, 7)),
        np.full((2, 7), np.nan),
        np.full((2, 7), np.inf),
    ],
)
def test_invalid_batch_is_rejected(value):
    with pytest.raises(ValueError):
        KUKAiiwaSolver().get_all_fk_mat(value)
    with pytest.raises(ValueError):
        KUKAiiwaSolver().get_fk(value)


@pytest.mark.parametrize("index", [-2, 7, 0.5, True, np.bool_(False)])
def test_batch_fk_rejects_invalid_link_index(index):
    with pytest.raises(IndexError):
        KUKAiiwaSolver().get_fk(np.empty((0, 7)), index)
