"""Vectorized Jacobians retain scalar conventions and match independent FK differences."""

import numpy as np
import pytest

from kuka_iiwa_solver import KUKAiiwaSolver


@pytest.mark.parametrize("count", [0, 1, 8, 64])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batch_jacobian_with_frames_offsets_and_strided_inputs(count, dtype):
    solver = KUKAiiwaSolver()
    solver.world_to_base[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    solver.world_to_base[:3, 3] = [0.3, -0.1, 0.2]
    solver.set_tcp(
        np.array([[1, 0, 0, 0.04], [0, 0, -1, 0.03], [0, 1, 0, -0.02], [0, 0, 0, 1.0]])
    )
    solver.dh_params[:, 3] = np.linspace(-0.1, 0.1, 7)
    solver.dh_params[:, 2] = np.linspace(0.01, 0.03, 7)
    joints = (
        np.random.default_rng(6305)
        .uniform(-1.5, 1.5, (count * 2, 7))
        .astype(dtype)[::2]
    )
    if count:
        joints[0] = -solver.dh_params[:, 3]
    actual = solver.get_jacobian(joints)
    assert actual.shape == (count, 6, 7) and actual.flags.c_contiguous
    for index, q in enumerate(joints):
        scalar = solver.get_jacobian(q)
        assert scalar.shape == (6, 7)
        np.testing.assert_allclose(actual[index], scalar, atol=1e-14, rtol=0)
    # Central differences of batched FK provide an independent derivative
    # check, including both world-frame linear and angular velocity rows.
    q = joints.astype(float)
    rotation = solver.get_fk(q)[:, :3, :3]
    step = 1e-6
    for joint in range(7):
        delta = np.eye(7)[joint] * step
        plus, minus = solver.get_fk(q + delta), solver.get_fk(q - delta)
        linear = (plus[:, :3, 3] - minus[:, :3, 3]) / (2 * step)
        omega = (
            (plus[:, :3, :3] - minus[:, :3, :3]) / (2 * step)
        ) @ rotation.transpose(0, 2, 1)
        angular = np.stack((omega[:, 2, 1], omega[:, 0, 2], omega[:, 1, 0]), axis=1)
        np.testing.assert_allclose(
            actual[:, :, joint],
            np.concatenate((linear, angular), axis=1),
            atol=2e-9,
            rtol=0,
        )
    preserved = actual.copy()
    solver.get_jacobian(np.ones((count + 1, 7)))
    joints[:] = 0
    np.testing.assert_array_equal(actual, preserved)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((2, 6)),
        np.zeros((1, 1, 7)),
        np.full((2, 7), np.nan),
        np.full((2, 7), np.inf),
    ],
)
def test_invalid_jacobian_batch_is_rejected(value):
    with pytest.raises(ValueError, match="qpos.*finite"):
        KUKAiiwaSolver().get_jacobian(value)


@pytest.mark.parametrize("step", [0, -1, np.inf, np.nan])
def test_empty_jacobian_batch_still_validates_step(step):
    with pytest.raises(ValueError, match="step"):
        KUKAiiwaSolver().get_jacobian(np.empty((0, 7)), step=step)
