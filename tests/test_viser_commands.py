"""Event ordering regressions; importing the loop does not require GUI extras."""

import time

import numpy as np
import pytest

from srs_viser.app import LabApp
from srs_viser.backend import Backend
from srs_viser.commands import CommandQueue
from srs_viser.model import Experiment


def make_loop():
    app = LabApp.__new__(LabApp)
    app.model = Experiment(Backend())
    app.commands = CommandQueue()
    app.last_time = time.perf_counter()
    app.render = lambda **kwargs: None
    return app


def test_drag_queue_replaces_only_adjacent_samples_and_drains_once():
    queue = CommandQueue()
    for value in range(10000):
        queue.put(("angle", value))
    queue.put(("joint", (0, 0.1)))
    queue.put(("joint", (1, 0.2)))
    queue.put(("joint", (0, 0.3)))
    queue.put(("angle", -10))
    queue.put(("reset", None))
    queue.put(("angle", 20))
    assert queue.drain() == [
        ("angle", 9999),
        ("joint", (0, 0.1)),
        ("joint", (1, 0.2)),
        ("joint", (0, 0.3)),
        ("angle", -10),
        ("reset", None),
        ("angle", 20),
    ]
    assert queue.drain() == []


@pytest.mark.parametrize(
    "events",
    [
        [("angle", 25), ("joint", (0, 0.5)), ("angle", -15)],
        [("joint", (1, -0.3)), ("angle", 25), ("joint", (1, -0.8))],
        [("angle", 25), ("preset", "Elbow down"), ("angle", -15)],
    ],
)
def test_interleaved_controls_match_ordered_execution(events):
    actual, expected = make_loop(), make_loop()
    for event in events:
        actual.commands.put(event)
        expected.apply(*event)
    actual.step(1 / 30)
    np.testing.assert_allclose(actual.model.q, expected.model.q, atol=1e-12)
    np.testing.assert_allclose(actual.model.target, expected.model.target, atol=1e-12)
    assert actual.model.branch == expected.model.branch


def test_auto_stop_forces_final_metrics_refresh():
    app = make_loop()
    app.model.mode = "jacobian"
    app.model.play = True
    app.model.center_step = lambda dt: False
    rendered = []
    app.render = lambda **kwargs: rendered.append(kwargs)
    assert app.step(1 / 30)
    assert not app.model.play
    assert rendered == [{"force_metrics": True}]


def test_pause_resume_keeps_sweep_phase_and_direction():
    actual, continuous = make_loop(), make_loop()
    for app in (actual, continuous):
        app.model.phase = np.pi
        app.model.play = True
    phase, center = actual.model.phase, actual.model.play_center
    actual.apply("play", False)
    q = actual.model.q.copy()
    for _ in range(10):
        assert not actual.step(1 / 30)
    np.testing.assert_array_equal(actual.model.q, q)
    assert actual.model.phase == phase and actual.model.play_center == center
    actual.apply("play", True)
    actual.step(1 / 30)
    continuous.step(1 / 30)
    np.testing.assert_allclose(actual.model.q, continuous.model.q, atol=1e-12)
    assert actual.model.offset_deg < 0


@pytest.mark.parametrize(
    "event", [("angle", 20), ("joint", (0, 0.4)), ("amplitude", 60)]
)
def test_pose_and_amplitude_edits_start_a_new_sweep(event):
    app = make_loop()
    app.model.phase = np.pi
    app.apply(*event)
    assert app.model.phase == 0 and app.model.play_center == app.model.offset_deg
    app.apply("play", True)
    q = app.model.q.copy()
    app.step(1 / 30)
    assert np.max(np.abs(app.model.q - q)) <= app.model.max_joint_speed / 30 + 1e-8
