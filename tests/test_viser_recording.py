"""Exported playback is independently reconstructible without GUI dependencies."""

import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from srs_viser.backend import Backend
from srs_viser.model import MODES, Experiment, pose_error
from srs_viser.recording import CSV_COLUMNS, write_playback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"


@pytest.fixture(params=["numpy"])
def backend(request):
    return Backend(request.param)


@pytest.mark.parametrize("mode", MODES)
def test_export_records_actual_time_joints_poses_and_speed(backend, mode):
    model = Experiment(backend, mode)
    model.path_plane, model.path_radius = "tool-xy", 0.015
    model.reset_trajectory()
    output = io.StringIO()
    report = write_playback(model, output, frames=35, dt=0.025)
    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert len(rows) == 36 and report["recorded_frames"] == 35
    assert report["completed"] and report["stop_reason"] is None
    assert not model.play
    assert report["simulated_seconds"] == pytest.approx(35 * 0.025)
    previous = None
    for index, row in enumerate(rows):
        row = {key: float(value) for key, value in row.items()}
        assert row["time_s"] == pytest.approx(index * 0.025)
        q = np.array([row[f"q{i + 1}_rad"] for i in range(7)])
        actual = backend.fk(q)
        target = np.eye(4)
        target[:3, 3] = [row[f"target_{axis}_m"] for axis in "xyz"]
        target[:3, :3] = [[row[f"target_r{i}{j}"] for j in range(3)] for i in range(3)]
        np.testing.assert_allclose(
            actual[:3, 3], [row[f"tcp_{axis}_m"] for axis in "xyz"], atol=1e-7
        )
        np.testing.assert_allclose(
            actual[:3, :3],
            [[row[f"tcp_r{i}{j}"] for j in range(3)] for i in range(3)],
            atol=1e-7,
        )
        assert max(pose_error(actual, target)) < 3e-6
        assert row["joint_limit_margin_rad"] >= 0
        speed = 0 if previous is None else float(np.max(np.abs(q - previous)) / 0.025)
        assert row["joint_speed_peak_rad_s"] == pytest.approx(speed)
        assert speed <= model.max_joint_speed + 4e-7
        assert tuple(row[k] for k in ("shoulder", "elbow", "wrist")) == model.branch
        previous = q


def test_blocked_recording_reports_only_accepted_frames(backend):
    model = Experiment(backend, "manifold")
    assert model.set_arm_angle(134)
    model.amplitude, model.frequency = 90, 0.5
    output = io.StringIO()
    report = write_playback(model, output, frames=30)
    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert not report["completed"] and "paused" in report["stop_reason"]
    assert 0 < report["recorded_frames"] < 30
    assert len(rows) == report["recorded_frames"] + 1
    assert float(rows[-1]["time_s"]) == report["simulated_seconds"]
    assert not model.play


def test_failed_output_pauses_playback(backend):
    class BrokenOutput:
        def write(self, text):
            raise OSError("disk full")

    model = Experiment(backend)
    model.play = True
    with pytest.raises(OSError, match="disk full"):
        write_playback(model, BrokenOutput(), frames=1)
    assert not model.play


@pytest.mark.parametrize(
    "kwargs", [{"dt": 0.1}, {"dt": float("nan")}, {"frames": 0}, {"frames": True}]
)
def test_invalid_recording_parameters_do_not_write_or_move(backend, kwargs):
    model = Experiment(backend)
    before, output = model.q.copy(), io.StringIO()
    with pytest.raises(ValueError):
        write_playback(model, output, **kwargs)
    assert output.getvalue() == ""
    np.testing.assert_array_equal(model.q, before)


def test_cli_export_needs_no_viewer_and_preserves_existing_file(tmp_path):
    code = """
import importlib.abc
import sys
class BlockViewer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'viser', 'trimesh', 'yourdfpy'}:
            raise ImportError('Viewer is unavailable')
sys.meta_path.insert(0, BlockViewer())
from srs_viser.app import main
main()
"""
    output = tmp_path / "motion.csv"
    command = [
        sys.executable,
        "-c",
        code,
        "--demo",
        "trajectory",
        "--plane",
        "world-xz",
        "--path",
        "circle",
        "--path-size",
        "0.015",
        "--frames",
        "30",
        "--export-csv",
        str(output),
    ]
    env = dict(os.environ, PYTHONPATH=str(SOURCE))
    result = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, check=True
    )
    report = json.loads(result.stdout[result.stdout.index("{\n") :])
    assert report["completed"] and report["recorded_frames"] == 30
    assert report["plane"] == "world-xz" and report["path"] == "circle"
    before = output.read_bytes()
    assert len(list(csv.DictReader(io.StringIO(before.decode())))) == 31
    repeated = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert repeated.returncode == 2 and "Cannot write recording" in repeated.stderr
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "args",
    [
        ["--fps", "10"],
        ["--joint-speed", "nan"],
        ["--frames", "0"],
        ["--plane", "invalid"],
    ],
)
def test_cli_rejects_invalid_motion_parameters_before_loading_backend(
    monkeypatch, args
):
    from srs_viser import app

    def unexpected_backend(*args):
        pytest.fail("invalid arguments constructed a solver")

    monkeypatch.setattr(app, "Backend", unexpected_backend)
    monkeypatch.setattr(sys, "argv", ["srs-viser-lab", *args])
    with pytest.raises(SystemExit) as error:
        app.main()
    assert error.value.code == 2
