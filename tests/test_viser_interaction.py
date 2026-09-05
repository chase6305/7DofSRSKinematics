"""Optional smoke test of the render loop when visualization extras are installed."""

import numpy as np
import pytest

pytest.importorskip("viser")
pytest.importorskip("trimesh")

from srs_viser.app import LabApp
from srs_viser.backend import Backend
from srs_viser.model import MODES, PRESETS
from srs_viser.motions import PROJECTIONS


def test_idle_coalescing_mode_switches_and_reset(monkeypatch):
    backend = Backend()
    app = LabApp(backend, port=0, meshes=False)
    try:
        count = app.render_count
        for _ in range(10):
            assert not app.step(1 / 30)
        assert app.render_count == count
        calls = []
        solve = backend.solve

        def counted(*args):
            calls.append(args)
            return solve(*args)

        monkeypatch.setattr(backend, "solve", counted)
        for angle in range(20):
            app.commands.put(("angle", angle))
        assert app.step(1 / 30)
        assert len(calls) == 1
        assert app.angle.value == pytest.approx(19)
        app.commands.put(("mode", MODES["branches"]))
        app.step(1 / 30)
        assert app.branch_table.visible
        assert not app.target.visible
        assert sum(bones.visible for _, bones, _, _, _ in app.gallery) > 0
        app.commands.put(("mode", MODES["jacobian"]))
        app.commands.put(("play", True))
        app.step(1 / 30)
        cost = app.model.centering_cost()
        app.step(1 / 30)
        assert app.model.centering_cost() < cost
        assert app.ellipsoid.visible
        assert app.target.visible
        app.commands.put(("reset", None))
        app.step(1 / 30)
        assert not app.play.value
        np.testing.assert_allclose(app.model.q, PRESETS["Bent arm"])
        # Programmatic GUI synchronization must not feed back into the solver.
        for _ in range(5):
            assert not app.step(1 / 30)
    finally:
        app.close()


def test_trajectory_and_manifold_controls_render_and_reanchor():
    app = LabApp(Backend(), port=0, meshes=False)
    try:
        app.commands.put(("mode", MODES["trajectory"]))
        app.commands.put(("path_kind", "Circle"))
        app.commands.put(("play", True))
        assert app.step(1 / 30)
        assert app.path_root.visible and app.path_trace.visible
        assert app.path_kind.value == "Circle"
        assert app.model.trajectory_steps == 1
        assert "Accepted steps" in app.metrics.content
        q = app.model.q.copy()
        app.commands.put(("mode", MODES["manifold"]))
        assert app.step(1 / 30)
        assert app.manifold_root.visible and not app.path_root.visible
        assert app.branch_table.visible
        assert "Feasible samples" in app.branch_table.content
        assert any(handle.visible for handle in app.manifold_curves)
        assert app.manifold_current.points.shape == (1, 3)
        np.testing.assert_array_equal(app.model.anchor_q, q)
        assert not app.play.value
        assert not app.step(1 / 30)
    finally:
        app.close()


def test_manifold_projection_cache_and_speed_controls(monkeypatch):
    import srs_viser.app as app_module

    calls = []
    segments = app_module.manifold_segments

    def counted(*args, **kwargs):
        calls.append(kwargs["axes"])
        return segments(*args, **kwargs)

    monkeypatch.setattr(app_module, "manifold_segments", counted)
    app = LabApp(Backend(), mode="manifold", port=0, meshes=False)
    try:
        data = app.model.manifold()
        assert len(calls) == 8
        app.commands.put(("max_joint_speed", 0.1))
        app.commands.put(("play", True))
        for _ in range(10):
            q = app.model.q.copy()
            assert app.step(1 / 30)
            assert np.max(np.abs(app.model.q - q)) <= 0.1 / 30 + 1e-8
        assert app.max_joint_speed.visible
        assert len(calls) == 8
        app.commands.put(("manifold_projection", "J5 / J6 / J7"))
        app.step(1 / 30)
        assert len(calls) == 16 and calls[-1] == (4, 5, 6)
        assert app.model.manifold() is data
        assert [h.text for h in app.manifold_axis_labels] == [
            "J5 [rad]",
            "J6 [rad]",
            "J7 [rad]",
        ]
        axes = PROJECTIONS[app.manifold_projection.value]
        np.testing.assert_array_equal(
            app.manifold_current.points,
            (app.model.q[None, list(axes)] * 0.22 + app.manifold_origin).astype(
                app.manifold_current.points.dtype
            ),
        )
        app.commands.put(("amplitude", 90))
        q = app.model.q.copy()
        app.step(1 / 30)
        assert np.max(np.abs(app.model.q - q)) <= 0.1 / 30 + 1e-8
        app.commands.put(("manifold_samples", 41))
        app.step(1 / 30)
        assert len(calls) == 24
        app.commands.put(("mode", MODES["branches"]))
        app.step(1 / 30)
        assert "Min margin" in app.branch_table.content
        app.commands.put(("mode", MODES["manifold"]))
        app.step(1 / 30)
        assert len(calls) == 24
        assert "Feasible samples" in app.branch_table.content
        app.commands.put(("reset", None))
        app.step(1 / 30)
        assert len(calls) == 32
    finally:
        app.close()


def test_path_plane_switch_reanchors_and_updates_displacement_plot():
    app = LabApp(Backend(), mode="trajectory", port=0, meshes=False)
    try:
        app.commands.put(("play", True))
        app.step(1 / 30)
        q, target = app.model.q.copy(), app.model.target.copy()
        app.commands.put(("path_plane", "World XZ"))
        app.step(1 / 30)
        np.testing.assert_array_equal(app.model.q, q)
        np.testing.assert_array_equal(app.model.trajectory.start, target)
        assert app.model.trajectory.phase == 0 and not app.model.play
        assert app.model.path_plane == "world-xz" and app.path_plane.visible
        assert "World XZ" in app.path_plot_title.text
        assert [h.text for h in app.path_plot_labels] == ["Δx", "Δz"]
        assert app.model.trajectory_steps == 0 and len(app.model.trail) == 1
        app.commands.put(("path_plane", "TCP XY"))
        app.commands.put(("play", True))
        app.step(1 / 30)
        assert app.model.trajectory_steps == 1
        assert [h.text for h in app.path_plot_labels] == ["ΔTCP x", "ΔTCP y"]
        np.testing.assert_allclose(
            app.model.trajectory.basis, target[:3, :2], atol=1e-7
        )
        app.commands.put(("mode", MODES["manifold"]))
        app.step(1 / 30)
        assert not app.path_plane.visible
    finally:
        app.close()
