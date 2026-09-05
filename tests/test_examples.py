"""The headless benchmark must work with only the core dependencies."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["continuous", "global"])
def test_headless_validation_without_visualization_dependencies(mode):
    script = """
import importlib.abc
import runpy
import sys

class BlockVisualization(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"trimesh", "viser", "yourdfpy"}:
            raise ImportError("Visualization dependency is unavailable")

sys.meta_path.insert(0, BlockVisualization())
sys.argv = ["examples/viser_demo.py", "--validate-only", "--validation-samples", "3", "--search-mode", sys.argv[1]]
runpy.run_path("examples/viser_demo.py", run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, mode],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert f"IK [{mode}] 3/3" in result.stdout


def test_benchmark_passes_search_mode_to_solver():
    spec = importlib.util.spec_from_file_location(
        "viser_demo", ROOT / "examples" / "viser_demo.py"
    )
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    class RecordingSolver(demo.KUKAiiwaSolver):
        def get_ik(self, target, seed, **kwargs):
            assert kwargs["search_mode"] == "global"
            assert kwargs["num_samples"] == 9
            return super().get_ik(target, seed, **kwargs)

    assert demo.benchmark(RecordingSolver(), 3, 9, "global")["successes"] == 3


def test_importing_library_preserves_application_logging():
    script = """
import logging
import sys
sys.path.insert(0, "src")
root = logging.getLogger()
handler = logging.NullHandler()
root.addHandler(handler)
root.setLevel(logging.ERROR)
import srs_kinematics
assert root.handlers == [handler]
assert root.level == logging.ERROR
"""
    subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True
    )
