"""Run fixed-orientation Cartesian trajectory tracking in Viser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srs_viser.app import main  # noqa: E402

if __name__ == "__main__":
    main("trajectory")
