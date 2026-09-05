"""Run the SRS arm-angle, branch-gallery or Jacobian experiment from a checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srs_viser.app import main  # noqa: E402

if __name__ == "__main__":
    main()
