"""
Start the Autonomous Cleaning Vehicle visualizer.

    python run.py                                 # control panel
    python run.py --version v2 --model q_learning # skip the panel
    python run.py --list                          # what is available

Works from a fresh checkout; no installation step is required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cleaning_vehicle.app.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
