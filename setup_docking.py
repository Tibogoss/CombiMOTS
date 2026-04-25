#!/usr/bin/env python3
"""Fresh-clone helper for configuring CombiMOTS docking dependencies."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "combimots"))

from pmcts.docking.setup_quickvina import main  # noqa: E402


if __name__ == "__main__":
    main()
