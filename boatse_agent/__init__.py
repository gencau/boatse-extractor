# boatse_agent/__init__.py
import sys
from pathlib import Path

# Ensure the repo root (parent of this package) is on sys.path so that
# the shared `utils/` package is importable regardless of working directory.
_repo_root = str(Path(__file__).parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from .agent_loop import BoatseAgent
