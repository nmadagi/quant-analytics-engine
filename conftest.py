"""Pytest configuration — ensures the repo root is importable in CI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
