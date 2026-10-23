"""Shared pytest configuration for the energysim test-suite."""
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"

# make ``fixtures`` / ``mocks`` importable from every test module
for p in (str(TESTS_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
