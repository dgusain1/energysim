# -*- coding: utf-8 -*-
"""Shared pytest configuration: make ``src/`` and ``tests/`` importable (fixtures/mocks/helpers)
and provide a private working-directory fixture."""
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"

for _p in (str(TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    """Run the test inside a private temporary working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
