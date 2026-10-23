# -*- coding: utf-8 -*-
"""Shared pytest configuration: make ``src/`` and ``tests/`` importable and
keep result files out of the working directory."""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "src"), _HERE):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    """Run the test inside a private temporary working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
