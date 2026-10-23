"""README.md's generated regions (features, install extras, parameter tables) must be in sync."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_generated_regions_are_in_sync():
    pytest.importorskip("yaml")
    gen = _load_script("generate_readme_features")
    current = (ROOT / "README.md").read_text(encoding="utf8").replace("\r\n", "\n")
    assert gen.sync(current) == current, (
        "README.md is out of date with features.yaml / pyproject.toml / the world() signatures. "
        "Run: python scripts/generate_readme_features.py"
    )


def test_readme_has_no_removed_api_claims():
    text = (ROOT / "README.md").read_text(encoding="utf8")
    for stale in ("save_state", "restore_state", "max_iterations", "convergence_tolerance", "'iterative'"):
        assert stale not in text.replace("save_state / restore_state were removed", ""), stale
