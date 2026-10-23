"""The claim registry (features.yaml) must be backed by real tests and real docs pages."""
from __future__ import annotations

import ast
import fnmatch
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
NEEDS_TESTS = {"stable", "beta", "experimental"}
STATUSES = NEEDS_TESTS | {"planned", "removed"}


def _load():
    return yaml.safe_load((ROOT / "features.yaml").read_text(encoding="utf8"))


@lru_cache(maxsize=None)
def _test_names(path: Path) -> frozenset:
    """Names of top-level test functions, test classes and their methods in *path*."""
    tree = ast.parse(path.read_text(encoding="utf8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(sub.name)
    return frozenset(names)


def _resolve(ref: str):
    """Return an error message if *ref* does not resolve, else None."""
    file_part, _, pattern = ref.partition("::")
    path = ROOT / file_part
    if not path.is_file():
        return f"test file does not exist: {file_part}"
    if not pattern:
        return None
    if not any(fnmatch.fnmatchcase(n, pattern) for n in _test_names(path)):
        return f"no test matching '{pattern}' in {file_part}"
    return None


FEATURES = _load()["features"]


def test_registry_is_well_formed():
    data = _load()
    groups = set(data["groups"])
    ids = [f["id"] for f in FEATURES]
    assert len(ids) == len(set(ids)), "duplicate feature ids"
    for f in FEATURES:
        for key in ("id", "group", "title", "description", "status", "since", "tests", "docs"):
            assert key in f, f"{f.get('id')}: missing '{key}'"
        assert f["status"] in STATUSES, f"{f['id']}: bad status {f['status']!r}"
        assert f["group"] in groups, f"{f['id']}: unknown group {f['group']!r}"
        assert "\n" not in f["description"].strip(), f"{f['id']}: description must be one line"


@pytest.mark.parametrize("feature", FEATURES, ids=[f["id"] for f in FEATURES])
def test_feature_is_backed(feature):
    refs = feature["tests"] or []
    if feature["status"] in NEEDS_TESTS:
        assert refs, f"{feature['id']} is {feature['status']} but lists no tests (a claim needs proof)"
    errors = [e for e in (_resolve(r) for r in refs) if e]
    assert not errors, f"{feature['id']}: " + "; ".join(errors)
    assert (ROOT / feature["docs"]).is_file(), f"{feature['id']}: docs page missing: {feature['docs']}"
