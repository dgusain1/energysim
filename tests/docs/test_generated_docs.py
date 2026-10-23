"""Committed generated documentation must equal a fresh regeneration from the code."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    pytest.importorskip("yaml")
    gen = _load_script("generate_reference_docs")
    files = gen.generate()
    out = tmp_path_factory.mktemp("generated_docs")
    gen.write(files, str(out))
    return gen, files, out


def test_reference_docs_are_in_sync(generated):
    gen, files, out = generated
    stale = []
    for rel in files:
        committed = DOCS / rel
        fresh = (out / rel).read_text(encoding="utf8")
        if not committed.exists() or committed.read_text(encoding="utf8").replace("\r\n", "\n") != fresh:
            stale.append(rel)
    assert not stale, (
        "Generated docs are out of date: " + ", ".join(sorted(stale))
        + "\nRegenerate with:  python scripts/generate_reference_docs.py"
    )


def test_units_doc_is_in_sync():
    gen = _load_script("generate_units_doc")
    committed = (DOCS / "units.md").read_text(encoding="utf8").replace("\r\n", "\n")
    assert gen.render() == committed, "docs/units.md is out of date. Run: python scripts/generate_units_doc.py"


def test_every_model_page_exists_and_documents_validity(generated):
    gen, files, _ = generated
    for cname, cls in gen.model_classes().items():
        doc = inspect.getdoc(cls) or ""
        assert "Validity" in doc, f"{cname}: the docstring must state the validity range (Notes -> 'Validity: ...')"


def test_every_model_variable_has_a_unit():
    gen = _load_script("generate_reference_docs")
    for cname, cls in gen.model_classes().items():
        d = cls.describe_class()
        missing = [v for v in d["inputs"] + d["outputs"] if v not in d["units"]]
        assert not missing, f"{cname}: variables without a declared unit: {missing}"
