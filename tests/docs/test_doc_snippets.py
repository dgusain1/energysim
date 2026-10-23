"""Execute every ```python fenced block of README.md and docs/**/*.md.

Rules
-----
* All blocks of one file run in order, in one shared namespace, inside a private temporary working directory.
* A block is skipped when its info string contains ``no-test`` or ``notest`` (``python {.no-test}``,
  ``python notest``), or when it imports a package that is not installed (``importorskip``).
* Pages listed in ``tests/docs/known_stale.txt`` are collected and run, but a failure is an ``xfail``
  (non-strict) instead of an error: the page's snippets are known to be out of date. Remove the page from
  that file once its snippets pass (an xpass is reported by ``-rX``).
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
import traceback
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STALE_FILE = Path(__file__).with_name("known_stale.txt")

_FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})\s*(?P<info>.*)$")
_STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
# import name -> package that must be importable
_SKIP_ROOTS = {"energysim"}


def known_stale() -> set:
    if not STALE_FILE.exists():
        return set()
    out = set()
    for ln in STALE_FILE.read_text(encoding="utf8").splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            out.add(ln.replace("\\", "/"))
    return out


def doc_files():
    files = [ROOT / "README.md"] + sorted((ROOT / "docs").rglob("*.md"))
    return [f for f in files if f.is_file()]


def _lang_and_tags(info: str):
    info = info.strip()
    m = re.match(r"^\{\s*(.*?)\s*\}$", info)
    if m:  # attr_list style: { .python .no-test }
        toks = m.group(1).split()
        lang = next((t[1:] for t in toks if t.startswith(".") and t[1:] not in ("no-test", "notest")), "")
        return lang.lower(), " ".join(toks).lower()
    toks = info.split(None, 1)
    return (toks[0].lower() if toks else ""), (toks[1].lower() if len(toks) > 1 else "")


def extract_blocks(path: Path):
    """Return ``[(line_number, code, skip_reason_or_None)]`` for every python fence in *path*."""
    lines = path.read_text(encoding="utf8").splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        fence, indent = m.group("fence"), len(m.group("indent").expandtabs(4))
        lang, tags = _lang_and_tags(m.group("info"))
        start = i + 1
        j = start
        while j < len(lines):
            mm = _FENCE.match(lines[j])
            if mm and mm.group("fence")[0] == fence[0] and len(mm.group("fence")) >= len(fence) \
                    and not mm.group("info").strip():
                break
            j += 1
        body = lines[start:j]
        i = j + 1
        if lang not in ("python", "py", "python3"):
            continue
        body = [ln[indent:] if ln[:indent].strip() == "" else ln for ln in body]
        reason = "tagged no-test" if ("no-test" in tags or "notest" in tags) else None
        blocks.append((start + 1, "\n".join(body) + "\n", reason))
    return blocks


def _missing_imports(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.append(node.module.split(".")[0])
    out = []
    for m in dict.fromkeys(mods):
        if m in _STDLIB or m in _SKIP_ROOTS or m == "__main__":
            continue
        if importlib.util.find_spec(m) is None:
            out.append(m)
    return out


def _cases():
    cases = []
    for f in doc_files():
        rel = f.relative_to(ROOT).as_posix()
        for lineno, code, reason in extract_blocks(f):
            cases.append(pytest.param(rel, lineno, code, reason, id=f"{rel}:{lineno}"))
    return cases


CASES = _cases()
_NAMESPACES: dict = {}
_SKIPPED_DEP: set = set()
_WORKDIRS: dict = {}


@pytest.fixture(autouse=True)
def _headless(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)
    monkeypatch.setattr(webbrowser, "open_new_tab", lambda *a, **k: True)


@pytest.mark.parametrize("rel,lineno,code,reason", CASES)
def test_snippet(rel, lineno, code, reason, tmp_path_factory, monkeypatch, request):
    if reason:
        pytest.skip(reason)
    missing = _missing_imports(code)
    if missing:
        _SKIPPED_DEP.add(rel)
        pytest.skip(f"optional dependency not installed: {', '.join(missing)}")

    workdir = _WORKDIRS.setdefault(rel, tmp_path_factory.mktemp(re.sub(r"\W+", "_", rel)))
    monkeypatch.chdir(workdir)
    ns = _NAMESPACES.setdefault(rel, {"__name__": "__doc_snippet__"})
    stale = rel in known_stale()
    try:
        exec(compile(code, f"{rel}:{lineno}", "exec"), ns)  # noqa: S102 - executing our own documentation
    except NameError as exc:
        if rel in _SKIPPED_DEP:
            pytest.skip(f"depends on a skipped block ({exc})")
        _fail(stale, exc, rel, lineno)
    except Exception as exc:  # noqa: BLE001
        _fail(stale, exc, rel, lineno)
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        plt.close("all")


def _fail(stale: bool, exc: BaseException, rel: str, lineno: int):
    tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    if stale:
        pytest.xfail(f"known stale page {rel}: {tb}")
    raise AssertionError(f"{rel}:{lineno}: snippet failed -> {tb}") from exc


def test_known_stale_entries_exist():
    existing = {f.relative_to(ROOT).as_posix() for f in doc_files()}
    unknown = known_stale() - existing
    assert not unknown, f"known_stale.txt lists pages that do not exist: {sorted(unknown)}"
