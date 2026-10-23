"""Download and unpack the Modelica Reference-FMUs for the test-suite.

The release archive is cached (``tests/.cache`` by default, override with the
``ENERGYSIM_TEST_CACHE`` env var) and verified against a pinned SHA-256 digest.
From the FMI 2.0 FMUs (which contain both a Model Exchange and a Co-Simulation
interface) this script derives single-interface FMUs for the *current
platform*:

    <cache>/fmus/<Name>_cs.fmu   (ModelExchange element removed)
    <cache>/fmus/<Name>_me.fmu   (CoSimulation element removed)

Usage::

    python tests/fixtures/fetch_reference_fmus.py        # download + extract

or, from tests::

    from fixtures.fetch_reference_fmus import get_reference_fmus
    fmus = get_reference_fmus()   # dict or None when offline / unsupported
"""
from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

VERSION = "0.0.39"
URL = f"https://github.com/modelica/Reference-FMUs/releases/download/v{VERSION}/Reference-FMUs-{VERSION}.zip"
SHA256 = "6863d55e5818e1ca4e4614c4d4ba4047a921b4495f6336e7002874ed791f6c2a"
MODELS = ("BouncingBall", "Dahlquist", "VanDerPol", "Stair", "Feedthrough")


def cache_dir() -> Path:
    env = os.environ.get("ENERGYSIM_TEST_CACHE")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / ".cache"


def fmi_platform() -> str | None:
    """FMI platform folder name of the *Python process* (None: unsupported)."""
    bits64 = sys.maxsize > 2 ** 32
    if sys.platform.startswith("win"):
        return "win64" if bits64 else "win32"
    if sys.platform.startswith("linux"):
        return "linux64" if bits64 else None
    if sys.platform == "darwin":
        return "darwin64"
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_archive(dest_dir: Path | None = None) -> Path:
    """Download (if needed) and verify the release archive; return its path."""
    dest_dir = Path(dest_dir) if dest_dir else cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / f"Reference-FMUs-{VERSION}.zip"
    if archive.exists() and _sha256(archive) == SHA256:
        return archive
    tmp = archive.with_suffix(".part")
    with urllib.request.urlopen(URL, timeout=60) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    digest = _sha256(tmp)
    if digest != SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {URL}: expected {SHA256}, got {digest}. Refusing to use the download.")
    tmp.replace(archive)
    return archive


def _derive(src: bytes, tag_to_remove: str, plat: str, name: str, out_path: Path) -> None:
    """Write a single-interface FMU holding only the current platform binary."""
    with zipfile.ZipFile(__import__("io").BytesIO(src)) as zin, \
            zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            fname = item.filename
            if fname.startswith("binaries/") and not fname.startswith(f"binaries/{plat}/"):
                continue
            if fname.startswith("sources/") or fname.startswith("documentation/"):
                continue
            data = zin.read(fname)
            if fname == "modelDescription.xml":
                text = data.decode("utf-8")
                text = re.sub(rf"<{tag_to_remove}\b.*?</{tag_to_remove}>", "", text, flags=re.S)
                data = text.encode("utf-8")
            zout.writestr(item, data)


def get_reference_fmus() -> dict | None:
    """Return ``{'Dahlquist': {'cs': Path, 'me': Path}, ...}`` or None if unavailable."""
    plat = fmi_platform()
    if plat is None:
        return None
    try:
        archive = download_archive()
    except Exception:
        return None
    out_dir = cache_dir() / "fmus" / f"{VERSION}-{plat}"
    result = {}
    with zipfile.ZipFile(archive) as z:
        for model in MODELS:
            cs, me = out_dir / f"{model}_cs.fmu", out_dir / f"{model}_me.fmu"
            if not (cs.exists() and me.exists()):
                out_dir.mkdir(parents=True, exist_ok=True)
                raw = z.read(f"2.0/{model}.fmu")
                _derive(raw, "ModelExchange", plat, model, cs)
                _derive(raw, "CoSimulation", plat, model, me)
            result[model] = {"cs": cs, "me": me}
    return result


if __name__ == "__main__":
    fmus = get_reference_fmus()
    if fmus is None:
        print("Reference FMUs unavailable (offline or unsupported platform).")
        sys.exit(1)
    for k, v in fmus.items():
        print(k, v["cs"], v["me"])
    print("platform:", fmi_platform(), "machine:", platform.machine())
