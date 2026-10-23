#!/usr/bin/env python
"""Manual PowerFactory check for energysim (run on a PC with a PowerFactory licence).

The CI suite only tests ``energysim.pfAdapter`` against a mock.  This script
exercises the *real* PowerFactory Python API with the same calls the adapter
makes, then (optionally) runs the adapter itself, and writes a PASS/FAIL
report to stdout and to a JSON file you can send back.

Example (Windows, PowerFactory 2024, project "IEEE14")::

    python tools/powerfactory_manual_check.py ^
        --project IEEE14 ^
        --pf-path "C:\\Program Files\\DIgSILENT\\PowerFactory 2024\\Python\\3.11" ^
        --element "Bus 05" --attr m:u ^
        --input-element "Load 05" --input-attr plini

Arguments
---------
--project        PowerFactory project name to activate (required)
--pf-path        directory that contains powerfactory.pyd (required unless the
                 module is already importable). Must match your Python version.
--element        loc_name of an element to read a result from (required)
--attr           result attribute to read, e.g. m:u, m:Psum:bus1 (required)
--input-element  element whose input will be changed (default: --element)
--input-attr     input attribute to change, e.g. plini or e:plini
                 (omit to skip the set-input steps)
--delta          relative change applied to the input (default 0.10 = +10%)
--test-dc        also run a DC load flow (ComLdf iopt_net=2) and restore AC
--skip-adapter   do not run the energysim adapter check
--json-out       output JSON file (default: powerfactory_check_result.json)

Exit code 0 when every mandatory step passed, 1 otherwise.
Nothing is saved to the project: the changed input is restored at the end
and the project is deactivated without saving.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback

RESULTS = []


def record(name, status, detail="", data=None):
    """status: PASS | FAIL | WARN | SKIP"""
    RESULTS.append({"step": name, "status": status, "detail": str(detail), "data": data})
    print(f"[{status:4}] {name}" + (f" -- {detail}" if detail else ""))


def step(name, mandatory=True):
    """Decorator: run fn(), record PASS/FAIL (exceptions become FAIL)."""
    def deco(fn):
        try:
            out = fn()
            if isinstance(out, tuple):
                detail, data = out
            else:
                detail, data = (out or ""), None
            record(name, "PASS", detail, data)
            return True
        except Exception as exc:  # noqa: BLE001
            record(name, "FAIL" if mandatory else "WARN", f"{type(exc).__name__}: {exc}",
                   traceback.format_exc(limit=3))
            return False
    return deco


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--pf-path", default=None)
    ap.add_argument("--element", required=True)
    ap.add_argument("--attr", required=True)
    ap.add_argument("--input-element", default=None)
    ap.add_argument("--input-attr", default=None)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--test-dc", action="store_true")
    ap.add_argument("--skip-adapter", action="store_true")
    ap.add_argument("--json-out", default="powerfactory_check_result.json")
    args = ap.parse_args(argv)

    started = time.time()
    env = {
        "python": sys.version.split()[0],
        "python_arch": platform.architecture()[0],
        "platform": platform.platform(),
        "pf_path": args.pf_path,
        "project": args.project,
        "element": args.element,
        "attr": args.attr,
    }
    print("=== energysim PowerFactory manual check ===")
    print(json.dumps(env, indent=2))

    state = {}

    if args.pf_path:
        if not os.path.isdir(args.pf_path):
            record("pf-path exists", "FAIL", f"not a directory: {args.pf_path}")
            return finish(args, env, started)
        sys.path.insert(0, args.pf_path)
        record("pf-path exists", "PASS", args.pf_path)

    @step("import powerfactory")
    def _():
        import powerfactory  # type: ignore
        state["pf"] = powerfactory
        return getattr(powerfactory, "__file__", "")

    if "pf" not in state:
        record("hint", "WARN", "Use the PowerFactory Python folder matching your Python minor version "
                              "(e.g. ...\\Python\\3.11 for Python 3.11) and a 64-bit interpreter.")
        return finish(args, env, started)
    pf = state["pf"]

    @step("GetApplication")
    def _():
        app = pf.GetApplication()
        if app is None:
            raise RuntimeError("GetApplication() returned None (no licence / PowerFactory not startable)")
        state["app"] = app

    if "app" not in state:
        return finish(args, env, started)
    app = state["app"]

    @step("ActivateProject")
    def _():
        rc = app.ActivateProject(args.project)
        state["activate_rc"] = rc
        if rc:
            raise RuntimeError(f"ActivateProject returned {rc!r} (non-zero = failure)")
        return f"rc={rc!r}", {"rc": repr(rc)}

    if state.get("activate_rc"):
        return finish(args, env, started)

    @step("GetActiveProject")
    def _():
        proj = app.GetActiveProject()
        if proj is None:
            raise RuntimeError("GetActiveProject() returned None")
        state["proj"] = proj
        return str(proj)

    @step("GetFromStudyCase('ComLdf')")
    def _():
        ldf = app.GetFromStudyCase("ComLdf")
        if ldf is None:
            raise RuntimeError("ComLdf not found - is a study case activated in the project?")
        state["ldf"] = ldf

    if "ldf" not in state:
        cleanup(app, state)
        return finish(args, env, started)
    ldf = state["ldf"]

    @step("ComLdf.Execute (AC load flow)")
    def _():
        rc = ldf.Execute()
        state["exec_rc"] = rc
        if rc:
            raise RuntimeError(f"Execute returned {rc!r}; load flow failed or did not converge")
        return f"rc={rc!r}", {"rc": repr(rc)}

    @step("GetCalcRelevantObjects('*')")
    def _():
        objs = app.GetCalcRelevantObjects("*")
        state["objs"] = list(objs)
        if not state["objs"]:
            raise RuntimeError("no calculation-relevant objects returned")
        return f"{len(state['objs'])} objects"

    @step("build loc_name cache / duplicates")
    def _():
        cache, dups = {}, {}
        for o in state.get("objs", []):
            n = o.GetAttribute("loc_name")
            if not n:
                continue
            if n in cache:
                dups[n] = dups.get(n, 1) + 1
            else:
                cache[n] = o
        state["cache"] = cache
        detail = f"{len(cache)} unique names"
        if dups:
            detail += f"; DUPLICATES (adapter uses first): {dict(list(dups.items())[:10])}"
        return detail, {"duplicates": dups}

    cache = state.get("cache", {})
    ielem = args.input_element or args.element

    @step(f"element '{args.element}' exists")
    def _():
        if args.element not in cache:
            raise KeyError(f"'{args.element}' not found. Examples: {list(cache)[:15]}")
        state["elem"] = cache[args.element]

    def read_attr(obj, attr):
        v = obj.GetAttribute(attr)
        if v is None:
            raise RuntimeError(f"GetAttribute('{attr}') returned None")
        return float(v)

    @step(f"GetAttribute('{args.attr}') is numeric")
    def _():
        v = read_attr(state["elem"], args.attr)
        state["base_value"] = v
        return f"{args.element}.{args.attr} = {v}", {"value": v}

    if args.input_attr:
        iattr = args.input_attr if args.input_attr.startswith(("e:", "m:", "c:", "s:")) else "e:" + args.input_attr

        @step(f"input element '{ielem}' exists")
        def _():
            if ielem not in cache:
                raise KeyError(f"'{ielem}' not found")
            state["ielem"] = cache[ielem]

        @step(f"GetAttribute('{iattr}') (original input)")
        def _():
            state["orig_in"] = read_attr(state["ielem"], iattr)
            return f"{ielem}.{iattr} = {state['orig_in']}", {"value": state["orig_in"]}

        if "orig_in" in state:
            new_in = state["orig_in"] * (1.0 + args.delta) if state["orig_in"] else args.delta

            @step(f"SetAttribute('{iattr}', {new_in})")
            def _():
                rc = state["ielem"].SetAttribute(iattr, float(new_in))
                state["set_rc"] = rc
                if isinstance(rc, int) and not isinstance(rc, bool) and rc != 0:
                    raise RuntimeError(f"SetAttribute returned {rc!r}. NOTE: energysim treats a non-zero "
                                       f"int return as failure; if this is a success code in your "
                                       f"PowerFactory version, report it.")
                return f"rc={rc!r}", {"rc": repr(rc)}

            @step("ComLdf.Execute after input change")
            def _():
                rc = ldf.Execute()
                if rc:
                    raise RuntimeError(f"Execute returned {rc!r}")
                return f"rc={rc!r}"

            @step(f"re-read '{args.element}.{args.attr}' after change")
            def _():
                v = read_attr(state["elem"], args.attr)
                state["new_value"] = v
                changed = abs(v - state["base_value"]) > 1e-12 * max(1.0, abs(v))
                if not changed:
                    record("result responded to input change", "WARN",
                           "result did not change - pick an --attr affected by --input-attr")
                else:
                    record("result responded to input change", "PASS",
                           f"{state['base_value']} -> {v}")
                return f"{v}", {"value": v}

            @step("restore original input", mandatory=False)
            def _():
                state["ielem"].SetAttribute(iattr, float(state["orig_in"]))
                rc = ldf.Execute()
                if rc:
                    raise RuntimeError(f"Execute after restore returned {rc!r}")
                v = read_attr(state["elem"], args.attr)
                if abs(v - state["base_value"]) > 1e-9 * max(1.0, abs(v)):
                    raise RuntimeError(f"result {v} differs from original {state['base_value']} after restore")
                return "input restored, result matches original"
    else:
        record("set-input steps", "SKIP", "no --input-attr given")

    if args.test_dc:
        @step("DC load flow (iopt_net=2) executes", mandatory=False)
        def _():
            old = ldf.GetAttribute("iopt_net")
            try:
                ldf.SetAttribute("iopt_net", 2)
                rc = ldf.Execute()
                if rc:
                    raise RuntimeError(f"DC Execute returned {rc!r}")
                v = read_attr(state["elem"], args.attr)
                return f"{args.element}.{args.attr} (DC) = {v}", {"value": v}
            finally:
                ldf.SetAttribute("iopt_net", old if old is not None else 0)
                ldf.Execute()

    if not args.skip_adapter:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
            from energysim.pfAdapter import pf_adapter  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            record("energysim adapter import", "WARN", f"skipped: {exc}")
        else:
            cleanup(app, state)   # the adapter activates the project itself

            @step("energysim pf_adapter: init/get/set/step/cleanup")
            def _():
                inputs = [f"{ielem}.{args.input_attr}"] if args.input_attr else []
                a = pf_adapter(project_name=args.project, inputs=inputs,
                               outputs=[f"{args.element}.{args.attr}"], pf_path=args.pf_path)
                a.init()
                try:
                    v0 = a.get_value([f"{args.element}.{args.attr}"])[0]
                    if inputs:
                        orig = a.get_value([f"{ielem}.e:{args.input_attr.split(':')[-1]}"])[0]
                        a.set_value(inputs, [orig * (1.0 + args.delta) if orig else args.delta])
                        a.step(0)
                        v1 = a.get_value([f"{args.element}.{args.attr}"])[0]
                        a.set_value(inputs, [orig])
                        a.step(0)
                    else:
                        v1 = v0
                finally:
                    a.cleanup()
                a.init()   # re-runnable
                v2 = a.get_value([f"{args.element}.{args.attr}"])[0]
                a.cleanup()
                if abs(v2 - v0) > 1e-9 * max(1.0, abs(v0)):
                    raise RuntimeError(f"re-init gave {v2}, first run gave {v0}")
                return f"v0={v0} v_changed={v1} v_reinit={v2}", {"v0": v0, "v1": v1, "v2": v2}
            state["proj"] = None
            state["app"] = None
            return finish(args, env, started)

    cleanup(app, state)
    return finish(args, env, started)


def cleanup(app, state):
    @step("Deactivate project (no save)", mandatory=False)
    def _():
        proj = state.get("proj")
        if proj is not None:
            proj.Deactivate()
            state["proj"] = None
        return "deactivated"


def finish(args, env, started):
    failed = [r for r in RESULTS if r["status"] == "FAIL"]
    warned = [r for r in RESULTS if r["status"] == "WARN"]
    overall = "FAIL" if failed else "PASS"
    print("\n=== SUMMARY ===")
    print(f"overall: {overall}   steps: {len(RESULTS)}   failed: {len(failed)}   warnings: {len(warned)}")
    for r in failed:
        print(f"  FAIL: {r['step']}: {r['detail']}")
    report = {"overall": overall, "environment": env, "seconds": round(time.time() - started, 2),
              "steps": RESULTS}
    try:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"JSON report written to {os.path.abspath(args.json_out)}")
    except OSError as exc:
        print(f"Could not write JSON report: {exc}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
