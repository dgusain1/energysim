"""Mock of DIgSILENT's ``powerfactory`` Python module for CI.

It reproduces just the API surface used by :mod:`energysim.pfAdapter`:

    GetApplication() -> app
    app.ActivateProject(name) -> 0 | non-zero
    app.GetActiveProject() -> project (has Deactivate())
    app.GetFromStudyCase('ComLdf') -> command (SetAttribute / GetAttribute / Execute)
    app.GetCalcRelevantObjects('*') -> list of data objects
    obj.GetAttribute(name) / obj.SetAttribute(name, value) -> 0

The "grid" is a trivial analytic model so tests can predict results:

    Bus1 m:u         = 1 - 0.002 * Load1.e:plini        (AC;  1.0 in DC)
    Load1 m:P:bus1   = Load1.e:plini
    Gen1  m:P:bus1   = -Load1.e:plini                   (balance)

Use ``reset()`` between tests; the module-level ``STATE`` lets tests inject
failures (``execute_rc``, ``fail_activate``, ``no_application``, duplicates).
"""
from __future__ import annotations

STATE = {}


def reset(**overrides):
    """Reset the mock to a clean project 'MockProject' (optionally overriding STATE keys)."""
    STATE.clear()
    STATE.update(
        projects={"MockProject"},
        active=None,
        execute_calls=0,
        execute_rc=0,
        fail_activate=False,
        no_application=False,
        no_comldf=False,
        duplicate_load=False,
        set_rc=0,
        deactivated=0,
        last_ldf_options={},
    )
    STATE.update(overrides)


reset()


class _Obj:
    def __init__(self, loc_name, attrs):
        self._attrs = dict(attrs)
        self._attrs["loc_name"] = loc_name

    def GetAttribute(self, name):
        if name not in self._attrs:
            raise Exception(f"mock powerfactory: attribute '{name}' not found on '{self._attrs['loc_name']}'")
        return self._attrs[name]

    def SetAttribute(self, name, value):
        if name not in self._attrs:
            raise Exception(f"mock powerfactory: attribute '{name}' not found on '{self._attrs['loc_name']}'")
        self._attrs[name] = value
        return STATE["set_rc"]


class _ComLdf(_Obj):
    def __init__(self, grid):
        super().__init__("Load Flow Calculation", {"iopt_net": 0})
        self._grid = grid

    def Execute(self):
        STATE["execute_calls"] += 1
        STATE["last_ldf_options"] = {"iopt_net": self._attrs["iopt_net"]}
        if STATE["execute_rc"]:
            return STATE["execute_rc"]
        self._grid.solve(dc=self._attrs["iopt_net"] == 2)
        return 0


class _Grid:
    def __init__(self):
        self.load = _Obj("Load1", {"e:plini": 10.0, "e:qlini": 2.0, "m:P:bus1": 0.0})
        self.bus = _Obj("Bus1", {"m:u": 1.0, "e:uknom": 20.0})
        self.gen = _Obj("Gen1", {"e:pgini": 0.0, "m:P:bus1": 0.0})
        self.objects = [self.load, self.bus, self.gen]
        if STATE["duplicate_load"]:
            self.objects.append(_Obj("Load1", {"e:plini": 99.0, "e:qlini": 0.0, "m:P:bus1": 0.0}))
        self.solve()

    def solve(self, dc=False):
        p = self.load._attrs["e:plini"]
        self.bus._attrs["m:u"] = 1.0 if dc else 1.0 - 0.002 * p
        self.load._attrs["m:P:bus1"] = p
        self.gen._attrs["m:P:bus1"] = -p


class _Project:
    def __init__(self, name):
        self.name = name

    def Deactivate(self):
        STATE["deactivated"] += 1
        STATE["active"] = None


class _App:
    def __init__(self):
        self._project = None
        self._grid = None
        self._ldf = None

    def ActivateProject(self, name):
        if STATE["fail_activate"] or name not in STATE["projects"]:
            return 1
        self._project = _Project(name)
        self._grid = _Grid()
        self._ldf = None if STATE["no_comldf"] else _ComLdf(self._grid)
        STATE["active"] = name
        return 0

    def GetActiveProject(self):
        return self._project

    def GetFromStudyCase(self, cls):
        return self._ldf if cls == "ComLdf" else None

    def GetCalcRelevantObjects(self, pattern="*", *args):
        return list(self._grid.objects) if self._grid else []


def GetApplication():
    if STATE["no_application"]:
        return None
    return _App()
