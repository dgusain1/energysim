# -*- coding: utf-8 -*-
"""
Pandapower adapter for energysim.

Variable naming
---------------
Variables are addressed as ``"<element name>.<column>"`` where the element
name is the ``name`` column of any element table (``load``, ``sgen``,
``gen``, ``storage``, ``ext_grid``, ``line``, ``trafo``, ``bus``).  The
split happens at the *last* dot whose left-hand side is a known element name,
so element names that themselves contain dots work.  Results are read from
the ``res_<table>`` DataFrame when the column exists there (e.g.
``line1.loading_percent``, ``line1.i_ka``, ``line1.p_from_mw``,
``trafo1.loading_percent``, ``bus1.vm_pu``) and otherwise from the element
table itself.  Element names that exist in several tables can be
disambiguated with a ``table:`` prefix, e.g. ``"bus:Grid.vm_pu"``.

Re-runnable
-----------
A pristine deep copy of the network is taken at construction.  The adapter
restores it on the next ``init()`` after ``cleanup()`` so a world can be
simulated repeatedly with identical results.  ``cleanup()`` never deletes the
network, so results stay inspectable after a run.
"""
from __future__ import annotations

import copy
import warnings

try:
    import pandapower as pp
except ImportError:
    pp = None

# Check once whether numba is available; if not, pass numba=False
# to all pp.run* calls to suppress the noisy print() warnings.
try:
    import numba  # noqa: F401
    _NUMBA = True
except ImportError:
    _NUMBA = False

from .base import SimulatorAdapter, SimulatorVariableError, SimulatorElementNotFoundError

# Element tables addressable by name.  Order = precedence when a name occurs
# in more than one table (the first table wins and a warning is emitted).
_KINDS = ('load', 'sgen', 'gen', 'storage', 'ext_grid', 'line', 'trafo', 'bus')
# Tables that must have non-empty names (legacy requirement).
_REQUIRED_NAMED = ('load', 'sgen', 'gen', 'ext_grid', 'bus')
_PICKLE_EXT = ('.p', '.pkl', '.pickle')


class pp_adapter(SimulatorAdapter):
    """Adapter for pandapower networks (AC and DC power flow)."""

    SET_PARAMS_BEFORE_INIT = False

    def __init__(self, network_name, net_loc=None, inputs=None, outputs=None, mode='ac', pf=None, model=None):
        if pp is None:
            raise ImportError(
                "pandapower is required for pp_adapter. "
                "Install it using: pip install pandapower (or pip install energysim[powerflow])"
            )
        '''
        Initialises the pandapower network adapter for energysim cosimulaiton object.
        Specify the following:
                network_name: unique name for your network
                net_loc: location of the pandapower network file (.json, or pickle .p/.pkl)
                model: in-memory pandapowerNet instance, factory callable, or import path
                inputs: list of all inputs. Specified as object_name.variable
                outputs: list of all output variables to record.
                mode: power flow calculation mode: 'ac' (default) or 'dc'.
        '''
        self.network_name = network_name
        target = model if model is not None else net_loc
        if target is None:
            raise ValueError(f"No network or model specified for pandapower simulator '{network_name}'.")

        from .utils import resolve_model_reference
        target = resolve_model_reference(target)

        if hasattr(target, 'bus') and hasattr(target, 'line'):
            self.network = target
            self.net_loc = getattr(target, 'name', None) or network_name
        elif isinstance(target, str):
            self.net_loc = target
            self.network = self._load_file(target)
        elif callable(target):
            self.network = target()
            self.net_loc = getattr(self.network, 'name', None) or network_name
        else:
            self.network = target
            self.net_loc = network_name

        if not self.check_names(self.network):
            warnings.warn(
                f"pandapower network '{network_name}' has unnamed load/sgen/gen/ext_grid/bus "
                f"elements. They cannot be addressed by energysim (only named elements can).",
                UserWarning, stacklevel=2)
        self.inputs = list(inputs or [])
        self.outputs = list(outputs or [])

        # Standardized powerflow mode: strictly 'ac' or 'dc'
        raw_mode = (mode if (mode != 'ac' or pf is None) else pf) or 'ac'
        raw_mode = raw_mode.lower()
        if raw_mode in ('ac', 'pf'):
            self.mode = 'ac'
        elif raw_mode in ('dc', 'dcpf'):
            self.mode = 'dc'
        else:
            raise ValueError(
                f"Unsupported calculation mode '{raw_mode}'. "
                f"energysim only supports 'ac' and 'dc' power flow modes."
            )
        self.pf = self.mode  # backward compatibility alias

        # Pristine snapshot used to make the adapter re-runnable.
        self._pristine = copy.deepcopy(self.network)
        self._needs_restore = False

        self._name_idx = {}
        self._name_type = {}
        self._build_caches()
        self._avail_cache = None

        # Fail early on typos in the declared inputs / outputs.
        for item in self.inputs:
            self._resolve(item, for_set=True)
        for item in self.outputs:
            self._resolve(item)

    # ------------------------------------------------------------------
    # Loading / caches
    # ------------------------------------------------------------------
    def _load_file(self, path):
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"pandapower network file not found: {path}")
        low = path.lower()
        if low.endswith('.json'):
            return pp.from_json(path)
        if low.endswith(_PICKLE_EXT):
            warnings.warn(
                f"Loading pandapower network '{path}' with pickle. Pickle files can execute "
                f"arbitrary code when loaded: only load files from a trusted source, or convert "
                f"the network to JSON (pandapower.to_json).",
                UserWarning, stacklevel=3)
            return pp.from_pickle(path)
        raise ValueError(
            f"Unsupported pandapower network file '{path}'. Use .json (recommended) or "
            f"a trusted pickle file ({', '.join(_PICKLE_EXT)})."
        )

    def check_names(self, network):
        for n in _REQUIRED_NAMED:
            df = getattr(network, n, None)
            if df is None or len(df) == 0:
                continue  # skip empty element tables
            if not (df["name"].notna().all() and (df["name"] != "").all()):
                return False
        return True

    def _build_caches(self):
        """Build name -> index / table caches (first table in _KINDS wins)."""
        self._name_idx = {}
        self._name_type = {}
        self._by_kind = {}
        duplicates = {}
        for kind in _KINDS:
            df = getattr(self.network, kind, None)
            if df is None or len(df) == 0 or 'name' not in df.columns:
                continue
            kmap = {}
            for idx, name in df['name'].items():
                if name is None or name == '' or name != name:  # None / '' / NaN
                    continue
                kmap[name] = idx
                if name in self._name_idx:
                    duplicates.setdefault(name, [self._name_type[name]]).append(kind)
                else:
                    self._name_idx[name] = idx
                    self._name_type[name] = kind
            self._by_kind[kind] = kmap
        if duplicates:
            detail = "; ".join(f"'{n}' in {kinds}" for n, kinds in duplicates.items())
            warnings.warn(
                f"pandapower network '{self.network_name}': element names occur in several tables ({detail}). "
                f"The first table wins; use a 'table:' prefix (e.g. 'bus:name.vm_pu') to disambiguate.",
                UserWarning, stacklevel=3)

    # ------------------------------------------------------------------
    # Parameter parsing
    # ------------------------------------------------------------------
    def _resolve(self, parameter, for_set=False):
        """Return ``(kind, idx, column)`` for ``'name.column'`` / ``'kind:name.column'``."""
        if not isinstance(parameter, str) or '.' not in parameter:
            raise SimulatorVariableError(
                f"Malformed pandapower variable '{parameter}'. Expected 'element_name.column'.")
        explicit_kind = None
        body = parameter
        if ':' in body:
            head, rest = body.split(':', 1)
            if head in _KINDS:
                explicit_kind, body = head, rest
        # Try split points from the left-most dot (names rarely contain dots,
        # but then the shortest matching element name wins deterministically).
        positions = [i for i, c in enumerate(body) if c == '.']
        for pos in positions:
            name, column = body[:pos], body[pos + 1:]
            if explicit_kind is not None:
                idx = self._by_kind.get(explicit_kind, {}).get(name)
                if idx is not None:
                    return explicit_kind, idx, column
            elif name in self._name_idx:
                return self._name_type[name], self._name_idx[name], column
        name = body.split('.', 1)[0]
        raise SimulatorElementNotFoundError(
            f"Could not find '{name}' as a component in network '{self.network_name}'."
            f" Make sure element names are correctly specified."
        )

    # ------------------------------------------------------------------
    # Life-cycle
    # ------------------------------------------------------------------
    def _restore_pristine(self):
        net = self.network
        fresh = copy.deepcopy(self._pristine)
        try:
            # keep the identity of the user's network object
            net.clear()
            net.update(fresh)
        except Exception:
            self.network = fresh
        self._needs_restore = False

    def _ensure_pristine(self):
        if self._needs_restore:
            self._restore_pristine()

    def _run(self):
        if self.mode == 'ac':
            return pp.runpp(self.network, numba=_NUMBA)
        return pp.rundcpp(self.network, numba=_NUMBA)

    def init(self):
        self._ensure_pristine()
        self._run()

    def cleanup(self):
        # Never delete the network: results must stay inspectable and the
        # adapter must be re-runnable. Restoration happens lazily.
        self._needs_restore = True

    # Backward compatibility alias
    cleanUp = cleanup

    # ------------------------------------------------------------------
    # set / get
    # ------------------------------------------------------------------
    def _poly_cost_row(self, kind, idx, parameter):
        if kind == 'bus':
            raise SimulatorVariableError(f"Bus variable '{parameter}' cannot have cost parameters.")
        pc = self.network.poly_cost
        matching = pc[(pc['et'] == kind) & (pc['element'] == idx)].index
        if len(matching) == 0:
            raise SimulatorVariableError(f"Cost parameter for {parameter} not found in poly_cost table.")
        return matching[0]

    def set_value(self, parameters, values):
        '''
        Must specify parameters and values in list format
        '''
        self._ensure_pristine()
        for parameter, value in zip(parameters, values):
            kind, idx, column = self._resolve(parameter, for_set=True)
            table = getattr(self.network, kind)
            if column in table.columns:
                table.at[idx, column] = value
            elif column == 'cp1_eur_per_mw':
                row = self._poly_cost_row(kind, idx, parameter)
                self.network.poly_cost.at[row, 'cp1_eur_per_mw'] = value
            else:
                raise SimulatorVariableError(
                    f"Parameter {parameter} not found in network {self.network_name}."
                )

    def get_value(self, parameters, time=None):
        '''
        Must specify parameter in a list format.
        '''
        out = []
        for parameter in parameters:
            kind, idx, column = self._resolve(parameter)
            res_table = getattr(self.network, f'res_{kind}', None)
            base_table = getattr(self.network, kind)
            if res_table is not None and column in res_table.columns:
                if idx in res_table.index:
                    out.append(res_table.at[idx, column])
                    continue
                if column not in base_table.columns:
                    raise SimulatorVariableError(
                        f"No result available for '{parameter}': run a power flow first "
                        f"(adapter.init() / step())."
                    )
            if column in base_table.columns:
                out.append(base_table.at[idx, column])
            elif column == 'cp1_eur_per_mw':
                row = self._poly_cost_row(kind, idx, parameter)
                out.append(self.network.poly_cost.at[row, 'cp1_eur_per_mw'])
            else:
                raise SimulatorVariableError(
                    f"Variable '{column}' does not exist in pandapower network '{self.network_name}' "
                    f"for element '{parameter.split('.')[0]}'."
                )
        return out

    def getOutput(self):
        return self.get_value(self.outputs)

    def setInput(self, inputValues):
        self.set_value(self.inputs, inputValues)

    def step(self, time):
        return self._run()

    # ------------------------------------------------------------------
    # Introspection (vectorised and cached)
    # ------------------------------------------------------------------
    def get_available_variables(self):
        """Return ``element.column`` names: inputs (element table columns)
        and outputs (result columns plus element table columns)."""
        if self._avail_cache is None:
            inputs_list, outputs_list = [], []
            for kind, kmap in self._by_kind.items():
                table = getattr(self.network, kind)
                cols = [c for c in table.columns if c != 'name']
                res = getattr(self.network, f'res_{kind}', None)
                res_cols = list(res.columns) if res is not None else []
                out_cols = res_cols + [c for c in cols if c not in res_cols]
                has_cost = (kind != 'bus' and getattr(self.network, 'poly_cost', None) is not None
                            and len(self.network.poly_cost) > 0)
                for name, idx in kmap.items():
                    names = [name + '.' + c for c in cols]
                    onames = [name + '.' + c for c in out_cols]
                    if has_cost:
                        pc = self.network.poly_cost
                        if ((pc['et'] == kind) & (pc['element'] == idx)).any():
                            names.append(name + '.cp1_eur_per_mw')
                            onames.append(name + '.cp1_eur_per_mw')
                    inputs_list.extend(names)
                    outputs_list.extend(onames)
            self._avail_cache = {'inputs': inputs_list, 'outputs': outputs_list}
        return {k: list(v) for k, v in self._avail_cache.items()}
