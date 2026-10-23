# -*- coding: utf-8 -*-
"""
World cosimulation canvas for energysim.
"""

from __future__ import annotations
from typing import Any, Optional, List, Dict

from .csv_adapter import csv_simulator
from .signalAdapter import signal_adapter
from .utils import (
    get_unit_conversion, infer_variable_unit, parse_unit,
    resolve_model_reference, looks_like_dotted_import, PICKLE_EXTS, MODEL_FILE_EXTS,
)
from ._safeexpr import build_signal
from .results import Results, Recorder, make_run_dir, scenario_hash, _have_pyarrow, _versions
from .base import (
    SimulatorAdapter, SimEntry,
    ConnectionSpecError,
    ValidationError, ValidationReport,
    ParsedConnection,
)
import os
import sys
import re
import math
import inspect
import itertools
import importlib
import importlib.util
import logging as _logging
import warnings
import time as _time
import datetime as _dt
from collections import deque
from fractions import Fraction
from functools import reduce
import numpy as np
import yaml
from tqdm import tqdm
import networkx as nx

logger = _logging.getLogger('energysim')

ITERATIVE_REMOVED_MSG = (
    "iterative coupling was removed in 3.0; use 'jacobi' or 'gauss-seidel', "
    "reduce t_macro, or merge coupled components"
)
_COUPLINGS = ('jacobi', 'gauss-seidel')
_EXTRAPOLATIONS = ('zero-order', 'linear')

#: Simulator types that execute arbitrary user code / external programs and
#: are therefore refused when a scenario is loaded with ``safe=True``.
UNSAFE_SIM_TYPES = frozenset({'external', 'script', 'pyscript', 'matlab', 'powerfactory'})

#: Recognised keys for :meth:`world.options`.
OPTION_KEYS = frozenset({'init', 'parameter_set', 'modify_signal', 'record_all'})

#: Keyword aliases for the model reference accepted by ``add_simulator``.
_MODEL_ALIAS_KEYS = (
    'sim_loc', 'net_loc', 'net', 'network', 'path', 'fmu_loc', 'fmu_path',
    'csv_location', 'csv_path', 'dss_loc', 'dss_path', 'project_name',
    'script_loc', 'script_path', 'container',
)

_CONTAINER_TYPES = ('container', 'fleet', 'vectorized', 'battery_fleet', 'building_fleet', 'pv_fleet',
                    'function_fleet')

_WORLD_KEYS = frozenset({'start_time', 'stop_time', 't_macro', 'coupling', 'extrapolation',
                         'power_unit', 'logging', 'start_datetime', 'seed', 'out_dir'})
RES_FILENAME_REMOVED_MSG = (
    "'res_filename' was removed in 3.0: simulate() returns a Results object (no HDF5 file is "
    "written). Use 'out_dir' to persist runs (Parquet/CSV + meta.json) or Results.save()/to_hdf()."
)
_REMOVED_WORLD_KEYS = frozenset({'max_iterations', 'convergence_tolerance'})

_ext_counter = itertools.count()


def _normalize_mode(mode):
    """Normalise a power-flow calculation mode to 'ac' / 'dc'."""
    if not mode:
        return mode
    m_low = str(mode).lower()
    if m_low in ('ac', 'pf', 'ldf', 'snapshot'):
        return 'ac'
    if m_low in ('dc', 'dcpf', 'lpf', 'direct'):
        return 'dc'
    raise ValueError(
        f"Unsupported calculation mode '{mode}'. "
        f"energysim only supports 'ac' and 'dc' power flow modes."
    )


def _to_fraction(x) -> Fraction:
    """Exact rational representation of a step size (handles 1e-5, numpy floats, ...)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x))
    return Fraction(str(float(x)))


def _filter_kwargs(func, candidates: Dict[str, Any]) -> Dict[str, Any]:
    """Return the subset of *candidates* accepted by *func*'s signature."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    params = sig.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(candidates)
    names = {p.name for p in params
             if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    return {k: v for k, v in candidates.items() if k in names}


def _is_adapter_like(obj) -> bool:
    return (not isinstance(obj, type)
            and all(callable(getattr(obj, m, None)) for m in ('step', 'get_value', 'set_value')))


class world():
    r"""
    Initialization
    ---------------
    ``world`` is your cosimulation canvas. It creates a ``world`` object on
    which simulators can be freely added. ``world`` accepts the
    following parameters :
        1. ``start_time`` : (0 by default).
        2. ``stop_time`` : (1000 by default). ``stop_time - start_time`` must
           be an integer multiple of the macro step, otherwise ``simulate()``
           raises ``ValueError`` (validated with a relative tolerance of 1e-9).
        3. ``logging`` : (False by default). When True, INFO messages of the
           ``'energysim'`` logger are printed to stderr.
        4. ``t_macro`` : (60 by default). Users can specify instance of information exchange.
           ``0`` means "least common multiple of all simulator step sizes".
        5. ``coupling`` : ``'jacobi'`` (default) or ``'gauss-seidel'``.
        6. ``start_datetime`` : optional wall-clock start (str / datetime); results are
           then indexed by datetime instead of seconds.
        7. ``seed`` : optional random seed, stored in the results metadata.
        8. ``out_dir`` : optional directory; every run then writes
           ``<out_dir>/<YYYYmmdd-HHMMSS>_<hash8>/`` (Parquet/CSV per simulator,
           ``meta.json``, ``scenario.yaml``).  By default nothing is written.

    >>> my_world = world(start_time=0, stop_time=1000, logging=True, t_macro=60)

    Specifying Simulators
    ---------------------
    After initializing the world cosimulation object, you can add simulators
    to the world by ``add_simulator()``. This is done by specifying
    the following :
        1. ``sim_type`` : 'fmu', 'powerflow', 'csv', 'signal', ...
        2. ``sim_name`` : It is essential that you assign the simulator a unique name.
        3. ``model`` (or legacy ``sim_loc``) : model object or location.
        4. ``outputs`` : specify the outputs that need to be recorded during simulation from the simulator.
        5. ``inputs`` : specify the inputs of the simulator.
        6. ``step_size`` : step size for simulator.

    >>> my_world.add_simulator(sim_type='fmu', sim_name='FMU1',
    >>> model=/path/to/sim, inputs=[], outputs=['var1','var2'], step_size=1)

    Connections between simulators
    -------------------------------
    >>> connections = {'sim1.output_variable1' : 'sim2.input_variable1',
    >>>    'sim3.output_variable2' : 'sim4.input_variable2',
    >>>    'sim1.output_variable3' : 'sim2.input_variable3',}
    >>> my_world.add_connections(connections)

    Initializing simulator variables
    --------------------------------
    >>> initializations = {'sim_name1' : (['sim_variables'], [values]),
    >>>                    'sim_name2' : (['sim_variables'], [values])}
    >>> my_world.options({'init' : initializations})

    Initial values are applied at the start of every run, either before
    or after ``adapter.init()`` depending on the adapter's
    ``SET_PARAMS_BEFORE_INIT`` flag.

    Simulate
    --------
    >>> my_world.simulate(pbar=True, record_all=False)

    ``simulate()`` can be called repeatedly; each call starts a fresh run
    (``configure()`` + ``init()`` on every adapter) and ``cleanup()`` is
    always called on every adapter, even on errors.

    Results
    -------
    >>> res = my_world.simulate()          # energysim.results.Results
    >>> res['sim1.var1']                   # pandas Series indexed by time
    >>> res.to_frame(); res.summary(); res.kpis(); res.save('out')

    Package Info
    ------------
    Author : Digvijay Gusain

    Email : digvijay.gusain29@gmail.com

    """

    #: rows buffered per simulator before they are converted to typed column chunks
    _results_chunk_rows = 65536

    def __init__(self, start_time=0, stop_time=1000, logging=False, t_macro=60,
                 coupling='jacobi', extrapolation='zero-order',
                 power_unit='kW', start_datetime=None, seed=None, out_dir=None):
        self.start_time = start_time
        self.stop_time = stop_time
        self.logging = logging
        if logging:
            _enable_verbose_logging()
        self.exchange = t_macro
        if power_unit:
            parse_unit(power_unit)
        self.power_unit = power_unit
        self.modify_signal = False
        self.modify_dict = {}
        self.init_dict = {}
        self._parameter_set = {}
        self.record_all = None
        self.G = None
        self.sim_dict = {}
        self.simulator_connections = {}
        self._parsed_connections = []
        self.simulator_dict = {}
        if start_datetime is not None:
            import pandas as pd
            try:
                pd.Timestamp(start_datetime)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid start_datetime {start_datetime!r}: {exc}") from None
        self.start_datetime = start_datetime
        self.seed = seed
        self.out_dir = out_dir
        self.last_results = None
        if coupling == 'iterative':
            raise ValueError(ITERATIVE_REMOVED_MSG)
        if coupling not in _COUPLINGS:
            raise ValueError(f"coupling must be 'jacobi' or 'gauss-seidel', got '{coupling}'")
        self.coupling = coupling
        if extrapolation not in _EXTRAPOLATIONS:
            raise ValueError(f"extrapolation must be 'zero-order' or 'linear', got '{extrapolation}'")
        self.extrapolation = extrapolation
        self._exchange_history = {}   # (sim_name, var_name) → deque(maxlen=2) of (time, value)
        self.macro_tstep = None
        self._total_steps = None
        # Execution order (populated by _compute_execution_order)
        self._execution_order = None
        self._dep_graph = None
        self._cycle_sccs = []  # strongly connected components with cycles

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def get_adapter(self, sim_name: str):
        """Retrieve the SimulatorAdapter instance for a registered simulator."""
        if sim_name not in self.simulator_dict:
            raise KeyError(f"Simulator '{sim_name}' not registered in world.")
        return self.simulator_dict[sim_name].adapter

    def get_model(self, sim_name: str):
        """
        Access the underlying native simulator/domain model object for 'glass-box' inspection or control.
        - Pandapower: returns `adapter.network` (the pandapower network DataFrame collection)
        - PyPSA: returns `adapter.network` (the PyPSA Network instance)
        - OpenDSS: returns `adapter._dss` (opendssdirect module handle)
        - PowerFactory: returns `adapter._project` (active PowerFactory project)
        - FMU: returns `adapter.fmu`
        - MATLAB: returns `adapter.engine`
        - Script: returns `adapter._module`
        - CSV: returns `adapter.data` (pandas DataFrame)
        - Signal: returns `adapter.signal`
        - External: returns `adapter`
        """
        adapter = self.get_adapter(sim_name)
        if hasattr(adapter, 'network'):
            return adapter.network
        if hasattr(adapter, '_dss'):
            return adapter._dss or adapter._get_dss()
        if hasattr(adapter, '_project'):
            return adapter._project
        if hasattr(adapter, 'fmu'):
            return adapter.fmu
        if hasattr(adapter, 'engine'):
            return adapter.engine
        if hasattr(adapter, '_module'):
            return adapter._module
        if hasattr(adapter, 'df'):
            return adapter.df
        if hasattr(adapter, 'data'):
            return adapter.data
        if hasattr(adapter, 'signal'):
            return adapter.signal
        return getattr(adapter, 'model', adapter)

    def _check_new_name(self, sim_name, kind='simulator'):
        if sim_name is None or (isinstance(sim_name, str) and not sim_name.strip()):
            raise ValueError(f"No name specified for {kind}. A unique sim_name must be given.")
        if not isinstance(sim_name, str):
            raise TypeError(f"Simulator name must be a string, got {type(sim_name).__name__}.")
        if sim_name in self.simulator_dict:
            raise ValueError(
                f"Simulator name '{sim_name}' already exists in the world. "
                f"Please specify a unique name."
            )

    @staticmethod
    def _check_units(units, power_unit=None):
        """Raise ValueError early for unknown units declared on a simulator."""
        if units is not None and not isinstance(units, dict):
            raise TypeError("units must be a dict mapping variable name -> unit string.")
        for var, u in (units or {}).items():
            try:
                parse_unit(u)
            except ValueError as exc:
                raise ValueError(f"Variable '{var}': {exc}") from None
        if power_unit:
            parse_unit(power_unit)

    @staticmethod
    def _pick_model(sim_name, model=None, **aliases):
        """Return the single model reference given via ``model`` or one of its aliases."""
        found = []
        for key, val in [('model', model)] + list(aliases.items()):
            if val is None or (isinstance(val, str) and val == ''):
                continue
            found.append((key, val))
        if not found:
            return None
        first_key, first = found[0]
        for key, val in found[1:]:
            same = val is first or (isinstance(val, str) and isinstance(first, str) and val == first)
            if not same:
                raise ValueError(
                    f"Conflicting model references for '{sim_name}': '{first_key}' and '{key}' "
                    f"were both given with different values."
                )
        return first

    def _register(self, sim_name, entry, label):
        self.simulator_dict[sim_name] = entry
        logger.info("Added %s simulator '%s' to the world", label, sim_name)

    # ------------------------------------------------------------------
    # add_* API
    # ------------------------------------------------------------------

    def add_signal(self, sim_name, signal=None, step_size=1, power_unit=None, units=None):
        """Add a signal source with a single output ``y``.

        ``signal`` may be a callable ``f(t)``, a number (constant), a safe
        expression string (``"5*sin(t/60)"``, optionally prefixed with
        ``lambda t:``) or a structured spec dict (``{type: sine, ...}``,
        see :mod:`energysim._safeexpr`).  Strings are never ``eval``-ed.
        """
        self._check_new_name(sim_name, 'signal')
        if signal is None:
            raise ValueError(
                f"Signal simulator '{sim_name}' needs a signal: a callable, a number, a safe "
                f"expression string or a structured spec dict."
            )
        self._check_units(units, power_unit)
        func = build_signal(signal)
        signal_obj = signal_adapter(signal_name=sim_name, signal=func)
        self._register(sim_name, SimEntry(
            sim_type='signal', adapter=signal_obj, step_size=step_size,
            outputs=['y'], inputs=[],
            power_unit=power_unit or self.power_unit,
            units=units or {},
            model=signal,
        ), 'signal')

    def add_simulator(self, sim_type='', sim_name='', sim_loc=None, inputs=None, outputs=None, step_size=1,
                      mode='ac', power_unit=None, units=None, model=None, allow_import=False, **kwargs):
        """
        Method to add simulator to ``world()`` object. ``add_simulator()`` takes the following arguments as inputs :

        1. ``sim_type`` : 'fmu', 'powerflow', 'pypsa', 'opendss', 'csv', 'signal', 'external', 'powerfactory',
           'matlab', 'script', 'model', 'container' / 'fleet' / 'battery_fleet' / 'building_fleet' / 'pv_fleet'
        2. ``sim_name`` : Unique name identifier for the simulator.
        3. ``model`` : In-memory Python object, file path, project name, or callable factory.
           Aliases (exactly one may be given): ``sim_loc``, ``net_loc``, ``net``, ``network``, ``path``,
           ``fmu_path``, ``csv_path``, ``dss_path``, ``script_path``, ``project_name``, ``container`` ...
        4. ``outputs`` : Specify the outputs to record during simulation.
        5. ``inputs`` : Specify the inputs for the simulator.
        6. ``step_size`` : Micro step size for simulator (set to 1 by default).
        7. ``mode`` : Powerflow calculation mode: 'ac' (default) or 'dc'.
        8. ``power_unit`` : Physical unit for power (default: 'kW', or native simulator unit).
        9. ``units`` : Optional dictionary mapping variable name -> unit string.
        10. ``allow_import`` : allow dotted-import model references such as
            ``'pandapower.networks.case14'`` to be imported and called (default False).
        """
        inputs = list(inputs or [])
        outputs = list(outputs or [])

        # Backward compatibility for legacy 'pf' parameter
        if 'pf' in kwargs:
            mode = kwargs.pop('pf')
        aliases = {k: kwargs.pop(k) for k in _MODEL_ALIAS_KEYS if k in kwargs}
        target = self._pick_model(sim_name, model=model, sim_loc=sim_loc, **aliases)
        mode = _normalize_mode(mode)

        st = (sim_type or '').lower()
        common = dict(inputs=inputs, outputs=outputs, step_size=step_size,
                      power_unit=power_unit, units=units)

        if st == 'fmu':
            self.add_fmu(sim_name, model=target,
                         solver_name=kwargs.pop('solver_name', 'Cvode'),
                         variable=kwargs.pop('variable', False),
                         validate=kwargs.pop('validate', True),
                         **common, **kwargs)

        elif st in ('powerflow', 'pandapower'):
            self.add_powerflow(sim_name, model=target, mode=mode, allow_import=allow_import,
                               **common, **kwargs)

        elif st == 'pypsa':
            self.add_pypsa(sim_name, model=target, mode=mode, allow_import=allow_import,
                           **common, **kwargs)

        elif st == 'opendss':
            self.add_opendss(sim_name, model=target, mode=mode, **common, **kwargs)

        elif st in ('script', 'pyscript'):
            self.add_script(script_name=sim_name, sim_name=sim_name, model=target,
                            function=kwargs.pop('function', None), **common, **kwargs)

        elif st == 'signal':
            signal_spec = target if target is not None else kwargs.pop('signal', None)
            self.add_signal(sim_name, signal_spec, step_size=step_size,
                            power_unit=power_unit, units=units)

        elif st == 'csv':
            self.add_csv(
                csv_name=sim_name, model=target, step_size=step_size, outputs=outputs,
                delimiter=kwargs.pop('delimiter', ','), power_unit=power_unit, units=units)

        elif st == 'external':
            self.add_external_simulator(sim_name=sim_name, model=target, **common, **kwargs)

        elif st == 'powerfactory':
            self.add_powerfactory(sim_name=sim_name, model=target, mode=mode,
                                  pf_path=kwargs.pop('pf_path', None), **common)

        elif st in _CONTAINER_TYPES:
            kind = kwargs.pop('kind', None) or kwargs.pop('fleet_type', None)
            if kind is None and st not in ('container', 'fleet', 'vectorized'):
                kind = st
            size = kwargs.pop('size', 100)
            self.add_container(sim_name=sim_name, container=target, kind=kind, size=size,
                               **common, **kwargs)

        elif st in ('model', 'component') or isinstance(target, SimulatorAdapter):
            self.add_model(sim_name=sim_name, model=target, **common, **kwargs)

        elif st == 'matlab':
            self.add_matlab(sim_name=sim_name, model=target, engine=kwargs.pop('engine', 'auto'), **common)

        else:
            raise ValueError(
                f"Simulator type '{sim_type}' not recognized for '{sim_name}'."
                f" Possible sim_type are 'container', 'fleet', 'battery_fleet', 'building_fleet', 'pv_fleet',"
                f" 'model', 'external', 'fmu', 'powerflow', 'pypsa', 'opendss', 'script', 'signal', 'csv',"
                f" 'powerfactory', 'matlab'."
            )

    def add_model(self, sim_name, model=None, inputs=None, outputs=None, step_size=1,
                  power_unit=None, units=None, **kwargs):
        """
        Add an inbuilt physical model component (or any custom SimulatorAdapter instance) to the world.

        ``model`` may be an adapter instance, or a class / factory called with ``**kwargs``.
        """
        target = self._pick_model(sim_name, model=model, sim_loc=kwargs.pop('sim_loc', None))
        if target is None:
            raise ValueError(f"No model instance or factory provided for '{sim_name}'.")
        self._check_new_name(sim_name, 'model')
        self._check_units(units, power_unit)

        if callable(target) and not _is_adapter_like(target):
            target = target(**kwargs)
        if not _is_adapter_like(target):
            raise TypeError(
                f"Model for '{sim_name}' must be a SimulatorAdapter (or provide step/get_value/set_value), "
                f"got {type(target).__name__}."
            )

        adapter = target
        adapter.step_size = step_size

        eff_inputs = list(inputs) if inputs else list(getattr(adapter, 'inputs', []) or [])
        eff_outputs = list(outputs) if outputs else list(getattr(adapter, 'outputs', []) or [])
        adapter.inputs = eff_inputs
        adapter.outputs = eff_outputs

        self._register(sim_name, SimEntry(
            sim_type='model', adapter=adapter, step_size=step_size,
            outputs=eff_outputs, inputs=eff_inputs,
            power_unit=power_unit or getattr(adapter, 'power_unit', None) or self.power_unit,
            units=units or dict(getattr(adapter, 'units', {}) or {}),
            model=target,
        ), 'model')

    def add_container(
        self,
        sim_name: Optional[str] = None,
        container: Any = None,
        kind: Optional[str] = None,
        size: int = 100,
        step_size: float = 900.0,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        power_unit: Optional[str] = None,
        units: Optional[Dict[str, str]] = None,
        model: Any = None,
        **kwargs
    ):
        """
        Add a Vectorized Container Simulator to the world.

        Parameters
        ----------
        sim_name : str, optional
            Unique name identifier for the container simulator (or pass via name=...).
        container : VectorizedContainerAdapter or None
            An existing container adapter instance, or class.
        kind : str, optional
            Fleet type: 'battery', 'building', 'pv', or 'function' (or pass via fleet_type=...).
        size : int
            Number of entities in the container (default: 100).
        step_size : float
            Micro-step size in seconds.
        """
        sim_name = sim_name or kwargs.pop('name', None) or kwargs.pop('sim_name', None)
        if not sim_name:
            raise ValueError("sim_name (or name) must be specified for container/fleet simulator.")
        kind = kind or kwargs.pop('fleet_type', None) or kwargs.pop('kind', None)
        kwargs.pop('fleet_type', None)
        kwargs.pop('kind', None)
        self._check_new_name(sim_name, 'container')
        self._check_units(units, power_unit)

        from .container import (
            VectorizedContainerAdapter,
            BatteryFleetAdapter,
            BuildingFleetAdapter,
            PVFleetAdapter,
            VectorizedFunctionAdapter,
        )

        target = self._pick_model(sim_name, model=container, container_model=model)
        k_low = (kind or '').lower()

        if isinstance(target, VectorizedContainerAdapter):
            adapter = target
            adapter.name = sim_name
            if step_size:
                adapter.step_size = step_size
        elif isinstance(target, type) and issubclass(target, VectorizedContainerAdapter):
            adapter = target(name=sim_name, size=size, step_size=step_size, **kwargs)
        elif target is not None:
            raise TypeError(
                f"Container for '{sim_name}' must be a VectorizedContainerAdapter instance or subclass, "
                f"got {type(target).__name__}."
            )
        elif 'battery' in k_low:
            adapter = BatteryFleetAdapter(name=sim_name, size=size, step_size=step_size, **kwargs)
        elif 'building' in k_low or 'thermal' in k_low:
            adapter = BuildingFleetAdapter(name=sim_name, size=size, step_size=step_size, **kwargs)
        elif 'pv' in k_low or 'solar' in k_low:
            adapter = PVFleetAdapter(name=sim_name, size=size, step_size=step_size, **kwargs)
        elif ('function' in k_low or callable(kwargs.get('update_func'))
              or callable(kwargs.get('update_fn')) or callable(kwargs.get('func'))):
            adapter = VectorizedFunctionAdapter(
                name=sim_name, size=size, inputs=inputs or None, step_size=step_size, **kwargs,
            )
        else:
            raise ValueError(
                f"Cannot build container '{sim_name}': specify kind='battery' | 'building' | 'pv' | "
                f"'function' (or a fleet sim_type), or pass a container adapter via model=/container=."
            )

        if outputs:
            adapter.outputs = list(outputs)
        if inputs:
            adapter.inputs = list(inputs)

        # Default outputs: use all available scalar aggregations if none provided
        if not adapter.outputs:
            avail = adapter.get_available_variables()
            adapter.outputs = [v for v in avail['outputs'] if v.startswith(('total_', 'mean_'))]

        self._register(sim_name, SimEntry(
            sim_type='container',
            adapter=adapter,
            step_size=adapter.step_size,
            outputs=adapter.outputs,
            inputs=adapter.inputs,
            power_unit=power_unit or self.power_unit,
            units=units or {},
            model=adapter,
        ), f'vectorized container (size={adapter.size})')

    # Convenience alias for add_container
    add_fleet = add_container

    # ------------------------------------------------------------------
    # External simulators
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file_module(path):
        """Load a .py file as a fresh module with a unique name (sys.path untouched)."""
        stem = os.path.splitext(os.path.basename(path))[0]
        mod_name = f"_energysim_external_{next(_ext_counter)}_{re.sub(r'[^0-9A-Za-z_]', '_', stem)}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load external simulator file {path!r}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(mod_name, None)
            raise
        return module

    @staticmethod
    def _import_legacy_module(sim_name, directory):
        """Legacy: import module ``sim_name`` from *directory* (sys.path restored afterwards)."""
        inserted = False
        if directory:
            directory = os.fspath(directory)
            if directory not in sys.path:
                sys.path.insert(0, directory)
                inserted = True
        try:
            return importlib.import_module(sim_name)
        finally:
            if inserted:
                try:
                    sys.path.remove(directory)
                except ValueError:
                    pass

    def _build_external(self, sim_name, target, inputs, outputs, step_size, kwargs):
        factory = None
        if target is not None and _is_adapter_like(target):
            return target
        if isinstance(target, type) or (callable(target) and not isinstance(target, str)):
            factory = target
        elif isinstance(target, (str, os.PathLike)):
            ref = os.fspath(target)
            path_part, _, attr = ref.rpartition(':')
            if path_part and path_part.lower().endswith('.py') and attr.isidentifier():
                if not os.path.isfile(path_part):
                    raise FileNotFoundError(f"External simulator file not found: {path_part!r}")
                module = self._load_file_module(path_part)
                try:
                    factory = getattr(module, attr)
                except AttributeError:
                    raise AttributeError(f"'{attr}' not found in {path_part!r}.") from None
            elif ref.lower().endswith('.py'):
                if not os.path.isfile(ref):
                    raise FileNotFoundError(f"External simulator file not found: {ref!r}")
                module = self._load_file_module(ref)
                factory = getattr(module, 'external_simulator', None)
                if factory is None:
                    raise AttributeError(
                        f"{ref!r} defines no 'external_simulator' class; use 'path/to/file.py:ClassName'."
                    )
            else:
                module = self._import_legacy_module(sim_name, ref)
                factory = module.external_simulator
        elif target is None:
            module = self._import_legacy_module(sim_name, None)
            factory = module.external_simulator
        else:
            raise TypeError(
                f"Unsupported external simulator model for '{sim_name}': {type(target).__name__}."
            )

        candidates = dict(kwargs)
        candidates.update(inputs=inputs, outputs=outputs, step_size=step_size, sim_name=sim_name)
        call_kwargs = _filter_kwargs(factory, candidates)
        return factory(**call_kwargs)

    def add_external_simulator(self, sim_name, sim_loc=None, inputs=None, outputs=None, step_size=900,
                               power_unit=None, units=None, model=None, **kwargs):
        """Add a user-defined simulator.

        ``model`` (or legacy ``sim_loc``) may be:

        a. a ``SimulatorAdapter`` instance;
        b. a class or factory callable (``inputs``, ``outputs``, ``step_size``,
           ``sim_name`` and extra kwargs are passed only if its signature accepts them);
        c. ``'path/to/file.py:ClassName'`` (loaded from file under a unique module name,
           ``sys.path`` untouched), or ``'path/to/file.py'`` defining ``external_simulator``;
        d. legacy: a directory containing module ``<sim_name>.py`` with class
           ``external_simulator`` (directory is put on ``sys.path`` only during import).
        """
        target = self._pick_model(sim_name, model=model, sim_loc=sim_loc)
        self._check_new_name(sim_name, 'external simulator')
        self._check_units(units, power_unit)
        inputs = list(inputs or [])
        outputs = list(outputs or [])
        simulator = self._build_external(sim_name, target, inputs, outputs, step_size, kwargs)
        if not _is_adapter_like(simulator):
            raise TypeError(
                f"External simulator '{sim_name}' must provide step(), get_value() and set_value()."
            )
        simulator.step_size = step_size
        eff_inputs = inputs or list(getattr(simulator, 'inputs', []) or [])
        eff_outputs = outputs or list(getattr(simulator, 'outputs', []) or [])
        self._register(sim_name, SimEntry(
            sim_type='external', adapter=simulator, step_size=step_size,
            outputs=eff_outputs, inputs=eff_inputs,
            power_unit=power_unit or self.power_unit,
            units=units or {},
            model=target,
        ), 'external')

    def add_powerfactory(self, sim_name, sim_loc=None, inputs=None,
                         outputs=None, step_size=900, mode='ac', pf=None,
                         pf_path=None, power_unit=None, units=None,
                         project_name=None, model=None, **kwargs):
        """Add a DIgSILENT PowerFactory model to the co-simulation."""
        self._check_new_name(sim_name, 'PowerFactory model')
        self._check_units(units, power_unit)
        p_name = self._pick_model(sim_name, model=model, project_name=project_name, sim_loc=sim_loc) or sim_name
        from .pfAdapter import pf_adapter  # lazy import
        adapter = pf_adapter(
            project_name=p_name,
            pf_loc=p_name,
            inputs=inputs or [],
            outputs=outputs or [],
            mode=mode,
            pf=pf,
            pf_path=pf_path,
        )
        adapter.step_size = step_size
        self._register(sim_name, SimEntry(
            sim_type='powerfactory', adapter=adapter,
            step_size=step_size, outputs=list(outputs or []),
            inputs=list(inputs or []), mode=adapter.mode,
            power_unit=power_unit or 'MW',
            units=units or {},
            model=p_name,
        ), 'PowerFactory')

    def add_matlab(self, sim_name, sim_loc=None, inputs=None, outputs=None,
                   step_size=60, engine='auto', power_unit=None, units=None,
                   script_path=None, model=None, **kwargs):
        """Add a MATLAB / Octave .m function to the co-simulation."""
        self._check_new_name(sim_name, 'MATLAB model')
        self._check_units(units, power_unit)
        target = self._pick_model(sim_name, model=model, script_path=script_path, sim_loc=sim_loc)
        from .matlabAdapter import matlab_adapter  # lazy import
        adapter = matlab_adapter(
            func_name=sim_name,
            script_loc=target,
            inputs=inputs or [],
            outputs=outputs or [],
            engine=engine,
        )
        adapter.step_size = step_size
        self._register(sim_name, SimEntry(
            sim_type='matlab', adapter=adapter,
            step_size=step_size, outputs=list(outputs or []),
            inputs=list(inputs or []),
            power_unit=power_unit or self.power_unit,
            units=units or {},
            model=target,
        ), 'MATLAB/Octave')

    def add_powerflow(self, network_name=None, net_loc=None, inputs=None, outputs=None, mode='ac', pf=None,
                      step_size=900, power_unit=None, units=None, net=None, path=None, model=None,
                      allow_import=False, **kwargs):
        """Add a pandapower network. Dotted import references require ``allow_import=True``."""
        network_name = network_name or kwargs.pop('sim_name', None)
        target = self._pick_model(network_name, model=model, net=net, path=path, net_loc=net_loc,
                                  sim_loc=kwargs.pop('sim_loc', None), network=kwargs.pop('network', None))
        self._check_new_name(network_name, 'power flow model')
        if target is None:
            raise ValueError(f"No network model or location specified for power flow model '{network_name}'.")
        self._check_units(units, power_unit)
        inputs = list(inputs or [])
        outputs = list(outputs or [])
        resolved = resolve_model_reference(target, allow_import=allow_import)
        from .ppAdapter import pp_adapter         # lazy: pandapower has heavy optional deps
        network = pp_adapter(network_name, net_loc=resolved, inputs=inputs, outputs=outputs, mode=mode, pf=pf,
                             model=resolved)
        network.step_size = step_size
        self._register(network_name, SimEntry(
            sim_type='powerflow', adapter=network, step_size=step_size,
            outputs=outputs, inputs=inputs, mode=network.mode,
            power_unit=power_unit or 'MW',
            units=units or {},
            model=target,
        ), 'powerflow')

    def add_pypsa(self, network_name=None, net_loc=None, inputs=None, outputs=None, mode='ac', pf=None,
                  step_size=900, power_unit=None, units=None, network=None, path=None, model=None,
                  allow_import=False, **kwargs):
        """Add a PyPSA network simulator to the world (Pattern B single-snapshot mode)."""
        network_name = network_name or kwargs.pop('sim_name', None)
        target = self._pick_model(network_name, model=model, network=network, path=path, net_loc=net_loc,
                                  sim_loc=kwargs.pop('sim_loc', None), net=kwargs.pop('net', None))
        self._check_new_name(network_name, 'PyPSA model')
        if target is None:
            raise ValueError(f"No network model or location specified for PyPSA model '{network_name}'.")
        self._check_units(units, power_unit)
        inputs = list(inputs or [])
        outputs = list(outputs or [])
        resolved = resolve_model_reference(target, allow_import=allow_import)
        from .pypsaAdapter import pypsa_adapter
        network_adapter = pypsa_adapter(
            network_name, net_loc=resolved, inputs=inputs, outputs=outputs, mode=mode, pf=pf,
            logger_level=kwargs.get('logger', 'CRITICAL'), model=resolved
        )
        network_adapter.step_size = step_size
        self._register(network_name, SimEntry(
            sim_type='pypsa', adapter=network_adapter, step_size=step_size,
            outputs=outputs, inputs=inputs, mode=network_adapter.mode,
            power_unit=power_unit or 'MW',
            units=units or {},
            model=target,
        ), 'PyPSA (Pattern B single-snapshot mode)')

    def add_opendss(self, dss_name=None, dss_loc=None, inputs=None, outputs=None, mode='ac', pf=None,
                    step_size=900, power_unit=None, units=None, dss_path=None, model=None, **kwargs):
        """Add an OpenDSS distribution network simulator to the world via OpenDSSDirect.py."""
        dss_name = dss_name or kwargs.pop('sim_name', None)
        target = self._pick_model(dss_name, model=model, dss_path=dss_path, dss_loc=dss_loc,
                                  sim_loc=kwargs.pop('sim_loc', None))
        self._check_new_name(dss_name, 'OpenDSS model')
        if target is None:
            raise ValueError(f"No DSS path or model specified for OpenDSS simulator '{dss_name}'.")
        self._check_units(units, power_unit)
        inputs = list(inputs or [])
        outputs = list(outputs or [])
        from .opendssAdapter import opendss_adapter
        dss_obj = opendss_adapter(dss_name, dss_location=target, inputs=inputs, outputs=outputs,
                                  step_size=step_size, mode=mode, pf=pf, model=target)
        self._register(dss_name, SimEntry(
            sim_type='opendss', adapter=dss_obj, step_size=step_size,
            outputs=outputs, inputs=inputs, mode=dss_obj.mode,
            power_unit=power_unit or 'kW',
            units=units or {},
            model=target,
        ), 'OpenDSS')

    def add_script(self, script_name=None, script_loc=None, inputs=None, outputs=None, step_size=900,
                   power_unit=None, units=None, script_path=None, model=None, sim_name=None,
                   function=None, **kwargs):
        """Add a Python script/module function simulator to the world.

        ``sim_name`` defaults to ``script_name``.  The function called each
        step is ``function`` if given, otherwise ``script_name`` (format
        ``'module.function'`` when the location is a directory, or just
        ``'function'`` when it is a ``.py`` file).
        """
        sim_id = sim_name or script_name
        target = self._pick_model(sim_id, model=model, script_path=script_path, script_loc=script_loc,
                                  sim_loc=kwargs.pop('sim_loc', None))
        self._check_new_name(sim_id, 'Python script simulator')
        if target is None:
            raise ValueError(f"No script path or model specified for Python script simulator '{sim_id}'.")
        self._check_units(units, power_unit)
        inputs = list(inputs or [])
        outputs = list(outputs or [])
        func_spec = function or kwargs.pop('func', None) or script_name
        from .pyScriptAdapter import py_adapter
        adapter = py_adapter(script_name=func_spec, script_loc=target, inputs=inputs, outputs=outputs)
        adapter.step_size = step_size
        self._register(sim_id, SimEntry(
            sim_type='script', adapter=adapter, step_size=step_size,
            outputs=outputs, inputs=inputs,
            power_unit=power_unit or self.power_unit,
            units=units or {},
            model=target,
        ), 'script')

    def add_fmu(
            self,
            fmu_name=None,
            fmu_loc=None,
            step_size=1,
            inputs=None,
            outputs=None,
            exist=False,
            solver_name='Cvode',
            variable=False,
            power_unit=None,
            units=None,
            fmu_path=None,
            model=None,
            validate=True,
            **kwargs):
        fmu_name = fmu_name or kwargs.pop('sim_name', None)
        target = self._pick_model(fmu_name, model=model, fmu_path=fmu_path, fmu_loc=fmu_loc,
                                  sim_loc=kwargs.pop('sim_loc', None))
        self._check_new_name(fmu_name, 'FMU simulator')
        if target is None:
            raise ValueError(f"No FMU path or model specified for FMU simulator '{fmu_name}'.")
        self._check_units(units, power_unit)
        inputs = list(inputs or [])
        outputs = list(outputs or [])
        # Lazy imports — fmpy is only pulled in when an FMU is actually added
        from fmpy.model_description import read_model_description
        from .csAdapter import FmuCsAdapter
        from .meAdapter import FmuMeAdapter

        m_desc = read_model_description(target, validate=validate)
        if m_desc.coSimulation is not None:
            fmu_temp = FmuCsAdapter(target,
                                    instanceName=fmu_name,
                                    step_size=step_size,
                                    inputs=inputs,
                                    outputs=outputs,
                                    exist=exist,
                                    validate=validate)
            fmu_temp.variable_step = variable
            fmu_temp.step_size = step_size
        else:
            fmu_temp = FmuMeAdapter(target,
                                    instanceName=fmu_name,
                                    step_size=step_size,
                                    inputs=inputs,
                                    outputs=outputs,
                                    solver_name=solver_name,
                                    validate=validate)

        self._register(fmu_name, SimEntry(
            sim_type='fmu', adapter=fmu_temp, step_size=step_size,
            outputs=outputs, inputs=inputs, variable_step=variable,
            power_unit=power_unit or 'W',
            units=units or {},
            model=target,
        ), 'FMU')

    def add_csv(self, csv_name=None, csv_location=None, step_size=1, outputs=None, delimiter=',',
                power_unit=None, units=None, csv_path=None, model=None, **kwargs):
        csv_name = csv_name or kwargs.pop('sim_name', None)
        target = self._pick_model(csv_name, model=model, csv_path=csv_path, csv_location=csv_location,
                                  sim_loc=kwargs.pop('sim_loc', None))
        self._check_new_name(csv_name, 'CSV simulator')
        if target is None:
            raise ValueError(f"No CSV path or model specified for CSV simulator '{csv_name}'.")
        self._check_units(units, power_unit)
        outputs = list(outputs or [])
        csv_obj = csv_simulator(csv_name, target, step_size, outputs, delimiter)
        csv_obj.step_size = step_size
        self._register(csv_name, SimEntry(
            sim_type='csv', adapter=csv_obj, step_size=step_size,
            outputs=outputs, inputs=[],
            power_unit=power_unit or self.power_unit,
            units=units or {},
            model=target,
        ), 'CSV')

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def _resolve_unit(self, entry, var):
        """Unit of *var* on *entry*: explicit ``units`` mapping, name suffix, or (power names only) power_unit."""
        if entry is None:
            return None
        if var in entry.units:
            return entry.units[var]
        return infer_variable_unit(var, default_unit=entry.power_unit or self.power_unit)

    def add_connections(self, connections=None):
        '''
        Adds connections between simulators. Connections are
        specified as dictionary with keys as sending variables,
        and values as receiving variables (or a list of dicts with
        ``from``/``to``/``scale``/``offset``/``from_unit``/``to_unit``).

        Unit conversion is applied automatically only when the units of
        *both* ends are known explicitly: via ``from_unit``/``to_unit`` on the
        connection, the simulator's ``units`` mapping, or a conventional
        variable-name suffix (``_mw``, ``_kw``, ``_kv``, ``_pu``, ``_kwh``, ...).
        A simulator's ``power_unit`` is only used for variables whose name
        looks like active power (``p``, ``p_cmd``, ``P_set``, ``power``...).
        Unknown units and incompatible dimensions raise ``ValueError``.
        A ``scale`` given without any unit key disables automatic conversion.
        '''
        if connections is None:
            connections = {}

        if not self.simulator_dict:
            raise ValueError("Cannot make connections with no simulators defined.")
        if not isinstance(connections, (dict, list)):
            raise TypeError(f"connections must be a dict or a list of dicts, got {type(connections).__name__}.")
        self.simulator_connections = connections

        if isinstance(connections, list):
            raw_connections = []
            for item in connections:
                if not isinstance(item, dict):
                    raise ConnectionSpecError(f"Connection list items must be dicts, got {item!r}.")
                raw_connections.append(item)
        else:
            raw_connections = []
            for k, v in connections.items():
                if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
                    item = {'from': k, 'to': v[0]}
                    item.update({kk: vv for kk, vv in v[1].items() if kk not in ('to', 'target')})
                    raw_connections.append(item)
                elif isinstance(v, dict):
                    item = {'from': k, 'to': v.get('to') or v.get('target')}
                    item.update({kk: vv for kk, vv in v.items() if kk not in ('to', 'target')})
                    raw_connections.append(item)
                else:
                    raw_connections.append({'from': k, 'to': v})

        # Pre-parse connections once so exchange_variable_values() doesn't
        # re-split strings on every time step.
        parsed_connections = []
        for item in raw_connections:
            output_key = item.get('from') or item.get('source')
            input_key = item.get('to') or item.get('target')
            user_scale = float(item.get('scale', 1.0))
            user_offset = float(item.get('offset', 0.0))
            from_unit = item.get('from_unit') or item.get('unit')
            to_unit = item.get('to_unit') or item.get('unit')

            # --- parse output side ---
            if isinstance(output_key, str):
                ou_sim, ou_var = self._split_var(output_key)
                parsed_output = (ou_sim, ou_var, output_key)
                first_ou_sim, first_ou_var = ou_sim, ou_var
            elif isinstance(output_key, (tuple, list)) and output_key:
                parsed_output = tuple(
                    (self._split_var(k)[0], self._split_var(k)[1], k)
                    for k in output_key
                )
                first_ou_sim, first_ou_var = parsed_output[0][0], parsed_output[0][1]
            else:
                raise ConnectionSpecError(f"Invalid connection source {output_key!r}.")
            # --- parse input side ---
            if isinstance(input_key, str):
                in_sim, in_var = self._split_var(input_key)
                parsed_input = (in_sim, in_var)
                first_in_sim, first_in_var = in_sim, in_var
            elif isinstance(input_key, (list, tuple)) and input_key:
                parsed_input = [
                    (self._split_var(k)[0], self._split_var(k)[1])
                    for k in input_key
                ]
                first_in_sim, first_in_var = parsed_input[0][0], parsed_input[0][1]
            else:
                raise ConnectionSpecError(f"Invalid connection target {input_key!r} (from {output_key!r}).")

            explicit_units = any(k in item for k in ('from_unit', 'to_unit', 'unit'))
            if 'scale' in item and not explicit_units:
                # A manual scale without units overrides automatic conversion.
                from_unit = to_unit = None
                unit_scale, unit_offset = 1.0, 0.0
            else:
                if not from_unit:
                    from_unit = self._resolve_unit(self.simulator_dict.get(first_ou_sim), first_ou_var)
                if not to_unit:
                    to_unit = self._resolve_unit(self.simulator_dict.get(first_in_sim), first_in_var)
                try:
                    unit_scale, unit_offset = get_unit_conversion(from_unit, to_unit)
                except ValueError as exc:
                    raise ValueError(f"Connection {output_key!r} -> {input_key!r}: {exc}") from None

            effective_scale = unit_scale * user_scale
            effective_offset = unit_offset * user_scale + user_offset

            parsed_connections.append(
                ParsedConnection(parsed_output, parsed_input, effective_scale, effective_offset,
                                 from_unit, to_unit, user_scale=user_scale, user_offset=user_offset)
            )
        self._parsed_connections = parsed_connections

        # Run validation checks
        self._validate_connections()

        # Build dependency graph and compute execution order
        self._build_dependency_graph()
        self._compute_execution_order()

        logger.info("Added %i connections between simulators to the World", len(self._parsed_connections))

    # ------------------------------------------------------------------
    # Connection validation
    # ------------------------------------------------------------------

    @staticmethod
    def _split_var(dotted_name):
        """Split 'sim_name.var.sub' → ('sim_name', 'var.sub')."""
        if not isinstance(dotted_name, str):
            raise ConnectionSpecError(f"Connection endpoint must be a string, got {dotted_name!r}.")
        parts = dotted_name.split('.', 1)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ConnectionSpecError(
                f"Connection string '{dotted_name}' is malformed — expected "
                f"format 'simulator_name.variable_name'."
            )
        return parts[0], parts[1]

    def _validate_connections(self):
        """Validate all parsed connections for correctness."""
        known_sims = set(self.simulator_dict.keys())
        read_only_types = {'csv', 'signal'}

        for conn in self._parsed_connections:
            output_, input_ = conn[0], conn[1]
            # --- validate output side ---
            if isinstance(output_, tuple) and isinstance(output_[0], tuple):
                # fan-in
                for ou_sim_name, ou_sim_var, ou_key in output_:
                    self._validate_sim_exists(ou_sim_name, known_sims, ou_key)
                    self._validate_variable(ou_sim_name, ou_sim_var, direction='output')
            else:
                ou_sim_name, ou_sim_var, ou_key = output_
                self._validate_sim_exists(ou_sim_name, known_sims, ou_key)
                self._validate_variable(ou_sim_name, ou_sim_var, direction='output')

            # --- validate input side ---
            if isinstance(input_, list):
                for in_sim_name, in_sim_var in input_:
                    self._validate_sim_exists(in_sim_name, known_sims,
                                              f"{in_sim_name}.{in_sim_var}")
                    self._validate_variable(in_sim_name, in_sim_var, direction='input')
                    self._warn_if_read_only(in_sim_name, in_sim_var, read_only_types)
            else:
                in_sim_name, in_sim_var = input_
                self._validate_sim_exists(in_sim_name, known_sims,
                                          f"{in_sim_name}.{in_sim_var}")
                self._validate_variable(in_sim_name, in_sim_var, direction='input')
                self._warn_if_read_only(in_sim_name, in_sim_var, read_only_types)

    def _validate_sim_exists(self, sim_name, known_sims, connection_str):
        """Raise ConnectionSpecError if simulator doesn't exist."""
        if sim_name not in known_sims:
            raise ConnectionSpecError(
                f"Simulator '{sim_name}' referenced in connection "
                f"'{connection_str}' does not exist. "
                f"Available simulators: {sorted(known_sims)}"
            )

    def _validate_variable(self, sim_name, var_name, direction='output'):
        """Emit a warning if a variable name doesn't match known variables."""
        entry = self.simulator_dict[sim_name]
        adapter = entry.adapter
        get_vars = getattr(adapter, 'get_available_variables', None)
        avail = get_vars() if callable(get_vars) else {}
        all_vars = set(avail.get('inputs', [])) | set(avail.get('outputs', []))
        if not all_vars:
            # Adapter doesn't expose variable metadata — skip validation
            return
        if var_name not in all_vars:
            # Strip index if present, e.g. 'soc[0]' -> 'soc'
            base_var = re.sub(r'\[\d+\]$', '', var_name)
            if base_var in all_vars:
                return
            # Strip aggregation prefix if present, e.g. 'total_soc' -> 'soc'
            agg_base = re.sub(r'^(total_|mean_|sum_|min_|max_)', '', var_name)
            if agg_base in all_vars or re.sub(r'\[\d+\]$', '', agg_base) in all_vars:
                return

            warnings.warn(
                f"Variable '{var_name}' not found in simulator '{sim_name}' "
                f"declared variables. Connection may fail at runtime. "
                f"Known variables: {sorted(all_vars)[:20]}{'...' if len(all_vars) > 20 else ''}",
                UserWarning, stacklevel=4
            )

    def _warn_if_read_only(self, sim_name, var_name, read_only_types):
        """Warn if connecting to a read-only simulator."""
        entry = self.simulator_dict[sim_name]
        if entry.sim_type in read_only_types:
            warnings.warn(
                f"Simulator '{sim_name}' is read-only (type='{entry.sim_type}'). "
                f"Connection to '{var_name}' will be silently ignored.",
                UserWarning, stacklevel=4
            )

    # ------------------------------------------------------------------
    # Dependency graph & topological ordering
    # ------------------------------------------------------------------

    def _build_dependency_graph(self):
        """Build a directed graph of simulator dependencies from connections."""
        self._dep_graph = nx.DiGraph()
        # Add all simulators as nodes (including unconnected ones)
        for sim_name in self.simulator_dict:
            self._dep_graph.add_node(sim_name)

        for conn in self._parsed_connections:
            output_, input_ = conn[0], conn[1]
            if isinstance(output_, tuple) and isinstance(output_[0], tuple):
                ou_sims = [p[0] for p in output_]
            else:
                ou_sims = [output_[0]]

            if isinstance(input_, list):
                in_sims = [p[0] for p in input_]
            else:
                in_sims = [input_[0]]

            for o_sim in ou_sims:
                for i_sim in in_sims:
                    self._dep_graph.add_edge(o_sim, i_sim)

    def _compute_execution_order(self):
        """Compute execution order via topological sort.

        For DAGs, produces a dependency-respecting order.
        For cyclic graphs, uses condensation (SCCs collapsed to super-nodes)
        to get a partial topological order, with simulators inside each SCC
        kept in registration order. Emits a warning for each detected cycle.
        """
        G = self._dep_graph
        if nx.is_directed_acyclic_graph(G):
            self._execution_order = list(nx.topological_sort(G))
            self._cycle_sccs = []
        else:
            # Find all strongly connected components (cycles)
            sccs = list(nx.strongly_connected_components(G))
            self._cycle_sccs = [scc for scc in sccs if len(scc) > 1]
            for scc in self._cycle_sccs:
                warnings.warn(
                    f"Algebraic loop detected involving simulators: {sorted(scc)}. "
                    f"Using Jacobi coupling (one-step delay) for these cyclic dependencies. "
                    f"Reduce t_macro or merge the coupled components to limit the delay error.",
                    UserWarning, stacklevel=3
                )
            condensation = nx.condensation(G)
            topo_order_condensed = list(nx.topological_sort(condensation))
            reg_order = list(self.simulator_dict.keys())
            self._execution_order = []
            for super_node in topo_order_condensed:
                members = condensation.nodes[super_node]['members']
                sorted_members = sorted(members, key=lambda x: reg_order.index(x))
                self._execution_order.extend(sorted_members)

        logger.info("Execution order: %s", self._execution_order)

    def get_fmu_name(self, name):
        if type(name).__name__ == 'str':
            return name.split('.')[0]
        if type(name).__name__ == 'tuple':
            return [x.split('.')[0] for x in name]

    def plot(self, plot_edge_labels=False, node_size=300, node_color='r'):
        """
        Plots approximate graph diagram for the energysim network.
        """
        import matplotlib.pyplot as plt
        # Build graph if not already built
        if self._dep_graph is None:
            self._build_dependency_graph()
        self.G = self._dep_graph.copy()
        # Add edge labels for visualization
        conns = self.simulator_connections if isinstance(self.simulator_connections, dict) else {}
        for key, value in conns.items():
            if isinstance(key, str) and isinstance(value, str):
                n1 = key.split('.')[0]
                n2 = value.split('.')[0]
                self.G.edges[n1, n2]['name'] = key + ' to ' + value
            elif isinstance(key, str) and isinstance(value, tuple):
                n1 = key.split('.')[0]
                for v in value:
                    if isinstance(v, str):
                        n2 = v.split('.')[0]
                        self.G.edges[n1, n2]['name'] = key + ' to ' + v
            elif isinstance(key, tuple) and isinstance(value, str):
                n2 = value.split('.')[0]
                for k in key:
                    n1 = k.split('.')[0]
                    self.G.edges[n1, n2]['name'] = k + ' to ' + value
        pos = nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos, node_size=node_size, node_color=node_color, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos, alpha=0.8, edge_color='b')
        nx.draw_networkx_labels(self.G, pos)
        if plot_edge_labels:
            edge_labels = nx.get_edge_attributes(self.G, 'name')
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        plt.tick_params(axis="x", which="both", bottom=False, top=False)
        plt.tick_params(axis="y", which="both", bottom=False, top=False)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Time grid
    # ------------------------------------------------------------------

    def get_lcm(self):
        """Least common multiple of all simulator step sizes (exact rational arithmetic)."""
        steps = [entry.step_size for entry in self.simulator_dict.values()]
        if not steps:
            raise ValueError("Cannot compute the step-size LCM: no simulators registered.")
        fracs = []
        for s in steps:
            f = _to_fraction(s)
            if f <= 0:
                raise ValueError(f"Step sizes must be > 0, got {s!r}.")
            fracs.append(f)

        def lcm2(a, b):
            return a * b // math.gcd(a, b)

        num = reduce(lcm2, (f.numerator for f in fracs))
        den = reduce(math.gcd, (f.denominator for f in fracs))
        return float(Fraction(num, den))

    def _macro_step(self):
        if self.exchange is None or self.exchange == 0:
            return self.get_lcm()
        if self.exchange < 0:
            raise ValueError(f"Macro step size t_macro ({self.exchange}) must be > 0.")
        return self.exchange

    def _compute_total_steps(self, macro):
        """Number of macro steps; raise ValueError unless the span is an integer multiple of *macro*.

        A relative tolerance of 1e-9 absorbs floating-point error
        (e.g. 0.3 / 0.1); anything else is a configuration error.
        """
        span = self.stop_time - self.start_time
        if macro <= 0:
            raise ValueError(f"Macro step ({macro}) must be > 0.")
        n_float = span / macro
        n = round(n_float)
        if abs(n_float - n) > 1e-9 * max(1.0, abs(n_float)):
            lo = self.start_time + math.floor(n_float) * macro
            hi = self.start_time + math.ceil(n_float) * macro
            raise ValueError(
                f"Simulation span stop_time - start_time = {span} is not an integer multiple of the "
                f"macro step {macro}. Use e.g. stop_time={lo:g} or stop_time={hi:g}."
            )
        if n < 1:
            raise ValueError(
                f"stop_time - start_time ({span}) must be at least one macro step ({macro})."
            )
        return int(n)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all state of the previous run (recorded-output map, exchange history, time grid).

        The configured simulators, connections and options are kept;
        ``last_results`` is cleared.
        """
        self.last_results = None
        self.sim_dict = {}
        self._exchange_history = {}
        self.macro_tstep = None
        self._total_steps = None

    @staticmethod
    def _normalize_params(params, sim_name):
        """Accept ``([names], [values])`` or ``{name: value}``; return a dict."""
        if isinstance(params, dict):
            return dict(params)
        if isinstance(params, (list, tuple)) and len(params) == 2:
            names, values = list(params[0]), list(params[1])
            if len(names) != len(values):
                raise ValueError(
                    f"Parameters for '{sim_name}': {len(names)} names but {len(values)} values."
                )
            return dict(zip(names, values))
        raise TypeError(
            f"Parameters for '{sim_name}' must be ([names], [values]) or {{name: value}}, got {params!r}."
        )

    def _collect_parameters(self):
        merged: Dict[str, Dict[str, Any]] = {}
        for source in (self.init_dict, self._parameter_set):
            for sim_name, params in (source or {}).items():
                if sim_name not in self.simulator_dict:
                    raise ValueError(f"Parameters given for unknown simulator '{sim_name}'.")
                merged.setdefault(sim_name, {}).update(self._normalize_params(params, sim_name))
        return merged

    def init(self):
        '''
        Prepares a fresh run: resets run state, calls ``configure(start, stop, macro)`` on every adapter, then
        ``init()`` on every adapter with initial parameters applied before or
        after ``init()`` according to ``adapter.SET_PARAMS_BEFORE_INIT``.
        '''
        logger.info("Simulation started..")
        self.reset()

        self.macro_tstep = self._macro_step()
        self._total_steps = self._compute_total_steps(self.macro_tstep)
        params = self._collect_parameters()

        self.sim_dict = {name: list(entry.outputs) for name, entry in self.simulator_dict.items()}

        for sim_name, entry in self.simulator_dict.items():
            configure = getattr(entry.adapter, 'configure', None)
            if callable(configure):
                try:
                    configure(self.start_time, self.stop_time, self.macro_tstep)
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not configure the {entry.sim_type} simulator '{sim_name}': {exc}"
                    ) from exc

        for sim_name, entry in self.simulator_dict.items():
            adapter = entry.adapter
            p = params.get(sim_name)
            before = bool(getattr(adapter, 'SET_PARAMS_BEFORE_INIT', False))
            if p and before:
                self._apply_parameters(sim_name, p)
            try:
                adapter.init()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not initialize the {entry.sim_type} simulator '{sim_name}': {exc}"
                ) from exc
            if p and not before:
                self._apply_parameters(sim_name, p)

    def _apply_parameters(self, sim_name, params):
        entry = self.simulator_dict[sim_name]
        setter = getattr(entry.adapter, 'set_parameters', None)
        if callable(setter):
            setter(dict(params))
        else:
            entry.adapter.set_value(list(params.keys()), list(params.values()))

    def _cleanup_all(self):
        """Call cleanup() on every adapter; errors are logged, never raised."""
        for sim_name, entry in self.simulator_dict.items():
            cleanup = getattr(entry.adapter, 'cleanup', None)
            if not callable(cleanup):
                continue
            try:
                cleanup()
            except Exception:
                logger.warning("Error during cleanup of simulator '%s'", sim_name, exc_info=True)

    def step_simulator(self, sim_name, entry, time, record_all, local_stop_time):
        """Step a single simulator from *time* to *local_stop_time*.

        Returns a list of [time, val1, val2, …] rows for recording.
        """
        adapter = entry.adapter
        outputs = entry.outputs
        tmp_var = []

        if record_all:
            # Delegate to the adapter's advance_with_recording(), which
            # respects internal solver logic (e.g. FMU ME adaptive stepping)
            tmp_var = adapter.advance_with_recording(time, local_stop_time, outputs)
        else:
            # Record snapshot at macro boundary BEFORE stepping
            if outputs:
                tmp = [time] + list(adapter.get_value(outputs, time))
            else:
                tmp = [time]
            tmp_var.append(tmp)
            # Hand the full interval to advance()
            adapter.advance(time, local_stop_time)

        return tmp_var

    def simulate(self, pbar=True, record_all=None, validate=True, out_dir=None):
        '''
        Simulates the world object and returns a :class:`~energysim.results.Results`.

        Supports two coupling modes (set via ``world(..., coupling=...)``):
        - ``'jacobi'``: Exchange all variables first, then step all simulators.
        - ``'gauss-seidel'``: Step simulators in topological order; after each
          simulator steps, immediately propagate its outputs to downstream
          simulators before they step (algebraic loops use a one-step delay).

        Every simulator gets one row per macro step (taken before stepping;
        with ``record_all`` one row per micro step) plus a final row at
        ``stop_time`` holding the state at the end of the run.

        ``record_all`` defaults to the value given via ``options({'record_all': ...})``
        or True.  ``out_dir`` (default: the world's ``out_dir``) persists the run
        to ``<out_dir>/<YYYYmmdd-HHMMSS>_<hash8>/``; otherwise nothing is written.
        The results are also stored as ``world.last_results``; if the run
        fails, the partial results (``meta['status'] == 'failed'``) are stored
        there and the error is re-raised.  ``cleanup()`` is always called on
        every adapter.  ``simulate()`` may be called repeatedly.
        '''
        if self.coupling == 'iterative':
            raise ValueError(ITERATIVE_REMOVED_MSG)
        if self.coupling not in _COUPLINGS:
            raise ValueError(f"coupling must be 'jacobi' or 'gauss-seidel', got '{self.coupling}'")
        if record_all is None:
            record_all = True if self.record_all is None else bool(self.record_all)
        if validate:
            self.validate(raise_on_error=True)

        self.last_results = None
        out_dir = self.out_dir if out_dir is None else out_dir
        scenario = self._scenario_snapshot()
        digest = scenario_hash(scenario)
        now = _dt.datetime.now()
        run_dir = make_run_dir(out_dir, digest, now=now) if out_dir else None
        stream = run_dir is not None and _have_pyarrow()
        recorder = Recorder({name: list(entry.outputs) for name, entry in self.simulator_dict.items()},
                            chunk_rows=self._results_chunk_rows,
                            stream_dir=run_dir if stream else None)
        run_info = dict(scenario=scenario, digest=digest, created=now, run_dir=run_dir,
                        record_all=record_all, t0=_time.perf_counter())

        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                try:
                    self.init()
                    exec_order = self._execution_order or list(self.simulator_dict.keys())
                    startTime = self.start_time
                    for time_idx in tqdm(range(self._total_steps), disable=not pbar):
                        time = startTime + time_idx * self.macro_tstep
                        local_stop_time = time + self.macro_tstep

                        if self.coupling == 'jacobi':
                            tmp_dict = self._step_jacobi(exec_order, time, local_stop_time, record_all)
                        else:
                            tmp_dict = self._step_gauss_seidel(exec_order, time, local_stop_time, record_all)

                        for sim_name, rows in tmp_dict.items():
                            recorder.add_rows(sim_name, rows)
                    self._record_final_state(recorder)
                finally:
                    self._cleanup_all()
        except BaseException as exc:
            try:
                self.last_results = self._finish_results(recorder, run_info, 'failed', caught, exc)
            except Exception:
                logger.warning("Could not assemble partial results of the failed run", exc_info=True)
            raise
        finally:
            for w in caught:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)

        self.last_results = self._finish_results(recorder, run_info, 'completed', caught)
        return self.last_results

    # ------------------------------------------------------------------
    # Results assembly
    # ------------------------------------------------------------------

    def _record_final_state(self, recorder):
        """Append one row per simulator at ``stop_time`` (state at the end of the run)."""
        t_end = self.stop_time
        for sim_name, entry in self.simulator_dict.items():
            outputs = entry.outputs
            try:
                row = [t_end] + (list(entry.adapter.get_value(outputs, t_end)) if outputs else [])
            except Exception as exc:
                warnings.warn(f"Could not record the final state of '{sim_name}' at t={t_end}: {exc}",
                              RuntimeWarning, stacklevel=2)
                continue
            recorder.add_row(sim_name, row)

    def _scenario_snapshot(self):
        """Serialisable scenario description used for the metadata and the run hash."""
        try:
            scenario = self.to_dict()
        except Exception as exc:  # pragma: no cover - defensive
            scenario = {'error': f"scenario not serialisable: {exc}"}
        scenario.get('world', {}).pop('out_dir', None)
        return scenario

    def _results_units(self):
        """``{'sim.var': unit}`` from SimEntry.units, connection units and name inference."""
        conn_units = {}
        for conn in self._parsed_connections:
            out = conn[0]
            fu = getattr(conn, 'from_unit', None)
            if fu and not (isinstance(out, tuple) and isinstance(out[0], tuple)):
                conn_units.setdefault((out[0], out[1]), fu)
        units = {}
        for sim_name, entry in self.simulator_dict.items():
            for var in entry.outputs:
                u = self._resolve_unit(entry, var)
                if u is None:
                    u = conn_units.get((sim_name, var))
                units[f"{sim_name}.{var}"] = u
        return units

    def _finish_results(self, recorder, info, status, caught, error=None):
        frames = recorder.finalize()
        units = self._results_units()
        start_dt = None
        if self.start_datetime is not None:
            import pandas as pd
            start_dt = pd.Timestamp(self.start_datetime).isoformat()
        created = info['created']
        meta = {
            'run_id': f"{created.strftime('%Y%m%d-%H%M%S')}_{info['digest'][:8]}",
            'created': created.isoformat(timespec='seconds'),
            'status': status,
            'error': None if error is None else f"{type(error).__name__}: {error}",
            'wall_time_s': round(_time.perf_counter() - info['t0'], 6),
            'versions': _versions(),
            'scenario': info['scenario'],
            'scenario_sha256': info['digest'],
            'seed': self.seed,
            'start_datetime': start_dt,
            'start_time': self.start_time,
            'stop_time': self.stop_time,
            't_macro': self.macro_tstep,
            'coupling': self.coupling,
            'extrapolation': self.extrapolation,
            'record_all': info['record_all'],
            'units': units,
            'warnings': [{'category': w.category.__name__, 'message': str(w.message)}
                         for w in list(caught)[:1000]],
            'n_rows': {s: recorder.n_rows(s) for s in recorder.columns},
            'run_dir': info['run_dir'],
        }
        res = Results(frames, meta=meta, units=units, start_datetime=self.start_datetime)
        run_dir = info['run_dir']
        if run_dir:
            if recorder.stream_dir is not None:
                res._write_meta(run_dir, 'parquet', dict(recorder.files))
            else:
                warnings.warn("pyarrow is not installed; the run directory stores CSV files instead "
                              "of Parquet (pip install pyarrow).", UserWarning, stacklevel=3)
                res.save(run_dir, format='csv')
            try:
                scenario_yaml = self.to_yaml()
            except Exception:
                scenario_yaml = repr(self.to_dict())
            with open(os.path.join(run_dir, 'scenario.yaml'), 'w', encoding='utf-8') as f:
                f.write(scenario_yaml)
        return res

    # ------------------------------------------------------------------
    # Coupling strategies
    # ------------------------------------------------------------------

    def _step_jacobi(self, exec_order, time, local_stop_time, record_all):
        """Jacobi coupling: exchange all, then step all."""
        self.exchange_variable_values(time)
        tmp_dict = {}
        for sim_name in exec_order:
            entry = self.simulator_dict[sim_name]
            tmp_dict[sim_name] = self.step_simulator(sim_name, entry, time, record_all, local_stop_time)
        return tmp_dict

    def _step_gauss_seidel(self, exec_order, time, local_stop_time, record_all):
        """Gauss-Seidel coupling: step in topological order,
        exchanging each simulator's outputs before the next one steps."""
        # For simulators in cyclic SCCs, exchange their inputs first (Jacobi within SCC)
        scc_members = set()
        for scc in self._cycle_sccs:
            scc_members.update(scc)

        # Initial exchange for cyclic simulators (they need stale values to start)
        if scc_members:
            self._exchange_for_sims(time, target_sims=scc_members)

        tmp_dict = {}
        for sim_name in exec_order:
            # For non-cyclic sims: exchange inputs from already-stepped upstream sims
            if sim_name not in scc_members:
                self._exchange_inputs_for(sim_name, time)
            entry = self.simulator_dict[sim_name]
            tmp_dict[sim_name] = self.step_simulator(sim_name, entry, time, record_all, local_stop_time)
        return tmp_dict

    # ------------------------------------------------------------------
    # Gauss-Seidel exchange helpers
    # ------------------------------------------------------------------

    def _exchange_inputs_for(self, target_sim, t):
        """Exchange only those connections whose input side is *target_sim*."""
        for conn in self._parsed_connections:
            output_, input_ = conn[0], conn[1]
            scale = getattr(conn, 'scale', 1.0)
            offset = getattr(conn, 'offset', 0.0)

            # Determine which input sims this connection targets
            if isinstance(input_, list):
                matching = [(in_sim, in_var) for in_sim, in_var in input_
                            if in_sim == target_sim]
                if not matching:
                    continue
            else:
                in_sim_name, in_sim_var = input_
                if in_sim_name != target_sim:
                    continue
                matching = [(in_sim_name, in_sim_var)]

            # Gather output value(s)
            tmp_var = self._gather_output_value(output_, t)
            if scale != 1.0 or offset != 0.0:
                tmp_var = [tmp_var[0] * scale + offset]

            # Set to matching inputs only
            for in_sim_name, in_sim_var in matching:
                self.simulator_dict[in_sim_name].adapter.set_value([in_sim_var], tmp_var)

    def _exchange_for_sims(self, t, target_sims):
        """Exchange connections whose input side is in target_sims set."""
        for sim_name in target_sims:
            self._exchange_inputs_for(sim_name, t)

    def _gather_output_value(self, output_, t):
        """Read output value(s) from a parsed connection's output side."""
        if isinstance(output_, tuple) and isinstance(output_[0], tuple):
            # Fan-in: sum multiple outputs
            tmps = []
            for ou_sim_name, ou_sim_var, ou_key in output_:
                tmp1 = self.simulator_dict[ou_sim_name].adapter.get_value([ou_sim_var], t)
                tmp = self.alter_signal(ou_key, tmp1)
                tmps.append(tmp[0])
            return [sum(tmps)]
        else:
            ou_sim_name, ou_sim_var, ou_key = output_
            tmp_var1 = self.simulator_dict[ou_sim_name].adapter.get_value([ou_sim_var], t)
            return self.alter_signal(ou_key, tmp_var1)

    def alter_signal(self, op_, tmp):
        '''
        Applies the ``modify_signal`` option (``[gain]`` or ``[gain, offset]``) to output *op_*.
        '''
        temp_var = tmp[0]
        mod = self.modify_dict.get(op_) if self.modify_dict else None
        if mod is None:
            return [temp_var]
        if len(mod) == 1:
            return [temp_var * mod[0]]
        if len(mod) == 2:
            return [temp_var * mod[0] + mod[1]]
        raise ValueError(f"Invalid modify_signal entry for '{op_}': {mod!r} (expected [gain] or [gain, offset]).")

    def exchange_variable_values(self, t):
        """
        Helper function for exchanging the variable values between simulators at particular time.
        Supports zero-order hold (default) and linear extrapolation, plus per-connection scaling.
        """
        for conn in self._parsed_connections:
            output_, input_ = conn[0], conn[1]
            scale = getattr(conn, 'scale', 1.0)
            offset = getattr(conn, 'offset', 0.0)

            # --- gather output value(s) ---
            tmp_var = self._gather_output_value(output_, t)

            # --- apply scale and offset ---
            if scale != 1.0 or offset != 0.0:
                tmp_var = [tmp_var[0] * scale + offset]

            # --- apply linear extrapolation if enabled ---
            if self.extrapolation == 'linear':
                tmp_var = self._apply_extrapolation(output_, t, tmp_var)

            # --- distribute to input(s) ---
            if isinstance(input_, list):
                for in_sim_name, in_sim_var in input_:
                    self.simulator_dict[in_sim_name].adapter.set_value([in_sim_var], tmp_var)
            else:
                in_sim_name, in_sim_var = input_
                self.simulator_dict[in_sim_name].adapter.set_value([in_sim_var], tmp_var)

    def _apply_extrapolation(self, output_, t, current_value):
        """Apply linear extrapolation using history of exchanged values."""
        # Build a hashable key for this output
        if isinstance(output_, tuple) and isinstance(output_[0], tuple):
            hist_key = tuple((s, v) for s, v, _ in output_)
        else:
            hist_key = (output_[0], output_[1])

        # Update history
        if hist_key not in self._exchange_history:
            self._exchange_history[hist_key] = deque(maxlen=2)
        self._exchange_history[hist_key].append((t, current_value[0]))

        history = self._exchange_history[hist_key]
        if len(history) >= 2:
            t_prev, v_prev = history[0]
            t_curr, v_curr = history[1]
            dt = t_curr - t_prev
            if abs(dt) > 1e-15:
                slope = (v_curr - v_prev) / dt
                # Extrapolate forward by one macro-step
                v_extrap = v_curr + slope * self.macro_tstep
                return [v_extrap]
        # Fall back to zero-order hold if not enough history
        return current_value

    def get_step_time(self, step_size, local_stop_time, current_time, final_tStep, int_start_time):
        """
        Helper function to obtain the variable time step for specified FMUs.
        """
        if current_time - int_start_time < 0.01 or local_stop_time - current_time < 0.01:
            new_step_size = 0.001
        else:
            new_step_size = final_tStep - 0.01
        return new_step_size

    def options(self, settings):
        """Set run options.

        Recognised keys (anything else raises ``ValueError``):

        - ``init``: ``{sim_name: ([names], [values])}`` or ``{sim_name: {name: value}}``
          initial values, applied at the start of every run.
        - ``parameter_set``: same format, merged with ``init`` (see :meth:`set_parameters`).
        - ``modify_signal``: ``{'sim.var': [gain]}`` or ``{'sim.var': [gain, offset]}``.
        - ``record_all``: bool, default for ``simulate(record_all=...)``.
        """
        if not isinstance(settings, dict):
            raise TypeError(f"options() expects a dict, got {type(settings).__name__}.")
        unknown = sorted(set(settings) - OPTION_KEYS)
        if unknown:
            raise ValueError(f"Unknown option(s) {unknown}. Valid options: {sorted(OPTION_KEYS)}.")
        if not self.simulator_dict:
            raise ValueError("Cannot add settings to world when no simulators are specified!")

        if 'init' in settings:
            init = settings['init'] or {}
            if not isinstance(init, dict):
                raise TypeError("options['init'] must be a dict {sim_name: ([names], [values])}.")
            for sim_name, params in init.items():
                if sim_name not in self.simulator_dict:
                    raise ValueError(f"options['init'] references unknown simulator '{sim_name}'.")
                self._normalize_params(params, sim_name)
            self.init_dict = init

        if 'parameter_set' in settings:
            self.set_parameters(settings['parameter_set'] or {})

        if 'modify_signal' in settings:
            mods = settings['modify_signal'] or {}
            if not isinstance(mods, dict):
                raise TypeError("options['modify_signal'] must be a dict {'sim.var': [gain, offset]}.")
            for key, mod in mods.items():
                if not isinstance(mod, (list, tuple)) or len(mod) not in (1, 2):
                    raise ValueError(
                        f"Invalid modify_signal entry for '{key}': {mod!r} (expected [gain] or [gain, offset])."
                    )
            self.modify_dict = mods

        if 'record_all' in settings:
            ra = settings['record_all']
            if not isinstance(ra, (bool, np.bool_)):
                raise TypeError(f"options['record_all'] must be a bool, got {ra!r}.")
            self.record_all = bool(ra)

    def set_parameters(self, parameter_dictionary):
        """Register parameter values ``{sim_name: ([names], [values])}`` (or ``{sim_name: {name: value}}``).

        Since 3.0 values are *not* applied immediately: they are applied at
        the start of every run, before or after ``adapter.init()`` as the
        adapter's ``SET_PARAMS_BEFORE_INIT`` flag requires.
        """
        if not isinstance(parameter_dictionary, dict):
            raise TypeError("set_parameters() expects a dict {sim_name: ([names], [values])}.")
        for sim_name, params in parameter_dictionary.items():
            if sim_name not in self.simulator_dict:
                raise ValueError(f"set_parameters(): unknown simulator '{sim_name}'.")
            normalized = self._normalize_params(params, sim_name)
            self._parameter_set.setdefault(sim_name, {}).update(normalized)

    def check_signal_output(self, fx):
        tmp = fx(1)
        if isinstance(tmp, list):
            return True
        logger.warning('Function must return list. Signal function will not be added.')
        return False

    def plot_results(self, xlab='Time(s)', ylab='', y=None, scientific=False):
        """Plot simulation results using matplotlib.

        Parameters
        ----------
        xlab : str
            X-axis label.
        ylab : str
            Y-axis label.
        y : list[str] or None
            Variables to plot, specified as 'sim_name.var_name' or 'var_name'.
            If empty/None, plots all recorded variables.
        scientific : bool
            Whether to use scientific notation for ticks.
        """
        import matplotlib.pyplot as plt
        if y is None:
            y = []
        res = self.results(dashboard=False)
        plt.figure(figsize=(10, 6))
        for sim_name, df in res.items():
            if not hasattr(df, 'columns') or 'time' not in df.columns:
                continue
            time_vector = df['time']
            vars_to_plot = [c for c in df.columns if c != 'time']
            if y:
                vars_to_plot = [c for c in vars_to_plot if f"{sim_name}.{c}" in y or c in y]
            for var in vars_to_plot:
                plt.plot(time_vector, df[var], label=f"{sim_name}.{var}")
        plt.ticklabel_format(useOffset=scientific)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def validate(self, raise_on_error: bool = False) -> ValidationReport:
        """
        Pre-flight semantic and graph validation of the co-simulation world.

        Checks:
          1. At least one simulator registered.
          2. Step size > 0 for all simulators and divisibility with the macro step;
             the simulation span must be an integer multiple of the macro step.
          3. Connection endpoints exist and are valid.
          4. Read-only simulators (CSV, signal) are not targets of inputs.
          5. Adapters with variable introspection are verified for variable name mismatches.
          6. Algebraic loops / cycles are identified with coupling strategy advice.

        Parameters
        ----------
        raise_on_error : bool, default False
            If True and validation errors are found, raises ValidationError.

        Returns
        -------
        ValidationReport
            Detailed report containing is_valid, errors, and warnings.
        """
        errors = []
        warnings_list = []

        # 1. Simulators checks
        if not self.simulator_dict:
            errors.append("No simulators registered in the world.")
            report = ValidationReport(is_valid=False, errors=errors, warnings=warnings_list)
            if raise_on_error:
                raise ValidationError(str(report))
            return report

        if self.stop_time < self.start_time:
            errors.append(f"stop_time ({self.stop_time}) must be >= start_time ({self.start_time}).")

        if self.coupling not in _COUPLINGS:
            errors.append(ITERATIVE_REMOVED_MSG if self.coupling == 'iterative'
                          else f"Unknown coupling '{self.coupling}'.")

        step_ok = True
        for sim_name, entry in self.simulator_dict.items():
            if entry.step_size is None or entry.step_size <= 0:
                errors.append(f"Simulator '{sim_name}' has invalid step_size: {entry.step_size} <= 0.")
                step_ok = False

        macro = None
        if self.exchange is not None and self.exchange < 0:
            errors.append(f"Macro step size t_macro ({self.exchange}) must be > 0.")
        elif step_ok:
            try:
                macro = self._macro_step()
            except ValueError as exc:
                errors.append(str(exc))

        if macro is not None:
            for sim_name, entry in self.simulator_dict.items():
                ratio = macro / entry.step_size
                if abs(ratio - round(ratio)) > 1e-6:
                    warnings_list.append(
                        f"Simulator '{sim_name}' step_size ({entry.step_size}s) does not divide "
                        f"macro step t_macro ({macro}s) evenly (ratio={ratio:.4f}). "
                        f"This may lead to time misalignment."
                    )
            if self.stop_time >= self.start_time:
                try:
                    self._compute_total_steps(macro)
                except ValueError as exc:
                    errors.append(str(exc))

        # 2. Connection checks
        read_only_types = {'csv', 'signal'}
        for conn in self._parsed_connections:
            output_, input_ = conn[0], conn[1]

            # check outputs
            if isinstance(output_, tuple) and isinstance(output_[0], tuple):
                outputs_to_check = output_
            else:
                outputs_to_check = [output_]

            for ou_sim, ou_var, ou_key in outputs_to_check:
                if ou_sim not in self.simulator_dict:
                    errors.append(f"Source simulator '{ou_sim}' in connection '{ou_key}' is not registered.")
                else:
                    adapter = self.simulator_dict[ou_sim].adapter
                    avail = adapter.get_available_variables()
                    all_vars = set(avail.get('inputs', [])) | set(avail.get('outputs', []))
                    if all_vars and ou_var not in all_vars:
                        warnings_list.append(
                            f"Output variable '{ou_var}' in simulator '{ou_sim}' not found in known variables."
                        )

            # check inputs
            if isinstance(input_, list):
                inputs_to_check = input_
            else:
                inputs_to_check = [input_]

            for in_sim, in_var in inputs_to_check:
                if in_sim not in self.simulator_dict:
                    errors.append(f"Target simulator '{in_sim}' in connection to '{in_sim}.{in_var}' is not registered.")
                else:
                    entry = self.simulator_dict[in_sim]
                    if entry.sim_type in read_only_types:
                        warnings_list.append(
                            f"Simulator '{in_sim}' is read-only (type='{entry.sim_type}'). "
                            f"Targeted input '{in_var}' will be ignored."
                        )
                    adapter = entry.adapter
                    avail = adapter.get_available_variables()
                    all_vars = set(avail.get('inputs', [])) | set(avail.get('outputs', []))
                    if all_vars and in_var not in all_vars:
                        warnings_list.append(
                            f"Input variable '{in_var}' in simulator '{in_sim}' not found in known variables."
                        )

        # 3. Topology & Coupling advice
        if self._dep_graph is not None and not nx.is_directed_acyclic_graph(self._dep_graph):
            sccs = [scc for scc in nx.strongly_connected_components(self._dep_graph) if len(scc) > 1]
            if self.coupling == 'gauss-seidel':
                how = ("Gauss-Seidel cannot resolve cyclic feedback without delay; inside the loop a "
                       "one macro-step (Jacobi) delay is used.")
            else:
                how = "Jacobi coupling resolves this by introducing a one macro-step delay."
            warnings_list.append(
                f"Algebraic feedback loop detected among simulators {sccs}. {how} "
                f"To reduce the coupling error, reduce t_macro or merge the coupled components "
                f"into a single simulator."
            )

        is_valid = len(errors) == 0
        report = ValidationReport(is_valid=is_valid, errors=errors, warnings=warnings_list)
        if raise_on_error and not is_valid:
            raise ValidationError(str(report))
        return report

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self):
        """Serialize the world configuration, simulators, and connections to a dictionary."""
        sims_list = []
        for name, entry in self.simulator_dict.items():
            sim_info = {
                'name': name,
                'type': entry.sim_type,
                'step_size': entry.step_size,
                'inputs': list(entry.inputs),
                'outputs': list(entry.outputs),
                'power_unit': entry.power_unit,
            }
            if entry.units:
                sim_info['units'] = dict(entry.units)
            if entry.sim_type == 'signal':
                spec = entry.model
                if isinstance(spec, (str, dict, int, float)) and not isinstance(spec, bool):
                    sim_info['signal'] = spec
                else:
                    logger.warning(
                        "Signal '%s' is a Python callable and cannot be serialised; use an expression "
                        "string or a structured spec to make it round-trip.", name)
            elif entry.model is not None and isinstance(entry.model, str):
                sim_info['model'] = entry.model
                sim_info['location'] = entry.model
            else:
                for attr in ('sim_loc', 'fmu_loc', 'net_loc', 'csv_location', 'dss_location', 'script_loc'):
                    if hasattr(entry.adapter, attr):
                        loc_val = getattr(entry.adapter, attr)
                        if isinstance(loc_val, str):
                            sim_info['model'] = loc_val
                            sim_info['location'] = loc_val
                            break

            if entry.mode:
                sim_info['mode'] = entry.mode
            sims_list.append(sim_info)

        conns_list = []
        for conn in self._parsed_connections:
            output_, input_ = conn[0], conn[1]
            scale = getattr(conn, 'user_scale', getattr(conn, 'scale', 1.0))
            offset = getattr(conn, 'user_offset', getattr(conn, 'offset', 0.0))
            from_unit = getattr(conn, 'from_unit', None)
            to_unit = getattr(conn, 'to_unit', None)

            if isinstance(output_, tuple) and isinstance(output_[0], tuple):
                from_val = [item[2] for item in output_]
            else:
                from_val = output_[2]

            if isinstance(input_, list):
                to_val = [f"{item[0]}.{item[1]}" for item in input_]
            else:
                to_val = f"{input_[0]}.{input_[1]}"

            conn_dict = {'from': from_val, 'to': to_val}
            if scale != 1.0:
                conn_dict['scale'] = scale
            if offset != 0.0:
                conn_dict['offset'] = offset
            if from_unit and to_unit:
                conn_dict['from_unit'] = from_unit
                conn_dict['to_unit'] = to_unit
            conns_list.append(conn_dict)

        return {
            'world': {
                'start_time': self.start_time,
                'stop_time': self.stop_time,
                't_macro': self.exchange,
                'coupling': self.coupling,
                'extrapolation': self.extrapolation,
                'res_filename': self.file_name,
                'power_unit': self.power_unit,
                'logging': self.logging,
            },
            'simulators': sims_list,
            'connections': conns_list,
        }

    def to_yaml(self, filepath=None):
        """Export world scenario configuration to YAML string or file."""
        data = self.to_dict()
        yaml_str = yaml.dump(data, sort_keys=False, default_flow_style=False)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
        return yaml_str

    @staticmethod
    def _check_safe_simulator(name, sim_type, model, options):
        """Reject code-executing simulator definitions in safe mode."""
        st = (sim_type or '').lower()
        if st in UNSAFE_SIM_TYPES:
            raise ValueError(
                f"Simulator '{name}': type '{sim_type}' runs arbitrary code and is not allowed in "
                f"safe mode."
            )
        refs = [('model', model)] + [(k, v) for k, v in (options or {}).items()]
        for key, val in refs:
            if not isinstance(val, (str, os.PathLike)):
                continue
            s = os.fspath(val)
            if s.lower().endswith(PICKLE_EXTS):
                raise ValueError(
                    f"Simulator '{name}': pickle file {s!r} ({key}) can execute arbitrary code when "
                    f"loaded and is not allowed in safe mode."
                )
            if key in ('model', 'location') + _MODEL_ALIAS_KEYS and looks_like_dotted_import(s):
                raise ValueError(
                    f"Simulator '{name}': dotted import reference {s!r} is not allowed in safe mode."
                )

    @classmethod
    def from_dict(cls, data, base_dir=None, safe=False, allow_import=False):
        """Construct a world instance from a dictionary configuration.

        Parameters
        ----------
        safe : bool
            Refuse simulator types that execute arbitrary code (external, script,
            matlab, powerfactory), dotted-import model references and pickle
            (``.p`` / ``.pkl``) model files.  Use for untrusted scenario files.
        allow_import : bool
            Allow dotted-import model references (``'pkg.module.factory'``) to
            be imported and called.  Ignored (False) in safe mode.
        """
        if not isinstance(data, dict):
            raise ValueError("Scenario must be a mapping with 'world', 'simulators' and 'connections'.")
        w_cfg = data.get('world', {}) or {}
        removed = sorted(set(w_cfg) & _REMOVED_WORLD_KEYS)
        if removed:
            raise ValueError(f"World keys {removed} are no longer supported: {ITERATIVE_REMOVED_MSG}.")
        unknown = sorted(set(w_cfg) - _WORLD_KEYS)
        if unknown:
            raise ValueError(f"Unknown world key(s) {unknown}. Valid keys: {sorted(_WORLD_KEYS)}.")
        if safe:
            allow_import = False
        w = cls(
            start_time=w_cfg.get('start_time', 0),
            stop_time=w_cfg.get('stop_time', 1000),
            t_macro=w_cfg.get('t_macro', 60),
            coupling=w_cfg.get('coupling', 'jacobi'),
            extrapolation=w_cfg.get('extrapolation', 'zero-order'),
            res_filename=w_cfg.get('res_filename', 'es_res.h5'),
            power_unit=w_cfg.get('power_unit', 'kW'),
            logging=w_cfg.get('logging', False),
        )

        for sim_cfg in data.get('simulators', []) or []:
            if not isinstance(sim_cfg, dict) or 'name' not in sim_cfg or 'type' not in sim_cfg:
                raise ValueError(f"Each simulator needs at least 'name' and 'type': {sim_cfg!r}")
            s_name = sim_cfg['name']
            s_type = sim_cfg['type']
            st = str(s_type).lower()
            s_model = sim_cfg.get('model') if sim_cfg.get('model') is not None else sim_cfg.get('location')
            kwargs = dict(sim_cfg.get('options', {}) or {})
            if 'signal' in sim_cfg:
                kwargs['signal'] = sim_cfg['signal']

            if safe:
                cls._check_safe_simulator(s_name, s_type, s_model, kwargs)

            if (st != 'signal' and base_dir and isinstance(s_model, str) and s_model
                    and not os.path.isabs(s_model)):
                candidate_path = os.path.normpath(os.path.join(base_dir, s_model))
                if os.path.exists(candidate_path) or s_model.lower().endswith(MODEL_FILE_EXTS):
                    s_model = candidate_path

            w.add_simulator(
                sim_type=s_type,
                sim_name=s_name,
                model=s_model,
                inputs=sim_cfg.get('inputs', []),
                outputs=sim_cfg.get('outputs', []),
                step_size=sim_cfg.get('step_size', 1.0),
                mode=sim_cfg.get('mode') or sim_cfg.get('pf_mode') or 'ac',
                power_unit=sim_cfg.get('power_unit'),
                units=sim_cfg.get('units'),
                allow_import=allow_import,
                **kwargs
            )

        conns = data.get('connections', [])
        if conns:
            w.add_connections(conns)

        return w

    @classmethod
    def from_yaml(cls, path_or_str, safe=False, allow_import=False):
        """Load a world from a YAML file path or YAML string (see :meth:`from_dict` for ``safe``)."""
        if os.path.isfile(path_or_str):
            base_dir = os.path.dirname(os.path.abspath(path_or_str))
            with open(path_or_str, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            base_dir = None
            data = yaml.safe_load(path_or_str)

        return cls.from_dict(data, base_dir=base_dir, safe=safe, allow_import=allow_import)


def _enable_verbose_logging():
    """Make INFO messages of the 'energysim' logger visible (used by ``world(logging=True)``)."""
    if logger.level == _logging.NOTSET or logger.level > _logging.INFO:
        logger.setLevel(_logging.INFO)
    if not any(getattr(h, '_energysim_default', False) for h in logger.handlers):
        handler = _logging.StreamHandler()
        handler.setFormatter(_logging.Formatter('%(message)s'))
        handler._energysim_default = True
        logger.addHandler(handler)


# PEP 8 alias
World = world
