# -*- coding: utf-8 -*-
"""Simulator registration (``add_*`` methods) of the world."""

from __future__ import annotations
from typing import Any, Optional, List, Dict

import os
import sys
import re
import importlib
import importlib.util

import numpy as np

from ..csv_adapter import csv_simulator
from ..signalAdapter import signal_adapter
from ..utils import (
    parse_unit,
    resolve_model_reference,
)
from .._safeexpr import build_signal
from ..base import (
    SimulatorAdapter, SimEntry,
)
from ._common import (logger, _MODEL_ALIAS_KEYS, _CONTAINER_TYPES, _ext_counter, _normalize_mode,
                      _filter_kwargs, _is_adapter_like)


class RegistryMixin:
    """Simulator registration (``add_*`` methods) of the world."""


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

    def _check_single_output_signal(self, sim_name, func):
        """Reject signals that return more than one value (probe at ``t = start_time``).

        A signal exposes the single output ``y``; a callable returning ``[v]`` is fine.
        A callable that raises on the probe is accepted (it is not probed again).
        """
        try:
            probe = func(self.start_time)
        except Exception:
            return
        try:
            n_values = len(probe) if isinstance(probe, (list, tuple, np.ndarray)) else 1
        except TypeError:
            n_values = 1
        if n_values > 1:
            raise ValueError(
                f"Signal '{sim_name}' returns {n_values} values; signals expose a single output "
                f"'y'; add one signal per output (a callable returning a list of one value is fine)."
            )

    def add_signal(self, sim_name, signal=None, step_size=1, power_unit=None, units=None, breakpoints=None):
        """Add a signal source with a single output ``y``.

        A signal returning more than one value is rejected with a ``ValueError``
        (probed once at ``t = start_time``): add one signal per output.

        ``signal`` may be a callable ``f(t)``, a number (constant), a safe
        expression string (``"5*sin(t/60)"``, optionally prefixed with
        ``lambda t:``) or a structured spec dict (``{type: sine, ...}``,
        see :mod:`energysim._safeexpr`).  Strings are never ``eval``-ed.

        ``breakpoints`` (times where the signal jumps) - together with the
        breakpoints of ``step`` / ``ramp`` / ``piecewise`` specs - are event
        hints: with ``t_macro_min < t_macro`` macro steps land exactly on them.
        """
        self._check_new_name(sim_name, 'signal')
        if signal is None:
            raise ValueError(
                f"Signal simulator '{sim_name}' needs a signal: a callable, a number, a safe "
                f"expression string or a structured spec dict."
            )
        self._check_units(units, power_unit)
        func = build_signal(signal)
        self._check_single_output_signal(sim_name, func)
        signal_obj = signal_adapter(signal_name=sim_name, signal=func, breakpoints=breakpoints)
        signal_obj.step_size = step_size   # record_all samples at the declared step, not 1 s
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
                            power_unit=power_unit, units=units, breakpoints=kwargs.pop('breakpoints', None))

        elif st == 'csv':
            self.add_csv(
                csv_name=sim_name, model=target, step_size=step_size, outputs=outputs,
                delimiter=kwargs.pop('delimiter', ','), power_unit=power_unit, units=units,
                align_to_rows=kwargs.pop('align_to_rows', False))

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
            units=dict(units or {}),   # the component's own units are resolved live (adapter.units)
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

        from ..container import (
            VectorizedContainerAdapter,
            BatteryFleetAdapter,
            BuildingFleetAdapter,
            PVFleetAdapter,
            HeatPumpFleetAdapter,
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
        elif target is not None and (hasattr(target, 'step_model') or hasattr(target, 'get_value')):
            # Parameterized ModelComponent or archetype passed directly to add_fleet / add_container
            t_name = type(target).__name__.lower()
            if hasattr(target, 'capacity_mwh') or 'battery' in k_low or 'ev' in k_low or 'bess' in k_low or 'battery' in t_name or 'bess' in t_name:
                adapter = BatteryFleetAdapter.from_model(target, size=size, name=sim_name, step_size=step_size, **kwargs)
            elif hasattr(target, 'p_dc_stc_mw') or 'pv' in k_low or 'solar' in k_low or 'pv' in t_name:
                adapter = PVFleetAdapter.from_model(target, size=size, name=sim_name, step_size=step_size, **kwargs)
            elif hasattr(target, 'eta_carnot') or hasattr(target, 'carnot_efficiency') or 'hp' in k_low or 'heat_pump' in k_low or 'heatpump' in t_name:
                adapter = HeatPumpFleetAdapter.from_model(target, size=size, name=sim_name, step_size=step_size, **kwargs)
            elif hasattr(target, 'c_th') or 'building' in k_low or 'house' in k_low or 'building' in t_name:
                adapter = BuildingFleetAdapter.from_model(target, size=size, name=sim_name, step_size=step_size, **kwargs)
            else:
                raise TypeError(
                    f"ModelComponent '{type(target).__name__}' does not have an automatic fleet adapter mapping. "
                    f"Use BatteryFleetAdapter, PVFleetAdapter, HeatPumpFleetAdapter, or BuildingFleetAdapter."
                )
        elif target is not None:
            raise TypeError(
                f"Container for '{sim_name}' must be a VectorizedContainerAdapter instance or subclass, "
                f"or a ModelComponent, got {type(target).__name__}."
            )
        elif 'battery' in k_low or 'ev' in k_low or 'bess' in k_low:
            adapter = BatteryFleetAdapter(name=sim_name, size=size, step_size=step_size, **kwargs)
        elif 'building' in k_low or 'thermal' in k_low or 'house' in k_low:
            adapter = BuildingFleetAdapter(name=sim_name, size=size, step_size=step_size, **kwargs)
        elif 'pv' in k_low or 'solar' in k_low:
            adapter = PVFleetAdapter(name=sim_name, size=size, step_size=step_size, **kwargs)
        elif 'heat_pump' in k_low or 'hp' in k_low:
            adapter = HeatPumpFleetAdapter(name=sim_name, size=size, step_size=step_size, **kwargs)
        elif ('function' in k_low or callable(kwargs.get('update_func'))
              or callable(kwargs.get('update_fn')) or callable(kwargs.get('func'))):
            adapter = VectorizedFunctionAdapter(
                name=sim_name, size=size, inputs=inputs or None, step_size=step_size, **kwargs,
            )
        else:
            raise ValueError(
                f"Cannot build container '{sim_name}': specify kind='battery' | 'building' | 'pv' | "
                f"'heat_pump' | 'function' (or a fleet sim_type), or pass a container adapter via model=/container=."
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
        from ..pfAdapter import pf_adapter  # lazy import
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
        from ..matlabAdapter import matlab_adapter  # lazy import
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
        from ..ppAdapter import pp_adapter         # lazy: pandapower has heavy optional deps
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
        from ..pypsaAdapter import pypsa_adapter
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
        from ..opendssAdapter import opendss_adapter
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
        from ..pyScriptAdapter import py_adapter
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
        from ..csAdapter import FmuCsAdapter
        from ..meAdapter import FmuMeAdapter

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
                power_unit=None, units=None, csv_path=None, model=None, align_to_rows=False, **kwargs):
        """Add a CSV profile (``time`` column + value columns).

        ``align_to_rows=True`` reports the next row time as an event hint, so
        that with ``t_macro_min < t_macro`` macro steps land on CSV rows.
        """
        csv_name = csv_name or kwargs.pop('sim_name', None)
        target = self._pick_model(csv_name, model=model, csv_path=csv_path, csv_location=csv_location,
                                  sim_loc=kwargs.pop('sim_loc', None))
        self._check_new_name(csv_name, 'CSV simulator')
        if target is None:
            raise ValueError(f"No CSV path or model specified for CSV simulator '{csv_name}'.")
        self._check_units(units, power_unit)
        outputs = list(outputs or [])
        csv_obj = csv_simulator(csv_name, target, step_size, outputs, delimiter, align_to_rows=align_to_rows)
        csv_obj.step_size = step_size
        self._register(csv_name, SimEntry(
            sim_type='csv', adapter=csv_obj, step_size=step_size,
            outputs=outputs, inputs=[],
            power_unit=power_unit or self.power_unit,
            units=units or {},
            model=target,
        ), 'CSV')

