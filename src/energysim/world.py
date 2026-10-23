# -*- coding: utf-8 -*-
"""
World cosimulation canvas for energysim.
"""

from __future__ import annotations
from typing import Any, Optional, List, Dict, Union

from .csv_adapter import csv_simulator
from .signalAdapter import signal_adapter
from .utils import (
    convert_hdf_to_dict, record_data, create_results_recorder,
    get_unit_scale, infer_variable_unit,
)
from .dashboard import generate_dashboard
from .base import (
    SimulatorAdapter, SimEntry,
    EnergysimError, SimulatorVariableError,
    SimulatorElementNotFoundError, ConnectionError,
    ValidationError, ValidationReport,
    ParsedConnection,
)
import os
import sys
import re
import math
import warnings
from collections import deque
import numpy as np
import yaml
from tqdm import tqdm
from functools import reduce
import importlib
import networkx as nx
import tables as tb
filters = tb.Filters(complevel=5, complib='zlib')


class world():
    r"""
    Initialization
    ---------------
    ``world`` is your cosimulation canvas. It creates a ``world`` object on
    which simulators can be freely added. ``world`` accepts the
    following parameters :
        1. ``start_time`` : (0 by default).
        2. ``stop_time`` : (1000 by default).
        3. ``logging`` : (False by default). Gives users update on simulation progress.
        4. ``t_macro`` : (60 by default). Users can specify instance of information exchange.

    >>> my_world = world(start_time=0, stop_time=1000, logging=True, t_macro=60)

    Specifying Simulators
    ---------------------
    After initializing the world cosimulation object, you can add simulators
    to the world by ``add_simulator()``. This is done by specifying
    the following :
        1. ``sim_type`` : 'fmu', 'powerflow', 'csv', 'signal'.
        2. ``sim_name`` : It is essential that you assign the simulator a unique name.
        3. ``sim_loc`` : Use a raw string address of simulator location.
        4. ``outputs`` : specify the outputs that need to be recorded during simulation from the simulator.
        5. ``inputs`` : specify the inputs of the simulator.
        6. ``step_size`` : step size for simulator (1e-3 by default).

    >>> my_world.add_simulator(sim_type='fmu', sim_name='FMU1',
    >>> sim_loc=/path/to/sim, inputs=[], outputs=['var1','var2'], step_size=1)

    Connections between simulators
    -------------------------------
    The connections between simulators can be specified with a dictionary
    as follows :

    >>> connections = {'sim1.output_variable1' : 'sim2.input_variable1',
    >>>    'sim3.output_variable2' : 'sim4.input_variable2',
    >>>    'sim1.output_variable3' : 'sim2.input_variable3',}
    >>> my_world.add_connections(connections)

    This dictionary can be passed onto the world object.

    Initializing simulator variables
    --------------------------------
    To provide initial values to the simulators, the initial values can be
    specified by providing a ``init`` dict :

    >>> initializations = {'sim_name1' : (['sim_variables'], [values]),
    >>>                    'sim_name2' : (['sim_variables'], [values])}
    >>> options = {'init' : initializations}
    >>> my_world.options(options)

    Simulate
    --------
    Finally, the ``simulate()`` function can be called to simulate
    the world. When ``record_all`` is True, ``energysim`` records
    the value of variables not only at macro time steps, but also at
    micro time steps specified by the user when adding the simulators.
    This allows the users to get a better understanding of simulators
    in between macro time steps. When set to False, variables are
    only recorded at macro time step. This is useful in case a long
    term simulation (for ex. a day) is performed, but one of the
    simulators has a time step in milli-seconds. ``pbar`` can be used
    to toggle the progress bar for the simulation::

    >>> my_world.simulate(pbar=True, record_all=False)

    Extracting Results
    ------------------
    Results can be extracted by calling ``results()`` function on
    ``my_world`` object. This returns a dictionary object with each
    simulators' results as pandas dataframe. Additionally,
    ``to_csv`` flag can be toggled to export results to csv files.

    >>> results = my_world.results(to_csv=True)

    Package Info
    ------------
    Author : Digvijay Gusain

    Email : digvijay.gusain29@gmail.com

    """

    def __init__(self, start_time=0, stop_time=1000, logging=False, t_macro=60,
                 res_filename='es_res.h5', coupling='jacobi', extrapolation='zero-order',
                 max_iterations=10, convergence_tolerance=1e-6, power_unit='kW'):
        self.start_time = start_time
        self.stop_time = stop_time
        self.logging = logging
        self.exchange = t_macro
        self.power_unit = power_unit
        self.modify_signal = False
        self.modify_dict = {}
        self.init_dict = {}
        self.G = None
        self.sim_dict = {}
        self.simulator_connections = {}
        self._parsed_connections = []
        self.simulator_dict = {}
        self.file_name = res_filename
        # Coupling strategy: 'jacobi', 'gauss-seidel', or 'iterative'
        assert coupling in ('jacobi', 'gauss-seidel', 'iterative'), (
            f"coupling must be 'jacobi', 'gauss-seidel', or 'iterative', got '{coupling}'"
        )
        self.coupling = coupling
        # Extrapolation of exchanged variables: 'zero-order' or 'linear'
        assert extrapolation in ('zero-order', 'linear'), (
            f"extrapolation must be 'zero-order' or 'linear', got '{extrapolation}'"
        )
        self.extrapolation = extrapolation
        self._exchange_history = {}   # (sim_name, var_name) → deque(maxlen=2) of (time, value)
        # Iterative coupling settings
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        # Execution order (populated by _compute_execution_order)
        self._execution_order = None
        self._dep_graph = None
        self._cycle_sccs = []  # strongly connected components with cycles

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

    def add_signal(self, sim_name, signal, step_size=1, power_unit=None, units=None):
        if isinstance(signal, str):
            signal = eval(signal)  # evaluate string lambda if loaded from YAML
        if sim_name not in self.simulator_dict.keys():
            signal_obj = signal_adapter(signal_name=sim_name, signal=signal)
            self.simulator_dict[sim_name] = SimEntry(
                sim_type='signal', adapter=signal_obj, step_size=step_size,
                outputs=['y'], inputs=[],
                power_unit=power_unit or self.power_unit,
                units=units or {},
                model=signal,
            )

    def add_simulator(self, sim_type='', sim_name='', sim_loc=None, inputs=None, outputs=None, step_size=1,
                      mode='ac', power_unit=None, units=None, model=None, **kwargs):
        """
        Method to add simulator to ``world()`` object. ``add_simulator()`` takes the following arguments as inputs :

        1. ``sim_type`` : 'fmu', 'powerflow', 'pypsa', 'opendss', 'csv', 'external', 'powerfactory', 'matlab', 'script'
        2. ``sim_name`` : Unique name identifier for the simulator.
        3. ``model`` : In-memory Python object, file path, project name, or callable factory.
        4. ``sim_loc`` : Backward-compatible alias for ``model``.
        5. ``outputs`` : Specify the outputs to record during simulation.
        6. ``inputs`` : Specify the inputs for the simulator.
        7. ``step_size`` : Micro step size for simulator (set to 1 by default).
        8. ``mode`` : Powerflow calculation mode: 'ac' (default) or 'dc'.
        9. ``power_unit`` : Physical unit for power (default: 'kW', or native simulator unit).
        10. ``units`` : Optional dictionary mapping variable name -> unit string.
        """
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []

        # Backward compatibility for legacy 'pf' parameter
        if 'pf' in kwargs:
            mode = kwargs.pop('pf')

        target_model = model if model is not None else sim_loc
        if target_model is None:
            for k in ('net', 'path', 'network', 'fmu_path', 'csv_path', 'dss_path', 'project_name', 'script_path'):
                if k in kwargs and kwargs[k] is not None:
                    target_model = kwargs.pop(k)
                    break

        # Normalize calculation mode (energysim supports 'ac' and 'dc')
        if mode:
            m_low = mode.lower()
            if m_low in ('ac', 'pf', 'ldf', 'snapshot'):
                mode = 'ac'
            elif m_low in ('dc', 'dcpf', 'lpf', 'direct'):
                mode = 'dc'
            else:
                raise ValueError(
                    f"Unsupported calculation mode '{mode}'. "
                    f"energysim only supports 'ac' and 'dc' power flow modes."
                )

        st = sim_type.lower()
        if st == 'fmu':
            solver_name = kwargs.get('solver_name', 'Cvode')
            variable = kwargs.get('variable', False)
            validate = kwargs.get('validate', True)
            self.add_fmu(sim_name, fmu_loc=target_model, model=target_model, inputs=inputs, outputs=outputs, step_size=step_size,
                         solver_name=solver_name, variable=variable, validate=validate,
                         power_unit=power_unit, units=units)

        elif st in ('powerflow', 'pandapower'):
            self.add_powerflow(sim_name, net_loc=target_model, model=target_model, inputs=inputs, outputs=outputs, step_size=step_size,
                               mode=mode, power_unit=power_unit, units=units, **kwargs)

        elif st == 'pypsa':
            self.add_pypsa(sim_name, net_loc=target_model, model=target_model, inputs=inputs, outputs=outputs, step_size=step_size,
                           mode=mode, power_unit=power_unit, units=units, **kwargs)

        elif st == 'opendss':
            self.add_opendss(sim_name, dss_loc=target_model, model=target_model, inputs=inputs, outputs=outputs, step_size=step_size,
                             mode=mode, power_unit=power_unit, units=units, **kwargs)

        elif st in ('script', 'pyscript'):
            self.add_script(sim_name, script_loc=target_model, model=target_model, inputs=inputs, outputs=outputs, step_size=step_size,
                            power_unit=power_unit, units=units, **kwargs)

        elif st == 'signal':
            signal_func = target_model if callable(target_model) else kwargs.get('signal', None)
            self.add_signal(sim_name, signal_func, step_size=step_size,
                            power_unit=power_unit, units=units)

        elif st == 'csv':
            delimiter = kwargs.get('delimiter', ',')
            self.add_csv(
                csv_name=sim_name,
                csv_location=target_model,
                model=target_model,
                step_size=step_size,
                outputs=outputs,
                delimiter=delimiter,
                power_unit=power_unit,
                units=units)

        elif st == 'external':
            self.add_external_simulator(
                sim_name=sim_name,
                sim_loc=target_model,
                model=target_model,
                inputs=inputs,
                outputs=outputs,
                step_size=step_size,
                power_unit=power_unit,
                units=units)

        elif st == 'powerfactory':
            pf_path = kwargs.get('pf_path', None)
            self.add_powerfactory(
                sim_name=sim_name,
                sim_loc=target_model,
                model=target_model,
                inputs=inputs,
                outputs=outputs,
                step_size=step_size,
                mode=mode,
                pf_path=pf_path,
                power_unit=power_unit,
                units=units)

        elif st in ('container', 'fleet', 'vectorized', 'battery_fleet', 'building_fleet', 'pv_fleet'):
            self.add_container(
                sim_name=sim_name,
                container=target_model,
                kind=kwargs.get('kind', st),
                size=kwargs.get('size', 100),
                step_size=step_size,
                inputs=inputs,
                outputs=outputs,
                power_unit=power_unit,
                units=units,
                **kwargs,
            )

        elif st in ('model', 'component') or isinstance(target_model, SimulatorAdapter):
            self.add_model(
                sim_name=sim_name,
                model=target_model,
                inputs=inputs,
                outputs=outputs,
                step_size=step_size,
                power_unit=power_unit,
                units=units,
                **kwargs,
            )

        elif st == 'matlab':
            engine = kwargs.get('engine', 'auto')
            self.add_matlab(
                sim_name=sim_name,
                sim_loc=target_model,
                model=target_model,
                inputs=inputs,
                outputs=outputs,
                step_size=step_size,
                engine=engine,
                power_unit=power_unit,
                units=units)

        else:
            print(
                f"Simulator type {sim_type} not recognized."
                f" Possible sim_type are 'container', 'fleet', 'model', 'external', 'fmu', 'powerflow', 'pypsa',"
                f" 'opendss', 'script', 'signal', 'csv', 'powerfactory', 'matlab'."
            )
            print(f"Simulator {sim_name} not added")

    def add_model(self, sim_name, model=None, inputs=None, outputs=None, step_size=1,
                  power_unit=None, units=None, **kwargs):
        """
        Add an inbuilt physical model component (or any custom SimulatorAdapter instance) to the world.
        """
        assert sim_name is not None, "No name specified for model simulator."
        target = model if model is not None else kwargs.get('sim_loc', None)
        assert target is not None, f"No model instance or factory provided for '{sim_name}'."

        if callable(target) and not isinstance(target, SimulatorAdapter):
            target = target(**kwargs)

        adapter = target
        adapter.step_size = step_size

        eff_inputs = list(inputs) if inputs is not None and len(inputs) > 0 else list(getattr(adapter, 'inputs', []))
        eff_outputs = list(outputs) if outputs is not None and len(outputs) > 0 else list(getattr(adapter, 'outputs', []))
        adapter.inputs = eff_inputs
        adapter.outputs = eff_outputs

        if sim_name not in self.simulator_dict.keys():
            self.simulator_dict[sim_name] = SimEntry(
                sim_type='model', adapter=adapter, step_size=step_size,
                outputs=eff_outputs, inputs=eff_inputs,
                power_unit=power_unit or getattr(adapter, 'power_unit', self.power_unit),
                units=units or getattr(adapter, 'units', {}),
                model=target,
            )
            if self.logging:
                print(f"Added model simulator '{sim_name}' to the world")
        else:
            print(f"Please specify a unique name for simulator. {sim_name} already exists.")

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
        from .container import (
            VectorizedContainerAdapter,
            BatteryFleetAdapter,
            BuildingFleetAdapter,
            PVFleetAdapter,
            VectorizedFunctionAdapter,
        )

        target = container or model
        adapter = None

        if isinstance(target, VectorizedContainerAdapter):
            adapter = target
            adapter.name = sim_name
            if step_size:
                adapter.step_size = step_size
        elif isinstance(target, type) and issubclass(target, VectorizedContainerAdapter):
            adapter = target(name=sim_name, size=size, step_size=step_size, **kwargs)
        else:
            k_low = (kind or '').lower()
            if 'battery' in k_low:
                adapter = BatteryFleetAdapter(name=sim_name, size=size, step_size=step_size, **kwargs)
            elif 'building' in k_low or 'thermal' in k_low:
                adapter = BuildingFleetAdapter(name=sim_name, size=size, step_size=step_size, **kwargs)
            elif 'pv' in k_low or 'solar' in k_low:
                adapter = PVFleetAdapter(name=sim_name, size=size, step_size=step_size, **kwargs)
            elif 'function' in k_low or callable(kwargs.get('update_func')):
                adapter = VectorizedFunctionAdapter(
                    name=sim_name, size=size,
                    update_func=kwargs.get('update_func'),
                    initial_states=kwargs.get('initial_states', {}),
                    inputs=inputs,
                    step_size=step_size,
                )
            else:
                adapter = VectorizedContainerAdapter(
                    name=sim_name, size=size, step_size=step_size, inputs=inputs, outputs=outputs
                )

        if outputs:
            adapter.outputs = list(outputs)
        if inputs:
            adapter.inputs = list(inputs)

        # Default outputs: use all available scalar aggregations if none provided
        if not adapter.outputs:
            avail = adapter.get_available_variables()
            adapter.outputs = [v for v in avail['outputs'] if v.startswith(('total_', 'mean_'))]

        if sim_name not in self.simulator_dict:
            self.simulator_dict[sim_name] = SimEntry(
                sim_type='container',
                adapter=adapter,
                step_size=adapter.step_size,
                outputs=adapter.outputs,
                inputs=adapter.inputs,
                power_unit=power_unit or self.power_unit,
                units=units or {},
                model=adapter,
            )
            if self.logging:
                print(f"Added Vectorized Container Simulator '{sim_name}' (size={adapter.size}) to the world")
        else:
            print(f"Please specify a unique name for simulator. {sim_name} already exists.")

    # Convenience alias for add_container
    add_fleet = add_container

    def add_external_simulator(self, sim_name, sim_loc=None, inputs=None, outputs=None, step_size=900,
                               power_unit=None, units=None, model=None, **kwargs):
        target = model if model is not None else sim_loc
        if isinstance(target, str) and target:
            sys.path.append(target)
        tmp = importlib.import_module(sim_name)
        external_simulator = tmp.external_simulator
        inputs = inputs or []
        outputs = outputs or []
        # check that the sim_name is unique
        if sim_name not in self.simulator_dict.keys():
            simulator = external_simulator(inputs=inputs, outputs=outputs)
            simulator.step_size = step_size
            self.simulator_dict[sim_name] = SimEntry(
                sim_type='external', adapter=simulator, step_size=step_size,
                outputs=outputs, inputs=inputs,
                power_unit=power_unit or self.power_unit,
                units=units or {},
                model=target,
            )
            if self.logging:
                print("Added external simulator '%s' to the world" % (sim_name))
        else:
            print(f"Please specify a unique name for simulator. {sim_name} already exists.")

    def add_powerfactory(self, sim_name, sim_loc=None, inputs=None,
                         outputs=None, step_size=900, mode='ac', pf=None,
                         pf_path=None, power_unit=None, units=None,
                         project_name=None, model=None, **kwargs):
        """Add a DIgSILENT PowerFactory model to the co-simulation."""
        p_name = project_name or model or sim_loc or sim_name
        from .pfAdapter import pf_adapter  # lazy import
        adapter = pf_adapter(
            project_name=p_name,
            pf_loc=sim_loc or p_name,
            inputs=inputs or [],
            outputs=outputs or [],
            mode=mode,
            pf=pf,
            pf_path=pf_path,
        )
        adapter.step_size = step_size
        if sim_name not in self.simulator_dict:
            self.simulator_dict[sim_name] = SimEntry(
                sim_type='powerfactory', adapter=adapter,
                step_size=step_size, outputs=outputs or [],
                inputs=inputs or [], mode=adapter.mode,
                power_unit=power_unit or 'MW',
                units=units or {},
                model=p_name,
            )
            if self.logging:
                print(f"Added PowerFactory simulator '{sim_name}' to the world")
        else:
            print(f"Please specify a unique name for simulator. {sim_name} already exists.")

    def add_matlab(self, sim_name, sim_loc=None, inputs=None, outputs=None,
                   step_size=60, engine='auto', power_unit=None, units=None,
                   script_path=None, model=None, **kwargs):
        """Add a MATLAB / Octave .m function to the co-simulation."""
        target = model if model is not None else (script_path if script_path is not None else sim_loc)
        from .matlabAdapter import matlab_adapter  # lazy import
        adapter = matlab_adapter(
            func_name=sim_name,
            script_loc=target,
            inputs=inputs or [],
            outputs=outputs or [],
            engine=engine,
        )
        adapter.step_size = step_size
        if sim_name not in self.simulator_dict:
            self.simulator_dict[sim_name] = SimEntry(
                sim_type='matlab', adapter=adapter,
                step_size=step_size, outputs=outputs or [],
                inputs=inputs or [],
                power_unit=power_unit or self.power_unit,
                units=units or {},
                model=target,
            )
            if self.logging:
                print(f"Added MATLAB/Octave simulator '{sim_name}' to the world")
        else:
            print(f"Please specify a unique name for simulator. {sim_name} already exists.")

    def add_powerflow(self, network_name=None, net_loc=None, inputs=None, outputs=None, mode='ac', pf=None, step_size=900,
                      power_unit=None, units=None, net=None, path=None, model=None, **kwargs):
        network_name = network_name or kwargs.pop('sim_name', None)
        target = model if model is not None else (net if net is not None else (path if path is not None else (net_loc if net_loc is not None else kwargs.pop('sim_loc', None))))
        assert network_name is not None, "No name specified for power flow model. Name must be specified."
        assert target is not None, "No network model or location specified for power flow model."
        inputs = inputs or []
        outputs = outputs or []
        from .ppAdapter import pp_adapter         # lazy: pandapower has heavy optional deps
        network = pp_adapter(network_name, net_loc=target, inputs=inputs, outputs=outputs, mode=mode, pf=pf, model=target)
        network.step_size = step_size
        if network_name not in self.simulator_dict.keys():
            self.simulator_dict[network_name] = SimEntry(
                sim_type='powerflow', adapter=network, step_size=step_size,
                outputs=outputs, inputs=inputs, mode=network.mode,
                power_unit=power_unit or 'MW',
                units=units or {},
                model=target,
            )
            if self.logging:
                print("Added powerflow simulator '%s' to the world" % (network_name))
        else:
            print(f"Please specify a unique name for simulator. {network_name} already exists.")

    def add_pypsa(self, network_name=None, net_loc=None, inputs=None, outputs=None, mode='ac', pf=None, step_size=900,
                  power_unit=None, units=None, network=None, path=None, model=None, **kwargs):
        """Add a PyPSA network simulator to the world (Pattern B single-snapshot mode)."""
        network_name = network_name or kwargs.pop('sim_name', None)
        target = model if model is not None else (network if network is not None else (path if path is not None else (net_loc if net_loc is not None else kwargs.pop('sim_loc', None))))
        assert network_name is not None, "No name specified for PyPSA model."
        assert target is not None, "No network model or location specified for PyPSA model."
        inputs = inputs or []
        outputs = outputs or []
        from .pypsaAdapter import pypsa_adapter
        network_adapter = pypsa_adapter(
            network_name, net_loc=target, inputs=inputs, outputs=outputs, mode=mode, pf=pf,
            logger_level=kwargs.get('logger', 'CRITICAL'), model=target
        )
        network_adapter.step_size = step_size
        if network_name not in self.simulator_dict.keys():
            self.simulator_dict[network_name] = SimEntry(
                sim_type='pypsa', adapter=network_adapter, step_size=step_size,
                outputs=outputs, inputs=inputs, mode=network_adapter.mode,
                power_unit=power_unit or 'MW',
                units=units or {},
                model=target,
            )
            if self.logging:
                print(f"Added PyPSA simulator '{network_name}' to the world (Pattern B single-snapshot mode)")
        else:
            print(f"Please specify a unique name for simulator. {network_name} already exists.")

    def add_opendss(self, dss_name=None, dss_loc=None, inputs=None, outputs=None, mode='ac', pf=None, step_size=900,
                    power_unit=None, units=None, dss_path=None, model=None, **kwargs):
        """Add an OpenDSS distribution network simulator to the world via OpenDSSDirect.py."""
        dss_name = dss_name or kwargs.pop('sim_name', None)
        target = model if model is not None else (dss_path if dss_path is not None else (dss_loc if dss_loc is not None else kwargs.pop('sim_loc', None)))
        assert dss_name is not None, "No name specified for OpenDSS model."
        assert target is not None, "No DSS path or model specified for OpenDSS simulator."
        inputs = inputs or []
        outputs = outputs or []
        from .opendssAdapter import opendss_adapter
        dss_obj = opendss_adapter(dss_name, dss_location=target, inputs=inputs, outputs=outputs, step_size=step_size, mode=mode, pf=pf, model=target)
        if dss_name not in self.simulator_dict.keys():
            self.simulator_dict[dss_name] = SimEntry(
                sim_type='opendss', adapter=dss_obj, step_size=step_size,
                outputs=outputs, inputs=inputs, mode=dss_obj.mode,
                power_unit=power_unit or 'kW',
                units=units or {},
                model=target,
            )
            if self.logging:
                print(f"Added OpenDSS simulator '{dss_name}' to the world")
        else:
            print(f"Please specify a unique name for simulator. {dss_name} already exists.")

    def add_script(self, script_name, script_loc=None, inputs=None, outputs=None, step_size=900,
                   power_unit=None, units=None, script_path=None, model=None, **kwargs):
        """Add a Python script/module function simulator to the world."""
        target = model if model is not None else (script_path if script_path is not None else (script_loc if script_loc is not None else kwargs.pop('sim_loc', None)))
        assert script_name is not None, "No name specified for Python script simulator."
        assert target is not None, "No script path or model specified for Python script simulator."
        inputs = inputs or []
        outputs = outputs or []
        from .pyScriptAdapter import py_adapter
        adapter = py_adapter(script_name=script_name, script_loc=target, inputs=inputs, outputs=outputs)
        adapter.step_size = step_size
        sim_id = kwargs.get('sim_name', script_name)
        if sim_id not in self.simulator_dict.keys():
            self.simulator_dict[sim_id] = SimEntry(
                sim_type='script', adapter=adapter, step_size=step_size,
                outputs=outputs, inputs=inputs,
                power_unit=power_unit or self.power_unit,
                units=units or {},
                model=target,
            )
            if self.logging:
                print(f"Added script simulator '{sim_id}' to the world")
        else:
            print(f"Please specify a unique name for simulator. {sim_id} already exists.")

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
            **kwargs):
        fmu_name = fmu_name or kwargs.pop('sim_name', None)
        target = model if model is not None else (fmu_path if fmu_path is not None else (fmu_loc if fmu_loc is not None else kwargs.pop('sim_loc', None)))
        assert fmu_name is not None, "No name specified for FMU simulator."
        assert target is not None, "No FMU path or model specified for FMU simulator."
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
        # Lazy imports — fmpy is only pulled in when an FMU is actually added
        from fmpy.model_description import read_model_description
        from .csAdapter import FmuCsAdapter
        from .meAdapter import FmuMeAdapter

        if 'validate' in kwargs.keys():
            validate = kwargs['validate']
        else:
            validate = True
        m_desc = read_model_description(target, validate=validate)
        fmi_type = 'CoSimulation' if m_desc.coSimulation is not None else 'ModelExchange'
        if fmi_type == 'CoSimulation':
            fmu_temp = FmuCsAdapter(target,
                                    instanceName=fmu_name,
                                    step_size=step_size,
                                    inputs=inputs,
                                    outputs=outputs,
                                    exist=exist,
                                    validate=validate)
            fmu_temp.variable_step = variable
            fmu_temp.step_size = step_size
        elif fmi_type == 'ModelExchange':
            fmu_temp = FmuMeAdapter(target,
                                    instanceName=fmu_name,
                                    step_size=step_size,
                                    inputs=inputs,
                                    outputs=outputs,
                                    solver_name=solver_name,
                                    validate=validate)

        if fmu_name not in self.simulator_dict.keys():
            self.simulator_dict[fmu_name] = SimEntry(
                sim_type='fmu', adapter=fmu_temp, step_size=step_size,
                outputs=outputs, inputs=inputs, variable_step=variable,
                power_unit=power_unit or 'W',
                units=units or {},
                model=target,
            )
            if self.logging:
                print("Added FMU simulator '%s' to the world" % (fmu_name))
        else:
            print(f"Please specify a unique name for simulator. {fmu_name} already exists.")

    def add_csv(self, csv_name=None, csv_location=None, step_size=1, outputs=None, delimiter=',',
                power_unit=None, units=None, csv_path=None, model=None, **kwargs):
        csv_name = csv_name or kwargs.pop('sim_name', None)
        target = model if model is not None else (csv_path if csv_path is not None else (csv_location if csv_location is not None else kwargs.pop('sim_loc', None)))
        assert csv_name is not None, "No name specified for CSV simulator."
        assert target is not None, "No CSV path or model specified for CSV simulator."
        outputs = outputs or []
        csv_obj = csv_simulator(csv_name, target, step_size, outputs, delimiter)
        csv_obj.step_size = step_size
        if csv_name not in self.simulator_dict.keys():
            self.simulator_dict[csv_name] = SimEntry(
                sim_type='csv', adapter=csv_obj, step_size=step_size,
                outputs=outputs, inputs=[],
                power_unit=power_unit or self.power_unit,
                units=units or {},
                model=target,
            )
            if self.logging:
                print("Added CSV simulator '%s' to the world" % (csv_name))

    def add_connections(self, connections=None):
        '''
        Adds connections between simulators. Connections are
        specified as dictionary with keys as sending variables,
        and values as receiving variables.
        '''
        if connections is None:
            connections = {}

        assert len(self.simulator_dict) > 0, "Cannot make connections with no simulators defined."
        self.simulator_connections = connections

        if isinstance(connections, list):
            raw_connections = connections
        elif isinstance(connections, dict):
            raw_connections = []
            for k, v in connections.items():
                if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
                    raw_connections.append({
                        'from': k,
                        'to': v[0],
                        'scale': v[1].get('scale', 1.0),
                        'offset': v[1].get('offset', 0.0),
                        'from_unit': v[1].get('from_unit') or v[1].get('unit'),
                        'to_unit': v[1].get('to_unit') or v[1].get('unit'),
                    })
                elif isinstance(v, dict):
                    raw_connections.append({
                        'from': k,
                        'to': v.get('to') or v.get('target'),
                        'scale': v.get('scale', 1.0),
                        'offset': v.get('offset', 0.0),
                        'from_unit': v.get('from_unit') or v.get('unit'),
                        'to_unit': v.get('to_unit') or v.get('unit'),
                    })
                else:
                    raw_connections.append({'from': k, 'to': v})
        else:
            raw_connections = []

        # Pre-parse connections once so exchange_variable_values() doesn't
        # re-split strings on every time step.
        self._parsed_connections = []
        for item in raw_connections:
            output_key = item.get('from') or item.get('source')
            input_key = item.get('to') or item.get('target')
            scale = float(item.get('scale', 1.0))
            offset = float(item.get('offset', 0.0))
            from_unit = item.get('from_unit') or item.get('unit')
            to_unit = item.get('to_unit') or item.get('unit')

            # --- parse output side ---
            if isinstance(output_key, str):
                ou_sim, ou_var = self._split_var(output_key)
                parsed_output = (ou_sim, ou_var, output_key)
                first_ou_sim, first_ou_var = ou_sim, ou_var
            elif isinstance(output_key, tuple):
                parsed_output = tuple(
                    (self._split_var(k)[0], self._split_var(k)[1], k)
                    for k in output_key
                )
                first_ou_sim, first_ou_var = parsed_output[0][0], parsed_output[0][1]
            # --- parse input side ---
            if isinstance(input_key, str):
                in_sim, in_var = self._split_var(input_key)
                parsed_input = (in_sim, in_var)
                first_in_sim, first_in_var = in_sim, in_var
            elif isinstance(input_key, (list, tuple)):
                parsed_input = [
                    (self._split_var(k)[0], self._split_var(k)[1])
                    for k in input_key
                ]
                first_in_sim, first_in_var = parsed_input[0][0], parsed_input[0][1]

            # Resolve physical units
            src_entry = self.simulator_dict.get(first_ou_sim)
            dst_entry = self.simulator_dict.get(first_in_sim)

            src_sim_type = src_entry.sim_type if src_entry else ''
            dst_sim_type = dst_entry.sim_type if dst_entry else ''
            src_default_unit = (src_entry.power_unit if src_entry else None) or self.power_unit
            dst_default_unit = (dst_entry.power_unit if dst_entry else None) or self.power_unit

            if not from_unit:
                if src_entry and first_ou_var in src_entry.units:
                    from_unit = src_entry.units[first_ou_var]
                else:
                    from_unit = infer_variable_unit(first_ou_var, sim_type=src_sim_type, default_unit=src_default_unit)

            if not to_unit:
                if dst_entry and first_in_var in dst_entry.units:
                    to_unit = dst_entry.units[first_in_var]
                else:
                    to_unit = infer_variable_unit(first_in_var, sim_type=dst_sim_type, default_unit=dst_default_unit)

            # Apply unit conversion scale if units differ and scale wasn't manually overridden without units
            if 'scale' in item and ('from_unit' not in item and 'to_unit' not in item and 'unit' not in item):
                unit_scale = 1.0
            else:
                unit_scale = get_unit_scale(from_unit, to_unit)

            effective_scale = scale * unit_scale

            self._parsed_connections.append(
                ParsedConnection(parsed_output, parsed_input, effective_scale, offset, from_unit, to_unit)
            )

        # Run validation checks
        self._validate_connections()

        # Build dependency graph and compute execution order
        self._build_dependency_graph()
        self._compute_execution_order()

        if self.logging:
            print("Added %i connections between simulators to the World" % (len(self.simulator_connections)))

    # ------------------------------------------------------------------
    # Connection validation
    # ------------------------------------------------------------------

    @staticmethod
    def _split_var(dotted_name):
        """Split 'sim_name.var.sub' → ('sim_name', 'var.sub')."""
        parts = dotted_name.split('.', 1)
        if len(parts) < 2:
            raise ConnectionError(
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
        """Raise ConnectionError if simulator doesn't exist."""
        if sim_name not in known_sims:
            raise ConnectionError(
                f"Simulator '{sim_name}' referenced in connection "
                f"'{connection_str}' does not exist. "
                f"Available simulators: {sorted(known_sims)}"
            )

    def _validate_variable(self, sim_name, var_name, direction='output'):
        """Emit a warning if a variable name doesn't match known variables."""
        entry = self.simulator_dict[sim_name]
        adapter = entry.adapter
        avail = adapter.get_available_variables()
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
                    f"Using Jacobi coupling (one-step delay) for these cyclic dependencies.",
                    UserWarning, stacklevel=3
                )
            # Condensation: collapse SCCs into super-nodes, topologically sort
            condensation = nx.condensation(G)
            # nx.condensation returns a DAG of super-nodes; each super-node
            # has a 'members' attribute (set of original nodes)
            topo_order_condensed = list(nx.topological_sort(condensation))
            # Expand super-nodes back to individual simulators
            # Preserve original registration order within each SCC
            reg_order = list(self.simulator_dict.keys())
            self._execution_order = []
            for super_node in topo_order_condensed:
                members = condensation.nodes[super_node]['members']
                # Sort members by registration order
                sorted_members = sorted(members, key=lambda x: reg_order.index(x))
                self._execution_order.extend(sorted_members)

        if self.logging:
            print(f"Execution order: {self._execution_order}")

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
        for key, value in self.simulator_connections.items():
            if isinstance(key, str) and isinstance(value, str):
                n1 = key.split('.')[0]
                n2 = value.split('.')[0]
                self.G.edges[n1, n2]['name'] = key + ' to ' + value
            elif isinstance(key, str) and isinstance(value, tuple):
                n1 = key.split('.')[0]
                for v in value:
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

    def get_lcm(self):
        ss_dict = [entry.step_size for entry in self.simulator_dict.values()]
        # Coerce each value to float so str() always produces a decimal component,
        # avoiding IndexError when a step size is a plain integer.
        str_parts = [str(float(i)).split('.') for i in ss_dict]
        max_order = int(max(len(p[1]) for p in str_parts))
        multiplier = 10 ** max_order
        # Work with integers to avoid floating-point modulo errors
        new_list = [int(round(float(i) * multiplier)) for i in ss_dict]

        def lcm2(a, b):
            return a * b // math.gcd(a, b)

        final_lcm = reduce(lcm2, new_list)
        return final_lcm / multiplier

    def init(self):
        '''
        Initialises simulator objects in world.
        '''
        if self.logging:
            print("Simulation started..")
            print("Simulation status:\n")

        self.sim_dict = {}
        for name, entry in self.simulator_dict.items():
            self.sim_dict[name] = entry.outputs

        # determine the final_tStep for World
        if self.exchange != 0:
            self.macro_tstep = self.exchange
        else:
            self.macro_tstep = self.get_lcm()

        create_results_recorder(file_name=self.file_name, sim_dict=self.sim_dict)

        # set initial conditions
        if self.init_dict:
            self.set_parameters(self.init_dict)

        # initialize simulators
        for sim_name, entry in self.simulator_dict.items():
            try:
                entry.adapter.init()
            except Exception:
                try:
                    from time import sleep
                    sleep(2)
                    entry.adapter.init()
                except Exception as e2:
                    print(
                        f"Could not initialize the {
                            entry.sim_type} simulator '{sim_name}': {e2}. Simulation stopped.")
                    raise RuntimeError(f"Initialization failed for '{sim_name}'") from e2

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

    def simulate(self, pbar=True, record_all=True, validate=True):
        '''
        Simulates the world object.

        Supports three coupling modes (set via ``world(..., coupling=...)``:
        - ``'jacobi'``: Exchange all variables first, then step all simulators.
        - ``'gauss-seidel'``: Step simulators in topological order; after each
          simulator steps, immediately propagate its outputs to downstream
          simulators before they step.
        - ``'iterative'``: Repeat exchange+step within each macro-step until
          output values converge (fixed-point iteration with rollback).
        '''
        if validate:
            self.validate(raise_on_error=True)

        startTime = self.start_time
        stopTime = self.stop_time
        self.init()
        assert (stopTime - startTime >=
                self.macro_tstep), (
            "difference between start and stop time > exchange"
            " value specified in world initialisation"
        )

        # Determine execution order
        exec_order = self._execution_order or list(self.simulator_dict.keys())

        total_steps = int((stopTime - startTime) / self.macro_tstep)
        with tb.open_file(filename=self.file_name, mode='a') as hdf_file:
            for time_idx in tqdm(range(total_steps), disable=not pbar):
                time = startTime + time_idx * self.macro_tstep
                local_stop_time = time + self.macro_tstep

                if self.coupling == 'jacobi':
                    tmp_dict = self._step_jacobi(exec_order, time, local_stop_time, record_all)
                elif self.coupling == 'gauss-seidel':
                    tmp_dict = self._step_gauss_seidel(exec_order, time, local_stop_time, record_all)
                elif self.coupling == 'iterative':
                    tmp_dict = self._step_iterative(exec_order, time, local_stop_time, record_all)

                record_data(hdf_file, tmp_dict)

        # Cleanup ALL adapters — not just FMUs
        for sim_name, entry in self.simulator_dict.items():
            try:
                entry.adapter.cleanup()
            except Exception:
                pass  # best-effort cleanup

        return 'Done'

    # ------------------------------------------------------------------
    # Coupling strategies
    # ------------------------------------------------------------------

    def _step_jacobi(self, exec_order, time, local_stop_time, record_all):
        """Jacobi coupling: exchange all, then step all."""
        self.exchange_variable_values(time)
        tmp_dict = {}
        for sim_name in exec_order:
            entry = self.simulator_dict[sim_name]
            tmp_var = self.step_simulator(sim_name, entry, time, record_all, local_stop_time)
            tmp_dict[sim_name] = np.array(tmp_var)
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
            tmp_var = self.step_simulator(sim_name, entry, time, record_all, local_stop_time)
            tmp_dict[sim_name] = np.array(tmp_var)
        return tmp_dict

    def _step_iterative(self, exec_order, time, local_stop_time, record_all):
        """Iterative coupling: repeat exchange+step until convergence."""
        # Save states for rollback
        saved_states = {}
        for sim_name in exec_order:
            adapter = self.simulator_dict[sim_name].adapter
            saved_states[sim_name] = adapter.save_state()

        prev_outputs = None
        converged = False

        for iteration in range(self.max_iterations):
            # Restore states for all simulators (except first iteration)
            if iteration > 0:
                for sim_name in exec_order:
                    adapter = self.simulator_dict[sim_name].adapter
                    if saved_states[sim_name] is not None:
                        adapter.restore_state(saved_states[sim_name])

            # Run one Gauss-Seidel pass
            tmp_dict = self._step_gauss_seidel(exec_order, time, local_stop_time, record_all)

            # Collect current outputs for convergence check
            curr_outputs = {}
            for sim_name in exec_order:
                entry = self.simulator_dict[sim_name]
                if entry.outputs:
                    vals = entry.adapter.get_value(entry.outputs, local_stop_time)
                    curr_outputs[sim_name] = np.array(vals, dtype=float)

            # Check convergence
            if prev_outputs is not None:
                max_residual = 0.0
                for sim_name, curr_vals in curr_outputs.items():
                    if sim_name in prev_outputs:
                        prev_vals = prev_outputs[sim_name]
                        denom = np.maximum(np.abs(curr_vals), 1.0)
                        residual = np.max(np.abs(curr_vals - prev_vals) / denom)
                        max_residual = max(max_residual, residual)

                if max_residual < self.convergence_tolerance:
                    converged = True
                    if self.logging:
                        print(f"  Iterative coupling converged at iteration {iteration + 1} "
                              f"(residual={max_residual:.2e})")
                    break

            prev_outputs = curr_outputs

        if not converged and self.max_iterations > 1:
            warnings.warn(
                f"Iterative coupling did not converge within {self.max_iterations} "
                f"iterations at time={time}. Accepting last iterate.",
                UserWarning, stacklevel=2
            )

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

    def results(self, to_csv=False, dashboard=True, dashboard_path=None):
        """
        Processes the results from the cosimulation and provides them in a dict+df form.

        Parameters
        ----------
        to_csv : bool
            If True, export results to CSV files.
        dashboard : bool
            If True (default), opens an interactive HTML dashboard in the
            browser showing all recorded variables.
        dashboard_path : str or None
            Optional file path for the dashboard HTML file. If None a
            temporary file is created.
        """
        if not self.sim_dict:
            raise RuntimeError("No simulation results available. Call simulate() first.")
        res = convert_hdf_to_dict(file_name=self.file_name, sim_dict=self.sim_dict, to_csv=to_csv)

        # --- Convert raw time (seconds) to a human-readable unit ----
        if isinstance(res, dict):
            time_label = self._convert_time_column(res)
        else:
            time_label = 'Time (s)'

        if dashboard:
            try:
                path = generate_dashboard(
                    res, output_path=dashboard_path,
                    auto_open=True, time_label=time_label,
                )
                print(f'Dashboard opened: {path}')
            except Exception as e:
                print(f'Could not open dashboard: {e}')
        return res

    @staticmethod
    def _convert_time_column(res):
        """Auto-detect time unit and convert the *time* column in-place.

        Returns the axis label string (e.g. ``'Time (h)'``).
        """
        import pandas as pd
        max_t = max(
            (df['time'].max() for df in res.values()
             if isinstance(df, pd.DataFrame) and 'time' in df.columns and len(df) > 0),
            default=0,
        )
        if max_t >= 7200:        # spans hours → show hours
            divisor, label = 3600, 'Time (h)'
        elif max_t >= 120:       # spans minutes → show minutes
            divisor, label = 60, 'Time (min)'
        else:
            divisor, label = 1, 'Time (s)'

        if divisor != 1:
            for df in res.values():
                if isinstance(df, pd.DataFrame) and 'time' in df.columns:
                    df['time'] = df['time'] / divisor
        return label

    def alter_signal(self, op_, tmp):
        '''
        Checks if the signal needs modification as aspecified in modify_signal dict under options.
        '''
        # check if signal modification is needed.
        temp_var = tmp[0]
        if self.modify_dict:
            flag = True if op_ in self.modify_dict.keys() else False
            if flag:
                if len(self.modify_dict[op_]) == 1:
                    ret_output = temp_var * self.modify_dict[op_][0]
                elif len(self.modify_dict[op_]) == 2:
                    ret_output = temp_var * self.modify_dict[op_][0] + self.modify_dict[op_][1]
                else:
                    print('Unknown signal modification on output %s. Not modification applied.' % (op_))
                    ret_output = temp_var
            else:
                ret_output = temp_var
        else:
            ret_output = temp_var
        return [ret_output]

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
        assert (len(self.simulator_dict) > 0), "Cannot add settings to world when no simulators are specified!"

        if 'parameter_set' in settings.keys():
            self.set_parameters(settings['parameter_set'])

        if 'modify_signal' in settings.keys():
            self.modify_dict = settings['modify_signal']

        if 'init' in settings.keys():
            self.init_dict = settings['init']

    def set_parameters(self, parameter_dictionary):
        for sim_name, params in parameter_dictionary.items():
            entry = self.simulator_dict[sim_name]
            parameter_list = list(params)[0]
            parameter_values = list(params)[1]
            # Uniform call — each adapter's set_parameters knows what to do
            # (FMU adapters call apply_start_values; others delegate to set_value)
            entry.adapter.set_parameters(dict(zip(parameter_list, parameter_values)))

    def check_signal_output(self, fx):
        tmp = fx(1)
        if isinstance(tmp, list):
            return True
        else:
            print('Function must return list. Signal function will not be added.')
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
          2. Step size > 0 for all simulators and divisibility with t_macro.
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

        if self.exchange <= 0:
            errors.append(f"Macro step size t_macro ({self.exchange}) must be > 0.")

        for sim_name, entry in self.simulator_dict.items():
            if entry.step_size <= 0:
                errors.append(f"Simulator '{sim_name}' has invalid step_size: {entry.step_size} <= 0.")
            else:
                # Divisibility check
                ratio = self.exchange / entry.step_size
                if abs(ratio - round(ratio)) > 1e-6:
                    warnings_list.append(
                        f"Simulator '{sim_name}' step_size ({entry.step_size}s) does not divide "
                        f"macro step t_macro ({self.exchange}s) evenly (ratio={ratio:.4f}). "
                        f"This may lead to time misalignment."
                    )

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
        if self._dep_graph is not None:
            if not nx.is_directed_acyclic_graph(self._dep_graph):
                sccs = [scc for scc in nx.strongly_connected_components(self._dep_graph) if len(scc) > 1]
                if self.coupling == 'jacobi':
                    warnings_list.append(
                        f"Algebraic feedback loop detected among simulators {sccs}. "
                        f"Jacobi coupling resolves this by introducing a 1-step macro delay."
                    )
                elif self.coupling == 'gauss-seidel':
                    warnings_list.append(
                        f"Algebraic feedback loop detected among simulators {sccs}. "
                        f"Gauss-Seidel without iterative relaxation cannot resolve cyclic feedback without delay. "
                        f"Consider coupling='iterative'."
                    )
                elif self.coupling == 'iterative':
                    for scc in sccs:
                        for sim_name in scc:
                            entry = self.simulator_dict.get(sim_name)
                            if entry and entry.adapter.save_state() is None:
                                warnings_list.append(
                                    f"Simulator '{sim_name}' in feedback loop does not support state rollback (save_state). "
                                    f"Iterative convergence is not guaranteed."
                                )

        is_valid = len(errors) == 0
        report = ValidationReport(is_valid=is_valid, errors=errors, warnings=warnings_list)
        if raise_on_error and not is_valid:
            raise ValidationError(str(report))
        return report

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
            if entry.model is not None and isinstance(entry.model, str):
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
            scale = getattr(conn, 'scale', 1.0)
            offset = getattr(conn, 'offset', 0.0)
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
            if from_unit:
                conn_dict['from_unit'] = from_unit
            if to_unit:
                conn_dict['to_unit'] = to_unit
            conns_list.append(conn_dict)

        return {
            'world': {
                'start_time': self.start_time,
                'stop_time': self.stop_time,
                't_macro': self.exchange,
                'coupling': self.coupling,
                'extrapolation': self.extrapolation,
                'max_iterations': self.max_iterations,
                'convergence_tolerance': self.convergence_tolerance,
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

    @classmethod
    def from_dict(cls, data, base_dir=None):
        """Construct a world instance from a dictionary configuration."""
        w_cfg = data.get('world', {})
        w = cls(
            start_time=w_cfg.get('start_time', 0),
            stop_time=w_cfg.get('stop_time', 1000),
            t_macro=w_cfg.get('t_macro', 60),
            coupling=w_cfg.get('coupling', 'jacobi'),
            extrapolation=w_cfg.get('extrapolation', 'zero-order'),
            res_filename=w_cfg.get('res_filename', 'es_res.h5'),
            max_iterations=w_cfg.get('max_iterations', 10),
            convergence_tolerance=float(w_cfg.get('convergence_tolerance', 1e-6)),
            power_unit=w_cfg.get('power_unit', 'kW'),
            logging=w_cfg.get('logging', False),
        )

        for sim_cfg in data.get('simulators', []):
            s_name = sim_cfg['name']
            s_type = sim_cfg['type']
            s_model = sim_cfg.get('model') if sim_cfg.get('model') is not None else sim_cfg.get('location', '')
            if base_dir and isinstance(s_model, str) and s_model and not os.path.isabs(s_model):
                candidate_path = os.path.normpath(os.path.join(base_dir, s_model))
                if os.path.exists(candidate_path) or any(s_model.lower().endswith(ext) for ext in (
                    '.fmu', '.p', '.pkl', '.json', '.csv', '.dss', '.m', '.py', '.nc'
                )):
                    s_model = candidate_path

            s_inputs = sim_cfg.get('inputs', [])
            s_outputs = sim_cfg.get('outputs', [])
            s_step = sim_cfg.get('step_size', 1.0)
            mode = sim_cfg.get('mode') or sim_cfg.get('pf_mode') or 'ac'
            power_unit = sim_cfg.get('power_unit')
            units = sim_cfg.get('units')
            kwargs = dict(sim_cfg.get('options', {}))

            w.add_simulator(
                sim_type=s_type,
                sim_name=s_name,
                model=s_model,
                inputs=s_inputs,
                outputs=s_outputs,
                step_size=s_step,
                mode=mode,
                power_unit=power_unit,
                units=units,
                **kwargs
            )

        conns = data.get('connections', [])
        if conns:
            w.add_connections(conns)

        return w

    @classmethod
    def from_yaml(cls, path_or_str):
        """Load and instantiate a world from a YAML file path or YAML formatted string."""
        if os.path.isfile(path_or_str):
            base_dir = os.path.dirname(os.path.abspath(path_or_str))
            with open(path_or_str, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            base_dir = None
            data = yaml.safe_load(path_or_str)

        return cls.from_dict(data, base_dir=base_dir)


# PEP 8 alias
World = world
