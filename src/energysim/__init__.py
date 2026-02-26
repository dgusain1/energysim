from .csv_adapter import csv_simulator
from .signalAdapter import signal_adapter
from .ParameterSweep import Sweep  # noqa: F401  (public API)
from .utils import convert_hdf_to_dict, record_data, create_results_recorder
from .dashboard import generate_dashboard
from .base import (  # noqa: F401  (public API)
    SimulatorAdapter, SimEntry,
    EnergysimError, SimulatorVariableError,
    SimulatorElementNotFoundError, ConnectionError,
)
import sys
import math
import warnings
from collections import deque
import numpy as np
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
                 max_iterations=10, convergence_tolerance=1e-6):
        self.start_time = start_time
        self.stop_time = stop_time
        self.logging = logging
        self.exchange = t_macro
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

    def add_signal(self, sim_name, signal, step_size=1):
        if sim_name not in self.simulator_dict.keys():
            signal_obj = signal_adapter(signal_name=sim_name, signal=signal)
            self.simulator_dict[sim_name] = SimEntry(
                sim_type='signal', adapter=signal_obj, step_size=step_size,
                outputs=['y'], inputs=[]
            )

    def add_simulator(self, sim_type='', sim_name='', sim_loc='', inputs=None, outputs=None, step_size=1, **kwargs):
        """
        Method to add simulator to ``world()`` object. ``add_simulator()`` takes the following arguments as inputs :

        1. ``sim_type`` : 'fmu', 'powerflow', 'csv', 'external', 'powerfactory', 'matlab'
        2. ``sim_name`` : It is essential that you assign the simulator a unique name.
        3. ``sim_loc`` : Use a raw string address of simulator location.
        4. ``outputs`` : specify the outputs that need to be recorded during simulation from the simulator.
        5. ``inputs`` : specify the inputs for the simulator.
        6. ``step_size`` : step size for simulator (set to 1e-3 by default).
        7. ``kwargs`` : Multiple options for FMUs and powerflow networks :
            - For FMUs, users can specify the following :
                - variable : boolean argument for variable stepping in simulation. Default is False.
                - validate : boolean argument for validating FMUs. Default is False.
            - For powerflow network, users can specify :
                - pf : 'pf' or 'opf' to toggle powerflow or
                  optimal power flow when available. Currently
                  only available for pandapower.
            - For PowerFactory, users can specify :
                - pf : 'ldf', 'shc', or 'rms' for load flow, short
                  circuit, or RMS simulation.
                - pf_path : path to the PowerFactory Python directory.
            - For MATLAB/Octave, users can specify :
                - engine : 'auto' (default), 'matlab', or 'octave'.
        """
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []

        if sim_type.lower() == 'fmu':
            solver_name = 'Cvode'
            if 'variable' in kwargs.keys():
                variable = kwargs['variable']
            else:
                variable = False
            if 'validate' in kwargs.keys():
                validate = kwargs['validate']
            else:
                validate = False
            self.add_fmu(sim_name, sim_loc, inputs=inputs, outputs=outputs, step_size=step_size,
                         solver_name=solver_name, variable=variable, validate=validate)

        elif sim_type.lower() == 'powerflow':
            if 'pf' in kwargs.keys():
                pf = kwargs['pf']
            else:
                pf = 'pf'

            self.add_powerflow(sim_name, sim_loc, inputs=inputs, outputs=outputs, step_size=step_size, pf=pf)

        elif sim_type.lower() == 'csv':
            if 'delimiter' in kwargs.keys():
                delimiter = kwargs['delimiter']
            else:
                delimiter = ','
            self.add_csv(
                csv_name=sim_name,
                csv_location=sim_loc,
                step_size=step_size,
                outputs=outputs,
                delimiter=delimiter)

        elif sim_type.lower() == 'external':
            self.add_external_simulator(
                sim_name=sim_name,
                sim_loc=sim_loc,
                inputs=inputs,
                outputs=outputs,
                step_size=step_size)

        elif sim_type.lower() == 'powerfactory':
            pf_mode = kwargs.get('pf', 'ldf')
            pf_path = kwargs.get('pf_path', None)
            self.add_powerfactory(
                sim_name=sim_name,
                sim_loc=sim_loc,
                inputs=inputs,
                outputs=outputs,
                step_size=step_size,
                pf=pf_mode,
                pf_path=pf_path)

        elif sim_type.lower() == 'matlab':
            engine = kwargs.get('engine', 'auto')
            self.add_matlab(
                sim_name=sim_name,
                sim_loc=sim_loc,
                inputs=inputs,
                outputs=outputs,
                step_size=step_size,
                engine=engine)

        else:
            print(
                f"Simulator type {sim_type} not recognized."
                f" Possible sim_type are 'external', 'fmu',"
                f" 'powerflow', 'csv', 'powerfactory', 'matlab'."
            )
            print(f"Simulator {sim_name} not added")

    def add_external_simulator(self, sim_name, sim_loc, inputs, outputs, step_size=900):
        sys.path.append(sim_loc)
        tmp = importlib.import_module(sim_name)
        external_simulator = tmp.external_simulator
        # check that the sim_name is unique
        if sim_name not in self.simulator_dict.keys():
            simulator = external_simulator(inputs=inputs, outputs=outputs)
            simulator.step_size = step_size
            self.simulator_dict[sim_name] = SimEntry(
                sim_type='external', adapter=simulator, step_size=step_size,
                outputs=outputs, inputs=inputs
            )
            if self.logging:
                print("Added external simulator '%s' to the world" % (sim_name))
        else:
            print(f"Please specify a unique name for simulator. {sim_name} already exists.")

    def add_powerfactory(self, sim_name, sim_loc, inputs=None,
                         outputs=None, step_size=900, pf='ldf',
                         pf_path=None):
        """Add a DIgSILENT PowerFactory model to the co-simulation.

        Parameters
        ----------
        sim_name : str
            Unique name AND PowerFactory project name.
        sim_loc : str
            Not used (PF projects are opened by name).  Pass ``''``.
        inputs : list[str]
            Input variables as ``'ElementName.attribute'``.
        outputs : list[str]
            Output variables to record.
        step_size : int
            Step size in seconds.
        pf : str
            Calculation type: ``'ldf'``, ``'shc'``, or ``'rms'``.
        pf_path : str or None
            Path to the PowerFactory Python directory.
        """
        from .pfAdapter import pf_adapter  # lazy import
        adapter = pf_adapter(
            project_name=sim_name,
            pf_loc=sim_loc,
            inputs=inputs or [],
            outputs=outputs or [],
            pf=pf,
            pf_path=pf_path,
        )
        adapter.step_size = step_size
        if sim_name not in self.simulator_dict:
            self.simulator_dict[sim_name] = SimEntry(
                sim_type='powerfactory', adapter=adapter,
                step_size=step_size, outputs=outputs or [],
                inputs=inputs or [], pf_mode=pf,
            )
            if self.logging:
                print(f"Added PowerFactory simulator '{sim_name}' to the world")
        else:
            print(f"Please specify a unique name for simulator. {sim_name} already exists.")

    def add_matlab(self, sim_name, sim_loc, inputs=None, outputs=None,
                   step_size=60, engine='auto'):
        """Add a MATLAB / Octave .m function to the co-simulation.

        Parameters
        ----------
        sim_name : str
            Unique name.  Must also be the function name (matches
            the ``.m`` filename without extension).
        sim_loc : str
            Directory containing the ``.m`` file.
        inputs : list[str]
            Input variable names (order must match function args).
        outputs : list[str]
            Output variable names (order must match function returns).
        step_size : int
            Step size in seconds.
        engine : str
            ``'auto'``, ``'matlab'``, or ``'octave'``.
        """
        from .matlabAdapter import matlab_adapter  # lazy import
        adapter = matlab_adapter(
            func_name=sim_name,
            script_loc=sim_loc,
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
            )
            if self.logging:
                print(f"Added MATLAB/Octave simulator '{sim_name}' to the world")
        else:
            print(f"Please specify a unique name for simulator. {sim_name} already exists.")

    def add_powerflow(self, network_name, net_loc, inputs=None, outputs=None, pf='pf', step_size=900, **kwargs):
        assert network_name is not None, "No name specified for power flow model. Name must be specified."
        assert net_loc is not None, "No location specified for power flow model. Can't read without."
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
        from .pypsaAdapter import pypsa_adapter  # lazy: pypsa has heavy optional deps
        from .ppAdapter import pp_adapter         # lazy: pandapower has heavy optional deps
        try:
            network = pypsa_adapter(network_name, net_loc, inputs=inputs, outputs=outputs,
                                    logger_level=kwargs.get('logger', 'DEBUG'))
        except Exception as e:
            if self.logging:
                print(f"pypsa_adapter failed ({e}), falling back to pp_adapter.")
            network = pp_adapter(network_name, net_loc, inputs=inputs, outputs=outputs, pf=pf)
        network.step_size = step_size
        if network_name not in self.simulator_dict.keys():
            self.simulator_dict[network_name] = SimEntry(
                sim_type='pf', adapter=network, step_size=step_size,
                outputs=outputs, inputs=inputs, pf_mode=pf
            )
            if self.logging:
                print("Added powerflow simulator '%s' to the world" % (network_name))
        else:
            print(f"Please specify a unique name for simulator. {network_name} already exists.")

    def add_fmu(
            self,
            fmu_name,
            fmu_loc,
            step_size,
            inputs=None,
            outputs=None,
            exist=False,
            solver_name='Cvode',
            variable=False,
            **kwargs):
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
        m_desc = read_model_description(fmu_loc, validate=validate)
        fmi_type = 'CoSimulation' if m_desc.coSimulation is not None else 'ModelExchange'
        if fmi_type == 'CoSimulation':
            fmu_temp = FmuCsAdapter(fmu_loc,
                                    instanceName=fmu_name,
                                    step_size=step_size,
                                    inputs=inputs,
                                    outputs=outputs,
                                    exist=exist,
                                    validate=validate)
            fmu_temp.variable_step = variable
            fmu_temp.step_size = step_size
        elif fmi_type == 'ModelExchange':
            fmu_temp = FmuMeAdapter(fmu_loc,
                                    instanceName=fmu_name,
                                    step_size=step_size,
                                    inputs=inputs,
                                    outputs=outputs,
                                    solver_name=solver_name,
                                    validate=validate)

        if fmu_name not in self.simulator_dict.keys():
            self.simulator_dict[fmu_name] = SimEntry(
                sim_type='fmu', adapter=fmu_temp, step_size=step_size,
                outputs=outputs, inputs=inputs, variable_step=variable
            )
            if self.logging:
                print("Added FMU simulator '%s' to the world" % (fmu_name))
        else:
            print(f"Please specify a unique name for simulator. {fmu_name} already exists.")

    def add_csv(self, csv_name, csv_location, step_size, outputs, delimiter):
        csv_obj = csv_simulator(csv_name, csv_location, step_size, outputs, delimiter)
        csv_obj.step_size = step_size
        if csv_name not in self.simulator_dict.keys():
            self.simulator_dict[csv_name] = SimEntry(
                sim_type='csv', adapter=csv_obj, step_size=step_size,
                outputs=outputs, inputs=[]
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

        # Pre-parse connections once so exchange_variable_values() doesn't
        # re-split strings on every time step.
        self._parsed_connections = []
        for output_key, input_key in connections.items():
            # --- parse output side ---
            if isinstance(output_key, str):
                ou_sim, ou_var = self._split_var(output_key)
                parsed_output = (ou_sim, ou_var, output_key)
            elif isinstance(output_key, tuple):
                parsed_output = tuple(
                    (self._split_var(item)[0], self._split_var(item)[1], item)
                    for item in output_key
                )
            # --- parse input side ---
            if isinstance(input_key, str):
                in_sim, in_var = self._split_var(input_key)
                parsed_input = (in_sim, in_var)
            elif isinstance(input_key, tuple):
                parsed_input = [
                    (self._split_var(item)[0], self._split_var(item)[1])
                    for item in input_key
                ]

            self._parsed_connections.append((parsed_output, parsed_input))

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

        for output_, input_ in self._parsed_connections:
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

        for key, value in self.simulator_connections.items():
            if isinstance(key, str) and isinstance(value, str):
                n1 = key.split('.')[0]
                n2 = value.split('.')[0]
                self._dep_graph.add_edge(n1, n2)
            elif isinstance(key, str) and isinstance(value, tuple):
                n1 = key.split('.')[0]
                for v in value:
                    n2 = v.split('.')[0]
                    self._dep_graph.add_edge(n1, n2)
            elif isinstance(key, tuple) and isinstance(value, str):
                n2 = value.split('.')[0]
                for k in key:
                    n1 = k.split('.')[0]
                    self._dep_graph.add_edge(n1, n2)

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

    def simulate(self, pbar=True, record_all=True):
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
        for output_, input_ in self._parsed_connections:
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
        Supports zero-order hold (default) and linear extrapolation.
        """
        for output_, input_ in self._parsed_connections:
            # --- gather output value(s) ---
            tmp_var = self._gather_output_value(output_, t)

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

    def plot_results(self, xlab='Time(s)', ylab='', y=[], scientific=False):
        import matplotlib.pyplot as plt
        locol = [x for x in self.list_of_columns if x != 'time']
        plt.figure(2)
        if self.interpolate_results:
            time_vector = self.results_dataframe.time
            if len(y) == 0:
                for item in locol:
                    plt.plot(time_vector, self.results_dataframe.loc[:, item], label=item)
            else:
                for item in y:
                    try:
                        plt.plot(time_vector, self.results_dataframe.loc[:, item], label=item)
                    except BaseException:
                        print("Could not find the variable %s in results dataframe. Make sure you have \
                              specified the variables as 'fmu.variable'. Skipping." % (item))
                        continue
            plt.ticklabel_format(useOffset=scientific)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.legend()
            plt.show()
        else:
            print(
                'Results can only be plotted natively if'
                ' interpolate_results option is set to True'
                ' in World() options.'
            )
