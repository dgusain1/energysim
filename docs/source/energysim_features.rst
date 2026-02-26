energysim features
==================

This page documents features beyond the basic workflow described on the
main page.


.. _coupling-modes:

Coupling modes
--------------

``energysim`` supports three coupling strategies, selected via the
``coupling`` parameter when creating the ``world`` object.


Jacobi coupling (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    my_world = world(coupling='jacobi', ...)

All variables are exchanged first, then all simulators are stepped.
This introduces a **one-step delay** in feedback loops — each simulator
sees the *previous* macro-step's outputs from its dependencies.

Best for: loosely coupled systems, parallel-friendly setups.


Gauss-Seidel coupling
^^^^^^^^^^^^^^^^^^^^^

::

    my_world = world(coupling='gauss-seidel', ...)

Simulators are stepped one at a time in **topological order**.  After
each simulator steps, its outputs are immediately available to
downstream simulators.  This eliminates the coupling delay for
feed-forward signal chains.

Within algebraic loops (cyclic dependencies), Gauss-Seidel falls back
to Jacobi coupling for the involved simulators.

Best for: predominantly feed-forward topologies, chains of sequential
models.


Iterative coupling
^^^^^^^^^^^^^^^^^^

::

    my_world = world(
        coupling='iterative',
        max_iterations=10,
        convergence_tolerance=1e-6,
    )

Each macro step is repeated (iterated) until the exchanged variables
converge:

1. Save the state of all simulators.
2. Perform a Gauss-Seidel step.
3. Compare current outputs to the previous iteration.
4. If converged (max relative change < ``convergence_tolerance``),
   accept the step.
5. Otherwise, restore the saved state and repeat (up to
   ``max_iterations``).

.. note::

   Iterative coupling requires adapters that support ``save_state()``
   and ``restore_state()``.  FMU Co-Simulation adapters implement this
   via ``getFMUstate()`` / ``setFMUstate()`` for FMI 2.0 FMUs with
   ``canGetAndSetFMUstate=true``.  External simulators can implement
   these methods by saving and restoring their internal state variables.

Best for: tightly coupled systems with algebraic loops (e.g. a
controller and plant with mutual feedback).


Extrapolation
-------------

The ``extrapolation`` parameter controls how exchanged values are
estimated between macro time-steps:

- ``'zero-order'`` (default) — hold the last exchanged value constant
  (zero-order hold).
- ``'linear'`` — linearly extrapolate from the two most recent values.

Example::

    my_world = world(
        coupling='gauss-seidel',
        extrapolation='linear',
        t_macro=900,
    )

Linear extrapolation can improve accuracy when variables change smoothly
between exchange points, but may cause instability if variables are noisy
or discontinuous.


Connection validation
---------------------

``add_connections()`` automatically validates all connections:

- **Simulator existence** — raises ``ConnectionError`` if a simulator
  name in the connection string is not registered.
- **Variable name checking** — warns if a variable name is not found
  in the adapter's known variables (via ``get_available_variables()``).
- **Read-only targets** — warns if a connection writes to a CSV or
  signal simulator, since those adapters discard incoming values.

Example of a validation error::

    my_world.add_connections({'sim1.y': 'nonexistent_sim.x'})
    # → ConnectionError: Simulator 'nonexistent_sim' not found

Example of a validation warning::

    my_world.add_connections({'signal.y': 'csv_data.temperature'})
    # → UserWarning: Simulator 'csv_data' is read-only (type='csv').
    #   Connection to 'temperature' will be silently ignored.


Dependency graph & algebraic loops
-----------------------------------

After connections are added, ``energysim`` builds a directed dependency
graph using NetworkX.  Simulators are automatically sorted in
**topological execution order** so that upstream simulators are stepped
before downstream ones.

If the dependency graph contains **cycles** (algebraic loops),
``energysim``:

1. Warns about the loop, listing the involved simulators.
2. Uses **strongly-connected-component (SCC) condensation** to find
   a safe execution order.
3. Within each SCC, uses Jacobi coupling (regardless of the global
   coupling mode).

You can inspect the computed execution order::

    my_world.add_connections(connections)
    # Prints: "Execution order: ['weather', 'pv', 'battery', 'grid']"


Variable introspection
----------------------

Some adapters support introspection of their available variables via
``get_available_variables()``.  This is used during connection
validation and can also be called directly::

    entry = my_world.simulator_dict["grid"]
    print(entry.adapter.get_available_variables())
    # {'inputs': ['Gen1.p_mw', 'Load1.p_mw', ...],
    #  'outputs': ['Bus1.vm_pu', 'Bus1.va_degree', ...]}

Currently implemented in:

- **ppAdapter** — returns element.variable names from gen, load, sgen,
  bus, and ext_grid tables.
- **csAdapter** — returns all variable names from the FMU's model
  description.
- **csv_adapter** — returns column names from the CSV file.


Adding signals
--------------

The ``add_signal()`` method attaches user-defined time-varying (or
constant) signals to the co-simulation.  This is useful when a simulator
input needs a constant set-point or a mathematical function of time::

    def my_signal(time):
        return [1]

    my_world.add_signal(
        sim_name='constant_signal',
        signal=my_signal,
        step_size=1)

The return value **must** be a single-element list.  More complex
signals are possible::

    import numpy as np

    def sine_signal(time):
        return [np.sin(2 * np.pi * time / 3600)]

    my_world.add_signal(sim_name='sine', signal=sine_signal, step_size=1)

In the connections dictionary::

    connections = {'constant_signal.y': 'sim1.input_variable1'}
    my_world.add_connections(connections)

.. note::

   Signal simulators are **read-only**.  If a connection writes to a
   signal simulator, ``set_value()`` emits a ``UserWarning`` and
   discards the value.


Modify signals before exchange
------------------------------

Sometimes an output needs unit conversion before being passed to another
simulator.  ``energysim`` supports this via the ``modify_signal`` option:

1. **Multiply by a constant**: ``'sim1.var1': [x]`` multiplies by *x*.
2. **Multiply and offset**: ``'sim1.var1': [x1, x2]`` computes
   ``value * x1 + x2``.

Example — convert Watts to MW::

    modifications = {'chp.e_power': [1 / 1e6]}
    options = {'modify_signal': modifications}
    my_world.options(options)


Optimal Power Flow
------------------

Pandapower networks can run OPF instead of standard power flow.
Specify ``pf='opf'`` (or ``'dcopf'``) when adding the simulator::

    my_world.add_simulator(
        sim_type='powerflow', sim_name='grid',
        sim_loc='network.p',
        inputs=['WF.p_mw'], outputs=['Bus1.vm_pu'],
        step_size=300, pf='opf')

For OPF, ensure the network has proper cost functions and that
controllable loads have ``controllable=True`` set.  This can be done
after the simulator is added::

    net = my_world.simulator_dict["grid"].adapter.network
    idx = net.load[net.load["name"] == "FlexLoad"].index
    net.load.loc[idx, "controllable"] = True


Validation of FMUs
------------------

By default, FMPy validates FMUs during initialisation.  To skip
validation (useful for faster startup or FMUs that fail validation but
simulate correctly)::

    my_world.add_simulator(
        sim_type='fmu', sim_name='plant',
        sim_loc='plant.fmu',
        inputs=['cmd'], outputs=['temp'],
        step_size=1, validate=False)


Recording all time-steps
------------------------

By default, ``energysim`` records output values once per macro
time-step.  To capture values at every micro-step, use
``record_all=True``::

    my_world.options({'record_all': True})

This calls ``adapter.advance_with_recording()`` instead of
``adapter.advance()``.  For most adapters, this records at every
``step_size`` interval.  For FMU Model Exchange, the ODE solver's
adaptive internal time points are used, providing higher-fidelity
traces without artificially constraining the solver.


Sensitivity analysis
--------------------

Parameter sweeps can be performed by updating the ``init`` option in a
loop::

    for v1, v2 in [(0.1, 0.2), (1, 2), (10, 20)]:
        inits = {
            'sim1': (['param_a'], [v1]),
            'sim2': (['param_b'], [v2]),
        }
        my_world.options({'init': inits})
        my_world.simulate(pbar=False)
        # extract and store results

A dedicated ``Sweep`` class is also available for automated parameter
sweeps with built-in plotting::

    from energysim import Sweep

    sweep = Sweep(my_world, sensitivity_info, kind='single')
    sweep.export_to_csv('sweep_results.csv')


PowerFactory integration
------------------------

DIgSILENT PowerFactory models (2020+) can be coupled into
co-simulations using the ``'powerfactory'`` sim type.  The adapter
supports load flow (``'ldf'``), short-circuit (``'shc'``), and RMS
transient (``'rms'``) calculation modes.

Variable naming convention:

- **Inputs** — use the PowerFactory *input* attribute prefix ``e:``,
  e.g. ``Load.plini`` (the adapter auto-prepends ``e:`` if missing).
- **Outputs** — use the *result* attribute prefix ``m:``,
  e.g. ``Bus1.m:u``, ``Line1.m:loading``.

The element name before the dot must match the object's ``loc_name``
in PowerFactory.

Example::

    my_world.add_simulator(
        sim_type='powerfactory',
        sim_name='IEEE13',
        sim_loc='IEEE13',
        inputs=['Load1.plini', 'Load2.plini'],
        outputs=['Bus1.m:u', 'Line1.m:loading'],
        step_size=900,
        pf='ldf',
        pf_path=r'C:\DIgSILENT\PowerFactory 2024\Python\3.11')

.. important::

   PowerFactory must be installed and licensed on the machine.  The
   ``pf_path`` argument is only needed if the PowerFactory Python
   directory is not already on ``sys.path``.


MATLAB / Octave integration
----------------------------

MATLAB ``.m`` **functions** (not scripts) can be integrated via the
``'matlab'`` sim type.  The adapter auto-detects whether to use the
MATLAB Engine for Python or GNU Octave (via ``oct2py``).

Every function must follow this contract::

    function [out1, out2] = my_model(in1, in2, time)

Rules:

- ``time`` is always the **last** input argument — ``energysim``
  appends it automatically.
- The number of declared inputs (minus ``time``) must match the
  ``inputs`` list.
- The number of declared outputs must match the ``outputs`` list.
- The file name (without ``.m``) must match the function name.

Use **persistent** variables to keep state across time-steps (see the
battery and building examples in ``examples/matlab_cosim/``).

Example::

    my_world.add_simulator(
        sim_type='matlab',
        sim_name='heatpump',
        sim_loc=r'models\heatpump.m',
        inputs=['P_electric', 'T_source', 'T_sink'],
        outputs=['Q_thermal', 'COP'],
        step_size=900,
        engine='auto')   # 'matlab', 'octave', or 'auto'

To force a particular engine, set ``engine='matlab'`` or
``engine='octave'``.

.. note::

   Install ``oct2py`` (``pip install oct2py``) and ensure GNU Octave is
   on PATH, or install the MATLAB Engine for Python from your MATLAB
   installation directory.


Interactive dashboard
---------------------

After simulation, ``results()`` can open an interactive HTML dashboard
in the browser::

    results = my_world.results(dashboard=True)

The dashboard features:

- **Variable browser** — tree view of all simulators and their
  variables.
- **Drag-and-drop** — drag variables onto subplots or click the
  **+** button.
- **Multiple subplots** — add, remove, and rename subplots.
- **Layout control** — switch between 1, 2, or 3 column layouts.
- **Auto-overview** — one-click plot of all variables grouped by
  simulator.
- **Export** — save the dashboard HTML file to share with others.

You can specify a custom path for the HTML file::

    results = my_world.results(
        dashboard=True,
        dashboard_path='my_dashboard.html')

.. note::

   The dashboard requires ``plotly``.  Install it with
   ``pip install plotly``.


System topology plot
--------------------

``energysim`` can visualise the co-simulation topology as a network
graph.  The ``plot()`` method reuses the internal dependency graph
(built during ``add_connections()``)::

    my_world.plot(plot_edge_labels=False, node_size=300, node_color='r')

If ``add_connections()`` has not been called yet, ``plot()`` will build
the graph on the fly from the current connections.


Error handling
--------------

All simulator errors now raise proper Python exceptions instead of
calling ``sys.exit()``:

+--------------------------------------+--------------------------------------------+
| Exception                            | When raised                                |
+======================================+============================================+
| ``ConnectionError``                  | Malformed connection string, missing       |
|                                      | simulator name                             |
+--------------------------------------+--------------------------------------------+
| ``SimulatorVariableError``           | Invalid variable name or type in a         |
|                                      | pandapower / FMU adapter                   |
+--------------------------------------+--------------------------------------------+
| ``SimulatorElementNotFoundError``    | Named element not found in a network       |
+--------------------------------------+--------------------------------------------+
| ``EnergysimError``                   | General energysim error (base class)       |
+--------------------------------------+--------------------------------------------+

These can be caught and handled in user code::

    from energysim.base import SimulatorVariableError

    try:
        my_world.simulate()
    except SimulatorVariableError as e:
        print(f"Variable error: {e}")


Accessing simulator internals
-----------------------------

After adding a simulator, its internal adapter object is accessible via
the ``simulator_dict``::

    # Access the pandapower network object
    net = my_world.simulator_dict["grid"].adapter.network

    # Access an FMU adapter
    fmu = my_world.simulator_dict["plant"].adapter

Each entry in ``simulator_dict`` is a ``SimEntry`` dataclass with:

- ``.adapter`` — the ``SimulatorAdapter`` instance.
- ``.outputs`` — list of output variable names.
- ``.inputs`` — list of input variable names (populated when connections
  are added).
- ``.step_size`` — micro time-step.
- ``.sim_type`` — ``'fmu'``, ``'pf'``, ``'csv'``, ``'signal'``, etc.
- ``.variable_step`` — whether variable stepping is enabled (FMU only).
- ``.pf_mode`` — power-flow mode (powerflow only).
