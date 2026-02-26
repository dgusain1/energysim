Architecture
============

This page describes the internal architecture of ``energysim`` (v2.2+).

Overview
--------

``energysim`` follows a **coordinator--adapter** pattern:

- The **coordinator** (``world`` class in ``__init__.py``) manages the
  simulation loop, time progression, variable exchange, dependency
  ordering, coupling strategy, and results collection.
- Each simulator is wrapped in an **adapter** that conforms to the
  ``SimulatorAdapter`` abstract base class defined in ``energysim.base``.

The coordinator never inspects the simulator type directly.  All
interaction goes through the uniform ``SimulatorAdapter`` interface.


Exception hierarchy
-------------------

``energysim`` uses structured exceptions (defined in ``energysim.base``)
instead of ``sys.exit()`` calls:

- ``EnergysimError(Exception)`` — base class for all energysim errors.
- ``SimulatorVariableError(EnergysimError)`` — invalid variable name or
  type mismatch when setting/getting adapter variables.
- ``SimulatorElementNotFoundError(EnergysimError)`` — a named element
  was not found in a pandapower or PyPSA network.
- ``ConnectionError(EnergysimError)`` — malformed or invalid connection
  specification (missing simulator, missing dot separator, etc.).

Import them as::

    from energysim.base import (
        EnergysimError,
        SimulatorVariableError,
        SimulatorElementNotFoundError,
        ConnectionError,
    )


SimulatorAdapter ABC
--------------------

Defined in ``energysim.base``, the abstract base class provides:

**Abstract methods** (must be implemented):

- ``step(time)`` — advance the model by one micro-step.
- ``get_value(parameters, time)`` — return a list of output values.
- ``set_value(parameters, values)`` — set input values.

**Default implementations** (override when needed):

- ``init()`` — no-op; override to initialise the model.
- ``cleanup()`` — no-op; override to release resources.
- ``advance(start_time, stop_time)`` — loops ``step()`` from
  *start_time* to *stop_time* at ``step_size`` intervals using
  **index-based** time computation (``t = start_time + i * step_size``)
  to avoid floating-point drift.  Complex adapters (e.g. FMU Model
  Exchange with CVode) override this to hand the full interval to their
  internal solver.
- ``set_parameters(params)`` — delegates to ``set_value()``.  FMU
  adapters override this to call ``apply_start_values()`` instead.
- ``advance_with_recording(start_time, stop_time, outputs)`` — advance
  the model *and* collect intermediate results at each micro-step.
  The default implementation calls ``advance()`` followed by
  ``get_value()``.  The FMU Model Exchange adapter overrides this to
  use the ODE solver's internal time points.  Used when
  ``record_all=True``.
- ``save_state() -> Any`` — snapshot the model state.  Used by
  iterative coupling to roll back if convergence is not achieved.
  Default returns ``None``.  The FMU CS adapter implements this via
  ``getFMUstate()`` for FMI 2.0 FMUs that support
  ``canGetAndSetFMUstate``.
- ``restore_state(state)`` — restore a previously saved state.  Default
  is a no-op.
- ``get_available_variables() -> Dict[str, List[str]]`` — return known
  variable names grouped by category.  Used by connection validation to
  warn about misspelled variable names.  Default returns an empty dict.
  Implemented in ``ppAdapter`` (element.variable names from
  gen/load/sgen/bus tables), ``csAdapter`` (all VRS keys), and
  ``csv_adapter`` (column names).


Built-in adapters
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Adapter
     - ``sim_type``
     - Description
   * - ``FmuCsAdapter``
     - ``'fmu'``
     - FMI Co-Simulation (1.0 & 2.0)
   * - ``FmuMeAdapter``
     - ``'fmu'``
     - FMI Model Exchange with ODE solver
   * - ``pp_adapter``
     - ``'powerflow'``
     - Pandapower network
   * - ``pypsa_adapter``
     - ``'powerflow'``
     - PyPSA network (experimental)
   * - ``csv_simulator``
     - ``'csv'``
     - Time-indexed CSV data (read-only)
   * - ``signal_adapter``
     - (via add_signal)
     - User-defined Python function (read-only)
   * - ``pyScriptAdapter``
     - ``'script'``
     - Standalone Python script
   * - ``external_simulator``
     - ``'external'``
     - User-defined class
   * - ``pf_adapter``
     - ``'powerfactory'``
     - DIgSILENT PowerFactory (ldf / shc / rms)
   * - ``matlab_adapter``
     - ``'matlab'``
     - MATLAB / GNU Octave ``.m`` functions

.. note::

   CSV and signal adapters are **read-only**.  If a connection writes to
   them, ``set_value()`` emits a ``UserWarning`` and discards the value
   instead of silently ignoring it.


SimEntry dataclass
------------------

Each simulator registered with ``world`` is stored as a ``SimEntry``::

    @dataclass
    class SimEntry:
        sim_type: str                    # 'fmu', 'pf', 'csv', 'signal', ...
        adapter: SimulatorAdapter        # the adapter instance
        step_size: float                 # micro-step size
        outputs: List[str]               # variable names to record
        inputs: List[str] = field(...)   # input variable names (from connections)
        variable_step: bool = False      # variable-step integration (CS FMU only)
        pf_mode: Optional[str] = None    # 'pf'/'opf'/... for powerflow

Access adapters via::

    entry = my_world.simulator_dict["grid"]
    entry.adapter          # the pp_adapter instance
    entry.adapter.network  # the pandapower network object
    entry.step_size        # 300
    entry.sim_type         # 'pf'
    entry.inputs           # ['PV.p_mw', 'Load.p_mw']
    entry.outputs          # ['Bus1.vm_pu', 'grid.p_mw']


Connection validation
---------------------

When ``add_connections()`` is called, ``energysim`` performs three
validation checks before parsing:

1. **Simulator existence** — every simulator name referenced in the
   connection strings must already be registered in ``simulator_dict``.
   Missing names raise ``ConnectionError``.

2. **Variable name validity** — if the adapter implements
   ``get_available_variables()``, variable names are checked against the
   known set.  Unrecognised names produce a ``UserWarning`` (not an
   error, since some adapters accept arbitrary variable names).

3. **Read-only target detection** — connections that write to CSV or
   signal simulators are warned about, since those adapters discard
   incoming values.


Dependency graph
----------------

After parsing connections, ``energysim`` builds a **directed graph**
(``nx.DiGraph``) where:

- **Nodes** are simulator names.
- **Edges** represent data flow (``source → destination``).

This graph is used for:

- **Topological ordering** — determining the execution order of
  simulators so that upstream simulators are stepped before downstream
  ones (when possible).
- **Algebraic loop detection** — identifying cycles via
  strongly-connected-component (SCC) analysis.

If the graph is a DAG (no cycles), simulators are sorted in topological
order.  If cycles exist, ``energysim`` performs SCC condensation:

1. Each SCC (group of mutually dependent simulators) is treated as a
   single node in the condensed DAG.
2. The condensed DAG is topologically sorted.
3. A ``UserWarning`` is emitted listing the simulators involved in each
   algebraic loop.
4. Within each SCC, Jacobi coupling is used (regardless of the global
   coupling mode).


Coupling modes
--------------

``energysim`` supports three coupling strategies, selected via the
``coupling`` parameter of ``world()``.

Jacobi (default)
^^^^^^^^^^^^^^^^

::

    my_world = world(coupling='jacobi', ...)

1. **Exchange all** variables between simulators based on the previous
   step's outputs.
2. **Step all** simulators from *t* to *t + t_macro* (in execution
   order).

This introduces a **one-step delay** in feedback loops — each
simulator sees the *previous* macro-step's outputs from its
dependencies.  This is simple, robust, and suitable for loosely coupled
systems.


Gauss-Seidel
^^^^^^^^^^^^^

::

    my_world = world(coupling='gauss-seidel', ...)

1. For each simulator in **topological order**:

   a. Exchange inputs from already-stepped simulators.
   b. Step the simulator from *t* to *t + t_macro*.

Downstream simulators immediately see the *current* macro-step's
outputs from upstream simulators.  This reduces the coupling delay for
feed-forward chains.  Within algebraic loops (SCCs), Jacobi coupling
is used as a fallback.


Iterative (fixed-point)
^^^^^^^^^^^^^^^^^^^^^^^

::

    my_world = world(
        coupling='iterative',
        max_iterations=10,
        convergence_tolerance=1e-6,
    )

1. **Save state** of all simulators (via ``save_state()``).
2. Perform a Gauss-Seidel step.
3. **Check convergence** — compare current outputs to previous
   iteration.  If the maximum relative change is below
   ``convergence_tolerance``, accept the step.
4. If not converged and iterations remain, **restore state** (via
   ``restore_state()``) and repeat from step 2.
5. If ``max_iterations`` is reached without convergence, accept the
   last result and emit a ``UserWarning``.

This is the most accurate mode for tightly coupled systems with
algebraic loops, but requires adapters that support state save/restore.
FMU CS adapters implement this via ``getFMUstate()``/``setFMUstate()``
for FMI 2.0 FMUs with ``canGetAndSetFMUstate=true``.


Extrapolation
-------------

Exchanged variable values can be extrapolated between macro time-steps:

- ``extrapolation='zero-order'`` (default) — hold the last exchanged
  value constant until the next exchange.  This is the standard
  zero-order hold.

- ``extrapolation='linear'`` — linearly extrapolate from the two most
  recent exchanged values.  A ``deque(maxlen=2)`` per variable stores
  ``(time, value)`` pairs.  Until two data points are available,
  zero-order hold is used as a fallback.


Simulation loop
---------------

The simulation loop in ``world.simulate()`` works as follows:

1. **Initialise** all adapters by calling ``adapter.init()``.
2. Compute the **execution order** from the dependency graph.
3. For each macro time-step, dispatch to the selected coupling method:

   - ``_step_jacobi()`` — exchange all, then step all.
   - ``_step_gauss_seidel()`` — step in order with immediate exchange.
   - ``_step_iterative()`` — iterate until convergence.

4. **Record outputs** to an HDF5 file after each macro step.
5. **Cleanup** all adapters by calling ``adapter.cleanup()``.

When ``record_all=True``, the coordinator calls
``adapter.advance_with_recording()`` instead of ``adapter.advance()``
to capture intermediate results at every micro-step.  This is
particularly important for FMU Model Exchange, where the ODE solver
may use adaptive time steps.


Lazy imports
------------

Simulator-specific dependencies (FMPy, pandapower, PyPSA, oct2py) are
imported lazily — only when the first simulator of that type is added.
This means core ``energysim`` works with just:

- NumPy
- Pandas
- Matplotlib
- NetworkX (core dependency since v2.2)
- tqdm
- PyTables

NetworkX was previously optional (only for plotting).  It is now a core
dependency because it is used for dependency graph construction and
topological ordering.


Pre-parsed connections
----------------------

Connections are parsed once in ``add_connections()`` and stored as a
list of ``(src_sim, src_var, dst_sim, dst_var)`` tuples.  This avoids
repeated string splitting during the simulation loop.

Malformed connection strings (missing ``.`` separator) now raise
``ConnectionError`` instead of causing cryptic index errors later.


Data flow summary
-----------------

::

    User code
      │
      ▼
    world()
      ├── add_simulator() / add_signal() / add_fmu() / add_powerflow()
      │     └── Creates SimEntry + adapter instance
      ├── add_connections()
      │     ├── _validate_connections()   → errors / warnings
      │     ├── _build_dependency_graph() → nx.DiGraph
      │     └── _compute_execution_order()→ topological sort / SCC
      ├── options()
      │     └── Sets init values, modify_signal, record_all, etc.
      └── simulate()
            ├── init all adapters
            ├── for each macro step:
            │     ├── _step_jacobi / _step_gauss_seidel / _step_iterative
            │     │     ├── exchange variables (with extrapolation)
            │     │     └── adapter.advance() or advance_with_recording()
            │     └── record to HDF5
            └── cleanup all adapters
