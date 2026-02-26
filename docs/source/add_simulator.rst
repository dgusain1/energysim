Adding simulators
=================

Simulators are added to the ``world`` object using ``add_simulator()``.
Seven simulator types are supported: **FMU**, **powerflow**, **CSV**,
**external**, **PowerFactory**, **MATLAB/Octave**, and **script**.
All of them are accessed through the same method.

Common parameters
-----------------

Every call to ``add_simulator()`` accepts these arguments:

- ``sim_type`` -- one of ``'fmu'``, ``'powerflow'``, ``'csv'``, ``'external'``,
  ``'powerfactory'``, ``'matlab'``, or ``'script'``.
- ``sim_name`` -- a unique name used to identify the simulator in connections and results.
- ``sim_loc`` -- path to the model or data file.
- ``inputs`` -- list of input variable names (used in the connections dictionary).
- ``outputs`` -- list of output variable names to record.
- ``step_size`` -- micro time-step in seconds.  This is the internal integration step for
  solvers.  For example, an FMU might use ``1`` while a power-flow network uses ``900``.

Example::

    my_world.add_simulator(
        sim_type='fmu', sim_name='plant',
        sim_loc='/path/to/model.fmu',
        inputs=['power_cmd'], outputs=['temperature'],
        step_size=1)


FMU simulators
--------------

FMUs conforming to FMI 1.0 or 2.0 are supported.  ``energysim``
auto-detects whether the FMU is **Co-Simulation** or **Model Exchange**
and loads the appropriate adapter.

Additional keyword arguments:

- ``validate`` *(bool, default True)* -- let FMPy validate the FMU before initialisation.
  Set to ``False`` for FMUs that fail validation but simulate correctly (common with
  OpenModelica exports).
- ``variable`` *(bool, default False)* -- enable variable-step integration
  (Co-Simulation only).

Example::

    my_world.add_simulator(
        sim_type='fmu', sim_name='electrolyser',
        sim_loc='electrolyser.fmu',
        inputs=['P_order', 'T_ambient'],
        outputs=['H2_production', 'efficiency'],
        step_size=1,
        validate=False)

.. note::
   FMPy is only imported when the first FMU is added, so ``energysim``
   works without FMPy installed as long as no FMU simulators are used.


Variable naming convention (FMU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FMU variables follow the Modelica dot-notation::

    sim_name.Component.SubComponent.Variable

For example: ``electrolyser.stack.voltage``.


Powerflow simulators
--------------------

Pandapower networks (saved as ``.p`` pickle files) can be added directly.
PyPSA networks are also supported experimentally.

Additional keyword arguments:

- ``pf`` *(str, default 'pf')* -- power-flow mode.  One of:

  - ``'pf'`` -- AC power flow
  - ``'dcpf'`` -- DC power flow
  - ``'opf'`` -- optimal power flow
  - ``'dcopf'`` -- DC optimal power flow

Example::

    my_world.add_simulator(
        sim_type='powerflow', sim_name='grid',
        sim_loc='network.p',
        inputs=['WF.p_mw', 'Load1.p_mw'],
        outputs=['Bus 0.vm_pu', 'WF.p_mw', 'Load1.p_mw'],
        step_size=300,
        pf='opf')

.. important::
   All load, generator, sgen, bus, and external-grid elements in the
   pandapower network **must have names**.  ``energysim`` matches
   variables by element name, not by index.  If a named element is not
   found, a ``SimulatorElementNotFoundError`` is raised.  If a variable
   attribute is invalid, a ``SimulatorVariableError`` is raised.

The pandapower adapter implements ``get_available_variables()`` for
connection validation — see :doc:`energysim_features`.


Variable naming convention (powerflow)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Variables are specified as ``ElementName.variable``:

**Inputs** (values that can be set):

- Generators: ``Gen1.p_mw``, ``Gen1.q_mvar``
- Loads: ``Load1.p_mw``, ``Load1.q_mvar``
- Sgens: ``PV.p_mw``, ``WF.p_mw``
- Cost: ``Load1.cp1_eur_per_mw`` (for OPF)
- Controllability bounds: ``Load1.max_p_mw``, ``Load1.min_p_mw``

**Outputs** (values that can be recorded):

- Bus voltages: ``Bus1.vm_pu``, ``Bus1.va_degree``
- Generator results: ``Gen1.p_mw``, ``Gen1.q_mvar``
- Load results: ``Load1.p_mw``, ``Load1.q_mvar``
- Sgen results: ``PV.p_mw``
- External grid: ``grid.p_mw``, ``grid.q_mvar``

.. note::
   For OPF, loads that should be dispatchable must have
   ``controllable=True`` set in the pandapower network.  You can do
   this after adding the simulator::

       net = my_world.simulator_dict["grid"].adapter.network
       idx = net.load[net.load["name"] == "FlexLoad"].index
       net.load.loc[idx, "controllable"] = True


CSV simulators
--------------

CSV files provide time-indexed data profiles (e.g. weather, demand).

Additional keyword arguments:

- ``delimiter`` *(str, default ',')* -- column delimiter.

Requirements:

- The CSV **must** have a ``time`` column.
- Time values must be at fixed intervals.

Example::

    my_world.add_simulator(
        sim_type='csv', sim_name='profiles',
        sim_loc='data.csv',
        step_size=900)

Variables are accessed by column name: ``profiles.wind_power``,
``profiles.temperature``, etc.

.. note::
   CSV simulators are **read-only**.  If a connection writes to a CSV
   simulator, ``set_value()`` emits a ``UserWarning`` and discards the
   value.  The CSV adapter implements ``get_available_variables()`` to
   return all column names for connection validation.


External simulators
-------------------

User-defined Python simulators can be integrated by subclassing
``SimulatorAdapter`` from ``energysim.base``.  See
:doc:`working_external_sims` for the full guide.

Example::

    my_world.add_simulator(
        sim_type='external', sim_name='my_sim',
        sim_loc='/path/to/directory',
        inputs=['cmd'], outputs=['result'],
        step_size=60)

The ``sim_name`` must match the Python filename (without ``.py``), and
the file must contain a class called ``external_simulator``.


PowerFactory simulators
-----------------------

DIgSILENT PowerFactory models can be integrated using the
``'powerfactory'`` sim type.  This requires PowerFactory 2020+ with
the Python API and a valid licence.

Additional keyword arguments:

- ``pf`` *(str, default 'ldf')* -- calculation type:

  - ``'ldf'`` -- load flow (``ComLdf``)
  - ``'shc'`` -- short circuit (``ComShc``)
  - ``'rms'`` -- RMS transient simulation (``ComInc`` + ``ComSim``)

- ``pf_path`` *(str or None)* -- path to the PowerFactory Python
  directory, e.g. ``r"C:\DIgSILENT\PowerFactory 2024\Python\3.11"``.
  Only needed when running outside the PF embedded interpreter.

Example::

    my_world.add_simulator(
        sim_type='powerfactory',
        sim_name='grid_pf',
        sim_loc='',
        inputs=['Load1.plini', 'Gen1.pgini'],
        outputs=['Bus1.m:u', 'Bus2.m:u', 'Line1.m:I:bus1'],
        step_size=900,
        pf='ldf',
        pf_path=r'C:\DIgSILENT\PowerFactory 2024\Python\3.11')

.. important::
   ``sim_name`` must match your PowerFactory **project name**.
   The adapter calls ``ActivateProject(sim_name)`` internally.


Variable naming convention (PowerFactory)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Variables are specified as ``ElementName.attribute``:

**Inputs** (written via ``SetAttribute``):

- Load active power: ``Load1.plini`` (auto-prefixed to ``e:plini``)
- Generator active power: ``Gen1.pgini``
- Any editable attribute: ``Element.e:attributeName``

**Outputs** (read via ``GetAttribute``):

- Bus voltage magnitude: ``Bus1.m:u``
- Bus voltage angle: ``Bus1.m:phiu``
- Line current: ``Line1.m:I:bus1``
- Generator output: ``Gen1.m:P:bus1``

Result attributes start with ``m:``.  For inputs, the ``e:`` prefix is
added automatically if omitted.


MATLAB / Octave simulators
---------------------------

MATLAB ``.m`` **functions** (not scripts) can be integrated using the
``'matlab'`` sim type.  The adapter auto-detects whether to use the
MATLAB Engine for Python or GNU Octave (via ``oct2py``).

Additional keyword arguments:

- ``engine`` *(str, default 'auto')* -- which backend to use:

  - ``'auto'`` -- try MATLAB first, fall back to Octave
  - ``'matlab'`` -- MATLAB Engine only
  - ``'octave'`` -- Oct2Py (GNU Octave) only

Example::

    my_world.add_simulator(
        sim_type='matlab',
        sim_name='heatpump',
        sim_loc=r'/path/to/m_files',
        inputs=['P_cmd', 'T_source', 'T_sink'],
        outputs=['Q_thermal', 'COP'],
        step_size=30,
        engine='auto')

.. important::
   ``sim_name`` must match the ``.m`` filename (without extension).
   The file must define a **function** (not a script).


Writing .m functions for energysim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your ``.m`` function must follow this signature::

    function [out1, out2, ...] = func_name(in1, in2, ..., time)

- **Input arguments** map to the ``inputs`` list **in order**, plus
  ``time`` (appended automatically by the adapter).
- **Output arguments** map to the ``outputs`` list **in order**.

The adapter validates at init time that:

1. The ``.m`` file is a function (not a script).
2. ``nargin`` matches ``len(inputs) + 1`` (the +1 is for ``time``).
3. ``nargout`` matches ``len(outputs)``.

Example ``.m`` function (``heatpump.m``)::

    function [Q_thermal, COP] = heatpump(P_cmd, T_source, T_sink, time)
    % HEATPUMP  Simple Carnot heat-pump model for energysim.
    %
    %   Inputs:
    %     P_cmd    - commanded electrical power [kW]
    %     T_source - source temperature [deg C]
    %     T_sink   - sink temperature [deg C]
    %     time     - simulation time [s] (unused here)
    %
    %   Outputs:
    %     Q_thermal - thermal output [kW]
    %     COP       - coefficient of performance [-]

        COP_carnot = (T_sink + 273.15) / max(T_sink - T_source, 0.1);
        COP = 0.45 * COP_carnot;
        Q_thermal = P_cmd * COP;
    end

Another example (``battery.m``)::

    function [SoC, P_actual] = battery(P_cmd, time)
    % BATTERY  Simple Li-ion battery with Coulomb counting.

        persistent soc_state;
        if isempty(soc_state)
            soc_state = 0.5;  % start at 50%
        end

        capacity_kwh = 13.5;
        max_power = 5.0;
        dt = 30;

        P_actual = max(-max_power, min(max_power, P_cmd));
        soc_state = soc_state - (P_actual * dt / 3600) / capacity_kwh;
        soc_state = max(0.05, min(0.95, soc_state));
        SoC = soc_state;
    end

.. note::
   Functions can use ``persistent`` variables to maintain state between
   steps — the MATLAB/Octave equivalent of Python instance variables.
