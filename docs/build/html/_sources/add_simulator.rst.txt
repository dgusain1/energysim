Adding simulators
=================

Adding simulators in energysim is done by calling the method ``add_simulator()``.
As mentioned, four main types of simulators can be added: FMU, powerflow networks, CSV, and external simulators.

The add_simulator requires **six** main arguments for all simulators. Optional arguments for FMUs, powerflows, and external simulators can be added if needed. These are:

	- ``sim_type`` : Specifies simulator type. Can be either of 'fmu', 'powerflow', 'csv', 'external'
	- ``sim_name`` : Each simulator must have a unique name. This is later required when specifying connections.
	- ``sim_loc`` : A raw string address of simulator location.
	- ``outputs`` : Variables that need to be recorded from the simulator during simulation.
	- ``inputs`` : Input variables to the simulator that are to be used while defining the connections.
	- ``step_size`` : Internal step size for simulator (1e-3 by default). Also known as micro-time steps. This is the time integration step required for solvers for the simulators. For example, some FMUs require integration time steps of 1e-3 secs, while for powerflow networks, the time step can be 15 mins (900s).

The ``add_simulator()`` works as follows::
	
	my_world.add_simulator(sim_type = sim_type, sim_name = sim_name, 
                            sim_loc = sim_loc, outputs = outputs, inputs = inputs,
                                step_size = step_size)

Additional arguments
^^^^^^^^^^^^^^^^^^^^
Apart from the six required arguments, users can also specify additional arguments for each simulator. These are:

    1. For FMUs: FMPy generally validates the FMUs before initialization process. Users can skip this validation by specifyng the argument `validate = False`. Generally, it has been observed, the FMUs packaged with OpenModelica fail validation. However, they can be simulated by setting the ``validate = False``.
    2. For powerflow: For powerflow simulators, users can specify whether to execute a AC powerflow or an optimal power flow, or a DCpowerflow. This can be done by providing the specifying the value of argument ``pf``. It can be set to "pf", "dcpf", "opf", or "dcopf". Please note, this is only available for pandapower networks currently. 
    3. For csv: For csv files, the users can specify the delimiter by providing the argument ``delimiter=','`` or whatever the delimiter is.

	
Variable naming convention
^^^^^^^^^^^^^^^^^^^^^^^^^^
Since variable extraction is an important part of cosimulation, it is important to become aware of the variable naming convention used for the simulators in ``energysim``.

FMUs
----
For FMUs, the variable naming convention is similar to how it is generally accessed within Modelica based environments. For example a variable_k nested within the FMU model can be accessed using::

    sim_name.Component_1.SubComponent_i.Variable_k

This is to be followed when specifying connections, initializations, or signal modifications. 

Powerflow networks
^^^^^^^^^^^^^^^^^^
Currently, ``energysim`` can only record bus voltages magnitude (V) and bus voltage angles (VA), along with active (P) and reactive power (Q) values for loads, static generators, external grid, generators. For inputs, it can set the active (P) and reactive power (Q) setpoints for loads, static generators, generators. All the elements within the network must have a valid name. This has to be ensured before importing the pandapower network within ``energysim`` environment.

Consider a network 'grid' with 3 buses named 'Bus1', 'Bus2', 'Bus3'. It has three loads named 'Load1', 'Load2', and 'Load3'. Similarly, a generator 'Gen1' is connected to one of the buses. 

The following quantities can be specified to receive inputs in the connections dictionary:

    - Gen1.P
    - Gen1.Q
    - Load1.P, Load2.P, Load3.P
    - Load1.Q, Load2.Q, Load3.Q

The following quantities can be speified as outputs in ``add_simulator`` to be recorded:

    - Bus1.V, Bus2.V, Bus3.V
    - Bus1.Va, Bus2.Va, Bus3.Va
    - Gen1.P
    - Gen1.Q
    - Load1.P, Load2.P, Load3.P
    - Load1.Q, Load2.Q, Load3.Q
    
It must be clarified that energysim can only retreive or set variables in simulators when the simulators name are an exact match. Please make sure that component names and variable names such as P, Q are exactly how they are specified here. 

CSV files
^^^^^^^^^
CSV simulators are used to attach csv data files to ``world``. The csv file must have clearly specified columns. One of the columns must be 'time'.

The output variables for CSV simulators are the column names. Consider a csv file given by:
    
| time,power
| 0,18
| 1,18
| 2,20
| 3,50
| 4,25
| 5,15

The "power" variable can be accessed using ``sim_name.power``. ``energysim`` can then automatically read the power variable from csv files corresponding to simulation time. If the current simulation time is between two time values,   ``energysim`` will read the value at time given by ``index = int(np.argwhere(time_array>current_time)[0] - 1)`` where ``time_array`` is the list of time values in the csv.
    
External simulators
^^^^^^^^^^^^^^^^^^^
Variables in external simulators can be accessed similar to other simulators::
    
     sim_name.var1
    
Users must make sure that the variable names within the simulator and that defined in ``energysim`` connections dict are the same.
    
    
    









