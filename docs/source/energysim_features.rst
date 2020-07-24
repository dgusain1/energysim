energysim features
==================

Although key functions were highlighted in the main page, ``energysim`` comes 
with additional inbuilt methods which allow users to take more control of the cosimulation.

Adding signals
^^^^^^^^^^^^^^
The ``add_signal()`` method of the ``world`` object provides the ability to add user-defined time (in)variant signals to the cosimulation objects. This is especially useful if some inputs of the cosimulation simulators need a constant signal, or a time varying signal (such as sin). In ``energysim``, this can be added by::

    def my_signal(time):
        return [1]
    
    my_world.add_signal(sim_name='constant_signal', signal = my_signal, step_size=1)

Users need to make sure that the return value from the my_signal part is within square brackets (**[ ]**), i.e. a **list**. The value returned must also be a single value. If multiple values are returned, the signal function will not be added. The signal function can also be more complex. For example, instead of ``return [1]``, the ``my_signal`` function can also ``return np.sin(2*np.pi*time)``.

In the connections dictionary, this signal can then be connected to other simulators by::

    connections = {'constant_signal.y' : 'sim1.input_variable1'}
    my_world.add_connections(connections)

The default step size is 1s for signals. However, it can be changed by specifying ``step_size`` argument in the ``add_signal`` method.

..
    Variable step_size
    ^^^^^^^^^^^^^^^^^^
    Time integration for FMUs can be computationally expensive. To overcome this, variable time stepping can be used. In energysim, a naive implementation of variable stepping is available. When adding the FMU simulators, users can specify ``variable = True`` along with other arguments in ``add_simulator()``::
    
        p2g_fmu_path = r'/path/to/fmu'
        my_world.add_simulator(sim_type='fmu', sim_name='p2g', sim_loc=p2g_fmu_path, 
            step_size=60, inputs=[], outputs = ['p2g.h2prod'], variable=True)
    
    When ``variable = True``, the ``energysim`` master checks the number of time steps the simulator will take between two message exchange time points. That is, it checks the ratio of "t_macro/sim_step_size". If this ratio is larger than 50, the simulator changes the integration time step to 0.001s around the t_macro. Therefore, between 
    This is only available for CoSimulation FMUs currently. 

Modify signals before exchange
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Many times, it is required in the simulation whereby output of a particular simulator needs to be "modified" before exchanging the value with a another simulator. This is fairly common in energy system integration simulations. For example, consider two simulators a CHP system and an electric network (EN). The power output from CHP simulator (an FMU) is obtained in Watts units. This power output of the CHP needs to be provided to the pandapower network. However, the EN accepts values only in MW. One way to address the problem is to change the output values of the CHP in the model itself and recompile the FMU. It may not always be possible to do so (FMU may be encrypted!). Therefore, ``energysim`` provides an inbuilt method to address such problems by supplying the ``modify_dict`` to ``world`` options. Two types of modifications can be applied: 1) multiply with a constant, and 2) multiply with a constant and add a constant. This is shown as follows ::
        
    modifications = {'sim1.var1':[x], #multiplies var1 of sim1 by x before variables are exchanged,
                     'sim2.var1':[x1, x2] #multiplies by x1 and adds x2
                    }
    
    options = {'init' : initializations,
                'modify_signal': modifications}
    
    my_world.options(options)

In the example highlighted above, the modification can be set as::
    
    #multiply electric power of chp by 1/1e6 to convert W -> MW before it is given to EN,
    modifications = {'chp.e_power':[1/1e6]}


Enabling sensitivity analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An important part of energy system analysis is the parameter sensitivity. In ``energysim``, this can be done by updating the ``init`` option to update parameters of the cosimulation.::
    
    #configure energysim with simulators
    
    for v1, v2 in [(0.1,0.2), (1,2), (10,20)]:
        sens = {'sim_name1' : (['sim_variables'], [values]),
    	        'sim_name2' : (['sim_variables'], [values])}
        	options = {'init' : sens}
        	my_world.options(options)
        res = my_world.simulate(pbar=False)
        #extract relevant results and store them

A more sophisticated functionality is planned to create an integrated sensitivity analysis with energysim.


Optimal Power Flow
^^^^^^^^^^^^^^^^^^
By default, the powerflow network added in ``energysim`` are solved for ac powerflow. However, users can specify in the ``add_simulator`` arguments to solve for opf. This is shown below::

    my_world.add_simulator(sim_type = 'powerflow', sim_name = 'grid', 
            sim_loc = grid_loc, inputs = ['wind1.P'], outputs=['Bus 1.V', 'Bus 12.V', 'wind1.P'], 
            step_size=3, pf = 'opf')

This feature is currently only available in pandapower networks.

Validation of FMUs
^^^^^^^^^^^^^^^^^^
Internally, FMPy checks the validity of FMUs. To speedup, this flag can be set as ``False`` while adding simulators.
        
System Topology Plot
^^^^^^^^^^^^^^^^^^^^
``energysim`` uses ``NetworkX`` to generate topology of the cosimulation based on the connections dictionary. This can be visualised by::
        	
        my_world.plot(plot_edge_labels=False, node_size=300, node_color='r')


