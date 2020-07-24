Working with external simulators
================================
One of the USPs of ``energysim`` is that it can be coupled to external simulators fairly easily. However, it is expected that the users are familiar with intermediate level of python programming. In this section, we show how users can interface their own simulators with ``energysim`` very easily.

In the main code, users can add their simulators to ``my_world`` by using::
    
    my_world.add_simulator(sim_type='external', sim_name = 'my_external_simulator', sim_loc = sim_loc, outputs=['var1', 'var2'], step_size=1)

Here the ``sim_name`` is the file name of the interfaced simulator *my_external_simulator.py*. This file has the following template:

.. include:: external_simulator.py
    :literal:

The four functions inside ``class external_simulator()`` are all that ``energysim`` requires to interface with the simulator. Users are free to make imports, and create other functions which can be called within this file. Let us go through each function and their definitions.

init() method
^^^^^^^^^^^^^
Note that this is different from the ``__init__()`` class. This method is needed to initialize the simulator. You can use it to, for example, establish connection to another software, or package. Basically, start-up the simulator.


step(time) method
^^^^^^^^^^^^^^^^^
The ``step`` method is used by ``energysim`` coordinator to perform time stepping for each simulator. The coordinator steps the simulator by deltaT = ``step_size`` defined while adding the simulator to ``world``. The step method is useful when the model consists consists of time-dependent equations and exibits dynamics behavior.


get_value(variable, time) method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``get_value`` method is used by the simulator to query ``variable`` value from the simulator. The coordinator queries the simulator by asking the value of ``variable`` at time=``time``. The ``variable`` is enclosed in a python list. The user must define in this method, how to obtain the value of that ``variable`` from its simulator.


set_value(variable, value) method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``set_value`` method is used by the simulator to set the variable to a particular value at the time instance when message are exchanged between simulators. In this method, users must specify how to set the ``variable`` to the specified ``value``. Both the ``variable`` and ``value`` are enclosed within a list.
