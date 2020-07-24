FAQ
===

OPFNotConvergedError
^^^^^^^^^^^^^^^^^^^^
OPF and DCOPF functionalities are subject to pandapower optimization. Therefore, you must make sure that OPF converence is met within the pandapower network before integrating it with ``energysim.world``. 

FMU Initialization Error
^^^^^^^^^^^^^^^^^^^^^^^^
If you get an initialization failed error for FMU, please check if it works independently. You can use the following code structure to check::

    from fmpy import *
    fmu_loc = /path/to/fmu
    res=simulate_fmu(fmu_loc)
    print(res)
    
If this works, but you still get initialization failed error with FMUs, you can try following remedies:
    
    1. Clear cache, temp folders.
    2. Restart application
    3. Run it with admin privileges