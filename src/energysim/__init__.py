from .csAdapter import FmuCsAdapter
from .meAdapter import FmuMeAdapter
from .csv_adapter import csv_simulator
from .pypsaAdapter import pypsa_adapter
from .ppAdapter import pp_adapter
from .signalAdapter import signal_adapter
from .utils import convert_hdf_to_dict, record_data, create_results_recorder
from fmpy.model_description import read_model_description
import sys
import numpy as np
import pypsa, logging as lg
import matplotlib.pyplot as plt
from tqdm import tqdm
pypsa.pf.logger.setLevel(lg.CRITICAL)
from functools import reduce
import importlib
import tables as tb
filters = tb.Filters(complevel=5, complib='zlib')
import gc

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
    Finally, the ``simulate()`` function can be called to simulate the world.
    When ``record_all`` is True, ``energysim`` records the value of variables not only at macro time steps, but also at micro time steps specified by the user when adding the simulators. This allows the users to get a better understanding of simulators in between macro time steps. When set to False, variables are only recorded at macro time step. This is useful in case a long term simulation (for ex. a day) is performed, but one of the simulators has a time step in milli-seconds. ``pbar`` can be used to toggle the progress bar for the simulation::

    >>> my_world.simulate(pbar=True, record_all=False)

    Extracting Results
    ------------------
    Results can be extracted by calling ``results()`` function on ``my_world`` object. This returns a dictionary object with each simulators' results as pandas dataframe. Additionally, ``to_csv`` flag can be toggled to export results to csv files.

    >>> results = my_world.results(to_csv=True)

    Package Info
    ------------
    Author : Digvijay Gusain

    Email : digvijay.gusain29@gmail.com

    Version : 2.1.x


    """
    
    def __init__(self, start_time = 0, stop_time = 1000, logging = False, t_macro = 60, res_filename='es_res.h5'):
        self.start_time = start_time
        self.stop_time = stop_time
        self.logging = logging
        self.exchange = t_macro
        self.modify_signal = False
        self.modify_dict = {}
        self.init_dict = {}
        self.G = True
        self.simulator_dict = {}
        self.file_name = res_filename 

    def add_signal(self, sim_name, signal, step_size=1):
        if sim_name not in self.simulator_dict.keys():
            signal_obj = signal_adapter(signal_name=sim_name, signal=signal)
            self.simulator_dict[sim_name] = ['signal', signal_obj, step_size, ['y']]

    def add_simulator(self, sim_type='', sim_name='', sim_loc='', inputs = [], outputs = [], step_size=1, **kwargs):
        """
        Method to add simulator to ``world()`` object. ``add_simulator()`` takes the following arguments as inputs :

        1. ``sim_type`` : 'fmu', 'powerflow', 'csv', 'external'
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
                - pf : 'pf' or 'opf' to toggle powerflow or optimal power flow when available. Currently only available for pandapower.
        """

        if sim_type.lower() == 'fmu':
            solver_name = 'Cvode'
            if 'variable' in kwargs.keys():
                variable = kwargs['variable']
            else:
                variable=False
            if 'validate' in kwargs.keys():
                validate=kwargs['validate']
            else:
                validate=False
            self.add_fmu(sim_name, sim_loc, inputs=inputs, outputs=outputs, step_size=step_size,
                         solver_name = solver_name, variable=variable, validate=validate)

        elif sim_type.lower() == 'powerflow':
            if 'pf' in kwargs.keys():
                pf = kwargs['pf']
            else:
                pf='pf'
                
            self.add_powerflow(sim_name, sim_loc, inputs=inputs, outputs=outputs, step_size=step_size, pf=pf)

        elif sim_type.lower() == 'csv':
            if 'delimiter' in kwargs.keys():
                delimiter=kwargs['delimiter']
            else:
                delimiter=','
            self.add_csv(csv_name=sim_name, csv_location=sim_loc, step_size=step_size, outputs=outputs, delimiter=delimiter)

        elif sim_type.lower() == 'external':
            elif sim_type.lower() == 'external':
            self.add_external_simulator(sim_name = sim_name, sim_loc = sim_loc, inputs = inputs, outputs = outputs, step_size = step_size)

        else:
            print(f"Simulator type {sim_type} not recognized. Possible sim_type are 'external', 'fmu', 'powerflow', 'csv' types.")
            print(f"Simulator {sim_name} not added")

    def add_external_simulator(self, sim_name, sim_loc, inputs, outputs, step_size = 900, **kwargs):
        sys.path.append(sim_loc)
        tmp = importlib.import_module(sim_name)
        external_simulator = tmp.external_simulator
        kwargs = kwargs
        #check that the sim_name is unique
        if sim_name not in self.simulator_dict.keys():
            simulator = external_simulator(inputs = inputs, outputs=outputs, options=kwargs)
            self.simulator_dict[sim_name] = ['external', simulator, step_size, outputs]
            if self.logging:
                print("Added external simulator '%s' to the world" %(sim_name))
        else:
            print(f"Please specify a unique name for simulator. {sim_name} already exists.")

    def add_powerflow(self, network_name, net_loc, inputs = [], outputs = [], pf = 'pf', step_size=900, **kwargs):
        assert network_name is not None, "No name specified for power flow model. Name must be specified."
        assert net_loc is not None, "No location specified for power flow model. Can't read without."
        try:
            network = pypsa_adapter(network_name, net_loc, inputs = inputs, outputs = outputs, logger_level = kwargs['logger'] if 'logger' in kwargs.keys() else 'DEBUG')
        except:
            network = pp_adapter(network_name, net_loc, inputs = inputs, outputs = outputs)
        if network_name not in self.simulator_dict.keys():
            self.simulator_dict[network_name] = ['pf', network, step_size, outputs, pf]
            if self.logging:
                print("Added powerflow simulator '%s' to the world" %(network_name))
        else:
            print(f"Please specify a unique name for simulator. {network_name} already exists.")

    def add_fmu(self, fmu_name, fmu_loc, step_size, inputs = [], outputs=[], exist=False, solver_name = 'Cvode', variable=False, **kwargs):
        if 'validate' in kwargs.keys():
            validate=kwargs['validate']
        else:
            validate=True
        m_desc = read_model_description(fmu_loc, validate=validate)
        fmi_type = 'CoSimulation' if m_desc.coSimulation is not None else 'ModelExchange'
        if fmi_type == 'CoSimulation':
            fmu_temp = FmuCsAdapter(fmu_loc,
                             instanceName=fmu_name,
                             step_size = step_size,
                             inputs = inputs,
                             outputs=outputs,
                             exist=exist,
                             validate=validate)
        elif fmi_type == 'ModelExchange':
            fmu_temp = FmuMeAdapter(fmu_loc,
                             instanceName=fmu_name,
                             step_size = step_size,
                             inputs = inputs,
                             outputs=outputs,
                             solver_name = solver_name,
                             validate=validate)

        if fmu_name not in self.simulator_dict.keys():
            self.simulator_dict[fmu_name] = ['fmu', fmu_temp, step_size, outputs, variable]
            if self.logging:
                print("Added FMU simulator '%s' to the world" %(fmu_name))
        else:
            print(f"Please specify a unique name for simulator. {fmu_name} already exists.")

    def add_csv(self, csv_name, csv_location, step_size, outputs, delimiter):
        csv_obj = csv_simulator(csv_name, csv_location, step_size, outputs, delimiter)
        if csv_name not in self.simulator_dict.keys():
            self.simulator_dict[csv_name] = ['csv', csv_obj, step_size, outputs]
            if self.logging:
                print("Added CSV simulator '%s' to the world" %(csv_name))

    def add_connections(self,connections={}):
        '''
        Adds connections between simulators. Connections are specified as dictionary with keys as sending variables, and values as receiving variables
        '''

        assert len(self.simulator_dict) > 0, "Cannot make connections with no simulators defined."
        self.simulator_connections = connections
        if self.logging:
            print("Added %i connections between simulators to the World" %(len(self.simulator_connections)))

    def get_fmu_name(self, name):
        if type(name).__name__ == 'str':
            return name.split('.')[0]
        if type(name).__name__ == 'tuple':
            return [x.split('.')[0] for x in name]

    def plot(self, plot_edge_labels=False, node_size=300, node_color='r'):
        """
        Plots approximate graph diagram for the energysim network.
        """
        import networkx as nx
        self.G=nx.DiGraph()
        for key, value in self.simulator_connections.items():

            if type(key).__name__ == 'str' and type(value).__name__ == 'str':
                n1 = self.get_fmu_name(key)
                n2 = self.get_fmu_name(value)
                self.G.add_node(n1)
                self.G.add_node(n2)
                self.G.add_edge(n1, n2, name=key + ' to ' + value)
            elif type(key).__name__ == 'str' and type(value).__name__ == 'tuple':
                n1 = self.get_fmu_name(key)
                self.G.add_node(n1)
                for v in value:
                    n2 = self.get_fmu_name(v)
                    self.G.add_node(n2)
                    self.G.add_edge(n1, n2, name=key + ' to ' + v)
            elif type(key).__name__ == 'tuple' and type(value).__name__ == 'str':
                n2 = self.get_fmu_name(value)
                self.G.add_node(n2)
                for k in n1:
                    n1 = self.get_fmu_name(k)
                    self.G.add_node(n1)
                    self.G.add_edge(n1, n2, name=k + ' to ' + value)
            else:
                print('There is a many to many dependance in the graph. Cannot create graph.')
        pos=nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G,pos, node_size = node_size, node_color=node_color, alpha=0.8, with_labels=True)
        nx.draw_networkx_edges(self.G,pos, alpha=0.8, edge_color='b')
        nx.draw_networkx_labels(self.G,pos)
        if plot_edge_labels:
            edge_labels = nx.get_edge_attributes(self.G,'name')
            nx.draw_networkx_edge_labels(self.G,pos, edge_labels=edge_labels)
        plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
        plt.tick_params(axis = "y", which = "both", bottom = False, top = False)
        plt.tight_layout()
        plt.show()

    def get_lcm(self):
        ss_dict = [x[2] for x in self.simulator_dict.values()]# if x[0] != 'signal']
        max_order = int(max([len(str(i).split('.')[1]) for i in ss_dict]))
        new_list = 10**(max_order)*np.array(ss_dict)

        def lcm1(a, b):
            greater = a if a>b else b
            while True:
                if greater % a == 0 and greater % b == 0:
                    lcm = greater
                    break
                greater += 1
            return lcm

        def get_lcm_for(your_list):
            return reduce(lambda x, y: lcm1(x, y), your_list)

        final_lcm = get_lcm_for(new_list)
        return final_lcm/10**max_order

    def init(self):
        '''
        Initialises simulator objects in world.
        '''
        if self.logging:
            print("Simulation started..")
            print("Simulation status:\n")
        
        self.sim_dict={}
        for i, j in self.simulator_dict.items():
            self.sim_dict[i] = j[3]
            
        #determine the final_tStep for World
        if self.exchange != 0:
            self.macro_tstep = self.exchange
        else:
            self.macro_tstep = self.get_lcm()

        create_results_recorder(file_name=self.file_name, sim_dict=self.sim_dict)
        
        #set initial conditions
        if self.init_dict:
            self.set_parameters(self.init_dict)
        
        #initialize simulators
        for sim_name, sim_values in self.simulator_dict.items():
            try:
                sim_values[1].init()
            except:
                try:
                    from time import sleep
                    sleep(2)
                    sim_values[1].init()
                except:
                    print(f"Could not initialize the {sim_values[0]} simulator {sim_name}. Check FAQs. Simulation stopped.")
                    sys.exit()
    
    def step_simulator(self, sim, time, record_all, local_stop_time):
        sim_name, sim_items = sim
        sim_type = sim_items[0]
        simulator = sim_items[1]
        sim_ss = sim_items[2]
        outputs = sim_items[3]
        temp_time = time
        tmp_var = []
        if len(outputs)>0 and record_all == False:
            tmp = [temp_time] + list(simulator.get_value(outputs, temp_time))
            tmp_var.append(tmp)
        elif len(outputs) == 0 and record_all == False:
            tmp = [temp_time]
            tmp_var.append(tmp)
        
        while temp_time < local_stop_time:
            if len(outputs)>0 and record_all:
                tmp = [temp_time] + list(simulator.get_value(outputs, temp_time))
                tmp_var.append(tmp)
            elif len(outputs) == 0 and record_all:
                tmp = [temp_time]
                tmp_var.append(tmp)
    
            if sim_type == 'fmu':
                if sim_items[4]:
                    stepsize = self.get_step_time(sim_ss, local_stop_time, temp_time, self.macro_tstep, time)
                    try:
                        simulator.step_advanced(min(temp_time, local_stop_time), stepsize)
                        temp_time += stepsize             #self.stepsize_dict[_fmu]
                    except:
                        print(f'Could not initiate variable step size for the FMU {sim_name}. Energysim will switch to fixed steps.')
                        simulator.step(min(temp_time, local_stop_time), sim_ss)
                        temp_time += sim_ss
                else:
                    simulator.step(min(temp_time, local_stop_time))
                    temp_time += sim_ss
            else:
                simulator.step(temp_time)
                temp_time += sim_ss
        return tmp_var

    def simulate(self, pbar = True, record_all=True):
        '''
        Simulates the world object
        '''
        startTime = self.start_time
        stopTime = self.stop_time
        self.init()
        assert (stopTime-startTime >= self.macro_tstep), "difference between start and stop time > exchange value specified in world initialisation"
        total_steps = int((stopTime-startTime)/self.macro_tstep)
        for time in tqdm(np.linspace(startTime, stopTime, total_steps, endpoint=False), disable = not pbar):
            #exchange values at t=0
            tmp_dict = {}
            self.exchange_variable_values(time)
            local_stop_time = time + self.macro_tstep#, stopTime)
            for sim_name, sim_items in self.simulator_dict.items():
                tmp_var = self.step_simulator((sim_name, sim_items), time, record_all, local_stop_time)
                tmp_dict[sim_name] = np.array(tmp_var)
                del tmp_var
                gc.collect()
                
            record_data(file_name=self.file_name, res_dict=tmp_dict)
            del tmp_dict
            gc.collect()
        
        for sim_name, sim_items in self.simulator_dict.items():
            if sim_items[0] == 'fmu':
                sim_items[1].cleanUp()

        return 'Done'

    def results(self, to_csv=False):
        """
        Processes the results from the cosimulation and provides them in a dict+df form.
        """
        
        res = convert_hdf_to_dict(file_name = self.file_name, sim_dict=self.sim_dict, to_csv=to_csv)
        return res

    
    def alter_signal(self, op_, tmp):
        '''
        Checks if the signal needs modification as aspecified in modify_signal dict under options.
        '''
        #check if signal modification is needed.
        temp_var = tmp[0]
        if self.modify_dict:
            flag = True if op_ in self.modify_dict.keys() else False
            if flag:
                if len(self.modify_dict[op_]) == 1:
                    ret_output = temp_var*self.modify_dict[op_][0]
                elif len(self.modify_dict[op_]) == 2:
                    ret_output = temp_var*self.modify_dict[op_][0] + self.modify_dict[op_][1]
                else:
                    print('Unknown signal modification on output %s. Not modification applied.' %(op_))
                    ret_output = temp_var
            else:
                ret_output = temp_var
        else:
            ret_output = temp_var
        return [ret_output]

    def exchange_variable_values(self, t):
        """
        Helper function for exchanging the variable values between simulators at particular time.
        """
        #parse connections
        for output_, input_ in self.simulator_connections.items():
            #data collector
            #check if it is tuple or single
            if type(output_).__name__ == 'str':
                ou_sim_name, ou_sim_var = output_.split('.')[0], output_.replace(output_.split('.')[0],'')[1:]
                tmp_var1 = self.simulator_dict[ou_sim_name][1].get_value([ou_sim_var], t)
                tmp_var = self.alter_signal(output_, tmp_var1)
            elif type(output_).__name__ == 'tuple':
                tmps = []
                for item in output_:
                    ou_sim_name, ou_sim_var = item.split('.')[0], item.replace(item.split('.')[0],'')[1:]
                    tmp1 = self.simulator_dict[ou_sim_name][1].get_value([ou_sim_var], t)
                    tmp = self.alter_signal(item, tmp1)
                    tmps.append(tmp[0])
                tmp_var = [sum(tmps)]

            if type(input_).__name__ == 'str':
                in_sim_name, in_sim_var = input_.split('.')[0], input_.replace(input_.split('.')[0],'')[1:]
                self.simulator_dict[in_sim_name][1].set_value([in_sim_var], tmp_var)
            elif type(input_).__name__ == 'tuple':
                for item in input_:
                    in_sim_name, in_sim_var = item.split('.')[0], item.replace(item.split('.')[0],'')[1:]
                    self.simulator_dict[in_sim_name][1].set_value([in_sim_var], tmp_var)

    def get_step_time(self, step_size, local_stop_time, current_time, final_tStep, int_start_time):
        """
        Helper function to obtain the variable time step for specified FMUs.
        """
        if current_time-int_start_time < 0.01 or local_stop_time-current_time < 0.01:
            new_step_size = 0.001
        else:
            new_step_size = final_tStep-0.01
        return new_step_size

    def options(self, settings):
        assert (len(self.simulator_dict) > 0),"Cannot add settings to world when no simulators are specified!"

        if 'parameter_set' in settings.keys():
            self.set_parameters(settings['parameter_set'])

        if 'modify_signal' in settings.keys():
            self.modify_dict = settings['modify_signal']

        if 'init' in settings.keys():
            self.init_dict = settings['init']


    def set_parameters(self, parameter_dictionary):
        _sims = parameter_dictionary.keys()
        for _sim in _sims:
            if self.simulator_dict[_sim][0] == 'fmu':
                parameter_list_derived = list(parameter_dictionary[_sim])[0]
                parameter_values_to_set = list(parameter_dictionary[_sim])[1]
                self.simulator_dict[_sim][1].set_start_values(init_dict = dict(zip(parameter_list_derived, parameter_values_to_set)))
            else:
                
                parameter_list_derived = list(parameter_dictionary[_sim])[0]
                parameter_values_to_set = list(parameter_dictionary[_sim])[1]
                self.simulator_dict[_sim][1].set_value(parameter_list_derived, parameter_values_to_set)

    def check_signal_output(self, fx):
        tmp = fx(1)
        if type(tmp) == 'list':
            return True
        else:
            print('Function must return list. Signal function will not be added.')
            return False

    def plot_results(self, xlab = 'Time(s)', ylab = '', y = [], scientific = False):
        locol = [x for x in self.list_of_columns if x!='time']
        plt.figure(2)
        if self.interpolate_results:
            time_vector = self.results_dataframe.time
            if len(y) == 0:
                for item in locol:
                    plt.plot(time_vector,self.results_dataframe.loc[:,item], label=item)
            else:
                for item in y:
                    try:
                        plt.plot(time_vector,self.results_dataframe.loc[:,item], label=item)
                    except:
                        print("Could not find the variable %s in results dataframe. Make sure you have \
                              specified the variables as 'fmu.variable'. Skipping." %(item))
                        continue
            plt.ticklabel_format(useOffset=scientific)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.legend()
            plt.show()
        else:
            print('Results can only be plotted natively if interpolate_results option is set to True in World() options.')
