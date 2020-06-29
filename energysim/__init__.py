# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 19:08:54 2018

@author: digvijaygusain
"""

__version__ = '1.0.0'
__author__ = 'Digvijay Gusain'

from .csAdapter import FmuCsAdapter
from .meAdapter import FmuMeAdapter
from .pypsaAdapter import pypsa_adapter
from .signalAdapter import signal_adapter
from .ppAdapter import pp_adapter
from .pyScriptAdapter import py_adapter
from fmpy.model_description import read_model_description
import sys
import numpy as np
import pandas as pd
import pypsa, logging as lg
from time import time as current_time, sleep
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
pypsa.pf.logger.setLevel(lg.CRITICAL)

__about__ = '''
    World is your cosimulation canvas. Initialise the World object with basic simulation 
    parameters such as:
        1. start_time (=0 by default) 
        2. stop_time (=100 by default)
        3. logging (False by default). Gives users update on simulation progress
        4. exchange (=0 by dfault). Users can specify instance of information exchange
        5. clean_up (True by default) cleans up temporary files created during the simulation (can be very large in some cases, hence recommended to keep it True)
        6. interpolate_results (True by default) FMUs with larger time steps have their outputs interpolated to match the time step of the lowest time step of specified FMUs.
    
    After specifying the world, you can add simulators to the world by add_simulator(). This is done by specifying the following:
	1. sim_type: 'fmu' / 'powerflow' / 'csv' / 'signal'
        2. sim_name: It is extremely essential that you assign the simulator a relevant name. This helps in specifying connections between different simulators later.
        3. sim_location: Use a raw string address of simulator file.
        4. outputs: specify the outputs that need to be recorded during simulation. 
	5. inputs: specify the inputs of the simulator.
        5. step_size (=1e-3 by default)
    
    When specifying signals, use signal = <signal>. The keyword is important.

    The connections between simulators can be specified with a dictionary as follows:
        {'_fmu1.output_variable1':'_fmu2.input_variable1',
        '_fmu1.output_variable2':'_fmu2.input_variable2',
        '_fmu1.output_variable3':'_fmu2.input_variable3',}
    
    Afterwards, simulate() function can be called to simulate the world. This returns a pandas dataframe
    with output values of all simulators as specified during add_simulator phase.
    
    NEW: World object can now be accessed as a single simulation entity. This allows users to embed World cosimulation into control schemes defined in Python as python functions. 
	after specifying World, instead of calling my_world.simulate(), users can make a python loop.
	
	def control(x):
		<do fancy stuff>
		return some_values
	my_world.init()
	for time in range(1,10):
		my_world.step(time)
		tmp = my_world.get_value(<variable1>)
		y = control(tmp)
		my_world.set_value(<variable2>, <y>)
	results = my_world.results()
    '''

class World():
    
    def __init__(self, start_time = 0, stop_time = 100, logging = False, exchange = 0, clean_up = True, interpolate_results = True):
        self.start_time = start_time
        self.stop_time = stop_time
        self.fmu_dict = {}
        self.stepsize_dict = {}
        self.powerflow_stepsize_dict = {}
        self.logging = logging
        self.exchange = exchange
        self.modify_signal = False
        self.modify_dict = {}
        self.signal_dict = {}
        self.signals_dict_new = {}
        self.init_dict = {}
        self.clean_up = clean_up
        self.all_outputs = []
        self.powerflow_dict = {}
        self.powerflow_outputs = {}
        self.powerflow_inputs = {}
        self.snapshots_dict = {}
        self.pf = ''
        self.interpolate_results = interpolate_results
        self.G = True
        self.csv_dict = {}
        self.variable_dict = {}
        
    def add_simulator(self, sim_type='', sim_name='', sim_loc='', inputs = [], outputs = [], step_size=1, **kwargs):
        if sim_type.lower() == 'fmu':
            if 'solver_name' in kwargs.keys():
                solver_name = kwargs['solver_name']
            else:
                solver_name = 'Cvode'
            if 'variable' in kwargs.keys():
                variable = kwargs['variable']
            else:
                variable=False
            if 'validate' in kwargs.keys():
                validate=kwargs['variable']
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
            
        elif sim_type.lower() == 'signal':
            if 'signal' in kwargs.keys():
                signal = kwargs['signal']
                self.add_signal(sim_name, signal)
            else:
                print(f"Signal needs to be specified to add signals. See documentation for defining signals.")
                print(f"Simulator {sim_name} not added.")
            
        elif sim_type.lower() == 'csv':
            self.add_csv(sim_name, sim_loc)
        else:
            print(f"Simulator type {sim_type} not recognized. Please use one of 'fmu', 'powerflow', 'signal', or 'csv' types.")
            print(f"Simulator {sim_name} not added")
        
    def add_powerflow(self, network_name, net_loc, inputs = [], outputs = [], pf = 'pf', step_size=900, logger = 'DEBUG'):
        self.stepsize_dict[network_name] = step_size
        assert network_name is not None, "No name specified for power flow model. Name must be specified."
        assert net_loc is not None, "No location specified for power flow model. Can't read without."
        try:
            network = pypsa_adapter(network_name, net_loc, inputs = inputs, outputs = outputs, logger_level = logger)
        except:
            network = pp_adapter(network_name, net_loc, inputs = inputs, outputs = outputs)
        self.powerflow_dict[network_name] = network
        self.pf = pf
        if self.logging:
            print("Added powerflow network '%s' to the world" %(network_name))

    
    def add_fmu(self, fmu_name, fmu_loc, step_size, inputs = [], outputs=[], exist=False, solver_name = 'Cvode', variable=False, **kwargs):
        if 'validate' in kwargs.keys():
            validate=kwargs['validate']
        else:
            validate=True
        self.stepsize_dict[fmu_name] = step_size
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
            self.fmu_dict[fmu_name] = fmu_temp
        elif fmi_type == 'ModelExchange':
            fmu_temp = FmuMeAdapter(fmu_loc,
                             instanceName=fmu_name,
                             step_size = step_size,
                             inputs = inputs,
                             outputs=outputs,
                             solver_name = solver_name,
                             validate=validate)
            self.fmu_dict[fmu_name] = fmu_temp
        
        self.all_outputs = [fmu_name + '.' + output for output in outputs]
        self.variable_dict[fmu_name] = variable
        if self.logging:
            print("Added a FMU '%s' to the world" %(fmu_name))
        
        
    
    def add_signal(self, signal_name, signal):
        signal_obj = signal_adapter(signal_name, signal)
        self.signal_dict[signal_name] = signal_obj
        if self.logging:
            print("Added a signal '%s' to the world" %(signal_name))
    
    def add_csv(self, csv_name, csv_location):
        df = pd.read_csv(csv_location)
        #analyse the df, and calculate step size
        if 'time' not in df.columns:
            print('No time column in csv file. Please convert csv file to required format. CSV not added.')
        else:
            autocorr = df.time.autocorr()
            if round(autocorr,4) != 1:
                print('energysim can only read csv with fixed time intervals. Current file does not have time stamps with fixed interval. Cant add csv.')
            else:
                dt = df.time[1] - df.time[0]
                self.stepsize_dict[csv_name] = dt
                self.csv_dict[csv_name] = df
                if self.logging:
                    print(f"Added csv: {csv_name} to World")
                
        
        
    def add_connections(self,connections={}):
        assert (len(self.fmu_dict) + len(self.powerflow_dict) > 0),"Cannot add connections when no FMUs specified!"
        if len(self.fmu_dict)+len(self.signal_dict)+len(self.powerflow_dict) == 1 and len(self.csv_dict) == 0:
            self.connections_between_fmus = {}
        else:
            self.connections_between_fmus = connections
        if self.logging:
            print("Added %i connections to the world" %(len(self.connections_between_fmus)))
    
    def get_fmu_name(self, name):
        if type(name).__name__ == 'str':
            return name.split('.')[0]
        if type(name).__name__ == 'tuple':
            return [x.split('.')[0] for x in name]
        
    def plot(self, plot_edge_labels=False):
        self.G=nx.DiGraph()
        for key, value in self.connections_between_fmus.items():
            
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
        nx.draw_networkx_nodes(self.G,pos, node_color='r', alpha=0.8, with_labels=True)
        nx.draw_networkx_edges(self.G,pos, alpha=0.8, edge_color='b')
        nx.draw_networkx_labels(self.G,pos)
        if plot_edge_labels:
            edge_labels = nx.get_edge_attributes(self.G,'name')
            nx.draw_networkx_edge_labels(self.G,pos, edge_labels=edge_labels)
        
        plt.show()
                
        
        
    def get_lcm(self):
        max_order = int(max([len(str(i).split('.')[1]) for i in self.stepsize_dict.values()]))
        new_list = 10**(max_order)*np.array(self.stepsize_dict.values())
        
        from functools import reduce    # need this line if you're using Python3.x
    
        def lcm(a, b):
            if a > b:
                greater = a
            else:
                greater = b
        
            while True:
                if greater % a == 0 and greater % b == 0:
                    lcm = greater
                    break
                greater += 1
        
            return lcm
        
        def get_lcm_for(your_list):
            return reduce(lambda x, y: lcm(x, y), your_list)
        
        final_lcm = get_lcm_for(new_list)
        self.big_time_step = final_lcm/10**max_order
        
        return self.big_time_step
    

    def perform_consistency_name_checks(self):
        a1 = self.csv_dict.keys()
        a2 = self.fmu_dict.keys()
        a3 = self.signal_dict.keys()
        a4 = self.powerflow_dict.keys()
        if len(set(a1).intersection(a2, a3, a4)) != 0:
            return False
        else:
            return True
        
    def create_df_for_pf(self, net_name, network):        
        colums_of_pf = ['time']
        for item in network.outputs:
            name = [net_name + '.' + item]
            colums_of_pf.extend(name)
        
        return pd.DataFrame(columns = colums_of_pf)
    
    def create_df_for_simulator(self, name, obj):
        columns_of_df = ['time']
        for item in obj.outputs:
            name = [name + '.' + item]
            columns_of_df.extend(name)
        return pd.DataFrame(columns = columns_of_df)
    
    def init(self):
        '''
        Initialises energysim object
        '''
        if self.logging:
            print("Simulation started..")
            print("Simulation status:\n")
        
        #create simulator list
        self.simulator_list = {}
        self.simulator_list.update(self.fmu_dict )
        self.simulator_list.update(self.powerflow_dict)
        
        
        self.create_results_dataframe()
        
        assert (len(self.fmu_dict) + len(self.powerflow_dict) > 0),"Cant run simulations when no simulators are specified!"
        if len(self.fmu_dict) + len(self.powerflow_dict) > 1:
            assert (len(self.connections_between_fmus) > 0),"Connections between FMUs are not specified!"
        
        self.res_dict = {}
        
        #initialise FMUs and their result dataframes
        for name, _fmu in self.fmu_dict.items():
            fmu_df = self.create_df_for_fmu(name)            
            self.res_dict[name] = fmu_df
            try:
                _fmu.setup()
                #check if initial vakues need to be set before initilisation
                if self.init_dict:
                    self.set_parameters(self.init_dict)
                    
                self.set_csv_signals(0)
                _fmu.init()
                print('Initialised FMU: %s'%(_fmu.instanceName))
            except:
                try:
                    print("Couldn't initialise %s. Trying again.." %(_fmu.instanceName))
                    _fmu.setup()
                    #check if initial vakues need to be set before initilisation
                    if self.init_dict:
                        self.set_parameters(self.init_dict)
                        
                    self.set_csv_signals(0)
                    _fmu.init()
                except:
                    print('Couldnt initialise %s. Simulation stopped.' %(_fmu.instanceName))
                    sys.exit()
        
        #initialise power flow and create its dataframe as well
        for net_name, network in self.powerflow_dict.items():
            net_df = self.create_df_for_pf(net_name, network)
            self.res_dict[net_name] = net_df
            self.set_csv_signals(0)
            network.init()
            print('Initialised powerflow network: %s'%(net_name))
        
        #determine when FMUs exchange information between themselves
        if self.exchange == 0 and len(self.fmu_dict.items()) + len(self.powerflow_dict.items())>1:
            self.final_tStep = self.get_lcm()
        elif self.exchange == 0 and len(self.fmu_dict.items()) + len(self.powerflow_dict.items()) == 1:
            self.final_tStep = list(self.stepsize_dict.values())[0]
        else:
            self.final_tStep = self.exchange
        
        
        if len(self.fmu_dict)+len(self.signal_dict)+len(self.powerflow_dict) > 1:
            self.do_exchange = True
        else:
            self.do_exchange = False
        
    def simulate(self, startTime=False, stopTime=False):
        '''
        Simulates the energysim object from startTime to stopTime with a step. If no input argument is specified, start, stop, and step times are derived from energysim iniitalization.
        '''
        check = self.perform_consistency_name_checks()
        if not check:
            print('Found more than one similar names for added fmu, signal, powerflow, or csv. Please use unique names for each add_xxx() method.')
            print('Exiting simulation.')
            sys.exit()
        
        if not startTime and not stopTime:
            flag = True
            dis = False
            startTime = self.start_time
            stopTime = self.stop_time
            self.init()
        else:
            flag = False
            dis = None
        assert (stopTime-startTime >= self.final_tStep), "difference between start and stop time > exchange value specified in world initialisation"
        total_steps = int((stopTime-startTime)/self.final_tStep)+1
        
        for time in tqdm(np.linspace(startTime, stopTime, total_steps), disable = dis):
            
            if self.do_exchange:
                self.exchange_values(time)
                
            for _fmu in self.fmu_dict.keys():                
                temp_time = time
                local_stop_time = min(temp_time + self.final_tStep, stopTime)
                while temp_time < local_stop_time:
                    self.set_csv_signals(temp_time)
                    temp_res = [temp_time] + self.fmu_dict[_fmu].getOutput()
                    self.res_dict[_fmu].loc[len(self.res_dict[_fmu].index)] = temp_res
                    
                    if _fmu in self.variable_dict.keys():                        
                        stepsize = self.get_step_time(self.stepsize_dict[_fmu], local_stop_time, temp_time, self.final_tStep, time)
                        try:
                            self.fmu_dict[_fmu].step_advanced(min(temp_time, local_stop_time), stepsize)
                            temp_time += stepsize             #self.stepsize_dict[_fmu]
                        except:
                            print(f'caught exception at time {temp_time}, changing step size from {stepsize} to {self.stepsize_dict[_fmu]}')
                            self.fmu_dict[_fmu].step_advanced(min(temp_time, local_stop_time), self.stepsize_dict[_fmu])
                            temp_time += self.stepsize_dict[_fmu]
                                                
                    else:
                        self.fmu_dict[_fmu].step(min(temp_time, local_stop_time))
                        temp_time += self.stepsize_dict[_fmu]
                
            for net_name, network in self.powerflow_dict.items():                
                if round(time%self.stepsize_dict[net_name]) == 0:
                    self.set_csv_signals(time)
                    network.step()
                    
                    temp_res = [time] + network.getOutput()
                    self.res_dict[net_name].loc[len(self.res_dict[net_name].index)] = temp_res
        
        if flag:
            self.clean_canvas()
            return self.results()
            
    def clean_canvas(self):            
        if self.clean_up:
            try:
                for _fmu in self.fmu_dict.values():
                    _fmu.cleanUp()
            except:
                print('Tried deleting temporary FMU files, but failed. Try manually.')
            for _net in self.powerflow_dict.values():
                _net.cleanUp()
        
    def sync(self, time):
        '''
        Manual call for exchange of values between simulators.
        '''
        self.exchange_values(time)
        
    def set_value(self, parameters, value):
        '''
        Must specify parameters and values in list format
        '''
        for item, val in zip(parameters, value):
            sim_name = item.split('.')[0]
            variable = item.replace(sim_name,'')[1:]
            self.simulator_list[sim_name].set_value([variable], [val])


    def get_value(self, parameter):
        '''
        Must specify parameter in a list format.
        '''
        outputs = []
        for item in parameter:
            sim_name = item.split('.')[0]
            variable = item.replace(sim_name,'')[1:]
            temp_value = self.simulator_list[sim_name].get_value([variable])
            outputs.append(temp_value[0])
        return outputs
    
    def get_new_time(self, stop, curr):        
        return (stop-curr)/2
        
    def get_step_time(self, step_size, local_stop_time, current_time, final_tStep, int_start_time):
        if current_time-int_start_time < 0.01 or local_stop_time-current_time < 0.01:
            new_step_size = 0.001
        else:
            new_step_size = final_tStep-0.01
        return new_step_size
    
    def results(self):
        if self.interpolate_results:
            try:
                self.process_results(self.res_dict)
                return self.results_dataframe
            except:
                return self.res_dict
        else:
            return self.res_dict
     
    
    def process_results(self, results_dict):
        if self.interpolate_results:
            from scipy.interpolate import interp1d
            temp1 = [(x, len(y)) for x, y in results_dict.items()]
            temp2 = dict(temp1)
            most_fmu_tsteps = max(temp2, key=temp2.get)
            self.new_tp = np.array(results_dict[most_fmu_tsteps].loc[:, 'time'])
            self.new_time_points = np.linspace(self.start_time, self.stop_time, len(np.asarray(self.new_tp <= self.stop_time).nonzero()[0]))
            
            self.results_dataframe.loc[:,'time'] = self.new_time_points
            
            #go through the results dict and create interpolated values.
            #define interpolation funciton
            for key, dataframe in results_dict.items():
                df_time = dataframe.loc[:,'time']
                df_time_temp = np.linspace(self.start_time, self.stop_time, len(np.asarray(df_time <= self.stop_time).nonzero()[0]))
                del df_time
                rem_columns = list(dataframe.columns)
                rem_columns.remove('time')
                for column_name in rem_columns:
                    y = np.array(dataframe.loc[:,column_name])[:len(df_time_temp)]
                    f = interp1d(df_time_temp, y, kind='previous')
                    new_datavalues = f(self.new_time_points)
                    self.results_dataframe.loc[:,column_name] = new_datavalues
        else:
            pass

        
    def create_df_for_fmu(self, fmu_name):
        colums_of_fmu = ['time']
        _fmu = self.fmu_dict[fmu_name]
        for item in _fmu.outputs:
            name = [_fmu.instanceName + '.' + item]
            colums_of_fmu.extend(name)
        
        fmu_df = pd.DataFrame(columns = colums_of_fmu)
        
        return fmu_df
    
        
    def set_csv_signals(self, t):
        for output_, input_ in self.connections_between_fmus.items():
            is_csv = True if type(output_).__name__ == 'str' and output_.split('.')[0] in self.csv_dict.keys() else False
            if is_csv:           # and type(output_).__name__ == 'str':
                
                if type(input_).__name__ == 'str':
                    input_ele_name = input_.split('.')[0]
                    is_powerflow = True if input_ele_name in self.powerflow_dict.keys() else False
                    is_fmu = True if input_ele_name in self.fmu_dict.keys() else False
                    if is_fmu:
                        input_fmu = self.fmu_dict[input_ele_name]
                        input_variable = input_.replace(input_ele_name,'')[1:]
                        
                        csv_name = output_.split('.')[0]
                        csv_variable = output_.split('.')[1]
                        csv_dt = self.stepsize_dict[csv_name]
                        temp_var = self.csv_dict[csv_name].at[int(t/csv_dt), csv_variable]
                        input_fmu.set_value([input_variable], [temp_var])
        
                    if is_powerflow:
                        network = self.powerflow_dict[input_ele_name]
                        network_variable = input_.replace(input_ele_name,'')[1:]
                        
                        csv_name = output_.split('.')[0]
                        csv_variable = output_.split('.')[1]
                        csv_dt = self.stepsize_dict[csv_name]
                        temp_var = self.csv_dict[csv_name].at[int(t/csv_dt), csv_variable]
                        network.set_value([network_variable], [temp_var])
                    
                if type(input_).__name__ == 'tuple':
                    for input__ in input_:
                        input_ele_name = input__.split('.')[0]
                        is_powerflow = True if input_ele_name in self.powerflow_dict.keys() else False
                        is_fmu = True if input_ele_name in self.fmu_dict.keys() else False
                        
                        if is_fmu:
                            input_fmu = self.fmu_dict[input_ele_name]
                            input_variable = input__.replace(input_ele_name,'')[1:]
                            
                            csv_name = output_.split('.')[0]
                            csv_variable = output_.split('.')[1]
                            csv_dt = self.stepsize_dict[csv_name]
                            temp_var = self.csv_dict[csv_name].at[int(t/csv_dt), csv_variable]
                            input_fmu.set_value([input_variable], [temp_var])
            
            
                        if is_powerflow:
                            network = self.powerflow_dict[input_ele_name]
                            network_variable = input__.replace(input_ele_name,'')[1:]
                            
                            csv_name = output_.split('.')[0]
                            csv_variable = output_.split('.')[1]
                            csv_dt = self.stepsize_dict[csv_name]
                            temp_var = self.csv_dict[csv_name].at[int(t/csv_dt), csv_variable]
                            
                            network.set_value([network_variable], [temp_var])
    
    def get_output_exchange(self, op_, t):
        ele_name = op_.split('.')[0]
        is_fmu = True if ele_name in self.fmu_dict.keys() else False
        is_pp_net = True if ele_name in self.powerflow_dict.keys() else False
        is_signal = True if ele_name in self.signal_dict.keys() else False
        is_csv = True if ele_name in self.csv_dict.keys() else False
        
        if is_fmu:
            output_fmu = self.fmu_dict[ele_name]
            output_variable =op_.replace(ele_name,'')[1:]
            temp_var = output_fmu.get_value([output_variable])[0]
        if is_pp_net:
            network = self.powerflow_dict[ele_name]
            output_variable =op_.replace(ele_name,'')[1:]
            temp_var = network.get_value([output_variable])[0]
        if is_signal:
            signal = self.signal_dict[ele_name]
            temp_var = signal.get_value(t)
        if is_csv:
            self.set_csv_signals(t)
            temp_var = 'its csv'
            
        if self.modify_dict:
            flag = True if op_ in self.modify_dict.keys() else False
            if flag:
                if len(self.modify_dict[op_]) == 1:
                    ret_output = temp_var*self.modify_dict[op_][0]
                elif len(self.modify_dict[op_]) == 2:
                    ret_output = temp_var*self.modify_dict[op_][0] + self.modify_dict[op_][1]
                else:
                    print('Unknown signal modification on output %s. Not applying modification.' %(op_))
                    ret_output = temp_var
            else:
                ret_output = temp_var
        else:
            ret_output = temp_var
        
        return ret_output
        
    
        
    def exchange_values(self, t):        
        for output_, input_ in self.connections_between_fmus.items():
            if type(output_).__name__ == 'str':
                temp_var = self.get_output_exchange(output_, t)
            elif type(output_).__name__ == 'tuple':
                temp_list = []
                for item in output_:
                    t_v = self.get_output_exchange(item, t)
                    temp_list.append(t_v)
                temp_var = sum(temp_list)
            
            if not type(temp_var).__name__ == 'str':
                if type(input_).__name__ == 'str':
                    ele_name = input_.split('.')[0]
                    pp_net = True if ele_name in self.powerflow_dict.keys() else False
                    fmu = True if ele_name in self.fmu_dict.keys() else False
                    
                    if fmu:
                        input_fmu = self.fmu_dict[ele_name]
                        input_variable = input_.replace(ele_name,'')[1:]
                        input_fmu.set_value([input_variable], [temp_var])
                    if pp_net:
                        input_pp = self.powerflow_dict[ele_name]
                        input_variable = input_.replace(ele_name,'')[1:]
                        input_pp.set_value([input_variable], [temp_var])
                    
                elif type(input_).__name__ == 'tuple':
                    for item in input_:
                        ele_name = item.split('.')[0]
                        pp_net = True if ele_name in self.powerflow_dict.keys() else False
                        fmu = True if ele_name in self.fmu_dict.keys() else False
                        
                        if fmu:
                            input_fmu = self.fmu_dict[ele_name]
                            input_variable = item.replace(ele_name,'')[1:]
                            input_fmu.set_value([input_variable], [temp_var])
                        if pp_net:
                            input_pp = self.powerflow_dict[ele_name]
                            input_variable = item.replace(ele_name,'')[1:]
                            input_pp.set_value([input_variable], [temp_var])
                    
    def create_results_dataframe(self):
        self.list_of_columns = ['time']
        for _fmu in self.fmu_dict.values():
            for item in _fmu.outputs:
                name = [_fmu.instanceName + '.' + item]
                self.list_of_columns.extend(name)
        self.results_dataframe = pd.DataFrame(columns = self.list_of_columns)
    
    
    def options(self, settings):
        assert (len(self.fmu_dict) + len(self.powerflow_dict) > 0),"Cannot add settings to world when no FMUs are specified!"
        
        if 'parameter_set' in settings.keys():
            self.set_parameters(settings['parameter_set'])
        
        if 'modify_signal' in settings.keys():
            self.modify_dict = settings['modify_signal']
        
        if 'init' in settings.keys():
            self.init_dict = settings['init']
    
    
    def set_parameters(self, parameter_dictionary):
        _sims = parameter_dictionary.keys()
        for _sim in _sims:
            parameter_list_derived = list(parameter_dictionary[_sim])[0]
            parameter_values_to_set = list(parameter_dictionary[_sim])[1]
            if _sim in self.fmu_dict.keys():
                self.fmu_dict[_sim].set_value(parameter_list_derived, parameter_values_to_set)
            if _sim in self.powerflow_dict.keys():
                self.powerflow_dict[_sim].set_value(parameter_list_derived, parameter_values_to_set)
            
        
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


        
        
