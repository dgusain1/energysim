# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:59:11 2019

@author: digvijaygusain

class for parameter sweep. Takes in World(), parameter names and values as lists.
Outputs nice figures based on seaborn for visualisation.
"""

import numpy as np, sys
import matplotlib.pyplot as plt
from time import sleep

class Sweep():
    '''
    With Sweep, you can perform parameter sweeps on cosimulation World created by FMUWorld.
    Sweep takes in three arguments:
        1. Cosimulation World object
        2. Sensitivity information
        3. Kind
        
    Sensitivity information is the most important parameter here since it defines what parameters should swept through and in what range.
    It is given as dictionary in the following form.
    
    1. Cosimulation World Object: World created using FMUWorld.
    
    2. sensitivity_info = {'parameters':[param1, param2, .., paramN],
                            'bounds':[[p1_l, p1_u, N*],
                                      [p2_l, p2_u, N*],
                                      [p3_l, p3_u, N*],
                                      [pN_l, pN_u, N*]],
                            'output_parameter': 'output_parameter_identifier'
                            }
        'parameters' lists the parameter names in the World object. These should be given as 'simulator_name.parameter_name'. simulator_name is the fmu, or powerflow object name in the World. 'parameter_name' is the parameter name, This can be further specified as object.variable. The definition follows a pattern similar to defining connections dictionary between World simulators.
        
        'bounds' specifies the bounds on each parameter listed. This is given as a numpy matrix of dimension Nx2, where N is the number of parameters specified. Each parameter has lower and upper bound specified as [p1_l, p1_u]. Additionally, Sweep accepts 'N' as a third value in this list which specifies the number of points between the bounds. This can be set differently for each parameter bounds. If not specified, a default value of 10 is chosen.  
        
        'output_parameter' specifies which parameter to look record. It is specified as per FMUWorld naming conventions. This should also be specified while creating the World object.
        
    3. Kind: specifies what type of plot should be created. Sweep supports contour plots for parametric (if two parameters are specified), and (default) simple line plots. All plots in defualt are given as multi axes line plots where x axes are parameters being varied, and y axis is the output_parameter. If kind = 'parametric', Sweep plots the output_parameter as a heat map, with x and y axes being the two parameters specified. For kind = 'parametric', the number of parameters must be exactly two.
        
    Once parameter sweep is completed, plots are generated and a result object is created. This can be exported as csv file with SweepObject.export_to_csv(csv_location).
    
    '''    
    
    
    def __init__(self, _cosimulation_world, _sensitivity_data, kind = 'single'):
        self.world = _cosimulation_world
        self.simulators_dict = {}
        self.simulators_dict.update(self.world.fmu_dict)
        self.simulators_dict.update(self.world.powerflow_dict)
        if len(self.world.powerflow_dict) > 0:
            self.powerflow_exists = True
        else:
            self.powerflow_exists = False
        self.info = _sensitivity_data
        self.kind = kind
        
        
    def sweep(self):
        if self.kind == 'single':
            res = self.single_parameter_sweep()
        elif self.kind == 'parametric':
            self.parametric_sweep()
        else:
            print('FMUWorld.Sweep currently supports only single and parametric sweeps.')
        return res
        
    def single_parameter_sweep(self):
        self.parameters_list = self.info['parameters']
#        print(f'param list = {self.parameters_list}')
        self.bounds_matrix = self.info['bounds']
        self.op_param = self.info['output_parameter']
        
        #process output parameter
        op_sim_name = self.op_param.split('.')[0]
        
        #re-order bounds matrix
        new_matrix = []
        for item in self.bounds_matrix:
            if len(item) == 2:
                new_item = np.linspace(item[0], item[1], 10)
            elif len(item) == 3:
                new_item = np.linspace(item[0], item[1], item[2])
            else:
                print('Bounds matrix should have elements with either two or three values to define lower, upper bounds and number of values.')
            new_matrix.append(new_item)
            
        new_matrix = np.array(new_matrix)        
        
        #store original parameters before changing anything
#        if self.powerflow_exists:
#            self.world.simulate()
        
        self.original_values = {}
        for parameter in self.parameters_list:
            sim_name = parameter.split('.')[0]
            sim_p = parameter.replace(sim_name,'')[1:]
            sim = self.simulators_dict[sim_name]
            tmp = sim.get_value([sim_p])
            self.original_values[parameter] = tmp[0]
        
        self.sweep_res = {}
        count = 0
        #create options list
        for parameter, value_list in zip(self.parameters_list, new_matrix):
            sim_name = parameter.split('.')[0]
            sim_param = parameter.replace(sim_name,'')[1:]
            self.sweep_res[sim_name] = {'data': [],
                                         'value': [],
                                         'y-axis': sim_name}
#            print(f"Simulation {round(count/len(self.parameters_list),1)}% complete.")
            cc = 0
            for value in list(value_list) + [self.original_values[parameter]]:
                
                self.my_options = {'init':{}} if len(self.world.init_dict) < 1 else {'init':self.world.init_dict}
                if sim_name not in self.my_options['init'].keys():
                    self.my_options['init'][sim_name] = ([], [])
                self.my_options['init'][sim_name][0].append(sim_param)
                self.my_options['init'][sim_name][1].append(value)
                self.world.options(self.my_options)
                results = self.world.simulate()
                
                a1 = results[op_sim_name].loc[:,['time', self.op_param]]
                self.sweep_res[sim_name]['data'].append(a1)
                self.sweep_res[sim_name]['value'].append(value)
                
                cc+=1
                if cc != len(list(value_list) + [self.original_values[parameter]]):
                    plt.figure(count)
                    plt.plot(a1['time'], a1[self.op_param], label=f'{parameter}: {value}')
                    plt.title(f"Sensitivity of {self.op_param} to {parameter}")
                    legend = plt.legend()
                    legend.draggable(state=True)
                            
            count+=1
        plt.show()
        
        return self.sweep_res
        
            
            
            
            
        
        
        
        
                
    def evaluation_function(self):
                
        
        
        
        
        pass        
    
    def parametric_sweep(self):
        pass
        
        
