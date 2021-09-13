# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:56:32 2019

@author: digvijaygusain

pypsa FMUWorld adapter
"""

import pypsa, sys, logging
pypsa.pf.logger.setLevel(logging.CRITICAL)

class pypsa_adapter():
    
    def __init__(self, network_name, net_loc, inputs = [], outputs = [], logger_level = 'CRITICAL'):
#        pypsa.pf.logger.setLevel(getattr(logging, logger_level))
        self.network_name = network_name
        self.net_loc = net_loc
        self.network = pypsa.Network()
        self.network.import_from_csv_folder(net_loc)
        assert len(self.network.snapshots) == 1, "Only one snapshot is supported."
        self.new_inputs, self.new_outputs = self.process_powerflow_ipop(self.network, inputs, outputs)
        self.outputs = outputs

    
    def init(self):
        self.network.lpf()
        self.network.pf(use_seed=True)
    
    def set_value(self, parameters, values):
        for parameter, value in zip(parameters, values):
            ele_name, input_variable = parameter.split('.')
            assert input_variable in ['P', 'Q'], "Powerflow input variable not valid. Use P, Q to  define variables."
            if ele_name in list(self.network.generators.index):
                adder, residual = 'generators', input_variable.lower()+'_set'
            elif ele_name in list(self.network.loads.index):
                adder, residual = 'loads', input_variable.lower()+'_set'
            elif ele_name in list(self.network.storage_units.index):
                adder, residual = 'storage_units', input_variable.lower()+'_set'
            else:
                print(f'Could not find {ele_name} as a component in {self.network} network. Make sure element names are correctly specified in get_value()')
                sys.exit()
            
            getattr(self.network, adder).at[ele_name, residual] = value
    
    def get_value(self, parameters, time):
        temp_parameter_list = [x.split('.') for x in parameters]
        temp_output = []
        for ele_name, output_variable in temp_parameter_list:
            if output_variable.lower() == 'v':
                x = 'v_mag_pu'
            elif output_variable.lower() == 'va':
                x = 'v_ang'
            if ele_name in list(self.network.generators.index):
                adder, residual = 'generators_t', output_variable.lower()
            elif ele_name in list(self.network.loads.index):
                adder, residual = 'loads_t', output_variable.lower()
            elif ele_name in list(self.network.buses.index):
                adder, residual = 'buses_t', x
            elif ele_name in list(self.network.storage_units.index):
                adder, residual = 'storage_units_t', output_variable.lower()
            else:
                print(f'Could not find {ele_name} as a component in {self.network} network. Make sure element names are correctly specified in get_value()')
                sys.exit()
                
            temp_var = getattr(getattr(self.network, adder), residual).at[list(getattr(getattr(self.network, adder), residual).index)[0], str(ele_name)]
            temp_output.append(temp_var)
        
        return temp_output
        
    
    def getOutput(self):
        return [getattr(getattr(self.network, adder), residual).at[list(getattr(getattr(self.network, adder), residual).index)[0], str(ele)] for ele, adder, residual in self.new_outputs]
        
    
    def setInput(self, inputValues):
        pass
    
    def step(self):  
#        pypsa.pf.logger.setLevel(logging.CRITICAL)
        self.network.lpf()
        self.network.pf(use_seed=True)
    
    def process_powerflow_ipop(self, network, inputs, outputs):
            new_inputs = []
            new_outputs = []
            for item in inputs:
                ele_name, input_variable = item.split('.')
                assert input_variable in ['P', 'Q'], "Powerflow input variable not valid. Use P, Q to  define variables."
                #check in generators
                if ele_name in list(network.generators.index):
                    adder, residual = 'generators', input_variable.lower()+'_set'
                elif ele_name in list(network.loads.index):
                    adder, residual = 'loads', input_variable.lower()+'_set'
                elif ele_name in list(network.storage_units.index):
                    adder, residual = 'storage_units', input_variable.lower()+'_set'
                else:
                    print(f'Only Generator, load, and storage P, Q inputs are supported. Couldnt find {ele_name} in either loads or generators. Quitting simulation.')
                    sys.exit()
                new_inputs.append((ele_name, adder, residual))
#                print('done input analysing')
            
            for item in outputs:
                ele_name, output_variable = item.split('.')
                assert output_variable in ['P', 'Q', 'V', 'Va'], "Powerflow output variable not valid. Use P, Q, V, Va to  define variables."
                
                if output_variable == 'V':
                    x = 'v_mag_pu'
                elif output_variable == 'Va':
                    x = 'v_ang'
                    
                if ele_name in list(network.generators.index):
                    adder, residual = 'generators_t', output_variable.lower()
                elif ele_name in list(network.loads.index):
                    adder, residual = 'loads_t', output_variable.lower()
                elif ele_name in list(network.buses.index):
                    adder, residual = 'buses_t', x
                elif ele_name in list(network.storage_units.index):
                    adder, residual = 'storage_units_t', input_variable.lower()
                else:
                    print(f'Only Generator, load, storage, and bus P, Q, outputs are supported. Couldnt find {ele_name} in specified network. Quitting simulation.')
                    sys.exit()
                new_outputs.append((ele_name, adder, residual))
#                print('done output analysing')
                
            return new_inputs, new_outputs   
    
    def cleanUp(self):
        del self.network