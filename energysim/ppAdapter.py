# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:07:17 2019

@author: Digvijay

Pandapower adapter for FMUWorld
"""


import pandapower as pp, sys

class pp_adapter():
    
    def __init__(self, network_name, net_loc, inputs = [], outputs = []):
        '''
        Initialises the pandapower network adapter for energysim cosimulaiton object. 
        Specify the following:
                network_name: unique name for your network
                net_loc: location of the pandapower network file (specified as pickle file)
                inputs: list of all inputs. Specified as object_name.variable
                outputs: list of all output variables to record.
        '''
        self.network_name = network_name
        self.net_loc = net_loc
        self.network = pp.from_pickle(net_loc)
        all_names_exist = self.check_names(self.network)
        assert all_names_exist, "All load, generator, sgen, bus, and external grid elements must have names. Simulation stopped"
        print('Successfully imported network')
        self.new_inputs, self.new_outputs = self.process_powerflow_ipop(self.network, inputs, outputs)
        self.outputs = outputs

    def check_names(self, network):
        for n in [ 'load', 'sgen', 'ext_grid', 'bus']:
            check = getattr(getattr(network, n),"name").any()
            if not check:
                return False
        return True
    
    def init(self):
        pp.runpp(self.network)
    
    def set_value(self, parameters, values):
        '''
        Must specify parameters and values in list format
        '''
        for parameter, value in zip(parameters, values):
            ele_name, input_variable = parameter.split('.')
            assert input_variable in ['P', 'Q'], "Powerflow input variable not valid. Use P, Q to  define variables."
            if ele_name in list(self.network.gen.name):
                adder, residual = 'gen', 'p_mw' if input_variable == 'P' else 'q_mvar'
            elif ele_name in list(self.network.load.name):
                adder, residual = 'load', 'p_mw' if input_variable == 'P' else 'q_mvar'
            elif ele_name in list(self.network.sgen.name):
                adder, residual = 'sgen', 'p_mw' if input_variable == 'P' else 'q_mvar'
            else:
                print(f'Could not find {ele_name} as a component in {self.network} network. Make sure element names are correctly specified in get_value()')
                sys.exit()
            
            getattr(self.network, adder).at[(getattr(self.network, adder).name == ele_name).idxmax(), residual] = value
    
    def get_value(self, parameters):
        '''
        Must specify parameter in a list format.
        '''
        temp_parameter_list = [x.split('.') for x in parameters]
        temp_output = []
        for ele_name, output_variable in temp_parameter_list:
            if output_variable.lower() == 'v':
                x = 'vm_pu'
            elif output_variable.lower() == 'va':
                x = 'va_degree'
            if ele_name in list(self.network.gen.name):
                adder, residual = 'res_gen', 'p_mw' if output_variable == 'P' else 'q_mvar'
            elif ele_name in list(self.network.load.name):
                adder, residual = 'res_load', 'p_mw' if output_variable == 'P' else 'q_mvar'
            elif ele_name in list(self.network.bus.name):
                adder, residual = 'res_bus', x
            elif ele_name in list(self.network.sgen.name):
                adder, residual = 'res_sgen', 'p_mw' if output_variable == 'P' else 'q_mvar'
            else:
                print(f'Could not find {ele_name} as a component in {self.network} network. Make sure element names are correctly specified in get_value()')
                sys.exit()
                
            temp_var = getattr(self.network, adder).at[(getattr(self.network, adder[4:]).name == ele_name).idxmax(), residual]
            temp_output.append(temp_var)
        
        return temp_output
        
    
    def getOutput(self):
        return [getattr(self.network, adder).at[(getattr(self.network, adder[4:]).name == ele).idxmax(), residual] for ele, adder, residual in self.new_outputs]
        
    
    def setInput(self, inputValues):
        pass
    
    def step(self):  
        a = pp.runpp(self.network)
        return a
    
    def process_powerflow_ipop(self, network, inputs, outputs):
            new_inputs = []
            new_outputs = []
            for item in inputs:
                ele_name, input_variable = item.split('.')
                assert input_variable in ['P', 'Q'], "Powerflow input variable not valid. Use P, Q to  define variables."
                #check in generators
                if ele_name in list(network.gen.name):
                    adder, residual = 'gen', 'p_mw' if input_variable == 'P' else 'q_mvar'
                elif ele_name in list(network.load.name):
                    adder, residual = 'load', 'p_mw' if input_variable == 'P' else 'q_mvar'
                elif ele_name in list(network.sgen.name):
                    adder, residual = 'sgen', 'p_mw' if input_variable == 'P' else 'q_mvar'
                else:
                    print(f'Only Generator, load, and sgen P, Q inputs are supported for pandapower nets. Couldnt find {ele_name} in either loads or generators. Quitting simulation.')
                    sys.exit()
                new_inputs.append((ele_name, adder, residual))
            
            for item in outputs:
                ele_name, output_variable = item.split('.')
                assert output_variable in ['P', 'Q', 'V', 'Va'], "Powerflow output variable not valid. Use P, Q, V, Va to  define variables."
                
                if output_variable == 'V':
                    x = 'vm_pu'
                elif output_variable == 'Va':
                    x = 'va_degree'
                    
                if ele_name in list(network.gen.name):
                    adder, residual = 'res_gen', 'p_mw' if output_variable == 'P' else 'q_mvar'
                elif ele_name in list(network.load.name):
                    adder, residual = 'res_load', 'p_mw' if output_variable == 'P' else 'q_mvar'
                elif ele_name in list(network.bus.name):
                    adder, residual = 'res_bus', x
                elif ele_name in list(network.sgen.name):
                    adder, residual = 'res_sgen', 'p_mw' if output_variable == 'P' else 'q_mvar'
                else:
                    print(f'Only Generator, load, storage, and bus P, Q, outputs are supported. Couldnt find {ele_name} in specified network. Quitting simulation.')
                    sys.exit()
                new_outputs.append((ele_name, adder, residual))
                
            return new_inputs, new_outputs   
    
    def cleanUp(self):
        del self.network