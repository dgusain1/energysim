# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:07:17 2019

@author: Digvijay

Python script adapter for energysim
"""


import sys
class py_adapter():
    
    def __init__(self, script_name, script_loc, inputs = [], outputs = []):
        '''
        script_name should be something1.something2
        here: something1 is filename.py, and something2 is the function that is called from filename.
        something1.py should contain a python function something2 that can be imported as from something1 import something2.
        '''
        self.script_name = script_name
        self.script_loc = script_loc
        self.inputs = inputs
        self.outputs = outputs
        self.init()

    def init(self):
        sys.path.append(self.script_loc)
        a = self.script_name.split('.')[0]
        exec('import ' + a)
        
    
    def set_value(self, parameters, values):
        self.my_dict = dict(zip(parameters, values))

    
    def get_value(self, parameters):
        return self.outvalues
    
    def step(self):
        exec('import ' + self.script_name.split('.')[0])
#        exec('import ' + self.script_name.split('.')[0])
        self.outvalues = ''
#        print(f'executing: {self.outvalues} = {a}{self.script_name.split('.')[1]}({self.my_dict})')
#        self.outvalues = getattr(self.script_name.split('.')[0],f'{self.my_dict}')
        exec(f'self.outvalues ={self.script_name}({self.my_dict})')
#        pass
#    
    def getOutput(self):
        return self.outvalues
    
    
#    def set_value(self, parameters, values):
#        for parameter, value in zip(parameters, values):
#            ele_name, input_variable = parameter.split('.')
#            assert input_variable in ['P', 'Q'], "Powerflow input variable not valid. Use P, Q to  define variables."
#            if ele_name in list(self.network.gen.name):
#                adder, residual = 'gen', 'p_mw' if input_variable == 'P' else 'q_mvar'
#            elif ele_name in list(self.network.load.name):
#                adder, residual = 'load', 'p_mw' if input_variable == 'P' else 'q_mvar'
#            elif ele_name in list(self.network.sgen.name):
#                adder, residual = 'sgen', 'p_mw' if input_variable == 'P' else 'q_mvar'
#            else:
#                print(f'Could not find {ele_name} as a component in {self.network} network. Make sure element names are correctly specified in get_value()')
#                sys.exit()
#            
#            getattr(self.network, adder).at[(getattr(self.network, adder).name == ele_name).idxmax(), residual] = value
#    
#    def get_value(self, parameters):
#        temp_parameter_list = [x.split('.') for x in parameters]
#        temp_output = []
#        for ele_name, output_variable in temp_parameter_list:
#            if output_variable.lower() == 'v':
#                x = 'vm_pu'
#            elif output_variable.lower() == 'va':
#                x = 'va_degree'
#            if ele_name in list(self.network.gen.name):
#                adder, residual = 'res_gen', 'p_mw' if output_variable == 'P' else 'q_mvar'
#            elif ele_name in list(self.network.load.name):
#                adder, residual = 'res_load', 'p_mw' if output_variable == 'P' else 'q_mvar'
#            elif ele_name in list(self.network.bus.name):
#                adder, residual = 'res_bus', x
#            elif ele_name in list(self.network.sgen.name):
#                adder, residual = 'res_sgen', 'p_mw' if output_variable == 'P' else 'q_mvar'
#            else:
#                print(f'Could not find {ele_name} as a component in {self.network} network. Make sure element names are correctly specified in get_value()')
#                sys.exit()
#                
#            temp_var = getattr(self.network, adder).at[(getattr(self.network, adder[4:]).name == ele_name).idxmax(), residual]
#            temp_output.append(temp_var)
#        
#        return temp_output
#        
#    
#    def getOutput(self):
##        a1 = []
##        for item in self.new_outputs:
##            a1.append(self.get_value(item))
##        return a1    
#        return [getattr(self.network, adder).at[(getattr(self.network, adder[4:]).name == ele).idxmax(), residual] for ele, adder, residual in self.new_outputs]
#        
#    
#    def setInput(self, inputValues):
#        pass
    
    
        
    
