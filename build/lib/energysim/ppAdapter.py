# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:07:17 2019

@author: Digvijay

Pandapower adapter for FMUWorld
"""


import pandapower as pp, sys

class pp_adapter():
    
    def __init__(self, network_name, net_loc, inputs = [], outputs = [], pf='pf'):
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
        self.new_inputs, self.new_outputs = self.process_powerflow_ipop(self.network, inputs, outputs)
        self.outputs = outputs
        self.pf = pf
        assert self.pf in ['pf', 'dcpf', 'opf', 'dcopf'], "PF option not recognized"
            

    def check_names(self, network):
        for n in [ 'load', 'sgen', 'ext_grid', 'bus']:
            check = getattr(getattr(network, n),"name").any()
            if not check:
                return False
        return True
    
    def init(self):
        if self.pf == 'pf':
            pp.runpp(self.network)
        elif self.pf == 'dcpf':
            pp.rundcpp(self.network)
        elif self.pf == 'opf':
            pp.runopp(self.network)
        else:
            pp.rundcopp(self.network)
        
            
    def check_input_var(self, ele, var):
        tmp = 'res_'+ele
        if var in list(getattr(getattr(self.network,tmp),'columns')):
            return True
        else:
            return False
        
    
    def set_value(self, parameters, values):
        '''
        Must specify parameters and values in list format
        '''
        for parameter, value in zip(parameters, values):
            ele_name, input_variable = parameter.split('.')
            if ele_name in list(self.network.gen.name):
                if input_variable in list(self.network.gen.columns):
                    adder, residual = 'gen', input_variable
                    self.network.gen.at[(self.network.gen.name==ele_name).idxmax(), input_variable] = value
                    getattr(self.network, adder).at[(getattr(self.network, adder).name == ele_name).idxmax(), residual] = value
                elif input_variable == 'cp1_eur_per_mw':
                    #index of load
                    idx = self.network.gen[self.network.gen['name']==ele_name].index.values[0]
                    tmp1 = self.network.poly_cost['et'] == 'gen'
                    tmp2 = self.network.poly_cost['element'] == idx
                    self.network.poly_cost.at[self.network.poly_cost[tmp1&tmp2].index.values[0],'cp1_eur_per_mw']=value
                else:
                    print(f"Parameter {parameter} not found in network {self.network_name}.")
                    sys.exit()
            elif ele_name in list(self.network.load.name):
                if input_variable in list(self.network.load.columns):
                    adder, residual = 'load', input_variable
                    getattr(self.network, adder).at[(getattr(self.network, adder).name == ele_name).idxmax(), residual] = value
                elif input_variable == 'cp1_eur_per_mw':
                    #index of load
                    idx = self.network.load[self.network.load['name']==ele_name].index.values[0]
                    tmp1 = self.network.poly_cost['et'] == 'load'
                    tmp2 = self.network.poly_cost['element'] == idx
                    self.network.poly_cost.at[self.network.poly_cost[tmp1&tmp2].index.values[0],'cp1_eur_per_mw']=value
                else:
                    print(f"Parameter {parameter} not found in network {self.network_name}.")
                    sys.exit()
            elif ele_name in list(self.network.sgen.name):
                if input_variable in list(self.network.sgen.columns):
                    adder, residual = 'sgen', input_variable
                    getattr(self.network, adder).at[(getattr(self.network, adder).name == ele_name).idxmax(), residual] = value
                elif input_variable == 'cp1_eur_per_mw':
                    #index of load
                    idx = self.network.sgen[self.network.sgen['name']==ele_name].index.values[0]
                    tmp1 = self.network.poly_cost['et'] == 'sgen'
                    tmp2 = self.network.poly_cost['element'] == idx
                    self.network.poly_cost.at[self.network.poly_cost[tmp1&tmp2].index.values[0],'cp1_eur_per_mw']=value

                else:
                    print(f"Parameter {parameter} not found in network {self.network_name}.")
                    sys.exit()
            else:
                print(f'Could not find {ele_name} as a component in {self.network} network. Make sure element names are correctly specified in get_value()')
                sys.exit()
            
    def get_value(self, parameters, *args):
        '''
        Must specify parameter in a list format.
        '''
        temp_parameter_list = [x.split('.') for x in parameters]
        temp_output = []
        for ele_name, output_variable in temp_parameter_list:
            if ele_name in list(self.network.gen.name):
                if output_variable in list(self.network.res_gen.columns):
                    adder, residual = 'res_gen', output_variable
                    temp_var = getattr(self.network, adder).at[(getattr(self.network, adder[4:]).name == ele_name).idxmax(), residual]
                    temp_output.append(temp_var)
                elif output_variable in list(self.network.gen.columns):
                    adder, residual = 'gen', output_variable
                    temp_var = getattr(self.network, adder).at[(getattr(self.network, adder).name == ele_name).idxmax(), residual]
                    temp_output.append(temp_var)
                elif output_variable == 'cp1_eur_per_mw':
                    #index of load
                    idx = self.network.gen[self.network.gen['name']==ele_name].index.values[0]
                    tmp1 = self.network.poly_cost['et'] == 'gen'
                    tmp2 = self.network.poly_cost['element'] == idx
                    temp_var = self.network.poly_cost.at[self.network.poly_cost[tmp1&tmp2].index.values[0],'cp1_eur_per_mw']
                    temp_output.append(temp_var)
                else:
                    print(f'Variable {output_variable} does not exist in pandapower network {self.network_name}.')
                    sys.exit()                    
            elif ele_name in list(self.network.load.name):
                if output_variable in list(self.network.res_load.columns):
                    adder, residual = 'res_load', output_variable
                    temp_var = getattr(self.network, adder).at[(getattr(self.network, adder[4:]).name == ele_name).idxmax(), residual]
                    temp_output.append(temp_var)
                elif output_variable in list(self.network.load.columns):
                    adder, residual = 'load', output_variable
                    temp_var = getattr(self.network, adder).at[(getattr(self.network, adder).name == ele_name).idxmax(), residual]
                    temp_output.append(temp_var)
                elif output_variable == 'cp1_eur_per_mw':
                    #index of load
                    idx = self.network.load[self.network.load['name']==ele_name].index.values[0]
                    tmp1 = self.network.poly_cost['et'] == 'load'
                    tmp2 = self.network.poly_cost['element'] == idx
                    temp_var = self.network.poly_cost.at[self.network.poly_cost[tmp1&tmp2].index.values[0],'cp1_eur_per_mw']
                    temp_output.append(temp_var)
                else:
                    print(f'Variable {output_variable} does not exist in pandapower network {self.network_name}.')
                    sys.exit()
            elif ele_name in list(self.network.bus.name):
                if output_variable in list(self.network.res_bus.columns):
                    adder, residual = 'res_bus', output_variable
                    temp_var = getattr(self.network, adder).at[(getattr(self.network, adder[4:]).name == ele_name).idxmax(), residual]
                    temp_output.append(temp_var)
                elif output_variable in list(self.network.bus.columns):
                    adder, residual = 'bus', output_variable
                    temp_var = getattr(self.network, adder).at[(getattr(self.network, adder).name == ele_name).idxmax(), residual]
                    temp_output.append(temp_var)
                elif output_variable == 'cp1_eur_per_mw':
                    #index of load
                    print(f'Bus variable cannot have cost parameters')
                    sys.exit()
                else:
                    print(f'Variable {output_variable} does not exist in pandapower network {self.network_name}.')
                    sys.exit()
            elif ele_name in list(self.network.sgen.name):
                if output_variable in list(self.network.res_sgen.columns):
                    adder, residual = 'res_sgen', output_variable
                    temp_var = getattr(self.network, adder).at[(getattr(self.network, adder[4:]).name == ele_name).idxmax(), residual]
                    temp_output.append(temp_var)
                elif output_variable in list(self.network.sgen.columns):
                    adder, residual = 'sgen', output_variable
                    temp_var = getattr(self.network, adder).at[(getattr(self.network, adder).name == ele_name).idxmax(), residual]
                    temp_output.append(temp_var)
                elif output_variable == 'cp1_eur_per_mw':
                    #index of load
                    idx = self.network.sgen[self.network.sgen['name']==ele_name].index.values[0]
                    tmp1 = self.network.poly_cost['et'] == 'sgen'
                    tmp2 = self.network.poly_cost['element'] == idx
                    temp_var = self.network.poly_cost.at[self.network.poly_cost[tmp1&tmp2].index.values[0],'cp1_eur_per_mw']
                    temp_output.append(temp_var)

                else:
                    print(f'Variable {output_variable} does not exist in pandapower network {self.network_name}.')
                    sys.exit()
            else:
                print(f'Could not find {ele_name} as a component in {self.network} network. Make sure element names are correctly specified in get_value()')
                sys.exit()
                

        
        return temp_output
        
    
    def getOutput(self):
        return self.get_value(self.outputs)
#        return [getattr(self.network, adder).at[(getattr(self.network, adder[4:]).name == ele).idxmax(), residual] for ele, adder, residual in self.new_outputs]
        
    
    def setInput(self, inputValues):
        pass
    
    def step(self, *args, **kwargs):  
        if self.pf == 'pf':
            a = pp.runpp(self.network)
        elif self.pf == 'dcpf':
            a = pp.rundcpp(self.network)
        elif self.pf == 'opf':
            a = pp.runopp(self.network)
        else:
            a = pp.rundcopp(self.network)
        return a
    
    def process_powerflow_ipop(self, network, inputs, outputs):
            new_inputs = []
            new_outputs = []
            for item in inputs:
                ele_name, input_variable = item.split('.')
                if ele_name in list(self.network.gen.name):
                    adder, residual = 'gen', input_variable
                elif ele_name in list(self.network.load.name):
                    adder, residual = 'load', input_variable
                elif ele_name in list(self.network.sgen.name):
                    adder, residual = 'sgen', input_variable
                else:
                    print(f'Couldnt find {ele_name}. Quitting simulation.')
                    sys.exit()
                new_inputs.append((ele_name, adder, residual))
            
            for item in outputs:
                ele_name, output_variable = item.split('.')
                if ele_name in list(self.network.gen.name):
                    if output_variable in list(self.network.res_gen.columns):
                        adder, residual = 'res_gen', output_variable
                    elif output_variable in list(self.network.gen.columns):
                        adder, residual = 'gen', output_variable
                    else:
                        print(f'Variable {output_variable} does not exist in pandapower network {self.network_name}.')
                        sys.exit()
                        
                elif ele_name in list(self.network.load.name):
                    if output_variable in list(self.network.res_load.columns):
                        adder, residual = 'res_load', output_variable
                    elif output_variable in list(self.network.load.columns):
                        adder, residual = 'load', output_variable
                    else:
                        print(f'Variable {output_variable} does not exist in pandapower network {self.network_name}.')
                        sys.exit()
    
                elif ele_name in list(self.network.bus.name):
                    if output_variable in list(self.network.res_bus.columns):
                        adder, residual = 'res_bus', output_variable
                    elif output_variable in list(self.network.bus.columns):
                        adder, residual = 'bus', output_variable
                    else:
                        print(f'Variable {output_variable} does not exist in pandapower network {self.network_name}.')
                        sys.exit()
                    
                elif ele_name in list(self.network.sgen.name):
                    if output_variable in list(self.network.res_sgen.columns):
                        adder, residual = 'res_sgen', output_variable
                    elif output_variable in list(self.network.sgen.columns):
                        adder, residual = 'sgen', output_variable
                    else:
                        print(f'Variable {output_variable} does not exist in pandapower network {self.network_name}.')
                        sys.exit()

                else:
                    print(f'Couldnt find {ele_name} in specified network. Quitting simulation.')
                    sys.exit()
                new_outputs.append((ele_name, adder, residual))
                
            return new_inputs, new_outputs   
     
    def cleanUp(self):
        del self.network
