#make necessary imports


class external_simulator():
    
    def __init__(self, sim_name, sim_loc, inputs = [], outputs = [], step_size=1):
        self.sim_name = sim_name
        self.sim_loc = sim_loc
        self.inputs = inputs
        self.outputs = outputs
        self.step_size = step_size
            
    def init(self):
        #specify simulator initialization command
        
        #remove pass after initialization has been set
        pass
    
    def set_value(self, variable, value):
        #this should set the simulator paramaters as values. Return cmd not reqd
        
        #remove the pass after specifying set_value
        pass
    
    def get_value(self, variable, time):
        #this should return a list of values from simulator as a list corresponding to parameters
        
        **Return reqd**
        
        #remove the pass after specifying get_value. 
        pass    
    
    def step(self, time):
        #use the time variable (if needed) to step the simulator to t=time
        
        #return is not required. remove the pass command afterwards.
        pass
    