#make necessary imports


class external_simulator():
    
    def __init__(self, inputs = [], outputs = [], **kwargs):
        self.inputs = inputs
        self.outputs = outputs
        
        #initalize sim object here
        
    
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
        
        #remove the pass after specifying get_value. **Return reqd**
        pass    
    
    def step(self, time):
        #use the time variable (if needed) to step the simulator to t=time
        
        #return is not required. remove the pass command afterwards.
        pass
    