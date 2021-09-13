import pandas as pd, numpy as np

class csv_simulator():
    
    def __init__(self, sim_name, sim_loc, step_size = 900, outputs = [], delimiter=','):
        #initalize sim object here
        self.df = pd.read_csv(sim_loc, delimiter=delimiter)
        self.step_size = step_size
        #analyse the df, and calculate step size
        assert 'time' in self.df.columns, 'No time column in csv file. Please convert csv file to required format. CSV not added.'
        autocorr = round(self.df.time.autocorr(), 4)
        assert round(autocorr,4) == 1, 'energysim can only read csv with fixed time intervals. Current file does not have time stamps with fixed interval. Cant add csv simulator.'
        self.time_array = self.df.time.to_numpy()
        
    
    def init(self):
        pass
    
    def set_value(self, parameters, values):
        #this should set the simulator paramaters as values. Return cmd not reqd
        
        #remove the pass after specifying set_value
        pass
    
    def get_value(self, parameters, time):
        #this should return a list of values from simulator as a list corresponding to parameters
        tmp = []
        for ele in parameters:
            index = int(np.argwhere(self.time_array>time)[0] - 1)
            tmp.append(self.df.at[index, ele])
        return tmp
        
#        temp_var = self.csv_dict[csv_name].at[int(t/csv_dt), csv_variable]
        #remove the pass after specifying get_value. **Return reqd**
        pass    
    
    def step(self, time):
        #use the time variable (if needed) to step the simulator to t=time
        
        #return is not required. remove the pass command afterwards.
        pass
    