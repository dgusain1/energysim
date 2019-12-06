# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:44:38 2019

@author: digvijaygusain
signal_adapter
"""

class signal_adapter():
    def __init__(self, signal_name, signal):
        self.signal_name = signal_name
        self.signal = signal
        
    
    def get_value(self, time):
        
        if len(self.signal) == 1:
            return self.signal[0]
        elif len(self.signal) == 3:
            return self.signal[0] if time < self.signal[1] else self.signal[2]
        elif len(self.signal) == 4:
            return self.signal[0] if time < self.signal[1] or time > self.signal[2] else self.signal[3]
        else:
            print('unrecognized signal format, using 0 values for signal %s.' %(self.signal_name))
            return 0