# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:44:38 2019

@author: digvijaygusain

Signal adapter for energysim.
"""

from .base import SimulatorAdapter
import warnings


class signal_adapter(SimulatorAdapter):
    def __init__(self, signal_name, signal):
        self.signal_name = signal_name
        self.signal = signal

    def init(self, *args, **kwargs):
        pass

    def step(self, time, *args, **kwargs):
        pass

    def set_value(self, *args, **kwargs):
        warnings.warn(
            f"Signal simulator '{self.signal_name}' is output-only."
            f" set_value() was called but values were discarded."
            f" Check your connections.",
            UserWarning, stacklevel=2
        )

    def get_value(self, parameters, time):
        result = self.signal(time)
        # Always return a list so the orchestrator can call list() on the result
        # without a TypeError when the signal lambda returns a plain scalar.
        if isinstance(result, (list, tuple)):
            return list(result)
        return [result]

#        if len(self.signal) == 1:
#            return [self.signal[0]]
#        elif len(self.signal) == 3:
#            return [self.signal[0] if time < self.signal[1] else self.signal[2]]
#        elif len(self.signal) == 4:
#            return [self.signal[0] if time < self.signal[1] or time > self.signal[2] else self.signal[3]]
#        else:
#            return [0]
