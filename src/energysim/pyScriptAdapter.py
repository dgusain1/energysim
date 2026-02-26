# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:07:17 2019

@author: Digvijay

Python script adapter for energysim.

Usage
-----
``script_name`` should be ``module_name.function_name``.  The module
must be importable from ``script_loc`` (which is added to ``sys.path``).
The function is called each step with a dict of current input values
and must return a list of output values.
"""

import sys
import importlib

from .base import SimulatorAdapter


class py_adapter(SimulatorAdapter):

    def __init__(self, script_name, script_loc, inputs=None, outputs=None):
        self.script_name = script_name
        self.script_loc = script_loc
        self.inputs = inputs or []
        self.outputs = outputs or []

        # Resolve module + function once up-front (no exec!)
        module_name, self._func_name = script_name.rsplit('.', 1)

        if script_loc not in sys.path:
            sys.path.insert(0, script_loc)

        self._module = importlib.import_module(module_name)
        self._func = getattr(self._module, self._func_name)

        # Internal state
        self._input_dict = {}
        self._output_values = []

    # ------------------------------------------------------------------
    # SimulatorAdapter interface
    # ------------------------------------------------------------------

    def init(self):
        pass  # module already loaded in __init__

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            self._input_dict[p] = v

    def get_value(self, parameters, time=None):
        return self._output_values

    def step(self, time):
        result = self._func(self._input_dict)
        if isinstance(result, (list, tuple)):
            self._output_values = list(result)
        else:
            self._output_values = [result]

    def cleanup(self):
        self._module = None
        self._func = None
#
#    def setInput(self, inputValues):
#        pass
