# -*- coding: utf-8 -*-
"""
Template for user-defined external simulators.

Copy this file, rename the class, and implement the abstract methods.
The ``world`` orchestrator will call ``init()`` → ``advance()`` (which
loops ``step()``) → ``cleanup()`` automatically.
"""

from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Minimal external simulator template."""

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []
        # Initialise your model here

    def init(self):
        # Specify simulator initialisation commands
        pass

    def set_value(self, parameters, values):
        # Set simulator parameters to the given values.
        # ``parameters`` and ``values`` are lists of equal length.
        pass

    def get_value(self, parameters, time):
        # Return a list of values corresponding to ``parameters``.
        pass

    def step(self, time):
        # Advance the model by one micro-step at ``time``.
        pass

    def cleanup(self):
        # Release any resources.
        pass
