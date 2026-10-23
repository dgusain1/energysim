# -*- coding: utf-8 -*-
"""
Template for user-defined external simulators.

Copy this file, rename the class, and implement the methods.  The ``world``
orchestrator calls, per run::

    configure(...)           # optional, receives start/stop/macro step
    init()
    advance(t0, t1)          # default implementation loops step(t) every step_size
    get_value(...) / set_value(...)   # at every macro/micro exchange
    cleanup()

``init()`` may be called again after ``cleanup()`` (a world can be simulated
more than once), so ``init()`` must (re)create any internal state.

``get_value`` must return a *list* with exactly one value per requested
parameter; this template returns ``0.0`` for variables that were never set,
so it works out of the box.
"""

from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Minimal external simulator template."""

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []
        self._values = {}
        # Create / load your model here (heavy objects belong in init())

    def init(self):
        # (Re)initialise the simulator state; called before every run.
        self._values = {}

    def set_value(self, parameters, values):
        # Set simulator inputs; ``parameters`` and ``values`` are lists of equal length.
        for name, val in zip(parameters, values):
            self._values[name] = val

    def get_value(self, parameters, time):
        # Return a list with one value per entry of ``parameters``.
        return [self._values.get(name, 0.0) for name in parameters]

    def step(self, time):
        # Advance the model by one micro-step (``self.step_size``) at ``time``.
        pass

    def cleanup(self):
        # Release any resources.
        pass

    def get_available_variables(self):
        return {'inputs': list(self.inputs), 'outputs': list(self.outputs)}
