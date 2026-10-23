# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:44:38 2019

@author: digvijaygusain

Signal adapter for energysim.
"""
from __future__ import annotations


from .base import SimulatorAdapter
import warnings


class signal_adapter(SimulatorAdapter):
    """Output-only signal ``y = f(t)``.

    ``breakpoints`` (and the breakpoints of structured ``step`` / ``ramp`` /
    ``piecewise`` specs) are reported through :meth:`next_event_time`, so
    event-driven macro steps can land exactly on them.
    """

    def __init__(self, signal_name, signal, breakpoints=None):
        if not callable(signal):
            raise TypeError(
                f"Signal '{signal_name}' must be callable f(t); build string / dict specs with "
                f"energysim._safeexpr.build_signal (world.add_signal does this automatically)."
            )
        self.signal_name = signal_name
        self.signal = signal
        self.user_breakpoints = sorted(float(b) for b in (breakpoints or []))
        bps = self.user_breakpoints + list(getattr(signal, 'breakpoints', None) or [])
        self.breakpoints = sorted({float(b) for b in bps})

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
        if not parameters:
            return []
        result = self.signal(time)
        # Always return a list so the orchestrator can call list() on the result
        # without a TypeError when the signal lambda returns a plain scalar.
        if isinstance(result, (list, tuple)):
            return list(result)
        return [result]

    def next_event_time(self, time):
        """First breakpoint strictly after *time* (``inf`` if none)."""
        for b in self.breakpoints:
            if b > time + 1e-9:
                return b
        return float('inf')

    def get_available_variables(self):
        return {'inputs': [], 'outputs': ['y']}
