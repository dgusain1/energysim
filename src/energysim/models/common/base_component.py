# -*- coding: utf-8 -*-
"""
Base ModelComponent for all energysim physical and empirical models.

Inherits from SimulatorAdapter, providing automated state tracking,
safe get_value / set_value variable routing, snapshotting for iterative
coupling, and micro-stepping loops.
"""

from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Sequence
from energysim.base import SimulatorAdapter


class ModelComponent(SimulatorAdapter):
    """
    Ergonomic base class for physical and empirical energy system models.

    Parameters
    ----------
    name : str, optional
        Unique component name.
    inputs : list[str], optional
        Declared input parameter names.
    outputs : list[str], optional
        Declared output parameter names.
    step_size : float, default 1.0
        Simulation micro-step size in seconds.
    """

    def __init__(
        self,
        name: str = "",
        inputs: Optional[Sequence[str]] = None,
        outputs: Optional[Sequence[str]] = None,
        step_size: float = 1.0,
        **kwargs,
    ):
        self.name = name
        self.inputs = list(inputs) if inputs is not None else []
        self.outputs = list(outputs) if outputs is not None else []
        self.step_size = float(step_size)

        # Internal state dictionary for continuous and discrete variables
        self._states: Dict[str, Any] = {}
        self._inputs_val: Dict[str, float] = {}
        self._outputs_val: Dict[str, Any] = {}
        self._state_snapshot: Optional[Dict[str, Any]] = None

    def register_variables(
        self,
        inputs: Optional[Sequence[str]] = None,
        outputs: Optional[Sequence[str]] = None,
    ) -> None:
        """Register default inputs and outputs."""
        if inputs:
            for inp in inputs:
                if inp not in self.inputs:
                    self.inputs.append(inp)
                default_val = getattr(self, inp, getattr(self, inp.lower(), 0.0))
                self._inputs_val.setdefault(inp, default_val)
        if outputs:
            for out in outputs:
                if out not in self.outputs:
                    self.outputs.append(out)
                self._outputs_val.setdefault(out, 0.0)

    def init(self) -> None:
        """Called once before simulation starts."""
        pass

    def set_value(self, parameters: Sequence[str], values: Sequence[Any]) -> None:
        """Receive inputs from world or coupled simulators."""
        for p, v in zip(parameters, values):
            self._inputs_val[p] = v
            # If the subclass has an attribute matching the parameter name, set it directly
            if hasattr(self, p):
                setattr(self, p, v)

    def get_value(self, parameters: Sequence[str], time: float) -> List[Any]:
        """Query component output variables."""
        results = []
        for p in parameters:
            if hasattr(self, p):
                val = getattr(self, p)
            elif p in self._outputs_val:
                val = self._outputs_val[p]
            elif p in self._states:
                val = self._states[p]
            elif p in self._inputs_val:
                val = self._inputs_val[p]
            else:
                val = 0.0
            results.append(val)
        return results

    def save_state(self) -> None:
        """Create a deep snapshot of model internal state."""
        self._state_snapshot = {
            "_states": copy.deepcopy(self._states),
            "_inputs_val": copy.deepcopy(self._inputs_val),
            "_outputs_val": copy.deepcopy(self._outputs_val),
        }

    def restore_state(self) -> None:
        """Restore internal state from previous snapshot."""
        if self._state_snapshot is not None:
            self._states = copy.deepcopy(self._state_snapshot["_states"])
            self._inputs_val = copy.deepcopy(self._state_snapshot["_inputs_val"])
            self._outputs_val = copy.deepcopy(self._state_snapshot["_outputs_val"])

    def step(self, time: float) -> None:
        """Advance the model by one micro-step."""
        self.step_model(self.step_size, time)

    def advance(self, start_time: float, stop_time: float) -> None:
        """Advance model across interval [start_time, stop_time]."""
        t = start_time
        dt = self.step_size
        while t < stop_time - 1e-9:
            step_dt = min(dt, stop_time - t)
            self.step_model(step_dt, t)
            t += step_dt

    def step_model(self, dt: float, time: float) -> None:
        """
        Subclasses implement this method with their physical/empirical logic.
        """
        raise NotImplementedError("Subclasses must implement step_model(dt, time).")

    def cleanup(self) -> None:
        """Release any resources."""
        pass
