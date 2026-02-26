# -*- coding: utf-8 -*-
"""
MATLAB / GNU Octave adapter for energysim.

Integrates ``.m`` **functions** (not scripts) into an energysim
co-simulation.  The adapter auto-detects whether to use the MATLAB
Engine for Python or Oct2Py (GNU Octave bridge) and validates at
init time that the target ``.m`` file is a proper function with the
expected number of inputs and outputs.

Function contract
-----------------
The ``.m`` file **must** be a function (not a script) with the
following signature::

    function [out1, out2, ...] = func_name(in1, in2, ..., time)

- Input arguments correspond to the ``inputs`` list given to
  ``add_simulator``, **in the same order**, followed by ``time``
  (the current simulation time in seconds, appended automatically).
- Output arguments correspond to the ``outputs`` list, **in order**.

Example ``.m`` function::

    function [Q_thermal, COP] = heatpump(P_cmd, T_source, T_sink, time)
        COP_carnot = (T_sink + 273.15) / max(T_sink - T_source, 0.1);
        COP = 0.45 * COP_carnot;
        Q_thermal = P_cmd * COP;
    end

Requirements
------------
- **MATLAB**: ``matlab.engine`` Python package (ships with MATLAB).
- **Octave**: ``oct2py`` (``pip install oct2py``) + GNU Octave on PATH.
"""

from .base import SimulatorAdapter


class matlab_adapter(SimulatorAdapter):
    """Adapter for MATLAB / GNU Octave .m functions."""

    def __init__(self, func_name, script_loc, inputs=None,
                 outputs=None, engine='auto'):
        """
        Parameters
        ----------
        func_name : str
            Name of the ``.m`` function (without extension).
            Must match the filename, e.g. ``'heatpump'`` for
            ``heatpump.m``.
        script_loc : str
            Directory containing the ``.m`` file.
        inputs : list[str]
            Input variable names (order must match function args).
        outputs : list[str]
            Output variable names (order must match function returns).
        engine : str
            ``'auto'`` (default): try MATLAB first, fall back to
            Octave.
            ``'matlab'``: use MATLAB Engine only.
            ``'octave'``: use Oct2Py only.
        """
        self.func_name = func_name
        self.script_loc = script_loc
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.engine_pref = engine.lower()

        assert self.engine_pref in ('auto', 'matlab', 'octave'), (
            f"engine must be 'auto', 'matlab', or 'octave', "
            f"got '{engine}'."
        )

        # Populated by init()
        self._engine = None       # matlab.engine or Oct2Py instance
        self._backend = None      # 'matlab' or 'octave'
        self._input_values = {}   # name -> last set value
        self._output_values = {}  # name -> last computed value

    # ------------------------------------------------------------------
    # Life-cycle
    # ------------------------------------------------------------------

    def init(self):
        """Start engine, add script path, validate the function."""
        self._start_engine()
        self._add_path(self.script_loc)
        self._validate_function()

    def cleanup(self):
        """Shut down the engine."""
        if self._engine is not None:
            try:
                if self._backend == 'matlab':
                    self._engine.quit()
                else:
                    self._engine.exit()
            except Exception:
                pass
        self._engine = None

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def step(self, time):
        """Call the .m function with current inputs + time."""
        # Build positional arg list in declared input order + time
        args = [
            self._input_values.get(name, 0.0)
            for name in self.inputs
        ]
        args.append(float(time))

        n_out = len(self.outputs)

        if self._backend == 'matlab':
            results = self._call_matlab(args, n_out)
        else:
            results = self._call_octave(args, n_out)

        # Store outputs
        if n_out == 1:
            self._output_values[self.outputs[0]] = self._to_float(
                results
            )
        elif n_out > 1:
            for name, val in zip(self.outputs, results):
                self._output_values[name] = self._to_float(val)

    def set_value(self, parameters, values):
        """Store input values for the next step() call."""
        for name, val in zip(parameters, values):
            self._input_values[name] = float(val)

    def get_value(self, parameters, time=None):
        """Return the latest output values."""
        return [
            self._output_values.get(name, 0.0)
            for name in parameters
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_engine(self):
        """Start MATLAB or Octave engine based on preference."""
        if self.engine_pref in ('auto', 'matlab'):
            try:
                import matlab.engine  # noqa
                self._engine = matlab.engine.start_matlab()
                self._backend = 'matlab'
                return
            except (ImportError, Exception) as exc:
                if self.engine_pref == 'matlab':
                    raise RuntimeError(
                        "MATLAB Engine for Python not available. "
                        "Install it or use engine='octave'."
                    ) from exc

        # Fall through to Octave
        if self.engine_pref in ('auto', 'octave'):
            try:
                from oct2py import Oct2Py  # noqa
                self._engine = Oct2Py()
                self._backend = 'octave'
                return
            except (ImportError, Exception) as exc:
                raise RuntimeError(
                    "Neither MATLAB Engine nor oct2py is available. "
                    "Install one of them:\n"
                    "  MATLAB: install matlab.engine from your "
                    "MATLAB installation\n"
                    "  Octave: pip install oct2py  (+ GNU Octave "
                    "on PATH)"
                ) from exc

    def _add_path(self, path):
        """Add a directory to the engine's search path.

        If *path* points to a ``.m`` file, add its parent directory
        instead.
        """
        import os
        if os.path.isfile(path):
            path = os.path.dirname(path)
        if self._backend == 'matlab':
            self._engine.addpath(path, nargout=0)
        else:
            self._engine.addpath(path)

    def _validate_function(self):
        """Verify that func_name is a function with correct arg counts.

        Checks:
        1. The .m file is a function (not a script) — ``nargin``
           returns >= 0 for functions, errors for scripts.
        2. ``nargin`` == ``len(inputs) + 1`` (the +1 is for time).
        3. ``nargout`` == ``len(outputs)``.
        """
        expected_nargin = len(self.inputs) + 1   # +1 for time
        expected_nargout = len(self.outputs)

        try:
            if self._backend == 'matlab':
                actual_nargin = int(
                    self._engine.nargin(self.func_name)
                )
                actual_nargout = int(
                    self._engine.nargout(self.func_name)
                )
            else:
                actual_nargin = int(
                    self._engine.feval('nargin', self.func_name)
                )
                actual_nargout = int(
                    self._engine.feval('nargout', self.func_name)
                )
        except Exception as exc:
            raise RuntimeError(
                f"'{self.func_name}' does not appear to be a valid "
                f"MATLAB/Octave function. Make sure "
                f"'{self.func_name}.m' exists in '{self.script_loc}'"
                f" and is a function (not a script)."
            ) from exc

        # nargin returns -N for varargin functions — accept those
        if actual_nargin >= 0 and actual_nargin != expected_nargin:
            raise ValueError(
                f"Function '{self.func_name}' expects "
                f"{actual_nargin} input(s) but energysim will pass "
                f"{expected_nargin} (inputs={self.inputs} + time). "
                f"Update the function signature or the inputs list."
            )
        if actual_nargout >= 0 and actual_nargout != expected_nargout:
            raise ValueError(
                f"Function '{self.func_name}' returns "
                f"{actual_nargout} output(s) but energysim expects "
                f"{expected_nargout} (outputs={self.outputs}). "
                f"Update the function signature or the outputs list."
            )

    def _call_matlab(self, args, n_out):
        """Call function via MATLAB Engine."""
        if n_out == 0:
            self._engine.feval(self.func_name, *args, nargout=0)
            return None
        return self._engine.feval(
            self.func_name, *args, nargout=n_out
        )

    def _call_octave(self, args, n_out):
        """Call function via Oct2Py."""
        if n_out == 0:
            self._engine.feval(self.func_name, *args)
            return None
        return self._engine.feval(
            self.func_name, *args, nout=n_out
        )

    @staticmethod
    def _to_float(val):
        """Convert a MATLAB/Octave return value to a Python float."""
        try:
            if hasattr(val, '__iter__') and not isinstance(val, str):
                import numpy as np
                arr = np.asarray(val).flatten()
                return float(arr[0]) if len(arr) > 0 else 0.0
            return float(val)
        except (TypeError, ValueError, IndexError):
            return 0.0
