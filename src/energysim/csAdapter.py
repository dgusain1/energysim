# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:13:38 2018

@author: digvijaygusain

FMU Co-Simulation adapter for energysim.

Life-cycle
----------
The adapter is *re-runnable*: ``cleanup()`` terminates and frees the FMU
instance and removes the temporary extraction directory; the next call that
needs the FMU (``set_parameters`` / ``set_value`` / ``init``) extracts and
instantiates it again.  Start values (``set_parameters``) must be applied
*before* ``init()``; adapters therefore declare ``SET_PARAMS_BEFORE_INIT = True``
and receive the experiment times through ``configure()``.
"""
from __future__ import annotations

try:
    from fmpy import read_model_description, extract, dump
    from fmpy.fmi1 import (
        FMU1Slave,
        fmi1CallbackFunctions, fmi1CallbackLoggerTYPE,
        fmi1CallbackAllocateMemoryTYPE, fmi1CallbackFreeMemoryTYPE,
    )
    from fmpy.fmi2 import (
        FMU2Slave,
        fmi2CallbackFunctions, fmi2CallbackLoggerTYPE,
        fmi2CallbackAllocateMemoryTYPE, fmi2CallbackFreeMemoryTYPE,
        printLogMessage, calloc, free,
    )
    from fmpy.simulation import Input, apply_start_values
    _FMPY_AVAILABLE = True
except ImportError:
    _FMPY_AVAILABLE = False
    read_model_description = extract = dump = None
    FMU1Slave = FMU2Slave = Input = apply_start_values = None
from random import random
import os
import shutil
import stat
import sys

from .base import SimulatorAdapter, SimulatorVariableError


def _rmtree(path):
    """Remove a directory tree.

    Deletes read-only files and retries briefly: on Windows a just-unloaded FMU
    shared library can stay locked for a few milliseconds.
    """
    import time

    def _make_writable_and_retry(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    for attempt in range(10):
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_make_writable_and_retry)
        else:
            shutil.rmtree(path, onerror=_make_writable_and_retry)
        if not os.path.exists(path):
            return
        time.sleep(0.05 * (attempt + 1))
    shutil.rmtree(path, ignore_errors=True)


def _require_fmpy():
    if not _FMPY_AVAILABLE:
        raise ImportError(
            "fmpy is required for FMU simulators. Install it using: "
            "pip install fmpy (or pip install energysim[fmu])"
        )


class _FmuAdapterBase(SimulatorAdapter):
    """Shared plumbing for the FMU CS and ME adapters (lazy, re-runnable)."""

    #: world applies start/parameter values *before* ``init()``
    SET_PARAMS_BEFORE_INIT = True
    #: 'coSimulation' or 'modelExchange' (attribute of the model description)
    _INTERFACE = 'coSimulation'

    def _setup_common(self, fmu_location, instanceName, start_time, tolerance, stop_time,
                      step_size, inputs, outputs, show_fmu_info, exist, validate):
        _require_fmpy()
        if fmu_location is None:
            raise ValueError("Must specify FMU location")
        self.fmu_location = fmu_location
        if instanceName is None:
            self.instanceName = 'fmu' + str(int(random() * 1000))
        else:
            self.instanceName = instanceName
        self.exist = exist
        self.tolerance = tolerance
        self.start_time = start_time
        self.stop_time = stop_time
        self.macro_step = None
        self.inputs = list(inputs or [])
        self.outputs = list(outputs or [])
        self.validate = validate
        if show_fmu_info:
            dump(self.fmu_location)

        # The model description is read exactly once.
        self.modelDescription = read_model_description(self.fmu_location, validate=self.validate)
        if self.modelDescription.fmiVersion not in ('1.0', '2.0'):
            raise ValueError(
                f"FMI version {self.modelDescription.fmiVersion} is not supported "
                f"(FMU '{fmu_location}'). energysim supports FMI 1.0 and 2.0."
            )
        self.is_fmi1 = self.modelDescription.fmiVersion == '1.0'
        if getattr(self.modelDescription, self._INTERFACE) is None:
            raise ValueError(
                f"FMU '{fmu_location}' does not provide a "
                f"{'Co-Simulation' if self._INTERFACE == 'coSimulation' else 'Model Exchange'} interface."
            )
        self.vrs = {v.name: [v.valueReference, v.type] for v in self.modelDescription.modelVariables}
        self._variability = {v.name: v.variability for v in self.modelDescription.modelVariables}

        self.fmu = None
        self.unzipDir = None
        self._owns_unzip = False
        self._callbacks = None
        self._initialized = False
        self._start_values = {}
        self._setup_times = None

    # ------------------------------------------------------------------
    # configuration from the world
    # ------------------------------------------------------------------
    def configure(self, start_time, stop_time, macro_step):
        """Receive experiment times from the world (called before init())."""
        self.start_time = start_time
        self.stop_time = stop_time
        self.macro_step = macro_step
        # An instance created for other experiment times must be re-created.
        if self.fmu is not None and not self._initialized and self._setup_times != start_time:
            self._free_instance()

    # ------------------------------------------------------------------
    # instance management
    # ------------------------------------------------------------------
    def _ensure_unzipped(self):
        if self.unzipDir is not None and os.path.isdir(self.unzipDir):
            return
        if getattr(self, 'exist', False) and self._INTERFACE == 'coSimulation':
            self.unzipDir = self.fmu_location
            self._owns_unzip = False
        else:
            self.unzipDir = extract(self.fmu_location)
            self._owns_unzip = True

    def _make_callbacks(self):
        if self.is_fmi1:
            cb = fmi1CallbackFunctions()
            cb.logger = fmi1CallbackLoggerTYPE(printLogMessage)
            cb.allocateMemory = fmi1CallbackAllocateMemoryTYPE(calloc)
            cb.freeMemory = fmi1CallbackFreeMemoryTYPE(free)
            cb.stepFinished = None
        else:
            cb = fmi2CallbackFunctions()
            cb.logger = fmi2CallbackLoggerTYPE(printLogMessage)
            cb.allocateMemory = fmi2CallbackAllocateMemoryTYPE(calloc)
            cb.freeMemory = fmi2CallbackFreeMemoryTYPE(free)
        return cb

    def _create_instance(self):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _ensure_instance(self):
        if self.fmu is not None:
            return
        try:
            self._ensure_unzipped()
            self._callbacks = self._make_callbacks()
            self._create_instance()
            self._setup_times = self.start_time
            self._initialized = False
        except Exception:
            self._free_instance()
            self._remove_unzip()
            raise

    def _free_instance(self):
        fmu, self.fmu = self.fmu, None
        if fmu is None:
            return
        if self._initialized:
            try:
                fmu.terminate()
            except Exception:
                pass
        try:
            fmu.freeInstance()
        except Exception:
            pass
        self._initialized = False

    def _remove_unzip(self):
        if self._owns_unzip and self.unzipDir and os.path.isdir(self.unzipDir):
            _rmtree(self.unzipDir)
        if self._owns_unzip:
            self.unzipDir = None
            self._owns_unzip = False

    def cleanup(self):
        """Terminate and free the FMU, remove the temp dir (the adapter can be init()-ed again)."""
        self._free_instance()
        self._remove_unzip()
        self._start_values = {}

    # Backward compatibility alias
    cleanUp = cleanup

    # ------------------------------------------------------------------
    # variables
    # ------------------------------------------------------------------
    def set_start_values(self, init_dict):
        self._ensure_instance()
        missing = apply_start_values(self.fmu, self.modelDescription, dict(init_dict))
        if missing:
            raise SimulatorVariableError(
                f"FMU '{self.instanceName}' has no settable variable(s) {sorted(missing)}.")
        self._start_values.update(init_dict)

    def set_parameters(self, params):
        """Override ABC: FMU uses apply_start_values (call before init())."""
        if params:
            self.set_start_values(params)

    def _vr(self, name):
        try:
            return self.vrs[name]
        except KeyError:
            raise SimulatorVariableError(
                f"Variable '{name}' not found in FMU '{self.instanceName}'."
            ) from None

    def set_value(self, parameterName, Value):
        '''
        Must specify parameters and values in list format
        '''
        self._ensure_instance()
        for name, val in zip(parameterName, Value):
            vr, typ = self._vr(name)
            if typ == 'Real':
                self.fmu.setReal([vr], [float(val)])
            elif typ in ('Integer', 'Enumeration'):
                self.fmu.setInteger([vr], [int(val)])
            elif typ == 'Boolean':
                if isinstance(val, str):
                    if val.lower() not in ('true', 'false'):
                        raise SimulatorVariableError(
                            f'The value "{val}" for variable "{name}" could not be converted to Boolean')
                    val = val.lower() == 'true'
                self.fmu.setBoolean([vr], [bool(val)])
            elif typ == 'String':
                self.fmu.setString([vr], [str(val)])
            else:
                raise SimulatorVariableError(f"Unsupported FMU variable type '{typ}' for '{name}'.")

    def get_value(self, parameterName, time=None):
        '''
        Must specify parameter in a list format.
        '''
        self._ensure_instance()
        values = []
        for name in parameterName:
            vr, typ = self._vr(name)
            if typ == 'Real':
                temp = self.fmu.getReal([vr])
            elif typ in ('Integer', 'Enumeration'):
                temp = self.fmu.getInteger([vr])
            elif typ == 'Boolean':
                temp = self.fmu.getBoolean([vr])
            elif typ == 'String':
                temp = self.fmu.getString([vr])
            else:
                raise SimulatorVariableError(f"Unsupported FMU variable type '{typ}' for '{name}'.")
            val = temp[0]
            if isinstance(val, bytes):
                val = val.decode('utf-8')
            if typ == 'Boolean':
                val = bool(val)
            elif typ in ('Integer', 'Enumeration'):
                val = int(val)
            values.append(val)
        return values

    def setInput(self, inputValues):
        self.set_value(self.inputs, inputValues)

    def getOutput(self):
        return self.get_value(self.outputs)

    def reset(self):
        if self.fmu is not None:
            self.fmu.reset()
            self._initialized = False

    def get_available_variables(self):
        """Variables from the model description.

        ``inputs``: causality input / parameter; ``outputs``: causality output;
        ``all``: every model variable (including local ones).
        """
        md = self.modelDescription
        inputs = [v.name for v in md.modelVariables if v.causality in ('input', 'parameter')]
        outputs = [v.name for v in md.modelVariables if v.causality == 'output']
        return {'inputs': inputs, 'outputs': outputs,
                'all': [v.name for v in md.modelVariables]}


class FmuCsAdapter(_FmuAdapterBase):
    '''
    FMU CoSimulation adapter for energysim
    '''

    eps = 1.0e-13
    _INTERFACE = 'coSimulation'

    def __init__(self, fmu_location,
                 instanceName=None,
                 start_time=0,
                 tolerance=1e-06,
                 stop_time=100,
                 step_size=1.0e-3,
                 inputs=None,
                 outputs=None,
                 show_fmu_info=False,
                 exist=False,
                 validate=True):
        self._setup_common(fmu_location, instanceName, start_time, tolerance, stop_time,
                           step_size, inputs, outputs, show_fmu_info, exist, validate)
        self.output_interval = step_size
        self.step_size = step_size
        self.variable_step = False

    def _create_instance(self):
        md = self.modelDescription
        cls = FMU1Slave if self.is_fmi1 else FMU2Slave
        self.fmu = cls(guid=md.guid,
                       unzipDirectory=self.unzipDir,
                       modelIdentifier=md.coSimulation.modelIdentifier,
                       instanceName=self.instanceName)
        if self.is_fmi1:
            self.fmu.instantiate(functions=self._callbacks)
        else:
            self.fmu.instantiate(callbacks=self._callbacks)
            self.fmu.setupExperiment(startTime=self.start_time, tolerance=self.tolerance)
        self.input = Input(self.fmu, md, None)
        self.t_next = self.start_time

    def init(self):
        if self._initialized:  # re-init without cleanup: start from scratch
            saved = dict(self._start_values)
            self._free_instance()
            self._ensure_instance()
            if saved:
                self.set_start_values(saved)
        else:
            self._ensure_instance()
        self.input.apply(self.start_time)
        if self.is_fmi1:
            self.fmu.initialize()
        else:
            self.fmu.enterInitializationMode()
            self.fmu.exitInitializationMode()
        self._initialized = True
        self.t_next = self.start_time

    def set_inital_inputs(self, starting_values):
        self.set_start_values(starting_values)

    def step_advanced(self, time, step_size=None):
        self.step_v2(time, self.output_interval if step_size is None else step_size)

    def step_v2(self, time, stepsize):
        self.input.apply(time)
        return self.fmu.doStep(currentCommunicationPoint=time, communicationStepSize=stepsize)

    def step_partial(self, time, dt):
        """Shorter final communication step (event-driven macro steps)."""
        return self.step_v2(time, dt)

    def step(self, time):
        # step ahead in time
        self.input.apply(time)
        return self.fmu.doStep(currentCommunicationPoint=time, communicationStepSize=self.output_interval)

    # ------------------------------------------------------------------
    # advance() — override for variable-step support
    # ------------------------------------------------------------------
    def advance(self, start_time, stop_time):
        """Advance from *start_time* to *stop_time*.

        With ``variable_step`` the step size follows a heuristic;
        otherwise ``doStep`` is called every ``output_interval`` (the
        final step is shortened so the FMU never overshoots *stop_time*).
        """
        t = start_time
        eps = 1e-13
        while t < stop_time - eps:
            if getattr(self, 'variable_step', False):
                ss = self._variable_step_size(t, stop_time)
            else:
                ss = min(self.output_interval, stop_time - t)
            self.step_v2(t, ss)
            t += ss

    def _variable_step_size(self, current_time, local_stop_time):
        """Heuristic for variable stepping (matches legacy get_step_time)."""
        macro = self.macro_step if self.macro_step else self.output_interval
        remaining = local_stop_time - current_time
        if remaining <= 1e-12:
            return 0.001
        # Near boundaries use a tiny step; otherwise use macro - small margin
        margin = 0.01
        if current_time < margin or remaining < margin:
            return min(0.001, remaining)
        step = macro - margin
        if step <= 0:
            step = min(macro, remaining)
        return min(max(step, 1e-6), remaining)

    def simulate(self, timeout=180):
        from fmpy import simulate_fmu

        return simulate_fmu(self.fmu_location,
                            start_time=self.start_time,
                            stop_time=self.stop_time,
                            timeout=timeout,
                            output_interval=self.output_interval,
                            output=self.outputs)
