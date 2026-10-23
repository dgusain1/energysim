# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:32:24 2018

@author: digvijaygusain

FMU Model Exchange adapter for energysim.

The adapter drives the FMU with an ODE solver (CVode through ``fmpy.sundials``
or the built-in fixed-step ``ForwardEuler``), handles time/state/step events
and is *re-runnable* (see :mod:`energysim.csAdapter` for the life-cycle).

Experiment times
----------------
``configure(start_time, stop_time, macro_step)`` is called by the world before
``init()``; the CVode ``maxStep`` is derived from the macro step (falling back
to the micro step when the adapter is used stand-alone).  Neither solver is
capped by ``stop_time``: simulations may run arbitrarily long.
"""
from __future__ import annotations

from ctypes import POINTER, c_double

try:
    from fmpy.fmi1 import FMU1Model, fmi1False
    from fmpy.fmi2 import FMU2Model, fmi2True, fmi2False
except ImportError:  # pragma: no cover
    FMU1Model = FMU2Model = fmi1False = fmi2True = fmi2False = None
import math

import numpy as np

from .base import EnergysimError
from .csAdapter import _FmuAdapterBase, Input, _FMPY_AVAILABLE  # noqa: F401

_SOLVERS = {'cvode': 'Cvode', 'euler': 'Euler'}


def _import_cvode_solver():
    """Import fmpy's CVodeSolver, with a workaround for x64 Python on ARM64 Windows.

    fmpy selects its bundled SUNDIALS libraries from ``platform.machine()``,
    which reports ARM64 for an (emulated) x64 interpreter on Windows-on-ARM
    where only the x86_64 libraries are shipped.
    """
    try:
        from fmpy.sundials import CVodeSolver
        return CVodeSolver
    except OSError as first_error:
        import sys
        import fmpy
        if (sys.platform.startswith('win') and 'AMD64' in sys.version
                and str(getattr(fmpy, 'platform_tuple', '')).startswith('aarch64')):
            saved = fmpy.platform_tuple
            fmpy.platform_tuple = 'x86_64-windows'
            try:
                from fmpy.sundials import CVodeSolver
                return CVodeSolver
            except OSError:
                pass
            finally:
                fmpy.platform_tuple = saved
        raise EnergysimError(
            "The CVode solver (SUNDIALS) could not be loaded on this platform "
            f"({first_error}). Use solver_name='Euler' instead."
        ) from first_error


class FmuMeAdapter(_FmuAdapterBase):
    '''
    FMU Model Exchange adapter for energysim
    '''

    _INTERFACE = 'modelExchange'

    def __init__(self, fmu_location,
                 instanceName=None,
                 start_time=0,
                 tolerance=1e-06,
                 stop_time=100,
                 step_size=1.0e-3,
                 inputs=None,
                 outputs=None,
                 solver_name='Cvode',
                 show_fmu_info=False,
                 exist=False,
                 validate=True):
        key = str(solver_name).lower()
        if key not in _SOLVERS:
            raise ValueError(
                f"Unknown solver_name '{solver_name}'. Must be one of 'Cvode' or 'Euler'.")
        self.solver_name = _SOLVERS[key]
        self._setup_common(fmu_location, instanceName, start_time, tolerance, stop_time,
                           step_size, inputs, outputs, show_fmu_info, exist, validate)
        self.step_size = step_size
        self.solver = None
        self.fixed_step = self.solver_name == 'Euler'
        self.t_next = start_time
        self._stop_limit = None
        self._terminated = False
        self._next_event_defined = False
        self._next_event_time = float('inf')

    # ------------------------------------------------------------------
    # instance
    # ------------------------------------------------------------------
    def _create_instance(self):
        md = self.modelDescription
        self.fmu_args = {'guid': md.guid,
                         'unzipDirectory': self.unzipDir,
                         'instanceName': self.instanceName,
                         'modelIdentifier': md.modelExchange.modelIdentifier}
        if self.exist:
            from fmpy.util import compile_dll
            import os
            self.fmu_args['libraryPath'] = compile_dll(
                model_description=md, sources_dir=os.path.join(self.unzipDir, 'sources'))
        cls = FMU1Model if self.is_fmi1 else FMU2Model
        self.fmu = cls(**self.fmu_args)
        if self.is_fmi1:
            self.fmu.instantiate(functions=self._callbacks)
            self.fmu.setTime(self.start_time)
        else:
            self.fmu.instantiate(callbacks=self._callbacks)
            self.fmu.setupExperiment(startTime=self.start_time)
        self.input = Input(self.fmu, md, None)
        self.solver = None
        self.t_next = self.start_time

    def _free_instance(self):
        self.solver = None
        super()._free_instance()

    # ------------------------------------------------------------------
    # initialisation
    # ------------------------------------------------------------------
    def init(self):
        if self._initialized:  # re-init without cleanup: start from scratch
            saved = dict(self._start_values)
            self._free_instance()
            self._ensure_instance()
            if saved:
                self.set_start_values(saved)
        else:
            self._ensure_instance()
        self._terminated = False
        self._next_event_defined = False
        self._next_event_time = float('inf')
        t0 = self.start_time
        try:
            if self.is_fmi1:
                self.input.apply(t0)
                res = self.fmu.initialize()
                self._read_event_info(res, fmi1=True)
            else:
                self.fmu.enterInitializationMode()
                self.input.apply(t0)
                self.fmu.exitInitializationMode()
                self._discrete_update()
                self.fmu.enterContinuousTimeMode()
            self._initialized = True
            self.set_solver()
        except Exception:
            self._initialized = False
            raise
        self.t_next = t0

    def _read_event_info(self, res, fmi1=False):
        """Store next-event info from an fmpy event call; return (iterate, terminate, changed)."""
        if isinstance(res, tuple):
            if fmi1:
                iteration_converged, _, state_values_changed, terminate, defined, t_ev = res
                self._next_event_defined, self._next_event_time = bool(defined), float(t_ev)
                return (not iteration_converged), bool(terminate), bool(state_values_changed)
            (needed, terminate, nominals_changed, values_changed, defined, t_ev) = res
            self._next_event_defined, self._next_event_time = bool(defined), float(t_ev)
            return bool(needed), bool(terminate), bool(nominals_changed or values_changed)
        # legacy fmpy: results live in fmu.eventInfo
        info = self.fmu.eventInfo
        if fmi1:
            self._next_event_defined = info.upcomingTimeEvent != fmi1False
            terminate = bool(info.terminateSimulation)
            return False, terminate, True
        self._next_event_defined = info.nextEventTimeDefined != fmi2False
        self._next_event_time = float(info.nextEventTime) if self._next_event_defined else float('inf')
        return (info.newDiscreteStatesNeeded == fmi2True,
                info.terminateSimulation == fmi2True, True)

    def _discrete_update(self):
        """Run the FMI 2.0 discrete-state iteration; return True if states changed."""
        changed = False
        needed, terminate = True, False
        while needed and not terminate:
            res = self.fmu.newDiscreteStates()
            needed, terminate, ch = self._read_event_info(res)
            changed |= ch
        if terminate:
            self._terminated = True
        return changed

    def set_solver(self):
        md = self.modelDescription
        fmu = self.fmu
        nx = md.numberOfContinuousStates
        solver_args = {
            'nx': nx,
            'nz': md.numberOfEventIndicators,
            'get_x': fmu.getContinuousStates,
            'set_x': fmu.setContinuousStates,
            'get_dx': fmu.getDerivatives,
            'get_z': fmu.getEventIndicators,
            'input': self.input,
        }
        if self.solver_name == 'Cvode':
            CVodeSolver = _import_cvode_solver()
            max_step = self.macro_step if self.macro_step else self.step_size
            self.solver = CVodeSolver(
                get_nominals=(fmu.getNominalContinuousStates if self.is_fmi1
                              else fmu.getNominalsOfContinuousStates),
                set_time=fmu.setTime,
                startTime=self.start_time,
                maxStep=float(max_step),
                relativeTolerance=self.tolerance,
                **solver_args)
            self.fixed_step = False
        else:
            self.solver = ForwardEuler(**solver_args)
            self.fixed_step = True

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------
    def step(self, time, csStep=False):
        """Integrate one micro-step starting at ``time``; returns the reached time."""
        if csStep:
            raise EnergysimError(
                'Variable step is only supported in cosimulation FMUs, not Model Exchange.'
            )
        if self._terminated:
            return time
        if self.fmu is None or self.solver is None:
            raise EnergysimError(f"FMU '{self.instanceName}' is not initialised (call init() first).")

        eps = 1.0e-13
        h = self.step_size
        if self.fixed_step:
            t_next = time + h
        else:  # next point on the micro-step grid
            t_next = (math.floor(time / h + 1e-9) + 1) * h
            if t_next < time + eps:
                t_next += h
        if self._stop_limit is not None and t_next > self._stop_limit:
            t_next = self._stop_limit

        t_input_event = self.input.nextEvent(time)
        input_event = t_input_event <= t_next
        if input_event:
            t_next = t_input_event

        time_event = (self._next_event_defined and time < self._next_event_time <= t_next + eps)
        if time_event:
            t_next = self._next_event_time

        if t_next - time > eps:
            state_event, _roots, new_time = self.solver.step(time, t_next)
        else:
            new_time, state_event = t_next, False

        self.fmu.setTime(new_time)
        self.input.apply(new_time, discrete=False)

        step_event = False
        if self.modelDescription.modelExchange.needsCompletedIntegratorStep:
            if self.is_fmi1:
                step_event = bool(self.fmu.completedIntegratorStep())
            else:
                step_event, terminate = self.fmu.completedIntegratorStep()
                step_event = bool(step_event)
                if terminate:
                    self._terminated = True
                    self.t_next = new_time
                    return new_time

        if input_event or time_event or state_event or step_event:
            if input_event and self.is_fmi1:
                self.input.apply(time=new_time, after_event=True)
            if self.is_fmi1:
                iterate = True
                while iterate and not self._terminated:
                    res = self.fmu.eventUpdate()
                    iterate, term, _ = self._read_event_info(res, fmi1=True)
                    if term:
                        self._terminated = True
            else:
                self.fmu.enterEventMode()
                if input_event:
                    self.input.apply(time=new_time, after_event=True)
                self._discrete_update()
                if not self._terminated:
                    self.fmu.enterContinuousTimeMode()
            self.solver.reset(new_time)

        self.t_next = new_time
        return new_time

    def terminate(self):
        if self.fmu is not None:
            self.fmu.terminate()

    # ------------------------------------------------------------------
    # advance() — hand whole intervals to the ODE solver
    # ------------------------------------------------------------------
    def advance(self, start_time: float, stop_time: float) -> None:
        """Advance the ME FMU from *start_time* to *stop_time* (never overshooting)."""
        t = start_time
        eps = 1e-13
        self._stop_limit = stop_time
        try:
            while t < stop_time - eps and not self._terminated:
                t = self._advance_once(t)
        finally:
            self._stop_limit = None

    def _advance_once(self, t):
        new_t = self.step(t)
        return new_t if new_t > t else t + self.step_size

    def advance_with_recording(self, start_time, stop_time, outputs):
        """Advance ME FMU while recording at each solver step."""
        rows = []
        t = start_time
        eps = 1e-13
        self._stop_limit = stop_time
        try:
            while t < stop_time - eps:
                rows.append([t] + list(self.get_value(outputs, t)) if outputs else [t])
                if self._terminated:
                    t = min(t + self.step_size, stop_time)
                    continue
                t = self._advance_once(t)
        finally:
            self._stop_limit = None
        return rows

    def simulate(self, timeout=180):
        from fmpy import simulate_fmu

        return simulate_fmu(self.fmu_location,
                            start_time=self.start_time,
                            stop_time=self.stop_time,
                            timeout=timeout,
                            step_size=self.step_size,
                            output=self.outputs)


class ForwardEuler(object):
    """Fixed-step explicit Euler integrator (same protocol as fmpy's)."""

    def __init__(self, nx, nz, get_x, set_x, get_dx, get_z, input=None):

        self.get_x = get_x
        self.set_x = set_x
        self.get_dx = get_dx
        self.get_z = get_z

        self.x = np.zeros(nx)
        self.dx = np.zeros(nx)
        self.z = np.zeros(nz)
        self.prez = np.zeros(nz)

        self._px = self.x.ctypes.data_as(POINTER(c_double))
        self._pdx = self.dx.ctypes.data_as(POINTER(c_double))
        self._pz = self.z.ctypes.data_as(POINTER(c_double))

        # initialize the event indicators
        self.get_z(self._pz, self.z.size)

    def step(self, t, tNext):

        # get the current states and derivatives
        self.get_x(self._px, self.x.size)
        self.get_dx(self._pdx, self.dx.size)

        # perform one step
        dt = tNext - t
        self.x += dt * self.dx

        # set the continuous states
        self.set_x(self._px, self.x.size)

        # check for state event (sign change of an event indicator)
        self.prez[:] = self.z
        self.get_z(self._pz, self.z.size)
        roots = np.zeros(self.z.size, dtype=np.int32)
        roots[(self.prez < 0) & (self.z >= 0)] = -1
        roots[(self.prez > 0) & (self.z <= 0)] = 1

        return bool(np.any(roots != 0)), roots, tNext

    def reset(self, time):
        # re-evaluate the event indicators after an event
        self.get_z(self._pz, self.z.size)
