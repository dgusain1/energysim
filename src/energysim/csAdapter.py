# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:13:38 2018

@author: digvijaygusain

FMU Co-Simulation adapter for energysim.
"""
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
from random import random

from .base import SimulatorAdapter


class FmuCsAdapter(SimulatorAdapter):
    '''
    FMU CoSimulation adapter for energysim
    '''

    eps = 1.0e-13

    def __init__(self, fmu_location,
                 instanceName=None,
                 start_time=0,
                 tolerance=1e-06,
                 stop_time=100,
                 step_size=1.0e-3,
                 inputs=[],
                 outputs=[],
                 show_fmu_info=False,
                 exist=False,
                 validate=True):
        assert (fmu_location is not None), "Must specify FMU location"
        self.fmu_location = fmu_location
        if instanceName is None:
            instanceID = int(random() * 1000)
            self.instanceName = 'fmu' + str(instanceID)
            print(self.instanceName)
        else:
            self.instanceName = instanceName
        self.exist = exist
        self.tolerance = tolerance
        self.start_time = start_time
        self.stop_time = stop_time
        self.output_interval = step_size
        self.inputs = inputs
        self.outputs = outputs
        self.validate = validate
        if show_fmu_info:
            dump(self.fmu_location)
        self.setup()

    def setup(self):
        self.t_next = self.start_time
        if self.exist:
            self.unzipDir = self.fmu_location
        else:
            self.unzipDir = extract(self.fmu_location)
        self.modelDescription = read_model_description(self.fmu_location, validate=self.validate)
        self.is_fmi1 = self.modelDescription.fmiVersion == '1.0'

        logger = printLogMessage

        if self.is_fmi1:
            callbacks = fmi1CallbackFunctions()
            callbacks.logger = fmi1CallbackLoggerTYPE(logger)
            callbacks.allocateMemory = fmi1CallbackAllocateMemoryTYPE(calloc)
            callbacks.freeMemory = fmi1CallbackFreeMemoryTYPE(free)
            callbacks.stepFinished = None
        else:
            callbacks = fmi2CallbackFunctions()
            callbacks.logger = fmi2CallbackLoggerTYPE(logger)
            callbacks.allocateMemory = fmi2CallbackAllocateMemoryTYPE(calloc)
            callbacks.freeMemory = fmi2CallbackFreeMemoryTYPE(free)

        # define var values for input and output variables
        self.vrs = {}
        for variable in self.modelDescription.modelVariables:
            self.vrs[variable.name] = [variable.valueReference, variable.type]

        if self.is_fmi1:
            self.fmu = FMU1Slave(guid=self.modelDescription.guid,
                                 unzipDirectory=self.unzipDir,
                                 modelIdentifier=self.modelDescription.coSimulation.modelIdentifier,
                                 instanceName=self.instanceName)
            self.fmu.instantiate(functions=callbacks)

        else:
            self.fmu = FMU2Slave(guid=self.modelDescription.guid,
                                 unzipDirectory=self.unzipDir,
                                 modelIdentifier=self.modelDescription.coSimulation.modelIdentifier,
                                 instanceName=self.instanceName)
            self.fmu.instantiate(callbacks=callbacks)
            self.fmu.setupExperiment(startTime=self.start_time, tolerance=self.tolerance)

        self.input = Input(self.fmu, self.modelDescription, None)

    def set_start_values(self, init_dict):
        apply_start_values(self.fmu, self.modelDescription, init_dict, apply_default_start_values=False)

    def set_parameters(self, params):
        """Override ABC: FMU uses apply_start_values."""
        self.set_start_values(params)

    def set_value(self, parameterName, Value):
        '''
        Must specify parameters and values in list format
        '''
        for i, j in zip(parameterName, Value):

            if self.vrs[i][1] == 'Real':
                self.fmu.setReal([self.vrs[i][0]], [j])
            elif self.vrs[i][1] in ['Integer', 'Enumeration']:
                self.fmu.setInteger([self.vrs[i][0]], [j])
            elif self.vrs[i][1] == 'Boolean':
                if isinstance(j, str):
                    if j.lower() not in ['true', 'false']:
                        raise Exception('The value "%s" for variable "%s" could not be converted to Boolean' %
                                        (j, i))
                    else:
                        j = j.lower() == 'true'
                self.fmu.setBoolean([self.vrs[i][0]], [bool(j)])
            elif self.vrs[i][1] == 'String':
                self.fmu.setString([self.vrs[i][0]], [j])

    def get_value(self, parameterName, time):
        '''
        Must specify parameter in a list format.
        '''
        values_ = []
        for i in parameterName:
            if self.vrs[i][1] == 'Real':
                temp = self.fmu.getReal([self.vrs[i][0]])
            elif self.vrs[i][1] in ['Integer', 'Enumeration']:
                temp = self.fmu.getInteger([self.vrs[i][0]])
            elif self.vrs[i][1] == 'Boolean':
                temp = self.fmu.getBoolean([self.vrs[i][0]])
            elif self.vrs[i][1] == 'String':
                temp = self.fmu.getString([self.vrs[i][0]])

            values_.append(temp[0])
        return values_

    def reset(self):
        self.fmu.reset()

    def init(self):
        if self.is_fmi1:
            # self.input.apply(0)
            self.fmu.initialize()
        else:
            self.fmu.enterInitializationMode()
            # input.apply(0)
            self.fmu.exitInitializationMode()

    def set_inital_inputs(self, starting_values):
        from fmpy.simulation import apply_start_values
        apply_start_values(fmu=self.fmu,
                           model_description=self.modelDescription,
                           start_values=starting_values,
                           apply_default_start_values=False)

    def setInput(self, inputValues):
        self.inputVariables = []
        for i in self.inputs:
            self.inputVariables.append(self.vrs[i][0])
        self.fmu.setReal(list(self.inputVariables), list(inputValues))

    def getOutput(self):
        self.outputVariables = []
        for i in self.outputs:
            self.outputVariables.append(self.vrs[i][0])

        return self.fmu.getReal(list(self.outputVariables))

    def step_advanced(self, time, step_size=None):
        # step ahead in time
        self.input.apply(time)
        if step_size is None:
            self.fmu.doStep(currentCommunicationPoint=time, communicationStepSize=self.output_interval)
        else:
            self.fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
# TODO -           while a!=0:
#                self.fmu.setFMUstate(state)
# print(f"Didnt work with stepsize = {step_size}, new stepsize = {step_size/2}.")
#                step_size = step_size/2
#                a = self.fmu.doStep(currentCommunicationPoint = time, communicationStepSize = step_size)
#            self.fmu.freeFMUstate(state)
#            print(f'returning to master: status = {a}, step size = {step_size}.')
#        return a, step_size

    def step_v2(self, time, stepsize):
        self.input.apply(time)
        return self.fmu.doStep(currentCommunicationPoint=time, communicationStepSize=stepsize)

    def step(self, time):
        # step ahead in time
        self.input.apply(time)
        return self.fmu.doStep(currentCommunicationPoint=time, communicationStepSize=self.output_interval)

    def cleanup(self):
        self.fmu.terminate()
        self.fmu.freeInstance()

    # Backward compatibility alias
    cleanUp = cleanup

    # ------------------------------------------------------------------
    # Introspection & state management
    # ------------------------------------------------------------------

    def get_available_variables(self):
        """Return all FMU variable names from the modelDescription."""
        all_vars = list(self.vrs.keys())
        return {'inputs': list(self.inputs), 'outputs': list(self.outputs),
                'all': all_vars}

    def save_state(self):
        """Save FMU state for rollback (iterative coupling).

        Only works for FMUs with canGetAndSetFMUstate=true.
        """
        try:
            if not self.is_fmi1 and getattr(
                    self.modelDescription.coSimulation, 'canGetAndSetFMUstate', False):
                return self.fmu.getFMUstate()
        except Exception:
            pass
        return None

    def restore_state(self, state):
        """Restore a previously saved FMU state."""
        if state is not None:
            try:
                self.fmu.setFMUstate(state)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # advance() — override for variable-step support
    # ------------------------------------------------------------------
    def advance(self, start_time, stop_time):
        """Advance from *start_time* to *stop_time*.

        If ``self.variable_step`` is True, delegates to
        ``step_advanced`` with a computed step size; otherwise loops
        ``step()`` at ``self.output_interval`` intervals (the default
        ABC behaviour but using the CS-specific step method).
        """
        t = start_time
        eps = 1e-13
        while t < stop_time - eps:
            if getattr(self, 'variable_step', False):
                ss = self._variable_step_size(t, stop_time)
                self.step_advanced(min(t, stop_time), ss)
                t += ss
            else:
                self.step(min(t, stop_time))
                t += self.output_interval
            if t > stop_time + eps:
                t = stop_time

    def _variable_step_size(self, current_time, local_stop_time):
        """Heuristic for variable stepping (matches legacy get_step_time)."""
        macro = self.output_interval  # fallback to micro-step
        if hasattr(self, '_macro_step'):
            macro = self._macro_step
        # Near boundaries use a tiny step; otherwise use macro - small margin
        margin = 0.01
        if current_time < margin or local_stop_time - current_time < margin:
            return 0.001
        return macro - margin

#        shutil.rmtree(self.unzipDir)

    def simulate(self, timeout=180):
        from fmpy import simulate_fmu

        result = simulate_fmu(self.fmu_location,
                              start_time=self.start_time,
                              stop_time=self.stop_time,
                              timeout=timeout,
                              output_interval=self.output_interval,
                              output=self.outputs)
        return result
