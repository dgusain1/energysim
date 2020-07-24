# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:32:24 2018

@author: digvijaygusain
"""
from fmpy import read_model_description, extract, dump
from fmpy.fmi1 import *
from fmpy.fmi1 import _FMU1
from fmpy.util import plot_result, download_test_file, auto_interval
import shutil, sys
from fmpy.simulation import Recorder, Input, apply_start_values
from random import random
from fmpy.fmi2 import *
import numpy as np

class FmuMeAdapter():
    '''
    FMU Model Exchange adapter for energysim
    '''
    
    def __init__(self, fmu_location, 
                 instanceName=None, 
                 start_time=0, 
                 tolerance=1e-06, 
                 stop_time = 100, 
                 step_size = 1.0e-3, 
                 inputs = [], 
                 outputs = [],
                 solver_name = 'Cvode',
                 show_fmu_info = False,
                 validate=True):

        assert (fmu_location is not None), "Must specify FMU location"
        self.fmu_location = fmu_location
        if instanceName is None:
            instanceID = int(random()*1000)
            self.instanceName = 'fmu'+str(instanceID)
            print('FMU instance created as: ' + self.instanceName)
        else:
            self.instanceName = instanceName
        self.tolerance = tolerance
        self.start_time = start_time
        self.stop_time = stop_time
        self.step_size = step_size
        self.inputs = inputs
        self.outputs = outputs
        self.solver_name = solver_name
        self.validate=validate
        if show_fmu_info:
            dump(self.fmu_location)

        self.setup()
        

        
    def setup(self):
        self.t_next = self.start_time
        self.unzipDir = extract(self.fmu_location)
        self.modelDescription = read_model_description(self.fmu_location, validate=self.validate)
        self.is_fmi1 = self.modelDescription.fmiVersion == '1.0'
        logger = printLogMessage
        if self.is_fmi1:
            callbacks = fmi1CallbackFunctions()
            callbacks.logger = fmi1CallbackLoggerTYPE(logger)
            callbacks.allocateMemory = fmi1CallbackAllocateMemoryTYPE(allocateMemory)
            callbacks.freeMemory = fmi1CallbackFreeMemoryTYPE(freeMemory)
            callbacks.stepFinished = None
        else:
            callbacks = fmi2CallbackFunctions()
            callbacks.logger = fmi2CallbackLoggerTYPE(logger)
            callbacks.allocateMemory = fmi2CallbackAllocateMemoryTYPE(allocateMemory)
            callbacks.freeMemory = fmi2CallbackFreeMemoryTYPE(freeMemory)
        
        #define var values for input and output variables
        self.vrs = {}
        for variable in self.modelDescription.modelVariables:
            self.vrs[variable.name] = [variable.valueReference, variable.type]
        
        if self.is_fmi1:
            self.fmu = FMU1Model(guid = self.modelDescription.guid,
                                 unzipDirectory=self.unzipDir,
                                 modelIdentifier=self.modelDescription.modelExchange.modelIdentifier,
                                 instanceName=self.instanceName)
            #instantiate FMU
            self.fmu.instantiate(functions=callbacks)
        else:
            self.fmu = FMU2Model(guid = self.modelDescription.guid,
                     unzipDirectory=self.unzipDir,
                     modelIdentifier=self.modelDescription.modelExchange.modelIdentifier,
                     instanceName=self.instanceName)
            #instantiate FMU
            self.fmu.instantiate(callbacks=callbacks)

    def set_value(self,parameterName,Value):
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
    

    def get_value(self,parameterName, time):
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
#        values_ = self.fmu.getReal(list(self.parameterVar))
        return values_
        
    def reset(self):
        self.fmu.reset()
        
    def init(self):
        #self.set_inital_inputs({})
        self.input = Input(self.fmu, self.modelDescription, None)
        if self.is_fmi1:
            self.fmu.setTime(self.start_time)
            self.input.apply(0)
            self.fmu.initialize()
        else:
            self.fmu.setupExperiment(startTime=self.start_time)
            self.fmu.enterInitializationMode()
            self.input.apply(0)
            self.fmu.exitInitializationMode()
    
            # event iteration
            self.fmu.eventInfo.newDiscreteStatesNeeded = fmi2True
            self.fmu.eventInfo.terminateSimulation = fmi2False
    
            while self.fmu.eventInfo.newDiscreteStatesNeeded == fmi2True and self.fmu.eventInfo.terminateSimulation == fmi2False:
                # update discrete states
                self.fmu.newDiscreteStates()
    
            self.fmu.enterContinuousTimeMode()        
        #self.fmu.initialize()
        self.set_solver()
        self.t_next = self.start_time
    
    def set_solver(self):
        solver_args = {
        'nx': self.modelDescription.numberOfContinuousStates,
        'nz': self.modelDescription.numberOfEventIndicators,
        'get_x': self.fmu.getContinuousStates,
        'set_x': self.fmu.setContinuousStates,
        'get_dx': self.fmu.getDerivatives,
        'get_z': self.fmu.getEventIndicators
        }
        
        if self.solver_name == 'Cvode':
            from fmpy.sundials import CVodeSolver
            self.solver = CVodeSolver(set_time=self.fmu.setTime,
                                 startTime=self.start_time,
                                 maxStep=(self.stop_time - self.start_time) / 50.,
                                 relativeTolerance=0.001,
                                 **solver_args)
            
            self.fixed_step = False
        
        if self.solver_name == 'Euler':
            self.solver = ForwardEuler(**solver_args)
            self.fixed_step = True

        
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
            
    
    def step(self, time, csStep=False):  
        
        if csStep:
            print('Variable step is only supported in cosimulation FMUs. Exiting simulation.')
            sys.exit()
        
        eps = 1.0e-13
        #step ahead in time
        if self.fixed_step:
            if time + self.step_size < self.stop_time + eps:
                self.t_next = time + self.step_size
#            else:
#                break
        else:
            if time + eps >= self.t_next:  # t_next has been reached
                # integrate to the next grid point
#                self.t_next = round(time / self.step_size) * self.step_size + self.step_size
                self.t_next = np.floor(time / self.step_size) * self.step_size + self.step_size
                if self.t_next < time + eps:
                    self.t_next += self.step_size
        
        #gets the time of input event
        t_input_event = self.input.nextEvent(time)
    
        # check for input event
        input_event = t_input_event <= self.t_next
    
        if input_event:
            self.t_next = t_input_event
        
        #check the time of next event.
        if self.is_fmi1:
            time_event = self.fmu.eventInfo.upcomingTimeEvent != fmi1False and self.fmu.eventInfo.nextEventTime <= self.t_next
        else:
            time_event = self.fmu.eventInfo.nextEventTimeDefined != fmi2False and self.fmu.eventInfo.nextEventTime <= self.t_next
            
#        time_event = self.fmu.eventInfo.upcomingTimeEvent != fmi1False and self.fmu.eventInfo.nextEventTime <= self.t_next
    
        if time_event and not self.fixed_step:
            self.t_next = self.fmu.eventInfo.nextEventTime
    
        if self.t_next - time > eps:
            # do one step
            state_event, time = self.solver.step(time, self.t_next)
        else:
            # skip
            time = self.t_next
            state_event = False
    
        # set the time
        self.fmu.setTime(time)
    
        # check for step event, e.g.dynamic state selection
        if self.is_fmi1:
            step_event = self.fmu.completedIntegratorStep()
        else:
            step_event, _ = self.fmu.completedIntegratorStep()
            step_event = step_event != fmi2False
            
        # handle events
        if input_event or time_event or state_event or step_event:
    
            #recorder.sample(time, force=True)
    
            if input_event:
                input.apply(time=time, after_event=True)
    
            # handle events
            if self.is_fmi1:
                self.fmu.eventUpdate()
            else:
                # handle events
                self.fmu.enterEventMode()

                self.fmu.eventInfo.newDiscreteStatesNeeded = fmi2True
                self.fmu.eventInfo.terminateSimulation = fmi2False

                # update discrete states
                while self.fmu.eventInfo.newDiscreteStatesNeeded != fmi2False and self.fmu.eventInfo.terminateSimulation == fmi2False:
                    self.fmu.newDiscreteStates()

                self.fmu.enterContinuousTimeMode()
    
            self.solver.reset(time)
    
    def terminate(self):
        self.fmu.terminate()
    
    def cleanUp(self):
        self.fmu.terminate()
        self.fmu.freeInstance()
        del self.solver
        shutil.rmtree(self.unzipDir)
    
    def simulate(self, timeout=180):
        from fmpy import simulate_fmu
            
        result = simulate_fmu(self.fmu_location, 
                              start_time = self.start_time, 
                              stop_time =self.stop_time, 
                              timeout = timeout, 
                              step_size = self.step_size, 
                              output = self.outputs)
        return result

class ForwardEuler(object):

    def __init__(self, nx, nz, get_x, set_x, get_dx, get_z):

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
        self._pprez = self.z.ctypes.data_as(POINTER(c_double))

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

        # check for state event
        self.prez[:] = self.z
        self.get_z(self._pz, self.z.size)
        stateEvent = np.any((self.prez * self.z) < 0)

        return stateEvent, tNext

    def reset(self, time):
        pass  # nothing to do
