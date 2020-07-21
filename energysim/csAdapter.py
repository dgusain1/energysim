# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:13:38 2018

@author: digvijaygusain
"""
import os, sys
from fmpy import read_model_description, extract, dump
from fmpy.fmi2 import *
from fmpy.util import plot_result, download_test_file, auto_interval
from fmpy import supported_platforms
import shutil
from fmpy.simulation import Recorder, Input, apply_start_values
from random import random


class FmuCsAdapter():
    '''
    FMU CoSimulation adapter for energysim
    '''
    
    eps = 1.0e-13
    
    def __init__(self, fmu_location, 
                 instanceName=None, 
                 start_time=0, 
                 tolerance=1e-06, 
                 stop_time = 100, 
                 step_size = 1.0e-3, 
                 inputs = [], 
                 outputs = [],
                 show_fmu_info = False,
                 exist=False,
                 validate=True):
        assert (fmu_location is not None), "Must specify FMU location"
        self.fmu_location = fmu_location
        if instanceName is None:
            instanceID = int(random()*1000)
            self.instanceName = 'fmu'+str(instanceID)
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
        self.validate=validate
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
            self.fmu = FMU1Slave(guid = self.modelDescription.guid,
                                 unzipDirectory=self.unzipDir,
                                 modelIdentifier=self.modelDescription.coSimulation.modelIdentifier,
                                 instanceName=self.instanceName)
            self.fmu.instantiate(functions=callbacks)
        else:
            self.fmu = FMU2Slave(guid = self.modelDescription.guid,
                     unzipDirectory=self.unzipDir,
                     modelIdentifier=self.modelDescription.coSimulation.modelIdentifier,
                     instanceName=self.instanceName)
            self.fmu.instantiate(callbacks=callbacks)
        
        self.input = Input(self.fmu, self.modelDescription, None)
        
        

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
        return values_
        
    def reset(self):
        self.fmu.reset()
        
    def init(self): 
        if self.is_fmi1:
            self.fmu.initialize()
        else:
            self.fmu.setupExperiment(startTime=self.start_time, tolerance=self.tolerance)
            self.fmu.enterInitializationMode()
            self.fmu.exitInitializationMode()
    
    
    def set_inital_inputs(self, starting_values):
        from fmpy.simulation import apply_start_values
        apply_start_values(fmu = self.fmu,
                           model_description = self.modelDescription,
                           start_values = starting_values,
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
        #step ahead in time
        self.input.apply(time)
        if step_size is None:
            self.fmu.doStep(currentCommunicationPoint = time, communicationStepSize = self.output_interval)
        else:
            self.fmu.doStep(currentCommunicationPoint = time, communicationStepSize = step_size)
# TODO -           while a!=0:
#                self.fmu.setFMUstate(state)
##                print(f"Didnt work with stepsize = {step_size}, new stepsize = {step_size/2}.")
#                step_size = step_size/2
#                a = self.fmu.doStep(currentCommunicationPoint = time, communicationStepSize = step_size)
#            self.fmu.freeFMUstate(state)                    
#            print(f'returning to master: status = {a}, step size = {step_size}.')
#        return a, step_size
    
    def step_v2(self,time, stepsize):
        self.input.apply(time)
        return self.fmu.doStep(currentCommunicationPoint = time, communicationStepSize = stepsize)
    
    def step(self, time):
        #step ahead in time
        self.input.apply(time)
        return self.fmu.doStep(currentCommunicationPoint = time, communicationStepSize = self.output_interval)
    
    def cleanUp(self):
        self.fmu.terminate()
        self.fmu.freeInstance()
        
        shutil.rmtree(self.unzipDir)
    
    def simulate(self, timeout=180):
        from fmpy import simulate_fmu
            
        result = simulate_fmu(self.fmu_location, 
                              start_time = self.start_time, 
                              stop_time =self.stop_time, 
                              timeout = timeout, 
                              output_interval = self.output_interval, 
                              output = self.outputs)
        return result
        
