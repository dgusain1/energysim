# -*- coding: utf-8 -*-
"""
energysim: Python package for cosimulation of multi-energy systems.

Provides a unified canvas (``world`` / ``World``) for orchestrating
heterogeneous energy system simulators including FMUs (Co-Simulation
and Model Exchange), pandapower, PyPSA, PowerFactory, MATLAB/Octave,
CSV profiles, signal generators, and custom Python scripts.
"""
from __future__ import annotations


from .world import world, World
from .base import (
    SimulatorAdapter,
    SimEntry,
    EnergysimError,
    SimulatorVariableError,
    SimulatorElementNotFoundError,
    ConnectionSpecError,
    ConnectionError,
    ValidationError,
    ValidationReport,
)
from .csv_adapter import csv_simulator
from .signalAdapter import signal_adapter
from .opendssAdapter import opendss_adapter
from .ParameterSweep import Sweep
from .ensemble import Ensemble, EnsembleResults, noisy_signal, NoisyInput
from .results import Results, register_kpi
from .dashboard import generate_dashboard

from .container import (
    VectorizedContainerAdapter,
    BatteryFleetAdapter,
    BuildingFleetAdapter,
    PVFleetAdapter,
    VectorizedFunctionAdapter,
)
from . import models
from . import networks
from . import control  # optimisation / MPC layer (scipy, pandapower, pypsa imported lazily)
from .models.common import ModelComponent, Curve1D, Surface2D

__version__ = "2.1.9.1"

__all__ = [
    "world",
    "World",
    "Sweep",
    "Ensemble",
    "EnsembleResults",
    "NoisyInput",
    "noisy_signal",
    "SimulatorAdapter",
    "SimEntry",
    "ModelComponent",
    "Curve1D",
    "Surface2D",
    "models",
    "networks",
    "control",
    "VectorizedContainerAdapter",
    "BatteryFleetAdapter",
    "BuildingFleetAdapter",
    "PVFleetAdapter",
    "VectorizedFunctionAdapter",
    "EnergysimError",
    "SimulatorVariableError",
    "SimulatorElementNotFoundError",
    "ConnectionSpecError",
    "ConnectionError",
    "ValidationError",
    "ValidationReport",
    "csv_simulator",
    "signal_adapter",
    "opendss_adapter",
    "generate_dashboard",
    "Results",
    "register_kpi",
    "__version__",
]

