# -*- coding: utf-8 -*-
"""
energysim: Python package for cosimulation of multi-energy systems.

Provides a unified canvas (``world`` / ``World``) for orchestrating
heterogeneous energy system simulators including FMUs (Co-Simulation
and Model Exchange), pandapower, PyPSA, PowerFactory, MATLAB/Octave,
CSV profiles, signal generators, and custom Python scripts.
"""

from .world import world, World
from .base import (
    SimulatorAdapter,
    SimEntry,
    EnergysimError,
    SimulatorVariableError,
    SimulatorElementNotFoundError,
    ConnectionError,
)
from .csv_adapter import csv_simulator
from .signalAdapter import signal_adapter
from .ParameterSweep import Sweep
from .utils import (
    convert_to_df,
    convert_to_csv,
    convert_hdf_to_dict,
    export_to_csv,
    record_data,
    create_results_recorder,
)
from .dashboard import generate_dashboard

__version__ = "2.1.9.1"

__all__ = [
    "world",
    "World",
    "Sweep",
    "SimulatorAdapter",
    "SimEntry",
    "EnergysimError",
    "SimulatorVariableError",
    "SimulatorElementNotFoundError",
    "ConnectionError",
    "csv_simulator",
    "signal_adapter",
    "generate_dashboard",
    "convert_to_df",
    "convert_to_csv",
    "convert_hdf_to_dict",
    "export_to_csv",
    "record_data",
    "create_results_recorder",
    "__version__",
]
