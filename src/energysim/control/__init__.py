# -*- coding: utf-8 -*-
"""
energysim.control: optimisation and model predictive control layer.

A controller is just another simulator: it reads measurements (plant
outputs, signals) and forecasts, solves an optimisation problem and
outputs set-points that the world connects to plant inputs.

* :class:`Controller` / :class:`RuleController` - base classes.
* :class:`MPCController` - receding-horizon economic MPC built from LP blocks
  (:mod:`energysim.control.blocks`), solved with HiGHS via scipy.
* :class:`OPFController` - pandapower optimal power flow as a controller.
* :class:`PyPSARollingHorizonController` - rolling-horizon PyPSA optimisation.
* Forecast providers: :class:`PerfectForecast`, :class:`PersistenceForecast`,
  :class:`CallableForecast`, :class:`NoisyForecast`.

scipy (``pip install energysim[control]``), pandapower and pypsa are
imported lazily, only when a solve is performed.
"""
from __future__ import annotations

from .base import Controller, RuleController, DIAGNOSTIC_OUTPUTS
from .forecast import (
    Forecast, PerfectForecast, PersistenceForecast, CallableForecast, NoisyForecast, as_forecast,
)
from .lp import (
    LinearProblem, LPResult, STATUS_OPTIMAL, STATUS_LIMIT, STATUS_INFEASIBLE, STATUS_UNBOUNDED,
    STATUS_ERROR, STATUS_NOT_SOLVED,
)
from .blocks import (
    Block, StorageBlock, ElectrolyzerBlock, HeatPumpBlock, BoilerBlock, GeneratorBlock, GridBlock,
    LoadBlock, RenewableBlock, PlanContext, block_from_component, register_block, CARRIER_UNITS,
)
from .mpc import MPCController, MPCPlan
from .opf import OPFController
from .pypsa_rolling import PyPSARollingHorizonController

__all__ = [
    "Controller", "RuleController", "DIAGNOSTIC_OUTPUTS",
    "Forecast", "PerfectForecast", "PersistenceForecast", "CallableForecast", "NoisyForecast", "as_forecast",
    "LinearProblem", "LPResult", "STATUS_OPTIMAL", "STATUS_LIMIT", "STATUS_INFEASIBLE", "STATUS_UNBOUNDED",
    "STATUS_ERROR", "STATUS_NOT_SOLVED",
    "Block", "StorageBlock", "ElectrolyzerBlock", "HeatPumpBlock", "BoilerBlock", "GeneratorBlock",
    "GridBlock", "LoadBlock", "RenewableBlock", "PlanContext", "block_from_component", "register_block",
    "CARRIER_UNITS", "MPCController", "MPCPlan", "OPFController", "PyPSARollingHorizonController",
]
