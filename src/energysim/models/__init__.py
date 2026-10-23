# -*- coding: utf-8 -*-
"""
Physical and Empirical Component Model Library for energysim.

Comprehensive non-power and sector-coupled components for microgrids,
BTM industrial energy systems, green hydrogen hubs, and district energy.
"""

from .common import (
    Curve1D,
    Surface2D,
    OperatingStateMachine,
    OperatingState,
    ModelComponent,
)

from .storage import (
    BatteryBESS,
    FlowBattery,
    Flywheel,
)

from .renewables import (
    PhysicsPV,
    WindTurbine,
    SolarThermalCollector,
)

from .hydrogen import (
    PEMElectrolyzer,
    AlkalineElectrolyzer,
    SOECElectrolyzer,
    H2Compressor,
    H2StorageTank,
    FuelCell_PEM,
    FuelCell_SOFC,
    MethanationReactor,
)

from .thermal import (
    GasTurbine_OCGT,
    CombinedCycle_CCGT,
    ReciprocatingGenset,
    CHP_SteamTurbine,
)

from .hvac_thermal import (
    IndustrialHeatPump,
    ElectricBoiler,
    ElectricChiller,
    AbsorptionChiller,
    ThermalStorageTank,
)

from .carbon import (
    CarbonCapture_Amine,
)

from .loads import (
    DataCenterLoad,
    IndustrialProcessSteam,
)

from . import catalog
from . import presets
archetypes = catalog

__all__ = [
    # Common
    "Curve1D",
    "Surface2D",
    "OperatingStateMachine",
    "OperatingState",
    "ModelComponent",
    # Storage
    "BatteryBESS",
    "FlowBattery",
    "Flywheel",
    # Renewables
    "PhysicsPV",
    "WindTurbine",
    "SolarThermalCollector",
    # Hydrogen
    "PEMElectrolyzer",
    "AlkalineElectrolyzer",
    "SOECElectrolyzer",
    "H2Compressor",
    "H2StorageTank",
    "FuelCell_PEM",
    "FuelCell_SOFC",
    "MethanationReactor",
    # Thermal
    "GasTurbine_OCGT",
    "CombinedCycle_CCGT",
    "ReciprocatingGenset",
    "CHP_SteamTurbine",
    # HVAC & Thermal
    "IndustrialHeatPump",
    "ElectricBoiler",
    "ElectricChiller",
    "AbsorptionChiller",
    "ThermalStorageTank",
    # Carbon
    "CarbonCapture_Amine",
    # Loads
    "DataCenterLoad",
    "IndustrialProcessSteam",
    # Equipment Catalog & Archetypes
    "catalog",
    "archetypes",
    "presets",
]
