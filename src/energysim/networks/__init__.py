# -*- coding: utf-8 -*-
"""
Network-lite layer for gas and district-heat networks.

A deliberately simple (non-hydraulic) graph model: producers, consumers,
storages and junctions joined by capacity-limited, lossy pipes, balanced each
step by a small LP (scipy/HiGHS) or, for radial networks without scipy, a
pure-Python simplex.  See :class:`EnergyNetwork`.
"""

from __future__ import annotations

from .base import EnergyNetwork, NetworkError
from .gas import GasNetwork
from .heat import HeatNetwork

__all__ = ["EnergyNetwork", "HeatNetwork", "GasNetwork", "NetworkError"]
