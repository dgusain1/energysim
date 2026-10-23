# -*- coding: utf-8 -*-
"""
Backward-compatibility alias for :mod:`energysim.models.catalog`.

The generic archetypes (``BESS_4h_2MW``, ``PEM_5MW``, ...) and ``PROVENANCE`` live in
``catalog``. The manufacturer-branded names re-exported here are deprecated aliases that
emit ``DeprecationWarning``; they are not backed by OEM datasheets.
"""

from .catalog import *  # noqa: F401,F403
from .catalog import __all__  # noqa: F401
