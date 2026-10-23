# -*- coding: utf-8 -*-
"""Shared constants and helpers of the world package."""

from __future__ import annotations
from typing import Any, Dict

import inspect
import itertools
import logging as _logging
from fractions import Fraction
import numpy as np


logger = _logging.getLogger('energysim')

ITERATIVE_REMOVED_MSG = (
    "iterative coupling was removed in 3.0; use 'jacobi' or 'gauss-seidel', "
    "reduce t_macro, or merge coupled components"
)
_COUPLINGS = ('jacobi', 'gauss-seidel')
_EXTRAPOLATIONS = ('zero-order', 'linear')

#: Simulator types that execute arbitrary user code / external programs and
#: are therefore refused when a scenario is loaded with ``safe=True``.
UNSAFE_SIM_TYPES = frozenset({'external', 'script', 'pyscript', 'matlab', 'powerfactory'})

#: Recognised keys for :meth:`world.options`.
OPTION_KEYS = frozenset({'init', 'parameter_set', 'modify_signal', 'record_all'})

#: Keyword aliases for the model reference accepted by ``add_simulator``.
_MODEL_ALIAS_KEYS = (
    'sim_loc', 'net_loc', 'net', 'network', 'path', 'fmu_loc', 'fmu_path',
    'csv_location', 'csv_path', 'dss_loc', 'dss_path', 'project_name',
    'script_loc', 'script_path', 'container',
)

_CONTAINER_TYPES = ('container', 'fleet', 'vectorized', 'battery_fleet', 'building_fleet', 'pv_fleet',
                    'function_fleet')

_WORLD_KEYS = frozenset({'start_time', 'stop_time', 't_macro', 'coupling', 'extrapolation',
                         'power_unit', 'logging', 'start_datetime', 'seed', 'out_dir',
                         'unit_system', 'unit_policy', 't_macro_min', 'record_all'})
RES_FILENAME_REMOVED_MSG = (
    "'res_filename' was removed in 3.0: simulate() returns a Results object (no HDF5 file is "
    "written). Use 'out_dir' to persist runs (Parquet/CSV + meta.json) or Results.save()/to_hdf()."
)
_REMOVED_WORLD_KEYS = frozenset({'max_iterations', 'convergence_tolerance'})

_ext_counter = itertools.count()


def _normalize_mode(mode):
    """Normalise a power-flow calculation mode to 'ac' / 'dc'."""
    if not mode:
        return mode
    m_low = str(mode).lower()
    if m_low in ('ac', 'pf', 'ldf', 'snapshot'):
        return 'ac'
    if m_low in ('dc', 'dcpf', 'lpf', 'direct'):
        return 'dc'
    raise ValueError(
        f"Unsupported calculation mode '{mode}'. "
        f"energysim only supports 'ac' and 'dc' power flow modes."
    )


def _to_fraction(x) -> Fraction:
    """Exact rational representation of a step size (handles 1e-5, numpy floats, ...)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x))
    return Fraction(str(float(x)))


def _filter_kwargs(func, candidates: Dict[str, Any]) -> Dict[str, Any]:
    """Return the subset of *candidates* accepted by *func*'s signature."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    params = sig.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(candidates)
    names = {p.name for p in params
             if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    return {k: v for k, v in candidates.items() if k in names}


def _is_adapter_like(obj) -> bool:
    return (not isinstance(obj, type)
            and all(callable(getattr(obj, m, None)) for m in ('step', 'get_value', 'set_value')))


def _enable_verbose_logging():
    """Make INFO messages of the 'energysim' logger visible (used by ``world(logging=True)``)."""
    if logger.level == _logging.NOTSET or logger.level > _logging.INFO:
        logger.setLevel(_logging.INFO)
    if not any(getattr(h, '_energysim_default', False) for h in logger.handlers):
        handler = _logging.StreamHandler()
        handler.setFormatter(_logging.Formatter('%(message)s'))
        handler._energysim_default = True
        logger.addHandler(handler)

