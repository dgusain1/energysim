# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:17:34 2020

@author: Digvijay
"""
from __future__ import annotations


import os
import re
from typing import Any, Optional, Tuple


_UNSAFE_NAME_CHARS = re.compile(r'[^A-Za-z0-9_.\-]')


def safe_filename_component(name: Any) -> str:
    """Return *name* reduced to a safe single path component.

    Path separators, drive colons, ``..`` and any character outside
    ``[A-Za-z0-9_.-]`` are replaced by ``_`` so simulator names can never
    escape the results directory.
    """
    s = _UNSAFE_NAME_CHARS.sub('_', str(name))
    while '..' in s:
        s = s.replace('..', '_')
    s = s.strip('. ')
    return s or '_'


# ---------------------------------------------------------------------------
# Unit Conversion Utilities (thin wrappers around energysim.units)
# ---------------------------------------------------------------------------
#
# ``energysim.units`` is the single source of truth for units.  The names
# below are kept for backwards compatibility and delegate to it:
#
# * lookup is case-sensitive first (so 'MV' megavolt != 'mV' millivolt); a
#   spelling-insensitive fallback is used only when it is unambiguous;
# * unknown units raise ``ValueError`` (never a silent factor of 1.0);
# * conversion is only allowed within one dimension.

from .units import (  # noqa: E402  (re-exported)
    UnitDef, KNOWN_UNITS, parse_unit, is_known_unit, infer_unit_from_name,
    get_unit_conversion as _units_get_unit_conversion,
)


def get_unit_conversion(from_unit: Optional[str], to_unit: Optional[str],
                        quantity: Optional[str] = None) -> Tuple[float, float]:
    """Return ``(scale, offset)`` with ``value_to = value_from * scale + offset``.

    If either unit is ``None`` / empty, no conversion is applied (``(1.0, 0.0)``).
    Unknown units raise ``ValueError``; units of different dimensions raise
    ``ValueError`` ("Incompatible unit conversion").  Affine units
    (degC <-> K) are supported through the offset.  Delegates to
    :func:`energysim.units.get_unit_conversion`.
    """
    return _units_get_unit_conversion(from_unit, to_unit, quantity)


def get_unit_scale(from_unit: Optional[str], to_unit: Optional[str],
                   quantity: Optional[str] = None) -> float:
    """
    Linear multiplication factor converting a value from `from_unit` to `to_unit`.

    Returns 1.0 when either unit is unspecified.  Raises ``ValueError`` for
    unknown units, incompatible dimensions, or affine conversions (degC <-> K;
    use :func:`get_unit_conversion` for those).
    """
    scale, offset = get_unit_conversion(from_unit, to_unit, quantity)
    if offset != 0.0:
        raise ValueError(
            f"Conversion '{from_unit}' -> '{to_unit}' needs an offset; use get_unit_conversion()."
        )
    return scale


# Names that conventionally denote active power: p, p_cmd, P_set, p_load, pgen, power, ...
_POWER_NAME = re.compile(
    r'^(p|p_(cmd|set|setpoint|ref|req|gen|load|el|elec|out|in|dispatch|demand)|'
    r'p(cmd|set|gen|load|ref)|.*power.*)$'
)

_NATIVE_POWER_UNIT = {
    'pandapower': 'MW', 'powerflow': 'MW', 'pypsa': 'MW', 'powerfactory': 'MW',
    'opendss': 'kW', 'fmu': 'W',
}


def looks_like_power(var_name: str) -> bool:
    """True if the variable name conventionally denotes active power (p, p_cmd, P_set, power, ...)."""
    last = var_name.strip().split('.')[-1].lower()
    return bool(_POWER_NAME.match(last))


def infer_variable_unit(var_name: str, sim_type: str = '', default_unit: Optional[str] = None) -> Optional[str]:
    """
    Infer the unit of *var_name*, or return ``None`` when it cannot be
    determined explicitly.

    1. Conventional suffixes (``_mw``, ``_kw``, ``_kv``, ``_pu``, ``_kwh``,
       ``_degc``, ...) and tool attribute names (``.kW``, ``.pgini``, ...),
       see :func:`energysim.units.infer_unit_from_name`.
    2. Only for variables whose *name* looks like active power (``p``,
       ``p_cmd``, ``P_set``, ``power`` ...): the simulator's native power unit
       (pandapower/PyPSA/PowerFactory: MW, OpenDSS: kW, FMU: W), else
       ``default_unit``.
    3. Otherwise ``None``.

    The world itself uses the full resolution order (declared -> component ->
    connection -> suffix -> default unit by quantity), see
    :meth:`energysim.world.resolve_variable_unit`.
    """
    if 'vm_pu' in var_name.strip().lower():
        return 'pu'
    unit = infer_unit_from_name(var_name)
    if unit is not None:
        return unit
    if looks_like_power(var_name):
        st = (sim_type or '').strip().lower()
        if st in _NATIVE_POWER_UNIT:
            return _NATIVE_POWER_UNIT[st]
        return default_unit
    return None


# ---------------------------------------------------------------------------
# Model references
# ---------------------------------------------------------------------------

MODEL_FILE_EXTS = ('.fmu', '.p', '.pkl', '.json', '.csv', '.dss', '.m', '.py', '.nc',
                   '.xlsx', '.sqlite', '.h5', '.hdf5', '.pfd')
PICKLE_EXTS = ('.p', '.pkl', '.pickle')
_DOTTED_REF = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$')


def looks_like_dotted_import(model_ref: Any) -> bool:
    """True if *model_ref* is a string like ``'pkg.module.func'`` that is not a path / file name."""
    if not isinstance(model_ref, str):
        return False
    if os.path.exists(model_ref) or model_ref.lower().endswith(MODEL_FILE_EXTS):
        return False
    return bool(_DOTTED_REF.match(model_ref))


def resolve_model_reference(model_ref: Any, allow_import: bool = False) -> Any:
    """
    Resolve a model reference which can be:

    - An in-memory Python object (returned as is)
    - An existing filesystem path / file name with a known extension (returned as is)
    - A dotted import string (e.g. ``'pandapower.networks.case14'``) which is
      imported and, if callable, called -- **only** when ``allow_import=True``.
      Without that opt-in a ``ValueError`` is raised instead of silently
      executing code.
    - Any other identifier string (e.g. PowerFactory project ``'IEEE13'``),
      returned as is.
    """
    if not isinstance(model_ref, str) or not looks_like_dotted_import(model_ref):
        return model_ref

    if not allow_import:
        raise ValueError(
            f"Model reference '{model_ref}' looks like a dotted Python import. Importing and "
            f"calling it requires explicit opt-in: pass allow_import=True (or build the model "
            f"in Python and pass the object)."
        )

    import importlib
    mod_name, func_name = model_ref.rsplit('.', 1)
    try:
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, func_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Could not import model reference '{model_ref}': {exc}") from exc
    if callable(obj):
        try:
            return obj()
        except TypeError:
            return obj
    return obj
