# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:17:34 2020

@author: Digvijay
"""
from __future__ import annotations


import tables as tb
import pandas as pd
import os
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
filters = tb.Filters(complevel=5, complib='zlib')


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


def convert_to_df(file_name, headers=None):
    '''Converts the input h5 file to dataframes'''
    if headers is None:
        headers = {}
    processed_res = {}
    if len(headers) > 0:
        with tb.open_file(filename=file_name, mode='r') as f:
            for key, value in headers.items():
                node = f.get_node('/', key) if f.__contains__(f'/{key}') else getattr(f.root, key)
                sim_data = node.read()
                df = pd.DataFrame(data=sim_data, columns=['time'] + value)
                processed_res[key] = df
    else:
        with tb.open_file(filename=file_name, mode='r') as f:
            nodes = list(f.root)
            for i in range(len(nodes)):
                sim_data = nodes[i].read() if hasattr(nodes[i], 'read') else list(nodes[i])
                df = pd.DataFrame(data=sim_data)
                processed_res[i] = df
    return processed_res


def convert_to_csv(file_name, headers=None):
    '''Converts the input h5 file to csv data files'''
    if headers is None:
        headers = {}
    if len(headers) > 0:
        with tb.open_file(filename=file_name, mode='r') as f:
            for key, value in headers.items():
                node = f.get_node('/', key) if f.__contains__(f'/{key}') else getattr(f.root, key)
                sim_data = node.read()
                df = pd.DataFrame(data=sim_data, columns=['time'] + value)
                df.to_csv(f'res_{safe_filename_component(key)}.csv', header=None)
    else:
        with tb.open_file(filename=file_name, mode='r') as f:
            nodes = list(f.root)
            for i in range(len(nodes)):
                sim_data = nodes[i].read() if hasattr(nodes[i], 'read') else list(nodes[i])
                df = pd.DataFrame(data=sim_data)
                df.to_csv(f'res_{i}.csv', header=None)


def convert_hdf_to_dict(file_name='es_res.h5', sim_dict=None, to_csv=False, **kwargs):
    if sim_dict is None:
        sim_dict = {}
    processed_res = {}
    with tb.open_file(filename=file_name, mode='r') as f:
        for sim_name, outputs in sim_dict.items():
            node = f.get_node('/', sim_name) if f.__contains__(f'/{sim_name}') else getattr(f.root, sim_name)
            sim_data = node.read()
            tmp = pd.DataFrame(data=sim_data, columns=['time'] + outputs)
            processed_res[sim_name] = tmp
    if to_csv:
        f_name = kwargs.get('res_folder_name', 'res')
        export_to_csv(processed_res, f_name=f_name)
    return processed_res


def export_to_csv(processed_res, f_name='res'):
    parent_dir = os.getcwd()
    tmp_path = os.path.join(parent_dir, f_name)
    os.makedirs(tmp_path, exist_ok=True)
    for sim_name, res_df in processed_res.items():
        res_df.to_csv(os.path.join(tmp_path, f'res_{safe_filename_component(sim_name)}.csv'))
    return f"Exported results to {tmp_path}."


def record_data(file_or_name, res_dict):
    """Utility function to record the data.

    Accepts either an already-open PyTables File object (preferred, avoids
    per-step open/close overhead) or a filename string (legacy fallback).
    """
    if isinstance(file_or_name, tb.file.File):
        for sim, data in res_dict.items():
            earray = file_or_name.get_node('/', sim) if file_or_name.__contains__(f'/{sim}') else getattr(file_or_name.root, sim)
            earray.append(sequence=data)
    else:
        with tb.open_file(filename=file_or_name, mode='a') as f:
            for sim, data in res_dict.items():
                earray = f.get_node('/', sim) if f.__contains__(f'/{sim}') else getattr(f.root, sim)
                earray.append(sequence=data)


def create_results_recorder(file_name, sim_dict):
    """Creates a hdf5 file to store results"""

    with tb.open_file(filename=file_name, mode='w') as f:
        for sim_name, outputs in sim_dict.items():
            # if len(outputs)>0:
            shape = (0, len(outputs) + 1)
            f.create_earray(
                where='/',
                name=sim_name,
                filters=filters,
                shape=shape,
                atom=tb.Float64Atom())  # Float32 silently loses precision; use Float64


# ---------------------------------------------------------------------------
# Unit Conversion Utilities
# ---------------------------------------------------------------------------
#
# Each unit maps to (dimension, factor, offset) such that
#     value_in_base = value * factor + offset
# Conversion is only allowed within one dimension.  Unknown units raise
# ``ValueError`` (never a silent factor of 1.0).
#
# Lookup is case-sensitive first (so 'MV' megavolt != 'mV' millivolt); a
# case-insensitive fallback is used only when it is unambiguous
# (e.g. 'mw' -> MW, 'kvar' -> kvar; 'mv' is rejected as ambiguous).

class UnitDef(NamedTuple):
    dimension: str
    factor: float
    offset: float = 0.0


_UNITS: Dict[str, UnitDef] = {}


def _reg(dimension: str, factor: float, *names: str, offset: float = 0.0) -> None:
    for n in names:
        _UNITS[n] = UnitDef(dimension, factor, offset)


# active power (base W)
_reg('power', 1.0, 'W', 'watt', 'watts')
_reg('power', 1e3, 'kW', 'kilowatt', 'kilowatts')
_reg('power', 1e6, 'MW', 'megawatt', 'megawatts')
_reg('power', 1e9, 'GW', 'gigawatt', 'gigawatts')
# reactive power (base var)
_reg('reactive_power', 1.0, 'var', 'vars')
_reg('reactive_power', 1e3, 'kvar', 'kvars')
_reg('reactive_power', 1e6, 'Mvar', 'mvars')
# voltage (base V)
_reg('voltage', 1e-3, 'mV', 'millivolt', 'millivolts')
_reg('voltage', 1.0, 'V', 'volt', 'volts')
_reg('voltage', 1e3, 'kV', 'kilovolt', 'kilovolts')
_reg('voltage', 1e6, 'MV', 'megavolt', 'megavolts')
# per-unit: its own dimension, never converted to/from volts
_reg('per_unit', 1.0, 'pu', 'p.u.')
# energy (base J)
_reg('energy', 1.0, 'J', 'joule', 'joules')
_reg('energy', 1e3, 'kJ')
_reg('energy', 1e6, 'MJ')
_reg('energy', 1e9, 'GJ')
_reg('energy', 3.6e3, 'Wh')
_reg('energy', 3.6e6, 'kWh')
_reg('energy', 3.6e9, 'MWh')
_reg('energy', 3.6e12, 'GWh')
# mass flow (base kg/s)
_reg('mass_flow', 1.0, 'kg/s')
_reg('mass_flow', 1.0 / 3600.0, 'kg/h')
_reg('mass_flow', 1000.0 / 3600.0, 't/h')
# volume flows: actual and normal volume are separate dimensions
_reg('volume_flow', 1.0, 'm3/s', 'm³/s')
_reg('volume_flow', 1.0 / 3600.0, 'm3/h', 'm³/h')
_reg('normal_volume_flow', 1.0, 'Nm3/h', 'Nm³/h')
# pressure (base Pa)
_reg('pressure', 1.0, 'Pa')
_reg('pressure', 1e3, 'kPa')
_reg('pressure', 1e5, 'bar')
_reg('pressure', 1e6, 'MPa')
# temperature (base K; degC is affine)
_reg('temperature', 1.0, 'K', 'kelvin')
_reg('temperature', 1.0, 'degC', '°C', 'celsius', offset=273.15)
# time (base s)
_reg('time', 1.0, 's', 'sec')
_reg('time', 60.0, 'min')
_reg('time', 3600.0, 'h', 'hr')
# dimensionless
_reg('dimensionless', 1.0, '-', '1')
_reg('dimensionless', 0.01, '%')

_UNITS_LOWER: Dict[str, List[str]] = {}
for _name in _UNITS:
    _UNITS_LOWER.setdefault(_name.lower(), []).append(_name)

KNOWN_UNITS = tuple(_UNITS)


def parse_unit(unit: str) -> UnitDef:
    """Return the :class:`UnitDef` for *unit*; raise ``ValueError`` if unknown or ambiguous."""
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(f"Invalid unit {unit!r}.")
    u = unit.strip()
    if u in _UNITS:
        return _UNITS[u]
    candidates = _UNITS_LOWER.get(u.lower(), [])
    defs = {_UNITS[c] for c in candidates}
    if len(defs) == 1:
        return _UNITS[candidates[0]]
    if len(defs) > 1:
        raise ValueError(
            f"Ambiguous unit '{unit}': could be any of {sorted(candidates)}. Use the exact spelling."
        )
    raise ValueError(f"Unknown unit '{unit}'. Known units: {', '.join(KNOWN_UNITS)}")


def is_known_unit(unit: str) -> bool:
    """True if *unit* is a recognised (unambiguous) unit string."""
    try:
        parse_unit(unit)
        return True
    except ValueError:
        return False


def get_unit_conversion(from_unit: Optional[str], to_unit: Optional[str]) -> Tuple[float, float]:
    """Return ``(scale, offset)`` with ``value_to = value_from * scale + offset``.

    If either unit is ``None`` / empty, no conversion is applied (``(1.0, 0.0)``).
    Unknown units raise ``ValueError``; units of different dimensions raise
    ``ValueError`` ("Incompatible unit conversion").  Affine units
    (degC <-> K) are supported through the offset.
    """
    if not from_unit or not to_unit:
        return 1.0, 0.0
    a = parse_unit(from_unit)
    b = parse_unit(to_unit)
    if a.dimension != b.dimension:
        raise ValueError(
            f"Incompatible unit conversion: cannot convert '{from_unit}' ({a.dimension}) "
            f"to '{to_unit}' ({b.dimension})."
        )
    scale = a.factor / b.factor
    offset = (a.offset - b.offset) / b.factor
    return scale, offset


def get_unit_scale(from_unit: Optional[str], to_unit: Optional[str]) -> float:
    """
    Linear multiplication factor converting a value from `from_unit` to `to_unit`.

    Returns 1.0 when either unit is unspecified.  Raises ``ValueError`` for
    unknown units, incompatible dimensions, or affine conversions (degC <-> K;
    use :func:`get_unit_conversion` for those).
    """
    scale, offset = get_unit_conversion(from_unit, to_unit)
    if offset != 0.0:
        raise ValueError(
            f"Conversion '{from_unit}' -> '{to_unit}' needs an offset; use get_unit_conversion()."
        )
    return scale


# Explicit unit suffixes recognised in (the last segment of) variable names.
# Order matters: longer suffixes first.
_SUFFIX_UNITS = (
    ('_nm3_h', 'Nm3/h'), ('_m3_h', 'm3/h'),
    ('_kg_s', 'kg/s'), ('_kg_h', 'kg/h'), ('_t_h', 't/h'),
    ('_mwh', 'MWh'), ('_kwh', 'kWh'), ('_gwh', 'GWh'), ('_wh', 'Wh'),
    ('_mvar', 'Mvar'), ('_kvar', 'kvar'),
    ('_gw', 'GW'), ('_mw', 'MW'), ('_kw', 'kW'),
    ('_kv', 'kV'), ('_pu', 'pu'),
    ('_degc', 'degC'), ('_kpa', 'kPa'), ('_mpa', 'MPa'), ('_bar', 'bar'),
)

# Dotted tool-specific attribute names (OpenDSS / PowerFactory)
_DOTTED_UNITS = (
    ('.pgini', 'MW'), ('.plini', 'MW'), ('.qgini', 'Mvar'), ('.qlini', 'Mvar'),
    ('.m:u', 'pu'), ('.kw', 'kW'), ('.kvar', 'kvar'), ('.kv', 'kV'),
)

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
       ``_degc``, ...) and tool attribute names (``.kW``, ``.pgini``, ...).
    2. Only for variables whose *name* looks like active power (``p``,
       ``p_cmd``, ``P_set``, ``power`` ...): the simulator's native power unit
       (pandapower/PyPSA/PowerFactory: MW, OpenDSS: kW, FMU: W), else
       ``default_unit``.
    3. Otherwise ``None`` -- the world will not auto-scale such a variable.
    """
    v_lower = var_name.strip().lower()
    if 'vm_pu' in v_lower:
        return 'pu'
    for suf, unit in _DOTTED_UNITS:
        if v_lower.endswith(suf):
            return unit
    last = v_lower.split('.')[-1]
    for suf, unit in _SUFFIX_UNITS:
        if last.endswith(suf):
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
