# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:17:34 2020

@author: Digvijay
"""
from __future__ import annotations


import tables as tb
import pandas as pd
import os
from typing import Any, Dict, List, Optional
filters = tb.Filters(complevel=5, complib='zlib')


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
                df.to_csv(f'res_{key}.csv', header=None)
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
        res_df.to_csv(os.path.join(tmp_path, f'res_{sim_name}.csv'))
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

_POWER_UNITS = {
    'w': 1e-3, 'watt': 1e-3, 'watts': 1e-3,
    'kw': 1.0, 'kilowatt': 1.0, 'kilowatts': 1.0,
    'mw': 1e3, 'megawatt': 1e3, 'megawatts': 1e3,
    'gw': 1e6, 'gigawatt': 1e6, 'gigawatts': 1e6,
}

_REACTIVE_POWER_UNITS = {
    'var': 1e-3, 'vars': 1e-3,
    'kvar': 1.0, 'kvars': 1.0,
    'mvar': 1e3, 'mvars': 1e3,
}

_VOLTAGE_UNITS = {
    'v': 1e-3, 'volt': 1e-3, 'volts': 1e-3,
    'kv': 1.0, 'kilovolt': 1.0, 'kilovolts': 1.0,
    'pu': 1.0, 'p.u.': 1.0,
}

_ENERGY_UNITS = {
    'j': 1.0 / 3.6e6, 'joule': 1.0 / 3.6e6, 'joules': 1.0 / 3.6e6,
    'wh': 1e-3,
    'kwh': 1.0,
    'mwh': 1e3,
}

_UNIT_CATEGORIES = [
    _POWER_UNITS,
    _REACTIVE_POWER_UNITS,
    _VOLTAGE_UNITS,
    _ENERGY_UNITS,
]


def get_unit_scale(from_unit: str, to_unit: str) -> float:
    """
    Compute linear multiplication factor to convert numeric value from `from_unit` to `to_unit`.
    Returns 1.0 if units are identical or unknown. Raises ValueError on category mismatch.
    """
    if not from_unit or not to_unit:
        return 1.0
    u_from = from_unit.strip().lower()
    u_to = to_unit.strip().lower()
    if u_from == u_to:
        return 1.0

    for cat in _UNIT_CATEGORIES:
        if u_from in cat and u_to in cat:
            return cat[u_from] / cat[u_to]

    # If both units were recognized in different categories
    cat_from = any(u_from in cat for cat in _UNIT_CATEGORIES)
    cat_to = any(u_to in cat for cat in _UNIT_CATEGORIES)
    if cat_from and cat_to:
        raise ValueError(f"Incompatible unit conversion: cannot convert '{from_unit}' to '{to_unit}'.")

    return 1.0


def infer_variable_unit(var_name: str, sim_type: str = '', default_unit: str = 'kW') -> str:
    """
    Infer the physical engineering unit from variable naming conventions or simulator native units.
    """
    v_lower = var_name.strip().lower()
    if v_lower.endswith(('_mw', '.p_mw', '.max_p_mw', '.min_p_mw', '.pgini', '.plini')):
        return 'MW'
    if v_lower.endswith(('_mvar', '.q_mvar', '.qgini', '.qlini')):
        return 'Mvar'
    if '.kw' in v_lower or v_lower.endswith(('_kw', '.p_kw', '.losses_kw', '.total_power_kw')):
        return 'kW'
    if '.kvar' in v_lower or v_lower.endswith(('_kvar', '.q_kvar')):
        return 'kvar'
    if 'vm_pu' in v_lower or v_lower.endswith(('.vm_pu', '_pu', '.v_mag_pu', '.m:u')):
        return 'pu'
    if '.kv' in v_lower or v_lower.endswith(('_kv', '.kv')):
        return 'kV'

    # Fall back based on simulator native defaults
    st = sim_type.strip().lower()
    if st in ('pandapower', 'powerflow', 'pypsa', 'powerfactory'):
        return 'MW'
    if st == 'opendss':
        return 'kW'
    if st == 'fmu':
        return 'W'

    return default_unit


def resolve_model_reference(model_ref: Any) -> Any:
    """
    Resolve a model reference which can be:
    - An in-memory Python object (returned as is)
    - An existing filesystem path (returned as is)
    - A dotted import string (e.g. 'pandapower.networks.case14' or 'my_module.build_grid')
      which is dynamically imported and evaluated if callable.
    - An identifier string (e.g. PowerFactory project name 'IEEE13')
    """
    if not isinstance(model_ref, str):
        return model_ref

    # If it's a file on disk or looks like a file path with known extension, return it as string
    if os.path.exists(model_ref) or any(model_ref.lower().endswith(ext) for ext in (
        '.fmu', '.p', '.pkl', '.json', '.csv', '.dss', '.m', '.py', '.nc'
    )):
        return model_ref

    # Check if it might be a dotted Python callable path (e.g. 'pandapower.networks.case14')
    if '.' in model_ref and not any(c in model_ref for c in ('/', '\\', ' ', ':')):
        try:
            mod_name, func_name = model_ref.rsplit('.', 1)
            import importlib
            mod = importlib.import_module(mod_name)
            obj = getattr(mod, func_name)
            if callable(obj):
                try:
                    return obj()
                except TypeError:
                    return obj
            return obj
        except (ImportError, AttributeError, ValueError):
            pass

    return model_ref

