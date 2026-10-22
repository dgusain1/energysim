# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:17:34 2020

@author: Digvijay
"""

import tables as tb
import pandas as pd
import os
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
