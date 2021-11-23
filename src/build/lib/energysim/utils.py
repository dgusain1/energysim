# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:17:34 2020

@author: Digvijay
"""

import tables as tb, pandas as pd, os
filters = tb.Filters(complevel=5, complib='zlib')

def convert_to_df(file_name, headers={}):
    '''Converts the input h5 file to dataframes'''
    processed_res = {}
    if len(headers)>0:
        with tb.open_file(filename=file_name, mode='r') as f:
            for key, value in headers.items():
                sim_data = list(getattr(f.root, key))
                df = pd.DataFrame(data=sim_data, columns = ['time'] + value)
                processed_res[key] = df
    else:
        with tb.open_file(filename=file_name, mode='r') as f:
            for i in range(len(list(f.root))):
                sim_data = list(list(f.root)[i])
                df = pd.DataFrame(data=sim_data)
                processed_res[i] = df
    return processed_res

def convert_to_csv(file_name, headers={}):
    '''Converts the input h5 file to csv data files'''
    if len(headers)>0:
        with tb.open_file(filename=file_name, mode='r') as f:
            for key, value in headers.items():
                sim_data = list(getattr(f.root, key))
                df = pd.DataFrame(data=sim_data, columns = ['time'] + value)
                df.to_csv(f'res_{key}.csv', header=None)
    else:
        with tb.open_file(filename=file_name, mode='r') as f:
            for i in range(len(list(f.root))):
                sim_data = list(list(f.root)[i])
                df = pd.DataFrame(data=sim_data)
                df.to_csv(f'res_{i}.csv', header=None)
        

def convert_hdf_to_dict(file_name='es_res.h5', sim_dict={}, to_csv = False, **kwargs):
    processed_res = {}
    with tb.open_file(filename=file_name, mode='r') as f:
        for sim_name, outputs in sim_dict.items():
            sim_data = list(getattr(f.root, sim_name))
            tmp = pd.DataFrame(data=sim_data, columns=['time'] + outputs)
            processed_res[sim_name] = tmp
    if to_csv:
        if 'res_folder_name' in kwargs.values():
            f_name = kwargs['res_folder_name']
        else:
            f_name = 'res'
        return export_to_csv(processed_res, f_name=f_name)
    else:
        return processed_res

def export_to_csv(processed_res, f_name='res'):
    parent_dir = os.getcwd()
    tmp_path = os.path.join(parent_dir, f_name)
    for sim_name, res_df in processed_res.items():
        if os.path.isdir(tmp_path):
            res_df.to_csv(os.path.join(tmp_path,f'res_{sim_name}.csv'))
        else:
            os.makedirs(tmp_path)
            res_df.to_csv(os.path.join(tmp_path,f'res_{sim_name}.csv'))
    return f"Exported results to {tmp_path}."

def record_data(file_name, res_dict):
    """Utility function to record the data"""
    
    with tb.open_file(filename=file_name, mode='a') as f:
        for sim, data in res_dict.items():
            earray= getattr(f.root, sim)
            earray.append(sequence=data)

def create_results_recorder(file_name, sim_dict):
    """Creates a hdf5 file to store results"""
    
    with tb.open_file(filename=file_name, mode='w') as f:
        for sim_name, outputs in sim_dict.items():
            # if len(outputs)>0:    
            shape = (0, len(outputs)+1)
            f.create_earray(
                where='/',
                name=sim_name,
                filters=filters,
                shape=shape,
                atom=tb.Float32Atom())