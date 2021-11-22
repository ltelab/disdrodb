#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:10:59 2021

@author: ghiggi
"""
import numpy as np 
    
## Kimbo
# - correct header names
# - dtype, attrs standards 
# - Check folder exists if force=True, 
# - coordinate standards ? 
#   get_velocity_bin_center(): 
    # - args: instrument=Parsivel,Thies .. if, elif 
# - click https://click.palletsprojects.com/en/8.0.x/ -> default type=bool

#-----------------------------------------------------------------------------. 
 
def _available_sensors():
    sensor_list = ['Parsivel', 'Parsivel2', 'ThiesLPM']
    return sensor_list

def _check_sensor_name(sensor_name): 
    if not isinstance(sensor_name, str):
        raise TypeError("'sensor_name' must be a string'")
    if sensor_name not in _available_sensors(): 
        raise ValueError("Valid sensor_name are {}".format(_available_sensors()))
    return 

def _write_to_parquet(df, fpath, force=False):    
    # TODO: schema args, and vompress options, chunks 
    # TODO: If force=False and dir exists, raise Error. If True remove and write 
    try:
        df.to_parquet(fpath , schema='infer')
    except (Exception) as e:
        raise ValueError("Can not convert to parquet file. The error is {}".format(e))

#-----------------------------------------------------------------------------.
# TODO correct values 
def _get_diameter_bin_center(sensor_name): 
    if sensor_name == "Parsivel":
        x = np.arange(0,32)
        
    elif sensor_name == "Parsivel2":
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        raise NotImplementedError
    else:
        raise ValueError("L0 bin characteristics for sensor {} are not yet defined".format(sensor_name))
    return x

def _get_diameter_bin_lower(sensor_name): 
    if sensor_name == "Parsivel":
        x = np.arange(0,32)
        
    elif sensor_name == "Parsivel2":
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        raise NotImplementedError
    else:
        raise ValueError("L0 bin characteristics for sensor {} are not yet defined".format(sensor_name))
    return x

def _get_diameter_bin_upper(sensor_name): 
    if sensor_name == "Parsivel":
        x = np.arange(0,32)
        
    elif sensor_name == "Parsivel2":
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        raise NotImplementedError
    else:
        raise ValueError("L0 bin characteristics for sensor {} are not yet defined".format(sensor_name))
    return x

def _get_diameter_bin_width(sensor_name): 
    if sensor_name == "Parsivel":
        x = np.arange(0,32)
        
    elif sensor_name == "Parsivel2":
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        raise NotImplementedError
    else:
        raise ValueError("L0 bin characteristics for sensor {} are not yet defined".format(sensor_name))
    return x
 
def _get_velocity_bin_center(sensor_name): 
    if sensor_name == "Parsivel":
        x = np.arange(0,32)
        
    elif sensor_name == "Parsivel2":
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        raise NotImplementedError
    else:
        raise ValueError("L0 bin characteristics for sensor {} are not yet defined".format(sensor_name))
    return x

def _get_velocity_bin_lower(sensor_name): 
    if sensor_name == "Parsivel":
        x = np.arange(0,32)
        
    elif sensor_name == "Parsivel2":
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        raise NotImplementedError
    else:
        raise ValueError("L0 bin characteristics for sensor {} are not yet defined".format(sensor_name))
    return x

def _get_velocity_bin_upper(sensor_name): 
    if sensor_name == "Parsivel":
        x = np.arange(0,32)
        
    elif sensor_name == "Parsivel2":
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        raise NotImplementedError
    else:
        raise ValueError("L0 bin characteristics for sensor {} are not yet defined".format(sensor_name))
    return x

def _get_velocity_bin_width(sensor_name): 
    if sensor_name == "Parsivel":
        x = np.arange(0,32)
        
    elif sensor_name == "Parsivel2":
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        raise NotImplementedError
    else:
        raise ValueError("L0 bin characteristics for sensor {} are not yet defined".format(sensor_name))
    return x
 
def _get_default_nc_encoding(chunks, dtype='float64', fill_value = np.nan):
    encoding_kwargs = {} 
    encoding_kwargs['dtype']  = dtype
    encoding_kwargs['scale_factor']  = 1.0
    encoding_kwargs['add_offset']  = 0.0
    encoding_kwargs['zlib']  = True
    encoding_kwargs['complevel']  = '4'
    encoding_kwargs['shuffle']  = True
    encoding_kwargs['fletcher32']  = False
    encoding_kwargs['contiguous']  = False
    encoding_kwargs['chunksizes']  = chunks
    encoding_kwargs['_FillValue']  = fill_value
    return encoding_kwargs
##----------------------------------------------------------------------------.


def get_L1_coords(sensor_name): 
    _check_sensor_name(sensor_name=sensor_name)
    coords = {} 
    coords["diameter_bin_center"] = _get_diameter_bin_center(sensor_name=sensor_name)
    coords["diameter_bin_lower"] = _get_diameter_bin_lower(sensor_name=sensor_name)
    coords["diameter_bin_upper"] = _get_diameter_bin_upper(sensor_name=sensor_name)
    coords["diameter_bin_width"] = _get_diameter_bin_width(sensor_name=sensor_name)
    coords["velocity_bin_center"] = _get_velocity_bin_center(sensor_name=sensor_name)
    coords["velocity_bin_lower"] = _get_velocity_bin_lower(sensor_name=sensor_name)
    coords["velocity_bin_upper"] = _get_velocity_bin_upper(sensor_name=sensor_name)
    coords["velocity_bin_width"] = _get_velocity_bin_width(sensor_name=sensor_name)
    return coords 
                 
def get_L1_chunks(sensor_name):
    _check_sensor_name(sensor_name=sensor_name)
    if sensor_name == "Parsivel":
        chunks_dict = {'FieldN': (1000,32),
                       'FieldV': (1000,32),
                       'RawData': (1000,32,32),
                      }
    elif sensor_name == "Parsivel2":
        raise NotImplementedError
    elif sensor_name == "ThiesLPM":
        raise NotImplementedError
    else:
        raise ValueError("L0 chunks for sensor {} are not yet defined".format(sensor_name))
    return chunks_dict

def get_L1_encodings_standards(sensor_name): 
    # Define variable names 
    vars = ['FieldN', 'FieldV', 'RawData']   
    # Get chunks based on sensor type
    chunks_dict = get_L1_chunks(sensor_name=sensor_name) 
    # Define encodings dictionary 
    encoding_dict = {} 
    for var in vars:
        encoding_dict[var] = _get_default_nc_encoding(chunks=chunks_dict[var],
                                                      dtype='int16',   # TODO !!! 
                                                      fill_value='-1') # TODO
    return encoding_dict 

     


 






