#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 12:53:14 2021

@author: kimbo

"""

import os
os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
import glob 
import shutil
import click
import time
import dask.array as da
import numpy as np 
import xarray as xr


from disdrodb.io import check_folder_structure
from disdrodb.io import check_valid_varname
from disdrodb.io import check_L0_standards
from disdrodb.io import check_L1_standards
from disdrodb.io import get_attrs_standards
from disdrodb.io import get_L0_dtype_standards
from disdrodb.io import get_dtype_standards_all_object
from disdrodb.io import get_flags
from disdrodb.io import _write_to_parquet
from disdrodb.io import col_dtype_check


from disdrodb.io import get_raw_field_nbins
from disdrodb.io import get_L1_coords
from disdrodb.io import rechunk_L1_dataset
from disdrodb.io import get_L1_zarr_encodings_standards
from disdrodb.io import get_L1_nc_encodings_standards

from disdrodb.logger import log
from disdrodb.logger import close_log

from disdrodb.sensor import Sensor


###############################
#### Perform L1 processing ####
###############################

def L1_process(verbose, processed_path, campaign_name, L0_processing, lazy, debug_on, sensor_name, attrs, keep_zarr, device_list, device = None):
    
    # Start logger
    global logger
    logger = log(processed_path, 'L1')

    
    if lazy: 
        import dask.dataframe as dd
    else: 
        import pandas as dd

    msg =f"L1 processing of device {device} started"
    if verbose:
        print(msg)
    logger.info(msg)

    ##-----------------------------------------------------------
    # Check the L0 df is available 
    # Path device folder parquet
    if device == None:
        df_fpath = os.path.join(processed_path + '/L0/' + campaign_name + '.parquet')
    else:
        df_fpath = os.path.join(processed_path + '/' + device + '/L0/' + campaign_name + '.parquet')
    
    if not L0_processing:
        if not os.path.exists(df_fpath):
            msg = "Need to run L0 processing. The {df_fpath} file is not available."
            logger.exception(msg)
            raise ValueError(msg)

    msg = f'Found parquet file: {df_fpath}'
    if verbose:
        print(msg)
    logger.info(msg)

    ##-----------------------------------------------------------
    # Read raw data from parquet file 
    msg = f'Start reading: {df_fpath}'
    if verbose:
        print(msg)
    logger.info(msg)

    df = dd.read_parquet(df_fpath)

    msg = f'Finish reading: {df_fpath}'
    if verbose:
        print(msg)
    logger.info(msg)

    ##-----------------------------------------------------------
    # Subset row sample to debug 
    if not lazy and debug_on:
        df = df.iloc[0:100,:] # df.head(100) 
        df = df.iloc[0:,:]
        
        msg = ' ***** Debug = True and Lazy = False, then only the first 100 rows are read *****'
        if verbose:
            print(msg)
        logger.info(msg)
       
    ##-----------------------------------------------------------
    # Retrieve raw data matrix 
    msg = f"Retrieve raw data matrix for device {device}"
    if verbose:
        print(msg)
    logger.info(msg)

    dict_data = {}
    n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)

    if not lazy and debug_on:
        n_timesteps = df.shape[0]
    else:
        n_timesteps = df.shape[0].compute()
        
    for key, n_bins in n_bins_dict.items(): 
        
        # Dask based 
        dd_series = df[key].astype(str).str.split(",")
        da_arr = da.stack(dd_series, axis=0)
        # Remove '' at the end 
        da_arr = da_arr[: , 0:n_bins_dict[key]]
        
        # Convert flag values to XXXX
        # dir(da_arr.str)
        # FieldN and FieldV --> -9.999, has floating number 
        
        if key == 'RawData':
            da_arr = da_arr.astype(int)
            try:
                da_arr = da_arr.reshape(n_timesteps, n_bins_dict['FieldN'], n_bins_dict['FieldV'])
            except Exception as e:
                msg = f'Error on retrive raw data matrix: {e}'
                logger.error(msg)
                print(msg)
                # raise SystemExit
        else:
            da_arr = da_arr.astype(float)                
        
        dict_data[key] = da_arr

                   
        # Pandas/Numpy based 
        # np_arr_str =  df[key].values.astype(str)
        # list_arr_str = np.char.split(np_arr_str,",")
        # arr_str = np.stack(list_arr_str, axis=0) 
        # arr = arr_str[:, 0:n_bins]
        # arr = arr.astype(float)                
        # if key == 'RawData':
        #     arr = arr.reshape(n_timesteps, n_bins_dict['FieldN'], n_bins_dict['FieldV'])
        # dict_data[key] = arr

    msg = f"Finish retrieve raw data matrix for device {device}"
    if verbose:
        print(msg)
    logger.info(msg)

    ##-----------------------------------------------------------
    # Define data variables for xarray Dataset 
    data_vars = {"FieldN": (["time", "diameter_bin_center"], dict_data['FieldN']),
                 "FieldV": (["time", "velocity_bin_center"], dict_data['FieldV']),
                 "RawData": (["time", "diameter_bin_center", "velocity_bin_center"], dict_data['RawData']),
                }

    # Define coordinates for xarray Dataset
    coords = get_L1_coords(sensor_name=sensor_name)
    coords['time'] = df['time'].values
    
    if device == None:
        coords['latitude'] = attrs['latitude']
        coords['longitude'] = attrs['longitude']
        coords['altitude'] = attrs['altitude']
        coords['crs'] = attrs['crs']
    else:
        coords['latitude'] = device_list[device].latitude
        coords['longitude'] = device_list[device].longitude
        coords['altitude'] = device_list[device].latitude
        coords['crs'] = device_list[device].crs
        coords['disdrodb_id'] = device_list[device].disdrodb_id

        attrs['latitude'] = device_list[device].latitude
        attrs['longitude'] = device_list[device].longitude
        attrs['altitude'] = device_list[device].latitude
        attrs['crs'] = device_list[device].crs
        attrs['disdrodb_id'] = device_list[device].disdrodb_id


    ##-----------------------------------------------------------
    # Create xarray Dataset
    try:
        ds = xr.Dataset(data_vars = data_vars, 
                        coords = coords, 
                        attrs = attrs,
                        )
    except Exception as e:
        msg = f'Error on creation xarray dataset: {e}'
        logger.error(msg)
        print(msg)
        # raise SystemExit

    ##-----------------------------------------------------------
    # Check L1 standards 
    check_L1_standards(ds)

    ##-----------------------------------------------------------    
    # Write to Zarr as intermediate storage 
    if keep_zarr:
        if device == None:
            tmp_zarr_fpath = os.path.join(processed_path + '/L1/' + campaign_name + '.zarr')
        else:
            tmp_zarr_fpath = os.path.join(processed_path + '/' + device + '/L1/' + campaign_name + '.zarr')
        ds = rechunk_L1_dataset(ds, sensor_name=sensor_name)
        zarr_encoding_dict = get_L1_zarr_encodings_standards(sensor_name=sensor_name)
        ds.to_zarr(tmp_zarr_fpath, encoding=zarr_encoding_dict, mode = "w")

    ##-----------------------------------------------------------  
    # Write L1 dataset to netCDF
    # Path for save into device folder
    if device == None:
        path = processed_path
    else:
        path = os.path.join(processed_path + '/' + device)
    
    L1_nc_fpath = path + '/L1/' + campaign_name + '.nc'
    ds = rechunk_L1_dataset(ds, sensor_name=sensor_name) # very important for fast writing !!!
    nc_encoding_dict = get_L1_nc_encodings_standards(sensor_name=sensor_name)
    
    try:
        if debug_on:
            ds.to_netcdf(L1_nc_fpath, engine="netcdf4")
        else:
            ds.to_netcdf(L1_nc_fpath, engine="netcdf4", encoding=nc_encoding_dict)
    except ValueError as e:
        msg = f'Error, try save withouth encoding: {e}'
        logger.error(msg)
        print(msg)
        ds.to_netcdf(L1_nc_fpath, engine="netcdf4")
    except Exception as e:
        msg = f'Error on save netCDF: {e}'
        logger.error(msg)
        print(msg)
        