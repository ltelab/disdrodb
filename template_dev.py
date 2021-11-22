#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:59:21 2021

@author: ghiggi
"""
import os
os.chdir("/home/ghiggi/Projects/parsiveldb")
import glob 
import argparse # prefer click ... https://click.palletsprojects.com/en/8.0.x/
import click
import time
import dask 
import numpy as np 
import xarray as xr 

from parsiveldb.io import check_valid_varname
from parsiveldb.io import check_L0_standards
from parsiveldb.io import check_L1_standards
from parsiveldb.io import check_L2_standards
from parsiveldb.io import get_attrs_standards
from parsiveldb.io import get_dtype_standards
from parsiveldb.io import get_L1_encodings_standards
 

# A file for each instrument 
# TODO: force=True ... remove existing file 
#       force=False ... raise error if file already exists 
 

base_dir = "/home/ghiggi/Data_sample_Ticino_2018/Ticino_2018/data/61"

debug_on = False
debug_on = True 
lazy = True
force = True
verbose = True

def main(base_dir, L0_processing=True, L1_processing=True, force=False, verbose=True, debug_on=True, lazy=True):
    #-------------------------------------------------------------------------.
    # Whether to use pandas or dask 
    if lazy: 
        import dask.dataframe as dd
    else: 
        import pandas as dd

    #-------------------------------------------------------------------------.
    # - Define directory and file paths 
    df_fpath = "/tmp/Ticino.parquet"   
    nc_fpath = "/tmp/Ticino.nc"   
    L1_nc_fpath = "/tmp/Ticino1.nc"  
    
    #-------------------------------------------------------------------------.
    # - Check if a directory already exists 
    # TODO 
    
    #-------------------------------------------------------------------------.
    # - Define instrument type 
    sensor_name = "Parsivel"
    
    # - Define attributes 
    attrs = get_attrs_standards()
    attrs['Title'] = 'aaa'
    attrs['lat'] = 1223
    attrs['lon'] = 1232
    attrs['crs'] = "WGS84"  # EPSG Code 
    attrs['sensor_name'] = sensor_name
 
    ###############################
    #### Perform L0 processing ####
    ###############################
    if L1_processing: 
        #----------------------------------------------------------------.
        if verbose:
            print("L0 processing of XXXX started")
        t_i = time.time() 
        # - Retrieve filepaths of raw data files 
        if debug_on: 
            file_list = glob.glob(os.path.join(base_dir,"*"))[0:3]
            
        else:  
            file_list = glob.glob(os.path.join(base_dir,"*"))
        #----------------------------------------------------------------.
        # - Define raw data headers 
        raw_data_columns = ['ID',
                            'Geo_coordinate_x',
                            'Geo_coordinate_y',
                            'Timestamp',
                            'Datalogger_temp',
                            'Datalogger_power',
                            'Datalogger_communication',
                            'Rain_intensity',
                            'Rain_amount_accumulated',
                            'Weather_code_SYNOP_according_table_4680',
                            'Weather_code_SYNOP_according_table_4677',
                            'Radar_reflectivity',
                            'MOR_visibility_precipitation',
                            'Signal_amplitude_laser_strip',
                            'Number_detected_particles',
                            'Temperature_sensor',
                            'Current_through_heating_system',
                            'Power_supply_voltage',
                            'Sensor_status',
                            'Rain_amount_absolute',
                            'Error_code',
                            'xxx',
                            'FieldN',
                            'FieldV',
                            'RawData',
                            'Unknow_column'
                            ]
        check_valid_varname(raw_data_columns)   
        ##------------------------------------------------------.      
        # Determine dtype based on standards 
        dtype_dict = get_dtype_standards()
        dtype_dict['FieldN'] = 'U'
        dtype_dict['FieldV'] = 'U'
        dtype_dict['RawData'] = 'U'
        dtype_dict = {column: dtype_dict[column] for column in raw_data_columns}
        
        ##------------------------------------------------------.
        # Define reader options 
        reader_kwargs = {}
        reader_kwargs['compression'] = 'gzip'
        reader_kwargs['delimiter'] = ','
        reader_kwargs["on_bad_lines"] = 'skip'
        reader_kwargs["engine"] = 'python'
        if lazy:
            reader_kwargs["blocksize"] = None
        reader_kwargs["on_bad_lines"] = 'skip'
        
        ##------------------------------------------------------.
        # Loop over all files 
        list_skipped_files = []
        list_df = []
        for filename in file_list:
            try:
                df = dd.read_csv(filename, 
                                 names = raw_data_columns, 
                                 dtype = dtype_dict,
                                 **reader_kwargs)
                
                # - Drop rows with all NaN
                # ---> TODO: find a row with all NA 
                # ---> TODO: remove rows with NA in specific columns 
                df = df.dropna(how='all') 
                
                # - Replace custom NA with standard flags 
                # TODO !!! 
                
                # - Append to the list of dataframe 
                list_df.append(df)
                
            except (Exception, ValueError) as e:
              msg = "{} has been skipped. The error is {}".format(filename, e)
              if verbose: 
                  print(msg)
              list_skipped_files.append(msg)
             
        if verbose:      
            print('{} files on {} has been skipped.'.format(len(list_skipped_files), len(file_list)))
        
        ##------------------------------------------------------.
        # Concatenate the dataframe 
        try:
            df = dd.concat(list_df, axis=0, ignore_index = True)
        except (AttributeError, TypeError) as e:
            raise ValueError("Can not create concat data files. Error: {}".format(e))
            
        ##------------------------------------------------------.                                   
        # Write to Parquet 
        print('Starting conversion to parquet file')
        _write_to_parquet(df, fpath=df_fpath)
        
        ##------------------------------------------------------.   
        # Check Parquet standards 
        check_L0_standards(df)
        
        ##------------------------------------------------------.   
        if verbose:
            print("L0 processing of XXXX ended in {}:.2f}".format(time.time() - t_i))
    
    #-------------------------------------------------------------------------.
    ###############################
    #### Perform L1 processing ####
    ###############################
    if L1_processing: 
        if verbose:
            print("L1 processing of XXXX started")
        t_i = time.time()    
        # Check the L0 df is available 
        if not L0_processing:
            if not os.path.exists(df_fpath):
                raise ValueError("Need to run L0 processing. The {} file is not available.".format(df_fpath))
        df_path = '' # TBD
        df = dd.read_parquet(df_fpath)
        
        df = df.head(10) 
        
        # Retrieve raw data matrix 
        dict_data = {}
        n_bins_dict = get_raw_field_nbins()
        for key, n_bins in n_bins_dict.items(): 
            np_arr_str =  df[key].values.compute().astype(str) # Dask
            # np_arr_str =  df[key].values.astype(str)           # Pandas 
            list_arr_str = np.char.split(np_arr_str,",")
            arr_str = np.stack(list_arr_str, axis=0) 
            arr = arr_str[:, 0:n_bins].astype(float)                
            if key == 'RawData':
                arr = arr.reshape(..., n_bins_dict['FieldN'], n_bins_dict['FieldV'])
            dict_data[key] = arr
        
        ##-----------------------------------------------------------
        # Define data variables for xarray Dataset 
        data_vars = {"FieldN": (["time", "diameter_bin_center"], dict_data['FieldN']),
                     "FieldV": (["time", "velocity_bin_center"], dict_data['FieldV']),
                     "RawData": (["time", "diameter_bin_center", "velocity_bin_center"], dict_data['RawData']),
                    }
        # Define coordinates for xarray Dataset
        coords = get_L1_coords(sensor_name=sensor_name)
        coords['time'] = df['Timestamp'].values
        coords['lat'] = attrs['lat']
        coords['lon'] = attrs['lon']
        coords['crs'] = attrs['crs']

        ##-----------------------------------------------------------
        # Create xarray Dataset
        ds = xr.Dataset(data_vars = data_vars, 
                        coords = coords, 
                        attrs = attrs,
                        )
        
        ##-----------------------------------------------------------
        # Save netCDF4 
        encoding_dict = get_L1_encodings_standards(sensor_name=sensor_name) # TODO: generalize to sensor_name 
        ds.to_netcdf(L1_nc_fpath, encoding=encoding_dict, format="NETCDF4", mode="w")
        
        ##-----------------------------------------------------------
        if verbose:
            print("L1 processing of XXXX ended in {}:.2f}".format(time.time() - t_i))
    
    #-------------------------------------------------------------------------.

# Parquet schema 

# TODO: maybe found a better way --> click
# https://click.palletsprojects.com/en/8.0.x/ 

if __name__ == '__main__':
    main() # if using click     
    # Otherwise:     
    parser = argparse.ArgumentParser(description='L0 and L1 data processing')
    parser.add_argument('--base_dir', type=str)
    parser.add_argument('--L0_processing', type=str, default='True')
    parser.add_argument('--L1_processing', type=str, default='True')
    parser.add_argument('--force', type=str, default='False')                    
    
    L0_processing=True, L1_processing=True, force=False
    
    args = parser.parse_args()
    if args.force == 'True':
        force = True
    else: 
        force = False
    if args.L0_processing == 'True':
        L0_processing = True
    else: 
        L0_processing = False 
     if args.L1_processing == 'True':
        L1_processing = True
    else: 
        L1_processing = False   
        
    main(base_dir = base_dir, 
         L0_processing=L0_processing, 
         L1_processing=L1_processing,
         force=force)
 
