#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:59:21 2021

@author: ghiggi
"""
import os
os.chdir("/home/ghiggi/Projects/parsiveldb")
import glob 
import shutil
import argparse # prefer click ... https://click.palletsprojects.com/en/8.0.x/
import click
import time
import dask 
import dask.array as da
import numpy as np 
import xarray as xr 

from parsiveldb.io import check_folder_structure
from parsiveldb.io import check_valid_varname
from parsiveldb.io import check_L0_standards
from parsiveldb.io import check_L1_standards
from parsiveldb.io import get_attrs_standards
from parsiveldb.io import get_dtype_standards
from parsiveldb.io import _write_to_parquet

from parsiveldb.io import get_raw_field_nbins
from parsiveldb.io import get_L1_coords
from parsiveldb.io import rechunk_L1_dataset
from parsiveldb.io import get_L1_zarr_encodings_standards
from parsiveldb.io import get_L1_nc_encodings_standards

### TODO 4 Kimbo tomorrow
# Implement click instead of using ArgumentParser 
# --> https://click.palletsprojects.com/en/8.0.x/ 


# To think eventually 
# --> Intermediate storage in Zarr: 74.32s
# --> Direct writing to netCDF: 61s 

# A file for each instrument 

## Infer Arrow schema from pandas
# import pyarrow as pa
# schema = pa.Schema.from_pandas(df)

base_dir = "/home/ghiggi/Data_sample_Ticino_2018/Ticino_2018/data/61"

debug_on = False
debug_on = False 
lazy = True
force = True
verbose = True
L0_processing = True
L1_processing = True
keep_zarr = False 

def main(base_dir, L0_processing=True, L1_processing=True, force=False, 
         verbose=True, debug_on=True, lazy=True, keep_zarr=False):
    #-------------------------------------------------------------------------.
    # Whether to use pandas or dask 
    if lazy: 
        import dask.dataframe as dd
    else: 
        import pandas as dd

    #-------------------------------------------------------------------------.
    # - Define directory and file paths 
    df_fpath = "/tmp/Ticino.parquet"   
    L1_nc_fpath = "/tmp/Ticino1.nc"  
    
    #-------------------------------------------------------------------------.
    # - Check if a directory already exists 
    # - TODO ADD: force=True ... remove existing file 
    #             force=False ... raise error if file already exists 
    
    # check_folder_structure(base_dir, campaign_name, force=True)

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
    attrs['disdrodb_id'] = ''
 
    ###############################
    #### Perform L0 processing ####
    ###############################
    if L1_processing: 
        #----------------------------------------------------------------.
        if verbose:
            print("L1 processing of {} started".format(attrs['disdrodb_id']))
        t_i = time.time() 
        # - Retrieve filepaths of raw data files 
        file_list = sorted(glob.glob(os.path.join(base_dir,"*")))
        if debug_on: 
            file_list = file_list[0:3]
        
        #----------------------------------------------------------------.
        # - Define raw data headers 
        raw_data_columns = ['id',
                            'latitude',
                            'longitude',
                            'time',
                            'temperature_sensor',
                            'datalogger_power',
                            'datalogger_sensor_status',
                            'rain_rate',
                            'acc_rain_amount',
                            'code_4680',
                            'code_4677',
                            'reflectivity_16bit',
                            'mor',
                            'amplitude',
                            'n_particles',
                            'heating_current',
                            'voltage',
                            'sensor_status',
                            'rain_amount_absolute',
                            'error_code',
                            'FieldN',
                            'FieldV',
                            'RawData',
                            'Unknow_column',
                            ]
        check_valid_varname(raw_data_columns)   
        
        ##------------------------------------------------------.      
        # Determine dtype based on standards 
        dtype_dict = get_dtype_standards()
        # dtype_dict['FieldN'] = 'U'
        # dtype_dict['FieldV'] = 'U'
        # dtype_dict['RawData'] = 'U'
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
            t_f = time.time() - t_i
            print("L0 processing of {} ended in {:.2f}s".format(attrs['disdrodb_id'], t_f))
    
    #-------------------------------------------------------------------------.
    ###############################
    #### Perform L1 processing ####
    ###############################
    if L1_processing: 
        ##-----------------------------------------------------------
        t_i = time.time()  
        if verbose:
            print("L1 processing of {} started".format(attrs['disdrodb_id']))
        
        ##-----------------------------------------------------------
        # Check the L0 df is available 
        if not L0_processing:
            if not os.path.exists(df_fpath):
                raise ValueError("Need to run L0 processing. The {} file is not available.".format(df_fpath))
        
        ##-----------------------------------------------------------
        # Read raw data from parquet file 
        df = dd.read_parquet(df_fpath)
        
        ##-----------------------------------------------------------
        # Subset row sample to debug 
        if not lazy and debug_on:
           df = df.iloc[0:100,:] # df.head(100) 
           
        ##-----------------------------------------------------------
        # Retrieve raw data matrix 
        dict_data = {}
        n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)
        n_timesteps = df.shape[0].compute()
        for key, n_bins in n_bins_dict.items(): 
            
            # Dask based 
            dd_series = df[key].astype(str).str.split(",")
            da_arr = da.stack(dd_series, axis=0)
            # Remove '' at the end 
            da_arr = da_arr[:, 0:n_bins_dict[key]]
            
            # Convert flag values to XXXX
            # dir(da_arr.str)
            # FieldN and FieldV --> -9.999, has floating number 
            
            if key == 'RawData':
                da_arr = da_arr.astype(int)
                da_arr = da_arr.reshape(n_timesteps, n_bins_dict['FieldN'], n_bins_dict['FieldV'])
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
            #     arr = arr.reshape(..., n_bins_dict['FieldN'], n_bins_dict['FieldV'])
            # dict_data[key] = arr
        
        ##-----------------------------------------------------------
        # Define data variables for xarray Dataset 
        data_vars = {"FieldN": (["time", "diameter_bin_center"], dict_data['FieldN']),
                     "FieldV": (["time", "velocity_bin_center"], dict_data['FieldV']),
                     "RawData": (["time", "diameter_bin_center", "velocity_bin_center"], dict_data['RawData']),
                    }
        
        # Define coordinates for xarray Dataset
        coords = get_L1_coords(sensor_name=sensor_name)
        coords['time'] = df['time'].values
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
        # Check L1 standards 
        check_L1_standards(ds)
        
        ##-----------------------------------------------------------    
        # Write to Zarr as intermediate storage 
        # tmp_zarr_fpath = "/tmp/Ticino1.zarr"
        # ds = rechunk_L1_dataset(ds, sensor_name=sensor_name)
        # zarr_encoding_dict = get_L1_zarr_encodings_standards(sensor_name=sensor_name)
        # ds.to_zarr(tmp_zarr_fpath, encoding=zarr_encoding_dict)  
        
        ##-----------------------------------------------------------  
        # Write L1 dataset to netCDF 
        ds = rechunk_L1_dataset(ds, sensor_name=sensor_name) # very important for fast writing !!!
        nc_encoding_dict = get_L1_nc_encodings_standards(sensor_name=sensor_name)
        ds.to_netcdf(L1_nc_fpath, engine="netcdf4", encoding=nc_encoding_dict)
        
        ##-----------------------------------------------------------
        if verbose:
            t_f = time.time() - t_i
            print("L1 processing of {} ended in {:.2f}s".format(attrs['disdrodb_id'], t_f))
    
    #-------------------------------------------------------------------------.

 

if __name__ == '__main__':
    main() # when using click 
    
    # Otherwise:     
    # parser = argparse.ArgumentParser(description='L0 and L1 data processing')
    # parser.add_argument('--base_dir', type=str)
    # parser.add_argument('--L0_processing', type=str, default='True')
    # parser.add_argument('--L1_processing', type=str, default='True')
    # parser.add_argument('--force', type=str, default='False')                    
    
    # L0_processing=True, L1_processing=True, force=False
    
    # args = parser.parse_args()
    # if args.force == 'True':
    #     force = True
    # else: 
    #     force = False
    # if args.L0_processing == 'True':
    #     L0_processing = True
    # else: 
    #     L0_processing = False 
    #  if args.L1_processing == 'True':
    #     L1_processing = True
    # else: 
    #     L1_processing = False   
        
    # main(base_dir = base_dir, 
    #      L0_processing=L0_processing, 
    #      L1_processing=L1_processing,
    #      force=force)
 
