#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:59:21 2021

@author: ghiggi
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 18:39:40 2021

@author: ghiggi
"""
import os
import glob 
import argparse # prefer click ... https://click.palletsprojects.com/en/8.0.x/
import click
import time

import dask 
# import pandas as dd
import dask.dataframe as dd
import numpy as np 
import xarray as xr 

from parsiveldb import XXXX
from parsiveldb import check_L0_standards
from parsiveldb import check_L1_standards
from parsiveldb import check_L2_standards
from parsiveldb import get_attrs_standards
from parsiveldb import get_dtype_standards
from parsiveldb import get_L1_encodings_standards
from parsiveldb import get_L1_chunks_standards

# A file for each instrument 
# TODO: force=True ... remove existing file 
#       force=False ... raise error if file already exists 
# Time execution 
print('- Ln, validation and test sets: {:.2f}s'.format(time.time() - t_i))


dirpath = "/home/ghiggi/Data_sample_Ticino_2018/Ticino_2018/data/61/*"
file_list = glob.glob(dirpath)[0:3]
file_list = glob.glob(dirpath)
debug_on = True 

def main(base_dir, L0_processing=True, L1_processing=True, force=False, verbose=True):
    #-------------------------------------------------------------------------.
    
    df_fpath = "/tmp/Ticino.parquet"   
    nc_fpath = "/tmp/Ticino.nc"   
    # - Define attributes 
    attrs = get_attrs_standards()
    attrs['Title'] = 'aaa'
    attrs['Title'] = ...

    ###############################
    #### Perform L0 processing ####
    ###############################
    if L1_processing: 
        if verbose:
            print("L0 processing of XXXX started")
        t_i = time.time() 
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
                            'FieldN',
                            'FieldV',
                            'RAW_data',
                            'Unknow_column'
                            ]
        check_valid_varname(raw_data_columns)   
        ##------------------------------------------------------.      
        # Determine dtype based on standards 
        dtype_dict = get_dtype_standards()
        dtype_dict = {column: dtype_dict[column] for column in raw_data_columns}
        ##------------------------------------------------------.
        # Define reader options 
        reader_kwargs = {}
        reader_kwargs['compression'] = 'gzip'
        reader_kwargs['delimiter'] = ','
        reader_kwargs["on_bad_lines"] = 'skip'
        reader_kwargs["engine"] = 'python'
        if not debug_on:
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
                df = df.dropna(how='all')  
                # - Replace custom NA with standard flags 
                # TODO !!! 
                # - Append to the list of dataframe 
                list_df.append(df)
                
            except (Exception, ValueError) as e:
              print("{} has been skipped. The error is {}".format(filename, e))
              list_skipped_files.append("{} has been skipped. The error is {}".format(filename, e))
            
        print('{} files on {} has been skipped.'.format(len(list_skipped_files), len(file_list)))
        
        ##------------------------------------------------------.
        # Concatenate the dataframe 
        try:
            df = dd.concat(temp_list, axis=0, ignore_index = True)
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
            print("L0 processing of XXXX ended in {}:.2f}".format(time.time() - t_i)))
    
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
        
        # df = df.head(10) 
        
        fields_length = {"Error_code": 32,
                         "FieldN": 32,
                         "FieldV": 1024,
                         }
        
        # fields_length = {"FieldN": 32,
        #                  "FieldV": 32,
        #                    "RAW_data": 1024,
        #                  }
        dict_data = {}
        for key, n_bins in fields_length.items(): 
            np_arr_str =  df[key].values.compute().astype(str) # Dask
            # np_arr_str =  df[key].values.astype(str)           # Pandas 
            list_arr_str = np.char.split(np_arr_str,",")
            arr_str = np.stack(list_arr_str, axis=0) 
            arr = arr_str[:, 0:n_bins].astype(float)               # <---- TO ADAPT 
            dict_data[key] = arr
        
        # dict_data['FieldV'].values.reshape(n_timesteps, 32, 32)
        # - Conversion to xarray 
        diameter_bin_center =
        coords = {
             "diameter_bin_center":  np.arange(0,32),
             "time": df['Timestamp'].values, 
                                               
        #     lat=(["x", "y"], lat),
        #     time=time,
        #     reference_time=reference_time,
        }
        ds = xr.Dataset(data_vars={
                            "FieldN": (["time", "diameter_bin_center"], dict_data['Error_code']),
                            "FieldV": (["time", "velocity_bin_center"], dict_data['FieldN']),
                            # "Raw": (["diameter_bin_center", "velocity_bin_center", "time"], precipitation),
                        },
                        coords = coords, 
                        attrs = attrs,
                    )
 
        # - Save netcdf 
        L1_nc_fpath = '' # TBD
        encoding = get_L1_encodings_standards()
        chunks = get_L1_chunks_standards()
        ds.to_netcdf(L1_nc_fpath, encoding=encoding, chunks=chunks)
        
        if verbose:
            print("L1 processing of XXXX ended in {}:.2f}".format(time.time() - t_i)))
    #-------------------------------------------------------------------------.

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
 
