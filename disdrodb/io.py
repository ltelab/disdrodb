#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:10:59 2021

@author: ghiggi
"""
import os
import shutil
import glob
import numpy as np 
import pandas as pd
import zarr
import dask.dataframe

from disdrodb.logger import log
from disdrodb.standards import get_diameter_bin_center
from disdrodb.standards import get_diameter_bin_lower
from disdrodb.standards import get_diameter_bin_upper
from disdrodb.standards import get_diameter_bin_width
from disdrodb.standards import get_velocity_bin_center
from disdrodb.standards import get_velocity_bin_lower
from disdrodb.standards import get_velocity_bin_upper
from disdrodb.standards import get_velocity_bin_width
from disdrodb.standards import get_raw_field_nbins
    
## Kimbo
# - dtype, attrs standards
# - coordinate standards ? 
#   get_velocity_bin_center(): 
    # - args: instrument=Parsivel,Thies .. if, elif 
# - Add L0, L1 and L2 folder 

#-----------------------------------------------------------------------------.

logger = None

def check_processed_folder(raw_dir):
    # Check if processed folder exist
    processed_campaign_dir = os.path.join(raw_dir)
    
    if not os.path.isdir(processed_campaign_dir):
        try:
            os.makedirs(processed_campaign_dir)
            return processed_campaign_dir
        except (Exception) as e:
            raise FileNotFoundError(f"Can not create folder <processed>. Error: {e}")
    else:
        return processed_campaign_dir


def check_folder_structure(raw_dir, campaign_name, processed_path):
    """Create the folder structure required for data processing"""
    # Define directories 
    # raw_campaign_dir = os.path.join(raw_dir, "raw", campaign_name)
    # In Ticino_2018 there is data folder and not raw
    
    processed_campaign_dir = check_processed_folder(processed_path)
    
    # Start logger
    global logger
    logger = log(processed_path, 'io')
    
    # Create station subfolder if need it
    for station_folder in glob.glob(os.path.join(raw_dir,"data", "*")):
        try:
            station_folder_path = os.path.join(processed_path, os.path.basename(os.path.normpath(station_folder)))
            os.makedirs(station_folder_path)
            logger.debug(f'Created {station_folder_path}')
            try:
                l0_folder_path = os.path.join(station_folder_path, 'l0')
                os.makedirs(l0_folder_path)
                logger.debug(f'Created {l0_folder_path}')
            except FileExistsError:
                logger.debug(f'Found {l0_folder_path}')
                pass
            except (Exception) as e:
                msg = f"Can not create folder l0 inside <station_folder_path>. Error: {e}"
                logger.exception(msg)
                raise FileNotFoundError(msg)
            try:
                l1_folder_path = os.path.join(station_folder_path, 'l1')
                os.makedirs(l1_folder_path)
                logger.debug(f'Created {l1_folder_path}')
            except FileExistsError:
                logger.debug(f'Found {l1_folder_path}')
                pass
            except (Exception) as e:
                msg = f"Can not create folder l01inside <l1_folder_path>. Error: {e}"
                logger.exception(msg)
                raise FileNotFoundError(msg)
        except FileExistsError:
            logger.debug(f'Found {station_folder_path}')
            pass
        except (Exception) as e:
            msg = f"Can not create folder the device folder inside <processed>. Error: {e}"
            logger.exception(msg)
            raise FileNotFoundError(msg)
            
    return 
            
def _available_sensors():
    sensor_list = ['Parsivel', 'Parsivel2', 'ThiesLPM']
    return sensor_list

def _check_sensor_name(sensor_name): 
    if not isinstance(sensor_name, str):
        logger.exception("'sensor_name' must be a string'")
        raise TypeError("'sensor_name' must be a string'")
    if sensor_name not in _available_sensors():
        logger.exception(f"Valid sensor_name are {_available_sensors()}")
        raise ValueError(f"Valid sensor_name are {_available_sensors()}")
    return 

def _write_to_parquet(df, path, campaign_name, force):  
    # Check if a file already exists (and remove if force=True)
    fpath = path + '/' + campaign_name + '.parquet'
    if os.path.exists(fpath):
        if not force:
            logger.error(f"--force is False and a file already exists at:{fpath}")
            raise ValueError(f"--force is False and a file already exists at:{fpath}")
        try:
            os.remove(fpath)
        except IsADirectoryError:
            try:
                os.rmdir(fpath)
            except OSError:
                try:
                    # shutil.rmtree(fpath.rpartition('.')[0])
                    for f in glob.glob(fpath + '/*'):
                        try:
                            os.remove(f)
                        except OSError as e:
                            msg = f"Can not delete file {f}, error: {e.strerror}"
                            logger.exception(msg)
                    os.rmdir(fpath)
                except (Exception) as e:
                    logger.error(f"Something wrong with: {fpath}")
                    raise ValueError(f"Something wrong with: {fpath}")
        logger.info(f"Deleted folder {fpath}")
        
    ##-------------------------------------------------------------------------.
    # Options 
    compression = 'snappy' # 'gzip', 'brotli, 'lz4', 'zstd'
    row_group_size = 100000 
    engine = "pyarrow"
    # Save to parquet 
    if isinstance(df, pd.DataFrame): 
        try:
            df.to_parquet(fpath,
                          engine = engine,
                          compression = compression,
                          row_group_size = row_group_size)
            logger.info(f'Converted data file in {path} to parquet')  
        except (Exception) as e:
            logger.exception(f"The Pandas DataFrame cannot be written as a parquet file, the error is {e}")
            raise ValueError(f"The Pandas DataFrame cannot be written as a parquet file, the error is {e}")
            
    elif isinstance(df, dask.dataframe.DataFrame): 
        # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html 
        try:
            df.repartition(npartitions=1)
            df.to_parquet(fpath , 
                          schema = 'infer', 
                          engine = engine, 
                          row_group_size = row_group_size,
                          compression = compression, 
                          write_metadata_file=False)
            logger.info(f'Converted data file in {path} to parquet')                 
        except (Exception) as e:
            logger.exception(f"The Dask DataFrame cannot be written as a parquet file, the error is {e}")
            raise ValueError(f"The Dask DataFrame cannot be written as a parquet file, the error is {e}")
    else:
        raise NotImplementedError("Pandas or Dask DataFrame is required.")
        
    return 

 
 

## Infer Arrow schema from pandas
# schema = pa.Schema.from_pandas(df)

## dtype = {'col': pd.api.types.CategoricalDtype(['a', 'b', 'c'])}


## Dask 
# schema = "infer"
# overwrite = force 
# partition_on 
# --> 

# - row_group_size = 100000


#-----------------------------------------------------------------------------.
# TODO correct values 


def _get_default_nc_encoding(chunks, dtype='float32'):
    encoding_kwargs = {} 
    encoding_kwargs['dtype']  = dtype
    encoding_kwargs['zlib']  = True
    encoding_kwargs['complevel']  = 4
    encoding_kwargs['shuffle']  = True
    encoding_kwargs['fletcher32']  = False
    encoding_kwargs['contiguous']  = False
    encoding_kwargs['chunksizes']  = chunks
 
    return encoding_kwargs

def _get_default_zarr_encoding(dtype='float32'):
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    encoding_kwargs = {} 
    encoding_kwargs['dtype']  = dtype
    encoding_kwargs['compressor']  = compressor 
    return encoding_kwargs

##----------------------------------------------------------------------------.
def get_L1_coords(sensor_name): 
    _check_sensor_name(sensor_name=sensor_name)
    coords = {} 
    coords["diameter_bin_center"] = get_diameter_bin_center(sensor_name=sensor_name)
    coords["diameter_bin_lower"] = (["diameter_bin_center"], get_diameter_bin_lower(sensor_name=sensor_name))
    coords["diameter_bin_upper"] = (["diameter_bin_center"], get_diameter_bin_upper(sensor_name=sensor_name))
    coords["diameter_bin_width"] = (["diameter_bin_center"], get_diameter_bin_width(sensor_name=sensor_name))
    coords["velocity_bin_center"] = (["velocity_bin_center"], get_velocity_bin_center(sensor_name=sensor_name))
    coords["velocity_bin_lower"] = (["velocity_bin_center"], get_velocity_bin_lower(sensor_name=sensor_name))
    coords["velocity_bin_upper"] = (["velocity_bin_center"], get_velocity_bin_upper(sensor_name=sensor_name))
    coords["velocity_bin_width"] = (["velocity_bin_center"], get_velocity_bin_width(sensor_name=sensor_name))
    return coords 
                 
def get_L1_chunks(sensor_name):
    _check_sensor_name(sensor_name=sensor_name)
    if sensor_name == "Parsivel":
        chunks_dict = {'FieldN': (5000,32),
                       'FieldV': (5000,32),
                       'RawData': (5000,32,32),
                      }
    elif sensor_name == "Parsivel2":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
        
    else:
        logger.exception(f'L0 chunks for sensor {sensor_name} are not yet defined')
        raise ValueError(f'L0 chunks for sensor {sensor_name} are not yet defined')
    return chunks_dict

def get_L1_dtype():
    # Float 32 or Float 64 (f4, f8)
    # (u)int 8 16, 32, 64   (u/i  1 2 4 8)
    dtype_dict = {'FieldN': 'float32',
                  'FieldV': 'float32',  
                  'RawData': 'int64',   # TODO: uint16? uint32 check largest number occuring, and if negative
                 }
    return dtype_dict

def get_L1_nc_encodings_standards(sensor_name): 
    # Define variable names 
    vars = ['FieldN', 'FieldV', 'RawData']   
    # Get chunks based on sensor type
    chunks_dict = get_L1_chunks(sensor_name=sensor_name) 
    dtype_dict = get_L1_dtype()
    # Define encodings dictionary 
    encoding_dict = {} 
    for var in vars:
        encoding_dict[var] = _get_default_nc_encoding(chunks=chunks_dict[var],
                                                      dtype=dtype_dict[var]) # TODO
        # encoding_dict[var]['scale_factor'] = 1.0
        # encoding_dict[var]['add_offset']  = 0.0
        # encoding_dict[var]['_FillValue']  = fill_value
        
    return encoding_dict 

def rechunk_L1_dataset(ds, sensor_name):
    chunks_dict = get_L1_chunks(sensor_name=sensor_name) 
    for var, chunk in chunks_dict.items():
       if chunk is not None: 
           ds[var] = ds[var].chunk(chunk) 
    return ds 
    
def get_L1_zarr_encodings_standards(sensor_name): 
    # Define variable names 
    vars = ['FieldN', 'FieldV', 'RawData']   
    dtype_dict = get_L1_dtype()
    # Define encodings dictionary 
    encoding_dict = {} 
    for var in vars:
        encoding_dict[var] = _get_default_zarr_encoding(dtype=dtype_dict[var]) # TODO        
    return encoding_dict 
     
def check_valid_varname(x):
    pass

def check_L0_standards(x):
    pass

def check_L1_standards(x):
    pass

def check_L2_standards(x): 
    pass

def get_attrs_standards(): 
    list_attrs = [# Description 
                  "title", "description",
                  "source", "history", "conventions",
                  "campaign_name", "project_name",
                  # Location
                  "station_id", "station_name", "station_number", 
                  "location", "country", "continent",
                  "latitude", "longitude", "altitude", "crs", "proj4", "EPSG"
                  "latitude_unit", "longitude_unit", "altitude_unit",
                  # Sensor info 
                  "sensor_name", "sensor_long_name", 
                  "sensor_wavelegth", "sensor_serial_number",
                  "firmware_IOP", "firmware_DSP", "firmware_version", 
                  "sensor_beam_width", "sensor_nominal_width", 
                  "temporal_resolution", "measurement_interval",
                  # Attribution 
                  "contributors", "authors", "institution",
                  "reference", "documentation",
                  "website","source_repository",
                  "doi", "contact", "contact_information",
                  # Source datatype 
                  "source_data_format", 
                  # DISDRO DB attrs 
                  "obs_type", "level", "disdrodb_id",
                 ]  
    attrs_dict = {key: '' for key in list_attrs}
    return attrs_dict
   
def get_L0_dtype_standards(): 
    dtype_dict = {                                 # Kimbo option
        "id": "uint32",
        "rain_rate_16bit": 'float32',
        "rain_rate_32bit": 'float32',
        "rain_accumulated_16bit":   'float32',
        "rain_accumulated_32bit":   'float32',
        
        "rain_amount_absolute_32bit": 'float32', 
        "reflectivity_16bit": 'float32',
        "reflectivity_32bit": 'float32',
        
        "rain_kinetic_energy"  :'float32',
        "snowfall_intensity": 'float32',
        
        "mor_visibility"    :'uint16',
        
        "weather_code_SYNOP_4680":'uint8',             
        "weather_code_SYNOP_4677":'uint8',              
        "weather_code_METAR_4678":'object', #TODO
        "weather_code_NWS":'object', #TODO
        
        "n_particles"     :'uint32',
        "n_particles_all": 'uint32',
        
        "sensor_temperature": 'uint8',
        "temperature_PBC" : 'object', #TODO
        "temperature_right" : 'object', #TODO
        "temperature_left":'object', #TODO
        
        "sensor_heating_current" : 'float32',
        "sensor_battery_voltage" : 'float32',
        "sensor_status"   : 'uint8',
        "laser_amplitude" :'uint32',
        "error_code"      : 'uint8',          

        # Custom ields       
        "Unknow_column": "object",
        "datalogger_temperature": "object",
        "datalogger_voltage": "object",
        'datalogger_error': 'uint8',
        
        # Data fields (TODO) 
        "FieldN": 'object',
        "FieldV": 'object',
        "RawData": 'object',
        
        # Coords 
        "latitude" : 'float32',
        "longitude" : 'float32',
        "altitude" : 'float32',
        
         # Dimensions
        'time': 'object',
        
        #Temp variables
        'temp': 'object',
        "Debug_data" : 'object',
        'All_0': 'object',
        'error_code?': 'object',
        'unknow2': 'object',
        'unknow3': 'object',
        'unknow4': 'object',
        'unknow5': 'object',
        'unknow': 'object',
        'unknow6': 'object',
        'power_supply_voltage': 'object',
        'A_voltage2?' : 'object'
        
    }
    return dtype_dict

def get_dtype_standards_all_object(): 
    dtype_dict = get_L0_dtype_standards()
    for i in dtype_dict:
        dtype_dict[i] = 'object'
        
    return dtype_dict

def get_flags(device):
    if device == 'Parsivel':
        flag_dict = {
            'sensor_status': [
                1,
                2,
                3
            ],
            'datalogger_error' : [
                1
            ],
            'error_code' : [
                1,
                2
            ]
            }
    return flag_dict

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
