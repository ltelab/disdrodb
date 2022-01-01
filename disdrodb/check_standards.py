#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------.
# Copyright (c) 2021-2022 DISDRODB developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#-----------------------------------------------------------------------------.
import logging
logger = logging.getLogger(__name__)


def available_sensor_name():
    sensor_list = ['Parsivel', 'Parsivel2', 'ThiesLPM']
    return sensor_list

def check_sensor_name(sensor_name): 
    if not isinstance(sensor_name, str):
        logger.exception("'sensor_name' must be a string'")
        raise TypeError("'sensor_name' must be a string'")
    if sensor_name not in available_sensor_name():
        msg = f"Valid sensor_name are {available_sensor_name()}"
        logger.exception(msg)
        raise ValueError(msg)
    return 
 
def check_L0_column_names(x):
    # Allow TO_BE_SPLITTED, TO_BE_PARSED
    pass

def check_L0_standards(x):
    # TODO:
    pass

def check_L1_standards(x):
    # TODO:
    pass

def check_L2_standards(x): 
    # TODO:
    pass

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

#### KIMBO CODE TO REVIEW
def get_dtype_range_values():
    
    import datetime
    
    dtype_range_values = {
            'id': [0, 4294967295],
            'rain_rate_16bit': [0, 9999.999],
            'rain_rate_32bit': [0, 9999.999],
            'rain_accumulated_16bit': [0, 300.00],
            'rain_accumulated_32bit': [0, 300.00],
            'rain_amount_absolute_32bit': [0, 999.999],
            'reflectivity_16bit': [-9.999, 99.999],
            'reflectivity_32bit': [-9.999, 99.999],
            'rain_kinetic_energy': [0, 999.999],
            'snowfall_intensity': [0, 999.999],
            'mor_visibility': [0, 20000],
            'weather_code_SYNOP_4680': [0, 99],
            'weather_code_SYNOP_4677': [0, 99],
            'n_particles': [0, 99999],  #For debug, [0, 99999]
            'n_particles_all': [0, 8192],
            'sensor_temperature': [-99, 100],
            'temperature_PBC': [-99, 100],
            'temperature_right': [-99, 100],
            'temperature_left': [-99, 100],
            'sensor_heating_current': [0, 4.00],
            'sensor_battery_voltage': [0, 30.0],
            'sensor_status': [0, 3],
            'laser_amplitude': [0, 99999],
            'error_code': [0,3],
            'datalogger_temperature': [-99, 100],
            'datalogger_voltage': [0, 30.0],
            'datalogger_error': [0,3],
            
            'latitude': [-90000, 90000],
            'longitude': [-180000, 180000],
            
            'time': [datetime.datetime(1900, 1, 1), datetime.datetime.now()]
           
            }
    return dtype_range_values


def get_dtype_max_digit():

    dtype_max_digit ={
            'id': [8],  #Maybe to change in the future
            'latitude': [9],
            'longitude': [15],
            'time': [19],
            'datalogger_temperature': [4],
            'datalogger_voltage': [4],
            'datalogger_error': [1],
            
            'rain_rate_16bit': [8],
            'rain_rate_32bit': [8],
            'rain_accumulated_16bit': [7],
            'rain_accumulated_32bit': [7],
            'rain_amount_absolute_32bit': [7],
            'reflectivity_16bit': [6],
            'reflectivity_32bit': [6],
            'rain_kinetic_energy': [7],
            'snowfall_intensity': [7],
            'mor_visibility': [4],
            'weather_code_SYNOP_4680': [2],
            'weather_code_SYNOP_4677': [2],
            'n_particles': [5],
            'n_particles_all': [8],
            'sensor_temperature': [3],
            'temperature_PBC': [3],
            'temperature_right': [3],
            'temperature_left': [3],
            'sensor_heating_current': [4],
            'sensor_battery_voltage': [4],
            'sensor_status': [1],
            'laser_amplitude': [5],
            'error_code': [1],
            'FieldN': [225],
            'FieldV': [225],
            'RawData': [4096],
        }

    return dtype_max_digit

def col_dtype_check(df, file_path, verbose = False):

    dtype_max_digit = get_dtype_max_digit()
    dtype_range_values = get_dtype_range_values()
    
    ignore_colums_range = [
        'FieldN',
        'FieldV',
        'RawData'
        ]
    
    try:
        df = df.compute()
    except AttributeError:
        #Not dask, so is pandas, ignore
        pass
    except Exception as e:
        msg = f'Error on read dataframe, error: {e}'
        if verbose:
            print(msg)
        logger.warning(msg)

    for col in df.columns:
        try:
            # Check if all nan
            if not df[col].isnull().all():
                # Check if data is longer than default in declared in dtype_max_digit
                if not df[col].astype(str).str.len().max() <= dtype_max_digit[col][0]:
                    msg1 = f'{col} has more than %s digits' % dtype_max_digit[col][0]
                    msg2 = f"File: {file_path}, the values {col} have too much digits (%s) in index: %s" % (dtype_max_digit[col][0], df.index[df[col].astype(str).str.len() >= dtype_max_digit[col][0]].tolist())
                    if verbose:
                        print(msg1)
                        print(msg2)
                    logger.warning(msg1)
                    logger.warning(msg2)
                    
                # Check if data is in default range declared in dtype_range_values
                if not df[col].between(dtype_range_values[col][0], dtype_range_values[col][1]).all():
                    msg = f'File: {file_path}, the values {col} in index are not in dtype range: %s' % (df.index[df[col].between(dtype_range_values[col][0], dtype_range_values[col][1]) == False].tolist())
                    if verbose:
                        print(msg)
                    logger.warning(msg)
                # Check exact length for FieldN, FieldV and RawData
                # if col == 'FieldN' or col == 'FieldV':
                #     # if df[col].astype(str).str.len().min() =! dtype_max_digit[col][0] or df[col].astype(str).str.len().max() =! dtype_max_digit[col][0]:
                #     if df[col].astype(str).str.len().min() != dtype_max_digit[col][0] or df[col].astype(str).str.len().max() != dtype_max_digit[col][0]:
                #         msg = f'File: {file_path}, the values {col} are not in range: %s' % (dtype_max_digit[col][0].tolist())
                #         if verbose:
                #             print(msg)
                #         logger.warning(msg)
            else:
                msg = f'File: {file_path}, {col} has all nan, check ignored'
                if verbose:
                    print(msg)
                logger.warning(msg)
            pass
                 
        except KeyError:
            if col in ignore_colums_range:
                msg = f'File: {file_path}, no range values for {col}, check ignored'
                if verbose:
                    print(msg)
                logger.warning(msg)
            pass
        except TypeError:
            msg = f'File: {file_path}, {col} is object, check ignored'
            if verbose:
                print(msg)
            logger.warning(msg)
            pass