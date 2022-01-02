#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 13:42:14 2022

@author: ghiggi
"""                          
##----------------------------------------------------------------------------.
#### Numeric EDA
filepath = file_list[0]
df = read_raw_data(filepath, column_names=column_names,  
                   reader_kwargs=reader_kwargs, lazy=False)
 
from disdrodb.dev_tools import print_df_summary_stats
print_df_summary_stats(df)

## TODO: 
# Check that min is < expected max 
# Check that max is > expected min 
# - Require removing na flag values (i.e. -99, ....)

##----------------------------------------------------------------------------. 
#### Character EDA 
filepath = file_list[0]
str_reader_kwargs = reader_kwargs.copy() 
str_reader_kwargs['dtype'] = str # or object 
df = read_raw_data(filepath, column_names=None,  
                   reader_kwargs=str_reader_kwargs, lazy=False)


arr = df.iloc[:,1]
check_constant_nchar(arr) 

# Check only on a random (i.e. second) row 
string = arr[1]

nchar = len(string)      
str_is_not_number(string)
str_is_number(string)
str_is_integer(string)
str_has_decimal_digits(string) # check if has a comma and right digits 
get_decimal_digits(string)     # digits right of the comma 
get_integer_digits(string)     # digits left of the comma 
get_n_digits(string)

####--------------------------------------------------------------------------.
#### Character checks 
def check_constant_nchar(arr):
    arr = np.asarray(arr)
    if arr.dtype != object: 
        arr = arr.astype(str)
    if arr.dtype.char != 'U':
        raise TypeError("Expecting str dtype.")
    # Get number of characters (include .)              
    str_nchars = np.char.str_len(arr)   
    str_nchars_unique = np.unique(str_nchars)
    if len(str_nchars_unique) != 1: 
        raise ValueError("Non-unique string length !")                  
 
def str_is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def str_is_not_number(string):
    return not str_is_number(string) 

def str_is_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False
    
def str_has_decimal_digits(string):
    if len(string.split(".")) == 2: 
        return True 
    else:
        return False 
        
def get_decimal_digits(string): 
    if str_has_decimal_digits(string):
        return len(string.split(".")[1])
    else:
        return 0
    
def get_integer_digits(string): 
    if str_is_integer(string):
        return len(string)
    if str_has_decimal_digits(string): 
        return len(string.split(".")[0])
    else:
        return 0 
    
def get_n_digits(string): 
    if str_is_not_number(string):
        return 0 
    if str_has_decimal_digits(string): 
        return len(string) - 1 # remove . 
    else: 
        return len(string) 
        

####--------------------------------------------------------------------------.
#### Instrument default string characteristics 
def get_field_natural_digits_dict(sensor_name):
    pass 

def get_field_decimal_digits_dict(sensor_name): 
    pass

def get_field_n_digits_dict(sensor_name):
    ### TODO: . count as digit 
    if sensor_name == "Parsivel":
        digits_dict = {
                # Optional 
                # 'id': [8],  # Maybe to change in the future
                # 'latitude': [9],
                # 'longitude': [15],
                # 'time': [19],
                'datalogger_temperature': 4,
                'datalogger_voltage': 4,
                'datalogger_error': 1,
                # Mandatory
                'rain_rate_16bit': 8,
                'rain_rate_32bit': 8,
                'rain_accumulated_16bit': 7,
                'rain_accumulated_32bit': 7,
                'rain_amount_absolute_32bit': 7,
                'reflectivity_16bit': 6,
                'reflectivity_32bit': 6,
                'rain_kinetic_energy': 7,
                'snowfall_intensity': 7,
                'mor_visibility': 4,
                'weather_code_SYNOP_4680': 2,
                'weather_code_SYNOP_4677': 2,
                'n_particles': 5,
                'n_particles_all': 8,
                'sensor_temperature': 3,
                'sensor_heating_current': 4,
                'sensor_battery_voltage': 4,
                'sensor_status': 4,
                'laser_amplitude': 5,
                'error_code': 1,
                'FieldN': 225,
                'FieldV': 225,
                'RawData': 4096,
        }
    elif sensor_name == "Parsivel2":
        digits_dict = {
                # Optional 
                # 'id': [8],  # Maybe to change in the future
                # 'latitude': [9],
                # 'longitude': [15],
                # 'time': [19],
                'datalogger_temperature': 4,
                'datalogger_voltage': 4,
                'datalogger_error': 1,
                # Mandatory
                'rain_rate_16bit': 8,
                'rain_rate_32bit': 8,
                'rain_accumulated_16bit': 7,
                'rain_accumulated_32bit': 7,
                'rain_amount_absolute_32bit': 7,
                'reflectivity_16bit': 6,
                'reflectivity_32bit': 6,
                'rain_kinetic_energy': 7,
                'snowfall_intensity': 7,
                'mor_visibility': 4,
                'weather_code_SYNOP_4680': 2,
                'weather_code_SYNOP_4677': 2,
                'n_particles': 5,
                'n_particles_all': 8,
                'sensor_temperature': 3,
                'temperature_PBC': 3,
                'temperature_right': 3,
                'temperature_left': 3,
                'sensor_heating_current': 4,
                'sensor_battery_voltage': 4,
                'sensor_status': 4,
                'laser_amplitude': 5,
                'error_code': 1,
                'FieldN': 225,
                'FieldV': 225,
                'RawData': 4096,
        }
    else: 
        raise NotImplementedError
            
    return digits_dict

####--------------------------------------------------------------------------.
#### Numeric checks 
# TODO: get_field_flag_values 
# TODO: get_field_value_realistic_range  # when removing flags 

from disdrodb.io import col_dtype_check
col_dtype_check(df, filename, verbose)

def get_field_value_range_dict():
    
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

####--------------------------------------------------------------------------.


def col_dtype_check(df, file_path, verbose = False):

    dtype_max_digit = get_field_n_digits_dict()
    dtype_range_values = get_field_value_range_dict()
    
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
                if not df[col].astype(str).str.len().max() <= dtype_max_digit[col]:
                    msg1 = f'{col} has more than %s digits' % dtype_max_digit[col]
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

###----------------------------------------------------------------------------.
# A function that makes numeric and character checks 
def check_reader_and_columns(filepath, 
                             column_names,  
                             reader_kwargs, 
                             lazy=False):
    # Define kwargs to have all columns as dtype 
    str_reader_kwargs = reader_kwargs.copy() 
    str_reader_kwargs['dtype'] = str # or object
    # Load numeric df 
    numeric_df = read_raw_data(filepath, column_names=column_names,  
                               reader_kwargs=reader_kwargs, lazy=lazy)
    # Load str df 
    str_df = read_raw_data(filepath, column_names=column_names,  
                        reader_kwargs=str_reader_kwargs, lazy=lazy)
    #-------------------------------------------------------------------------.
    # TODO all checks
    # - Option to raise Error when incosistency encountered 
    # - Option to add all errors in a list 
    # - Option to print (verbose) 
    # - List column names analyzed
    # - List column names not analyzed 
    # - List column names without inconsistencies 