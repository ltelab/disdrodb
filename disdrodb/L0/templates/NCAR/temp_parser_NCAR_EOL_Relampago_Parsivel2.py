#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:02:15 2022

@author: kimbo
"""

# -----------------------------------------------------------------------------.
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
# -----------------------------------------------------------------------------.

### THIS SCRIPT PROVIDE A TEMPLATE FOR PARSER FILE DEVELOPMENT
#   FROM RAW DATA FILES
# - Please copy such template and modify it for each parser ;)
# - Additional functions/tools to ease parser development are welcome

# -----------------------------------------------------------------------------.
import os
import logging
import pandas as pd

# Directory
from disdrodb.L0.io import (
    check_directories,
    get_campaign_name,
    create_directory_structure,
)

# Tools to develop the parser
from disdrodb.L0.template_tools import (
    check_column_names,
    infer_df_str_column_names,
    print_df_first_n_rows,
    print_df_random_n_rows,
    print_df_column_names,
    print_valid_L0_column_names,
    get_df_columns_unique_values_dict,
    print_df_columns_unique_values,
    print_df_summary_stats,
)

# L0A processing
from disdrodb.L0.L0A_processing import (
    read_raw_data,
    get_file_list,
    read_L0A_raw_file_list,
    cast_column_dtypes,
    write_df_to_parquet,  # TODO: add code to write to parquet a single file in 8.3 ... to check it works
)

# Metadata
from disdrodb.L0.metadata import read_metadata

# Standards
from disdrodb.L0.check_standards import check_sensor_name, check_L0A_column_names

# Logger
from disdrodb.utils.logger import create_logger

##------------------------------------------------------------------------.
######################################
#### 1. Define campaign filepaths ####
######################################
raw_dir = "/home/kimbo/data/Campagne/DISDRODB/Raw/NCAR/RELAMPAGO/SAO_BORJA_OTT"
processed_dir = (
    "/home/kimbo/data/Campagne/DISDRODB/Processed/NCAR/RELAMPAGO/SAO_BORJA_OTT"
)
force = False
force = True
lazy = True
lazy = False
verbose = True
debugging_mode = False

####--------------------------------------------------------------------------.
#############################################
#### 2. Here run code to not be modified ####
#############################################
# Initial directory checks
raw_dir, processed_dir = check_directories(raw_dir, processed_dir, force=force)

# Retrieve campaign name
campaign_name = get_campaign_name(raw_dir)

# -------------------------------------------------------------------------.
# Define logging settings
# create_logger(processed_dir, 'parser_' + campaign_name)

# Retrieve logger
# logger = logging.getLogger(campaign_name)
# logger.info('### Script start ###')

# -------------------------------------------------------------------------.
# Create directory structure
create_directory_structure(raw_dir, processed_dir)

# -------------------------------------------------------------------------.
# List stations
list_stations_id = os.listdir(os.path.join(raw_dir, "data"))

####--------------------------------------------------------------------------.
######################################################
#### 3. Select the station for parser development ####
######################################################
station_id = list_stations_id[0]

####--------------------------------------------------------------------------.
##########################################################################
#### 4. List files to process  [TO CUSTOMIZE AND THEN MOVE TO PARSER] ####
##########################################################################
glob_pattern = os.path.join("data", station_id, "*")  # CUSTOMIZE THIS
file_list = get_file_list(
    raw_dir=raw_dir,
    glob_pattern=glob_pattern,
    verbose=verbose,
    debugging_mode=debugging_mode,
)


####--------------------------------------------------------------------------.
##########################################################################
#### 4.1 Retrive metadata from yml files ####
##########################################################################
# Retrieve metadata
attrs = read_metadata(raw_dir=raw_dir, station_id=station_id)

# Retrieve sensor name
sensor_name = attrs["sensor_name"]
check_sensor_name(sensor_name)


####--------------------------------------------------------------------------.
#########################################################################
#### 5. Define reader options [TO CUSTOMIZE AND THEN MOVE TO PARSER] ####
#########################################################################
# Important: document argument need/behaviour

reader_kwargs = {}
# - Define delimiter
reader_kwargs["delimiter"] = "\\n"

# - Avoid first column to become df index !!!
reader_kwargs["index_col"] = False

# - Define behaviour when encountering bad lines
reader_kwargs["on_bad_lines"] = "skip"

# - Define parser engine
#   - C engine is faster
#   - Python engine is more feature-complete
reader_kwargs["engine"] = "python"

# - Define on-the-fly decompression of on-disk data
#   - Available: gzip, bz2, zip
reader_kwargs["compression"] = "infer"

# - Strings to recognize as NA/NaN and replace with standard NA flags
#   - Already included: ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’,
#                       ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’,
#                       ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
reader_kwargs["na_values"] = ["na", "", "error"]

# - Define max size of dask dataframe chunks (if lazy=True)
#   - If None: use a single block for each file
#   - Otherwise: "<max_file_size>MB" by which to cut up larger files
reader_kwargs["blocksize"] = None  # "50MB"

# Skip first row as columns names
reader_kwargs["header"] = None

# - Define encoding
reader_kwargs["encoding"] = "ISO-8859-1"

####--------------------------------------------------------------------------.
####################################################
#### 6. Open a single file and explore the data ####
####################################################
# - Do not assign column names yet to the columns
# - Do not assign a dtype yet to the columns
# - Possibily look at multiple files ;)
# filepath = file_list[0]
# str_reader_kwargs = reader_kwargs.copy()
# str_reader_kwargs['dtype'] = str # or object
# df_str = read_raw_data(filepath,
#                        column_names=None,
#                        reader_kwargs=str_reader_kwargs,
#                        lazy=False)

# # df = pd.read_csv(filepath, header=None)

# a = df_str.to_numpy()
# a = a.reshape(int(len(a)/97),97)
# df = pd.DataFrame(a)

# for col in df:
#     df[col] = df[col].str[3:]

# import numpy as np
# df.columns = np.arange(1,98)


# df.replace("", np.nan, inplace=True)
# df.dropna(how='all', axis=1, inplace=True)

# col = {1: 'rainfall_rate_32bit',
#     2: 'rainfall_accumulated_32bit',
#     3: 'weather_code_synop_4680',
#     4: 'weather_code_synop_4677',
#     5: 'weather_code_metar_4678',
#     6: 'weather_code_nws',
#     7: 'reflectivity_32bit',
#     8: 'mor_visibility',
#     9: 'sample_interval',
#     10: 'laser_amplitude',
#     11: 'number_particles',
#     12: 'sensor_temperature',
#     13: 'sensor_serial_number',
#     14: 'firmware_iop',
#     15: 'firmware_dsp',
#     16: 'sensor_heating_current',
#     17: 'sensor_battery_voltage',
#     18: 'sensor_status',
#     19: 'start_time',
#     20: 'sensor_time',
#     21: 'sensor_date',
#     22: 'station_name',
#     23: 'station_number',
#     24: 'rainfall_amount_absolute_32bit',
#     25: 'error_code',
#     26: 'sensor_temperature_pcb',
#     27: 'sensor_temperature_receiver',
#     28: 'sensor_temperature_trasmitter',
#     30: 'rainfall_rate_16_bit_30',
#     31: 'rainfall_rate_16_bit_1200',
#     32: 'rainfall_accumulated_16bit',
#     33: 'reflectivity_16bit',
#     34: 'rain_kinetic_energy',
#     35: 'snowfall_rate',
#     60: 'number_particles_all',
#     61: 'list_particles',
#     90: 'raw_drop_concentration',
#     91: 'raw_drop_average_velocity',
#     92: 'raw_drop_number'}

# df = df.rename(col, axis=1)

# col_to_drop = [40,41,50,51,93,94,95,96,97]
# df = df.drop(columns=col_to_drop)

# df['weather_code_metar_4678'] = df['weather_code_metar_4678'].str.strip()
# df['weather_code_nws'] = df['weather_code_nws'].str.strip()


# Print first rows
# print_df_first_n_rows(df_str, n = 1, column_names=False)
# print_df_first_n_rows(df_str, n = 5, column_names=False)
# print_df_random_n_rows(df_str, n= 5, column_names=False)  # this likely the more useful

# Retrieve number of columns
# print(len(df_str.columns))

# Look at unique values
# print_df_columns_unique_values(df_str, column_indices=None, column_names=False) # all

# print_df_columns_unique_values(df_str, column_indices=0, column_names=False) # single column

# print_df_columns_unique_values(df_str, column_indices=slice(0,15), column_names=False) # a slice of columns

# get_df_columns_unique_values_dict(df_str, column_indices=slice(0,15), column_names=False) # get dictionary

# Retrieve number of columns
# print(len(df_str.columns))

# Infer columns based on string patterns
# infer_df_str_column_names(df_str, sensor_name=sensor_name)

# Alternatively an empty list of column_names to infer
# ['Unknown' + str(i+1) for i in range(len(df_str.columns))]

# Print valid column names
# - If other names are required, add the key to disdrodb/L0/configs/<sensor_name>/L0A_dtype.yml
# print_valid_L0_column_names(sensor_name)

# Instrument manufacturer defaults

# get_OTT_Parsivel_dict()
# get_OTT_Parsivel2_dict()

####---------------------------------------------------------------------------.
######################################################################
#### 7. Define dataframe columns [TO CUSTOMIZE AND MOVE TO PARSER] ###
######################################################################
# - If a column must be splitted in two (i.e. lat_lon), use a name like: TO_SPLIT_lat_lon
column_names = ["temp"]

# # - Check name validity
# # check_column_names(column_names, sensor_name)

# # - Read data
# filepath = file_list[0]
# df = read_raw_data(filepath=filepath,
#                    column_names=column_names,
#                    reader_kwargs=reader_kwargs,
#                    lazy=False)

# # - Look at the columns and data
# print_df_column_names(df)
# print_df_random_n_rows(df, n= 5)

# # - Check it loads also lazily in dask correctly
# df1 = read_raw_data(filepath=filepath,
#                    column_names=column_names,
#                    reader_kwargs=reader_kwargs,
#                    lazy=True)
# df1 = df1.compute()

# # - Look at the columns and data
# print_df_column_names(df1)
# print_df_random_n_rows(df1, n= 5)

# # - Check are equals
# assert df.equals(df1)

# - Look at values statistics
# print_df_summary_stats(df)

# # - Look at unique values
# print_df_columns_unique_values(df, column_indices=None, column_names=True) # all

# print_df_columns_unique_values(df, column_indices=0, column_names=True) # single column

# print_df_columns_unique_values(df, column_indices=slice(0,10), column_names=True) # a slice of columns

# get_df_columns_unique_values_dict(df, column_indices=slice(0,15), column_names=True) # get dictionary

####---------------------------------------------------------------------------.
#########################################################
#### 8. Implement ad-hoc processing of the dataframe ####
#########################################################
# - This must be done once that reader_kwargs and column_names are correctly defined
# - Try the following code with various file and with both lazy=True and lazy=False
filepath = file_list[0]  # Select also other files here  1,2, ...
lazy = True  # Try also with True when work with False

# ------------------------------------------------------.
#### 8.1 Run following code portion without modifying anthing
# - This portion of code represent what is done by read_L0A_raw_file_list in L0_proc.py
df = read_raw_data(
    filepath=filepath, column_names=column_names, reader_kwargs=reader_kwargs, lazy=lazy
)

# ------------------------------------------------------.
# Check if file empty
if len(df.index) == 0:
    raise ValueError(f"{filepath} is empty and has been skipped.")

# Check column number
if len(df.columns) != len(column_names):
    raise ValueError(f"{filepath} has wrong columns number, and has been skipped.")

# ---------------------------------------------------------------------------.
#### 8.2 Ad-hoc code [TO CUSTOMIZE]
# --> Here specify columns to drop, to split and other type of ad-hoc processing
# --> This portion of code will need to be enwrapped (in the parser file)
#     into a function called df_sanitizer_fun(df, lazy=True). See below ...

# # Example: split erroneous columns
# df_tmp = df['TO_BE_SPLITTED'].astype(str).str.split(',', n=1, expand=True)
# df_tmp.columns = ['datalogger_voltage','rainfall_rate_32bit']
# df = df.drop(columns=['TO_BE_SPLITTED'])
# df = dd.concat([df, df_tmp], axis = 1, ignore_unknown_divisions=True)
# del df_tmp

# Reshape dataframe
a = df.to_numpy()
a = a.reshape(int(len(a) / 97), 97)
df = pd.DataFrame(a)

# Remove number before data
for col in df:
    df[col] = df[col].str[3:]

# Rename columns
import numpy as np

df.columns = np.arange(1, 98)

col = {
    1: "rainfall_rate_32bit",
    2: "rainfall_accumulated_32bit",
    3: "weather_code_synop_4680",
    4: "weather_code_synop_4677",
    5: "weather_code_metar_4678",
    6: "weather_code_nws",
    7: "reflectivity_32bit",
    8: "mor_visibility",
    9: "sample_interval",
    10: "laser_amplitude",
    11: "number_particles",
    12: "sensor_temperature",
    13: "sensor_serial_number",
    14: "firmware_iop",
    15: "firmware_dsp",
    16: "sensor_heating_current",
    17: "sensor_battery_voltage",
    18: "sensor_status",
    19: "start_time",
    20: "sensor_time",
    21: "sensor_date",
    22: "station_name",
    23: "station_number",
    24: "rainfall_amount_absolute_32bit",
    25: "error_code",
    26: "sensor_temperature_pcb",
    27: "sensor_temperature_receiver",
    28: "sensor_temperature_trasmitter",
    30: "rainfall_rate_16_bit_30",
    31: "rainfall_rate_16_bit_1200",
    32: "rainfall_accumulated_16bit",
    33: "reflectivity_16bit",
    34: "rain_kinetic_energy",
    35: "snowfall_rate",
    60: "number_particles_all",
    61: "list_particles",
    90: "raw_drop_concentration",
    91: "raw_drop_average_velocity",
    92: "raw_drop_number",
}

df = df.rename(col, axis=1)

# Cast time
df["time"] = pd.to_datetime(
    df["sensor_date"] + "-" + df["sensor_time"], format="%d.%m.%Y-%H:%M:%S"
)
df = df.drop(columns=["sensor_date", "sensor_time"])

# Drop useless columns
df.replace("", np.nan, inplace=True)
df.dropna(how="all", axis=1, inplace=True)
col_to_drop = [40, 41, 50, 51, 93, 94, 95, 96, 97]
df = df.drop(columns=col_to_drop)

# Trim weather_code_metar_4678 and weather_code_nws
df["weather_code_metar_4678"] = df["weather_code_metar_4678"].str.strip()
df["weather_code_nws"] = df["weather_code_nws"].str.strip()

# Delete invalid columsn by check_L0A
col_to_drop = [
    "sensor_temperature_trasmitter",
    "sensor_temperature_pcb",
    "rainfall_rate_16_bit_1200",
    "sensor_temperature_receiver",
    "snowfall_rate",
    "rain_kinetic_energy",
    "rainfall_rate_16_bit_30",
]
df = df.drop(columns=col_to_drop)

# ---------------------------------------------------------------------------.
#### 8.3 Run following code portion without modifying anthing
# - This portion of code represent what is done by read_L0A_raw_file_list in disdrodb.L0.L0A_processing.py

##----------------------------------------------------.
# Check column names met DISDRODB standards after custom processing (later done by df_sanitizer_fun)
check_L0A_column_names(df, sensor_name=sensor_name)

##----------------------------------------------------.
# Cast dataframe to dtypes
# - Determine dtype based on standards
df = cast_column_dtypes(df, sensor_name=sensor_name)

# ---------------------------------------------------------------------------.
#### 8.4 Check the dataframe looks as desired
print_df_column_names(df)
print_df_random_n_rows(df, n=5)
print_df_columns_unique_values(df, column_indices=2, column_names=True)
print_df_columns_unique_values(df, column_indices=slice(0, 20), column_names=True)

####------------------------------------------------------------------------------.
################################################
#### 9. Simulate parser file code execution ####
################################################
#### 9.1 Define sanitizer function [TO CUSTOMIZE]
# --> df_sanitizer_fun = None  if not necessary ...


def df_sanitizer_fun(df, lazy=False):
    # Import dask or pandas
    # No lazy mode for now
    if lazy:
        import pandas as dd

        df = df.compute()
    else:
        import pandas as dd

    # Reshape dataframe
    a = df.to_numpy()
    a = a.reshape(int(len(a) / 97), 97)
    df = pd.DataFrame(a)

    # Remove number before data
    for col in df:
        df[col] = df[col].str[3:]

    # Rename columns
    import numpy as np

    df.columns = np.arange(1, 98)

    col = {
        1: "rainfall_rate_32bit",
        2: "rainfall_accumulated_32bit",
        3: "weather_code_synop_4680",
        4: "weather_code_synop_4677",
        5: "weather_code_metar_4678",
        6: "weather_code_nws",
        7: "reflectivity_32bit",
        8: "mor_visibility",
        9: "sample_interval",
        10: "laser_amplitude",
        11: "number_particles",
        12: "sensor_temperature",
        13: "sensor_serial_number",
        14: "firmware_iop",
        15: "firmware_dsp",
        16: "sensor_heating_current",
        17: "sensor_battery_voltage",
        18: "sensor_status",
        19: "start_time",
        20: "sensor_time",
        21: "sensor_date",
        22: "station_name",
        23: "station_number",
        24: "rainfall_amount_absolute_32bit",
        25: "error_code",
        26: "sensor_temperature_pcb",
        27: "sensor_temperature_receiver",
        28: "sensor_temperature_trasmitter",
        30: "rainfall_rate_16_bit_30",
        31: "rainfall_rate_16_bit_1200",
        32: "rainfall_accumulated_16bit",
        33: "reflectivity_16bit",
        34: "rain_kinetic_energy",
        35: "snowfall_rate",
        60: "number_particles_all",
        61: "list_particles",
        90: "raw_drop_concentration",
        91: "raw_drop_average_velocity",
        92: "raw_drop_number",
    }

    df = df.rename(col, axis=1)

    # Cast time
    df["time"] = dd.to_datetime(
        df["sensor_date"] + "-" + df["sensor_time"], format="%d.%m.%Y-%H:%M:%S"
    )
    df = df.drop(columns=["sensor_date", "sensor_time"])

    # Drop useless columns
    df.replace("", np.nan, inplace=True)
    df.dropna(how="all", axis=1, inplace=True)
    col_to_drop = [40, 41, 50, 51, 93, 94, 95, 96, 97]
    df = df.drop(columns=col_to_drop)

    # Trim weather_code_metar_4678 and weather_code_nws
    df["weather_code_metar_4678"] = df["weather_code_metar_4678"].str.strip()
    df["weather_code_nws"] = df["weather_code_nws"].str.strip()

    # Delete invalid columsn by check_L0A
    col_to_drop = [
        "sensor_temperature_trasmitter",
        "sensor_temperature_pcb",
        "rainfall_rate_16_bit_1200",
        "sensor_temperature_receiver",
        "snowfall_rate",
        "rain_kinetic_energy",
        "rainfall_rate_16_bit_30",
    ]
    df = df.drop(columns=col_to_drop)

    return df


##------------------------------------------------------.
#### 9.2 Launch code as in the parser file
# - Try with increasing number of files
# - Try first with lazy=False, then lazy=True
lazy = True  # True
subset_file_list = file_list[0:5]
df = read_L0A_raw_file_list(
    file_list=subset_file_list,
    column_names=column_names,
    reader_kwargs=reader_kwargs,
    df_sanitizer_fun=df_sanitizer_fun,
    lazy=lazy,
    sensor_name=sensor_name,
    verbose=verbose,
)

##------------------------------------------------------.
#### 9.3 Check everything looks goods
df = df.compute()  # if lazy = True
print_df_column_names(df)
print_df_random_n_rows(df, n=5)
print_df_columns_unique_values(df, column_indices=2, column_names=True)
print_df_columns_unique_values(df, column_indices=slice(0, 20), column_names=True)

####--------------------------------------------------------------------------.
