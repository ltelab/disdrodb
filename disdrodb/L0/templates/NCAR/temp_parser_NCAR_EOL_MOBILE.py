#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 09:48:57 2022

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
raw_dir = "/home/kimbo/data/Campagne/DISDRODB/Raw/NCAR/PECAN/NCAR_EOL_MOBILE"
processed_dir = (
    "/home/kimbo/data/Campagne/DISDRODB/Processed/NCAR/PECAN/NCAR_EOL_MOBILE"
)
force = False
force = True
lazy = True
lazy = False
verbose = True
debugging_mode = True
sensor_name = "Parsivel"

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
glob_pattern = os.path.join("data", station_id, "*.dat*")  # CUSTOMIZE THIS
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
reader_kwargs["delimiter"] = "no_need_it"

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
# reader_kwargs['compression'] = 'xz'

# - Strings to recognize as NA/NaN and replace with standard NA flags
#   - Already included: ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’,
#                       ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’,
#                       ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
reader_kwargs["na_values"] = [
    "na",
    "",
    "error",
    "NA",
    "-.-",
    " NA",
]

# - Define max size of dask dataframe chunks (if lazy=True)
#   - If None: use a single block for each file
#   - Otherwise: "<max_file_size>MB" by which to cut up larger files
reader_kwargs["blocksize"] = None  # "50MB"

# Cast all to string
reader_kwargs["dtype"] = str

# Skip first row as columns names
reader_kwargs["header"] = None

# # Skip header
reader_kwargs["skiprows"] = 7

# Define encoding
reader_kwargs["encoding"] = "ISO-8859-1"

####--------------------------------------------------------------------------.
####################################################
#### 6. Open a single file and explore the data ####
####################################################
# - Do not assign column names yet to the columns
# - Do not assign a dtype yet to the columns
# - Possibily look at multiple files ;)
filepath = file_list[0]
str_reader_kwargs = reader_kwargs.copy()
str_reader_kwargs["dtype"] = str  # or object
df_str = read_raw_data(
    filepath, column_names=None, reader_kwargs=str_reader_kwargs, lazy=False
)

# Print first rows
print_df_first_n_rows(df_str, n=1, column_names=False)
print_df_first_n_rows(df_str, n=5, column_names=False)
print_df_random_n_rows(df_str, n=5, column_names=False)  # this likely the more useful

# Retrieve number of columns
print(len(df_str.columns))

# Look at unique values
# print_df_columns_unique_values(df_str, column_indices=None, column_names=False) # all

# print_df_columns_unique_values(df_str, column_indices=0, column_names=False) # single column

# print_df_columns_unique_values(df_str, column_indices=slice(0,15), column_names=False) # a slice of columns

# get_df_columns_unique_values_dict(df_str, column_indices=slice(0,15), column_names=False) # get dictionary

# Retrieve number of columns
print(len(df_str.columns))

# Infer columns based on string patterns
infer_df_str_column_names(df_str, sensor_name=sensor_name)

# Alternatively an empty list of column_names to infer
["Unknown" + str(i + 1) for i in range(len(df_str.columns))]

# Print valid column names
# - If other names are required, add the key to disdrodb/L0/configs/<sensor_name>/L0A_dtype.yml
# print_valid_L0_column_names(sensor_name)

# Instrument manufacturer defaults
# from disdrodb.standards import get_OTT_Parsivel_dict, get_OTT_Parsivel2_dict
# get_OTT_Parsivel_dict()
# get_OTT_Parsivel2_dict()

####---------------------------------------------------------------------------.
######################################################################
#### 7. Define dataframe columns [TO CUSTOMIZE AND MOVE TO PARSER] ###
######################################################################
# - If a column must be splitted in two (i.e. lat_lon), use a name like: TO_SPLIT_lat_lon
column_names = ["TO_SPLIT"]

# - Check name validity
# check_column_names(column_names, sensor_name)

# - Read data
filepath = file_list[0]
df = read_raw_data(
    filepath=filepath,
    column_names=column_names,
    reader_kwargs=reader_kwargs,
    lazy=False,
)

# - Look at the columns and data
print_df_column_names(df)
print_df_random_n_rows(df, n=5)

# - Check it loads also lazily in dask correctly
df1 = read_raw_data(
    filepath=filepath, column_names=column_names, reader_kwargs=reader_kwargs, lazy=True
)
df1 = df1.compute()

# - Look at the columns and data
print_df_column_names(df1)
print_df_random_n_rows(df1, n=5)

# - Check are equals
assert df.equals(df1)

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
lazy = False  # Try also with True when work with False

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


# Different column and divider into raw order between stations

header_1 = "Date,Time,Intensity (mm/h),Precipitation since start (mm),Weather code SYNOP WaWa,Radar reflectivity (dBz),MOR Visibility (m),Number of detected particles,Temperature in sensor (°C),Weather code METAR/SPECI,Weather code NWS,Signal amplitude of Laserband,Heating current (A),Sensor voltage (V),Spectrum"
header_2 = "Date,Time,Intensity (mm/h),Precipitation since start (mm),Weather code SYNOP WaWa,Weather code METAR/SPECI,Weather code NWS,Radar reflectivity (dBz),MOR Visibility (m),Signal amplitude of Laserband,Number of detected particles,Temperature in sensor (°C),Heating current (A),Sensor voltage (V),Spectrum"

if df.head().loc[df["TO_SPLIT"] == header_1].empty:
    df = df[~df.eq(header_2).any(1)]
    columns = [
        "date",
        "time_temp",
        "rainfall_rate_32bit",
        "precipitation_since_start_TO_DROP",
        "weather_code_synop_4680",
        "weather_code_metar_4678",
        "weather_code_nws",
        "reflectivity_32bit",
        "mor_visibility",
        "laser_amplitude",
        "number_particles",
        "sensor_temperature",
        "sensor_heating_current",
        "sensor_battery_voltage",
        "raw_drop_number",
    ]

else:
    df = df[~df.eq(header_1).any(1)]
    columns = [
        "date",
        "time_temp",
        "rainfall_rate_32bit",
        "precipitation_since_start_TO_DROP",
        "weather_code_synop_4680",
        "reflectivity_32bit",
        "mor_visibility",
        "laser_amplitude",
        "sensor_temperature",
        "weather_code_metar_4678",
        "weather_code_nws",
        "number_particles",
        "sensor_heating_current",
        "sensor_battery_voltage",
        "raw_drop_number",
    ]

# Split into columns and assign name
df = df["TO_SPLIT"].str.split(",", expand=True, n=14)
df.columns = columns

# Replace divider into raw_drop_number
df["raw_drop_number"] = df["raw_drop_number"].str.replace(";", ",")


# Parse time and drop precipitation_since_start_TO_DROP column
df["time"] = df["date"] + "-" + df["time_temp"]
df["time"] = dd.to_datetime(df["time"], format="%d.%m.%Y-%H:%M:%S")
df = df.drop(columns=["date", "time_temp", "precipitation_since_start_TO_DROP"])

# Set NaN into <SPECTRUM>ZERO</SPECTRUM> in raw_drop_number
import numpy as np

df["raw_drop_number"] = df["raw_drop_number"].replace(
    "<SPECTRUM>ZERO</SPECTRUM>", np.NaN
)

# Cannot implement dask for this loop, so only pandas for now
for i, v in df.iterrows():
    # Check if v is NaN
    if not v["raw_drop_number"] != v["raw_drop_number"]:

        v["raw_drop_number"] = v["raw_drop_number"].replace("<SPECTRUM>", "")
        v["raw_drop_number"] = v["raw_drop_number"].replace("</SPECTRUM>", "")

        # Add 0 digits to raw_drop_number
        temp_raw = v["raw_drop_number"].split(",")
        raw = ""
        for n in temp_raw:
            if n == "":
                raw += "000,"
            else:
                raw += "%03d" % int(n) + ","
        df.loc[i, "raw_drop_number"] = raw[:-4]

# Assign 0 value into raw_drop_number NaN values
df["raw_drop_number"] = df["raw_drop_number"].fillna(
    "000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,"
)


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
    # Cannot implement dask for this loop, so only pandas for now
    import numpy as np

    if lazy:
        import pandas as dd

        df = df.compute()
    else:
        import pandas as dd

    # Different column and divider into raw order between stations
    header_1 = "Date,Time,Intensity (mm/h),Precipitation since start (mm),Weather code SYNOP WaWa,Radar reflectivity (dBz),MOR Visibility (m),Number of detected particles,Temperature in sensor (°C),Weather code METAR/SPECI,Weather code NWS,Signal amplitude of Laserband,Heating current (A),Sensor voltage (V),Spectrum"
    header_2 = "Date,Time,Intensity (mm/h),Precipitation since start (mm),Weather code SYNOP WaWa,Weather code METAR/SPECI,Weather code NWS,Radar reflectivity (dBz),MOR Visibility (m),Signal amplitude of Laserband,Number of detected particles,Temperature in sensor (°C),Heating current (A),Sensor voltage (V),Spectrum"

    if df.head().loc[df["TO_SPLIT"] == header_1].empty:
        df = df[~df.eq(header_2).any(1)]
        columns = [
            "date",
            "time_temp",
            "rainfall_rate_32bit",
            "precipitation_since_start_TO_DROP",
            "weather_code_synop_4680",
            "reflectivity_32bit",
            "mor_visibility",
            "number_particles",
            "sensor_temperature",
            "weather_code_metar_4678",
            "weather_code_nws",
            "laser_amplitude",
            "sensor_heating_current",
            "sensor_battery_voltage",
            "raw_drop_number",
        ]

    else:
        df = df[~df.eq(header_1).any(1)]
        columns = [
            "date",
            "time_temp",
            "rainfall_rate_32bit",
            "precipitation_since_start_TO_DROP",
            "weather_code_synop_4680",
            "reflectivity_32bit",
            "mor_visibility",
            "laser_amplitude",
            "sensor_temperature",
            "weather_code_metar_4678",
            "weather_code_nws",
            "number_particles",
            "sensor_heating_current",
            "sensor_battery_voltage",
            "raw_drop_number",
        ]

    # Split into columns and assign name
    df = df["TO_SPLIT"].str.split(",", expand=True, n=14)
    df.columns = columns

    # Replace divider into raw_drop_number
    df["raw_drop_number"] = df["raw_drop_number"].str.replace(";", ",")

    # Parse time and drop precipitation_since_start_TO_DROP column
    df["time"] = df["date"] + "-" + df["time_temp"]
    df["time"] = dd.to_datetime(df["time"], format="%d.%m.%Y-%H:%M:%S")
    df = df.drop(columns=["date", "time_temp", "precipitation_since_start_TO_DROP"])

    # Set NaN into <SPECTRUM>ZERO</SPECTRUM> in raw_drop_number
    import numpy as np

    df["raw_drop_number"] = df["raw_drop_number"].replace(
        "<SPECTRUM>ZERO</SPECTRUM>", np.NaN
    )

    # Cannot implement dask for this loop, so only pandas for now
    for i, v in df.iterrows():
        # Check if v is NaN
        if not v["raw_drop_number"] != v["raw_drop_number"]:

            v["raw_drop_number"] = v["raw_drop_number"].replace("<SPECTRUM>", "")
            v["raw_drop_number"] = v["raw_drop_number"].replace("</SPECTRUM>", "")

            # Add 0 digits to raw_drop_number
            temp_raw = v["raw_drop_number"].split(",")
            raw = ""
            for n in temp_raw:
                if n == "":
                    raw += "000,"
                else:
                    raw += "%03d" % int(n) + ","
            df.loc[i, "raw_drop_number"] = raw[:-4]

    # Assign 0 value into raw_drop_number NaN values
    df["raw_drop_number"] = df["raw_drop_number"].fillna(
        "000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,"
    )

    return df


##------------------------------------------------------.
#### 9.2 Launch code as in the parser file
# - Try with increasing number of files
# - Try first with lazy=False, then lazy=True
lazy = True  # True
subset_file_list = file_list[0:10]
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
