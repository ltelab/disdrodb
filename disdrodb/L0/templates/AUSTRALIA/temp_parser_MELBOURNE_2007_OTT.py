#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 05:49:01 2022

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
import glob
import dask.dataframe as dd

# Directory
from disdrodb.L0.io import (
    check_directories,
    get_campaign_name,
    create_directory_structure,
)


# L0A processing
from disdrodb.L0.L0A_processing import read_L0A_raw_file_list

# Metadata
from disdrodb.L0.metadata import read_metadata


# Tools to develop the parser
from disdrodb.L0.template_tools import (
    infer_df_str_column_names,
    print_df_random_n_rows,
    print_df_column_names,
    print_df_columns_unique_values,
    print_df_column_names,
)


# Wrong dependency, to be changed
from disdrodb.data_encodings import get_L0_dtype_standards




##------------------------------------------------------------------------.
######################################
#### 1. Define campaign filepaths ####
######################################
raw_dir = "/home/kimbo/data/Campagne/Raw/MELBURNE/MELBOURNE_2007_OTT"
processed_dir = "/home/kimbo/data/Campagne/Processed/MELBURNE/MELBOURNE_2007_OTT"

l0_processing = True
l1_processing = True
force = True
verbose = True
debugging_mode = True
lazy = True
write_netcdf = True

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

# # Retrieve logger
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
glob_pattern = os.path.join("data", station_id, "*.txt*")  # CUSTOMIZE THIS
device_path = os.path.join(raw_dir, glob_pattern)
file_list = sorted(glob.glob(device_path, recursive=True))
# -------------------------------------------------------------------------.
# All files into the campaing
all_stations_files = sorted(
    glob.glob(os.path.join(raw_dir, "data", "*/*.txt*"), recursive=True)
)
# file_list = ['/SharedVM/Campagne/EPFL/Raw/EPFL_ROOF_2011/data/10/10_ascii_20110905.dat']
# file_list = get_file_list(raw_dir=raw_dir,
#                           glob_pattern=glob_pattern,
#                           verbose=verbose,
#                           debugging_mode=debugging_mode)


# Retrieve metadata
attrs = read_metadata(raw_dir=raw_dir, station_id=station_id)

# Retrieve sensor name
sensor_name = attrs["sensor_name"]
# check_sensor_name(sensor_name)

####--------------------------------------------------------------------------.
#########################################################################
#### 5. Define reader options [TO CUSTOMIZE AND THEN MOVE TO PARSER] ####
#########################################################################
# Important: document argument need/behaviour

reader_kwargs = {}

# - Define delimiter
reader_kwargs["delimiter"] = "!"

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
reader_kwargs["na_values"] = ["na", "", "error", "NA", "NP   "]

# - Define max size of dask dataframe chunks (if lazy=True)
#   - If None: use a single block for each file
#   - Otherwise: "<max_file_size>MB" by which to cut up larger files
reader_kwargs["blocksize"] = None  # "50MB"

# Cast all to string
reader_kwargs["dtype"] = str

# Skip first row as columns names
reader_kwargs["header"] = None


####--------------------------------------------------------------------------.
####################################################
#### 6. Open a single file and explore the data ####
####################################################
# - Do not assign column names yet to the columns
# - Do not assign a dtype yet to the columns
# - Possibily look at multiple files ;)
# filepath = file_list[0]
# filepath = file_list[0]
# str_reader_kwargs = reader_kwargs.copy()
# df = pd.read_csv(filepath, sep=',', skiprows=3, header=None)
# # Drop rows too short (date logger rows)
# df = df.loc[df[0].astype(str).str.len() > 620]


# column_names_temp = ['temp']
# df = read_raw_data(filepath,
#                    column_names=column_names_temp,
#                    reader_kwargs=str_reader_kwargs,
#                    lazy=False)

# # Data format:
# # -2015-01-09 00:02:16
# # 0000.063;0012.33;51;51;  -DZ; ...

# # Save time into df_time
# df_time = df.loc[df['temp'].astype(str).str.len() == 20]
# df_time['temp'] = pd.to_datetime(df_time['temp'], format='-%Y-%m-%d %H:%M:%S')


# # Drop header's log and corrupted rows
# df = df.loc[df['temp'].astype(str).str.len() > 620]

# # Split first 80 columns
# df = df['temp'].str.split(';', n=16, expand=True)

# Split raws columns
# df['raw_drop_concentration'] = df[16].str[:224]
# df['raw_drop_average_velocity'] = df[16].str[224:448]
# df['raw_drop_number'] = df[16].str[448:]

####---------------------------------------------------------------------------.
######################################################################
#### 7. Define dataframe columns [TO CUSTOMIZE AND MOVE TO PARSER] ###
######################################################################
# - If a column must be splitted in two (i.e. lat_lon), use a name like: TO_BE_SPLITTED_lat_lon

column_names_temp = ["temp"]

column_names = [
    "rainfall_rate_32bit",
    "rainfall_accumulated_32bit",
    "weather_code_synop_4680",
    "weather_code_synop_4677",
    "weather_code_metar_4678",
    "reflectivity_32bit",
    "mor_visibility",
    "laser_amplitude",
    "number_particles",
    "unknow2",
    "datalogger_temperature",
    "sensor_status",
    "station_name",
    "unknow3",
    "unknow4",
    "error_code",
    "TO_BE_SPLITTED",
]

# df.columns = column_names

# # Clean raw_drop_number (ignore last 5 column)
# # Split raws columns
# df['raw_drop_concentration'] = df['TO_BE_SPLITTED'].str[:224]
# df['raw_drop_average_velocity'] = df['TO_BE_SPLITTED'].str[224:448]
# df['raw_drop_number'] = df['TO_BE_SPLITTED'].str[448:]

# # Concat df and df_time
# df = df.reset_index(drop=True)
# df_time = df_time.reset_index(drop=True)
# df = pd.concat([df_time, df], axis=1, sort=False)

# df = df.dropna()


####---------------------------------------------------------------------------.
#########################################################
#### 8. Implement ad-hoc processing of the dataframe ####
#########################################################
# - This must be done once that reader_kwargs and column_names are correctly defined
# - Try the following code with various file and with both lazy=True and lazy=False
filepath = file_list[0]  # Select also other files here  1,2, ...
# filepath = all_stations_files
lazy = False  # Try also with True when work with False

# ------------------------------------------------------.
#### 8.1 Run following code portion without modifying anthing
# - This portion of code represent what is done by read_L0A_raw_file_list in L0_proc.py
# df = read_raw_data(filepath=filepath,
#                     column_names=column_names_temp,
#                    reader_kwargs=reader_kwargs,
#                    lazy=lazy)

# ------------------------------------------------------.
# Check if file empty
# if len(df.index) == 0:
#     raise ValueError(f"{filepath} is empty and has been skipped.")

# # Check column number
# if len(df.columns) != len(column_names):
#     raise ValueError(f"{filepath} has wrong columns number, and has been skipped.")

# ---------------------------------------------------------------------------.
#### 8.2 Ad-hoc code [TO CUSTOMIZE]
# --> Here specify columns to drop, to split and other type of ad-hoc processing
# --> This portion of code will need to be enwrapped (in the parser file)
#     into a function called df_sanitizer_fun(df, lazy=True). See below ...


# ---------------------------------------------------------------------------.
#### 8.3 Run following code portion without modifying anthing
# - This portion of code represent what is done by read_L0A_raw_file_list in L0_proc.py

# ## Keep only clean data
# # - This type of filtering will be done in the background automatically ;)
# # Remove rows with bad data
# # df = df[df.sensor_status == 0]
# # Remove rows with error_code not 000
# # df = df[df.error_code == 0]

##----------------------------------------------------.
# Cast dataframe to dtypes
# - Determine dtype based on standards
# dtype_dict = get_L0_dtype_standards(station_name)
# for column in df.columns:
#     try:
#         df[column] = df[column].astype(dtype_dict[column])
#     except KeyError:
#         # If column dtype is not into get_L0_dtype_standards, assign object
#         df[column] = df[column].astype('object')


####------------------------------------------------------------------------------.
################################################
#### 9. Simulate parser file code execution ####
################################################
#### 9.1 Define sanitizer function [TO CUSTOMIZE]
# --> df_sanitizer_fun = None  if not necessary ...


def df_sanitizer_fun(df, lazy=lazy):
    # Import dask or pandas
    if lazy:
        import dask.dataframe as dd
    else:
        import pandas as dd

    # Data format:
    # -2015-01-09 00:02:16
    # 0000.063;0012.33;51;51;  -DZ; ...

    # Save time into df_time
    df_time = df.loc[df["temp"].astype(str).str.len() == 20]
    df_time["temp"] = dd.to_datetime(df_time["temp"], format="-%Y-%m-%d %H:%M:%S")
    df_time.columns = ["time"]

    # Drop header's log and corrupted rows
    df = df.loc[df["temp"].astype(str).str.len() > 620]

    # Split first 80 columns
    df = df["temp"].str.split(";", n=16, expand=True)

    column_names = [
        "rainfall_rate_32bit",
        "rainfall_accumulated_32bit",
        "weather_code_synop_4680",
        "weather_code_synop_4677",
        "weather_code_metar_4678",
        "reflectivity_32bit",
        "mor_visibility",
        "laser_amplitude",
        "number_particles",
        "unknow2",
        "datalogger_temperature",
        "sensor_status",
        "station_name",
        "unknow3",
        "unknow4",
        "error_code",
        "TO_BE_SPLITTED",
    ]

    df.columns = column_names

    # Split raws columns
    df["raw_drop_concentration"] = df["TO_BE_SPLITTED"].str[:224]
    df["raw_drop_average_velocity"] = df["TO_BE_SPLITTED"].str[224:448]
    df["raw_drop_number"] = df["TO_BE_SPLITTED"].str[448:]

    # Concat df and df_time
    df = df.reset_index(drop=True)
    df_time = df_time.reset_index(drop=True)
    df = dd.concat([df_time, df], axis=1)

    # Drop last columns (all nan)
    df = df.dropna(thresh=(len(df.columns) - 19), how="all")

    # Columns to drop
    columns_to_drop = [
        "TO_BE_SPLITTED",
        "weather_code_metar_4678",
        "datalogger_temperature",
        "sensor_status",
        "station_name",
        "error_code",
        "unknow2",
        "unknow3",
        "unknow4",
    ]

    df = df.drop(columns=columns_to_drop)

    return df


##------------------------------------------------------.
#### 9.2 Launch code as in the parser file
# - Try with increasing number of files
# - Try first with lazy=False, then lazy=True
lazy = False  # True
# subset_file_list = file_list[0]
subset_file_list = all_stations_files[:3]
df = read_L0A_raw_file_list(
    file_list=subset_file_list,
    column_names=column_names_temp,
    reader_kwargs=reader_kwargs,
    df_sanitizer_fun=df_sanitizer_fun,
    verbose=verbose,
    sensor_name=sensor_name,
    lazy=lazy,
)

##------------------------------------------------------.
#### 9.3 Check everything looks goods
# df = df.compute() # if lazy = True
print_df_column_names(df)
print_df_random_n_rows(df, n=5)
print_df_columns_unique_values(df, column_indices=2, column_names=True)
print_df_columns_unique_values(df, column_indices=slice(0, 20), column_names=True)


infer_df_str_column_names(df, "Parsivel")

####--------------------------------------------------------------------------.
##------------------------------------------------------.
#### 10. Conversion to parquet
parquet_dir = os.path.join(processed_dir, "L0", campaign_name + "_s10.parquet")

# Define writing options
compression = "snappy"  # 'gzip', 'brotli, 'lz4', 'zstd'
row_group_size = 100000
engine = "pyarrow"

df2 = df.to_parquet(
    parquet_dir,
    # schema = 'infer',
    engine=engine,
    row_group_size=row_group_size,
    compression=compression,
)
# df2 = df3.to_parquet(
#     parquet_dir,
#     # schema="infer",
#     engine=engine,
#     row_group_size=row_group_size,
#     compression=compression,
#     # write_metadata_file=False,
# )
##------------------------------------------------------.
#### 10.1 Read parquet file
df2 = dd.read_parquet(parquet_dir)
df2 = df2.compute()
print(df2)
