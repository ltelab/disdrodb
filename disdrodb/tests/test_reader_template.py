#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 22:57:09 2022

@author: ghiggi
"""
from disdrodb.L0.L0A_processing import read_raw_data
from disdrodb.L0.L0B_processing import retrieve_L1_raw_arrays, create_L0B_from_L0A

lazy = False  # should we test also True !

##----------------------------------------------------------------------------.
###########################
#### CUSTOMIZABLE CODE ####
###########################
### Copy here the column_names definition

##----------------------------------------------------------------------------.
### Copy here the reader_kwargs definition

##----------------------------------------------------------------------------.
### Copy here the df_sanitzer

##----------------------------------------------------------------------------.
# Define filepath and sensor_name of a single sample data file
filepath = "/home/ghiggi/Desktop/Projects/disdrodb_scripts/data/spectropluvio_0a_Lz1LpnF60secPrainratePsize_v01_20220804_000000_1440.txt"
sensor_name = "OTT_Parsivel"
attrs = {}
attrs["sensor_name"] = sensor_name
attrs["latitude"] = "-9999"
attrs["longitude"] = "-9999"
attrs["altitude"] = "-9999"
attrs["crs"] = "dummy"

# Run this to check it works
df = read_raw_data(filepath, column_names, reader_kwargs, lazy=lazy)
df = df_sanitizer_fun(df, lazy=lazy)
print(df)

dict_data = retrieve_L1_raw_arrays(df, sensor_name, lazy=lazy, verbose=False)

# Note: here the dtype of the 1D variable is object. Expected.
ds = create_L0B_from_L0A(df, attrs, lazy=True, verbose=False)
