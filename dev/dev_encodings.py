#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:46:47 2022

@author: ghiggi
"""
import yaml
from disdrodb.standards import get_sensor_variables

sensor_name = "OTT_Parsivel"

# Defaults encodings
encoding_kwargs = {}
encoding_kwargs["dtype"] = "float32"
encoding_kwargs["zlib"] = True
encoding_kwargs["complevel"] = 3
encoding_kwargs["shuffle"] = True
encoding_kwargs["fletcher32"] = False
encoding_kwargs["contiguous"] = False
encoding_kwargs["chunksizes"] = 5000

# _FillValue
# scale_factor
# add_offset

# Define custom encodings
from disdrodb.data_encodings import get_L0_dtype_standards

dtype_L0 = get_L0_dtype_standards()
variables = get_sensor_variables(sensor_name)
encodings_dict = {}
for variable in variables:
    encodings_dict[variable] = encoding_kwargs.copy()
    encodings_dict[variable]["dtype"] = dtype_L0[variable]

encodings_dict["FieldN"]["chunksizes"] = [5000, 32]
encodings_dict["FieldV"]["chunksizes"] = [5000, 32]
encodings_dict["RawData"]["chunksizes"] = [5000, 32, 32]
encodings_dict["RawData"]["dtype"] = "int64"
encodings_dict["FieldN"]["dtype"] = "float32"
encodings_dict["FieldV"]["dtype"] = "float32"
encodings_dict["weather_code_METAR_4678"]["dtype"] = "str"
encodings_dict["weather_code_NWS"]["dtype"] = "str"

with open("/home/ghiggi/L1_netcdf_encodings.yml", "w") as f:
    yaml.dump(encodings_dict, f, sort_keys=False)

# Open dictionary
from disdrodb.standards import get_L1_netcdf_encoding_dict

encoding_dict = get_L1_netcdf_encoding_dict(sensor_name)


#### Zarr
# def _get_default_zarr_encoding(dtype="float32"):
#     compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
#     encoding_kwargs = {}
#     encoding_kwargs["dtype"] = dtype
#     encoding_kwargs["compressor"] = compressor
#     return encoding_kwargs


# def get_L1_zarr_encodings_standards(sensor_name):
#     # Define variable names
#     vars = ["FieldN", "FieldV", "RawData"]
#     dtype_dict = get_L1_dtype()
#     # Define encodings dictionary
#     encoding_dict = {}
#     for var in vars:
#         encoding_dict[var] = _get_default_zarr_encoding(dtype=dtype_dict[var])  # TODO

#     return encoding_dict
from disdrodb.standards import get_L0_dtype

get_L0_dtype(sensor_name)
