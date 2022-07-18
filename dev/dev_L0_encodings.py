#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:46:47 2022

@author: ghiggi
"""
################################################################## 
### Script helping in initializing the L0_encodings.yml file  ####
################################################################## 
import yaml
from disdrodb.standards import get_sensor_variables

sensor_name = "OTT_Parsivel2"

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
from disdrodb.L0.standards import get_L0_dtype 

dtype_L0 = get_L0_dtype(sensor_name=sensor_name)
variables = get_sensor_variables(sensor_name)
encodings_dict = {}
for variable in variables:
    encodings_dict[variable] = encoding_kwargs.copy()
    encodings_dict[variable]["dtype"] = dtype_L0[variable]

encodings_dict["raw_drop_concentration"]["chunksizes"] = [5000, 32]
encodings_dict["raw_drop_average_velocity"]["chunksizes"] = [5000, 32]
encodings_dict["raw_drop_number"]["chunksizes"] = [5000, 32, 32]
encodings_dict["raw_drop_number"]["dtype"] = "int64"
encodings_dict["raw_drop_concentration"]["dtype"] = "float32"
encodings_dict["raw_drop_average_velocity"]["dtype"] = "float32"
encodings_dict["weather_code_metar_4678"]["dtype"] = "str"
encodings_dict["weather_code_nws"]["dtype"] = "str"

dst_L0_encodings_fpath = "/home/ghiggi/L0_encodings.yml" 
with open(dst_L0_encodings_fpath, "w") as f:
    yaml.dump(encodings_dict, f, sort_keys=False)

# Open dictionary
from disdrodb.standards import get_L0_encoding_dict

encoding_dict = get_L0_encoding_dict(sensor_name)

 
 