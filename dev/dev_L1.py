#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:51:22 2022

@author: ghiggi
"""
import os
import pandas as pd
import dask.dataframe as dd

import numpy as np
from disdrodb.L1_proc import retrieve_L1_raw_arrays
from disdrodb.standards import get_raw_field_nbins
from disdrodb.L1_proc import convert_L0_raw_fields_arr_flags
from disdrodb.L1_proc import set_raw_fields_arr_dtype
from disdrodb.L1_proc import check_array_lengths_consistency

fpath = "/home/ghiggi/Downloads/DISDRO_DATA/EPFL/COMMON_2011/L0/COMMON_2011_s40.parquet"

df = pd.read_parquet(fpath)
df.columns
df["raw_drop_number"].iloc[0]


### COMMON S40: Problem writing LO
## --> S41 Works?

# Parsivel --> OTT_Parsivel

# ---------------------------------------


### DAV0S_2009_2011: Problem in L1
# - S50 problematic
# - S60, 70?


fpath = "/home/ghiggi/Downloads/DISDRO_DATA/EPFL/DAVOS_2009_2011/L0/DAVOS_2009_2011_s50.parquet"
fpath = "/home/ghiggi/Downloads/DISDRO_DATA/EPFL/PLATO_2019/L0/PLATO_2019_s10.parquet"

df = pd.read_parquet(fpath)
df.columns
df["raw_drop_number"].iloc[0]

# EPFL_ROOF_2011  # Problem in L1
# S10 problematic
# S11?

# HPICONET_2020
# S12 problematic
# --> 13? 30, 31, 32, 33?

# HYMEX 2012
# S10, S11
# S13 problematic
# S30, 31, 32, 33?

# PLATO_2019
# - S10 problematic

# RIELTZOBACK 2011
# - S60 problematic
# - S61, 62, 63, 70

# SAMOYLNOV_2017_2019
# - S01 Problematic
# - S02? 20, 62, 63?

# https://docs.dask.org/en/stable/generated/dask.dataframe.from_pandas.html

import os
import pandas as pd
import dask.dataframe as dd

import numpy as np
import xarray as xr

from disdrodb.L1_proc import retrieve_L1_raw_arrays
from disdrodb.standards import get_raw_field_nbins
from disdrodb.L1_proc import convert_L0_raw_fields_arr_flags
from disdrodb.L1_proc import set_raw_fields_arr_dtype
from disdrodb.L1_proc import check_array_lengths_consistency
from disdrodb.L1_proc import get_L1_coords
from disdrodb.L1_proc import create_L1_dataset_from_L0
from disdrodb.L1_proc import write_L1_to_netcdf

# fpath = "/home/ghiggi/Downloads/DISDRO_DATA/EPFL/SAMOYLOV_2017_2019/L0/SAMOYLOV_2017_2019_s01.parquet"
fpath = "/home/ghiggi/Downloads/DISDRO_DATA/EPFL/DAVOS_2009_2011/L0/DAVOS_2009_2011_s50.parquet"
# fpath = "/home/ghiggi/Downloads/DISDRO_DATA/EPFL/PLATO_2019/L0/PLATO_2019_s10.parquet"
fpath = "/home/ghiggi/Downloads/DISDRO_DATA/EPFL/HYMEX_2012/L0/HYMEX_2012_s13.parquet"
fpath = (
    "/home/ghiggi/Downloads/DISDRO_DATA/EPFL/HPICONET_2010/L0/HPICONET_2010_s12.parquet"
)
fpath = "/home/ghiggi/Downloads/DISDRO_DATA/EPFL/EPFL_ROOF_2011/L0/EPFL_ROOF_2011_s10.parquet"
l1_processing = True
lazy = True
lazy = False

verbose = True
debugging_mode = False
sensor_name = "OTT_Parsivel"

attrs = {}
attrs["sensor_name"] = sensor_name
attrs["latitude"] = 2334
attrs["longitude"] = 232434
attrs["altitude"] = 342
attrs["crs"] = "prjjfs"

if lazy:
    df = dd.read_parquet(fpath)
    print(len(df))
else:
    df = pd.read_parquet(fpath)
    print(len(df))

# df = df.iloc[0:100000]
ds = create_L1_dataset_from_L0(df, attrs, lazy=lazy, verbose=verbose)
write_L1_to_netcdf(ds, fpath="/tmp/try8.nc", sensor_name=sensor_name)

# ------------------------------------------------------------------------------.
### DEBUG

df = check_array_lengths_consistency(df, sensor_name=sensor_name, lazy=lazy)
print(len(df))


dict_data = retrieve_L1_raw_arrays(df, sensor_name, lazy=lazy, verbose=verbose)


df_series = df[key].astype(str).str.split(",")

np.count(df_series.apply(len))

len(df_series.iloc[714])


df_series.iloc[12404][-20:-1]
arr = np.stack(df_series, axis=0)

df_series.iloc[0][-10:]

df["raw_drop_number"].iloc[0]
# Retrieve raw fields matrix bins dictionary
n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)
# Retrieve number of timesteps
if lazy:
    n_timesteps = df.shape[0].compute()
else:
    n_timesteps = df.shape[0]

# Retrieve available arrays
dict_data = {}
unavailable_keys = []

key = "raw_drop_concentration"
key = "raw_drop_average_velocity"
key = "raw_drop_number"

for key, n_bins in n_bins_dict.items():
    # Check key is available in dataframe
    if key not in df.columns:
        unavailable_keys.append(key)
        continue
    # Parse the string splitting at ,
    df_series = df[key].astype(str).str.split(",")
    # Create array
    if lazy:
        arr = da.stack(df_series, axis=0)
    else:
        arr = np.stack(df_series, axis=0)
    # Remove '' at the last array position
    arr = arr[:, 0 : n_bins_dict[key]]
    # Deal with flag values (-9.9999)
    arr = convert_L0_raw_fields_arr_flags(arr, key=key)
    # Set dtype of the matrix
    arr = set_raw_fields_arr_dtype(arr, key=key)
    # For key='raw_drop_number', reshape to 2D matrix
    if key == "raw_drop_number":
        arr = reshape_L0_raw_drop_number(arr, n_bins_dict, n_timesteps)
    # Add array to dictionary
    dict_data[key] = arr

# Retrieve unavailable keys from raw spectrum
if len(unavailable_keys) > 0:
    if "raw_drop_number" not in list(dict_data.keys()):
        raise ValueError(
            "The raw spectrum is required to compute unavaible N_D and N_V."
        )
    if "raw_drop_concentration" in unavailable_keys:
        dict_data["raw_drop_concentration"] = get_drop_concentration(dict_data["raw_drop_number"])
    if "raw_drop_average_velocity" in unavailable_keys:
        dict_data["raw_drop_average_velocity"] = get_drop_average_velocity(dict_data["raw_drop_number"])


# -----------------------------------------------------------------.
#### - Create xarray Dataset
ds = create_L1_dataset_from_L0(df=df, attrs=attrs, lazy=lazy, verbose=verbose)


df = pd.read_parquet(fpath)
df.columns
df["raw_drop_number"].iloc[0]
