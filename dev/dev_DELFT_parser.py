#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 09:46:30 2022

@author: ghiggi
"""
import pandas as pd 

fpath = "/home/ghiggi/Downloads/part.0.parquet"

df = pd.read_parquet(fpath)

df.columns



df.iloc[0,:]["number_particles_all"]
df.iloc[0,:]['raw_drop_concentration']
df.iloc[0,:]['raw_drop_average_velocity']
df.iloc[0,:]['raw_drop_number']


attrs = {}
sensor_name = "OTT_Parsivel2"
attrs['sensor_name'] = "OTT_Parsivel2"
lazy=True
lazy=False
verbose=False














todrop = ['firmware_iop',
          'firmware_dps',
          'date_time_measurement_start',
          'sensor_time',
          'sensor_date',
          'station_name',
          'station_number',
          'list_particles',
          'epoch_time']

df = df.drop(todrop, axis=1)

from disdrodb.check_standards import check_L0_standards 
sensor_name = "OTT_Parsivel2"
verbose = True
check_L0_standards(fpath=fpath,
                   sensor_name=sensor_name,
                   verbose=verbose)


fpath = "/home/ghiggi/Downloads/data/10/20220101.csv"
filepath = fpath 
df = pd.read_csv(fpath)

df.columns
from disdrodb.L0_proc import read_raw_drop_number
lazy=True
df = read_raw_drop_number(
    filepath=filepath,
    column_names=columns_names_temporary,
    reader_kwargs=reader_kwargs,
    lazy=lazy,
)

if df_sanitizer_fun is not None:
    df = df_sanitizer_fun(df, lazy=lazy)
        
df.compute().iloc[0,:]
df.iloc[0,:]
column_names=columns_names_temporary

df1 = df.compute()
import numpy as np

df1['raw_drop_average_velocity'].astype(str).str.split(",")
df_series = df1["raw_drop_average_velocity"].astype(str).str.split(",")
arr_lengths = df_series.apply(len)
idx, count = np.unique(arr_lengths, return_counts=True)
n_max_vals = idx[np.argmax(count)]
