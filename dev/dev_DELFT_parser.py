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



df.iloc[0,:]["n_particles_all"]
df.iloc[0,:]['FieldN']
df.iloc[0,:]['FieldV']
df.iloc[0,:]['RawData']


attrs = {}
sensor_name = "OTT_Parsivel2"
attrs['sensor_name'] = "OTT_Parsivel2"
lazy=True
lazy=False
verbose=False














todrop = ['firmware_IOP',
          'firmware_DSP',
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

lazy=True
df = read_raw_data(
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