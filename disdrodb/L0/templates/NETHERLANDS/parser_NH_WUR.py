#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:52:03 2022

@author: ghiggi
"""
import xarray as xr

# fpath = "/ltenas3/0_Data/DISDRODB/TODO_Raw/NETHERLANDS/WUREX14/source/unknown/temp_files/10/Disdrometer_20140930.nc"
# ds = xr.open_dataset(fpath)

fpath = (
    "/ltenas3/0_Data/DISDRODB/TODO_Raw/NETHERLANDS/WUREX14/data/WURex14_disdrometers.h5"
)

import h5py

f = h5py.File(fpath, "r")

list(f.items())

list(f["biotechnion/auxiliary"].items())
list(f["biotechnion/channel_1"].items())

stations = list(f.keys())
for station_name in stations:
    print(station_name)
    attrs = xr.open_dataset(fpath, group=station_name).attrs
    print(attrs)
    ds_aux = xr.open_dataset(fpath, group=station_name + "/auxiliary")
    ds_main = xr.open_dataset(fpath, group=station_name + "/channel_1")
