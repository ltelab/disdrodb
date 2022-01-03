#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:30:50 2022

@author: ghiggi
"""
import xarray as xr 
ds = xr.open_dataset(fpath)
ds.coords
ds.dims
ds.data_vars
 
da = ds['FieldV'].compute()
da.dtype
da.time.dtype
da.velocity_bin_center

da.as_dtype(float)
da.plot.imshow(x="time", y="velocity_bin_center")

arr = da.values
import matplotlib.pyplot as plt 
plt.imshow(arr, aspect="auto")

 


df = read_L0_data(processed_dir, station_id, lazy=False, verbose=verbose, debugging_mode=debugging_mode)
                

datetime_series = pd.to_datetime(df['time'], format = '%dd-%Mm-%YYYY HH:MM:SS')  


dd.to_datetime('01-08-2018  00:00:00', format='%m-%d-%Y %H:%M:%S')
pd.to_datetime('01-08-2018  00:00:00', format='%m-%d-%Y %H:%M:%S')
 
tt = df['time']


t1 = pd.to_datetime(tt, format='%m-%d-%Y %H:%M:%S')
t2 = pd.to_datetime(tt, format='%d-%m-%Y %H:%M:%S')
np.unique(t1.dt.month)
np.unique(t1.dt.day)



