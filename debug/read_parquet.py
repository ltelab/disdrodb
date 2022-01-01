#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:41:47 2021

@author: kimbo
"""

import os
os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
import pandas as pd
import dask.dataframe as dd
import os
import numpy as np
# from disdrodb.io import col_dtype_check


campagna = 'EPFL_Roof_2008'
path = f"/SharedVM/Campagne/ltnas3/Processed/{campagna}/01/L0"
file = campagna + '.parquet'
file_path = os.path.join(path, file)

df = dd.read_parquet(file_path)

df = df.compute()
print(df)

campagna = 'Ticino_2018'
path = f"/SharedVM/Campagne/ltnas3/Processed/{campagna}/"
file = campagna + '.parquet'
file_path = os.path.join(path, file)

df2 = dd.read_parquet(file_path)

df2 = df2.compute()



# df = df.iloc[0:100,:] # df.head(100) 
# np_arr_str =  df['RawData'].values.astype(str)
# list_arr_str = np.char.split(np_arr_str,",")
# print(len(list_arr_str[0]))

# df2 = df2.iloc[0:100,:] # df.head(100) 
# np_arr_str2 =  df2['RawData'].values.astype(str)
# list_arr_str2 =  np.char.split(np_arr_str2,",")
# print(len(list_arr_str2[0]))


# col_dtype_check(df, path, verbose=True)

