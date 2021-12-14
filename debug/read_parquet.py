#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:41:47 2021

@author: kimbo
"""

import pandas as pd
import dask.dataframe as dd
import os

path = r'/SharedVM/Campagne/ltnas3/Processed/Payerne_2014/10/l0'
campagna = 'Payerne_2014'
file = campagna + '.parquet'
file_path = os.path.join(path, file)

df = dd.read_parquet(file_path)

df = df.compute()


path = r'/SharedVM/Campagne/ltnas3/Processed/Payerne_2014/20/l0'
campagna = 'Payerne_2014'
file = campagna + '.parquet'
file_path = os.path.join(path, file)

df2 = dd.read_parquet(file_path)

df2 = df2.compute()

path = r'/SharedVM/Campagne/ltnas3/Processed/Ticino_2018/'
campagna = 'Ticino_2018'
file = campagna + '.parquet'
file_path = os.path.join(path, file)

df3 = dd.read_parquet(file_path)

df3 = df3.compute()

# a = {}

# for col in list(df.iloc[:,:20]):
#     a[col] = df[col].unique()
    
# for key, value in a.items():
#     print(key, ' : ', value)