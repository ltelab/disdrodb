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



file_path = '/home/kimbo/data/Campagne/DISDRODB/Processed/DELFT/L0A/PAR001/DELFT_sPAR001.parquet'

df1 = pd.read_parquet(file_path)

df2 = df1.iloc[0:100]

df = df.compute()
print(df)

