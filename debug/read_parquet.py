#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:41:47 2021

@author: kimbo
"""

import pandas as pd
import dask.dataframe as dd
import os

path = r'/SharedVM/Campagne/ltnas3/Processed/Ticino_2018/'
campagna = 'Ticino_2018'
file = campagna + '.parquet'
file_path = os.path.join(path, file)

df = dd.read_parquet(file_path)

df2 = df.compute()