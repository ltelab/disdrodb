#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:33:39 2022

@author: ghiggi
"""
import numpy as np
import pandas as pd
import xarray as xr
from xarray.core import dtypes


def regularize_dataset(
    ds: xr.Dataset, range_freq: str, tolerance=None, method=None, fill_value=dtypes.NA
):
    """Regularize a dataset across time dimension with uniform resolution."""
    start = ds.time.values[0]
    end = ds.time.values[-1]
    # start_date = pd.to_datetime(start).date() # to start at 00
    # end_date = pd.to_datetime(end).date() + datetime.timedelta(hours=23, minutes=57, seconds=30)
    new_time_index = pd.date_range(
        start=pd.to_datetime(start), end=pd.to_datetime(end), freq=range_freq
    )

    # Regularize dataset and fill with NA values
    ds_reindexed = ds.reindex(
        {"time": new_time_index},
        method=method,  # do not fill gaps
        tolerance=tolerance,  # mismatch in seconds
        fill_value=fill_value,
    )
    return ds_reindexed
