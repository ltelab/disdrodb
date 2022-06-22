#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:31:55 2022

@author: ghiggi
"""

create_L0_summary_statistics
# - regularize timeseries
# - number of dry/rainy minutes
# - timebar plot with 0,>1, NA, no data rain rate (ARM STYLE)
# - timebar data quality

diameter_min
diameter_max
diameter_sum

# - Years coverage
# - Total minutes
# - Total DSD minutes 
# - Total rain events
# - Other stats TBD 


def regularize_dataset(ds: xr.Dataset, range_freq: str = "2min30s"):
    
    start = ds.time.to_numpy()[0]
    end = ds.time.to_numpy()[-1]

    full_range = pd.date_range(start=pd.to_datetime(start).date(), 
                               end=pd.to_datetime(pd.to_datetime(end).date()) + datetime.timedelta(hours=23, minutes=57, seconds=30), 
                               freq=range_freq).to_numpy()
    
    return ds.reindex({"time": full_range}, fill_value={"precip": 65535, "mask": 0})