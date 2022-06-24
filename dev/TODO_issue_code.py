#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:18:36 2022

@author: ghiggi
"""
import os
import yaml
import pandas as pd
import pandas.io.parsers
import logging
 
import yaml 
from disdrodb.L0.issue import create_issue_yml
from disdrodb.logger import log_debug

logger = logging.getLogger(__name__)

#--------------------------------------------------
# Open desired timestamps.yml
src_fpath = "/home/ghiggi/Projects/disdrodb/dev/src_timestamps.yml" 
with open(src_fpath, "r") as f:
    issue_dict = yaml.safe_load(f)

verbose = True

####--------------------------------------------------------------------------- 
#### CODE in l0/issue.py  
import pandas as pd

timestamps = issue_dict.get("timestamp", False)  
timestamps = pd.to_datetime(timestamps, format='%Y-%m %H:%M')
 
print(timestamps)

# DEV 
def check_timestamps_resolution(timestamps, resolution="s", verbose=False):
    # Ensure seconds resolution !!!
    # TODO 
    for timestamp in timestamps: 
        
# DateTimeIndex
timestamps.resolution 
timestamps.astype('datetime64[s]')
timestamps.astype('M8[ns]')
timestamps.dtype
timestamps.dt.ceil(freq="s")
timestamps.microseconds = 0


length_desired_resolution = len('2018-12-07 14:15:00')
print(length_desired_resolution)

# ALREADY WORKS 
def check_timestamps_format(timestamps, verbose=False):
    """Check correct timestamps format."""
    datetime_format = '%Y-%m-%d %H:%M:%S'
    # Try parse directly all timestamps
    try:
        timestamps = pd.to_datetime(timestamps, format=datetime_format)
    except:
        print("aaaa")
        # If throwing error, loop over all timesteps and record the errors
        list_errors = []
        for timestamp in timestamps:
            try:
                 _ = pd.to_datetime(timestamp, format=datetime_format)
            except ValueError as err:
                list_errors.append(err)
                pass
        # - Log the error 
        for i,_ in enumerate(list_errors):
            msg = list_errors[i].args[0]
            log_debug(logger, msg, verbose)
        # - Raise error 
        msg = "Please correct the format of timestamps in the issue YAML file."
        logger.error(msg)
        raise ValueError(msg) 
    # Ensure seconds resolution !!!
    # TODO 
    return timestamps     

check_timestamps(timestamps, verbose=True)          
           
def check_issue_compliance(issue_dict, verbose=False):
    # Check valid input 
    if isinstance(issue_dict, dict): 
        raise TypeError("issue_dict must be a dictionary.")
        
    # Check dictionary keys are valid  
    valid_keys = ["timestamp", "time_period"]
    dict_keys =  np.array(list(issue_dict.keys()))
    unvalid_keys = dict_keys[np.isin(dict_keys, valid_keys, invert=True)]
    if len(idx_unvalid_keys) > 0:
        msg = f"The keys {unvalid_keys} in the issue dictionary are unvalid. Valid keys are {valid_keys}."
        log_error(logger, msg, verbose)
        raise ValueError(msg)
        
    # -----------------------------------.   
    # Check valid timestamp 
    timestamps = issue_dict.get("timestamp", False)
    if timestamps:
        # TODO: check_timestamps_resolution(timestamps, resolution="s")
        timestamps = check_timestamps_format(timestamps, verbose=verbose)
        issue_dict['timestamp'] = timestamps
    
    # [OPTIONAL] Check timestamps are sorted ... 
    
    # -----------------------------------.      
    # Check valid time_periods 
    time_periods = issue_dict.get("time_period", False)
    if time_periods:
    # - TODO: Check list of list 
    # - TODO: Check valid timestamp 
    # - TODO: Check valid datetime resolution
        pass
    
    return issue_dict
    
####--------------------------------------------------------------------------- 
    try:
        timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S')
    except ValueError:

        timestamp = pd.DatetimeIndex(time_temp)
    except Exception:
        # Something wrong to get timestamp info
        msg = "Error on drop bad timestamp: something wrong to get timestamp information"
        logger.error(msg)
        raise NotImplementedError(" - " + msg)   
     if len(period) == 2:
         try:
             range_dates = pd.to_datetime(period, format='%Y-%m-%d %H:%M:%S')
         except ValueError:
             msg = "Warning in time_period, the date {} not in the format '%d-%m-%Y %H:%M:%S', but try to parse date.".format(period)
             log_debug(logger, msg, verbose)
             pass
         except Exception:
             msg = "Something wrong with {} into time_period, so skipped".format(period)
             logger.error(msg)
             if verbose:
                 print(' - ' + msg)
             pass
         if range_dates[0] > range_dates[1]:
             msg = "The initial date is bigger than the final one in: {}".format(period)
             logger.warning(msg)
             if verbose:
                 print(' - ' + msg)
             # Swap date position
             range_dates[0], range_dates[1] = range_dates[1], range_dates[0]
             
#### -------------------------------------------------------------------------.
#### CODE into  L0 proc 
# import numpy as np 
# list_idx_to_keep = []
# idx_to_keep = np.array([True, True, False]) 
# list_idx_to_keep.append(idx_to_keep)

def is_values_outside_periods(series, list_periods):
    """
    Get bool array indicating when values are outside the list of periods.
    
    Note: It expects values to be a pd.Series
    """
    # Get valid indices looping over each period 
    list_idx_to_keep = []
    for period in time_periods:
        idx_to_keep = ~series.between(period[0], period[1], inclusive=True)
        list_idx_to_keep.append(list_idx_to_keep)
    # Select rows where only True values occurs 
    arr_idx_to_keep = np.vstack(list_idx_to_keep)
    idx_to_keep = np.all(arr_idx_to_keep, axis=0)  
    return idx_to_keep

def remove_problematic_timestamp(df, issue_dict, verbose):
        '''Drop dataframe rows listed in the issue dictionary.'''
        # ---------------------------------------------
        # Check df has time column in correct format 
        # TODO: (and resolution)
               
        # ---------------------------------------------
        # Initialize counter 
        n_rows_dropped = 0 
        # ---------------------------------------------
        # Remove problematic timestamps  
        n_rows = len(df)
        timestamps = issue_dict.get('timestamp', False)
        if timestamps:
            df_timestamps = df['time'] #.compute()
            df = df[~df_timestamps.isin(timestamps)] # TODO: this works correctly in lazy mode ? 
            n_rows_dropped = n_rows - len(df)
            # Log info messages 
            msg = "{n_rows_dropped} single timestamps were dropped.".format(n_rows_dropped)
            log_info(logger, msg, verbose)
            # Log timesteps not in dataframe time coverage 
            bad_issue_timestamps = timestamps[np.isin(timestamps, df_timestamps, invert=True)]
            if len(bad_issue_timestamps) > 0: 
                msg = f"The following timestamps in the issue yml file are not present in the raw data: {timestamps}"
                log_warning(logger, msg, verbose)
        
        # ---------------------------------------------
        # Check there are rows left 
        n_rows = len(df)
        if n_rows == 0: 
            msg = "No rows left after removing problematic timestamps. Please check the issue YAML file."
            logger.error(msg)
            raise ValueError(msg)
        
        # ---------------------------------------------
        # Remove problematic time periods          
        time_periods = issue_dict.get('time_period', False)
        if time_periods: 
            df_timestamps = df['time']#.compute()
            n_rows = len(df_timestamps)
            idx_to_keep = is_values_outside_periods(df_timestamps, list_periods)
            df = df[idx_to_keep]
            n_dropped_timestamps = np.sum(idx_to_keep)
            n_rows_dropped =+ n_dropped_timestamps
            msg = f"{n_dropped_timestamps} timestamps were dropped when addresssing problematic time peridos."
            log_info(logger, msg, verbose)
          
       # ---------------------------------------------  
       # Check there are rows left 
       n_rows = len(df)
       if n_rows == 0: 
           msg = "No rows left after removing problematic time periods. Please check the issue YAML file."
           logger.error(msg)
           raise ValueError(msg)  
    
       # --------------------------------------------- 
       # Report info message
       msg = f"In total, {n_rows_dropped} rows were dropped."
       log_info(logger, msg, verbose)
       # --------------------------------------------- 
      
       return df

 