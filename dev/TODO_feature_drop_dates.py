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

logger = logging.getLogger(__name__)

def read_bad_timestamp_yml(raw_dir):
    """Read drop_dates yaml file and return the dictionary."""
    fpath = os.path.join(raw_dir, 'issue','bad_timestamp.yml')
    # Check yaml file exists
    if not os.path.exists(fpath):
        msg = "{} not exist".format(fpath)
        create_bad_timestamp_yml(fpath)
        d = None
        logger.exception(msg)
        raise ValueError(msg)
    # Open dictionary
    with open(fpath, "r") as f:
        d = yaml.safe_load(f)
    return d

def create_bad_timestamp_yml(fpath):
    write_metadata(attributes_list_bad_timestamp_yml(), fpath)
    msg = 'Created {}'.format(fpath)
    logger.debug(msg)

def write_metadata(data, fpath):
    """Write dictionary to YAML file."""
    with open(fpath, "w") as f:
        # Template for rop_dates.yml
        f.write("# This file is used to store dates to drop by the parser, the time format used is the isoformat (YYYY-mm-dd HH:MM:SS). \n")
        f.write("# timestamp: list of timestamps \n")
        f.write("# time_period: list of list ranges of dates \n")
        f.write("# Example: \n")
        f.write("# timestamp: ['2018-12-07 14:15','2018-12-07 14:17','2018-12-07 14:19', '2018-12-07 14:25'] \n")
        f.write("# time_period: [['2018-08-01 12:00:00', '2018-08-01 14:00:00'], \n")
        f.write("#               ['2018-08-01 15:44:30', '2018-08-01 15:59:31'], \n")
        f.write("#               ['2018-08-02 12:44:30', '2018-08-02 12:59:31']] \n")

        yaml.dump(data, f)


def attributes_list_bad_timestamp_yml():
    """Attributes for drop_dates.yml creation"""
    attrs = {}
    attrs['timestamp'] = []
    attrs['time_period'] = []
    return attrs


def drop_timestamp_df(df, bad_timestamp, verbose):
        '''Drop rows on dataframe based on timestamp or time_period list of dates'''
        # Check dates into timestamp
        try:
            timestamp = bad_timestamp.get('timestamp')
            try:
                timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S')
            except ValueError:
                # Loop all for check error
                time_temp = []
                invalid_timestamp = []
                for date in timestamp:
                    try:
                        time_temp.append(pd.to_datetime(date, format='%Y-%m-%d %H:%M:%S'))
                    except ValueError:
                        # Not the right format, try to cast to timestamp anyway
                        try:
                            time_temp.append(pd.to_datetime(date))
                            msg = "Warning in timestamp, the date {} not in the format '%d-%m-%Y %H:%M:%S', but try to parse date.".format(date)
                            logger.warning(msg)
                            if verbose:
                                print(" - " + msg)
                        except Exception:
                            invalid_timestamp.append(date)
                            msg = "Error in timestamp, invalid value: {}".format(date)
                            logger.warning(msg)
                            if verbose:
                                print(" - " + msg)
                        pass
                timestamp = pd.DatetimeIndex(time_temp)
            except Exception:
                # Something wrong to get timestamp info
                msg = "Error on drop bad timestamp: something wrong to get timestamp information"
                logger.error(msg)
                raise NotImplementedError(" - " + msg)


            drop_dates = 0
            # Drop rows with date in df
            try:
                count_temp_dropped_dates_1 = 0
                count_temp_dropped_dates_1 = len(df)
                df = df[~df['time'].isin(timestamp)]
                count_temp_dropped_dates_1 = count_temp_dropped_dates_1 - len(df)
            except TypeError:
                for date in timestamp:
                    try:
                        df = df[~df['time'].isin(date)]
                        count_temp_dropped_dates_1 += 1
                        drop_dates += count_temp_dropped_dates_1
                    except NotImplementedError:
                        msg = "Cannot remove {} timestamp, so skipped".format(date)
                        invalid_timestamp.append(date)
                        logger.warning(msg)
                        if verbose:
                            print(" - " + msg)
                        pass
            if count_temp_dropped_dates_1:
                msg = "{} dates were dropped into timestamp".format(count_temp_dropped_dates_1)
                logger.debug(msg)
                if verbose:
                    print(' - ' + msg)
        except AttributeError:
            msg = "No dates into timestamp, so skip"
            logger.debug(msg)
            if verbose:
                print(' - ' + msg)
            pass

        try:
            count_temp_dropped_dates_2 = 0
            time_period = bad_timestamp.get('time_period')
            if time_period:
                for period in time_period:
                    if len(period) == 2:
                        try:
                            range_dates = pd.to_datetime(period, format='%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            msg = "Warning in time_period, the date {} not in the format '%d-%m-%Y %H:%M:%S', but try to parse date.".format(period)
                            logger.warning(msg)
                            if verbose:
                                print(' - ' + msg)
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
                        # Drop rows with date in df
                        count_temp_dropped_dates_2 = len(df[df['time'].between(period[0],period[1])])
                        # df = df[~(df['time'] >= range_dates[0]) & (df['time'] <= range_dates[1])]
                        # Check if drop dates
                        if not count_temp_dropped_dates_2 == 0:
                            df = df[~df['time'].between(period[0],period[1])]
                            drop_dates += count_temp_dropped_dates_2
                        else:
                            msg = "{} time period hasn't drop any date".format(period)
                            logger.warning(msg)
                            if verbose:
                                print(" - " + msg)

                        if count_temp_dropped_dates_2:
                            msg = "{} dates were dropped with time_period {}".format(count_temp_dropped_dates_2, period)
                            logger.debug(msg)
                            if verbose:
                                print(' - ' + msg)
                    else:
                        msg = "Error in time_period, wrong number's dates for define time period (need to be 2 timestamps) into: {}, so skipped".format(period)
                        logger.error(msg)
                        print(' - ' + msg)
                        pass
        except AttributeError:
            msg = "No dates range into time_period"
            logger.debug(msg)
            if verbose:
                print(' - ' + msg)
            pass


        msg = "{} totals dates were dropped by bad_timestamps.yml".format(drop_dates)
        logger.debug(msg)
        if verbose:
            print(' - ' + msg)

        # Check if df is empty
        if len(df) == 0:
            msg = "The df is empty, please check dates into bad_timestamp.yml"
            logger.error(msg)
            raise NotImplementedError(msg)

        return df
    
    
# -------------------------------------------------------------------------.
#### In check raw_dir 
    # Check there is /issue folder
    issue_dir = os.path.join(raw_dir, "issue")
    if "issue" not in list_subfolders:
        # - Create directory with empty issue files to be filled.
        _create_directory(issue_dir)

        bad_timestamp_path = os.path.join(issue_dir, 'bad_timestamp.yml')

        if not os.path.exists(bad_timestamp_path):
            create_bad_timestamp_yml(bad_timestamp_path) # To change name function in the future?
            logger.info(" - Created {}".format(bad_timestamp_path))
        else:
            pass
    else:
        bad_timestamp_path = os.path.join(issue_dir, 'bad_timestamp.yml')

        if not os.path.exists(bad_timestamp_path):
            create_bad_timestamp_yml(bad_timestamp_path) # To change name function in the future?
            logger.info(" - Created {}".format(bad_timestamp_path))

# -------------------------------------------------------------------------.
##### In read_L0_raw_file_list
    # ------------------------------------------------------.
    # Retrive bad dates into drop_dates.yml and remove timestamp
    # bad_timestamp_path = '' # Implement funtion to get bad_timestamp.yml's path
    bad_timestamp = read_bad_timestamp_yml(raw_dir)
    if bad_timestamp is not None:
        df = drop_timestamp_df(df, bad_timestamp, verbose)