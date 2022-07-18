#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2022 DISDRODB developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
import logging
import os
import shutil
import glob
import numpy as np
import pandas as pd
import dask.dataframe as dd
from disdrodb.utils.logger import log_info, log_warning
logger = logging.getLogger(__name__)

####---------------------------------------------------------------------------.
#### Directory/Filepaths Defaults
def infer_institute_from_fpath(fpath):
    idx_start = fpath.rfind("DISDRODB")
    disdrodb_fpath = fpath[idx_start:]
    institute = disdrodb_fpath.split("/")[2]   
    return institute 

def infer_campaign_from_fpath(fpath):
    idx_start = fpath.rfind("DISDRODB")
    disdrodb_fpath = fpath[idx_start:]
    campaign = disdrodb_fpath.split("/")[3] 
    return campaign 

def infer_station_id_from_fpath(fpath):
    idx_start = fpath.rfind("DISDRODB")
    disdrodb_fpath = fpath[idx_start:]
    station_id = disdrodb_fpath.split("/")[5]
    # Optional strip .yml if fpath point to YAML file 
    station_id.strip(".yml")  
    return station_id 


def get_campaign_name(base_dir):
    """Return the campaign name from 'raw_dir' or processed_dir' paths."""
    base_dir = parse_fpath(base_dir)
    campaign_name = os.path.basename(base_dir).upper()
    return campaign_name

def get_L0A_dir(processed_dir, station_id): 
    dir_path = os.path.join(processed_dir, "L0A", station_id)
    return dir_path

def get_L0A_fname(campaign_name, station_id, suffix=""):
    if suffix != "":
        suffix = "_" + suffix
    fname = campaign_name + "_s" + station_id + suffix + ".parquet"
    return fname


def get_L0A_fpath(processed_dir, station_id, suffix=""):
    campaign_name = get_campaign_name(processed_dir)
    fname = get_L0A_fname(campaign_name, station_id, suffix=suffix)
    dir_path = get_L0A_dir(processed_dir, station_id)
    fpath = os.path.join(dir_path, fname)
    return fpath

def get_L0B_dir(processed_dir, station_id): 
    dir_path = os.path.join(processed_dir, "L0B", station_id)
    return dir_path

def get_L0B_fname(campaign_name, station_id, suffix=""):
    if suffix != "":
        suffix = "_" + suffix
    # TODO: _s make sense with station_id... but if station_name a bit orrible
    fname = campaign_name + "_s" + station_id + suffix + ".nc"
    return fname


def get_L0B_fpath(processed_dir, station_id, suffix=""):
    campaign_name = get_campaign_name(processed_dir)
    fname = get_L0B_fname(campaign_name, station_id, suffix=suffix)
    dir_path = get_L0B_dir(processed_dir, station_id)
    fpath = os.path.join(dir_path, fname)
    return fpath

####--------------------------------------------------------------------------.
#### Directory/File Creation/Deletion


def _create_directory(path):
    if not isinstance(path, str):
        raise TypeError("'path' must be a strig.")
    try:
        os.makedirs(path)
        logger.debug(f"Created directory {path}.")
    except FileExistsError:
        logger.debug(f"Directory {path} already exists.")
        pass
    except (Exception) as e:
        dir_name = os.path.basename(path)
        msg = f"Can not create folder {dir_name} inside <path>. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)


def _remove_if_exists(fpath, force=False):
    if os.path.exists(fpath):
        if not force:
            msg = f"--force is False and a file already exists at:{fpath}"
            logger.error(msg)
            raise ValueError(msg)
        try:
            os.remove(fpath)
        except IsADirectoryError:
            try:
                os.rmdir(fpath)
            except OSError:
                try:
                    # shutil.rmtree(fpath.rpartition('.')[0])
                    for f in glob.glob(fpath + "/*"):
                        try:
                            os.remove(f)
                        except OSError as e:
                            msg = f"Can not delete file {f}, error: {e.strerror}"
                            logger.exception(msg)
                    os.rmdir(fpath)
                except:
                    msg = f"Something wrong with: {fpath}"
                    logger.error(msg)
                    raise ValueError(msg)
        logger.info(f"Deleted folder {fpath}")


####--------------------------------------------------------------------------.
#### Directory checks
def parse_fpath(fpath):
    """Ensure fpath does not end with /."""
    if not isinstance(fpath, str):
        raise TypeError("'parse_fpath' expects a directory/filepath string.")
    if fpath[-1] == "/":
        print("{} should not end with /.".format(fpath))
        fpath = fpath[:-1]
    return fpath

####--------------------------------------------------------------------------.
#### L0 processing directory checks
def check_raw_dir(raw_dir, verbose=False):
    """Check validity of raw_dir.

    Steps:
    1. Check that 'raw_dir' is a valid directory path
    2. Check that 'raw_dir' follows the expect directory structure
    3. Check that each station_id directory contains data
    4. Check that for each station_id the mandatory metadata are specified.

    """
    from disdrodb.L0.metadata import create_metadata
    from disdrodb.L0.metadata import check_metadata_compliance
    from disdrodb.L0.issue import create_issue_yml
    from disdrodb.L0.issue import check_issue_compliance
    # -------------------------------------------------------------------------.
    # Check input argument
    if not isinstance(raw_dir, str):
        raise TypeError("Provide 'raw_dir' as a string'.")
    if not os.path.exists(raw_dir):
        raise ValueError("'raw_dir' {} directory does not exist.".format(raw_dir))
    if not os.path.isdir(raw_dir):
        raise ValueError("'raw_dir' {} is not a directory.".format(raw_dir))
    # -------------------------------------------------------------------------.
    #### Check there is /data subfolders
    list_subfolders = os.listdir(raw_dir)
    if len(list_subfolders) == 0:
        raise ValueError("There are not subfolders in {}".format(raw_dir))
    if "data" not in list_subfolders:
        raise ValueError(
            "'raw_dir' {} should have the /data subfolder.".format(raw_dir)
        )
    
    # -------------------------------------------------------------------------.
    #### Check there are subfolders corresponding to station to process
    raw_data_dir= os.path.join(raw_dir, "data")
    list_data_station_id = os.listdir(raw_data_dir)
    if len(list_data_station_id) == 0:
        raise ValueError("No station directories within {}".format(raw_data_dir))
    
    # -------------------------------------------------------------------------.
    #### Check there are data files in each list_data_station_id
    list_raw_data_station_dir= [
        os.path.join(raw_data_dir, station_id) for station_id in list_data_station_id
    ]
    list_nfiles_per_station = [
        len(glob.glob(os.path.join(path, "*"))) for path in list_raw_data_station_dir
    ]
    idx_0_files = np.where(np.array(list_nfiles_per_station) == 0)[0]
    if len(idx_0_files) > 0:
        empty_station_dir = [list_raw_data_station_dir[idx] for idx in idx_0_files]
        raise ValueError(
            "The following data directories are empty: {}".format(empty_station_dir)
        )
        
    # -------------------------------------------------------------------------.
    #### Check there is /metadata subfolders
    metadata_dir = os.path.join(raw_dir, "metadata")
    if "metadata" not in list_subfolders:
        # - Create metadata directory 
        _create_directory(metadata_dir)
        # - Create default metadata yml file for each station (since the folder didn't existed)
        list_metadata_fpath = [
            os.path.join(metadata_dir, station_id + ".yml")
            for station_id in list_data_station_id
        ]
        _ = [create_metadata(fpath) for fpath in list_metadata_fpath]
        msg = "'raw_dir' {} should have the /metadata subfolder. ".format(raw_dir)
        msg1 = "It has been now created with also empty metadata files to be filled for each station."
        raise ValueError(msg + msg1)

    # -------------------------------------------------------------------------.
    #### Check there are metadata file for each station_id in /metadata
    list_metadata_fpath = glob.glob(os.path.join(metadata_dir, "*.yml"))
    list_metadata_fname = [os.path.basename(fpath) for fpath in list_metadata_fpath]
    list_metadata_station_id = [fname[:-4] for fname in list_metadata_fname]
    
    # - Check there is metadata for each station
    missing_data_station_idx = np.where(
        np.isin(list_data_station_id, list_metadata_station_id, invert=True)
    )[0]
    # - If missing, create the defaults files and raise an error
    if len(missing_data_station_idx) > 0:
        list_missing_station_id = [
            list_data_station_id[idx] for idx in missing_data_station_idx
        ]
        list_missing_metadata_fpath = [
            os.path.join(metadata_dir, station_id + ".yml")
            for station_id in list_missing_station_id
        ]
        _ = [create_metadata(fpath) for fpath in list_missing_metadata_fpath]
        msg = "The metadata files for the following station_id were missing: {}".format(
            list_missing_station_id
        )
        raise ValueError(msg + " Now have been created to be filled.")
        
    # - Check not excess metadata compared to present stations
    excess_metadata_station_idx = np.where(np.isin(list_metadata_station_id, list_data_station_id, invert=True))[0]
    if len(excess_metadata_station_idx) > 0:
        list_excess_station_id = [
            list_metadata_station_id[idx] for idx in excess_metadata_station_idx
        ]
        print(
            "There are the following metadata files without corresponding data: {}".format(
                list_excess_station_id
            )
        )
        
    # -------------------------------------------------------------------------.
    #### Check metadata compliance
    _ = [check_metadata_compliance(fpath) for fpath in list_metadata_fpath]
    # TODO: MISSING IMPLEMENTATION OF check_metadata_compliance
    # -------------------------------------------------------------------------.
    #### Check there is /issue subfolder
    issue_dir = os.path.join(raw_dir, "issue")
    if "issue" not in list_subfolders:
        # - Create issue directory 
        _create_directory(issue_dir)
        # - Create issue yml file for each station (since the folder didn't existed)
        list_issue_fpath = [os.path.join(issue_dir, station_id + ".yml")
            for station_id in list_data_station_id
        ]
        _ = [create_issue_yml(fpath) for fpath in list_issue_fpath]
        msg = "The /issue subfolder has been now created to document and then remove timesteps with problematic data."
        logger.info(msg)
    # -------------------------------------------------------------------------.
    #### Check there are issue file for each station_id in /issue 
    list_issue_fpath = glob.glob(os.path.join(issue_dir, "*.yml"))
    list_issue_fname = [os.path.basename(fpath) for fpath in list_issue_fpath]
    list_issue_station_id = [fname[:-4] for fname in list_issue_fname]
    
    # - Check there is issue for each station
    missing_data_station_idx = np.where(
        np.isin(list_data_station_id, list_issue_station_id, invert=True)
    )[0]
    # - If missing, create the defaults files and raise an error
    if len(missing_data_station_idx) > 0:
        list_missing_station_id = [
            list_data_station_id[idx] for idx in missing_data_station_idx
        ]
        list_missing_issue_fpath = [
            os.path.join(issue_dir, station_id + ".yml")
            for station_id in list_missing_station_id
        ]
        _ = [create_issue_yml(fpath) for fpath in list_missing_issue_fpath]
        msg = "The issue files for the following station_id were missing: {}".format(
            list_missing_station_id
        )
        
    # - Check not excess issue compared to present stations
    excess_issue_station_idx = np.where(np.isin(list_issue_station_id, list_data_station_id, invert=True))[0]
    if len(excess_issue_station_idx) > 0:
        list_excess_station_id = [list_issue_station_id[idx] for idx in excess_issue_station_idx]
        msg =  f"There are the following issue files without corresponding data: {list_excess_station_id}"
        log_warning(logger, msg, verbose)

    # -------------------------------------------------------------------------.
    #### Check issue compliance
    _ = [check_issue_compliance(fpath) for fpath in list_issue_fpath]
    # TODO: MISSING IMPLEMENTATION OF check_issue_compliance
    # -------------------------------------------------------------------------.
    return None
    
def check_processed_dir(processed_dir, force=False):
    """Check that 'processed_dir' is a valid directory path."""
    if not isinstance(processed_dir, str):
        raise TypeError("Provide 'processed_dir' as a string'.")
    #------------------------------    
    # Check processed_dir has "DISDRODB/Processed" to avoid deleting precious stuffs 
    if processed_dir.find("DISDRODB/Processed") == -1: 
        msg = "Expecting 'processed_dir' to contain the pattern */DISDRODB/Processed/*."
        logger.error(msg)
        raise ValueError(msg)
        
    # Check processed_dir does not end with "DISDRODB/Processed" 
    # - It must contain also the <campaign_name> directory  
    if processed_dir.endswith("Processed") or processed_dir.endswith("Processed/"):
        msg = "Expecting 'processed_dir' to contain the pattern */DISDRODB/Processed/<campaign_name>."
        logger.error(msg)
        raise ValueError(msg)

    #------------------------------
    # If forcing overwriting 
    if force: 
        # Check processed_dir is a directory before removing the content 
        if os.path.exists(processed_dir):
            if not os.path.isdir(processed_dir):
                msg = f"'processed_dir' {processed_dir} already exist but is not a directory."
                logger.error(msg)
                raise ValueError(msg) 
            # Remove content of existing processed_dir
            # TODO: maybe remove also the campaign name directory, not only the content of it
            shutil.rmtree(processed_dir)  # TODO: !! TOO DANGEROUS ??? !!!

    #------------------------------
    # If avoiding overwriting 
    if not force:
        if os.path.exists(processed_dir):
            msg = f"'processed_dir' {processed_dir} already exists and force=False."
            logger.error(msg)
            raise ValueError(msg)
            
    #------------------------------        
    # Recreate processed_dir
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    else:
        msg = "Please report the BUG. This should not happen."  
        logger.error(msg)
        raise ValueError(msg)
           

def check_campaign_name(raw_dir, processed_dir):
    """Check that 'raw_dir' and 'processed_dir' have same campaign_name."""
    upper_campaign_name = os.path.basename(raw_dir).upper()
    raw_campaign_name = os.path.basename(raw_dir)
    processed_campaign_name = os.path.basename(processed_dir)
    if raw_campaign_name != processed_campaign_name:
        msg = f"'raw_dir' and 'processed_dir' must ends with same <campaign_name> {upper_campaign_name}"
        logger.error(msg)
        raise ValueError(msg)
            
    if raw_campaign_name != upper_campaign_name:
        msg = f"'raw_dir' and 'processed_dir' must ends with UPPERCASE <campaign_name> {upper_campaign_name}"
        logger.error(msg)
        raise ValueError(msg)

    return upper_campaign_name


def check_directories(raw_dir, processed_dir, force=False):
    """Check that the specified directories respect the standards."""
    raw_dir = parse_fpath(raw_dir)
    processed_dir = parse_fpath(processed_dir)
    check_raw_dir(raw_dir)
    check_processed_dir(processed_dir, force=force)
    check_campaign_name(raw_dir, processed_dir)
    return raw_dir, processed_dir

def check_metadata_dir(processed_path):
    # Create metadata folder
    try:
        metadata_folder_path = os.path.join(processed_path, "metadata")
        os.makedirs(metadata_folder_path)
        logger.debug(f"Created {metadata_folder_path}.")
    except FileExistsError:
        logger.debug(f"{metadata_folder_path} already existed.")
        pass
    except (Exception) as e:
        msg = f"Folder metadata can not be created in {metadata_folder_path}>. \n The error is: {e}."
        logger.exception(msg)
        raise FileNotFoundError(msg)

def copy_metadata_from_raw_dir(raw_dir, processed_dir):
    '''Copy yaml files in raw_dir/metadata into processed_dir/metadata'''
    # Get src and dst metadata directory 
    raw_metadata_dir = os.path.join(raw_dir, "metadata")
    processed_metadata_dir = os.path.join(processed_dir, "metadata")
    # Retrieve metadata fpaths in raw directory
    raw_metadata_fpaths = glob.glob(os.path.join(raw_metadata_dir, "*.yml"))
    # Copy all metadata yml files into the "processed" folder
    for raw_metadata_fpath in raw_metadata_fpaths:
        # Check if is a files
        if os.path.isfile(raw_metadata_fpath):
            metadata_fname = os.path.basename(raw_metadata_fpath)
            processed_metadata_fpath = os.path.join(processed_metadata_dir, metadata_fname)
            try:
                # Copy every file
                shutil.copy(raw_metadata_fpath, processed_metadata_fpath)
                msg = f'{metadata_fname} copied into {processed_metadata_dir}.' 
                logger.info(msg)
            except (Exception) as e:
                msg = f'Something went wrong when copying {metadata_fname} into {processed_metadata_dir}.\n The error is: {e}.'
                logger.error(msg)
        else:
            msg = f'Cannot copy {metadata_fname} into {processed_metadata_dir}.'
            logger.error(msg)
            raise ValueError(msg)
    metadata_fnames = [os.path.basename(metadata_fpath) for metadata_fpath in raw_metadata_fpaths]
    msg = f'The metadata of stations ({metadata_fnames}) have been copied into {processed_metadata_dir}.' 
    logger.info(msg)
    
####--------------------------------------------------------------------------.
#### L0 processing directory structure


def create_directory_structure(raw_dir, processed_dir):
    """Create directory structure for L0A and L0B processing."""
    #-----------------------------------------------------.
    #### Create metadata folder inside processed_dir
    check_metadata_dir(processed_dir)
    copy_metadata_from_raw_dir(raw_dir, processed_dir)
    
    #-----------------------------------------------------.
    #### Create info folder inside processed_dir
    try:
        info_folder_path = os.path.join(processed_dir, "info")
        os.makedirs(info_folder_path)
        logger.debug(f"Created {info_folder_path}")
    except FileExistsError:
        logger.debug(f"Found {info_folder_path}")
        pass
    except (Exception) as e:
        msg = f"Can not create folder metadata inside <info_folder_path>. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)
        
    #-----------------------------------------------------.
    #### Create L0A folder inside processed_dir
    try:
        L0A_folder_path = os.path.join(processed_dir, "L0A")
        os.makedirs(L0A_folder_path)
        logger.debug(f"Created {L0A_folder_path}")
    except FileExistsError:
        logger.debug(f"Found {L0A_folder_path}")
        pass
    except (Exception) as e:
        msg = f"Can not create folder L0A inside {processed_dir}. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)
        
    #-----------------------------------------------------.
    #### Create L0B folder inside processed_dir
    try:
        L0B_folder_path = os.path.join(processed_dir, "L0B")
        os.makedirs(L0B_folder_path)
        logger.debug(f"Created {L0B_folder_path}")
    except FileExistsError:
        logger.debug(f"Found {L0B_folder_path}")
        pass
    except (Exception) as e:
        msg = f"Can not create folder L0B inside {processed_dir}. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)
    return


####--------------------------------------------------------------------------.
#### DISDRODB L0A Readers
def _read_L0A(fpath, lazy=True, verbose=False): 
    # Log
    msg = f" - Reading L0 Apache Parquet file at {fpath} started."
    log_info(logger, msg, verbose)
    # Read
    if lazy:
        df = dd.read_parquet(fpath)
    else:
        df = pd.read_parquet(fpath)
    # Log
    msg = f" - Reading L0 Apache Parquet file at {fpath} ended."
    log_info(logger, msg, verbose)
    return df 

def read_L0A_dataframe(fpaths, lazy=True, verbose=False, debugging_mode=False):
    """
    Read DISDRODB L0A Apache Parquet file(s). 

    Parameters
    ----------
    fpaths : str or list
        Either a list or a single filepath .
    verbose : bool
        Whether to print detailed processing information into terminal. 
        The default is False.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        If fpaths is a list, it reads only the first 3 files 
        The default is False.
    lazy : bool
        Whether to read the dataframe lazily with dask. 
        If lazy=True, it returns a dask.dataframe.
        If lazy=False, it returns a pandas.DataFrame  
        The default is True.
    """
    from disdrodb.L0.L0A_processing import concatenate_dataframe
    # ----------------------------------------
    # Check fpaths validity
    if not isinstance(fpaths, (list,str)): 
        raise TypeError("Expecting fpaths to be a string or a list of strings.")
    # TODO:
    # - CHECK ENDS WITH .parquet  
    # ----------------------------------------
    # If list of length 1, convert to string
    if isinstance(fpaths, list) and len(fpaths) == 1:
        fpaths = fpaths[0]
    # ---------------------------------------------------
    # If more than 1 fpath, read and concantenate first
    if isinstance(fpaths, list): 
        if debugging_mode: 
            fpaths = fpaths[0:3] # select first 3 fpaths
        list_df = [_read_L0A(fpath, lazy=lazy, verbose=verbose) for fpath in fpaths]
        df = concatenate_dataframe(list_df, verbose=verbose, lazy=lazy)
    # Else read the single file 
    else: 
        df = _read_L0A(fpaths, lazy=lazy, verbose=verbose)
    # ---------------------------------------------------
    # Return dataframe 
    return df 

####---------------------------------------------------------------------------.
#### TO BE DEPRECATED 
#### Back-compatibility stuffs 
# TODO: DEPRECATE 
def get_L0_fname(campaign_name, station_id, suffix=""):
    return get_L0A_fname(campaign_name, station_id, suffix)


def get_L0_fpath(processed_dir, station_id, suffix=""):
   return get_L0A_fpath(processed_dir, station_id, suffix)


def get_L1_netcdf_fname(campaign_name, station_id, suffix=""):
    return get_L0B_fname(campaign_name, station_id, suffix)


def get_L1_netcdf_fpath(processed_dir, station_id, suffix=""):
    return get_L0B_fpath(processed_dir, station_id, suffix)


def check_L0_is_available(processed_dir, station_id, suffix=""):
    fpath = get_L0A_fpath(processed_dir, station_id, suffix=suffix)
    if not os.path.exists(fpath):
        msg = f"Need to run L0 processing first. The Apache Parquet file {fpath} is not available."
        logger.exception(msg)
        raise ValueError(msg)
    # Log
    msg = f"Found parquet file: {fpath}"
    logger.info(msg)


def read_L0_data(processed_dir, station_id, suffix="", 
                 lazy=True, verbose=False, debugging_mode=False):
    """Read L0 Apache Parquet into dataframe.
    
    If debugging_mode = True, return just a subset of total rows.
    """
    # Check L0 is available
    check_L0_is_available(processed_dir, station_id, suffix=suffix)
    # Define fpath
    fpath = get_L0A_fpath(processed_dir, station_id, suffix=suffix)
    # Read L0A Apache Parquet file 
    df = _read_L0A(fpath, lazy=lazy, verbose=verbose)
    # Subset dataframe if debugging_mode = True
    if debugging_mode:
        if not lazy:
            df = df.iloc[0:100, :]
        else:
            NotImplementedError
    return df


####--------------------------------------------------------------------------.

