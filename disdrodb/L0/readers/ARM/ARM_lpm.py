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
"""Reader for ARM Thies LPM sensor."""

import os
import time

# Directory
from disdrodb.L0.io import check_directories
from disdrodb.L0.io import get_campaign_name
from disdrodb.L0.io import create_directory_structure

# Logger
from disdrodb.utils.logger import create_l0_logger
from disdrodb.utils.logger import close_logger

# Metadata
from disdrodb.L0.metadata import read_metadata
from disdrodb.L0.check_standards import check_sensor_name

# L0 processing
from disdrodb.L0.L0A_processing import get_file_list
from disdrodb.L0.io import get_L0B_fpath
from disdrodb.L0.L0B_processing import write_L0B


from disdrodb.L0.L0_processing import reader_generic_docstring, is_documented_by


@is_documented_by(reader_generic_docstring)
def reader(
    raw_dir,
    processed_dir,
    l0a_processing,
    l0b_processing,
    keep_l0a,
    force=True,
    verbose=True,
    debugging_mode=False,
    lazy=True,
    single_netcdf=False,
):

    # Define functions to reformat ARM netCDFs
    def reformat_ARM_files(file_list, attrs):
        """
        Reformat LPM ARM netCDF files.

        Parameters
        ----------
        file_list : list
            Filepaths of NetCDFs to combine and reformat.
        attrs : dict
            DISDRODB metadata about the station.
        """
        from disdrodb.L0.utils_nc import xr_concat_datasets
        from disdrodb.L0.L0B_processing import get_coords
        from disdrodb.L0.auxiliary import get_ARM_LPM_dict
        from disdrodb.L0.standards import set_DISDRODB_L0_attrs

        sensor_name = attrs["sensor_name"]
        # --------------------------------------------------------
        #### Open netCDFs
        file_list = sorted(file_list)
        try:
            ds = xr_concat_datasets(file_list)
        except Exception as e:
            msg = f"Error in concatenating netCDF datasets. The error is: \n {e}"
            raise RuntimeError(msg)

        # --------------------------------------------------------
        # Select DISDRODB variable and rename
        dict_ARM = get_ARM_LPM_dict(sensor_name=sensor_name)
        vars_ARM = set(dict_ARM.keys())
        vars_ds = set(ds.data_vars)
        vars_selection = vars_ds.intersection(vars_ARM)
        dict_ARM_selection = {k: dict_ARM[k] for k in vars_selection}
        ds = ds[vars_selection]
        ds = ds.rename(dict_ARM_selection)

        # --------------------------------------------------------
        # Rename dimensions
        dict_dims = {
            "particle_diameter": "diameter_bin_center",
            "particle_fall_velocity": "velocity_bin_center",
        }
        ds = ds.rename_dims(dict_dims)
        ds = ds.rename(dict_dims)
        # --------------------------------------------------------
        # Update coordinates
        coords = get_coords(attrs["sensor_name"])
        coords["crs"] = attrs["crs"]
        coords["longitude"] = attrs["longitude"]
        coords["latitude"] = attrs["latitude"]
        coords["altitude"] = attrs["altitude"]
        ds = ds.drop(["latitude", "longitude", "altitude"])
        ds = ds.assign_coords(coords)

        # --------------------------------------------------------
        # Set DISDRODB attributes
        ds = set_DISDRODB_L0_attrs(ds, attrs)

        # --------------------------------------------------------
        return ds

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in raw_dir/data/<station_id>
    raw_data_glob_pattern = "*.nc"

    ####-------------------------------------------------------------------.
    ####################
    #### FIXED CODE ####
    ####################
    # ---------------------------------------------------------------------.
    # Initial directory checks
    raw_dir, processed_dir = check_directories(raw_dir, processed_dir, force=force)

    # Retrieve campaign name
    campaign_name = get_campaign_name(raw_dir)

    # --------------------------------------------------------------------.
    # Define logging settings
    logger = create_l0_logger(processed_dir, campaign_name)

    # ---------------------------------------------------------------------.
    # Create directory structure
    create_directory_structure(raw_dir, processed_dir)

    # ---------------------------------------------------------------------.
    # Get station list
    list_stations_id = os.listdir(os.path.join(raw_dir, "data"))

    # ---------------------------------------------------------------------.
    #### Loop over each station_id directory and process the files
    # station_id = list_stations_id[0]
    for station_id in list_stations_id:
        # ---------------------------------------------------------------------.
        t_i = time.time()
        msg = f" - Processing of station_id {station_id} has started"
        if verbose:
            print(msg)
        logger.info(msg)
        # ---------------------------------------------------------------------.
        # Retrieve metadata
        attrs = read_metadata(raw_dir=raw_dir, station_id=station_id)

        # Retrieve sensor name
        sensor_name = attrs["sensor_name"]
        check_sensor_name(sensor_name)

        # Retrieve list of files to process
        glob_pattern = os.path.join("data", station_id, raw_data_glob_pattern)
        file_list = get_file_list(
            raw_dir=raw_dir,
            glob_pattern=glob_pattern,
            verbose=verbose,
            debugging_mode=debugging_mode,
        )

        # -----------------------------------------------------------------.
        #### - Reformat netCDF to DISDRODB standards
        ds = reformat_ARM_files(file_list=file_list, attrs=attrs)

        # -----------------------------------------------------------------.
        #### - Save to DISDRODB netCDF standard
        fpath = get_L0B_fpath(processed_dir, station_id)
        ds = ds.compute()
        write_L0B(ds, fpath=fpath, sensor_name=sensor_name)

        # -----------------------------------------------------------------.
        # End L0 processing
        t_f = time.time() - t_i
        msg = " - NetCDF standardization of station_id {} ended in {:.2f}s".format(
            station_id, t_f
        )
        if verbose:
            print(msg)
        logger.info(msg)
        msg = " --------------------------------------------------"
        if verbose:
            print(msg)
        logger.info(msg)
        # -----------------------------------------------------------------.
    # -----------------------------------------------------------------.
    msg = "### Script finish ###"
    print("\n  " + msg + "\n")
    logger.info(msg)

    close_logger(logger)
    # -----------------------------------------------------------------.


# -----------------------------------------------------------------.
