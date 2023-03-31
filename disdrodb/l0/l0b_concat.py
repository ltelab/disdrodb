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
import os
import glob
import logging
from disdrodb.l0.io import get_L0B_dir, get_L0B_fpath
from disdrodb.utils.netcdf import xr_concat_datasets
from disdrodb.utils.scripts import _execute_cmd
from disdrodb.utils.logger import (
    create_file_logger,
    close_logger,
    log_info,
    log_warning,
    # log_debug,
    log_error,
)

logger = logging.getLogger(__name__)


def _concatenate_netcdf_files(processed_dir, station_name, remove=False, verbose=False):
    """Concatenate all L0B netCDF files into a single netCDF file.

    The single netCDF file is saved at <processed_dir>/L0B.
    """
    from disdrodb.l0.l0b_processing import write_l0b

    # Create logger
    filename = f"concatenatation_{station_name}"
    logger = create_file_logger(
        processed_dir=processed_dir,
        product_level="L0B",
        station_name="",  # locate outside the station directory
        filename=filename,
        parallel=False,
    )

    # -------------------------------------------------------------------------.
    # Retrieve L0B files
    L0B_dir_path = get_L0B_dir(processed_dir, station_name)
    file_list = sorted(glob.glob(os.path.join(L0B_dir_path, "*.nc")))

    # -------------------------------------------------------------------------.
    # Check there are at least two files
    n_files = len(file_list)
    if n_files == 0:
        msg = f"No L0B file is available for concatenation in {L0B_dir_path}."
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)

    if n_files == 1:
        msg = f"Only a single file is available for concatenation in {L0B_dir_path}."
        log_warning(logger=logger, msg=msg, verbose=verbose)
        raise ValueError(msg)

    # -------------------------------------------------------------------------.
    # Concatenate the files
    ds = xr_concat_datasets(file_list)

    # -------------------------------------------------------------------------.
    # Define the filepath of the concatenated L0B netCDF
    single_nc_fpath = get_L0B_fpath(ds, processed_dir, station_name, l0b_concat=True)
    force = True  # TODO add as argument
    write_l0b(ds, fpath=single_nc_fpath, force=force)

    # -------------------------------------------------------------------------.
    # Close file and delete
    ds.close()
    del ds

    # -------------------------------------------------------------------------.
    # If remove = True, remove all the single files
    if remove:
        log_info(logger=logger, msg="Removal of single L0B files started.", verbose=verbose)
        _ = [os.remove(fpath) for fpath in file_list]
        log_info(logger=logger, msg="Removal of single L0B files ended.", verbose=verbose)

    # -------------------------------------------------------------------------.
    # Close the file logger
    close_logger(logger)

    # Return the dataset
    return None


####---------------------------------------------------------------------------.
#### Wrappers of run_disdrodb_l0b_concat_station call


def run_disdrodb_l0b_concat_station(
    disdrodb_dir,
    data_source,
    campaign_name,
    station_name,
    remove_l0b=False,
    verbose=False,
):
    """Concatenate the L0B files of a single DISDRODB station.

    This function runs the run_disdrodb_l0b_concat_station script in the terminal.
    """
    cmd = " ".join(
        [
            "run_disdrodb_l0b_concat_station",
            disdrodb_dir,
            data_source,
            campaign_name,
            station_name,
            "--remove_l0b",
            str(remove_l0b),
            "--verbose",
            str(verbose),
        ]
    )
    _execute_cmd(cmd)


def run_disdrodb_l0b_concat(
    disdrodb_dir,
    data_sources=None,
    campaign_names=None,
    station_names=None,
    remove_l0b=False,
    verbose=False,
):
    """Concatenate the L0B files of the DISDRODB archive.

    This function is called by the run_disdrodb_l0b_concat script.
    """
    from disdrodb.api.io import available_stations

    list_info = available_stations(
        disdrodb_dir,
        product_level="L0B",
        data_sources=data_sources,
        campaign_names=campaign_names,
    )
    # If no stations available, raise an error
    if len(list_info) == 0:
        raise ValueError("No stations to concatenate!")

    # Filter by provided stations
    if station_names is not None:
        list_info = [info for info in list_info if info[2] in station_names]
        # If nothing left, raise an error
        if len(list_info) == 0:
            raise ValueError("No stations to concatenate given the provided `station` argument!")

    # Print message
    n_stations = len(list_info)
    print(f"Concatenation of {n_stations} L0B stations started.")

    # Start the loop to launch the concatenation of each station
    for data_source, campaign_name, station_name in list_info:
        run_disdrodb_l0b_concat_station(
            disdrodb_dir=disdrodb_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            remove_l0b=remove_l0b,
            verbose=verbose,
        )

    print(f"Concatenation of {n_stations} L0B stations ended.")


####--------------------------------------------------------------------------.
