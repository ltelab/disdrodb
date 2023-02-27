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
import sys
import click
from disdrodb.utils.scripts import parse_arg_to_list
from disdrodb.l0.l0_processing import (
    click_l0_processing_options,
    click_l0_stations_options,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click.argument("disdrodb_dir", metavar="<disdrodb_dir>")
@click_l0_stations_options
@click_l0_processing_options
def run_disdrodb_l0a(
    disdrodb_dir,
    # L0 disdrodb stations options
    data_sources=None,
    campaign_names=None,
    station_names=None,
    # Processing options
    force: bool = False,
    verbose: bool = True,
    parallel: bool = True,
    debugging_mode: bool = False,
):
    """Run the L0A processing of DISDRODB stations.

    This function enable to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
    disdrodb_dir : str
        Base directory of DISDRODB
        Format: <...>/DISDRODB
    data_sources : str
        Name of data source(s) to process.
        The name(s) must be UPPER CASE.
        If campaign_names and station are not specified, process all stations.
        To specify multiple data sources, write i.e.: --data_sources 'GPM EPFL NCAR'
    campaign_names : str
        Name of the campaign(s) to process.
        The name(s) must be UPPER CASE.
        To specify multiple campaigns, write i.e.: --campaign_names 'IPEX IMPACTS'
    station_names : str
        Station names.
        To specify multiple stations, write i.e.: --station_names 'station1 station2'
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is False.
    parallel : bool
        If True, the files are processed simultanously in multiple processes.
        Each process will use a single thread.
        By default, the number of process is defined with os.cpu_count().
        However, you can customize it by typing: DASK_NUM_WORKERS=4 run_disdrodb_l0a
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        It processes just the first 3 raw data files for each station.
        The default is False.
    """
    from disdrodb.l0.l0_processing import run_disdrodb_l0a

    # Parse data_sources, campaign_names and station arguments
    print(data_sources)
    print(campaign_names)
    print(station_names)

    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    # Run processing
    run_disdrodb_l0a(
        disdrodb_dir=disdrodb_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )

    return None


if __name__ == "__main__":
    run_disdrodb_l0a()
