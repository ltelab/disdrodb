# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2023 DISDRODB developers
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
"""DISDRODB CLI routine wrappers."""
import datetime
import time
from typing import Optional

from disdrodb.api.search import available_stations, get_required_product
from disdrodb.utils.cli import _execute_cmd

####--------------------------------------------------------------------------.
#### Run DISDRODB Station Processing


def run_l0_station(
    data_source,
    campaign_name,
    station_name,
    # L0 archive options
    l0a_processing: bool = True,
    l0b_processing: bool = True,
    l0c_processing: bool = True,
    remove_l0a: bool = False,
    remove_l0b: bool = False,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Run the L0 processing of a specific DISDRODB station from the terminal.

    Parameters
    ----------
    data_source : str
        Institution name (when campaign data spans more than 1 country),
        or country (when all campaigns (or sensor networks) are inside a given country).
        Must be UPPER CASE.
    campaign_name : str
        Campaign name. Must be UPPER CASE.
    station_name : str
        Station name
    l0a_processing : bool
        Whether to launch processing to generate L0A Apache Parquet file(s) from raw data.
        The default value is ``True``.
    l0b_processing : bool
        Whether to launch processing to generate L0B netCDF4 file(s) from L0A data.
        The default value is ``True``.
    l0b_processing : bool
        Whether to launch processing to generate L0C netCDF4 file(s) from L0B data.
        The default value is ``True``.
    l0c_processing : bool
        Whether to launch processing to generate L0C netCDF4 file(s) from L0C data.
        The default is True.
    remove_l0a : bool
        Whether to keep the L0A files after having generated the L0B netCDF products.
        The default value is ``False``.
    remove_l0b : bool
        Whether to remove the L0B files after having produced L0C netCDF files.
        The default is False.
    force : bool
        If ``True``, overwrite existing data into destination directories.
        If ``False``, raise an error if there are already data into destination directories.
        The default value is ``False``.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default value is ``True``.
    parallel : bool
        If ``True``, the files are processed simultaneously in multiple processes.
        Each process will use a single thread to avoid issues with the HDF/netCDF library.
        By default, the number of process is defined with ``os.cpu_count()``.
        If ``False``, the files are processed sequentially in a single process.
        If ``False``, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If ``True``, it reduces the amount of data to process.
        For L0A, it processes just the first 3 raw data files for each station.
        For L0B, it processes just the first 100 rows of 3 L0A files for each station.
        The default value is ``False``.
    data_archive_dir : str (optional)
        The directory path where the DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    # ---------------------------------------------------------------------.
    t_i = time.time()
    print(f"L0 processing of station {station_name} has started.")

    # ------------------------------------------------------------------.
    # L0A processing
    if l0a_processing:
        run_l0a_station(
            # DISDRODB root directories
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            # Station arguments
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # Processing options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )
    # ------------------------------------------------------------------.
    # L0B processing
    if l0b_processing:
        run_l0b_station(
            # DISDRODB root directories
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            # Station arguments
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # L0B processing options
            remove_l0a=remove_l0a,
            # Processing options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )

    # ------------------------------------------------------------------.
    # L0C processing
    if l0c_processing:
        run_l0c_station(
            # DISDRODB root directories
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            # Station arguments
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # L0C processing options
            remove_l0b=remove_l0b,
            # Processing options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )

    # -------------------------------------------------------------------------.
    # End of L0 processing for all stations
    timedelta_str = str(datetime.timedelta(seconds=round(time.time() - t_i)))
    print(f"L0 processing of stations {station_name} completed in {timedelta_str}")


def run_l0a_station(
    # Station arguments
    data_source,
    campaign_name,
    station_name,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """
    Run the L0A processing of a station by invoking the disdrodb_run_l0a_station command in the terminal.

    Parameters
    ----------
    data_source : str
        The name of the data source.
    campaign_name : str
        The name of the campaign.
    station_name : str
        The name of the station.
    force : bool, optional
        If ``True``, overwrite existing data in destination directories.
        The default value is ``False``.
    verbose : bool, optional
        If ``True``, print detailed processing information to the terminal.
        The default value is ``False``.
    debugging_mode : bool, optional
        If ``True``, reduce the amount of data to process for debugging.
        The default value is ``False``.
    parallel : bool, optional
        If ``True``, process files in multiple processes simultaneously.
        The default value is ``True``.
    data_archive_dir
        The directory path where the local DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    metadata_archive_dir
        The directory path where the DISDRODB Metadata Archive is located.
        The directory path must end with ``<...>/DISDRODB-METADATA/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    # Define command
    cmd = " ".join(
        [
            "disdrodb_run_l0a_station",
            # Station arguments
            data_source,
            campaign_name,
            station_name,
            # DISDRODB root directories
            "--data_archive_dir",
            str(data_archive_dir),
            "--metadata_archive_dir",
            str(metadata_archive_dir),
            # Processing options
            "--force",
            str(force),
            "--verbose",
            str(verbose),
            "--debugging_mode",
            str(debugging_mode),
            "--parallel",
            str(parallel),
        ],
    )
    # Execute command
    _execute_cmd(cmd)


def run_l0b_station(
    # Station arguments
    data_source,
    campaign_name,
    station_name,
    # L0B processing options
    remove_l0a: bool = False,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """
    Run the L0B processing of a station by invoking the disdrodb_run_l0b_station command in the terminal.

    Parameters
    ----------
    data_archive_dir
        The directory path where the local DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    metadata_archive_dir
        The directory path where the DISDRODB Metadata Archive is located.
        The directory path must end with ``<...>/DISDRODB-METADATA/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    data_source : str
        The name of the data source.
    campaign_name : str
        The name of the campaign.
    station_name : str
        The name of the station.
    remove_l0a : bool, optional
        Whether to keep the L0A files after generating L0B netCDF files.
        The default value is ``False``.
    force : bool, optional
        If ``True``, overwrite existing data in destination directories.
        The default value is ``False``.
    verbose : bool, optional
        If ``True``, print detailed processing information to the terminal.
        The default value is ``False``.
    debugging_mode : bool, optional
        If ``True``, reduce the amount of data processed for debugging.
        The default value is ``False``.
    parallel : bool, optional
        If ``True``, process files in multiple processes simultaneously.
        The default value is ``True``.
    """
    # Define command
    cmd = " ".join(
        [
            "disdrodb_run_l0b_station",
            # Station arguments
            data_source,
            campaign_name,
            station_name,
            # DISDRODB root directories
            "--data_archive_dir",
            str(data_archive_dir),
            "--metadata_archive_dir",
            str(metadata_archive_dir),
            # L0B processing options
            "--remove_l0a",
            str(remove_l0a),
            # Processing options
            "--force",
            str(force),
            "--verbose",
            str(verbose),
            "--debugging_mode",
            str(debugging_mode),
            "--parallel",
            str(parallel),
        ],
    )
    # Execute command
    _execute_cmd(cmd)


def run_l0c_station(
    # Station arguments
    data_source,
    campaign_name,
    station_name,
    # L0C options
    remove_l0b: bool = False,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """
    Run the L0C processing of a station by invoking the disdrodb_run_l0c_station command in the terminal.

    Parameters
    ----------
    data_source : str
        The name of the data source.
    campaign_name : str
        The name of the campaign.
    station_name : str
        The name of the station.
    remove_l0b : bool, optional
        Whether to remove the L0B files after generating L0C netCDF files.
        The default value is ``False``.
    force : bool, optional
        If ``True``, overwrite existing data in destination directories.
        The default value is ``False``.
    verbose : bool, optional
        If ``True``, print detailed processing information to the terminal.
        The default value is ``False``.
    debugging_mode : bool, optional
        If ``True``, reduce the amount of data processed for debugging.
        The default value is ``False``.
    parallel : bool, optional
        If ``True``, process files in multiple processes simultaneously.
        The default value is ``True``.
    data_archive_dir
        The directory path where the local DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    metadata_archive_dir
        The directory path where the DISDRODB Metadata Archive is located.
        The directory path must end with ``<...>/DISDRODB-METADATA/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    # Define command
    cmd = " ".join(
        [
            "disdrodb_run_l0c_station",
            # Station arguments
            data_source,
            campaign_name,
            station_name,
            # DISDRODB root directories
            "--data_archive_dir",
            str(data_archive_dir),
            "--metadata_archive_dir",
            str(metadata_archive_dir),
            # L0C processing options
            "--remove_l0b",
            str(remove_l0b),
            # Processing options
            "--force",
            str(force),
            "--verbose",
            str(verbose),
            "--debugging_mode",
            str(debugging_mode),
            "--parallel",
            str(parallel),
        ],
    )
    # Execute command
    _execute_cmd(cmd)


def run_l1_station(
    # Station arguments
    data_source,
    campaign_name,
    station_name,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """
    Run the L1 processing of a station by invoking the disdrodb_run_l1_station command in the terminal.

    Parameters
    ----------
    data_source : str
        The name of the data source.
    campaign_name : str
        The name of the campaign.
    station_name : str
        The name of the station.
    force : bool, optional
        If ``True``, overwrite existing data in destination directories.
        The default value is ``False``.
    verbose : bool, optional
        If ``True``, print detailed processing information to the terminal.
        The default value is ``False``.
    debugging_mode : bool, optional
        If ``True``, reduce the amount of data processed for debugging.
        The default value is ``False``.
    parallel : bool, optional
        If ``True``, process files in multiple processes simultaneously.
        The default value is ``True``.
    data_archive_dir
        The directory path where the local DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    metadata_archive_dir
        The directory path where the DISDRODB Metadata Archive is located.
        The directory path must end with ``<...>/DISDRODB-METADATA/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    # Define command
    cmd = " ".join(
        [
            "disdrodb_run_l1_station",
            # Station arguments
            data_source,
            campaign_name,
            station_name,
            # DISDRODB root directories
            "--data_archive_dir",
            str(data_archive_dir),
            "--metadata_archive_dir",
            str(metadata_archive_dir),
            # Processing options
            "--force",
            str(force),
            "--verbose",
            str(verbose),
            "--debugging_mode",
            str(debugging_mode),
            "--parallel",
            str(parallel),
        ],
    )
    # Execute command
    _execute_cmd(cmd)


def run_l2e_station(
    # Station arguments
    data_source,
    campaign_name,
    station_name,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """
    Run the L2E processing of a station by invoking the disdrodb_run_l2e_station command in the terminal.

    Parameters
    ----------
    data_source : str
        The name of the data source.
    campaign_name : str
        The name of the campaign.
    station_name : str
        The name of the station.
    force : bool, optional
        If ``True``, overwrite existing data in destination directories.
        The default value is ``False``.
    verbose : bool, optional
        If ``True``, print detailed processing information to the terminal.
        The default value is ``False``.
    debugging_mode : bool, optional
        If ``True``, reduce the amount of data processed for debugging.
        The default value is ``False``.
    parallel : bool, optional
        If ``True``, process files in multiple processes simultaneously.
        The default value is ``True``.
    data_archive_dir
        The directory path where the local DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    metadata_archive_dir
        The directory path where the DISDRODB Metadata Archive is located.
        The directory path must end with ``<...>/DISDRODB-METADATA/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    # Define command
    cmd = " ".join(
        [
            "disdrodb_run_l2e_station",
            # Station arguments
            data_source,
            campaign_name,
            station_name,
            # DISDRODB root directories
            "--data_archive_dir",
            str(data_archive_dir),
            "--metadata_archive_dir",
            str(metadata_archive_dir),
            # Processing options
            "--force",
            str(force),
            "--verbose",
            str(verbose),
            "--debugging_mode",
            str(debugging_mode),
            "--parallel",
            str(parallel),
        ],
    )
    # Execute command
    _execute_cmd(cmd)


def run_l2m_station(
    # Station arguments
    data_source,
    campaign_name,
    station_name,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """
    Run the L2M processing of a station by invoking the disdrodb_run_l2m_station command in the terminal.

    Parameters
    ----------
    data_source : str
        The name of the data source.
    campaign_name : str
        The name of the campaign.
    station_name : str
        The name of the station.
    force : bool, optional
        If ``True``, overwrite existing data in destination directories.
        The default value is ``False``.
    verbose : bool, optional
        If ``True``, print detailed processing information to the terminal.
        The default value is ``False``.
    debugging_mode : bool, optional
        If ``True``, reduce the amount of data processed for debugging.
        The default value is ``False``.
    parallel : bool, optional
        If ``True``, process files in multiple processes simultaneously.
        The default value is ``True``.
    data_archive_dir
        The directory path where the local DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    metadata_archive_dir
        The directory path where the DISDRODB Metadata Archive is located.
        The directory path must end with ``<...>/DISDRODB-METADATA/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    # Define command
    cmd = " ".join(
        [
            "disdrodb_run_l2m_station",
            # Station arguments
            data_source,
            campaign_name,
            station_name,
            # DISDRODB root directories
            "--data_archive_dir",
            str(data_archive_dir),
            "--metadata_archive_dir",
            str(metadata_archive_dir),
            # Processing options
            "--force",
            str(force),
            "--verbose",
            str(verbose),
            "--debugging_mode",
            str(debugging_mode),
            "--parallel",
            str(parallel),
        ],
    )
    # Execute command
    _execute_cmd(cmd)


####--------------------------------------------------------------------------.
#### Run DISDRODB Archive Processing


def run_l0a(
    data_sources=None,
    campaign_names=None,
    station_names=None,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Run the L0A processing of DISDRODB stations.

    This function allows to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
    data_sources : list
        Name of data source(s) to process.
        The name(s) must be UPPER CASE.
        If campaign_names and station are not specified, process all stations.
        The default value is ``None``.
    campaign_names : list
        Name of the campaign(s) to process.
        The name(s) must be UPPER CASE.
        The default value is ``None``.
    station_names : list
        Station names to process.
        The default value is ``None``.
    force : bool
        If ``True``, overwrite existing data into destination directories.
        If ``False``, raise an error if there are already data into destination directories.
        The default value is ``False``.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default value is ``True``.
    parallel : bool
        If ``True``, the files are processed simultaneously in multiple processes.
        By default, the number of process is defined with ``os.cpu_count()``.
        If ``False``, the files are processed sequentially in a single process.
    debugging_mode : bool
        If ``True``, it processes just the first 3 raw data files.
        The default value is ``False``.
    data_archive_dir : str (optional)
        The directory path where the DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    metadata_archive_dir
        The directory path where the DISDRODB Metadata Archive is located.
        The directory path must end with ``<...>/DISDRODB-METADATA/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    # Define products
    product = "L0A"
    required_product = get_required_product(product)

    # Get list of available stations
    list_info = available_stations(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Stations arguments
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # Search options
        product=required_product,
        raise_error_if_empty=True,
    )

    # Print message
    n_stations = len(list_info)
    print(f"{product} processing of {n_stations} stations started.")

    # Loop over stations
    for data_source, campaign_name, station_name in list_info:
        print(f"{product} processing of {data_source} {campaign_name} {station_name} station started.")
        # Run processing
        run_l0a_station(
            # DISDRODB root directories
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            # Station arguments
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # Process options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )
        print(f"{product} processing of {data_source} {campaign_name} {station_name} station ended.")


def run_l0b(
    data_sources=None,
    campaign_names=None,
    station_names=None,
    # L0B processing options
    remove_l0a: bool = False,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Run the L0B processing of DISDRODB stations.

    This function allows to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB L0A stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
    data_sources : list
        Name of data source(s) to process.
        The name(s) must be UPPER CASE.
        If campaign_names and station are not specified, process all stations.
        The default value is ``None``.
    campaign_names : list
        Name of the campaign(s) to process.
        The name(s) must be UPPER CASE.
        The default value is ``None``.
    station_names : list
        Station names to process.
        The default value is ``None``.
    remove_l0a : bool
        Whether to keep the L0A files after having generated the L0B netCDF products.
        The default value is ``False``.
    force : bool
        If ``True``, overwrite existing data into destination directories.
        If ``False``, raise an error if there are already data into destination directories.
        The default value is ``False``.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default value is ``True``.
    parallel : bool
        If ``True``, the files are processed simultaneously in multiple processes.
        By default, the number of process is defined with ``os.cpu_count()``.
        If ``False``, the files are processed sequentially in a single process.
    debugging_mode : bool
        If ``True``, it reduces the amount of data to process.
        For L0B, it processes just the first 100 rows of 3 L0A files.
        The default value is ``False``.
    data_archive_dir : str (optional)
        The directory path where the DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    metadata_archive_dir
        The directory path where the DISDRODB Metadata Archive is located.
        The directory path must end with ``<...>/DISDRODB-METADATA/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    # Define products
    product = "L0B"
    required_product = get_required_product(product)

    # Get list of available stations
    list_info = available_stations(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Stations arguments
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # Search options
        product=required_product,
        raise_error_if_empty=True,
    )

    # Print message
    n_stations = len(list_info)
    print(f"{product} processing of {n_stations} stations started.")

    # Loop over stations
    for data_source, campaign_name, station_name in list_info:
        print(f"{product} processing of {data_source} {campaign_name} {station_name} station started.")
        # Run processing
        run_l0b_station(
            # DISDRODB root directories
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            # Station arguments
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # L0B options
            remove_l0a=remove_l0a,
            # Process options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )
        print(f"{product} processing of {data_source} {campaign_name} {station_name} station ended.")


def run_l0c(
    data_sources=None,
    campaign_names=None,
    station_names=None,
    # L0C options
    remove_l0b: bool = False,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Run the L0C processing of DISDRODB stations.

    This function allows to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
    data_sources : list
        Name of data source(s) to process.
        The name(s) must be UPPER CASE.
        If campaign_names and station are not specified, process all stations.
        The default value is ``None``.
    campaign_names : list
        Name of the campaign(s) to process.
        The name(s) must be UPPER CASE.
        The default value is ``None``.
    station_names : list
        Station names to process.
        The default value is ``None``.
    remove_l0b : bool
         Whether to remove the L0B files after having produced L0C netCDF files.
         The default value is ``False``.
    force : bool
        If ``True``, overwrite existing data into destination directories.
        If ``False``, raise an error if there are already data into destination directories.
        The default value is ``False``.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default value is ``False``.
    parallel : bool
        If ``True``, the files are processed simultaneously in multiple processes.
        Each process will use a single thread to avoid issues with the HDF/netCDF library.
        By default, the number of process is defined with ``os.cpu_count()``.
        If ``False``, the files are processed sequentially in a single process.
        If ``False``, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If ``True``, it reduces the amount of data to process.
        For L1B, it processes just 3 L0B files.
        The default value is ``False``.
    data_archive_dir : str (optional)
        The directory path where the DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    metadata_archive_dir
        The directory path where the DISDRODB Metadata Archive is located.
        The directory path must end with ``<...>/DISDRODB-METADATA/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    # Define products
    product = "L0C"
    required_product = get_required_product(product)

    # Get list of available stations
    list_info = available_stations(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Stations arguments
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # Search options
        product=required_product,
        raise_error_if_empty=True,
    )

    # Print message
    n_stations = len(list_info)
    print(f"{product} processing of {n_stations} stations started.")

    # Loop over stations
    for data_source, campaign_name, station_name in list_info:
        print(f"{product} processing of {data_source} {campaign_name} {station_name} station started.")
        # Run processing
        run_l0c_station(
            # DISDRODB root directories
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            # Station arguments
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # L0C options
            remove_l0b=remove_l0b,
            # Process options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )
        print(f"{product} processing of {data_source} {campaign_name} {station_name} station ended.")


def run_l0(
    data_sources=None,
    campaign_names=None,
    station_names=None,
    # L0 archive options
    l0a_processing: bool = True,
    l0b_processing: bool = True,
    l0c_processing: bool = True,
    remove_l0a: bool = False,
    remove_l0b: bool = False,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Run the L0 processing of DISDRODB stations.

    This function allows to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
    data_sources : list
        Name of data source(s) to process.
        The name(s) must be UPPER CASE.
        If campaign_names and station are not specified, process all stations.
        The default value is ``None``.
    campaign_names : list
        Name of the campaign(s) to process.
        The name(s) must be UPPER CASE.
        The default value is ``None``.
    station_names : list
        Station names to process.
        The default value is ``None``.
    l0a_processing : bool
        Whether to launch processing to generate L0A Apache Parquet file(s) from raw data.
        The default value is ``True``.
    l0b_processing : bool
        Whether to launch processing to generate L0B netCDF4 file(s) from L0A data.
        The default value is ``True``.
    l0c_processing : bool
        Whether to launch processing to generate L0C netCDF4 file(s) from L0B data.
        The default value is ``True``.
    remove_l0a : bool
        Whether to keep the L0A files after having generated the L0B netCDF products.
        The default value is ``False``.
    remove_l0b : bool
        Whether to remove the L0B files after having produced all L0C netCDF files.
        The default value is ``False``.
    force : bool
        If ``True``, overwrite existing data into destination directories.
        If ``False``, raise an error if there are already data into destination directories.
        The default value is ``False``.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default value is ``False``.
    parallel : bool
        If ``True``, the files are processed simultaneously in multiple processes.
        Each process will use a single thread to avoid issues with the HDF/netCDF library.
        By default, the number of process is defined with ``os.cpu_count()``.
        If ``False``, the files are processed sequentially in a single process.
        If ``False``, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If ``True``, it reduces the amount of data to process.
        For L0A, it processes just the first 3 raw data files.
        For L0B, it processes just the first 100 rows of 3 L0A files.
        The default value is ``False``.
    data_archive_dir : str (optional)
        The directory path where the DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    metadata_archive_dir
        The directory path where the DISDRODB Metadata Archive is located.
        The directory path must end with ``<...>/DISDRODB-METADATA/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    # Define starting product
    if l0c_processing:
        required_product = get_required_product("L0C")
    if l0b_processing:
        required_product = get_required_product("L0B")
    if l0a_processing:
        required_product = get_required_product("L0A")

    # Get list of available stations
    list_info = available_stations(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Stations arguments
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # Search options
        product=required_product,
        raise_error_if_empty=True,
    )

    # Print message
    n_stations = len(list_info)
    print(f"L0 processing of {n_stations} stations started.")

    # Loop over stations
    for data_source, campaign_name, station_name in list_info:
        print(f"L0 processing of {data_source} {campaign_name} {station_name} station started.")
        # Run processing
        run_l0_station(
            # DISDRODB root directories
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            # Station arguments
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # L0 archive options
            l0a_processing=l0a_processing,
            l0b_processing=l0b_processing,
            l0c_processing=l0c_processing,
            remove_l0a=remove_l0a,
            remove_l0b=remove_l0b,
            # Process options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )
        print(f"L0 processing of {data_source} {campaign_name} {station_name} station ended.")


def run_l1(
    data_sources=None,
    campaign_names=None,
    station_names=None,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Run the L1 processing of DISDRODB stations.

    This function allows to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
    data_sources : list
        Name of data source(s) to process.
        The name(s) must be UPPER CASE.
        If campaign_names and station are not specified, process all stations.
        The default value is ``None``.
    campaign_names : list
        Name of the campaign(s) to process.
        The name(s) must be UPPER CASE.
        The default value is ``None``.
    station_names : list
        Station names to process.
        The default value is ``None``.
    force : bool
        If ``True``, overwrite existing data into destination directories.
        If ``False``, raise an error if there are already data into destination directories.
        The default value is ``False``.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default value is ``False``.
    parallel : bool
        If ``True``, the files are processed simultaneously in multiple processes.
        Each process will use a single thread to avoid issues with the HDF/netCDF library.
        By default, the number of process is defined with ``os.cpu_count()``.
        If ``False``, the files are processed sequentially in a single process.
        If ``False``, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If ``True``, it reduces the amount of data to process.
        For L1B, it processes just 3 L0B files.
        The default value is ``False``.
    data_archive_dir : str (optional)
        The directory path where the DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    metadata_archive_dir
        The directory path where the DISDRODB Metadata Archive is located.
        The directory path must end with ``<...>/DISDRODB-METADATA/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    product = "L1"
    required_product = get_required_product(product)

    # Get list of available stations
    list_info = available_stations(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Stations arguments
        product=required_product,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # Search options
        available_data=False,  # Check for station product directory is present only
        raise_error_if_empty=True,
    )

    # Print message
    n_stations = len(list_info)
    print(f"{product} processing of {n_stations} stations started.")

    # Loop over stations
    for data_source, campaign_name, station_name in list_info:
        print(f"{product} processing of {data_source} {campaign_name} {station_name} station started.")
        # Run processing
        run_l1_station(
            # DISDRODB root directories
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            # Station arguments
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # Process options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )
        print(f"{product} processing of {data_source} {campaign_name} {station_name} station ended.")


def run_l2e(
    data_sources=None,
    campaign_names=None,
    station_names=None,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Run the L2E processing of DISDRODB stations.

    This function allows to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
    data_sources : list
        Name of data source(s) to process.
        The name(s) must be UPPER CASE.
        If campaign_names and station are not specified, process all stations.
        The default value is ``None``.
    campaign_names : list
        Name of the campaign(s) to process.
        The name(s) must be UPPER CASE.
        The default value is ``None``.
    station_names : list
        Station names to process.
        The default value is ``None``.
    force : bool
        If ``True``, overwrite existing data into destination directories.
        If ``False``, raise an error if there are already data into destination directories.
        The default value is ``False``.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default value is ``False``.
    parallel : bool
        If ``True``, the files are processed simultaneously in multiple processes.
        Each process will use a single thread to avoid issues with the HDF/netCDF library.
        By default, the number of process is defined with ``os.cpu_count()``.
        If ``False``, the files are processed sequentially in a single process.
        If ``False``, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If ``True``, it reduces the amount of data to process.
        For L2E, it processes just 3 L1 files.
        The default value is ``False``.
    data_archive_dir : str (optional)
        The directory path where the DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    metadata_archive_dir
        The directory path where the DISDRODB Metadata Archive is located.
        The directory path must end with ``<...>/DISDRODB-METADATA/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    product = "L2E"
    required_product = get_required_product(product)

    # Get list of available stations
    list_info = available_stations(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Stations arguments
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # Search options
        product=required_product,
        raise_error_if_empty=True,
    )

    # Print message
    n_stations = len(list_info)
    print(f"{product} processing of {n_stations} stations started.")

    # Loop over stations
    for data_source, campaign_name, station_name in list_info:
        print(f"{product} processing of {data_source} {campaign_name} {station_name} station started.")
        # Run processing
        run_l2e_station(
            # DISDRODB root directories
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            # Station arguments
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # Process options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )
        print(f"{product} processing of {data_source} {campaign_name} {station_name} station ended.")


def run_l2m(
    data_sources=None,
    campaign_names=None,
    station_names=None,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Run the L2M processing of DISDRODB stations.

    This function allows to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
    data_sources : list
        Name of data source(s) to process.
        The name(s) must be UPPER CASE.
        If campaign_names and station are not specified, process all stations.
        The default value is ``None``.
    campaign_names : list
        Name of the campaign(s) to process.
        The name(s) must be UPPER CASE.
        The default value is ``None``.
    station_names : list
        Station names to process.
        The default value is ``None``.
    force : bool
        If ``True``, overwrite existing data into destination directories.
        If ``False``, raise an error if there are already data into destination directories.
        The default value is ``False``.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default value is ``False``.
    parallel : bool
        If ``True``, the files are processed simultaneously in multiple processes.
        Each process will use a single thread to avoid issues with the HDF/netCDF library.
        By default, the number of process is defined with ``os.cpu_count()``.
        If ``False``, the files are processed sequentially in a single process.
        If ``False``, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If ``True``, it reduces the amount of data to process.
        For L2MB, it processes just 3 L0B files.
        The default value is ``False``.
    data_archive_dir : str (optional)
        The directory path where the DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.
    metadata_archive_dir
        The directory path where the DISDRODB Metadata Archive is located.
        The directory path must end with ``<...>/DISDRODB-METADATA/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    product = "L2M"
    required_product = get_required_product(product)

    # Get list of available stations
    list_info = available_stations(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Stations arguments
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # Search options
        product=required_product,
        raise_error_if_empty=True,
    )

    # Print message
    n_stations = len(list_info)
    print(f"{product} processing of {n_stations} stations started.")

    # Loop over stations
    for data_source, campaign_name, station_name in list_info:
        print(f"{product} processing of {data_source} {campaign_name} {station_name} station started.")
        # Run processing
        run_l2m_station(
            # DISDRODB root directories
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            # Station arguments
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # Process options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )
        print(f"{product} processing of {data_source} {campaign_name} {station_name} station ended.")


####--------------------------------------------------------------------------.
