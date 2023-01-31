import os
import glob
import logging
import xarray as xr
from disdrodb.L0.io import _check_directory_exist
from disdrodb.L0.io import get_L0B_dir, get_L0B_fpath
from disdrodb.L0.utils_nc import xr_concat_datasets
from disdrodb.utils.logger import (
    create_file_logger,
    close_logger,
    log_info,
    log_warning,
    # log_debug,
    log_error,
)

logger = logging.getLogger(__name__)


def _concatenate_L0B_files(processed_dir, station_id, remove=False, verbose=False):
    """Concatenate all L0B netCDF files into a single netCDF file.

    The single netCDF file is saved at <processed_dir>/L0B.
    """
    from disdrodb.L0.L0B_processing import write_L0B

    # Create logger
    filename = f"concatenatation_{station_id}"
    logger = create_file_logger(
        processed_dir=processed_dir,
        product_level="L0B",
        station_id="",  # locate outside the station directory
        filename=filename,
        parallel=False,
    )

    # -------------------------------------------------------------------------.
    # Retrieve L0B files
    L0B_dir_path = get_L0B_dir(processed_dir, station_id)
    file_list = sorted(glob.glob(os.path.join(L0B_dir_path, "*.nc")))

    # -------------------------------------------------------------------------.
    # Check there are at least two files
    n_files = len(file_list)
    if n_files == 0:
        msg = f"No L0B file is available for concatenation in {L0B_dir_path}."
        log_error(logger=logger, msg=msg, verbose=verbose)
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
    single_nc_fpath = get_L0B_fpath(ds, processed_dir, station_id, single_netcdf=True)
    write_L0B(ds, fpath=single_nc_fpath)

    # -------------------------------------------------------------------------.
    # Close file and delete
    ds.close()
    del ds

    # -------------------------------------------------------------------------.
    # If remove = True, remove all the single files
    if remove:
        log_info(
            logger=logger, msg="Removal of single L0B files started.", verbose=verbose
        )
        _ = [os.remove(fpath) for fpath in file_list]
        log_info(
            logger=logger, msg="Removal of single L0B files ended.", verbose=verbose
        )

    # -------------------------------------------------------------------------.
    # Close the file logger
    close_logger(logger)

    # Return the dataset
    return None


def _concatenate_L0B_station(
    disdrodb_dir, data_source, campaign_name, station, remove=False, verbose=False
):
    """This function concatenate the L0B files of a single DISDRODB station.

    This function is called to run the command:  run_disdrodb_l0b_concat_station
    """
    # Retrieve processed_dir
    processed_dir = os.path.join(disdrodb_dir, "Processed", data_source, campaign_name)
    _check_directory_exist(processed_dir)
    # Run concatenation
    _concatenate_L0B_files(
        processed_dir=processed_dir, station_id=station, remove=remove, verbose=verbose
    )


####---------------------------------------------------------------------------.
#### Wrappers of run_disdrodb_l0b_concat_station call


def _execute_cmd(cmd):
    """Execute command in the terminal, streaming output in python console."""
    from subprocess import Popen, PIPE, CalledProcessError

    with Popen(cmd, shell=True, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end="")  # process line here

    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)


def concatenate_L0B_station(
    disdrodb_dir, data_source, campaign_name, station, remove=False, verbose=False
):
    """Concatenate the L0B files of a single DISDRODB station.

    This function calls run_disdrodb_l0b_concat_station in the terminal.
    """
    cmd = " ".join(
        [
            "run_disdrodb_l0b_concat_station",
            disdrodb_dir,
            data_source,
            campaign_name,
            station,
            "--remove",
            str(remove),
            "--verbose",
            str(verbose),
        ]
    )
    _execute_cmd(cmd)


def concatenate_L0B(
    disdrodb_dir,
    data_sources=None,
    campaign_names=None,
    station=None,
    remove=False,
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
    if station is not None:
        list_info = [info for info in list_info if info[2] in station]
        # If nothing left, raise an error
        if len(list_info) == 0:
            raise ValueError(
                "No stations to concatenate given the provided `station` argument!"
            )

    # Print message
    n_stations = len(list_info)
    print(f"Concatenation of {n_stations} L0B stations started.")

    # Start the loop to launch the concatenation of each station
    for data_source, campaign_name, station in list_info:
        concatenate_L0B_station(
            disdrodb_dir=disdrodb_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station=station,
            remove=remove,
            verbose=verbose,
        )

    print(f"Concatenation of {n_stations} L0B stations ended.")


####--------------------------------------------------------------------------.


def concatenate_L0B_files(processed_dir, station_id, remove=False, verbose=False):
    """Concatenate the L0B files of a single DISDRODB station.

    This function calls run_disdrodb_l0b_concat_station in the terminal.
    It is used by L0_processing.run_L0 functtion if single_netcf=True.
    """
    from disdrodb.L0.io import get_data_source, get_campaign_name, get_disdrodb_dir

    disdrodb_dir = get_disdrodb_dir(processed_dir)
    data_source = get_data_source(processed_dir)
    campaign_name = get_campaign_name(processed_dir)
    concatenate_L0B_station(
        disdrodb_dir=disdrodb_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station=station_id,
        remove=remove,
        verbose=verbose,
    )
