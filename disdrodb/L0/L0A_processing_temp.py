#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:16:16 2023

@author: ghiggi
"""
# - specify only arguments: data_source, campaign_name, station_ids
# --> reader_name is taken from the yaml file !

# - ask for disdrodb_base_dir
# -->  instead of raw/processed_dir

# -----------------------------------------------------------------------------.
# TODO generate_l0_archive
# - search for yml file, search for stations with data, processes it
# - add station_ids --> reader, run_l0


# args
# def _run_disdrodb_l0a_station(disdrodb_dir, data_source, campaign_name, station):

#     run_reader(
#         data_source: str,
#         reader_name: str,
#         raw_dir: str,
#         processed_dir: str,
#         station_ids: list = None,
#         l0a_processing: bool = True,
#         l0b_processing: bool = True,
#         keep_l0a: bool = False,
#         force: bool = False,
#         verbose: bool = False,
#         debugging_mode: bool = False,
#         parallel: bool = True,
#         single_netcdf: bool = True,
#     )


disdrodb_dir = "/home/ghiggi/DISDRODB/"
disdrodb_dir = "/tmp/data/DISDRODB/"

# data_sources = "EPFL"
campaign_names = "LOCARNO_2018"

list_info = available_stations(
    disdrodb_dir,
    product_level=product_level,
    data_sources=data_sources,
    campaign_names=campaign_names,
)
# Start the loop to launch the concatenation of each station
data_source, campaign_name, station = list_info[0]


# L0A requires raw dir
# L0B requires only processed dir

# disdrodb.L0.io currently expect processing of full archive
# --> station_ids
# --> be selective to station !!!


# Check directories and create directory structure
# TODO: flexible for station, type of processing

raw_dir, processed_dir = check_directories(raw_dir, processed_dir, force=force)
create_directory_structure(raw_dir, processed_dir)

# ----------------------------------------------------------------------------.


def check_reader_arguments(reader):
    # TODO: check reader arguments !!!!
    # --> This should be also called for all readers in the CI
    pass


def _get_reader(disdrodb_dir, product_level, data_source, campaign_name, station):
    """Retrieve reader form station metadata information."""
    from disdrodb.L0.L0_reader import get_reader
    from disdrodb.api.io import get_metadata_dict

    # Get metadata
    metadata = get_metadata_dict(
        disdrodb_dir=disdrodb_dir,
        product_level=product_level,
        data_source=data_source,
        campaign_name=campaign_name,
        station=station,
    )
    # Check reader key is within the dictionary
    if "reader" not in metadata:
        raise ValueError(
            f"The `reader` key is not available in the metadata of the {data_source} {campaign_name} {station} station."
        )

    # Retrieve reader
    # TODO: DEFINE CONVENTION !!!
    # reader: <data_source/reader_name> in disdrodb.L0.readers
    reader_data_source_name = metadata.get("reader")
    reader_data_source = reader_data_source_name.split("/")[0]
    reader_name = reader_data_source_name.split("/")[1]

    reader = get_reader(reader_data_source, reader_name)

    # Check reader argument
    check_reader_arguments(reader)

    return reader


def run_disdrodb_l0(
    disdrodb_dir,
    data_sources=None,
    campaign_names=None,
    station=None,
    l0a_processing: bool = True,
    l0b_processing: bool = True,
    keep_l0a: bool = False,
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    single_netcdf: bool = True,
):
    from disdrodb.api.io import available_stations

    if l0a_processing:
        product_level = "RAW"
    elif l0b_processing:
        product_level = "L0A"
    else:
        # TODO: potentially can be used to run single_netcdf
        raise ValueError("At least l0a_processing or l0b_processing must be True.")

    # Get list of available stations
    list_info = available_stations(
        disdrodb_dir,
        product_level=product_level,
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
    print(f"L0 processing of {n_stations} stations started.")

    for data_source, campaign_name, station in list_info:
        print(
            f"L0 processing of {data_source} {campaign_name} {station} station started."
        )
        # Get reader
        reader = _get_reader(
            disdrodb_dir=disdrodb_dir,
            product_level=product_level,
            data_source=data_source,
            campaign_name=campaign_name,
            station=station,
        )
        # Run processing
        # TODO: run_disdrodb_l0_station
        reader(
            raw_dir,
            processed_dir,
            station_ids=station,  # TODO: REQUIRE MODIFICATION OF DISDRODB.L0.io
            # -----------------------------------.
            l0a_processing=l0a_processing,
            l0b_processing=l0b_processing,
            keep_l0a=keep_l0a,
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            bool=False,
            parallel=parallel,
            single_netcdf=single_netcdf,
        )
        print(
            f"L0 processing of {data_source} {campaign_name} {station} station ended."
        )


####---------------------------------------------------------------------------.
#### Wrappers to run L0 processing


def run_disdrodb_l0a(
    disdrodb_dir,
    data_sources=None,
    campaign_names=None,
    station=None,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
):

    run_disdrodb_l0(
        disdrodb_dir=disdrodb_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station=station,
        # L0A settings
        l0a_processing=True,
        # L0B settings
        l0b_processing=False,
        keep_l0a=True,
        single_netcdf=False,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )


def run_disdrodb_l0b(
    disdrodb_dir,
    data_sources=None,
    campaign_names=None,
    station=None,
    # L0B settings
    keep_l0a: bool = True,
    single_netcdf: bool = False,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
):

    run_disdrodb_l0(
        disdrodb_dir=disdrodb_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station=station,
        # L0A settings
        l0a_processing=False,
        # L0B settings
        l0b_processing=True,
        keep_l0a=keep_l0a,
        single_netcdf=single_netcdf,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )
