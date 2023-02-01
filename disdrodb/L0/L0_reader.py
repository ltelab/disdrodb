#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:50:39 2023

@author: ghiggi
"""
import os
import logging

logger = logging.getLogger(__name__)


def get_available_readers() -> dict:
    """Returns the readers description included into the current release of DISDRODB.

    Returns
    -------
    dict
        The dictionary has the following schema {"data_source": {"campaign_name": "reader file path"}}
    """
    # Current file path
    lo_folder_path = os.path.dirname(__file__)

    # Readers folder path
    reader_folder_path = os.path.join(lo_folder_path, "readers")

    # List of readers folder
    list_of_reader_folder = [
        f.path for f in os.scandir(reader_folder_path) if f.is_dir()
    ]

    # Create dictionary
    dict_reader = {}
    for path_folder in list_of_reader_folder:
        data_source = os.path.basename(path_folder)
        dict_reader[data_source] = {}
        for path_python_file in [
            f.path
            for f in os.scandir(path_folder)
            if f.is_file() and f.path.endswith(".py")
        ]:
            reader_name = (
                os.path.basename(path_python_file)
                .replace("reader_", "")
                .replace(".py", "")
            )
            dict_reader[data_source][reader_name] = path_python_file

    return dict_reader


def check_data_source(data_source: str) -> str:
    """Check if the provided data source exists within the available readers.

    Please run get_available_readers() to get the list of all available reader.

    Parameters
    ----------
    data_source : str
        Data source name  - Institution name (when campaign data spans more than 1 country) or country (when all campaigns (or sensor networks) are inside a given country)

    Returns
    -------
    str
        if data source exists : return the correct data source name
        if data source does not exist : error

    Raises
    ------
    ValueError
        Error if the data source name provided has not been found.
    """

    dict_all_readers = get_available_readers()

    correct_data_source_list = list(
        set(dict_all_readers.keys()).intersection([data_source, data_source.upper()])
    )

    if correct_data_source_list:
        correct_data_source = correct_data_source_list[0]
    else:
        msg = f"Data source {data_source} has not been found within the available readers."
        logger.exception(msg)
        raise ValueError(msg)

    return correct_data_source


def get_available_readers_by_data_source(data_source: str) -> dict:
    """Return the available readers by data source.

    Parameters
    ----------
    data_source : str
        Data source name - Institution name (when campaign data spans more than 1 country) or country (when all campaigns (or sensor networks) are inside a given country)

    Returns
    -------
    dict
        Dictionary that conatins the campaigns for the requested data source.

    """
    correct_data_source = check_data_source(data_source)

    if correct_data_source:
        dict_data_source = get_available_readers().get(correct_data_source)

    return dict_data_source


def check_reader_name(data_source: str, reader_name: str) -> str:
    """Check if the provided data source exists and reader names exists within the available readers.

    Please run get_available_readers() to get the list of all available reader.

    Parameters
    ----------
    data_source : str
        Data source name - Institution name (when campaign data spans more than 1 country) or country (when all campaigns (or sensor networks) are inside a given country)
    reader_name : str
        Campaign name

    Returns
    -------
    str
        If True : returns the reader name
        If False : Error - return None

    Raises
    ------
    ValueError
        Error if the reader name provided for the campaign has not been found.
    """

    correct_data_source = check_data_source(data_source)

    if correct_data_source:
        dict_reader_names = get_available_readers_by_data_source(correct_data_source)

        correct_reader_name_list = list(
            set(dict_reader_names.keys()).intersection(
                [reader_name, reader_name.upper()]
            )
        )

        if correct_reader_name_list:
            correct_reader_name = correct_reader_name_list[0]
        else:
            msg = (
                f"Reader {reader_name} has not been found within the available readers"
            )
            logger.exception(msg)
            raise ValueError(msg)

    return correct_reader_name


def get_reader(data_source: str, reader_name: str) -> object:
    """Returns the reader function based on input parameters.

    Parameters
    ----------
    data_source : str
        Institution name (when campaign data spans more than 1 country) or country (when all campaigns (or sensor networks) are inside a given country)
    reader_name : str
        Campaign name

    Returns
    -------
    object
        The reader() function

    """

    correct_data_source = check_data_source(data_source)
    correct_reader_name = check_reader_name(data_source, reader_name)

    if correct_reader_name:
        full_name = (
            f"disdrodb.L0.readers.{correct_data_source}.{correct_reader_name}.reader"
        )
        module_name, unit_name = full_name.rsplit(".", 1)
        my_reader = getattr(__import__(module_name, fromlist=[""]), unit_name)

    return my_reader


def check_reader_arguments(reader):
    # TODO: check reader arguments !!!!
    # --> This should be also called for all readers in the CI
    pass


# TODO: rename as get_reader after refactoring above
def _get_new_reader(disdrodb_dir, data_source, campaign_name, station_name):
    """Retrieve reader form station metadata information."""
    from disdrodb.L0.L0_reader import get_reader
    from disdrodb.api.io import get_metadata_dict

    # Get metadata
    metadata = get_metadata_dict(
        disdrodb_dir=disdrodb_dir,
        product_level="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Check reader key is within the dictionary
    if "reader" not in metadata:
        raise ValueError(
            f"The `reader` key is not available in the metadata of the {data_source} {campaign_name} {station_name} station."
        )

    # ------------------------------------------------------------------------.
    # Retrieve reader
    # - Convention: reader: <data_source/reader_name> in disdrodb.L0.readers
    reader_data_source_name = metadata.get("reader")
    reader_data_source = reader_data_source_name.split("/")[0]
    reader_name = reader_data_source_name.split("/")[1]

    reader = get_reader(
        reader_data_source, reader_name
    )  # TODO: redefine get_reader for new pattern

    # ------------------------------------------------------------------------.
    # Check reader argument
    check_reader_arguments(reader)

    return reader


####--------------------------------------------------------------------------.
#### Readers Docs


def is_documented_by(original):
    """Wrapper function to apply generic docstring to the decorated function.

    Parameters
    ----------
    original : function
        Function to take the docstring from.
    """

    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


def reader_generic_docstring():
    """Script to convert the raw data to L0A format.

    Parameters
    ----------
    raw_dir : str
        The directory path where all the raw content of a specific campaign is stored.
        The path must have the following structure:
            <...>/DISDRODB/Raw/<data_source>/<campaign_name>'.
        Inside the raw_dir directory, it is required to adopt the following structure:
        - /data/<station_name>/<raw_files>
        - /metadata/<station_name>.yaml
        Important points:
        - For each <station_name> there must be a corresponding YAML file in the metadata subfolder.
        - The <campaign_name> must semantically match between:
           - the raw_dir and processed_dir directory paths;
           - with the key 'campaign_name' within the metadata YAML files.
        - The campaign_name are expected to be UPPER CASE.
    processed_dir : str
        The desired directory path for the processed DISDRODB L0A and L0B products.
        The path should have the following structure:
            <...>/DISDRODB/Processed/<data_source>/<campaign_name>'
        For testing purpose, this function exceptionally accept also a directory path simply ending
        with <campaign_name> (i.e. /tmp/<campaign_name>).
    station_name : str
        Station name
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is True.
    parallel : bool
        If True, the files are processed simultanously in multiple processes.
        The number of simultaneous processes can be customized using the dask.distributed LocalCluster.
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        It processes just the first 3 raw data files.
        The default is False.
    """
