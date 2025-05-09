#!/usr/bin/env python3

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
"""Define DISDRODB L0 readers routines."""
import inspect
import logging
import os
from collections import defaultdict

from disdrodb.api.checks import check_data_sources, check_sensor_name
from disdrodb.utils.directories import list_files
from disdrodb.utils.list import flatten_list

logger = logging.getLogger(__name__)


####--------------------------------------------------------------------------.
#### Search readers


def define_readers_directory(sensor_name="") -> str:
    """Returns the path to the ``disdrodb.l0.readers`` directory within the disdrodb package."""
    from disdrodb import __root_path__

    reader_dir = os.path.join(__root_path__, "disdrodb", "l0", "readers", sensor_name)
    return reader_dir


def define_reader_path(sensor_name, reader_reference):
    """Define the reader path based on the reader reference name."""
    # Retrieve path to directory with sensor readers
    reader_dir = define_readers_directory(sensor_name)
    # Define reader path
    reader_path = os.path.join(reader_dir, *reader_reference.split("/")) + ".py"
    return reader_path


def list_readers_paths(sensor_name) -> list:
    """Returns the file paths of the available readers for a given sensor in ``disdrodb.l0.readers.{sensor_name}``."""
    # Retrieve path to directory with sensor readers
    reader_dir = define_readers_directory(sensor_name)
    # List readers
    readers_paths = list_files(reader_dir, glob_pattern="*.py", recursive=True)
    return readers_paths


def list_readers_references(sensor_name):
    """Returns the readers references available for a given sensor in ``disdrodb.l0.readers.{sensor_name}``."""
    # Retrieve path to directory with sensor readers
    reader_dir = define_readers_directory(sensor_name)
    # List readers paths
    readers_paths = list_readers_paths(sensor_name)
    # Derive readers references
    readers_references = [
        path.replace(reader_dir, "").lstrip(os.path.sep).rstrip(".py").replace("\\", "/") for path in readers_paths
    ]
    return readers_references


def get_specific_readers_references(sensor_name):
    """Returns a dictionary with the readers references available for each data source."""
    # List reader references
    readers_references = list_readers_references(sensor_name)
    # Group reader by data source
    # - Discard generic readers references
    specific_reader_references = [
        reader_reference.split("/") for reader_reference in readers_references if len(reader_reference.split("/")) == 2
    ]
    data_sources_readers_dict = defaultdict(list)
    for data_source, reader_name in specific_reader_references:
        data_sources_readers_dict[data_source].append(f"{data_source}/{reader_name}")
    data_sources_readers_dict = dict(data_sources_readers_dict)
    return data_sources_readers_dict


def get_specific_readers_path(sensor_name):
    """Returns a dictionary with the file paths of the available readers for each data source."""
    data_sources_readers_dict = get_specific_readers_references(sensor_name)
    data_sources_readers_dict = {
        data_source: [
            define_reader_path(sensor_name=sensor_name, reader_reference=reader_reference)
            for reader_reference in readers_references
        ]
        for data_source, readers_references in data_sources_readers_dict.items()
    }
    return data_sources_readers_dict


def available_readers(sensor_name, data_sources=None, return_path=False):
    """Retrieve available readers information."""
    check_sensor_name(sensor_name)
    # Return all available readers for a specific sensor_name
    if data_sources is None and not return_path:
        return list_readers_references(sensor_name)
    if data_sources is None and return_path:
        return list_readers_paths(sensor_name)
    # Return all available readers for a specific sensor_name and set of data sources
    data_sources = check_data_sources(data_sources)
    if return_path:
        dict_readers_paths = get_specific_readers_path(sensor_name)
        dict_readers_paths = {data_source: dict_readers_paths[data_source] for data_source in data_sources}
        return flatten_list(list(dict_readers_paths.values()))
    # Return dictionary of paths otherwise
    dict_readers_references = get_specific_readers_references(sensor_name)
    dict_readers_references = {data_source: dict_readers_references[data_source] for data_source in data_sources}
    return flatten_list(list(dict_readers_references.values()))


####--------------------------------------------------------------------------.
#### Reader Function Checks


def check_reader_reference(reader_reference):
    """Check the reader_reference value."""
    if isinstance(reader_reference, type(None)):
        raise TypeError("`reader_reference` is None. Specify the reader reference name !")
    if not isinstance(reader_reference, str):
        raise TypeError(f"`reader_reference` must be a string. Got type {type(reader_reference)}.")
    if reader_reference == "":
        raise ValueError("`reader_reference` is an empty string. Specify the reader reference name !")
    if len(reader_reference.split("/")) > 2:
        raise ValueError("`reader_reference` expects to be composed by maximum one `/` (<DATA_SOURCE>/<CUSTOM_NAME>).")
    return reader_reference


def check_reader_exists(reader_reference, sensor_name):
    """Check the reader exists."""
    valid_readers_references = available_readers(sensor_name)
    if reader_reference not in valid_readers_references:
        msg = (
            f"{sensor_name} reader '{reader_reference}' does not exists. Valid readers are {valid_readers_references}."
        )
        raise ValueError(msg)


def check_reader_arguments(reader):
    """Check the reader function have the expected input arguments."""
    expected_arguments = ["filepath", "logger"]
    signature = inspect.signature(reader)
    reader_arguments = sorted(signature.parameters.keys())
    if reader_arguments != expected_arguments:
        raise ValueError(f"The reader must be defined with the following arguments: {expected_arguments}")
    # Verify 'logger' default
    logger_param = signature.parameters.get("logger")
    if logger_param.default is inspect._empty:
        raise ValueError(
            "The 'logger' argument must have a default value (None).",
        )
    if logger_param.default is not None:
        raise ValueError(
            f"The default value for 'logger' must be None, got {logger_param.default!r}.",
        )


def check_metadata_reader(metadata):
    """Check the metadata ``reader`` key is available and points to an existing disdrodb reader."""
    data_source = metadata.get("data_source", "")
    campaign_name = metadata.get("campaign_name", "")
    station_name = metadata.get("station_name", "")
    # Check the reader is specified
    if "reader" not in metadata:
        raise ValueError(
            "The `reader` key is not specified in the metadata of the"
            f" {data_source} {campaign_name} {station_name} station.",
        )
    if "sensor_name" not in metadata:
        raise ValueError(
            "The `sensor_name` is not specified in the metadata of the"
            f" {data_source} {campaign_name} {station_name} station.",
        )
    # If the reader name is specified, test it is valid.
    # --> Reader location: disdrodb.l0.readers.{sensor_name}.{reader_reference}
    # --> reader_reference typically defined as "{DATA_SOURCE}"/"{CAMPAIGN_NAME}_{OPTIONAL_SUFFIX}"
    reader_reference = metadata["reader"]
    sensor_name = metadata["sensor_name"]
    _ = get_reader(reader_reference, sensor_name=sensor_name)


def check_software_readers():
    """Check the validity of all readers included in disdrodb software ."""
    import disdrodb

    sensors_names = disdrodb.available_sensor_names()
    for sensor_name in sensors_names:
        readers_references = available_readers(sensor_name=sensor_name, return_path=False)
        for reader_reference in readers_references:
            try:
                _ = get_reader(reader_reference=reader_reference, sensor_name=sensor_name)
            except Exception as e:
                raise ValueError(f"Invalid {sensor_name} {reader_reference}.py reader: {e}")


####--------------------------------------------------------------------------.
#### Reader Retrieval


def get_reader(reader_reference, sensor_name):
    """Retrieve the reader function.

    Parameters
    ----------
    reader_reference : str
        The reader reference name.
        The reader is located at ``disdrodb.l0.readers.{sensor_name}.{reader_reference}``.
        The reader_reference naming convention is ``"{DATA_SOURCE}"/"{CAMPAIGN_NAME}_{OPTIONAL_SUFFIX}"``.
    sensor_name : str
        The sensor name.

    Returns
    -------
    callable
        The ``reader()`` function.

    """
    # Check reader reference value
    reader_reference = check_reader_reference(reader_reference)

    # Check reader exists
    check_reader_exists(reader_reference=reader_reference, sensor_name=sensor_name)

    # Replace "/" with "." to define reader reference path
    reader_reference = reader_reference.replace("/", ".")

    # Import reader function
    # --> This will not raise error if check_reader_exists pass !
    full_name = f"disdrodb.l0.readers.{sensor_name}.{reader_reference}.reader"
    module_name, unit_name = full_name.rsplit(".", 1)
    reader = getattr(__import__(module_name, fromlist=[""]), unit_name)

    # Check reader function validity
    check_reader_arguments(reader)

    # Return readere function
    return reader


def get_station_reader(data_source, campaign_name, station_name, metadata_archive_dir=None):
    """Retrieve the reader function of a specific DISDRODB station."""
    from disdrodb.metadata import read_station_metadata

    # Get metadata
    metadata = read_station_metadata(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Retrieve reader function using metadata information
    reader = get_reader_from_metadata(metadata)

    # Return the reader function
    return reader


def get_reader_from_metadata(metadata):
    """Retrieve the reader function based on the metadata information.

    The reader_reference naming convention is ``"{DATA_SOURCE}"/"{CAMPAIGN_NAME}_{OPTIONAL_SUFFIX}"``.
    The reader is located at ``disdrodb.l0.readers.{sensor_name}.{reader_reference}``.
    """
    # Check validity of metadata reader key
    check_metadata_reader(metadata)

    # Extract reader information from metadata
    reader_reference = metadata.get("reader")
    sensor_name = metadata.get("sensor_name")

    # Retrieve reader function
    reader = get_reader(reader_reference=reader_reference, sensor_name=sensor_name)
    return reader


####--------------------------------------------------------------------------.
#### Readers Docstring


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
    """Reader to convert a raw data file to DISDRODB L0A or L0B format.

    Raw text files are read and converted to a ``pandas.DataFrame`` (L0A format).
    Raw netCDF files are read and converted to a ``xarray.Dataset`` (L0B format).

    Parameters
    ----------
    filepath : str
        Filepath of the raw data file to be processed.
    logger: logging.Logger, optional
        Logger to use for logging messages.
        Default is ``None``, which means no logger is used.
    """
