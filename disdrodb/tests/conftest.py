#!/usr/bin/env python3
"""Shared preparation and utilities for testing.

This module is executed automatically by pytest.

"""
import os
import shutil

import pytest

from disdrodb import __root_path__
from disdrodb.utils.yaml import write_yaml


def create_fake_metadata_directory(base_dir, product, data_source="DATA_SOURCE", campaign_name="CAMPAIGN_NAME"):
    from disdrodb.l0.create_directories import create_metadata_directory

    return create_metadata_directory(
        base_dir=base_dir, product=product, data_source=data_source, campaign_name=campaign_name
    )


def create_fake_station_dir(
    base_dir, product, data_source="DATA_SOURCE", campaign_name="CAMPAIGN_NAME", station_name="station_name"
):
    from disdrodb.l0.create_directories import create_station_directory

    return create_station_directory(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )


def create_fake_metadata_file(
    base_dir,
    product="RAW",
    metadata_dict={},
    data_source="DATA_SOURCE",
    campaign_name="CAMPAIGN_NAME",
    station_name="station_name",
):
    from disdrodb.api.io import define_metadata_filepath

    # Define metadata filepath
    metadata_fpath = define_metadata_filepath(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=False,
    )
    # Create metadata directory
    os.makedirs(os.path.dirname(metadata_fpath), exist_ok=True)

    # Define defaults fields
    if "data_source" not in metadata_dict:
        metadata_dict["data_source"] = data_source
    if "campaign_name" not in metadata_dict:
        metadata_dict["campaign_name"] = campaign_name
    if "station_name" not in metadata_dict:
        metadata_dict["station_name"] = station_name

    # Write metadata
    write_yaml(metadata_dict, metadata_fpath)

    # Return filepath
    return str(metadata_fpath)


def create_fake_raw_data_file(
    base_dir,
    product="RAW",
    data_source="DATA_SOURCE",
    campaign_name="CAMPAIGN_NAME",
    station_name="station_name",
    filename="test_data.txt",
):
    # Define station data directory
    from disdrodb.l0.create_directories import create_station_directory

    station_dir = create_station_directory(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Define filepath
    filepath = os.path.join(station_dir, filename)
    # filepath = station_dir / filename

    # Write fake data
    # filepath.touch()
    with open(filepath, "w") as f:
        f.write("This is some fake text.")

    # Return filepath as string
    return str(filepath)


@pytest.fixture
def create_test_config_files(request):
    """Create the specified config files into a temporary "test" directory.

    This fixture facilitates the creation of configuration files from provided dictionaries.
    The files are created in a temporary directory at disdrodb/l0/configs/test,
    that is automatically cleaned up after the test completes, regardless of whether
    the test passes or fails.

    Parameters
    ----------
    request : _pytest.fixtures.SubRequest
        The request object provided by pytest. It must contain a parameter `param`
        with a dictionary structure {"<config_file_name>.yml": <config_dict>}.
        `request.param` is used to extract the configuration data and file names
        for the configuration files to be created.

    """

    config_dicts = request.param
    for file_name, dictionary in config_dicts.items():
        config_folder = os.path.join(__root_path__, "disdrodb", "l0", "configs")

        test_folder = os.path.join(config_folder, "test")
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        test_file_path = os.path.join(test_folder, file_name)
        write_yaml(dictionary, test_file_path)

    yield
    os.remove(test_file_path)
    shutil.rmtree(test_folder)
