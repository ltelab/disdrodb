#!/usr/bin/env python3
"""Shared preparation and utilities for testing.

This module is executed automatically by pytest.

"""
import os
import shutil

import pytest

from disdrodb import __root_path__
from disdrodb.metadata.download import download_metadata_archive
from disdrodb.utils.yaml import write_yaml


def create_fake_metadata_directory(metadata_dir, data_source="DATA_SOURCE", campaign_name="CAMPAIGN_NAME"):
    from disdrodb.api.create_directories import create_metadata_directory

    return create_metadata_directory(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
    )


def create_fake_station_dir(
    base_dir,
    product,
    data_source="DATA_SOURCE",
    campaign_name="CAMPAIGN_NAME",
    station_name="station_name",
):
    from disdrodb.api.create_directories import create_station_directory

    return create_station_directory(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )


def create_fake_metadata_file(
    metadata_dir,
    metadata_dict=None,
    data_source="DATA_SOURCE",
    campaign_name="CAMPAIGN_NAME",
    station_name="station_name",
):
    from disdrodb.api.path import define_metadata_filepath
    from disdrodb.metadata.writer import get_default_metadata_dict

    # Define metadata filepath
    if metadata_dict is None:
        metadata_dict = {}
    metadata_filepath = define_metadata_filepath(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Create metadata directory
    os.makedirs(os.path.dirname(metadata_filepath), exist_ok=True)

    # Define defaults fields
    if "data_source" not in metadata_dict:
        metadata_dict["data_source"] = data_source
    if "campaign_name" not in metadata_dict:
        metadata_dict["campaign_name"] = campaign_name
    if "station_name" not in metadata_dict:
        metadata_dict["station_name"] = station_name
    if "sensor_name" not in metadata_dict:
        metadata_dict["sensor_name"] = "OTT_Parsivel"

    # Update over default metadata dict
    default_metadata_dict = get_default_metadata_dict()
    default_metadata_dict.update(metadata_dict)

    # Write metadata
    write_yaml(default_metadata_dict, metadata_filepath)

    # Return filepath
    return str(metadata_filepath)


def create_fake_issue_file(
    metadata_dir,
    issue_dict=None,
    data_source="DATA_SOURCE",
    campaign_name="CAMPAIGN_NAME",
    station_name="station_name",
):
    from disdrodb.api.path import define_issue_filepath
    from disdrodb.issue.writer import write_issue

    # Define issue filepath
    if issue_dict is None:
        issue_dict = {}
    issue_filepath = define_issue_filepath(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=False,
    )
    # Create issue directory
    os.makedirs(os.path.dirname(issue_filepath), exist_ok=True)

    # Write issue
    write_issue(
        filepath=issue_filepath,
        timesteps=issue_dict.get("timesteps", None),
        time_periods=issue_dict.get("time_periods", None),
    )
    # Return filepath
    return str(issue_filepath)


def get_default_product_kwargs(product, product_kwargs=None):
    from disdrodb import PRODUCTS_ARGUMENTS

    product_kwargs = {} if product_kwargs is None else product_kwargs
    if product not in PRODUCTS_ARGUMENTS:
        return product_kwargs
    # Define default test product_kwargs
    default_kwargs = {
        "model_name": "GAMMA_ML",
        "sample_interval": 30,
        "rolling": False,
    }
    # Set missing product kwargs
    required_args = PRODUCTS_ARGUMENTS[product]
    product_kwargs = {arg: product_kwargs.get(arg, default_kwargs[arg]) for arg in required_args}
    return product_kwargs


def create_fake_raw_data_file(
    base_dir,
    product="RAW",
    data_source="DATA_SOURCE",
    campaign_name="CAMPAIGN_NAME",
    station_name="station_name",
    filename="test_data.txt",
    **product_kwargs,
):
    from disdrodb.api.create_directories import create_data_directory

    product_kwargs = get_default_product_kwargs(product, product_kwargs)

    # Define station data directory
    data_dir = create_data_directory(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        **product_kwargs,
    )

    # Define filepath
    filepath = os.path.join(data_dir, filename)

    # Write fake data
    # filepath.touch()
    with open(filepath, "w") as f:
        f.write("This is some fake text.")

    # Return filepath as string
    return str(filepath)


@pytest.fixture
def create_test_config_files(request):  # noqa: PT004
    """Create the specified config files into a temporary "test" directory.

    This fixture facilitates the creation of configuration files from provided dictionaries.
    The files are created in a temporary directory at disdrodb/l0/configs/test,
    that is automatically cleaned up after the test completes, regardless of whether
    the test passes or fails.

    Parameters
    ----------
    request : _pytest.fixtures.SubRequest
        The request object provided by pytest. It must contain a parameter `param`
        with a dictionary structure {"<config_filename>.yml": <config_dict>}.
        `request.param` is used to extract the configuration data and file names
        for the configuration files to be created.

    """
    config_dicts = request.param
    for filename, dictionary in config_dicts.items():
        config_dir = os.path.join(__root_path__, "disdrodb", "l0", "configs")

        test_dir = os.path.join(config_dir, "test")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        test_filepath = os.path.join(test_dir, filename)
        write_yaml(dictionary, test_filepath)

    yield

    os.remove(test_filepath)
    shutil.rmtree(test_dir)


@pytest.fixture(scope="session")
def disdrodb_metadata_dir(tmp_path_factory):
    """Download the DISDRODB Metadata Archive once per pytest session.

    It return the metadata root directory pointing to it.
    """
    # Define directory where to download repository
    return download_metadata_archive(tmp_path_factory.mktemp("original_metadata_archive_repo"))
