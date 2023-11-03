#!/usr/bin/env python3
"""Shared preparation and utilities for testing.

This module is executed automatically by pytest.

"""
import os
import shutil

import pytest

from disdrodb import __root_path__
from disdrodb.utils.yaml import write_yaml


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
