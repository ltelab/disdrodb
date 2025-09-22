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
"""DISDRODB Configuration File functions."""

import os
import shutil
from typing import Optional

from disdrodb.utils.yaml import read_yaml, write_yaml


def _define_config_filepath():
    # Retrieve user home directory
    home_directory = os.path.expanduser("~")
    # Define path where .config_disdrodb.yaml file should be located
    filepath = os.path.join(home_directory, ".config_disdrodb.yml")
    return filepath


def define_configs(
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
    scattering_table_dir: Optional[str] = None,
    products_configs_dir: Optional[str] = None,
    folder_partitioning: Optional[str] = None,
    zenodo_token: Optional[str] = None,
    zenodo_sandbox_token: Optional[str] = None,
):
    """
    Defines the DISDRODB configuration file with the given credentials and base directory.

    Parameters
    ----------
    data_archive_dir : str
        The directory path where the DISDRODB Data Archive is located.
    metadata_archive_dir : str
        The directory path where the DISDRODB Metadata Archive is located.
    scattering_table_dir : str
        The directory path where to store DISDRODB T-Matrix scattering tables.
    products_configs_dir : str
        The directory path where the custom DISDRODB products configurations files are defined.
    folder_partitioning : str
        The folder partitioning scheme used in the DISDRODB Data Archive.
        Allowed values are:
        - "": No additional subdirectories, files are saved directly in <station_dir>.
        - "year": Files are stored under a subdirectory for the year (<station_dir>/2025).
        - "year/month": Files are stored under subdirectories by year and month (<station_dir>/2025/04).
        - "year/month/day": Files are stored under subdirectories by year, month and day (<station_dir>/2025/04/01).
        - "year/month_name": Files are stored under subdirectories by year and month name (<station_dir>/2025/April).
        - "year/quarter": Files are stored under subdirectories by year and quarter (<station_dir>/2025/Q2).
    zenodo__token: str
        Zenodo Access Token. It is required to upload stations data to Zenodo.
    zenodo_sandbox_token: str
        Zenodo Sandbox Access Token. It is required to upload stations data to Zenodo Sandbox.

    Notes
    -----
    This function write or update the DISDRODB config YAML file.
    The DISDRODB config YAML file is located in the user's home directory at ``~/.config_disdrodb.yml``.
    The configuration file is used to run the various DISDRODB operations.

    """
    import disdrodb
    from disdrodb.api.checks import (
        check_data_archive_dir,
        check_folder_partitioning,
        check_metadata_archive_dir,
        check_scattering_table_dir,
    )

    # Define path to .config_disdrodb.yaml file
    filepath = _define_config_filepath()

    # If the config exists, read it and update it ;)
    if os.path.exists(filepath):
        config_dict = read_yaml(filepath)
        action_msg = "updated"
    else:
        config_dict = {}
        action_msg = "written"

    # Add DISDRODB Data Archive Directory
    if data_archive_dir is not None:
        config_dict["data_archive_dir"] = check_data_archive_dir(data_archive_dir)

    # Add DISDRODB Metadata Archive Directory
    if metadata_archive_dir is not None:
        config_dict["metadata_archive_dir"] = check_metadata_archive_dir(metadata_archive_dir)

    # Add DISDRODB Scattering Table Directory
    if scattering_table_dir is not None:
        os.makedirs(scattering_table_dir, exist_ok=True)
        config_dict["scattering_table_dir"] = check_scattering_table_dir(scattering_table_dir)

    # Add DISDRODB Folder Partitioning
    if folder_partitioning is not None:
        config_dict["folder_partitioning"] = check_folder_partitioning(folder_partitioning)

    # Add Zenodo Access Tokens
    if zenodo_token is not None:
        config_dict["zenodo_token"] = zenodo_token
    if zenodo_sandbox_token is not None:
        config_dict["zenodo_sandbox_token"] = zenodo_sandbox_token

    if products_configs_dir is not None:
        config_dict["products_configs_dir"] = products_configs_dir

    # Write the DISDRODB config file
    write_yaml(config_dict, filepath, sort_keys=False)

    print(f"The DISDRODB config file has been {action_msg} successfully!")

    # Now read the config file and set it as the active configuration
    # - This avoid the need to restart a python session to take effect !
    config_dict = read_configs()
    disdrodb.config.update(config_dict)


def read_configs() -> dict[str, str]:
    """
    Reads the DISDRODB configuration file and returns a dictionary with the configuration settings.

    Returns
    -------
    dict
        A dictionary containing the configuration settings for the DISDRODB.

    Raises
    ------
    ValueError
        If the configuration file has not been defined yet. Use ``disdrodb.define_configs()`` to
        specify the configuration file path and settings.

    Notes
    -----
    This function reads the YAML configuration file located at ``~/.config_disdrodb.yml``.
    """
    # Define path where .config_disdrodb.yaml file should be located
    filepath = _define_config_filepath()
    if not os.path.exists(filepath):
        raise ValueError("The DISDRODB config file has not been specified. Use disdrodb.define_configs to specify it !")
    # Read the DISDRODB config file
    config_dict = read_yaml(filepath)
    return config_dict


def get_data_archive_dir(data_archive_dir=None):
    """Return the DISDRODB base directory."""
    import disdrodb
    from disdrodb.api.checks import check_data_archive_dir

    if data_archive_dir is None:
        data_archive_dir = disdrodb.config.get("data_archive_dir", None)
    if data_archive_dir is None:
        raise ValueError("The DISDRODB Data Archive Directory is not specified.")
    data_archive_dir = check_data_archive_dir(data_archive_dir)  # ensure Path converted to str
    return data_archive_dir


def get_metadata_archive_dir(metadata_archive_dir=None):
    """Return the DISDRODB Metadata Archive Directory."""
    import disdrodb
    from disdrodb.api.checks import check_metadata_archive_dir

    if metadata_archive_dir is None:
        metadata_archive_dir = disdrodb.config.get("metadata_archive_dir", None)
    if metadata_archive_dir is None:
        raise ValueError("The DISDRODB Metadata Archive Directory is not specified.")
    metadata_archive_dir = check_metadata_archive_dir(metadata_archive_dir)  # ensure Path converted to str
    return metadata_archive_dir


def get_scattering_table_dir(scattering_table_dir=None):
    """Return the directory where DISDRODB save pyTMatrix scattering tables."""
    import disdrodb
    from disdrodb.api.checks import check_scattering_table_dir

    if scattering_table_dir is None:
        scattering_table_dir = disdrodb.config.get("scattering_table_dir", None)
    if scattering_table_dir is None:
        raise ValueError("The directory where to save DISDRODB T-Matrix scattering tables is not specified.")
    scattering_table_dir = check_scattering_table_dir(scattering_table_dir)  # ensure Path converted to str
    return scattering_table_dir


def get_folder_partitioning():
    """Return the folder partitioning."""
    import disdrodb
    from disdrodb.api.checks import check_folder_partitioning

    # Get the folder partitioning
    folder_partitioning = disdrodb.config.get("folder_partitioning", None)
    if folder_partitioning is None:
        raise ValueError("The folder partitioning is not specified.")
    check_folder_partitioning(folder_partitioning)
    return folder_partitioning


def get_zenodo_token(sandbox: bool):
    """Return the Zenodo access token."""
    import disdrodb

    if sandbox:
        host = "sandbox.zenodo.org"
        token_name = "zenodo_sandbox_token"
    else:
        host = "zenodo.org"
        token_name = "zenodo_token"

    # token = read_configs().get(token_name, None)
    token = disdrodb.config.get(token_name, None)

    if token is None:
        print(f"The '{token_name}' is not yet specified in the DISDRODB config file !")
        print(f"1. Generate the token at https://{host}/account/settings/applications/tokens/new/")
        print(f"2. Add the {token_name} to the .config_disdrodb.yml file:")
        print("")
        print("   import disdrodb")
        print(f"   disdrodb.define_config({token_name}=<your_token>)")
        raise ValueError(f"Missing {token_name} in the DISDRODB config file !")

    return token


def get_default_products_configs_dir():
    """Return the directory path where DISDRODB products default configuration files are stored."""
    import disdrodb

    products_configs_dir = os.path.join(disdrodb.__root_path__, "disdrodb", "etc", "products")
    return products_configs_dir


def get_products_configs_dir():
    """Return the DISDRODB products configuration directory."""
    import disdrodb

    if os.environ.get("PYTEST_CURRENT_TEST"):
        products_configs_dir = os.path.join(disdrodb.__root_path__, "disdrodb", "tests", "products")
    else:
        products_configs_dir = disdrodb.config.get("products_configs_dir", get_default_products_configs_dir())
    return products_configs_dir


def copy_default_products_configs(products_configs_dir):
    """Copy the default DISDRODB products configuration directory to a custom location.

    This function duplicates the entire directory of default products settings
    (located at ``disdrodb/etc/products``) into the user-specified
    ``products_configs_dir``. Once copied, you can safely edit these files without
    modifying the library's built-in defaults. To have DISDRODB use your
    custom settings, point the global configuration at this new directory
    (e.g by specifying ``products_configs_dir`` with the ``disdrodb.define_configs`` function).

    Parameters
    ----------
    products_configs_dir:
        Directory where the default products configuration files
        will be copied. This directory must not already exist, and later
        needs to be referenced in your DISDRODB global configuration.

    Returns
    -------
    products_configs_dir
        The path to the newly created custom product configuration directory.

    """
    source_dir_path = get_default_products_configs_dir()
    if os.path.exists(products_configs_dir):
        raise FileExistsError(f"The {products_configs_dir} directory already exists!")
    products_configs_dir = shutil.copytree(source_dir_path, products_configs_dir)
    return products_configs_dir
