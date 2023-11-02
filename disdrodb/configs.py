#!/usr/bin/env python3
"""
Created on Thu Nov  2 15:39:01 2023

@author: ghiggi
"""
import os
from typing import Dict

from disdrodb.utils.yaml import read_yaml, write_yaml


def define_disdrodb_configs(
    disdrodb_dir: str,
):
    """
    Defines the DISDRODB configuration file with the given credentials and base directory.

    Parameters
    ----------
    disdrodb_dir : str
        The base directory where DISDRODB Metadata Archive is located.

    Notes
    -----
    This function writes a YAML file to the user's home directory at ~/.config_disdrodb.yml
    The configuration file is used to run the various DISDRODB operations.

    """
    config_dict = {}
    config_dict["disdrodb_dir"] = disdrodb_dir

    # Retrieve user home directory
    home_directory = os.path.expanduser("~")

    # Define path to .config_disdrodb.yaml file
    fpath = os.path.join(home_directory, ".config_disdrodb.yml")

    # Write the DISDRODB config file
    write_yaml(config_dict, fpath, sort_keys=False)

    print("The DISDRODB config file has been written successfully!")
    return


def read_disdrodb_configs() -> Dict[str, str]:
    """
    Reads the DISDRODB configuration file and returns a dictionary with the configuration settings.

    Returns
    -------
    dict
        A dictionary containing the configuration settings for the DISDRODB.

    Raises
    ------
    ValueError
        If the configuration file has not been defined yet. Use `disdrodb.define_configs()` to
        specify the configuration file path and settings.

    Notes
    -----
    This function reads the YAML configuration file located at ~/.config_disdrodb.yml.
    """
    # Retrieve user home directory
    home_directory = os.path.expanduser("~")
    # Define path where .config_disdrodb.yaml file should be located
    fpath = os.path.join(home_directory, ".config_disdrodb.yml")
    if not os.path.exists(fpath):
        raise ValueError("The DISDRODB config file has not been specified. Use disdrodb.define_configs to specify it !")
    # Read the DISDRODB config file
    config_dict = read_yaml(fpath)
    return config_dict


def _get_config_key(key):
    """Return the config key if `value` is None."""
    value = read_disdrodb_configs().get(key, None)
    if value is None:
        raise ValueError(f"The DISDRODB {key} parameter has not been defined ! ")
    return value


def get_disdrodb_dir():
    """Return the DISDRODB base directory."""
    import disdrodb

    disdrodb_dir = disdrodb.config["dir"]
    return disdrodb_dir
