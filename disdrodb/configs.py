#!/usr/bin/env python3
"""
Created on Thu Nov  2 15:39:01 2023

@author: ghiggi
"""
import os
from typing import Dict

from disdrodb.utils.yaml import read_yaml, write_yaml


def _define_config_filepath():
    # Retrieve user home directory
    home_directory = os.path.expanduser("~")
    # Define path where .config_disdrodb.yaml file should be located
    filepath = os.path.join(home_directory, ".config_disdrodb.yml")
    return filepath


def define_disdrodb_configs(base_dir: str = None, zenodo_token: str = None, zenodo_sandbox_token: str = None):
    """
    Defines the DISDRODB configuration file with the given credentials and base directory.

    Parameters
    ----------
    base_dir : str
        The base directory where DISDRODB Metadata Archive is located.
    zenodo__token: str
        Zenodo Access Token. It is required to upload stations data to Zenodo.
    zenodo_sandbox_token: str
        Zenodo Sandbox Access Token. It is required to upload stations data to Zenodo Sandbox.

    Notes
    -----
    This function write or update the DISDRODB config YAML file
    The DISDRODB config YAML file is located in the user's home directory at ~/.config_disdrodb.yml
    The configuration file is used to run the various DISDRODB operations.

    """
    # Define path to .config_disdrodb.yaml file
    filepath = _define_config_filepath()

    # If the config exists, read it and update it ;)
    if os.path.exists(filepath):
        config_dict = read_yaml(filepath)
        action_msg = "updated"
    else:
        config_dict = {}
        action_msg = "written"

    # Add DISDRODB Base Directory
    if base_dir is not None:
        config_dict["base_dir"] = base_dir

    # Add Zenodo Access Tokens
    if zenodo_token is not None:
        config_dict["zenodo_token"] = zenodo_token
    if zenodo_sandbox_token is not None:
        config_dict["zenodo_sandbox_token"] = zenodo_sandbox_token

    # Write the DISDRODB config file
    write_yaml(config_dict, filepath, sort_keys=False)

    print(f"The DISDRODB config file has been {action_msg} successfully!")
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
    # Define path where .config_disdrodb.yaml file should be located
    filepath = _define_config_filepath()
    if not os.path.exists(filepath):
        raise ValueError("The DISDRODB config file has not been specified. Use disdrodb.define_configs to specify it !")
    # Read the DISDRODB config file
    config_dict = read_yaml(filepath)
    return config_dict


def get_base_dir(base_dir=None):
    """Return the DISDRODB base directory."""
    import disdrodb

    if base_dir is None:
        base_dir = disdrodb.config.get("base_dir", None)
    if base_dir is None:
        raise ValueError("The DISDRODB Base Directory is not specified.")
    base_dir = str(base_dir)  # convert Path to str
    return base_dir


def get_zenodo_token(sandbox: bool):
    """Return the Zenodo Access Token."""
    import disdrodb

    if sandbox:
        host = "sandbox.zenodo.org"
        token_name = "zenodo_sandbox_token"
    else:
        host = "zenodo.org"
        token_name = "zenodo_token"

    # token = read_disdrodb_configs().get(token_name, None)
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
