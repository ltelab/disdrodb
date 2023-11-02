#!/usr/bin/env python3
"""
Created on Thu Nov  2 15:42:45 2023

@author: ghiggi
"""
import yaml


def read_yaml(fpath: str) -> dict:
    """Read a YAML file into a dictionary.

    Parameters
    ----------
    fpath : str
        Input YAML file path.

    Returns
    -------
    dict
        Dictionary with the attributes read from the YAML file.
    """
    with open(fpath) as f:
        dictionary = yaml.safe_load(f)
    return dictionary


def write_yaml(dictionary, fpath, sort_keys=False):
    """Write a dictionary into a YAML file.

    Parameters
    ----------
    dictionary : dict
        Dictionary to write into a YAML file.
    """
    with open(fpath, "w") as f:
        yaml.dump(dictionary, f, sort_keys=sort_keys)
    return None
