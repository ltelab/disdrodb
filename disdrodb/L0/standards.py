#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2022 DISDRODB developers
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

# Retrieve sensor standards and configs

# -----------------------------------------------------------------------------.
import os
import yaml
import logging
import datetime

logger = logging.getLogger(__name__)

PRODUCT_VERSION = "V0"
SOFTWARE_VERSION = "V0"
EPOCH = "seconds since 1970-01-01 00:00:00"


# Notes:
# - L0A_encodings currently specify only the dtype. This could be expanded in the future.
# - disdrodb.configs ... the netcdf chunk size could be an option to be specified


def read_config_yml(sensor_name: str, filename: str) -> dict:
    """Read a config yaml file and return the dictionary.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    filename : str
        Name of the file.

    Returns
    -------
    dict
        Content of the config file.

    Raises
    ------
    ValueError
        Error if file does not exist.
    """

    # Get config path
    config_sensor_dir_path = get_configs_dir(sensor_name)
    fpath = os.path.join(config_sensor_dir_path, filename)
    # Check yaml file exists
    if not os.path.exists(fpath):
        msg = f"{filename} not available in {config_sensor_dir_path}"
        logger.exception(msg)
        raise ValueError(msg)
    # Open dictionary
    with open(fpath, "r") as f:
        d = yaml.safe_load(f)
    return d


def get_configs_dir(sensor_name: str) -> str:
    """Retrieve configs directory.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    str
        Config directory.

    Raises
    ------
    ValueError
        Error if the config directory does not exist.
    """

    dir_path = os.path.dirname(__file__)
    config_dir_path = os.path.join(dir_path, "configs")
    config_sensor_dir_path = os.path.join(config_dir_path, sensor_name)
    if not os.path.exists(config_sensor_dir_path):
        list_sensors = sorted(os.listdir(config_dir_path))
        print(f"Available sensor_name are {list_sensors}")
        raise ValueError(
            f"The config directory {config_sensor_dir_path} does not exist."
        )
    return config_sensor_dir_path


def get_available_sensor_name() -> sorted:
    """Get available names of sensors.

    Returns
    -------
    sorted
        Sorted list of the available sensors
    """

    dir_path = os.path.dirname(__file__)
    config_dir_path = os.path.join(dir_path, "configs")
    # TODO: here add checks that contains all required yaml file
    return sorted(os.listdir(config_dir_path))


def get_variables_dict(sensor_name: str) -> dict:
    """Get a dictionary containing the variable name of the sensor field numbers.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Variables names
    """

    return read_config_yml(sensor_name=sensor_name, filename="variables.yml")


def get_sensor_variables(sensor_name: str) -> list:
    """Get sensor variable names list.

    Parameters
    ----------
    sensor_name : str
         Name of the sensor.

    Returns
    -------
    list
        List of the variables values
    """

    return list(get_variables_dict(sensor_name).values())


def get_data_format_dict(sensor_name: str) -> dict:
    """Get a dictionary containing the data format of each sensor variable.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Data format of each sensor variable
    """

    return read_config_yml(sensor_name=sensor_name, filename="L0_data_format.yml")


def get_description_dict(sensor_name: str) -> dict:
    """Get a dictionary containing the description of each sensor variable.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Description of each sensor variable.
    """

    return read_config_yml(sensor_name=sensor_name, filename="variable_description.yml")


def get_long_name_dict(sensor_name: str) -> dict:
    """Get a dictionary containing the long name of each sensor variable.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Long name of each sensor variable.
    """

    return read_config_yml(sensor_name=sensor_name, filename="variable_longname.yml")


def get_units_dict(sensor_name: str) -> dict:
    """Get a dictionary containing the unit of each sensor variable.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Unit of each sensor variable
    """

    return read_config_yml(sensor_name=sensor_name, filename="variable_units.yml")


def get_sensor_name_dict(sensor_name: str) -> dict:
    """Get a dictionary containing the description of each sensor variable.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Description of each sensor variable
    """

    d = read_config_yml(sensor_name=sensor_name, filename="variable_description.yml")
    return d


def get_diameter_bins_dict(sensor_name: str) -> dict:
    """Get dictionary with sensor_name diameter bins information.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        sensor_name diameter bins information
    """

    d = read_config_yml(sensor_name=sensor_name, filename="diameter_bins.yml")
    # TODO: Check dict contains center, bounds and width keys
    return d


def get_velocity_bins_dict(sensor_name: str) -> dict:
    """Get velocity with sensor_name diameter bins information.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Sensor_name diameter bins information
    """

    d = read_config_yml(sensor_name=sensor_name, filename="velocity_bins.yml")
    return d


def get_L0A_dtype(sensor_name: str) -> dict:
    """Get a dictionary containing the L0A dtype.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        L0A dtype
    """

    # Note: This function could extract the info from get_L0A_encodings_dict in future.
    d = read_config_yml(sensor_name=sensor_name, filename="L0A_encodings.yml")
    return d


def get_L0A_encodings_dict(sensor_name: str) -> dict:
    """Get a dictionary containing the L0A encodings

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        L0A encodings
    """

    # - L0A_encodings currently specify only the dtype. This could be expanded in the future.
    d = read_config_yml(sensor_name=sensor_name, filename="L0A_encodings.yml")
    return d


def get_L0B_encodings_dict(sensor_name: str) -> dict:
    """Get a dictionary containing the encoding to write L0B netCDFs.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Encoding to write L0B netCDFs
    """

    d = read_config_yml(sensor_name=sensor_name, filename="L0B_encodings.yml")

    # Ensure chunksize is a list
    for var in d.keys():
        if not isinstance(d[var]["chunksizes"], (list, type(None))):
            d[var]["chunksizes"] = [d[var]["chunksizes"]]

    # Sanitize encodings
    for var in d.keys():
        # Ensure contiguous=True if chunksizes is None
        if isinstance(d[var]["chunksizes"], type(None)) and not d[var]["contiguous"]:
            # These changes are required to enable netCDF writing
            d[var]["contiguous"] = True
            d[var]["fletcher32"] = False
            d[var]["zlib"] = False
            print(f"Set contiguous=True for variable {var} because chunksizes=None")
            print(f"Set fletcher32=False for variable {var} because contiguous=True")
            print(f"Set zlib=False for variable {var} because contiguous=True")
        # Ensure contiguous=False if chunksizes is not None
        if d[var]["contiguous"] and not isinstance(d[var]["chunksizes"], type(None)):
            d[var]["contiguous"] = False
            print(
                f"Set contiguous=False for variable {var} because chunksizes is defined!"
            )

    return d


def get_time_encoding() -> dict:
    """Create time encoding

    Returns
    -------
    dict
        Time encoding
    """
    encoding = {}
    encoding["units"] = EPOCH
    encoding["calendar"] = "proleptic_gregorian"
    return encoding


def set_DISDRODB_L0_attrs(ds, attrs: dict):
    """Define DISDRODB L0 attributes

    Parameters
    ----------
    ds : xarray dataset
        Dataset
    attrs : dict
        Attributes

    Returns
    -------
    xarray dataset
        Dataset
    """
    sensor_name = attrs["sensor_name"]
    # ----------------------------------
    # Set global attributes (from metadata)
    ds.attrs = attrs

    # ----------------------------------
    # Set variable attributes (from config standards)
    description_dict = get_sensor_name_dict(sensor_name)
    units_dict = get_units_dict(sensor_name)
    long_name_dict = get_long_name_dict(sensor_name)
    # data_format_dict = get_data_format_dict(sensor_name : str)
    for var in list(ds.data_vars):
        attrs_var = {}
        attrs_var["long_name"] = long_name_dict[var]
        attrs_var["description"] = description_dict[var]
        attrs_var["unit"] = units_dict[var]
        # TODO:
        # attrs['valid_min'] =  data_format_dict
        # attrs['valid_max'] =  data_format_dict
        ds[var].attrs = attrs_var

    # ----------------------------------
    # Set coordinate attributes
    # TODO
    # lat/lon - degrees N/E...
    # altitude - meters above sea level

    # ----------------------------------
    # Add DISDRODB processing info
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    ds.attrs["disdrodb_processing_date"] = current_time
    ds.attrs["disdrodb_product_version"] = PRODUCT_VERSION
    ds.attrs["disdrodb_software_version"] = SOFTWARE_VERSION
    ds.attrs["disdrodb_product_level"] = "L0"
    # ds.attrs['disdrodb_id'] = # TODO XXXX

    return ds


####-------------------------------------------------------------------------.
#############################################
#### Get diameter and velocity bins info ####
#############################################


def get_diameter_bin_center(sensor_name: str) -> list:
    """Get diameter bin center.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor

    Returns
    -------
    list
        Diameter bin center
    """

    diameter_dict = get_diameter_bins_dict(sensor_name)
    diameter_bin_center = list(diameter_dict["center"].values())
    return diameter_bin_center


def get_diameter_bin_lower(sensor_name: str) -> list:
    """Get diameter bin lower bound.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor

    Returns
    -------
    list
        Diameter bin lower bound
    """

    diameter_dict = get_diameter_bins_dict(sensor_name)
    lower_bounds = [v[0] for v in diameter_dict["bounds"].values()]
    return lower_bounds


def get_diameter_bin_upper(sensor_name: str) -> list:
    """Get diameter bin upper bound.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor

    Returns
    -------
    list
        Diameter bin upper bound
    """

    diameter_dict = get_diameter_bins_dict(sensor_name)
    upper_bounds = [v[1] for v in diameter_dict["bounds"].values()]
    return upper_bounds


def get_diameter_bin_width(sensor_name: str) -> list:
    """Get diameter bin width.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor

    Returns
    -------
    list
        Diameter bin width
    """

    diameter_dict = get_diameter_bins_dict(sensor_name)
    diameter_bin_width = list(diameter_dict["width"].values())
    return diameter_bin_width


def get_velocity_bin_center(sensor_name: str) -> list:
    """Get velocity bin center.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor

    Returns
    -------
    list
        Velocity bin center
    """

    velocity_dict = get_velocity_bins_dict(sensor_name)
    if velocity_dict is not None:
        velocity_bin_center = list(velocity_dict["center"].values())
    else:
        return None
    return velocity_bin_center


def get_velocity_bin_lower(sensor_name: str) -> list:
    """Get velocity bin lower bound.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor

    Returns
    -------
    list
        Velocity bin lower bound.
    """

    velocity_dict = get_velocity_bins_dict(sensor_name)
    if velocity_dict is not None:
        lower_bounds = [v[0] for v in velocity_dict["bounds"].values()]
    else:
        return None
    return lower_bounds


def get_velocity_bin_upper(sensor_name: str) -> list:
    """Get velocity bin upper bound.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor

    Returns
    -------
    list
        Velocity bin upper bound
    """

    velocity_dict = get_velocity_bins_dict(sensor_name)
    if velocity_dict is not None:
        upper_bounds = [v[1] for v in velocity_dict["bounds"].values()]
    else:
        return None
    return upper_bounds


def get_velocity_bin_width(sensor_name: str) -> list:
    """Get velocity bin width.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor

    Returns
    -------
    list
        Velocity bin width
    """

    velocity_dict = get_velocity_bins_dict(sensor_name)
    if velocity_dict is not None:
        velocity_bin_width = list(velocity_dict["width"].values())
    else:
        return None
    return velocity_bin_width


def get_raw_field_nbins(sensor_name: str) -> dict:
    """Get the raw field number of bins

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Field definition.
    """

    diameter_dict = get_diameter_bins_dict(sensor_name)
    velocity_dict = get_velocity_bins_dict(sensor_name)
    n_d = len(diameter_dict["center"])
    # For instruments measuring size and velocity (i.e. OTT Parsivel, ThiesLPM)
    if velocity_dict is not None:
        n_v = len(velocity_dict["center"])
        nbins_dict = {
            "raw_drop_concentration": n_d,
            "raw_drop_average_velocity": n_v,
            "raw_drop_number": n_d * n_v,
        }
    # For instruments measuring only size (i.e. RD80)
    else:
        nbins_dict = {
            "raw_drop_number": n_d,
        }
    return nbins_dict


def get_raw_field_dim_order(sensor_name: str) -> dict:
    """Get dimention order  dictionnary

    Parameters
    ----------
    sensor_name : str
        Name of the sensor

    Returns
    -------
    dict
        Dimention order  dictionnary

    Raises
    ------
    NotImplementedError
        Name of the sensor not implemented.
    """
    # TODO: this should go into a config file ...
    # TODO: also think to set dimensions as diameter and velocity ... TO DISCUSS
    if sensor_name in ["OTT_Parsivel", "OTT_Parsivel2", "ThiesLPM"]:
        dim_dict = {
            "raw_drop_concentration": ["diameter_bin_center"],
            "raw_drop_average_velocity": ["velocity_bin_center"],
            "raw_drop_number": ["diameter_bin_center", "velocity_bin_center"],
        }
    elif sensor_name in ["RD80"]:
        dim_dict = {"raw_drop_number": ["diameter_bin_center"]}
    else:
        raise NotImplementedError()
    return dim_dict


def get_raw_spectrum_ndims(sensor_name: str):
    encoding_dict = get_L0B_encodings_dict(sensor_name)
    ndim = len(encoding_dict["raw_drop_number"]["chunksizes"]) - 1
    return ndim


# -----------------------------------------------------------------------------.
