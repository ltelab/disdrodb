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
import numpy as np

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


def get_data_range_dict(sensor_name: str) -> dict:
    """Get the variable data range.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with the expected data value range for each data field.
        It excludes variables without specified data_range.
    """

    data_format_dict = get_data_format_dict(sensor_name)
    dict_data_range = {}
    for k in data_format_dict.keys():
        data_range = data_format_dict[k]["data_range"]
        if data_range is not None:
            dict_data_range[k] = data_range
    return dict_data_range


####-------------------------------------------------------------------------.
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


####-------------------------------------------------------------------------.
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

    d = read_config_yml(sensor_name=sensor_name, filename="bins_diameter.yml")
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

    d = read_config_yml(sensor_name=sensor_name, filename="bins_velocity.yml")
    return d


####-------------------------------------------------------------------------.
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


####-------------------------------------------------------------------------.
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


def get_coords_attrs_dict(ds):
    """Return dictionary with DISDRODB coordinates attributes."""
    attrs_dict = {}
    # Define diameter attributes
    attrs_dict["diameter_bin_center"] = {
        "name": "diameter_bin_center",
        "standard_name": "diameter_bin_center",
        "long_name": "diameter_bin_center",
        "units": "mm",
        "description": "Bin center drop diameter value",
    }
    attrs_dict["diameter_bin_width"] = {
        "name": "diameter_bin_width",
        "standard_name": "diameter_bin_width",
        "long_name": "diameter_bin_width",
        "units": "mm",
        "description": "Drop diameter bin width",
    }
    attrs_dict["diameter_bin_upper"] = {
        "name": "diameter_bin_upper",
        "standard_name": "diameter_bin_upper",
        "long_name": "diameter_bin_upper",
        "units": "mm",
        "description": "Bin upper bound drop diameter value",
    }
    attrs_dict["velocity_bin_lower"] = {
        "name": "velocity_bin_lower",
        "standard_name": "velocity_bin_lower",
        "long_name": "velocity_bin_lower",
        "units": "mm",
        "description": "Bin lower bound drop diameter value",
    }
    # Define velocity attributes
    attrs_dict["velocity_bin_center"] = {
        "name": "velocity_bin_center",
        "standard_name": "velocity_bin_center",
        "long_name": "velocity_bin_center",
        "units": "m/s",
        "description": "Bin center drop fall velocity value",
    }
    attrs_dict["velocity_bin_width"] = {
        "name": "velocity_bin_width",
        "standard_name": "velocity_bin_width",
        "long_name": "velocity_bin_width",
        "units": "m/s",
        "description": "Drop fall velocity bin width",
    }
    attrs_dict["velocity_bin_upper"] = {
        "name": "velocity_bin_upper",
        "standard_name": "velocity_bin_upper",
        "long_name": "velocity_bin_upper",
        "units": "m/s",
        "description": "Bin upper bound drop fall velocity value",
    }
    attrs_dict["velocity_bin_lower"] = {
        "name": "velocity_bin_lower",
        "standard_name": "velocity_bin_lower",
        "long_name": "velocity_bin_lower",
        "units": "m/s",
        "description": "Bin lower bound drop fall velocity value",
    }
    # Define geolocation attributes
    attrs_dict["latitude"] = {
        "name": "latitude",
        "standard_name": "latitude",
        "long_name": "Latitude",
        "units": "degrees_north",
    }
    attrs_dict["longitude"] = {
        "name": "longitude",
        "standard_name": "longitude",
        "long_name": "Longitude",
        "units": "degrees_east",
    }
    attrs_dict["altitude"] = {
        "name": "altitude",
        "standard_name": "altitude",
        "long_name": "Altitude",
        "units": "m",
        "description": "Altitude above sea level",
    }

    # Define crs attributes
    # TODO
    # - CF compliant
    # - wkt
    # - add grid_mapping name

    # Define time attributes
    attrs_dict["time"] = {
        "name": "time",
        "standard_name": "time",
        "long_name": "time",
        "description": "UTC Time",
    }

    return attrs_dict


def set_disdrodb_attrs(ds, product_level: str):
    """Add DISDRODB processing information to the netCDF global attributes.

    Parameters
    ----------
    ds : xarray dataset
        Dataset
    product_level: str
        DISDRODB product_level

    Returns
    -------
    xarray dataset
        Dataset
    """
    # Add DISDRODB processing info
    now = datetime.datetime.utcnow()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    ds.attrs["disdrodb_processing_date"] = current_time
    ds.attrs["disdrodb_product_version"] = PRODUCT_VERSION
    ds.attrs["disdrodb_software_version"] = SOFTWARE_VERSION
    ds.attrs["disdrodb_product_level"] = product_level
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


####-------------------------------------------------------------------------.
# TODO: to improve


def get_dims_size_dict(sensor_name: str) -> dict:
    """Get the number of bins for each dimension.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with the number of bins for each dimension.
    """
    # Retrieve number of bins
    diameter_dict = get_diameter_bins_dict(sensor_name)
    velocity_dict = get_velocity_bins_dict(sensor_name)
    n_diameter_bins = len(diameter_dict["center"])
    if velocity_dict is None:
        n_velocity_bins = 0
    else:
        n_velocity_bins = len(velocity_dict["center"])
    # Define the dictionary
    dims_size_dict = {
        "diameter_bin_center": n_diameter_bins,
        "velocity_bin_center": n_velocity_bins,
    }
    return dims_size_dict


def get_n_diameter_bins(sensor_name):
    """Get the number of diameter bins."""
    return get_dims_size_dict(sensor_name)["diameter_bin_center"]


def get_n_velocity_bins(sensor_name):
    """Get the number of velocity bins."""
    return get_dims_size_dict(sensor_name)["velocity_bin_center"]


def get_raw_field_dim_order(sensor_name: str) -> dict:
    """Get the dimention order of the raw fields.

    The order of dimension specified for raw_drop_number controls the
    reshaping of the precipitation raw spectrum.

    Examples:
        OTT Parsivel spectrum [v1d1 ... v1d32, v2d1, ..., v2d32]
        --> dims_order = ["velocity_bin_center", "diameter_bin_center"]
        Thies LPM spectrum [v1d1 ... v20d1, v1d2, ..., v20d2]
        --> dims_order = ["diameter_bin_center", "velocity_bin_center"]

    Parameters
    ----------
    sensor_name : str
        Name of the sensor

    Returns
    -------
    dict
        Dimension order dictionary

    Raises
    ------
    NotImplementedError
        Name of the sensor not implemented.
    """

    if sensor_name in ["OTT_Parsivel", "OTT_Parsivel2"]:
        dim_dict = {
            "raw_drop_concentration": ["diameter_bin_center"],
            "raw_drop_average_velocity": ["velocity_bin_center"],
            "raw_drop_number": ["velocity_bin_center", "diameter_bin_center"],
        }
    elif sensor_name in ["Thies_LPM"]:
        dim_dict = {
            "raw_drop_number": ["diameter_bin_center", "velocity_bin_center"],
        }
    elif sensor_name in ["RD_80"]:
        dim_dict = {"raw_drop_number": ["diameter_bin_center"]}
    else:
        raise NotImplementedError()
    return dim_dict


# TODO: RENAME
def get_raw_field_nbins(sensor_name: str) -> dict:
    """Get the raw field number of values.

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
    # For instruments measuring size and velocity (i.e. OTT Parsivel, Thies_LPM)
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


def get_raw_spectrum_ndims(sensor_name: str):
    encoding_dict = get_L0B_encodings_dict(sensor_name)
    ndim = len(encoding_dict["raw_drop_number"]["chunksizes"]) - 1
    return ndim


def get_valid_coordinates_names(sensor_name):
    common_coords = [
        "time",
        "latitude",
        "longitude",
        # "altitude",
        # crs,
        "diameter_bin_center",
        "diameter_bin_width",
        "diameter_bin_lower",
        "diameter_bin_upper",
    ]
    if sensor_name in ["OTT_Parsivel", "OTT_Parsivel2", "Thies_LPM"]:
        velocity_coordinates = [
            "velocity_bin_center",
            "velocity_bin_width",
            "velocity_bin_lower",
            "velocity_bin_upper",
        ]
        coordinates = common_coords + velocity_coordinates
    elif sensor_name in ["RD_80"]:
        coordinates = common_coords
    else:
        raise NotImplementedError()
    return coordinates


def get_valid_dimension_names(sensor_name):
    if sensor_name in ["OTT_Parsivel", "OTT_Parsivel2", "Thies_LPM"]:
        dimensions = ["time", "velocity_bin_center", "diameter_bin_center"]
    elif sensor_name in ["RD_80"]:
        dimensions = ["time", "diameter_bin_center"]
    else:
        raise NotImplementedError()
    return dimensions


def get_valid_variable_names(sensor_name):
    variables = list(get_L0B_encodings_dict(sensor_name).keys())
    return variables


def get_valid_names(sensor_name):
    variables = get_valid_variable_names(sensor_name)
    coordinates = get_valid_dimension_names(sensor_name)
    dimensions = get_valid_coordinates_names(sensor_name)
    names = np.unique(variables + coordinates + dimensions).tolist()
    return names


def get_variables_dimension(sensor_name: str):
    encoding_dict = get_L0B_encodings_dict(sensor_name)
    variables = list(encoding_dict.keys())
    raw_field_dims = get_raw_field_dim_order(sensor_name)
    var_dim_dict = {}
    for var in variables:
        print(var)
        chunk_sizes = encoding_dict[var]["chunksizes"]
        if len(chunk_sizes) == 1:
            var_dim_dict[var] = ["time"]
        else:
            var_dim_dict[var] = raw_field_dims[var] + ["time"]
    return var_dim_dict


# -----------------------------------------------------------------------------.
