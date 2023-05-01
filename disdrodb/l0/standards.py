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
import importlib

logger = logging.getLogger(__name__)

PRODUCT_VERSION = "V0"
SOFTWARE_VERSION = "V" + importlib.metadata.version("disdrodb")
CONVENTIONS = "CF-1.10, ACDD-1.3"
EPOCH = "seconds since 1970-01-01 00:00:00"


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
        raise ValueError(f"The config directory {config_sensor_dir_path} does not exist.")
    return config_sensor_dir_path


####--------------------------------------------------------------------------.


def available_sensor_name() -> sorted:
    """Get available names of sensors.

    Returns
    -------
    sorted
        Sorted list of the available sensors
    """

    dir_path = os.path.dirname(__file__)
    config_dir_path = os.path.join(dir_path, "configs")
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

    return read_config_yml(sensor_name=sensor_name, filename="raw_data_format.yml")


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


####--------------------------------------------------------------------------.
#### Variables validity dictionary


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
        It excludes variables without specified data_range key.
    """

    data_format_dict = get_data_format_dict(sensor_name)
    dict_data_range = {}
    for k in data_format_dict.keys():
        if "data_range" in data_format_dict[k]:
            data_range = data_format_dict[k]["data_range"]
            if data_range is not None:
                dict_data_range[k] = data_range
    return dict_data_range


def get_nan_flags_dict(sensor_name: str) -> dict:
    """Get the variable nan_flags.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with the expected nan_flags list for each data field.
        It excludes variables without specified nan_flags key.
    """

    data_format_dict = get_data_format_dict(sensor_name)
    dict_nan_flags = {}
    for k in data_format_dict.keys():
        if "nan_flags" in data_format_dict[k]:
            nan_flags = data_format_dict[k]["nan_flags"]
            if nan_flags is not None:
                if not isinstance(nan_flags, list):
                    nan_flags = [nan_flags]
                dict_nan_flags[k] = nan_flags
    return dict_nan_flags


def get_valid_values_dict(sensor_name: str) -> dict:
    """Get the list of valid values for a variable.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with the expected values for specific variables.
        It excludes variables without specified valid_values key.
    """
    data_format_dict = get_data_format_dict(sensor_name)
    dict_valid_values = {}
    for k in data_format_dict.keys():
        if "valid_values" in data_format_dict[k]:
            valid_values = data_format_dict[k]["valid_values"]
            if valid_values is not None:
                if not isinstance(valid_values, list):
                    valid_values = [valid_values]
                dict_valid_values[k] = valid_values
    return dict_valid_values


####--------------------------------------------------------------------------.
#### Get variable string format
def get_field_ndigits_natural_dict(sensor_name: str) -> dict:
    """Get number of digits on the left side of the comma from the instrument default string standards.

    Example: 123,45 -> 123 --> 3 natural digits

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with the expected number of natural digits for each data field.
    """

    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["n_naturals"] for k, v in data_dict.items()}
    return d


def get_field_ndigits_decimals_dict(sensor_name: dict) -> dict:
    """Get number of digits on the right side of the comma from the instrument default string standards.

    Example: 123,45 -> 45 --> 2 decimal digits
    Parameters
    ----------
    sensor_name : dict
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with the expected number of decimal digits for each data field.
    """

    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["n_decimals"] for k, v in data_dict.items()}
    return d


def get_field_ndigits_dict(sensor_name: str) -> dict:
    """Get number of digits from the instrument default string standards.

    Important note: it excludes the comma but it counts the minus sign !!!


    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    Returns
    -------
    dict
        Dictionary with the expected number of digits for each data field.
    """

    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["n_digits"] for k, v in data_dict.items()}
    return d


def get_field_nchar_dict(sensor_name: str) -> dict:
    """Get the total number of characters from the instrument default string standards.

    Important note: it accounts also for the comma and the minus sign !!!


    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with the expected number of characters for each data field.
    """

    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["n_characters"] for k, v in data_dict.items()}
    return d


####-------------------------------------------------------------------------.
#### Variable attributes


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

    return read_config_yml(sensor_name=sensor_name, filename="variable_long_name.yml")


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
#### Coordinates attributes


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
        "description": "Elevation above sea level",
    }
    # Define time attributes
    attrs_dict["time"] = {
        "name": "time",
        "standard_name": "time",
        "long_name": "time",
        "description": "UTC Time",
    }

    return attrs_dict


####-------------------------------------------------------------------------.
#### DISDRODB attributes


def set_disdrodb_attrs(ds, product_level: str):
    """Add DISDRODB processing information to the netCDF global attributes.

    It assumes stations metadata are already added the dataset.

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
    # Add dataset conventions
    ds.attrs["Conventions"] = CONVENTIONS

    # Add featureType
    platform_type = ds.attrs["platform_type"]
    if platform_type == "fixed":
        ds.attrs["featureType"] = "timeSeries"
    else:
        ds.attrs["featureType"] = "trajectory"

    # Add time_coverage_start and time_coverage_end
    ds.attrs["time_coverage_start"] = str(ds["time"].data[0])
    ds.attrs["time_coverage_end"] = str(ds["time"].data[-1])

    # DISDRODDB attributes
    # - Add DISDRODB processing info
    now = datetime.datetime.utcnow()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    ds.attrs["disdrodb_processing_date"] = current_time
    # - Add DISDRODB product and version
    ds.attrs["disdrodb_product_version"] = PRODUCT_VERSION
    ds.attrs["disdrodb_software_version"] = SOFTWARE_VERSION
    ds.attrs["disdrodb_product_level"] = product_level

    return ds


####-------------------------------------------------------------------------.
#### Coordinates information


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
    return d


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


def get_n_diameter_bins(sensor_name):
    """Get the number of diameter bins."""
    # Retrieve number of bins
    diameter_dict = get_diameter_bins_dict(sensor_name)
    n_diameter_bins = len(diameter_dict["center"])
    return n_diameter_bins


def get_n_velocity_bins(sensor_name):
    """Get the number of velocity bins."""
    # Retrieve number of bins
    velocity_dict = get_velocity_bins_dict(sensor_name)
    if velocity_dict is None:
        n_velocity_bins = 0
    else:
        n_velocity_bins = len(velocity_dict["center"])
    return n_velocity_bins


####-------------------------------------------------------------------------.
#### Encodings


def get_l0a_dtype(sensor_name: str) -> dict:
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

    # Note: This function could extract the info from l0a_encodings in future.
    d = read_config_yml(sensor_name=sensor_name, filename="l0a_encodings.yml")
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
    d = read_config_yml(sensor_name=sensor_name, filename="l0a_encodings.yml")
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

    d = read_config_yml(sensor_name=sensor_name, filename="l0b_encodings.yml")

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
            print(f"Set contiguous=False for variable {var} because chunksizes is defined!")

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


####-------------------------------------------------------------------------.
#### L0B processing tools


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
    n_diameter_bins = get_n_diameter_bins(sensor_name)
    n_velocity_bins = get_n_velocity_bins(sensor_name)
    # Define the dictionary
    dims_size_dict = {
        "diameter_bin_center": n_diameter_bins,
        "velocity_bin_center": n_velocity_bins,
    }
    return dims_size_dict


def get_raw_array_dims_order(sensor_name: str) -> dict:
    """Get the dimension order of the raw fields.

    The order of dimension specified for raw_drop_number controls the
    reshaping of the precipitation raw spectrum.

    Examples:
        OTT Parsivel spectrum [v1d1 ... v1d32, v2d1, ..., v2d32]
        --> dimension_order = ["velocity_bin_center", "diameter_bin_center"]
        Thies LPM spectrum [v1d1 ... v20d1, v1d2, ..., v20d2]
        --> dimension_order = ["diameter_bin_center", "velocity_bin_center"]

    Parameters
    ----------
    sensor_name : str
        Name of the sensor

    Returns
    -------
    dict
        Dimension order dictionary

    """
    # Retrieve data format dictionary
    data_format = get_data_format_dict(sensor_name)
    # Retrieve the dimension order for each array variable
    dim_dict = {}
    for var, var_dict in data_format.items():
        if "dimension_order" in var_dict:
            dim_dict[var] = var_dict["dimension_order"]
    return dim_dict


def get_raw_array_nvalues(sensor_name: str) -> dict:
    """Get a dictionary with the number of values expected for each raw array.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Field definition.
    """
    # Retrieve data format dictionary
    data_format = get_data_format_dict(sensor_name)
    # Retrieve the number of values for each array variable
    nvalues_dict = {}
    for var, var_dict in data_format.items():
        if "n_values" in var_dict:
            nvalues_dict[var] = var_dict["n_values"]
    return nvalues_dict


def get_variables_dimension(sensor_name: str):
    """Returns a dictionary with the variable dimensions of a L0B product."""
    encoding_dict = get_L0B_encodings_dict(sensor_name)
    variables = list(encoding_dict.keys())
    raw_field_dims = get_raw_array_dims_order(sensor_name)
    var_dim_dict = {}
    for var in variables:
        chunk_sizes = encoding_dict[var]["chunksizes"]
        if len(chunk_sizes) == 1:
            var_dim_dict[var] = ["time"]
        else:
            var_dim_dict[var] = raw_field_dims[var] + ["time"]
    return var_dim_dict


####-------------------------------------------------------------------------.
#### Valid names


def get_valid_variable_names(sensor_name):
    """Get list of valid variables."""
    variables = list(get_L0B_encodings_dict(sensor_name).keys())
    return variables


def get_valid_dimension_names(sensor_name):
    """Get list of valid dimension names."""
    # Retrieve dimension order dictionary
    dims_dict = get_raw_array_dims_order(sensor_name=sensor_name)
    # Retrieve possible dimensions
    list_dimensions = list(dims_dict.values())  # for each array variable
    list_dimensions = [item for sublist in list_dimensions for item in sublist]
    valid_dims = np.unique(list_dimensions).tolist()
    dimensions = ["time"] + valid_dims
    return dimensions


def get_valid_coordinates_names(sensor_name):
    """Get list of valid coordinates."""
    # Define diameter and velocity coordinates
    velocity_coordinates = [
        "velocity_bin_center",
        "velocity_bin_width",
        "velocity_bin_lower",
        "velocity_bin_upper",
    ]
    diameter_coordinates = [
        "diameter_bin_center",
        "diameter_bin_width",
        "diameter_bin_lower",
        "diameter_bin_upper",
    ]
    # Define common coordinates
    coordinates = [
        "time",
        "latitude",
        "longitude",
        # "altitude",
        # crs,
    ]
    # Since diameter is always present, add to valid coordinates
    coordinates = coordinates + diameter_coordinates
    # Add velocity if velocity_bin_center is a valid dimension
    valid_dims = get_valid_dimension_names(sensor_name)
    if "velocity_bin_center" in valid_dims:
        coordinates = coordinates + velocity_coordinates
    # Return valid coordinates
    return coordinates


def get_valid_names(sensor_name):
    variables = get_valid_variable_names(sensor_name)
    coordinates = get_valid_dimension_names(sensor_name)
    dimensions = get_valid_coordinates_names(sensor_name)
    names = np.unique(variables + coordinates + dimensions).tolist()
    return names


# -----------------------------------------------------------------------------.
