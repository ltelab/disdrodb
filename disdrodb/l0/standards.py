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
"""Retrieve L0 sensor standards."""

import logging

import numpy as np

from disdrodb.api.checks import check_sensor_name
from disdrodb.api.configs import read_config_file

logger = logging.getLogger(__name__)


####-------------------------------------------------------------------------.
#### Sensor variables


def get_sensor_logged_variables(sensor_name: str) -> list:
    """Get the sensor logged variables list.

    Parameters
    ----------
    sensor_name : str
         Name of the sensor.

    Returns
    -------
    list
        List of the variables logged by the sensor.
    """
    return list(get_data_format_dict(sensor_name).keys())


def allowed_l0_variables(sensor_name: str) -> list:
    """Get the list of allowed L0 variables for a given sensor."""
    sensor_variables = list(get_l0a_dtype(sensor_name))
    weather_variables = ["air_temperature", "relative_humidity", "wind_speed", "wind_direction"]
    allowed_variables = [*sensor_variables, *weather_variables, "time", "latitude", "longitude", "altitude"]
    allowed_variables = sorted(np.unique(allowed_variables).tolist())
    return allowed_variables


####--------------------------------------------------------------------------.
#### Variables validity dictionary


def _ensure_list_value(value):
    """Ensure the output value is a list."""
    if not isinstance(value, list):
        value = [value]
    return value


def get_data_format_dict(sensor_name: str) -> dict:
    """Get a dictionary containing the data format of each sensor variable.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Data format of each sensor variable.
    """
    return read_config_file(sensor_name=sensor_name, product="L0A", filename="raw_data_format.yml")


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
    for k in data_format_dict:
        data_range = data_format_dict[k].get("data_range", None)
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
    for k in data_format_dict:
        nan_flags = data_format_dict[k].get("nan_flags", None)
        if nan_flags is not None:
            dict_nan_flags[k] = _ensure_list_value(nan_flags)
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
    for k in data_format_dict:
        valid_values = data_format_dict[k].get("valid_values", None)
        if valid_values is not None:
            dict_valid_values[k] = _ensure_list_value(valid_values)
    return dict_valid_values


####--------------------------------------------------------------------------.
#### Get variable string format
def get_field_ndigits_natural_dict(sensor_name: str) -> dict:
    """Get number of digits on the left side of the comma from the instrument default string standards.

    Example: 123,45 -> 123 --> 3 natural digits.

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

    Example: 123,45 -> 45 --> 2 decimal digits.

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
#### Variable CF Attributes


def get_l0b_cf_attrs_dict(sensor_name: str) -> dict:
    """Get a dictionary containing the CF attributes of each sensor variable.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        CF attributes of each sensor variable.
        For each variable, the 'units', 'description', and 'long_name' attributes are specified.
    """
    return read_config_file(sensor_name=sensor_name, product="L0A", filename="l0b_cf_attrs.yml")


####-------------------------------------------------------------------------.
#### Bin Coordinates Information


def get_diameter_bins_dict(sensor_name: str) -> dict:
    """Get dictionary with ``sensor_name`` diameter bins information.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Sensor diameter bins information.
    """
    d = read_config_file(sensor_name=sensor_name, product="L0A", filename="bins_diameter.yml")
    return d


def get_diameter_bin_center(sensor_name: str) -> list:
    """Get diameter bin center.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    list
        Diameter bin center.
    """
    diameter_dict = get_diameter_bins_dict(sensor_name)
    diameter_bin_center = list(diameter_dict["center"].values())
    return diameter_bin_center


def get_diameter_bin_lower(sensor_name: str) -> list:
    """Get diameter bin lower bound.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    list
        Diameter bin lower bound.
    """
    diameter_dict = get_diameter_bins_dict(sensor_name)
    lower_bounds = [v[0] for v in diameter_dict["bounds"].values()]
    return lower_bounds


def get_diameter_bin_upper(sensor_name: str) -> list:
    """Get diameter bin upper bound.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    list
        Diameter bin upper bound.
    """
    diameter_dict = get_diameter_bins_dict(sensor_name)
    upper_bounds = [v[1] for v in diameter_dict["bounds"].values()]
    return upper_bounds


def get_diameter_bin_width(sensor_name: str) -> list:
    """Get diameter bin width.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    list
        Diameter bin width.
    """
    diameter_dict = get_diameter_bins_dict(sensor_name)
    diameter_bin_width = list(diameter_dict["width"].values())
    return diameter_bin_width


def get_velocity_bins_dict(sensor_name: str) -> dict:
    """Get velocity with ``sensor_name`` diameter bins information.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Sensor velocity bins information.
    """
    d = read_config_file(sensor_name=sensor_name, product="L0A", filename="bins_velocity.yml")
    return d


def get_velocity_bin_center(sensor_name: str) -> list:
    """Get velocity bin center.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    list
        Velocity bin center.
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
        Name of the sensor.

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
        Name of the sensor.

    Returns
    -------
    list
        Velocity bin upper bound.
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
        Name of the sensor.

    Returns
    -------
    list
        Velocity bin width.
    """
    velocity_dict = get_velocity_bins_dict(sensor_name)
    if velocity_dict is not None:
        velocity_bin_width = list(velocity_dict["width"].values())
    else:
        return None
    return velocity_bin_width


def get_bin_coords_dict(sensor_name: str) -> dict:
    """Retrieve diameter (and velocity) bin coordinates.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with coordinates arrays.
    """
    check_sensor_name(sensor_name)
    coords = {}
    # Retrieve diameter coords
    coords["diameter_bin_center"] = get_diameter_bin_center(sensor_name=sensor_name)
    coords["diameter_bin_lower"] = (
        ["diameter_bin_center"],
        get_diameter_bin_lower(sensor_name=sensor_name),
    )
    coords["diameter_bin_upper"] = (
        ["diameter_bin_center"],
        get_diameter_bin_upper(sensor_name=sensor_name),
    )
    coords["diameter_bin_width"] = (
        ["diameter_bin_center"],
        get_diameter_bin_width(sensor_name=sensor_name),
    )
    # Retrieve velocity coords (if available)
    if get_velocity_bin_center(sensor_name=sensor_name) is not None:
        coords["velocity_bin_center"] = (
            ["velocity_bin_center"],
            get_velocity_bin_center(sensor_name=sensor_name),
        )
        coords["velocity_bin_lower"] = (
            ["velocity_bin_center"],
            get_velocity_bin_lower(sensor_name=sensor_name),
        )
        coords["velocity_bin_upper"] = (
            ["velocity_bin_center"],
            get_velocity_bin_upper(sensor_name=sensor_name),
        )
        coords["velocity_bin_width"] = (
            ["velocity_bin_center"],
            get_velocity_bin_width(sensor_name=sensor_name),
        )
    return coords


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
    n_velocity_bins = 0 if velocity_dict is None else len(velocity_dict["center"])
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
        Dictionary with the L0A dtype.
    """
    # Note: This function could extract the info from l0a_encodings in future.
    d = read_config_file(sensor_name=sensor_name, product="L0A", filename="l0a_encodings.yml")
    return d


def get_l0a_encodings_dict(sensor_name: str) -> dict:
    """Get a dictionary containing the L0A encodings.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        L0A encodings.
    """
    # - l0a_encodings.yml currently specify only the dtype. This could be expanded in the future.
    d = read_config_file(sensor_name=sensor_name, product="L0A", filename="l0a_encodings.yml")
    return d


def _check_contiguous_chunksize_agrees(encoding_dict, var):
    chunksizes = encoding_dict[var].get("chunksizes", None)
    contiguous = encoding_dict[var].get("contiguous", False)
    if isinstance(chunksizes, list) and len(chunksizes) >= 1 and contiguous:
        raise ValueError(
            f"Invalid encodings for variable {var}. 'chunksizes' are specified but 'contiguous' is set to True !",
        )


def _if_no_chunksizes_set_contiguous(encoding_dict, var):
    if isinstance(encoding_dict[var].get("chunksizes", None), type(None)) and not encoding_dict[var].get(
        "contiguous",
        False,
    ):
        encoding_dict[var]["contiguous"] = True
        print(f"Set contiguous=True for variable {var} because chunksizes=None")
    return encoding_dict


def _set_contiguous_encoding_options(encoding_dict, var):
    if encoding_dict[var].get("contiguous", False):
        encoding_dict[var]["fletcher32"] = False
        encoding_dict[var]["zlib"] = False
        print(f"Set fletcher32=False for variable {var} because contiguous=True")
        print(f"Set zlib=False for variable {var} because contiguous=True")
    return encoding_dict


def _ensure_valid_chunksizes(encoding_dict, var):
    if not isinstance(encoding_dict[var].get("chunksizes", None), type(None)):
        encoding_dict[var]["chunksizes"] = _ensure_list_value(encoding_dict[var]["chunksizes"])
        encoding_dict[var]["contiguous"] = False
    else:
        encoding_dict[var]["chunksizes"] = []
    return encoding_dict


def _ensure_valid_netcdf_encoding_dict(encoding_dict):
    for var in encoding_dict:
        _check_contiguous_chunksize_agrees(encoding_dict, var)
        # Ensure valid arguments for contiguous (unchunked) arrays
        encoding_dict = _if_no_chunksizes_set_contiguous(encoding_dict, var)
        encoding_dict = _set_contiguous_encoding_options(encoding_dict, var)
        # Ensure chunksizes is a list
        encoding_dict = _ensure_valid_chunksizes(encoding_dict, var)
    return encoding_dict


def get_l0b_encodings_dict(sensor_name: str) -> dict:
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
    encoding_dict = read_config_file(sensor_name=sensor_name, product="L0A", filename="l0b_encodings.yml")
    encoding_dict = _ensure_valid_netcdf_encoding_dict(encoding_dict)
    return encoding_dict


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

    Examples
    --------
        OTT Parsivel spectrum [d1v1 ... d32v1, d1v2, ..., d32v2] (diameter increases first)
        --> dimension_order = ["velocity_bin_center", "diameter_bin_center"]
        Thies LPM spectrum [v1d1 ... v20d1, v1d2, ..., v20d2]  (velocity increases first)
        --> dimension_order = ["diameter_bin_center", "velocity_bin_center"]
        PWS 100 spectrum [d1v1 ... d1v34, d2v1, ..., d2v34] (velocity increases first)
        --> dimension_order = ["diameter_bin_center", "velocity_bin_center"]

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dimension order dictionary.

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
    encoding_dict = get_l0b_encodings_dict(sensor_name)
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
#### Valid DISDRODB L0B names


def get_valid_variable_names(sensor_name):
    """Get list of valid variables."""
    encoding_dict = read_config_file(sensor_name=sensor_name, product="L0A", filename="l0b_encodings.yml")
    variables = list(encoding_dict.keys())
    return variables


def get_valid_dimension_names(sensor_name):
    """Get list of valid dimension names for DISDRODB L0B."""
    # Retrieve dimension order dictionary
    dims_dict = get_raw_array_dims_order(sensor_name=sensor_name)
    # Retrieve possible dimensions
    list_dimensions = list(dims_dict.values())  # for each array variable
    list_dimensions = [item for sublist in list_dimensions for item in sublist]
    valid_dims = np.unique(list_dimensions).tolist()
    dimensions = ["time", *valid_dims]
    return dimensions


def get_valid_coordinates_names(sensor_name):
    """Get list of valid coordinates for DISDRODB L0B."""
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
    """Return the list of valid variable and coordinates names for DISDRODB L0B."""
    variables = get_valid_variable_names(sensor_name)
    coordinates = get_valid_dimension_names(sensor_name)
    dimensions = get_valid_coordinates_names(sensor_name)
    names = np.unique(variables + coordinates + dimensions).tolist()
    return names


# -----------------------------------------------------------------------------.
