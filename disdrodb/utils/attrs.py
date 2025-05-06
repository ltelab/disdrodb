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
"""DISDRODB netCDF4 attributes utilities."""
import datetime

from disdrodb import ARCHIVE_VERSION, CONVENTIONS, SOFTWARE_VERSION

####---------------------------------------------------------------------.
#### Variable attributes


def set_attrs(ds, attrs_dict):
    """Set attributes to the variables of the xr.Dataset."""
    for var in attrs_dict:
        if var in ds:
            ds[var].attrs.update(attrs_dict[var])
    return ds


####---------------------------------------------------------------------.
#### Coordinates attributes


def get_coords_attrs_dict():
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


def set_coordinate_attributes(ds):
    """Set coordinates attributes."""
    # Get attributes dictionary
    attrs_dict = get_coords_attrs_dict()
    # Set attributes
    ds = set_attrs(ds, attrs_dict)
    return ds


####-------------------------------------------------------------------------.
#### DISDRODB Global Attributes


def set_disdrodb_attrs(ds, product: str):
    """Add DISDRODB processing information to the netCDF global attributes.

    It assumes stations metadata are already added the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset
    product: str
        DISDRODB product.

    Returns
    -------
    xarray dataset
        Dataset.
    """
    # Add dataset conventions
    ds.attrs["Conventions"] = CONVENTIONS

    # Add featureType
    if "platform_type" in ds.attrs:
        platform_type = ds.attrs["platform_type"]
        if platform_type == "fixed":
            ds.attrs["featureType"] = "timeSeries"
        else:
            ds.attrs["featureType"] = "trajectory"

    # Update DISDRODDB attributes
    ds = update_disdrodb_attrs(ds=ds, product=product)
    return ds


def update_disdrodb_attrs(ds, product: str):
    """Add DISDRODB processing information to the netCDF global attributes.

    It assumes stations metadata are already added the dataset.

    Parameters
    ----------
    ds : xarray dataset.
        Dataset
    product: str
        DISDRODB product.

    Returns
    -------
    xarray dataset
        Dataset.
    """
    # Add time_coverage_start and time_coverage_end
    ds.attrs["time_coverage_start"] = str(ds["time"].data[0])
    ds.attrs["time_coverage_end"] = str(ds["time"].data[-1])

    # DISDRODDB attributes
    # - Add DISDRODB processing info
    now = datetime.datetime.utcnow()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    ds.attrs["disdrodb_processing_date"] = current_time
    # - Add DISDRODB product and version
    ds.attrs["disdrodb_product_version"] = ARCHIVE_VERSION
    ds.attrs["disdrodb_software_version"] = SOFTWARE_VERSION
    ds.attrs["disdrodb_product"] = product
    return ds
