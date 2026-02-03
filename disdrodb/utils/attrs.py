# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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

import os

from disdrodb.api.checks import get_current_utc_time
from disdrodb.constants import ARCHIVE_VERSION, CONVENTIONS, COORDINATES, SOFTWARE_VERSION
from disdrodb.utils.warnings import suppress_warnings
from disdrodb.utils.yaml import read_yaml

####---------------------------------------------------------------------.
#### Variable and coordinates attributes


def get_attrs_dict():
    """Get attributes dictionary for DISDRODB product variables and coordinates."""
    import disdrodb

    configs_path = os.path.join(disdrodb.package_dir, "etc", "configs")
    attrs_dict = read_yaml(os.path.join(configs_path, "attributes.yaml"))
    return attrs_dict


def set_attrs(ds, attrs_dict):
    """Set attributes to the variables and coordinates of the xr.Dataset."""
    for var in attrs_dict:
        if var in ds:
            ds[var].attrs.update(attrs_dict[var])
    return ds


####---------------------------------------------------------------------.
#### Coordinates attributes


def set_coordinate_attributes(ds):
    """Set coordinates attributes."""
    # Get attributes dictionary
    attrs_dict = get_attrs_dict()
    coords_dict = {coord: attrs_dict[coord] for coord in COORDINATES if coord in attrs_dict}
    # Set attributes
    ds = set_attrs(ds, coords_dict)
    return ds


####-------------------------------------------------------------------------.
#### DISDRODB Global Attributes


def update_disdrodb_attrs(ds, product: str):
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
    xarray.Dataset
        Dataset.
    """
    attrs = ds.attrs.copy()

    # ----------------------------------------------
    # Drop metadata not relevant for DISDRODB products
    keys_to_drop = [
        "disdrodb_reader",
        "disdrodb_data_url",
        "raw_data_glob_pattern",
        "raw_data_format",
    ]
    for key in keys_to_drop:
        _ = attrs.pop(key, None)

    # ----------------------------------------------
    # Add time_coverage_start and time_coverage_end
    if "time" in ds.dims:
        encoding = ds["time"].encoding
        ds["time"] = ds["time"].dt.floor("s")  # ensure no sub-second values
        with suppress_warnings():
            ds["time"] = ds["time"].astype("datetime64[s]")
        ds["time"].encoding = encoding  # otherwise time encoding get lost !

        attrs["time_coverage_start"] = str(ds["time"].data[0])
        attrs["time_coverage_end"] = str(ds["time"].data[-1])

    # ----------------------------------------------
    # Set DISDRODDB attributes
    # - Add DISDRODB processing info
    now = get_current_utc_time()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    attrs["disdrodb_processing_date"] = current_time
    # - Add DISDRODB product and version
    attrs["disdrodb_product_version"] = ARCHIVE_VERSION
    attrs["disdrodb_software_version"] = SOFTWARE_VERSION
    attrs["disdrodb_product"] = product

    # ----------------------------------------------
    # Finalize attributes dictionary
    # - Sort attributes alphabetically
    attrs = dict(sorted(attrs.items()))
    # - Set attributes
    ds.attrs = attrs
    return ds


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
    xarray.Dataset
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
