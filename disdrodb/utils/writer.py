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
"""DISDRODB product writers."""

import os

import xarray as xr

from disdrodb.utils.attrs import set_disdrodb_attrs
from disdrodb.utils.directories import create_directory, remove_if_exists


def write_product(ds: xr.Dataset, filepath: str, product: str, force: bool = False) -> None:
    """Save the xarray dataset into a NetCDF file.

    Parameters
    ----------
    ds  : xarray.Dataset
        Input xarray dataset.
    filepath : str
        Output file path.
    product: str
        DISDRODB product name.
    force : bool, optional
        Whether to overwrite existing data.
        If ``True``, overwrite existing data into destination directories.
        If ``False``, raise an error if there are already data into destination directories. This is the default.
    """
    # Create station directory if does not exist
    create_directory(os.path.dirname(filepath))

    # Check if the file already exists
    # - If force=True --> Remove it
    # - If force=False --> Raise error
    remove_if_exists(filepath, force=force)

    # Update attributes
    ds = set_disdrodb_attrs(ds, product=product)

    # Write netcdf
    ds.to_netcdf(filepath, engine="netcdf4")
