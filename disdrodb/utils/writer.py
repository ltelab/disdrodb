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
"""DISDRODB product writers."""

import os

import xarray as xr

from disdrodb.utils.attrs import get_attrs_dict, set_attrs, set_disdrodb_attrs
from disdrodb.utils.directories import create_directory, remove_if_exists
from disdrodb.utils.encoding import get_encodings_dict, set_encodings


def finalize_product(ds, product=None) -> xr.Dataset:
    """Finalize DISDRODB product."""
    # Add variables attributes
    attrs_dict = get_attrs_dict()
    ds = set_attrs(ds, attrs_dict=attrs_dict)

    # Add variables encoding
    encodings_dict = get_encodings_dict()
    ds = set_encodings(ds, encodings_dict=encodings_dict)

    # Add DISDRODB global attributes
    # - e.g. in generate_l2_radar it inherit from input dataset !
    if product is not None:
        ds = set_disdrodb_attrs(ds, product=product)
    return ds


def write_product(ds: xr.Dataset, filepath: str, force: bool = False) -> None:
    """Save the xarray dataset into a NetCDF file.

    Parameters
    ----------
    ds  : xarray.Dataset
        Input xarray dataset.
    filepath : str
        Output file path.
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

    # Write netcdf
    ds.to_netcdf(filepath, engine="netcdf4")
