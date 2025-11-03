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
"""Include functions helping for DISDRODB product manipulations."""

import numpy as np

from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.utils.xarray import unstack_datarray_dimension


def filter_diameter_bins(ds, minimum_diameter=None, maximum_diameter=None):
    """
    Filter the dataset to include only diameter bins within specified bounds.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing diameter bin data.
    minimum_diameter : float, optional
        The minimum diameter to be included, in millimeters.
        Defaults to the minimum value in `ds["diameter_bin_lower"]`.
    maximum_diameter : float, optional
        The maximum diameter to be included, in millimeters.
        Defaults to the maximum value in `ds["diameter_bin_upper"]`.

    Returns
    -------
    xarray.Dataset
        The filtered dataset containing only the specified diameter bins.
    """
    # Put data into memory
    ds["diameter_bin_lower"] = ds["diameter_bin_lower"].compute()
    ds["diameter_bin_upper"] = ds["diameter_bin_upper"].compute()

    # Initialize default arguments
    if minimum_diameter is None:
        minimum_diameter = ds["diameter_bin_lower"].min().item()
    if maximum_diameter is None:
        maximum_diameter = ds["diameter_bin_upper"].max().item()

    # Select bins which overlap the specified diameters
    valid_indices = np.logical_and(
        ds["diameter_bin_upper"] > minimum_diameter,
        ds["diameter_bin_lower"] < maximum_diameter,
    )
    ds = ds.isel({DIAMETER_DIMENSION: valid_indices})

    if ds.sizes[DIAMETER_DIMENSION] == 0:
        msg = f"Filtering using {minimum_diameter=} removes all diameter bins."
        raise ValueError(msg)
    return ds


def filter_velocity_bins(ds, minimum_velocity=None, maximum_velocity=None):
    """
    Filter the dataset to include only velocity bins within specified bounds.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing velocity bin data.
    minimum_velocity : float, optional
        The minimum velocity to include in the filter, in meters per second.
        Defaults to the minimum value in `ds["velocity_bin_lower"]`.
    maximum_velocity : float, optional
        The maximum velocity to include in the filter, in meters per second.
        Defaults to the maximum value in `ds["velocity_bin_upper"]`.

    Returns
    -------
    xarray.Dataset
        The filtered dataset containing only the specified velocity bins.
    """
    # Put data into memory
    ds["velocity_bin_lower"] = ds["velocity_bin_lower"].compute()
    ds["velocity_bin_upper"] = ds["velocity_bin_upper"].compute()

    # Initialize default arguments
    if minimum_velocity is None:
        minimum_velocity = ds["velocity_bin_lower"].min().item()
    if maximum_velocity is None:
        maximum_velocity = ds["velocity_bin_upper"].max().item()

    # Select bins which overlap the specified velocities
    valid_indices = np.logical_and(
        ds["velocity_bin_upper"] > minimum_velocity,
        ds["velocity_bin_lower"] < maximum_velocity,
    )

    ds = ds.isel({VELOCITY_DIMENSION: valid_indices})
    if ds.sizes[VELOCITY_DIMENSION] == 0:
        msg = f"Filtering using {minimum_velocity=} removes all velocity bins."
        raise ValueError(msg)
    return ds


def get_diameter_bin_edges(ds):
    """Retrieve diameter bin edges."""
    bin_edges = np.append(ds["diameter_bin_lower"].to_numpy(), ds["diameter_bin_upper"].to_numpy()[-1])
    return bin_edges


def get_velocity_bin_edges(ds):
    """Retrieve velocity bin edges."""
    bin_edges = np.append(ds["velocity_bin_lower"].to_numpy(), ds["velocity_bin_upper"].to_numpy()[-1])
    return bin_edges


def convert_from_decibel(x):
    """Convert dB to unit."""
    return np.power(10.0, 0.1 * x)  # x/10


def convert_to_decibel(x):
    """Convert unit to dB."""
    return 10 * np.log10(x)


def unstack_radar_variables(ds):
    """Unstack radar variables."""
    from disdrodb.scattering import RADAR_VARIABLES

    for var in RADAR_VARIABLES:
        if var in ds:
            ds_unstack = unstack_datarray_dimension(ds[var], dim="frequency", prefix="", suffix="_")
            ds.update(ds_unstack)
            ds = ds.drop_vars(var)
    if "frequency" in ds.dims:
        ds = ds.drop_dims("frequency")
    return ds


def get_diameter_coords_dict_from_bin_edges(diameter_bin_edges):
    """Get dictionary with all relevant diameter coordinates."""
    if np.size(diameter_bin_edges) < 2:
        raise ValueError("Expecting at least 2 values defining bin edges.")
    diameter_bin_center = diameter_bin_edges[:-1] + np.diff(diameter_bin_edges) / 2
    diameter_bin_width = np.diff(diameter_bin_edges)
    diameter_bin_lower = diameter_bin_edges[:-1]
    diameter_bin_upper = diameter_bin_edges[1:]
    coords_dict = {
        "diameter_bin_center": (DIAMETER_DIMENSION, diameter_bin_center),
        "diameter_bin_width": (DIAMETER_DIMENSION, diameter_bin_width),
        "diameter_bin_lower": (DIAMETER_DIMENSION, diameter_bin_lower),
        "diameter_bin_upper": (DIAMETER_DIMENSION, diameter_bin_upper),
    }
    return coords_dict


def resample_drop_number_concentration(drop_number_concentration, diameter_bin_edges, method="linear"):
    """Resample drop number concentration N(D) DataArray to high resolution diameter bins."""
    diameters_bin_center = diameter_bin_edges[:-1] + np.diff(diameter_bin_edges) / 2

    da = drop_number_concentration.interp(coords={"diameter_bin_center": diameters_bin_center}, method=method)
    coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
    da = da.assign_coords(coords_dict)
    return da
