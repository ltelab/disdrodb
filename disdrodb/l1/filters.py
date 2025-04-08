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
"""Utilities for filtering the disdrometer raw drop spectra."""

import numpy as np
import xarray as xr


def filter_diameter_bins(ds, minimum_diameter=None, maximum_diameter=None):
    """
    Filter the dataset to include only diameter bins within specified bounds.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing diameter bin data.
    minimum_diameter : float, optional
        The minimum diameter to include in the filter, in millimeters.
        Defaults to the minimum value in `ds["diameter_bin_lower"]`.
    maximum_diameter : float, optional
        The maximum diameter to include in the filter, in millimeters.
        Defaults to the maximum value in `ds["diameter_bin_upper"]`.

    Returns
    -------
    xarray.Dataset
        The filtered dataset containing only the specified diameter bins.
    """
    # Initialize default arguments
    if minimum_diameter is None:
        minimum_diameter = ds["diameter_bin_lower"].min().item()
    if maximum_diameter is None:
        maximum_diameter = ds["diameter_bin_upper"].max().item()
    # Select valid bins
    valid_indices = np.logical_and(
        ds["diameter_bin_lower"] >= minimum_diameter,
        ds["diameter_bin_upper"] <= maximum_diameter,
    )
    ds = ds.isel({"diameter_bin_center": valid_indices})
    # Update history
    history = ds.attrs.get("history", "")
    ds.attrs["history"] = (
        history + f" Selected drops with diameters between {minimum_diameter} and {maximum_diameter} mm \n"
    )
    return ds


def filter_velocity_bins(ds, minimum_velocity=0, maximum_velocity=12):
    """
    Filter the dataset to include only velocity bins within specified bounds.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing velocity bin data.
    minimum_velocity : float, optional
        The minimum velocity to include in the filter, in meters per second.
        Defaults to 0 m/s.
    maximum_velocity : float, optional
        The maximum velocity to include in the filter, in meters per second.
        Defaults to 12 m/s.

    Returns
    -------
    xarray.Dataset
        The filtered dataset containing only the specified velocity bins.
    """
    # Initialize default arguments
    if minimum_velocity is None:
        minimum_velocity = ds["velocity_bin_lower"].min().item()
    if maximum_velocity is None:
        maximum_velocity = ds["velocity_bin_upper"].max().item()
    # Select valid bins
    valid_indices = np.logical_and(
        ds["velocity_bin_lower"] >= minimum_velocity,
        ds["velocity_bin_upper"] <= maximum_velocity,
    )
    ds = ds.isel({"velocity_bin_center": valid_indices})
    # Update history
    history = ds.attrs.get("history", "")
    ds.attrs["history"] = (
        history + f" Selected drops with fall velocity between {minimum_velocity} and {maximum_velocity} m/s \n"
    )
    return ds


def define_spectrum_mask(
    drop_number,
    fall_velocity,
    above_velocity_fraction=None,
    above_velocity_tolerance=None,
    below_velocity_fraction=None,
    below_velocity_tolerance=None,
    small_diameter_threshold=1,  # 1,   # 2
    small_velocity_threshold=2.5,  # 2.5, # 3
    maintain_smallest_drops=False,
):
    """Define a mask for the drop spectrum based on fall velocity thresholds.

    Parameters
    ----------
    drop_number : xarray.DataArray
        Array of drop counts per diameter and velocity bins.
    fall_velocity : array-like
        The expected terminal fall velocities for drops of given sizes.
    above_velocity_fraction : float, optional
        Fraction of terminal fall velocity above which drops are considered too fast.
        Either specify ``above_velocity_fraction`` or ``above_velocity_tolerance``.
    above_velocity_tolerance : float, optional
        Absolute tolerance above which drops terminal fall velocities are considered too fast.
        Either specify ``above_velocity_fraction`` or ``above_velocity_tolerance``.
    below_velocity_fraction : float, optional
        Fraction of terminal fall velocity below which drops are considered too slow.
        Either specify ``below_velocity_fraction`` or ``below_velocity_tolerance``.
    below_velocity_tolerance : float, optional
        Absolute tolerance below which drops terminal fall velocities are considered too slow.
         Either specify ``below_velocity_fraction`` or ``below_velocity_tolerance``.
    maintain_smallest : bool, optional
        If True, ensures that the small drops in the spectrum are retained in the mask.
        The smallest drops are characterized by ``small_diameter_threshold``
        and ``small_velocity_threshold`` arguments.
        Defaults to False.
    small_diameter_threshold : float, optional
        The diameter threshold to use for keeping the smallest drop.
        Defaults to 1 mm.
    small_velocity_threshold : float, optional
        The fall velocity threshold to use for keeping the smallest drops.
        Defaults to 2.5 m/s.

    Returns
    -------
    xarray.DataArray
        A boolean mask array indicating valid bins according to the specified criteria.

    """
    # Ensure it creates a 2D mask if the fall_velocity does not vary over time
    if "time" in drop_number.dims and "time" not in fall_velocity.dims:
        drop_number = drop_number.isel(time=0)

    # Check arguments
    if above_velocity_fraction is not None and above_velocity_tolerance is not None:
        raise ValueError("Either specify 'above_velocity_fraction' or 'above_velocity_tolerance'.")
    if below_velocity_fraction is not None and below_velocity_tolerance is not None:
        raise ValueError("Either specify 'below_velocity_fraction' or 'below_velocity_tolerance'.")

    # Define above/below velocity thresholds
    if above_velocity_fraction is not None:
        above_fall_velocity = fall_velocity * (1 + above_velocity_fraction)
    elif above_velocity_tolerance is not None:
        above_fall_velocity = fall_velocity + above_velocity_tolerance
    else:
        above_fall_velocity = np.inf
    if below_velocity_fraction is not None:
        below_fall_velocity = fall_velocity * (1 - below_velocity_fraction)
    elif below_velocity_tolerance is not None:
        below_fall_velocity = fall_velocity - below_velocity_tolerance
    else:
        below_fall_velocity = 0

    # Define velocity 2D array
    velocity_lower = xr.ones_like(drop_number) * drop_number["velocity_bin_lower"]
    velocity_upper = xr.ones_like(drop_number) * drop_number["velocity_bin_upper"]

    # Define mask
    mask = np.logical_and(
        np.logical_or(velocity_lower >= below_fall_velocity, velocity_upper >= below_fall_velocity),
        np.logical_or(velocity_lower <= above_fall_velocity, velocity_upper <= above_fall_velocity),
    )

    # Maintant smallest drops
    if maintain_smallest_drops:
        mask_smallest = np.logical_and(
            drop_number["diameter_bin_upper"] < small_diameter_threshold,
            drop_number["velocity_bin_upper"] < small_velocity_threshold,
        )
        mask = np.logical_or(mask, mask_smallest)

    return mask
