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
"""Core functions for DISDRODB L1 production."""

import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.fall_velocity import get_rain_fall_velocity_from_ds
from disdrodb.l1.filters import define_rain_spectrum_mask, filter_diameter_bins, filter_velocity_bins
from disdrodb.l1.resampling import add_sample_interval
from disdrodb.l1_env.routines import load_env_dataset
from disdrodb.l2.empirical_dsd import (  # TODO: maybe move out of L2
    add_bins_metrics,
    get_min_max_diameter,
)
from disdrodb.utils.time import ensure_sample_interval_in_seconds, infer_sample_interval
from disdrodb.utils.writer import finalize_product


def generate_l1(
    ds,
    # Fall velocity option
    fall_velocity_model="Beard1976",
    # Diameter-Velocity Filtering Options
    minimum_diameter=0,
    maximum_diameter=10,
    minimum_velocity=0,
    maximum_velocity=12,
    above_velocity_fraction=0.5,
    above_velocity_tolerance=None,
    below_velocity_fraction=0.5,
    below_velocity_tolerance=None,
    small_diameter_threshold=1,  # 2
    small_velocity_threshold=2.5,  # 3
    maintain_smallest_drops=True,
):
    """Generate DISDRODB L1 Dataset from DISDRODB L0C Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        DISDRODB L0C dataset.
    fall_velocity_model : str, optional
        Method to compute fall velocity.
        The default method is ``"Beard1976"``.
    minimum_diameter : float, optional
        Minimum diameter for filtering. The default value is 0 mm.
    maximum_diameter : float, optional
        Maximum diameter for filtering. The default value is 10 mm.
    minimum_velocity : float, optional
        Minimum velocity for filtering. The default value is 0 m/s.
    maximum_velocity : float, optional
        Maximum velocity for filtering. The default value is 12 m/s.
    above_velocity_fraction : float, optional
        Fraction of drops above velocity threshold. The default value is 0.5.
    above_velocity_tolerance : float or None, optional
        Tolerance for above velocity filtering. The default value is ``None``.
    below_velocity_fraction : float, optional
        Fraction of drops below velocity threshold. The default value is 0.5.
    below_velocity_tolerance : float or None, optional
        Tolerance for below velocity filtering. The default value is ``None``.
    small_diameter_threshold : float, optional
        Threshold for small diameter drops. The default value is 1.
    small_velocity_threshold : float, optional
        Threshold for small velocity drops. The default value is 2.5.
    maintain_smallest_drops : bool, optional
        Whether to maintain the smallest drops. The default value is ``True``.

    Returns
    -------
    xarray.Dataset
        DISDRODB L1 dataset.
    """
    # Retrieve source attributes
    attrs = ds.attrs.copy()

    # Determine if the velocity dimension is available
    has_velocity_dimension = VELOCITY_DIMENSION in ds.dims

    # Retrieve sensor_name
    # - If not present, don't drop Parsivels first two bins
    sensor_name = attrs.get("sensor_name", "")

    # ---------------------------------------------------------------------------
    # Retrieve sample interval
    # --> sample_interval is a coordinate of L0C products
    if "sample_interval" in ds:
        sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"].data)
    else:
        # This line is not called in the DISDRODB processing chain !
        sample_interval = infer_sample_interval(ds, verbose=False)

    # ---------------------------------------------------------------------------
    # Retrieve ENV dataset or take defaults
    # - Used only for Beard fall velocity currently !
    # - It checks and includes default geolocation if missing
    # - For mobile disdrometer, infill missing geolocation with backward and forward filling
    ds_env = load_env_dataset(ds)

    # ---------------------------------------------------------------------------
    # Initialize L1 dataset
    ds_l1 = xr.Dataset()

    # Add raw_drop_number variable to L1 dataset
    ds_l1["raw_drop_number"] = ds["raw_drop_number"]

    # Add sample interval as coordinate (in seconds)
    ds_l1 = add_sample_interval(ds_l1, sample_interval=sample_interval)

    # Add optional variables to L1 dataset
    optional_variables = ["time_qc", "qc_resampling"]
    for var in optional_variables:
        if var in ds:
            ds_l1[var] = ds[var]

    # -------------------------------------------------------------------------------------------
    # Filter dataset by diameter and velocity bins
    if sensor_name in ["PARSIVEL", "PARSIVEL2"]:
        # - Remove first two bins because never reports data !
        # - If not removed, can alter e.g. L2M model fitting
        ds_l1 = filter_diameter_bins(ds=ds_l1, minimum_diameter=0.2495)  # it includes the 0.2495-0.3745 bin

    # - Filter diameter bins
    ds_l1 = filter_diameter_bins(ds=ds_l1, minimum_diameter=minimum_diameter, maximum_diameter=maximum_diameter)
    # - Filter velocity bins
    if has_velocity_dimension:
        ds_l1 = filter_velocity_bins(ds=ds_l1, minimum_velocity=minimum_velocity, maximum_velocity=maximum_velocity)

    # -------------------------------------------------------------------------------------------
    # Compute fall velocity
    ds_l1["fall_velocity"] = get_rain_fall_velocity_from_ds(ds=ds_l1, ds_env=ds_env, model=fall_velocity_model)
    fall_velocity_lower = get_rain_fall_velocity_from_ds(
        ds=ds_l1,
        ds_env=ds_env,
        model=fall_velocity_model,
        diameter="diameter_bin_lower",
    )
    fall_velocity_upper = get_rain_fall_velocity_from_ds(
        ds=ds_l1,
        ds_env=ds_env,
        model=fall_velocity_model,
        diameter="diameter_bin_upper",
    )

    # -------------------------------------------------------------------------------------------
    # Define filtering mask according to fall velocity
    if has_velocity_dimension:
        mask = define_rain_spectrum_mask(
            drop_number=ds_l1["raw_drop_number"],
            fall_velocity_lower=fall_velocity_lower,
            fall_velocity_upper=fall_velocity_upper,
            above_velocity_fraction=above_velocity_fraction,
            above_velocity_tolerance=above_velocity_tolerance,
            below_velocity_fraction=below_velocity_fraction,
            below_velocity_tolerance=below_velocity_tolerance,
            small_diameter_threshold=small_diameter_threshold,
            small_velocity_threshold=small_velocity_threshold,
            maintain_smallest_drops=maintain_smallest_drops,
        )

    # -------------------------------------------------------------------------------------------
    # Retrieve drop number and drop_counts arrays
    if has_velocity_dimension:
        drop_number = ds_l1["raw_drop_number"].where(mask, 0)  # 2D (diameter, velocity)
        drop_counts = drop_number.sum(dim=VELOCITY_DIMENSION)  # 1D (diameter)
        drop_counts_raw = ds_l1["raw_drop_number"].sum(dim=VELOCITY_DIMENSION)  # 1D (diameter)
    else:
        drop_number = ds_l1["raw_drop_number"]  # 1D (diameter)
        drop_counts = ds_l1["raw_drop_number"]  # 1D (diameter)
        drop_counts_raw = ds_l1["raw_drop_number"]

    # Add drop number and drop_counts
    ds_l1["drop_number"] = drop_number
    ds_l1["drop_counts"] = drop_counts

    # -------------------------------------------------------------------------------------------
    # Compute minimum and max drop diameter observed
    min_drop_diameter, max_drop_diameter = get_min_max_diameter(drop_counts)

    # Add drop statistics
    ds_l1["Dmin"] = min_drop_diameter
    ds_l1["Dmax"] = max_drop_diameter
    ds_l1["N"] = drop_counts.sum(dim=DIAMETER_DIMENSION)
    ds_l1["Nraw"] = drop_counts_raw.sum(dim=DIAMETER_DIMENSION)
    ds_l1["Nremoved"] = ds_l1["Nraw"] - ds_l1["N"]

    # Add bins statistics
    ds_l1 = add_bins_metrics(ds_l1)

    # -------------------------------------------------------------------------------------------
    # Add quality flags
    # TODO: snow_flags, insects_flag, ...

    #### ----------------------------------------------------------------------------.
    #### Finalize dataset
    # Add global attributes
    ds_l1.attrs = attrs

    # Add variables attributes and encodings
    ds_l1 = finalize_product(ds_l1, product="L1")
    return ds_l1
