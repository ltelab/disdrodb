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

from disdrodb import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.l1.encoding_attrs import get_attrs_dict, get_encoding_dict
from disdrodb.l1.fall_velocity import get_raindrop_fall_velocity
from disdrodb.l1.filters import define_spectrum_mask, filter_diameter_bins, filter_velocity_bins
from disdrodb.l1.resampling import add_sample_interval
from disdrodb.l1_env.routines import load_env_dataset
from disdrodb.l2.empirical_dsd import (  # TODO: maybe move out of L2
    compute_qc_bins_metrics,
    get_min_max_diameter,
)
from disdrodb.utils.attrs import set_attrs
from disdrodb.utils.encoding import set_encodings
from disdrodb.utils.time import ensure_sample_interval_in_seconds, infer_sample_interval


def generate_l1(
    ds,
    # Fall velocity option
    fall_velocity_method="Beard1976",
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
    """Generate the DISDRODB L1 dataset from the DISDRODB L0C dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        DISDRODB L0C dataset.
    fall_velocity_method : str, optional
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
        DISRODB L1 dataset.
    """
    # Take as input an L0 !

    # Retrieve source attributes
    attrs = ds.attrs.copy()

    # Determine if the velocity dimension is available
    has_velocity_dimension = VELOCITY_DIMENSION in ds.dims

    # Initialize L2 dataset
    ds_l1 = xr.Dataset()

    # Retrieve sample interval
    # --> sample_interval is a coordinate of L0C products
    if "sample_interval" in ds:
        sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"].data)
    else:
        # This line is not called in the DISDRODB processing chain !
        sample_interval = infer_sample_interval(ds, verbose=False)

    # Re-add sample interval as coordinate (in seconds)
    ds = add_sample_interval(ds, sample_interval=sample_interval)

    # ---------------------------------------------------------------------------
    # Retrieve ENV dataset or take defaults
    # --> Used only for Beard fall velocity currently !
    ds_env = load_env_dataset(ds)

    # -------------------------------------------------------------------------------------------
    # Filter dataset by diameter and velocity bins
    # - Filter diameter bins
    ds = filter_diameter_bins(ds=ds, minimum_diameter=minimum_diameter, maximum_diameter=maximum_diameter)
    # - Filter velocity bins
    if has_velocity_dimension:
        ds = filter_velocity_bins(ds=ds, minimum_velocity=minimum_velocity, maximum_velocity=maximum_velocity)

    # -------------------------------------------------------------------------------------------
    # Compute fall velocity
    fall_velocity = get_raindrop_fall_velocity(
        diameter=ds["diameter_bin_center"],
        method=fall_velocity_method,
        ds_env=ds_env,  # mm
    )

    # Add fall velocity
    ds_l1["fall_velocity"] = fall_velocity

    # -------------------------------------------------------------------------------------------
    # Define filtering mask according to fall velocity
    if has_velocity_dimension:
        mask = define_spectrum_mask(
            drop_number=ds["raw_drop_number"],
            fall_velocity=fall_velocity,
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
        drop_number = ds["raw_drop_number"].where(mask)  # 2D (diameter, velocity)
        drop_counts = drop_number.sum(dim=VELOCITY_DIMENSION)  # 1D (diameter)
        drop_counts_raw = ds["raw_drop_number"].sum(dim=VELOCITY_DIMENSION)  # 1D (diameter)

    else:
        drop_number = ds["raw_drop_number"]  # 1D (diameter)
        drop_counts = ds["raw_drop_number"]  # 1D (diameter)
        drop_counts_raw = ds["raw_drop_number"]

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
    ds_l1["Nremoved"] = drop_counts_raw.sum(dim=DIAMETER_DIMENSION) - ds_l1["N"]

    # Add bins statistics
    ds_l1.update(compute_qc_bins_metrics(ds_l1))

    # -------------------------------------------------------------------------------------------
    # Add quality flags
    # TODO: snow_flags, insects_flag, ...

    # -------------------------------------------------------------------------------------------
    #### Add L0C coordinates that might got lost
    if "time_qc" in ds:
        ds_l1 = ds_l1.assign_coords({"time_qc": ds["time_qc"]})

    #### ----------------------------------------------------------------------------.
    #### Add encodings and attributes
    # Add variables attributes
    attrs_dict = get_attrs_dict()
    ds_l1 = set_attrs(ds_l1, attrs_dict=attrs_dict)

    # Add variables encoding
    encoding_dict = get_encoding_dict()
    ds_l1 = set_encodings(ds_l1, encoding_dict=encoding_dict)

    # Add global attributes
    ds_l1.attrs = attrs
    return ds_l1
