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
"""Core functions for DISDRODB L1 production."""

import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION, METEOROLOGICAL_VARIABLES, VELOCITY_DIMENSION
from disdrodb.l1.classification import (
    classify_raw_spectrum,
    get_temperature,
    map_precip_flag_to_precipitation_type,
)
from disdrodb.l1.resampling import add_sample_interval
from disdrodb.l1_env.routines import load_env_dataset
from disdrodb.utils.manipulations import filter_diameter_bins
from disdrodb.utils.time import ensure_sample_interval_in_seconds, infer_sample_interval
from disdrodb.utils.writer import finalize_product


def generate_l1(
    ds,
    **kwargs,  # noqa
):
    """Generate DISDRODB L1 Dataset from DISDRODB L0C Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        DISDRODB L0C dataset.

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

    # --------------------------------------------------------------------------
    # Retrieve sample interval
    # --> sample_interval is a coordinate of L0C products
    if "sample_interval" in ds:
        sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"].data)
    else:
        # This line is not called in the DISDRODB processing chain !
        sample_interval = infer_sample_interval(ds, verbose=False)

    # --------------------------------------------------------------------------
    # Retrieve ENV dataset or take defaults
    # - Used only for Beard fall velocity currently !
    # - It checks and includes default geolocation if missing
    # - For mobile disdrometer, infill missing geolocation with backward and forward filling
    ds_env = load_env_dataset(ds)

    # --------------------------------------------------------------------------
    # Initialize L1 dataset
    ds_l1 = xr.Dataset()

    # Add raw_drop_number variable to L1 dataset
    ds_l1["raw_drop_number"] = ds["raw_drop_number"]

    # Add sample interval as coordinate (in seconds)
    ds_l1 = add_sample_interval(ds_l1, sample_interval=sample_interval)

    # Add optional variables to L1 dataset
    optional_variables = ["qc_time", "qc_resampling", *METEOROLOGICAL_VARIABLES]
    for var in optional_variables:
        if var in ds:
            ds_l1[var] = ds[var]

    # --------------------------------------------------------------------------
    # Filter dataset by diameter and velocity bins
    if sensor_name in ["PARSIVEL", "PARSIVEL2"]:
        # - Remove first two bins because never reports data !
        # - Could be removed also in L2E, but we save disk space here
        # - If not removed, can alter e.g. L2M model fitting
        ds_l1 = filter_diameter_bins(ds=ds_l1, minimum_diameter=0.2495)  # it includes the 0.2495-0.3745 bin

    # --------------------------------------------------------------------------
    # If (diameter, velocity) spectrum is available, run hydrometeor classification
    if has_velocity_dimension:
        temperature, snow_temperature_upper_limit = get_temperature(ds)
        temperature = temperature.compute() if temperature is not None else None
        ds_hc = classify_raw_spectrum(
            ds=ds_l1,
            ds_env=ds_env,
            sensor_name=sensor_name,
            sample_interval=sample_interval,
            temperature=temperature,
            rain_temperature_lower_limit=-5,
            snow_temperature_upper_limit=snow_temperature_upper_limit,
        )
        ds_l1.update(ds_hc)

    # Otherwise, if no 2D spectrum, do nothing (temporary)
    # --> Specialized QC for RD-80 or ODM-470 not yet implemented
    else:
        # If OceanRain ODM470 data, translate precip_flag to precipitation_type
        if sensor_name == "ODM470" and "precip_flag" in ds:
            ds_l1["precipitation_type"] = map_precip_flag_to_precipitation_type(ds["precip_flag"])
        ds_l1["n_particles"] = ds_l1["raw_drop_number"].sum(dim=DIAMETER_DIMENSION)
        pass

    #### ----------------------------------------------------------------------.
    #### Finalize dataset
    # Add global attributes
    ds_l1.attrs = attrs

    # Add variables attributes and encodings
    ds_l1 = finalize_product(ds_l1, product="L1")
    return ds_l1
