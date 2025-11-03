"""Utility functions to create DISDRODB products dataset templates."""

import numpy as np
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.l2.processing import generate_l2e


def create_template_dataset(with_velocity=True):
    """Create DISDRODB L1 basic dataset."""
    # Define coordinates
    time = xr.DataArray(np.array(["2018-01-01 00:01:00", "2018-01-01 00:02:00"], dtype="M8[ns]"), dims="time")
    diameter_bin_center = xr.DataArray(np.array([0.2, 0.4, 0.6, 0.8]), dims=DIAMETER_DIMENSION)
    diameter_bin_width = xr.DataArray(np.array([0.2, 0.2, 0.2, 0.2]), dims=DIAMETER_DIMENSION)
    diameter_bin_lower = xr.DataArray(np.array([0.1, 0.3, 0.5, 0.7]), dims=DIAMETER_DIMENSION)
    diameter_bin_upper = xr.DataArray(np.array([0.3, 0.5, 0.7, 0.9]), dims=DIAMETER_DIMENSION)

    velocity_bin_center = xr.DataArray(np.array([0.2, 0.5, 1]), dims=VELOCITY_DIMENSION)
    velocity_bin_width = xr.DataArray(np.array([0.4, 0.2, 0.8]), dims=VELOCITY_DIMENSION)
    velocity_bin_lower = xr.DataArray(np.array([0, 0.4, 0.6]), dims=VELOCITY_DIMENSION)
    velocity_bin_upper = xr.DataArray(np.array([0.4, 0.6, 1.4]), dims=VELOCITY_DIMENSION)

    # Define variables
    drop_number = xr.DataArray(np.ones((2, 3, 4)), dims=("time", VELOCITY_DIMENSION, DIAMETER_DIMENSION))

    # L1 flags
    qc_resampling = xr.DataArray(np.ones(2), dims=("time"))
    qc_time = xr.DataArray(np.zeros(2), dims=("time"))

    # Create dataset
    ds = xr.Dataset(
        data_vars={
            "raw_drop_number": drop_number,
            "qc_resampling": qc_resampling,
            "qc_time": qc_time,
        },
        coords={
            "time": time,
            "sample_interval": 60,
            "altitude": 0,
            "latitude": 42,
            "longitude": 32,
            "diameter_bin_center": diameter_bin_center,
            "diameter_bin_lower": diameter_bin_lower,
            "diameter_bin_upper": diameter_bin_upper,
            "diameter_bin_width": diameter_bin_width,
            "velocity_bin_center": velocity_bin_center,
            "velocity_bin_width": velocity_bin_width,
            "velocity_bin_lower": velocity_bin_lower,
            "velocity_bin_upper": velocity_bin_upper,
        },
    )

    # Finalize attribute
    if not with_velocity:
        ds = ds.sum(dim=VELOCITY_DIMENSION)
        ds.attrs["sensor_name"] = "RD80"
    else:
        ds.attrs["sensor_name"] = "PARSIVEL2"
    return ds


def create_template_l0c_dataset(with_velocity=True):
    """Create DISDRODB L0C basic dataset."""
    ds = create_template_dataset(with_velocity=with_velocity)
    return ds


def create_template_l2e_dataset(with_velocity=True):
    """Create DISDRODB L2E basic dataset."""
    ds = create_template_dataset(with_velocity=with_velocity)
    ds = ds.drop_vars(names=["fall_velocity", "drop_number_concentration"], errors="ignore")
    ds = generate_l2e(ds)

    ds["fall_velocity"] = xr.DataArray(
        np.array([[0.5, 1, 1.5, 2], [0.5, 1, 1.5, 2]]),
        dims=("time", DIAMETER_DIMENSION),
    )
    ds["drop_number"] = xr.DataArray(np.ones((2, 3, 4)), dims=("time", VELOCITY_DIMENSION, DIAMETER_DIMENSION))
    ds["drop_number_concentration"] = xr.DataArray(
        np.array([[0, 10000, 5000, 500], [0, 10000, 5000, 500]]),
        dims=("time", "diameter_bin_center"),
    )
    # Compute N
    ds["N"] = ds["drop_number"].sum(dim=(VELOCITY_DIMENSION, DIAMETER_DIMENSION))

    # Remove velocity dimension if asked
    if not with_velocity:
        ds = ds.sum(dim=VELOCITY_DIMENSION)
    return ds
