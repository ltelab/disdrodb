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
"""Testing xarray GOF metrics."""
import numpy as np
import xarray as xr
from disdrodb.constants import DIAMETER_DIMENSION

from disdrodb.psd.gof_metrics import (
    compute_gof_stats,  
)


def make_test_data(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
):
    """
    Generate obs, pred, and diameter_bin_width DataArrays for GoF tests.

    Parameters
    ----------
    obs_data : array_like (time, bins)
        obs values per time step and bin.
    pred_data : array_like (time, bins)
        pred values per time step and bin.

    Returns
    -------
    obs : xr.DataArray
    pred : xr.DataArray
    diameter_bin_width : xr.DataArray
    """
    # Convert inputs
    obs = np.asarray(obs_data)
    pred = np.asarray(pred_data)
    if obs.shape != pred.shape:
        raise ValueError("obs_data and pred_data must have the same shape")

    n_times, n_bins = obs.shape

    # Common coordinates (invariant across tests)
    bin_centers = np.arange(1, n_bins + 1, dtype=float)
    bin_widths = np.ones(n_bins, dtype=float)

    # Build coordinates
    diameter_bin_center = xr.DataArray(
        data=bin_centers,
        dims=[DIAMETER_DIMENSION],
        name="diameter_bin_center",
    )
    diameter_bin_width = xr.DataArray(
        data=bin_widths,
        dims=[DIAMETER_DIMENSION],
        name="diameter_bin_width",
    )
    time = xr.DataArray(
        data=np.arange(n_times),
        dims=["time"],
        coords={"time": np.arange(n_times)},
        name="time",
    )

    # Create DataArrays
    obs = xr.DataArray(
        data=obs,
        dims=["time", DIAMETER_DIMENSION],
        coords={
            "time": time,
            "diameter_bin_center": diameter_bin_center,
            "diameter_bin_width": diameter_bin_width,
        },
        name="obs_values",
    )
    pred = xr.DataArray(
        data=pred,
        dims=["time", DIAMETER_DIMENSION],
        coords={
            "time": time,
            "diameter_bin_center": diameter_bin_center,
            "diameter_bin_width": diameter_bin_width,
        },
        name="pred_values",
    )

    return obs, pred


class TestComputeGoFStats:
    """Test suite for compute_gof_stats function."""

    def test_zero_zero_case(self):
        """Test that equal (0) observed and predicted values should yield zero errors."""
        # Generate test inputs
        data = np.zeros((2, 3))
        obs, pred = make_test_data(data, data)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred).isel(time=0)

        # Assert values
        assert ds["MAE"].values == 0
        assert ds["MaxAE"].values == 0
        assert ds["PeakDiff"].values == 0
        assert ds["NtDiff"].values == 0
        assert ds["RelMaxAE"].values == 0
        assert ds["RelPeakDiff"].values == 0
        assert ds["KLDiv"].values == 0

        assert np.isnan(ds["R2"].values)  # standard deviation of arrays with unique values is 0
        assert np.isnan(ds["DmodeDiff"].values)  # for zero arrays, diameter mode is NaN

    def test_one_one_case(self):
        """Test that equal observed and predicted values should yield zero errors."""
        # Generate test inputs
        data = np.ones((2, 3))
        obs, pred = make_test_data(data, data)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred).isel(time=0)

        # Assert values
        assert np.isnan(ds["R2"].values)  # standard deviation of arrays with unique values is 0
        assert ds["MAE"].values == 0
        assert ds["MaxAE"].values == 0
        assert ds["RelMaxAE"].values == 0
        assert ds["PeakDiff"].values == 0
        assert ds["RelPeakDiff"].values == 0
        assert ds["DmodeDiff"].values == 0
        assert ds["NtDiff"].values == 0
        assert ds["KLDiv"].values == 0

    def test_one_zero_case(self):
        """Test the case where observed values are zeros."""
        # Generate test inputs
        obs = np.zeros((2, 3))
        pred = np.ones((2, 3))
        obs, pred = make_test_data(obs, pred)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred).isel(time=0)

        # Assert values
        assert np.isnan(ds["R2"].values)  # standard deviation of arrays with unique values is 0
        assert ds["MAE"].values == 1
        assert ds["MaxAE"].values == 1
        assert np.isnan(ds["RelMaxAE"].values)  # division by 0 obs_max
        assert ds["PeakDiff"].values == -1
        assert np.isnan(ds["RelPeakDiff"].values)  # division by 0 obs_max
        assert np.isnan(ds["DmodeDiff"].values)  # for zero arrays, diameter mode is NaN
        assert ds["NtDiff"].values == 3
        assert np.isnan(ds["KLDiv"].values)  # pk of zeros arrays is NaN (pk=obs/Nt_obs=0/0)

    def test_zero_one_case(self):
        """Test the case where predicted values are zeros."""
        # Generate test inputs
        obs = np.ones((2, 3))
        pred = np.zeros((2, 3))
        obs, pred = make_test_data(obs, pred)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred).isel(time=0)

        # Assert values
        assert np.isnan(ds["R2"].values)  # standard deviation of arrays with unique values is 0
        assert ds["MAE"].values == 1
        assert ds["MaxAE"].values == 1
        assert ds["RelMaxAE"].values == 1
        assert ds["PeakDiff"].values == 1
        assert ds["RelPeakDiff"].values == 1
        assert np.isnan(ds["DmodeDiff"].values)  # for zero arrays, diameter mode is NaN
        assert ds["NtDiff"].values == -3
        assert np.isnan(ds["KLDiv"].values)  # qk of zeros arrays is NaN (qk=pred/Nt_pred=0/0)

    def test_real_case(self):
        """Test the case where pred values are zeros."""
        # Generate test inputs
        obs = np.arange(0, 6).reshape(2, 3)
        pred = obs + 1
        obs, pred = make_test_data(obs, pred)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred).isel(time=0)

        # Assert values
        assert ds["MAE"].values == 1
        assert ds["MaxAE"].values == 1
        assert ds["RelMaxAE"].values == 0.5

        assert ds["PeakDiff"].values == -1
        assert ds["RelPeakDiff"].values == -0.5
        assert ds["DmodeDiff"].values == 0

        assert ds["NtDiff"].values == 3
        assert ds["KLDiv"].values == 0.19
        assert ds["R2"].values == 1

    def test_single_nan_case(self):
        """Test the case where there are NaN values."""
        # Generate test inputs
        obs = np.arange(0, 6, dtype=float).reshape(2, 3)
        pred = obs + 1
        pred[0, 2] = np.nan

        obs, pred = make_test_data(obs, pred)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred)

        # Assert values
        for var in ds.data_vars:
            assert np.isnan(ds[var].isel(time=0).values)

        # Assert values
        for var in ds.data_vars:
            assert not np.isnan(ds[var].isel(time=1).values)

    def test_all_nan_case(self):
        """Test the case where all values are NaN values."""
        # Generate test inputs
        obs = np.zeros((2, 3)) * np.nan
        pred = np.zeros((2, 3)) * np.nan
        obs, pred = make_test_data(obs, pred)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred).isel(time=0)

        # Assert values
        for var in ds.data_vars:
            assert np.isnan(ds[var].values)

    def test_dask_case(self):
        """Test that lazy dask computations are allowed."""
        # Generate test inputs
        obs = np.arange(0, 6).reshape(2, 3)
        pred = obs + 1
        obs, pred = make_test_data(obs, pred)
        obs = obs.chunk({"time": 1})
        pred = pred.chunk({"time": 1})

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred)
        for var in ds.data_vars:
            assert hasattr(ds[var].data, "chunks")
        ds = ds.compute().isel(time=0)

        # Assert values
        assert ds["MAE"].values == 1
        assert ds["MaxAE"].values == 1
        assert ds["RelMaxAE"].values == 0.5

        assert ds["PeakDiff"].values == -1
        assert ds["RelPeakDiff"].values == -0.5
        assert ds["DmodeDiff"].values == 0

        assert ds["NtDiff"].values == 3
        assert ds["KLDiv"].values == 0.19
        assert ds["R2"].values == 1

    def test_only_diameter_dimension(self):
        """Test case where only diameter dimension is present."""
        # Generate test inputs
        obs = np.arange(0, 6).reshape(2, 3)
        pred = obs + 1
        obs, pred = make_test_data(obs, pred)

        # Remove time dimension
        obs = obs.isel(time=0)
        pred = pred.isel(time=0)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred)
        assert isinstance(ds, xr.Dataset)