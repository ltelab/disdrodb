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
    compute_jensen_shannon_distance,
    compute_kl_divergence,
    compute_kolmogorov_smirnov_distance,
    compute_wasserstein_distance,
)
from disdrodb.utils.warnings import suppress_warnings


def make_test_data(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
):
    """
    Generate obs and pred DataArrays with diameter coordinates for GoF tests.

    Creates test xarray.DataArray objects with proper dimensions and coordinates
    required by compute_gof_stats, including time, diameter_bin_center, and
    diameter_bin_width.

    Parameters
    ----------
    obs_data : np.ndarray
        Observed values with shape (n_times, n_bins).
    pred_data : np.ndarray
        Predicted values with shape (n_times, n_bins).

    Returns
    -------
    obs : xr.DataArray
        Observations DataArray with dimensions [time, diameter_dimension].
    pred : xr.DataArray
        Predictions DataArray with dimensions [time, diameter_dimension].

    Raises
    ------
    ValueError
        If obs_data and pred_data have different shapes.
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
        """Test equal zero observed and predicted values yield zero errors.

        When obs = pred = 0 everywhere, most metrics should be zero except
        R2 and DmodeDiff which are undefined (NaN) for zero arrays.
        """
        data = np.zeros((2, 3))
        obs, pred = make_test_data(data, data)
        ds = compute_gof_stats(obs, pred).isel(time=0)
        assert ds["MAE"].values == 0
        assert ds["MaxAE"].values == 0
        assert ds["PeakDiff"].values == 0
        assert ds["NtDiff"].values == 0
        assert ds["RelMaxAE"].values == 0
        assert ds["RelPeakDiff"].values == 0
        assert np.isnan(ds["KLDiv"].values)
        assert np.isnan(ds["JSD"].values)
        assert np.isnan(ds["WD"].values)
        assert np.isnan(ds["KS"].values)
        #  assert np.isnan(ds["KS_pvalue"].values)
        assert np.isnan(ds["R2"].values)
        assert np.isnan(ds["DmodeDiff"].values)

    def test_equal_no_variance_case(self):
        """Test equal constant values between observed and predicted values.

        When obs = pred = 1 everywhere, all error metrics should be zero.
        R2 is NaN because the arrays have no variance.
        """
        data = np.ones((2, 3))
        obs, pred = make_test_data(data, data)
        ds = compute_gof_stats(obs, pred).isel(time=0)
        assert np.isnan(ds["R2"].values)  # standard deviation of arrays with unique values is 0
        assert ds["MAE"].values == 0
        assert ds["MaxAE"].values == 0
        assert ds["RelMaxAE"].values == 0
        assert ds["PeakDiff"].values == 0
        assert ds["RelPeakDiff"].values == 0
        assert ds["DmodeDiff"].values == 0
        assert ds["NtDiff"].values == 0
        assert ds["KLDiv"].values == 0
        assert ds["JSD"].values == 0
        assert ds["WD"].values == 0
        assert ds["KS"].values == 0
        # assert ds["KS_pvalue"].values == 1

    def test_equal_with_variance_case(self):
        """Test equal constant values between observed and predicted values.

        When obs = pred = 1 everywhere, all error metrics should be zero.
        R2 is NaN because the arrays have no variance.
        """
        data = np.array([np.arange(1, 5), np.arange(1, 5)])
        obs, pred = make_test_data(data, data)
        ds = compute_gof_stats(obs, pred).isel(time=0)
        assert ds["R2"].values == 1
        assert ds["MAE"].values == 0
        assert ds["MaxAE"].values == 0
        assert ds["RelMaxAE"].values == 0
        assert ds["PeakDiff"].values == 0
        assert ds["RelPeakDiff"].values == 0
        assert ds["DmodeDiff"].values == 0
        assert ds["NtDiff"].values == 0
        assert ds["KLDiv"].values == 0
        assert ds["JSD"].values == 0
        assert ds["WD"].values == 0
        assert ds["KS"].values == 0
        # assert ds["KS_pvalue"].values == 1

    def test_one_zero_case(self):
        """Test case where observed values are zero."""
        obs = np.zeros((2, 3))
        pred = np.ones((2, 3))
        obs, pred = make_test_data(obs, pred)
        ds = compute_gof_stats(obs, pred).isel(time=0)
        assert np.isnan(ds["R2"].values)  # standard deviation of arrays with unique values is 0
        assert ds["MAE"].values == 1
        assert ds["MaxAE"].values == 1
        assert np.isnan(ds["RelMaxAE"].values)
        assert ds["PeakDiff"].values == -1
        assert np.isnan(ds["RelPeakDiff"].values)
        assert np.isnan(ds["DmodeDiff"].values)  # for zero arrays, diameter mode is NaN
        assert ds["NtDiff"].values == 3
        assert np.isnan(ds["KLDiv"].values)
        assert np.isnan(ds["JSD"].values)
        assert np.isnan(ds["WD"].values)
        assert np.isnan(ds["KS"].values)
        # assert np.isnan(ds["KS_pvalue"].values)

    def test_zero_one_case(self):
        """Test case where predicted values are zero."""
        obs = np.ones((2, 3))
        pred = np.zeros((2, 3))
        obs, pred = make_test_data(obs, pred)
        ds = compute_gof_stats(obs, pred).isel(time=0)
        assert np.isnan(ds["R2"].values)  # standard deviation of arrays with unique values is 0
        assert ds["MAE"].values == 1
        assert ds["MaxAE"].values == 1
        assert ds["RelMaxAE"].values == 1
        assert ds["PeakDiff"].values == 1
        assert ds["RelPeakDiff"].values == 1
        assert np.isnan(ds["DmodeDiff"].values)  # for zero arrays, diameter mode is NaN
        assert ds["NtDiff"].values == -3
        assert np.isnan(ds["KLDiv"].values)
        assert np.isnan(ds["JSD"].values)
        assert np.isnan(ds["WD"].values)
        assert np.isnan(ds["KS"].values)
        # assert np.isnan(ds["KS_pvalue"].values)

    def test_real_case(self):
        """Test realistic case with varied observed and predicted values.

        When pred = obs + 1, all metrics should reflect a consistent small error.
        Perfect correlation (R2=1) expected due to linear relationship.
        """
        obs = np.arange(0, 6).reshape(2, 3)
        pred = obs + 1
        obs, pred = make_test_data(obs, pred)
        ds = compute_gof_stats(obs, pred).isel(time=0)
        assert ds["MAE"].values == 1
        assert ds["MaxAE"].values == 1
        assert ds["RelMaxAE"].values == 0.5
        assert ds["PeakDiff"].values == -1
        assert ds["RelPeakDiff"].values == -0.5
        assert ds["DmodeDiff"].values == 0
        assert ds["NtDiff"].values == 3
        assert ds["KLDiv"].values == 0.19
        assert ds["R2"].values == 1
        assert ds["JSD"].values == 0.25
        assert ds["WD"].values == 0.33
        assert ds["KS"].values == 0.17
        # assert ds["KS_pvalue"].values == 1

    def test_single_nan_case(self):
        """Test handling of NaN values in predictions.

        When a single NaN appears in one time step, that entire time step
        should have NaN results. Other time steps should compute normally.
        """
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
        """Test handling of completely NaN data.

        When both obs and pred are all NaN, all computed metrics should be NaN.
        """
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
        """Test compatibility with dask lazy arrays.

        Ensures compute_gof_stats works with chunked dask arrays and produces
        correct results after computation.
        """
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
        with suppress_warnings():
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
        assert ds["JSD"].values == 0.25
        assert ds["WD"].values == 0.33
        assert ds["KS"].values == 0.17
        # assert ds["KS_pvalue"].values == 1

    def test_only_diameter_dimension(self):
        """Test compute_gof_stats with only diameter dimension (no time).

        Ensures the function works correctly when only the diameter_dimension
        is present, without a time dimension.
        """
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


class TestComputeKLDivergence:
    """Test suite for compute_kl_divergence function."""

    def test_identical_distributions(self):
        """Test KL divergence of identical distributions is zero."""
        pk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        qk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        kl = compute_kl_divergence(pk, qk, dim="bin")
        assert np.isclose(kl.values, 0.0, atol=1e-10)

    def test_zero_zero_case(self):
        """Test KL divergence when both distributions are zero."""
        pk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        qk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        kl = compute_kl_divergence(pk, qk, dim="bin")
        assert np.isnan(kl.values)

    def test_zero_nonzero_case(self):
        """Test KL divergence when pk is zero but qk is nonzero."""
        pk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        qk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        kl = compute_kl_divergence(pk, qk, dim="bin")
        assert np.isnan(kl.values)

    def test_nonzero_zero_case(self):
        """Test KL divergence when pk is nonzero but qk is zero."""
        pk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        qk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        kl = compute_kl_divergence(pk, qk, dim="bin")
        assert np.isnan(kl.values)

    def test_different_distributions(self):
        """Test KL divergence of different distributions is positive."""
        pk = xr.DataArray([0.5, 0.5, 0.0], dims=["bin"])
        qk = xr.DataArray([0.25, 0.25, 0.5], dims=["bin"])
        kl = compute_kl_divergence(pk, qk, dim="bin")
        assert np.isclose(kl.values, 0.69314718, atol=1e-10)

    def test_multidimensional(self):
        """Test KL divergence with multidimensional input."""
        pk = xr.DataArray(
            [[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]],
            dims=["time", "bin"],
        )
        qk = xr.DataArray(
            [[0.2, 0.3, 0.5], [0.25, 0.25, 0.5]],
            dims=["time", "bin"],
        )
        kl = compute_kl_divergence(pk, qk, dim="bin")
        assert kl.shape == (2,)
        assert np.isclose(kl.values[0], 0.0, atol=1e-10)
        assert np.isclose(kl.values[1], 0.21801191, atol=1e-10)


class TestComputeJensenShannonDistance:
    """Test suite for compute_jensen_shannon_distance function."""

    def test_identical_distributions(self):
        """Test JS distance of identical distributions is zero."""
        pk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        qk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        jsd = compute_jensen_shannon_distance(pk, qk, dim="bin")
        assert np.isclose(jsd.values, 0.0, atol=1e-10)

    def test_zero_zero_case(self):
        """Test JS distance when both distributions are zero."""
        pk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        qk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        jsd = compute_jensen_shannon_distance(pk, qk, dim="bin")
        assert np.isnan(jsd.values)

    def test_zero_nonzero_case(self):
        """Test JS distance when pk is zero but qk is nonzero."""
        pk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        qk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        jsd = compute_jensen_shannon_distance(pk, qk, dim="bin")
        assert np.isnan(jsd.values)

    def test_nonzero_zero_case(self):
        """Test JS distance when pk is nonzero but qk is zero."""
        pk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        qk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        jsd = compute_jensen_shannon_distance(pk, qk, dim="bin")
        assert np.isnan(jsd.values)

    def test_different_distributions(self):
        """Test JS distance of different distributions is positive."""
        pk = xr.DataArray([0.5, 0.5, 0.0], dims=["bin"])
        qk = xr.DataArray([0.25, 0.25, 0.5], dims=["bin"])
        jsd = compute_jensen_shannon_distance(pk, qk, dim="bin")
        assert np.isclose(jsd.values, 0.4645014, atol=1e-10)

    def test_symmetry(self):
        """Test JS distance is symmetric."""
        pk = xr.DataArray([0.5, 0.5, 0.0], dims=["bin"])
        qk = xr.DataArray([0.25, 0.25, 0.5], dims=["bin"])
        jsd_pk_qk = compute_jensen_shannon_distance(pk, qk, dim="bin")
        jsd_qk_pk = compute_jensen_shannon_distance(qk, pk, dim="bin")
        assert np.isclose(jsd_pk_qk.values, jsd_qk_pk.values)


class TestComputeWassersteinDistance:
    """Test suite for compute_wasserstein_distance function."""

    def test_identical_distributions(self):
        """Test Wasserstein distance of identical distributions is zero."""
        pk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        qk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        D = xr.DataArray([1.0, 2.0, 3.0], dims=["bin"])
        dD = xr.DataArray([1.0, 1.0, 1.0], dims=["bin"])
        wd = compute_wasserstein_distance(pk, qk, D, dD, dim="bin")
        assert np.isclose(wd.values, 0.0, atol=1e-10)

    def test_zero_zero_case(self):
        """Test Wasserstein distance when both distributions are zero."""
        pk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        qk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        D = xr.DataArray([1.0, 2.0, 3.0], dims=["bin"])
        dD = xr.DataArray([1.0, 1.0, 1.0], dims=["bin"])
        wd = compute_wasserstein_distance(pk, qk, D, dD, dim="bin")
        assert np.isnan(wd.values)

    def test_zero_nonzero_case(self):
        """Test Wasserstein distance when pk is zero but qk is nonzero."""
        pk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        qk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        D = xr.DataArray([1.0, 2.0, 3.0], dims=["bin"])
        dD = xr.DataArray([1.0, 1.0, 1.0], dims=["bin"])
        wd = compute_wasserstein_distance(pk, qk, D, dD, dim="bin")
        assert np.isnan(wd.values)

    def test_nonzero_zero_case(self):
        """Test Wasserstein distance when pk is nonzero but qk is zero."""
        pk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        qk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        D = xr.DataArray([1.0, 2.0, 3.0], dims=["bin"])
        dD = xr.DataArray([1.0, 1.0, 1.0], dims=["bin"])
        wd = compute_wasserstein_distance(pk, qk, D, dD, dim="bin")
        assert np.isnan(wd.values)

    def test_shifted_distributions(self):
        """Test Wasserstein distance of shifted distributions is positive."""
        pk = xr.DataArray([1.0, 0.0, 0.0], dims=["bin"])
        qk = xr.DataArray([0.0, 1.0, 0.0], dims=["bin"])
        D = xr.DataArray([1.0, 2.0, 3.0], dims=["bin"])
        dD = xr.DataArray([1.0, 1.0, 1.0], dims=["bin"])
        wd = compute_wasserstein_distance(pk, qk, D, dD, dim="bin")
        assert np.isclose(wd.values, 1)

    def test_left_riemann(self):
        """Test Wasserstein distance with left Riemann integration."""
        pk = xr.DataArray([1.0, 0.0, 0.0], dims=["bin"])
        qk = xr.DataArray([0.0, 1.0, 0.0], dims=["bin"])
        D = xr.DataArray([1.0, 2.0, 3.0], dims=["bin"])
        dD = xr.DataArray([1.0, 1.0, 1.0], dims=["bin"])
        wd = compute_wasserstein_distance(pk, qk, D, dD, dim="bin", integration="left_riemann")
        assert np.isclose(wd.values, 1)

    def test_symmetry(self):
        """Test Wasserstein distance is symmetric."""
        pk = xr.DataArray([0.5, 0.5, 0.0], dims=["bin"])
        qk = xr.DataArray([0.25, 0.25, 0.5], dims=["bin"])
        D = xr.DataArray([1.0, 2.0, 3.0], dims=["bin"])
        dD = xr.DataArray([1.0, 1.0, 1.0], dims=["bin"])
        wd_pk_qk = compute_wasserstein_distance(pk, qk, D, dD, dim="bin")
        wd_qk_pk = compute_wasserstein_distance(qk, pk, D, dD, dim="bin")
        assert np.isclose(wd_pk_qk.values, wd_qk_pk.values)


class TestComputeKolmogorovSmirnovDistance:
    """Test suite for compute_kolmogorov_smirnov_distance function."""

    def test_identical_distributions(self):
        """Test KS distance of identical distributions is zero."""
        pk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        qk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        ks, pval = compute_kolmogorov_smirnov_distance(pk, qk, dim="bin")
        assert np.isclose(ks.values, 0.0, atol=1e-10)
        assert np.isclose(pval.values, 1.0, atol=1e-10)

    def test_zero_zero_case(self):
        """Test KS distance when both distributions are zero."""
        pk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        qk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        ks, pval = compute_kolmogorov_smirnov_distance(pk, qk, dim="bin")
        assert np.isnan(ks.values)
        assert np.isnan(pval.values)

    def test_zero_nonzero_case(self):
        """Test KS distance when pk is zero but qk is nonzero."""
        pk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        qk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        ks, pval = compute_kolmogorov_smirnov_distance(pk, qk, dim="bin")
        assert np.isnan(ks.values)
        assert np.isnan(pval.values)

    def test_nonzero_zero_case(self):
        """Test KS distance when pk is nonzero but qk is zero."""
        pk = xr.DataArray([0.2, 0.3, 0.5], dims=["bin"])
        qk = xr.DataArray([0.0, 0.0, 0.0], dims=["bin"])
        ks, pval = compute_kolmogorov_smirnov_distance(pk, qk, dim="bin")
        assert np.isnan(ks.values)
        assert np.isnan(pval.values)

    def test_different_distributions(self):
        """Test KS distance of different distributions is positive."""
        pk = xr.DataArray([0.5, 0.5, 0.0, 0.0, 0.0], dims=["bin"])
        qk = xr.DataArray([0.25, 0.25, 0.5, 0.5, 0.5], dims=["bin"])
        ks, pval = compute_kolmogorov_smirnov_distance(pk, qk, dim="bin")
        assert np.isclose(ks.values, 1, atol=1e-10)
        assert np.isclose(pval.values, 0.46701296, atol=1e-10)

    def test_completely_different_distributions(self):
        """Test KS distance of completely different distributions."""
        pk = xr.DataArray([1.0, 0.0, 0.0], dims=["bin"])
        qk = xr.DataArray([0.0, 0.0, 1.0], dims=["bin"])
        ks, pval = compute_kolmogorov_smirnov_distance(pk, qk, dim="bin")
        assert ks.values == 1.0

    def test_p_value_range(self):
        """Test that p-value is always between 0 and 1."""
        pk = xr.DataArray([0.5, 0.5, 0.0], dims=["bin"])
        qk = xr.DataArray([0.25, 0.25, 0.5], dims=["bin"])
        ks, pval = compute_kolmogorov_smirnov_distance(pk, qk, dim="bin")
        assert 0 <= pval.values <= 1
