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
"""Define Goodness-Of-Fit metrics for xarray objects."""

import numpy as np
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION
from disdrodb.utils.warnings import suppress_warnings


def compute_kl_divergence(pk, qk, dim, eps=1e-12):
    """Compute Kullback-Leibler (KL) divergence.

    Compares two probability distributions.
    When KL < 0.1 the two distributions are similar.
    When KL < 0.01 the two distributions are nearly indistinguishable.

    Parameters
    ----------
    pk : xarray.DataArray
        Observed / true / empirical probability distribution
    qk : xarray.DataArray
        Predicted / model / approximating probability distribution
    dim : str
        Name of the bin dimension.
    eps : float, optional
        Small value for numerical stability. Default is 1e-12.

    Returns
    -------
    xarray.DataArray
        Kullback-Leibler (KL) divergence.

    """
    # Get row masses before computing CDFs
    pk_mass = pk.sum(dim=dim)
    qk_mass = qk.sum(dim=dim)

    # Regularize probability to avoid division by zero
    qk_regularized = xr.where(qk == 0, eps, qk)

    # Compute log probability ratio
    log_prob_ratio = np.log(pk / qk_regularized)

    # Compute divergence (zero out where pk=0)
    kl = (pk * log_prob_ratio).where(pk > 0, other=0.0).sum(dim=dim, skipna=False)

    # Clip tiny negative values due to numerical noise
    kl = xr.where(kl >= 0.0, kl, 0.0)

    # Handle edge cases when P=0 or Q=0
    kl = xr.where((pk_mass == 0) | (qk_mass == 0), np.nan, kl)
    return kl


def compute_jensen_shannon_distance(pk, qk, dim, eps=1e-12):
    """Compute Jensen-Shannon distance.

    Symmetric and finite version of KL divergence.
    The Jensen-Shannon distance is the square root of the Jensen-Shannon divergence.
    Values are defined between 0 and np.sqrt(ln(2)) = 0.83256

    Parameters
    ----------
    pk : xarray.DataArray
        Observed / true probability distribution
    qk : xarray.DataArray
        Predicted / model probability distribution
    dim : str
        Name of the bin dimension
    eps : float, optional
        Small value for numerical stability. Default is 1e-12.

    Returns
    -------
    xarray.DataArray
        Jensen-Shannon distance
    """
    # Mixture distribution
    mk = 0.5 * (pk + qk)

    # KL(P || M)
    kl_pm = compute_kl_divergence(pk=pk, qk=mk, dim=dim, eps=eps)

    # KL(Q || M)
    kl_qm = compute_kl_divergence(pk=qk, qk=mk, dim=dim, eps=eps)

    # Jensen-Shannon divergence [0, ln(2)]
    js_div = 0.5 * (kl_pm + kl_qm)
    js_div = np.maximum(js_div, 0.0)  # clip tiny negative values to zero (numerical safety)

    # Jensen-Shannon distance
    js_distance = np.sqrt(js_div)
    return js_distance


def compute_wasserstein_distance(
    pk,
    qk,
    D,
    dD,
    dim,
    integration="bin",
):
    """Compute Wasserstein-1 distance (Earth Mover's Distance) between two distributions.

    Parameters
    ----------
    pk : xarray.DataArray
        Observed / true probability distribution
    qk : xarray.DataArray
        Predicted / model probability distribution
    D : xarray.DataArray
        Bin centers
    dD : xarray.DataArray
        Bin widths
    dim : str
        Name of the bin dimension
    integration : {"bin", "left_riemann"}, optional
        Integration scheme for the Wasserstein integral:

        - "bin":
            Histogram-based Wasserstein distance.
            Interprets distributions as piecewise-constant densities over bins of width dD
            and integrates the CDF difference over those intervals.
            Default is "bin".

        - "left_riemann":
            Discrete-support Wasserstein distance.
            Interprets probability mass as concentrated at bin centers D and
            integrates using spacing between support points, consistent with
            ``scipy.stats.wasserstein_distance``.

    Returns
    -------
    xarray.DataArray
        Wasserstein-1 distance
    """
    # Get row masses before computing CDFs
    pk_mass = pk.sum(dim=dim)
    qk_mass = qk.sum(dim=dim)

    # CDFs
    cdf_p = pk.cumsum(dim)
    cdf_q = qk.cumsum(dim)

    # Absolute CDF difference
    diff = abs(cdf_p - cdf_q)

    if integration == "bin":
        # Histogram-based Wasserstein (density interpretation)
        wd = (diff * dD).sum(dim=dim)

    elif integration == "left_riemann":
        # Discrete-support Wasserstein (SciPy-style)
        # Evaluate |CDF difference| at left support points D_i
        diff_left = diff.isel({dim: slice(None, -1)})

        # Compute spacing between support points and
        # explicitly assign left coordinates to avoid misalignment
        dx = D.diff(dim)
        dx = dx.assign_coords({dim: D.isel({dim: slice(None, -1)})})
        wd = (diff_left * dx).sum(dim=dim)
    else:
        raise ValueError("integration must be 'bin' or 'left_riemann'")

    # Handle edge cases (Wasserstein distance is undefined when P=0 or Q=0)
    wd = xr.where((pk_mass == 0) | (qk_mass == 0), np.nan, wd)
    return wd


def compute_kolmogorov_smirnov_distance(pk, qk, dim):
    """Compute Kolmogorov-Smirnov (KS) distance and p-value.

    The Kolmogorov-Smirnov (KS) distance is bounded between 0 and 1,
    where 0 indicates that the two distributions are identical.
    The associated KS test p-value ranges from 0 to 1,
    with a value of 1 indicating no evidence against the null hypothesis that the distributions are identical.

    Parameters
    ----------
    pk : xarray.DataArray
        Observed / true probability distribution
    qk : xarray.DataArray
        Predicted / model probability distribution
    dim : str
        Name of the bin dimension

    Returns
    -------
    ks_statistic : xarray.DataArray
        Kolmogorov-Smirnov statistic (maximum CDF difference)
        If 0, the two distributions are identical.
    ks_p_value : xarray.DataArray
        Kolmogorov-Smirnov Test p-value (asymptotic approximation)
        A p-value of 0 means “strong evidence against equality.”
        A p-value of 1 means “no evidence against equality.”
        Identical distributions show a pvalue of 1.
        Similar distributions show a pvalue close to 1.
    """
    # Get row masses before computing CDFs
    pk_mass = pk.sum(dim=dim)
    qk_mass = qk.sum(dim=dim)

    # CDFs
    cdf_p = pk.cumsum(dim)
    cdf_q = qk.cumsum(dim)

    # KS statistic
    ks = np.abs(cdf_p - cdf_q).max(dim=dim)

    # Effective sample sizes (Rényi-2 effective N)
    n_eff_p = 1.0 / (pk**2).sum(dim=dim)
    n_eff_q = 1.0 / (qk**2).sum(dim=dim)

    # Combined effective sample size
    n_eff = (n_eff_p * n_eff_q) / (n_eff_p + n_eff_q)

    # Asymptotic KS p-value approximation
    p_value = 2.0 * np.exp(-2.0 * (ks * np.sqrt(n_eff)) ** 2)
    p_value = p_value.clip(0.0, 1.0)

    # Handle edge cases (KS distance is undefined when P=0 or Q=0)
    ks = xr.where((pk_mass == 0) | (qk_mass == 0), np.nan, ks)
    p_value = xr.where((pk_mass == 0) | (qk_mass == 0), np.nan, p_value)

    return ks, p_value


def compute_gof_stats(obs, pred, dim=DIAMETER_DIMENSION):
    """
    Compute various goodness-of-fit (GoF) statistics between observed and predicted values.

    Computes a comprehensive set of GOF metrics including Pearson correlation,
    error statistics, and distribution distance metrics (KL, JS, Wasserstein, KS).

    Parameters
    ----------
    obs : xarray.DataArray
        Observations DataArray with at least dimension ``dim``.
        Should contain 'diameter_bin_center' and 'diameter_bin_width' coordinates.
    pred : xarray.DataArray
        Predictions DataArray with at least dimension ``dim``.
    dim : str, optional
        DataArray dimension over which to compute GOF statistics.
        Default is DIAMETER_DIMENSION.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the following computed GoF statistics:

        - R2: Coefficient of determination (squared Pearson correlation)
        - MAE: Mean Absolute Error
        - MaxAE: Maximum Absolute Error
        - RelMaxAE: Relative Maximum Absolute Error
        - PeakDiff: Difference at distribution peak (N(D) max)
        - RelPeakDiff: Relative difference at peak
        - DmodeDiff: Difference in mode diameters
        - NtDiff: Difference in total number concentration
        - KLDiv: Kullback-Leibler divergence
        - JSD: Jensen-Shannon distance
        - WD: Wasserstein-1 distance
        - KS: Kolmogorov-Smirnov statistic
        - KS_pvalue: Kolmogorov-Smirnov Test p-value
    """
    # TODO: add censoring option (by setting values to np.nan?)
    from disdrodb.l2.empirical_dsd import get_mode_diameter

    # Retrieve diameter and diameter bin width
    diameter = obs["diameter_bin_center"]
    diameter_bin_width = obs["diameter_bin_width"]

    # Compute errors
    error = obs - pred

    # Compute max obs and pred
    obs_max = obs.max(dim=dim, skipna=False)
    pred_max = pred.max(dim=dim, skipna=False)

    # Compute NaN mask
    mask_nan = np.logical_or(np.isnan(obs_max), np.isnan(pred_max))

    # Compute GOF statistics
    with suppress_warnings():
        # Compute Pearson Correlation
        # - if same values and a single values return NaN (std of constant value is not defined)
        pearson_r = xr.corr(obs, pred, dim=dim)

        # Compute Mean Absolute Error (MAE)
        mae = np.abs(error).mean(dim=dim, skipna=False)

        # Compute maximum absolute error
        max_error = np.abs(error).max(dim=dim, skipna=False)
        relative_max_error = xr.where(max_error == 0, 0, xr.where(obs_max == 0, np.nan, max_error / obs_max))

        # Compute deviation of N(D) at distribution mode
        mode_deviation = obs_max - pred_max
        mode_relative_deviation = xr.where(
            mode_deviation == 0,
            0,
            xr.where(obs_max == 0, np.nan, mode_deviation / obs_max),
        )

        # Compute diameter difference of the distribution mode
        diameter_mode_pred = get_mode_diameter(pred, diameter)
        diameter_mode_obs = get_mode_diameter(obs, diameter)
        diameter_mode_deviation = diameter_mode_obs - diameter_mode_pred

        # Compute difference in total number concentration
        total_number_concentration_obs = (obs * diameter_bin_width).sum(dim=dim, skipna=False)
        total_number_concentration_pred = (pred * diameter_bin_width).sum(dim=dim, skipna=False)
        total_number_concentration_difference = total_number_concentration_pred - total_number_concentration_obs

        # Compute pdf per bin
        pk_pdf = obs / total_number_concentration_obs
        qk_pdf = pred / total_number_concentration_pred

        # Compute probabilities per bin
        pk = pk_pdf * diameter_bin_width
        pk = pk / pk.sum(dim=dim, skipna=False)  # this might not be necessary
        qk = qk_pdf * diameter_bin_width
        qk = qk / qk.sum(dim=dim, skipna=False)  # this might not be necessary

        # Keep prob mass to 0 if total concentration is 0
        pk = xr.where(total_number_concentration_obs == 0, 0, pk)
        qk = xr.where(total_number_concentration_pred == 0, 0, qk)

        # Compute Kullback-Leibler divergence
        kl_divergence = compute_kl_divergence(pk=pk, qk=qk, dim=dim)

        # Compute Jensen-Shannon distance
        js_distance = compute_jensen_shannon_distance(pk=pk, qk=qk, dim=dim)

        # Compute Wasserstein-1 distance
        wd = compute_wasserstein_distance(pk=pk, qk=qk, D=diameter, dD=diameter_bin_width, dim=dim)

        # Compute Kolmogorov-Smirnov distance
        ks_stat, ks_p_value = compute_kolmogorov_smirnov_distance(pk=pk, qk=qk, dim=dim)

    # Create an xarray.Dataset to hold the computed statistics
    ds = xr.Dataset(
        {
            "R2": pearson_r**2,  # Squared Pearson correlation coefficient
            "MAE": mae,  # Mean Absolute Error
            "MaxAE": max_error,  # Maximum Absolute Error
            "RelMaxAE": relative_max_error,  # Relative Maximum Absolute Error
            "PeakDiff": mode_deviation,  # Difference at distribution peak
            "RelPeakDiff": mode_relative_deviation,  # Relative difference at peak
            "DmodeDiff": diameter_mode_deviation,  # Difference in mode diameters
            "NtDiff": total_number_concentration_difference,
            "KLDiv": kl_divergence,  # Kullback-Leibler divergence
            "JSD": js_distance,  # Jensen-Shannon distance
            "WD": wd,  # Wasserstein-1 distance
            "KS": ks_stat,  # Kolmogorov-Smirnov statistic
            "KS_pvalue": ks_p_value,  # Kolmogorov-Smirnov Test p-value
        },
    )
    # Round
    ds = ds.round(2)
    # Mask where input obs or pred is NaN
    ds = ds.where(~mask_nan)
    return ds
