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
"""Routines for PSD fitting."""

import warnings
from contextlib import contextmanager

import numpy as np
import scipy.optimize
import scipy.stats as ss
import xarray as xr
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import gamma, gammainc, gammaln  # Regularized lower incomplete gamma function

from disdrodb.l2.empirical_dsd import get_mode_diameter
from disdrodb.psd.models import NormalizedGammaPSD


@contextmanager
def suppress_warnings():
    """Context manager suppressing RuntimeWarnings and UserWarnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        yield


####--------------------------------------------------------------------------------------.
#### Measures of fit
def compute_gof_stats(drop_number_concentration, psd):
    """
    Compute various goodness-of-fit (GoF) statistics between observed and predicted values.

    Parameters
    ----------
    - drop_number_concentration: xarray.DataArray with dimensions ('time', 'diameter_bin_center')
    - psd: instance of PSD class

    Returns
    -------
    - ds: xarray.Dataset containing the computed GoF statistics
    """
    # Retrieve diameter bin width
    diameter = drop_number_concentration["diameter_bin_center"]
    diameter_bin_width = drop_number_concentration["diameter_bin_width"]

    # Define observed and predicted values and compute errors
    observed_values = drop_number_concentration
    fitted_values = psd(diameter)  # .transpose(*observed_values.dims)
    error = observed_values - fitted_values

    # Compute GOF statistics
    with suppress_warnings():
        # Compute Pearson correlation
        pearson_r = xr.corr(observed_values, fitted_values, dim="diameter_bin_center")

        # Compute MSE
        mse = (error**2).mean(dim="diameter_bin_center")

        # Compute maximum error
        max_error = error.max(dim="diameter_bin_center")
        relative_max_error = error.max(dim="diameter_bin_center") / observed_values.max(dim="diameter_bin_center")

        # Compute difference in total number concentration
        total_number_concentration_obs = (observed_values * diameter_bin_width).sum(dim="diameter_bin_center")
        total_number_concentration_pred = (fitted_values * diameter_bin_width).sum(dim="diameter_bin_center")
        total_number_concentration_difference = total_number_concentration_pred - total_number_concentration_obs

        # Compute Kullback-Leibler divergence
        # - Compute pdf per bin
        pk_pdf = observed_values / total_number_concentration_obs
        qk_pdf = fitted_values / total_number_concentration_pred

        # - Compute probabilities per bin
        pk = pk_pdf * diameter_bin_width
        pk = pk / pk.sum(dim="diameter_bin_center")  # this might not be necessary
        qk = qk_pdf * diameter_bin_width
        qk = qk / qk.sum(dim="diameter_bin_center")  # this might not be necessary

        # - Compute divergence
        log_prob_ratio = np.log(pk / qk)
        log_prob_ratio = log_prob_ratio.where(np.isfinite(log_prob_ratio))
        kl_divergence = (pk * log_prob_ratio).sum(dim="diameter_bin_center")

        # Other statistics that can be computed also from different diameter discretization
        # - Compute max deviation at distribution mode
        max_deviation = observed_values.max(dim="diameter_bin_center") - fitted_values.max(dim="diameter_bin_center")
        max_relative_deviation = max_deviation / fitted_values.max(dim="diameter_bin_center")

        # - Compute diameter difference of the distribution mode
        diameter_mode_deviation = get_mode_diameter(observed_values) - get_mode_diameter(fitted_values)

    # Create an xarray.Dataset to hold the computed statistics
    ds = xr.Dataset(
        {
            "r2": pearson_r**2,  # Squared Pearson correlation coefficient
            "mse": mse,  # Mean Squared Error
            "max_error": max_error,  # Maximum Absolute Error
            "relative_max_error": relative_max_error,  # Relative Maximum Error
            "total_number_concentration_difference": total_number_concentration_difference,
            "kl_divergence": kl_divergence,  # Kullback-Leibler divergence
            "max_deviation": max_deviation,  # Deviation at distribution mode
            "max_relative_deviation": max_relative_deviation,  # Relative deviation at mode
            "diameter_mode_deviation": diameter_mode_deviation,  # Difference in mode diameters
        },
    )
    return ds


####--------------------------------------------------------------------------------------.
#### Maximum Likelihood


def get_expected_probabilities(params, cdf_func, pdf_func, bin_edges, probability_method, normalized=False):
    """
    Compute the expected probabilities for each bin given the distribution parameters.

    Parameters
    ----------
    params : array-like
        Parameters for the CDF or PDF function.
    cdf_func : callable
        Cumulative distribution function (CDF) that takes bin edges and parameters as inputs.
    pdf_func : callable
        Probability density function (PDF) that takes a value and parameters as inputs.
    bin_edges : array-like
        Edges of the bins for which to compute the probabilities.
    probability_method : {'cdf', 'pdf'}
        Method to compute the probabilities. If 'cdf', use the CDF to compute probabilities.
        If 'pdf', integrate the PDF over each bin range.
    normalized : bool, optional
        If True, normalize the probabilities to sum to 1. Default is False.

    Returns
    -------
    expected_probabilities : numpy.ndarray
        Array of expected probabilities for each bin.

    Notes
    -----
    - If the 'cdf' method is used, the probabilities are computed as the difference in CDF values at the bin edges.
    - If the 'pdf' method is used, the probabilities are computed by integrating the PDF over each bin range.
    - Any zero or negative probabilities are replaced with a very small positive number (1e-10) to ensure optimization.
    - If `normalized` is True, the probabilities are normalized to sum to 1.

    """
    if probability_method == "cdf":
        # Compute the CDF at bin edges
        cdf_vals = cdf_func(bin_edges, params)
        # Compute probabilities for each bin
        expected_probabilities = np.diff(cdf_vals)
        # Replace any zero or negative probabilities with a very small positive number
        # --> Otherwise do not optimize ...
        expected_probabilities = np.maximum(expected_probabilities, 1e-10)
    # Or integrate PDF over the bin range
    else:  # probability_method == "pdf":
        # For each bin, integrate the PDF over the bin range
        expected_probabilities = np.array(
            [quad(lambda x: pdf_func(x, params), bin_edges[i], bin_edges[i + 1])[0] for i in range(len(bin_edges) - 1)],
        )
    if normalized:
        # Normalize probabilities to sum to 1
        total_probability = np.sum(expected_probabilities)
        expected_probabilities /= total_probability
    return expected_probabilities


def get_adjusted_nt(cdf, params, Nt, bin_edges):
    """Adjust Nt for the proportion of missing drops. See Johnson's et al., 2013 Eqs. 3 and 4."""
    # Estimate proportion of missing drops (Johnson's 2011 Eqs. 3)
    # --> Alternative: p = 1 - np.sum(pdf(diameter, params)* diameter_bin_width)  # [-]
    p = 1 - np.diff(cdf([bin_edges[0], bin_edges[-1]], params)).item()  # [-]

    # Adjusts Nt for the proportion of drops not observed
    return Nt / (1 - p)  # [m-3]


def compute_negative_log_likelihood(
    params,
    bin_edges,
    counts,
    cdf_func,
    pdf_func,
    param_constraints=None,
    probability_method="cdf",
    likelihood="multinomial",
    truncated_likelihood=True,
):
    """
    General negative log-likelihood function for fitting distributions to binned data.

    Parameters
    ----------
    params : array-like
        Parameters of the distribution.
    bin_edges : array-like
        Edges of the bins (length N+1).
    counts : array-like
        Observed counts in each bin (length N).
    cdf_func : callable
        Cumulative distribution function of the distribution.
    pdf_func : callable
        Probability density function of the distribution.
    param_constraints : callable, optional
        Function that checks if parameters are valid.
    probability_method : str, optional
        Method to compute expected probabilities, either 'cdf' or 'pdf'. Default is 'cdf'.
    likelihood : str, optional
        Type of likelihood to compute, either 'multinomial' or 'poisson'. Default is 'multinomial'.
    truncated_likelihood : bool, optional
        Whether to normalize the expected probabilities. Default is True.
    nll : float
        Negative log-likelihood value.

    Returns
    -------
    nll: float
      The negative log-likelihood value.
    """
    # Check if parameters are valid
    if param_constraints is not None and not param_constraints(params):
        return np.inf

    # Compute (unormalized) expected probabilities using CDF
    expected_probabilities = get_expected_probabilities(
        params=params,
        cdf_func=cdf_func,
        pdf_func=pdf_func,
        bin_edges=bin_edges,
        probability_method=probability_method,
        normalized=truncated_likelihood,
    )

    # Ensure expected probabilities are valid
    if np.any(expected_probabilities <= 0):
        return np.inf

    # Compute negative log-likelihood
    if likelihood == "poisson":
        n_total = np.sum(counts)
        expected_counts = expected_probabilities * n_total
        expected_counts = np.maximum(expected_counts, 1e-10)  # Avoid zero expected counts
        nll = -np.sum(counts * np.log(expected_counts) - expected_counts)
    else:  # likelihood == "multinomial":
        # Compute likelihood
        nll = -np.sum(counts * np.log(expected_probabilities))
    return nll


def estimate_lognormal_parameters(
    counts,
    bin_edges,
    probability_method="cdf",
    likelihood="multinomial",
    truncated_likelihood=True,
    output_dictionary=True,
    optimizer="Nelder-Mead",
):
    """
    Estimate the parameters of a lognormal distribution given histogram data.

    Parameters
    ----------
    counts : array-like
        The counts for each bin in the histogram.
    bin_edges : array-like
        The edges of the bins.
    probability_method : str, optional
        The method to compute probabilities, either ``"cdf"`` or ``"pdf"``. The default is ``"cdf"``.
    likelihood : str, optional
        The likelihood function to use, either ``"multinomial"`` or ``"poisson"``.
        The default is ``"multinomial"``.
    truncated_likelihood : bool, optional
        Whether to use truncated likelihood. The default is ``True``.
    output_dictionary : bool, optional
        Whether to return the output as a dictionary.
        If False, returns a numpy array. The default is ``True``
    optimizer : str, optional
        The optimization method to use. Default is ``"Nelder-Mead"``.

    Returns
    -------
    dict or numpy.ndarray
        The estimated parameters of the lognormal distribution.
        If ``output_dictionary`` is ``True``, returns a dictionary with keys ``Nt``, ``mu``, and ``sigma``.
        If ``output_dictionary`` is ``False``,returns a numpy array with values [Nt, mu, sigma].

    Notes
    -----
    The lognormal distribution is defined as:
    N(D) = Nt / (sqrt(2 * pi) * sigma * D) * exp(-(ln(D) - mu)**2 / (2 * sigma**2))
    where Nt is the total number of counts, mu is the mean of the log of the distribution,
    and sigma is the standard deviation of the log of the distribution.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm
    """
    # LogNormal
    # - mu = log(scale)
    # - loc = 0

    # Initialize bad results
    null_output = (
        {"Nt": np.nan, "mu": np.nan, "sigma": np.nan} if output_dictionary else np.array([np.nan, np.nan, np.nan])
    )

    # Define the CDF and PDF functions for the lognormal distribution
    def lognorm_cdf(x, params):
        sigma, scale = params
        return ss.lognorm.cdf(x, sigma, loc=0, scale=scale)

    def lognorm_pdf(x, params):
        sigma, scale = params
        return ss.lognorm.pdf(x, sigma, loc=0, scale=scale)

    # Define valid parameters for the lognormal distribution
    def param_constraints(params):
        sigma, scale = params
        return sigma > 0 and scale > 0

    # Definite initial guess for the parameters
    initial_params = [1.0, 1.0]  # sigma, scale

    # Define bounds for sigma and scale
    bounds = [(1e-6, None), (1e-6, None)]

    # Minimize the negative log-likelihood
    with suppress_warnings():
        result = minimize(
            compute_negative_log_likelihood,
            initial_params,
            args=(
                bin_edges,
                counts,
                lognorm_cdf,
                lognorm_pdf,
                param_constraints,
                probability_method,
                likelihood,
                truncated_likelihood,
            ),
            bounds=bounds,
            method=optimizer,
        )

    # Check if the fit had success
    if not result.success:
        return null_output

    # Define Nt
    Nt = np.sum(counts).item()

    # Retrieve parameters
    params = result.x
    if truncated_likelihood:
        Nt = get_adjusted_nt(cdf=lognorm_cdf, params=params, Nt=Nt, bin_edges=bin_edges)
    sigma, scale = params
    mu = np.log(scale)

    # Define output
    output = {"Nt": Nt, "mu": mu, "sigma": sigma} if output_dictionary else np.array([Nt, mu, sigma])
    return output


def estimate_exponential_parameters(
    counts,
    bin_edges,
    probability_method="cdf",
    likelihood="multinomial",
    truncated_likelihood=True,
    output_dictionary=True,
    optimizer="Nelder-Mead",
):
    """
    Estimate the parameters of an exponential distribution given histogram data.

    Parameters
    ----------
    counts : array-like
        The counts for each bin in the histogram.
    bin_edges : array-like
        The edges of the bins.
    probability_method : str, optional
        The method to compute probabilities, either ``"cdf"`` or ``"pdf"``. The default is ``"cdf"``.
    likelihood : str, optional
        The likelihood function to use, either ``"multinomial"`` or ``"poisson"``.
        The default is ``"multinomial"``.
    truncated_likelihood : bool, optional
        Whether to use truncated likelihood. The default is ``True``.
    output_dictionary : bool, optional
        Whether to return the output as a dictionary.
        If False, returns a numpy array. The default is ``True``
    optimizer : str, optional
        The optimization method to use. Default is ``"Nelder-Mead"``.

    Returns
    -------
    dict or numpy.ndarray
        The estimated parameters of the exponential distribution.
        If ``output_dictionary`` is ``True``, returns a dictionary with keys ``N0`` and ``Lambda``.
        If `output_dictionary` is ``False``, returns a numpy array with [N0, Lambda].

    Notes
    -----
    The exponential distribution is defined as:
        N(D) = N0 * exp(-Lambda * D) = Nt * Lambda * exp(-Lambda * D)
    where Lambda = 1 / scale and N0 = Nt * Lambda.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html
    """
    # Initialize bad results
    null_output = {"N0": np.nan, "Lambda": np.nan} if output_dictionary else np.array([np.nan, np.nan])

    # Define the CDF and PDF functions for the exponential distribution
    def exp_cdf(x, params):
        scale = params[0]
        return ss.expon.cdf(x, loc=0, scale=scale)

    def exp_pdf(x, params):
        scale = params[0]
        return ss.expon.pdf(x, loc=0, scale=scale)

    # Define valid parameters for the exponential distribution
    def param_constraints(params):
        scale = params[0]
        return scale > 0

    # Definite initial guess for the scale parameter
    initial_params = [1.0]  # scale

    # Define bounds for scale
    bounds = [(1e-6, None)]

    # Minimize the negative log-likelihood
    with suppress_warnings():
        result = minimize(
            compute_negative_log_likelihood,
            initial_params,
            args=(
                bin_edges,
                counts,
                exp_cdf,
                exp_pdf,
                param_constraints,
                probability_method,
                likelihood,
                truncated_likelihood,
            ),
            bounds=bounds,
            method=optimizer,
        )

    # Check if the fit had success
    if not result.success:
        return null_output

    # Define Nt
    Nt = np.sum(counts).item()

    # Retrieve parameters
    params = result.x
    if truncated_likelihood:
        Nt = get_adjusted_nt(cdf=exp_cdf, params=params, Nt=Nt, bin_edges=bin_edges)
    scale = params[0]
    Lambda = 1 / scale
    N0 = Nt * Lambda

    # Define output
    output = {"N0": N0, "Lambda": Lambda} if output_dictionary else np.array([N0, Lambda])
    return output


def estimate_gamma_parameters(
    counts,
    bin_edges,
    probability_method="cdf",
    likelihood="multinomial",
    truncated_likelihood=True,
    output_dictionary=True,
    optimizer="Nelder-Mead",
):
    """
    Estimate the parameters of a gamma distribution given histogram data.

    Parameters
    ----------
    counts : array-like
        The counts for each bin in the histogram.
    bin_edges : array-like
        The edges of the bins.
    probability_method : str, optional
        The method to compute probabilities, either ``"cdf"`` or ``"pdf"``. The default is ``"cdf"``.
    likelihood : str, optional
        The likelihood function to use, either ``"multinomial"`` or ``"poisson"``.
        The default is ``"multinomial"``.
    truncated_likelihood : bool, optional
        Whether to use truncated likelihood. The default is ``True``.
    output_dictionary : bool, optional
        Whether to return the output as a dictionary.
        If False, returns a numpy array. The default is ``True``
    optimizer : str, optional
        The optimization method to use. Default is ``"Nelder-Mead"``.

    Returns
    -------
    dict or numpy.ndarray
        The estimated parameters of the gamma distribution.
        If ``output_dictionary`` is ``True``, returns a dictionary with keys ``N0``, ``mu`` and ``Lambda``.
        If `output_dictionary` is ``False``, returns a numpy array with [N0, mu, Lambda].

    Notes
    -----
    The gamma distribution is defined as:
        N(D) = N0 * D**mu * exp(-Lambda*D)
    where Lambda = 1/scale, and mu = a - 1 with ``a`` being the shape parameter of the gamma distribution.
    N0 is defined as N0 = Nt*Lambda**(mu+1)/gamma(mu+1).

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

    """
    # Initialize bad results
    null_output = (
        {"N0": np.nan, "mu": np.nan, "lambda": np.nan} if output_dictionary else np.array([np.nan, np.nan, np.nan])
    )

    # Define the CDF and PDF functions for the gamma distribution
    def gamma_cdf(x, params):
        a, scale = params
        return ss.gamma.cdf(x, a, loc=0, scale=scale)

    def gamma_pdf(x, params):
        a, scale = params
        return ss.gamma.pdf(x, a, loc=0, scale=scale)

    # Define valid parameters for the gamma distribution
    def param_constraints(params):
        a, scale = params
        return a > 0 and scale > 0

    # Definite initial guess for the parameters
    initial_params = [1.0, 1.0]  # a, scale (mu=a-1, a=mu+1)

    # Define bounds for a and scale
    bounds = [(1e-6, None), (1e-6, None)]

    # Minimize the negative log-likelihood
    with suppress_warnings():
        result = minimize(
            compute_negative_log_likelihood,
            initial_params,
            args=(
                bin_edges,
                counts,
                gamma_cdf,
                gamma_pdf,
                param_constraints,
                probability_method,
                likelihood,
                truncated_likelihood,
            ),
            method=optimizer,
            bounds=bounds,
        )

    # Check if the fit had success
    if not result.success:
        return null_output

    # Define Nt
    Nt = np.sum(counts).item()

    # Retrieve parameters
    params = result.x
    if truncated_likelihood:
        Nt = get_adjusted_nt(cdf=gamma_cdf, params=params, Nt=Nt, bin_edges=bin_edges)
    a, scale = params
    mu = a - 1
    Lambda = 1 / scale

    # Compute N0
    # - Use logarithmic computations to prevent overflow
    # - N0 = Nt * Lambda ** (mu + 1) / gamma(mu + 1)
    # with suppress_warnings():
    log_N0 = np.log(Nt) + (mu + 1) * np.log(Lambda) - gammaln(mu + 1)
    N0 = np.exp(log_N0)
    if not np.isfinite(N0):
        N0 = np.nan

    # Define output
    output = {"N0": N0, "mu": mu, "Lambda": Lambda} if output_dictionary else np.array([N0, mu, Lambda])
    return output


def get_gamma_parameters(
    ds,
    probability_method="cdf",
    likelihood="multinomial",
    truncated_likelihood=True,
    optimizer="Nelder-Mead",
):
    """
    Estimate gamma distribution parameters for drop size distribution (DSD) data.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing drop size distribution data. It must include the following variables:
        - ``drop_number_concentration``: The number concentration of drops.
        - ``diameter_bin_width``": The width of each diameter bin.
        - ``diameter_bin_lower``: The lower bounds of the diameter bins.
        - ``diameter_bin_upper``: The upper bounds of the diameter bins.
        - ``diameter_bin_center``: The center values of the diameter bins.
    probability_method : str, optional
        Method to compute probabilities. The default is ``cdf``.
    likelihood : str, optional
        Likelihood function to use for fitting. The default is ``multinomial``.
    truncated_likelihood : bool, optional
        Whether to use truncated likelihood. The default is ``True``.
    optimizer : str, optional
        Optimization method to use. The default is ``Nelder-Mead``.

    Returns
    -------
    xarray.Dataset
        Dataset containing the estimated gamma distribution parameters:
        - ``N0``: Intercept parameter.
        - ``mu``: Shape parameter.
        - ``Lambda``: Scale parameter.
        The dataset will also have an attribute ``disdrodb_psd_model`` set to ``GammaPSD``.

    Notes
    -----
    The function uses `xr.apply_ufunc` to fit the lognormal distribution parameters
    in parallel, leveraging Dask for parallel computation.

    """
    # Define inputs
    counts = ds["drop_number_concentration"] * ds["diameter_bin_width"]
    diameter_breaks = np.append(ds["diameter_bin_lower"].data, ds["diameter_bin_upper"].data[-1])

    # Define kwargs
    kwargs = {
        "output_dictionary": False,
        "bin_edges": diameter_breaks,
        "probability_method": probability_method,
        "likelihood": likelihood,
        "truncated_likelihood": truncated_likelihood,
        "optimizer": optimizer,
    }

    # Fit distribution in parallel
    da_params = xr.apply_ufunc(
        estimate_gamma_parameters,
        counts,
        kwargs=kwargs,
        input_core_dims=[["diameter_bin_center"]],
        output_core_dims=[["parameters"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"parameters": 3}},  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )

    # Add parameters coordinates
    da_params = da_params.assign_coords({"parameters": ["N0", "mu", "Lambda"]})

    # Create parameters dataset
    ds_params = da_params.to_dataset(dim="parameters")

    # Add DSD model name to the attribute
    ds_params.attrs["disdrodb_psd_model"] = "GammaPSD"

    return ds_params


def get_lognormal_parameters(
    ds,
    probability_method="cdf",
    likelihood="multinomial",
    truncated_likelihood=True,
    optimizer="Nelder-Mead",
):
    """
    Estimate lognormal distribution parameters for drop size distribution (DSD) data.

    Parameters
    ----------
    ds : xarray.Dataset
    Input dataset containing drop size distribution data. It must include the following variables:
    - ``drop_number_concentration``: The number concentration of drops.
    - ``diameter_bin_width``": The width of each diameter bin.
    - ``diameter_bin_lower``: The lower bounds of the diameter bins.
    - ``diameter_bin_upper``: The upper bounds of the diameter bins.
    - ``diameter_bin_center``: The center values of the diameter bins.
    probability_method : str, optional
        Method to compute probabilities. The default is ``cdf``.
    likelihood : str, optional
        Likelihood function to use for fitting. The default is ``multinomial``.
    truncated_likelihood : bool, optional
        Whether to use truncated likelihood. The default is ``True``.
    optimizer : str, optional
        Optimization method to use. The default is ``Nelder-Mead``.

    Returns
    -------
    xarray.Dataset
        Dataset containing the estimated lognormal distribution parameters:
        - ``Nt``: Total number concentration.
        - ``mu``: Mean of the lognormal distribution.
        - ``sigma``: Standard deviation of the lognormal distribution.
        The resulting dataset will have an attribute ``disdrodb_psd_model`` set to ``LognormalPSD``.

    Notes
    -----
    The function uses `xr.apply_ufunc` to fit the lognormal distribution parameters
    in parallel, leveraging Dask for parallel computation.

    """
    # Define inputs
    counts = ds["drop_number_concentration"] * ds["diameter_bin_width"]
    diameter_breaks = np.append(ds["diameter_bin_lower"].data, ds["diameter_bin_upper"].data[-1])

    # Define kwargs
    kwargs = {
        "output_dictionary": False,
        "bin_edges": diameter_breaks,
        "probability_method": probability_method,
        "likelihood": likelihood,
        "truncated_likelihood": truncated_likelihood,
        "optimizer": optimizer,
    }

    # Fit distribution in parallel
    da_params = xr.apply_ufunc(
        estimate_lognormal_parameters,
        counts,
        kwargs=kwargs,
        input_core_dims=[["diameter_bin_center"]],
        output_core_dims=[["parameters"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"parameters": 3}},  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )

    # Add parameters coordinates
    da_params = da_params.assign_coords({"parameters": ["Nt", "mu", "sigma"]})

    # Create parameters dataset
    ds_params = da_params.to_dataset(dim="parameters")

    # Add DSD model name to the attribute
    ds_params.attrs["disdrodb_psd_model"] = "LognormalPSD"

    return ds_params


def get_exponential_parameters(
    ds,
    probability_method="cdf",
    likelihood="multinomial",
    truncated_likelihood=True,
    optimizer="Nelder-Mead",
):
    """
    Estimate the parameters of an exponential particle size distribution (PSD) from the given dataset.

    Fitting this model is equivalent to fitting a GammaPSD model fixing ``mu`` to 0.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing drop number concentration data and diameter information.
        It must include the following variables:
        - ``drop_number_concentration``: The number concentration of drops.
        - ``diameter_bin_width``": The width of each diameter bin.
        - ``diameter_bin_lower``: The lower bounds of the diameter bins.
        - ``diameter_bin_upper``: The upper bounds of the diameter bins.
        - ``diameter_bin_center``: The center values of the diameter bins.
    probability_method : str, optional
        Method to compute probabilities. The default is ``cdf``.
    likelihood : str, optional
        Likelihood function to use for fitting. The default is ``multinomial``.
    truncated_likelihood : bool, optional
        Whether to use truncated likelihood. The default is ``True``.
    optimizer : str, optional
        Optimization method to use. The default is ``Nelder-Mead``.

    Returns
    -------
    xarray.Dataset
        Dataset containing the estimated expontial distribution parameters:
        - ``N0``: Intercept parameter.
        - ``Lambda``: Scale parameter.
        The resulting dataset will have an attribute ``disdrodb_psd_model`` set to ``ExponentialPSD``.

    Notes
    -----
    The function uses `xr.apply_ufunc` to fit the exponential distribution parameters
    in parallel, leveraging Dask for parallel computation.

    """
    # Define inputs
    counts = ds["drop_number_concentration"] * ds["diameter_bin_width"]
    diameter_breaks = np.append(ds["diameter_bin_lower"].data, ds["diameter_bin_upper"].data[-1])

    # Define kwargs
    kwargs = {
        "output_dictionary": False,
        "bin_edges": diameter_breaks,
        "probability_method": probability_method,
        "likelihood": likelihood,
        "truncated_likelihood": truncated_likelihood,
        "optimizer": optimizer,
    }

    # Fit distribution in parallel
    da_params = xr.apply_ufunc(
        estimate_exponential_parameters,
        counts,
        kwargs=kwargs,
        input_core_dims=[["diameter_bin_center"]],
        output_core_dims=[["parameters"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"parameters": 2}},  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )

    # Add parameters coordinates
    da_params = da_params.assign_coords({"parameters": ["N0", "Lambda"]})

    # Create parameters dataset
    ds_params = da_params.to_dataset(dim="parameters")

    # Add DSD model name to the attribute
    ds_params.attrs["disdrodb_psd_model"] = "ExponentialPSD"

    return ds_params


####-------------------------------------------------------------------------------------------------------------------.


def _estimate_gamma_parameters_johnson(
    drop_number_concentration,
    diameter,
    diameter_breaks,
    output_dictionary=True,
    method="Nelder-Mead",
    mu=0.5,
    Lambda=3,
    **kwargs,
):
    """Maximum likelihood estimation of Gamma model.

    N(D) = N_t * lambda**(mu+1) / gamma(mu+1) D**mu exp(-lambda*D)

    Args:
        spectra: The DSD for which to find parameters [mm-1 m-3].
        widths: Class widths for each DSD bin [mm].
        diams: Class-centre diameters for each DSD bin [mm].
        mu: Initial value for shape parameter mu [-].
        lambda_param: Initial value for slope parameter lambda [mm^-1].
        kwargs: Extra arguments for the optimization process.

    Returns
    -------
        Dictionary with estimated mu, lambda, and N0.
        mu (shape) N0 (scale) lambda(slope)

    Notes
    -----
    The last bin counts are not accounted in the fitting procedure !

    References
    ----------
    Johnson, R. W., D. V. Kliche, and P. L. Smith, 2011: Comparison of Estimators for Parameters of Gamma Distributions
    with Left-Truncated Samples. J. Appl. Meteor. Climatol., 50, 296-310, https://doi.org/10.1175/2010JAMC2478.1

    Johnson, R.W., Kliche, D., & Smith, P.L. (2010).
    Maximum likelihood estimation of gamma parameters for coarsely binned and truncated raindrop size data.
    Quarterly Journal of the Royal Meteorological Society, 140. DOI:10.1002/qj.2209

    """
    # Initialize bad results
    if output_dictionary:
        null_output = {"mu": np.nan, "lambda": np.nan, "N0": np.nan}
    else:
        null_output = np.array([np.nan, np.nan, np.nan])

    # Initialize parameters
    # --> Ideally with method of moments estimate
    # --> See equation 8 of Johnson's 2013
    x0 = [mu, Lambda]

    # Compute diameter_bin_width
    diameter_bin_width = np.diff(diameter_breaks)

    # Convert drop_number_concentration from mm-1 m-3 to m-3.
    spectra = np.asarray(drop_number_concentration) * diameter_bin_width

    # Define cost function
    # - Parameter to be optimized on first positions
    def _cost_function(parameters, spectra, diameter_breaks):
        # Assume spectra to be in unit [m-3] (drop_number_concentration*diameter_bin_width) !
        mu, Lambda = parameters
        # Precompute gamma integrals between various diameter bins
        # - gamminc(mu+1) already divides the integral by gamma(mu+1) !
        pgamma_d = gammainc(mu + 1, Lambda * diameter_breaks)
        # Compute probability with interval
        delta_pgamma_bins = pgamma_d[1:] - pgamma_d[:-1]
        # Compute normalization over interval
        denominator = pgamma_d[-1] - pgamma_d[0]
        # Compute cost function
        # a = mu - 1, x = lambda
        if mu > -1 and Lambda > 0:
            cost = np.sum(-spectra * np.log(delta_pgamma_bins / denominator))
            return cost
        return np.inf

    # Minimize the cost function
    with suppress_warnings():
        bounds = [(0, None), (0, None)]  # Force mu and lambda to be non-negative
        res = minimize(
            _cost_function,
            x0=x0,
            args=(spectra, diameter_breaks),
            method=method,
            bounds=bounds,
            **kwargs,
        )

    # Check if the fit had success
    if not res.success:
        return null_output

    # Extract parameters
    mu = res.x[0]  # [-]
    Lambda = res.x[1]  # [mm-1]

    # Estimate tilde_N_T using the total drop concentration
    tilde_N_T = np.sum(drop_number_concentration * diameter_bin_width)  # [m-3]

    # Estimate proportion of missing drops (Johnson's 2011 Eqs. 3)
    D = diameter
    p = 1 - np.sum((Lambda ** (mu + 1)) / gamma(mu + 1) * D**mu * np.exp(-Lambda * D) * diameter_bin_width)  # [-]

    # Convert tilde_N_T to N_T using Johnson's 2013 Eqs. 3 and 4.
    # - Adjusts for the proportion of drops not observed
    N_T = tilde_N_T / (1 - p)  # [m-3]

    # Compute N0
    N0 = N_T * (Lambda ** (mu + 1)) / gamma(mu + 1)  # [m-3 * mm^(-mu-1)]

    # Compute Dm
    # Dm = (mu + 4)/ Lambda

    # Compute Nw
    # Nw = N0* D^mu / f(mu) , with f(mu of the Normalized PSD)

    # Define output
    output = {"mu": mu, "Lambda": Lambda, "N0": N0} if output_dictionary else np.array([mu, Lambda, N0])
    return output


def get_gamma_parameters_johnson2014(ds, method="Nelder-Mead"):
    """Deprecated model. See Gamma Model with truncated_likelihood and 'pdf'."""
    drop_number_concentration = ds["drop_number_concentration"]
    diameter = ds["diameter_bin_center"]
    diameter_breaks = np.append(ds["diameter_bin_lower"].data, ds["diameter_bin_upper"].data[-1])
    # Define kwargs
    kwargs = {
        "output_dictionary": False,
        "diameter_breaks": diameter_breaks,
        "method": method,
    }
    da_params = xr.apply_ufunc(
        _estimate_gamma_parameters_johnson,
        drop_number_concentration,
        diameter,
        # diameter_bin_width,
        kwargs=kwargs,
        input_core_dims=[["diameter_bin_center"], ["diameter_bin_center"]],  # ["diameter_bin_center"],
        output_core_dims=[["parameters"]],
        vectorize=True,
    )

    # Add parameters coordinates
    da_params = da_params.assign_coords({"parameters": ["mu", "Lambda", "N0"]})

    # Convert to skill Dataset
    ds_params = da_params.to_dataset(dim="parameters")
    return ds_params


####-----------------------------------------------------------------------------------------.
#### Grid Search
# Optimize Normalized Gamma


def _estimate_normalized_gamma_w_d50(Nd, D, D50, Nw, order=2):
    # Define cost function
    # - Parameter to be optimized on first position
    def _cost_function(parameters, Nd, D, D50, Nw):
        mu = parameters
        cost = np.nansum(np.power(np.abs(Nd - NormalizedGammaPSD.formula(D=D, D50=D50, Nw=Nw, mu=mu)), order))
        return cost

    # Optimize for mu
    with suppress_warnings():
        res = scipy.optimize.minimize_scalar(_cost_function, bounds=(-1, 20), args=(Nd, D, D50, Nw), method="bounded")
    if not res.success or res.x > 20:
        return np.nan
    return res.x


def get_normalized_gamma_parameters(ds, order=2):
    r"""Estimate $\mu$ of a Normalized Gamma distribution for a single observed DSD.

    The D50 and Nw parameters of the Normalized Gamma distribution are derived empirically from the observed DSD.
    $\mu$ is derived by minimizing the errors between the observed DSD and modelled Normalized Gamma distribution.

    Parameters
    ----------
    Nd : array_like
        A drop size distribution
    D50: optional, float
        Median drop diameter in mm. If none is given, it will be estimated.
    Nw: optional, float
        Normalized Intercept Parameter. If none is given, it will be estimated.
    order: optional, float
        Order to which square the error when computing the sum of errors.
        Order = 2 is equivalent to minimize the mean squared error (MSE) (L2 norm). The default is 2.
        Order = 1 is equivalent to minimize the mean absolute error (MAE) (L1 norm).
        Higher orders typically stretch higher the gamma distribution.

    Returns
    -------
    mu: integer
        Best estimate for DSD shape parameter $\mu$.

    References
    ----------
    Bringi and Chandrasekar 2001.
    """
    # TODO: another order to square the D
    # TODO: minimize rain rate / reflectivity / lwc ...

    # Extract variables
    Nd = ds["drop_number_concentration"]
    D = ds["diameter_bin_center"]

    # Define D50 and Nw using empirical moments
    D50 = ds["D50"]
    Nw = ds["Nw"]

    # TODO: define here normalized (on ntot and lwc) gamma formula (with Dm or D50)
    # - D50 or Dm
    # - Normalized to Ntot or LWC.

    # Define kwargs
    kwargs = {
        "order": order,
    }
    # Estimate mu
    mu = xr.apply_ufunc(
        _estimate_normalized_gamma_w_d50,
        Nd,
        D,
        D50,
        Nw,
        kwargs=kwargs,
        input_core_dims=[["diameter_bin_center"], ["diameter_bin_center"], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
    )
    # Create Dataset
    ds = xr.Dataset()
    ds["Nw"] = Nw
    ds["mu"] = mu
    ds["D50"] = D50

    # Add DSD model name to the attribute
    ds.attrs["disdrodb_psd_model"] = "NormalizedGammaPSD"
    return ds


####-----------------------------------------------------------------.
#### Methods of Moments


def get_exponential_parameters_Zhang2008(moment_l, moment_m, l, m):  # noqa: E741
    """Calculate Exponential DSD parameters using the method of moments (MOM).

    The choice of moments is given in the parameters.

    Parameters
    ----------
    moment_l: float
        First moment to use.
    moment_l: float
        Second moment to use.
    l : float
        Moment order.
    m : float
        Moment order,

    References
    ----------
    [1] Zhang, et. al., 2008, Diagnosing the Intercept Parameter for Exponential Raindrop Size
        Distribution Based on Video Disdrometer Observations: Model Development. J. Appl.
        Meteor. Climatol.,
        https://doi.org/10.1175/2008JAMC1876.1
    """
    num = moment_l * gamma(m + 1)
    den = moment_m * gamma(l + 1)
    Lambda = np.power(num / den, (1 / (m - l)))
    N0 = moment_l * np.power(Lambda, l + 1) / gamma(l + 1)
    return Lambda, N0


def get_exponential_parameters_M34(moment_3, moment_4):
    """Compute exponential distribution parameters following Testud 2001.

    References
    ----------
    Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001:
    The Concept of “Normalized” Distribution to Describe Raindrop Spectra:
    A Tool for Cloud Physics and Cloud Remote Sensing.
    J. Appl. Meteor. Climatol., 40, 1118-1140,
    https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2
    """
    N0 = 256 / gamma(4) * moment_3**5 / moment_4**4
    Dm = moment_4 / moment_3
    Lambda = 4 / Dm
    return Lambda, N0


# M246 DEFAULT FOR GAMMA ?
# -----------------------------

# LMOM (Johnson et al., 2014)


def get_gamma_parameters_M012(M0, M1, M2):
    """Compute gamma distribution parameters following Cao et al., 2009.

    References
    ----------
    Cao, Q., and G. Zhang, 2009:
    Errors in Estimating Raindrop Size Distribution Parameters Employing Disdrometer  and Simulated Raindrop Spectra.
    J. Appl. Meteor. Climatol., 48, 406-425, https://doi.org/10.1175/2008JAMC2026.1.
    """
    G = M1**3 / M0 / M2
    mu = 1 / (1 - G) - 2
    Lambda = M0 / M1 * (mu + 1)
    N0 = Lambda ** (mu + 1) * M0 / gamma(mu + 1)
    return mu, Lambda, N0


def get_gamma_parameters_M234(M2, M3, M4):
    """Compute gamma distribution parameters following Cao et al., 2009.

    References
    ----------
    Cao, Q., and G. Zhang, 2009:
    Errors in Estimating Raindrop Size Distribution Parameters Employing Disdrometer  and Simulated Raindrop Spectra.
    J. Appl. Meteor. Climatol., 48, 406-425, https://doi.org/10.1175/2008JAMC2026.1.
    """
    G = M3**2 / M2 / M4
    mu = 1 / (1 - G) - 4
    Lambda = M2 / M3 * (mu + 3)
    N0 = Lambda ** (mu + 3) * M2 / gamma(mu + 3)
    return mu, Lambda, N0


def get_gamma_parameters_M246(M2, M4, M6):
    """Compute gamma distribution parameters following Ulbrich 1998.

    References
    ----------
    Ulbrich, C. W., and D. Atlas, 1998:
    Rainfall Microphysics and Radar Properties: Analysis Methods for Drop Size Spectra.
    J. Appl. Meteor. Climatol., 37, 912-923,
    https://doi.org/10.1175/1520-0450(1998)037<0912:RMARPA>2.0.CO;2

    Cao, Q., and G. Zhang, 2009:
    Errors in Estimating Raindrop Size Distribution Parameters Employing Disdrometer  and Simulated Raindrop Spectra.
    J. Appl. Meteor. Climatol., 48, 406-425, https://doi.org/10.1175/2008JAMC2026.1.

    Thurai, M., Williams, C.R., Bringi, V.N., 2014:
    Examining the correlations between drop size distribution parameters using data
    from two side-by-side 2D-video disdrometers.
    Atmospheric Research, 144, 95-110, https://doi.org/10.1016/j.atmosres.2014.01.002.
    """
    G = M4**2 / M2 / M6

    # TODO: Different formulas !
    # Thurai et al., 2014 (A4), Ulbrich et al., 1998 (2)
    #  mu = ((7.0 - 11.0 * G) -
    #  np.sqrt((7.0 - 11.0 * G) ** 2.0 - 4.0 * (G - 1.0) * (30.0 * G - 12.0)) / (2.0 * (G - 1.0)))
    mu = (7.0 - 11.0 * G) - np.sqrt(G**2 + 89 * G + 1) / (2.0 * (G - 1.0))

    # Cao et al., 2009 (B3)
    # --> Wrong ???
    mu = (7.0 - 11.0 * G) - np.sqrt(G**2 + 14 * G + 1) / (2.0 * (G - 1.0))

    Lambda = np.sqrt((4 + mu) * (3 + mu) * M2 / M4)
    # Cao et al., 2009
    N0 = M2 * Lambda ** (3 + mu) / gamma(3 + mu)
    # # Thurai et al., 2014
    # N0 = M3 * Lambda ** (4 + mu) / gamma(4 + mu)
    # # Ulbrich et al., 1998
    # N0 = M6 * Lambda ** (7.0 + mu) / gamma(7 + mu)
    return mu, Lambda, N0


def get_gamma_parameters_M456(M4, M5, M6):
    """Compute gamma distribution parameters following Cao et al., 2009.

    References
    ----------
    Cao, Q., and G. Zhang, 2009:
    Errors in Estimating Raindrop Size Distribution Parameters Employing Disdrometer  and Simulated Raindrop Spectra.
    J. Appl. Meteor. Climatol., 48, 406-425, https://doi.org/10.1175/2008JAMC2026.1.
    """
    G = M5**2 / M4 / M6
    mu = 1 / (1 - G) - 6
    Lambda = M4 / M5 * (mu + 5)
    N0 = Lambda ** (mu + 5) * M4 / gamma(mu + 5)
    return mu, Lambda, N0


def get_gamma_parameters_M346(M3, M4, M6):
    """Compute gamma distribution parameters following Kozu 1991.

    References
    ----------
    Kozu, T., and K. Nakamura, 1991:
    Rainfall Parameter Estimation from Dual-Radar Measurements
    Combining Reflectivity Profile and Path-integrated Attenuation.
    J. Atmos. Oceanic Technol., 8, 259-270, https://doi.org/10.1175/1520-0426(1991)008<0259:RPEFDR>2.0.CO;2

    Tokay, A., and D. A. Short, 1996:
    Evidence from Tropical Raindrop Spectra of the Origin of Rain from
    Stratiform versus Convective Clouds.
    J. Appl. Meteor. Climatol., 35, 355-371,
    https://doi.org/10.1175/1520-0450(1996)035<0355:EFTRSO>2.0.CO;2

    Cao, Q., and G. Zhang, 2009:
    Errors in Estimating Raindrop Size Distribution Parameters Employing Disdrometer  and Simulated Raindrop Spectra.
    J. Appl. Meteor. Climatol., 48, 406-425, https://doi.org/10.1175/2008JAMC2026.1.
    """
    G = M4**3 / M3**2 / M6

    # Kozu
    mu = (5.5 * G - 4 + np.sqrt(G * (G * 0.25 + 2))) / (1 - G)

    # Cao et al., 2009 (equivalent)
    # mu = (11 * G - 8 + np.sqrt(G * (G + 8))) / (2 * (1 - G))

    Lambda = (mu + 4) * M3 / M4
    N0 = Lambda ** (mu + 4) * M3 / gamma(mu + 4)
    return mu, Lambda, N0


def get_lognormal_parameters_M346(M3, M4, M6):
    """Compute lognormal distribution parameters following Kozu1991.

    References
    ----------
    Kozu, T., and K. Nakamura, 1991:
    Rainfall Parameter Estimation from Dual-Radar Measurements
    Combining Reflectivity Profile and Path-integrated Attenuation.
    J. Atmos. Oceanic Technol., 8, 259-270, https://doi.org/10.1175/1520-0426(1991)008<0259:RPEFDR>2.0.CO;2
    """
    L3 = np.log(M3)
    L4 = np.log(M4)
    L6 = np.log(M6)
    Nt = np.exp((24 * L3 - 27 * L4 - 6 * L6) / 3)
    mu = (-10 * L3 + 13.5 * L4 - 3.5 * L6) / 3
    sigma = (2 * L3 - 3 * L4 + L6) / 3
    return mu, sigma, Nt


####--------------------------------------------------------------------------------------.
#### Wrappers for fitting


distribution_dictionary = {
    "gamma": get_gamma_parameters,
    "lognormal": get_lognormal_parameters,
    "exponential": get_exponential_parameters,
    "normalized_gamma": get_normalized_gamma_parameters,
}


def available_distributions():
    """Return the lst of available PSD distributions."""
    return list(distribution_dictionary)


def estimate_model_parameters(
    ds,
    distribution,
    probability_method="cdf",
    likelihood="multinomial",
    truncated_likelihood=True,
    optimizer="Nelder-Mead",
    order=2,
):
    """
    Estimate model parameters for a given distribution.

    Parameters
    ----------
     ds : xarray.Dataset
        Input dataset containing drop number concentration data and diameter information.
        It must include the following variables:
        - ``drop_number_concentration``: The number concentration of drops.
        - ``diameter_bin_width``": The width of each diameter bin.
        - ``diameter_bin_lower``: The lower bounds of the diameter bins.
        - ``diameter_bin_upper``: The upper bounds of the diameter bins.
        - ``diameter_bin_center``: The center values of the diameter bins.
    distribution : str
        The type of distribution to fit. See ``available_distributions()`` for the available distributions.
    probability_method : str, optional
        Method to compute probabilities. The default is ``cdf``.
    likelihood : str, optional
        Likelihood function to use for fitting. The default is ``multinomial``.
    truncated_likelihood : bool, optional
        Whether to use truncated likelihood. The default is ``True``.
    optimizer : str, optional
        Optimization method to use. The default is ``Nelder-Mead``.
    order : int, optional
        The order parameter for the ``normalized_gamma`` distribution.
        The default value is 2.

    Returns
    -------
    xarray.Dataset
        The dataset containing the estimated parameters.

    """
    # TODO: initialize with moment methods
    # TODO: optimize for rain rate ...

    # Likelihood poisson ...
    # Truncated Maximum Likelihood method (TML)
    # Constrained TML
    # Integrating the PDFs of the distribution over the bin ranges. This accounts for the irregular bin widths

    # Check valid distribution
    valid_distribution = list(distribution_dictionary)
    if distribution not in valid_distribution:
        raise ValueError(f"Invalid distribution {distribution}. Valid values are {valid_distribution}.")

    # Check valid likelihood
    valid_likelihood = ["multinomial", "poisson"]
    if likelihood not in valid_likelihood:
        raise ValueError(f"Invalid likelihood {likelihood}. Valid values are {valid_likelihood}.")

    # Check valid probability_method
    valid_probability_method = ["cdf", "pdf"]
    if likelihood not in valid_likelihood:
        raise ValueError(
            f"Invalid probability_method {probability_method}. Valid values are {valid_probability_method}.",
        )

    # Retrieve estimation function
    func = distribution_dictionary[distribution]

    # Retrieve parameters
    if distribution in ["gamma", "lognormal", "exponential"]:
        ds_params = func(
            ds=ds,
            probability_method=probability_method,
            likelihood=likelihood,
            truncated_likelihood=truncated_likelihood,
            optimizer=optimizer,
        )
    else:  # normalized_gamma
        ds_params = get_normalized_gamma_parameters(ds=ds, order=order)

    # Return dataset with parameters
    return ds_params
