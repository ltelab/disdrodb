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
"""Routines for PSD fitting."""

import numpy as np
import scipy.stats as ss
import xarray as xr
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import gamma, gammaln  # Regularized lower incomplete gamma function

from disdrodb.constants import DIAMETER_DIMENSION
from disdrodb.fall_velocity import get_rain_fall_velocity_from_ds
from disdrodb.l2.empirical_dsd import (
    get_median_volume_drop_diameter,
    get_moment,
    get_normalized_intercept_parameter_from_moments,
    get_total_number_concentration,
)
from disdrodb.psd.grid_search import (
    check_censoring,
    check_target,
    check_transformation,
    # check_loss,
    compute_weighted_loss,
)
from disdrodb.psd.models import (
    ExponentialPSD,
    GammaPSD,
    GeneralizedGammaPSD,
    LognormalPSD,
    NormalizedGammaPSD,
    NormalizedGeneralizedGammaPSD,
)
from disdrodb.utils.manipulations import get_diameter_bin_edges
from disdrodb.utils.warnings import suppress_warnings

# gamma(>171) return inf !

####--------------------------------------------------------------------------------------.
#### Notes
## Variable requirements for fitting PSD Models
# - drop_number_concentration and diameter coordinates
# - Always recompute other parameters to ensure not use model parameters of L2M

# ML: None

# MOM: moments
# --> get_moment(drop_number_concentration, diameter, diameter_bin_width, moment)

# GS: fall_velocity if target optimization is R (rain)
# - NormalizedGamma: "Nw", "D50"
# --> get_normalized_intercept_parameter_from_moments(moment_3, moment_4)
# --> get_median_volume_drop_diameter(drop_number_concentration, diameter, diameter_bin_width):
# --> get_mean_volume_drop_diameter(moment_3, moment_4)  (Dm)

# - LogNormal,Exponential, Gamma: Nt
# --> get_total_number_concentration(drop_number_concentration, diameter_bin_width)

####--------------------------------------------------------------------------------------.
#### Maximum Likelihood (ML)


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
    # --> Alternative:
    # - p = 1 - np.sum(pdf(diameter, params)* diameter_bin_width)  # [-]
    # - p = 1 - np.sum((Lambda ** (mu + 1)) / gamma(mu + 1) * D**mu * np.exp(-Lambda * D) * diameter_bin_width)  # [-]
    p = 1 - np.diff(cdf([bin_edges[0], bin_edges[-1]], params)).item()  # [-]
    # Adjusts Nt for the proportion of missing drops
    nt_adj = np.nan if np.isclose(p, 1, atol=1e-12) else Nt / (1 - p)  # [m-3]
    return nt_adj


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
        obs counts in each bin (length N).
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
    mu,
    sigma,
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
    mu: float
        The initial guess of the mean of the log of the distribution.
        A good default value is 0.
    sigma: float
        The initial guess of the standard deviation of the log distribution.
        A good default value is 1.
    bin_edges : array-like
        The edges of the bins.
    probability_method : str, optional
        The method to compute probabilities, either ``"cdf"`` or ``"pdf"``. The default value is ``"cdf"``.
    likelihood : str, optional
        The likelihood function to use, either ``"multinomial"`` or ``"poisson"``.
        The default value is ``"multinomial"``.
    truncated_likelihood : bool, optional
        Whether to use truncated likelihood. The default value is ``True``.
    output_dictionary : bool, optional
        Whether to return the output as a dictionary.
        If False, returns a numpy array. The default value is ``True``
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
    # Definite initial guess for the parameters
    scale = np.exp(mu)  # mu = np.log(scale)
    initial_params = [sigma, scale]

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
    Lambda,
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
    Lambda : float
        The initial guess of the scale parameter.
        scale = 1 / lambda correspond to the scale parameter of the scipy.stats.expon distribution.
        A good default value is 1.
    bin_edges : array-like
        The edges of the bins.
    probability_method : str, optional
        The method to compute probabilities, either ``"cdf"`` or ``"pdf"``. The default value is ``"cdf"``.
    likelihood : str, optional
        The likelihood function to use, either ``"multinomial"`` or ``"poisson"``.
        The default value is ``"multinomial"``.
    truncated_likelihood : bool, optional
        Whether to use truncated likelihood. The default value is ``True``.
    output_dictionary : bool, optional
        Whether to return the output as a dictionary.
        If False, returns a numpy array. The default value is ``True``
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
    # Definite initial guess for parameters
    scale = 1 / Lambda
    initial_params = [scale]

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
    mu,
    Lambda,
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
    mu: float
        The initial guess of the shape parameter.
        a = mu + 1 correspond to the shape parameter of the scipy.stats.gamma distribution.
        A good default value is 0.
    lambda: float
        The initial guess of the scale parameter.
        scale = 1 / lambda correspond to the scale parameter of the scipy.stats.gamma distribution.
        A good default value is 1.
    bin_edges : array-like
        The edges of the bins.
    probability_method : str, optional
        The method to compute probabilities, either ``"cdf"`` or ``"pdf"``. The default value is ``"cdf"``.
    likelihood : str, optional
        The likelihood function to use, either ``"multinomial"`` or ``"poisson"``.
        The default value is ``"multinomial"``.
    truncated_likelihood : bool, optional
        Whether to use truncated likelihood. The default value is ``True``.
    output_dictionary : bool, optional
        Whether to return the output as a dictionary.
        If False, returns a numpy array. The default value is ``True``
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
    # Define initial guess for parameters
    a = mu + 1  # (mu = a-1, a = mu+1) (a > 0 --> mu=-1)
    scale = 1 / Lambda
    initial_params = [a, scale]

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
    # mu = -0.99 is a vertical line essentially ...
    def param_constraints(params):
        a, scale = params
        return a > 0.1 and scale > 0  # using a > 0 cause some troubles

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
    # - N0 = Nt * Lambda ** (mu + 1) / gamma(mu + 1)  # [m-3 * mm^(-mu-1)]
    with suppress_warnings():
        log_N0 = np.log(Nt) + (mu + 1) * np.log(Lambda) - gammaln(mu + 1)
        N0 = np.exp(log_N0)

    # Set parameters to np.nan if any of the parameters is not a finite number
    if not np.isfinite(N0) or not np.isfinite(mu) or not np.isfinite(Lambda):
        return null_output

    # Define output
    output = {"N0": N0, "mu": mu, "Lambda": Lambda} if output_dictionary else np.array([N0, mu, Lambda])
    return output


def _get_initial_lognormal_parameters(ds, mom_method=None):
    default_mu = 0  # mu = np.log(scale)
    default_sigma = 1
    if mom_method is None or mom_method == "None":
        ds_init = xr.Dataset(
            {
                "mu": default_mu,
                "sigma": default_sigma,
            },
        )
    else:
        ds_init = get_mom_parameters(
            ds=ds,
            psd_model="LognormalPSD",
            mom_methods=mom_method,
        )
        # If initialization results in some not finite number, set default value
        ds_init["mu"] = xr.where(
            np.logical_and(np.isfinite(ds_init["mu"]), ds_init["mu"] > 0),
            ds_init["mu"],
            default_mu,
        )
        ds_init["sigma"] = xr.where(np.isfinite(ds_init["sigma"]), ds_init["sigma"], default_sigma)
    return ds_init


def _get_initial_exponential_parameters(ds, mom_method=None):
    default_lambda = 1  # lambda = 1 /scale
    if mom_method is None or mom_method == "None":
        ds_init = xr.Dataset(
            {
                "Lambda": default_lambda,
            },
        )
    else:
        ds_init = get_mom_parameters(
            ds=ds,
            psd_model="ExponentialPSD",
            mom_methods=mom_method,
        )
        # If initialization results in some not finite number, set default value
        ds_init["Lambda"] = xr.where(np.isfinite(ds_init["Lambda"]), ds_init["Lambda"], default_lambda)
    return ds_init


def _get_initial_gamma_parameters(ds, mom_method=None):
    default_mu = 0  #  a = mu + 1  |   mu = a - 1
    default_lambda = 1  #  scale = 1 / Lambda
    if mom_method is None or mom_method == "None":
        ds_init = xr.Dataset(
            {
                "mu": default_mu,
                "Lambda": default_lambda,
            },
        )
    else:
        ds_init = get_mom_parameters(
            ds=ds,
            psd_model="GammaPSD",
            mom_methods=mom_method,
        )
        # If initialization results in some not finite number, set default value
        ds_init["mu"] = xr.where(
            np.logical_and(np.isfinite(ds_init["mu"]), ds_init["mu"] > -1),
            ds_init["mu"],
            default_mu,
        )
        ds_init["Lambda"] = xr.where(np.isfinite(ds_init["Lambda"]), ds_init["Lambda"], default_lambda)
    return ds_init


def get_gamma_parameters(
    ds,
    init_method=None,
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
        - The moments M0...M6 variables required to compute the initial parameters
          with the specified mom_method.
    init_method: str or list
        The method(s) of moments used to initialize the gamma parameters.
        If None (or 'None'), the scale parameter is set to 1 and mu to 0 (a=1).
    probability_method : str, optional
        Method to compute probabilities. The default value is ``cdf``.
    likelihood : str, optional
        Likelihood function to use for fitting. The default value is ``multinomial``.
    truncated_likelihood : bool, optional
        Whether to use truncated likelihood. The default value is ``True``.
        See Johnson et al., 2011 and 2011 for more information.
    optimizer : str, optional
        Optimization method to use. The default value is ``Nelder-Mead``.

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

    References
    ----------
    Johnson, R. W., D. V. Kliche, and P. L. Smith, 2011: Comparison of Estimators for Parameters of Gamma Distributions
    with Left-Truncated Samples. J. Appl. Meteor. Climatol., 50, 296-310, https://doi.org/10.1175/2010JAMC2478.1

    Johnson, R.W., Kliche, D., & Smith, P.L. (2010).
    Maximum likelihood estimation of gamma parameters for coarsely binned and truncated raindrop size data.
    Quarterly Journal of the Royal Meteorological Society, 140. DOI:10.1002/qj.2209

    """
    # Define inputs
    counts = ds["drop_number_concentration"] * ds["diameter_bin_width"]
    diameter_breaks = get_diameter_bin_edges(ds)

    # Define initial parameters (mu, Lambda)
    ds_init = _get_initial_gamma_parameters(ds, mom_method=init_method)

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
        ds_init["mu"],
        ds_init["Lambda"],
        kwargs=kwargs,
        input_core_dims=[[DIAMETER_DIMENSION], [], []],
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
    init_method=None,
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
        Method to compute probabilities. The default value is ``cdf``.
    likelihood : str, optional
        Likelihood function to use for fitting. The default value is ``multinomial``.
    truncated_likelihood : bool, optional
        Whether to use truncated likelihood. The default value is ``True``.
    optimizer : str, optional
        Optimization method to use. The default value is ``Nelder-Mead``.

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
    diameter_breaks = get_diameter_bin_edges(ds)

    # Define initial parameters (mu, sigma)
    ds_init = _get_initial_lognormal_parameters(ds, mom_method=init_method)

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
        ds_init["mu"],
        ds_init["sigma"],
        kwargs=kwargs,
        input_core_dims=[[DIAMETER_DIMENSION], [], []],
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
    init_method=None,
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
        Method to compute probabilities. The default value is ``cdf``.
    likelihood : str, optional
        Likelihood function to use for fitting. The default value is ``multinomial``.
    truncated_likelihood : bool, optional
        Whether to use truncated likelihood. The default value is ``True``.
    optimizer : str, optional
        Optimization method to use. The default value is ``Nelder-Mead``.

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
    counts = ds["drop_number_concentration"] * ds["diameter_bin_width"]  # mm-1 m-3 --> m-3
    diameter_breaks = get_diameter_bin_edges(ds)

    # Define initial parameters (Lambda)
    ds_init = _get_initial_exponential_parameters(ds, mom_method=init_method)

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
        ds_init["Lambda"],
        kwargs=kwargs,
        input_core_dims=[[DIAMETER_DIMENSION], []],
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


####-----------------------------------------------------------------------------------------.
#### Grid Search (GS)
#### - Optimization utilities


def define_param_range(center, step, bounds, factor=2, refinement=20):
    """
    Create a refined parameter search range around a center value, constrained to bounds.

    Parameters
    ----------
    center : float
        Center of the range (e.g., current best estimate).
    step : float
        Coarse step size used in the first search.
    bounds : tuple of (float, float)
        Lower and upper bounds (can include -np.inf, np.inf).
    factor : float, optional
        How wide the refined range extends from the center (in multiples of step).
        Default = 2.
    refinement : int, optional
        Factor to refine the step size (smaller step = finer grid).
        Default = 20.

    Returns
    -------
    np.ndarray
        Array of values constrained to bounds.
    """
    lower = max(center - factor * step, bounds[0])
    upper = min(center + factor * step, bounds[1])
    new_step = step / refinement
    return np.arange(lower, upper, new_step)


#### - Optimization routines
def apply_exponential_gs(
    Nt,
    ND_obs,
    V,
    # Coords
    D,
    dD,
    # PSD parameters
    Lambda,
    # Optimization options
    objectives,
    # Output options
    return_loss=False,
):
    """Estimate ExponentialPSD model parameters using Grid Search.

    This function performs a grid search optimization to find the best parameters
    (N0, Lambda) for the ExponentialPSD model by minimizing a weighted
    cost function across one or more objectives.

    Parameters
    ----------
    Nt : float
        Total number concentration.
    ND_obs : ndarray
        Observed PSD data [#/mm/m3].
    V : ndarray
        Fall velocity [m/s].
    D : ndarray
        Diameter bins [mm].
    dD : ndarray
        Diameter bin widths [mm].
    Lambda : int, float or numpy.ndarray
        Lambda parameter values to search.
    objectives: list of dicts
        target : str, optional
            Target quantity to optimize. Valid options:
            - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
            - ``"H(x)"`` : Normalized drop number concentration [-]
            - ``"R"`` : Rain rate [mm h⁻¹]
            - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
            - ``"LWC"`` : Liquid water content [g m⁻³]
            - ``"M<p>"`` : Moment of order p
        transformation : str, optional
            Transformation applied to the target quantity before computing the loss.
            Valid options:
            - ``"identity"`` : No transformation
            - ``"log"`` : Logarithmic transformation (default)
            - ``"sqrt"`` : Square root transformation
        censoring : str
            Specifies whether the observed particle size distribution (PSD) is
            treated as censored at the edges of the diameter range due to
            instrumental sensitivity limits:
            - ``"none"`` : No censoring is applied. All diameter bins are used.
            - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
              range of diameter bins with non-zero observed concentrations is
              retained.
        loss : int, optional
            Loss function.  To be specified only if target is ``"N(D)"`` or ``"H(x)"``.
            Valid options are:
            - ``SSE``: Sum of Squared Errors
            - ``SAE``: Sum of Absolute Errors
            - ``MAE``: Mean Absolute Error
            - ``MSE``: Mean Squared Error
            - ``RMSE``: Root Mean Squared Error
            - ``relMAE``: Relative Mean Absolute Error
            - ``KL``: Kullback-Leibler Divergence
            - ``WD``: Wasserstein Distance
            - ``JS``: Jensen-Shannon Distance
            - ``KS``: Kolmogorov-Smirnov Statistic
        loss_weight: int, optional
            Weight of this objective when multiple objectives are used.
            Must be specified if len(objectives) > 1.
    return_loss : bool, optional
        If True, return both the loss surface and parameters.
        Default is False.

    Returns
    -------
    parameters : ndarray
        Best parameters as [N0, Lambda].
        An array of NaN values is returned if no valid solution is found.
    total_loss : ndarray, optional
        1D array of total loss values.
        Only returned if return_loss=True.

    Notes
    -----
    - When multiple objectives are provided, losses are normalized and weighted
    - The best parameters correspond to the minimum total weighted loss
    """
    # Convert lambda to array if needed
    if not isinstance(Lambda, np.ndarray):
        Lambda = np.atleast_1d(Lambda)

    # Perform grid search
    with suppress_warnings():
        # Compute N(D)
        N0_arr = Nt * Lambda
        ND_preds = ExponentialPSD.formula(D=D[None, :], N0=N0_arr[:, None], Lambda=Lambda[:, None])

        # Compute loss
        total_loss = compute_weighted_loss(
            ND_obs=ND_obs,
            ND_preds=ND_preds,
            D=D,
            dD=dD,
            V=V,
            objectives=objectives,
        )

    # Define best parameters
    if not np.all(np.isnan(total_loss)):
        best_index = np.nanargmin(total_loss)
        N0 = N0_arr[best_index].item()
        Lambda_best = Lambda[best_index].item()
        parameters = np.array([N0, Lambda_best])
    else:
        parameters = np.array([np.nan, np.nan])

    # If asked, return cost function
    if return_loss:
        return total_loss, parameters

    return parameters


def apply_gamma_gs(
    Nt,
    ND_obs,
    V,
    # Coords
    D,
    dD,
    # PSD parameters
    mu,
    Lambda,
    # Optimization options
    objectives,
    # Output options
    return_loss=False,
):
    """Estimate GammaPSD model parameters using Grid Search.

    This function performs a grid search optimization to find the best parameters
    (mu, Lambda) for the GammaPSD model by minimizing a weighted
    cost function across one or more objectives.

    Parameters
    ----------
    Nt : float
        Total number concentration.
    ND_obs : ndarray
        Observed PSD data [#/mm/m3].
    V : ndarray
        Fall velocity [m/s].
    D : ndarray
        Diameter bins [mm].
    dD : ndarray
        Diameter bin widths [mm].
    mu : int, float or numpy.ndarray
        mu parameter values to search.
    Lambda : int, float or numpy.ndarray
        Lambda parameter values to search.
    objectives: list of dicts
        target : str, optional
            Target quantity to optimize. Valid options:
            - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
            - ``"H(x)"`` : Normalized drop number concentration [-]
            - ``"R"`` : Rain rate [mm h⁻¹]
            - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
            - ``"LWC"`` : Liquid water content [g m⁻³]
            - ``"M<p>"`` : Moment of order p
        transformation : str, optional
            Transformation applied to the target quantity before computing the loss.
            Valid options:
            - ``"identity"`` : No transformation
            - ``"log"`` : Logarithmic transformation (default)
            - ``"sqrt"`` : Square root transformation
        censoring : str
            Specifies whether the observed particle size distribution (PSD) is
            treated as censored at the edges of the diameter range due to
            instrumental sensitivity limits:
            - ``"none"`` : No censoring is applied. All diameter bins are used.
            - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
              range of diameter bins with non-zero observed concentrations is
              retained.
        loss : int, optional
            Loss function.  To be specified only if target is ``"N(D)"`` or ``"H(x)"``.
            Valid options are:
            - ``SSE``: Sum of Squared Errors
            - ``SAE``: Sum of Absolute Errors
            - ``MAE``: Mean Absolute Error
            - ``MSE``: Mean Squared Error
            - ``RMSE``: Root Mean Squared Error
            - ``relMAE``: Relative Mean Absolute Error
            - ``KL``: Kullback-Leibler Divergence
            - ``WD``: Wasserstein Distance
            - ``JS``: Jensen-Shannon Distance
            - ``KS``: Kolmogorov-Smirnov Statistic
        loss_weight: int, optional
            Weight of this objective when multiple objectives are used.
            Must be specified if len(objectives) > 1.
    return_loss : bool, optional
        If True, return both the loss surface and parameters.
        Default is False.

    Returns
    -------
    parameters : ndarray
        Best parameters as [N0, Lambda, mu].
        An array of NaN values is returned if no valid solution is found.
    total_loss : ndarray, optional
        2D array of total loss values reshaped to (len(mu), len(Lambda)).
        Only returned if return_loss=True.

    Notes
    -----
    - When multiple objectives are provided, losses are normalized and weighted
    - The best parameters correspond to the minimum total weighted loss
    """
    # Define combinations of parameters for grid search
    mu_grid, Lambda_grid = np.meshgrid(
        mu,
        Lambda,
        indexing="xy",
    )
    mu_arr = mu_grid.ravel()
    Lambda_arr = Lambda_grid.ravel()

    # Perform grid search
    with suppress_warnings():
        # Compute N(D)
        N0 = np.exp(np.log(Nt) + (mu_arr[:, None] + 1) * np.log(Lambda_arr[:, None]) - gammaln(mu_arr[:, None] + 1))
        ND_preds = GammaPSD.formula(D=D[None, :], N0=N0, Lambda=Lambda_arr[:, None], mu=mu_arr[:, None])

        # Compute loss
        total_loss = compute_weighted_loss(
            ND_obs=ND_obs,
            ND_preds=ND_preds,
            D=D,
            dD=dD,
            V=V,
            objectives=objectives,
        )

    # Define best parameters
    if not np.all(np.isnan(total_loss)):
        best_index = np.nanargmin(total_loss)
        N0_best = N0[best_index].item()
        mu_best = mu_arr[best_index].item()
        Lambda_best = Lambda_arr[best_index].item()
        parameters = np.array([N0_best, Lambda_best, mu_best])
    else:
        parameters = np.array([np.nan, np.nan, np.nan])

    # If asked, return cost function
    if return_loss:
        total_loss = total_loss.reshape(mu_grid.shape)
        return total_loss, parameters

    return parameters


def apply_generalized_gamma_gs(
    Nt,
    ND_obs,
    V,
    # Coords
    D,
    dD,
    # PSD parameters
    mu,
    c,
    Lambda,
    # Optimization options
    objectives,
    # Output options
    return_loss=False,
):
    """Estimate GeneralizedGammaPSD model parameters using Grid Search.

    This function performs a grid search optimization to find the best parameters
    (mu, c, Lambda) for the GeneralizedGammaPSD model by minimizing a weighted
    cost function across one or more objectives.

    Parameters
    ----------
    Nt : float
        Total number concentration.
    ND_obs : ndarray
        Observed PSD data [#/mm/m3].
    V : ndarray
        Fall velocity [m/s].
    D : ndarray
        Diameter bins [mm].
    dD : ndarray
        Diameter bin widths [mm].
    mu : int, float or numpy.ndarray
        mu parameter values to search.
    c : int, float or numpy.ndarray
        c parameter values to search.
    Lambda : int, float or numpy.ndarray
        Lambda parameter values to search.
    objectives: list of dicts
        target : str, optional
            Target quantity to optimize. Valid options:
            - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
            - ``"H(x)"`` : Normalized drop number concentration [-]
            - ``"R"`` : Rain rate [mm h⁻¹]
            - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
            - ``"LWC"`` : Liquid water content [g m⁻³]
            - ``"M<p>"`` : Moment of order p
        transformation : str, optional
            Transformation applied to the target quantity before computing the loss.
            Valid options:
            - ``"identity"`` : No transformation
            - ``"log"`` : Logarithmic transformation (default)
            - ``"sqrt"`` : Square root transformation
        censoring : str
            Specifies whether the observed particle size distribution (PSD) is
            treated as censored at the edges of the diameter range due to
            instrumental sensitivity limits:
            - ``"none"`` : No censoring is applied. All diameter bins are used.
            - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
              range of diameter bins with non-zero observed concentrations is
              retained.
        loss : int, optional
            Loss function.  To be specified only if target is ``"N(D)"`` or ``"H(x)"``.
            Valid options are:
            - ``SSE``: Sum of Squared Errors
            - ``SAE``: Sum of Absolute Errors
            - ``MAE``: Mean Absolute Error
            - ``MSE``: Mean Squared Error
            - ``RMSE``: Root Mean Squared Error
            - ``relMAE``: Relative Mean Absolute Error
            - ``KL``: Kullback-Leibler Divergence
            - ``WD``: Wasserstein Distance
            - ``JS``: Jensen-Shannon Distance
            - ``KS``: Kolmogorov-Smirnov Statistic
        loss_weight: int, optional
            Weight of this objective when multiple objectives are used.
            Must be specified if len(objectives) > 1.
    return_loss : bool, optional
        If True, return both the loss surface and parameters.
        Default is False.

    Returns
    -------
    parameters : ndarray
        Best parameters as [Lambda, mu, c].
        An array of NaN values is returned if no valid solution is found.
    total_loss : ndarray, optional
        3D array of total loss values reshaped to (len(mu), len(Lambda), len(c)).
        Only returned if return_loss=True.

    Notes
    -----
    - When multiple objectives are provided, losses are normalized and weighted
    - The best parameters correspond to the minimum total weighted loss
    """
    # Define combinations of parameters for grid search
    mu_grid, Lambda_grid, c_grid = np.meshgrid(
        mu,
        Lambda,
        c,
        indexing="xy",
    )
    mu_arr = mu_grid.ravel()
    Lambda_arr = Lambda_grid.ravel()
    c_arr = c_grid.ravel()

    # Perform grid search
    with suppress_warnings():
        # Compute N(D)
        ND_preds = GeneralizedGammaPSD.formula(
            D=D[None, :],
            Nt=Nt,
            Lambda=Lambda_arr[:, None],
            mu=mu_arr[:, None],
            c=c_arr[:, None],
        )

        # Compute loss
        total_loss = compute_weighted_loss(
            ND_obs=ND_obs,
            ND_preds=ND_preds,
            D=D,
            dD=dD,
            V=V,
            objectives=objectives,
        )

    # Define best parameters
    if not np.all(np.isnan(total_loss)):
        best_index = np.nanargmin(total_loss)
        mu_best = mu_arr[best_index].item()
        c_best = c_arr[best_index].item()
        Lambda_best = Lambda_arr[best_index].item()
        parameters = np.array([Nt, Lambda_best, mu_best, c_best])
    else:
        parameters = np.array([np.nan, np.nan, np.nan, np.nan])

    # If asked, return cost function
    if return_loss:
        total_loss = total_loss.reshape(mu_grid.shape)
        return total_loss, parameters

    return parameters


def apply_lognormal_gs(
    Nt,
    ND_obs,
    V,
    # Coords
    D,
    dD,
    # PSD parameters
    mu,
    sigma,
    # Optimization options
    objectives,
    # Output options
    return_loss=False,
):
    """Estimate LognormalPSD model parameters using Grid Search.

    This function performs a grid search optimization to find the best parameters
    (mu, sigma) for the LognormalPSD model by minimizing a weighted
    cost function across one or more objectives.

    Parameters
    ----------
    Nt : float
        Total number concentration.
    ND_obs : ndarray
        Observed PSD data [#/mm/m3].
    V : ndarray
        Fall velocity [m/s].
    D : ndarray
        Diameter bins [mm].
    dD : ndarray
        Diameter bin widths [mm].
    mu : int, float or numpy.ndarray
        mu parameter values to search.
    sigma : int, float or numpy.ndarray
        sigma parameter values to search.
    objectives: list of dicts
        target : str, optional
            Target quantity to optimize. Valid options:
            - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
            - ``"H(x)"`` : Normalized drop number concentration [-]
            - ``"R"`` : Rain rate [mm h⁻¹]
            - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
            - ``"LWC"`` : Liquid water content [g m⁻³]
            - ``"M<p>"`` : Moment of order p
        transformation : str, optional
            Transformation applied to the target quantity before computing the loss.
            Valid options:
            - ``"identity"`` : No transformation
            - ``"log"`` : Logarithmic transformation (default)
            - ``"sqrt"`` : Square root transformation
        censoring : str
            Specifies whether the observed particle size distribution (PSD) is
            treated as censored at the edges of the diameter range due to
            instrumental sensitivity limits:
            - ``"none"`` : No censoring is applied. All diameter bins are used.
            - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
              range of diameter bins with non-zero observed concentrations is
              retained.
        loss : int, optional
            Loss function.  To be specified only if target is ``"N(D)"`` or ``"H(x)"``.
            Valid options are:
            - ``SSE``: Sum of Squared Errors
            - ``SAE``: Sum of Absolute Errors
            - ``MAE``: Mean Absolute Error
            - ``MSE``: Mean Squared Error
            - ``RMSE``: Root Mean Squared Error
            - ``relMAE``: Relative Mean Absolute Error
            - ``KL``: Kullback-Leibler Divergence
            - ``WD``: Wasserstein Distance
            - ``JS``: Jensen-Shannon Distance
            - ``KS``: Kolmogorov-Smirnov Statistic
        loss_weight: int, optional
            Weight of this objective when multiple objectives are used.
            Must be specified if len(objectives) > 1.
    return_loss : bool, optional
        If True, return both the loss surface and parameters.
        Default is False.

    Returns
    -------
    parameters : ndarray
        Best parameters as [mu, sigma].
        An array of NaN values is returned if no valid solution is found.
    total_loss : ndarray, optional
        2D array of total loss values reshaped to (len(mu), len(sigma)).
        Only returned if return_loss=True.

    Notes
    -----
    - When multiple objectives are provided, losses are normalized and weighted
    - The best parameters correspond to the minimum total weighted loss
    """
    # Define combinations of parameters for grid search
    mu_grid, sigma_grid = np.meshgrid(
        mu,
        sigma,
        indexing="xy",
    )
    mu_arr = mu_grid.ravel()
    sigma_arr = sigma_grid.ravel()

    # Perform grid search
    with suppress_warnings():
        # Compute N(D)
        ND_preds = LognormalPSD.formula(D=D[None, :], Nt=Nt, mu=mu_arr[:, None], sigma=sigma_arr[:, None])

        # Compute loss
        total_loss = compute_weighted_loss(
            ND_obs=ND_obs,
            ND_preds=ND_preds,
            D=D,
            dD=dD,
            V=V,
            objectives=objectives,
        )

    # Define best parameters
    if not np.all(np.isnan(total_loss)):
        best_index = np.nanargmin(total_loss)
        mu_best = mu_arr[best_index].item()
        sigma_best = sigma_arr[best_index].item()
        parameters = np.array([Nt, mu_best, sigma_best])
    else:
        parameters = np.array([np.nan, np.nan, np.nan])

    # If asked, return cost function
    if return_loss:
        total_loss = total_loss.reshape(mu_grid.shape)
        return total_loss, parameters

    return parameters


def apply_normalized_gamma_gs(
    Nw,
    D50,
    ND_obs,
    V,
    # Coords
    D,
    dD,
    # PSD parameters
    mu,
    # Optimization options
    objectives,
    # Output options
    return_loss=False,
):
    """Estimate NormalizedGammaPSD model parameters using Grid Search.

    This function performs a grid search optimization to find the best parameter
    (mu) for the NormalizedGammaPSD model by minimizing a weighted
    cost function across one or more objectives.

    Parameters
    ----------
    Nw : float
        Normalized intercept parameter.
    D50 : float
        Median volume diameter parameter.
    ND_obs : ndarray
        Observed PSD data [#/mm/m3].
    V : ndarray
        Fall velocity [m/s].
    D : ndarray
        Diameter bins [mm].
    dD : ndarray
        Diameter bin widths [mm].
    mu : int, float or numpy.ndarray
        mu parameter values to search.
    objectives: list of dicts
        target : str, optional
            Target quantity to optimize. Valid options:
            - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
            - ``"H(x)"`` : Normalized drop number concentration [-]
            - ``"R"`` : Rain rate [mm h⁻¹]
            - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
            - ``"LWC"`` : Liquid water content [g m⁻³]
            - ``"M<p>"`` : Moment of order p
        transformation : str, optional
            Transformation applied to the target quantity before computing the loss.
            Valid options:
            - ``"identity"`` : No transformation
            - ``"log"`` : Logarithmic transformation (default)
            - ``"sqrt"`` : Square root transformation
        censoring : str
            Specifies whether the observed particle size distribution (PSD) is
            treated as censored at the edges of the diameter range due to
            instrumental sensitivity limits:
            - ``"none"`` : No censoring is applied. All diameter bins are used.
            - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
              range of diameter bins with non-zero observed concentrations is
              retained.
        loss : int, optional
            Loss function.  To be specified only if target is ``"N(D)"`` or ``"H(x)"``.
            Valid options are:
            - ``SSE``: Sum of Squared Errors
            - ``SAE``: Sum of Absolute Errors
            - ``MAE``: Mean Absolute Error
            - ``MSE``: Mean Squared Error
            - ``RMSE``: Root Mean Squared Error
            - ``relMAE``: Relative Mean Absolute Error
            - ``KL``: Kullback-Leibler Divergence
            - ``WD``: Wasserstein Distance
            - ``JS``: Jensen-Shannon Distance
            - ``KS``: Kolmogorov-Smirnov Statistic
        loss_weight: int, optional
            Weight of this objective when multiple objectives are used.
            Must be specified if len(objectives) > 1.
    return_loss : bool, optional
        If True, return both the loss surface and parameters.
        Default is False.

    Returns
    -------
    parameters : ndarray
        Best parameters as [Nw, mu, D50].
        An array of NaN values is returned if no valid solution is found.
    total_loss : ndarray, optional
        1D array of total loss values.
        Only returned if return_loss=True.

    Notes
    -----
    - When multiple objectives are provided, losses are normalized and weighted
    - The best parameters correspond to the minimum total weighted loss
    """
    # Convert mu to array if needed
    mu_arr = np.atleast_1d(mu) if not isinstance(mu, np.ndarray) else mu

    # Perform grid search
    with suppress_warnings():
        # Compute N(D)
        ND_preds = NormalizedGammaPSD.formula(D=D[None, :], D50=D50, Nw=Nw, mu=mu_arr[:, None])

        # Compute loss
        total_loss = compute_weighted_loss(
            ND_obs=ND_obs,
            ND_preds=ND_preds,
            D=D,
            dD=dD,
            V=V,
            objectives=objectives,
            Nc=Nw,
        )

    # Define best parameters
    if not np.all(np.isnan(total_loss)):
        best_index = np.nanargmin(total_loss)
        mu_best = mu_arr[best_index].item()
        parameters = np.array([Nw, D50, mu_best])
    else:
        parameters = np.array([np.nan, np.nan, np.nan])

    # If asked, return cost function
    if return_loss:
        return total_loss, parameters

    return parameters


def apply_normalized_generalized_gamma_gs(
    Nc,
    Dc,
    ND_obs,
    V,
    # Coords
    D,
    dD,
    # PSD parameters
    i,
    j,
    mu,
    c,
    # Optimization options
    objectives,
    # Output options
    return_loss=False,
):
    """Estimate NormalizedGeneralizedGammaPSD model parameters using Grid Search.

    This function performs a grid search optimization to find the best parameters
    (mu, c) for the NormalizedGeneralizedGammaPSD model by minimizing a weighted
    cost function across one or more objectives.

    Parameters
    ----------
    Nc : float
        Normalized intercept parameter.
    Dc : float
        Normalized characteristic diameter parameter.
    ND_obs : ndarray
        Observed PSD data [#/mm/m3].
    V : ndarray
        Fall velocity [m/s].
    D : ndarray
        Diameter bins [mm].
    dD : ndarray
        Diameter bin widths [mm].
    i : int
        Moment order i of the NormalizedGeneralizedGammaPSD.
    j : int
        Moment order j of the NormalizedGeneralizedGammaPSD.
    mu : int, float or numpy.ndarray
        mu parameter values to search.
    c : int, float or numpy.ndarray
        c parameter values to search.
    objectives: list of dicts
        target : str, optional
            Target quantity to optimize. Valid options:
            - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
            - ``"H(x)"`` : Normalized drop number concentration [-]
            - ``"R"`` : Rain rate [mm h⁻¹]
            - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
            - ``"LWC"`` : Liquid water content [g m⁻³]
            - ``"M<p>"`` : Moment of order p
        transformation : str, optional
            Transformation applied to the target quantity before computing the loss.
            Valid options:
            - ``"identity"`` : No transformation
            - ``"log"`` : Logarithmic transformation (default)
            - ``"sqrt"`` : Square root transformation
        censoring : str
            Specifies whether the observed particle size distribution (PSD) is
            treated as censored at the edges of the diameter range due to
            instrumental sensitivity limits:
            - ``"none"`` : No censoring is applied. All diameter bins are used.
            - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
              range of diameter bins with non-zero observed concentrations is
              retained.
        loss : int, optional
            Loss function.  To be specified only if target is ``"N(D)"`` or ``"N(D)"``.
            Valid options are:
            - ``SSE``: Sum of Squared Errors
            - ``SAE``: Sum of Absolute Errors
            - ``MAE``: Mean Absolute Error
            - ``MSE``: Mean Squared Error
            - ``RMSE``: Root Mean Squared Error
            - ``relMAE``: Relative Mean Absolute Error
            - ``KL``: Kullback-Leibler Divergence
            - ``WD``: Wasserstein Distance
            - ``JS``: Jensen-Shannon Distance
            - ``KS``: Kolmogorov-Smirnov Statistic
        loss_weight: int, optional
            Weight of this objective when multiple objectives are used.
            Must be specified if len(objectives) > 1.
    return_loss : bool, optional
        If True, return both the loss surface and parameters.
        Default is False.

    Returns
    -------
    parameters : ndarray
        Best parameters as [Nc, Dc, mu, c].
        An array of NaN values is returned if no valid solution is found.
    total_loss : ndarray, optional
        2D array of total loss values reshaped to (len(mu), len(c)).
        Only returned if return_loss=True.

    Notes
    -----
    - When multiple objectives are provided, losses are normalized and weighted
    - The best parameters correspond to the minimum total weighted loss
    """
    # Thurai 2018: mu [-3, 1], c [0-6]

    # Define combinations of parameters for grid search
    mu_grid, c_grid = np.meshgrid(
        mu,
        c,
        indexing="xy",
    )
    mu_arr = mu_grid.ravel()
    c_arr = c_grid.ravel()

    # Perform grid search
    with suppress_warnings():

        # Compute N(D)
        ND_preds = NormalizedGeneralizedGammaPSD.formula(
            D=D[None, :],
            i=i,
            j=j,
            Nc=Nc,
            Dc=Dc,
            mu=mu_arr[:, None],
            c=c_arr[:, None],
        )

        # Compute loss
        total_loss = compute_weighted_loss(
            ND_obs=ND_obs,
            ND_preds=ND_preds,
            D=D,
            dD=dD,
            V=V,
            objectives=objectives,
            Nc=Nc,
        )

    # Define best parameters
    if not np.all(np.isnan(total_loss)):
        best_index = np.nanargmin(total_loss)
        mu, c = mu_arr[best_index].item(), c_arr[best_index].item()
        parameters = np.array([Nc, Dc, mu, c])
    else:
        parameters = np.array([np.nan, np.nan, np.nan, np.nan])

    # If asked, return cost function
    if return_loss:
        total_loss = total_loss.reshape(mu_grid.shape)
        return total_loss, parameters
    return parameters


def get_exponential_parameters_gs(
    ds,
    Lambda=None,
    objectives=None,
    return_loss=False,
):
    """Estimate Exponential PSD parameters using Grid Search optimization.

    The parameter ``N_t`` is computed empirically from the observed DSD,
    while the shape parameter ``Lambda`` is estimated through
    grid search by minimizing the error between observed and modeled quantities.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing PSD observations. Must include:
        - ``drop_number_concentration`` : Drop number concentration [m⁻³ mm⁻¹]
        - ``diameter_bin_center`` : Diameter bin centers [mm]
        - ``diameter_bin_width`` : Diameter bin widths [mm]
        - ``fall_velocity`` : Drop fall velocity [m s⁻¹] (required if target='R')
    Lambda : int, float or numpy.ndarray
        Lambda parameter values to search.
    objectives: list of dicts
        target : str, optional
            Target quantity to optimize. Valid options:
            - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
            - ``"H(x)"`` : Normalized drop number concentration [-]
            - ``"R"`` : Rain rate [mm h⁻¹]
            - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
            - ``"LWC"`` : Liquid water content [g m⁻³]
            - ``"M<p>"`` : Moment of order p
        transformation : str, optional
            Transformation applied to the target quantity before computing the loss.
            Valid options:
            - ``"identity"`` : No transformation
            - ``"log"`` : Logarithmic transformation (default)
            - ``"sqrt"`` : Square root transformation
        censoring : str
            Specifies whether the observed particle size distribution (PSD) is
            treated as censored at the edges of the diameter range due to
            instrumental sensitivity limits:
            - ``"none"`` : No censoring is applied. All diameter bins are used.
            - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
              range of diameter bins with non-zero observed concentrations is
              retained.
        loss : int, optional
            Loss function.  To be specified only if target is ``"N(D)"`` or ``"H(x)"``.
            Valid options are:
            - ``SSE``: Sum of Squared Errors
            - ``SAE``: Sum of Absolute Errors
            - ``MAE``: Mean Absolute Error
            - ``MSE``: Mean Squared Error
            - ``RMSE``: Root Mean Squared Error
            - ``relMAE``: Relative Mean Absolute Error
            - ``KL``: Kullback-Leibler Divergence
            - ``WD``: Wasserstein Distance
            - ``JS``: Jensen-Shannon Distance
            - ``KS``: Kolmogorov-Smirnov Statistic
        loss_weight: int, optional
            Weight of this objective when multiple objectives are used.
            Must be specified if len(objectives) > 1.
    return_loss : bool, optional
        If True, return both the loss surface and parameters.
        Default is False.

    Returns
    -------
    ds_params : xarray.Dataset
        Dataset containing the estimated Exponential distribution parameters.
    """
    # Compute required variables
    Nt = get_total_number_concentration(
        drop_number_concentration=ds["drop_number_concentration"],
        diameter_bin_width=ds["diameter_bin_width"],
    )

    # Define search space
    if Lambda is None:
        Lambda = np.arange(0.01, 10, step=0.01)

    # Define kwargs
    kwargs = {
        "D": ds["diameter_bin_center"].data,
        "dD": ds["diameter_bin_width"].data,
        "objectives": objectives,
        "return_loss": return_loss,
        "Lambda": Lambda,
    }

    # Define function to create parameters dataset
    def _create_parameters_dataset(da_parameters):
        # Add parameters coordinates
        da_parameters = da_parameters.assign_coords({"parameters": ["N0", "Lambda"]})

        # Create parameters dataset
        ds_parameters = da_parameters.to_dataset(dim="parameters")

        # Add DSD model name to the attribute
        ds_parameters.attrs["disdrodb_psd_model"] = "ExponentialPSD"
        return ds_parameters

    # Return cost function if asked
    if return_loss:
        da_cost_function, da_parameters = xr.apply_ufunc(
            apply_exponential_gs,
            # Variables varying over time
            Nt,
            ds["drop_number_concentration"],
            ds["fall_velocity"],
            # Other options
            kwargs=kwargs,
            # Settings
            input_core_dims=[[], [DIAMETER_DIMENSION], [DIAMETER_DIMENSION]],
            output_core_dims=[["Lambda_values"], ["parameters"]],
            vectorize=True,
            dask="parallelized",
            # Lengths of the new output_core_dims dimensions.
            dask_gufunc_kwargs={"output_sizes": {"Lambda_values": len(Lambda), "parameters": 2}},
            output_dtypes=["float64", "float64"],
        )
        ds_parameters = _create_parameters_dataset(da_parameters)
        ds_parameters["cost_function"] = da_cost_function
        ds_parameters = ds_parameters.assign_coords({"Lambda_values": Lambda})
        return ds_parameters

    # Otherwise return just best parameters
    da_parameters = xr.apply_ufunc(
        apply_exponential_gs,
        # Variables varying over time
        Nt,
        ds["drop_number_concentration"],
        ds["fall_velocity"],
        # Other options
        kwargs=kwargs,
        # Settings
        input_core_dims=[[], [DIAMETER_DIMENSION], [DIAMETER_DIMENSION]],
        output_core_dims=[["parameters"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"parameters": 2}},  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )
    ds_parameters = _create_parameters_dataset(da_parameters)
    return ds_parameters


def get_gamma_parameters_gs(
    ds,
    mu=None,
    Lambda=None,
    objectives=None,
    return_loss=False,
):
    """Estimate Gamma PSD parameters using Grid Search optimization.

    The parameter ``N_t`` is computed empirically from the observed DSD,
    while the shape parameters ``mu`` and ``Lambda`` are estimated through
    grid search by minimizing the error between observed and modeled quantities.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing PSD observations. Must include:
        - ``drop_number_concentration`` : Drop number concentration [m⁻³ mm⁻¹]
        - ``diameter_bin_center`` : Diameter bin centers [mm]
        - ``diameter_bin_width`` : Diameter bin widths [mm]
        - ``fall_velocity`` : Drop fall velocity [m s⁻¹] (required if target='R')
    mu : int, float or numpy.ndarray
        mu parameter values to search.
    Lambda : int, float or numpy.ndarray
        Lambda parameter values to search.
    objectives: list of dicts
        target : str, optional
            Target quantity to optimize. Valid options:
            - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
            - ``"H(x)"`` : Normalized drop number concentration [-]
            - ``"R"`` : Rain rate [mm h⁻¹]
            - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
            - ``"LWC"`` : Liquid water content [g m⁻³]
            - ``"M<p>"`` : Moment of order p
        transformation : str, optional
            Transformation applied to the target quantity before computing the loss.
            Valid options:
            - ``"identity"`` : No transformation
            - ``"log"`` : Logarithmic transformation (default)
            - ``"sqrt"`` : Square root transformation
        censoring : str
            Specifies whether the observed particle size distribution (PSD) is
            treated as censored at the edges of the diameter range due to
            instrumental sensitivity limits:
            - ``"none"`` : No censoring is applied. All diameter bins are used.
            - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
              range of diameter bins with non-zero observed concentrations is
              retained.
        loss : int, optional
            Loss function.  To be specified only if target is ``"N(D)"`` or ``"H(x)"``.
            Valid options are:
            - ``SSE``: Sum of Squared Errors
            - ``SAE``: Sum of Absolute Errors
            - ``MAE``: Mean Absolute Error
            - ``MSE``: Mean Squared Error
            - ``RMSE``: Root Mean Squared Error
            - ``relMAE``: Relative Mean Absolute Error
            - ``KL``: Kullback-Leibler Divergence
            - ``WD``: Wasserstein Distance
            - ``JS``: Jensen-Shannon Distance
            - ``KS``: Kolmogorov-Smirnov Statistic
        loss_weight: int, optional
            Weight of this objective when multiple objectives are used.
            Must be specified if len(objectives) > 1.
    return_loss : bool, optional
        If True, return both the loss surface and parameters.
        Default is False.

    Returns
    -------
    ds_params : xarray.Dataset
        Dataset containing the estimated Gamma distribution parameters.
    """
    # Compute required variables
    Nt = get_total_number_concentration(
        drop_number_concentration=ds["drop_number_concentration"],
        diameter_bin_width=ds["diameter_bin_width"],
    )

    # Define search space
    if mu is None:
        mu = np.arange(0, 40, step=0.1)
    if Lambda is None:
        Lambda = np.arange(0, 60, step=0.1)

    # Define kwargs
    kwargs = {
        "D": ds["diameter_bin_center"].data,
        "dD": ds["diameter_bin_width"].data,
        "objectives": objectives,
        "return_loss": return_loss,
        "mu": mu,
        "Lambda": Lambda,
    }

    # Define function to create parameters dataset
    def _create_parameters_dataset(da_parameters):
        # Add parameters coordinates
        da_parameters = da_parameters.assign_coords({"parameters": ["N0", "Lambda", "mu"]})

        # Create parameters dataset
        ds_parameters = da_parameters.to_dataset(dim="parameters")

        # Add DSD model name to the attribute
        ds_parameters.attrs["disdrodb_psd_model"] = "GammaPSD"
        return ds_parameters

    # Return cost function if asked
    if return_loss:
        # Define lengths of the new output_core_dims dimensions.
        output_dict_size = {
            "mu_values": len(mu),
            "Lambda_values": len(Lambda),
            "parameters": 3,
        }
        # Compute cost function and parameters
        da_cost_function, da_parameters = xr.apply_ufunc(
            apply_gamma_gs,
            # Variables varying over time
            Nt,
            ds["drop_number_concentration"],
            ds["fall_velocity"],
            # Other options
            kwargs=kwargs,
            # Settings
            input_core_dims=[[], [DIAMETER_DIMENSION], [DIAMETER_DIMENSION]],
            output_core_dims=[["Lambda_values", "mu_values"], ["parameters"]],
            vectorize=True,
            dask="parallelized",
            # Lengths of the new output_core_dims dimensions.
            dask_gufunc_kwargs={"output_sizes": output_dict_size},
            output_dtypes=["float64", "float64"],
        )
        ds_parameters = _create_parameters_dataset(da_parameters)
        ds_parameters["cost_function"] = da_cost_function
        ds_parameters = ds_parameters.assign_coords({"mu_values": mu, "Lambda_values": Lambda})
        return ds_parameters

    # Otherwise return just best parameters
    da_parameters = xr.apply_ufunc(
        apply_gamma_gs,
        # Variables varying over time
        Nt,
        ds["drop_number_concentration"],
        ds["fall_velocity"],
        # Other options
        kwargs=kwargs,
        # Settings
        input_core_dims=[[], [DIAMETER_DIMENSION], [DIAMETER_DIMENSION]],
        output_core_dims=[["parameters"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"parameters": 3}},  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )
    ds_parameters = _create_parameters_dataset(da_parameters)
    return ds_parameters


def get_generalized_gamma_parameters_gs(
    ds,
    mu=None,
    c=None,
    Lambda=None,
    objectives=None,
    return_loss=False,
):
    """Estimate Generalized Gamma PSD parameters using Grid Search optimization.

    The parameter ``N_t`` is computed empirically from the observed DSD,
    while the shape parameters ``mu``, ``c``, and ``Lambda`` are estimated through
    grid search by minimizing the error between observed and modeled quantities.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing PSD observations. Must include:
        - ``drop_number_concentration`` : Drop number concentration [m⁻³ mm⁻¹]
        - ``diameter_bin_center`` : Diameter bin centers [mm]
        - ``diameter_bin_width`` : Diameter bin widths [mm]
        - ``fall_velocity`` : Drop fall velocity [m s⁻¹] (required if target='R')
    mu : int, float or numpy.ndarray
        mu parameter values to search.
    c : int, float or numpy.ndarray
        c parameter values to search.
    Lambda : int, float or numpy.ndarray
        Lambda parameter values to search.
    objectives: list of dicts
        target : str, optional
            Target quantity to optimize. Valid options:
            - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
            - ``"H(x)"`` : Normalized drop number concentration [-]
            - ``"R"`` : Rain rate [mm h⁻¹]
            - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
            - ``"LWC"`` : Liquid water content [g m⁻³]
            - ``"M<p>"`` : Moment of order p
        transformation : str, optional
            Transformation applied to the target quantity before computing the loss.
            Valid options:
            - ``"identity"`` : No transformation
            - ``"log"`` : Logarithmic transformation (default)
            - ``"sqrt"`` : Square root transformation
        censoring : str
            Specifies whether the observed particle size distribution (PSD) is
            treated as censored at the edges of the diameter range due to
            instrumental sensitivity limits:
            - ``"none"`` : No censoring is applied. All diameter bins are used.
            - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
              range of diameter bins with non-zero observed concentrations is
              retained.
        loss : int, optional
            Loss function.  To be specified only if target is ``"N(D)"`` or ``"H(x)"``.
            Valid options are:
            - ``SSE``: Sum of Squared Errors
            - ``SAE``: Sum of Absolute Errors
            - ``MAE``: Mean Absolute Error
            - ``MSE``: Mean Squared Error
            - ``RMSE``: Root Mean Squared Error
            - ``relMAE``: Relative Mean Absolute Error
            - ``KL``: Kullback-Leibler Divergence
            - ``WD``: Wasserstein Distance
            - ``JS``: Jensen-Shannon Distance
            - ``KS``: Kolmogorov-Smirnov Statistic
        loss_weight: int, optional
            Weight of this objective when multiple objectives are used.
            Must be specified if len(objectives) > 1.
    return_loss : bool, optional
        If True, return both the loss surface and parameters.
        Default is False.

    Returns
    -------
    ds_params : xarray.Dataset
        Dataset containing the estimated Generalized Gamma distribution parameters.
    """
    # Compute required variables
    Nt = get_total_number_concentration(
        drop_number_concentration=ds["drop_number_concentration"],
        diameter_bin_width=ds["diameter_bin_width"],
    )

    # Define search space
    if mu is None:
        mu = np.arange(0, 30, step=0.1)
    if c is None:
        c = np.arange(0, 10, step=0.2)
    if Lambda is None:
        Lambda = np.arange(0, 40, step=0.1)

    # Define kwargs
    kwargs = {
        "D": ds["diameter_bin_center"].data,
        "dD": ds["diameter_bin_width"].data,
        "objectives": objectives,
        "return_loss": return_loss,
        "mu": mu,
        "c": c,
        "Lambda": Lambda,
    }

    # Define function to create parameters dataset
    def _create_parameters_dataset(da_parameters):
        # Add parameters coordinates
        da_parameters = da_parameters.assign_coords({"parameters": ["Nt", "Lambda", "mu", "c"]})

        # Create parameters dataset
        ds_parameters = da_parameters.to_dataset(dim="parameters")

        # Add DSD model name to the attribute
        ds_parameters.attrs["disdrodb_psd_model"] = "GeneralizedGammaPSD"
        return ds_parameters

    # Return cost function if asked
    if return_loss:
        # Define lengths of the new output_core_dims dimensions.
        output_dict_size = {
            "mu_values": len(mu),
            "Lambda_values": len(Lambda),
            "c_values": len(c),
            "parameters": 4,
        }
        # Compute
        da_cost_function, da_parameters = xr.apply_ufunc(
            apply_generalized_gamma_gs,
            # Variables varying over time
            Nt,
            ds["drop_number_concentration"],
            ds["fall_velocity"],
            # Other options
            kwargs=kwargs,
            # Settings
            input_core_dims=[[], [DIAMETER_DIMENSION], [DIAMETER_DIMENSION]],
            output_core_dims=[["Lambda_values", "mu_values", "c_values"], ["parameters"]],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={"output_sizes": output_dict_size},
            output_dtypes=["float64", "float64"],
        )
        ds_parameters = _create_parameters_dataset(da_parameters)
        ds_parameters["cost_function"] = da_cost_function
        ds_parameters = ds_parameters.assign_coords({"mu_values": mu, "Lambda_values": Lambda, "c_values": c})
        return ds_parameters

    # Otherwise return just best parameters
    da_parameters = xr.apply_ufunc(
        apply_generalized_gamma_gs,
        # Variables varying over time
        Nt,
        ds["drop_number_concentration"],
        ds["fall_velocity"],
        # Other options
        kwargs=kwargs,
        # Settings
        input_core_dims=[[], [DIAMETER_DIMENSION], [DIAMETER_DIMENSION]],
        output_core_dims=[["parameters"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"parameters": 4}},  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )
    ds_parameters = _create_parameters_dataset(da_parameters)
    return ds_parameters


def get_lognormal_parameters_gs(
    ds,
    mu=None,
    sigma=None,
    objectives=None,
    return_loss=False,
):
    """Estimate Lognormal PSD parameters using Grid Search optimization.

    The parameter ``N_t`` is computed empirically from the observed DSD,
    while the shape parameters ``mu`` and ``sigma`` are estimated through
    grid search by minimizing the error between observed and modeled quantities.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing PSD observations. Must include:
        - ``drop_number_concentration`` : Drop number concentration [m⁻³ mm⁻¹]
        - ``diameter_bin_center`` : Diameter bin centers [mm]
        - ``diameter_bin_width`` : Diameter bin widths [mm]
        - ``fall_velocity`` : Drop fall velocity [m s⁻¹] (required if target='R')
    mu : int, float or numpy.ndarray
        mu parameter values to search.
    sigma : int, float or numpy.ndarray
        sigma parameter values to search.
    objectives: list of dicts
        target : str, optional
            Target quantity to optimize. Valid options:
            - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
            - ``"H(x)"`` : Normalized drop number concentration [-]
            - ``"R"`` : Rain rate [mm h⁻¹]
            - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
            - ``"LWC"`` : Liquid water content [g m⁻³]
            - ``"M<p>"`` : Moment of order p
        transformation : str, optional
            Transformation applied to the target quantity before computing the loss.
            Valid options:
            - ``"identity"`` : No transformation
            - ``"log"`` : Logarithmic transformation (default)
            - ``"sqrt"`` : Square root transformation
        censoring : str
            Specifies whether the observed particle size distribution (PSD) is
            treated as censored at the edges of the diameter range due to
            instrumental sensitivity limits:
            - ``"none"`` : No censoring is applied. All diameter bins are used.
            - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
              range of diameter bins with non-zero observed concentrations is
              retained.
        loss : int, optional
            Loss function.  To be specified only if target is ``"N(D)"`` or ``"H(x)"``.
            Valid options are:
            - ``SSE``: Sum of Squared Errors
            - ``SAE``: Sum of Absolute Errors
            - ``MAE``: Mean Absolute Error
            - ``MSE``: Mean Squared Error
            - ``RMSE``: Root Mean Squared Error
            - ``relMAE``: Relative Mean Absolute Error
            - ``KL``: Kullback-Leibler Divergence
            - ``WD``: Wasserstein Distance
            - ``JS``: Jensen-Shannon Distance
            - ``KS``: Kolmogorov-Smirnov Statistic
        loss_weight: int, optional
            Weight of this objective when multiple objectives are used.
            Must be specified if len(objectives) > 1.
    return_loss : bool, optional
        If True, return both the loss surface and parameters.
        Default is False.

    Returns
    -------
    ds_params : xarray.Dataset
        Dataset containing the estimated Lognormal distribution parameters.
    """
    # Compute required variables
    Nt = get_total_number_concentration(
        drop_number_concentration=ds["drop_number_concentration"],
        diameter_bin_width=ds["diameter_bin_width"],
    )

    # Define search space
    if mu is None:
        mu = np.arange(-4, 1, step=0.1)
    if sigma is None:
        sigma = np.arange(0, 3, step=0.2)

    # Define kwargs
    kwargs = {
        "D": ds["diameter_bin_center"].data,
        "dD": ds["diameter_bin_width"].data,
        "objectives": objectives,
        "return_loss": return_loss,
        "mu": mu,
        "sigma": sigma,
    }

    # Define function to create parameters dataset
    def _create_parameters_dataset(da_parameters):
        # Add parameters coordinates
        da_parameters = da_parameters.assign_coords({"parameters": ["Nt", "mu", "sigma"]})

        # Create parameters dataset
        ds_parameters = da_parameters.to_dataset(dim="parameters")

        # Add DSD model name to the attribute
        ds_parameters.attrs["disdrodb_psd_model"] = "LognormalPSD"
        return ds_parameters

    # Return cost function if asked
    if return_loss:
        da_cost_function, da_parameters = xr.apply_ufunc(
            apply_lognormal_gs,
            # Variables varying over time
            Nt,
            ds["drop_number_concentration"],
            ds["fall_velocity"],
            # Other options
            kwargs=kwargs,
            # Settings
            input_core_dims=[[], [DIAMETER_DIMENSION], [DIAMETER_DIMENSION]],
            output_core_dims=[["sigma_values", "mu_values"], ["parameters"]],
            vectorize=True,
            dask="parallelized",
            # Lengths of the new output_core_dims dimensions.
            dask_gufunc_kwargs={"output_sizes": {"mu_values": len(mu), "sigma_values": len(sigma), "parameters": 3}},
            output_dtypes=["float64", "float64"],
        )
        ds_parameters = _create_parameters_dataset(da_parameters)
        ds_parameters["cost_function"] = da_cost_function
        ds_parameters = ds_parameters.assign_coords({"mu_values": mu, "sigma_values": sigma})
        return ds_parameters

    # Otherwise return just best parameters
    da_parameters = xr.apply_ufunc(
        apply_lognormal_gs,
        # Variables varying over time
        Nt,
        ds["drop_number_concentration"],
        ds["fall_velocity"],
        # Other options
        kwargs=kwargs,
        # Settings
        input_core_dims=[[], [DIAMETER_DIMENSION], [DIAMETER_DIMENSION]],
        output_core_dims=[["parameters"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"parameters": 3}},  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )
    ds_parameters = _create_parameters_dataset(da_parameters)
    return ds_parameters


def get_normalized_gamma_parameters_gs(
    ds,
    mu=None,
    objectives=None,
    return_loss=False,
):
    """Estimate Normalized Gamma PSD parameters using Grid Search optimization.

    The parameters ``N_w`` and ``D50`` are computed empirically from the observed DSD
    moments, while the shape parameter ``mu`` is estimated through
    grid search by minimizing the error between observed and modeled quantities.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing PSD observations. Must include:
        - ``drop_number_concentration`` : Drop number concentration [m⁻³ mm⁻¹]
        - ``diameter_bin_center`` : Diameter bin centers [mm]
        - ``diameter_bin_width`` : Diameter bin widths [mm]
        - ``fall_velocity`` : Drop fall velocity [m s⁻¹] (required if target='R')
    mu : int, float or numpy.ndarray
        mu parameter values to search.
    objectives: list of dicts
        target : str, optional
            Target quantity to optimize. Valid options:
            - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
            - ``"H(x)"`` : Normalized drop number concentration [-]
            - ``"R"`` : Rain rate [mm h⁻¹]
            - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
            - ``"LWC"`` : Liquid water content [g m⁻³]
            - ``"M<p>"`` : Moment of order p
        transformation : str, optional
            Transformation applied to the target quantity before computing the loss.
            Valid options:
            - ``"identity"`` : No transformation
            - ``"log"`` : Logarithmic transformation (default)
            - ``"sqrt"`` : Square root transformation
        censoring : str
            Specifies whether the observed particle size distribution (PSD) is
            treated as censored at the edges of the diameter range due to
            instrumental sensitivity limits:
            - ``"none"`` : No censoring is applied. All diameter bins are used.
            - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
              range of diameter bins with non-zero observed concentrations is
              retained.
        loss : int, optional
            Loss function.  To be specified only if target is ``"N(D)"`` or ``"H(x)"``.
            Valid options are:
            - ``SSE``: Sum of Squared Errors
            - ``SAE``: Sum of Absolute Errors
            - ``MAE``: Mean Absolute Error
            - ``MSE``: Mean Squared Error
            - ``RMSE``: Root Mean Squared Error
            - ``relMAE``: Relative Mean Absolute Error
            - ``KL``: Kullback-Leibler Divergence
            - ``WD``: Wasserstein Distance
            - ``JS``: Jensen-Shannon Distance
            - ``KS``: Kolmogorov-Smirnov Statistic
        loss_weight: int, optional
            Weight of this objective when multiple objectives are used.
            Must be specified if len(objectives) > 1.
    return_loss : bool, optional
        If True, return both the loss surface and parameters.
        Default is False.

    Returns
    -------
    ds_params : xarray.Dataset
        Dataset containing the estimated Normalized Gamma distribution parameters.
    """
    # Compute required variables
    drop_number_concentration = ds["drop_number_concentration"]
    diameter_bin_width = ds["diameter_bin_width"]
    diameter = ds["diameter_bin_center"] / 1000  # conversion from mm to m
    m3 = get_moment(
        drop_number_concentration=drop_number_concentration,
        diameter=diameter,  # m
        diameter_bin_width=diameter_bin_width,  # mm
        moment=3,
    )
    m4 = get_moment(
        drop_number_concentration=drop_number_concentration,
        diameter=diameter,  # m
        diameter_bin_width=diameter_bin_width,  # mm
        moment=4,
    )
    Nw = get_normalized_intercept_parameter_from_moments(moment_3=m3, moment_4=m4)
    D50 = get_median_volume_drop_diameter(
        drop_number_concentration=drop_number_concentration,
        diameter=diameter,  # m
        diameter_bin_width=diameter_bin_width,  # mm
    )

    # Define search space
    if mu is None:
        mu = np.arange(-4, 30, step=0.01)

    # Define kwargs
    kwargs = {
        "D": ds["diameter_bin_center"].data,
        "dD": ds["diameter_bin_width"].data,
        "objectives": objectives,
        "return_loss": return_loss,
        "mu": mu,
    }

    # Define function to create parameters dataset
    def _create_parameters_dataset(da_parameters):
        # Add parameters coordinates
        da_parameters = da_parameters.assign_coords({"parameters": ["Nw", "D50", "mu"]})

        # Create parameters dataset
        ds_parameters = da_parameters.to_dataset(dim="parameters")

        # Add DSD model name to the attribute
        ds_parameters.attrs["disdrodb_psd_model"] = "NormalizedGammaPSD"
        return ds_parameters

    # Return cost function if asked
    if return_loss:
        da_cost_function, da_parameters = xr.apply_ufunc(
            apply_normalized_gamma_gs,
            # Variables varying over time
            Nw,
            D50,
            ds["drop_number_concentration"],
            ds["fall_velocity"],
            # Other options
            kwargs=kwargs,
            # Settings
            input_core_dims=[[], [], [DIAMETER_DIMENSION], [DIAMETER_DIMENSION]],
            output_core_dims=[["mu_values"], ["parameters"]],
            vectorize=True,
            dask="parallelized",
            # Lengths of the new output_core_dims dimensions.
            dask_gufunc_kwargs={"output_sizes": {"mu_values": len(mu), "parameters": 3}},
            output_dtypes=["float64", "float64"],
        )
        ds_parameters = _create_parameters_dataset(da_parameters)
        ds_parameters["cost_function"] = da_cost_function
        ds_parameters = ds_parameters.assign_coords({"mu_values": mu})
        return ds_parameters

    # Otherwise return just best parameters
    da_parameters = xr.apply_ufunc(
        apply_normalized_gamma_gs,
        # Variables varying over time
        Nw,
        D50,
        ds["drop_number_concentration"],
        ds["fall_velocity"],
        # Other options
        kwargs=kwargs,
        # Settings
        input_core_dims=[[], [], [DIAMETER_DIMENSION], [DIAMETER_DIMENSION]],
        output_core_dims=[["parameters"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"parameters": 3}},  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )
    ds_parameters = _create_parameters_dataset(da_parameters)
    return ds_parameters


def get_normalized_generalized_gamma_parameters_gs(
    ds,
    i,
    j,
    mu=None,
    c=None,
    objectives=None,
    return_loss=False,
):
    """Estimate Normalized Generalized Gamma PSD parameters using Grid Search optimization.

    The parameters ``N_c`` and ``Dc`` are computed empirically from the observed DSD
    moments, while the shape parameters ``mu`` and ``c`` are estimated through
    grid search by minimizing the error between observed and modeled quantities.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing PSD observations. Must include:
        - ``drop_number_concentration`` : Drop number concentration [m⁻³ mm⁻¹]
        - ``diameter_bin_center`` : Diameter bin centers [mm]
        - ``diameter_bin_width`` : Diameter bin widths [mm]
        - ``fall_velocity`` : Drop fall velocity [m s⁻¹] (required if target='R')
    i : int
        Moment order i of the NormalizedGeneralizedGammaPSD.
    j : int
        Moment order j of the NormalizedGeneralizedGammaPSD.
    mu : int, float or numpy.ndarray
        mu parameter values to search.
    c : int, float or numpy.ndarray
        c parameter values to search.
    objectives: list of dicts
        target : str, optional
            Target quantity to optimize. Valid options:
            - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
            - ``"H(x)"`` : Normalized drop number concentration [-]
            - ``"R"`` : Rain rate [mm h⁻¹]
            - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
            - ``"LWC"`` : Liquid water content [g m⁻³]
            - ``"M<p>"`` : Moment of order p
        transformation : str, optional
            Transformation applied to the target quantity before computing the loss.
            Valid options:
            - ``"identity"`` : No transformation
            - ``"log"`` : Logarithmic transformation (default)
            - ``"sqrt"`` : Square root transformation
        censoring : str
            Specifies whether the observed particle size distribution (PSD) is
            treated as censored at the edges of the diameter range due to
            instrumental sensitivity limits:
            - ``"none"`` : No censoring is applied. All diameter bins are used.
            - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
              range of diameter bins with non-zero observed concentrations is
              retained.
        loss : int, optional
            Loss function.  To be specified only if target is ``"N(D)"`` or ``"N(D)"``.
            Valid options are:
            - ``SSE``: Sum of Squared Errors
            - ``SAE``: Sum of Absolute Errors
            - ``MAE``: Mean Absolute Error
            - ``MSE``: Mean Squared Error
            - ``RMSE``: Root Mean Squared Error
            - ``relMAE``: Relative Mean Absolute Error
            - ``KL``: Kullback-Leibler Divergence
            - ``WD``: Wasserstein Distance
            - ``JS``: Jensen-Shannon Distance
            - ``KS``: Kolmogorov-Smirnov Statistic
        loss_weight: int, optional
            Weight of this objective when multiple objectives are used.
            Must be specified if len(objectives) > 1.
    return_loss : bool, optional
        If True, return both the loss surface and parameters.
        Default is False.

    Returns
    -------
    ds_params : xarray.Dataset
        Dataset containing the estimated Normalized Generalized Gamma distribution parameters.
    """
    # Compute required variables
    drop_number_concentration = ds["drop_number_concentration"]
    diameter_bin_width = ds["diameter_bin_width"]
    diameter = ds["diameter_bin_center"] / 1000  # conversion from mm to m
    Mi = get_moment(
        drop_number_concentration=drop_number_concentration,
        diameter=diameter,  # m
        diameter_bin_width=diameter_bin_width,  # mm
        moment=i,
    )
    Mj = get_moment(
        drop_number_concentration=drop_number_concentration,
        diameter=diameter,  # m
        diameter_bin_width=diameter_bin_width,  # mm
        moment=j,
    )
    Dc = NormalizedGeneralizedGammaPSD.compute_Dc(i=i, j=j, Mi=Mi, Mj=Mj)
    Nc = NormalizedGeneralizedGammaPSD.compute_Nc(i=i, j=j, Mi=Mi, Mj=Mj)

    # Define search space
    if mu is None:
        mu = np.arange(-7, 30, step=0.01)
    if c is None:
        c = np.arange(0.01, 10, step=0.01)

    # Define kwargs
    kwargs = {
        "i": i,
        "j": j,
        "D": ds["diameter_bin_center"].data,
        "dD": ds["diameter_bin_width"].data,
        "objectives": objectives,
        "return_loss": return_loss,
        "mu": mu,
        "c": c,
    }

    # Define function to create parameters dataset
    def _create_parameters_dataset(da_parameters, i, j):
        # Add parameters coordinates
        da_parameters = da_parameters.assign_coords({"parameters": ["Nc", "Dc", "mu", "c"]})

        # Create parameters dataset
        ds_parameters = da_parameters.to_dataset(dim="parameters")

        # Add Nc and Dc
        ds_parameters["Dc"].attrs["moment_orders"] = f"{i}, {j}"
        ds_parameters["Nc"].attrs["moment_orders"] = f"{i}, {j}"

        # Add DSD model name to the attribute
        ds_parameters.attrs["disdrodb_psd_model"] = "NormalizedGeneralizedGammaPSD"
        ds_parameters.attrs["disdrodb_psd_model_kwargs"] = f"{{'i': {i}, 'j': {j}}}"
        return ds_parameters

    # Return cost function if asked
    if return_loss:
        da_cost_function, da_parameters = xr.apply_ufunc(
            apply_normalized_generalized_gamma_gs,
            # Variables varying over time
            Nc,
            Dc,
            ds["drop_number_concentration"],
            ds["fall_velocity"],
            # Other options
            kwargs=kwargs,
            # Settings
            input_core_dims=[[], [], [DIAMETER_DIMENSION], [DIAMETER_DIMENSION]],
            output_core_dims=[["c_values", "mu_values"], ["parameters"]],
            vectorize=True,
            dask="parallelized",
            # Lengths of the new output_core_dims dimensions.
            dask_gufunc_kwargs={"output_sizes": {"mu_values": len(mu), "c_values": len(c), "parameters": 4}},
            output_dtypes=["float64", "float64", "float64"],
        )
        ds_parameters = _create_parameters_dataset(da_parameters, i=i, j=j)
        ds_parameters["cost_function"] = da_cost_function
        ds_parameters = ds_parameters.assign_coords({"mu_values": mu, "c_values": c})
        return ds_parameters

    # Otherwise return just best parameters
    da_parameters = xr.apply_ufunc(
        apply_normalized_generalized_gamma_gs,
        # Variables varying over time
        Nc,
        Dc,
        ds["drop_number_concentration"],
        ds["fall_velocity"],
        # Other options
        kwargs=kwargs,
        # Settings
        input_core_dims=[[], [], [], [], [DIAMETER_DIMENSION], [DIAMETER_DIMENSION]],
        output_core_dims=[["parameters"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"parameters": 4}},  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )
    ds_parameters = _create_parameters_dataset(da_parameters, i=i, j=j)
    return ds_parameters


####-----------------------------------------------------------------.
#### Methods of Moments (MOM)
# - M246 DEFAULT FOR GAMMA ?
# - LMOM (Johnson et al., 2014)


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
    if l == m:
        raise ValueError("Equal l and m moment orders are not allowed.")
    num = moment_l * gamma(m + 1)
    den = moment_m * gamma(l + 1)
    Lambda = np.power(num / den, (1 / (m - l)))
    N0 = moment_l * np.power(Lambda, l + 1) / gamma(l + 1)
    return N0, Lambda


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
    return N0, Lambda


# def get_gamma_parameters_M012(M0, M1, M2):
#     """Compute gamma distribution parameters following Cao et al., 2009.

#     References
#     ----------
#     Cao, Q., and G. Zhang, 2009:
#     Errors in Estimating Raindrop Size Distribution Parameters Employing Disdrometer  and Simulated Raindrop Spectra.
#     J. Appl. Meteor. Climatol., 48, 406-425, https://doi.org/10.1175/2008JAMC2026.1.
#     """
#     # TODO: really bad results. check formula !
#     G = M1**3 / M0 / M2
#     mu = 1 / (1 - G) - 2
#     Lambda = M0 / M1 * (mu + 1)
#     N0 = Lambda ** (mu + 1) * M0 / gamma(mu + 1)
#     return N0, mu, Lambda


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
    return N0, mu, Lambda


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
    return N0, mu, Lambda


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
    return N0, mu, Lambda


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
    return N0, mu, Lambda


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
    return Nt, mu, sigma


def _compute_moments(ds, moments):
    list_moments = [
        get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,  # m
            diameter_bin_width=ds["diameter_bin_width"],  # mm
            moment=int(moment.replace("M", "")),
        )
        for moment in moments
    ]
    return list_moments


def _get_gamma_parameters_mom(ds: xr.Dataset, mom_method: str) -> xr.Dataset:
    # Get the correct function and list of variables for the requested method
    func, needed_moments = MOM_METHODS_DICT["GammaPSD"][mom_method]

    # Compute required moments
    arrs = _compute_moments(ds, moments=needed_moments)

    # Apply the function. This will produce (mu, Lambda, N0) with the same coords/shapes as input data
    N0, mu, Lambda = func(*arrs)

    # Return a new Dataset containing the results
    ds = xr.Dataset(
        {
            "N0": N0,
            "mu": mu,
            "Lambda": Lambda,
        },
        coords=ds.coords,
    )
    return ds


def _get_lognormal_parameters_mom(ds: xr.Dataset, mom_method: str) -> xr.Dataset:
    # Get the correct function and list of variables for the requested method
    func, needed_moments = MOM_METHODS_DICT["LognormalPSD"][mom_method]

    # Compute required moments
    arrs = _compute_moments(ds, moments=needed_moments)

    # Apply the function. This will produce (mu, Lambda, N0) with the same coords/shapes as input data
    Nt, mu, sigma = func(*arrs)

    # Return a new Dataset containing the results
    ds = xr.Dataset(
        {
            "Nt": Nt,
            "mu": mu,
            "sigma": sigma,
        },
        coords=ds.coords,
    )
    return ds


def _get_exponential_parameters_mom(ds: xr.Dataset, mom_method: str) -> xr.Dataset:
    # Get the correct function and list of variables for the requested method
    func, needed_moments = MOM_METHODS_DICT["ExponentialPSD"][mom_method]

    # Compute required moments
    arrs = _compute_moments(ds, moments=needed_moments)

    # Apply the function. This will produce (mu, Lambda, N0) with the same coords/shapes as input data
    N0, Lambda = func(*arrs)

    # Return a new Dataset containing the results
    ds = xr.Dataset(
        {
            "N0": N0,
            "Lambda": Lambda,
        },
        coords=ds.coords,
    )
    return ds


####--------------------------------------------------------------------------------------.
#### Routines dictionary

####--------------------------------------------------------------------------------------.
ATTRS_PARAMS_DICT = {
    "GammaPSD": {
        "N0": {
            "description": "Intercept parameter of the Gamma PSD",
            "standard_name": "particle_size_distribution_intercept",
            "units": "mm**(-1-mu) m-3",
            "long_name": "GammaPSD intercept parameter",
        },
        "mu": {
            "description": "Shape parameter of the Gamma PSD",
            "standard_name": "particle_size_distribution_shape",
            "units": "",
            "long_name": "GammaPSD shape parameter",
        },
        "Lambda": {
            "description": "Slope (rate) parameter of the Gamma PSD",
            "standard_name": "particle_size_distribution_slope",
            "units": "mm-1",
            "long_name": "GammaPSD slope parameter",
        },
    },
    "NormalizedGammaPSD": {
        "Nw": {
            "standard_name": "normalized_intercept_parameter",
            "units": "mm-1 m-3",
            "long_name": "NormalizedGammaPSD Normalized Intercept Parameter",
        },
        "mu": {
            "description": "Dimensionless shape parameter controlling the curvature of the Normalized Gamma PSD",
            "standard_name": "particle_size_distribution_shape",
            "units": "",
            "long_name": "NormalizedGammaPSD Shape Parameter ",
        },
        "D50": {
            "standard_name": "median_volume_diameter",
            "units": "mm",
            "long_name": "NormalizedGammaPSD Median Volume Drop Diameter",
        },
    },
    "LognormalPSD": {
        "Nt": {
            "standard_name": "number_concentration_of_rain_drops_in_air",
            "units": "m-3",
            "long_name": "Total Number Concentration",
        },
        "mu": {
            "description": "Mean of the Lognormal PSD",
            "units": "log(mm)",
            "long_name": "Mean of the Lognormal PSD",
        },
        "sigma": {
            "standard_name": "Standard Deviation of the Lognormal PSD",
            "units": "",
            "long_name": "Standard Deviation of the Lognormal PSD",
        },
    },
    "ExponentialPSD": {
        "N0": {
            "description": "Intercept parameter of the Exponential PSD",
            "standard_name": "particle_size_distribution_intercept",
            "units": "mm-1 m-3",
            "long_name": "ExponentialPSD intercept parameter",
        },
        "Lambda": {
            "description": "Slope (rate) parameter of the Exponential PSD",
            "standard_name": "particle_size_distribution_slope",
            "units": "mm-1",
            "long_name": "ExponentialPSD slope parameter",
        },
    },
}

PSD_MODELS = list(ATTRS_PARAMS_DICT)

MOM_METHODS_DICT = {
    "GammaPSD": {
        # "M012": (get_gamma_parameters_M012, ["M0", "M1", "M2"]),
        "M234": (get_gamma_parameters_M234, ["M2", "M3", "M4"]),
        "M246": (get_gamma_parameters_M246, ["M2", "M4", "M6"]),
        "M456": (get_gamma_parameters_M456, ["M4", "M5", "M6"]),
        "M346": (get_gamma_parameters_M346, ["M3", "M4", "M6"]),
    },
    "LognormalPSD": {
        "M346": (get_lognormal_parameters_M346, ["M3", "M4", "M6"]),
    },
    "ExponentialPSD": {
        "M234": (get_exponential_parameters_M34, ["M3", "M4"]),
    },
}


OPTIMIZATION_ROUTINES_DICT = {
    "MOM": {
        "GammaPSD": _get_gamma_parameters_mom,
        "LognormalPSD": _get_lognormal_parameters_mom,
        "ExponentialPSD": _get_exponential_parameters_mom,
    },
    "GS": {
        "GammaPSD": get_gamma_parameters_gs,
        "NormalizedGammaPSD": get_normalized_gamma_parameters_gs,
        "LognormalPSD": get_lognormal_parameters_gs,
        "ExponentialPSD": get_exponential_parameters_gs,
        "GeneralizedGammaPSD": get_generalized_gamma_parameters_gs,
    },
    "ML": {
        "GammaPSD": get_gamma_parameters,
        "LognormalPSD": get_lognormal_parameters,
        "ExponentialPSD": get_exponential_parameters,
    },
}


def available_mom_methods(psd_model):
    """Implemented MOM methods for a given PSD model."""
    if psd_model not in MOM_METHODS_DICT:
        raise NotImplementedError(f"No MOM methods available for {psd_model}")
    return list(MOM_METHODS_DICT[psd_model])


def available_optimization(psd_model):
    """Implemented fitting methods for a given PSD model."""
    return [opt for opt in list(OPTIMIZATION_ROUTINES_DICT) if psd_model in OPTIMIZATION_ROUTINES_DICT[opt]]


####--------------------------------------------------------------------------------------.
#### Argument checkers


def check_psd_model(psd_model, optimization):
    """Check valid psd_model argument."""
    valid_psd_models = list(OPTIMIZATION_ROUTINES_DICT[optimization])
    if psd_model not in valid_psd_models:
        msg = (
            f"{optimization} optimization is not available for 'psd_model' {psd_model}. "
            f"Accepted PSD models are {valid_psd_models}."
        )
        raise NotImplementedError(msg)


def check_likelihood(likelihood):
    """Check valid likelihood argument."""
    valid_likelihood = ["multinomial", "poisson"]
    if likelihood not in valid_likelihood:
        raise ValueError(f"Invalid 'likelihood' {likelihood}. Valid values are {valid_likelihood}.")
    return likelihood


def check_truncated_likelihood(truncated_likelihood):
    """Check valid truncated_likelihood argument."""
    if not isinstance(truncated_likelihood, bool):
        raise TypeError(f"Invalid 'truncated_likelihood' argument {truncated_likelihood}. Must be True or False.")
    return truncated_likelihood


def check_probability_method(probability_method):
    """Check valid probability_method argument."""
    # Check valid probability_method
    valid_probability_method = ["cdf", "pdf"]
    if probability_method not in valid_probability_method:
        raise ValueError(
            f"Invalid 'probability_method' {probability_method}. Valid values are {valid_probability_method}.",
        )
    return probability_method


def check_optimizer(optimizer):
    """Check valid optimizer argument."""
    # Check valid probability_method
    valid_optimizer = ["Nelder-Mead", "Powell", "L-BFGS-B"]
    if optimizer not in valid_optimizer:
        raise ValueError(
            f"Invalid 'optimizer' {optimizer}. Valid values are {valid_optimizer}.",
        )
    return optimizer


def check_mom_methods(mom_methods, psd_model, allow_none=False):
    """Check valid mom_methods arguments."""
    if isinstance(mom_methods, (str, type(None))):
        mom_methods = [mom_methods]
    mom_methods = [str(v) for v in mom_methods]  # None --> 'None'
    valid_mom_methods = available_mom_methods(psd_model)
    if allow_none:
        valid_mom_methods = [*valid_mom_methods, "None"]
    invalid_mom_methods = np.array(mom_methods)[np.isin(mom_methods, valid_mom_methods, invert=True)]
    if len(invalid_mom_methods) > 0:
        raise ValueError(
            f"Unknown mom_methods '{invalid_mom_methods}' for {psd_model}. Choose from {valid_mom_methods}.",
        )
    return mom_methods


def check_optimization(optimization):
    """Check valid optimization argument."""
    valid_optimization = list(OPTIMIZATION_ROUTINES_DICT)
    if optimization not in valid_optimization:
        raise ValueError(
            f"Invalid 'optimization' {optimization}. Valid procedure are {valid_optimization}.",
        )
    return optimization


def check_optimization_kwargs(optimization_kwargs, optimization, psd_model):
    """Check valid optimization_kwargs."""
    dict_arguments = {
        "ML": {
            "init_method": None,
            "probability_method": check_probability_method,
            "likelihood": check_likelihood,
            "truncated_likelihood": check_truncated_likelihood,
            "optimizer": check_optimizer,
        },
        "GS": {
            "target": check_target,
            "transformation": check_transformation,
            "error_order": None,
            "censoring": check_censoring,
        },
        "MOM": {
            "mom_methods": None,
        },
    }
    optimization = check_optimization(optimization)
    check_psd_model(psd_model=psd_model, optimization=optimization)

    # Retrieve the expected arguments for the given optimization method
    expected_arguments = dict_arguments.get(optimization, {})

    # Check for missing arguments in optimization_kwargs
    # missing_args = [arg for arg in expected_arguments if arg not in optimization_kwargs]
    # if missing_args:
    #     raise ValueError(f"Missing required arguments for {optimization} optimization: {missing_args}")

    # Validate arguments values
    _ = [
        check(optimization_kwargs[arg])
        for arg, check in expected_arguments.items()
        if callable(check) and arg in optimization_kwargs
    ]

    # Further special checks
    if optimization == "MOM" and "mom_methods" in optimization_kwargs:
        _ = check_mom_methods(mom_methods=optimization_kwargs["mom_methods"], psd_model=psd_model)
    if optimization == "ML" and optimization_kwargs.get("init_method", None) is not None:
        _ = check_mom_methods(mom_methods=optimization_kwargs["init_method"], psd_model=psd_model, allow_none=True)


####--------------------------------------------------------------------------------------.
#### Wrappers for fitting


def _finalize_attributes(ds_params, psd_model, optimization, optimization_kwargs):
    ds_params.attrs["disdrodb_psd_model"] = psd_model
    ds_params.attrs["disdrodb_psd_optimization"] = optimization
    ds_params.attrs["disdrodb_psd_optimization_kwargs"] = ", ".join(
        [f"{k}: {v}" for k, v in optimization_kwargs.items()],
    )
    return ds_params


def get_mom_parameters(ds: xr.Dataset, psd_model: str, mom_methods=None) -> xr.Dataset:
    """
    Compute PSD model parameters using various method-of-moments (MOM) approaches.

    The method is specified by the `mom_methods` abbreviations, e.g. 'M012', 'M234', 'M246'.

    Parameters
    ----------
    ds : xarray.Dataset
        An xarray Dataset with the required moments M0...M6 as data variables.
    mom_methods: str or list (optional)
        See valid values with disdrodb.psd.available_mom_methods(psd_model)
        If None (the default), compute model parameters with all available MOM methods.

    Returns
    -------
    xarray.Dataset
        A Dataset containing mu, Lambda, and N0 variables.
        If multiple mom_methods are specified, the dataset has the dimension mom_method.

    """
    # Check inputs
    check_psd_model(psd_model=psd_model, optimization="MOM")
    if mom_methods is None:
        mom_methods = available_mom_methods(psd_model)
    mom_methods = check_mom_methods(mom_methods, psd_model=psd_model)

    # Retrieve function
    func = OPTIMIZATION_ROUTINES_DICT["MOM"][psd_model]

    # Compute parameters
    if len(mom_methods) == 1:
        ds_params = func(ds=ds, mom_method=mom_methods[0])
    else:
        list_ds = [func(ds=ds, mom_method=mom_method) for mom_method in mom_methods]
        ds_params = xr.concat(list_ds, dim="mom_method")
        ds_params = ds_params.assign_coords({"mom_method": mom_methods})

    # Add model attributes
    optimization_kwargs = {"mom_methods": mom_methods}
    ds_params = _finalize_attributes(
        ds_params=ds_params,
        psd_model=psd_model,
        optimization="MOM",
        optimization_kwargs=optimization_kwargs,
    )
    return ds_params


def get_ml_parameters(
    ds,
    psd_model,
    init_method=None,
    probability_method="cdf",
    likelihood="multinomial",
    truncated_likelihood=True,
    optimizer="Nelder-Mead",
):
    """
    Estimate model parameters for a given distribution using Maximum Likelihood.

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
    psd_model : str
        The PSD model to fit. See ``available_psd_models()``.
    init_method: str or list
        The method(s) of moments used to initialize the PSD model parameters.
        Multiple methods can be specified. See ``available_mom_methods(psd_model)``.
    probability_method : str, optional
        Method to compute probabilities. The default value is ``cdf``.
    likelihood : str, optional
        Likelihood function to use for fitting. The default value is ``multinomial``.
    truncated_likelihood : bool, optional
        Whether to use Truncated Maximum Likelihood (TML). The default value is ``True``.
    optimizer : str, optional
        Optimization method to use. The default value is ``Nelder-Mead``.

    Returns
    -------
    xarray.Dataset
        The dataset containing the estimated parameters.

    """
    # -----------------------------------------------------------------------------.
    # Check arguments
    check_psd_model(psd_model, optimization="ML")
    likelihood = check_likelihood(likelihood)
    probability_method = check_probability_method(probability_method)
    optimizer = check_optimizer(optimizer)

    # Check valid init_method
    init_method = check_mom_methods(mom_methods=init_method, psd_model=psd_model, allow_none=True)

    # Retrieve estimation function
    func = OPTIMIZATION_ROUTINES_DICT["ML"][psd_model]

    # Compute parameters
    if init_method is None or len(init_method) == 1:
        ds_params = func(
            ds=ds,
            init_method=init_method[0],
            probability_method=probability_method,
            likelihood=likelihood,
            truncated_likelihood=truncated_likelihood,
            optimizer=optimizer,
        )
    else:
        list_ds = [
            func(
                ds=ds,
                init_method=method,
                probability_method=probability_method,
                likelihood=likelihood,
                truncated_likelihood=truncated_likelihood,
                optimizer=optimizer,
            )
            for method in init_method
        ]
        ds_params = xr.concat(list_ds, dim="init_method")
        ds_params = ds_params.assign_coords({"init_method": init_method})

    # Add model attributes
    optimization_kwargs = {
        "init_method": init_method,
        "probability_method": "probability_method",
        "likelihood": likelihood,
        "truncated_likelihood": truncated_likelihood,
        "optimizer": optimizer,
    }
    ds_params = _finalize_attributes(
        ds_params=ds_params,
        psd_model=psd_model,
        optimization="ML",
        optimization_kwargs=optimization_kwargs,
    )

    # Return dataset with parameters
    return ds_params


def get_gs_parameters(ds, psd_model, target="N(D)", transformation="log", error_order=1, censoring="none"):
    """Estimate PSD model parameters using Grid Search optimization.

    This function estimates particle size distribution (PSD) model parameters
    by minimizing the error between observed and modeled PSD quantities through
    a grid search over the parameter space.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing PSD observations. Must include:
        - ``drop_number_concentration`` : Drop number concentration [m⁻³ mm⁻¹]
        - ``diameter_bin_center`` : Diameter bin centers [mm]
        - ``diameter_bin_width`` : Diameter bin widths [mm]
        - ``fall_velocity`` : Drop fall velocity [m s⁻¹] (required if target='R')
    psd_model : str
        Name of the PSD model to fit. Valid options are:
        - ``"GammaPSD"`` : Gamma distribution
        - ``"NormalizedGammaPSD"`` : Normalized gamma distribution
        - ``"LognormalPSD"`` : Lognormal distribution
        - ``"ExponentialPSD"`` : Exponential distribution
    target : str, optional
        Target quantity to optimize. Valid options:
        - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
        - ``"R"`` : Rain rate [mm h⁻¹]
        - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
        - ``"LWC"`` : Liquid water content [g m⁻³]
    transformation : str, optional
        Transformation applied to the target quantity before computing the error.
        Valid options:
        - ``"identity"`` : No transformation
        - ``"log"`` : Logarithmic transformation (default)
        - ``"sqrt"`` : Square root transformation
    error_order : int, optional
        Order of the error metric (p-norm). Default is 1 (L1 norm)(MAE).
        Use 2 for L2 norm (MSEs).
    censoring : {"none", "left", "right", "both"}, optional
        Specifies whether the observed PSD is treated as censored at
        the diameter edges due to instrumental sensitivity limits.
        - ``"none"`` : No censoring is applied. All diameter bins are used.
        - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
          the spectrum where the observed number concentration is zero are
          removed prior to cost-function evaluation.
        - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
          the spectrum where the observed number concentration is zero are
          removed prior to cost-function evaluation.
        - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
          range of diameter bins with non-zero observed concentrations is
          retained.

    Returns
    -------
    ds_params : xarray.Dataset
        Dataset containing the estimated PSD model parameters.
        Variables depend on the selected ``psd_model``:
        - ``GammaPSD`` : ``N0``, ``mu``, ``Lambda``
        - ``NormalizedGammaPSD`` : ``Nw``, ``mu``, ``Dm``
        - ``LognormalPSD`` : ``Nt``, ``mu``, ``sigma``
        - ``ExponentialPSD`` : ``N0``, ``Lambda``

        Each parameter variable includes attributes describing the parameter
        name, units, and optimization metadata.

    Notes
    -----
    Grid search optimization explores a predefined parameter space to find
    the combination that minimizes the specified error metric. This method
    is more robust than gradient-based methods but can be computationally
    expensive for high-dimensional parameter spaces.

    If ``drop_number_concentration`` values are all zeros or contain
    non-finite values, the output PSD parameters are set to NaN.
    """
    # Check valid psd_model
    check_psd_model(psd_model, optimization="GS")

    # Check valid target
    target = check_target(target)

    # Check valid censoring
    censoring = check_censoring(censoring)

    # Check valid transformation
    transformation = check_transformation(transformation)

    # Check fall velocity is available if target R
    if "fall_velocity" not in ds:
        ds["fall_velocity"] = get_rain_fall_velocity_from_ds(ds)

    # Retrieve estimation function
    func = OPTIMIZATION_ROUTINES_DICT["GS"][psd_model]

    # Estimate parameters
    ds_params = func(ds, target=target, transformation=transformation, error_order=error_order)

    # Add model attributes
    optimization_kwargs = {
        "target": target,
        "transformation": transformation,
        "error_order": error_order,
    }
    ds_params = _finalize_attributes(
        ds_params=ds_params,
        psd_model=psd_model,
        optimization="GS",
        optimization_kwargs=optimization_kwargs,
    )
    # Return dataset with parameters
    return ds_params


def sanitize_drop_number_concentration(drop_number_concentration):
    """Sanitize drop number concentration array.

    If N(D) is all zero or contain not finite values, set everything to np.nan
    """
    # Condition 1: all zeros along diameter_bin_center
    all_zero = (drop_number_concentration == 0).all(dim="diameter_bin_center")

    # Condition 2: any non-finite along diameter_bin_center
    any_nonfinite = (~np.isfinite(drop_number_concentration)).any(dim="diameter_bin_center")

    # Combine conditions
    invalid = all_zero | any_nonfinite

    # Replace entire profile with NaN where invalid
    drop_number_concentration = drop_number_concentration.where(~invalid, np.nan)
    return drop_number_concentration


def estimate_model_parameters(
    ds,
    psd_model,
    optimization,
    optimization_kwargs=None,
):
    """Estimate particle size distribution model parameters.

    This is the main interface function for fitting PSD models to observed data.
    It supports three optimization methods: Maximum Likelihood (ML), Method of
    Moments (MOM), and Grid Search (GS).

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing PSD observations. Must include:

        - ``drop_number_concentration`` : Drop number concentration [m⁻³ mm⁻¹]
        - ``diameter_bin_center`` : Diameter bin centers [mm]
        - ``diameter_bin_width`` : Diameter bin widths [mm]

        Additional variables required for specific optimization methods:

        - For ML: ``diameter_bin_lower``, ``diameter_bin_upper``
        - For GS with target='R': ``fall_velocity`` (auto-computed if missing)
        - For MOM: Moment variables ``M0``, ``M1``, ..., ``M6`` (depending on method)
    psd_model : str
        Name of the PSD model to fit. Valid options:

        - ``"GammaPSD"`` : Gamma distribution
        - ``"NormalizedGammaPSD"`` : Normalized gamma distribution
        - ``"LognormalPSD"`` : Lognormal distribution
        - ``"ExponentialPSD"`` : Exponential distribution

        Use ``available_optimization(psd_model)`` to check which optimization
        methods are available for a given model.
    optimization : str
        Optimization method to use. Valid options:

        - ``"ML"`` : Maximum Likelihood estimation
        - ``"MOM"`` : Method of Moments
        - ``"GS"`` : Grid Search
    optimization_kwargs : dict, optional
        Dictionary of keyword arguments specific to the chosen optimization method.

        For ``optimization="ML"``:

        - ``init_method`` : str or list, Method(s) of moments for parameter initialization
        - ``probability_method`` : str, Method to compute probabilities (default: 'cdf')
        - ``likelihood`` : str, Likelihood function ('multinomial' or 'poisson', default: 'multinomial')
        - ``truncated_likelihood`` : bool, Use truncated likelihood (default: True)
        - ``optimizer`` : str, Optimization algorithm (default: 'Nelder-Mead')

        For ``optimization="GS"``:

        - ``target`` : str, Target quantity to optimize ('ND', 'R', 'Z', 'LWC', default: 'ND')
        - ``transformation`` : str, Error transformation ('identity', 'log', 'sqrt', default: 'log')
        - ``error_order`` : int, Error metric order (default: 1)
        - ``censoring`` : str, Censoring type ('none', 'left', 'right', 'both', default: 'none')

        For ``optimization="MOM"``:

        - ``mom_methods`` : str or list, Method(s) of moments to use (e.g., 'M234')

    Returns
    -------
    ds_params : xarray.Dataset
        Dataset containing the estimated PSD model parameters with attributes.
        Variables depend on the selected ``psd_model``:

        - ``GammaPSD`` : ``N0``, ``mu``, ``Lambda``
        - ``NormalizedGammaPSD`` : ``Nw``, ``mu``, ``Dm``
        - ``LognormalPSD`` : ``Nt``, ``mu``, ``sigma``
        - ``ExponentialPSD`` : ``N0``, ``Lambda``

        Each parameter variable includes attributes with parameter name, units,
        and optimization metadata.

        Dataset attributes include:

        - ``disdrodb_psd_model`` : The fitted PSD model name
        - ``disdrodb_psd_optimization`` : The optimization method used
        - ``disdrodb_psd_optimization_kwargs`` : String representation of kwargs
    """
    # Check inputs arguments
    optimization_kwargs = {} if optimization_kwargs is None else optimization_kwargs
    optimization = check_optimization(optimization)
    check_optimization_kwargs(optimization_kwargs=optimization_kwargs, optimization=optimization, psd_model=psd_model)

    # Check N(D)
    # --> If all 0, set to np.nan
    # --> If any is not finite --> set to np.nan
    if "drop_number_concentration" not in ds:
        raise ValueError("'drop_number_concentration' variable not present in input xarray.Dataset.")
    ds["drop_number_concentration"] = sanitize_drop_number_concentration(ds["drop_number_concentration"])

    # Define function
    dict_func = {
        "ML": get_ml_parameters,
        "MOM": get_mom_parameters,
        "GS": get_gs_parameters,
    }
    func = dict_func[optimization]

    # Retrieve parameters
    ds_params = func(ds, psd_model=psd_model, **optimization_kwargs)

    # Add parameters attributes (and units)
    for var, attrs in ATTRS_PARAMS_DICT[psd_model].items():
        ds_params[var].attrs = attrs
    return ds_params
