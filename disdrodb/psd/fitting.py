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
import numpy as np
import scipy.stats as ss
import xarray as xr
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import gamma, gammainc, gammaln  # Regularized lower incomplete gamma function

from disdrodb.psd.models import ExponentialPSD, GammaPSD, LognormalPSD, NormalizedGammaPSD
from disdrodb.utils.warnings import suppress_warnings

# gamma(>171) return inf !


####--------------------------------------------------------------------------------------.
#### Goodness of fit (GOF)
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
    from disdrodb.l2.empirical_dsd import get_mode_diameter

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
        diameter_mode_deviation = get_mode_diameter(observed_values, diameter) - get_mode_diameter(
            fitted_values,
            diameter,
        )

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
    # --> Alternative: p = 1 - np.sum(pdf(diameter, params)* diameter_bin_width)  # [-]
    p = 1 - np.diff(cdf([bin_edges[0], bin_edges[-1]], params)).item()  # [-]
    # Adjusts Nt for the proportion of drops not observed
    #   p = np.clip(p, 0, 1 - 1e-12)
    if np.isclose(p, 1, atol=1e-12):
        return np.nan
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
    a,
    scale,
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
    a: float
        The shape parameter of the scipy.stats.gamma distribution.
        A good default value is 1.
    scale: float
        The scale parameter of the scipy.stats.gamma distribution.
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

    # Definite initial guess for the parameters
    initial_params = [a, scale]  # (mu=a-1, a=mu+1)

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
    with suppress_warnings():
        log_N0 = np.log(Nt) + (mu + 1) * np.log(Lambda) - gammaln(mu + 1)
        N0 = np.exp(log_N0)

    # Set parameters to np.nan if any of the parameters is not a finite number
    if not np.isfinite(N0) or not np.isfinite(mu) or not np.isfinite(Lambda):
        return null_output

    # Define output
    output = {"N0": N0, "mu": mu, "Lambda": Lambda} if output_dictionary else np.array([N0, mu, Lambda])
    return output


def _get_initial_gamma_parameters(ds, mom_method=None):
    if mom_method is None:
        ds_init = xr.Dataset(
            {
                "a": xr.ones_like(ds["M1"]),
                "scale": xr.ones_like(ds["M1"]),
            },
        )
    else:
        ds_init = get_mom_parameters(
            ds=ds,
            psd_model="GammaPSD",
            mom_methods=mom_method,
        )
        ds_init["a"] = ds_init["mu"] + 1
        ds_init["scale"] = 1 / ds_init["Lambda"]
        # If initialization results in some not finite number, set default value
        ds_init["a"] = xr.where(np.isfinite(ds_init["a"]), ds_init["a"], ds["M1"])
        ds_init["scale"] = xr.where(np.isfinite(ds_init["scale"]), ds_init["scale"], ds["M1"])
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
        If None, the scale parameter is set to 1 and mu to 0 (a=1).
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

    # Define initial parameters (a, scale)
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
        ds_init["a"],
        ds_init["scale"],
        kwargs=kwargs,
        input_core_dims=[["diameter_bin_center"], [], []],
        output_core_dims=[["parameters"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"parameters": 3}},  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )

    ds_init.isel(velocity_method=0, time=-3)

    # Add parameters coordinates
    da_params = da_params.assign_coords({"parameters": ["N0", "mu", "Lambda"]})

    # Create parameters dataset
    ds_params = da_params.to_dataset(dim="parameters")

    # Add DSD model name to the attribute
    ds_params.attrs["disdrodb_psd_model"] = "GammaPSD"
    return ds_params


def get_lognormal_parameters(
    ds,
    init_method=None,  # noqa: ARG001
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
    init_method=None,  # noqa: ARG001
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
    """Deprecated Maximum likelihood estimation of Gamma model.

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
    with suppress_warnings():
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
#### Grid Search (GS)


def _compute_rain_rate(ND, D, dD, V):
    axis = 1 if ND.ndim == 2 else None
    rain_rate = np.pi / 6 * np.sum(ND * V * (D / 1000) ** 3 * dD, axis=axis) * 3600 * 1000
    return rain_rate  # mm/h


def _compute_lwc(ND, D, dD, rho_w=1000):
    axis = 1 if ND.ndim == 2 else None
    lwc = np.pi / 6.0 * (rho_w * 1000) * np.sum((D / 1000) ** 3 * ND * dD, axis=axis)
    return lwc  # g/m3


def _compute_z(ND, D, dD):
    axis = 1 if ND.ndim == 2 else None
    z = np.sum(((D) ** 6 * ND * dD), axis=axis)  # mm⁶·m⁻³
    Z = 10 * np.log10(z)
    return Z


def _compute_cost_function(ND_obs, ND_preds, D, dD, V, target, transformation, error_order):
    # Assume ND_obs of shape (D bins) and ND_preds of shape (# params, D bins)
    if target == "ND":
        if transformation == "identity":
            errors = np.mean(np.abs(ND_obs[None, :] - ND_preds) ** error_order, axis=1)
        if transformation == "log":
            errors = np.mean(np.abs(np.log(ND_obs[None, :] + 1) - np.log(ND_preds + 1)) ** error_order, axis=1)
        if transformation == "np.sqrt":
            errors = np.mean(np.abs(np.sqrt(ND_obs[None, :]) - np.sqrt(ND_preds)) ** error_order, axis=1)
    elif target == "Z":
        errors = np.abs(_compute_z(ND_obs, D, dD) - _compute_z(ND_preds, D, dD))
    elif target == "R":
        errors = np.abs(_compute_rain_rate(ND_obs, D, dD, V) - _compute_rain_rate(ND_preds, D, dD, V))
    elif target == "LWC":
        errors = np.abs(_compute_lwc(ND_obs, D, dD) - _compute_lwc(ND_preds, D, dD))
    else:
        raise ValueError("Invalid target")
    return errors


def apply_exponential_gs(
    Nt,
    ND_obs,
    V,
    # Coords
    D,
    dD,
    # Error options
    target,
    transformation,
    error_order,
):
    """Apply Grid Search for the ExponentialPSD distribution."""
    # Define set of mu values
    lambda_arr = np.arange(0.01, 20, step=0.01)

    # Perform grid search
    with suppress_warnings():
        # Compute ND
        N0_arr = Nt * lambda_arr
        ND_preds = ExponentialPSD.formula(D=D[None, :], N0=N0_arr[:, None], Lambda=lambda_arr[:, None])

        # Compute errors
        errors = _compute_cost_function(
            ND_obs=ND_obs,
            ND_preds=ND_preds,
            D=D,
            dD=dD,
            V=V,
            target=target,
            transformation=transformation,
            error_order=error_order,
        )

    # Identify best parameter set
    best_index = np.argmin(errors)
    return np.array([N0_arr[best_index].item(), lambda_arr[best_index].item()])


def _apply_gamma_gs(mu_values, lambda_values, Nt, ND_obs, D, dD, V, target, transformation, error_order):
    """Routine for GammaPSD parameters grid search."""
    # Define combinations of parameters for grid search
    combo = np.meshgrid(mu_values, lambda_values, indexing="xy")
    mu_arr = combo[0].ravel()
    lambda_arr = combo[1].ravel()

    # Perform grid search
    with suppress_warnings():
        # Compute ND
        N0 = np.exp(np.log(Nt) + (mu_arr[:, None] + 1) * np.log(lambda_arr[:, None]) - gammaln(mu_arr[:, None] + 1))
        ND_preds = GammaPSD.formula(D=D[None, :], N0=N0, Lambda=lambda_arr[:, None], mu=mu_arr[:, None])

        # Compute errors
        errors = _compute_cost_function(
            ND_obs=ND_obs,
            ND_preds=ND_preds,
            D=D,
            dD=dD,
            V=V,
            target=target,
            transformation=transformation,
            error_order=error_order,
        )

    # Best parameter
    best_index = np.argmin(errors)
    return N0[best_index].item(), mu_arr[best_index].item(), lambda_arr[best_index].item()


def apply_gamma_gs(
    Nt,
    ND_obs,
    V,
    # Coords
    D,
    dD,
    # Error options
    target,
    transformation,
    error_order,
):
    """Estimate GammaPSD model parameters using Grid Search."""
    # Define initial set of parameters
    mu_step = 0.5
    lambda_step = 0.5
    mu_values = np.arange(0.01, 20, step=mu_step)
    lambda_values = np.arange(0, 60, step=lambda_step)

    # First round of GS
    N0, mu, Lambda = _apply_gamma_gs(
        mu_values=mu_values,
        lambda_values=lambda_values,
        Nt=Nt,
        ND_obs=ND_obs,
        D=D,
        dD=dD,
        V=V,
        target=target,
        transformation=transformation,
        error_order=error_order,
    )

    # Second round of GS
    mu_values = np.arange(mu - mu_step * 2, mu + mu_step * 2, step=mu_step / 20)
    lambda_values = np.arange(Lambda - lambda_step * 2, Lambda + lambda_step * 2, step=lambda_step / 20)
    N0, mu, Lambda = _apply_gamma_gs(
        mu_values=mu_values,
        lambda_values=lambda_values,
        Nt=Nt,
        ND_obs=ND_obs,
        D=D,
        dD=dD,
        V=V,
        target=target,
        transformation=transformation,
        error_order=error_order,
    )

    return np.array([N0, mu, Lambda])


def _apply_lognormal_gs(mu_values, sigma_values, Nt, ND_obs, D, dD, V, target, transformation, error_order):
    """Routine for LognormalPSD parameters grid search."""
    # Define combinations of parameters for grid search
    combo = np.meshgrid(mu_values, sigma_values, indexing="xy")
    mu_arr = combo[0].ravel()
    sigma_arr = combo[1].ravel()

    # Perform grid search
    with suppress_warnings():
        # Compute ND
        ND_preds = LognormalPSD.formula(D=D[None, :], Nt=Nt, mu=mu_arr[:, None], sigma=sigma_arr[:, None])

        # Compute errors
        errors = _compute_cost_function(
            ND_obs=ND_obs,
            ND_preds=ND_preds,
            D=D,
            dD=dD,
            V=V,
            target=target,
            transformation=transformation,
            error_order=error_order,
        )

    # Best parameter
    best_index = np.argmin(errors)
    return Nt, mu_arr[best_index].item(), sigma_arr[best_index].item()


def apply_lognormal_gs(
    Nt,
    ND_obs,
    V,
    # Coords
    D,
    dD,
    # Error options
    target,
    transformation,
    error_order,
):
    """Estimate LognormalPSD model parameters using Grid Search."""
    # Define initial set of parameters
    mu_step = 0.5
    sigma_step = 0.5
    mu_values = np.arange(0.01, 20, step=mu_step)  # TODO: define realistic values
    sigma_values = np.arange(0, 20, step=sigma_step)  # TODO: define realistic values

    # First round of GS
    Nt, mu, sigma = _apply_lognormal_gs(
        mu_values=mu_values,
        sigma_values=sigma_values,
        Nt=Nt,
        ND_obs=ND_obs,
        D=D,
        dD=dD,
        V=V,
        target=target,
        transformation=transformation,
        error_order=error_order,
    )

    # Second round of GS
    mu_values = np.arange(mu - mu_step * 2, mu + mu_step * 2, step=mu_step / 20)
    sigma_values = np.arange(sigma - sigma_step * 2, sigma + sigma_step * 2, step=sigma_step / 20)
    Nt, mu, sigma = _apply_lognormal_gs(
        mu_values=mu_values,
        sigma_values=sigma_values,
        Nt=Nt,
        ND_obs=ND_obs,
        D=D,
        dD=dD,
        V=V,
        target=target,
        transformation=transformation,
        error_order=error_order,
    )

    return np.array([Nt, mu, sigma])


def apply_normalized_gamma_gs(
    Nw,
    D50,
    ND_obs,
    V,
    # Coords
    D,
    dD,
    # Error options
    target,
    transformation,
    error_order,
):
    """Estimate NormalizedGammaPSD model parameters using Grid Search."""
    # Define set of mu values
    mu_arr = np.arange(0.01, 20, step=0.01)

    # Perform grid search
    with suppress_warnings():
        # Compute ND
        ND_preds = NormalizedGammaPSD.formula(D=D[None, :], D50=D50, Nw=Nw, mu=mu_arr[:, None])

        # Compute errors
        errors = _compute_cost_function(
            ND_obs=ND_obs,
            ND_preds=ND_preds,
            D=D,
            dD=dD,
            V=V,
            target=target,
            transformation=transformation,
            error_order=error_order,
        )

    # Identify best parameter set
    mu = mu_arr[np.argmin(errors)]
    return np.array([Nw, mu, D50])


def get_exponential_parameters_gs(ds, target="ND", transformation="log", error_order=1):
    """Estimate the parameters of an Exponential distribution using Grid Search."""
    # "target": ["ND", "LWC", "Z", "R"]
    # "transformation": "log", "identity", "sqrt",  # only for drop_number_concentration
    # "error_order": 1,     # MAE/MSE ... only for drop_number_concentration

    # Define kwargs
    kwargs = {
        "D": ds["diameter_bin_center"].data,
        "dD": ds["diameter_bin_width"].data,
        "target": target,
        "transformation": transformation,
        "error_order": error_order,
    }

    # Fit distribution in parallel
    da_params = xr.apply_ufunc(
        apply_exponential_gs,
        # Variables varying over time
        ds["Nt"],
        ds["drop_number_concentration"],
        ds["fall_velocity"],
        # Other options
        kwargs=kwargs,
        # Settings
        input_core_dims=[[], ["diameter_bin_center"], ["diameter_bin_center"]],
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


def get_gamma_parameters_gs(ds, target="ND", transformation="log", error_order=1):
    """Compute Grid Search to identify mu and Lambda Gamma distribution parameters."""
    # "target": ["ND", "LWC", "Z", "R"]
    # "transformation": "log", "identity", "sqrt",  # only for drop_number_concentration
    # "error_order": 1,     # MAE/MSE ... only for drop_number_concentration

    # Define kwargs
    kwargs = {
        "D": ds["diameter_bin_center"].data,
        "dD": ds["diameter_bin_width"].data,
        "target": target,
        "transformation": transformation,
        "error_order": error_order,
    }

    # Fit distribution in parallel
    da_params = xr.apply_ufunc(
        apply_gamma_gs,
        # Variables varying over time
        ds["Nt"],
        ds["drop_number_concentration"],
        ds["fall_velocity"],
        # Other options
        kwargs=kwargs,
        # Settings
        input_core_dims=[[], ["diameter_bin_center"], ["diameter_bin_center"]],
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


def get_lognormal_parameters_gs(ds, target="ND", transformation="log", error_order=1):
    """Compute Grid Search to identify mu and sigma lognormal distribution parameters."""
    # "target": ["ND", "LWC", "Z", "R"]
    # "transformation": "log", "identity", "sqrt",  # only for drop_number_concentration
    # "error_order": 1,     # MAE/MSE ... only for drop_number_concentration

    # Define kwargs
    kwargs = {
        "D": ds["diameter_bin_center"].data,
        "dD": ds["diameter_bin_width"].data,
        "target": target,
        "transformation": transformation,
        "error_order": error_order,
    }

    # Fit distribution in parallel
    da_params = xr.apply_ufunc(
        apply_lognormal_gs,
        # Variables varying over time
        ds["Nt"],
        ds["drop_number_concentration"],
        ds["fall_velocity"],
        # Other options
        kwargs=kwargs,
        # Settings
        input_core_dims=[[], ["diameter_bin_center"], ["diameter_bin_center"]],
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


def get_normalized_gamma_parameters_gs(ds, target="ND", transformation="log", error_order=1):
    r"""Estimate $\mu$ of a Normalized Gamma distribution using Grid Search.

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
    ds_params : xarray.Dataset
        Dataset containing the estimated Normalized Gamma distribution parameters.
    """
    # "target": ["ND", "LWC", "Z", "R"]
    # "transformation": "log", "identity", "sqrt",  # only for drop_number_concentration
    # "error_order": 1,     # MAE/MSE ... only for drop_number_concentration

    # Define kwargs
    kwargs = {
        "D": ds["diameter_bin_center"].data,
        "dD": ds["diameter_bin_width"].data,
        "target": target,
        "transformation": transformation,
        "error_order": error_order,
    }

    # Fit distribution in parallel
    da_params = xr.apply_ufunc(
        apply_normalized_gamma_gs,
        # Variables varying over time
        ds["Nw"],
        ds["D50"],
        ds["drop_number_concentration"],
        ds["fall_velocity"],
        # Other options
        kwargs=kwargs,
        # Settings
        input_core_dims=[[], [], ["diameter_bin_center"], ["diameter_bin_center"]],
        output_core_dims=[["parameters"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"parameters": 3}},  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )

    # Add parameters coordinates
    da_params = da_params.assign_coords({"parameters": ["Nw", "mu", "D50"]})

    # Create parameters dataset
    ds_params = da_params.to_dataset(dim="parameters")

    # Add DSD model name to the attribute
    ds_params.attrs["disdrodb_psd_model"] = "NormalizedGammaPSD"
    return ds_params


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


def get_gamma_parameters_M012(M0, M1, M2):
    """Compute gamma distribution parameters following Cao et al., 2009.

    References
    ----------
    Cao, Q., and G. Zhang, 2009:
    Errors in Estimating Raindrop Size Distribution Parameters Employing Disdrometer  and Simulated Raindrop Spectra.
    J. Appl. Meteor. Climatol., 48, 406-425, https://doi.org/10.1175/2008JAMC2026.1.
    """
    # TODO: really bad results. check formula !
    G = M1**3 / M0 / M2
    mu = 1 / (1 - G) - 2
    Lambda = M0 / M1 * (mu + 1)
    N0 = Lambda ** (mu + 1) * M0 / gamma(mu + 1)
    return N0, mu, Lambda


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


def _get_gamma_parameters_mom(ds: xr.Dataset, mom_method: str) -> xr.Dataset:
    # Get the correct function and list of variables for the requested method
    func, needed_moments = MOM_METHODS_DICT["GammaPSD"][mom_method]

    # Extract the required arrays from the dataset
    arrs = [ds[var_name] for var_name in needed_moments]

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

    # Extract the required arrays from the dataset
    arrs = [ds[var_name] for var_name in needed_moments]

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

    # Extract the required arrays from the dataset
    arrs = [ds[var_name] for var_name in needed_moments]

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
    },
    "ML": {
        "GammaPSD": get_gamma_parameters,
        "LognormalPSD": get_lognormal_parameters,
        "ExponentialPSD": get_exponential_parameters,
    },
}


def available_mom_methods(psd_model):
    """Implemented MOM methods for a given PSD model."""
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
        raise ValueError(msg)


def check_target(target):
    """Check valid target argument."""
    valid_targets = ["ND", "R", "Z", "LWC"]
    if target not in valid_targets:
        raise ValueError(f"Invalid 'target' {target}. Valid targets are {valid_targets}.")
    return target


def check_transformation(transformation):
    """Check valid transformation argument."""
    valid_transformation = ["identity", "log", "sqrt"]
    if transformation not in valid_transformation:
        raise ValueError(
            f"Invalid 'transformation' {transformation}. Valid transformations are {transformation}.",
        )
    return transformation


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


def check_mom_methods(mom_methods, psd_model):
    """Check valid mom_methods arguments."""
    if isinstance(mom_methods, str):
        mom_methods = [mom_methods]
    valid_mom_methods = available_mom_methods(psd_model)
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
    missing_args = [arg for arg in expected_arguments if arg not in optimization_kwargs]
    if missing_args:
        raise ValueError(f"Missing required arguments for {optimization} optimization: {missing_args}")

    # Validate argument values
    _ = [check(optimization_kwargs[arg]) for arg, check in expected_arguments.items() if callable(check)]

    # Further special checks
    if optimization == "MOM":
        _ = check_mom_methods(mom_methods=optimization_kwargs["mom_methods"], psd_model=psd_model)
    if optimization == "ML" and optimization_kwargs["init_method"] is not None:
        _ = check_mom_methods(mom_methods=optimization_kwargs["init_method"], psd_model=psd_model)


####--------------------------------------------------------------------------------------.
#### Wrappers for fitting


def get_mom_parameters(ds: xr.Dataset, psd_model: str, mom_methods: str) -> xr.Dataset:
    """
    Compute PSD model parameters using various method-of-moments (MOM) approaches.

    The method is specified by the `mom_methods` acronym, e.g. 'M012', 'M234', 'M246'.

    Parameters
    ----------
    ds : xarray.Dataset
        An xarray Dataset with the required moments M0...M6 as data variables.
    mom_methods: str or list
        Valid MOM methods are {'M012', 'M234', 'M246', 'M456', 'M346'}.

    Returns
    -------
    xarray.Dataset
        A Dataset containing mu, Lambda, and N0 variables.
        If multiple mom_methods are specified, the dataset has the dimension mom_method.

    """
    # Check inputs
    check_psd_model(psd_model=psd_model, optimization="MOM")
    mom_methods = check_mom_methods(mom_methods, psd_model=psd_model)

    # Retrieve function
    func = OPTIMIZATION_ROUTINES_DICT["MOM"][psd_model]

    # Compute parameters
    if len(mom_methods) == 1:
        ds = func(ds=ds, mom_method=mom_methods[0])
        ds.attrs["mom_method"] = mom_methods[0]
        return ds
    list_ds = [func(ds=ds, mom_method=mom_method) for mom_method in mom_methods]
    ds = xr.concat(list_ds, dim="mom_method")
    ds = ds.assign_coords({"mom_method": mom_methods})
    return ds


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
        See ``available_mom_methods(psd_model)``.
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
    if init_method is not None:
        init_method = check_mom_methods(mom_methods=init_method, psd_model=psd_model)

    # Retrieve estimation function
    func = OPTIMIZATION_ROUTINES_DICT["ML"][psd_model]

    # Retrieve parameters
    ds_params = func(
        ds=ds,
        init_method=init_method,
        probability_method=probability_method,
        likelihood=likelihood,
        truncated_likelihood=truncated_likelihood,
        optimizer=optimizer,
    )
    # Return dataset with parameters
    return ds_params


def get_gs_parameters(ds, psd_model, target="ND", transformation="log", error_order=1):
    """Retrieve PSD model parameters using Grid Search."""
    # Check valid psd_model
    check_psd_model(psd_model, optimization="GS")

    # Check valid target
    target = check_target(target)

    # Check valid transformation
    transformation = check_transformation(transformation)

    # Retrieve estimation function
    func = OPTIMIZATION_ROUTINES_DICT["GS"][psd_model]

    # Estimate parameters
    ds_params = func(ds, target=target, transformation=transformation, error_order=error_order)

    # Return dataset with parameters
    return ds_params


def estimate_model_parameters(
    ds,
    psd_model,
    optimization,
    optimization_kwargs,
):
    """Routine to estimate PSD model parameters."""
    optimization = check_optimization(optimization)
    check_optimization_kwargs(optimization_kwargs=optimization_kwargs, optimization=optimization, psd_model=psd_model)

    # Define function
    dict_func = {
        "ML": get_ml_parameters,
        "MOM": get_mom_parameters,
        "GS": get_gs_parameters,
    }
    func = dict_func[optimization]

    # Retrieve parameters
    ds_params = func(ds, psd_model=psd_model, **optimization_kwargs)

    # Finalize attributes
    ds_params.attrs["disdrodb_psd_model"] = psd_model
    ds_params.attrs["disdrodb_psd_optimization"] = optimization
    if optimization == "GS":
        ds_params.attrs["disdrodb_psd_optimization_target"] = optimization_kwargs["target"]

    return ds_params
