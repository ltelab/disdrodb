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
"""Definition of PSD models.

The class implementation is inspired by pytmatrix.psd and pyradsim.psd modules
and adapted to allow efficient vectorized computations with xarray.

Source code:
- https://github.com/jleinonen/pytmatrix/blob/master/pytmatrix/psd.py
- https://github.com/wolfidan/pyradsim/blob/master/pyradsim/psd.py

"""
import importlib

import dask.array
import numpy as np
import xarray as xr
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.special import gamma as gamma_f

from disdrodb import DIAMETER_DIMENSION
from disdrodb.utils.warnings import suppress_warnings

# Check if pytmatrix is available
# - We import pytmatrix.PSD class to pass isinstance(obj, PSD) checks in pytmatrix
if importlib.util.find_spec("pytmatrix") is not None:
    from pytmatrix.psd import PSD
else:

    class PSD:
        """Dummy."""

        pass


def available_psd_models():
    """Return a list of available PSD models."""
    return list(PSD_MODELS_DICT)


def check_psd_model(psd_model):
    """Check validity of a PSD model."""
    available_models = available_psd_models()
    if psd_model not in available_models:
        raise ValueError(f"{psd_model} is an invalid PSD model. Valid models are: {available_models}.")
    return psd_model


def check_input_parameters(parameters):
    """Check valid input parameters."""
    for param, value in parameters.items():
        if not (is_scalar(value) or isinstance(value, xr.DataArray)):
            raise TypeError(f"Parameter {param} must be a scalar or xarray.DataArray, not {type(value)}")
    return parameters


def check_diameter_inputs(D):
    """Check valid diameter input."""
    if isinstance(D, xr.DataArray) or is_scalar(D):
        return D
    if isinstance(D, (tuple, list)):
        D = np.asanyarray(D)
    if isinstance(D, (np.ndarray, dask.array.Array)):
        if D.ndim != 1:
            raise ValueError("Expecting a 1-dimensional diameter array.")
        if D.size == 0:
            raise ValueError("Expecting a non-empty diameter array.")
        return xr.DataArray(D, dims=DIAMETER_DIMENSION)
    raise TypeError(f"Invalid diameter type: {type(D)}")


def get_psd_model(psd_model):
    """Retrieve the PSD Class."""
    return PSD_MODELS_DICT[psd_model]


def get_psd_model_formula(psd_model):
    """Retrieve the PSD formula."""
    return PSD_MODELS_DICT[psd_model].formula


def create_psd(psd_model, parameters):  # TODO: check name around
    """Define a PSD from a dictionary or xr.Dataset of parameters."""
    psd_class = get_psd_model(psd_model)
    psd = psd_class.from_parameters(parameters)
    return psd


def get_required_parameters(psd_model):
    """Retrieve the list of parameters required by a PSD model."""
    psd_class = get_psd_model(psd_model)
    return psd_class.required_parameters()


def is_scalar(value):
    """Determines if the input value is a scalar."""
    return isinstance(value, (float, int)) or (isinstance(value, (np.ndarray, xr.DataArray)) and value.size == 1)


class XarrayPSD(PSD):
    """PSD class template allowing vectorized computations with xarray.

    We currently inherit from pytmatrix PSD to allow scattering simulations:
    --> https://github.com/ltelab/pytmatrix-lte/blob/880170b4ca62a04e8c843619fa1b8713b9e11894/pytmatrix/psd.py#L321
    """

    def __call__(self, D):
        """Compute the PSD."""
        D = check_diameter_inputs(D)
        with suppress_warnings():
            return self.formula(D=D, **self.parameters)

    def has_scalar_parameters(self):
        """Check if the PSD object contains only a single set of parameters."""
        return np.all([is_scalar(value) for value in self.parameters.values()])

    def has_xarray_parameters(self):
        """Check if the PSD object contains at least one xarray parameter."""
        return any(isinstance(value, xr.DataArray) for param, value in self.parameters.items())

    def isel(self, **kwargs):
        """Subset the parameters by index using xarray.isel.

        If the PSD has xarray parameters, returns a new PSD with subset parameters.
        Otherwise raises an error.
        """
        if not self.has_xarray_parameters():
            raise ValueError("isel() can only be used when PSD model parameters are xarray DataArrays")

        # Subset each xarray parameter
        new_params = {}
        for param, value in self.parameters.items():
            if isinstance(value, xr.DataArray):
                new_params[param] = value.isel(**kwargs)
            else:
                new_params[param] = value

        # Create new PSD instance
        return self.__class__.from_parameters(new_params)

    def sel(self, **kwargs):
        """Subset the parameters by label using xarray.sel.

        If the PSD has xarray parameters, returns a new PSD with subset parameters.
        Otherwise raises an error.
        """
        if not self.has_xarray_parameters():
            raise ValueError("sel() can only be used when PSD model parameters are xarray DataArrays")

        # Subset each xarray parameter
        new_params = {}
        for param, value in self.parameters.items():
            if isinstance(value, xr.DataArray):
                new_params[param] = value.sel(**kwargs)
            else:
                new_params[param] = value

        # Create new PSD instance
        return self.__class__.from_parameters(new_params)

    def __eq__(self, other):
        """Check if two objects are equal."""
        # Check class equality
        if not isinstance(other, self.__class__):
            return False
        # Get required parameters
        params = self.required_parameters()
        # Check scalar parameters case
        if self.has_scalar_parameters() and other.has_scalar_parameters():
            return all(self.parameters[param] == other.parameters[param] for param in params)
        # Check array parameters case
        return all(np.all(self.parameters[param] == other.parameters[param]) for param in params)

    # def moment(self, D, dD, order):
    #     """
    #     Compute the moments of the Particle Size Distribution (PSD).

    #     Parameters
    #     ----------
    #     D: array-like
    #         Diameter bin center in m.
    #     dD: array-like
    #         Diameter bin width in mm.
    #     order : int
    #         The order of the moment to compute.

    #     Returns
    #     -------
    #     float
    #         The computed moment of the PSD.

    #     Notes
    #     -----
    #     The method uses numerical integration (trapezoidal rule) to compute the moment.
    #     """
    #     return np.trapezoid(D**order * self.__call__(D), x=D, dx=dD)


class LognormalPSD(XarrayPSD):
    """Lognormal drop size distribution (DSD).

    Callable class to provide a lognormal PSD with the given parameters.

    The PSD form is:

    N(D) = Nt/(sqrt(2*pi)*sigma*D)) * exp(-(ln(D)-mu)**2 / (2*sigma**2))

    # g = sigma
    # theta = 0

    Attributes
    ----------
        Nt:
        g:
        theta:
        mu:
        sigma:

    """

    def __init__(self, Nt=1.0, mu=0.0, sigma=1.0):
        self.Nt = Nt
        self.mu = mu
        self.sigma = sigma
        self.parameters = {"Nt": self.Nt, "mu": self.mu, "sigma": self.sigma}
        check_input_parameters(self.parameters)

    @property
    def name(self):
        """Return name of the PSD."""
        return "LognormalPSD"

    @staticmethod
    def formula(D, Nt, mu, sigma):
        """Calculates the Lognormal PSD values."""
        coeff = Nt / (np.sqrt(2.0 * np.pi) * sigma * (D))
        return coeff * np.exp(-((np.log(D) - mu) ** 2) / (2.0 * sigma**2))

    @staticmethod
    def from_parameters(parameters):
        """Initialize LognormalPSD from a dictionary or xr.Dataset.

        Args:
            parameters (dict or xr.Dataset): Parameters to initialize the class.

        Returns
        -------
            LognormalPSD: An instance of LognormalPSD initialized with the parameters.
        """
        Nt = parameters["Nt"]
        mu = parameters["mu"]
        sigma = parameters["sigma"]
        return LognormalPSD(Nt=Nt, mu=mu, sigma=sigma)

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD."""
        return ["Nt", "mu", "sigma"]

    def parameters_summary(self):
        """Return a string with the parameter summary."""
        if self.has_scalar_parameters():
            summary = "".join(
                [
                    f"{self.name}\n",
                    f"$Nt = {self.Nt:.2f}$\n",
                    f"$\\sigma = {self.sigma:.2f}$\n" f"$\\mu = {self.mu:.2f}$\n\n",
                ],
            )
        else:
            summary = "" f"{self.name} with N-d parameters \n"
        return summary


class ExponentialPSD(XarrayPSD):
    """Exponential particle size distribution (PSD).

    Callable class to provide an exponential PSD with the given
    parameters. The attributes can also be given as arguments to the
    constructor.

    The PSD form is:
    N(D) = N0 * exp(-Lambda*D)

    Attributes
    ----------
        N0: the intercept parameter.
        Lambda: the inverse scale parameter

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.
    """

    def __init__(self, N0=1.0, Lambda=1.0):
        # Define parameters
        self.N0 = N0
        self.Lambda = Lambda
        self.parameters = {"N0": self.N0, "Lambda": self.Lambda}
        check_input_parameters(self.parameters)

    @property
    def name(self):
        """Return name of the PSD."""
        return "ExponentialPSD"

    @staticmethod
    def formula(D, N0, Lambda):
        """Calculates the Exponential PSD values."""
        return N0 * np.exp(-Lambda * D)

    @staticmethod
    def from_parameters(parameters):
        """Initialize ExponentialPSD from a dictionary or xr.Dataset.

        Args:
            parameters (dict or xr.Dataset): Parameters to initialize the class.

        Returns
        -------
            ExponentialPSD: An instance of ExponentialPSD initialized with the parameters.
        """
        N0 = parameters["N0"]
        Lambda = parameters["Lambda"]
        return ExponentialPSD(N0=N0, Lambda=Lambda)

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD."""
        return ["N0", "Lambda"]

    def parameters_summary(self):
        """Return a string with the parameter summary."""
        if self.has_scalar_parameters():
            summary = "".join(
                [
                    f"{self.name}\n",
                    f"$N0 = {self.N0:.2f}$\n",
                    f"$\\lambda = {self.Lambda:.2f}$\n\n",
                ],
            )
        else:
            summary = "" f"{self.name} with N-d parameters \n"
        return summary


class GammaPSD(ExponentialPSD):
    """Gamma particle size distribution (PSD).

    Callable class to provide an gamma PSD with the given
    parameters. The attributes can also be given as arguments to the
    constructor.

    The PSD form is:
    N(D) = N0 * D**mu * exp(-Lambda*D)

    Attributes
    ----------
        N0: the intercept parameter [mm**(-1-mu) m**-3] (scale parameter)
        Lambda: the inverse scale parameter [mm-1] (slope parameter)
        mu: the shape parameter [-]

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.

    References
    ----------
    Ulbrich, C. W., 1985: The Effects of Drop Size Distribution Truncation on
    Rainfall Integral Parameters and Empirical Relations.
    J. Appl. Meteor. Climatol., 24, 580-590, https://doi.org/10.1175/1520-0450(1985)024<0580:TEODSD>2.0.CO;2
    """

    def __init__(self, N0=1.0, mu=0.0, Lambda=1.0):
        # Define parameters
        self.N0 = N0
        self.Lambda = Lambda
        self.mu = mu
        self.parameters = {"N0": self.N0, "mu": self.mu, "Lambda": self.Lambda}
        check_input_parameters(self.parameters)

    @property
    def name(self):
        """Return name of the PSD."""
        return "GammaPSD"

    @staticmethod
    def formula(D, N0, Lambda, mu):
        """Calculates the Gamma PSD values."""
        return N0 * np.exp(mu * np.log(D) - Lambda * D)

    @staticmethod
    def from_parameters(parameters):
        """Initialize GammaPSD from a dictionary or xr.Dataset.

        Args:
            parameters (dict or xr.Dataset): Parameters to initialize the class.

        Returns
        -------
            GammaPSD: An instance of GammaPSD initialized with the parameters.
        """
        N0 = parameters["N0"]
        Lambda = parameters["Lambda"]
        mu = parameters["mu"]
        return GammaPSD(N0=N0, Lambda=Lambda, mu=mu)

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD."""
        return ["N0", "mu", "Lambda"]

    def parameters_summary(self):
        """Return a string with the parameter summary."""
        if self.has_scalar_parameters():
            summary = "".join(
                [
                    f"{self.name}\n",
                    f"$\\mu = {self.mu:.2f}$\n",
                    f"$N0 = {self.N0:.2f}$\n",
                    f"$\\lambda = {self.Lambda:.2f}$\n\n",
                ],
            )
        else:
            summary = "" f"{self.name} with N-d parameters \n"
        return summary


class NormalizedGammaPSD(XarrayPSD):
    """Normalized gamma particle size distribution (PSD).

    Callable class to provide a normalized gamma PSD with the given
    parameters. The attributes can also be given as arguments to the
    constructor.

    The PSD form is:

    N(D) = Nw * f(mu) * (D/D50)**mu * exp(-(mu+3.67)*D/D50)
    f(mu) = 6/(3.67**4) * (mu+3.67)**(mu+4)/Gamma(mu+4)

    An alternative formulation as function of Dm:
    # Testud (2001), Bringi (2001), Williams et al., 2014, Dolan 2018
    # --> Normalized with respect to liquid water content (mass) --> Nx=D3/Dm4
    N(D) = Nw * f1(mu) * (D/Dm)**mu * exp(-(mu+4)*D/Dm)   # Nw * f(D; Dm, mu)
    f1(mu) = 6/(4**4) * (mu+4)**(mu+4)/Gamma(mu+4)

    Note: gamma(4) = 6

    An alternative formulation as function of Dm:
    # Tokay et al., 2010
    # Illingworth et al., 2002 (see eq10 to derive full formulation!)
    # --> Normalized with respect to total concentration --> Nx = #/Dm
    N(D) = Nt* * f2(mu) * (D/Dm)**mu * exp(-(mu+4)*D/Dm)
    f2(mu) = (mu+4)**(mu+1)/Gamma(mu+1)

    Attributes
    ----------
        D50: the median volume diameter.
        Nw: the intercept parameter.
        mu: the shape parameter.

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.

    References
    ----------
    Willis, P. T., 1984: Functional Fits to Some Observed Drop Size Distributions and Parameterization of Rain.
    J. Atmos. Sci., 41, 1648-1661, https://doi.org/10.1175/1520-0469(1984)041<1648:FFTSOD>2.0.CO;2

    Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001: The Concept of “Normalized” Distribution
    to Describe Raindrop Spectra: A Tool for Cloud Physics and Cloud Remote Sensing.
    J. Appl. Meteor. Climatol., 40, 1118-1140, https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2

    Illingworth, A. J., and T. M. Blackman, 2002:
    The Need to Represent Raindrop Size Spectra as Normalized Gamma Distributions for
    the Interpretation of Polarization Radar Observations.
    J. Appl. Meteor. Climatol., 41, 286-297, https://doi.org/10.1175/1520-0450(2002)041<0286:TNTRRS>2.0.CO;2

    Bringi, V. N., G. Huang, V. Chandrasekar, and E. Gorgucci, 2002:
    A Methodology for Estimating the Parameters of a Gamma Raindrop Size Distribution Model from
    Polarimetric Radar Data: Application to a Squall-Line Event from the TRMM/Brazil Campaign.
    J. Atmos. Oceanic Technol., 19, 633-645, https://doi.org/10.1175/1520-0426(2002)019<0633:AMFETP>2.0.CO;2

    Bringi, V. N., V. Chandrasekar, J. Hubbert, E. Gorgucci, W. L. Randeu, and M. Schoenhuber, 2003:
    Raindrop Size Distribution in Different Climatic Regimes from Disdrometer and Dual-Polarized Radar Analysis.
    J. Atmos. Sci., 60, 354-365, https://doi.org/10.1175/1520-0469(2003)060<0354:RSDIDC>2.0.CO;2

    Tokay, A., and P. G. Bashor, 2010: An Experimental Study of Small-Scale Variability of Raindrop Size Distribution.
    J. Appl. Meteor. Climatol., 49, 2348-2365, https://doi.org/10.1175/2010JAMC2269.1

    """

    def __init__(self, Nw=1.0, D50=1.0, mu=0.0):
        self.D50 = D50
        self.mu = mu
        self.Nw = Nw
        self.parameters = {"Nw": Nw, "D50": D50, "mu": mu}
        check_input_parameters(self.parameters)

    @property
    def name(self):
        """Return the PSD name."""
        return "NormalizedGammaPSD"

    @staticmethod
    def formula(D, Nw, D50, mu):
        """Calculates the NormalizedGamma PSD values."""
        d_ratio = D / D50
        nf = Nw * 6.0 / 3.67**4 * (3.67 + mu) ** (mu + 4) / gamma_f(mu + 4)
        # return nf * d_ratio ** mu * np.exp(-(mu + 3.67) * d_ratio)
        return nf * np.exp(mu * np.log(d_ratio) - (3.67 + mu) * d_ratio)

    @staticmethod
    def from_parameters(parameters):
        """Initialize NormalizedGammaPSD from a dictionary or xr.Dataset.

        Args:
            parameters (dict or xr.Dataset): Parameters to initialize the class.

        Returns
        -------
            NormalizedGammaPSD: An instance of NormalizedGammaPSD initialized with the parameters.
        """
        D50 = parameters["D50"]
        Nw = parameters["Nw"]
        mu = parameters["mu"]
        return NormalizedGammaPSD(D50=D50, Nw=Nw, mu=mu)

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD."""
        return ["Nw", "D50", "mu"]

    def parameters_summary(self):
        """Return a string with the parameter summary."""
        if self.has_scalar_parameters():
            summary = "".join(
                [
                    f"{self.name}\n",
                    f"$\\mu = {self.mu:.2f}$\n",
                    f"$Nw = {self.Nw:.2f}$\n",
                    f"$D50 = {self.D50:.2f}$\n",
                ],
            )
        else:
            summary = "" f"{self.name} with N-d parameters \n"
        return summary


PSD_MODELS_DICT = {
    "LognormalPSD": LognormalPSD,
    "ExponentialPSD": ExponentialPSD,
    "GammaPSD": GammaPSD,
    "NormalizedGammaPSD": NormalizedGammaPSD,
}


def define_interpolator(bin_edges, bin_values, interp_method):
    """
    Returns an interpolation function that takes one argument D.

    Parameters
    ----------
      interp_method (str): Interpolation method: 'step_left', 'step_right', 'linear' or 'pchip'.
      bin_edges (array-like): Sorted array of bin edge values.
      bin_values (array-like): Array of bin values corresponding to each bin.

    Returns
    -------
    callable
      A function f(D) that returns the interpolated values.
    """
    # Ensure bin_edges and bin_values are NumPy arrays
    bin_edges = np.asarray(bin_edges)
    bin_values = np.asarray(bin_values)
    bin_center = bin_edges[:-1] + np.diff(bin_edges) / 2
    # Define a dictionary of lambda functions for each method.
    # - Each lambda accepts only the variable D.
    methods = {
        # 'linear': Linear interpolation between bin values.
        "linear": lambda D: interp1d(
            bin_center,
            bin_values,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(D),
        # 'pchip': Uses the PCHIP interpolator which preserves monotonicity.
        "pchip": lambda D: PchipInterpolator(
            bin_center,
            bin_values,
            extrapolate="extrapolate",
        )(D),
        # 'binary': Uses np.searchsorted for a vectorized direct bin lookup.
        "step_left": lambda D: _stepwise_interpolator(bin_edges, bin_values, D, side="left"),
        "step_right": lambda D: _stepwise_interpolator(bin_edges, bin_values, D, side="right"),
    }
    return methods[interp_method]


def _stepwise_interpolator(bin_edges, bin_values, D, side="left"):
    # Use np.searchsorted binary search to determine the insertion indices.
    # With side='right', it returns the index of the first element greater than D
    # Subtracting by 1 gives the bin to the left of D.
    indices = np.searchsorted(bin_edges, D, side=side) - 1
    indices = np.minimum(indices, len(bin_values) - 1)  # enable left inclusion of bin edge max
    # Prepare an array for the results. For D outside the valid range the value is 0.
    result = np.zeros_like(D, dtype=bin_values.dtype)
    # Define valid indices
    valid = (bin_edges[0] < D) & (bin_edges[-1] >= D)
    # For valid entries, assign the corresponding bin value from self.bin_psd.
    result[valid] = bin_values[indices[valid]]
    return result


class BinnedPSD(PSD):
    """Binned Particle Size Distribution (PSD).

    This class represents a binned particle size distribution (PSD) that computes PSD values
    based on provided bin edges and corresponding PSD values. The PSD is evaluated via interpolation
    using one of several available methods.

    Parameters
    ----------
    bin_edges : array_like
        A sequence of n+1 bin edge values defining the bins. The edges must be monotonically increasing.
    bin_psd : array_like
        A sequence of n PSD values corresponding to the intervals defined by bin_edges.
    interp_method : {'step_left', 'step_right', 'linear', 'pchip'}, optional
        The interpolation method used to compute the PSD values. The default is 'step_left'.

    For any input diameter (or diameters) D:
        - If D lies outside the range (bin_edges[0], bin_edges[-1]), the PSD value is set to 0.
        - The interpolation function is defined internally based on the chosen method.
        - PSD values are clipped to ensure they are non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> bin_edges = [0.0, 1.0, 2.0, 3.0, 4.0]
    >>> bin_psd = [10.0, 20.0, 30.0, 0.0]
    >>> D = np.linspace(0, 3.5, 100)
    >>>
    >>> # Using linear interpolation
    >>> psd_linear = BinnedPSD(bin_edges, bin_psd, interp_method="linear")
    >>> psd_values = psd_linear(D)
    >>>
    >>> # Values for D outside (bin_edges[0], bin_edges[-1]) are set to 0
    """

    def __init__(self, bin_edges, bin_psd, interp_method="step_left"):
        # Check array size
        if len(bin_edges) != (len(bin_psd) + 1):
            raise ValueError("There must be n+1 bin edges for n bins.")
        # Assign psd values and edges
        self.bin_edges = np.asanyarray(bin_edges)
        self.bin_psd = np.asanyarray(bin_psd)
        self.interp_method = interp_method

    def __call__(self, D):
        """Compute the PSD.

        Parameters
        ----------
        D : float
            The diameter for which to calculate the PSD.

        Returns
        -------
        array-like
            The PSD value(s) corresponding to the given diameter(s) D.
            if D values are outside the range of bin edges, 0 values are returned.

        """
        # Ensure D is numpy array of correct dimension
        D = np.asanyarray(check_diameter_inputs(D))
        # Define interpolator
        interpolator = define_interpolator(
            bin_edges=self.bin_edges,
            bin_values=self.bin_psd,
            interp_method=self.interp_method,
        )
        # Interpolate
        values = interpolator(D)
        # Mask outside bin edges
        values[~(self.bin_edges[0] < D) & (self.bin_edges[-1] >= D)] = 0
        # Clip values above 0
        # - Extrapolation of some interpolator
        values = np.clip(values, a_min=0, a_max=None)
        if D.size == 1:
            return values.item()
        return values

    def __eq__(self, other):
        """Check Binned PSD equality."""
        if other is None:
            return False
        if not isinstance(other, self.__class__):
            return False
        return (
            len(self.bin_edges) == len(other.bin_edges)
            and (self.bin_edges == other.bin_edges).all()
            and (self.bin_psd == other.bin_psd).all()
        )


####-----------------------------------------------------------------.
#### Moments Computations from PSD parameters


def get_exponential_moment(N0, Lambda, moment):
    """Compute exponential distribution moments."""
    return N0 * gamma_f(moment + 1) / Lambda ** (moment + 1)


def get_gamma_moment_v1(N0, mu, Lambda, moment):
    """Compute gamma distribution moments.

    References
    ----------
    Kozu, T., and K. Nakamura, 1991:
    Rainfall Parameter Estimation from Dual-Radar Measurements
    Combining Reflectivity Profile and Path-integrated Attenuation.
    J. Atmos. Oceanic Technol., 8, 259-270, https://doi.org/10.1175/1520-0426(1991)008<0259:RPEFDR>2.0.CO;2
    """
    # Zhang et al 2001: N0 * gamma_f(mu + moment + 1) * Lambda ** (-(mu + moment + 1))
    return N0 * gamma_f(mu + moment + 1) / Lambda ** (mu + moment + 1)


def get_gamma_moment_v2(Nt, mu, Lambda, moment):
    """Compute gamma distribution moments.

    References
    ----------
    Kozu, T., and K. Nakamura, 1991:
    Rainfall Parameter Estimation from Dual-Radar Measurements
    Combining Reflectivity Profile and Path-integrated Attenuation.
    J. Atmos. Oceanic Technol., 8, 259-270, https://doi.org/10.1175/1520-0426(1991)008<0259:RPEFDR>2.0.CO;2
    """
    return Nt * gamma_f(mu + moment + 1) / gamma_f(mu + 1) / Lambda**moment


def get_lognormal_moment(Nt, sigma, mu, moment):
    """Compute lognormal distribution moments.

    References
    ----------
    Kozu, T., and K. Nakamura, 1991:
    Rainfall Parameter Estimation from Dual-Radar Measurements
    Combining Reflectivity Profile and Path-integrated Attenuation.
    J. Atmos. Oceanic Technol., 8, 259-270, https://doi.org/10.1175/1520-0426(1991)008<0259:RPEFDR>2.0.CO;2
    """
    return Nt * np.exp(moment * mu + 1 / 2 * moment * sigma**2)
