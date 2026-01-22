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
from scipy.special import gammaln

from disdrodb.constants import DIAMETER_DIMENSION
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
        return D  # If xr.DataArray(D, dims=DIAMETER_DIMENSION) make pytmatrix failing !
    raise TypeError(f"Invalid diameter type: {type(D)}")


def get_psd_model(psd_model):
    """Retrieve the PSD Class."""
    return PSD_MODELS_DICT[psd_model]


def get_psd_model_formula(psd_model):
    """Retrieve the PSD formula."""
    return PSD_MODELS_DICT[psd_model].formula


def create_psd(psd_model, parameters):
    """Define a PSD from a dictionary or xr.Dataset of parameters."""
    psd_class = get_psd_model(psd_model)
    psd = psd_class.from_parameters(parameters)
    return psd


def create_psd_from_dataset(ds_params):
    """Define PSD from DISDRODB L2M product."""
    if "disdrodb_psd_model" not in ds_params.attrs:
        raise ValueError("Expecting a DISDRODB L2M product with attribute 'disdrodb_psd_model'.")
    return create_psd(ds_params.attrs["disdrodb_psd_model"], ds_params)


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
        if self.has_xarray_parameters() and not np.isscalar(D):
            D = xr.DataArray(D, dims=DIAMETER_DIMENSION)
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
        D: the particle diameter in millimeter.

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
        D: the particle diameter in millimeter.

    Returns (call):
        The PSD value for the given diameter.

    References
    ----------
    Ulbrich, C. W., 1983.
    Natural Variations in the Analytical Form of the Raindrop Size Distribution.
    J. Appl. Meteor. Climatol., 22, 1764-1775.
    https://doi.org/10.1175/1520-0450(1983)022<1764:NVITAF>2.0.CO;2.
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

    def compute_Dm(mu, Lambda):
        """Compute Dm from PSD parameters."""
        return (mu + 4) / Lambda

    def compute_sigma_m(mu, Lambda):
        """Compute sigma_m from PSD parameters."""
        return (mu + 4) ** 0.5 / Lambda


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
        D: the particle diameter in millimeter.

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


class GeneralizedGammaPSD(XarrayPSD):
    """Generalized Gamma particle size distribution (PSD).

    Callable class to provide a generalized gamma PSD with the given parameters.

    The PSD form is:

    N(D; N_t, Λ, μ, c) = N_t * (c*Λ/Γ(μ+1)) * (Λ*D)^(c(μ+1)-1) * exp(-(Λ*D)^c)

    Where:
    - N_t: total concentration [m^-3]
    - Λ: inverse scale parameter [mm^-1]
    - μ: shape parameter (μ > -1)
    - c: shape parameter (c ≠ 0)

    Attributes
    ----------
        Nt: total concentration parameter
        Lambda: inverse scale parameter
        mu: shape parameter
        c: shape parameter

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.

    References
    ----------
    Lee, G. W., I. Zawadzki, W. Szyrmer, D. Sempere-Torres, and R. Uijlenhoet, 2004:
    A General Approach to Double-Moment Normalization of Drop Size Distributions.
    J. Appl. Meteor. Climatol., 43, 264-281, https://doi.org/10.1175/1520-0450(2004)043<0264:AGATDN>2.0.CO;2
    """

    def __init__(self, Nt=1.0, Lambda=1.0, mu=0.0, c=1.0):
        self.Nt = Nt
        self.Lambda = Lambda
        self.mu = mu
        self.c = c
        self.parameters = {
            "Nt": self.Nt,
            "Lambda": self.Lambda,
            "mu": self.mu,
            "c": self.c,
        }
        check_input_parameters(self.parameters)

    @property
    def name(self):
        """Return the PSD name."""
        return "GeneralizedGammaPSD"

    @staticmethod
    def formula(D, Nt, Lambda, mu, c):
        """Calculates the Generalized Gamma PSD values.

        Parameters
        ----------
        D : array-like
            Particle diameter
        Nt : float or array-like
            Total concentration parameter [m^-3]
        Lambda : float or array-like
            Inverse scale parameter [???]
        mu : float or array-like
            Shape parameter (μ > -1)
        c : float or array-like
            Shape parameter (c ≠ 0)

        Returns
        -------
        array-like
            PSD values
        """
        # N(D) = N_t * (c*Λ/Γ(μ+1)) * (Λ*D)^(c(μ+1)-1) * exp(-(Λ*D)^c)
        lambda_d = Lambda * D
        intercept = Nt * c * Lambda / gamma_f(mu + 1)
        power_term = lambda_d ** (c * (mu + 1) - 1)
        exp_term = np.exp(-(lambda_d**c))
        return intercept * power_term * exp_term

    @staticmethod
    def from_parameters(parameters):
        """Initialize GeneralizedGammaPSD from a dictionary or xr.Dataset.

        Args:
            parameters (dict or xr.Dataset): Parameters to initialize the class.
                Required: Nt, Lambda, mu, c

        Returns
        -------
            GeneralizedGammaPSD: An instance of GeneralizedGammaPSD
            initialized with the parameters.
        """
        Nt = parameters["Nt"]
        Lambda = parameters["Lambda"]
        mu = parameters["mu"]
        c = parameters["c"]
        return GeneralizedGammaPSD(Nt=Nt, Lambda=Lambda, mu=mu, c=c)

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD."""
        return ["Nt", "Lambda", "mu", "c"]

    def parameters_summary(self):
        """Return a string with the parameter summary."""
        if self.has_scalar_parameters():
            summary = "".join(
                [
                    f"{self.name}\n",
                    f"$N_t = {self.Nt:.2f}$\n",
                    f"$\\lambda = {self.Lambda:.2f}$\n",
                    f"$\\mu = {self.mu:.2f}$\n",
                    f"$c = {self.c:.2f}$\n",
                ],
            )
        else:
            summary = "" f"{self.name} with N-d parameters \n"
        return summary


class NormalizedGeneralizedGammaPSD(XarrayPSD):
    """Normalized Generalized Gamma particle size distribution (PSD).

    Callable class to provide a normalized generalized gamma PSD with the given
    parameters.

    The PSD form is:

    N(D; Mi, Mj, μ, c) = N_c' * c * Γ_i^((j+c(μ+1))/(i-j)) *
                            Γ_j^((-i-c(μ+1))/(i-j)) *
                            (D/D_c')^(c(μ+1)) *
                            exp(-(Γ_i/Γ_j)^(c/(i-j)) * (D/D_c')^c)
    with
    - N_c' = Mi^((j+1)/(j-i)) * Mj^((i+1)/(i-j))
    - D_c' = (Mj / Mi)^(1/(j-i))

    where:
    - Mi = Γ_i (moment parameter i)
    - Mj = Γ_j (moment parameter j)
    - μ: shape parameter, with μ > -1
    - c: shape parameter, c!=0,
    - N_c': normalized intercept parameter
    - D_c': characteristic diameter parameter

    Attributes
    ----------
        i: moment index i
        j: moment index j
        N_c: normalized intercept parameter (computed from i, j, Mi, Mj)
        D_c: characteristic diameter parameter (computed from i, j, Mi, Mj)
        c: shape parameter c
        mu: shape parameter μ

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.

    References
    ----------
    Lee, G. W., I. Zawadzki, W. Szyrmer, D. Sempere-Torres, and R. Uijlenhoet, 2004.
    A General Approach to Double-Moment Normalization of Drop Size Distributions.
    J. Appl. Meteor. Climatol., 43, 264-281, https://doi.org/10.1175/1520-0450(2004)043<0264:AGATDN>2.0.CO;2.
    """

    def __init__(self, i=1.0, j=0.0, Nc=1, Dc=1.0, c=1.0, mu=0.0):
        self.i = i
        self.j = j
        self.Nc = Nc
        self.Dc = Dc
        self.c = c
        self.mu = mu
        self.parameters = {
            "i": self.i,
            "j": self.j,
            "Nc": self.Nc,
            "Dc": self.Dc,
            "c": self.c,
            "mu": self.mu,
        }
        check_input_parameters(self.parameters)

    @staticmethod
    def compute_Nc(i, j, Mi, Mj):
        """Compute N_c' from i, j, Mi, Mj.

        N_c' = Mi^((j+1)/(j-i)) * Mj^((i+1)/(i-j))

        Parameters
        ----------
        i : float or array-like
            Moment index i
        j : float or array-like
            Moment index j
        Mi : float or array-like
            Moment parameter Mi (Γ_i)
        Mj : float or array-like
            Moment parameter Mj (Γ_j)

        Returns
        -------
        float or array-like
            The normalized intercept parameter N_c' with units m-3 mm-1.
        """
        exponent_i = (j + 1) / (j - i)
        exponent_j = (i + 1) / (i - j)
        return (Mi**exponent_i) * (Mj**exponent_j)

    @staticmethod
    def compute_Dc(i, j, Mi, Mj):
        """Compute D_c' from i, j, Mi, Mj.

        D_c' = (Mj / Mi)^(1/(j-i))

        Parameters
        ----------
        i : float or array-like
            Moment index i
        j : float or array-like
            Moment index j
        Mi : float or array-like
            Moment parameter Mi (Γ_i)
        Mj : float or array-like
            Moment parameter Mj (Γ_j)

        Returns
        -------
        float or array-like
            The characteristic diameter parameter D_c' with units mm.
        """
        exponent = 1.0 / (j - i)
        return (Mj / Mi) ** exponent

    @property
    def name(self):
        """Return the PSD name."""
        return "NormalizedGeneralizedGammaPSD"

    @staticmethod
    def formula(D, i, j, Nc, Dc, c, mu):
        """Calculates the Normalized Generalized Gamma PSD values.

        N_c' and D_c' are computed internally from the parameters.

        Parameters
        ----------
        D : array-like
            Particle diameter
        i : float
            Moment index i
        j : float
            Moment index j
        Nc : float
            General characteristic intercept (mm-1 m-3)
        Dc : float
            General characteristic diameter (mm)
        c : float
            Shape parameter c
        mu : float
            Shape parameter μ

        Returns
        -------
        array-like
            PSD values
        """
        # Compute x
        x = D / Dc

        # ---------------------------------------------------------------
        # Compute lngamma i and j
        gammaln_i = gammaln(mu + 1 + (i / c))
        gammaln_j = gammaln(mu + 1 + (j / c))

        # Compute gamma i and j
        # gamma_i = gamma_f(mu + 1  + i / c)
        # gamma_j = gamma_f(mu + 1  + j / c)

        # Calculate normalization coefficient
        # Equation: c * Γ_i^((j+c(μ+1))/(i-j)) * Γ_j^((-i-c(μ+1))/(i-j))
        pow_i = (j + c * (mu + 1)) / (i - j)
        pow_j = (-i - c * (mu + 1)) / (i - j)
        norm_coeff = c * np.exp(pow_i * gammaln_i + pow_j * gammaln_j)
        # norm_coeff = c * (gamma_i ** pow_i) * (gamma_j ** pow_j)

        # Compute ratio gammas
        # ratio_gammas = gamma_i / gamma_j
        ratio_gammas = np.exp(gammaln_i - gammaln_j)

        # Calculate the full PSD formula
        # N_c' * norm_coeff * (D/D_c')^(c(μ+1)) * exp(-(Γ_i/Γ_j)^(c/(i-j)) * (D/D_c')^c)
        exponent_power = (ratio_gammas) ** (c / (i - j))
        power_term = x ** (c * (mu + 1) - 1)
        exp_term = np.exp(-exponent_power * (x**c))
        return Nc * norm_coeff * power_term * exp_term

    @staticmethod
    def from_parameters(parameters):
        """Initialize NormalizedGeneralizedGammaPSD from a dictionary or xr.Dataset.

        Args:
            parameters (dict or xr.Dataset): Parameters to initialize the class.
                Required: i, j, Nc, Dc, c, mu

        Returns
        -------
            NormalizedGeneralizedGammaPSD: An instance of NormalizedGeneralizedGammaPSD
            initialized with the parameters.
        """
        if "disdrodb_psd_model_kwargs" in parameters.attrs:
            model_kwargs = eval(parameters.attrs["disdrodb_psd_model_kwargs"])
            i = model_kwargs["i"]
            j = model_kwargs["j"]
        else:
            i = parameters["i"]
            j = parameters["j"]
        Dc = parameters["Dc"]
        Nc = parameters["Nc"]
        c = parameters["c"]
        mu = parameters["mu"]
        return NormalizedGeneralizedGammaPSD(i=i, j=j, Nc=Nc, Dc=Dc, c=c, mu=mu)

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD."""
        return ["i", "j", "Nc", "Dc", "c", "mu"]

    def parameters_summary(self):
        """Return a string with the parameter summary."""
        if self.has_scalar_parameters():
            summary = "".join(
                [
                    f"{self.name}\n",
                    f"$i = {self.i:.2f}$\n",
                    f"$j = {self.j:.2f}$\n",
                    f"$c = {self.c:.2f}$\n",
                    f"$\\mu = {self.mu:.2f}$\n",
                    f"$N_c' = {self.Nc:.2f}$\n",
                    f"$D_c' = {self.Dc:.2f}$\n",
                ],
            )
        else:
            summary = "" f"{self.name} with N-d parameters \n"
        return summary


####-------------------------------------------------------------------------.
#### PSD_MODELS_DICT
PSD_MODELS_DICT = {
    "LognormalPSD": LognormalPSD,
    "ExponentialPSD": ExponentialPSD,
    "GammaPSD": GammaPSD,
    "GeneralizedGammaPSD": GeneralizedGammaPSD,
    "NormalizedGammaPSD": NormalizedGammaPSD,
    "NormalizedGeneralizedGammaPSD": NormalizedGeneralizedGammaPSD,
}


####-------------------------------------------------------------------------.
#### BinnedPSD


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
