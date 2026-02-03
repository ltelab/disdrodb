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

import ast
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
        """Dummy PSD class placeholder when pytmatrix is not available.

        This class serves as a placeholder when the pytmatrix library is not installed,
        allowing the module to be imported without errors while maintaining the class
        hierarchy for PSD models.
        """

        pass


def available_psd_models():
    """Return a list of available PSD models.

    Returns
    -------
    list of str
        List of available PSD model names.
    """
    return list(PSD_MODELS_DICT)


def check_psd_model(psd_model):
    """Check validity of a PSD model.

    Parameters
    ----------
    psd_model : str
        Name of the PSD model to validate.

    Returns
    -------
    str
        The validated PSD model name.

    Raises
    ------
    ValueError
        If the PSD model is not valid.
    """
    available_models = available_psd_models()
    if psd_model not in available_models:
        raise ValueError(f"{psd_model} is an invalid PSD model. Valid models are: {available_models}.")
    return psd_model


def check_input_parameters(parameters):
    """Check validity of input parameters.

    Parameters
    ----------
    parameters : dict
        Dictionary of PSD parameters to validate.

    Returns
    -------
    dict
        The validated parameters dictionary.

    Raises
    ------
    TypeError
        If any parameter is not a scalar or xarray.DataArray.
    """
    for param, value in parameters.items():
        if not (is_scalar(value) or isinstance(value, xr.DataArray)):
            raise TypeError(f"Parameter {param} must be a scalar or xarray.DataArray, not {type(value)}")
    return parameters


def check_diameter_inputs(D):
    """Check validity of diameter input.

    Parameters
    ----------
    D : int, float, array-like, or xarray.DataArray
        Diameter values to validate [mm].

    Returns
    -------
    int, float, numpy.ndarray, dask.array.Array, or xarray.DataArray
        The validated diameter input.

    Raises
    ------
    ValueError
        If the diameter array is not 1-dimensional or is empty.
    TypeError
        If the diameter type is invalid.
    """
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
    """Retrieve the PSD class.

    Parameters
    ----------
    psd_model : str
        Name of the PSD model.

    Returns
    -------
    type
        The PSD class corresponding to the model name.
    """
    return PSD_MODELS_DICT[psd_model]


def get_psd_model_formula(psd_model):
    """Retrieve the PSD formula function.

    Parameters
    ----------
    psd_model : str
        Name of the PSD model.

    Returns
    -------
    callable
        The static formula method of the PSD class.
    """
    return PSD_MODELS_DICT[psd_model].formula


def create_psd(psd_model, parameters):
    """Create a PSD instance from model name and parameters.

    Parameters
    ----------
    psd_model : str
        Name of the PSD model.
    parameters : dict or xarray.Dataset
        Dictionary or Dataset containing the PSD parameters.

    Returns
    -------
    XarrayPSD
        An instance of the specified PSD model initialized with the given parameters.
    """
    psd_class = get_psd_model(psd_model)
    psd = psd_class.from_parameters(parameters)
    return psd


def get_required_parameters(psd_model):
    """Retrieve the list of parameters required by a PSD model.

    Parameters
    ----------
    psd_model : str
        Name of the PSD model.

    Returns
    -------
    list of str
        List of required parameter names for the specified PSD model.
    """
    psd_class = get_psd_model(psd_model)
    return psd_class.required_parameters()


def create_psd_from_dataset(ds_params):
    """Create a PSD instance from a DISDRODB L2M product.

    Parameters
    ----------
    ds_params : xarray.Dataset
        DISDRODB L2M dataset containing PSD parameters and metadata.
        Must have 'disdrodb_psd_model' attribute.

    Returns
    -------
    XarrayPSD
        An instance of the PSD model specified in the dataset attributes.

    Raises
    ------
    ValueError
        If the dataset does not contain 'disdrodb_psd_model' attribute.
    """
    if "disdrodb_psd_model" not in ds_params.attrs:
        raise ValueError("Expecting a DISDRODB L2M product with attribute 'disdrodb_psd_model'.")
    return create_psd(ds_params.attrs["disdrodb_psd_model"], ds_params)


def get_parameters_from_dataset(ds):
    """Extract PSD parameters from DISDRODB L2M dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        DISDRODB L2M dataset containing PSD parameters.
        Must have 'disdrodb_psd_model' attribute.

    Returns
    -------
    xarray.Dataset
        Dataset containing only the PSD parameter variables.

    Raises
    ------
    ValueError
        If the dataset does not contain 'disdrodb_psd_model' attribute.
    """
    if "disdrodb_psd_model" not in ds.attrs:
        raise ValueError("Expecting a DISDRODB L2M product with attribute 'disdrodb_psd_model'.")
    psd_model = ds.attrs["disdrodb_psd_model"]
    # Retrieve psd parameters list
    required_parameters = get_required_parameters(psd_model)
    required_parameters = set(required_parameters) - {"i", "j"}
    return ds[required_parameters]


def is_scalar(value):
    """Determine if the input value is a scalar.

    Parameters
    ----------
    value : any
        Value to check.

    Returns
    -------
    bool
        True if the value is a scalar, False otherwise.

    Notes
    -----
    A value is considered scalar if it is an int, float, or a numpy/xarray
    array with exactly one element.
    """
    return isinstance(value, (float, int)) or (isinstance(value, (np.ndarray, xr.DataArray)) and value.size == 1)


def compute_Nc(i, j, Mi, Mj):
    r"""Compute double moment normalization intercept parameter N_c.

    The normalized intercept parameter is calculated as:

    .. math::

        N_c = M_i^{\frac{j + 1}{j - i}} M_j^{\frac{i + 1}{i - j}}

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
        The normalized intercept parameter N_c with units m-3 mm-1.
    """
    exponent_i = (j + 1) / (j - i)
    exponent_j = (i + 1) / (i - j)
    return (Mi**exponent_i) * (Mj**exponent_j)


def compute_Dc(i, j, Mi, Mj):
    r"""Compute double moment normalization characteristic diameter D_c.

    The characteristic diameter is calculated as:

    .. math::

        D_c = \left(\frac{M_j}{M_i}\right)^{\frac{1}{j - i}}

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
        The characteristic diameter parameter D_c with units mm.
    """
    exponent = 1.0 / (j - i)
    return (Mj / Mi) ** exponent


class XarrayPSD(PSD):
    """PSD class template allowing vectorized computations with xarray.

    This class serves as a base template for Particle Size Distribution (PSD) models
    that support vectorized computations with xarray.DataArray objects. It extends
    the pytmatrix PSD class to maintain compatibility with scattering simulations.

    Notes
    -----
    This class inherits from pytmatrix PSD to enable scattering simulations.
    See: https://github.com/ltelab/pytmatrix-lte/blob/880170b4ca62a04e8c843619fa1b8713b9e11894/pytmatrix/psd.py#L321

    The class supports both scalar and xarray.DataArray parameters, enabling
    efficient vectorized operations across multiple dimensions.
    """

    def __call__(self, D, zero_below=1e-3):
        """Compute the PSD values for given diameters.

        Parameters
        ----------
        D : scalar, array-like, or xarray.DataArray
            Particle diameter(s) [mm].
        zero_below : float, optional
            Threshold below which PSD values are set to zero.
            Default is 1e-3.

        Returns
        -------
        scalar, numpy.ndarray, or xarray.DataArray
            PSD values N(D) [m^-3 mm^-1] corresponding to the input diameter(s).
        """
        D = check_diameter_inputs(D)
        if self.has_xarray_parameters() and not np.isscalar(D):
            D = xr.DataArray(D, dims=DIAMETER_DIMENSION)
        with suppress_warnings():
            nd = self.formula(D=D, **self.parameters)

        # Clip values to ensure non-negative PSD (and set values < zero_below to 0)
        nd = nd.where(nd >= zero_below, 0) if isinstance(nd, xr.DataArray) else np.where(nd < zero_below, 0, nd)
        return nd

    def has_scalar_parameters(self):
        """Check if the PSD object contains only scalar parameters.

        Returns
        -------
        bool
            True if all parameters are scalars, False otherwise.
        """
        return np.all([is_scalar(value) for value in self.parameters.values()])

    def has_xarray_parameters(self):
        """Check if the PSD object contains at least one xarray parameter.

        Returns
        -------
        bool
            True if at least one parameter is an xarray.DataArray, False otherwise.
        """
        return any(isinstance(value, xr.DataArray) for param, value in self.parameters.items())

    def isel(self, **kwargs):
        """Subset the parameters by index using xarray.isel.

        Parameters
        ----------
        **kwargs : dict
            Indexing arguments passed to xarray.DataArray.isel().

        Returns
        -------
        XarrayPSD
            A new PSD instance with subset parameters.

        Raises
        ------
        ValueError
            If the PSD does not have xarray parameters.
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

        Parameters
        ----------
        **kwargs : dict
            Indexing arguments passed to xarray.DataArray.sel().

        Returns
        -------
        XarrayPSD
            A new PSD instance with subset parameters.

        Raises
        ------
        ValueError
            If the PSD does not have xarray parameters.
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
        """Check if two PSD objects are equal.

        Parameters
        ----------
        other : XarrayPSD
            Another PSD object to compare with.

        Returns
        -------
        bool
            True if the objects have the same class and parameter values, False otherwise.
        """
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
    r"""Lognormal particle size distribution (PSD).

    This class implements a lognormal PSD model, which is commonly used to
    describe particle size distributions in atmospheric sciences.

    The PSD is defined by the formula:

    .. math::

        N(D) = \frac{N_t}{\sqrt{2\pi} \sigma D} \exp\left(-\frac{(\ln(D) - \mu)^2}{2\sigma^2}\right)

    Parameters
    ----------
    Nt : float or xarray.DataArray, optional
        Total concentration parameter [m^-3].
        Default is 1.0.
    mu : float or xarray.DataArray, optional
        Location parameter of the underlying normal distribution [-].
        Default is 0.0.
    sigma : float or xarray.DataArray, optional
        Scale parameter (standard deviation) of the underlying normal distribution [-].
        Default is 1.0.

    Attributes
    ----------
    Nt : float or xarray.DataArray
        Total concentration parameter.
    mu : float or xarray.DataArray
        Location parameter.
    sigma : float or xarray.DataArray
        Scale parameter.
    parameters : dict
        Dictionary containing all PSD parameters.

    Notes
    -----
    The lognormal distribution is characterized by the fact that the logarithm
    of the variable follows a normal distribution.
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
        """Calculate the Lognormal PSD values.

        Parameters
        ----------
        D : array-like
            Particle diameter [mm].
        Nt : float or array-like
            Total concentration parameter [m^-3].
        mu : float or array-like
            Location parameter [-].
        sigma : float or array-like
            Scale parameter [-].

        Returns
        -------
        array-like
            PSD values N(D) [m^-3 mm^-1].
        """
        coeff = Nt / (np.sqrt(2.0 * np.pi) * sigma * (D))
        return coeff * np.exp(-((np.log(D) - mu) ** 2) / (2.0 * sigma**2))

    @staticmethod
    def from_parameters(parameters):
        """Initialize LognormalPSD from a dictionary or xarray.Dataset.

        Parameters
        ----------
        parameters : dict or xarray.Dataset
            Parameters to initialize the class. Must contain 'Nt', 'mu', and 'sigma'.

        Returns
        -------
        LognormalPSD
            An instance of LognormalPSD initialized with the parameters.
        """
        Nt = parameters["Nt"]
        mu = parameters["mu"]
        sigma = parameters["sigma"]
        return LognormalPSD(Nt=Nt, mu=mu, sigma=sigma)

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD.

        Returns
        -------
        list of str
            List of required parameter names.
        """
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
    r"""Exponential particle size distribution (PSD).

    This class implements an exponential PSD model, which is one of the simplest
    forms used to describe particle size distributions.

    The PSD is defined by the formula:

    .. math::

        N(D) = N_0 \exp(-\Lambda D)

    Parameters
    ----------
    N0 : float or xarray.DataArray, optional
        Intercept parameter [m^-3 mm^-1].
        Default is 1.0.
    Lambda : float or xarray.DataArray, optional
        Inverse scale parameter (slope parameter) [mm^-1].
        Default is 1.0.

    Attributes
    ----------
    N0 : float or xarray.DataArray
        Intercept parameter.
    Lambda : float or xarray.DataArray
        Inverse scale parameter.
    parameters : dict
        Dictionary containing all PSD parameters.

    Notes
    -----
    The exponential distribution is a special case of the gamma distribution
    with shape parameter mu = 0.
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
        """Calculate the Exponential PSD values.

        Parameters
        ----------
        D : array-like
            Particle diameter [mm].
        N0 : float or array-like
            Intercept parameter [m^-3 mm^-1].
        Lambda : float or array-like
            Inverse scale parameter [mm^-1].

        Returns
        -------
        array-like
            PSD values N(D) [m^-3 mm^-1].
        """
        return N0 * np.exp(-Lambda * D)

    @staticmethod
    def from_parameters(parameters):
        """Initialize ExponentialPSD from a dictionary or xarray.Dataset.

        Parameters
        ----------
        parameters : dict or xarray.Dataset
            Parameters to initialize the class. Must contain 'N0' and 'Lambda'.

        Returns
        -------
        ExponentialPSD
            An instance of ExponentialPSD initialized with the parameters.
        """
        N0 = parameters["N0"]
        Lambda = parameters["Lambda"]
        return ExponentialPSD(N0=N0, Lambda=Lambda)

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD.

        Returns
        -------
        list of str
            List of required parameter names.
        """
        return ["N0", "Lambda"]

    def parameters_summary(self):
        """Return a string with the parameter summary.

        Returns
        -------
        str
            Formatted string summarizing the PSD parameters.
        """
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
    r"""Gamma particle size distribution (PSD).

    This class implements a gamma PSD model, which is widely used to describe
    raindrop size distributions and other particle size distributions in
    atmospheric sciences.

    The PSD is defined by the formula:

    .. math::

        N(D) = N_0 D^{\mu} \exp(-\Lambda D)

    Parameters
    ----------
    N0 : float or xarray.DataArray, optional
        Intercept parameter (scale parameter) [m^-3 mm^(-1-mu)].
        Default is 1.0.
    mu : float or xarray.DataArray, optional
        Shape parameter [-].
        Default is 0.0.
    Lambda : float or xarray.DataArray, optional
        Inverse scale parameter (slope parameter) [mm^-1].
        Default is 1.0.

    Attributes
    ----------
    N0 : float or xarray.DataArray
        Intercept parameter.
    mu : float or xarray.DataArray
        Shape parameter.
    Lambda : float or xarray.DataArray
        Inverse scale parameter.
    parameters : dict
        Dictionary containing all PSD parameters.

    Notes
    -----
    The gamma distribution reduces to the exponential distribution when mu = 0.
    This formulation is particularly useful for representing natural variations
    in raindrop size distributions.

    References
    ----------
    Ulbrich, C. W., 1983.
    Natural Variations in the Analytical Form of the Raindrop Size Distribution.
    J. Appl. Meteor. Climatol., 22, 1764-1775,
    https://doi.org/10.1175/1520-0450(1983)022<1764:NVITAF>2.0.CO;2
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
        """Calculate the Gamma PSD values.

        Parameters
        ----------
        D : array-like
            Particle diameter [mm].
        N0 : float or array-like
            Intercept parameter [m^-3 mm^(-1-mu)].
        Lambda : float or array-like
            Inverse scale parameter [mm^-1].
        mu : float or array-like
            Shape parameter [-].

        Returns
        -------
        array-like
            PSD values N(D) [m^-3 mm^-1].
        """
        return N0 * np.exp(mu * np.log(D) - Lambda * D)

    @staticmethod
    def from_parameters(parameters):
        """Initialize GammaPSD from a dictionary or xarray.Dataset.

        Parameters
        ----------
        parameters : dict or xarray.Dataset
            Parameters to initialize the class. Must contain 'N0', 'Lambda', and 'mu'.

        Returns
        -------
        GammaPSD
            An instance of GammaPSD initialized with the parameters.
        """
        N0 = parameters["N0"]
        Lambda = parameters["Lambda"]
        mu = parameters["mu"]
        return GammaPSD(N0=N0, Lambda=Lambda, mu=mu)

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD.

        Returns
        -------
        list of str
            List of required parameter names.
        """
        return ["N0", "mu", "Lambda"]

    def parameters_summary(self):
        """Return a string with the parameter summary.

        Returns
        -------
        str
            Formatted string summarizing the PSD parameters.
        """
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

    @staticmethod
    def compute_Dm(mu, Lambda):
        """Compute mass-weighted mean diameter from PSD parameters.

        Parameters
        ----------
        mu : float or array-like
            Shape parameter [-].
        Lambda : float or array-like
            Inverse scale parameter [mm^-1].

        Returns
        -------
        float or array-like
            Mass-weighted mean diameter Dm [mm].
        """
        return (mu + 4) / Lambda

    @staticmethod
    def compute_sigma_m(mu, Lambda):
        """Compute standard deviation of mass-weighted distribution.

        Parameters
        ----------
        mu : float or array-like
            Shape parameter [-].
        Lambda : float or array-like
            Inverse scale parameter [mm^-1].

        Returns
        -------
        float or array-like
            Standard deviation sigma_m [mm].
        """
        return (mu + 4) ** 0.5 / Lambda


class NormalizedGammaPSD(XarrayPSD):
    r"""Normalized gamma particle size distribution (PSD).

    Callable class implementing a normalized gamma particle size distribution
    parameterized by a characteristic diameter and shape parameter. The PSD
    can be evaluated by calling the instance with particle diameters.

    Notes
    -----
    The normalized gamma PSD is defined as:

    .. math::

        N(D) = N_w \ f(\mu) \left( \frac{D}{D_{50}} \right)^{\mu} \exp\!\left[-(\mu + 3.67)\frac{D}{D_{50}}\right]

    with

    .. math::

        f(\mu) = \frac{6}{3.67^4} \frac{(\mu + 3.67)^{\mu + 4}}{\Gamma(\mu + 4)}

    where:

    - :math:`D` is the particle diameter,
    - :math:`D_{50}` is the median volume diameter,
    - :math:`N_w` is the intercept parameter,
    - :math:`\mu` is the shape parameter,
    - :math:`\Gamma(\cdot)` denotes the gamma function.

    Alternative formulation using the mass-weighted mean diameter :math:`D_m`
    (Testud et al., 2001; Bringi et al., 2001; Williams et al., 2014; Dolan et al., 2018):

    .. math::

        N(D) = N_w \, f_1(\mu) \left( \frac{D}{D_m} \right)^{\mu} \exp\!\left[-(\mu + 4)\frac{D}{D_m}\right]

    with

    .. math::

        f_1(\mu) = \frac{6}{4^4} \frac{(\mu + 4)^{\mu + 4}}{\Gamma(\mu + 4)}

    This formulation corresponds to a normalization with respect to liquid
    water content.

    Another alternative formulation normalized by total number concentration
    (Tokay et al., 2010; Illingworth et al., 2002):

    .. math::

        N(D) = N_t \, f_2(\mu) \left( \frac{D}{D_m} \right)^{\mu} \exp\!\left[-(\mu + 4)\frac{D}{D_m}\right]

    with

    .. math::

        f_2(\mu) = \frac{(\mu + 4)^{\mu + 1}}{\Gamma(\mu + 1)}

    Note that :math:`\Gamma(4) = 6`.

    Attributes
    ----------
    D50 : float or xarray.DataArray
        Median volume diameter.
    Nw : float or xarray.DataArray
        Intercept parameter.
    mu : float or xarray.DataArray
        Shape parameter.

    Parameters
    ----------
    D : float or array-like
        Particle diameter (same units as :math:`D_{50}` or :math:`D_m`).

    Returns
    -------
    float or array-like
        Particle size distribution value evaluated at diameter ``D``.

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
        """Calculate the Normalized Gamma PSD values.

        Parameters
        ----------
        D : array-like
            Particle diameter [mm].
        Nw : float or array-like
            Intercept parameter [m^-3 mm^-1].
        D50 : float or array-like
            Median volume diameter [mm].
        mu : float or array-like
            Shape parameter [-].

        Returns
        -------
        array-like
            PSD values N(D) [m^-3 mm^-1].
        """
        d_ratio = D / D50
        nf = Nw * 6.0 / 3.67**4 * (3.67 + mu) ** (mu + 4) / gamma_f(mu + 4)
        # return nf * d_ratio ** mu * np.exp(-(mu + 3.67) * d_ratio)
        return nf * np.exp(mu * np.log(d_ratio) - (3.67 + mu) * d_ratio)

    @staticmethod
    def from_parameters(parameters):
        """Initialize NormalizedGammaPSD from a dictionary or xarray.Dataset.

        Parameters
        ----------
        parameters : dict or xarray.Dataset
            Parameters to initialize the class. Must contain 'Nw', 'D50', and 'mu'.

        Returns
        -------
        NormalizedGammaPSD
            An instance of NormalizedGammaPSD initialized with the parameters.
        """
        D50 = parameters["D50"]
        Nw = parameters["Nw"]
        mu = parameters["mu"]
        return NormalizedGammaPSD(D50=D50, Nw=Nw, mu=mu)

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD.

        Returns
        -------
        list of str
            List of required parameter names.
        """
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
    r"""Generalized gamma particle size distribution (PSD).

    This class implements a generalized gamma PSD model, which extends the standard
    gamma distribution by introducing an additional shape parameter c. This provides
    greater flexibility in representing diverse particle size distributions.

    The PSD is defined by the formula:

    .. math::

        N(D; N_t, \\Lambda, \\mu, c) = N_t \\frac{c\\Lambda}{\\Gamma(\\mu+1)} (\\Lambda D)^{c(\\mu+1)-1} \\exp[-(\\Lambda D)^c]

    Parameters
    ----------
    Nt : float or xarray.DataArray, optional
        Total concentration parameter [m^-3].
        Default is 1.0.
    Lambda : float or xarray.DataArray, optional
        Inverse scale parameter (slope parameter) [mm^-1].
        Default is 1.0.
    mu : float or xarray.DataArray, optional
        Shape parameter, must satisfy mu > -1 [-].
        Default is 0.0.
    c : float or xarray.DataArray, optional
        Additional shape parameter, must satisfy c ≠ 0 [-].
        Default is 1.0.

    Attributes
    ----------
    Nt : float or xarray.DataArray
        Total concentration parameter.
    Lambda : float or xarray.DataArray
        Inverse scale parameter.
    mu : float or xarray.DataArray
        Shape parameter.
    c : float or xarray.DataArray
        Additional shape parameter.
    parameters : dict
        Dictionary containing all PSD parameters.

    Notes
    -----
    The generalized gamma distribution reduces to the standard gamma distribution
    when c = 1. The parameter c provides additional flexibility in controlling
    the shape of the distribution, particularly useful for representing
    diverse atmospheric particle populations.

    References
    ----------
    Lee, G. W., I. Zawadzki, W. Szyrmer, D. Sempere-Torres, and R. Uijlenhoet, 2004.
    A General Approach to Double-Moment Normalization of Drop Size Distributions.
    J. Appl. Meteor. Climatol., 43, 264-281,
    https://doi.org/10.1175/1520-0450(2004)043<0264:AGATDN>2.0.CO;2
    """  # noqa: E501

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
        """Initialize GeneralizedGammaPSD from a dictionary or xarray.Dataset.

        Parameters
        ----------
        parameters : dict or xarray.Dataset
            Parameters to initialize the class. Must contain 'Nt', 'Lambda', 'mu', and 'c'.

        Returns
        -------
        GeneralizedGammaPSD
            An instance of GeneralizedGammaPSD initialized with the parameters.
        """
        Nt = parameters["Nt"]
        Lambda = parameters["Lambda"]
        mu = parameters["mu"]
        c = parameters["c"]
        return GeneralizedGammaPSD(Nt=Nt, Lambda=Lambda, mu=mu, c=c)

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD.

        Returns
        -------
        list of str
            List of required parameter names.
        """
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
    r"""Normalized generalized gamma particle size distribution (PSD).

    This class implements a normalized generalized gamma PSD model based on the
    double-moment normalization framework. This formulation uses two moments of
    the distribution to derive normalized parameters, providing a flexible
    representation of particle size distributions.

    The PSD is defined by the formula:

    .. math::

        N(D; M_i, M_j, \mu, c) = N_c \, c \,
            \Gamma_i^{\frac{j + c(\mu + 1)}{i - j}}
            \Gamma_j^{\frac{-i - c(\mu + 1)}{i - j}}
            \left(\frac{D}{D_c}\right)^{c(\mu + 1) - 1}
            \exp\left[
                -\left(\frac{\Gamma_i}{\Gamma_j}\right)^{\frac{c}{i - j}}
                \left(\frac{D}{D_c}\right)^c
            \right]

    where the normalization parameters are defined as:

    .. math::

        N_c = M_i^{\frac{j + 1}{j - i}} M_j^{\frac{i + 1}{i - j}}

    .. math::

        D_c = \left(\frac{M_j}{M_i}\right)^{\frac{1}{j - i}}

    with :math:`M_i = \Gamma_i` and :math:`M_j = \Gamma_j` representing the i-th and j-th
    moments of the distribution.

    Parameters
    ----------
    i : float or int, optional
        Moment index i [-].
        Default is 1.0.
    j : float or int, optional
        Moment index j [-].
        Default is 0.0.
    Nc : float or xarray.DataArray, optional
        Normalized intercept parameter [m^-3 mm^-1].
        Default is 1.0.
    Dc : float or xarray.DataArray, optional
        Characteristic diameter parameter [mm].
        Default is 1.0.
    c : float or xarray.DataArray, optional
        Shape parameter, must satisfy c ≠ 0 [-].
        Default is 1.0.
    mu : float or xarray.DataArray, optional
        Shape parameter, must satisfy mu > -1 [-].
        Default is 0.0.

    Attributes
    ----------
    i : float or int
        Moment index i.
    j : float or int
        Moment index j.
    Nc : float or xarray.DataArray
        Normalized intercept parameter computed from moments.
    Dc : float or xarray.DataArray
        Characteristic diameter parameter computed from moments.
    c : float or xarray.DataArray
        Shape parameter.
    mu : float or xarray.DataArray
        Shape parameter.
    parameters : dict
        Dictionary containing all PSD parameters.

    Notes
    -----
    The double-moment normalization framework uses two arbitrary moments of the
    distribution to compute the normalization parameters Nc and Dc. This approach
    provides a unified framework for comparing different PSD models and relating
    them to observable quantities.

    The moment indices i and j are typically chosen based on the moments that can
    be most reliably measured or estimated from observations. Common choices include
    (i=3, j=4) or (i=3, j=6) for radar applications.

    References
    ----------
    Lee, G. W., I. Zawadzki, W. Szyrmer, D. Sempere-Torres, and R. Uijlenhoet, 2004:
    A General Approach to Double-Moment Normalization of Drop Size Distributions.
    J. Appl. Meteor. Climatol., 43, 264-281,
    https://doi.org/10.1175/1520-0450(2004)043<0264:AGATDN>2.0.CO;2
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
        r"""Compute N_c from i, j, Mi, Mj.

        .. math::

            N_c = M_i^{\frac{j + 1}{j - i}} M_j^{\frac{i + 1}{i - j}}

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
            The normalized intercept parameter N_c with units m-3 mm-1.
        """
        return compute_Nc(i=i, j=j, Mi=Mi, Mj=Mj)

    @staticmethod
    def compute_Dc(i, j, Mi, Mj):
        r"""Compute D_c from i, j, Mi, Mj.

        .. math::

            D_c = \left(\frac{M_j}{M_i}\right)^{\frac{1}{j - i}}

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
            The characteristic diameter parameter D_c with units mm.
        """
        return compute_Dc(i=i, j=j, Mi=Mi, Mj=Mj)

    @property
    def name(self):
        """Return the PSD name."""
        return "NormalizedGeneralizedGammaPSD"

    @staticmethod
    def normalized_formula(x, i, j, c, mu):
        """Calculates N(D)/Nc from x=D/Dc.

        This formula is useful to fit a single normalized PSD shape to data
        in the double normalization framework.

        Parameters
        ----------
        x : array-like
            Normalized particle diameter: x = D/Dc
        i : float
            Moment index i
        j : float
            Moment index j
        c : float
            Shape parameter c
        mu : float
            Shape parameter μ

        Returns
        -------
        array-like
            N(D)/Nc values
        """
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
        # N_c * norm_coeff * (D/D_c)^(c(μ+1)) * exp(-(Γ_i/Γ_j)^(c/(i-j)) * (D/D_c)^c)
        exponent_power = (ratio_gammas) ** (c / (i - j))
        power_term = x ** (c * (mu + 1) - 1)
        exp_term = np.exp(-exponent_power * (x**c))
        return norm_coeff * power_term * exp_term

    @staticmethod
    def formula(D, i, j, Nc, Dc, c, mu):
        """Calculates the Normalized Generalized Gamma PSD N(D) values.

        N_c and D_c are computed internally from the parameters.

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

        norm_nd = NormalizedGeneralizedGammaPSD.normalized_formula(
            x=x,
            i=i,
            j=j,
            c=c,
            mu=mu,
        )
        return Nc * norm_nd

    @staticmethod
    def from_parameters(parameters):
        """Initialize NormalizedGeneralizedGammaPSD from a dictionary or xarray.Dataset.

        Parameters
        ----------
        parameters : dict or xarray.Dataset
            Parameters to initialize the class. Must contain 'i', 'j', 'Nc', 'Dc', 'c', and 'mu'.
            The moment indices 'i' and 'j' can also be provided in the 'disdrodb_psd_model_kwargs'
            attribute if parameters is an xarray.Dataset.

        Returns
        -------
        NormalizedGeneralizedGammaPSD
            An instance of NormalizedGeneralizedGammaPSD initialized with the parameters.
        """
        if hasattr(parameters, "attrs") and "disdrodb_psd_model_kwargs" in parameters.attrs:
            model_kwargs = ast.literal_eval(parameters.attrs["disdrodb_psd_model_kwargs"])
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
        """Return the required parameters of the PSD.

        Returns
        -------
        list of str
            List of required parameter names.
        """
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
                    f"$N_c = {self.Nc:.2f}$\n",
                    f"$D_c = {self.Dc:.2f}$\n",
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
    """Create an interpolation function for binned data.

    Parameters
    ----------
    bin_edges : array-like
        Sorted array of n+1 bin edge values [mm].
    bin_values : array-like
        Array of n bin values corresponding to each bin.
    interp_method : str
        Interpolation method:

        - 'step_left': Piecewise constant, left-continuous
        - 'step_right': Piecewise constant, right-continuous
        - 'linear': Linear interpolation
        - 'pchip': Piecewise Cubic Hermite Interpolating Polynomial

    Returns
    -------
    callable
        A function f(D) that returns the interpolated values for diameter D.
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
    """Binned particle size distribution (PSD).

    This class represents a binned PSD that computes values through interpolation
    between discretized bin values. This approach is useful for representing
    empirically measured or discretized PSDs.

    The PSD values are computed via interpolation from discrete bin values using
    various methods. Values outside the defined bin range are set to zero, and
    all returned values are non-negative.

    Parameters
    ----------
    bin_edges : array-like
        Sequence of n+1 bin edge values defining the bins [mm].
        Must be monotonically increasing.
    bin_psd : array-like
        Sequence of n PSD values corresponding to the intervals defined by bin_edges [m^-3 mm^-1].
    interp_method : str, optional
        Interpolation method for computing PSD values between bin centers. Valid methods can be:

        - 'step_left': Use the value from the left bin (piecewise constant, left-continuous)
        - 'step_right': Use the value from the right bin (piecewise constant, right-continuous)
        - 'linear': Linear interpolation between bin centers
        - 'pchip': Piecewise Cubic Hermite Interpolating Polynomial, preserves monotonicity

        Default is 'step_left'.

    Attributes
    ----------
    bin_edges : numpy.ndarray
        Bin edge values.
    bin_psd : numpy.ndarray
        PSD values for each bin.
    interp_method : str
        Selected interpolation method.

    Notes
    -----
    - Values for diameters D outside the range (bin_edges[0], bin_edges[-1]) are set to 0
    - Interpolation is performed using bin centers computed as the midpoint of each bin
    - All PSD values are clipped to be non-negative after interpolation
    - The 'pchip' method is recommended when smoothness and monotonicity preservation are important

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
    >>> # Using step interpolation
    >>> psd_step = BinnedPSD(bin_edges, bin_psd, interp_method="step_left")
    >>> psd_values_step = psd_step(D)
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
        """Check Binned PSD equality.

        Parameters
        ----------
        other : BinnedPSD or None
            Another BinnedPSD object to compare with.

        Returns
        -------
        bool
            True if both objects have the same bin edges and PSD values, False otherwise.
        """
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
    """Compute moments of the exponential distribution.

    Parameters
    ----------
    N0 : float or array-like
        Intercept parameter [m^-3 mm^-1].
    Lambda : float or array-like
        Inverse scale parameter [mm^-1].
    moment : int or float
        Moment order.

    Returns
    -------
    float or array-like
        The computed moment value.
    """
    return N0 * gamma_f(moment + 1) / Lambda ** (moment + 1)


def get_gamma_moment_v1(N0, mu, Lambda, moment):
    """Compute moments of the gamma distribution (version 1).

    Parameters
    ----------
    N0 : float or array-like
        Intercept parameter [m^-3 mm^(-1-mu)].
    mu : float or array-like
        Shape parameter [-].
    Lambda : float or array-like
        Inverse scale parameter [mm^-1].
    moment : int or float
        Moment order.

    Returns
    -------
    float or array-like
        The computed moment value.

    References
    ----------
    Kozu, T., and K. Nakamura, 1991:
    Rainfall Parameter Estimation from Dual-Radar Measurements
    Combining Reflectivity Profile and Path-integrated Attenuation.
    J. Atmos. Oceanic Technol., 8, 259-270,
    https://doi.org/10.1175/1520-0426(1991)008<0259:RPEFDR>2.0.CO;2
    """
    # Zhang et al 2001: N0 * gamma_f(mu + moment + 1) * Lambda ** (-(mu + moment + 1))
    return N0 * gamma_f(mu + moment + 1) / Lambda ** (mu + moment + 1)


def get_gamma_moment_v2(Nt, mu, Lambda, moment):
    """Compute moments of the gamma distribution (version 2).

    Parameters
    ----------
    Nt : float or array-like
        Total concentration parameter [m^-3].
    mu : float or array-like
        Shape parameter [-].
    Lambda : float or array-like
        Inverse scale parameter [mm^-1].
    moment : int or float
        Moment order.

    Returns
    -------
    float or array-like
        The computed moment value.

    References
    ----------
    Kozu, T., and K. Nakamura, 1991:
    Rainfall Parameter Estimation from Dual-Radar Measurements
    Combining Reflectivity Profile and Path-integrated Attenuation.
    J. Atmos. Oceanic Technol., 8, 259-270, https://doi.org/10.1175/1520-0426(1991)008<0259:RPEFDR>2.0.CO;2
    """
    return Nt * gamma_f(mu + moment + 1) / gamma_f(mu + 1) / Lambda**moment


def get_lognormal_moment(Nt, sigma, mu, moment):
    """Compute moments of the lognormal distribution.

    Parameters
    ----------
    Nt : float or array-like
        Total concentration parameter [m^-3].
    sigma : float or array-like
        Scale parameter [-].
    mu : float or array-like
        Location parameter [-].
    moment : int or float
        Moment order.

    Returns
    -------
    float or array-like
        The computed moment value.

    References
    ----------
    Kozu, T., and K. Nakamura, 1991:
    Rainfall Parameter Estimation from Dual-Radar Measurements
    Combining Reflectivity Profile and Path-integrated Attenuation.
    J. Atmos. Oceanic Technol., 8, 259-270, https://doi.org/10.1175/1520-0426(1991)008<0259:RPEFDR>2.0.CO;2
    """
    return Nt * np.exp(moment * mu + 1 / 2 * moment * sigma**2)
