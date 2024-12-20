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

import numpy as np
import xarray as xr
from pytmatrix.psd import PSD
from scipy.special import gamma

# psd.log_likelihood
# psd.moment(order)
# psd.mean
# psd.variance
# psd.mode

# TODO
# - psd.isel(**kwargs)
# - psd.sel(**kwargs)

# __eq__
# --> Generalize using self.parameters and deep diff


# ------------------------------------------------------------------------------------------------------------.


def available_psd_models():
    """Return a list of available PSD models."""
    return list(PSD_MODELS_DICT)


def check_psd_model(psd_model):
    """Check validity of a PSD model."""
    available_models = available_psd_models()
    if psd_model not in available_models:
        raise ValueError(f"{psd_model} is an invalid PSD model. Valid models are: {available_models}.")
    return psd_model


def get_psd_model(psd_model):
    """Retrieve the PSD Class."""
    return PSD_MODELS_DICT[psd_model]


def create_psd(psd_model, parameters):  # TODO: check name around
    """Define a PSD from a dictionary or xr.Dataset of parameters."""
    psd_class = get_psd_model(psd_model)
    psd = psd_class.from_parameters(parameters)
    return psd


def get_required_parameters(psd_model):
    """Retrieve the list of parameters required by a PSD model."""
    psd_class = get_psd_model(psd_model)
    return psd_class.required_parameters()


def clip_values(D, values, Dmax=np.inf):
    """Clip values outside the [Dmin,Dmax) interval to 0."""
    # Handle scalar input
    if np.isscalar(D):
        if Dmax < D or D == 0.0:
            return 0.0
        return values

    # Handle numpy array input
    if isinstance(values, np.ndarray):
        mask = (Dmax < D) | (D == 0)
        values = np.where(mask, 0, values)

    # Handle xarray.DataArray input
    elif isinstance(values, xr.DataArray):
        values = xr.where(np.logical_or(Dmax < D, D == 0), 0, values)
        values = values.where(~np.isnan(values).any(dim="diameter_bin_center"))
    else:
        raise TypeError("Input 'D' and 'values' must be a scalar, numpy array or an xarray.DataArray.")
    return values


def is_scalar(value):
    """Determines if the input value is a scalar."""
    return isinstance(value, (float, int)) or isinstance(value, (np.ndarray, xr.DataArray)) and value.size == 1


class XarrayPSD(PSD):
    """PSD class template allowing vectorized computations with xarray.

    We currently inherit from pytmatrix PSD to allow scattering simulations:
    --> https://github.com/ltelab/pytmatrix-lte/blob/880170b4ca62a04e8c843619fa1b8713b9e11894/pytmatrix/psd.py#L321
    """

    def __eq__(self, other):
        """Check if two objects are equal."""
        return False

    def has_scalar_parameters(self):
        """Check if the PSD object contains only a single set of parameters."""
        return np.all(is_scalar(value) for param, value in self.parameters.items())

    def formula(self, D, **parameters):
        """PSD formula."""
        pass

    def __call__(self, D):
        """Compute the PSD."""
        values = self.formula(D=D, **self.parameters)
        return clip_values(D=D, values=values, Dmax=self.Dmax)

    def moment(self, order, nbins_diam=1024):
        """
        Compute the moments of the Particle Size Distribution (PSD).

        Parameters
        ----------
        order : int
            The order of the moment to compute.
        nbins_diam : int, optional
            The number of bins to use for the diameter range (default is 1024).

        Returns
        -------
        float
            The computed moment of the PSD.

        Notes
        -----
        The method uses numerical integration (trapezoidal rule) to compute the moment.
        """
        dbins = np.linspace(self.Dmin, self.Dmax, nbins_diam)
        dD = dbins[1] - dbins[0]
        return np.trapz(dbins**order * self.__call__(dbins), dx=dD)


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

    def __init__(self, Nt=1.0, mu=0.0, sigma=1.0, Dmin=0, Dmax=None):
        self.Nt = Nt
        self.mu = mu
        self.sigma = sigma
        self.Dmin = Dmin
        self.Dmax = mu + sigma * 4 if Dmax is None else Dmax

        self.parameters = {"Nt": Nt, "mu": self.mu, "sigma": self.sigma}

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD."""
        return ["Nt", "mu", "sigma"]

    @property
    def name(self):
        """Return name of the PSD."""
        return "LognormalPSD"

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

    @staticmethod
    def formula(D, Nt, mu, sigma):
        """Calculates the Lognormal PSD values."""
        coeff = Nt / (np.sqrt(2.0 * np.pi) * sigma * (D))
        expon = np.exp(-((np.log(D) - mu) ** 2) / (2.0 * sigma**2))
        return coeff * expon

    # def __eq__(self, other):
    #     try:
    #         return isinstance(other, ExponentialPSD) and \
    #             (self.N0 == other.N0) and (self.Lambda == other.Lambda) and \
    #             (self.Dmax == other.Dmax)
    #     except AttributeError:
    #         return False

    # params dictionary !


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
        Dmax: the maximum diameter to consider (defaults to 11/Lambda, i.e. approx. 3*D50, if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.
        Returns 0 for all diameters larger than Dmax.
    """

    def __init__(self, N0=1.0, Lambda=1.0, Dmin=0, Dmax=None):
        self.N0 = N0
        self.Lambda = Lambda
        self.Dmax = 11.0 / Lambda if Dmax is None else Dmax
        self.Dmin = Dmin
        self.parameters = {"N0": self.N0, "Lambda": self.Lambda}

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD."""
        return ["N0", "Lambda"]

    @property
    def name(self):
        """Return name of the PSD."""
        return "ExponentialPSD"

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

    @staticmethod
    def formula(D, N0, Lambda):
        """Calculates the Exponential PSD values."""
        return N0 * np.exp(-Lambda * D)

    def __eq__(self, other):
        """Check if two objects are equal."""
        try:
            return (
                isinstance(other, ExponentialPSD)
                and (self.N0 == other.N0)
                and (self.Lambda == other.Lambda)
                and (self.Dmax == other.Dmax)
            )
        except AttributeError:
            return False


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
        Dmax: the maximum diameter to consider (defaults to 11/Lambda,
            i.e. approx. 3*D50, if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.
        Returns 0 for all diameters larger than Dmax.

    References
    ----------
    Ulbrich, C. W., 1985: The Effects of Drop Size Distribution Truncation on
    Rainfall Integral Parameters and Empirical Relations.
    J. Appl. Meteor. Climatol., 24, 580-590, https://doi.org/10.1175/1520-0450(1985)024<0580:TEODSD>2.0.CO;2
    """

    def __init__(self, N0=1.0, mu=0.0, Lambda=1.0, Dmin=0, Dmax=None):
        super().__init__(N0=N0, Lambda=Lambda, Dmin=Dmin, Dmax=Dmax)
        self.mu = mu
        self.parameters = {"N0": self.N0, "mu": mu, "Lambda": self.Lambda}

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD."""
        return ["N0", "mu", "Lambda"]

    @property
    def name(self):
        """Return name of the PSD."""
        return "GammaPSD"

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

    @staticmethod
    def formula(D, N0, Lambda, mu):
        """Calculates the Gamma PSD values."""
        return N0 * np.exp(mu * np.log(D) - Lambda * D)

    def __eq__(self, other):
        """Check if two objects are equal."""
        try:
            return super().__eq__(other) and self.mu == other.mu
        except AttributeError:
            return False


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
        Dmax: the maximum diameter to consider (defaults to 3*D50 when
            if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.
        Returns 0 for all diameters larger than Dmax.

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

    def __init__(self, Nw=1.0, D50=1.0, mu=0.0, Dmin=0, Dmax=None):
        self.D50 = D50
        self.mu = mu
        self.Dmin = Dmin
        self.Dmax = 3.0 * D50 if Dmax is None else Dmax
        self.Nw = Nw
        self.parameters = {"Nw": Nw, "D50": D50, "mu": mu}

    @staticmethod
    def required_parameters():
        """Return the required parameters of the PSD."""
        return ["Nw", "D50", "mu"]

    @property
    def name(self):
        """Return the PSD name."""
        return "NormalizedGammaPSD"

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
    def formula(D, Nw, D50, mu):
        """Calculates the NormalizedGamma PSD values."""
        d_ratio = D / D50
        nf = Nw * 6.0 / 3.67**4 * (3.67 + mu) ** (mu + 4) / gamma(mu + 4)
        return nf * np.exp(mu * np.log(d_ratio) - (3.67 + mu) * d_ratio)

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

    def __eq__(self, other):
        """Check if two objects are equal."""
        try:
            return (
                isinstance(other, NormalizedGammaPSD)
                and (self.D50 == other.D50)
                and (self.Nw == other.Nw)
                and (self.mu == other.mu)
                and (self.Dmax == other.Dmax)
            )
        except AttributeError:
            return False


PSD_MODELS_DICT = {
    "LognormalPSD": LognormalPSD,
    "ExponentialPSD": ExponentialPSD,
    "GammaPSD": GammaPSD,
    "NormalizedGammaPSD": NormalizedGammaPSD,
}


class BinnedPSD(PSD):
    """Binned gamma particle size distribution (PSD).

    Callable class to provide a binned PSD with the given bin edges and PSD
    values.

    Args (constructor):
        The first argument to the constructor should specify n+1 bin edges,
        and the second should specify n bin_psd values.

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.
        Returns 0 for all diameters outside the bins.
    """

    def __init__(self, bin_edges, bin_psd):
        if len(bin_edges) != len(bin_psd) + 1:
            raise ValueError("There must be n+1 bin edges for n bins.")

        self.bin_edges = bin_edges
        self.bin_psd = bin_psd

    def psd_for_D(self, D):
        """
        Calculate the particle size distribution (PSD) for a given diameter D.

        Parameters
        ----------
        D : float
            The diameter for which to calculate the PSD.

        Returns
        -------
        float
            The PSD value corresponding to the given diameter D. Returns 0.0 if D is outside the range of bin edges.

        Notes
        -----
        This method uses a binary search algorithm to find the appropriate bin for the given diameter D.
        """
        if not (self.bin_edges[0] < D <= self.bin_edges[-1]):
            return 0.0

        # binary search for the right bin
        start = 0
        end = len(self.bin_edges)
        while end - start > 1:
            half = (start + end) // 2
            if self.bin_edges[start] < D <= self.bin_edges[half]:
                end = half
            else:
                start = half

        return self.bin_psd[start]

    def __call__(self, D):
        """Compute the PSD."""
        if np.shape(D) == ():  # D is a scalar
            return self.psd_for_D(D)
        return np.array([self.psd_for_D(d) for d in D])

    def __eq__(self, other):
        """Check PSD equality."""
        if other is None:
            return False
        return (
            len(self.bin_edges) == len(other.bin_edges)
            and (self.bin_edges == other.bin_edges).all()
            and (self.bin_psd == other.bin_psd).all()
        )


####-----------------------------------------------------------------.
#### Moments Computation


def get_exponential_moment(N0, Lambda, moment):
    """Compute exponential distribution moments."""
    return N0 * gamma(moment + 1) / Lambda ** (moment + 1)


def get_gamma_moment_v1(N0, mu, Lambda, moment):
    """Compute gamma distribution moments.

    References
    ----------
    Kozu, T., and K. Nakamura, 1991:
    Rainfall Parameter Estimation from Dual-Radar Measurements
    Combining Reflectivity Profile and Path-integrated Attenuation.
    J. Atmos. Oceanic Technol., 8, 259-270, https://doi.org/10.1175/1520-0426(1991)008<0259:RPEFDR>2.0.CO;2
    """
    # Zhang et al 2001: N0 * gamma(mu + moment + 1) * Lambda ** (-(mu + moment + 1))
    return N0 * gamma(mu + moment + 1) / Lambda ** (mu + moment + 1)


def get_gamma_moment_v2(Nt, mu, Lambda, moment):
    """Compute gamma distribution moments.

    References
    ----------
    Kozu, T., and K. Nakamura, 1991:
    Rainfall Parameter Estimation from Dual-Radar Measurements
    Combining Reflectivity Profile and Path-integrated Attenuation.
    J. Atmos. Oceanic Technol., 8, 259-270, https://doi.org/10.1175/1520-0426(1991)008<0259:RPEFDR>2.0.CO;2
    """
    return Nt * gamma(mu + moment + 1) / gamma(mu + 1) / Lambda**moment


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
