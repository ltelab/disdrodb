# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
#
# temperaturehis program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# temperaturehis program is distributed in the hope that it will be useful,
# but WItemperatureHOUtemperature ANY WARRANtemperatureY; without even the implied warranty of
# MERCHANtemperatureABILItemperatureY or FItemperatureNESS FOR A PARtemperatureICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""Implement particle permittivity models."""

import numpy as np
import xarray as xr

from disdrodb.utils.warnings import suppress_warnings

# Definitions
# - Complex_refractive_index: m
# - Complex dielectric constant = complex relative permittivity: eps
# - Rayleigh dielectric factor: Kw_sqr

# Other useful codes for ice/snow future extension:
# - pytmatrix: https://github.com/ltelab/pytmatrix-lte/blob/main/pytmatrix/refractive.py#L66
# - pyradsim: https://github.com/wolfidan/pyradsim/blob/master/pyradsim/permittivity_models.py
# - cosmo_pol: https://github.com/wolfidan/cosmo_pol/blob/master/cosmo_pol/hydrometeors/dielectric.py#L49
# - m_func for snow, melting: https://github.com/wolfidan/cosmo_pol/blob/master/cosmo_pol/hydrometeors/hydrometeors.py#L544

####-------------------------------------------------------------------------------------.
#### Wrappers


def available_permittivity_models():
    """Return a list of the available raindrops complex refractive index models."""
    return list(REFRACTIVE_INDEX_MODELS)


def get_refractive_index_function(permittivity_model):
    """Return the specified model estimating the complex refractive index of rain drops.

    The complex refractive index of a hydrometeor (e.g., water droplet, ice particle, graupel)
    governs how radar waves interact with it.
    The real part determines how much the radar wave slows down inside the particle (phase shift)
    The imaginary part determines how much the radar wave is absorbed and attenuated by the particle
    The imaginary part thus describes how much energy is lost as the wave travels through the particle.
    The imaginary part thus controls radar attenuation and depolarization effects.
    A large imaginary part leads to weaker returned signals, especially at shorter wavelengths.

    The square root of the complex refractive index corresponds to the complex relative permittivity,
    also known as the complex dielectric constant.

    Parameters
    ----------
    model : str
        The model to use for calculating the complex refractive index. Available models are:
        'Liebe1991', 'Liebe1991v2', 'Ellison2007', 'Turner2016', 'Turner2016SLC'.

    Returns
    -------
    callable
        A function which compute the complex refractive index for given temperature and frequency.

    Notes
    -----
    This function serves as a wrapper to various complex refractive index models for raindrops.
    It returns the appropriate model based on the `model` parameter.

    """
    permittivity_model = check_permittivity_model(permittivity_model)
    return REFRACTIVE_INDEX_MODELS[permittivity_model]


def check_permittivity_model(permittivity_model):
    """Check validity of the specified complex refractive index model."""
    available_models = available_permittivity_models()
    if permittivity_model not in available_models:
        raise ValueError(f"{permittivity_model} is an invalid permittivity model. Valid models: {available_models}.")
    return permittivity_model


def get_refractive_index(temperature, frequency, permittivity_model):
    """
    Compute the complex refractive index of raindrops using the specified permittivity model.

    The complex refractive index of a hydrometeor (e.g., water droplet, ice particle, graupel)
    governs how radar waves interact with it.
    The real part determines how much the radar wave slows down inside the particle (phase shift)
    The imaginary part determines how much the radar wave is absorbed and attenuated by the particle
    The imaginary part thus describes how much energy is lost as the wave travels through the particle.
    The imaginary part thus controls radar attenuation and depolarization effects.
    A large imaginary part leads to weaker returned signals, especially at shorter wavelengths.

    The square root of the complex refractive index corresponds to the complex relative permittivity,
    also known as the complex dielectric constant.

    Parameters
    ----------
    temperature : array-like
        Temperature in degree Celsius.
    frequency: float
        Frequency in GHz.
    permittivity_model : str
        The permittivity model to use for calculating the complex refractive index.
        Available models are: 'Liebe1991', 'Liebe1991v2', 'Ellison2007', 'Turner2016', 'Turner2016SLC'.
        See available models with ``disdrodb.scattering.available_permittivity_models()``.

    Returns
    -------
    m : array-like
        Complex refractive index of raindrop at given temperature and frequency.

    Notes
    -----
    This function serves as a wrapper to various permittivity models for raindrops.
    It selects and applies the appropriate model based on the `permittivity_model` parameter.

    Examples
    --------
    >>> temperature = np.array([0.5, 1.0, 2.0, 3.0])
    >>> frequency = 5.6  # GhZ  (C band)
    >>> m = get_refractive_index(temperature=temperature, frequency=frequency, permittivity_model="Liebe1991")

    """
    # Ensure input is numpy array or xr.DataArray
    frequency = ensure_array(frequency)
    temperature = ensure_array(temperature)

    # If both inputs are numpy (or dask) arrays with size > 1 → raise error
    if (
        not isinstance(temperature, xr.DataArray)
        and not isinstance(frequency, xr.DataArray)
        and np.size(temperature) > 1
        and np.size(frequency) > 1
    ):
        raise ValueError(
            "get_refractive_index does not support broadcasting plain numpy/dask arrays "
            "when both `temperature` and `frequency` have size > 1. "
            "Please provide both input as xarray.DataArray objects "
            "with different dimensions to enable labeled broadcasting.",
        )

    # Retrieve refractive_index function
    func = get_refractive_index_function(permittivity_model)

    # Retrieve refractive_index
    refractive_index = func(temperature=temperature, frequency=frequency)

    # Add attributes
    if isinstance(refractive_index, xr.DataArray):
        refractive_index.name = "refractive_index"
        refractive_index.attrs["units"] = ""
        refractive_index.attrs["model"] = permittivity_model
    return refractive_index


####----------------------------------------------------------------------------------------
#### Liquid Water Refractive Index Models


def ensure_array(arr):
    """Ensure data to be a numpy array or xarray DataArray."""
    if isinstance(arr, xr.DataArray):
        return arr
    return np.asanyarray(arr)


def check_temperature_validity_range(temperature, vmin, vmax, permittivity_model):
    """Check temperature validity range."""
    if np.logical_or(temperature < vmin, temperature > vmax).any():
        raise ValueError(
            f"The {permittivity_model} refractive index model is only valid between {vmin} and {vmax} degree Celsius.",
        )
    return temperature


def check_frequency_validity_range(frequency, vmin, vmax, permittivity_model):
    """Check frequency validity range."""
    if np.logical_or(frequency < vmin, frequency > vmax).any():
        raise ValueError(
            f"The {permittivity_model} refractive index model is only valid between {vmin} and {vmax} GHz.",
        )
    return frequency


def get_rain_refractive_index_liebe1991_single(temperature, frequency):
    """Compute the complex refractive index according to the single Debye model of Liebe et al. (1991).

    Parameters
    ----------
    temperature : array-like
        Temperature in degree Celsius.
    frequency : array-like
        Frequency in GHz.

    Returns
    -------
    m : array-like
        Complex refractive index at requested temperature and frequency.

    Notes
    -----
    The code of this function has been derived from RainSense code of Thomas van Leth available at
    https://github.com/temperatureCvanLeth/RainSense/blob/master/rainsense/scattering.py#L149

    References
    ----------
    H. J. Liebe, G. A. Hufford, and T. Manabe (1991).
    A model for the complex permittivity of water at frequencies below 1 THz.
    Journal of Atmospheric and Oceanic Technology, 27(2), 333-344.
    Int. J. Infrared Millim. Waves, 12(7), 659-675.
    https://doi.org/10.1007/BF01008897
    """
    # Ensure input is numpy array or xr.DataArray
    frequency = ensure_array(frequency)
    temperature = ensure_array(temperature)

    # Check frequency and temperature within validity range
    temperature = check_temperature_validity_range(temperature, vmin=0, vmax=100, permittivity_model="Liebe1991single")
    frequency = check_frequency_validity_range(frequency, vmin=0, vmax=100, permittivity_model="Liebe1991single")

    # Conversion of temperature to Kelvin
    temperature = temperature + 273.15

    # Compute static dielectric constant (eq. 1)
    theta = 1 - 300 / temperature
    eps_0 = 77.66 - 103.3 * theta

    # Compute the complex dielectric constant (eq. 2)
    eps_1 = 0.066 * eps_0
    gamma_D = 20.27 + 146.5 * theta + 314 * theta**2
    eps = (eps_0 - eps_1) / (1 - 1j * frequency / gamma_D) + eps_1

    # Compute the refractive index
    m = np.sqrt(eps)
    return m


def get_rain_refractive_index_liebe1991(temperature, frequency):
    """Compute the complex refractive index according to the double Debye model of Liebe et al. (1991).

    Parameters
    ----------
    temperature : array-like
        Temperature in degree Celsius.
    frequency : array-like
        Frequency in GHz.

    Returns
    -------
    m : array-like
        Complex refractive index at requested temperature and frequency.

    Notes
    -----
    The code of this function has been derived from pyradsim code of Daniel Wolfensberger available at
    https://github.com/wolfidan/pyradsim/blob/master/pyradsim/permittivity_models.py#L37
    The Liebe et al. (1991) replaces the work of Ray et al. (1972).

    References
    ----------
    H. J. Liebe, G. A. Hufford, and T. Manabe (1991).
    A model for the complex permittivity of water at frequencies below 1 THz.
    Journal of Atmospheric and Oceanic Technology, 27(2), 333-344.
    Int. J. Infrared Millim. Waves, 12(7), 659-675.
    https://doi.org/10.1007/BF01008897

    Peter S. Ray (1972).
    Broadband Complex Refractive Indices of Ice and Water.
    Applied Optics, 11(8), 1836-1844.
    https://doi.org/10.1364/AO.11.001836
    """
    # Ensure input is numpy array or xr.DataArray
    frequency = ensure_array(frequency)
    temperature = ensure_array(temperature)

    # Check frequency and temperature within validity range
    temperature = check_temperature_validity_range(temperature, vmin=0, vmax=40, permittivity_model="Liebe1991")
    frequency = check_frequency_validity_range(frequency, vmin=0, vmax=1000, permittivity_model="Liebe1991")

    # Conversion of temperature to Kelvin
    temperature = temperature + 273.15

    # Compute static dielectric constant (eq. 1)
    theta = 1 - 300 / temperature
    eps_0 = 77.66 - 103.3 * theta

    # Compute the complex dielectric constant (e4, eq5)
    eps_1 = 0.0671 * eps_0
    eps_2 = 3.52 + 7.52 * theta
    gamma_1 = 20.20 + 146.5 * theta + 316 * theta**2
    gamma_2 = 39.8 * gamma_1

    term1 = eps_0 - eps_1
    term2 = 1 + (frequency / gamma_1) ** 2
    term3 = 1 + (frequency / gamma_2) ** 2
    term4 = eps_1 - eps_2
    term5 = eps_2

    eps_real = term1 / term2 + term4 / term3 + term5
    eps_imag = (term1 / term2) * (frequency / gamma_1) + (term4 / term3) * (frequency / gamma_2)

    eps = eps_real + 1j * eps_imag

    # Compute the refractive index
    m = np.sqrt(eps)
    return m


def get_rain_refractive_index_ellison2007(temperature, frequency):
    """Compute the complex refractive index according to Ellison (2005) model.

    Parameters
    ----------
    temperature : array-like
       Temperature in degree Celsius.
    frequency : array-like
       Frequency in GHz.

    Returns
    -------
    m : array-like
       Complex refractive index at requested temperature and frequency.

    Notes
    -----
    The model is designed to operate only up to 1000 GHz and temperature ranging from 0 degC to 100 degC.
    The code of this function has been derived from Davide Ori raincoat code available at
    https://github.com/OPTIMICe-team/raincoat/blob/master/raincoat/scatTable/water.py#L160

    References
    ----------
    W. J. Ellison (2007).
    Permittivity of Pure Water, at Standard Atmospheric Pressure, over the
    Frequency Range 0-25 THz and the Temperature Range 0-100 °C.
    J. Phys. Chem. Ref. Data, 36, 1-18.
    https://doi.org/10.1063/1.2360986
    """
    # Ensure input is numpy array or xr.DataArray
    frequency = ensure_array(frequency)
    temperature = ensure_array(temperature)

    # Check frequency and temperature within validity range
    temperature = check_temperature_validity_range(temperature, vmin=0, vmax=100, permittivity_model="Ellison2007")
    frequency = check_frequency_validity_range(frequency, vmin=0, vmax=1000, permittivity_model="Ellison2007")

    # Conversion of frequency to Hz
    frequency = frequency / 1e-9

    # Here below we assume temperature in Celsius, frequency in Hz
    T = temperature

    # Compute the complex dielectric constant
    a0 = 5.7230
    a1 = 2.2379e-2
    a2 = -7.1237e-4
    a3 = 5.0478
    a4 = -7.0315e-2
    a5 = 6.0059e-4
    a6 = 3.6143
    a7 = 2.8841e-2
    a8 = 1.3652e-1
    a9 = 1.4825e-3
    a10 = 2.4166e-4

    es = (37088.6 - 82.168 * T) / (421.854 + T)
    einf = a6 + a7 * T
    e1 = a0 + T * (a1 + T * a2)  # a0+a1*T+a2*T*T
    ni1 = (45.0 + T) / (a3 + T * (a4 + T * a5))  # (a3+a4*T+a5*T*T)
    ni2 = (45.0 + T) / (a8 + T * (a9 + T * a10))  # (a8+a9*T+a10*T*T)
    A1 = frequency * 1.0e-9 / ni1
    A2 = frequency * 1.0e-9 / ni2

    eps_real = (es - e1) / (1 + A1 * A1) + (e1 - einf) / (1 + A2 * A2) + einf
    eps_imag = (es * A1 - e1 * A1) / (1 + A1 * A1) + (e1 * A2 - einf * A2) / (1 + A2 * A2)

    eps = eps_real + 1j * eps_imag

    # Compute the refractive index
    with suppress_warnings():
        m = np.sqrt(eps)  # on windows sqrt of eps=np.complex128(nan+nanj) raise warning
    return m


def get_rain_refractive_index_turner2016(frequency, temperature):
    """Compute the complex refractive index using the Turner-Kneifel-Cadeddu (TKC) model.

    The TKC supercooled liquid water absorption model was built using both laboratory observations
    (primarily at warm temperature) and field data observed by MWRs at multiple frequency at
    supercool temperature. The field data were published in Kneifel et al. (2014).

    The strength of the TKC model is the use of an optimal estimation framework to
    determine the empirical coefficients of the double-Debye model.
    A full description of this model is given in Turner et al. (2016).

    Parameters
    ----------
    temperature : array-like
        Temperature in degree Celsius.
    frequency : array-like
        Frequency in GHz.

    Returns
    -------
    m : array-like
        Complex refractive index at given temperature and frequency.

    Notes
    -----
    The code of this function has been checked against Joseph Hardin pyDSD and Davide Ori raincoat codes
    available at:
    https://github.com/josephhardinee/PyDSD/blob/main/pydsd/utility/dielectric.py#L36
    https://github.com/OPTIMICe-team/raincoat/blob/master/raincoat/scatTable/water.py#L54

    References
    ----------
    Turner, D.D., S. Kneifel, and M.P. Cadeddu (2016).
    An improved liquid water absorption model in the microwave for supercooled liquid clouds.
    J. Atmos. Oceanic Technol., 33(1), 33-44.
    https://doi.org/10.1175/JTECH-D-15-0074.1.

    Kneifel, S., S. Redl, E. Orlandi, U. Löhnert, M. P. Cadeddu, D. D. Turner, and M. Chen (2014).
    Absorption Properties of Supercooled Liquid Water between 31 and 225 GHz:
    Evaluation of Absorption Models Using Ground-Based Observations.
    J. Appl. Meteor. Climatol., 53, 1028-1045.
    https://doi.org/10.1175/JAMC-D-13-0214.1
    """
    # Ensure input is numpy array or xr.DataArray
    frequency = ensure_array(frequency)
    temperature = ensure_array(temperature)

    # Check frequency and temperature within validity range
    temperature = check_temperature_validity_range(temperature, vmin=-40, vmax=50, permittivity_model="Turner2016")
    frequency = check_frequency_validity_range(frequency, vmin=0.5, vmax=500, permittivity_model="Turner2016")

    # Conversion of frequency to Hz
    frequency = frequency / 1e-9

    # Define coefficients
    a = [8.111e01, 2.025]
    b = [4.434e-3, 1.073e-2]
    c = [1.302e-13, 1.012e-14]
    d = [6.627e02, 6.089e02]
    tc = 1.342e2
    s = [8.79144e1, -4.04399e-1, 9.58726e-4, -1.32802e-6]

    def A_i(i, temperature, frequency):
        """Compute the relaxation terms A_i (Eq 7) of the double Debye model."""
        delta = a[i] * np.exp(-1 * b[i] * temperature)  # (Eq 9)
        tau = c[i] * np.exp(d[i] / (temperature + tc))  # (Eq 10)

        return (tau**2 * delta) / (1 + (2 * np.pi * frequency * tau) ** 2)  # (Eq 7)

    def B_i(i, temperature, frequency):
        """Compute the relaxation terms B_i (Eq 7) of the double Debye model."""
        delta = a[i] * np.exp(-1 * b[i] * temperature)  # (Eq 9)
        tau = c[i] * np.exp(d[i] / (temperature + tc))  # (Eq 10)

        return (tau * delta) / (1 + (2 * np.pi * frequency * tau) ** 2)  # (Eq 8)

    # Compute the static dielectric permittivity (Eq 6)
    es = s[0] + s[1] * temperature + s[2] * temperature**2 + s[3] * temperature**3

    # Compute the complex dielectric constant
    eps_real = es - (2 * np.pi * frequency) ** 2 * (
        A_i(0, temperature, frequency) + A_i(1, temperature, frequency)
    )  # (Eq 4)
    eps_imag = 2 * np.pi * frequency * (B_i(0, temperature, frequency) + B_i(1, temperature, frequency))  # (Eq 5)

    eps = eps_real + 1j * eps_imag

    # Compute the refractive index
    m = np.sqrt(eps)
    return m


####----------------------------------------------------------------------------------------
def get_rayleigh_dielectric_factor(m):
    r"""Compute the Rayleigh dielectric factor :math:`|K|^2` from the complex refractive index.

    The magnitude squared of the complex dielectric constant factor for liquid water,
    relative to the surrounding medium (typically air).

    This factor is used to compute the radar reflectivity.

    Parameters
    ----------
    m : complex
        Complex refractive index.

    Returns
    -------
    float
        Dielectric factor :math:`|K|^2` used in Rayleigh scattering.
        Often also called the radar dieletric factor.
        In pytmatrix, correspond to the ``Kw_sqr`` argument of the Scatterer object.
    """
    eps = m**2
    K_complex = (eps - 1.0) / (eps + 2.0)
    return np.abs(K_complex) ** 2


####-------------------------------------------------------------------------------------.
REFRACTIVE_INDEX_MODELS = {
    "Liebe1991": get_rain_refractive_index_liebe1991,
    "Liebe1991single": get_rain_refractive_index_liebe1991_single,
    "Ellison2007": get_rain_refractive_index_ellison2007,
    "Turner2016": get_rain_refractive_index_turner2016,
}
