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
"""Theoretical models to estimate the graupel fall velocity based on particle maximum diameter in mm."""

import numpy as np
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION
from disdrodb.l0.l0b_processing import ensure_valid_geolocation
from disdrodb.l1_env.routines import load_env_dataset
from disdrodb.physics.wrappers import retrieve_air_density, retrieve_air_dynamic_viscosity, retrieve_air_pressure
from disdrodb.utils.warnings import suppress_warnings


def get_graupel_density(diameter):
    """Estimate the graupel particle density.

    Use the Heymsfield & Wright (2014) parameterization.

    At 5mm, graupel density only 140 kg/m3 ...

    Parameters
    ----------
    diameter : array-like or float
        Particle maximum diameter in millimeters [mm].

    Returns
    -------
    graupel_density : xarray.DataArray or numpy.ndarray
        Graupel density [kg/m3].

    References
    ----------
    Heymsfield, A. J., and Wright, R., 2014.
    Graupel and Hail Terminal Velocities: Does a Supercritical Reynolds Number Apply?
    J. Atmos. Sci., 71, 3392-3403, https://doi.org/10.1175/JAS-D-14-0034.1.
    """
    graupel_density = 0.18 * (diameter * 0.1) ** 0.33 * 1000
    return graupel_density


def get_fall_velocity_lee_2015(diameter):
    """
    Compute terminal fall velocity of lump graupel.

    Use Lee et al., 2015 empirical formula.

    Parameters
    ----------
    diameter : array-like or float
        Particle maximum diameter in millimeters [mm].

    Returns
    -------
    fall_velocity : numpy.ndarray or xarray.DataArray
        Terminal fall velocity [m s⁻¹].
    """
    fall_velocity = 1.10 * diameter**0.28
    return fall_velocity


def get_fall_velocity_locatelli_1974_lump(diameter):
    """
    Compute terminal fall velocity of lump graupel.

    Use Locatelli and Hobbs 1974 empirical formula.

    Parameters
    ----------
    diameter : array-like or float
        Particle maximum diameter in millimeters [mm].

    Returns
    -------
    fall_velocity : numpy.ndarray or xarray.DataArray
        Terminal fall velocity [m s⁻¹].

    Reference
    ---------
    Locatelli, J. D., and P. V. Hobbs (1974).
    Fall speeds and masses of solid precipitation particles
    J. Geophys. Res., 79(15), 2185-2197, doi:10.1029/JC079i015p02185.
    """
    fall_velocity = 1.3 * diameter**0.66  # Dmax [0.5-3]
    # fall_velocity = 1.16*diameter**0.46,  # Dmax [0.5-2]
    # fall_velocity = 1.5*diameter**0.37    # Dmax [0.5-1]
    return fall_velocity


def get_fall_velocity_locatelli_1974_conical(diameter):
    """
    Compute terminal fall velocity of cone-shaped graupel.

    Use Locatelli and Hobbs 1974 empirical formula.

    Parameters
    ----------
    diameter : array-like or float
        Particle maximum diameter in millimeters [mm].

    Returns
    -------
    fall_velocity : numpy.ndarray or xarray.DataArray
        Terminal fall velocity [m s⁻¹].

    Reference
    ---------
    Locatelli, J. D., and P. V. Hobbs (1974).
    Fall speeds and masses of solid precipitation particles
    J. Geophys. Res., 79(15), 2185-2197, doi:10.1029/JC079i015p02185.
    """
    fall_velocity = 1.20 * diameter**0.65
    return fall_velocity


def get_fall_velocity_locatelli_1974_hexagonal(diameter):
    """
    Compute terminal fall velocity of hexagonal graupel.

    Use Locatelli and Hobbs 1974 empirical formula.

    Parameters
    ----------
    diameter : array-like or float
        Particle maximum diameter in millimeters [mm].

    Returns
    -------
    fall_velocity : numpy.ndarray or xarray.DataArray
        Terminal fall velocity [m s⁻¹].

    Reference
    ---------
    Locatelli, J. D., and P. V. Hobbs (1974).
    Fall speeds and masses of solid precipitation particles
    J. Geophys. Res., 79(15), 2185-2197, doi:10.1029/JC079i015p02185.
    """
    fall_velocity = 1.10 * diameter**0.57
    return fall_velocity


def get_fall_velocity_heymsfield_2014(diameter):
    """
    Compute terminal velocity of graupel and small hail.

    Use Heymsfield & Wright (2014) empirical formula.

    Parameters
    ----------
    diameter : array-like or float
        Particle maximum diameter in millimeters [mm].

    Returns
    -------
    fall_velocity : numpy.ndarray or xarray.DataArray
        Terminal fall velocity [m s⁻¹].
    Reference
    ---------
    Heymsfield, A. J., and Wright, R., 2014.
    Graupel and hail terminal velocities: Observations and theory.
    Journal of the Atmospheric Sciences* , 71(1), 339-353.
    """
    fall_velocity = 4.88 * (0.1 * diameter) ** 0.84
    return fall_velocity


def get_fall_velocity_heymsfield_2018(diameter):
    """Get graupel fall velocity from Heymsfield et al., 2018.

    Use the Heymsfield et al., 2018 parameterization.

    Parameters
    ----------
    diameter : array-like or float
        Particle maximum diameter in millimeters [mm].


    Returns
    -------
    fall_velocity : xarray.DataArray or numpy.ndarray
        Terminal fall velocity [m s⁻¹].

    References
    ----------
    Heymsfield, A., M. Szakáll, A. Jost, I. Giammanco, and R. Wright, 2018.
    A Comprehensive Observational Study of Graupel and Hail Terminal Velocity, Mass Flux, and Kinetic Energy.
    J. Atmos. Sci., 75, 3861-3885, https://doi.org/10.1175/JAS-D-18-0035.1.

    Heymsfield, A., M. Szakáll, A. Jost, I. Giammanco, R. Wright, and J. Brimelow, 2020.
    CORRIGENDUM.
    J. Atmos. Sci., 77, 405-412, https://doi.org/10.1175/JAS-D-19-0185.1.
    """
    # Original incorrect formula from Heymsfield et al., 2018
    # fall_velocity = 6.35 * (0.1 * diameter) ** 0.87  # 0.87 in Table 3. 0.97 in Eq 8

    # Corrected formula from Heymsfield et al., 2020 (Corrigendum)
    fall_velocity = 7.59 * (0.1 * diameter) ** 0.89
    return fall_velocity


####----------------------------------------------------------------------
#### Heymsfield_2014 graupel model


def get_graupel_heymsfield_2014_fall_velocity(
    diameter,
    graupel_density=500.0,
    air_density=1.225,
    air_dynamic_viscosity=1.81e-5,
    g=9.81,
):
    r"""
    Compute the terminal fall velocity of sleet, graupel and small hail particles.

    Use the Heymsfield & Wright (2014) parameterization.

    Parameters
    ----------
    diameter : array-like or float
        Particle maximum diameter in millimeters [mm].
    air_density : float, optional
        Air density [kg m⁻³]. Default is 1.225 (typical at sea level, 15°C).
    graupel_density : float, optional
        Bulk density of the ice particle [kg m⁻³].
        Defaults to 500 kg m⁻³ (typical for graupel; hail can be ~900).
    air_dynamic_viscosity : float, optional
        Dynamic viscosity of air [kg m⁻¹ s⁻¹].
        Default is 1.81e-5 (air at 15°C, 1 atm).
    g : float, optional
        Acceleration due to gravity [m s⁻²]. Default is 9.81.

    Returns
    -------
    fall_velocity : xarray.DataArray
        Terminal fall velocity [m s⁻¹].

    Notes
    -----
    The relationship is based on empirical fits to the Reynolds number (Re)
    as a function of a dimensionless variable X:

    .. math::

        X = \\frac{4}{3} D^3 \\rho_b g \\frac{\\rho_a}{\\eta_a^2}

    Two regimes are used to compute the Reynolds number :math:`Re` as a function of
    the dimensionless parameter :math:`X`:

    .. math::

        Re =
        \\begin{cases}
            0.106\\, X^{0.693}, & \\text{for } X < 6.77 \\times 10^{4} \\\\
            0.55\\,  X^{0.545}, & \\text{for } X \\ge 6.77 \\times 10^{4}
        \\end{cases}

    The terminal fall velocity :math:`v_t` is then obtained as:

    .. math::

        v_t = \\frac{Re\\, \\eta_a}{\\rho_a\\, D}

    where:


    - :math:`Re` — Reynolds number (dimensionless)
    - :math:`\\eta_a` — dynamic viscosity of air [kg m⁻¹ s⁻¹]
    - :math:`\\rho_a` — air density [kg m⁻³]
    - :math:`D` — particle diameter [m]
    - :math:`\\rho_b` — bulk density of the particle [kg m⁻³]
    - :math:`g` — gravitational acceleration [m s⁻²]

    References
    ----------
    Heymsfield, A. J., and Wright, R., 2014.
    Graupel and Hail Terminal Velocities: Does a Supercritical Reynolds Number Apply?
    J. Atmos. Sci., 71, 3392-3403, https://doi.org/10.1175/JAS-D-14-0034.1.
    """
    diameter = xr.DataArray(diameter / 1000)

    # Compute Davies (or Best) number
    X = (4 / 3) * diameter**3 * graupel_density * g * air_density / air_dynamic_viscosity**2

    Re = xr.where(
        X < 6.77e4,
        0.106 * X**0.693,
        0.55 * X**0.545,
    )

    fall_velocity = Re * air_dynamic_viscosity / (air_density * diameter)
    return fall_velocity


def retrieve_graupel_heymsfield2014_fall_velocity(
    diameter,
    ds_env,
    graupel_density=500.0,
):
    """
    Compute the terminal fall velocity of sleet, graupel and small hail particles.

    Use the Heymsfield & Wright (2014) parameterization.

    Parameters
    ----------
    diameter : array-like
        Diameter of the graupel particles in millimeters.
    ds_env : xarray.Dataset
        A dataset containing the following environmental variables:
        - 'altitude' :  Altitude in meters (m).
        - 'latitude' :  Latitude in degrees.
        - 'temperature' : Temperature in degrees Kelvin (K).
        - 'relative_humidity' :  Relative humidity between 0 and 1.
        - 'sea_level_air_pressure' : Standard atmospheric pressure at sea level in Pascals (Pa).
        The default is 101_325 Pa.
        - 'air_pressure': Air pressure in Pascals (Pa). If None, air_pressure at altitude is inferred.
        - 'lapse_rate' : Atmospheric lapse rate in degrees Celsius or Kelvin per meter (°C/m).
        The default is 0.0065 K/m.
        - 'gas_constant_dry_air': Gas constant for dry air in J/(kg*K). The default is 287.04 is J/(kg*K).
    graupel_density : float, optional
        Bulk density of the ice particle [kg m⁻³].
        Defaults to 500 kg m⁻³ (typical for graupel; hail can be ~900).

    Returns
    -------
    fall_velocity : array-like
        Terminal fall velocity for the graupel particles.
    """
    air_viscosity = retrieve_air_dynamic_viscosity(ds_env)
    air_density = retrieve_air_density(ds_env)
    fall_velocity = get_graupel_heymsfield_2014_fall_velocity(
        diameter=diameter,
        graupel_density=graupel_density,
        air_density=air_density,
        air_dynamic_viscosity=air_viscosity,
    )
    return fall_velocity


####------------------------------------------------------------------------------------
#### Wrappers


GRAUPEL_FALL_VELOCITY_MODELS = {
    "Lee2015": get_fall_velocity_lee_2015,
    "Locatelli1974Lump": get_fall_velocity_locatelli_1974_lump,
    "Locatelli1974Conical": get_fall_velocity_locatelli_1974_conical,
    "Locatelli1974Hexagonal": get_fall_velocity_locatelli_1974_hexagonal,
    "Heymsfield2014": get_fall_velocity_heymsfield_2014,
    "Heymsfield2018": get_fall_velocity_heymsfield_2018,
}


def available_graupel_fall_velocity_models():
    """Return a list of the available graupel fall velocity models."""
    return list(GRAUPEL_FALL_VELOCITY_MODELS)


def check_graupel_fall_velocity_model(model):
    """Check validity of the specified graupel fall velocity model."""
    available_models = available_graupel_fall_velocity_models()
    if model not in available_models:
        raise ValueError(f"{model} is an invalid graupel fall velocity model. Valid models: {available_models}.")
    return model


def get_graupel_fall_velocity_model(model):
    """Return the specified graupel fall velocity model.

    Parameters
    ----------
    model : str
        The model to use for calculating the rain drop fall velocity. Available models are:
       'Lee2015', 'Locatelli1974Lump', 'Locatelli1974Conical', 'Locatelli1974Hexagonal',
       'Heymsfield2014', 'Heymsfield2018'.

    Returns
    -------
    callable
        A function which compute the graupel fall velocity model
        given the rain drop diameter in mm.

    Notes
    -----
    This function serves as a wrapper to various graupel fall velocity models.
    It returns the appropriate model based on the `model` parameter.
    """
    model = check_graupel_fall_velocity_model(model)
    return GRAUPEL_FALL_VELOCITY_MODELS[model]


def get_graupel_fall_velocity(diameter, model, ds_env=None, minimum_diameter=0.5, maximum_diameter=5):
    """Calculate the fall velocity of graupel based on their diameter.

    Parameters
    ----------
    diameter : array-like
        The diameter of the graupel in millimeters.
    model : str
        The model to use for calculating the graupel fall velocity. Must be one of the following:
       'Lee2015', 'Locatelli1974Lump', 'Locatelli1974Conical', 'Locatelli1974Hexagonal',
       'Heymsfield2014', 'Heymsfield2018'.
    ds_env : xarray.Dataset, optional
        A dataset containing the following environmental variables:
        - 'altitude' (m)
        - 'latitude' (°)
        - 'temperature' : Temperature in degrees Kelvin (K).
        - 'relative_humidity' :  Relative humidity. A value between 0 and 1.
        - 'sea_level_air_pressure' : Sea level air pressure in Pascals (Pa).
        - 'lapse_rate' : Lapse rate in degrees Celsius per meter (°C/m).
        If not specified, sensible default values are used.

    Returns
    -------
    fall_velocity : xarray.DataArray
        The calculated graupel fall velocities per diameter.

    Notes
    -----
    The 'Beard1976' model requires additional environmental parameters.
    These parameters can be provided through the `ds_env` argument.
    If not provided, default values are be used.

    For D < 0.12, Atlas1973 relationship results output V = 0 m/s  !
    For D < 0.05, VanDijk2002 relationship results output V = 0 m/s !
    For D < 0.02, Brandes relationship results output V = 0 m/s !

    """
    # Check valid method
    model = check_graupel_fall_velocity_model(model)

    # Copy diameter
    if isinstance(diameter, xr.DataArray):
        diameter = diameter.copy()
    else:
        diameter = np.atleast_1d(diameter)
        diameter = xr.DataArray(diameter, dims=DIAMETER_DIMENSION, coords={DIAMETER_DIMENSION: diameter.copy()})

    # Initialize ds_env if None
    # --> Ensure valid altitude and geolocation
    # - altitude requiredto correct for elevation (air_density)
    # - latitude required for gravity
    if ds_env is None:
        ds_env = load_env_dataset()
        for coord in ["altitude", "latitude"]:
            ds_env = ensure_valid_geolocation(ds_env, coord=coord, errors="raise")

    # Retrieve fall velocity
    func = get_graupel_fall_velocity_model(model)
    with suppress_warnings():  # e.g. when diameter = 0 for Beard1976
        fall_velocity = func(diameter, ds_env=ds_env) if model == "Beard1976" else func(diameter)

    # Correct for altitude
    air_pressure = retrieve_air_pressure(ds_env)
    correction_factor = (101325 / air_pressure) ** 0.545
    fall_velocity = fall_velocity * correction_factor

    # Set to NaN for diameter outside [0.5, 5]
    fall_velocity = fall_velocity.where(fall_velocity["diameter_bin_lower"] >= minimum_diameter)
    fall_velocity = fall_velocity.where(fall_velocity["diameter_bin_upper"] <= maximum_diameter)

    # Ensure fall velocity is > 0 to avoid division by zero
    # - Some models, at small diameter, can return negative/zero fall velocity
    fall_velocity = fall_velocity.where(fall_velocity > 0)

    # Add attributes
    fall_velocity.name = "fall_velocity"
    fall_velocity.attrs["units"] = "m/s"
    fall_velocity.attrs["model"] = model
    return fall_velocity.squeeze()
