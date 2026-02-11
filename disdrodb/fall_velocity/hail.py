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
"""Theoretical models to estimate hailstones fall velocity based on particle maximum diameter in mm."""

import numpy as np
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION
from disdrodb.l0.l0b_processing import ensure_valid_geolocation
from disdrodb.l1_env.routines import load_env_dataset
from disdrodb.physics.wrappers import retrieve_air_pressure
from disdrodb.utils.warnings import suppress_warnings


def get_fall_velocity_laurie_1960(diameter):
    """Get hailstones fall velocity based on Laurie 1960 data.

    The parametrizazion is reported in Table 3 of Heymsfield et al., 2018.

    Parameters
    ----------
    diameter : array-like or float
        Particle maximum diameter in millimeters [mm].


    Returns
    -------
    fall_velocity : array-like or float
        Terminal fall velocity [m s⁻¹].

    References
    ----------
    Heymsfield, A., M. Szakáll, A. Jost, I. Giammanco, and R. Wright, 2018.
    A Comprehensive Observational Study of Graupel and Hail Terminal Velocity, Mass Flux, and Kinetic Energy.
    J. Atmos. Sci., 75, 3861-3885, https://doi.org/10.1175/JAS-D-18-0035.1.
    """
    fall_velocity = 13.95 * (0.1 * diameter) ** 0.51
    return fall_velocity


def get_fall_velocity_knight_1983_low_density(diameter):
    """Get low-density hailstones fall velocity based on Knight et al. 1983.

    The parametrization is reported in Figure 3 of Knight et al. 1983.
    It's valid for hail stone density between 0.31 and 0.61 g cm-3.

    Parameters
    ----------
    diameter : array-like or float
        Particle maximum diameter in millimeters [mm].


    Returns
    -------
    fall_velocity : array-like or float
        Terminal fall velocity [m s⁻¹].

    References
    ----------
    Knight, N. C., and A. J. Heymsfield, 1983.
    Measurement and Interpretation of Hailstone Density and Terminal Velocity.
    J. Atmos. Sci., 40, 1510-1516. https://doi.org/10.1175/1520-0469(1983)040<1510:MAIOHD>2.0.CO;2.
    """
    fall_velocity = 8.445 * (0.1 * diameter) ** 0.553
    return fall_velocity


def get_fall_velocity_knight_1983_high_density(diameter):
    """Get low-density hailstones fall velocity based on Knight et al. 1983.

    The parametrization is reported in Figure 6 of Knight et al. 1983.
    It's valid for hail stone density around 0.82 g cm-3.

    Parameters
    ----------
    diameter : array-like or float
        Particle maximum diameter in millimeters [mm].


    Returns
    -------
    fall_velocity : array-like or float
        Terminal fall velocity [m s⁻¹].

    References
    ----------
    Knight, N. C., and A. J. Heymsfield, 1983.
    Measurement and Interpretation of Hailstone Density and Terminal Velocity.
    J. Atmos. Sci., 40, 1510-1516. https://doi.org/10.1175/1520-0469(1983)040<1510:MAIOHD>2.0.CO;2.
    """
    fall_velocity = 10.58 * (0.1 * diameter) ** 0.267
    return fall_velocity


def get_fall_velocity_heymsfield_2014(diameter):
    """Get hail fall velocity from Heymsfield et al., 2014.

    Use the Heymsfield et al., 2014 parameterization.

    Parameters
    ----------
    diameter : array-like or float
        Maximum Particle maximum diameter in millimeters [mm].

    Returns
    -------
    fall_velocity : xarray.DataArray or numpy.ndarray
        Terminal fall velocity [m s⁻¹].

    References
    ----------
    Heymsfield, A. J., I. M. Giammanco, and R. Wright (2014).
    Terminal velocities and kinetic energies of natural hailstones.
    Geophys. Res. Lett., 41, 8666-8672, https://doi.org/10.1002/2014GL062324
    """
    fall_velocity = 12.28 * (0.1 * diameter) ** 0.57  # Dmax > 1.3 mm
    return fall_velocity


def get_fall_velocity_heymsfield_2018(diameter):
    """Get hailstones fall velocity from Heymsfield et al., 2018.

    Parameters
    ----------
    diameter : array-like or float
        Particle maximum diameter in millimeters [mm].

    Returns
    -------
    fall_velocity : array-like or float
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
    # fall_velocity = 6.1 * (0.1 * diameter) ** 0.72  # eq 7 and 15

    # Corrected formula from Heymsfield et al., 2020 (Corrigendum)
    fall_velocity = 8.39 * (0.1 * diameter) ** 0.67
    return fall_velocity


def get_fall_velocity_fehlmann_2020(diameter):
    """Get hailstones fall velocity from Fehlmann et al., 2020."""
    fall_velocity = 3.74 * diameter**0.5
    return fall_velocity


####------------------------------------------------------------------------------------
#### Wrappers


HAIL_FALL_VELOCITY_MODELS = {
    "Laurie1960": get_fall_velocity_laurie_1960,
    "Knight1983LD": get_fall_velocity_knight_1983_low_density,
    "Knight1983HD": get_fall_velocity_knight_1983_high_density,
    "Heymsfield2014": get_fall_velocity_heymsfield_2014,
    "Heymsfield2018": get_fall_velocity_heymsfield_2018,
    "Fehlmann2020": get_fall_velocity_fehlmann_2020,
}


def available_hail_fall_velocity_models():
    """Return a list of the available hail fall velocity models."""
    return list(HAIL_FALL_VELOCITY_MODELS)


def check_hail_fall_velocity_model(model):
    """Check validity of the specified hail fall velocity model."""
    available_models = available_hail_fall_velocity_models()
    if model not in available_models:
        raise ValueError(f"{model} is an invalid hail fall velocity model. Valid models: {available_models}.")
    return model


def get_hail_fall_velocity_model(model):
    """Return the specified hail fall velocity model.

    Parameters
    ----------
    model : str
        The model to use for calculating the rain drop fall velocity. Available models are:
       'Laurie1960', 'Knight1983LD', 'Knight1983HD', 'Heymsfield2014', 'Heymsfield2018', 'Fehlmann2020'.

    Returns
    -------
    callable
        A function which compute the hail fall velocity model
        given the rain drop diameter in mm.

    Notes
    -----
    This function serves as a wrapper to various hail fall velocity models.
    It returns the appropriate model based on the `model` parameter.
    """
    model = check_hail_fall_velocity_model(model)
    return HAIL_FALL_VELOCITY_MODELS[model]


def get_hail_fall_velocity(diameter, model, ds_env=None, minimum_diameter=4):
    """Calculate the fall velocity of hails based on their diameter.

    Parameters
    ----------
    diameter : array-like
        The diameter of the hails in millimeters.
    model : str
        The model to use for calculating the hail fall velocity. Must be one of the following:
       'Laurie1960', 'Knight1983LD', 'Knight1983HD', 'Heymsfield2014', 'Heymsfield2018', 'Fehlmann2020'.
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
        The calculated hail fall velocities per diameter.

    """
    # Check valid method
    model = check_hail_fall_velocity_model(model)

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
    func = get_hail_fall_velocity_model(model)
    with suppress_warnings():  # e.g. when diameter = 0
        fall_velocity = func(diameter)

    # Correct for altitude
    air_pressure = retrieve_air_pressure(ds_env)
    correction_factor = (101325 / air_pressure) ** 0.545
    fall_velocity = fall_velocity * correction_factor

    # Set to NaN for diameter outside [5, ...)
    fall_velocity = fall_velocity.where(diameter > minimum_diameter)

    # Ensure fall velocity is > 0 to avoid division by zero
    # - Some models, at small diameter, can return negative/zero fall velocity
    fall_velocity = fall_velocity.where(fall_velocity > 0)

    # Add attributes
    fall_velocity.name = "fall_velocity"
    fall_velocity.attrs["units"] = "m/s"
    fall_velocity.attrs["model"] = model
    return fall_velocity.squeeze()
