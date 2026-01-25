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
"""Theoretical models to estimate the raindrop fall velocity based on drop diameter in mm."""

import numpy as np
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION
from disdrodb.l0.l0b_processing import ensure_valid_geolocation
from disdrodb.l1_env.routines import load_env_dataset
from disdrodb.physics.wrappers import retrieve_air_density
from disdrodb.utils.warnings import suppress_warnings


def get_fall_velocity_atlas_1973(diameter):
    """
    Compute the fall velocity of raindrops using the Atlas et al. (1973) relationship.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in millimeters.

    Returns
    -------
    fall_velocity : array-like
        Fall velocities corresponding to the input diameters, in meters per second.

    References
    ----------
    Atlas, D., Srivastava, R. C., & Sekhon, R. S. (1973).
    Doppler radar characteristics of precipitation at vertical incidence.
    Reviews of Geophysics, 11(1), 1-35.
    https://doi.org/10.1029/RG011i001p00001

    Gunn, R., & Kinzer, G. D. (1949).
    The terminal velocity of fall for water droplets in stagnant air.
    Journal of Meteorology, 6(4), 243-248.
    https://doi.org/10.1175/1520-0469(1949)006<0243:TTVOFF>2.0.CO;2

    """
    fall_velocity = 9.65 - 10.3 * np.exp(-0.6 * diameter)  # clip to 0 !
    fall_velocity = fall_velocity.clip(min=0, max=None)
    return fall_velocity


def get_fall_velocity_lhermitte1988(diameter):
    """
    Compute the fall velocity of raindrops using the Lhermitte et al. (1988) relationship.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in millimeters.

    Returns
    -------
    fall_velocity : array-like
        Fall velocities corresponding to the input diameters, in meters per second.

    References
    ----------
    Roger M. Lhermitte, 1988.
    Observation of rain at vertical incidence with a 94 GHz Doppler radar: An insight on Mie scattering.
    Geophysical Research Letter, 15(10), 1125-1128.
    https://doi.org/10.1029/GL015i010p01125
    """
    fall_velocity = 9.25 * (1 - np.exp(-(0.068 * diameter**2 + 0.488 * diameter)))  # Ladino 2025
    # fall_velocity = 9.25 * (1 - np.exp(-(6.8 * (diameter*10)**2 + 4.88*(diameter*10)))) # Lhermitte 1988 formula wrong
    fall_velocity = fall_velocity.clip(min=0, max=None)
    return fall_velocity


def get_fall_velocity_brandes_2002(diameter):
    """
    Compute the fall velocity of raindrops using the Brandes et al. (2002) relationship.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in millimeters.

    Returns
    -------
    fall_velocity : array-like
        Fall velocities in meters per second.

    References
    ----------
    Brandes, E. A., Zhang, G., & Vivekanandan, J. (2002).
    Experiments in rainfall estimation with a polarimetric radar in a subtropical environment.
    Journal of Applied Meteorology, 41(6), 674-685.
    https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2

    """
    fall_velocity = -0.1021 + 4.932 * diameter - 0.9551 * diameter**2 + 0.07934 * diameter**3 - 0.002362 * diameter**4
    fall_velocity = fall_velocity.clip(min=0, max=None)
    return fall_velocity


def get_fall_velocity_uplinger_1981(diameter):
    """
    Compute the fall velocity of raindrops using Uplinger (1981) relationship.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in millimeters.
        Valid for diameters between 0.1 mm and 7 mm.

    Returns
    -------
    fall_velocity : array-like
        Fall velocities in meters per second.

    References
    ----------
    Uplinger, C. W. (1981). A new formula for raindrop terminal velocity.
    In Proceedings of the 20th Conference on Radar Meteorology (pp. 389-391).
    AMS.

    """
    # Valid between 0.1 and 7 mm
    fall_velocity = 4.874 * diameter * np.exp(-0.195 * diameter)  # 4.854?
    fall_velocity = fall_velocity.clip(min=0, max=None)
    return fall_velocity


def get_fall_velocity_van_dijk_2002(diameter):
    """
    Compute the fall velocity of raindrops using van Dijk et al. (2002) relationship.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in millimeters.

    Returns
    -------
    fall_velocity : array-like
        Fall velocities in meters per second.

    References
    ----------
    van Dijk, A. I. J. M., Bruijnzeel, L. A., & Rosewell, C. J. (2002).
    Rainfall intensity-kinetic energy relationships: a critical literature appraisal.
    Journal of Hydrology, 261(1-4), 1-23.
    https://doi.org/10.1016/S0022-1694(02)00020-3

    """
    fall_velocity = -0.254 + 5.03 * diameter - 0.912 * diameter**2 + 0.0561 * diameter**3
    fall_velocity = fall_velocity.clip(min=0, max=None)
    return fall_velocity


####---------------------------------------------------------------------------.
#### Beard model


def get_raindrop_reynolds_number(diameter, temperature, air_density, water_density, g):
    """Compute raindrop Reynolds number.

    It quantifies the relative strength of the convective inertia and linear viscous
    forces acting on the drop at terminal velocity.

    Estimates Reynolds number for drops with diameter between 19 um and 7 mm.
    Coefficients are taken from Table 1 of Beard 1976.

    Reference: Beard 1976; Pruppacher & Klett 1978
    See also Table A1 in Rahman et al., 2020.

    Parameters
    ----------
    diameter : float
        Diameter of the raindrop in meters.
    temperature : float
        Temperature in Kelvin.
    air_density : float
        Density of air in kg/m^3.
    water_density : float
        Density of water in kg/m^3.
    g : float
        Gravitational acceleration in m/s^2.

    Returns
    -------
    float
        Reynolds number for the raindrop.
    """
    from disdrodb.physics.atmosphere import get_air_dynamic_viscosity
    from disdrodb.physics.water import get_pure_water_surface_tension

    # Define mask for small and large particles
    small_diam_mask = diameter < 1.07e-3  # < 1mm

    # Compute properties
    pure_water_surface_tension = get_pure_water_surface_tension(temperature)  # N/m
    air_viscosity = get_air_dynamic_viscosity(temperature)  # kg/(m*s) (aka Pa*s).
    delta_density = water_density - air_density

    # Compute Davies number for small droplets
    davis_number = 4 * air_density * delta_density * g * diameter**3 / (3 * air_viscosity**2)

    # Compute the slip correction (is approx 1 and can be discarded)
    # l0 = 6.62*1e-8  # m
    # v0 = 0.01818  # g / m / s
    # p0 = 101_325_25 # Pa
    # t0 = 293.15 # K
    # c_sc = 1 + 2.51*l0*(air_viscosity/v0)*(air_pressure/p0)*((temperature/t0)**3)/diameter

    # Compute modified Bond and physical property numbers for large droplets
    bond_number = 4 * delta_density * g * diameter**2 / (3 * pure_water_surface_tension)
    property_number = pure_water_surface_tension**3 * air_density**2 / (air_viscosity**4 * delta_density * g)

    # Compute Reynolds_number_for small particles (diameter < 0.00107) (1 mm)
    b = [-3.18657, 0.992696, -0.00153193, -0.000987059, -0.000578878, 0.0000855176, -0.00000327815]
    x = np.log(davis_number)
    y = b[0] + sum(b * x**i for i, b in enumerate(b[1:], start=1))
    reynolds_number_small = np.exp(y)  # TODO: miss C_sc = slip correction factor ?

    # Compute Reynolds_number_for large particles (diameter >= 0.00107)
    b = [-5.00015, 5.23778, -2.04914, 0.475294, -0.0542819, 0.00238449]
    log_property_number = np.log(property_number) / 6
    x = np.log(bond_number) + log_property_number
    y = b[0] + sum(b * x**i for i, b in enumerate(b[1:], start=1))
    reynolds_number_large = np.exp(log_property_number + y)

    # Define final reynolds number
    reynolds_number = xr.where(small_diam_mask, reynolds_number_small, reynolds_number_large)
    return reynolds_number


def get_drag_coefficient(diameter, air_density, water_density, fall_velocity, g=9.81):
    """
    Computes the drag coefficient for a raindrop.

    Parameters
    ----------
    diameter : float
        Diameter of the raindrop in meters.
    air_density : float
        Density of air in kg/m^3.
    water_density : float
        Density of water in kg/m^3.
    fall_velocity : float
        Terminal fall velocity of the raindrop in m/s.
    g : float
        Gravitational acceleration in m/s^2.

    Returns
    -------
    float
        Drag coefficient of the raindrop.
    """
    delta_density = water_density - air_density
    drag_coefficient = 4 * delta_density * g * diameter / (3 * air_density * fall_velocity**2)
    return drag_coefficient


def get_raindrop_beard1976_fall_velocity(diameter, temperature, air_density, water_density, g):
    """
    Computes the terminal fall velocity of a raindrop in still air.

    Reference: Beard 1976; Pruppacher & Klett 1978

    Parameters
    ----------
    diameter : float
        Diameter of the raindrop in millimeters.
    temperature : float
        Temperature in Kelvin.
    air_density : float
        Density of air in kg/m^3.
    water_density : float
        Density of water in kg/m^3.
    g : float
        Gravitational acceleration in m/s^2.

    Returns
    -------
    float
        Terminal fall velocity of the raindrop in m/s.
    """
    from disdrodb.physics.atmosphere import get_air_dynamic_viscosity

    # Convert diameter to meter
    diameter = diameter / 1000

    # Compute air viscotiy and reynolds number
    air_viscosity = get_air_dynamic_viscosity(temperature)
    reynolds_number = get_raindrop_reynolds_number(
        diameter=diameter,
        temperature=temperature,
        air_density=air_density,
        water_density=water_density,
        g=g,
    )
    # Compute fall velocity
    fall_velocity = air_viscosity * reynolds_number / (air_density * diameter)
    return fall_velocity


def retrieve_raindrop_beard_fall_velocity(
    diameter,
    ds_env,
):
    """
    Computes the terminal fall velocity for liquid raindrops using the Beard (1976) model.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in millimeters.
    ds_env : xr.Dataset
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
        - 'gas_constant_dry_air': Gas constant for dry air in J/(kg*K).
        The default is 287.04 is J/(kg*K).

    Returns
    -------
    fall_velocity : array-like
        Terminal fall velocity for liquid raindrops.
    """
    from disdrodb.physics.atmosphere import (
        get_air_density,
        get_air_pressure_at_height,
        get_gravitational_acceleration,
        get_vapor_actual_pressure,
    )
    from disdrodb.physics.water import get_water_density

    # Retrieve relevant variables from ENV dataset
    altitude = ds_env["altitude"]
    latitude = ds_env["latitude"]
    temperature = ds_env["temperature"]
    relative_humidity = ds_env["relative_humidity"]
    air_pressure = ds_env.get("air_pressure", None)
    sea_level_air_pressure = ds_env.get("sea_level_air_pressure", 101_325)
    gas_constant_dry_air = ds_env.get("gas_constant_dry_air", 287.04)
    lapse_rate = ds_env.get("lapse_rate", 0.0065)

    # Retrieve air pressure at altitude if not specified
    if air_pressure is None:
        air_pressure = get_air_pressure_at_height(
            altitude=altitude,
            latitude=latitude,
            temperature=temperature,
            sea_level_air_pressure=sea_level_air_pressure,
            lapse_rate=lapse_rate,
            gas_constant_dry_air=gas_constant_dry_air,
        )

    # Retrieve vapour pressure (from relative humidity)
    vapor_pressure = get_vapor_actual_pressure(
        relative_humidity=relative_humidity,
        temperature=temperature,
    )

    # Retrieve air density
    air_density = get_air_density(
        temperature=temperature,
        air_pressure=air_pressure,
        vapor_pressure=vapor_pressure,
        gas_constant_dry_air=gas_constant_dry_air,
    )

    # Retrieve water density
    water_density = get_water_density(
        temperature=temperature,
        air_pressure=air_pressure,
        sea_level_air_pressure=sea_level_air_pressure,
    )

    # Retrieve accurate gravitational_acceleration
    g = get_gravitational_acceleration(altitude=altitude, latitude=latitude)

    # Compute fall velocity
    fall_velocity = get_raindrop_beard1976_fall_velocity(
        diameter=diameter,
        temperature=temperature,
        air_density=air_density,
        water_density=water_density,
        g=g,
    )

    # drag_coefficient = get_drag_coefficient(diameter=diameter,
    #                                         air_density=air_density,
    #                                         water_density=water_density,
    #                                         g=g.
    #                                         fall_velocity=fall_velocity)

    # Clip output
    fall_velocity = fall_velocity.clip(min=0, max=None)
    return fall_velocity


#### --------------------------------------------------------------------------------------
#### WRAPPERS
RAIN_FALL_VELOCITY_MODELS = {
    "Atlas1973": get_fall_velocity_atlas_1973,
    "Beard1976": retrieve_raindrop_beard_fall_velocity,
    "Uplinger1981": get_fall_velocity_uplinger_1981,
    "Lhermitte1988": get_fall_velocity_lhermitte1988,
    "Brandes2002": get_fall_velocity_brandes_2002,
    "VanDijk2002": get_fall_velocity_van_dijk_2002,
}


def available_rain_fall_velocity_models():
    """Return a list of the available raindrop fall velocity models."""
    return list(RAIN_FALL_VELOCITY_MODELS)


def check_rain_fall_velocity_model(model):
    """Check validity of the specified raindrop fall velocity model."""
    available_models = available_rain_fall_velocity_models()
    if model not in available_models:
        raise ValueError(f"{model} is an invalid raindrop fall velocity model. Valid models: {available_models}.")
    return model


def get_rain_fall_velocity_model(model):
    """Return the specified raindrop fall velocity model.

    Parameters
    ----------
    model : str
        The model to use for calculating the rain drop fall velocity. Available models are:
       'Atlas1973', 'Beard1976', 'Brandes2002', 'Uplinger1981', 'VanDijk2002'.

    Returns
    -------
    callable
        A function which compute the raindrop fall velocity model
        given the rain drop diameter in mm.

    Notes
    -----
    This function serves as a wrapper to various raindrop fall velocity models.
    It returns the appropriate model based on the `model` parameter.
    """
    model = check_rain_fall_velocity_model(model)
    return RAIN_FALL_VELOCITY_MODELS[model]


def get_rain_fall_velocity(diameter, model, ds_env=None):
    """Calculate the fall velocity of raindrops based on their diameter.

    Parameters
    ----------
    diameter : array-like
        The diameter of the raindrops in millimeters.
    model : str
        The model to use for calculating the raindrop fall velocity. Must be one of the following:
        'Atlas1973', 'Beard1976', 'Brandes2002', 'Uplinger1981', 'VanDijk2002'.
    ds_env : xr.Dataset, optional
        Only required if model is 'Beard1976'.
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
    fall_velocity : xr.DataArray
        The calculated raindrop fall velocities per diameter.

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
    model = check_rain_fall_velocity_model(model)

    # Copy diameter
    if isinstance(diameter, xr.DataArray):
        diameter = diameter.copy()
    else:
        diameter = np.atleast_1d(diameter)
        diameter = xr.DataArray(diameter, dims=DIAMETER_DIMENSION, coords={DIAMETER_DIMENSION: diameter.copy()})

    # Initialize ds_env
    if ds_env is None:
        ds_env = load_env_dataset()

    # Ensure valid altitude and geolocation
    # - altitude required by Beard
    # - latitude required for gravity
    for coord in ["altitude", "latitude"]:
        ds_env = ensure_valid_geolocation(ds_env, coord=coord, errors="raise")

    # Retrieve fall velocity
    func = get_rain_fall_velocity_model(model)
    with suppress_warnings():  # e.g. when diameter = 0 for Beard1976
        fall_velocity = func(diameter, ds_env=ds_env) if model == "Beard1976" else func(diameter)

    # Correct for altitude
    if model != "Beard1976":
        air_density_height = retrieve_air_density(ds_env)
        air_density_sea_surface = 1.225  # kg/m3 (International Standard Atmosphere air density at sea level)
        correction_factor = (air_density_sea_surface / air_density_height) ** (diameter * 0.025 + 0.375)
        fall_velocity = fall_velocity * correction_factor

    # Set to NaN for diameter outside [0, 10)
    fall_velocity = fall_velocity.where(diameter < 10).where(diameter > 0)

    # Ensure fall velocity is > 0 to avoid division by zero
    # - Some models, at small diameter, can return negative/zero fall velocity
    fall_velocity = fall_velocity.where(fall_velocity > 0)

    # Add attributes
    fall_velocity.name = "fall_velocity"
    fall_velocity.attrs["units"] = "m/s"
    fall_velocity.attrs["model"] = model
    return fall_velocity.squeeze()


def get_rain_fall_velocity_from_ds(ds, ds_env=None, model="Beard1976", diameter="diameter_bin_center"):
    """Compute the raindrop fall velocity.

    Parameters
    ----------
    ds : xarray.Dataset
        DISDRODB dataset with the ``'diameter_bin_center'`` coordinate.
        The ``'altitude'`` and ``'latitude'`` coordinates are used if ``model='Beard1976'``.
    model : str, optional
        Model to compute rain drop fall velocity.
        The default model is ``"Beard1976"``.
    ds_env : xr.Dataset, optional
        Only required if model is 'Beard1976'.
        A dataset containing the following environmental variables:
        - 'temperature' : Temperature in degrees Kelvin (K).
        - 'relative_humidity' :  Relative humidity. A value between 0 and 1.
        - 'sea_level_air_pressure' : Sea level air pressure in Pascals (Pa).
        - 'lapse_rate' : Lapse rate in degrees Celsius per meter (°C/m).
        If not specified, sensible default values are used.

    Returns
    -------
    xarray.DataArray
        Rain drop fall velocity DataArray.

    Notes
    -----
    The 'Beard1976' model requires additional environmental parameters.
    These parameters can be provided through the `ds_env` argument.
    If not provided, default values are be used.

    For D < 0.12, Atlas1973 relationship results output V = 0 m/s
    For D < 0.05, VanDijk2002 relationship results output V = 0 m/s
    For D < 0.02, Brandes relationship results output V = 0 m/s

    """
    from disdrodb.constants import DIAMETER_DIMENSION
    from disdrodb.l1_env.routines import load_env_dataset

    # Check if diameter dimension exists
    if DIAMETER_DIMENSION not in ds.dims:
        raise ValueError(f"Diameter dimension '{DIAMETER_DIMENSION}' not found in dataset dimensions.")

    # Retrieve ENV dataset
    # - It checks and includes default geolocation if missing
    # - For mobile disdrometer, infill missing geolocation with backward and forward filling
    if ds_env is None:
        ds_env = load_env_dataset(ds)

    # Compute raindrop fall velocity
    fall_velocity = get_rain_fall_velocity(diameter=ds[diameter], model=model, ds_env=ds_env)  # mn
    return fall_velocity
