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
"""Utilities to estimate the drop fall velocity using the Beard model."""


import numpy as np
import xarray as xr


def get_gravitational_acceleration(latitude, altitude=0):
    """
    Computes gravitational acceleration at a given altitude and latitude.

    Parameters
    ----------
    altitude : float
        Altitude in meters. The default is 0 m (sea level).
    latitude : float
        Latitude in degrees.

    Returns
    -------
    float
        Gravitational acceleration in m/s^2.
    """
    g0 = 9.806229 - 0.025889372 * np.cos(2 * np.deg2rad(latitude))
    return g0 - 2.879513 * altitude / 1e6


def get_air_pressure_at_height(
    altitude,
    latitude,
    temperature,
    sea_level_air_pressure=101_325,
    lapse_rate=0.0065,
    gas_constant_dry_air=287.04,
):
    """
    Computes the air pressure at a given height in a standard atmosphere.

    According to the hypsometric formula of Brutsaert 1982; Ulaby et al. 1981

    Parameters
    ----------
    altitude : float
        Altitude in meters.
    latitude : float
        Latitude in degrees.
    temperature : float
        Temperature at altitude in Kelvin.
    sea_level_air_pressure : float, optional
        Standard atmospheric pressure at sea level in Pascals. The default is 101_325 Pascals.
    lapse_rate : float, optional
        Standard atmospheric lapse rate in K/m. The default is 0.0065 K/m.
    gas_constant_dry_air : float, optional
        Gas constant for dry air in J/(kg*K). The default is 287.04 J/(kg*K).

    Returns
    -------
    float
        Air pressure in Pascals.
    """
    g = get_gravitational_acceleration(altitude=altitude, latitude=latitude)
    return sea_level_air_pressure * np.exp(
        -g / (lapse_rate * gas_constant_dry_air) * np.log(1 + lapse_rate * altitude / temperature),
    )


def get_air_temperature_at_height(altitude, sea_level_temperature, lapse_rate=0.0065):
    """
    Computes the air temperature at a given height in a standard atmosphere.

    Reference: Brutsaert 1982; Ulaby et al. 1981

    Parameters
    ----------
    altitude : float
        Altitude in meters.
    sea_level_temperature : float
        Standard temperature at sea level in Kelvin.
    lapse_rate : float, optional
        Standard atmospheric lapse rate in K/m. The default is 0.0065 K/m.

    Returns
    -------
    float
        Air temperature in Kelvin.
    """
    return sea_level_temperature - lapse_rate * altitude


def get_vapor_actual_pressure_at_height(
    altitude,
    sea_level_temperature,
    sea_level_relative_humidity,
    sea_level_air_pressure=101_325,
    lapse_rate=0.0065,
):
    """
    Computes the vapor pressure using Yamamoto's exponential relationship.

    Reference: Brutsaert 1982

    Parameters
    ----------
    altitude : float
        Altitude in meters.
    sea_level_temperature : float
        Standard temperature at sea level in Kelvin.
    sea_level_relative_humidity : float
        Relative humidity at sea level. A value between 0 and 1.
    sea_level_air_pressure : float, optional
        Standard atmospheric pressure at sea level in Pascals. The default is 101_325 Pascals.
    lapse_rate : float, optional
        Standard atmospheric lapse rate in K/m. The default is 0.0065 K/m.

    Returns
    -------
    float
        Vapor pressure in Pascals.
    """
    temperature_at_altitude = get_air_temperature_at_height(
        altitude=altitude,
        sea_level_temperature=sea_level_temperature,
        lapse_rate=lapse_rate,
    )
    esat = get_vapor_saturation_pressure(sea_level_temperature)
    actual_vapor = sea_level_relative_humidity / (1 / esat - (1 - sea_level_relative_humidity) / sea_level_air_pressure)
    return actual_vapor * np.exp(-(5.8e3 * lapse_rate / (temperature_at_altitude**2) + 5.5e-5) * altitude)


def get_vapor_saturation_pressure(temperature):
    """
    Computes the saturation vapor pressure over water as a function of temperature.

    Use formulation and coefficients of Wexler (1976, 1977).
    References: Brutsaert 1982; Pruppacher & Klett 1978; Flatau & al. 1992

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Saturation vapor pressure in Pascal.
    """
    # Polynomial coefficients
    g = [
        -0.29912729e4,
        -0.60170128e4,
        0.1887643854e2,
        -0.28354721e-1,
        0.17838301e-4,
        -0.84150417e-9,
        0.44412543e-12,
        0.2858487e1,
    ]
    # Perform polynomial accumulation using Horner rule
    esat = g[6]
    for i in [5, 4, 3, 2]:
        esat = esat * temperature + g[i]
    esat = esat + g[7] * np.log(temperature)
    for i in [1, 0]:
        esat = esat * temperature + g[i]
    return np.exp(esat / (temperature**2))


def get_vapor_actual_pressure(relative_humidity, temperature):
    """
    Computes the actual vapor pressure over water.

    Parameters
    ----------
    relative_humidity : float
        Relative humidity. A value between 0 and 1.
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Actual vapor pressure in Pascal.
    """
    esat = get_vapor_saturation_pressure(temperature)
    return relative_humidity * esat


def get_pure_water_density(temperature):
    """
    Computes the density of pure water at standard pressure.

    For temperatures above freezing uses Kell formulation.
    For temperatures below freezing use Dorsch & Boyd formulation.

    References: Pruppacher & Klett 1978; Weast & Astle 1980

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Density of pure water in kg/m^3.
    """
    # Convert to Celsius
    temperature = temperature - 273.15

    # Define mask
    above_freezing_mask = temperature > 0

    # Compute density above freezing temperature
    c = [9.9983952e2, 1.6945176e1, -7.9870401e-3, -4.6170461e-5, 1.0556302e-7, -2.8054253e-10, 1.6879850e-2]
    density = c[0] + sum(c * temperature**i for i, c in enumerate(c[1:6], start=1))
    density_above_0 = density / (1 + c[6] * temperature)

    # Compute density below freezing temperature
    c = [999.84, 0.086, -0.0108]
    density_below_0 = c[0] + sum(c * temperature**i for i, c in enumerate(c[1:], start=1))

    # Define final density
    density = xr.where(above_freezing_mask, density_above_0, density_below_0)
    return density


def get_pure_water_compressibility(temperature):
    """
    Computes the isothermal compressibility of pure ordinary water.

    Reference: Kell, Weast & Astle 1980

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Compressibility of water in Pascals.
    """
    # Convert to Celsius
    temperature = temperature - 273.15

    # Compute compressibility
    c = [5.088496e1, 6.163813e-1, 1.459187e-3, 2.008438e-5, -5.857727e-8, 4.10411e-10, 1.967348e-2]
    compressibility = c[0] + sum(c * temperature**i for i, c in enumerate(c[1:6], start=1))
    compressibility = compressibility / (1 + c[6] * temperature) * 1e-11
    return compressibility


def get_pure_water_surface_tension(temperature):
    """
    Computes the surface tension of pure ordinary water against air.

    Reference: Pruppacher & Klett 1978

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Surface tension in N/m.
    """
    sigma = 0.0761 - 0.000155 * (temperature - 273.15)
    return sigma


def get_air_dynamic_viscosity(temperature):
    """
    Computes the dynamic viscosity of dry air.

    Reference: Beard 1977; Pruppacher & Klett 1978

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Dynamic viscosity of dry air in kg/(m*s) (aka Pa*s).
    """
    # Convert to Celsius
    temperature = temperature - 273.15

    # Define mask
    above_freezing_mask = temperature > 0

    # Compute viscosity above freezing temperature
    viscosity_above_0 = (1.721 + 0.00487 * temperature) / 1e5

    # Compute viscosity below freezing temperature
    viscosity_below_0 = (1.718 + 0.0049 * temperature - 1.2 * temperature**2 / 1e5) / 1e5

    # Define final viscosity
    viscosity = xr.where(above_freezing_mask, viscosity_above_0, viscosity_below_0)
    return viscosity


def get_air_density(temperature, air_pressure, vapor_pressure, gas_constant_dry_air=287.04):
    """
    Computes the air density according to the equation of state for moist air.

    Reference: Brutsaert 1982

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.
    air_pressure : float
        Air pressure in Pascals.
    vapor_pressure : float
        Vapor pressure in Pascals.
    gas_constant_dry_air : float, optional
        Gas constant for dry air in J/(kg*K). The default is 287.04 J/(kg*K).

    Returns
    -------
    float
        Air density in kg/m^3.
    """
    # # Define constant for water vapor in J/(kg·K)
    # gas_constant_water_vapor=461.5

    # # Partial pressure of dry air (Pa)
    # pressure_dry_air = air_pressure - vapor_pressure

    # # Density of dry air (kg/m^3)
    # density_dry_air = pressure_dry_air / (gas_constant_dry_air * temperature)

    # # Density of water vapor (kg/m^3)
    # density_water_vapor = vapor_pressure / (gas_constant_water_vapor * temperature)

    # # Total air density (kg/m^3)
    # air_density = density_dry_air + density_water_vapor

    return air_pressure * (1 - 0.378 * vapor_pressure / air_pressure) / (gas_constant_dry_air * temperature)


def get_water_density(temperature, air_pressure, sea_level_air_pressure=101_325):
    """
    Computes the density of water according to Weast & Astle 1980.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.
    air_pressure : float
        Air pressure in Pascals.
    sea_level_air_pressure : float
        Standard atmospheric pressure at sea level in Pascals.
        The default is 101_325 Pascal.
    freezing_temperature : float, optional
        Freezing temperature of water in Kelvin. The default is 273.15 K.

    Returns
    -------
    float
        Water density in kg/m^3.
    """
    delta_pressure = sea_level_air_pressure - air_pressure
    water_compressibility = get_pure_water_compressibility(temperature)
    return get_pure_water_density(temperature) * np.exp(-1 * water_compressibility * delta_pressure)


def get_raindrop_reynolds_number(diameter, temperature, air_density, water_density, g):
    """Compute raindrop Reynolds number.

    It quantifies the relative strength of the convective inertia and linear viscous
    forces acting on the drop at terminal velocity.

    Estimates Reynolds number for drops with diameter between 19 um and 7 mm.
    Coefficients are taken from Table 1 of Beard 1976.

    Reference: Beard 1976; Pruppacher & Klett 1978

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
    # Define mask for small and large particles
    small_diam_mask = diameter < 1.07e-3  # < 1mm

    # Compute properties
    pure_water_surface_tension = get_pure_water_surface_tension(temperature)  # N/m
    air_viscosity = get_air_dynamic_viscosity(temperature)  # kg/(m*s) (aka Pa*s).
    delta_density = water_density - air_density

    # Compute Davis number for small droplets
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
    # --> First 9 bins of Parsivel ...
    b = [-3.18657, 0.992696, -0.00153193, -0.000987059, -0.000578878, 0.0000855176, -0.00000327815]
    x = np.log(davis_number)
    y = b[0] + sum(b * x**i for i, b in enumerate(b[1:], start=1))
    reynolds_number_small = np.exp(y)  # TODO: miss C_sc = slip correction factor ?

    # Compute Reynolds_number_for large particles (diameter >= 0.00107)
    b = [-5.00015, 5.23778, -2.04914, 0.475294, -0.0542819, 0.00238449]
    log_property_number = np.log(property_number) / 6
    x = np.log(bond_number) + log_property_number
    y = b[0]
    y = b[0] + sum(b * x**i for i, b in enumerate(b[1:], start=1))
    reynolds_number_large = np.exp(log_property_number + y)

    # Define final reynolds number
    reynolds_number = xr.where(small_diam_mask, reynolds_number_small, reynolds_number_large)
    return reynolds_number


def get_fall_velocity_beard_1976(diameter, temperature, air_density, water_density, g):
    """
    Computes the terminal fall velocity of a raindrop in still air.

    Reference: Beard 1976; Pruppacher & Klett 1978

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
        Terminal fall velocity of the raindrop in m/s.
    """
    air_viscosity = get_air_dynamic_viscosity(temperature)
    reynolds_number = get_raindrop_reynolds_number(
        diameter=diameter,
        temperature=temperature,
        air_density=air_density,
        water_density=water_density,
        g=g,
    )
    fall_velocity = air_viscosity * reynolds_number / (air_density * diameter)
    return fall_velocity


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


def retrieve_fall_velocity(
    diameter,
    altitude,
    latitude,
    temperature,
    relative_humidity,
    air_pressure=None,
    sea_level_air_pressure=101_325,
    gas_constant_dry_air=287.04,
    lapse_rate=0.0065,
):
    """
    Computes the terminal fall velocity and drag coefficients for liquid raindrops.

    Parameters
    ----------
    diameter : float
        Diameter of the raindrop in meters.
    altitude : float
        Altitude in meters.
    temperature : float
        Temperature in Kelvin.
    relative_humidity : float
        Relative humidity. A value between 0 and 1.
    latitude : float
        Latitude in degrees.
    air_pressure : float
        Air pressure in Pascals.
        If None, air_pressure at altitude is inferred assuming
        a standard atmospheric pressure at sea level.
    sea_level_air_pressure : float
        Standard atmospheric pressure at sea level in Pascals.
        The default is 101_325 Pascal.
    gas_constant_dry_air : float, optional
        Gas constant for dry air in J/(kg*K). The default is 287.04 is J/(kg*K).
    lapse_rate : float, optional
        Standard atmospheric lapse rate in K/m. The default is 0.0065 K/m.

    Returns
    -------
    tuple
        Terminal fall velocity and drag coefficients for liquid raindrops.
    """
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

    # Retrieve air density and water density
    air_density = get_air_density(
        temperature=temperature,
        air_pressure=air_pressure,
        vapor_pressure=vapor_pressure,
        gas_constant_dry_air=gas_constant_dry_air,
    )
    water_density = get_water_density(
        temperature=temperature,
        air_pressure=air_pressure,
        sea_level_air_pressure=sea_level_air_pressure,
    )

    # Retrieve accurate gravitational_acceleration
    g = get_gravitational_acceleration(altitude=altitude, latitude=latitude)

    # Compute fall velocity
    fall_velocity = get_fall_velocity_beard_1976(
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

    return fall_velocity


####-----------------------------------------------------------------------------------------
#### OLD CODE


# def get_fall_velocity_beard_1977(diameter):
#     """
#     Compute the fall velocity of raindrops using the Beard (1977) relationship.

#     Parameters
#     ----------
#     diameter : array-like
#         Diameter of the raindrops in millimeters.
#         Valid up to 7 mm (0.7 cm).

#     Returns
#     -------
#     fall_velocity : array-like
#         Fall velocities in meters per second.

#     Notes
#     -----
#     This method uses an exponential function based on the work of Beard (1977),
#     valid at sea level conditions (pressure = 1 atm, temperature = 20°C,
#     air density = 1.194 kg/m³).

#     References
#     ----------
#     Beard, K. V. (1977).
#     Terminal velocity adjustment for cloud and precipitation drops aloft.
#     Journal of the Atmospheric Sciences, 34(8), 1293-1298.
#     https://doi.org/10.1175/1520-0469(1977)034<1293:TVAFCA>2.0.CO;2

#     """
#     diameter_cm = diameter/1000
#     c = [7.06037, 1.74951, 4.86324, 6.60631, 4.84606, 2.14922, 0.58714, 0.096348, 0.00869209, 0.00033089]
#     log_diameter = np.log(diameter_cm)
#     y = c[0] + sum(c * log_diameter**i for i, c in enumerate(c[1:], start=1))
#     fall_velocity = np.exp(y)
#     return fall_velocity


# def get_fall_velocity_beard_1977(diameter, temperature, air_pressure, gas_constant_dry_air=287.04):
#     """
#     Computes the terminal fall velocity of a raindrop in still air.

#     This function is based on the Table 4 coefficients of Kenneth V. Beard (1977),
#     "Terminal Velocity and Shape of Cloud and Precipitation Drops Aloft",
#     Journal of the Atmospheric Sciences, Vol. 34, pp. 1293-1298.

#     Note: This approximation is valid at sea level with conditions:
#           Pressure = 1 atm, Temperature = 20°C, (saturated) air density = 1.194 kg/m³.

#     Parameters
#     ----------
#     diameter : array-like
#         Array of equivolume drop diameters in meters.

#     Returns
#     -------
#     fall_velocity : array-like
#         Array of terminal fall velocity in meters per second (m/s).
#         For diameters greater than 7 mm, the function returns NaN.

#     """
#     # PROBLEMATIC
#     # Compute sea level velocity
#     c = [7.06037, 1.74951, 4.86324, 6.60631, 4.84606, 2.14922, 0.58714, 0.096348, 0.00869209, 0.00033089]
#     log_diameter = np.log(diameter / 1000 * 10)
#     y = c[0] + sum(c * log_diameter**i for i, c in enumerate(c[1:], start=1))
#     v0 = np.exp(y)

#     # Compute fall velocity
#     t_20 = 273.15 + 20
#     eps_s = get_air_dynamic_viscosity(t_20) / get_air_dynamic_viscosity(temperature) - 1
#     eps_c = -1 + (
#         np.sqrt(
#             get_air_density(
#                 temperature=t_20,
#                 air_pressure=101325,
#                 vapor_pressure=0,
#                 gas_constant_dry_air=gas_constant_dry_air,
#             )
#             / get_air_density(
#                 temperature=temperature,
#                 air_pressure=air_pressure,
#                 vapor_pressure=0,
#                 gas_constant_dry_air=gas_constant_dry_air,
#             ),
#         )
#     )
#     a = 1.104 * eps_s
#     b = (1.058 * eps_c - 1.104 * eps_s) / 5.01
#     x = np.log(diameter) + 5.52
#     f = (a + b * x) + 1
#     fall_velocity = v0 * f
#     # fall_velocity.plot()

#     eps = 1.104 * eps_s + (1.058 * eps_c - 1.104 * eps_s) * np.log(diameter / 1e-3) / 5.01
#     # eps = 1.104 * eps_s + (1.058 * eps_c - 1.104 * eps_s) * np.log(diameter / 4e-5) / 5.01
#     fall_velocity = 0.01 * v0 * (1 + eps)
#     return fall_velocity
