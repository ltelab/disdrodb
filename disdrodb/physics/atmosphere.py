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
"""DISDRODB atmospheric physics module."""

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
