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
"""DISDRODB water physics module."""

import numpy as np
import xarray as xr


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
