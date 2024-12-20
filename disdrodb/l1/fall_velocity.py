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
"""Theoretical models to estimate the drop fall velocity."""


import numpy as np


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

    Atlas, D., & Ulbrich, C. W. (1977).
    Path- and area-integrated rainfall measurement by microwave attenuation in the 1-3 cm band.
    Journal of Applied Meteorology, 16(12), 1322-1331.
    https://doi.org/10.1175/1520-0450(1977)016<1322:PAAIRM>2.0.CO;2

    Gunn, R., & Kinzer, G. D. (1949).
    The terminal velocity of fall for water droplets in stagnant air.
    Journal of Meteorology, 6(4), 243-248.
    https://doi.org/10.1175/1520-0469(1949)006<0243:TTVOFF>2.0.CO;2

    """
    fall_velocity = 9.65 - 10.3 * np.exp(-0.6 * diameter)  # clip to 0 !
    fall_velocity = np.clip(fall_velocity, 0, None)
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
    fall_velocity = 4.874 * diameter * np.exp(-0.195 * diameter)
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
    return fall_velocity


def get_fall_velocity_beard_1976(diameter, ds_env):
    """Calculate the fall velocity of a particle using the Beard (1976) model.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in millimeters.
    ds_env : xr.Dataset
        A dataset containing the following environmental variables:
        - 'altitude' :  Altitude in meters (m).
        - 'latitude' :  Latitude in degrees.
        - 'temperature' : Temperature in degrees Celsius (°C).
        - 'relative_humidity' :  Relative humidity in percentage (%).
        - 'sea_level_air_pressure' : Sea level air pressure in Pascals (Pa).
        - 'lapse_rate' : Lapse rate in degrees Celsius per meter (°C/m).

    Returns
    -------
    fall_velocity : array-like
        The calculated fall velocities of the raindrops.
    """
    from disdrodb.l1.beard_model import retrieve_fall_velocity

    # Input diameter in mmm
    fall_velocity = retrieve_fall_velocity(
        diameter=diameter / 1000,  # diameter expected in m !!!
        altitude=ds_env["altitude"],
        latitude=ds_env["latitude"],
        temperature=ds_env["temperature"],
        relative_humidity=ds_env["relative_humidity"],
        # TODO: add air_pressure # TODO
        sea_level_air_pressure=ds_env["sea_level_air_pressure"],
        lapse_rate=ds_env["lapse_rate"],
    )
    return fall_velocity


def ensure_valid_coordinates(ds, default_altitude=0, default_latitude=0, default_longitude=0):
    """Ensure dataset valid coordinates for altitude, latitude, and longitude.

    Invalid values are np.nan and -9999.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset for which to ensure valid geolocation coordinates.
    default_altitude : float, optional
        The default value to use for invalid altitude values. Defaults to 0.
    default_latitude : float, optional
        The default value to use for invalid latitude values. Defaults to 0.
    default_longitude : float, optional
        The default value to use for invalid longitude values. Defaults to 0.

    Returns
    -------
    xarray.Dataset
        The dataset with invalid coordinates replaced by default values.

    """
    invalid_altitude = np.logical_or(np.isnan(ds["altitude"]), ds["altitude"] == -9999)
    ds["altitude"] = ds["altitude"].where(~invalid_altitude, default_altitude)

    invalid_lat = np.logical_or(np.isnan(ds["latitude"]), ds["latitude"] == -9999)
    ds["latitude"] = ds["latitude"].where(~invalid_lat, default_latitude)

    invalid_lon = np.logical_or(np.isnan(ds["longitude"]), ds["longitude"] == -9999)
    ds["longitude"] = ds["longitude"].where(~invalid_lon, default_longitude)
    return ds


def get_raindrop_fall_velocity(diameter, method, ds_env=None):
    """Calculate the fall velocity of raindrops based on their diameter.

    Parameters
    ----------
    diameter : array-like
        The diameter of the raindrops in millimeters.
    method : str
        The method to use for calculating the fall velocity. Must be one of the following:
        'Atlas1973', 'Beard1976', 'Brandes2002', 'Uplinger1981', 'VanDijk2002'.
    ds_env : xr.Dataset, optional
        A dataset containing the following environmental variables:
        - 'altitude' :  Altitude in meters (m).
        - 'latitude' :  Latitude in degrees.
        - 'temperature' : Temperature in degrees Celsius (°C).
        - 'relative_humidity' :  Relative humidity. A value between 0 and 1.
        - 'sea_level_air_pressure' : Sea level air pressure in Pascals (Pa).
        - 'lapse_rate' : Lapse rate in degrees Celsius per meter (°C/m).
        It is required for for the 'Beard1976' method.

    Returns
    -------
    fall_velocity : array-like
        The calculated fall velocities of the raindrops.

    Notes
    -----
    The 'Beard1976' method requires additional environmental parameters such as altitude and latitude.
    These parameters can be provided through the `ds_env` argument. If not provided, default values will be used.
    """
    # Input diameter in mm
    dict_methods = {
        "Atlas1973": get_fall_velocity_atlas_1973,
        "Beard1976": get_fall_velocity_beard_1976,
        "Brandes2002": get_fall_velocity_brandes_2002,
        "Uplinger1981": get_fall_velocity_uplinger_1981,
        "VanDijk2002": get_fall_velocity_van_dijk_2002,
    }
    # Check valid method
    available_methods = list(dict_methods)
    if method not in dict_methods:
        raise ValueError(f"{method} is an invalid fall velocity method. Valid methods: {available_methods}.")
    # Copy diameter
    diameter = diameter.copy()
    # Ensure valid altitude and geolocation (if missing set defaults)
    # - altitude required by Beard
    # - latitude required for gravity
    ds_env = ensure_valid_coordinates(ds_env)
    # Retrieve fall velocity
    func = dict_methods[method]
    fall_velocity = func(diameter, ds_env=ds_env) if method == "Beard1976" else func(diameter)
    return fall_velocity
