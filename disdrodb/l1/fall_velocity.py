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
"""Theoretical models to estimate the raindrop fall velocity based on drop diameter in mm."""
import numpy as np
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION
from disdrodb.l0.l0b_processing import ensure_valid_geolocation
from disdrodb.l1_env.routines import load_env_dataset
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
    fall_velocity = 4.874 * diameter * np.exp(-0.195 * diameter)
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
        - 'temperature' : Temperature in degrees Kelvin (K).
        - 'relative_humidity' :  Relative humidity in percentage (%).
        - 'sea_level_air_pressure' : Sea level air pressure in Pascals (Pa).
        - 'air_pressure': Air pressure in Pascals (Pa).
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
        air_pressure=ds_env.get("air_pressure", None),
        sea_level_air_pressure=ds_env["sea_level_air_pressure"],
        lapse_rate=ds_env["lapse_rate"],
    )
    fall_velocity = fall_velocity.clip(min=0, max=None)
    return fall_velocity


RAINDROP_FALL_VELOCITY_MODELS = {
    "Atlas1973": get_fall_velocity_atlas_1973,
    "Beard1976": get_fall_velocity_beard_1976,
    "Brandes2002": get_fall_velocity_brandes_2002,
    "Uplinger1981": get_fall_velocity_uplinger_1981,
    "VanDijk2002": get_fall_velocity_van_dijk_2002,
}


def available_raindrop_fall_velocity_models():
    """Return a list of the available raindrop fall velocity models."""
    return list(RAINDROP_FALL_VELOCITY_MODELS)


def check_raindrop_fall_velocity_model(model):
    """Check validity of the specified raindrop fall velocity model."""
    available_models = available_raindrop_fall_velocity_models()
    if model not in available_models:
        raise ValueError(f"{model} is an invalid raindrop fall velocity model. Valid models: {available_models}.")
    return model


def get_raindrop_fall_velocity_model(model):
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
    model = check_raindrop_fall_velocity_model(model)
    return RAINDROP_FALL_VELOCITY_MODELS[model]


def get_raindrop_fall_velocity(diameter, model, ds_env=None):
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
    model = check_raindrop_fall_velocity_model(model)

    # Copy diameter
    if isinstance(diameter, xr.DataArray):
        diameter = diameter.copy()
    else:
        diameter = np.atleast_1d(diameter)
        diameter = xr.DataArray(diameter, dims=DIAMETER_DIMENSION, coords={DIAMETER_DIMENSION: diameter.copy()})

    # Initialize ds_env if None and method == "Beard1976"
    if model == "Beard1976":
        if ds_env is None:
            ds_env = load_env_dataset()

        # Ensure valid altitude and geolocation
        # - altitude required by Beard
        # - latitude required for gravity
        for coord in ["altitude", "latitude"]:
            ds_env = ensure_valid_geolocation(ds_env, coord=coord, errors="raise")

    # Retrieve fall velocity
    func = get_raindrop_fall_velocity_model(model)
    with suppress_warnings():  # e.g. when diameter = 0 for Beard1976
        fall_velocity = func(diameter, ds_env=ds_env) if model == "Beard1976" else func(diameter)

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


def get_raindrop_fall_velocity_from_ds(ds, ds_env=None, model="Beard1976"):
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
    fall_velocity = get_raindrop_fall_velocity(diameter=ds["diameter_bin_center"], model=model, ds_env=ds_env)  # mn

    return fall_velocity
