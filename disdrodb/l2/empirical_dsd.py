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
"""Functions for computation of DSD parameters."""

import numpy as np
import xarray as xr


def get_effective_sampling_area(sensor_name, diameter):
    """Compute the effective sampling area of the disdrometer."""
    if sensor_name in ["OTT_Parsivel", "OTT_Parsivel2"]:
        # Calculate sampling area for each diameter bin (S_i)
        L = 180 / 1000  # Length of the Parsivel beam in m (180 mm)
        B = 30 / 1000  # Width of the Parsivel beam in m (30mm)
        sampling_area = L * (B - diameter / 1000 / 2)
    elif sensor_name in "Thies_LPM":
        # TODO: provided as variable varying with time?
        L = 228 / 1000  # Length of the Parsivel beam in m (228 mm)
        B = 20 / 1000  # Width of the Parsivel beam in m (20 mm)
        sampling_area = L * (B - diameter / 1000 / 2)
    elif sensor_name in "RD80":
        sampling_area = 1  # TODO
    else:
        raise NotImplementedError
    return sampling_area


def _get_spectrum_dims(ds):
    if "velocity_bin_center" in ds.dims:
        dims = ["diameter_bin_center", "velocity_bin_center"]
    else:
        dims = ["diameter_bin_center"]
    return dims


def get_drop_volume(diameter):
    """
    Compute the volume of a droplet assuming it is spherical.

    Parameters
    ----------
    diameter : float or array-like
        The diameter of the droplet(s). Can be a scalar or an array of diameters.

    Returns
    -------
    array-like
        The volume of the droplet(s) calculated in cubic units based on the input diameter(s).

    Notes
    -----
    The volume is calculated using the formula for the volume of a sphere:
    V = (π/6) * d^3, where d is the diameter of the droplet.
    """
    return np.pi / 6 * diameter**3  # /6 = 4/3*(0.5**3)


####-------------------------------------------------------------------------------------------------------------------.


def get_drop_average_velocity(drop_number):
    r"""
    Calculate the drop average velocity \\( v_m(D))) \\) per diameter class.

    Parameters
    ----------
    drop_number : xarray.DataArray
        Array of drop counts \\( n(D,v) \\) per diameter (and velocity, if available) bins
        over the time integration period.

    Returns
    -------
    average_velocity : xarray.DataArray
        Array of drop average velocity \\( v_m(D))) \\) in m·s⁻¹ .
    """
    velocity = xr.ones_like(drop_number) * drop_number["velocity_bin_center"]
    average_velocity = ((velocity * drop_number).sum(dim="velocity_bin_center")) / drop_number.sum(
        dim="velocity_bin_center",
    )
    # average_velocity = average_velocity.where(average_velocity > 0, 0)
    return average_velocity


def get_drop_number_concentration(drop_number, velocity, diameter_bin_width, sampling_area, sample_interval):
    r"""
    Calculate the volumetric drop number concentration \\( N(D) \\) per diameter class.

    Computes the drop number concentration \\( N(D) \\) [m⁻³·mm⁻¹] for each diameter
    class based on the measured drop counts and sensor parameters. This represents
    the number of drops per unit volume per unit diameter interval.
    It is also referred to as the drop size distribution N(D) per cubic metre per millimetre [m-3 mm-1]

    Parameters
    ----------
    velocity : xarray.DataArray
        Array of drop fall velocities \\( v(D) \\) corresponding to each diameter bin in meters per second (m/s).
    diameter_bin_width : xarray.DataArray
        Width of each diameter bin \\( \\Delta D \\) in millimeters (mm).
    drop_number : xarray.DataArray
        Array of drop counts \\( n(D,v) \\) per diameter (and velocity, if available)
        bins over the time integration period.
    sample_interval : float or xarray.DataArray
        Time over which the drops are counted \\( \\Delta t \\) in seconds (s).
    sampling_area : float or xarray.DataArray
        The effective sampling area \\( A \\) of the sensor in square meters (m²).

    Returns
    -------
    drop_number_concentration : xarray.DataArray or ndarray
        Array of drop number concentrations \\( N(D) \\) in m⁻³·mm⁻¹, representing
        the number of drops per unit volume per unit diameter interval.

    Notes
    -----
    The drop number concentration \\( N(D) \\) is calculated using:

    .. math::

        N(D) = \frac{n(D)}{A_{\text{eff}}(D) \\cdot \\Delta D \\cdot \\Delta t \\cdot v(D)}

    where:

    - \\( n(D,v) \\): Number of drops counted in diameter (and velocity) bins.
    - \\( A_{\text{eff}}(D) \\): Effective sampling area of the sensor for diameter \\( D \\) in square meters (m²).
    - \\( \\Delta D \\): Diameter bin width in millimeters (mm).
    - \\( \\Delta t \\): Time integration period in seconds (s).
    - \\( v(D) \\): Fall velocity of drops in diameter bin \\( D \\) in meters per second (m/s).

    The effective sampling area \\( A_{\text{eff}}(D) \\) depends on the sensor and may vary with drop diameter.
    """
    # Ensure velocity is 2D (diameter, velocity)
    velocity = xr.ones_like(drop_number) * velocity

    # Compute drop number concentration
    # - For disdrometer with velocity bins
    if "velocity_bin_center" in drop_number.dims:
        drop_number_concentration = (drop_number / velocity).sum(dim=["velocity_bin_center"]) / (
            sampling_area * diameter_bin_width * sample_interval
        )
    # - For impact disdrometers
    else:
        drop_number_concentration = drop_number / (sampling_area * diameter_bin_width * sample_interval * velocity)
    return drop_number_concentration


# def get_drop_number_concentration1(drop_counts, velocity, diameter_bin_width, sampling_area, sample_interval):
#     r"""
#     Calculate the volumetric drop number concentration \\( N(D) \\) per diameter class.

#     Computes the drop number concentration \\( N(D) \\) [m⁻³·mm⁻¹] for each diameter
#     class based on the measured drop counts and sensor parameters. This represents
#     the number of drops per unit volume per unit diameter interval.
#     It is also referred to as the drop size distribution N(D) per cubic metre per millimetre [m-3 mm-1]

#     Parameters
#     ----------
#     velocity : xarray.DataArray
#         Array of drop fall velocities \\( v(D) \\) corresponding to each diameter bin in meters per second (m/s).
#     diameter_bin_width : xarray.DataArray
#         Width of each diameter bin \\( \\Delta D \\) in millimeters (mm).
#     drop_counts : xarray.DataArray
#         Array of drop counts \\( n(D) \\) per diameter bin over the time integration period.
#     sample_interval : float or xarray.DataArray
#         Time over which the drops are counted \\( \\Delta t \\) in seconds (s).
#     sampling_area : xarray.DataArray

#     Returns
#     -------
#     drop_number_concentration : xarray.DataArray or ndarray
#         Array of drop number concentrations \\( N(D) \\) in m⁻³·mm⁻¹, representing
#         the number of drops per unit volume per unit diameter interval.

#     Notes
#     -----
#     The drop number concentration \\( N(D) \\) is calculated using:

#     .. math::

#         N(D) = \frac{n(D)}{A_{\text{eff}}(D) \\cdot \\Delta D \\cdot \\Delta t \\cdot v(D)}

#     where:

#     - \\( n(D) \\): Number of drops counted in diameter bin \\( D \\).
#     - \\( A_{\text{eff}}(D) \\): Effective sampling area of the sensor for diameter \\( D \\) in square meters (m²).
#     - \\( \\Delta D \\): Diameter bin width in millimeters (mm).
#     - \\( \\Delta t \\): Time integration period in seconds (s).
#     - \\( v(D) \\): Fall velocity of drops in diameter bin \\( D \\) in meters per second (m/s).

#     The effective sampling area \\( A_{\text{eff}}(D) \\) depends on the sensor and may vary with drop diameter.
#     """
#     drop_number_concentration = drop_counts / (sampling_area * diameter_bin_width * sample_interval * velocity)
#     return drop_number_concentration


def get_total_number_concentration(drop_number_concentration, diameter_bin_width):
    r"""
    Compute the total number concentration \\( N_t \\) from the drop size distribution.

    Calculates the total number concentration \\( N_t \\) [m⁻³] by integrating the
    drop number concentration over all diameter bins.

    Parameters
    ----------
    drop_number_concentration : xarray.DataArray
        Array of drop number concentrations \\( N(D) \\) in m⁻³·mm⁻¹.
    diameter_bin_width : xarray.DataArray
        Width of each diameter bin \\( \\Delta D \\) in millimeters (mm).

    Returns
    -------
    total_number_concentration : xarray.DataArray or ndarray
        Total number concentration \\( N_t \\) in m⁻³, representing the total number
        of drops per unit volume.

    Notes
    -----
    The total number concentration \\( N_t \\) is calculated by integrating over the diameter bins:

    .. math::

        N_t = \\sum_{\text{bins}} N(D) \\cdot \\Delta D

    where:

    - \\( N(D) \\): Drop number concentration in each diameter bin [m⁻³·mm⁻¹].
    - \\( \\Delta D \\): Diameter bin width in millimeters (mm).

    """
    total_number_concentration = (drop_number_concentration * diameter_bin_width).sum(dim="diameter_bin_center")
    return total_number_concentration


def get_moment(drop_number_concentration, diameter, diameter_bin_width, moment):
    r"""
    Calculate the m-th moment of the drop size distribution.

    Computes the m-th moment of the drop size distribution (DSD), denoted as E[D**m],
    where D is the drop diameter and m is the order of the moment. This is useful
    in meteorology and hydrology for characterizing precipitation. For example,
    weather radar measurements correspond to the sixth moment of the DSD (m = 6).

    Parameters
    ----------
    drop_number_concentration : xarray.DataArray
        The drop number concentration N(D) for each diameter bin,
        typically in units of number per cubic meter per millimeter (m⁻³ mm⁻¹).
    diameter : xarray.DataArray
        The equivalent volume diameters D of the drops in each bin, in meters (m).
    diameter_bin_width : xarray.DataArray
        The width dD of each diameter bin, in millimeters (mm).
    moment : int or float
        The order m of the moment to compute.

    Returns
    -------
    moment_value : xarray.DataArray
        The computed m-th moment of the drop size distribution, typically in units
        dependent on the input units, such as mmᵐ m⁻³.

    Notes
    -----
    The m-th moment is calculated using the formula:

    .. math::

        M_m = \\sum_{\text{bins}} N(D) \\cdot D^m \\cdot dD

    where:

    - \\( M_m \\) is the m-th moment of the DSD.
    - \\( N(D) \\) is the drop number concentration for diameter \\( D \\).
    - \\( D^m \\) is the diameter raised to the power of \\( m \\).
    - \\( dD \\) is the diameter bin width.

    This computation integrates over the drop size distribution to provide a
    scalar value representing the statistical momen
    """
    return ((diameter * 1000) ** moment * drop_number_concentration * diameter_bin_width).sum(dim="diameter_bin_center")


####------------------------------------------------------------------------------------------------------------------
#### Rain and Reflectivity


def get_rain_rate(drop_counts, sampling_area, diameter, sample_interval):
    r"""
    Compute the rain rate \\( R \\) [mm/h] based on the drop size distribution and drop velocities.

    This function calculates the rain rate by integrating over the drop size distribution (DSD),
    considering the volume of water falling per unit time and area. It uses the number of drops
    counted in each diameter class, the effective sampling area of the sensor, the diameters of the
    drops, and the time interval over which the drops are counted.

    Parameters
    ----------
    drop_counts : xarray.DataArray
        Array representing the number of drops per diameter class \\( n(D) \\) in each bin.
    sample_interval : float or xarray.DataArray
        The time duration over which drops are counted \\( \\Delta t \\) in seconds (s).
    sampling_area : float or xarray.DataArray
        The effective sampling area \\( A \\) of the sensor in square meters (m²).
    diameter : xarray.DataArray
        Array of drop diameters \\( D \\) in meters (m).

    Returns
    -------
    rain_rate : xarray.DataArray
        The computed rain rate \\( R \\) in millimeters per hour (mm/h), which represents the volume
        of water falling per unit area per unit time.

    Notes
    -----
    The rain rate \\( R \\) is calculated using the following formula:

    .. math::

        R = \frac{\\pi}{6} \times 10^{-3} \times 3600 \times
        \\sum_{\text{bins}} n(D) \cdot A(D) \cdot D^3 \cdot \\Delta t

    Where:
    - \\( n(D) \\) is the number of drops in each diameter class.
    - \\( A(D) \\) is the effective sampling area.
    - \\( D \\) is the drop diameter.
    - \\( \\Delta t \\) is the time interval for drop counts.

    This formula incorporates a conversion factor to express the rain rate in millimeters per hour.
    """
    rain_rate = (
        np.pi
        / 6
        / sample_interval
        * (drop_counts / sampling_area * diameter**3).sum(dim="diameter_bin_center")
        * 3600
        * 1000
    )

    # 0.6 or / 6 --> Different variant across articles and codes !!! (pydsd 0.6, raupach 2015, ...)
    # -->  1/6 * 3600 = 600 = 0.6  * 1e3 = 6 * 1e2
    # -->  1/6 * 3600 * 1000 = 0.6 * 1e6 = 6 * 1e5 --> 6 * 1e-4 (if diameter in mm)
    # rain_rate = np.pi * 0.6 * 1e3 / sample_interval * (
    #   (drop_counts * diameter**3 / sampling_area).sum(dim="diameter_bin_center") * 1000))
    # rain_rate = np.pi / 6 / sample_interval * (
    #   (drop_counts * diameter**3 / sampling_area).sum(dim="diameter_bin_center") * 1000 * 3600)

    return rain_rate


def get_rain_rate_from_dsd(drop_number_concentration, velocity, diameter, diameter_bin_width):
    r"""
    Compute the rain rate \\( R \\) [mm/h] based on the drop size distribution and raindrop velocities.

    Calculates the rain rate by integrating over the drop size distribution (DSD),
    considering the volume of water falling per unit time and area.

    Parameters
    ----------
    drop_number_concentration : xarray.DataArray
        Array of drop number concentrations \\( N(D) \\) in m⁻³·mm⁻¹.
    velocity : xarray.DataArray
        Array of drop fall velocities \\( v(D) \\) corresponding to each diameter bin in meters per second (m/s).
    diameter : xarray.DataArray
        Array of drop diameters \\( D \\) in meters (m).
    diameter_bin_width : xarray.DataArray
        Width of each diameter bin \\( \\Delta D \\) in millimeters (mm).

    Returns
    -------
    rain_rate : xarray.DataArray
        The rain rate \\( R \\) in millimeters per hour (mm/h), representing the volume
        of water falling per unit area per unit time.

    Notes
    -----
    The rain rate \\( R \\) is calculated using:

    .. math::

        R = \frac{\\pi}{6} \times 10^{-3} \times 3600 \times
          \\sum_{\text{bins}} N(D) \\cdot v(D) \\cdot D^3 \\cdot \\Delta D

    where:

    - \\( N(D) \\): Drop number concentration [m⁻³·mm⁻¹].
    - \\( v(D) \\): Fall velocity of drops in diameter bin \\( D \\) [m/s].
    - \\( D \\): Drop diameter [mm].
    - \\( \\Delta D \\): Diameter bin width [mm].
    - The factor \\( \frac{\\pi}{6} \\) converts the diameter cubed to volume of a sphere.
    - The factor \\( 10^{-3} \\) converts from mm³ to m³.
    - The factor \\( 3600 \\) converts from seconds to hours.

    """
    # The following formula assume diameter in mm !!!
    rain_rate = (
        np.pi
        / 6
        * (drop_number_concentration * velocity * diameter**3 * diameter_bin_width).sum(dim="diameter_bin_center")
        * 3600
        * 1000
    )

    # Alternative formulation
    # 3600*1000/6 = 6e5
    # 1e-9 for mm to meters conversion
    # --> 6 * 1 e-4
    # rain_rate = 6 * np.pi * 1e-4 * (
    #   (drop_number_concentration * velocity * diameter**3 * diameter_bin_width).sum(dim="diameter_bin_center")
    # )
    return rain_rate


def get_rain_accumulation(rain_rate, sample_interval):
    """
    Calculate the total rain accumulation over a specified time period.

    Parameters
    ----------
    rain_rate : float or array-like
        The rain rate in millimeters per hour (mm/h).
    sample_interval : int
        The time over which to accumulate rain, specified in seconds.

    Returns
    -------
    float or numpy.ndarray
        The total rain accumulation in millimeters (mm) over the specified time period.

    """
    rain_accumulation = rain_rate / 3600 * sample_interval
    return rain_accumulation


def get_equivalent_reflectivity_factor(drop_number_concentration, diameter, diameter_bin_width):
    r"""
    Compute the equivalent reflectivity factor in decibels relative to 1 mm⁶·m⁻³ (dBZ).

    The equivalent reflectivity (in mm⁶·m⁻³) is obtained from the sixth moment of the drop size distribution (DSD).
    The reflectivity factor is expressed in decibels relative to 1 mm⁶·m⁻³ using the formula:

    .. math::

        Z = 10 \cdot \log_{10}(z)

    where \\( z \\) is the reflectivity in linear units of the DSD.

    To convert back the reflectivity factor to linear units (mm⁶·m⁻³), use the formula:

    .. math::

        z = 10^{(Z/10)}

    Parameters
    ----------
    drop_number_concentration : xarray.DataArray
        Array representing the concentration of droplets per diameter class in number per unit volume.
    diameter : xarray.DataArray
        Array of droplet diameters in meters (m).
    diameter_bin_width : xarray.DataArray
        Array representing the width of each diameter bin in millimeters (mm).

    Returns
    -------
    xarray.DataArray
        The equivalent reflectivity factor in decibels (dBZ).

    Notes
    -----
    The function computes the sixth moment of the DSD using the formula:

    .. math::

        z = \\sum n(D) \cdot D^6 \cdot \\Delta D

    where \\( n(D) \\) is the drop number concentration, \\( D \\) is the drop diameter, and
    \\( \\Delta D \\) is the diameter bin width.

    """
    # Compute reflectivity in mm⁶·m⁻³
    z = ((diameter * 1000) ** 6 * drop_number_concentration * diameter_bin_width).sum(dim="diameter_bin_center")
    invalid_mask = z > 0
    z = z.where(invalid_mask)
    # Compute equivalent reflectivity factor in dBZ
    # - np.log10(np.nan) returns -Inf !
    # --> We mask again after the log
    Z = 10 * np.log10(z)
    Z = Z.where(invalid_mask)
    return Z


####------------------------------------------------------------------------------------------------------------------
#### Liquid Water Content / Mass Parameters


def get_mass_spectrum(drop_number_concentration, diameter, water_density=1000):
    """
    Calculate the rain drop mass spectrum m(D) in g/m3 mm-1.

    It represents the mass of liquid water as a function of raindrop diameter.

    Parameters
    ----------
    drop_number_concentration : array-like
        The concentration of droplets (number of droplets per unit volume) in each diameter bin.
    diameter : array-like
        The diameters of the droplets for each bin, in meters (m).


    Returns
    -------
    array-like
        The calculated rain drop mass spectrum in grams per cubic meter per diameter (g/m3 mm-1).

    """
    # Convert water density from kg/m3 to g/m3
    water_density = water_density * 1000

    # Calculate the volume constant for the water droplet formula
    vol_constant = np.pi / 6.0 * water_density

    # Calculate the mass spectrum (lwc per diameter bin)
    return vol_constant * (diameter**3 * drop_number_concentration)  #  [g/m3 mm-1]


def get_liquid_water_content(drop_number_concentration, diameter, diameter_bin_width, water_density=1000):
    """
    Calculate the liquid water content based on drop number concentration and drop diameter.

    Parameters
    ----------
    drop_number_concentration : array-like
        The concentration of droplets (number of droplets per unit volume) in each diameter bin.
    diameter : array-like
        The diameters of the droplets for each bin, in meters (m).
    diameter_bin_width : array-like
        The width of each diameter bin, in millimeters (mm).
    water_density : float, optional
        The density of water in kg/m^3. The default is 1000 kg/m3.

    Returns
    -------
    array-like
        The calculated liquid water content in grams per cubic meter (g/m3).

    """
    # Convert water density from kg/m3 to g/m3
    water_density = water_density * 1000

    # Calculate the volume constant for the water droplet formula
    vol_constant = np.pi / 6.0 * water_density

    # Calculate the liquid water content
    lwc = vol_constant * (diameter**3 * drop_number_concentration * diameter_bin_width).sum(dim="diameter_bin_center")
    return lwc


def get_mom_liquid_water_content(moment_3, water_density=1000):
    r"""
    Calculate the liquid water content (LWC) from the third moment of the DSD.

    LWC represents the mass of liquid water per unit volume of air.

    Parameters
    ----------
    moment_3 : float or array-like
        The third moment of the drop size distribution, \\( M_3 \\), in units of
        [m⁻³·mm³] (number per cubic meter times diameter cubed).
    water_density : float, optional
        The density of water in kilograms per cubic meter (kg/m³).
        Default is 1000 kg/m³ (approximate density of water at 20°C).

    Returns
    -------
    lwc : float or array-like
        The liquid water content in grams per cubic meter (g/m³).

    Notes
    -----
    The liquid water content is calculated using the formula:

    .. math::

        \text{LWC} = \frac{\\pi \rho_w}{6} \\cdot M_3

    where:

    - \\( \text{LWC} \\) is the liquid water content [g/m³].
    - \\( \rho_w \\) is the density of water [g/mm³].
    - \\( M_3 \\) is the third moment of the DSD [m⁻³·mm³].

    Examples
    --------
    Compute the liquid water content from the third moment:

    >>> moment_3 = 1e6  # Example value in [m⁻³·mm³]
    >>> lwc = get_liquid_water_content_from_moments(moment_3)
    >>> print(f"LWC: {lwc:.4f} g/m³")
    LWC: 0.0005 g/m³
    """
    # Convert water density from kg/m³ to g/mm³
    water_density = water_density * 1e-6  # [kg/m³] * 1e-6 = [g/mm³]
    # Calculate LWC [g/m3]
    lwc = (np.pi * water_density / 6) * moment_3  # [g/mm³] * [m⁻³·mm³] = [g/m³]
    return lwc


####--------------------------------------------------------------------------------------------------------
#### Diameter parameters


def _get_last_xr_valid_idx(da_condition, dim, fill_value=None):
    """
    Get the index of the last True value along a specified dimension in an xarray DataArray.

    This function finds the last index along the given dimension where the condition is True.
    If all values are False or NaN along that dimension, the function returns ``fill_value``.

    Parameters
    ----------
    da_condition : xarray.DataArray
        A boolean DataArray where True indicates valid or desired values.
        Should have the dimension specified in `dim`.
    dim : str
        The name of the dimension along which to find the last True index.
    fill_value : int or float
        The fill value when all values are False or NaN along the specified dimension.
        The default is ``dim_size - 1``.

    Returns
    -------
    last_idx : xarray.DataArray
        An array containing the index of the last True value along the specified dimension.
        If all values are False or NaN, the corresponding entry in `last_idx` will be NaN.

    Notes
    -----
    The function works by reversing the DataArray along the specified dimension and using
    `argmax` to find the first True value in the reversed array. It then calculates the
    corresponding index in the original array. To handle cases where all values are False
    or NaN (and `argmax` would return 0), the function checks if there is any True value
    along the dimension and assigns NaN to `last_idx` where appropriate.

    Examples
    --------
    >>> import xarray as xr
    >>> da = xr.DataArray([[False, False, True], [False, False, False]], dims=["time", "diameter_bin_center"])
    >>> last_idx = _get_last_xr_valid_idx(da, "diameter_bin_center")
    >>> print(last_idx)
    <xarray.DataArray (time: 2)>
    array([2., nan])
    Dimensions without coordinates: time

    In this example, for the first time step, the last True index is 2.
    For the second time step, all values are False, so the function returns NaN.

    """
    # Get the size of the 'diameter_bin_center' dimension
    dim_size = da_condition.sizes[dim]

    # Define default fillvalue
    if fill_value is None:
        fill_value = dim_size - 1

    # Reverse the mask along 'diameter_bin_center'
    da_condition_reversed = da_condition.isel({dim: slice(None, None, -1)})

    # Check if there is any True value along the dimension for each slice
    has_true = da_condition.any(dim=dim)

    # Find the first non-zero index in the reversed array
    last_idx_from_end = da_condition_reversed.argmax(dim=dim)

    # Calculate the last True index in the original array
    last_idx = xr.where(
        has_true,
        dim_size - last_idx_from_end - 1,
        fill_value,
    )
    return last_idx


def get_min_max_diameter(drop_counts):
    """
    Get the minimum and maximum diameters where drop_counts is non-zero.

    Parameters
    ----------
    drop_counts : xarray.DataArray
        Drop counts with dimensions ("time", "diameter_bin_center") and
        coordinate "diameter_bin_center".

    Returns
    -------
    min_drop_diameter : xarray.DataArray
        Minimum diameter where drop_counts is non-zero, for each time step.
    max_drop_diameter : xarray.DataArray
        Maximum diameter where drop_counts is non-zero, for each time step.
    """
    # Create a boolean mask where drop_counts is non-zero
    non_zero_mask = drop_counts > 0

    # Find the first non-zero index along 'diameter_bin_center' for each time
    # - Return 0 if all False, zero or NaN
    first_non_zero_idx = non_zero_mask.argmax(dim="diameter_bin_center")

    # Calculate the last non-zero index in the original array
    last_non_zero_idx = _get_last_xr_valid_idx(da_condition=non_zero_mask, dim="diameter_bin_center")

    # Get the 'diameter_bin_center' coordinate
    diameters = drop_counts["diameter_bin_center"]

    # Retrieve the diameters corresponding to the first and last non-zero indices
    min_drop_diameter = diameters.isel(diameter_bin_center=first_non_zero_idx.astype(int))
    max_drop_diameter = diameters.isel(diameter_bin_center=last_non_zero_idx.astype(int))

    # Identify time steps where all drop_counts are zero
    is_all_zero_or_nan = ~non_zero_mask.any(dim="diameter_bin_center")

    # Mask with NaN where no drop or all values are NaN
    min_drop_diameter = min_drop_diameter.where(~is_all_zero_or_nan)
    max_drop_diameter = max_drop_diameter.where(~is_all_zero_or_nan)

    return min_drop_diameter, max_drop_diameter


def get_mode_diameter(drop_number_concentration):
    """Get raindrop diameter with highest occurrence."""
    diameter = drop_number_concentration["diameter_bin_center"]
    # If all NaN, set to 0 otherwise argmax fail when all NaN data
    idx_all_nan_mask = np.isnan(drop_number_concentration).all(dim="diameter_bin_center")
    drop_number_concentration = drop_number_concentration.where(~idx_all_nan_mask, 0)
    # Find index where all 0
    # --> argmax will return 0
    idx_all_zero = (drop_number_concentration == 0).all(dim="diameter_bin_center")
    # Find the diameter index corresponding the "mode"
    idx_observed_mode = drop_number_concentration.argmax(dim="diameter_bin_center")
    # Find the diameter corresponding to the "mode"
    diameter_mode = diameter.isel({"diameter_bin_center": idx_observed_mode})
    diameter_mode = diameter_mode.drop(
        ["diameter_bin_width", "diameter_bin_lower", "diameter_bin_upper", "diameter_bin_center"],
    )
    # Set to np.nan where data where all NaN or all 0
    idx_mask = np.logical_or(idx_all_nan_mask, idx_all_zero)
    diameter_mode = diameter_mode.where(~idx_mask)
    return diameter_mode


####-------------------------------------------------------------------------------------------------------------------.
#### Mass diameters


def get_mean_volume_drop_diameter(moment_3, moment_4):
    r"""
    Calculate the volume-weighted mean volume diameter \\( D_m \\) from DSD moments.

    The mean volume diameter of a drop size distribution (DSD) is computed using
    the third and fourth moments.

    The volume-weighted mean volume diameter is also referred as the mass mean diameter.
    It represents the first moment of the mass spectrum.

    Parameters
    ----------
    moment_3 : float or array-like
        The third moment of the drop size distribution, \\( M_3 \\), in units of
        [m⁻³·mm³].
    moment_4 : float or array-like
        The fourth moment of the drop size distribution, \\( M_4 \\), in units of
        [m⁻³·mm⁴].

    Returns
    -------
    D_m : float or array-like
        The mean volume diameter in millimeters (mm).

    Notes
    -----
    The mean volume diameter is calculated using the formula:

    .. math::

        D_m = \frac{M_4}{M_3}

    where:

    - \\( D_m \\) is the mean volume diameter [mm].
    - \\( M_3 \\) is the third moment of the DSD [m⁻³·mm³].
    - \\( M_4 \\) is the fourth moment of the DSD [m⁻³·mm⁴].

    Examples
    --------
    Compute the mean volume diameter from the third and fourth moments:

    >>> moment_3 = 1e6  # Example value in [m⁻³·mm³]
    >>> moment_4 = 5e6  # Example value in [m⁻³·mm⁴]
    >>> D_m = get_mean_volume_drop_diameter(moment_3, moment_4)
    >>> print(f"Mean Volume Diameter D_m: {D_m:.4f} mm")
    Mean Volume Diameter D_m: 5.0000 mm

    """
    D_m = moment_4 / moment_3  # Units: [mm⁴] / [mm³] = [mm]
    return D_m


def get_std_volume_drop_diameter(drop_number_concentration, diameter_bin_width, diameter, mean_volume_diameter):
    r"""
    Calculate the standard deviation of the mass-weighted drop diameter (σₘ).

    This parameter is often also referred as the mass spectrum standard deviation.
    It quantifies the spread or variability of DSD.

    Parameters
    ----------
    drop_number_concentration : xarray.DataArray
        The drop number concentration \\( N(D) \\) for each diameter bin, typically in units of
        number per cubic meter per millimeter (m⁻³·mm⁻¹).
    diameter : xarray.DataArray
        The equivalent volume diameters \\( D \\) of the drops in each bin, in meters (m).
    diameter_bin_width : xarray.DataArray
        The width \\( \\Delta D \\) of each diameter bin, in millimeters (mm).
    mean_volume_diameter : xarray.DataArray
        The mean volume diameter \\( D_m \\), in millimeters (mm). This is typically computed using the
        third and fourth moments or directly from the DSD.

    Returns
    -------
    sigma_m : xarray.DataArray or float
        The standard deviation of the mass-weighted drop diameter, \\( \\sigma_m \\),
        in millimeters (mm).

    Notes
    -----
    The standard deviation of the mass-weighted drop diameter is calculated using the formula:

    .. math::

        \\sigma_m = \\sqrt{\frac{\\sum [N(D) \\cdot (D - D_m)^2 \\cdot D^3
        \\cdot \\Delta D]}{\\sum [N(D) \\cdot D^3 \\cdot \\Delta D]}}

    where:

    - \\( N(D) \\) is the drop number concentration for diameter \\( D \\) [m⁻³·mm⁻¹].
    - \\( D \\) is the drop diameter [mm].
    - \\( D_m \\) is the mean volume diameter [mm].
    - \\( \\Delta D \\) is the diameter bin width [mm].
    - The numerator computes the weighted variance of diameters.
    - The weighting factor \\( D^3 \\) accounts for mass (since mass ∝ \\( D^3 \\)).

    **Physical Interpretation:**

    - A smaller \\( \\sigma_m \\) indicates that the mass is concentrated around the
      mean mass-weighted diameter, implying less variability in drop sizes.
    - A larger \\( \\sigma_m \\) suggests a wider spread of drop sizes contributing
      to the mass, indicating greater variability.

    References
    ----------
    - Smith, P. L., Johnson, R. W., & Kliche, D. V. (2019). On Use of the Standard
      Deviation of the Mass Distribution as a Parameter in Raindrop Size Distribution
      Functions. *Journal of Applied Meteorology and Climatology*, 58(4), 787-796.
      https://doi.org/10.1175/JAMC-D-18-0086.1
    - Williams, C. R., and Coauthors, 2014: Describing the Shape of Raindrop Size Distributions Using Uncorrelated
      Raindrop Mass Spectrum Parameters. J. Appl. Meteor. Climatol., 53, 1282-1296, https://doi.org/10.1175/JAMC-D-13-076.1.
    """
    const = drop_number_concentration * diameter_bin_width * diameter**3
    numerator = ((diameter * 1000 - mean_volume_diameter) ** 2 * const).sum(dim="diameter_bin_center")
    sigma_m = np.sqrt(numerator / const.sum(dim="diameter_bin_center"))
    return sigma_m


def get_median_volume_drop_diameter(drop_number_concentration, diameter, diameter_bin_width, water_density=1000):
    r"""
    Compute the median volume drop diameter (D50).

    The median volume drop diameter (D50) is defined as the diameter at which half of the total liquid water content
    is contributed by drops smaller than D50, and half by drops larger than D50.

    Drops smaller (respectively larger) than D50 contribute to half of the
    total rainwater content in the sampled volume.
    D50 is sensitive to the concentration of large drops.

    Often referred also as D50 (50 for 50 percentile of the distribution).

    Parameters
    ----------
    drop_number_concentration : xarray.DataArray
        The drop number concentration \( N(D) \) for each diameter bin, typically in units of
        number per cubic meter per millimeter (m⁻³·mm⁻¹).
    diameter : xarray.DataArray
        The equivalent volume diameters \( D \) of the drops in each bin, in meters (m).
    diameter_bin_width : xarray.DataArray
        The width \( \Delta D \) of each diameter bin, in millimeters (mm).
    water_density : float, optional
        The density of water in kg/m^3. The default is 1000 kg/m3.

    Returns
    -------
    xarray.DataArray
        Median volume drop diameter (D50) [mm].
        The drop diameter that divides the volume of water contained in the sample into two equal parts.

    """
    d50 = get_quantile_volume_drop_diameter(
        drop_number_concentration=drop_number_concentration,
        diameter=diameter,
        diameter_bin_width=diameter_bin_width,
        fraction=0.5,
        water_density=water_density,
    )
    return d50


def get_quantile_volume_drop_diameter(
    drop_number_concentration,
    diameter,
    diameter_bin_width,
    fraction,
    water_density=1000,
):
    r"""
    Compute the diameter corresponding to a specified fraction of the cumulative liquid water content (LWC).

    This function calculates the diameter \( D_f \) at which the cumulative LWC reaches
    a specified fraction \( f \) of the total LWC for each drop size distribution (DSD).
    When \( f = 0.5 \), it computes the median volume drop diameter.


    Parameters
    ----------
    drop_number_concentration : xarray.DataArray
        The drop number concentration \( N(D) \) for each diameter bin, typically in units of
        number per cubic meter per millimeter (m⁻³·mm⁻¹).
    diameter : xarray.DataArray
        The equivalent volume diameters \( D \) of the drops in each bin, in meters (m).
    diameter_bin_width : xarray.DataArray
        The width \( \Delta D \) of each diameter bin, in millimeters (mm).
    fraction : float
        The fraction \( f \) of the total liquid water content to compute the diameter for.
        Default is 0.5, which computes the median volume diameter (D50).
        For other percentiles, use 0.1 for D10, 0.9 for D90, etc. Must be between 0 and 1 (exclusive).
    water_density : float, optional
        The density of water in kg/m^3. The default is 1000 kg/m3.

    Returns
    -------
    D_f : xarray.DataArray
        The diameter \( D_f \) corresponding to the specified fraction \( f \) of cumulative LWC,
        in millimeters (mm). For `fraction=0.5`, this is the median volume drop diameter D50.

    Notes
    -----
    The calculation involves computing the cumulative sum of the liquid water content
    contributed by each diameter bin and finding the diameter at which the cumulative
    sum reaches the specified fraction \( f \) of the total liquid water content.

    Linear interpolation is used between the two diameter bins where the cumulative LWC
    crosses the target LWC fraction.

    """
    # Check fraction
    if not (0 < fraction < 1):
        raise ValueError("Fraction must be between 0 and 1 (exclusive)")

    # Convert water density from kg/m3 to g/m3
    water_density = water_density * 1000

    # Compute LWC per diameter bin [g/m3]
    lwc_per_diameter = np.pi / 6.0 * water_density * (diameter**3 * drop_number_concentration * diameter_bin_width)

    # Compute rain rate per diameter [mm/hr]
    # rain_rate_per_diameter = np.pi / 6 * (
    # (drop_number_concentration * velocity * diameter**3 * diameter_bin_width) * 3600 * 1000
    # )

    # Compute the cumulative sum of LWC along the diameter bins
    cumulative_lwc = lwc_per_diameter.cumsum(dim="diameter_bin_center")

    # ------------------------------------------------------.
    # Retrieve total lwc and target lwc
    total_lwc = cumulative_lwc.isel(diameter_bin_center=-1)
    target_lwc = total_lwc * fraction

    # Retrieve idx half volume is reached
    # --> If all NaN or False, argmax and _get_last_xr_valid_idx(fill_value=0) return 0 !
    idx_upper = (cumulative_lwc >= target_lwc).argmax(dim="diameter_bin_center")
    idx_lower = _get_last_xr_valid_idx(
        da_condition=(cumulative_lwc <= target_lwc),
        dim="diameter_bin_center",
        fill_value=0,
    )

    # Define mask when fraction fall exactly at a diameter bin center
    # - Also related to the case of only values in the first bin.
    solution_is_bin_center = idx_upper == idx_lower

    # Define diameter increment from lower bin center
    y1 = cumulative_lwc.isel(diameter_bin_center=idx_lower)
    y2 = cumulative_lwc.isel(diameter_bin_center=idx_upper)
    yt = target_lwc
    d1 = diameter.isel(diameter_bin_center=idx_lower)  # m
    d2 = diameter.isel(diameter_bin_center=idx_upper)  # m
    d_increment = (d2 - d1) * (yt - y1) / (y2 - y1)

    # Define quantile diameter
    d = xr.where(solution_is_bin_center, d1, d1 + d_increment)

    # Set NaN where total sum is 0 or all NaN
    mask_invalid = np.logical_or(total_lwc == 0, np.isnan(total_lwc))
    d = d.where(~mask_invalid)

    # Convert diameter to mm
    d = d * 1000

    return d


####-----------------------------------------------------------------------------------------------------
#### Normalized Gamma Parameters


def get_normalized_intercept_parameter(liquid_water_content, mean_volume_diameter, water_density=1000):
    r"""
    Calculate the normalized intercept parameter \\( N_w \\) of the drop size distribution.

    A higher \\( N_w \\) indicates a higher concentration of smaller drops.
    The \\( N_w \\) is used in models to represent the DSD when assuming a normalized gamma distribution.

    Parameters
    ----------
    liquid_water_content : float or array-like
        Liquid water content \\( LWC \\) in grams per cubic meter (g/m³).
    mean_volume_diameter : float or array-like
        Mean volume diameter \\( D_m \\) in millimeters (mm).
    water_density : float, optional
        Density of water \\( \rho_w \\) in kilograms per cubic meter (kg/m³).
        The default is 1000 kg/m³.

    Returns
    -------
    Nw : xarray.DataArray or float
        Normalized intercept parameter \\( N_w \\) in units of m⁻3·mm⁻¹.

    Notes
    -----
    The normalized intercept parameter \\( N_w \\) is calculated using the formula:

    .. math::

        N_w = \frac{256}{\\pi \rho_w} \\cdot \frac{W}{D_m^4}

    where:

    - \\( N_w \\) is the normalized intercept parameter.
    - \\( W \\) is the liquid water content in g/m³.
    - \\( D_m \\) is the mean volume diameter in mm.
    - \\( \rho_w \\) is the density of water in kg/m³.
    """
    # Conversion to g/m3
    water_density = water_density * 1000  # g/m3

    # Compute Nw
    # --> 1e9 is used to convert from mm-4 to m-3 mm-1
    # - 256 = 4**4
    # - lwc = (np.pi * water_density / 6) * moment_3
    Nw = (256.0 / (np.pi * water_density)) * liquid_water_content / mean_volume_diameter**4 * 1e9
    return Nw


def get_mom_normalized_intercept_parameter(moment_3, moment_4):
    r"""
    Calculate the normalized intercept parameter \\( N_w \\) of the drop size distribution.

    moment_3 : float or array-like
        The third moment of the drop size distribution, \\( M_3 \\), in units of
        [m⁻³·mm³] (number per cubic meter times diameter cubed).

    moment_4 : float or array-like
        The foruth moment of the drop size distribution, \\( M_3 \\), in units of
        [m⁻³·mm4].

    Returns
    -------
    Nw : xarray.DataArray or float
        Normalized intercept parameter \\( N_w \\) in units of m⁻3·mm⁻¹.

    References
    ----------
    Testud, J., S. Oury, R. A. Black, P. Amayenc, and X. Dou, 2001:
    The Concept of “Normalized” Distribution to Describe Raindrop Spectra:
    A Tool for Cloud Physics and Cloud Remote Sensing.
    J. Appl. Meteor. Climatol., 40, 1118-1140,
    https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2

    """
    Nw = 256 / 6 * moment_3**5 / moment_4**4
    return Nw


####--------------------------------------------------------------------------------------------------------
#### Kinetic Energy Parameters


def get_min_max_drop_kinetic_energy(drop_number, diameter, velocity, water_density=1000):
    r"""
    Calculate the minimum and maximum kinetic energy of raindrops in a drop size distribution (DSD).

    This function computes the kinetic energy of individual raindrops based on their diameters and
    fall velocities and returns the minimum and maximum values among these drops for each time step.

    Parameters
    ----------
    drop_number : xarray.DataArray
        The number of drops in each diameter (and velocity, if available) bin(s).
    diameter : xarray.DataArray
        The equivalent volume diameters \\( D \\) of the drops in each bin, in meters (m).
    velocity : xarray.DataArray or float
        The fall velocities \\( v \\) of the drops in each bin, in meters per second (m/s).
    water_density : float, optional
        The density of water \\( \rho_w \\) in kilograms per cubic meter (kg/m³).
        Default is 1000 kg/m³.

    Returns
    -------
    min_drop_kinetic_energy : xarray.DataArray
        The minimum kinetic energy among the drops present in the DSD, in joules (J).
    max_drop_kinetic_energy : xarray.DataArray
        The maximum kinetic energy among the drops present in the DSD, in joules (J).

    Notes
    -----
    The kinetic energy \\( KE \\) of an individual drop is calculated using:

    .. math::

        KE = \frac{1}{2} \\cdot m \\cdot v^2

    where:

    - \\( m \\) is the mass of the drop, calculated as:

      .. math::

          m = \frac{\\pi}{6} \\cdot \rho_w \\cdot D^3

      with \\( D \\) being the drop diameter.

    - \\( v \\) is the fall velocity of the drop.
    """
    # Ensure velocity is 2D (diameter, velocity)
    velocity = xr.ones_like(drop_number) * velocity

    # # Compute the mass of each drop: m = (π/6) * rho_w * D^3
    # mass = (np.pi / 6) * water_density * diameter**3  # Units: kg

    # # Compute kinetic energy: KE = 0.5 * m * v^2
    # ke = 0.5 * mass * velocity**2  # Units: J

    # Compute kinetic energy
    ke = 1 / 12 * water_density * np.pi * diameter**3 * velocity**2

    # Select kinetic energies where drops are present
    ke = ke.where(drop_number > 0)

    # Compute min, mean and maximum drop kinetic energy
    max_drop_kinetic_energy = ke.max(dim=_get_spectrum_dims(ke))
    min_drop_kinetic_energy = ke.min(dim=_get_spectrum_dims(ke))
    return min_drop_kinetic_energy, max_drop_kinetic_energy


def get_kinetic_energy_density_flux(
    drop_number,
    diameter,
    velocity,
    sampling_area,
    sample_interval,
    water_density=1000,
):
    r"""
    Calculate the kinetic energy flux density (KE) of rainfall over time.

    This function computes the total kinetic energy of raindrops passing through the sensor's sampling area
    per unit time and area, resulting in the kinetic energy flux density
    in joules per square meter per hour (J·m⁻²·h⁻¹).

    Typical values range between 0 and 5000 J·m⁻²·h⁻¹ .
       KE = E * R

    Parameters
    ----------
    drop_number : xarray.DataArray
        The number of drops in each diameter (and velocity, if available) bin(s).
    diameter : xarray.DataArray
        The equivalent volume diameters \\( D \\) of the drops in each bin, in meters (m).
    velocity : xarray.DataArray or float
        The fall velocities \\( v \\) of the drops in each bin, in meters per second (m/s).
        Values are broadcasted to match the dimensions of `drop_number`.
    sampling_area : float
        The effective sampling area \\( A \\) of the sensor in square meters (m²).
    sample_interval : float
        The time over which the drops are counted \\( \\Delta t \\) in seconds (s).
    water_density : float, optional
        The density of water \\( \rho_w \\) in kilograms per cubic meter (kg/m³).
        Default is 1000 kg/m³.

    Returns
    -------
    kinetic_energy_flux : xarray.DataArray
        The kinetic energy flux density of rainfall in joules per square meter per hour (J·m⁻²·h⁻¹).
        Dimensions are reduced to ('time',).

    Notes
    -----
    The kinetic energy flux density \\( KE \\) is calculated using:

    .. math::

        KE = \frac{1}{2} \\cdot \frac{\rho_w \\pi}{6} \\cdot \frac{1}{\\Delta t} \\cdot 3600 \\cdot \\sum_{i,j}
        \\left( \frac{n_{ij} \\cdot D_i^3 \\cdot v_j^2}{A} \right)

    where:

    - \\( n_{ij} \\) is the number of drops in diameter bin \\( i \\) and velocity bin \\( j \\).
    - \\( D_i \\) is the diameter of bin \\( i \\).
    - \\( v_j \\) is the velocity of bin \\( j \\).
    - \\( A \\) is the sampling area.
    - \\( \\Delta t \\) is the time integration period in seconds.
    - The factor \\( 3600 \\) converts the rate to per hour.

    """
    # Ensure velocity is 2D (diameter, velocity)
    velocity = xr.ones_like(drop_number) * velocity

    # # Compute rain drop kinetic energy [J]
    # ke = 0.5 *  water_density * np.pi / 6 * diameter **3 * velocity**2
    # # Compute total kinetic energy in [J / m2]
    # total_kinetic_energy =  (ke * drop_number / sampling_area).sum(dim=["diameter_bin_center", "velocity_bin_center"])
    # # Compute kinetic energy density flux (KE) (J/m2/h)
    # kinetic_energy_flux = total_kinetic_energy / sample_interval * 3600

    # Compute kinetic energy flux density (KE) (J/m2/h)
    kinetic_energy_flux = (
        water_density
        * np.pi
        / 12
        / sample_interval
        * 3600
        * ((drop_number * diameter**3 * velocity**2) / sampling_area).sum(
            dim=_get_spectrum_dims(drop_number),
        )
    )
    return kinetic_energy_flux


def get_rainfall_kinetic_energy(drop_number, diameter, velocity, rain_accumulation, sampling_area, water_density=1000):
    r"""
    Calculate the kinetic energy per unit rainfall depth (E) in joules per square meter per millimeter (J·m⁻²·mm⁻¹).

    This function computes the kinetic energy of the rainfall per millimeter of rain, providing a measure of the
    energy associated with each unit of rainfall depth. This parameter is useful for understanding the potential
    impact of raindrop erosion and the intensity of rainfall events.

    The values typically range between 0 and 40 J·m⁻²·mm⁻¹.
    E is related to the kinetic energy flux density (KE) by the rain rate: E = KE/R .

    Parameters
    ----------
    drop_number : xarray.DataArray
        The number of drops in each diameter (and velocity, if available) bin(s).
    diameter : xarray.DataArray
        The equivalent volume diameters \\( D \\) of the drops in each bin, in meters (m).
    velocity : xarray.DataArray or float
        The fall velocities \\( v \\) of the drops in each bin, in meters per second (m/s).
        Values are broadcasted to match the dimensions of `drop_number`.
    rain_accumulation : xarray.DataArray or float
        The total rainfall accumulation over the time integration period, in millimeters (mm).
    sampling_area : float
        The effective sampling area \\( A \\) of the sensor in square meters (m²).
    water_density : float, optional
        The density of water \\( \rho_w \\) in kilograms per cubic meter (kg/m³).
        Default is 1000 kg/m³.

    Returns
    -------
    E : xarray.DataArray
        The kinetic energy per unit rainfall depth in joules per square meter per millimeter (J·m⁻²·mm⁻¹).
        Dimensions are reduced to ('time',).

    Notes
    -----
    The kinetic energy per unit rainfall depth \\( E \\) is calculated using:

    .. math::

        E = \frac{1}{2} \\cdot \frac{\\pi}{6} \\cdot \frac{\rho_w}{R} \\cdot \\sum_{i,j}
        \\left( \frac{n_{ij} \\cdot D_i^3 \\cdot v_j^2}{A} \right)

    where:

    - \\( n_{ij} \\) is the number of drops in diameter bin \\( i \\) and velocity bin \\( j \\).
    - \\( D_i \\) is the diameter of bin \\( i \\).
    - \\( v_j \\) is the velocity of bin \\( j \\).
    - \\( A \\) is the sampling area.
    - \\( R \\) is the rainfall accumulation over the integration period (mm).
    """
    # Ensure velocity has the same dimensions as drop_number
    velocity = xr.ones_like(drop_number) * velocity
    # Compute rainfall kinetic energy per unit rainfall depth
    E = (
        0.5
        * np.pi
        / 6
        * water_density
        / rain_accumulation
        * ((drop_number * diameter**3 * velocity**2) / sampling_area).sum(
            dim=_get_spectrum_dims(drop_number),
        )
    )
    return E
