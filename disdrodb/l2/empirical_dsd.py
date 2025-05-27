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
"""Functions for computation of DSD parameters.

The functions of this module expects xarray.DataArray objects as input.
Zeros and NaN values input arrays are correctly processed.
Infinite values should be removed beforehand or otherwise are propagated throughout the computations.
"""
import numpy as np
import xarray as xr

from disdrodb import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.api.checks import check_sensor_name
from disdrodb.utils.xarray import (
    remove_diameter_coordinates,
    remove_velocity_coordinates,
    xr_get_last_valid_idx,
)


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
    return np.pi / 6 * diameter**3  # 1/6 = 4/3*(0.5**3)


def get_drop_average_velocity(drop_number):
    r"""
    Calculate the drop average velocity \\( v_m(D))) \\) per diameter class.

    The average velocity is obtained by weighting by the number of drops in each velocity bin.
    If in a given diameter bin no drops are recorded, the resulting average drop size velocity for
    such bin will be set to NaN.

    Parameters
    ----------
    drop_number : xarray.DataArray
        Array of drop counts \\( n(D,v) \\) per diameter (and velocity, if available) bins
        over the time integration period.
        The DataArray must have the ``velocity_bin_center`` coordinate.

    Returns
    -------
    average_velocity : xarray.DataArray
        Array of drop average velocity \\( v_m(D))) \\) in m·s⁻¹ .
        At timesteps with zero drop counts, it returns NaN.
    """
    velocity = xr.ones_like(drop_number) * drop_number["velocity_bin_center"]
    average_velocity = ((velocity * drop_number).sum(dim=VELOCITY_DIMENSION, skipna=False)) / drop_number.sum(
        dim=VELOCITY_DIMENSION,
        skipna=False,
    )
    return average_velocity


def count_bins_with_drops(ds):
    """Count the number of diameter bins with data."""
    # Select useful variable
    candidate_variables = ["drop_counts", "drop_number_concentration", "drop_number"]
    available_variables = [var for var in candidate_variables if var in ds]
    if len(available_variables) == 0:
        raise ValueError(f"One of these variables is required: {candidate_variables}")
    da = ds[available_variables[0]]
    if VELOCITY_DIMENSION in da.dims:
        da = da.sum(dim=VELOCITY_DIMENSION)
    # Count number of bins with data
    da = (da > 0).sum(dim=DIAMETER_DIMENSION)
    # TODO: remove this in future !
    if "velocity_method" in da.dims:
        da = da.max(dim="velocity_method")
    return da


def _compute_qc_bins_metrics(arr):
    # Find indices of non-zero elements
    arr = arr.copy()
    arr[np.isnan(arr)] = 0
    non_zero_indices = np.nonzero(arr)[0]
    if non_zero_indices.size == 0:
        return np.array([0, len(arr), 1, len(arr)])

    # Define bins interval with drops
    start_idx, end_idx = non_zero_indices[0], non_zero_indices[-1]
    segment = arr[start_idx : end_idx + 1]

    # Compute number of bins with drops
    total_bins = segment.size

    # Compute number of missing bins (zeros)
    n_missing_bins = int(np.sum(segment == 0))

    # Compute fraction of bins with missing drops
    fraction_missing = n_missing_bins / total_bins

    # Identify longest with with consecutive zeros
    zero_mask = (segment == 0).astype(int)
    # - Pad with zeros at both ends to detect edges
    padded = np.pad(zero_mask, (1, 1), "constant", constant_values=0)
    diffs = np.diff(padded)
    # - Start and end indices of runs
    run_starts = np.where(diffs == 1)[0]
    run_ends = np.where(diffs == -1)[0]
    run_lengths = run_ends - run_starts
    max_consecutive_missing = run_lengths.max() if run_lengths.size > 0 else 0

    # Define output
    output = np.array([total_bins, n_missing_bins, fraction_missing, max_consecutive_missing])
    return output


def compute_qc_bins_metrics(ds):
    """
    Compute quality-control metrics for drop-count bins along the diameter dimension.

    This function selects the first available drop-related variable from the dataset,
    optionally collapses over velocity methods and the velocity dimension, then
    computes four metrics per time step:

      1. Nbins: total number of diameter bins between the first and last non-zero count
      2. Nbins_missing: number of bins with zero or NaN counts in that interval
      3. Nbins_missing_fraction: fraction of missing bins (zeros) in the interval
      4. Nbins_missing_consecutive: maximum length of consecutive missing bins

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing one of the following variables:
        'drop_counts', 'drop_number_concentration', or 'drop_number'.
        If a 'velocity_method' dimension exists, only the first method is used.
        If a velocity dimension (specified by VELOCITY_DIMENSION) exists, it is summed over.

    Returns
    -------
    xr.Dataset
        Dataset with a new 'metric' dimension of size 4 and coordinates:
        ['Nbins', 'Nbins_missing', 'Nbins_missing_fraction', 'Nbins_missing_consecutive'],
        indexed by 'time'.
    """
    # Select useful variable
    candidate_variables = ["drop_counts", "drop_number_concentration", "drop_number"]
    available_variables = [var for var in candidate_variables if var in ds]
    if len(available_variables) == 0:
        raise ValueError(f"One of these variables is required: {candidate_variables}")
    da = ds[available_variables[0]]
    if "velocity_method" in da.dims:
        da = da.isel(velocity_method=0)
        da = da.drop_vars("velocity_method")
    if VELOCITY_DIMENSION in da.dims:
        da = da.sum(dim=VELOCITY_DIMENSION)

    # Compute QC metrics
    da_qc_bins = xr.apply_ufunc(
        _compute_qc_bins_metrics,
        da,
        input_core_dims=[[DIAMETER_DIMENSION]],
        output_core_dims=[["metric"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"metric": 4}},
    )

    # Assign meaningful labels to the qc 'metric' dimension
    variables = ["Nbins", "Nbins_missing", "Nbins_missing_fraction", "Nbins_missing_consecutive"]
    ds_qc_bins = da_qc_bins.assign_coords(metric=variables).to_dataset(dim="metric")
    return ds_qc_bins


####-------------------------------------------------------------------------------------------------------------------.
#### DSD Spectrum, Concentration, Moments


def get_effective_sampling_area(sensor_name, diameter):
    """Compute the effective sampling area in m2 of the disdrometer.

    The diameter must be provided in meters !
    """
    check_sensor_name(sensor_name)
    if sensor_name in ["PARSIVEL", "PARSIVEL2"]:
        # Calculate sampling area for each diameter bin (S_i)
        L = 180 / 1000  # Length of the Parsivel beam in m (180 mm)
        B = 30 / 1000  # Width of the Parsivel beam in m (30mm)
        sampling_area = L * (B - diameter / 2)
        return sampling_area
    if sensor_name == "LPM":
        # Calculate sampling area for each diameter bin (S_i)
        L = 228 / 1000  # Length of the Parsivel beam in m (228 mm)
        B = 20 / 1000  # Width of the Parsivel beam in m (20 mm)
        sampling_area = L * (B - diameter / 2)
        return sampling_area
    if sensor_name == "PWS100":
        sampling_area = 0.004  # m2  # TODO: L * (B - diameter / 2) ?
        return sampling_area
    if sensor_name == "RD80":
        sampling_area = 0.005  # m2
        return sampling_area
    raise NotImplementedError(f"Effective sampling area for {sensor_name} must yet to be specified in the software.")


def get_bin_dimensions(xr_obj):
    """Return the dimensions of the drop spectrum."""
    return sorted([k for k in [DIAMETER_DIMENSION, VELOCITY_DIMENSION] if k in xr_obj.dims])


def get_drop_number_concentration(drop_number, velocity, diameter_bin_width, sampling_area, sample_interval):
    r"""
    Calculate the volumetric drop number concentration \\( N(D) \\) per diameter class.

    Computes the drop number concentration \\( N(D) \\) [m⁻³·mm⁻¹] for each diameter
    class based on the measured drop counts and sensor parameters.
    This represents the number of drops per unit volume per unit diameter interval.
    It is also referred to as the drop size distribution N(D) per cubic metre per millimetre [m-3 mm-1]

    Parameters
    ----------
    velocity : xarray.DataArray
        Array of drop fall velocities \\( v(D) \\) corresponding to each diameter bin in meters per second (m/s).
        Typically the estimated fall velocity is used.
        But one can also pass the velocity bin center of the optical disdrometer, which get broadcasted
        along the diameter bin dimension.
    diameter_bin_width : xarray.DataArray
        Width of each diameter bin \\( \\Delta D \\) in millimeters (mm).
    drop_number : xarray.DataArray
        Array of drop counts \\(  n(D) or n(D,v) \\) per diameter (and velocity if available)
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
    if VELOCITY_DIMENSION in drop_number.dims:
        drop_number_concentration = (drop_number / velocity).sum(dim=VELOCITY_DIMENSION, skipna=False) / (
            sampling_area * diameter_bin_width * sample_interval
        )
    # - For impact disdrometers
    else:
        drop_number_concentration = (drop_number / velocity) / (sampling_area * diameter_bin_width * sample_interval)
    return drop_number_concentration


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
    total_number_concentration = (drop_number_concentration * diameter_bin_width).sum(
        dim=DIAMETER_DIMENSION,
        skipna=False,
    )
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
    return ((diameter * 1000) ** moment * drop_number_concentration * diameter_bin_width).sum(
        dim=DIAMETER_DIMENSION,
        skipna=False,
    )


####------------------------------------------------------------------------------------------------------------------
#### Rain Rate and Accumulation


def get_rain_rate_from_drop_number(drop_number, sampling_area, diameter, sample_interval):
    r"""
    Compute the rain rate \\( R \\) [mm/h] based on the drop size distribution and drop velocities.

    This function calculates the rain rate by integrating over the drop size distribution (DSD),
    considering the volume of water falling per unit time and area. It uses the number of drops
    counted in each diameter class, the effective sampling area of the sensor, the diameters of the
    drops, and the time interval over which the drops are counted.

    Parameters
    ----------
    drop_number : xarray.DataArray
        Array representing the number of drops per diameter class
        and, optionally, velocity class \\( n(D, (v)) \\).
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

        R = \frac{\\pi}{6} \times 10^{3} \times 3600 \times
            \\sum_{\text{bins}} n(D) \cdot A(D) \cdot D^3 \cdot \\Delta t

          = \\pi \times 0.6 \times 10^{6} \times
            \\sum_{\text{bins}} n(D) \cdot A(D) \cdot D^3 \cdot \\Delta t

           = \\pi \times 6 \times 10^{5} \times
             \\sum_{\text{bins}} n(D) \cdot A(D) \cdot D^3 \cdot \\Delta t

    Where:
    - \\( n(D) \\) is the number of drops in each diameter class.
    - \\( A(D) \\) is the effective sampling area.
    - \\( D \\) is the drop diameter.
    - \\( \\Delta t \\) is the time interval for drop counts.

    This formula incorporates a conversion factor to express the rain rate in millimeters per hour.

    In the literature, when the diameter is expected in millimeters, the formula is given
    as:
    .. math::

        R = \\pi  \times {6} \times 10^{-4}  \times
        \\sum_{\text{bins}} n(D) \cdot A(D) \cdot D^3 \cdot \\Delta t

    """
    dim = get_bin_dimensions(drop_number)
    rain_rate = (
        np.pi
        / 6
        / sample_interval
        * (drop_number * (diameter**3 / sampling_area)).sum(dim=dim, skipna=False)
        * 3600
        * 1000
    )
    return rain_rate


def get_rain_rate(drop_number_concentration, velocity, diameter, diameter_bin_width):
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
    if VELOCITY_DIMENSION in velocity.dims:
        raise ValueError(f"The 'velocity' DataArray must not have the {VELOCITY_DIMENSION} dimension.")

    rain_rate = (
        6
        * np.pi
        * 1e5
        * (drop_number_concentration * (velocity * diameter**3 * diameter_bin_width)).sum(
            dim=DIAMETER_DIMENSION,
            skipna=False,
        )
    )
    return rain_rate


def get_rain_rate_spectrum(drop_number_concentration, velocity, diameter):
    r"""
    Compute the rain rate per diameter class.

    It represents the rain rate as a function of raindrop diameter.
    The total rain rate can be obtained by multiplying the spectrum with
    the diameter bin width and summing over the diameter bins.

    Parameters
    ----------
    drop_number_concentration : xarray.DataArray
        Array of drop number concentrations \\( N(D) \\) in m⁻³·mm⁻¹.
    velocity : xarray.DataArray
        Array of drop fall velocities \\( v(D) \\) corresponding to each diameter bin in meters per second (m/s).
    diameter : xarray.DataArray
        Array of drop diameters \\( D \\) in meters (m).

    Returns
    -------
    xarray.DataArray
        The rain rate spectrum in millimeters per hour per mm, representing the volume
        of water falling per unit area per unit time per unit diameter.

    """
    rain_rate = 6 * np.pi * 1e5 * (drop_number_concentration * (velocity * diameter**3))
    return rain_rate


def get_rain_rate_contribution(drop_number_concentration, velocity, diameter, diameter_bin_width):
    r"""Compute the rain rate contribution per diameter class.

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
    xarray.DataArray
        The rain rate contribution percentage per diameter class.

    """
    rain_rate_spectrum = (6 * np.pi * 1e5 * (velocity * diameter**3 * diameter_bin_width)) * drop_number_concentration

    rain_rate_total = rain_rate_spectrum.sum(dim=DIAMETER_DIMENSION, skipna=False)
    rain_rate_contribution = rain_rate_spectrum / rain_rate_total * 100
    return rain_rate_contribution


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


####------------------------------------------------------------------------------------------------------------------
#### Reflectivity
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
    z = (drop_number_concentration * ((diameter * 1000) ** 6 * diameter_bin_width)).sum(
        dim=DIAMETER_DIMENSION,
        skipna=False,
    )
    invalid_mask = z > 0
    z = z.where(invalid_mask)
    # Compute equivalent reflectivity factor in dBZ
    # - np.log10(np.nan) returns -Inf !
    # --> We mask again after the log
    Z = 10 * np.log10(z)
    Z = Z.where(invalid_mask)
    # Clip reflectivity at -60 dBZ
    Z = Z.clip(-60, None)
    return Z


def get_equivalent_reflectivity_spectrum(drop_number_concentration, diameter):
    r"""
    Compute the equivalent reflectivity per diameter class.

    The equivalent reflectivity per unit diameter Z(D) [in mm⁶·m⁻³ / mm] is expressed in decibels
    using the formula:

    .. math::

        Z(D) = 10 \cdot \log_{10}(z(D))

    where \\( z(D) \\) is the equivalent reflectivity spectrum in linear units of the DSD.

    To convert back the reflectivity factor to linear units (mm⁶·m⁻³ / mm), use the formula:

    .. math::

        z(D) = 10^{(Z(D)/10)}

    To obtain the total equivalent reflectivity factor (z) one has to multiply z(D) with the diameter
    bins intervals and summing over the diameter bins.

    Parameters
    ----------
    drop_number_concentration : xarray.DataArray
        Array representing the concentration of droplets per diameter class in number per unit volume.
    diameter : xarray.DataArray
        Array of droplet diameters in meters (m).

    Returns
    -------
    xarray.DataArray
        The equivalent reflectivity spectrum in decibels (dBZ).

    """
    # Compute reflectivity in mm⁶·m⁻³
    z = drop_number_concentration * ((diameter * 1000) ** 6)
    invalid_mask = z > 0
    z = z.where(invalid_mask)
    # Compute equivalent reflectivity factor in dBZ
    # - np.log10(np.nan) returns -Inf !
    # --> We mask again after the log
    Z = 10 * np.log10(z)
    Z = Z.where(invalid_mask)
    # Clip reflectivity at -60 dBZ
    Z = Z.clip(-60, None)
    return Z


####------------------------------------------------------------------------------------------------------------------
#### Liquid Water Content / Mass Parameters


def get_liquid_water_spectrum(drop_number_concentration, diameter, water_density=1000):
    """
    Calculate the mass spectrum W(D) per diameter class.

    It represents the mass of liquid water as a function of raindrop diameter.
    The integrated liquid water content can be obtained by multiplying
    the spectrum with the diameter bins intervals and summing over the diameter bins.

    Parameters
    ----------
    drop_number_concentration : array-like
        The concentration of droplets (number of droplets per unit volume) in each diameter bin.
    diameter : array-like
        The diameters of the droplets for each bin, in meters (m).

    Returns
    -------
    array-like
        The calculated rain drop mass spectrum in grams per cubic meter per unit diameter (g/m3/mm).

    """
    # Convert water density from kg/m3 to g/m3
    water_density = water_density * 1000

    # Calculate the mass spectrum (LWC per diameter bin)
    return (np.pi / 6.0 * water_density * diameter**3) * drop_number_concentration  #  [g/m3 mm-1]


# def get_mass_flux(drop_number_concentration, diameter, velocity, diameter_bin_width, water_density=1000):
#     """
#     Calculate the mass flux based on drop number concentration and drop diameter and velocity.

#     Parameters
#     ----------
#     drop_number_concentration : array-like
#         The concentration of droplets (number of droplets per unit volume) in each diameter bin.
#     diameter : array-like
#         The diameters of the droplets for each bin, in meters (m).
#     water_density : float, optional
#         The density of water in kg/m^3. The default is 1000 kg/m3.

#     Returns
#     -------
#     array-like
#         The calculated mass in grams per cubic meter per second (g/m3/s).

#     """
# if VELOCITY_DIMENSION in velocity.dims:
#     raise ValueError("The 'velocity' DataArray must not have the {VELOCITY_DIMENSION} dimension.")
# # Convert water density from kg/m3 to g/m3
# water_density = water_density * 1000

# # Calculate the volume constant for the water droplet formula
# vol_constant = np.pi / 6.0 * water_density

# # Calculate the mass flux
# # TODO: check equal to density * R
# mass_flux = (vol_constant *
#          (drop_number_concentration *
#           (diameter**3 * velocity)).sum(dim=DIAMETER_DIMENSION, skipna=False)
#  )
# return mass_flux


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

    # Calculate the liquid water content
    lwc = (
        np.pi
        / 6.0
        * water_density
        * (drop_number_concentration * (diameter**3 * diameter_bin_width)).sum(
            dim=DIAMETER_DIMENSION,
            skipna=False,
        )
    )
    return lwc


def get_liquid_water_content_from_moments(moment_3, water_density=1000):
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
#### Diameter Statistics


def get_min_max_diameter(drop_counts):
    """
    Get the minimum and maximum diameters where drop_counts is non-zero.

    Parameters
    ----------
    drop_counts : xarray.DataArray
        Drop counts with dimensions ("time", "diameter_bin_center") and
        coordinate "diameter_bin_center".
        It assumes the diameter coordinate to be monotonically increasing !

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
    first_non_zero_idx = non_zero_mask.argmax(dim=DIAMETER_DIMENSION)

    # Calculate the last non-zero index in the original array
    last_non_zero_idx = xr_get_last_valid_idx(da_condition=non_zero_mask, dim=DIAMETER_DIMENSION)

    # Get the 'diameter_bin_center' coordinate
    diameters = drop_counts["diameter_bin_center"]

    # Retrieve the diameters corresponding to the first and last non-zero indices
    min_drop_diameter = diameters.isel({DIAMETER_DIMENSION: first_non_zero_idx.astype(int)})
    max_drop_diameter = diameters.isel({DIAMETER_DIMENSION: last_non_zero_idx.astype(int)})

    # Identify time steps where all drop_counts are zero
    is_all_zero_or_nan = ~non_zero_mask.any(dim=DIAMETER_DIMENSION)

    # Mask with NaN where no drop or all values are NaN
    min_drop_diameter = min_drop_diameter.where(~is_all_zero_or_nan)
    max_drop_diameter = max_drop_diameter.where(~is_all_zero_or_nan)

    # Remove diameter coordinates
    min_drop_diameter = remove_diameter_coordinates(min_drop_diameter)
    max_drop_diameter = remove_diameter_coordinates(max_drop_diameter)

    return min_drop_diameter, max_drop_diameter


def get_mode_diameter(drop_number_concentration, diameter):
    """Get raindrop diameter with highest occurrence."""
    # If all NaN, set to 0 otherwise argmax fail when all NaN data
    idx_all_nan_mask = np.isnan(drop_number_concentration).all(dim=DIAMETER_DIMENSION)
    drop_number_concentration = drop_number_concentration.where(~idx_all_nan_mask, 0)
    # Find index where all 0
    # --> argmax will return 0
    idx_all_zero = (drop_number_concentration == 0).all(dim=DIAMETER_DIMENSION)
    # Find the diameter index corresponding the "mode"
    idx_observed_mode = drop_number_concentration.argmax(dim=DIAMETER_DIMENSION)
    # Find the diameter corresponding to the "mode"
    diameter_mode = diameter.isel({DIAMETER_DIMENSION: idx_observed_mode})
    # Remove diameter coordinates
    diameter_mode = remove_diameter_coordinates(diameter_mode)
    # Set to np.nan where data where all NaN or all 0
    idx_mask = np.logical_or(idx_all_nan_mask, idx_all_zero)
    diameter_mode = diameter_mode.where(~idx_mask)
    return diameter_mode


####-------------------------------------------------------------------------------------------------------------------.
#### Mass Distribution Diameters


def get_mean_volume_drop_diameter(moment_3, moment_4):
    r"""
    Calculate the volume-weighted mean volume diameter \\( D_m \\) from DSD moments.

    The mean volume diameter of a drop size distribution (DSD) is computed using
    the third and fourth moments.

    The volume-weighted mean volume diameter is also referred as the mass mean diameter.
    It represents the first moment of the mass spectrum.

    If no drops are recorded, the output values is NaN.

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
    # Note:
    # - 0/0 return NaN
    # - <number>/0 return Inf
    D_m = moment_4 / moment_3  # Units: [mm⁴] / [mm³] = [mm]
    return D_m


def get_std_volume_drop_diameter(moment_3, moment_4, moment_5):
    r"""
    Calculate the standard deviation of the mass-weighted drop diameter (σₘ).

    This parameter is often also referred as the mass spectrum standard deviation.
    It quantifies the spread or variability of DSD.

    If drops are recorded in just one bin, the standard deviation of the mass-weighted drop diameter
    is set to 0.
    If no drops are recorded, the output values is NaN.

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
    # # Full formula
    # const = drop_number_concentration * diameter_bin_width * diameter**3
    # numerator = ((diameter * 1000 - mean_volume_diameter) ** 2 * const).sum(dim=DIAMETER_DIMENSION, skipna=False)
    # variance_m = numerator / const.sum(dim=DIAMETER_DIMENSION, skipna=False))

    # Compute variance using moment formula
    variance_m = (moment_3 * moment_5 - moment_4**2) / moment_3**2

    # Set to 0 when very low values (resulting from numerical errors)
    # --> For example should return 0 when drops only recorded in 1 bin !
    variance_m = xr.where(np.logical_and(variance_m < 1e-5, ~np.isnan(variance_m)), 0, variance_m)

    # Compute standard deviation
    sigma_m = np.sqrt(variance_m)
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


def _get_quantile_volume_drop_diameter(
    drop_number_concentration,
    diameter,
    diameter_bin_width,
    fraction,
    water_density=1000,
):
    # Check fraction value(s)
    fraction = np.atleast_1d(fraction)
    for value in fraction:
        if not (0 < value < 1):
            raise ValueError("Fraction values must be between 0 and 1 (exclusive)")

    # Create fraction DataArray
    fraction = xr.DataArray(fraction, coords={"quantile": fraction}, dims="quantile")

    # Convert water density from kg/m3 to g/m3
    water_density = water_density * 1000

    # Compute LWC per diameter bin [g/m3]
    mass_spectrum = get_liquid_water_spectrum(
        drop_number_concentration=drop_number_concentration,
        diameter=diameter,
        water_density=water_density,
    )
    lwc_per_diameter = diameter_bin_width * mass_spectrum

    # Compute the cumulative sum of LWC along the diameter bins
    cumulative_lwc = lwc_per_diameter.cumsum(dim=DIAMETER_DIMENSION, skipna=False)

    # Check if single bin
    is_single_bin = (lwc_per_diameter != 0).sum(dim=DIAMETER_DIMENSION) == 1

    # Retrieve total lwc and target lwc
    total_lwc = cumulative_lwc.isel({DIAMETER_DIMENSION: -1})
    target_lwc = total_lwc * fraction

    # Retrieve bin indices between which the quantile of the volume is reached
    # --> If all NaN or False, argmax and xr_get_last_valid_idx(fill_value=0) return 0 !
    idx_upper = (cumulative_lwc >= target_lwc).argmax(dim=DIAMETER_DIMENSION).astype(int)
    idx_lower = xr_get_last_valid_idx(
        da_condition=(cumulative_lwc <= target_lwc),
        dim=DIAMETER_DIMENSION,
        fill_value=0,
    ).astype(int)

    # Retrieve cumulative LWC values at such bins and target LWC
    y1 = cumulative_lwc.isel({DIAMETER_DIMENSION: idx_lower})
    y2 = cumulative_lwc.isel({DIAMETER_DIMENSION: idx_upper})
    yt = target_lwc

    # ------------------------------------------------------.
    ## Case with multiple bins
    # Define interpolation slope, avoiding division by zero if y1 equals y2.
    # - When target LWC exactly equal to cumulative_lwc value of a bin --> Diameter is the bin center diameter !
    slope = xr.where(y1 == y2, 0, (yt - y1) / (y2 - y1))

    # Define diameter increment from lower bin center
    d1 = diameter.isel(diameter_bin_center=idx_lower)  # m
    d2 = diameter.isel(diameter_bin_center=idx_upper)  # m
    d_increment = (d2 - d1) * slope

    # Define quantile diameter
    quantile_diameter_multi_bin = d1 + d_increment

    ## ------------------------------------------------------.
    ## Case with single bin
    # When no accumulation has yet occurred (y1==0), use the upper bin for both indices.
    idx_lower = xr.where(y1 == 0, idx_upper, idx_lower)
    # Identify the bin center diameter
    d = diameter.isel(diameter_bin_center=idx_lower)  # m
    d_width = diameter_bin_width.isel({DIAMETER_DIMENSION: idx_lower}) / 1000  # m
    d_lower = d - d_width / 2
    quantile_diameter_single_bin = d_lower + d_width * fraction

    ## ------------------------------------------------------.
    # Define quantile diameter
    quantile_diameter = xr.where(is_single_bin, quantile_diameter_single_bin, quantile_diameter_multi_bin)

    # Set NaN where total sum is 0 or all NaN
    mask_invalid = np.logical_or(total_lwc == 0, np.isnan(total_lwc))
    quantile_diameter = quantile_diameter.where(~mask_invalid)

    # Convert diameter to mm
    quantile_diameter = quantile_diameter * 1000

    # If only 1 fraction specified, squeeze and drop quantile coordinate
    if quantile_diameter.sizes["quantile"] == 1:
        quantile_diameter = quantile_diameter.drop_vars("quantile").squeeze()

    # Drop meaningless coordinates
    quantile_diameter = remove_diameter_coordinates(quantile_diameter)
    quantile_diameter = remove_velocity_coordinates(quantile_diameter)
    return quantile_diameter


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
    fraction : float or numpy.ndarray
        The fraction \( f \) of the total liquid water content to compute the diameter for.
        Default is 0.5, which computes the median volume diameter (D50).
        For other percentiles, use 0.1 for D10, 0.9 for D90, etc.
        Values must be between 0 and 1 (exclusive).
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
    # Dask array backend
    if hasattr(drop_number_concentration.data, "chunks"):
        fraction = np.atleast_1d(fraction)
        if fraction.size > 1:
            dask_gufunc_kwargs = {"output_sizes": {"quantile": fraction.size}}
            output_core_dims = [["quantile"]]
        else:
            dask_gufunc_kwargs = None
            output_core_dims = ((),)
        quantile_diameter = xr.apply_ufunc(
            _get_quantile_volume_drop_diameter,
            drop_number_concentration,
            kwargs={
                "fraction": fraction,
                "diameter": diameter.compute(),
                "diameter_bin_width": diameter_bin_width.compute(),
                "water_density": water_density,
            },
            input_core_dims=[[DIAMETER_DIMENSION]],
            vectorize=True,
            dask="parallelized",
            output_core_dims=output_core_dims,
            dask_gufunc_kwargs=dask_gufunc_kwargs,
            output_dtypes=["float64"],
        )
        if fraction.size > 1:
            quantile_diameter = quantile_diameter.assign_coords({"quantile": fraction})
        return quantile_diameter

    # Numpy array backed
    quantile_diameter = _get_quantile_volume_drop_diameter(
        drop_number_concentration=drop_number_concentration,
        diameter=diameter,
        diameter_bin_width=diameter_bin_width,
        fraction=fraction,
        water_density=water_density,
    )
    return quantile_diameter


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


def get_normalized_intercept_parameter_from_moments(moment_3, moment_4):
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
    The Concept of “Normalized” Distribution to Describe Raindrop spectrum:
    A Tool for Cloud Physics and Cloud Remote Sensing.
    J. Appl. Meteor. Climatol., 40, 1118-1140,
    https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2

    """
    Nw = 256 / 6 * moment_3**5 / moment_4**4
    return Nw


####--------------------------------------------------------------------------------------------------------
#### Kinetic Energy Parameters


def get_kinetic_energy_spectrum(
    drop_number_concentration,
    velocity,
    diameter,
    sample_interval,
    water_density=1000,
):
    r"""Compute the rainfall kinetic energy per diameter class.

    To obtain the Total Kinetic Energy (TKE) one has to multiply KE(D) with the diameter
     bins intervals and summing over the diameter bins.

    Parameters
    ----------
    drop_number_concentration : xarray.DataArray
        Array of drop number concentrations \\( N(D) \\) in m⁻³·mm⁻¹.
    velocity : xarray.DataArray or float
        The fall velocities \\( v \\) of the drops, in meters per second (m/s).
    diameter : xarray.DataArray
        The equivalent volume diameters \\( D \\) of the drops in each bin, in meters (m).
    sample_interval : float
        The time over which the drops are counted \\( \\Delta t \\) in seconds (s).
    water_density : float, optional
        The density of water \\( \rho_w \\) in kilograms per cubic meter (kg/m³).
        Default is 1000 kg/m³.

    Returns
    -------
    xr.DataArray
        Kinetic Energy Spectrum [J/m2/mm]
    """
    KE_spectrum = (
        np.pi / 12 * water_density * sample_interval * (drop_number_concentration * (diameter**3 * velocity**3))
    )
    return KE_spectrum


def get_kinetic_energy_variables(
    drop_number_concentration,
    velocity,
    diameter,
    diameter_bin_width,
    sample_interval,
    water_density=1000,
):
    r"""Compute rainfall kinetic energy descriptors from the drop number concentration.

    Parameters
    ----------
    drop_number_concentration : xarray.DataArray
        Array of drop number concentrations \\( N(D) \\) in m⁻³·mm⁻¹.
    velocity : xarray.DataArray or float
        The fall velocities \\( v \\) of the drops, in meters per second (m/s).
    diameter : xarray.DataArray
        The equivalent volume diameters \\( D \\) of the drops in each bin, in meters (m).
    diameter_bin_width : xarray.DataArray
        Width of each diameter bin \\( \\Delta D \\) in millimeters (mm).
    sample_interval : float
        The time over which the drops are counted \\( \\Delta t \\) in seconds (s).
    water_density : float, optional
        The density of water \\( \rho_w \\) in kilograms per cubic meter (kg/m³).
        Default is 1000 kg/m³.

    Returns
    -------
    xarray.Dataset
        Xarray Dataset with relevant rainfall kinetic energy variables:
        - TKE: Total Kinetic Energy [J/m2]
        - KED: Kinetic Energy per unit rainfall Depth [J·m⁻²·mm⁻¹]. Typical values range between 0 and 40 J·m⁻²·mm⁻¹.
        - KEF: Kinetic Energy Flux [J·m⁻²·h⁻¹]. Typical values range between 0 and 5000 J·m⁻²·h⁻¹.
        KEF is related to the KED by the rain rate: KED = KEF/R .
    """
    # Check velocity DataArray does not have a velocity dimension
    if VELOCITY_DIMENSION in velocity.dims:
        raise ValueError(f"The 'velocity' DataArray must not have the '{VELOCITY_DIMENSION}' dimension.")

    # Compute rain rate
    R = get_rain_rate(
        drop_number_concentration=drop_number_concentration,
        velocity=velocity,
        diameter=diameter,
        diameter_bin_width=diameter_bin_width,
    )

    # Compute total kinetic energy in [J/m2]
    TKE = (
        np.pi
        / 12
        * water_density
        * sample_interval
        * (drop_number_concentration * diameter**3 * velocity**3 * diameter_bin_width).sum(
            dim=DIAMETER_DIMENSION,
            skipna=False,
        )
    )

    # Compute Kinetic Energy Flux (KEF) [J/m2/h]
    KEF = TKE / sample_interval * 3600

    # Compute Kinetic Energy per Rainfall Depth [J/m2/mm]
    KED = KEF / R
    KED = xr.where(R == 0, 0, KED)  # Ensure KED is 0 when R (and thus drop number is 0)

    # Create dataset
    dict_vars = {
        "TKE": TKE,
        "KEF": KEF,
        "KED": KED,
    }
    ds = xr.Dataset(dict_vars)
    return ds


def get_kinetic_energy_variables_from_drop_number(
    drop_number,
    velocity,
    sampling_area,
    diameter,
    sample_interval,
    water_density=1000,
):
    r"""Compute rainfall kinetic energy descriptors from the measured drop number spectrum.

    Parameters
    ----------
    drop_number : xarray.DataArray
        The number of drops in each diameter (and velocity, if available) bin(s).
    velocity : xarray.DataArray or float
        The fall velocities \\( v \\) of the drops in each bin, in meters per second (m/s).
        Values are broadcasted to match the dimensions of `drop_number`.
    diameter : xarray.DataArray
        The equivalent volume diameters \\( D \\) of the drops in each bin, in meters (m).
    sampling_area : float
        The effective sampling area \\( A \\) of the sensor in square meters (m²).
    sample_interval : float
        The time over which the drops are counted \\( \\Delta t \\) in seconds (s).
    water_density : float, optional
        The density of water \\( \rho_w \\) in kilograms per cubic meter (kg/m³).
        Default is 1000 kg/m³.

    Returns
    -------
    xarray.Dataset
        Xarray Dataset with relevant rainfall kinetic energy variables:
        - TKE: Total Kinetic Energy [J/m2]
        - KED: Kinetic Energy per unit rainfall Depth [J·m⁻²·mm⁻¹]. Typical values range between 0 and 40 J·m⁻²·mm⁻¹.
        - KEF: Kinetic Energy Flux [J·m⁻²·h⁻¹]. Typical values range between 0 and 5000 J·m⁻²·h⁻¹.
        KEF is related to the KED by the rain rate: KED = KEF/R .

    Notes
    -----
    KED provides a measure of the energy associated with each unit of rainfall depth.
    KED is useful for analyze the potential impact of raindrop erosion as a function of
    the intensity of rainfall events.

    The kinetic energy of a rain drop is defined as:

    .. math::

        KE(D) = \frac{1}{2} · m_{drop} · v_{drop}^2 = \frac{\\pi \rho_{w}}{12} · D^3 · v^2

    The Total Kinetic Energy (TKE) is calculated using:

    .. math::

        TKE = \\sum_{i,j} \\left({n_{ij} · KE(D_{i}) \right)
            = \frac{\\pi \rho_{w}}{12 · A} \\sum_{i,j} \\left( {n_{ij} · D_{i}^3 · v_{j}^2}} \right)

    The Kinetic Energy Flux (KEF) is calculated using:

    .. math::

        KEF = \frac{TKE}{\\Delta t } · 3600

    KED is calculated using:

    .. math::

        KED = \frac{KEF}{R} \\cdot \frac{\\pi}{6} \\cdot \frac{\rho_w}{R} \\cdot \\sum_{i,j}
        \\left( \frac{n_{ij} \\cdot D_i^3 \\cdot v_j^2}{A} \right)

    where:

    - \\( n_{ij} \\) is the number of drops in diameter bin \\( i \\) and velocity bin \\( j \\).
    - \\( D_i \\) is the diameter of bin \\( i \\).
    - \\( v_j \\) is the velocity of bin \\( j \\).
    - \\( A \\) is the sampling area.
    - \\( \\Delta t \\) is the time integration period in seconds.
    - \\( R \\) is the rainfall rate in mm/hr.

    """
    # Get drop number core dimensions
    dim = get_bin_dimensions(drop_number)

    # Ensure velocity is 2D if drop number has velocity dimension
    # - if measured velocity --> already 2D
    # - if estimated velocity --> broadcasted to 2D
    velocity = xr.ones_like(drop_number) * velocity

    # Compute rain rate
    R = get_rain_rate_from_drop_number(
        drop_number=drop_number,
        sampling_area=sampling_area,
        diameter=diameter,
        sample_interval=sample_interval,
    )

    # Compute drop size kinetic energy per diameter (and velocity bin)
    KE = np.pi / 12 * water_density * diameter**3 * velocity**2  # [J]

    # Compute total kinetic energy in [J/m2]
    TKE = (KE * drop_number / sampling_area).sum(dim=dim, skipna=False)

    # Compute Kinetic Energy Flux (KEF) [J/m2/h]
    KEF = TKE / sample_interval * 3600

    # Compute Kinetic Energy per Rainfall Depth [J/m2/mm]
    KED = KEF / R
    KED = xr.where(R == 0, 0, KED)  # Ensure KED is 0 when R (and thus drop number is 0)

    # Create dataset
    dict_vars = {
        "TKE": TKE,
        "KEF": KEF,
        "KED": KED,
    }
    ds = xr.Dataset(dict_vars)
    return ds


####-------------------------------------------------------------------------------------------------------.
#### Wrapper ####


def compute_integral_parameters(
    drop_number_concentration,
    velocity,
    diameter,
    diameter_bin_width,
    sample_interval,
    water_density,
):
    """
    Compute integral parameters of a drop size distribution (DSD).

    Parameters
    ----------
    drop_number_concentration : xr.DataArray
        Drop number concentration in each diameter bin [#/m3/mm].
    velocity : xr.DataArray
        Fall velocity of drops in each diameter bin [m/s].
        The presence of a velocity_method dimension enable to compute the parameters
        with different velocity estimates.
    diameter : array-like
        Diameter of drops in each bin in m.
    diameter_bin_width : array-like
        Width of each diameter bin in mm.
    sample_interval : float
        Time interval over which the samples are collected in seconds.
    water_density : float or array-like
        Density of water [kg/m3].

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the computed integral parameters:
        - Nt : Total number concentration [#/m3]
        - M1 to M6 : Moments of the drop size distribution
        - Z : Reflectivity factor [dBZ]
        - W : Liquid water content [g/m3]
        - D10 : Diameter at the 10th quantile of the cumulative LWC distribution [mm]
        - D50 : Median volume drop diameter [mm]
        - D90 : Diameter at the 90th quantile of the cumulative LWC distribution [mm]
        - Dmode : Diameter at which the distribution peaks [mm]
        - Dm : Mean volume drop diameter [mm]
        - sigma_m : Standard deviation of the volume drop diameter [mm]
        - Nw : Normalized intercept parameter [m-3·mm⁻¹]
        - R : Rain rate [mm/h]
        - P : Rain accumulation [mm]
        - TKE: Total Kinetic Energy [J/m2]
        - KED: Kinetic Energy per unit rainfall Depth [J·m⁻²·mm⁻¹].
        - KEF: Kinetic Energy Flux [J·m⁻²·h⁻¹].
    """
    # Initialize dataset
    ds = xr.Dataset()

    # Compute total number concentration (Nt) [#/m3]
    ds["Nt"] = get_total_number_concentration(
        drop_number_concentration=drop_number_concentration,
        diameter_bin_width=diameter_bin_width,
    )

    # Compute rain rate
    ds["R"] = get_rain_rate(
        drop_number_concentration=drop_number_concentration,
        velocity=velocity,
        diameter=diameter,
        diameter_bin_width=diameter_bin_width,
    )

    # Compute rain accumulation (P) [mm]
    ds["P"] = get_rain_accumulation(rain_rate=ds["R"], sample_interval=sample_interval)

    # Compute moments (m0 to m6)
    for moment in range(0, 7):
        ds[f"M{moment}"] = get_moment(
            drop_number_concentration=drop_number_concentration,
            diameter=diameter,
            diameter_bin_width=diameter_bin_width,
            moment=moment,
        )

    # Compute Liquid Water Content (LWC) (W) [g/m3]
    # ds["W"] = get_liquid_water_content(
    #     drop_number_concentration=drop_number_concentration,
    #     diameter=diameter,
    #     diameter_bin_width=diameter_bin_width,
    #     water_density=water_density,
    # )

    ds["W"] = get_liquid_water_content_from_moments(moment_3=ds["M3"], water_density=water_density)

    # Compute reflectivity in dBZ
    ds["Z"] = get_equivalent_reflectivity_factor(
        drop_number_concentration=drop_number_concentration,
        diameter=diameter,
        diameter_bin_width=diameter_bin_width,
    )

    # Compute the diameter at which the distribution peak
    ds["Dmode"] = get_mode_diameter(drop_number_concentration, diameter=diameter) * 1000  # Output converted to mm

    # Compute mean_volume_diameter (Dm) [mm]
    ds["Dm"] = get_mean_volume_drop_diameter(moment_3=ds["M3"], moment_4=ds["M4"])

    # Compute σₘ[mm]
    ds["sigma_m"] = get_std_volume_drop_diameter(
        moment_3=ds["M3"],
        moment_4=ds["M4"],
        moment_5=ds["M5"],
    )

    # Compute normalized_intercept_parameter (Nw) [m-3·mm⁻¹]
    # ds["Nw"] = get_normalized_intercept_parameter(
    #     liquid_water_content=liquid_water_content,
    #     mean_volume_diameter=mean_volume_diameter,
    #     water_density=water_density,
    # )

    ds["Nw"] = get_normalized_intercept_parameter_from_moments(moment_3=ds["M3"], moment_4=ds["M4"])

    # Compute median volume_drop_diameter
    # --> Equivalent to get_quantile_volume_drop_diameter with fraction = 0.5
    # ds["D50"] = get_median_volume_drop_diameter(
    #     drop_number_concentration=drop_number_concentration,
    #     diameter=diameter,
    #     diameter_bin_width=diameter_bin_width,
    #     water_density=water_density,
    # )

    # Compute volume_drop_diameter for the 10th and 90th quantile of the cumulative LWC distribution
    fractions = [0.1, 0.5, 0.9]
    d_q = get_quantile_volume_drop_diameter(
        drop_number_concentration=drop_number_concentration,
        diameter=diameter,
        diameter_bin_width=diameter_bin_width,
        fraction=fractions,
        water_density=water_density,
    )
    for fraction in fractions:
        var = f"D{round(fraction*100)!s}"  # D10, D50, D90
        ds[var] = d_q.sel(quantile=fraction).drop_vars("quantile")

    # Compute kinetic energy variables
    ds_ke = get_kinetic_energy_variables(
        drop_number_concentration=drop_number_concentration,
        velocity=velocity,
        diameter=diameter,
        diameter_bin_width=diameter_bin_width,
        sample_interval=sample_interval,
        water_density=water_density,
    )
    ds.update(ds_ke)
    return ds


def compute_spectrum_parameters(
    drop_number_concentration,
    velocity,
    diameter,
    sample_interval,
    water_density=1000,
):
    """
    Compute drop size spectrum of rain rate, kinetic energy, mass and reflectivity.

    Parameters
    ----------
    drop_number_concentration : xr.DataArray
        Drop number concentration in each diameter bin [#/m3/mm].
    velocity : xr.DataArray
        Fall velocity of drops in each diameter bin [m/s].
        The presence of a velocity_method dimension enable to compute the parameters
        with different velocity estimates.
    diameter : array-like
        Diameter of drops in each bin in m.
    sample_interval : float
        Time interval over which the samples are collected in seconds.
    water_density : float or array-like
        Density of water [kg/m3].

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the following spectrum:
        - KE_spectrum : Kinetic Energy spectrum [J/m2/mm]
        - R_spectrum : Rain Rate spectrum [mm/h/mm]
        - W_spectrum : Mass spectrum [g/m3/mm]
        - Z_spectrum : Reflectivity spectrum [dBZ of mm6/m3/mm]
    """
    # Initialize dataset
    ds = xr.Dataset()
    ds["KE_spectrum"] = get_kinetic_energy_spectrum(
        drop_number_concentration,
        velocity=velocity,
        diameter=diameter,
        sample_interval=sample_interval,
        water_density=water_density,
    )
    ds["R_spectrum"] = get_rain_rate_spectrum(drop_number_concentration, velocity=velocity, diameter=diameter)
    ds["W_spectrum"] = get_liquid_water_spectrum(
        drop_number_concentration,
        diameter=diameter,
        water_density=water_density,
    )
    ds["Z_spectrum"] = get_equivalent_reflectivity_spectrum(drop_number_concentration, diameter=diameter)
    return ds
