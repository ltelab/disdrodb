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
"""Implement drop axis ratio theoretical models."""

import numpy as np
import xarray as xr


def available_axis_ratio():
    """Return a list of the available drop axis ratio methods."""
    return list(AXIS_RATIO_METHODS)


def get_axis_ratio_method(method):
    """Return the specified drop axis ratio method."""
    method = check_axis_ratio(method)
    return AXIS_RATIO_METHODS[method]


def check_axis_ratio(method):
    """Check validity of the specified drop axis ratio method."""
    available_methods = available_axis_ratio()
    if method not in available_methods:
        raise ValueError(f"{method} is an invalid axis-ratio method. Valid methods: {available_methods}.")
    return method


def get_axis_ratio(diameter, method):
    """
    Compute the axis ratio of raindrops using the specified method.

    Parameters
    ----------
    diameter : array-like
        Raindrops diameter in mm.
    method : str
        The method to use for calculating the axis ratio. Available methods are:
        'Thurai2005', 'Thurai2007', 'Battaglia2010', 'Brandes2002',
        'Pruppacher1970', 'Beard1987', 'Andsager1999'.

    Returns
    -------
    axis_ratio : array-like
        Calculated axis ratios corresponding to the input diameters.

    Raises
    ------
    ValueError
        If the specified method is not one of the available methods.

    Notes
    -----
    This function serves as a wrapper to various axis ratio models for raindrops.
    It selects and applies the appropriate model based on the `method` parameter.

    Examples
    --------
    >>> diameter = np.array([0.5, 1.0, 2.0, 3.0])
    >>> axis_ratio = get_axis_ratio(diameter, method="Brandes2002")

    """
    # Retrieve axis ratio function
    func = get_axis_ratio_method(method)

    # Retrieve axis ratio
    axis_ratio = func(diameter)

    # Clip values between 0 and 1
    axis_ratio = np.clip(axis_ratio, 0, 1)
    return axis_ratio


def get_axis_ratio_andsager_1999(diameter):
    """
    Compute the axis ratio of raindrops using the Andsager et al. (1999) method.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in millimeters.

    Returns
    -------
    axis_ratio : array-like
        Calculated axis ratios corresponding to the input diameters.

    Notes
    -----
    This function calculates the axis ratio of raindrops based on the method described
    in Andsager et al. (1999). For diameters between 1.1 mm and 4.4 mm, it uses the
    average axis-ratio relationship given by Kubesh and Beard (1993):

        axis_ratio = 1.012 - 0.144 * D - 1.03 * D^2

    For diameters outside this range (0.1 mm to 1.1 mm and 4.4 mm to 7.0 mm),
    it uses the equilibrium shape equation from Beard and Chuang (1987).

    References
    ----------
    Andsager, K., Beard, K. V., & Laird, N. F. (1999).
        Laboratory measurements of axis ratios for large raindrops.
        Journal of the Atmospheric Sciences, 56(15), 2673-2683.

    Kubesh, R. J., & Beard, K. V. (1993).
        Laboratory measurements of spontaneous oscillations for moderate-size raindrops.
        Journal of the Atmospheric Sciences, 50(7), 1089-1098.

    Beard, K. V., & Chuang, C. (1987).
        A new model for the equilibrium shape of raindrops.
        Journal of the Atmospheric Sciences, 44(11), 1509-1524.

    """
    # Convert diameter to centimeters
    diameter_cm = diameter * 0.1

    # Axis ratio for diameters outside 1.1 mm to 4.4 mm using equilibrium model
    axis_ratio_equilibrium = get_axis_ratio_beard_1987(diameter)

    # Axis ratio for diameters between 1.1 mm and 4.4 mm using Kubesh & Beard (1993) model
    axis_ratio_kubesh = 1.012 - 0.144 * diameter_cm - 1.03 * diameter_cm**2

    # Combine models based on diameter ranges
    axis_ratio = xr.where(
        (diameter_cm >= 1.1) & (diameter_cm < 4.4),
        axis_ratio_kubesh,
        axis_ratio_equilibrium,
    )

    return axis_ratio


def get_axis_ratio_battaglia_2010(diameter):
    """
    Compute the axis ratio of raindrops using the Battaglia et al. (2010) method.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in millimeters.

    Returns
    -------
    axis_ratio : array-like
        Calculated axis ratios corresponding to the input diameters.

    Notes
    -----
    - For diameters less than or equal to 1 mm, the axis ratio is constant at 1.0.
    - For diameters greater than or equal to 5 mm, the axis ratio is constant at 0.7.
    - Between 1 mm and 5 mm, the axis ratio varies linearly.

    The axis ratio is calculated using the equation:

        axis_ratio = 1.075 - 0.075 * D

    where **D** is the diameter in millimeters.

    References
    ----------
    Battaglia, A., Rustemeier, E., Tokay, A., Blahak, U., & Simmer, C. (2010).
    PARSIVEL Snow Observations: A Critical Assessment.
    Journal of Atmospheric and Oceanic Technology, 27(2), 333-344.
    https://doi.org/10.1175/2009JTECHA1332.1

    """
    axis_ratio = 1.075 - 0.075 * diameter
    axis_ratio = xr.where(diameter <= 1, 1.0, axis_ratio)
    axis_ratio = xr.where(diameter >= 5, 0.7, axis_ratio)
    return axis_ratio


def get_axis_ratio_beard_1987(diameter):
    """
    Compute the axis ratio of raindrops using the Beard and Chuang (1987) method.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in centimeters.

    Returns
    -------
    axis_ratio : array-like
        Calculated axis ratios corresponding to the input diameters.

    Notes
    -----
    The formula is a polynomial fit to the numerical model of Beard and Chuang (1987), with
    drop diameters between 1 and 7 mm.

    References
    ----------
    Beard, K. V., & Chuang, C. (1987).
    A new model for the equilibrium shape of raindrops.
    Journal of the Atmospheric Sciences, 44(11), 1509-1524.
    https://doi.org/10.1175/1520-0469(1987)044<1509:ANMFTE>2.0.CO;2
    """
    return 1.0048 + 5.7e-04 * diameter - 2.628e-02 * diameter**2 + 3.682e-03 * diameter**3 - 1.677e-04 * diameter**4


def get_axis_ratio_brandes_2002(diameter):
    """
    Compute the axis ratio of raindrops using the Brandes et al. (2002) method.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in millimeters.

    Returns
    -------
    axis_ratio : array-like
        Calculated axis ratios corresponding to the input diameters.

    References
    ----------
    Brandes, E. A., Zhang, G., & Vivekanandan, J. (2002).
    Experiments in rainfall estimation with a polarimetric radar in a subtropical environment.
    Journal of Applied Meteorology, 41(6), 674-685.
    https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2

    Brandes, et  al. 2005: On the Influence of Assumed Drop Size Distribution Form
    on Radar-Retrieved Thunderstorm Microphysics. J. Appl. Meteor. Climatol., 45, 259-268.
    """
    # Valid for drop diameters between 0.1 to 8.1 mm
    axis_ratio = 0.9951 + 0.0251 * diameter - 0.03644 * diameter**2 + 0.005303 * diameter**3 - 0.0002492 * diameter**4
    return axis_ratio


def get_axis_ratio_pruppacher_1970(diameter):
    """
    Compute the axis ratio of raindrops using the Pruppacher and Pitter (1971) method.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in millimeters.

    Returns
    -------
    axis_ratio : array-like
        Calculated axis ratios corresponding to the input diameters.

    Notes
    -----
    This formula is a linear fit to wind tunnel data of Pruppacher and Pitter (1971) with
    drop diameters between 1 and 9 mm.

    References
    ----------
    Pruppacher, H. R., & Pitter, R. L. (1971).
    A Semi-Empirical Determination of the Shape of Cloud and Precipitation Drops.
    Journal of the Atmospheric Sciences, 28(1), 86-94.
    https://doi.org/10.1175/1520-0469(1971)028<0086:ASEDOT>2.0.CO;2
    """
    axis_ratio = 1.03 - 0.062 * diameter
    return axis_ratio


def get_axis_ratio_thurai_2005(diameter):
    """
    Compute the axis ratio of raindrops using the Thurai et al. (2005) method.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in millimeters.

    Returns
    -------
    axis_ratio : array-like
        Calculated axis ratios corresponding to the input diameters.

    References
    ----------
    Thurai, M., and V. N. Bringi, 2005: Drop Axis Ratios from a 2D Video Disdrometer.
    J. Atmos. Oceanic Technol., 22, 966-978, https://doi.org/10.1175/JTECH1767.1

    """
    # Valid between 1 and 5 mm
    axis_ratio = 0.9707 + 4.26e-2 * diameter - 4.29e-2 * diameter**2 + 6.5e-3 * diameter**3 - 3e-4 * diameter**4
    return axis_ratio


def get_axis_ratio_thurai_2007(diameter):
    """Compute the axis ratio of raindrops using the Thurai et al. (2007) method.

    Parameters
    ----------
    diameter : array-like
        Diameter of the raindrops in millimeters.

    Returns
    -------
    axis_ratio : array-like
        Calculated axis ratios corresponding to the input diameters.

    References
    ----------
    Thurai, M., G. J. Huang, V. N. Bringi, W. L. Randeu, and M. Schönhuber, 2007:
    Drop Shapes, Model Comparisons, and Calculations of Polarimetric Radar Parameters in Rain.
    J. Atmos. Oceanic Technol., 24, 1019-1032, https://doi.org/10.1175/JTECH2051.1

    """
    # Assume spherical drop when diameter < 0.7 mm
    axis_ratio_below_0_7 = 1
    # Beard and Kubesh (1991) for drops diameter between 0.7 mm and 1.5 mm
    axis_ratio_below_1_5 = (
        1.173 - 0.5165 * diameter + 0.4698 * diameter**2 - 0.1317 * diameter**3 - 8.5e-3 * diameter**4
    )
    # Formula fitted on measurements of Thurai et al., 2005 for drop diameter above 1.5 mm
    # --> This is very similar to Pruppacher1970 !
    axis_ratio_above_1_5 = (
        1.065 - 6.25e-2 * diameter - 3.99e-3 * diameter**2 + 7.66e-4 * diameter**3 - 4.095e-5 * diameter**4
    )
    # Combine axis ratio
    axis_ratio_below_1_5 = xr.where(diameter > 0.7, axis_ratio_below_1_5, axis_ratio_below_0_7)
    axis_ratio = xr.where(diameter > 1.5, axis_ratio_above_1_5, axis_ratio_below_1_5)
    return axis_ratio


AXIS_RATIO_METHODS = {
    "Thurai2005": get_axis_ratio_thurai_2005,
    "Thurai2007": get_axis_ratio_thurai_2007,
    "Battaglia2010": get_axis_ratio_battaglia_2010,
    "Brandes2002": get_axis_ratio_brandes_2002,
    "Pruppacher1970": get_axis_ratio_pruppacher_1970,
    "Beard1987": get_axis_ratio_beard_1987,
    "Andsager1999": get_axis_ratio_andsager_1999,
}
