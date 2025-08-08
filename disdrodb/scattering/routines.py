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
"""Implement PSD scattering routines."""

import itertools
import logging
import os

import dask
import numpy as np
import xarray as xr

from disdrodb.configs import get_scattering_table_dir
from disdrodb.constants import DIAMETER_DIMENSION
from disdrodb.l1.filters import filter_diameter_bins
from disdrodb.psd.models import BinnedPSD, create_psd, get_required_parameters
from disdrodb.scattering.axis_ratio import check_axis_ratio_model, get_axis_ratio_model
from disdrodb.scattering.permittivity import (
    check_permittivity_model,
    get_rayleigh_dielectric_factor,
    get_refractive_index,
)
from disdrodb.utils.logger import log_info
from disdrodb.utils.manipulations import get_diameter_bin_edges
from disdrodb.utils.warnings import suppress_warnings

logger = logging.getLogger(__name__)

RADAR_OPTIONS = [
    "frequency",
    "diameter_max",
    "num_points",
    "canting_angle_std",
    "axis_ratio_model",
    "permittivity_model",
    "water_temperature",
    "elevation_angle",
]

# Common radar frequencies (in GHz)
frequency_dict = {
    "S": 2.70,  # e.g. NEXRAD radars
    "C": 5.4,  # e.g. MeteoSwiss Rad4Alp radars
    "X": 9.4,  # e.g. LTE MXPOL radar
    "Ku": 13.6,  # e.g. DPR-Ku
    "K": 24.2,  # e.g. MRR-PRO
    "Ka": 35.5,  # e.g. DPR-Ka
    "W": 94.05,  # e.g. CloudSat, EarthCare
}


def available_radar_bands():
    """Return a list of the available radar bands."""
    return list(frequency_dict)


def check_radar_band(radar_band):
    """Check the validity of the specified radar band."""
    available_bands = available_radar_bands()
    if radar_band not in available_bands:
        raise ValueError(f"{radar_band} is an invalid radar band. Valid radar bands: {available_bands}.")
    return radar_band


def _check_frequency(frequency):
    """Check the validity of the specified frequency."""
    if isinstance(frequency, str):
        frequency = check_radar_band(frequency)
        frequency = frequency_dict[frequency]
        return frequency
    if not isinstance(frequency, (int, float)):
        raise TypeError(f"Frequency {frequency} must be a string or a number.")
    return frequency


def ensure_numerical_frequency(frequency):
    """Ensure that the frequencies are numerical values in GHz."""
    if isinstance(frequency, (str, int, float)):
        frequency = [frequency]
    frequency = np.array([_check_frequency(f) for f in frequency])
    return frequency.squeeze()


# Wavelength, Frequency Conversion
def wavelength_to_frequency(wavelength):
    """Convert wavelength in millimeters to frequency in GHz."""
    c = 299_792_458  # speed of light in m/s
    frequency = c / np.array(wavelength) / 1e6
    return frequency


def frequency_to_wavelength(frequency):
    """Convert frequency in GHz to wavelength millimeters."""
    c = 299_792_458  # speed of light in m/s
    wavelength = c / np.array(frequency) / 1e6
    return wavelength


def get_backward_geometry(elevation_angle):
    """Define backward geometry given a radar elevation angle."""
    # - Format (thet0, thet0, phi0, phi0, alpha, beta
    # - thet0, thet0, thet: The zenith angles of incident and scattered radiation (default to 90)
    # - phi0, phi: The azimuth angles of incident and scattered radiation (default to 0 and 180)
    # - alpha, beta: Defaults to 0.0, 0.0. Valid values: alpha = [0, 360] beta = [0, 180]

    # Retrieve zenith angle of incident beam (from vertical)
    theta = 90.0 - elevation_angle

    # Return (thet0, thet0, phi0, phi0, alpha, beta) tuple
    return (theta, 180 - theta, 0.0, 180, 0.0, 0.0)


def get_forward_geometry(elevation_angle):
    """Define forward geometry given a radar elevation angle."""
    # - Format (thet0, thet0, phi0, phi0, alpha, beta
    # - thet0, thet0, thet: The zenith angles of incident and scattered radiation (default to 90)
    # - phi0, phi: The azimuth angles of incident and scattered radiation (default to 0 and 180)
    # - alpha, beta: Defaults to 0.0, 0.0. Valid values: alpha = [0, 360] beta = [0, 180]

    # Retrieve zenith angle of incident beam (from vertical)
    theta = 90.0 - elevation_angle

    # Return (thet0, thet0, phi0, phi0, alpha, beta) tuple
    return (theta, theta, 0.0, 0.0, 0.0, 0.0)


# from pytmatrix import tmatrix_aux
# get_backward_geometry(0)
# tmatrix_aux.geom_horiz_back
# get_backward_geometry(90)
# tmatrix_aux.geom_vert_back   # phi0 varies (180 instead of pytmatrix 0)

# get_forward_geometry(0)
# tmatrix_aux.geom_horiz_forw
# get_forward_geometry(90)     # theta and thet0 are 0 instead of 180
# tmatrix_aux.geom_vert_forw


def initialize_scatterer(
    wavelength,
    refractive_index,
    num_points=1024,
    diameter_max=8,
    canting_angle_std=7,
    axis_ratio_model="Thurai2007",
    elevation_angle=0,
):
    """Initialize T-matrix scatterer object for a given frequency.

    Load a scatterer object with cached scattering table.

    If the scattering table does not exist at the specified location, it will be
    created and saved to disk. If the file is found, it will be loaded and used
    to configure the scatterer.

    Parameters
    ----------
    wavelength : float
        Radar wavelength in mm.
    refractive_index: complex
        Water refractive index.
    num_points: int
        Number of bins into which discretize the PSD.
    diameter_max : float
        Maximum drop diameter in millimeters for the scattering table.
    canting_angle_std : float, optional
        Standard deviation of the canting angle distribution in degrees,
        by default 7.
    axis_ratio_model: str
        Axis ratio model used to shape hydrometeors. The default is ``"Thurai2007"``.
        See available models with ``disdrodb.scattering.available_axis_ratio_models()``.
    elevation_angle: str
        Radar elevation angle in degrees.
        Specify 90 degrees for vertically pointing radars.
        The default is 0 degrees.
    scattering_table_dir : str or Path, optional
        Directory path where T-Matrix scattering tables are stored. If None, the default
        location will be used.
    verbose: bool
        Whether to verbose the computation of the scattering table. The default is False.

    Returns
    -------
    scatterer : Scatterer
        A scatterer object with the PSD integrator configured and scattering
        table loaded or generated.
    """
    from pytmatrix import orientation
    from pytmatrix.psd import PSDIntegrator
    from pytmatrix.tmatrix import Scatterer

    # Retrieve custom axis ratio function
    axis_ratio_func = get_axis_ratio_model(axis_ratio_model)

    # Define radar dielectric factor
    Kw_sqr = get_rayleigh_dielectric_factor(refractive_index)

    # Define backward and forward geometries
    # - Format (thet0, thet0, phi0, phi0, alpha, beta
    backward_geom = get_backward_geometry(elevation_angle)
    forward_geom = get_forward_geometry(elevation_angle)

    # ---------------------------------------------------------------.
    # For W band limits diameter_max up to 9.5, otherwise the kernel dies !
    if wavelength < 3.5:
        diameter_max = min(diameter_max, 9.5)

    # ---------------------------------------------------------------.
    # Initialize Scatterer class
    # - By specifying m, we assume same refractive index for all particles diameters
    scatterer = Scatterer(wavelength=wavelength, m=refractive_index, Kw_sqr=Kw_sqr)

    # - Define geometry
    scatterer.set_geometry(backward_geom)

    # - Define orientation methods
    # --> Alternatives: orient_averaged_adaptive, orient_single,
    # --> Speed: orient_single > orient_averaged_fixed > orient_averaged_adaptive
    scatterer.orient = orientation.orient_averaged_fixed

    # - Define particle orientation PDF for orientational averaging
    # --> The standard deviation of the angle with respect to vertical orientation (the canting angle).
    scatterer.or_pdf = orientation.gaussian_pdf(std=canting_angle_std)

    # ---------------------------------------------------------------.
    # Initialize PSDIntegrator
    scatterer.psd_integrator = PSDIntegrator()
    # - Define axis_ratio_func
    # --> The Scatterer class expects horizontal to vertical
    # --> Axis ratio model are defined to return vertical to horizontal aspect ratio !
    scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0 / axis_ratio_func(D)
    # - Define function to compute refrative index (as function of D)
    # scatterer.psd_integrator.m_func = None    # Use constant value of scatterer.m
    # - Define number of points over which to integrate
    scatterer.psd_integrator.num_points = num_points
    # - Define maximum drop diameter
    scatterer.psd_integrator.D_max = diameter_max
    # - Define geometries
    # --> convention: first is backward, second is forward
    scatterer.psd_integrator.geometries = (backward_geom, forward_geom)
    return scatterer


def calculate_scatterer(
    wavelength,
    refractive_index,
    num_points=1024,
    diameter_max=8,
    canting_angle_std=7,
    axis_ratio_model="Thurai2007",
    elevation_angle=0,
):
    """Initialize T-matrix scatterer object for a given frequency.

    Load a scatterer object with cached scattering table.

    If the scattering table does not exist at the specified location, it will be
    created and saved to disk. If the file is found, it will be loaded and used
    to configure the scatterer.

    Parameters
    ----------
    wavelength : float
        Radar wavelength in millimeters.
    num_points: int
        Number of bins into which discretize the PSD.
    diameter_max : float
        Maximum drop diameter in millimeters for the scattering table.
    canting_angle_std : float, optional
        Standard deviation of the canting angle distribution in degrees,
        by default 7.
    axis_ratio_model : str, optional
        Axis ratio model used to shape hydrometeors. The default is ``"Thurai2007"``.
        See available models with ``disdrodb.scattering.available_axis_ratio_models()``.
    elevation_angle: str
        Radar elevation angle in degrees.
        Specify 90 degrees for vertically pointing radars.
        The default is 0 degrees.

    Returns
    -------
    scatterer : Scatterer
        A scatterer object with the PSD integrator configured and scattering
        table loaded or generated.
    """
    # ---------------------------------------------------------------.
    # Initialize Scatterer class
    scatterer = initialize_scatterer(
        wavelength=wavelength,
        refractive_index=refractive_index,
        num_points=num_points,
        diameter_max=diameter_max,
        canting_angle_std=canting_angle_std,
        axis_ratio_model=axis_ratio_model,
        elevation_angle=elevation_angle,
    )

    # ---------------------------------------------------------------.
    # Calculate scattering table
    scatterer.psd_integrator.init_scatter_table(scatterer)
    return scatterer


def load_scatterer(
    frequency,
    num_points=1024,
    diameter_max=8,
    canting_angle_std=7,
    axis_ratio_model="Thurai2007",
    permittivity_model="Turner2016",
    water_temperature=10,
    elevation_angle=0,
    scattering_table_dir=None,
    verbose=False,
):
    """
    Load a scatterer object with cached scattering table.

    If the scattering table does not exist at the specified location, it will be
    created and saved to disk. If the file is found, it will be loaded and used
    to configure the scatterer.

    Parameters
    ----------
    frequency : float
        Radar frequency in GHz.
    num_points: int
        Number of bins into which discretize the PSD.
    diameter_max : float
        Maximum drop diameter in millimeters for the scattering table.
    canting_angle_std : float, optional
        Standard deviation of the canting angle distribution in degrees,
        by default 7.
    axis_ratio_model : str, optional
        Axis ratio model used to shape hydrometeors. The default is ``"Thurai2007"``.
        See available models with ``disdrodb.scattering.available_axis_ratio_models()``.
    permittivity_model : str
        Permittivity model to use to compute the refractive index and the
        rayleigh_dielectric_factor. The default is ``Turner2016``.
        See available models with ``disdrodb.scattering.available_permittivity_models()``.
    water_temperature : float
        Water temperature in degree Celsius to be used in the permittivity model.
        The default is 10 degC.
    elevation_angle: str
        Radar elevation angle in degrees.
        Specify 90 degrees for vertically pointing radars.
        The default is 0 degrees.
    scattering_table_dir : str or Path, optional
        Directory path where T-Matrix scattering tables are stored. If None, the default
        location will be used.
    verbose: bool
        Whether to verbose the computation of the scattering table. The default is False.

    Returns
    -------
    scatterer : Scatterer
        A scatterer object with the PSD integrator configured and scattering
        table loaded or generated.
    """
    # Define wavelength (in mm)
    wavelength = frequency_to_wavelength(frequency)

    # Define complex refractive index
    refractive_index = get_refractive_index(
        frequency=frequency,
        temperature=water_temperature,
        permittivity_model=permittivity_model,
    )

    # Retrieve scattering table directory
    scattering_table_dir = get_scattering_table_dir(scattering_table_dir)

    # Define a filename based on the key parameters
    filename = "_".join(
        [
            "ScatteringTable",
            f"wl-{wavelength:.2f}",
            f"el-{elevation_angle:.1f}",
            f"dmax-{diameter_max:.1f}",
            f"npts-{num_points}",
            f"m-{refractive_index:.3f}",
            f"cant-{canting_angle_std:.1f}",
            f"ar-{axis_ratio_model}.pkl",
        ],
    )
    scatter_table_filepath = os.path.join(scattering_table_dir, filename)

    # Load or create scattering table
    if os.path.exists(scatter_table_filepath):
        scatterer = initialize_scatterer(
            wavelength=wavelength,
            refractive_index=refractive_index,
            num_points=num_points,
            diameter_max=diameter_max,
            canting_angle_std=canting_angle_std,
            axis_ratio_model=axis_ratio_model,
            elevation_angle=elevation_angle,
        )
        _ = scatterer.psd_integrator.load_scatter_table(scatter_table_filepath)

    else:
        if verbose:
            msg = f"- Computing pyTmatrix {filename}"
            log_info(logger=logger, msg=msg, verbose=verbose)

        # Calculate scatterer
        scatterer = calculate_scatterer(
            wavelength=wavelength,
            refractive_index=refractive_index,
            num_points=num_points,
            diameter_max=diameter_max,
            canting_angle_std=canting_angle_std,
            axis_ratio_model=axis_ratio_model,
            elevation_angle=elevation_angle,
        )

        scatterer.psd_integrator.save_scatter_table(scatter_table_filepath)
    return scatterer


####----------------------------------------------------------------------
#### Scattering functions


def compute_radar_variables(scatterer):
    """Compute radar variables for a given scatter object with a specified PSD.

    To speed up computations, this function should input a scatterer object with
    a preinitialized scattering table.
    """
    from pytmatrix import radar

    with suppress_warnings():
        radar_vars = {}
        # Retrieve backward and forward_geometries
        # - Convention (first is backward, second is forward)
        backward_geom = scatterer.psd_integrator.geometries[0]
        forward_geom = scatterer.psd_integrator.geometries[1]

        # Set backward scattering for reflectivity calculations
        scatterer.set_geometry(backward_geom)

        radar_vars["DBZH"] = 10 * np.log10(radar.refl(scatterer, h_pol=True))  # dBZ
        radar_vars["DBZV"] = 10 * np.log10(radar.refl(scatterer, h_pol=False))  # dBZ

        radar_vars["ZDR"] = 10 * np.log10(radar.Zdr(scatterer))  # dB
        if ~np.isfinite(radar_vars["ZDR"]):
            radar_vars["ZDR"] = np.nan

        radar_vars["LDR"] = 10 * np.log10(radar.ldr(scatterer))  # dBZ
        if ~np.isfinite(radar_vars["LDR"]):
            radar_vars["LDR"] = np.nan

        radar_vars["RHOHV"] = radar.rho_hv(scatterer)  # deg/km
        radar_vars["DELTAHV"] = radar.delta_hv(scatterer) * 180.0 / np.pi  # [deg]

        # Set forward scattering for attenuation and phase calculations
        scatterer.set_geometry(forward_geom)
        radar_vars["KDP"] = radar.Kdp(scatterer)  # deg/km
        radar_vars["AH"] = radar.Ai(scatterer, h_pol=True)  # dB/km
        radar_vars["AV"] = radar.Ai(scatterer, h_pol=False)  # dB/km
        radar_vars["ADR"] = radar_vars["AH"] - radar_vars["AV"]  # dB/km
    return radar_vars


# Radar variables computed by DISDRODB
# - Must reflect dictionary order output of compute_radar_variables
RADAR_VARIABLES = ["DBZH", "DBZV", "ZDR", "LDR", "RHOHV", "DELTAHV", "KDP", "AH", "AV", "ADR"]


def _initialize_null_output(output_dictionary):
    if output_dictionary:
        return dict.fromkeys(RADAR_VARIABLES, np.nan)
    return np.zeros(len(RADAR_VARIABLES)) * np.nan


def _estimate_empirical_radar_parameters(
    drop_number_concentration,
    bin_edges,
    scatterer,
    output_dictionary,
):
    # Assign PSD model to the scatterer object
    scatterer.psd = BinnedPSD(bin_edges, drop_number_concentration)

    # Get radar variables
    with suppress_warnings():
        try:
            radar_vars = compute_radar_variables(scatterer)
            output = radar_vars if output_dictionary else np.array(list(radar_vars.values()))
        except Exception:
            output = _initialize_null_output(output_dictionary)
    return output


def _estimate_model_radar_parameters(
    parameters,
    psd_model,
    psd_parameters_names,
    scatterer,
    output_dictionary,
):
    # Assign PSD model to the scatterer object
    parameters = dict(zip(psd_parameters_names, parameters))
    scatterer.psd = create_psd(psd_model, parameters)

    # Get radar variables
    with suppress_warnings():
        try:
            radar_vars = compute_radar_variables(scatterer)
            output = radar_vars if output_dictionary else np.array(list(radar_vars.values()))
        except Exception:
            output = _initialize_null_output(output_dictionary)
    return output


def get_psd_parameters(ds):
    """Return a xr.Dataset with only the PSD parameters as variable."""
    psd_model = ds.attrs["disdrodb_psd_model"]
    required_parameters = get_required_parameters(psd_model)
    missing_parameters = [param for param in required_parameters if param not in ds]
    if len(missing_parameters) > 0:
        raise ValueError(f"The {psd_model} parameters {missing_parameters} are not present in the dataset.")
    return ds[required_parameters]


def get_model_radar_parameters(
    ds,
    frequency,
    num_points=1024,
    diameter_max=10,
    canting_angle_std=7,
    axis_ratio_model="Thurai2007",
    permittivity_model="Turner2016",
    water_temperature=10,
    elevation_angle=0,
):
    """Compute radar parameters from a PSD model.

    This function retrieve values for a single set of parameter only !

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the parameters of the PSD model.
        The dataset attribute disdrodb_psd_model specifies the PSD model to use.
    frequency : float
        Frequency in GHz for which to compute the radar parameters.
    canting_angle_std : float, optional
        Standard deviation of the canting angle.  The default value is 10.
    diameter_max : float, optional
        Maximum diameter. The default value is 8 mm.
    axis_ratio_model : str, optional
        Axis ratio model used to shape hydrometeors. The default is ``"Thurai2007"``.
        See available models with ``disdrodb.scattering.available_axis_ratio_models()``.
    permittivity_model : str
        Permittivity model to use to compute the refractive index and the
        rayleigh_dielectric_factor. The default is ``Turner2016``.
        See available models with ``disdrodb.scattering.available_permittivity_models()``.
    water_temperature : float
        Water temperature in degree Celsius to be used in the permittivity model.
        The default is 10 degC.
    elevation_angle: str
        Radar elevation angle in degrees.
        Specify 90 degrees for vertically pointing radars.
        The default is 0 degrees.

    Returns
    -------
    xarray.Dataset
        Dataset containing the computed radar parameters.
    """
    # Retrieve psd model and parameters.
    psd_model = ds.attrs["disdrodb_psd_model"]
    required_parameters = get_required_parameters(psd_model)
    ds_parameters = get_psd_parameters(ds)

    # Check argument validity
    axis_ratio_model = check_axis_ratio_model(axis_ratio_model)
    permittivity_model = check_permittivity_model(permittivity_model)

    # Create DataArray with PSD parameters
    da_parameters = ds_parameters.to_array(dim="psd_parameters").compute()

    # Initialize scattering table
    scatterer = load_scatterer(
        frequency=frequency,
        num_points=num_points,
        diameter_max=diameter_max,
        canting_angle_std=canting_angle_std,
        axis_ratio_model=axis_ratio_model,
        permittivity_model=permittivity_model,
        water_temperature=water_temperature,
        elevation_angle=elevation_angle,
    )

    # Define kwargs
    kwargs = {
        "output_dictionary": False,
        "psd_model": psd_model,
        "psd_parameters_names": required_parameters,
        "scatterer": scatterer,
    }

    # Loop over each PSD (not in parallel --> dask="forbidden")
    da_radar = xr.apply_ufunc(
        _estimate_model_radar_parameters,
        da_parameters,
        kwargs=kwargs,
        input_core_dims=[["psd_parameters"]],
        output_core_dims=[["radar_variables"]],
        vectorize=True,
        dask="forbidden",
        dask_gufunc_kwargs={
            "output_sizes": {"radar_variables": len(RADAR_VARIABLES)},
        },  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )

    # Finalize radar dataset (add name, coordinates)
    ds_radar = _finalize_radar_dataset(
        da_radar=da_radar,
        frequency=frequency,
        num_points=num_points,
        diameter_max=diameter_max,
        canting_angle_std=canting_angle_std,
        axis_ratio_model=axis_ratio_model,
        permittivity_model=permittivity_model,
        water_temperature=water_temperature,
        elevation_angle=elevation_angle,
    )
    return ds_radar


def get_empirical_radar_parameters(
    ds,
    frequency,
    num_points=1024,
    diameter_max=10,
    canting_angle_std=7,
    axis_ratio_model="Thurai2007",
    permittivity_model="Turner2016",
    water_temperature=10,
    elevation_angle=0,
):
    """Compute radar parameters from an empirical drop number concentration.

    This function retrieve values for a single set of parameter only !

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the drop number concentration variable.
    frequency : float
        Frequency in GHz for which to compute the radar parameters.
    canting_angle_std : float, optional
        Standard deviation of the canting angle.  The default value is 10.
    diameter_max : float, optional
        Maximum diameter. The default value is 8 mm.
    axis_ratio_model : str, optional
        Axis ratio model used to shape hydrometeors. The default is ``"Thurai2007"``.
        See available models with ``disdrodb.scattering.available_axis_ratio_models()``.
    permittivity_model : str
        Permittivity model to use to compute the refractive index and the
        rayleigh_dielectric_factor. The default is ``Turner2016``.
        See available models with ``disdrodb.scattering.available_permittivity_models()``.
    water_temperature : float
        Water temperature in degree Celsius to be used in the permittivity model.
        The default is 10 degC.
    elevation_angle: str
        Radar elevation angle in degrees.
        Specify 90 degrees for vertically pointing radars.
        The default is 0 degrees.

    Returns
    -------
    xarray.Dataset
        Dataset containing the computed radar parameters.
    """
    # Subset dataset based on diameter max
    ds = filter_diameter_bins(ds=ds, maximum_diameter=diameter_max)

    # Define inputs
    da_drop_number_concentration = ds["drop_number_concentration"].compute()

    # Set all zeros drop number concentration to np.nan
    # --> Otherwise inf can appear in the output
    # --> Note that if a single np.nan is present, the output simulation will be NaN values
    valid_obs = da_drop_number_concentration.sum(dim=DIAMETER_DIMENSION) != 0
    da_drop_number_concentration = da_drop_number_concentration.where(valid_obs)

    # Define bin edges
    bin_edges = get_diameter_bin_edges(ds)

    # Check argument validity
    axis_ratio_model = check_axis_ratio_model(axis_ratio_model)
    permittivity_model = check_permittivity_model(permittivity_model)

    # Initialize scattering table
    scatterer = load_scatterer(
        frequency=frequency,
        num_points=num_points,
        diameter_max=diameter_max,
        canting_angle_std=canting_angle_std,
        axis_ratio_model=axis_ratio_model,
        permittivity_model=permittivity_model,
        water_temperature=water_temperature,
        elevation_angle=elevation_angle,
    )

    # Define kwargs
    kwargs = {
        "output_dictionary": False,
        "bin_edges": bin_edges,
        "scatterer": scatterer,
    }

    # Loop over each PSD (not in parallel --> dask="forbidden")
    # - It costs much more to initiate the scatterer rather than looping over timesteps !
    da_radar = xr.apply_ufunc(
        _estimate_empirical_radar_parameters,
        da_drop_number_concentration,
        kwargs=kwargs,
        input_core_dims=[["diameter_bin_center"]],
        output_core_dims=[["radar_variables"]],
        vectorize=True,
        dask="forbidden",
        dask_gufunc_kwargs={
            "output_sizes": {"radar_variables": len(RADAR_VARIABLES)},
        },  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )

    # Finalize radar dataset (add name, coordinates)
    ds_radar = _finalize_radar_dataset(
        da_radar=da_radar,
        frequency=frequency,
        num_points=num_points,
        diameter_max=diameter_max,
        canting_angle_std=canting_angle_std,
        axis_ratio_model=axis_ratio_model,
        permittivity_model=permittivity_model,
        water_temperature=water_temperature,
        elevation_angle=elevation_angle,
    )
    return ds_radar


def _finalize_radar_dataset(
    da_radar,
    frequency,
    num_points,
    diameter_max,
    canting_angle_std,
    axis_ratio_model,
    permittivity_model,
    water_temperature,
    elevation_angle,
):
    # Add parameters coordinates
    da_radar = da_radar.assign_coords({"radar_variables": RADAR_VARIABLES})

    # Create parameters dataset
    ds_radar = da_radar.to_dataset(dim="radar_variables")

    # Expand dimensions for later merging in get_radar_parameters()
    dims_dict = {
        "frequency": [frequency],
        "diameter_max": [diameter_max],
        "num_points": [num_points],
        "axis_ratio_model": [axis_ratio_model],
        "canting_angle_std": [canting_angle_std],
        "permittivity_model": [permittivity_model],
        "water_temperature": [water_temperature],
        "elevation_angle": [elevation_angle],
    }
    ds_radar = ds_radar.expand_dims(dim=dims_dict)
    return ds_radar


####----------------------------------------------------------------------
#### Wrapper for L2E and L2M products


def ensure_rounded_unique_array(arr, decimals=None):
    """Ensure that the input array is a unique, rounded array."""
    arr = np.atleast_1d(arr)
    if decimals is not None:
        arr = arr.round(decimals)
    return np.unique(arr)


def get_list_simulations_params(
    frequency,
    num_points,
    diameter_max,
    canting_angle_std,
    axis_ratio_model,
    permittivity_model,
    water_temperature,
    elevation_angle,
):
    """Return list with the set of parameters required for each simulation."""
    # Ensure numeric frequencies
    frequency = ensure_numerical_frequency(frequency)

    # Ensure arguments are unique set of values
    # - Otherwise problems with non-unique xarray dataset coordinates
    frequency = ensure_rounded_unique_array(frequency, decimals=2)
    num_points = ensure_rounded_unique_array(num_points, decimals=0)
    diameter_max = ensure_rounded_unique_array(diameter_max, decimals=1)
    canting_angle_std = ensure_rounded_unique_array(canting_angle_std, decimals=1)
    axis_ratio_model = ensure_rounded_unique_array(axis_ratio_model)
    permittivity_model = ensure_rounded_unique_array(permittivity_model)
    water_temperature = ensure_rounded_unique_array(water_temperature, decimals=1)
    elevation_angle = ensure_rounded_unique_array(elevation_angle, decimals=1)

    # Check parameters validity
    axis_ratio_model = [check_axis_ratio_model(model) for model in axis_ratio_model]
    permittivity_model = [check_permittivity_model(model) for model in permittivity_model]

    # Order frequency from lowest to highest
    # --> ['S', 'C', 'X', 'Ku', 'K', 'Ka', 'W']
    frequency = sorted(frequency)

    # Retrieve combination of parameters
    list_params = [
        {
            "frequency": freq.item(),
            "diameter_max": d_max.item(),
            "num_points": n_p.item(),
            "canting_angle_std": cas.item(),
            "axis_ratio_model": ar.item(),
            "permittivity_model": perm.item(),
            "water_temperature": t_w.item(),
            "elevation_angle": el.item(),
        }
        for freq, d_max, n_p, cas, ar, perm, t_w, el in itertools.product(
            frequency,
            diameter_max,
            num_points,
            canting_angle_std,
            axis_ratio_model,
            permittivity_model,
            water_temperature,
            elevation_angle,
        )
    ]
    return list_params


def get_radar_parameters(
    ds,
    frequency=None,
    num_points=1024,
    diameter_max=8,
    canting_angle_std=7,
    axis_ratio_model="Thurai2007",
    permittivity_model="Turner2016",
    water_temperature=10,
    elevation_angle=0,
    parallel=True,
):
    """Compute radar parameters from empirical drop number concentration or PSD model.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the drop number concentration variable.
    frequency : str, float, or list of str and float, optional
        Frequencies in GHz for which to compute the radar parameters.
        Alternatively, also strings can be used to specify common radar frequencies.
        If ``None``, the common radar frequencies will be used.
        See ``disdrodb.scattering.available_radar_bands()``.
    num_points: int or lis tof integer, optional
        Number of bins into which discretize the PSD.
    diameter_max : float or list of float, optional
        Maximum diameter. The default value is 8 mm.
    canting_angle_std : float or list of float, optional
        Standard deviation of the canting angle.  The default value is 7.
    axis_ratio_model : str or list of str, optional
        Models to compute the axis ratio. The default model is ``Thurai2007``.
        See available models with ``disdrodb.scattering.available_axis_ratio_models()``.
    permittivity_model : str str or list of str, optional
        Permittivity model to use to compute the refractive index and the
        rayleigh_dielectric_factor. The default is ``Turner2016``.
        See available models with ``disdrodb.scattering.available_permittivity_models()``.
    water_temperature : float or list of float, optional
        Water temperature in degree Celsius to be used in the permittivity model.
        The default is 10 degC.
    elevation_angle: float or list of float, optional
        Radar elevation angles in degrees.
        Specify 90 degrees for vertically pointing radars.
        The default is 0 degrees.
    parallel : bool, optional
        Whether to compute radar variables in parallel.
        The default value is ``True``.

    Returns
    -------
    xarray.Dataset
        Dataset containing the computed radar parameters.
    """
    # Decide whether to simulate radar parameters based on empirical PSD or model PSD
    if "disdrodb_psd_model" not in ds.attrs and "drop_number_concentration" not in ds:
        raise ValueError("The input dataset is not a DISDRODB L2E or L2M product.")

    # Model-based simulation
    if "disdrodb_psd_model" in ds.attrs:
        func = get_model_radar_parameters
        ds_subset = get_psd_parameters(ds).compute()
    # Empirical PSD simulation
    else:
        func = get_empirical_radar_parameters
        ds_subset = ds[["drop_number_concentration"]].compute()

    # Define default frequencies if not specified
    if frequency is None:
        frequency = available_radar_bands()

    # Define parameters for all requested simulations
    list_params = get_list_simulations_params(
        frequency=frequency,
        num_points=num_points,
        diameter_max=diameter_max,
        canting_angle_std=canting_angle_std,
        axis_ratio_model=axis_ratio_model,
        permittivity_model=permittivity_model,
        water_temperature=water_temperature,
        elevation_angle=elevation_angle,
    )

    # Compute radar variables for each configuration in parallel
    # - The function expects the data into memory (no dask arrays !)
    if parallel:
        list_ds = [dask.delayed(func)(ds_subset, **params) for params in list_params]
        list_ds = dask.compute(*list_ds)
    else:
        list_ds = [func(ds_subset, **params) for params in list_params]

    # Merge into a single dataset
    ds_radar = xr.merge(list_ds)

    # Order frequency from lowest to highest
    # --> ['S', 'C', 'X', 'Ku', 'K', 'Ka', 'W']
    frequency = sorted(ds_radar["frequency"].to_numpy())
    ds_radar = ds_radar.sel(frequency=frequency)

    # Map default frequency to classical radar band
    # --> This transform the frequency coordinate to dtype object
    ds_radar = _replace_common_frequency_with_radar_band(ds_radar)

    # Copy global attributes from input dataset
    ds_radar.attrs = ds.attrs.copy()

    # Remove single dimensions and add scattering settings information for single dimensions
    scattering_string = ""
    for param in RADAR_OPTIONS:
        if ds_radar.sizes[param] == 1:
            value = ds_radar[param].item()
            scattering_string += f"param: {value}; "

    if scattering_string != "":
        ds_radar.attrs["disdrodb_scattering_options"] = scattering_string
    ds_radar = ds_radar.squeeze()
    return ds_radar


def _map_frequency_to_band(f):
    """Function to map frequency value to radar band."""
    for band, val in frequency_dict.items():
        if np.isclose(f, val):
            return band
    return f


def _replace_common_frequency_with_radar_band(ds_radar):
    """Replace dataset coordinates with radar band if the case."""
    # Map frequencies to radar bands
    frequency = ds_radar["frequency"].to_numpy()
    frequency = [_map_frequency_to_band(f) for f in frequency]
    # Update dataset with new coordinate labels
    ds_radar = ds_radar.assign_coords({"frequency": ("frequency", frequency)})
    return ds_radar
