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
from pytmatrix import orientation, radar, refractive, tmatrix_aux
from pytmatrix.psd import PSDIntegrator
from pytmatrix.tmatrix import Scatterer

from disdrodb import DIAMETER_DIMENSION, get_scattering_table_dir
from disdrodb.l1.filters import filter_diameter_bins
from disdrodb.psd.models import BinnedPSD, create_psd, get_required_parameters
from disdrodb.scattering.axis_ratio import check_axis_ratio, get_axis_ratio_method
from disdrodb.utils.logger import log_info
from disdrodb.utils.manipulations import get_diameter_bin_edges
from disdrodb.utils.warnings import suppress_warnings

logger = logging.getLogger(__name__)

# Wavelengths for which the refractive index is defined in pytmatrix (in mm)
wavelength_dict = {
    "S": tmatrix_aux.wl_S,
    "C": tmatrix_aux.wl_C,
    "X": tmatrix_aux.wl_X,
    "Ku": tmatrix_aux.wl_Ku,
    "Ka": tmatrix_aux.wl_Ka,
    "W": tmatrix_aux.wl_W,
}


def available_radar_bands():
    """Return a list of the available radar bands."""
    return list(wavelength_dict)


def check_radar_band(radar_band):
    """Check the validity of the specified radar band."""
    available_bands = available_radar_bands()
    if radar_band not in available_bands:
        raise ValueError(f"{radar_band} is an invalid radar band. Valid radar bands: {available_bands}.")
    return radar_band


def get_radar_wavelength(radar_band):
    """Get the wavelength in mm of a radar band."""
    wavelength = wavelength_dict[radar_band]
    return wavelength


def initialize_scatterer(wavelength, num_points=1024, diameter_max=8, canting_angle_std=7, axis_ratio="Thurai2007"):
    """Initialize T-matrix scatterer object for a given wavelength."""
    # Retrieve custom axis ratio function
    axis_ratio_func = get_axis_ratio_method(axis_ratio)

    # Define water permittivity: complex refractive index
    # - Here we currently assume 10 °C
    # - m_w_0C and m_w_20C are also available
    # - As function of radar frequency !
    # TODO: should be another dimension ? Or use scatterer.psd_integrator.m_func?
    water_refractive_index = refractive.m_w_10C[wavelength]

    # Define radar dielectric factor
    # - Here we use pyTmatrix default
    # - Impacts only reflectivity calculation, not scattering table generation
    # --> TODO: to be adapted as function of water_refractive_index?
    Kw_sqr = 0.93

    # ---------------------------------------------------------------.
    # For W band, diameter_max up to 9.5, otherwise kernel dies !
    if wavelength < 3.5:
        diameter_max = min(diameter_max, 9.5)

    # ---------------------------------------------------------------.
    # Initialize Scatterer class
    # - By specifying m, we assume same refractive index for all particles diameters
    # - thet0, thet0, thet: The zenith angles of incident and scattered radiation (default to 90)
    # - phi0, phi: The azimuth angles of incident and scattered radiation (default to 0 and 180)
    # - alpha, beta: Defaults to 0.0, 0.0. Valid values: alpha = [0, 360] beta = [0, 180]
    scatterer = Scatterer(wavelength=wavelength, m=water_refractive_index, Kw_sqr=Kw_sqr)
    # - Define particle orientation PDF for orientational averaging
    # --> The standard deviation of the angle with respect to vertical orientation (the canting angle).
    scatterer.or_pdf = orientation.gaussian_pdf(std=canting_angle_std)

    # - Define orientation methods
    # --> Alternatives: orient_averaged_adaptive, orient_single,
    # --> Speed: orient_single > orient_averaged_fixed > orient_averaged_adaptive
    scatterer.orient = orientation.orient_averaged_fixed

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
    scatterer.psd_integrator.geometries = (tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_horiz_forw)
    # ---------------------------------------------------------------.
    # Initialize scattering table
    scatterer.psd_integrator.init_scatter_table(scatterer)
    return scatterer


def load_scatterer(
    wavelength,
    diameter_max,
    num_points=1024,
    canting_angle_std=7,
    axis_ratio="Thurai2007",
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
    wavelength : float
        Radar wavelength in millimeters.
    num_points: int
        Number of bins into which discretize the PSD.
    diameter_max : float
        Maximum drop diameter in millimeters for the scattering table.
    canting_angle_std : float, optional
        Standard deviation of the canting angle distribution in degrees,
        by default 7.
    axis_ratio : str or callable, optional
        Axis ratio model used to shape hydrometeors. Can be a string identifier
        (e.g., "Thurai2007") or a callable. Default is "Thurai2007".
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
    # Retrieve scattering table directory
    scattering_table_dir = get_scattering_table_dir(scattering_table_dir)

    # Define a filename based on the key parameters
    filename = "_".join(
        [
            "ScatterTable",
            f"wl-{wavelength:.2f}",
            f"dmax-{diameter_max:.1f}",
            f"npts-{num_points}",
            f"cant-{canting_angle_std:.1f}",
            f"ar-{axis_ratio}.pkl",
        ],
    )
    scatter_table_filepath = os.path.join(scattering_table_dir, filename)

    # Load or create scattering table
    if os.path.exists(scatter_table_filepath):
        water_refractive_index = refractive.m_w_10C[wavelength]
        scatterer = Scatterer(wavelength=wavelength, m=water_refractive_index)
        scatterer.psd_integrator = PSDIntegrator()
        _ = scatterer.psd_integrator.load_scatter_table(scatter_table_filepath)

    else:
        if verbose:
            msg = f"- Computing pyTmatrix scattering table {filename}"
            log_info(logger=logger, msg=msg, verbose=verbose)

        # Initialize scatterer
        scatterer = initialize_scatterer(
            wavelength=wavelength,
            canting_angle_std=canting_angle_std,
            diameter_max=diameter_max,
            axis_ratio=axis_ratio,
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
    radar_vars = {}

    # Set backward scattering for reflectivity calculations
    scatterer.set_geometry(tmatrix_aux.geom_horiz_back)

    radar_vars["DBZH"] = 10 * np.log10(radar.refl(scatterer, h_pol=True))  # dBZ
    radar_vars["DBZV"] = 10 * np.log10(radar.refl(scatterer, h_pol=False))  # dBZ

    radar_vars["ZDR"] = 10 * np.log10(radar.Zdr(scatterer))  # dB
    radar_vars["LDR"] = 10 * np.log10(radar.ldr(scatterer))  # dBZ
    radar_vars["RHOHV"] = radar.rho_hv(scatterer)  # deg/km
    radar_vars["DELTAHV"] = radar.delta_hv(scatterer) * 180.0 / np.pi  # [deg]

    # Set forward scattering for attenuation and phase calculations
    scatterer.set_geometry(tmatrix_aux.geom_horiz_forw)
    radar_vars["KDP"] = radar.Kdp(scatterer)  # deg/km
    radar_vars["AH"] = radar.Ai(scatterer, h_pol=True)  # dB/km
    radar_vars["AV"] = radar.Ai(scatterer, h_pol=False)  # dB/km
    radar_vars["ADR"] = radar_vars["AH"] - radar_vars["AV"]  # dB/km
    return radar_vars


# Radar variables computed by DISDRODB
# - Must reflect dictionary order output of compute_radar_variables
RADAR_VARIABLES = ["DBZH", "DBZH", "ZDR", "LDR", "RHOHV", "DELTAHV", "KDP", "AH", "AV", "ADR"]


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
        radar_vars = compute_radar_variables(scatterer)
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
    radar_band,
    num_points=1024,
    diameter_max=10,
    canting_angle_std=7,
    axis_ratio="Thurai2007",
):
    """Compute radar parameters from a PSD model.

    This function retrieve values for a single set of parameter only !

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the parameters of the PSD model.
        The dataset attribute disdrodb_psd_model specifies the PSD model to use.
    radar_band : str
        Radar band to be used.
    canting_angle_std : float, optional
        Standard deviation of the canting angle.  The default value is 10.
    diameter_max : float, optional
        Maximum diameter. The default value is 8 mm.
    axis_ratio : str, optional
        Method to compute the axis ratio. The default method is ``Thurai2007``.

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
    axis_ratio = check_axis_ratio(axis_ratio)
    radar_band = check_radar_band(radar_band)

    # Retrieve wavelengths in mm
    wavelength = get_radar_wavelength(radar_band)

    # Create DataArray with PSD parameters
    da_parameters = ds_parameters.to_array(dim="psd_parameters").compute()

    # Initialize scattering table
    scatterer = load_scatterer(
        wavelength=wavelength,
        num_points=num_points,
        diameter_max=diameter_max,
        canting_angle_std=canting_angle_std,
        axis_ratio=axis_ratio,
    )

    # Define kwargs
    kwargs = {
        "output_dictionary": False,
        "psd_model": psd_model,
        "psd_parameters_names": required_parameters,
        "scatterer": scatterer,
    }

    # Loop over each PSD (not in parallel --> dask="forbidden")
    # - It costs much more to initiate the scatterer rather than looping over timesteps !
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
        radar_band=radar_band,
        num_points=num_points,
        diameter_max=diameter_max,
        canting_angle_std=canting_angle_std,
        axis_ratio=axis_ratio,
    )
    return ds_radar


def get_empirical_radar_parameters(
    ds,
    radar_band,
    num_points=1024,
    diameter_max=10,
    canting_angle_std=7,
    axis_ratio="Thurai2007",
):
    """Compute radar parameters from an empirical drop number concentration.

    This function retrieve values for a single set of parameter only !

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the drop number concentration variable.
    radar_band : str
        Radar band to be used.
    canting_angle_std : float, optional
        Standard deviation of the canting angle.  The default value is 10.
    diameter_max : float, optional
        Maximum diameter. The default value is 8 mm.
    axis_ratio : str, optional
        Method to compute the axis ratio. The default method is ``Thurai2007``.

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
    axis_ratio = check_axis_ratio(axis_ratio)
    radar_band = check_radar_band(radar_band)

    # Retrieve wavelengths in mm
    wavelength = get_radar_wavelength(radar_band)

    # Initialize scattering table
    scatterer = load_scatterer(
        wavelength=wavelength,
        num_points=num_points,
        diameter_max=diameter_max,
        canting_angle_std=canting_angle_std,
        axis_ratio=axis_ratio,
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
        radar_band=radar_band,
        num_points=num_points,
        diameter_max=diameter_max,
        canting_angle_std=canting_angle_std,
        axis_ratio=axis_ratio,
    )
    return ds_radar


def _finalize_radar_dataset(da_radar, radar_band, num_points, diameter_max, canting_angle_std, axis_ratio):
    # Add parameters coordinates
    da_radar = da_radar.assign_coords({"radar_variables": RADAR_VARIABLES})

    # Create parameters dataset
    ds_radar = da_radar.to_dataset(dim="radar_variables")

    # Expand dimensions for later merging in get_radar_parameters()
    dims_dict = {
        "radar_band": [radar_band],
        "diameter_max": [diameter_max],
        "num_points": [num_points],
        "axis_ratio": [axis_ratio],
        "canting_angle_std": [canting_angle_std],
    }
    ds_radar = ds_radar.expand_dims(dim=dims_dict)
    return ds_radar


####----------------------------------------------------------------------
#### Wrapper for L2E and L2M products


def get_list_simulations_params(radar_band, num_points, diameter_max, canting_angle_std, axis_ratio):
    """Return list with the set of parameters required for each simulation."""
    # Ensure parameters are list
    radar_band = np.atleast_1d(radar_band)
    num_points = np.atleast_1d(num_points)
    diameter_max = np.atleast_1d(diameter_max)
    canting_angle_std = np.atleast_1d(canting_angle_std)
    axis_ratio = np.atleast_1d(axis_ratio)

    # Check parameters validity
    axis_ratio = [check_axis_ratio(method) for method in axis_ratio]
    radar_band = [check_radar_band(band) for band in radar_band]

    # Order radar band from longest to shortest wavelength
    # - ["S", "C", "X", "Ku", "Ka", "W"]
    radar_band = sorted(radar_band, key=lambda x: wavelength_dict[x])[::-1]

    # Retrieve combination of parameters
    list_params = [
        {
            "radar_band": rb.item(),
            "canting_angle_std": cas.item(),
            "axis_ratio": ar.item(),
            "diameter_max": d_max.item(),
            "num_points": n_p.item(),
        }
        for rb, cas, ar, d_max, n_p in itertools.product(
            radar_band,
            canting_angle_std,
            axis_ratio,
            diameter_max,
            num_points,
        )
    ]
    return list_params


def get_radar_parameters(
    ds,
    radar_band=None,
    num_points=1024,
    diameter_max=8,
    canting_angle_std=7,
    axis_ratio="Thurai2007",
    parallel=True,
):
    """Compute radar parameters from empirical drop number concentration or PSD model.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the drop number concentration variable.
    radar_band : str or list of str, optional
        Radar band(s) to be used.
        If ``None`` (the default), all available radar bands are used.
    canting_angle_std : float or list of float, optional
        Standard deviation of the canting angle.  The default value is 7.
    diameter_max : float or list of float, optional
        Maximum diameter. The default value is 8 mm.
    axis_ratio : str or list of str, optional
        Method to compute the axis ratio. The default method is ``Thurai2007``.
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

    # Initialize radar band if not provided
    if radar_band is None:
        radar_band = available_radar_bands()

    # Define parameters for all requested simulations
    list_params = get_list_simulations_params(
        radar_band=radar_band,
        num_points=num_points,
        diameter_max=diameter_max,
        canting_angle_std=canting_angle_std,
        axis_ratio=axis_ratio,
    )

    # Compute radar variables for each configuration in parallel
    # - The function expects the data into memory (no dask arrays !)
    if parallel:
        list_ds = [dask.delayed(func)(ds_subset, **params) for params in list_params]
        list_ds = dask.compute(*list_ds)
    else:
        list_ds = [func(ds_subset, **params) for params in list_params]

    # Merge into a single dataset
    # - Order radar bands from longest to shortest wavelength
    ds_radar = xr.merge(list_ds)
    ds_radar = ds_radar.sel(radar_band=radar_band)

    # Copy global attributes from input dataset
    ds_radar.attrs = ds.attrs.copy()

    # Remove single dimensions and add scattering settings information for single dimensions
    parameters = ["radar_band", "num_points", "diameter_max", "canting_angle_std", "axis_ratio"]
    scattering_string = ""
    for param in parameters:
        if ds_radar.sizes[param] == 1:
            value = ds_radar[param].item()
            scattering_string += f"param: {value}; "

    if scattering_string != "":
        ds_radar.attrs[f"disdrodb_scattering_{param}"] = scattering_string
    ds_radar = ds_radar.squeeze()
    return ds_radar
