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
"""Implement DISDRODB L2 processing."""

import numpy as np
import xarray as xr

from disdrodb.l1.encoding_attrs import get_attrs_dict, get_encoding_dict
from disdrodb.l1.fall_velocity import get_raindrop_fall_velocity
from disdrodb.l1_env.routines import load_env_dataset
from disdrodb.l2.empirical_dsd import (
    compute_integral_parameters,
    compute_qc_bins_metrics,
    compute_spectrum_parameters,
    get_drop_average_velocity,
    get_drop_number_concentration,
    get_effective_sampling_area,
    get_kinetic_energy_variables_from_drop_number,
    get_rain_accumulation,
    get_rain_rate_from_drop_number,
)
from disdrodb.psd import create_psd, estimate_model_parameters
from disdrodb.psd.fitting import compute_gof_stats
from disdrodb.utils.attrs import set_attrs
from disdrodb.utils.decorators import check_pytmatrix_availability
from disdrodb.utils.encoding import set_encodings
from disdrodb.utils.time import ensure_sample_interval_in_seconds


def define_diameter_array(diameter_min=0, diameter_max=10, diameter_spacing=0.05):
    """
    Define an array of diameters and their corresponding bin properties.

    Parameters
    ----------
    diameter_min : float, optional
        The minimum diameter value. The default value is 0 mm.
    diameter_max : float, optional
        The maximum diameter value. The default value is 10 mm.
    diameter_spacing : float, optional
        The spacing between diameter values. The default value is 0.05 mm.

    Returns
    -------
    xr.DataArray
        A DataArray containing the center of each diameter bin, with coordinates for
        the bin width, lower bound, upper bound, and center.

    """
    diameters_bounds = np.arange(diameter_min, diameter_max + diameter_spacing / 2, step=diameter_spacing)
    diameters_bin_lower = diameters_bounds[:-1]
    diameters_bin_upper = diameters_bounds[1:]
    diameters_bin_width = diameters_bin_upper - diameters_bin_lower
    diameters_bin_center = diameters_bin_lower + diameters_bin_width / 2
    da = xr.DataArray(
        diameters_bin_center,
        dims="diameter_bin_center",
        coords={
            "diameter_bin_width": ("diameter_bin_center", diameters_bin_width),
            "diameter_bin_lower": ("diameter_bin_center", diameters_bin_lower),
            "diameter_bin_upper": ("diameter_bin_center", diameters_bin_upper),
            "diameter_bin_center": ("diameter_bin_center", diameters_bin_center),
        },
    )
    return da


def define_velocity_array(ds):
    """
    Create the fall velocity DataArray using various methods.

    If 'velocity_bin_center' is a dimension in the dataset, returns a Dataset
    with 'measured_velocity', 'average_velocity', and 'fall_velocity' as variables.
    Otherwise, returns the 'fall_velocity' DataArray from the input dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing velocity variables.

    Returns
    -------
    velocity: xarray.DataArray
    """
    drop_number = ds["drop_number"]
    if "velocity_bin_center" in ds.dims:
        velocity = xr.Dataset(
            {
                "fall_velocity": xr.ones_like(drop_number) * ds["fall_velocity"],
                "measured_velocity": xr.ones_like(drop_number) * ds["velocity_bin_center"],
            },
        ).to_array(dim="velocity_method")
    else:
        velocity = ds["fall_velocity"]
    return velocity


####--------------------------------------------------------------------------
#### L2 Empirical Parameters


def generate_l2_empirical(ds, ds_env=None, compute_spectra=False):
    """Generate the DISDRODB L2E dataset from the DISDRODB L1 dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        DISDRODB L1 dataset.
    ds_env : xarray.Dataset, optional
    Environmental dataset used for fall velocity and water density estimates.
    If None, a default environment dataset will be loaded.

    Returns
    -------
    xarray.Dataset
        DISRODB L2E dataset.
    """
    # Initialize L2E dataset
    ds_l2 = xr.Dataset()

    # Retrieve attributes
    attrs = ds.attrs.copy()

    # -------------------------------------------------------
    #### Preprocessing
    # Discard all timesteps without measured drops
    # - This allow to speed up processing
    # - Regularization can be done at the end
    ds = ds.isel(time=ds["N"] > 0)

    # Count number of diameter bins with data
    if "Nbins" not in ds:
        # Add bins statistics
        ds.update(compute_qc_bins_metrics(ds))

    # Retrieve ENV dataset or take defaults
    # --> Used for fall velocity and water density estimates
    if ds_env is None:
        ds_env = load_env_dataset(ds)

    # TODO: Derive water density as function of ENV (temperature, ...)
    # -->  (T == 10){density_water <- 999.7}else if(T == 20){density_water <- 998.2}else{density_water <- 995.7}
    water_density = 1000  # kg / m3

    # Determine if the velocity dimension is available
    has_velocity_dimension = "velocity_bin_center" in ds.dims

    # -------------------------------------------------------
    # Extract variables from L1
    sensor_name = ds.attrs["sensor_name"]
    diameter = ds["diameter_bin_center"] / 1000  # m
    diameter_bin_width = ds["diameter_bin_width"]  # mm
    drop_number = ds["drop_number"]
    sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"])  # s

    # Compute sampling area [m2]
    sampling_area = get_effective_sampling_area(sensor_name=sensor_name, diameter=diameter)  # m2

    # Copy relevant L1 variables to L2 product
    variables = [
        "drop_number",  # 2D V x D
        "drop_counts",  # 1D D
        "sample_interval",
        "N",
        "Nremoved",
        "Dmin",
        "Dmax",
        "fall_velocity",
    ]

    variables = [var for var in variables if var in ds]
    ds_l2.update(ds[variables])

    # -------------------------------------------------------------------------------------------
    # Compute and add drop average velocity if an optical disdrometer (i.e OTT Parsivel or ThiesLPM)
    # - We recompute it because if the input dataset is aggregated, it must be updated !
    if has_velocity_dimension:
        ds["drop_average_velocity"] = get_drop_average_velocity(ds["drop_number"])

    # -------------------------------------------------------------------------------------------
    # Define velocity array with dimension 'velocity_method'
    velocity = define_velocity_array(ds)

    # Compute drop number concentration (Nt) [#/m3/mm]
    drop_number_concentration = get_drop_number_concentration(
        drop_number=drop_number,
        velocity=velocity,
        diameter_bin_width=diameter_bin_width,
        sample_interval=sample_interval,
        sampling_area=sampling_area,
    )
    ds_l2["drop_number_concentration"] = drop_number_concentration

    # -------------------------------------------------------
    #### Compute L2 spectra
    if compute_spectra:
        ds_spectrum = compute_spectrum_parameters(
            drop_number_concentration,
            velocity=ds["fall_velocity"],
            diameter=diameter,
            sample_interval=sample_interval,
            water_density=water_density,
        )
        ds_l2.update(ds_spectrum)

    # ----------------------------------------------------------------------------
    #### Compute L2 integral parameters from drop_number_concentration
    ds_parameters = compute_integral_parameters(
        drop_number_concentration=drop_number_concentration,
        velocity=ds["fall_velocity"],
        diameter=diameter,
        diameter_bin_width=diameter_bin_width,
        sample_interval=sample_interval,
        water_density=water_density,
    )

    # -------------------------------------------------------
    #### Compute R and P from drop number (without velocity assumptions)
    # - Rain rate and accumulation computed with this method are not influenced by the fall velocity of drops !
    ds_l2["Rm"] = get_rain_rate_from_drop_number(
        drop_number=drop_number,
        sampling_area=sampling_area,
        diameter=diameter,
        sample_interval=sample_interval,
    )
    # Compute rain accumulation (P) [mm]
    ds_l2["Pm"] = get_rain_accumulation(rain_rate=ds_l2["Rm"], sample_interval=sample_interval)

    # -------------------------------------------------------
    #### Compute KE integral parameters directly from drop_number
    # - The kinetic energy variables can be computed using the actual measured fall velocity by the sensor.
    if has_velocity_dimension:
        ds_ke = get_kinetic_energy_variables_from_drop_number(
            drop_number=drop_number,
            diameter=diameter,
            velocity=velocity,
            sampling_area=sampling_area,
            sample_interval=sample_interval,
            water_density=water_density,
        )
        # Combine integral parameters
        ke_vars = list(ds_ke.data_vars)
        ds_ke = ds_ke.expand_dims(dim={"source": ["drop_number"]}, axis=-1)
        ds_ke_dsd = ds_parameters[ke_vars].expand_dims(dim={"source": ["drop_number_concentration"]}, axis=-1)
        ds_ke = xr.concat((ds_ke_dsd, ds_ke), dim="source")
        ds_parameters = ds_parameters.drop_vars(ke_vars)
        for var in ke_vars:
            ds_parameters[var] = ds_ke[var]

    # ----------------------------------------------------------------------------
    #### Finalize L2 Dataset
    # Add DSD integral parameters
    ds_l2.update(ds_parameters)

    # ----------------------------------------------------------------------------.
    #### Add encodings and attributes
    # Add variables attributes
    attrs_dict = get_attrs_dict()
    ds_l2 = set_attrs(ds_l2, attrs_dict=attrs_dict)

    # Add variables encoding
    encoding_dict = get_encoding_dict()
    ds_l2 = set_encodings(ds_l2, encoding_dict=encoding_dict)

    # Add global attributes
    ds_l2.attrs = attrs

    return ds_l2


####--------------------------------------------------------------------------
#### L2 Model Parameters


def generate_l2_model(
    ds,
    ds_env=None,
    fall_velocity_method="Beard1976",
    # PSD discretization
    diameter_min=0,
    diameter_max=10,
    diameter_spacing=0.05,
    # Fitting options
    psd_model=None,
    optimization=None,
    optimization_kwargs=None,
    # Filtering options
    min_nbins=4,
    remove_timesteps_with_few_bins=False,
    mask_timesteps_with_few_bins=False,
    # GOF metrics options
    gof_metrics=True,
):
    """
    Generate the DISDRODB L2M dataset from a DISDRODB L2E dataset.

    This function estimates PSD model parameters and successively computes DSD integral parameters.
    Optionally, radar variables at various bands are simulated using T-matrix simulations.
    Goodness-of-fit metrics of the PSD can also be optionally included into the output dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        DISDRODB L2E dataset.
    ds_env : xarray.Dataset, optional
        Environmental dataset used for fall velocity and water density estimates.
        If None, a default environment dataset will be loaded.
    diameter_min : float, optional
        Minimum PSD diameter. The default value is 0 mm.
    diameter_max : float, optional
        Maximum PSD diameter. The default value is 8 mm.
    diameter_spacing : float, optional
        PSD diameter spacing. The default value is 0.05 mm.
    psd_model : str
        The PSD model to fit. See ``available_psd_models()``.
    optimization : str, optional
        The fitting optimization procedure. Either "GS" (Grid Search), "ML (Maximum Likelihood)
        or "MOM" (Method of Moments).
    optimization_kwargs : dict, optional
        Dictionary with arguments to customize the fitting procedure.
    gof_metrics : bool, optional
        Whether to add goodness-of-fit metrics to the output dataset. The default is True.

    Returns
    -------
    xarray.Dataset
        DISDRODB L2M dataset.
    """
    # ----------------------------------------------------------------------------.
    #### NOTES
    # - Final processing: Optionally filter dataset only when PSD has fitted ?
    # --> but good to have everything to compare across models

    # ----------------------------------------------------------------------------.
    # Retrieve attributes
    attrs = ds.attrs.copy()

    # -------------------------------------------------------
    # Derive water density as function of ENV (temperature, ...)
    # TODO --> Add into ds_env !
    # -->  (T == 10){density_water <- 999.7}else if(T == 20){density_water <- 998.2}else{density_water <- 995.7}
    water_density = 1000  # kg / m3

    ####------------------------------------------------------.
    #### Preprocessing
    # Count number of diameter bins with data
    if "Nbins" not in ds:
        # Add bins statistics
        ds.update(compute_qc_bins_metrics(ds))

    # Identify timesteps with enough diameter bins with counted trops
    valid_timesteps = ds["Nbins"] >= min_nbins

    # Drop such timesteps if asked
    if remove_timesteps_with_few_bins:
        mask_timesteps_with_few_bins = False
        ds = ds.isel(time=valid_timesteps, drop=False)

    # Retrieve ENV dataset or take defaults
    # --> Used for fall velocity and water density estimates
    if ds_env is None:
        ds_env = load_env_dataset(ds)

    ####------------------------------------------------------.
    #### Define default PSD optimization arguments
    if psd_model is None and optimization is None:
        psd_model = "NormalizedGammaPSD"
        optimization = "GS"
        optimization_kwargs = {
            "target": "ND",
            "transformation": "identity",
            "error_order": 1,  # MAE
        }

    ####------------------------------------------------------.
    #### Retrieve PSD parameters
    ds_psd_params = estimate_model_parameters(
        ds=ds,
        psd_model=psd_model,
        optimization=optimization,
        optimization_kwargs=optimization_kwargs,
    )

    ####------------------------------------------------------.
    #### Mask timesteps with few bins if asked
    if mask_timesteps_with_few_bins:
        ds_psd_params = ds_psd_params.where(valid_timesteps)

    ####-------------------------------------------------------
    #### Create PSD
    psd_name = ds_psd_params.attrs["disdrodb_psd_model"]
    psd = create_psd(psd_name, parameters=ds_psd_params)

    ####-------------------------------------------------------
    #### Compute integral parameters
    # Define diameter array
    diameter = define_diameter_array(
        diameter_min=diameter_min,
        diameter_max=diameter_max,
        diameter_spacing=diameter_spacing,
    )
    diameter_bin_width = diameter["diameter_bin_width"]

    # Retrieve time of integration
    # - If dataset is opened with decode_timedelta=False, sample_interval is already in seconds !
    sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"])

    # Retrieve drop number concentration
    drop_number_concentration = psd(diameter)

    # Retrieve fall velocity for each new diameter bin
    velocity = get_raindrop_fall_velocity(diameter=diameter, method=fall_velocity_method, ds_env=ds_env)  # mm

    # Compute integral parameters
    ds_params = compute_integral_parameters(
        drop_number_concentration=drop_number_concentration,
        velocity=velocity,
        diameter=diameter / 1000,  # in meters !
        diameter_bin_width=diameter_bin_width,
        sample_interval=sample_interval,
        water_density=water_density,
    )

    #### ----------------------------------------------------------------------------
    #### Create L2 Dataset
    # Update with PSD parameters
    ds_params.update(ds_psd_params)

    # Add GOF statistics if asked
    if gof_metrics:
        ds_gof = compute_gof_stats(drop_number_concentration=ds["drop_number_concentration"], psd=psd)
        ds_params.update(ds_gof)

    #### ----------------------------------------------------------------------------.
    #### Add encodings and attributes
    # Add variables attributes
    attrs_dict = get_attrs_dict()
    ds_params = set_attrs(ds_params, attrs_dict=attrs_dict)

    # Add variables encoding
    encoding_dict = get_encoding_dict()
    ds_params = set_encodings(ds_params, encoding_dict=encoding_dict)

    # Add global attributes
    ds_params.attrs = attrs
    ds_params.attrs["disdrodb_psd_model"] = psd_name

    # Return dataset
    return ds_params


####-------------------------------------------------------------------------------------------------------------------.
#### L2 Radar Parameters


@check_pytmatrix_availability
def generate_l2_radar(
    ds,
    radar_band=None,
    canting_angle_std=7,
    diameter_max=10,
    axis_ratio="Thurai2007",
    parallel=True,
):
    """Simulate polarimetric radar variables from empirical drop number concentration or the estimated PSD.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the drop number concentration variable or the PSD parameters.
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
    # Import here to avoid pytmatrix has mandatory dependency
    # - It is only needed for radar simulation
    from disdrodb.scattering import get_radar_parameters

    # Retrieve radar variables from L2E drop number concentration or from estimated L2M PSD model
    ds_radar = get_radar_parameters(
        ds=ds,
        radar_band=radar_band,
        canting_angle_std=canting_angle_std,
        diameter_max=diameter_max,
        axis_ratio=axis_ratio,
        parallel=parallel,
    )

    #### ----------------------------------------------------------------------------.
    #### Add encodings and attributes
    # Add variables attributes
    attrs_dict = get_attrs_dict()
    ds_radar = set_attrs(ds_radar, attrs_dict=attrs_dict)

    # Add variables encoding
    encoding_dict = get_encoding_dict()
    ds_radar = set_encodings(ds_radar, encoding_dict=encoding_dict)

    # Return dataset
    return ds_radar
