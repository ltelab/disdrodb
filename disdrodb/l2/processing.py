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

from disdrodb.constants import DIAMETER_DIMENSION
from disdrodb.fall_velocity import get_rain_fall_velocity
from disdrodb.l1_env.routines import load_env_dataset
from disdrodb.l2.empirical_dsd import (
    BINS_METRICS,
    add_bins_metrics,
    compute_integral_parameters,
    compute_spectrum_parameters,
    get_drop_number_concentration,
    get_effective_sampling_area,
    get_kinetic_energy_variables_from_drop_number,
    get_rain_accumulation,
    get_rain_rate_from_drop_number,
)
from disdrodb.psd import create_psd, estimate_model_parameters
from disdrodb.psd.fitting import compute_gof_stats
from disdrodb.utils.decorators import check_pytmatrix_availability
from disdrodb.utils.time import ensure_sample_interval_in_seconds
from disdrodb.utils.writer import finalize_product


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
#### Timesteps filtering functions


def select_timesteps_with_drops(ds, minimum_ndrops=0):
    """Select timesteps with at least the specified number of drops."""
    valid_timesteps = ds["N"].to_numpy() >= minimum_ndrops
    if not valid_timesteps.any().item():
        raise ValueError(f"No timesteps with N >= {minimum_ndrops}.")
    if "time" in ds.dims:
        ds = ds.isel(time=valid_timesteps, drop=False)
    return ds


def select_timesteps_with_minimum_nbins(ds, minimum_nbins):
    """Select timesteps with at least the specified number of diameter bins with drops."""
    if minimum_nbins == 0:
        return ds
    valid_timesteps = ds["Nbins"].to_numpy() >= minimum_nbins
    if not valid_timesteps.any().item():
        raise ValueError(f"No timesteps with Nbins >= {minimum_nbins}.")
    if "time" in ds.dims:
        ds = ds.isel(time=valid_timesteps, drop=False)
    return ds


def select_timesteps_with_minimum_rain_rate(ds, minimum_rain_rate):
    """Select timesteps with at least the specified rain rate."""
    if minimum_rain_rate == 0:
        return ds
    # Ensure dimensionality of R is 1
    # - Collapse velocity_method
    dims_to_agg = set(ds["R"].dims) - {"time"}
    da_r = ds["R"].max(dim=dims_to_agg)
    # Determine valid timesteps
    valid_timesteps = da_r.to_numpy() >= minimum_rain_rate
    if not valid_timesteps.any().item():
        raise ValueError(f"No timesteps with rain rate (R) >= {minimum_rain_rate} mm/hr.")
    if "time" in ds.dims:
        ds = ds.isel(time=valid_timesteps, drop=False)
    return ds


####--------------------------------------------------------------------------
#### L2 Empirical Parameters


def _ensure_present(container, required, kind):
    """Raise a ValueError if any of `required` are missing from the `container`."""
    missing = [item for item in required if item not in container]
    if missing:
        raise ValueError(f"Dataset is missing required {kind}: {', '.join(missing)}")


def check_l2e_input_dataset(ds):
    """Check dataset validity for L2E production."""
    from disdrodb.scattering import RADAR_OPTIONS

    # Check minimum required variables, coordinates and dimensions are presents
    required_variables = ["drop_number", "fall_velocity"]
    required_coords = [
        "diameter_bin_center",
        "diameter_bin_width",
        "sample_interval",
    ]
    required_attributes = ["sensor_name"]
    required_dims = [DIAMETER_DIMENSION]
    _ensure_present(list(ds.data_vars), required=required_variables, kind="variables")
    _ensure_present(list(ds.coords), required=required_coords, kind="coords")
    _ensure_present(list(ds.dims), required=required_dims, kind="dimensions")
    _ensure_present(list(ds.attrs), required=required_attributes, kind="attributes")

    # Remove dimensions and coordinates generated by L2E routine
    # - This allow to recursively repass L2E product to the generate_l2e function
    unallowed_dims = [dim for dim in ds.dims if dim in ["source", "velocity_method", *RADAR_OPTIONS]]
    ds = ds.drop_dims(unallowed_dims)
    unallowed_coords = [coord for coord in ds.coords if coord in ["source", "velocity_method", *RADAR_OPTIONS]]
    ds = ds.drop_vars(unallowed_coords)
    return ds


def generate_l2e(
    ds,
    ds_env=None,
    compute_spectra=False,
    compute_percentage_contribution=False,
    minimum_ndrops=1,
    minimum_nbins=1,
    minimum_rain_rate=0.01,
):
    """Generate the DISDRODB L2E dataset from the DISDRODB L1 dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        DISDRODB L1 dataset.
        Alternatively, a xarray dataset with at least:

            - variables: drop_number, fall_velocity
            - dimension: DIAMETER_DIMENSION
            - coordinates: diameter_bin_center, diameter_bin_width, sample_interval
            - attributes: sensor_name

    ds_env : xarray.Dataset, optional
        Environmental dataset used for fall velocity and water density estimates.
        If None, a default environment dataset will be loaded.

    Returns
    -------
    xarray.Dataset
        DISDRODB L2E dataset.
    """
    # Check and prepapre input dataset
    ds = check_l2e_input_dataset(ds)

    # -------------------------------------------------------
    # Initialize L2E dataset
    ds_l2 = xr.Dataset()

    # Retrieve attributes
    attrs = ds.attrs.copy()

    # -------------------------------------------------------
    #### Preprocessing
    # Select timesteps with at least the specified number of drops
    ds = select_timesteps_with_drops(ds, minimum_ndrops=minimum_ndrops)

    # Add bins metrics to resampled data if missing
    ds = add_bins_metrics(ds)

    # Remove timesteps with not enough bins with drops
    ds = select_timesteps_with_minimum_nbins(ds, minimum_nbins=minimum_nbins)

    # Retrieve ENV dataset or take defaults
    # --> Used for fall velocity and water density estimates
    if ds_env is None:
        ds_env = load_env_dataset(ds)
    water_density = ds_env.get("water_density", 1000)  # kg / m3

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
        "raw_drop_number",  # 2D V x D
        "drop_number",  # 2D V x D
        "drop_counts",  # 1D D
        "sample_interval",
        "N",
        "Nremoved",
        "Dmin",
        "Dmax",
        "fall_velocity",
        "qc_resampling",
        "time_qc",
    ]

    variables = [var for var in variables if var in ds]
    ds_l2.update(ds[variables])
    ds_l2.update(ds[BINS_METRICS])

    # -------------------------------------------------------------------------------------------
    # Compute and add drop average velocity if an optical disdrometer (i.e OTT Parsivel or ThiesLPM)
    # - We recompute it because if the input dataset is aggregated, it must be updated !
    # if has_velocity_dimension:
    #     ds["drop_average_velocity"] = get_drop_average_velocity(ds["drop_number"])

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
    #### Compute R, LWC, KE and Z spectra
    if compute_spectra:
        ds_spectrum = compute_spectrum_parameters(
            drop_number_concentration,
            velocity=ds["fall_velocity"],
            diameter=diameter,
            sample_interval=sample_interval,
            water_density=water_density,
        )
        ds_l2.update(ds_spectrum)

    if compute_percentage_contribution:
        # TODO: Implement percentage contribution computation
        pass

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
    # The kinetic energy variables can be computed using the actual measured fall velocity by the sensor.
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

    # Add DSD integral parameters
    ds_l2.update(ds_parameters)

    # ----------------------------------------------------------------------------
    #### Finalize L2 Dataset
    ds_l2 = select_timesteps_with_minimum_rain_rate(ds_l2, minimum_rain_rate=minimum_rain_rate)

    # Add global attributes
    ds_l2.attrs = attrs

    # Add variables attributes and encodings
    ds_l2 = finalize_product(ds_l2, product="L2E")
    return ds_l2


####--------------------------------------------------------------------------
#### L2 Model Parameters


def _get_default_optimization(psd_model):
    """PSD model defaults."""
    defaults = {
        "ExponentialPSD": "ML",
        "GammaPSD": "ML",
        "LognormalPSD": "ML",
        "NormalizedGammaPSD": "GS",
    }
    optimization = defaults[psd_model]
    return optimization


def check_l2m_input_dataset(ds):
    """Check dataset validity for L2M production."""
    # Retrieve drop_number concentration (if not available) from drop_number
    # --> This allow people to use directly L1 datasets to generate L2M datasets
    if "drop_number_concentration" not in ds:
        if "drop_number" in ds:
            check_l2e_input_dataset(ds)
            sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"])
            sampling_area = get_effective_sampling_area(
                sensor_name=ds.attrs["sensor_name"],
                diameter=ds["diameter_bin_center"] / 1000,
            )  # m2
            # Compute drop number concentration (Nt) [#/m3/mm]
            ds["drop_number_concentration"] = get_drop_number_concentration(
                drop_number=ds["drop_number"],
                velocity=define_velocity_array(ds),  # fall_velocity (and optionally also velocity_bin_center)
                diameter_bin_width=ds["diameter_bin_width"],  # mm
                sample_interval=sample_interval,
                sampling_area=sampling_area,
            )
        else:
            raise ValueError("Please provide DISDRODB L1 or L2E dataset !")

    # Check minimum required variables, coordinates and dimensions are presents
    required_variables = ["drop_number_concentration"]
    required_coords = [
        "diameter_bin_center",
        "diameter_bin_width",
        "diameter_bin_lower",
        "diameter_bin_upper",
        "sample_interval",
    ]
    required_dims = [DIAMETER_DIMENSION]
    _ensure_present(list(ds.data_vars), required=required_variables, kind="variables")
    _ensure_present(list(ds.coords), required=required_coords, kind="coords")
    _ensure_present(list(ds.dims), required=required_dims, kind="dimensions")
    return ds


def generate_l2m(
    ds,
    psd_model,
    # Fitting options
    optimization=None,
    optimization_kwargs=None,
    # PSD discretization
    diameter_min=0,
    diameter_max=10,
    diameter_spacing=0.05,
    # Processing options
    ds_env=None,
    fall_velocity_model="Beard1976",
    # Filtering options
    minimum_ndrops=1,
    minimum_nbins=3,
    minimum_rain_rate=0.01,
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
    psd_model : str
        The PSD model to fit. See ``disdrodb.psd.available_psd_models()``.
    ds_env : xarray.Dataset, optional
        Environmental dataset used for fall velocity and water density estimates.
        If None, a default environment dataset will be loaded.
    diameter_min : float, optional
        Minimum PSD diameter. The default value is 0 mm.
    diameter_max : float, optional
        Maximum PSD diameter. The default value is 8 mm.
    diameter_spacing : float, optional
        PSD diameter spacing. The default value is 0.05 mm.
    optimization : str, optional
        The fitting optimization procedure. Either "GS" (Grid Search), "ML (Maximum Likelihood)
        or "MOM" (Method of Moments).
    optimization_kwargs : dict, optional
        Dictionary with arguments to customize the fitting procedure.
    minimum_nbins: int
        Minimum number of bins with drops required to fit the PSD model.
        The default value is 5.
    gof_metrics : bool, optional
        Whether to add goodness-of-fit metrics to the output dataset. The default is True.

    Returns
    -------
    xarray.Dataset
        DISDRODB L2M dataset.
    """
    ####------------------------------------------------------.
    #### Define default PSD model and optimization
    psd_model = "NormalizedGammaPSD" if psd_model is None else psd_model
    optimization = _get_default_optimization(psd_model) if optimization is None else optimization

    # ----------------------------------------------------------------------------.
    #### Preprocessing
    # Retrieve attributes
    attrs = ds.attrs.copy()

    # Check and prepare dataset
    ds = check_l2m_input_dataset(ds)

    # Retrieve measurement interval
    # - If dataset is opened with decode_timedelta=False, sample_interval is already in seconds !
    sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"])

    # Select timesteps with at least the specified number of drops
    ds = select_timesteps_with_drops(ds, minimum_ndrops=minimum_ndrops)

    # Add bins metrics if missing
    ds = add_bins_metrics(ds)

    # Remove timesteps with not enough bins with drops
    ds = select_timesteps_with_minimum_nbins(ds, minimum_nbins=minimum_nbins)

    # Retrieve ENV dataset or take defaults
    # --> Used for fall velocity and water density estimates
    if ds_env is None:
        ds_env = load_env_dataset(ds)
    water_density = ds_env.get("water_density", 1000)  # kg / m3

    ####------------------------------------------------------.
    #### Retrieve PSD parameters
    ds_psd_params = estimate_model_parameters(
        ds=ds,
        psd_model=psd_model,
        optimization=optimization,
        optimization_kwargs=optimization_kwargs,
    )
    psd_fitting_attrs = ds_psd_params.attrs

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

    # Retrieve drop number concentration
    drop_number_concentration = psd(diameter)

    # Retrieve fall velocity for each new diameter bin
    velocity = get_rain_fall_velocity(diameter=diameter, model=fall_velocity_model, ds_env=ds_env)  # mm

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
        ds_gof = compute_gof_stats(
            obs=ds["drop_number_concentration"],  # empirical N(D)
            pred=psd(ds["diameter_bin_center"]),  # fitted N(D) on empirical diameter bins !
        )
        ds_params.update(ds_gof)

    # Add empirical drop_number_concentration and fall velocity
    # - To reuse output dataset to create another L2M dataset or to compute other GOF metrics
    ds_params["drop_number_concentration"] = ds["drop_number_concentration"]
    ds_params["fall_velocity"] = ds["fall_velocity"]
    ds_params["N"] = ds["N"]
    ds_params.update(ds[BINS_METRICS])

    #### ----------------------------------------------------------------------------.
    #### Finalize dataset
    ds_params = select_timesteps_with_minimum_rain_rate(ds_params, minimum_rain_rate=minimum_rain_rate)

    # Add global attributes
    ds_params.attrs = attrs
    ds_params.attrs.update(psd_fitting_attrs)

    # Add variables attributes and encodings
    ds_params = finalize_product(ds_params, product="L2M")

    # Return dataset
    return ds_params


####-------------------------------------------------------------------------------------------------------------------.
#### L2 Radar Parameters


@check_pytmatrix_availability
def generate_l2_radar(
    ds,
    frequency=None,
    num_points=1024,
    diameter_max=10,
    canting_angle_std=7,
    axis_ratio_model="Thurai2007",
    permittivity_model="Turner2016",
    water_temperature=10,
    elevation_angle=0,
    parallel=True,
):
    """Simulate polarimetric radar variables from empirical drop number concentration or the estimated PSD.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the drop number concentration variable or the PSD parameters.
    frequency : str, float, or list of str and float, optional
        Frequencies in GHz for which to compute the radar parameters.
        Alternatively, also strings can be used to specify common radar frequencies.
        If ``None``, the common radar frequencies will be used.
        See ``disdrodb.scattering.available_radar_bands()``.
    num_points: int or list of integer, optional
        Number of bins into which discretize the PSD.
    diameter_max : float or list of float, optional
        Maximum diameter. The default value is 10 mm.
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
    elevation_angle : float or list of float, optional
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
    # Import here to avoid pytmatrix has mandatory dependency
    # - It is only needed for radar simulation
    from disdrodb.scattering import get_radar_parameters

    # Retrieve radar variables from L2E drop number concentration or from estimated L2M PSD model
    ds_radar = get_radar_parameters(
        ds=ds,
        frequency=frequency,
        num_points=num_points,
        diameter_max=diameter_max,
        canting_angle_std=canting_angle_std,
        axis_ratio_model=axis_ratio_model,
        permittivity_model=permittivity_model,
        water_temperature=water_temperature,
        elevation_angle=elevation_angle,
        parallel=parallel,
    )

    #### ----------------------------------------------------------------------------.
    #### Finalize dataset
    # Add variables attributes and encodings
    ds_radar = finalize_product(ds_radar)

    # Return dataset
    return ds_radar
