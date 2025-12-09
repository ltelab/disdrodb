# -----------------------------------------------------------------------------.
"""DISDRODB hydrometeor classification and QC module."""
import numpy as np
import pandas as pd
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.fall_velocity import get_hail_fall_velocity
from disdrodb.fall_velocity.graupel import retrieve_graupel_heymsfield2014_fall_velocity
from disdrodb.fall_velocity.rain import get_rain_fall_velocity
from disdrodb.l2.empirical_dsd import get_effective_sampling_area, get_rain_rate_from_drop_number
from disdrodb.l2.processing import define_rain_spectrum_mask
from disdrodb.utils.manipulations import filter_diameter_bins, filter_velocity_bins
from disdrodb.utils.time import ensure_sample_interval_in_seconds
from disdrodb.utils.xarray import xr_remap_numeric_array

# Define possible variable available and corresponding snow_temperature_limit
DICT_TEMPERATURES = {
    "air_temperature": 6,  # generic and PWS100
    "air_temperature_min": 6,  # PWS100
    "temperature_ambient": 6,  # LPM
    "temperature_interior": 10,  # LPM
    "sensor_temperature": 10,  # PARSIVEL and SWS250
}
TEMPERATURE_VARIABLES = list(DICT_TEMPERATURES)


def get_temperature(ds):
    """Retrieve temperature variable from L0C product."""
    # Check temperature variable is available, otherwise return None
    if not any(var in ds.data_vars for var in DICT_TEMPERATURES):
        return None, None

    # Define temperature available
    for var, thr in DICT_TEMPERATURES.items():
        if var in ds:
            temperature = ds[var]
            snow_temperature_limit = thr
            break

    # Fill NaNs
    temperature = temperature.ffill("time").bfill("time")
    return temperature, snow_temperature_limit


def define_qc_temperature(temperature, sample_interval, threshold_minutes=360):
    """Define segment-based QC rule for temperature.

    Return a qc array with 1 when temperature is constant for over threshold_minutes.
    Return a qc of 2 if temperature is not available.
    """
    # If all NaN, return flag equal to 2
    if np.all(np.isnan(temperature)):
        return xr.full_like(temperature, 2)

    # Fill NaNs
    temperature = temperature.ffill("time").bfill("time")

    # Round temperature
    temperature = temperature.round(0)

    # Initialize flag
    qc_flag = xr.zeros_like(temperature, dtype=int)

    # Assign 1 when temperature changes, 0 otherwise
    change_points = np.concatenate(([True], np.diff(temperature.values) != 0))

    # Assign segment IDs
    segment_id = np.cumsum(change_points)

    # Compute duration per segment
    df = pd.DataFrame(
        {
            "segment": segment_id,
            "time": temperature["time"].to_numpy(),
        },
    )

    # Count samples per segment
    segment_length = df.groupby("segment").size().rename("count").to_frame()

    # Compute duration based on sample_interval
    segment_length["duration"] = segment_length["count"] * int(sample_interval)

    # Flag segments that are constant for over threshold_minutes
    threshold_seconds = threshold_minutes * 60
    long_segments = segment_length[segment_length["duration"] >= threshold_seconds].index

    # Define QC flag: 1 = no variation, 0 = normal
    mask = np.isin(segment_id, long_segments)
    qc_flag.data = xr.where(mask, 1, 0)
    return qc_flag


def define_qc_margin_fallers(spectrum, fall_velocity_upper, above_velocity_fraction=None, above_velocity_tolerance=2):
    """Define QC mask for margin fallers and splashing."""
    if above_velocity_fraction is not None:
        above_fall_velocity = fall_velocity_upper * (1 + above_velocity_fraction)
    elif above_velocity_tolerance is not None:
        above_fall_velocity = fall_velocity_upper + above_velocity_tolerance
    else:
        above_fall_velocity = np.inf

    # Define mask
    velocity_lower = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["velocity_bin_lower"]
    diameter_upper = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["diameter_bin_upper"]

    mask = np.logical_and(diameter_upper <= 5, velocity_lower >= above_fall_velocity)
    return mask


def define_qc_rain_strong_wind_mask(spectrum):
    """Define QC mask for strong wind artefacts in heavy rainfall.

    Based on Katia Friedrich et al. 2013.
    """
    diameter_lower = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["diameter_bin_lower"]
    velocity_upper = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["velocity_bin_upper"]

    # Define mask
    mask = np.logical_and(
        diameter_lower >= 5,
        velocity_upper < 1,
    )
    return mask


def qc_spikes_isolated_precip(hydrometeor_type, sample_interval):
    """
    Identify isolated precipitation spikes based on hydrometeor classification.

    This quality control (QC) routine flags short, isolated precipitation detections
    (spikes) that are not supported by neighboring precipitating timesteps within
    a defined time window. The test helps remove spurious single-sample precipitation
    detections caused by instrument noise or transient misclassification.

    The algorithm:
        1. Identifies potential precipitation timesteps where `hydrometeor_type>= 1`.
        2. Computes time differences between consecutive potential precipitation samples.
        3. Flags a timestep as a spike when both the previous and next precipitation
           detections are separated by more than the configured time window.
        4. Skips the QC test entirely if the temporal resolution exceeds 2 minutes.

    Parameters
    ----------
    hydrometeor_type: xr.DataArray
        Hydrometeor type classification array with a ``time`` coordinate.
        Precipitation presence is defined where ``hydrometeor_type>= 1``.
    sample_interval : float or int
        Nominal sampling interval of the dataset in **seconds**.
        If ``sample_interval >= 120`` (2 minutes), the QC test is skipped.

    Returns
    -------
    flag_spikes : xr.DataArray of int
        Binary QC flag array (same dimensions as input) with:
            * 0 : no spike detected
            * 1 : isolated precipitation spike

    Notes
    -----
    - The time window is currently fixed to ±60 seconds for typical 1-minute
      sampling intervals but can be adapted to scale with `sample_interval`.
    - Designed to work with irregular time coordinates; relies on actual
      timestamp differences instead of fixed rolling windows.
    - For datasets with coarse sampling (> 2 minutes), the function
      returns a zero-valued flag (QC not applied).
    """
    # Define potential precipitating timesteps
    is_potential_precip = xr.where((hydrometeor_type >= 1), 1, 0)

    # Initialize QC flag
    flag_spikes = xr.zeros_like(is_potential_precip, dtype=int)
    flag_spikes.attrs.update(
        {
            "long_name": "Isolated precipitation spike flag",
            "standard_name": "flag_spikes",
            "units": "1",
            "flag_values": [0, 1],
            "flag_meanings": "no_spike isolated_precipitation_spike",
            "description": (
                "Quality control flag indicating isolated precipitation detections, "
                "without neighboring precipitating timesteps. "
                "If the sampling interval is 2 minutes or coarser, the QC test is skipped."
            ),
        },
    )

    # Skip QC for coarse temporal data (> 2 min)
    if sample_interval >= 120:
        return flag_spikes

    # Define time window based on sample interval
    time_window = 60

    # Extract arrays
    times = pd.to_datetime(is_potential_precip["time"].to_numpy())
    mask_potential_precip = is_potential_precip.to_numpy() == 1

    # If no precipitation, skip and return 0 array
    if not np.any(mask_potential_precip):
        return flag_spikes

    # Get potential precipition indices and timestamps
    precip_idx = np.where(mask_potential_precip)[0]
    precip_times = times[precip_idx].astype("datetime64[s]").astype("int64").astype("float64")

    # Compute Δt to previous and next precip (vectorized)
    dt_prev = np.empty_like(precip_times)
    dt_next = np.empty_like(precip_times)
    dt_prev[0] = np.inf
    dt_prev[1:] = precip_times[1:] - precip_times[:-1]
    dt_next[-1] = np.inf
    dt_next[:-1] = precip_times[1:] - precip_times[:-1]

    # Create same-size arrays aligned with original time dimension
    # - Fill NaN for non-precip indices
    delta_prev = np.full_like(mask_potential_precip, np.inf, dtype=float)
    delta_next = np.full_like(mask_potential_precip, np.inf, dtype=float)
    delta_prev[precip_idx] = dt_prev
    delta_next[precip_idx] = dt_next

    # Identify isolated spikes
    isolated = (mask_potential_precip) & (delta_prev > time_window) & (delta_next > time_window)
    flag_spikes.data[isolated] = 1
    return flag_spikes


def define_hail_mask(spectrum, ds_env, minimum_diameter=5):
    """Define hail mask."""
    # Define velocity limits
    fall_velocity_lower = get_hail_fall_velocity(
        spectrum["diameter_bin_lower"],
        model="Heymsfield2018",
        ds_env=ds_env,
        minimum_diameter=minimum_diameter - 1,
    )
    fall_velocity_upper = get_hail_fall_velocity(
        spectrum["diameter_bin_upper"],
        model="Laurie1960",
        ds_env=ds_env,
        minimum_diameter=minimum_diameter - 1,
    )

    # Define spectrum mask
    diameter_lower = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["diameter_bin_lower"]
    velocity_lower = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["velocity_bin_lower"]
    velocity_upper = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["velocity_bin_upper"]

    mask_velocity = np.logical_and(
        velocity_lower >= fall_velocity_lower,
        velocity_upper <= fall_velocity_upper,
    )

    mask_diameter = diameter_lower >= minimum_diameter
    mask = np.logical_and(mask_diameter, mask_velocity)
    return mask


def define_graupel_mask(
    spectrum,
    ds_env,
    minimum_diameter=0.5,
    maximum_diameter=5,
    minimum_density=50,
    maximum_density=600,
):
    """Define graupel mask."""
    # Define velocity limits
    fall_velocity_lower = retrieve_graupel_heymsfield2014_fall_velocity(
        diameter=spectrum["diameter_bin_lower"],
        graupel_density=minimum_density,
        ds_env=ds_env,
    )
    fall_velocity_upper = retrieve_graupel_heymsfield2014_fall_velocity(
        diameter=spectrum["diameter_bin_upper"],
        graupel_density=maximum_density,
        ds_env=ds_env,
    )
    # fall_velocity_upper = get_graupel_fall_velocity(
    #     diameter=spectrum["diameter_bin_upper"],
    #     model="Locatelli1974Lump",
    #     ds_env=ds_env,
    #     minimum_diameter=minimum_diameter-1,
    #     maximum_diameter=maximum_diameter+1,
    # )
    # Define spectrum mask
    diameter_lower = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["diameter_bin_lower"]
    diameter_upper = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["diameter_bin_upper"]

    velocity_lower = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["velocity_bin_lower"]
    velocity_upper = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["velocity_bin_upper"]
    mask_velocity = np.logical_and(
        velocity_lower >= fall_velocity_lower,
        velocity_upper <= fall_velocity_upper,
    )
    mask_diameter = np.logical_and(
        diameter_lower >= minimum_diameter,
        diameter_upper <= maximum_diameter,
    )
    mask = np.logical_and(mask_diameter, mask_velocity)
    return mask


def classify_raw_spectrum(
    ds,
    ds_env,
    sample_interval,
    sensor_name,
    temperature=None,
    rain_temperature_lower_limit=-5,
    snow_temperature_upper_limit=5,
):
    """Run precipitation and hydrometeor type classification."""
    # ------------------------------------------------------------------
    # Filter LPM to avoid being impacted by noise in first bins
    if sensor_name == "LPM":
        # Remove first two diameter bins (very noisy)
        ds = filter_diameter_bins(ds=ds, minimum_diameter=0.375)
        # Remove first velocity bin
        ds = filter_velocity_bins(ds=ds, minimum_velocity=0.2)
    if sensor_name == "PWS100":
        # Remove first two bin
        ds = filter_diameter_bins(ds=ds, minimum_diameter=0.2)
        # Remove first two bin
        ds = filter_velocity_bins(ds=ds, minimum_velocity=0.2)

    # ------------------------------------------------------------------
    # Retrieve raw spectrum
    raw_spectrum = ds["raw_drop_number"]

    #### Define masks
    spectrum_template = raw_spectrum.isel(time=0, missing_dims="ignore")
    diameter_lower = raw_spectrum["diameter_bin_lower"].broadcast_like(spectrum_template)  # [mm]
    diameter_upper = raw_spectrum["diameter_bin_upper"].broadcast_like(spectrum_template)

    # Define spectrum areas
    B1 = (diameter_lower >= 0.0) & (diameter_upper <= 0.5)
    B2 = (diameter_upper > 0.5) & (diameter_upper <= 5.0)
    B3 = (diameter_upper > 5.0) & (diameter_upper <= 8.0)
    B4 = diameter_upper > 8.0

    # Define liquid masks
    # - Compute raindrop fall velocity for lower and upper diameter limits
    raindrop_fall_velocity_upper = get_rain_fall_velocity(
        diameter=ds["diameter_bin_upper"],
        model="Beard1976",
        ds_env=ds_env,
    )
    raindrop_fall_velocity_lower = get_rain_fall_velocity(
        diameter=ds["diameter_bin_lower"],
        model="Beard1976",
        ds_env=ds_env,
    )
    liquid_mask = define_rain_spectrum_mask(
        drop_number=raw_spectrum,
        fall_velocity_lower=raindrop_fall_velocity_lower,
        fall_velocity_upper=raindrop_fall_velocity_upper,
        above_velocity_fraction=None,
        above_velocity_tolerance=2,
        below_velocity_fraction=None,
        below_velocity_tolerance=3,
        maintain_drops_smaller_than=1,  # 1,   # 2
        maintain_drops_slower_than=2.5,  # 2.5, # 3
        maintain_smallest_drops=False,
    )

    drizzle_mask = liquid_mask & B1
    drizzle_rain_mask = liquid_mask & B2
    rain_mask = liquid_mask & B3  # potentially mixed with small hail

    # Define graupel masks
    graupel_mask = define_graupel_mask(
        raw_spectrum,
        ds_env=ds_env,
        minimum_diameter=0.9,
        maximum_diameter=5.5,
        minimum_density=50,
        maximum_density=900,
    )
    graupel_mask = np.logical_and(graupel_mask, ~liquid_mask)
    graupel_hd_mask = define_graupel_mask(
        raw_spectrum,
        ds_env=ds_env,
        minimum_diameter=0.9,
        maximum_diameter=5.5,
        minimum_density=400,
        maximum_density=900,
    )
    graupel_hd_mask = np.logical_and(graupel_hd_mask, graupel_mask)
    graupel_ld_mask = np.logical_and(graupel_mask, ~graupel_hd_mask)

    # graupel_mask.plot.pcolormesh(x="diameter_bin_center")
    # liquid_mask.plot.pcolormesh(x="diameter_bin_center")
    # graupel_hd_mask.plot.pcolormesh(x="diameter_bin_center")
    # graupel_ld_mask.plot.pcolormesh(x="diameter_bin_center")

    # Define hail mask
    hail_mask = define_hail_mask(raw_spectrum, ds_env=ds_env, minimum_diameter=5)
    hail_mask = np.logical_and(hail_mask, ~graupel_mask)

    small_hail_mask = hail_mask & B3  # [5,8]
    large_hail_mask = hail_mask & B4  # > 8

    # Define snow masks
    velocity_upper = xr.ones_like(raw_spectrum.isel(time=0, missing_dims="ignore")) * raw_spectrum["velocity_bin_upper"]
    snow_mask_full = velocity_upper <= 6.5

    # - Without rain and hail
    snow_mask = np.logical_and(snow_mask_full, ~liquid_mask)
    snow_mask = np.logical_and(snow_mask, ~hail_mask)
    snow_mask = np.logical_and(snow_mask, diameter_lower >= 1)

    # - Without rain, hail and graupel
    # snow_small_mask = snow_mask & (diameter_upper <= 5.0)
    snow_large_mask = snow_mask & (diameter_upper > 5.0)

    # Define snow grain mask
    snow_grains_mask = (velocity_upper <= 2.5) & (diameter_upper <= 2.2)  # ice crystals & prisms (0.1 < D < 1 or 2 mm)

    # ---------------------------------------------------------------------
    # Check mask cover all space without leaving empty bins
    # FUTURE: CHECK IF THERE ARE CASES WHERE EMPTY BINS STILL OCCURS

    # from functools import reduce
    # sum_mask = reduce(np.logical_or, [hail_mask, liquid_mask, graupel_mask])
    # sum_mask.plot.pcolormesh(x="diameter_bin_center")

    # ---------------------------------------------------------------------
    # Estimate rain rate using particles with D <=5  (D > 5 can be contaminated by noise or hail)
    # - Extract sample interval
    sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"])  # s
    # - Extract diameter in m
    diameter = ds["diameter_bin_center"] / 1000  # m
    # - Compute sampling area [m2]
    sampling_area = get_effective_sampling_area(sensor_name=sensor_name, diameter=diameter)  # m2
    # - Compute dummy rainfall rate (on D from 0 to 5) to avoid 'hail' contamination
    rainfall_rate_mask = drizzle_mask + drizzle_rain_mask
    rainfall_rate = get_rain_rate_from_drop_number(
        drop_number=raw_spectrum.where(rainfall_rate_mask, 0),  # if any NaN --> return NaN
        sampling_area=sampling_area,
        diameter=diameter,
        sample_interval=sample_interval,
    )
    # ---------------------------------------------------------------------
    # Estimate gross snowfall rate
    # FUTURE:
    # - Compute over snow mask area with and without rainy area
    # - Use Lempio lump parametrization
    # - Required to define weather codes

    # ---------------------------------------------------------------------
    # Define wind artefacts mask (Friedrich et al., 2013)
    strong_wind_mask = define_qc_rain_strong_wind_mask(spectrum_template)

    # Define margin fallers mask
    margin_fallers_mask = define_qc_margin_fallers(
        spectrum_template,
        fall_velocity_upper=raindrop_fall_velocity_upper,
        # above_velocity_fraction=0.6,
        above_velocity_tolerance=2,
    )

    # Define splash mask
    splash_mask = (diameter_lower >= 0.0) & (diameter_upper <= 6) & (velocity_upper <= 0.6)

    # ---------------------------------------------------------------------
    # Define liquid, snow, and graupel mask (robust to splash)
    liquid_mask_without_splash = liquid_mask & ~splash_mask
    snow_mask_without_splash = snow_mask & ~splash_mask
    # graupel_mask_without_splash = graupel_mask & ~splash_mask
    graupel_ld_mask_without_splash = graupel_ld_mask & ~splash_mask

    # ---------------------------------------------------------------------
    #### Compute statistics
    dims = [DIAMETER_DIMENSION, VELOCITY_DIMENSION]
    n_particles = raw_spectrum.sum(dim=dims)
    # n_particles_1 = raw_spectrum.where(B1).sum(dim=dims)
    # n_particles_2 = raw_spectrum.where(B2).sum(dim=dims)
    # n_particles_3 = raw_spectrum.where(B3).sum(dim=dims)
    # n_particles_4 = raw_spectrum.where(B4).sum(dim=dims)

    ## ----
    # Rain
    n_drizzle = raw_spectrum.where(drizzle_mask).sum(dim=dims)
    n_drizzle_rain = raw_spectrum.where(drizzle_rain_mask).sum(dim=dims)
    n_rain = raw_spectrum.where(rain_mask).sum(dim=dims)
    n_liquid = n_drizzle + n_drizzle_rain + n_rain
    n_liquid_robust = raw_spectrum.where(liquid_mask_without_splash).sum(dim=dims)

    ## ----
    # Hail
    # n_hail = raw_spectrum.where(hail_mask).sum(dim=dims)
    n_small_hail = raw_spectrum.where(small_hail_mask).sum(dim=dims)
    n_large_hail = raw_spectrum.where(large_hail_mask).sum(dim=dims)

    ## ----
    # Graupel
    n_graupel = raw_spectrum.where(graupel_mask).sum(dim=dims)
    # n_graupel_robust = raw_spectrum.where(graupel_mask_without_splash).sum(dim=dims)
    n_graupel_ld = raw_spectrum.where(graupel_ld_mask_without_splash).sum(dim=dims)
    n_graupel_hd = raw_spectrum.where(graupel_hd_mask).sum(dim=dims)

    ## ----
    # Snow
    n_snow = raw_spectrum.where(snow_mask).sum(dim=dims)
    n_snow_robust = raw_spectrum.where(snow_mask_without_splash).sum(dim=dims)

    # n_snow_small = raw_spectrum.where(snow_small_mask).sum(dim=dims)
    n_snow_large = raw_spectrum.where(snow_large_mask).sum(dim=dims)
    n_snow_grains = raw_spectrum.where(snow_grains_mask).sum(dim=dims)

    ## ----
    # Auxiliary
    n_wind_artefacts = raw_spectrum.where(strong_wind_mask).sum(dim=dims)
    n_margin_fallers = raw_spectrum.where(margin_fallers_mask).sum(dim=dims)
    n_splashing = raw_spectrum.where(splash_mask).sum(dim=dims)

    ## ----
    # Bins statistics
    # n_bins = (raw_spectrum.where((~splash_mask) > 0)).sum(dim=dims)
    n_liquid_bins = (raw_spectrum.where(liquid_mask_without_splash) > 0).sum(dim=dims)
    n_snow_bins = (raw_spectrum.where(snow_mask_without_splash) > 0).sum(dim=dims)  # without rainy area
    # n_snow_large_bins = (raw_spectrum.where(snow_large_mask) > 0).sum(dim=dims) # only > 5 mm
    # n_graupel_bins = (raw_spectrum.where(graupel_mask_without_splash) > 0).sum(dim=dims)

    # Bins fractions
    fraction_rain_bins = xr.where(n_particles == 0, 0, n_liquid_bins / (n_liquid_bins + n_snow_bins))
    # fraction_snow_bins = xr.where(n_particles == 0, 0, n_snow_bins / (n_liquid_bins + n_snow_bins))

    ## ----
    # Particles fractions
    # fraction_drizzle_rel = xr.where(n_particles_1 == 0, 0, n_drizzle / n_particles_1)
    fraction_drizzle_tot = xr.where(n_particles == 0, 0, n_drizzle / n_particles)

    # fraction_drizzle_rain_rel = xr.where(n_particles_2 == 0, 0, n_drizzle_rain / n_particles_2)
    fraction_drizzle_rain_tot = xr.where(n_particles == 0, 0, (n_drizzle + n_drizzle_rain) / n_particles)

    # fraction_rain_rel = xr.where(n_particles_3 == 0, 0, n_rain / n_particles_3)
    fraction_rain_tot = xr.where(n_particles == 0, 0, n_liquid / n_particles)
    # fraction_liquid = fraction_rain_tot

    # fraction_graupel_only_rel = xr.where(n_particles_2 == 0, 0, n_graupel / n_particles_2)
    fraction_graupel_only_tot = xr.where(n_particles == 0, 0, n_graupel / n_particles)

    # fraction_hail = xr.where(n_particles_4 == 0, 0, n_hail / n_particles_4)

    # fraction_snow_large_rel = xr.where((n_particles_3 + n_particles_4) == 0, 0,
    #                                     n_snow_large / (n_particles_3+n_particles_4))
    # fraction_snow_large_tot = xr.where(n_particles == 0, 0, n_snow_large / n_particles)
    fraction_snow_tot = xr.where(n_particles == 0, 0, n_snow / n_particles)
    fraction_snow_grains_tot = xr.where(n_particles == 0, 0, n_snow_grains / n_particles)

    fraction_splash = xr.where(n_particles == 0, 0, n_splashing / n_particles)

    # fraction_rain_graupel_tot = xr.where(n_particles == 0, 0,  (n_liquid + n_graupel) / n_particles)

    # graupel_liquid_ratio = xr.where(n_particles == 0, 0,  n_graupel_robust/n_liquid_robust)
    solid_liquid_ratio = xr.where(n_particles == 0, 0, n_snow_robust / n_liquid_robust)

    # -----------------------------------------------------------------------------------------------.
    #### Classification logic
    # Class        |Conditions       | WMO 4680
    # -------------------------------------
    # Drizzle      |D < 0.5 mm       |
    # Rain         |D > 0.5, D < 10  |
    # Snow         |D > 1, V < 6     | 71-73
    # Snow grains  |D < 1            | 77
    # Ice Crystals |0 < D < 2        |
    # Graupel      |D >1 , D < 5     | 74-76

    # Snow grains are within the drizzle class (<0.5 mm)!
    # --> Temperature required to classify them

    # Graupel class
    # - Ice pellets / Sleets (frozen raindrops, T<0) (1 <= D <= 5 mm)
    # - Graupel (GS) (Snow pellet coated with ice, T > 0) (2 <= D <= 5 mm)
    #                           # WMO 4680
    # --------------------------------------------------------------
    # Initialize label
    label = xr.ones_like(ds["time"], dtype=float) * -1  # [mm]

    # No precipitation
    label = xr.where(n_particles == 0, 0, label)

    # Graupel only
    cond = (fraction_graupel_only_tot > 0.7) & (n_snow_large < 1)
    label = xr.where(cond & (label == -1), 8, label)

    # Liquid only
    # - Drizzle (D < 0.5 mm)
    cond = (fraction_drizzle_tot > 0.1) & (n_graupel < 1) & (n_snow_large < 1) & (n_drizzle_rain < 1)
    label = xr.where(cond & (label == -1), 1, label)

    # - Drizzle + Rain (0.5-5 mm)
    cond = (fraction_drizzle_rain_tot > 0.1) & (n_graupel < 1) & (n_snow_large < 1) & (n_rain < 1)
    label = xr.where(cond & (label == -1), 2, label)

    # - Rain (D > 5 mm)
    cond = (fraction_rain_tot > 0.1) & (n_graupel < 1) & (n_snow_large < 1)
    label = xr.where(cond & (label == -1), 3, label)

    # Snow only
    cond = fraction_snow_tot > 0.6  # TODO: extend to use snow_mask_full
    label = xr.where(cond & (label == -1), 5, label)

    # ---------------------------------
    # Rain (R > 3 mm/hr) with some graupel
    cond = (fraction_rain_bins >= 0.75) & (rainfall_rate > 3)
    label = xr.where(cond & (label == -1), 31, label)  # mixed

    # ---------------------------------
    #  (label == -1).sum()
    # (cond & (label == -1)).sum()

    # ---------------------------------
    # Mixed
    # --> FUTURE: Better clarified the meaning
    # --> FUTURE: R computed with particles only above 3 m/s would help disentagle snow from mixed !
    # --> When R > 1 mm/hr and no splash - solid_liquid_ratio > XXX
    n_snow_bins_thr = 6
    fraction_splash_thr = 0.1
    solid_liquid_ratio_thr = 0.05
    cond = (
        (solid_liquid_ratio >= solid_liquid_ratio_thr)
        & (rainfall_rate > 1)
        & (n_snow_bins > n_snow_bins_thr)
        & (fraction_splash < fraction_splash_thr)
    )
    label = xr.where(cond & (label == -1), 4, label)  # mixed

    cond = (
        (solid_liquid_ratio >= solid_liquid_ratio_thr)
        & (rainfall_rate > 1)
        & (n_snow_bins <= n_snow_bins_thr)
        & (fraction_splash < fraction_splash_thr)
    )
    label = xr.where(cond & (label == -1), 21, label)  # Set as rain !

    # - When R > 1 mm/hr and no splash - solid_liquid_ratio < XXX
    cond = (solid_liquid_ratio < solid_liquid_ratio_thr) & (rainfall_rate > 1) & (fraction_splash < fraction_splash_thr)
    label = xr.where(cond & (label == -1), 21, label)  # Set as rain !

    # ---------------------------------
    # Non-hydrometeors class
    cond = (fraction_splash >= 0.5) & (rainfall_rate < 1.5)
    label = xr.where(cond & (label == -1), -2, label)

    cond = (fraction_splash >= 0.4) & (fraction_splash <= 0.5) & (rainfall_rate <= 0.2)
    label = xr.where(cond & (label == -1), -2, label)

    # ---------------------------------
    # - When R > 1mm/hr, with splash
    cond = (rainfall_rate > 1) & (fraction_splash >= 0.1) & (solid_liquid_ratio >= 0.2)
    label = xr.where(cond & (label == -1), 41, label)  # mixed

    cond = (rainfall_rate > 1) & (fraction_splash >= 0.1) & (solid_liquid_ratio < 0.2)
    label = xr.where(cond & (label == -1), 23, label)  # rainfall

    # ---------------------------------
    # - Noisy Rain (solid_liquid_ratio < 0.03)
    cond = (solid_liquid_ratio <= 0.05) & (n_snow_robust <= 2)
    label = xr.where(cond & (label == -1), 22, label)  # Set noisy rain

    # ---------------------------------
    # Ice Crystals
    cond = fraction_snow_grains_tot >= 0.95
    label = xr.where(cond & (label == -1), 6, label)

    # Remaining snow
    label = xr.where(label == -1, 51, label)

    # ------------------------------------------------------------------------.
    # Improve classification using temperature information if available
    if temperature is not None:
        temperature = temperature.compute()
        qc_temperature = define_qc_temperature(temperature, sample_interval=sample_interval, threshold_minutes=1440)

        is_surely_rain = (temperature >= snow_temperature_upper_limit) & (qc_temperature == 0)
        is_surely_snow = (temperature <= rain_temperature_lower_limit) & (qc_temperature == 0)
        is_mixed = label.isin([4, 41])
        is_snow = label.isin([5, 51])
        is_drizzle = label.isin([1])
        is_snow_grain = label.isin([6])
        is_rain = label.isin([2, 21, 22, 23, 3])
        is_graupel = label == 8

        # Improve mixed classification (4, 41)
        # - If T > snow_temperature_upper_limit --> rain
        # - If T < -5 rain_temperature_lower_limit --> snow
        label = xr.where(is_surely_rain & is_mixed, 24, label)
        label = xr.where(is_surely_snow & is_mixed, 52, label)

        # Improve snow classification
        # - If T > snow_temperature_upper_limit --> No hydrometeors
        label = xr.where(is_surely_rain & is_snow, -21, label)

        # Improve drizzle classification
        label = xr.where(is_surely_snow & is_drizzle, 61, label)

        # Improve snow grains classification
        # --> If T > snow_temperature_upper_limit --> No hydrometeors
        label = xr.where(is_surely_rain & is_snow_grain, -21, label)

        # Improve rain classification
        # If T < rain_temperature_lower_limit --> No hydrometeors
        label = xr.where(is_surely_snow & is_rain, -21, label)

        # Improve graupel classification
        # If T < rain_temperature_lower_limit --> Ice pellets / Sleets
        label = xr.where(is_surely_snow & is_graupel, 7, label)

    # ------------------------------------------------------------------------.
    # Define hydrometeor_typevariable
    # -2 No hydrometeor
    # -1 Undefined
    # 0 No precipitation
    # 1 Drizzle
    # 2 Drizzle+Rain
    # 3 Rain
    # 4 Mixed (when no only graupel, and rain)
    # 5 Snow  (when not only graupel, and no rain)
    # 6 Snow grains / ice crystals / needles  (only if temperature is available <-- drizzle)
    # 7 Ice pellets / Sleets (only if temperature is available)
    # 8 Graupel --> flag_graupel
    # 9 Hail --> flag_hail

    hydrometeor_type = label.copy()
    # No hydrometeor
    hydrometeor_type = xr.where(label.isin([-2, -21]), -2, hydrometeor_type)
    # Drizzle
    hydrometeor_type = xr.where(hydrometeor_type.isin([1]), 1, hydrometeor_type)
    # Drizzle+Rain
    hydrometeor_type = xr.where(hydrometeor_type.isin([2, 21, 22, 23, 24]), 2, hydrometeor_type)
    # Rain
    hydrometeor_type = xr.where(hydrometeor_type.isin([3, 31]), 3, hydrometeor_type)
    # Mixed
    hydrometeor_type = xr.where(hydrometeor_type.isin([4]), 4, hydrometeor_type)
    # Snow
    hydrometeor_type = xr.where(hydrometeor_type.isin([5, 51, 52]), 5, hydrometeor_type)
    # Snow grains
    hydrometeor_type = xr.where(hydrometeor_type.isin([6]), 6, hydrometeor_type)
    # Ice Pellets
    hydrometeor_type = xr.where(hydrometeor_type.isin([7]), 7, hydrometeor_type)
    # Graupel
    hydrometeor_type = xr.where(hydrometeor_type.isin([8]), 8, hydrometeor_type)
    # Add CF-attributes
    hydrometeor_type.attrs.update(
        {
            "long_name": "hydrometeor type classification",
            "standard_name": "hydrometeor_classification",
            "units": "1",
            "flag_values": [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "flag_meanings": (
                "no_hydrometeor undefined no_precipitation "
                "drizzle drizzle_and_rain rain mixed "
                "snow snow_grains ice_pellets graupel hail"
            ),
        },
    )

    # ------------------------------------------------------------------------.
    #### Define precipitation type variable
    precipitation_type = xr.ones_like(ds["time"], dtype=float) * -1
    precipitation_type = xr.where(hydrometeor_type.isin([0]), 0, precipitation_type)
    precipitation_type = xr.where(hydrometeor_type.isin([1, 2, 3]), 0, precipitation_type)
    precipitation_type = xr.where(hydrometeor_type.isin([5, 6, 7, 8]), 1, precipitation_type)
    precipitation_type = xr.where(hydrometeor_type.isin([4]), 2, precipitation_type)
    precipitation_type.attrs.update(
        {
            "long_name": "precipitation phase classification",
            "standard_name": "precipitation_phase",
            "units": "1",
            "flag_values": [-2, -1, 0, 1, 2],
            "flag_meanings": "undefined no_precipitation rainfall snowfall mixed_phase",
        },
    )

    # ------------------------------------------------------------------------.
    #### Define flag graupel
    flag_graupel = xr.ones_like(ds["time"], dtype=float) * 0
    flag_graupel = xr.where(
        (((precipitation_type == 0) & (n_graupel_ld > 2)) | ((hydrometeor_type == 8) & (n_graupel_ld > 0))),
        1,
        flag_graupel,
    )
    flag_graupel = xr.where(
        (((precipitation_type == 0) & (n_graupel_hd > 2)) | ((hydrometeor_type == 8) & (n_graupel_hd > 0))),
        2,
        flag_graupel,
    )
    flag_graupel.attrs.update(
        {
            "long_name": "graupel occurrence flag",
            "standard_name": "graupel_flag",
            "units": "1",
            "flag_values": [0, 1, 2],
            "flag_meanings": "no_graupel low_density_graupel high_density_graupel",
            "description": (
                "Flag indicating the presence of graupel. "
                "The flag is set when hydrometeor classification identifies graupel (class=8) or "
                "rainfall with graupel particles. "
                "Low-density graupel (value = 1) corresponds to density < 400 kg/m3 "
                "while high-density graupel corresponds to density > 400 kg/m3."
            ),
        },
    )

    # ------------------------------------------------------------------------.
    #### Define flag hail
    # FUTURE:
    # - Small hail: check if attached to rain body or not
    # - Check how much is detached
    flag_hail = xr.ones_like(ds["time"], dtype=float) * 0
    flag_hail = xr.where(((precipitation_type == 0) & (n_small_hail >= 1) & (rainfall_rate > 1)), 1, flag_hail)
    flag_hail = xr.where(((precipitation_type == 0) & (n_large_hail >= 1) & (rainfall_rate > 1)), 2, flag_hail)
    flag_hail.attrs.update(
        {
            "long_name": "hail occurrence and size flag",
            "standard_name": "hail_flag",
            "units": "1",
            "flag_values": [0, 1, 2],
            "flag_meanings": "no_hail small_hail large_hail",
            "description": (
                "Flag indicating the presence and estimated size of hail. "
                "Set to 1 for small hail when precipitation type indicates rain. "
                "Set to 2 for large hail (>8 mm) under similar conditions."
            ),
        },
    )
    # ------------------------------------------------------------------------.
    #### Define WMO codes
    # FUTURE: Use hydrometeor_typeand flag_hail values [1,2]
    # Require snowfall rate estimate

    # ------------------------------------------------------------------------
    #### Define QC splashing, strong_wind, margin_fallers, spikes
    # FUTURE: flag_spikes can be used for non hydrometeor classification,
    # --> But caution because observing the below show true rainfall signature
    # --> raw_spectrum.isel(time=(flag_spikes == 0) & (precipitation_type == 0)).disdrodb.plot_spectrum()

    flag_splashing = xr.where((precipitation_type == 0) & (fraction_splash >= 0.1), 1, 0)
    flag_wind_artefacts = xr.where((precipitation_type == 0) & (n_wind_artefacts >= 1), 1, 0)
    flag_noise = xr.where((hydrometeor_type == -2), 1, 0)
    flag_spikes = qc_spikes_isolated_precip(hydrometeor_type, sample_interval=sample_interval)

    # ------------------------------------------------------------------------.
    #### Define n_particles_<hydro_class>
    n_graupel_ld_final = xr.where(flag_graupel == 1, n_graupel_ld, 0)
    n_graupel_hd_final = xr.where(flag_graupel == 2, n_graupel_hd, 0)

    n_small_hail_final = xr.where(flag_hail == 1, n_small_hail, 0)
    n_large_hail_final = xr.where(flag_hail == 2, n_large_hail, 0)
    n_margin_fallers_final = xr.where(precipitation_type == 0, n_margin_fallers, 0)
    n_splashing_final = xr.where(flag_splashing == 1, n_splashing, 0)

    # ------------------------------------------------------------------------.
    # Create HC and QC dataset
    ds_class = ds[["time"]]

    # ds_class["label"] = label
    ds_class["precipitation_type"] = precipitation_type
    ds_class["hydrometeor_type"] = hydrometeor_type

    ds_class["n_particles"] = n_particles

    ds_class["n_low_density_graupel"] = n_graupel_ld_final
    ds_class["n_high_density_graupel"] = n_graupel_hd_final

    ds_class["n_small_hail"] = n_small_hail_final
    ds_class["n_large_hail"] = n_large_hail_final
    ds_class["n_margin_fallers"] = n_margin_fallers_final
    ds_class["n_splashing"] = n_splashing_final

    # fraction_splash
    # fraction_margin_fallers

    # ds_class["mask_graupel"] = graupel_mask_without_splash
    # ds_class["mask_splashing"] = mask_splashing

    ds_class["flag_hail"] = flag_hail
    ds_class["flag_graupel"] = flag_graupel

    ds_class["flag_noise"] = flag_noise
    ds_class["flag_spikes"] = flag_spikes
    ds_class["flag_splashing"] = flag_splashing
    ds_class["flag_wind_artefacts"] = flag_wind_artefacts
    return ds_class


####--------------------------------------------------------------
#### Other utilities
def map_precip_flag_to_precipitation_type(precip_flag):
    """Map OCEANRAIN precip_flag to DISDRODB precipitation_type."""
    mapping_dict = {
        0: 0,  # rain → rainfall
        1: 1,  # snow → snowfall
        2: 2,  # mixed_phase → mixed
        3: -1,  # true_zero_value → no_precipitation
        4: -2,  # inoperative → undefined
        5: -2,  # harbor_time_no_data → undefined
    }
    precipitation_type = xr_remap_numeric_array(precip_flag, mapping_dict)
    precipitation_type.attrs.update(
        {
            "long_name": "precipitation phase classification",
            "standard_name": "precipitation_phase",
            "units": "1",
            "flag_values": [-2, -1, 0, 1, 2],
            "flag_meanings": "undefined no_precipitation rainfall snowfall mixed_phase",
        },
    )
    return precipitation_type
