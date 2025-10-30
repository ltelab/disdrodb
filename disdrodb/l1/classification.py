# -----------------------------------------------------------------------------.
"""DISDRODB hydrometeor classification module."""

import numpy as np
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.fall_velocity import get_hail_fall_velocity 
from disdrodb.fall_velocity.hail import retrieve_graupel_heymsfield2014_fall_velocity


def define_qc_margin_fallers(spectrum, fall_velocity, above_velocity_fraction=0.6):
    """Define QC mask for margin fallers and splashing."""
    above_fall_velocity = fall_velocity * (1 + above_velocity_fraction)
        
    diameter_upper = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["diameter_bin_upper"]
    velocity_lower = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["velocity_bin_lower"]

    mask = np.logical_and(diameter_upper <= 10, velocity_lower > above_fall_velocity) 
    return mask 
     
 
def define_qc_rain_strong_wind_mask(spectrum):
    """Define QC mask for strong wind artefacts in heavy rainfall.
    
    Based on Katia Friedrich et al. 2013.
    """
    diameter_lower = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["diameter_bin_lower"]
    velocity_upper = xr.ones_like(spectrum.isel(time=0, missing_dims="ignore")) * spectrum["velocity_bin_upper"]
    
    # Define mask
    mask = np.logical_and(
        diameter_lower > 5,
        velocity_upper < 1,
    )
    return mask



def define_hail_mask(spectrum, ds_env, minimum_diameter=5):
    """Define hail mask."""
    # Define velocity limits
    fall_velocity_lower = get_hail_fall_velocity(spectrum["diameter_bin_lower"], 
                                                 model="Heymsfield2018", 
                                                 ds_env=ds_env, 
                                                 minimum_diameter=minimum_diameter-1)
    fall_velocity_upper = get_hail_fall_velocity(spectrum["diameter_bin_upper"], 
                                                 model="Laurie1960", 
                                                 ds_env=ds_env, 
                                                 minimum_diameter=minimum_diameter-1)

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


def define_graupel_mask(spectrum, ds_env, minimum_diameter=0.5, maximum_diameter=5): 
    """Define graupel mask."""
    # Define velocity limits
    fall_velocity_lower = retrieve_graupel_heymsfield2014_fall_velocity(
        diameter=spectrum["diameter_bin_lower"],
        graupel_density=50,
        ds_env=ds_env, 
    )
    fall_velocity_upper = retrieve_graupel_heymsfield2014_fall_velocity(
        diameter=spectrum["diameter_bin_upper"],
        graupel_density=600,
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

