#!/usr/bin/env python3
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
"""Reader for ARM OTT PARSIVEL2 sensor."""
from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0b_nc_processing import open_raw_netcdf_file, replace_custom_nan_flags, standardize_raw_dataset


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""
    ##------------------------------------------------------------------------.
    #### Open the netCDF
    ds = open_raw_netcdf_file(filepath=filepath, logger=logger)

    ##------------------------------------------------------------------------.
    #### Adapt the dataframe to adhere to DISDRODB L0 standards
    # Define dictionary mapping dataset variables to select and rename
    dict_names = {
        ### Dimensions
        "particle_size": "diameter_bin_center",
        "raw_fall_velocity": "velocity_bin_center",
        ### Variables
        # 'base_time': 'base_time',
        # 'time_offset'
        # 'class_size_width',
        # 'fall_velocity_calculated',
        # "lat": "latitude",
        # "lon": "longitude",
        # "alt": "altitude",
        "weather_code": "weather_code_synop_4680",
        "equivalent_radar_reflectivity_ott": "reflectivity_32bit",
        "mor_visibility": "mor_visibility",
        "sensor_temperature": "sensor_temperature",
        "laserband_amplitude": "laser_amplitude",
        "heating_current": "sensor_heating_current",
        "sensor_voltage": "sensor_battery_voltage",
        "number_detected_particles": "number_particles",
        "snow_depth_intensity": "snowfall_rate",  # Only available > 2019
        "number_density_drops": "raw_drop_concentration",
        "raw_spectrum": "raw_drop_number",
        ### ARM retrievals
        # 'moment1',
        # 'moment2',
        # 'moment3',
        # 'moment4',
        # 'moment5',
        # 'moment6',
        # 'diameter_min':
        # 'diameter_max':
        # 'median_volume_diameter'
        # 'intercept_parameter',
        # 'slope_parameter'
        # 'liquid_water_content',
        # 'liquid_water_distribution_mean'
        # 'precip_rate':
        # 'equivalent_radar_reflectivity',
        # Possible QC variables
        # 'qc_time',
        # 'qc_precip_rate':
        # 'qc_number_detected_particles':
        # 'qc_mor_visibility':
        # 'qc_heating_current':
        # 'qc_snow_depth_intensity':
        # 'qc_laserband_amplitude':
        # 'qc_weather_code':
        # 'qc_equivalent_radar_reflectivity_ott':
        # 'qc_sensor_voltage':
    }

    # Rename dataset variables and columns and infill missing variables
    ds = standardize_raw_dataset(ds=ds, dict_names=dict_names, sensor_name="PARSIVEL2")

    # Replace nan flags with np.nan
    # - ARM use the -9999 flags
    nan_flags_variables = [
        "sensor_temperature",
        "laser_amplitude",
        "mor_visibility",
        "number_particles",
        "weather_code_synop_4680",
        "raw_drop_number",
    ]
    dict_nan_flags = {var: [-9999] for var in nan_flags_variables}
    ds = replace_custom_nan_flags(ds, dict_nan_flags=dict_nan_flags, logger=logger)

    # Return the dataset adhering to DISDRODB L0B standards
    return ds
