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
"""Reader for DELFT OTT PARSIVEL2 sensor in netCDF format."""

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0b_nc_processing import open_raw_netcdf_file, standardize_raw_dataset


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
        "diameter": "diameter_bin_center",
        "velocity": "velocity_bin_center",
        ### Variables
        "rainfall_rate_32bit": "rainfall_rate_32bit",
        "synop_WaWa": "weather_code_synop_4680",
        "synop_WW": "weather_code_synop_4677",
        "radar_reflectivity": "reflectivity_32bit",
        "visibility": "mor_visibility",
        "interval": "sample_interval",
        "sig_laser": "laser_amplitude",
        "n_particles": "number_particles",
        "T_sensor": "sensor_temperature",
        "I_heating": "sensor_heating_current",
        "V_power_supply": "sensor_battery_voltage",
        "state_sensor": "sensor_status",
        "error_code": "error_code",
        "kinetic_energy": "rain_kinetic_energy",
        "snowfall_rate": "snowfall_rate",
        "fall_velocity": "raw_drop_average_velocity",
        "number_concentration": "raw_drop_concentration",
        "data_raw": "raw_drop_number",
    }

    # Rename dataset variables and columns and infill missing variables
    ds = standardize_raw_dataset(ds=ds, dict_names=dict_names, sensor_name="PARSIVEL2")

    # Ensure sensor_temperature in Celsius degree (as logged by sensor)
    ds["sensor_temperature"] = ds["sensor_temperature"] - 273.15

    # Return the dataset adhering to DISDRODB L0B standards
    return ds
