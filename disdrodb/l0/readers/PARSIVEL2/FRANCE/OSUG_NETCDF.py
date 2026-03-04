#!/usr/bin/env python3
# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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
"""Reader for OSUG DISDRODB L0C netCDFs."""

import xarray as xr

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
    # Add time coordinate
    ds["time"] = ds["time"].astype("M8[s]")

    # Add sample interval as function of time
    ds["sample_interval"] = xr.ones_like(ds["time"], dtype=float) * ds["sample_interval"].item()
    ds = ds.reset_coords("sample_interval")

    # Define dictionary mapping dataset variables to select and rename
    dict_vars = {
        "sample_interval": "sample_interval",
        "rainfall_rate_32bit": "rainfall_rate_32bit",
        "rainfall_accumulated_32bit": "rainfall_accumulated_32bit",
        "weather_code_synop_4680": "weather_code_synop_4680",
        "weather_code_synop_4677": "weather_code_synop_4677",
        "reflectivity_32bit": "reflectivity_32bit",
        "mor_visibility": "mor_visibility",
        "laser_amplitude": "laser_amplitude",
        "number_particles": "number_particles",
        "sensor_temperature": "sensor_temperature",
        "sensor_heating_current": "sensor_heating_current",
        "sensor_battery_voltage": "sensor_battery_voltage",
        "sensor_temperature_receiver": "sensor_temperature_receiver",
        "sensor_temperature_trasmitter": "sensor_temperature_trasmitter",
        "sensor_status": "sensor_status",
        "error_code": "error_code",
        "sensor_serial_number": "sensor_serial_number",
        "rain_kinetic_energy": "rain_kinetic_energy",
        "raw_drop_concentration": "raw_drop_concentration",
        "raw_drop_average_velocity": "raw_drop_average_velocity",
        "raw_drop_number": "raw_drop_number",
        "air_temperature": "air_temperature",
        "relative_humidity": "relative_humidity",
        "wind_speed": "wind_speed",
        "wind_direction": "wind_direction",
    }
    dict_vars = {k: v for k, v in dict_vars.items() if k in ds}
    dict_names = {
        ### Dimensions
        "diameter_bin_center": "diameter_bin_center",
        "velocity_bin_center": "velocity_bin_center",
        ### Variables
    }
    dict_names.update(dict_vars)

    # Rename dataset variables and columns and infill missing variables
    ds = standardize_raw_dataset(ds=ds, dict_names=dict_names, sensor_name="PARSIVEL2")

    # Return the dataset adhering to DISDRODB L0B standards
    return ds
