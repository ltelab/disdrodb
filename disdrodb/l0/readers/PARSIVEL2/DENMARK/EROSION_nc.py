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
"""Reader for the EROSION campaign in Denmark."""
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
        "particle_diameter": "diameter_bin_center",
        "particle_fall_speed": "velocity_bin_center",
        ### Variables
        "rain_rate": "rainfall_rate_32bit",
        "rain_amount": "rainfall_accumulated_32bit",
        "SYNOP_4680": "weather_code_synop_4680",
        "SYNOP_4677": "weather_code_synop_4677",
        "reflectivity": "reflectivity_32bit",
        "visibility": "mor_visibility",
        "particles": "number_particles",
        "laser_amplitude": "laser_amplitude",
        "sample_interval": "sample_interval",
        "power_voltage": "sensor_battery_voltage",
        "temp_sensor": "sensor_temperature",
        "heat_status": "sensor_heating_current",
        "status": "sensor_status",
        "rain_kinetic_energy": "rain_kinetic_energy",
        "number_of_particles_classified": "raw_drop_number",
    }

    # Rename dataset variables and columns and infill missing variables
    ds = standardize_raw_dataset(ds=ds, dict_names=dict_names, sensor_name="PARSIVEL2")

    # Return the dataset adhering to DISDRODB L0B standards
    return ds
