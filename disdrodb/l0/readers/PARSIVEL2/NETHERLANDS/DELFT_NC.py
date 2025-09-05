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
    # Add time coordinate
    ds["time"] = ds["time_as_string"].astype("M8[s]")
    ds["time"].attrs.pop("comment", None)
    ds["time"].attrs.pop("units", None)
    ds = ds.set_coords("time")

    # Define dictionary mapping dataset variables to select and rename
    dict_names = {
        ### Dimensions
        "diameter_classes": "diameter_bin_center",
        "velocity_classes": "velocity_bin_center",
        ### Variables
        "rainfall_rate_32bit": "rainfall_rate_32bit",
        "weather_code_synop_4680": "weather_code_synop_4680",
        "weather_code_synop_4677": "weather_code_synop_4677",
        "weather_code_metar_4678": "weather_code_metar_4678",
        "weather_code_nws": "weather_code_nws",
        "reflectivity_32bit": "reflectivity_32bit",
        "mor_visibility": "mor_visibility",
        "laser_amplitude": "laser_amplitude",
        "number_particles_validated": "number_particles",
        "sensor_temperature": "sensor_temperature",
        "error_code": "error_code",
        "kinetic_energy": "rain_kinetic_energy",
        "fieldV": "raw_drop_average_velocity",
        "fieldN": "raw_drop_concentration",
        "raw_data": "raw_drop_number",
    }

    # Rename dataset variables and columns and infill missing variables
    ds = standardize_raw_dataset(ds=ds, dict_names=dict_names, sensor_name="PARSIVEL2")

    # Replace NaN flags
    ds["raw_drop_number"] = ds["raw_drop_number"].where(ds["raw_drop_number"] < 65437)  # dummy flag 65437.0

    # Return the dataset adhering to DISDRODB L0B standards
    return ds
