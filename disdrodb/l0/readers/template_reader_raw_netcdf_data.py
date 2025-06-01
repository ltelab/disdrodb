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
"""DISDRODB reader template for raw text data."""
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

    ##---------------------------------------------------------------------.
    #### Adapt the dataframe to adhere to DISDRODB L0 standards
    # Define dictionary mapping dataset variables and coordinates to keep (and rename)
    # - If the platform is moving, keep longitude, latitude and altitude
    # - If the platform is fixed, remove longitude, latitude and altitude coordinates
    #   --> The geolocation information must be specified in the station metadata !
    dict_names = {
        # Dimensions
        "<raw_dataset_diameter_dimension>": "diameter_bin_center",  # [TO ADAPT]
        "<raw_dataset_velocity_dimension>": "velocity_bin_center",  # [TO ADAPT]
        # Variables
        # - Add here other variables accepted by DISDRODB L0 standards
        "<precipitation_spectrum>": "raw_drop_number",  # [TO ADAPT]
    }

    # Rename dataset variables and columns and infill missing variables
    sensor_name = "LPM"  # [SPECIFY HERE THE SENSOR FOR WHICH THE READER IS DESIGNED]
    ds = standardize_raw_dataset(ds=ds, dict_names=dict_names, sensor_name=sensor_name)

    # Replace occurrence of NaN flags with np.nan
    # - Define a dictionary specifying the value(s) of NaN flags for each variable
    # - The code here below is just an example that requires to be adapted !
    # - This step might not be required with your data !
    dict_nan_flags = {"<raw_drop_number>": [-9999, -999]}
    ds = replace_custom_nan_flags(ds, dict_nan_flags=dict_nan_flags, logger=logger)

    # [ADD ADDITIONAL REQUIRED CUSTOM CODE HERE]

    return ds
