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
"""Reader for OCEANRAIN ODM470 data in netCDF format."""
import numpy as np

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0b_nc_processing import open_raw_netcdf_file, standardize_raw_dataset
from disdrodb.utils.warnings import suppress_warnings


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""
    ##------------------------------------------------------------------------.
    #### Open the netCDF
    with suppress_warnings():
        ds = open_raw_netcdf_file(filepath=filepath, logger=logger)

    ##------------------------------------------------------------------------.
    #### Adapt the dataframe to adhere to DISDRODB L0 standards
    # Reset time encoding
    ds["time"].encoding = {}

    # Retrieve spectrum
    bins = "bin" + np.arange(1, 129).astype(str)
    raw_spectrum = ds[bins].to_array(dim="diameter_bin_center")
    ds["raw_spectrum"] = raw_spectrum.transpose("time", ...)
    ds = ds.drop_vars(bins)

    # Define dictionary mapping dataset variables to select and rename
    dict_names = {
        ### Dimensions
        "diameter_bin_center": "diameter_bin_center",
        ### Variables
        "reference_voltage": "reference_voltage",
        "relative_wind_speed": "relative_wind_speed",
        "ODM470_precipitation_rate_R": "precipitation_rate",
        "precip_flag": "precip_flag",
        "raw_spectrum": "raw_drop_number",
    }

    # Rename dataset variables and columns and infill missing variables
    ds = standardize_raw_dataset(ds=ds, dict_names=dict_names, sensor_name="ODM470")

    # Return the dataset adhering to DISDRODB L0B standards
    return ds
