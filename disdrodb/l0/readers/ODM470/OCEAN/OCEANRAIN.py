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
"""Reader for OceanRAIN ODM470 R and W files in netCDF format."""
import os

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
    import xarray as xr

    ##------------------------------------------------------------------------.
    #### Open the OceanRAIN-R netCDF
    with suppress_warnings():
        ds_r = open_raw_netcdf_file(filepath=filepath, logger=logger)

    ##------------------------------------------------------------------------.
    #### Adapt the dataset to adhere to DISDRODB L0 standards
    # Reset time encoding
    ds_r["time"].encoding = {}

    # Retrieve raw spectrum
    bins = "bin" + np.arange(1, 129).astype(str)
    raw_spectrum = ds_r[bins].to_array(dim="diameter_bin_center")
    ds_r["raw_spectrum"] = raw_spectrum.transpose("time", ...)
    ds_r = ds_r.drop_vars(bins)

    # Define dictionary mapping dataset variables to select and rename
    dict_names = {
        ### Dimensions
        "diameter_bin_center": "diameter_bin_center",
        "raw_spectrum": "raw_drop_number",
    }

    # Rename dataset variables and columns and infill missing variables
    ds_r = standardize_raw_dataset(ds=ds_r, dict_names=dict_names, sensor_name="ODM470")

    # Drop duplicate times from da before reindexing
    _, index = np.unique(ds_r["time"], return_index=True)
    ds_r = ds_r.isel(time=index)

    # -----------------------------------------------------------------
    # Retrieve path to OCEANRAIN-W file
    w_filepath = filepath.replace("OceanRAIN-R", "OceanRAIN-W")
    if not os.path.exists(w_filepath):
        raise ValueError("OceanRAIN-W product not available. Please download that too.")

    # Open OCEANRAIN-W file
    ds_w = open_raw_netcdf_file(filepath=w_filepath, logger=logger)

    # Remove bad encodings
    ds_w["reference_voltage"].encoding = {}
    ds_w["relative_wind_speed"].encoding = {}
    # ds_w["relative_wind_speed_ODM470"].encoding = {}

    # Select variable of interest
    dict_names = {
        "air_temperature": "air_temperature",
        "relative_humidity": "relative_humidity",
        # 'air_pressure': "air_pressure",
        # 'dew_point_temperature': "dew_point_temperature",
        "relative_wind_speed_ODM470": "relative_wind_speed",  # used for N(D) computations
        # "relative_wind_speed": "relative_wind_speed",
        # "relative_wind_direction": "relative_wind_direction",
        "true_wind_speed": "wind_speed",
        "true_wind_direction": "wind_direction",
        "reference_voltage": "reference_voltage",
        "ODM470_precipitation_rate_R": "precipitation_rate",
        "precip_flag": "precip_flag",
        # 'ww_present_weather_code': "weather_code_synop_4677", # 99 in many datasets
    }
    ds_w = ds_w[list(dict_names)]
    ds_w = ds_w.rename(dict_names)

    # Set to NaN wind speed values < 0 (e.g. -5.1)
    ds_w["relative_wind_speed"] = ds_w["relative_wind_speed"].where(ds_w["relative_wind_speed"] >= 0)
    ds_w["wind_speed"] = ds_w["wind_speed"].where(ds_w["wind_speed"] >= 0)

    # Drop duplicate times from OCEAN-RAIN W
    _, index = np.unique(ds_w["time"], return_index=True)
    ds_w = ds_w.isel(time=index)

    # -----------------------------------------------------------------
    # Reindex raw drop number to match OCEAN-RAIN W
    ds_w["raw_drop_number"] = ds_r["raw_drop_number"].reindex(time=ds_w["time"])

    # Set raw_drop_number to 0 where precip_flag is 3 (true zero value)
    ds_w["raw_drop_number"] = xr.where(ds_w["precip_flag"] == 3, 0, ds_w["raw_drop_number"])

    # Remove timesteps with inoperative ODM
    ds_w = ds_w.isel(time=~ds_w["precip_flag"].isin([4, 5]))

    # Replace precip_flag 3 with -1
    # --> This allow max precip_flag resampling
    ds_w["precip_flag"] = ds_w["precip_flag"].where(ds_w["precip_flag"] != 3, -1)
    ds_w["precip_flag"].attrs["flag_meanings"] = "true_zero_value rain snow mixed_phase"
    ds_w["precip_flag"].attrs["flag_values"] = np.array([-1, 0, 1, 2], dtype=int)

    # Return the dataset adhering to DISDRODB L0B standards
    return ds_w
