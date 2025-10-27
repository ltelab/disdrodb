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
"""Reader for RWANDA DELFT Thies LPM sensor in netCDF format."""

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
    ds["time"].attrs.pop("comment", None)
    ds["time"].attrs.pop("units", None)
    ds = ds.set_coords("time")

    # Define dictionary mapping dataset variables to select and rename
    dict_names = {
        ### Dimensions
        "diameter_classes": "diameter_bin_center",
        "velocity_classes": "velocity_bin_center",
        ### Variables
        "weather_code_synop_4680": "weather_code_synop_4680",
        "weather_code_synop_4677": "weather_code_synop_4677",
        "weather_code_metar_4678": "weather_code_metar_4678",
        "liquid_precip_intensity": "rainfall_rate",
        "solid_precip_intensity": "snowfall_rate",
        "all_precip_intensity": "precipitation_rate",
        "reflectivity": "reflectivity",
        "visibility": "mor_visibility",
        "measurement_quality": "quality_index",
        "maximum_diameter_hail": "max_hail_diameter",
        "status_laser": "laser_status",
        "status_output_laser_power": "control_output_laser_power_status",
        "interior_temperature": "temperature_interior",
        "temperature_of_laser_driver": "laser_temperature",
        "mean_value_laser_current": "laser_current_average",
        "control_voltage": "control_voltage",
        "optical_control_output": "optical_control_voltage_output",
        "voltage_sensor_supply": "sensor_voltage_supply",
        "current_heating_laser_head": "current_heating_pane_transmitter_head",
        "current_heating_receiver_head": "current_heating_pane_receiver_head",
        "ambient_temperature": "temperature_ambient",
        "voltage_heating_supply": "current_heating_voltage_supply",
        "current_heating_housing": "current_heating_house",
        "current_heating_heads": "current_heating_heads",
        "current_heating_carriers": "current_heating_carriers",
        "number_of_all_measured_particles": "number_particles",
        "number_of_particles_slower_than_0.15": "number_particles_min_speed",
        "number_of_particles_faster_than_20": "number_particles_max_speed",
        "number_of_particles_smaller_than_0.15": "number_particles_min_diameter",
        "number_of_particles_with_unknown_classification": "number_particles_unknown_classification",
        "total_volume_gross_particles_unknown_classification": "number_particles_unknown_classification_internal_data",
        "number_of_particles_class_1": "number_particles_class_1",
        "total_volume_gross_of_class_1": "number_particles_class_1_internal_data",
        "number_of_particles_class_2": "number_particles_class_2",
        "total_volume_gross_of_class_2": "number_particles_class_2_internal_data",
        "number_of_particles_class_3": "number_particles_class_3",
        "total_volume_gross_of_class_3": "number_particles_class_3_internal_data",
        "number_of_particles_class_4": "number_particles_class_4",
        "total_volume_gross_of_class_4": "number_particles_class_4_internal_data",
        "number_of_particles_class_5": "number_particles_class_5",
        "total_volume_gross_of_class_5": "number_particles_class_5_internal_data",
        "number_of_particles_class_6": "number_particles_class_6",
        "total_volume_gross_of_class_6": "number_particles_class_6_internal_data",
        "number_of_particles_class_7": "number_particles_class_7",
        "total_volume_gross_of_class_7": "number_particles_class_7_internal_data",
        "number_of_particles_class_8": "number_particles_class_8",
        "total_volume_gross_of_class_8": "number_particles_class_8_internal_data",
        "number_of_particles_class_9": "number_particles_class_9",
        "total_volume_gross_of_class_9": "number_particles_class_9_internal_data",
        "raw_data": "raw_drop_number",
    }

    # Rename dataset variables and columns and infill missing variables
    ds = standardize_raw_dataset(ds=ds, dict_names=dict_names, sensor_name="LPM")

    # Return the dataset adhering to DISDRODB L0B standards
    return ds
