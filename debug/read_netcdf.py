#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:49:23 2021

@author: kimbo
"""

import pandas as pd
import dask.dataframe as dd
import os
import xarray as xr
import netCDF4

campagna = 'PARSIVEL_2007'
path = f'/SharedVM/Campagne/EPFL/Processed/{campagna}/L1'
file = campagna + '_s10.nc'
file_path = os.path.join(path, file)

ds1 = xr.open_dataset(file_path)

campagna = 'anxldM1.b1.20191201.000000'
path = "/SharedVM/Campagne/ARM/anxldM1"
file = campagna + '.cdf'
file_path = os.path.join(path, file)

ds = xr.open_dataset(file_path)

net = ds.rename(dict_ARM)

# ds['time'] = ds['time'].astype('M8')

# ds['FieldN'].plot(x = 'time', y = 'diameter_bin_center')

# ds.plot(x = 'time', y = 'diameter_bin_center')

df = ds.to_dataframe()

df1 = df.head(1000)

print(ds)

ds.close()

dict_ARM_to_l0 =    {'time': 'time',
                    'time_offset': 'time_offset_OldName',
                    'precip_rate': 'rain_rate_32bit',
                    'qc_precip_rate': 'qc_precip_rate_OldName',
                    'weather_code': 'weather_code_SYNOP_4680',
                    'qc_weather_code': 'qc_weather_code_OldName',
                    'equivalent_radar_reflectivity_ott': 'reflectivity_32bit',
                    'qc_equivalent_radar_reflectivity_ott': 'qc_equivalent_radar_reflectivity_ott_OldName',
                    'number_detected_particles': 'n_particles',
                    'qc_number_detected_particles': 'qc_number_detected_particles_OldName',
                    'mor_visibility': 'mor_visibility_OldName',
                    'qc_mor_visibility': 'qc_mor_visibility_OldName',
                    'snow_depth_intensity': 'snow_depth_intensity_OldName',
                    'qc_snow_depth_intensity': 'qc_snow_depth_intensity_OldName',
                    'laserband_amplitude': 'laser_amplitude',
                    'qc_laserband_amplitude': 'qc_laserband_amplitude_OldName',
                    'sensor_temperature': 'sensor_temperature',
                    'heating_current': 'sensor_heating_current',
                    'qc_heating_current': 'qc_heating_current_OldName',
                    'sensor_voltage': 'sensor_battery_voltage',
                    'qc_sensor_voltage': 'qc_sensor_voltage_OldName',
                    'class_size_width': 'class_size_width_OldName',
                    'fall_velocity_calculated': 'fall_velocity_calculated_OldName',
                    'raw_spectrum': 'raw_spectrum_OldName',
                    'liquid_water_content': 'liquid_water_content_OldName',
                    'equivalent_radar_reflectivity': 'equivalent_radar_reflectivity_OldName',
                    'intercept_parameter': 'intercept_parameter_OldName',
                    'slope_parameter': 'slope_parameter_OldName',
                    'median_volume_diameter': 'median_volume_diameter_OldName',
                    'liquid_water_distribution_mean': 'liquid_water_distribution_mean_OldName',
                    'number_density_drops': 'number_density_drops_OldName',
                    'diameter_min': 'diameter_min_OldName',
                    'diameter_max': 'diameter_max_OldName',
                    'moment1': 'moment1_OldName',
                    'moment2': 'moment2_OldName',
                    'moment3': 'moment3_OldName',
                    'moment4': 'moment4_OldName',
                    'moment5': 'moment5_OldName',
                    'moment6': 'moment6_OldName',
                    'lat': 'latitude',
                    'lon': 'longitude',
                    'alt': 'altitude',
                    }


dict_ARM_description = {
                        {'base_time': '2019-12-01 00:00:00 0:00', 'long_name': 'Base time in Epoch', 'ancillary_variables': 'time_offset'},
                        {'time_offset': 'Time offset from base_time', 'ancillary_variables': 'base_time'},
                        {'precip_rate': 'Precipitation intensity', 'units': 'mm/hr', 'valid_min': 0.0, 'valid_max': 99.999, 'standard_name': 'lwe_precipitation_rate', 'ancillary_variables': 'qc_precip_rate'},
                        {'qc_precip_rate': 'Quality check results on field: Precipitation intensity', 'units': '1', 'description': 'This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.', 'flag_method': 'bit', 'bit_1_description': 'Value is equal to missing_value.', 'bit_1_assessment': 'Bad', 'bit_2_description': 'Value is less than the valid_min.', 'bit_2_assessment': 'Bad', 'bit_3_description': 'Value is greater than the valid_max.', 'bit_3_assessment': 'Bad'},
                        {'weather_code': 'SYNOP WaWa Table 4680', 'units': '1', 'valid_min': 0, 'valid_max': 90, 'ancillary_variables': 'qc_weather_code'},
                        {'qc_weather_code': 'Quality check results on field: SYNOP WaWa Table 4680', 'units': '1', 'description': 'This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.', 'flag_method': 'bit', 'bit_1_description': 'Value is equal to missing_value.', 'bit_1_assessment': 'Bad', 'bit_2_description': 'Value is less than the valid_min.', 'bit_2_assessment': 'Bad', 'bit_3_description': 'Value is greater than the valid_max.', 'bit_3_assessment': 'Bad'},
                        {'equivalent_radar_reflectivity_ott': "Radar reflectivity from the manufacturer's software", 'units': 'dBZ', 'valid_min': -60.0, 'valid_max': 250.0, 'ancillary_variables': 'qc_equivalent_radar_reflectivity_ott'},
                        {'qc_equivalent_radar_reflectivity_ott': "Quality check results on field: Radar reflectivity from the manufacturer's software", 'units': '1', 'description': 'This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.', 'flag_method': 'bit', 'bit_1_description': 'Value is equal to missing_value.', 'bit_1_assessment': 'Bad', 'bit_2_description': 'Value is less than the valid_min.', 'bit_2_assessment': 'Bad', 'bit_3_description': 'Value is greater than the valid_max.', 'bit_3_assessment': 'Bad'},
                        {'number_detected_particles': 'Number of particles detected', 'units': 'count', 'valid_min': 0, 'valid_max': 99999, 'ancillary_variables': 'qc_number_detected_particles'},
                        {'qc_number_detected_particles': 'Quality check results on field: Number of particles detected', 'units': '1', 'description': 'This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.', 'flag_method': 'bit', 'bit_1_description': 'Value is equal to missing_value.', 'bit_1_assessment': 'Bad', 'bit_2_description': 'Value is less than the valid_min.', 'bit_2_assessment': 'Bad', 'bit_3_description': 'Value is greater than the valid_max.', 'bit_3_assessment': 'Bad'},
                        {'mor_visibility': 'Meteorological optical range visibility', 'units': 'm', 'valid_min': 0, 'valid_max': 9999, 'standard_name': 'visibility_in_air', 'ancillary_variables': 'qc_mor_visibility'},
                        {'qc_mor_visibility': 'Quality check results on field: Meteorological optical range visibility', 'units': '1', 'description': 'This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.', 'flag_method': 'bit', 'bit_1_description': 'Value is equal to missing_value.', 'bit_1_assessment': 'Bad', 'bit_2_description': 'Value is less than the valid_min.', 'bit_2_assessment': 'Bad', 'bit_3_description': 'Value is greater than the valid_max.', 'bit_3_assessment': 'Bad'},
                        {'snow_depth_intensity': 'New snow height', 'units': 'mm/hr', 'valid_min': 0.0, 'valid_max': 99.999, 'ancillary_variables': 'qc_snow_depth_intensity', 'comment': 'This value is valid on a short period of one hour and its purpose is to provide new snow height on railways or roads for the purposes of safety.  It is not equivalent to the WMO definition of snow intensity nor does if follow from WMO observation guide lines.'},
                        {'qc_snow_depth_intensity': 'Quality check results on field: New snow height', 'units': '1', 'description': 'This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.', 'flag_method': 'bit', 'bit_1_description': 'Value is equal to missing_value.', 'bit_1_assessment': 'Bad', 'bit_2_description': 'Value is less than the valid_min.', 'bit_2_assessment': 'Bad', 'bit_3_description': 'Value is greater than the valid_max.', 'bit_3_assessment': 'Bad'},
                        {'laserband_amplitude': 'Laserband amplitude', 'units': 'count', 'valid_min': 0, 'valid_max': 99999, 'ancillary_variables': 'qc_laserband_amplitude'},
                        {'qc_laserband_amplitude': 'Quality check results on field: Laserband amplitude', 'units': '1', 'description': 'This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.', 'flag_method': 'bit', 'bit_1_description': 'Value is equal to missing_value.', 'bit_1_assessment': 'Bad', 'bit_2_description': 'Value is less than the valid_min.', 'bit_2_assessment': 'Bad', 'bit_3_description': 'Value is greater than the valid_max.', 'bit_3_assessment': 'Bad'},
                        {'sensor_temperature': 'Temperature in sensor', 'units': 'degC', 'valid_min': -100, 'valid_max': 100},
                        {'heating_current': 'Heating current', 'units': 'A', 'valid_min': 0.0, 'valid_max': 9.9, 'ancillary_variables': 'qc_heating_current'},
                        {'qc_heating_current': 'Quality check results on field: Heating current', 'units': '1', 'description': 'This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.', 'flag_method': 'bit', 'bit_1_description': 'Value is equal to missing_value.', 'bit_1_assessment': 'Bad', 'bit_2_description': 'Value is less than the valid_min.', 'bit_2_assessment': 'Bad', 'bit_3_description': 'Value is greater than the valid_max.', 'bit_3_assessment': 'Bad'},
                        {'sensor_voltage': 'Sensor voltage', 'units': 'V', 'valid_min': 0.0, 'valid_max': 99.9, 'ancillary_variables': 'qc_sensor_voltage'},
                        {'qc_sensor_voltage': 'Quality check results on field: Sensor voltage', 'units': '1', 'description': 'This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.', 'flag_method': 'bit', 'bit_1_description': 'Value is equal to missing_value.', 'bit_1_assessment': 'Bad', 'bit_2_description': 'Value is less than the valid_min.', 'bit_2_assessment': 'Bad', 'bit_3_description': 'Value is greater than the valid_max.', 'bit_3_assessment': 'Bad'},
                        {'class_size_width': 'Class size width', 'units': 'mm'},
                        {'fall_velocity_calculated': 'Fall velocity calculated after Lhermite', 'units': 'm/s'},
                        {'raw_spectrum': 'Raw drop size distribution', 'units': 'count'},
                        {'liquid_water_content': 'Liquid water content', 'units': 'mm^3/m^3'},
                        {'equivalent_radar_reflectivity': 'Radar reflectivity calculated by the ingest', 'units': 'dBZ'},
                        {'intercept_parameter': 'Intercept parameter, assuming an ideal Marshall-Palmer type distribution', 'units': '1/(m^3 mm)'},
                        {'slope_parameter': 'Slope parameter, assuming an ideal Marshall-Palmer type distribution', 'units': '1/mm'},
                        {'median_volume_diameter': 'Median volume diameter, assuming an ideal Marshall-Palmer type distribution', 'units': 'mm'},
                        {'liquid_water_distribution_mean': 'Liquid water distribution mean, assuming an ideal Marshall-Palmer type distribution', 'units': 'mm'},
                        {'number_density_drops': 'Number density of drops of the diameter corresponding to a particular drop size class per unit volume', 'units': '1/(m^3 mm)'},
                        {'diameter_min': 'Diameter of smallest drop observed', 'units': 'mm'},
                        {'diameter_max': 'Diameter of largest drop observed', 'units': 'mm'},
                        {'moment1': 'Moment 1 from the observed distribution', 'units': 'mm/m^3'},
                        {'moment2': 'Moment 2 from the observed distribution', 'units': 'mm^2/m^3'},
                        {'moment3': 'Moment 3 from the observed distribution', 'units': 'mm^3/m^3'},
                        {'moment4': 'Moment 4 from the observed distribution', 'units': 'mm^4/m^3'},
                        {'moment5': 'Moment 5 from the observed distribution', 'units': 'mm^5/m^3'},
                        {'moment6': 'Moment 6 from the observed distribution', 'units': 'mm^6/m^3'},
                        {'lat': 'North latitude', 'units': 'degree_N', 'valid_min': -90.0, 'valid_max': 90.0, 'standard_name': 'latitude'},
                        {'lon': 'East longitude', 'units': 'degree_E', 'valid_min': -180.0, 'valid_max': 180.0, 'standard_name': 'longitude'},
                        {'alt': 'Altitude above mean sea level', 'units': 'm', 'standard_name': 'altitude'}
                        }













































