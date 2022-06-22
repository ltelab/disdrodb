#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 09:29:40 2022

@author: ghiggi
"""
  
 

#-----------------------------------------------------------------------------.

### THIES DESCRIPTION 

#     field_dict = {
#                     {'No': '1', 'Column': '1', 'Len': '1', 'Description': 'STX (start identifier)'},
#                     {'No': '2', 'Column': '2-3', 'Len': '2', 'Description': 'Device address (factory setting „00“) (NN)'},
#                     {'No': '3', 'Column': '5-8', 'Len': '4', 'Description': 'Serial number (NNNN)'},
#                     {'No': '4', 'Column': '10-13', 'Len': '5', 'Description': 'Software-Version (N.NN)'},
#                     {'No': '5', 'Column': '15-22', 'Len': '8', 'Description': 'Date of the sensor (tt.mm.jj)'},
#                     {'No': '6', 'Column': '24-31', 'Len': '8', 'Description': 'Time of the sensor (on request) (hh:mm:ss)'},
#                     {'No': '7', 'Column': '33-34', 'Len': '2', 'Description': '5M SYNOP Tab.4677 (5 minutes mean value) (NN)'},
#                     {'No': '8', 'Column': '36-37', 'Len': '2', 'Description': '5M SYNOP Tab.4680 (5 minutes mean value) (NN)'},
#                     {'No': '9', 'Column': '39-43', 'Len': '5', 'Description': '5M METAR Tab.4678 (5 minutes mean value) (AAAAA)'},
#                     {'No': '10', 'Column': '45-51', 'Len': '7', 'Description': '5M Intensität [mm/h] (5 minutes mean value) (NNN.NNN)'},
#                     {'No': '11', 'Column': '53-54', 'Len': '2', 'Description': '1M SYNOP Tab.4677 (1 minute value) (NN)'},
#                     {'No': '12', 'Column': '56-57', 'Len': '2', 'Description': '1M SYNOP Tab.4680 (1 minute value) (NN)'},
#                     {'No': '13', 'Column': '59-63', 'Len': '5', 'Description': '1M METAR Tab.4678 (1 minute value) (AAAAA)'},
#                     {'No': '14', 'Column': '65-71', 'Len': '7', 'Description': '1M Intensity [mm/h] total precipitation (1 minute value) (NNN.NNN)'},
#                     {'No': '15', 'Column': '73-79', 'Len': '7', 'Description': '1M Intensity [mm/h] liquid precipitation (1 minute value) (NNN.NNN)'},
#                     {'No': '16', 'Column': '81-87', 'Len': '7', 'Description': '1M Intensity [mm/h] solid precipitation (1 minute value) (NNN.NNN)'},
#                     {'No': '17', 'Column': '89-95', 'Len': '7', 'Description': 'Precipitation amount [mm] (Reset with command „RA“) (NNNN.NN)'},
#                     {'No': '18', 'Column': '97-101', 'Len': '5', 'Description': '1M Visibility in precipitation [0...99999m] (1 minute value) (NNNNN)'},
#                     {'No': '19', 'Column': '103-106', 'Len': '4', 'Description': '1M Radar reflectivity [-9.9...99.9dBZ] (1 minute value) (NN.N)'},
#                     {'No': '20', 'Column': '108-110', 'Len': '3', 'Description': '1M Measuring quality [0...100%] (1 minute value) (NNN)'},
#                     {'No': '21', 'Column': '112-114' '3', 'Description': '1M Maximum diameter hail [mm] (1 minute value) (N.N))'},
#                     {'No': '22', 'Column': '116', 'Len': '1', 'Description': 'Status Laser (OK/on:0, off:1)'},
#                     {'No': '23', 'Column': '118', 'Len': '1', 'Description': 'Static signal (OK:0, Error:1)'},
#                     {'No': '24', 'Column': '120', 'Len': '1', 'Description': 'Status Laser temperature (analogue) (OK:0, Error:1)'},
#                     {'No': '25', 'Column': '122', 'Len': '1', 'Description': 'Status Laser temperature (digital) (OK:0, Error:1)'},
#                     {'No': '26', 'Column': '124', 'Len': '1', 'Description': 'Status Laser current (analogue) (OK:0, Error:1)'},
#                     {'No': '27', 'Column': '126', 'Len': '1', 'Description': 'Status Laser current (digital) (OK:0, Error:1)'},
#                     {'No': '28', 'Column': '128', 'Len': '1', 'Description': 'Status Sensor supply (OK:0, Error:1)'},
#                     {'No': '29', 'Column': '130', 'Len': '1', 'Description': 'Status Current pane heating laser head (OK:0, warning:1)'},
#                     {'No': '30', 'Column': '132', 'Len': '1', 'Description': 'Status Current pane heating receiver head (OK:0, warning:1)'},
#                     {'No': '31', 'Column': '134', 'Len': '1', 'Description': 'Status Temperature sensor (OK:0, warning:1)'},
#                     {'No': '32', 'Column': '136', 'Len': '1', 'Description': 'Status Heating supply (OK:0, warning:1)'},
#                     {'No': '33', 'Column': '138', 'Len': '1', 'Description': 'Status Current heating housing (OK:0, warning:1)'},
#                     {'No': '34', 'Column': '140', 'Len': '1', 'Description': 'Status Current heating heads (OK:0, warning:1)'},
#                     {'No': '35', 'Column': '142', 'Len': '1', 'Description': 'Status Current heating carriers (OK:0, warning:1)'},
#                     {'No': '36', 'Column': '144', 'Len': '1', 'Description': 'Status Control output laser power (OK:0, warning:1)'},
#                     {'No': '37', 'Column': '146', 'Len': '1', 'Description': 'Reserve Status ( 0)'},
#                     {'No': '38', 'Column': '148-150', 'Len': '3', 'Description': 'Interior temperature [°C] (NNN)'},
#                     {'No': '39', 'Column': '152-153', 'Len': '2', 'Description': 'Temperature of laser driver 0-80°C (NN)'},
#                     {'No': '40', 'Column': '155-158', 'Len': '4', 'Description': 'Mean value laser current [1/100 mA] (NNNN)'},
#                     {'No': '41', 'Column': '160-163', 'Len': '4', 'Description': 'Control voltage [mV] (reference value: 4010±5) (NNNN)'},
#                     {'No': '42', 'Column': '165-168', 'Len': '4', 'Description': 'Optical control output [mV] (2300 … 6500) (NNNN)'},
#                     {'No': '43', 'Column': '170-172', 'Len': '3', 'Description': 'Voltage sensor supply [1/10V] (NNN)'},
#                     {'No': '44', 'Column': '174-176', 'Len': '3', 'Description': 'Current pane heating laser head [mA] (NNN)'},
#                     {'No': '45', 'Column': '178-180', 'Len': '3', 'Description': 'Current pane heating receiver head [mA] (NNN)'},
#                     {'No': '46', 'Column': '182-186', 'Len': '5', 'Description': 'Ambient temperature [°C] (NNN.N)'},
#                     {'No': '47', 'Column': '188-190', 'Len': '3', 'Description': 'Voltage Heating supply [1/10 V] (only 5.4110.x1.xxx, otherwise “999”) (NNN)'},
#                     {'No': '48', 'Column': '192-195', 'Len': '4', 'Description': 'Current heating housing [mA] (only 5.4110.x1.xxx, otherwise “9999”) (NNNN)'},
#                     {'No': '49', 'Column': '197-200', 'Len': '4', 'Description': 'Current heating heads [mA] (only 5.4110.x1.xxx, otherwise “9999”) (NNNN)'},
#                     {'No': '50', 'Column': '202-205', 'Len': '4', 'Description': 'Current heating carriers [mA] (only 5.4110.x1.xxx, otherwise “9999”) (NNNN)'},
#                     {'No': '51', 'Column': '207-211', 'Len': '5', 'Description': 'Number of all measured particles (NNNNN)'},
#                     {'No': '52', 'Column': '213-221', 'Len': '9', 'Description': '„00000.000“ (internal data)'},
#                     {'No': '53', 'Column': '223-227', 'Len': '5', 'Description': 'Number of particles < minimal speed (0.15m/s) (NNNNN)'},
#                     {'No': '54', 'Column': '229-237', 'Len': '9', 'Description': '„00000.000“ (internal data)'},
#                     {'No': '55', 'Column': '239-243', 'Len': '5', 'Description': 'Number of particles > maximal speed (20m/s) (NNNNN)'},
#                     {'No': '56', 'Column': '245-253', 'Len': '9', 'Description': '„00000.000“ (internal data)'},
#                     {'No': '57', 'Column': '255-259', 'Len': '5', 'Description': 'Number of particles < minimal diameter (0.15mm) (NNNNN)'},
#                     {'No': '58', 'Column': '261-269', 'Len': '9', 'Description': '„00000.000“ (internal data)'},
#                     {'No': '59', 'Column': '271-275', 'Len': '5', 'Description': 'Number of particles no hydrometeor'},
#                     {'No': '60', 'Column': '277-285', 'Len': '9', 'Description': 'Total volume (gross) of this class'},
#                     {'No': '61', 'Column': '287-291', 'Len': '5', 'Description': 'Number of particles with unknown classification'},
#                     {'No': '62', 'Column': '293-301', 'Len': '9', 'Description': 'Total volume (gross) of this class'},
#                     {'No': '63', 'Column': '303-307', 'Len': '5', 'Description': 'Number of particles class 1'},
#                     {'No': '64', 'Column': '309-317', 'Len': '9', 'Description': 'Total volume (gross) of class 1'},
#                     {'No': '65', 'Column': '319-323', 'Len': '5', 'Description': 'Number of particles class 2'},
#                     {'No': '66', 'Column': '325-333', 'Len': '9', 'Description': 'Total volume (gross) of class 2'},
#                     {'No': '67', 'Column': '335-339', 'Len': '5', 'Description': 'Number of particles class 3'},
#                     {'No': '68', 'Column': '341-349', 'Len': '9', 'Description': 'Total volume (gross) of class 3'},
#                     {'No': '69', 'Column': '351-355', 'Len': '5', 'Description': 'Number of particles class 4'},
#                     {'No': '70', 'Column': '357-365', 'Len': '9', 'Description': 'Total volume (gross) of class 4'},
#                     {'No': '71', 'Column': '367-371', 'Len': '5', 'Description': 'Number of particles class 5'},
#                     {'No': '72', 'Column': '373-381', 'Len': '9', 'Description': 'Total volume (gross) of class 5'},
#                     {'No': '73', 'Column': '383-387', 'Len': '5', 'Description': 'Number of particles class 6'},
#                     {'No': '74', 'Column': '389-397', 'Len': '9', 'Description': 'Total volume (gross) of class 6'},
#                     {'No': '75', 'Column': '399-403', 'Len': '5', 'Description': 'Number of particles class 7'},
#                     {'No': '76', 'Column': '405-413', 'Len': '9', 'Description': 'Total volume (gross) of class 7'},
#                     {'No': '77', 'Column': '415-419', 'Len': '5', 'Description': 'Number of particles class 8'},
#                     {'No': '78', 'Column': '421-429', 'Len': '9', 'Description': 'Total volume (gross) of class 8'},
#                     {'No': '79', 'Column': '431-435', 'Len': '5', 'Description': 'Number of particles class 9'},
#                     {'No': '80', 'Column': '437-445', 'Len': '9', 'Description': 'Total volume (gross) of class 9'},
#                     {'No': '81', 'Column': '447-449', 'Len': '3', 'Description': 'Precipitation spectrum'},
#                     {'No': '520', 'Column': '2203-2205', 'Len': '3', 'Description': 'Diameter and speed (NNN)'},
#                     {'No': '521', 'Column': '2228-2229', 'Len': '2', 'Description': 'Checksum (AA)'},
#                     {'No': '522', 'Column': '2231-2232', 'Len': '2', 'Description': 'CRLF'},
#                     {'No': '523', 'Column': '2233', 'Len': '1', 'Description': 'ETX (End identifier)'},
#         }
 
 

 
 

#### TODO: check valid min, valid max correspond to data_range in CONFIGS for OTT Parsivel
#     dict_ARM_description = {
#         {
#             "base_time": "2019-12-01 00:00:00 0:00",
#             "long_name": "Base time in Epoch",
#             "ancillary_variables": "time_offset",
#         },
#         {
#             "time_offset": "Time offset from base_time",
#             "ancillary_variables": "base_time",
#         },
#         {
#             "precip_rate": "Precipitation intensity",
#             "units": "mm/hr",
#             "valid_min": 0.0,
#             "valid_max": 99.999,
#             "standard_name": "lwe_precipitation_rate",
#             "ancillary_variables": "qc_precip_rate",
#         },
#         {
# #         {
#             "weather_code": "SYNOP WaWa Table 4680",
#             "units": "1",
#             "valid_min": 0,
#             "valid_max": 90,
#             "ancillary_variables": "qc_weather_code",
#         },

#             "equivalent_radar_reflectivity_ott": "Radar reflectivity from the manufacturer's software",
#             "units": "dBZ",
#             "valid_min": -60.0,
#             "valid_max": 250.0,
#             "ancillary_variables": "qc_equivalent_radar_reflectivity_ott",
#         },
#  
#             "number_detected_particles": "Number of particles detected",
#             "units": "count",
#             "valid_min": 0,
#             "valid_max": 99999,
#             "ancillary_variables": "qc_number_detected_particles",
#         },
 
#         {
#             "mor_visibility": "Meteorological optical range visibility",
#             "units": "m",
#             "valid_min": 0,
#             "valid_max": 9999,
#             "standard_name": "visibility_in_air",
#             "ancillary_variables": "qc_mor_visibility",
#         },
 
#         {
#             "snow_depth_intensity": "New snow height",
#             "units": "mm/hr",
#             "valid_min": 0.0,
#             "valid_max": 99.999,
#             "ancillary_variables": "qc_snow_depth_intensity",
#             "comment": "This value is valid on a short period of one hour and its purpose is to provide new snow height on railways or roads for the purposes of safety.  It is not equivalent to the WMO definition of snow intensity nor does if follow from WMO observation guide lines.",
#         },
 
#         {
#             "laserband_amplitude": "Laserband amplitude",
#             "units": "count",
#             "valid_min": 0,
#             "valid_max": 99999,
#             "ancillary_variables": "qc_laserband_amplitude",
#         },
 
#         {
#             "sensor_temperature": "Temperature in sensor",
#             "units": "degC",
#             "valid_min": -100,
#             "valid_max": 100,
#         },
#         {
#             "heating_current": "Heating current",
#             "units": "A",
#             "valid_min": 0.0,
#             "valid_max": 9.9,
#             "ancillary_variables": "qc_heating_current",
#         },
          {
#             "sensor_voltage": "Sensor voltage",
#             "units": "V",
#             "valid_min": 0.0,
#             "valid_max": 99.9,
#             "ancillary_variables": "qc_sensor_voltage",
#         },
#          {
#             "lat": "North latitude",
#             "units": "degree_N",
#             "valid_min": -90.0,
#             "valid_max": 90.0,
#             "standard_name": "latitude",
#         },
#         {
#             "lon": "East longitude",
#             "units": "degree_E",
#             "valid_min": -180.0,
#             "valid_max": 180.0,
#             "standard_name": "longitude",
#         },
#         {
#             "alt": "Altitude above mean sea level",
#             "units": "m",
#             "standard_name": "altitude",
#         },
#     }
 
