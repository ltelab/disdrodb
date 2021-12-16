#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 13:52:33 2021

@author: kimbo
"""

class Sensor:
    def __init__(self,
                disdrodb_id,
                sensor_name,
                sensor_long_name,
                sensor_beam_width,
                sensor_nominal_width,
                measurement_interval,
                temporal_resolution,
                sensor_wavelegth,
                sensor_serial_number,
                firmware_IOP,
                firmware_DSP,
                station_id,
                station_name,
                station_number,
                location,
                country,
                continent,
                latitude,
                longitude,
                altitude,
                latitude_unit,
                longitude_unit,
                altitude_unit,
                crs,
                EPSG,
                proj4_string,
                 ):
        self.disdrodb_id = disdrodb_id
        self.sensor_name = sensor_name
        self.sensor_long_name = sensor_long_name
        self.sensor_beam_width = sensor_beam_width
        self.sensor_nominal_width = sensor_nominal_width
        self.measurement_interval = measurement_interval
        self.temporal_resolution = temporal_resolution
        self.sensor_wavelegth = sensor_wavelegth
        self.sensor_serial_number = sensor_serial_number
        self.firmware_IOP = firmware_IOP
        self.firmware_DSP = firmware_DSP
        self.station_id = station_id
        self.station_name = station_name
        self.station_number = station_number
        self.location = location
        self.country = country
        self.continent = continent
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.latitude_unit = latitude_unit
        self.longitude_unit = longitude_unit
        self.altitude_unit = altitude_unit
        self.crs = crs
        self.EPSG = EPSG
        self.proj4_string = proj4_string