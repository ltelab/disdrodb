#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 17:02:42 2021

@author: kimbo
"""

class Campaign:
    def __init__(   self,
                    list_parameters
                 ):
        self.authors = list_parameters['authors']
        self.campaign_name = list_parameters['campaign_name']
        self.contact = list_parameters['contact']
        self.contact_information = list_parameters['contact_information']
        self.continent = list_parameters['continent']
        self.contributors = list_parameters['contributors']
        self.conventions = list_parameters['conventions']
        self.country = list_parameters['country']
        self.description = list_parameters['description']
        self.documentation = list_parameters['documentation']
        self.doi = list_parameters['doi']
        self.EPSG = list_parameters['EPSG']
        self.EPSGlatitude_unit = list_parameters['EPSGlatitude_unit']
        self.history = list_parameters['history']
        self.institution = list_parameters['institution']
        self.level = list_parameters['level']
        self.obs_type = list_parameters['obs_type']
        self.proj4 = list_parameters['proj4']
        self.proj4_string = list_parameters['proj4_string']
        self.reference = list_parameters['reference']
        self.source = list_parameters['source']
        self.source_data_format = list_parameters['source_data_format']
        self.source_repository = list_parameters['source_repository']
        self.title = list_parameters['title']
        self.website = list_parameters['website']


class Sensor:
    def __init__(self,
                list_parameters,
                path
                 ):
            self.altitude = list_parameters['altitude']
            self.altitude_unit = list_parameters['altitude_unit']
            self.crs = list_parameters['crs']
            self.disdrodb_id = list_parameters['disdrodb_id']
            self.firmware_DSP = list_parameters['firmware_DSP']
            self.firmware_IOP = list_parameters['firmware_IOP']
            self.latitude = list_parameters['latitude']
            self.latitude_unit = list_parameters['latitude_unit']
            self.location = list_parameters['location']
            self.longitude = list_parameters['longitude']
            self.longitude_unit = list_parameters['longitude_unit']
            self.measurement_interval = list_parameters['measurement_interval']
            self.project_name = list_parameters['project_name']
            self.sensor_beam_width = list_parameters['sensor_beam_width']
            self.sensor_long_name = list_parameters['sensor_long_name']
            self.sensor_name = list_parameters['sensor_name']
            self.sensor_nominal_width = list_parameters['sensor_nominal_width']
            self.sensor_serial_number = list_parameters['sensor_serial_number']
            self.sensor_wavelegth = list_parameters['sensor_wavelegth']
            self.station_id = list_parameters['station_id']
            self.station_name = list_parameters['station_name']
            self.station_number = list_parameters['station_number']
            self.temporal_resolution = list_parameters['temporal_resolution']
            self.path = path


"""

Sample JSON attributes template

{
    "campaing": {
        "authors": [],
        "campaign_name": "",
        "contact": "",
        "contact_information": "http://lte.epfl.ch",
        "continent": "Europe",
        "contributors": [],
        "conventions": "",
        "country": "Switzerland",
        "description": "",
        "documentation": "",
        "doi": "",
        "EPSG": 4326,
        "EPSGlatitude_unit": "",
        "history": "",
        "institution": "Laboratoire de Teledetection Environnementale -  Ecole Polytechnique Federale de Lausanne",
        "level": "L0",
        "obs_type": "raw",
        "proj4": "",
        "proj4_string": "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        "reference": "",
        "source": "",
        "source_data_format": "raw_data",
        "source_repository": "",
        "title": "",
        "website": ""
    },
    "device": [
        {
            "altitude": 0,
            "altitude_unit": "MetersAboveSeaLevel",
            "crs": "WGS84",
            "disdrodb_id": 0,
            "firmware_DSP": "",
            "firmware_IOP": "",
            "firmware_version": "",
            "latitude": 0,
            "latitude_unit": "DegreesNorth",
            "location": "",
            "longitude": 0,
            "longitude_unit": "DegreesEast",
            "measurement_interval": 30,
            "project_name": "",
            "sensor_beam_width": 180,
            "sensor_long_name": "OTT Hydromet Parsivel",
            "sensor_name": "Parsivel",
            "sensor_nominal_width": 180,
            "sensor_serial_number": "",
            "sensor_wavelegth": "650 nm",
            "station_id": 0,
            "station_name": "",
            "station_number": 0,
            "temporal_resolution": 30
        }
    ]
}

"""