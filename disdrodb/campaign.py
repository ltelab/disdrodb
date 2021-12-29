#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 17:48:47 2021

@author: kimbo
"""

class Campaign:
    def __init__(   self,
                    campaign_name,
                    country,
                    continent,
                    title,
            	   	description,
            	   	institution,
            	   	source,
            	   	history,
            	   	conventions,
                    crs,
                    proj4,
                    EPSGlatitude_unit,
                    contributors,
            	   	authors,
            	   	reference,
            	   	documentation,
            	   	website,
            	   	source_repository,
            	   	doi,
            	   	contact,
            	   	contact_information,
            	   	source_data_formatraw_data,
            	   	obs_typeraw,
            	   	levelL0,
                    EPSG,
                    proj4_string,
                 ):
        self.campaign_name = campaign_name
        self.country = country
        self.continent = continent
        self.continent = continent
        self.title = title
        self.description = description
        self.institution = institution
        self.source = source
        self.history = history
        self.conventions = conventions
        self.crs = crs
        self.proj4 = proj4
        self.EPSGlatitude_unit = EPSGlatitude_unit
        self.contributors = contributors
        self.authors = authors
        self.reference = reference
        self.documentation = documentation
        self.website = website
        self.source_repository = source_repository
        self.doi = doi
        self.contact = contact
        self.contact_information = contact_information
        self.source_data_formatraw_data = source_data_formatraw_data
        self.obs_typeraw = obs_typeraw
        self.levelL0 = levelL0
        self.EPSG = EPSG
        self.proj4_string = proj4_string