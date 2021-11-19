#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:03:47 2021

@author: ghiggi
"""
#####################
### Variable Name ###
#####################         ### Suggested name changes 
rain_rate               
acc_rain_amount               # --> rain_acc / rain_accum
reflectivity_32bit       
# reflectivity_16bit  
MOR                           # --> mor_visibility
amplitude                     # --> JG: laser_amplitude ? GG: signal_amplitude (I guess)  
n_particles
n_all_particles               # Difference with the above? More explicit name?
temperature_sensor  
heating_current               # --> heating_current_sensor or sensor_heating_current
voltage                       # JG: which voltage? is it sensor_voltage --> GG: Power supply voltage --> supply_voltage / power_voltage / battery_voltage
sensor_status                      
error_code                    # is it a sensor error code or data acquisition error code
temperature_PBC     
temperature_right   
temperature_left    
kinetic_energy                # precip_kinetic_energy?
snowfall_intensity  
code_4680                     # wanna add some info on this codes in the name? some are METAR, some are NWS, etc
code_4677                     # SYNOP_4680, SYNOP_4677
code_4678                     # SYNOP_4678 (METAR/SPECI_4678) ?
code_NWS                      # NWS   (prefix: precip_type_<...> --> precip_type_SYNOP_4677
# --> http://www.czo.psu.edu/downloads/Metadataworksheets/LPM_SYNOP_METAR_key.pdf

# Other options 
Nd / d / drop_diameter_counts / Field_N # ???
Vd / v / drop_velocity_counts / Field_V # ???    
N / n / raw_data  / Field_Raw     # ??? pcm Parsivel conditional matrix

# Core dimensions 
time / timestep  # start or end?   --> TIME should be always END in my opinion for weather measurements (jgr)
diameter_bin_center   # --> bin or class ? 
velocity_bin_center

## Coords
lat     # OR  latitude                
lon     # OR  longitude                 
altitude                         
crs
diameter_bin_width    
velocity_bin_width  
diameter_bin_lower
diameter_bin_upper 
velocity_bin_lower
velocity_bin_upper 

### Datalogger EPFL 
# 'Datalogger_temp': 'object',
# 'Datalogger_power': 'object', 
# 'Datalogger_communication': 'uint8', 
 
### Acronyms used by Kimbo   
# 'Signal_amplitude_laser_strip': amplitude
# 'Number_detected_particles':  # n_particles or n_all_particles --> Check based on OTT code
# 'Current_through_heating_system': heating_current
# 'Power_supply_voltage': voltage

### Tim variables 
# 'PrecipCode',
# 'ParsivelStatusCode',
# 'ParsivelIntensity',
# 'CorrectedVolumetricDrops',
# 'RawDrops',
# 'VolumetricDrops',

### PCM 
# - https://github.com/DISDRONET/PyDSD/blob/master/pydsd/parsivel_conditional_matrix.txt 

##----------------------------------------------------------------------------.
####################
### Attributes   ###
#################### 
# - Default: '
# - Lowercase ! 
attrs = {"title"      : 'Parsivel disdrometer observations from Ardeche region, France, for HYMEX campaign in autumn 2012.',
         "description":  'OTT Parsivel data collected during the HYMEX Campaign. See http://hymex.org for details.',
         "institution": 'Laboratoire de Teledetection Environnementale -  Ecole Polytechnique Federale de Lausanne',
         "source"     : 'surface observation',  # what we want to describe?
         "source": 'Parsivel observations of drop counts per velocity/equivolume diameter class, filtered for quality control.',
         "history": '', # created the xxxx-xx-xx ?
         "conventions": '',
         
         'campaign_name': 
         "site_name": '',    
         'station_id': 10,
         'station_name': 'Mirabel',
         'location': '',
         'country': '',
         'continent: '',
         'crs': 'WGS84',
         'latitude': '',
         'longitude': '',
         'latitude_unit': 'DegreesNorth',
         'longitude_unit': 'DegreesEast',
         'altitude_unit': 'MetersAboveSeaLevel',
            
          # Very imporant 
          # - Limited set to be checked on runtime for inclusion in DB 
          # - I would define values such as Parsivel, Parsivel2, ThiesLPM, ...
         "sensor_name": 'Parsivel2',  # instrument name?      
         "sensor_long_name": 'OTT Hydromet Parsivel2',
         "instrument_version": '', # included in sensor_name no?
         "sensor_wavelegth": '', # change between Parsivel version ... 
         "sensor_serial_number": '',  
         "firmware_IOP": '',  
         "firmware_DSP": '', 
         "firmware_version": '', 
         
         # Quantity useful for downstream computations (JG)
         # - Example: conversion from areal DSD to volumetric DSD
         "sensor_beam_width": '', #  Parsivel2 > Parsivel
         'temporal_resolution': 30 # "measurement_interval", default in seconds ? ,
        
         # Attribution 
         "project_name": 'https://ruisdael-observatory.nl/',  # rendunant with campaign_name? 
         "contributors": 'Marc Schleiss, Saverio Guzzo, Rob Mackenzie', 
         "authors": '', # or authors
         'reference': 'XXX et al., ... ',
         'documentation': '',
         'website': '', 
         'source_repository': '',
         'doi': '',
         'contact': '',
         'contact_information': 'http://lte.epfl.ch',
         
         # DISDRO DB attrs 
        'obs_type': '', # raw/preprocessed/postprocessed?
        'level': 'L0', 
      
              
         # To derive 
         # - Years coverage
         # - Total minutes
         # - Total rain events
         # - Other stats TBD 
        
     }  

def get_default_attrs(): 
    attrs_list = ['title','description'] # to add all the others 
    dict_attrs = {var: '' for var in attrs_list}
    return dict_attrs

# How to define attributes 
attrs = get_default_attrs()
attrs['title'] = 'Parsivel disdrometer observations from Ardeche region"
# .... 

##----------------------------------------------------------------------------.
#######################
### Variable dtypes ###
#######################
def var_dtypes(): 
    dtypes = {                                 # Kimbo option
        "rain_rate": 'float32',
        "acc_rain_amount":   'float32',
        "reflectivity_16bit": 'float16',
        "reflectivity_32bit": 'float32',
        "mor"             :'float32',          #  uint16 
        "amplitude"       :'float32',          #  uint32
        "n_particles"     :'int32',            # 'uint32' 
        "n_all_particles": 'int32',            # 'uint32'  
        "temperature_sensor": 'int16',         #  int8
        "heating_current" : 'float32',
        "voltage"         : 'float32',
        "sensor_status"   : 'int8',
        "error_code"      : 'int8',  
        
        "temperature_PBC" : 'int8',
        "temperature_right" : 'int8',
        "temperature_left":'int8',
        "kinetic_energy"  :'float32',
        "snowfall_intensity": 'float32',
        
        "code_4680"      :'int8',             # o uint8
        "code_4677"      :'int8',             # o uint8
        "code_4678"      :'U',
        "code_NWS"       :'U',
        
        # Data fields (TODO) (Log scale?)
        "v",
        "d",
        "n",
        
        # Coords 
        "latitude" : 'float32',
        "longitude" : 'float32',
        "altitude" : 'float32',
         # Dimensions
        'timestep': 'datetime64[ns]', 
        
    }
    return dtypes
  
    #  'Datalogger_temp': 'object',
    #  'Datalogger_power': 'object', 
    #  'Datalogger_communication': 'uint8', 
 
    
# columns = ["Number_detected_particles", "Current_through_heating_system"]
# dtype_subset = {var:var_type()[var] for var in columns}
 

 
 


##---------------------------------------------------------------------------.
### NAN Flags 
# - Determine standard flags for each variable 
# --> -1, -99, nan ... what to use for int (99, or -1)?

def var_na_flag(): 
    flag = {
     'longitude': 'nan', 
     'latitude': 'nan', 
     'timestep': 'nan', 
     'Datalogger_temp': -1,
     'Datalogger_power': 'object', 
     'Datalogger_communication': 'uint8', 
     'Rain_intensity': 'float', 
    }
    return flag 





###---------------------------------------------------------------------------.
def OTT_parsivel_dict(): 
    """
    Get a dictionary containing the variable name of OTT parsivel field numbers.
   
    Returns
    -------
    field_dict : dictionary
        Dictionary with the variable name of OTT parsivel field numbers.
    """ 
    field_dict = {"01": "rain_rate", 
                   "02": "acc_rain_amount", 
                   "03": "code_4680", 
                   "04": "code_4677", 
                   "05": "code_4678", 
                   "06": "code_NWS", 
                   "07": "reflectivity", 
                   "08": "MOR", 
                   "10": "amplitude", 
                   "11": "n_particles", 
                   "12": "temperature_sensor", 
                   "16": "heating_current", 
                   "17": "voltage", 
                   "18": "sensor_status",
                   "25": "error_code", 
                   "26": "temperature_PCB", 
                   "27": "temperature_right", 
                   "28": "temperature_left", 
                   "34": "kinetic_energy", 
                   "35": "snowfall_intensity",     
                   "90": "ND", 
                   "91": "VD", 
                   "93": "N",
                 }
     return field_dict

# c("09","Sample interval",5,"s","single_number")
# c("13","Sensor serial number",6,"","character_string")
# c("14","Firmware IOP",6,"","character_string")
# c("15","Firmware DSP",6,"","character_string")
 
# c("24","Rain amount absolute 32 bit",7,"mm","single_number")
# c("30","Rain intensity 16 bit max 30 mm/h",6,"mm/h","single_number")
# c("31","Rain intensity 16 bit max 1200 mm/h",6,"mm/h","single_number")
# c("32","Rain amount accumulated 16 bit",7,"mm","single_number")
# c("33","Radar reflectivity 16 bit",5,"dBZ","single_number")
# c("60","Number of all particles detected",8,"","single_number")
# c("61","List of all particles detected",13,"","list")
 
def var_units_dict(): 
    """
    Get a dictionary containing the units of the variables
   
    Returns
    -------
    units : dictionary
        Dictionary with the units of the variables
    """ 
    # TODO BE EXPANDED 
    units_dict = {"rain_rate": "mm/h", 
                  "acc_rain_amount": "mm", 
                  "code_4680": "",
                  "code_4677": "", 
                  "code_4678": "", 
                  "code_NWS": "", 
                  "reflectivity": "dBZ", 
                  "MOR": "m", 
                  "amplitude": "",
                  "n_particles": "", 
                  "temperature_sensor": "degree_Celsius", 
                  "heating_current": "A", 
                  "voltage": "V", 
                  "sensor_status": "", 
                  "error_code": "", 
                  "temperature_PCB": "degree_Celsius",
                  "temperature_right": "degree_Celsius", 
                  "temperature_left": "degree_Celsius", 
                  "kinetic_energy": "J/(m2*h)", 
                  "snowfall_intensity": "mm/h",  
                  "ND": "1/(m3*mm)",
                  "VD": "m/s", 
                  "N": "", 
                  }
     return units_dict
    
def var_explanations():
    """
    Get a dictionary containing verbose explanation of the variables 
    
    Returns
    -------
    explanations : dictionary
        Dictionary with the explanation of the variables (keys)
    """ 
    # TODO BE EXPANDED 
    name_dict = { 'timestep':'Datetime object of the measurement',
                "rain_rate": "Rainfall rate", 
                "acc_rain_amount": "accumulate rain amount", 
                "code_4680": "SYNOP weather code according to table 4680 of Parsivel documentation",
                "code_4677": "SYNOP weather code according to table 4677 of Parsivel documentation",
                "code_4678": "METAR/SPECI weather code according to table 4678 of Parsivel documentation",
                "code_NWS": "NWS weather code according to Parsivel documentation", 
                "reflectivity": "radar reflectivity",
                "MOR": "Meteorological Optical Range in precipitation", 
                "amplitude": "Signal amplitude of laser strip",
                "n_particles": "Number of particles detected and validated", 
                "temperature_sensor": "Temperature in sensor housing", 
                "heating_current": "Sensor head heating current", 
                "voltage": "Power supply voltage", 
                "sensor_status": "Sensor status", 
                "error_code": "Error code", 
                "temperature_PCB": "Temperature in printed circuit board", 
                "temperature_right": "Temperature in right sensor head", 
                "temperature_left": "Temperature in left sensor head", 
                "kinetic_energy": "Kinetic energy", 
                "snowfall_intensity": "Volume equivalent snow depth intensity", 
                "ND": "Particle number concentrations per diameter class", 
                "VD": "Average particle velocities for each diameter class", 
                "N": "Drop counts per diameter and velocity class", 
                }
     return name_dict

def attrs_explanations():
    """
    Get a dictionary containing verbose explanation of the attributes 
    
    Returns
    -------
    explanations : dictionary
        Dictionary with the explanation of the attributes (keys)
    """ 
    # TODO BE UPDATED 
    explanations = {
             'datetime':'Datetime object of the measurement',
             'index':         'Index ranging from 0 to N, where N is the number of observations in the database. For unique identifications better is to use flake_id',
             'flake_id':      'Unique identifier of each measurement. It combines the datetime of measurement with the temporary internal flake number given by the MASC',
             'flake_number_tmp':'Temporary flake number. Incremental, but it resets upon reboot of the instrument. ',
             'pix_size':      'Pixel size',
             'quality_xhi':   'Quality index of the ROI. Very good images above values of 9.  Reference is https://doi.org/10.5194/amt-10-1335-2017 (see Appendix B)',
             'cam_id':        'ID of the CAM: 0, 1 or 2',

             'n_roi'   :      'Number of ROIs initially identified in the raw image of one camera. Note that all the processing downstream is referred to only one (the main) ROI',
             'flake_n_roi'   :'Average value of n_roi (see n_roi definition) over the three cameras ',

             'area'    :      'ROI area. Descriptor 1 of https://doi.org/10.5194/amt-10-1335-2017 (see Appendix A)',
             'perim'   :      'ROI perimeter. Descriptor 2 of https://doi.org/10.5194/amt-10-1335-2017 (see Appendix A)',
             'Dmean'   :      'ROI mean diameter. Mean value of x-width and y-height. Descriptor 3 of https://doi.org/10.5194/amt-10-1335-2017 (see Appendix A)',
             'Dmax'    :      'ROI maximum dimension. Descriptor 4 of https://doi.org/10.5194/amt-10-1335-2017 (see Appendix A)',
             }
    return explanations

def get_OTT_Parsivel_bins():
    diameter_center = np.array([  
        0.06,
        0.19,
        0.32,
        0.45,
        0.58,
        0.71,
        0.84,
        0.96,
        1.09,
        1.22,
        1.42,
        1.67,
        1.93,
        2.19,
        2.45,
        2.83,
        3.35,
        3.86,
        4.38,
        4.89,
        5.66,
        6.7,
        7.72,
        8.76,
        9.78,
        11.33,
        13.39,
        15.45,
        17.51,
        19.57,
        22.15,
        25.24,
    ]
    diameter_width = np.array([
        0.129,
        0.129,
        0.129,
        0.129,
        0.129,
        0.129,
        0.129,
        0.129,
        0.129,
        0.129,
        0.257,
        0.257,
        0.257,
        0.257,
        0.257,
        0.515,
        0.515,
        0.515,
        0.515,
        0.515,
        1.030,
        1.030,
        1.030,
        1.030,
        1.030,
        2.060,
        2.060,
        2.060,
        2.060,
        2.060,
        3.090,
        3.090,
        ])
    velocity_center = np.array([
        0.05,
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
        1.1,
        1.3,
        1.5,
        1.7,
        1.9,
        2.2,
        2.6,
        3,
        3.4,
        3.8,
        4.4,
        5.2,
        6.0,
        6.8,
        7.6,
        8.8,
        10.4,
        12.0,
        13.6,
        15.2,
        17.6,
        20.8,
    ])
    velocity_width = np.array([ 
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        1.6,
        1.6,
        1.6,
        1.6,
        1.6,
        3.2,
        3.2,
    ])

    
    
