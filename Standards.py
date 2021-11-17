#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:03:47 2021

@author: ghiggi
"""
#####################
### Variable Name ###
#####################                                     
rain_rate               
acc_rain_amount   
reflectivity_32bit       
# reflectivity_16bit  
MOR                 
amplitude           
n_particles        
n_all_particles     
temperature_sensor  
heating_current     
voltage             
sensor_status       
error_code          
temperature_PBC     
temperature_right   
temperature_left    
kinetic_energy      
snowfall_intensity  
code_4680              
code_4677           
code_4678            
code_NWS           

# Other options 
Nd / d / drop_diameter_counts / Field_N # ???
Vd / v / drop_velocity_counts / Field_V # ???    
N / n / raw_data  / Field_Raw     # ??? 


diameter_bin_width / diameter_class_width / diameter_width   (left or right)
velocity_bin_width # .... 
diameter_bin_center / diameter_class_center     
velocity_bin_center # ... 
diameter_bin_lower, diameter_bin_upper  # same for velocity? 

# Dimensions 
time / timestep  # start or end? 
# diameter/ velocity ... bin lower, center or upper? which core dimension TODO !!!!

## Coords
lat                latitude                
lon                longitude                 
altitude           altitude                   ?? attrs ? 
crs

### Suggested name changes 
# acc_rain_amount --> rain_acc 
# MOR --> mor_visibility 
# amplitide --> signal amplitude 
# voltage --> supply_voltage / power_voltage
# code_4680 --> SYNOP_4680
# code_4677 --> SYNOP_4677
# code_4678 --> SYNOP_4678
# code_NWS --> NWS

### Datalogger EPFL 
# 'Datalogger_temp': 'object',
# 'Datalogger_power': 'object', 
# 'Datalogger_communication': 'uint8', 
 
### Changes 4 Kimbo   
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
         "site_name": '',
         "sensor_name": '',  # instrument name? 
         "instrument_version": '',
         "project_name": 'https://ruisdael-observatory.nl/',
         "contributors": 'Marc Schleiss, Saverio Guzzo, Rob Mackenzie', 
         "authors": '', # or authors
         "sensor_type": 'OTT Hydromet Parsivel2',
         "sensor_serial_number": '',  
         "firmware_IOP": '',  
         "firmware_DSP": '', 
         # firmware_version
         
         # Suggested 
        'campaign_name': 
        'station_id': 10,
        'station_name': 'Mirabel',
        'location': '',
        'crs': 'WGS84',
        'latitude_unit': 'DegreesNorth',
        'longitude_unit': 'DegreesEast',
        'altitude_unit': 'MetersAboveSeaLevel',
        'country': '', 
        
        'obs_type': '', # raw/preprocessed/postprocessed?
        'level': 'L0', 
        'temporal_resolution': '30 seconds.',
        'reference': 'XXX et al., ... ',
        'documentation': '',
        'website': '', 
        'source_repository': '',
        'doi': '',
        'contact': '',
        'contact_information': 'http://lte.epfl.ch',
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

