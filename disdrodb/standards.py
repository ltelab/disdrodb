#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:26:51 2021

@author: ghiggi
"""
import numpy as np 
import logging
logger = logging.getLogger(__name__)

def get_OTT_Parsivel_diameter_bin_center():
    """
    Returns a 32x1 array containing the center of the diameter bin (in mm)
    of a OTT Parsivel disdrometer. 
    """
    classD = np.zeros((32,1))*np.nan
    classD[0,:] = [0.062]
    classD[1,:] = [0.187]
    classD[2,:] = [0.312]
    classD[3,:] = [0.437]
    classD[4,:] = [0.562]
    classD[5,:] = [0.687]
    classD[6,:] = [0.812]
    classD[7,:] = [0.937]
    classD[8,:] = [1.062]
    classD[9,:] = [1.187]
    classD[10,:] = [1.375]
    classD[11,:] = [1.625]
    classD[12,:] = [1.875]
    classD[13,:] = [2.125]
    classD[14,:] = [2.375]
    classD[15,:] = [2.750]
    classD[16,:] = [3.250]
    classD[17,:] = [3.750]
    classD[18,:] = [4.250]
    classD[19,:] = [4.750]
    classD[20,:] = [5.500]
    classD[21,:] = [6.500]
    classD[22,:] = [7.500]
    classD[23,:] = [8.500]
    classD[24,:] = [9.500]
    classD[25,:] = [11.000]
    classD[26,:] = [13.000]
    classD[27,:] = [15.000]
    classD[28,:] = [17.000]
    classD[29,:] = [19.000]
    classD[30,:] = [21.500]
    classD[31,:] = [24.500]
    return classD.flatten()

def get_OTT_Parsivel2_diameter_bin_center():
    return get_OTT_Parsivel_diameter_bin_center()
    
def get_OTT_Parsivel_diameter_bin_bounds():
    """
    Returns a 32x2 array containing the lower/upper dimater limits (in mm)
    of a OTT Parsivel disdrometer. 
    """
    classD = np.zeros((32,2))*np.nan
    classD[0,:] = [0,0.1245]
    classD[1,:] = [0.1245,0.2495]
    classD[2,:] = [0.2495,0.3745]
    classD[3,:] = [0.3745,0.4995]
    classD[4,:] = [0.4995,0.6245]
    classD[5,:] = [0.6245,0.7495]
    classD[6,:] = [0.7495,0.8745]
    classD[7,:] = [0.8745,0.9995]
    classD[8,:] = [0.9995,1.1245]
    classD[9,:] = [1.1245,1.25]
    classD[10,:] = [1.25,1.50]
    classD[11,:] = [1.50,1.75]
    classD[12,:] = [1.75,2.00]
    classD[13,:] = [2.00,2.25]
    classD[14,:] = [2.25,2.50]
    classD[15,:] = [2.50,3.00]
    classD[16,:] = [3.00,3.50]
    classD[17,:] = [3.50,4.00]
    classD[18,:] = [4.00,4.50]
    classD[19,:] = [4.50,5.00]
    classD[20,:] = [5.00,6.00]
    classD[21,:] = [6.00,7.00]
    classD[22,:] = [7.00,8.00]
    classD[23,:] = [8.00,9.00]
    classD[24,:] = [9.00,10.0]
    classD[25,:] = [10.0,12.0]
    classD[26,:] = [12.0,14.0]
    classD[27,:] = [14.0,16.0]
    classD[28,:] = [16.0,18.0]
    classD[29,:] = [18.0,20.0]
    classD[30,:] = [20.0,23.0]
    classD[31,:] = [23.0,26.0]
    
    return classD 

def get_OTT_Parsivel2_diameter_bin_center():
    return get_OTT_Parsivel_diameter_bin_bounds()

def get_OTT_Parsivel_velocity_bin_center():
    """
    Returns a 32x1 array containing the center of the diameter bin (in m/s)
    of a OTT Parsivel disdrometer. 
    """
    classV = np.zeros((32,1))*np.nan
    classV[0,:] = [0.050]
    classV[1,:] = [0.150]
    classV[2,:] = [0.250]
    classV[3,:] = [0.350]
    classV[4,:] = [0.450]
    classV[5,:] = [0.550]
    classV[6,:] = [0.650]
    classV[7,:] = [0.750]
    classV[8,:] = [0.850]
    classV[9,:] = [0.950]
    classV[10,:] = [1.100]
    classV[11,:] = [1.300]
    classV[12,:] = [1.500]
    classV[13,:] = [1.700]
    classV[14,:] = [1.900]
    classV[15,:] = [2.200]
    classV[16,:] = [2.600]
    classV[17,:] = [3.000]
    classV[18,:] = [3.400]
    classV[19,:] = [3.800]
    classV[20,:] = [4.400]
    classV[21,:] = [5.200]
    classV[22,:] = [6.000]
    classV[23,:] = [6.800]
    classV[24,:] = [7.600]
    classV[25,:] = [8.800]
    classV[26,:] = [10.400]
    classV[27,:] = [12.000]
    classV[28,:] = [13.600]
    classV[29,:] = [15.200]
    classV[30,:] = [17.600]
    classV[31,:] = [20.800]
    return classV.flatten()

def get_OTT_Parsivel2_velocity_bin_center():
    return get_OTT_Parsivel_velocity_bin_center()

def get_OTT_Parsivel_velocity_bin_bounds():
    """
    Returns a 32x2 array containing the lower/upper velocity limits (in m/s)
    of a OTT Parsivel disdrometer. 
    """
    classV = np.zeros((32,2))*np.nan
    classV[0,:] = [0,0.1]
    classV[1,:] = [0.1,0.2]
    classV[2,:] = [0.2,0.3]
    classV[3,:] = [0.3,0.4]
    classV[4,:] = [0.4,0.5]
    classV[5,:] = [0.5,0.6]
    classV[6,:] = [0.6,0.7]
    classV[7,:] = [0.7,0.8]
    classV[8,:] = [0.8,0.9]
    classV[9,:] = [0.9,1.0]
    classV[10,:] = [1.0,1.2]
    classV[11,:] = [1.2,1.4]
    classV[12,:] = [1.4,1.6]
    classV[13,:] = [1.6,1.8]
    classV[14,:] = [1.8,2.0]
    classV[15,:] = [2.0,2.4]
    classV[16,:] = [2.4,2.8]
    classV[17,:] = [2.8,3.2]
    classV[18,:] = [3.2,3.6]
    classV[19,:] = [3.6,4.0]
    classV[20,:] = [4.0,4.8]
    classV[21,:] = [4.8,5.6]
    classV[22,:] = [5.6,6.4]
    classV[23,:] = [6.4,7.2]
    classV[24,:] = [7.2,8.0]
    classV[25,:] = [8.0,9.6]
    classV[26,:] = [9.6,11.2]
    classV[27,:] = [11.2,12.8]
    classV[28,:] = [12.8,14.4]
    classV[29,:] = [14.4,16.0]
    classV[30,:] = [16.0,19.2]
    classV[31,:] = [19.2,22.4]
    
    return classV

# def get_OTT_Parsivel_velocity_bin_bounds(): 
#     return get_OTT_Parsivel_velocity_bin_bounds()

def get_OTT_Parsivel_diameter_bin_width():
    """
    Returns a 32x1 array containing the width of the diameter bin (in mm)
    of a OTT Parsivel disdrometer. 
    """
    classD = np.concatenate((np.ones(5)*0.125,
                             np.ones(5)*0.125,
                             np.ones(5)*0.250,
                             np.ones(5)*0.500,
                             np.ones(5)*1.000,
                             np.ones(5)*2.000,
                             np.ones(2)*3.000))
    return classD

def get_OTT_Parsivel2_diameter_bin_width():
    """
    Returns a 32x1 array containing the width of the diameter bin (in mm)
    of a OTT Parsivel2 disdrometer. 
    """
    classD = np.concatenate((np.ones(5)*0.125,
                             np.ones(5)*0.125,
                             np.ones(5)*0.250,
                             np.ones(5)*0.500,
                             np.ones(5)*1.000,
                             np.ones(5)*2.000,
                             np.ones(2)*3.000))
    return classD

def get_OTT_Parsivel_velocity_bin_width():
    """
    Returns a 32x1 array containing the width of the velocity bin (in m/s)
    of a OTT Parsivel disdrometer. 
    """
    classV = np.concatenate((np.ones(5)*0.100,
                             np.ones(5)*0.100,
                             np.ones(5)*0.200,
                             np.ones(5)*0.400,
                             np.ones(5)*0.800,
                             np.ones(5)*1.600,
                             np.ones(2)*3.200))
    return classV

def get_OTT_Parsivel2_velocity_bin_width():
    """
    Returns a 32x1 array containing the width of the velocity bin (in m/s)
    of a OTT Parsivel2 disdrometer. 
    """
    classV = np.concatenate((np.ones(5)*0.100,
                             np.ones(5)*0.100,
                             np.ones(5)*0.200,
                             np.ones(5)*0.400,
                             np.ones(5)*0.800,
                             np.ones(5)*1.600,
                             np.ones(2)*3.200))
    return classV

#-----------------------------------------------------------------------------.
def get_OTT_Parsivel_dict(): 
    """
    Get a dictionary containing the variable name of OTT Parsivel field numbers.
   
    Returns
    -------
    field_dict : dictionary
        Dictionary with the variable name of OTT Parsivel field numbers.
    """ 
    field_dict = {"01": "rain_rate_32bit", 
                  "02": "rain_accumulated_32bit", 
                  "03": "weather_code_SYNOP_4680", 
                  "04": "weather_code_SYNOP_4677", 
                  "05": "weather_code_METAR_4678", 
                  "06": "weather_code_NWS", 
                  "07": "reflectivity_32bit", 
                  "08": "mor_visibility", 
                  "09": "sample_interval",
                  "10": "laser_amplitude", 
                  "11": "n_particles", 
                  "12": "sensor_temperature", 
                  
                  # "13": "sensor_serial_number", 
                  # "14": "firmware_iop",
                  # "14": "firmware_dsp",
                  
                  "16": "sensor_heating_current", 
                  "17": "sensor_battery_voltage", 
                  "18": "sensor_status",
                  
                  # "19": "start_time",
                  # "20": "sensor_time", 
                  # "21": "sensor_date", 
                  # "22": "station_name",
                  # "23": "station_number",
                  
                  "24": "rain_amount_absolute_32bit", 
                  
                  "25": "error_code", 
                  
                  "30": "rain_rate_16bit", 
                  "31": "rain_rate_12bit", 
                  "32": "rain_accumulated_16bit", 
                  "33": "reflectivity_16bit", 
                   
                  "90": "ND", 
                  "91": "VD", 
                  "93": "N",
                 }
    return field_dict

#--------------------------------------------------------
def get_OTT_Parsivel2_dict(): 
    """
    Get a dictionary containing the variable name of OTT Parsivel2 field numbers.
   
    Returns
    -------
    field_dict : dictionary
        Dictionary with the variable name of OTT Parsivel2 field numbers.
    """ 
    field_dict = {"01": "rain_rate_32bit", 
                  "02": "rain_accumulated_32bit", 
                  "03": "weather_code_SYNOP_4680", 
                  "04": "weather_code_SYNOP_4677", 
                  "05": "weather_code_METAR_4678", 
                  "06": "weather_weather_code_NWS", 
                  "07": "reflectivity_32bit", 
                  "08": "mor_visibility", 
                  "09": "sample_interval",
                  "10": "laser_amplitude", 
                  "11": "n_particles", 
                  "12": "sensor_temperature", 
                  
                  # "13": "sensor_serial_number", 
                  # "14": "firmware_iop",
                  # "14": "firmware_dsp",
                  
                  "16": "sensor_heating_current", 
                  "17": "sensor_battery_voltage", 
                  "18": "sensor_status",
                  
                  # "19": "start_time",
                  # "20": "sensor_time", 
                  # "21": "sensor_date", 
                  # "22": "station_name",
                  # "23": "station_number",
                  
                  "24": "rain_amount_absolute_32bit", 
                  "25": "error_code", 
                  
                  "26": "temperature_PCB",      # only Parsivel 2
                  "27": "temperature_right",    # only Parsivel 2
                  "28": "temperature_left",     # only Parsivel 2
                  
                  "30": "rain_rate_16bit_30",   # change from Parsivel 
                  "31": "rain_rate_16bit_1200", # change from Parsivel 
                  "32": "rain_accumulated_16bit", 
                  "33": "reflectivity_16bit", 
                  
                  "34": "rain_kinetic_energy",  # only Parsivel 2
                  "35": "snowfall_intensity",   # only Parsivel 2
                  
                  "60": "n_particles_all",      # only Parsivel 2
                  "61": "list_particles",       # only Parsivel 2
                  
                  "90": "ND", 
                  "91": "VD", 
                  "93": "N",
                 }
    return field_dict

#-----------------------------------------------------------------------------.
def var_units_dict(): 
    """
    Get a dictionary containing the units of the variables
   
    Returns
    -------
    units : dictionary
        Dictionary with the units of the variables
    """ 
    # TODO BE UPDATED AND EXPANDED  
    units_dict = {"rain_rate_32bit": "mm/h", 
                  "rain_accumulated_32bit": "mm", 
                  "weather_code_SYNOP_4680": "",
                  "weather_code_SYNOP_4677": "", 
                  "weather_code_METAR_4678": "", 
                  "weather_code_NWS": "", 
                  "reflectivity_32bit": "dBZ", 
                  "mor_visibility": "m", 
                  "laser_amplitude": "",
                  "n_particles": "", 
                  "sensor_temperature": "degree celsius", 
                  "sensor_heating_current": "A", 
                  "sensor_battery_voltage": "V", 
                  "sensor_status": "", 
                  "error_code": "", 
                  "temperature_PCB": "degree celsius",
                  "temperature_right": "degree celsius", 
                  "temperature_left": "degree celsius", 
                  "rain_kinetic_energy": "J/(m2*h)", 
                  "snowfall_intensity": "mm/h",  
                  "ND": "1/(m3*mm)",
                  "VD": "m/s", 
                  "N": "", 
                  }
    return units_dict
    
def get_var_explanations():
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
                "rain_accumulated_32bit": "Accumulated rain amount over the measurement interval", 
                "reflectivity_32bit": "Radar reflectivity",
                "mor_visibility": "Meteorological Optical Range in precipitation", 
                "rain_kinetic_energy": "Rain Kinetic energy", 
                "snowfall_intensity": "Volume equivalent snow depth intensity", 
                
                "weather_code_SYNOP_4680": "SYNOP weather code according to table 4680 of Parsivel documentation",
                "weather_code_SYNOP_4677": "SYNOP weather code according to table 4677 of Parsivel documentation",
                "weather_code_METAR_4678": "METAR/SPECI weather code according to table 4678 of Parsivel documentation",
                "weather_code_NWS": "NWS weather code according to Parsivel documentation", 
                
                "laser_amplitude": "Signal amplitude of the laser strip. A way to monitor if windows are dirty or not.",
               
                "temperature_PCB": "Temperature in printed circuit board", 
                "temperature_right": "Temperature in right sensor head", 
                "temperature_left": "Temperature in left sensor head",
                
                "sensor_temperature": "Temperature in sensor housing", 
                "sensor_heating_current": "Sensor head heating current. Optimum heating output of the sensor head heating system can be guaranteed with a power supply voltage > 20 V ", 
                "sensor_battery_voltage": "Power supply voltage. ", 
                "sensor_status": "Sensor status", 
                "error_code": "Error code", 
                                
                "n_particles": "Number of particles detected and validated", 
                "n_particles_all": "Number of all particles detected", 
                    
                "ND": "Particle number concentrations per diameter class", 
                "VD": "Average particle velocities for each diameter class", 
                "N": "Drop counts per diameter and velocity class", 
                }
    return name_dict

def get_attrs_explanations():
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
             # 'index':         'Index ranging from 0 to N, where N is the number of observations in the database. For unique identifications better is to use flake_id',
             # 'flake_id':      'Unique identifier of each measurement. It combines the datetime of measurement with the temporary internal flake number given by the MASC',
             # 'flake_number_tmp':'Temporary flake number. Incremental, but it resets upon reboot of the instrument. ',
             # 'pix_size':      'Pixel size',
             # 'quality_xhi':   'Quality index of the ROI. Very good images above values of 9.  Reference is https://doi.org/10.5194/amt-10-1335-2017 (see Appendix B)',
             # 'cam_id':        'ID of the CAM: 0, 1 or 2',

             # 'n_roi'   :      'Number of ROIs initially identified in the raw image of one camera. Note that all the processing downstream is referred to only one (the main) ROI',
             # 'flake_n_roi'   :'Average value of n_roi (see n_roi definition) over the three cameras ',

             # 'area'    :      'ROI area. Descriptor 1 of https://doi.org/10.5194/amt-10-1335-2017 (see Appendix A)',
             # 'perim'   :      'ROI perimeter. Descriptor 2 of https://doi.org/10.5194/amt-10-1335-2017 (see Appendix A)',
             # 'Dmean'   :      'ROI mean diameter. Mean value of x-width and y-height. Descriptor 3 of https://doi.org/10.5194/amt-10-1335-2017 (see Appendix A)',
             # 'Dmax'    :      'ROI maximum dimension. Descriptor 4 of https://doi.org/10.5194/amt-10-1335-2017 (see Appendix A)',
             }
    return explanations


#-----------------------------------------------------------------------------.
def get_diameter_bin_center(sensor_name): 
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_diameter_bin_center()
    elif sensor_name == "Parsivel2":
        x = get_OTT_Parsivel2_diameter_bin_center()
    elif sensor_name == "ThiesLPM":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
    else:
        logger.exception(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
        raise ValueError(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
    return x

def get_diameter_bin_lower(sensor_name): 
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_diameter_bin_bounds()[:,0]
    elif sensor_name == "Parsivel2":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError 
    elif sensor_name == "ThiesLPM":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
    else:
        logger.exception(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
        raise ValueError(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
    return x

def get_diameter_bin_upper(sensor_name): 
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_diameter_bin_bounds()[:,1]
    elif sensor_name == "Parsivel2":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
    else:
        logger.exception(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
        raise ValueError(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
    return x

def get_diameter_bin_width(sensor_name): 
   if sensor_name == "Parsivel":
       x = get_OTT_Parsivel_diameter_bin_width()
       
   elif sensor_name == "Parsivel2":
       logger.exception(f'Not implemented {sensor_name} device')
       raise NotImplementedError
       
   elif sensor_name == "ThiesLPM":
       logger.exception(f'Not implemented {sensor_name} device')
       raise NotImplementedError
   else:
       logger.exception(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
       raise ValueError(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
   return x
 
def get_velocity_bin_center(sensor_name): 
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_velocity_bin_center()
        
    elif sensor_name == "Parsivel2":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
    else:
        logger.exception(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
        raise ValueError(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
    return x

def get_velocity_bin_lower(sensor_name): 
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_velocity_bin_bounds()[:,0]
        
    elif sensor_name == "Parsivel2":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
    else:
        logger.exception(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
        raise ValueError(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
    return x

def get_velocity_bin_upper(sensor_name): 
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_velocity_bin_bounds()[:,1]
        
    elif sensor_name == "Parsivel2":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
    else:
        logger.exception(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
        raise ValueError(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
    return x

def get_velocity_bin_width(sensor_name): 
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_velocity_bin_width() 
        
    elif sensor_name == "Parsivel2":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
    else:
        logger.exception(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
        raise ValueError(f'L0 bin characteristics for sensor {sensor_name} are not yet defined')
    return x
 
def get_raw_field_nbins(sensor_name): 
    if sensor_name == "Parsivel":
        nbins_dict = {"FieldN": 32,
                      "FieldV": 32,
                      "RawData": 1024,
                     }
    elif sensor_name == "Parsivel2":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
        
    elif sensor_name == "ThiesLPM":
        logger.exception(f'Not implemented {sensor_name} device')
        raise NotImplementedError
    else:
        logger.exception(f'Bin characteristics for sensor {sensor_name} are not yet defined')
        raise ValueError(f'Bin characteristics for sensor {sensor_name} are not yet defined')
    return nbins_dict


def get_var_explanations_ARM():
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
    return dict_ARM_description