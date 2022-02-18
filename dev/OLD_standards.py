#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 09:29:40 2022

@author: ghiggi
"""
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
    classD = np.zeros((32, 1)) * np.nan
    classD[0, :] = [0.062]
    classD[1, :] = [0.187]
    classD[2, :] = [0.312]
    classD[3, :] = [0.437]
    classD[4, :] = [0.562]
    classD[5, :] = [0.687]
    classD[6, :] = [0.812]
    classD[7, :] = [0.937]
    classD[8, :] = [1.062]
    classD[9, :] = [1.187]
    classD[10, :] = [1.375]
    classD[11, :] = [1.625]
    classD[12, :] = [1.875]
    classD[13, :] = [2.125]
    classD[14, :] = [2.375]
    classD[15, :] = [2.750]
    classD[16, :] = [3.250]
    classD[17, :] = [3.750]
    classD[18, :] = [4.250]
    classD[19, :] = [4.750]
    classD[20, :] = [5.500]
    classD[21, :] = [6.500]
    classD[22, :] = [7.500]
    classD[23, :] = [8.500]
    classD[24, :] = [9.500]
    classD[25, :] = [11.000]
    classD[26, :] = [13.000]
    classD[27, :] = [15.000]
    classD[28, :] = [17.000]
    classD[29, :] = [19.000]
    classD[30, :] = [21.500]
    classD[31, :] = [24.500]
    return classD.flatten()


def get_OTT_Parsivel2_diameter_bin_center():
    return get_OTT_Parsivel_diameter_bin_center()


def get_ThiesLPM_diameter_bin_center():
    """
    Returns a 22x1 array containing the center of the diameter bin (in mm)
    of a Thies laser disdrometer.
    """
    classD = np.zeros((22, 1)) * np.nan
    classD[0, :] = [0.125]
    classD[1, :] = [0.250]
    classD[2, :] = [0.375]
    classD[3, :] = [0.500]
    classD[4, :] = [0.750]
    classD[5, :] = [1.000]
    classD[6, :] = [1.250]
    classD[7, :] = [1.500]
    classD[8, :] = [1.750]
    classD[9, :] = [2.000]
    classD[10, :] = [2.500]
    classD[11, :] = [3.000]
    classD[12, :] = [3.500]
    classD[13, :] = [4.000]
    classD[14, :] = [4.500]
    classD[15, :] = [5.000]
    classD[16, :] = [5.500]
    classD[17, :] = [6.000]
    classD[18, :] = [6.500]
    classD[19, :] = [7.000]
    classD[20, :] = [7.500]
    classD[21, :] = [8.000]
    return classD.flatten()


def get_OTT_Parsivel_diameter_bin_bounds():
    """
    Returns a 32x2 array containing the lower/upper dimater limits (in mm)
    of a OTT Parsivel disdrometer.
    """
    classD = np.zeros((32, 2)) * np.nan
    classD[0, :] = [0, 0.1245]
    classD[1, :] = [0.1245, 0.2495]
    classD[2, :] = [0.2495, 0.3745]
    classD[3, :] = [0.3745, 0.4995]
    classD[4, :] = [0.4995, 0.6245]
    classD[5, :] = [0.6245, 0.7495]
    classD[6, :] = [0.7495, 0.8745]
    classD[7, :] = [0.8745, 0.9995]
    classD[8, :] = [0.9995, 1.1245]
    classD[9, :] = [1.1245, 1.25]
    classD[10, :] = [1.25, 1.50]
    classD[11, :] = [1.50, 1.75]
    classD[12, :] = [1.75, 2.00]
    classD[13, :] = [2.00, 2.25]
    classD[14, :] = [2.25, 2.50]
    classD[15, :] = [2.50, 3.00]
    classD[16, :] = [3.00, 3.50]
    classD[17, :] = [3.50, 4.00]
    classD[18, :] = [4.00, 4.50]
    classD[19, :] = [4.50, 5.00]
    classD[20, :] = [5.00, 6.00]
    classD[21, :] = [6.00, 7.00]
    classD[22, :] = [7.00, 8.00]
    classD[23, :] = [8.00, 9.00]
    classD[24, :] = [9.00, 10.0]
    classD[25, :] = [10.0, 12.0]
    classD[26, :] = [12.0, 14.0]
    classD[27, :] = [14.0, 16.0]
    classD[28, :] = [16.0, 18.0]
    classD[29, :] = [18.0, 20.0]
    classD[30, :] = [20.0, 23.0]
    classD[31, :] = [23.0, 26.0]

    return classD


def get_OTT_Parsivel2_diameter_bin_center():
    return get_OTT_Parsivel_diameter_bin_bounds()


def get_ThiesLPM_diameter_bin_bounds():
    """
    Returns a 22x2 array containing the lower/upper dimater limits (in mm)
    of a Thies laser disdrometer.
    """
    classD = np.zeros((22, 2)) * np.nan
    classD[0, :] = [0, 0.125]
    classD[1, :] = [0.125, 0.250]
    classD[2, :] = [0.250, 0.375]
    classD[3, :] = [0.375, 0.500]
    classD[4, :] = [0.500, 0.750]
    classD[5, :] = [0.750, 1.000]
    classD[6, :] = [1.000, 1.250]
    classD[7, :] = [1.250, 1.500]
    classD[8, :] = [1.500, 1.750]
    classD[9, :] = [1.750, 2.000]
    classD[10, :] = [2.000, 2.500]
    classD[11, :] = [2.500, 3.000]
    classD[12, :] = [3.000, 3.500]
    classD[13, :] = [3.500, 4.000]
    classD[14, :] = [4.000, 4.500]
    classD[15, :] = [4.500, 5.000]
    classD[16, :] = [5.000, 5.500]
    classD[17, :] = [5.500, 6.000]
    classD[18, :] = [6.000, 6.500]
    classD[19, :] = [6.500, 7.000]
    classD[20, :] = [7.000, 7.500]
    classD[21, :] = [8.00, 999]  # To infinite by the documentation

    return classD


def get_OTT_Parsivel_velocity_bin_center():
    """
    Returns a 32x1 array containing the center of the diameter bin (in m/s)
    of a OTT Parsivel disdrometer.
    """
    classV = np.zeros((32, 1)) * np.nan
    classV[0, :] = [0.050]
    classV[1, :] = [0.150]
    classV[2, :] = [0.250]
    classV[3, :] = [0.350]
    classV[4, :] = [0.450]
    classV[5, :] = [0.550]
    classV[6, :] = [0.650]
    classV[7, :] = [0.750]
    classV[8, :] = [0.850]
    classV[9, :] = [0.950]
    classV[10, :] = [1.100]
    classV[11, :] = [1.300]
    classV[12, :] = [1.500]
    classV[13, :] = [1.700]
    classV[14, :] = [1.900]
    classV[15, :] = [2.200]
    classV[16, :] = [2.600]
    classV[17, :] = [3.000]
    classV[18, :] = [3.400]
    classV[19, :] = [3.800]
    classV[20, :] = [4.400]
    classV[21, :] = [5.200]
    classV[22, :] = [6.000]
    classV[23, :] = [6.800]
    classV[24, :] = [7.600]
    classV[25, :] = [8.800]
    classV[26, :] = [10.400]
    classV[27, :] = [12.000]
    classV[28, :] = [13.600]
    classV[29, :] = [15.200]
    classV[30, :] = [17.600]
    classV[31, :] = [20.800]
    return classV.flatten()


def get_OTT_Parsivel2_velocity_bin_center():
    return get_OTT_Parsivel_velocity_bin_center()


def get_ThiesLPM_velocity_bin_center():
    """
    Returns a 20x1 array containing the center of the diameter bin (in m/s)
    of a Thies laser disdrometer.
    """
    classV = np.zeros((20, 1)) * np.nan
    classV[0, :] = [0.100]
    classV[1, :] = [0.300]
    classV[2, :] = [0.500]
    classV[3, :] = [0.700]
    classV[4, :] = [0.900]
    classV[5, :] = [1.200]
    classV[6, :] = [1.600]
    classV[7, :] = [2.000]
    classV[8, :] = [2.400]
    classV[9, :] = [2.800]
    classV[10, :] = [3.200]
    classV[11, :] = [3.800]
    classV[12, :] = [4.600]
    classV[13, :] = [5.400]
    classV[14, :] = [6.200]
    classV[15, :] = [7.000]
    classV[16, :] = [7.800]
    classV[17, :] = [8.600]
    classV[18, :] = [9.500]
    classV[19, :] = [10.000]  # I'm not sure about this
    return classV.flatten()


def get_OTT_Parsivel_velocity_bin_bounds():
    """
    Returns a 32x2 array containing the lower/upper velocity limits (in m/s)
    of a OTT Parsivel disdrometer.
    """
    classV = np.zeros((32, 2)) * np.nan
    classV[0, :] = [0, 0.1]
    classV[1, :] = [0.1, 0.2]
    classV[2, :] = [0.2, 0.3]
    classV[3, :] = [0.3, 0.4]
    classV[4, :] = [0.4, 0.5]
    classV[5, :] = [0.5, 0.6]
    classV[6, :] = [0.6, 0.7]
    classV[7, :] = [0.7, 0.8]
    classV[8, :] = [0.8, 0.9]
    classV[9, :] = [0.9, 1.0]
    classV[10, :] = [1.0, 1.2]
    classV[11, :] = [1.2, 1.4]
    classV[12, :] = [1.4, 1.6]
    classV[13, :] = [1.6, 1.8]
    classV[14, :] = [1.8, 2.0]
    classV[15, :] = [2.0, 2.4]
    classV[16, :] = [2.4, 2.8]
    classV[17, :] = [2.8, 3.2]
    classV[18, :] = [3.2, 3.6]
    classV[19, :] = [3.6, 4.0]
    classV[20, :] = [4.0, 4.8]
    classV[21, :] = [4.8, 5.6]
    classV[22, :] = [5.6, 6.4]
    classV[23, :] = [6.4, 7.2]
    classV[24, :] = [7.2, 8.0]
    classV[25, :] = [8.0, 9.6]
    classV[26, :] = [9.6, 11.2]
    classV[27, :] = [11.2, 12.8]
    classV[28, :] = [12.8, 14.4]
    classV[29, :] = [14.4, 16.0]
    classV[30, :] = [16.0, 19.2]
    classV[31, :] = [19.2, 22.4]

    return classV


# def get_OTT_Parsivel_velocity_bin_bounds():
#     return get_OTT_Parsivel_velocity_bin_bounds()


def get_ThiesLPM_velocity_bin_bounds():
    """
    Returns a 20x2 array containing the lower/upper velocity limits (in m/s)
    of a Thies laser disdrometer.
    """
    classV = np.zeros((20, 2)) * np.nan
    classV[0, :] = [0, 0.200]
    classV[1, :] = [0.200, 0.400]
    classV[2, :] = [0.400, 0.600]
    classV[3, :] = [0.600, 0.800]
    classV[4, :] = [0.800, 1.000]
    classV[5, :] = [1.000, 1.400]
    classV[6, :] = [1.400, 1.800]
    classV[7, :] = [1.800, 2.200]
    classV[8, :] = [2.200, 2.600]
    classV[9, :] = [2.600, 3.000]
    classV[10, :] = [3.000, 3.400]
    classV[11, :] = [3.400, 4.200]
    classV[12, :] = [4.200, 5.000]
    classV[13, :] = [5.000, 5.800]
    classV[14, :] = [5.800, 6.600]
    classV[15, :] = [6.600, 7.400]
    classV[16, :] = [7.400, 8.200]
    classV[17, :] = [8.200, 9.000]
    classV[18, :] = [9.000, 10.000]
    classV[19, :] = [10.000, 9999]

    return classV


def get_OTT_Parsivel_diameter_bin_width():
    """
    Returns a 32x1 array containing the width of the diameter bin (in mm)
    of a OTT Parsivel disdrometer.
    """
    classD = np.concatenate(
        (
            np.ones(5) * 0.125,
            np.ones(5) * 0.125,
            np.ones(5) * 0.250,
            np.ones(5) * 0.500,
            np.ones(5) * 1.000,
            np.ones(5) * 2.000,
            np.ones(2) * 3.000,
        )
    )
    return classD


def get_OTT_Parsivel2_diameter_bin_width():
    """
    Returns a 32x1 array containing the width of the diameter bin (in mm)
    of a OTT Parsivel2 disdrometer.
    """
    classD = np.concatenate(
        (
            np.ones(5) * 0.125,
            np.ones(5) * 0.125,
            np.ones(5) * 0.250,
            np.ones(5) * 0.500,
            np.ones(5) * 1.000,
            np.ones(5) * 2.000,
            np.ones(2) * 3.000,
        )
    )
    return classD


def get_ThiesLPM_diameter_bin_width():
    """
    Returns a 22x1 array containing the width of the diameter bin (in mm)
    of a Thies laser disdrometer.
    """
    classD = np.concatenate(
        (
            np.ones(3) * 0.125,
            np.ones(6) * 0.250,
            np.ones(12) * 0.500,
            # 0.500 to Infinite by the documentation
        )
    )
    return classD


def get_OTT_Parsivel_velocity_bin_width():
    """
    Returns a 32x1 array containing the width of the velocity bin (in m/s)
    of a OTT Parsivel disdrometer.
    """
    classV = np.concatenate(
        (
            np.ones(5) * 0.100,
            np.ones(5) * 0.100,
            np.ones(5) * 0.200,
            np.ones(5) * 0.400,
            np.ones(5) * 0.800,
            np.ones(5) * 1.600,
            np.ones(2) * 3.200,
        )
    )
    return classV


def get_OTT_Parsivel2_velocity_bin_width():
    """
    Returns a 32x1 array containing the width of the velocity bin (in m/s)
    of a OTT Parsivel2 disdrometer.
    """
    classV = np.concatenate(
        (
            np.ones(5) * 0.100,
            np.ones(5) * 0.100,
            np.ones(5) * 0.200,
            np.ones(5) * 0.400,
            np.ones(5) * 0.800,
            np.ones(5) * 1.600,
            np.ones(2) * 3.200,
        )
    )
    return classV


def get_ThiesLPM_velocity_bin_width():
    """
    Returns a 20x1 array containing the width of the velocity bin (in m/s)
    of a Thies laser disdrometer.
    """
    classV = np.concatenate(
        (
            np.ones(5) * 0.200,
            np.ones(6) * 0.400,
            np.ones(7) * 0.800,
            np.ones(1) * 1.000,
            np.ones(1) * 10.000,
        )
    )
    return classV


# -----------------------------------------------------------------------------.
def get_OTT_Parsivel_dict():
    """
    Get a dictionary containing the variable name of OTT Parsivel field numbers.

    Returns
    -------
    field_dict : dictionary
        Dictionary with the variable name of OTT Parsivel field numbers.
    """
    field_dict = {
        "01": "rain_rate_32bit",
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


# --------------------------------------------------------
def get_OTT_Parsivel2_dict():
    """
    Get a dictionary containing the variable name of OTT Parsivel2 field numbers.

    Returns
    -------
    field_dict : dictionary
        Dictionary with the variable name of OTT Parsivel2 field numbers.
    """
    field_dict = {
        "01": "rain_rate_32bit",
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
        "26": "temperature_PCB",  # only Parsivel 2
        "27": "temperature_right",  # only Parsivel 2
        "28": "temperature_left",  # only Parsivel 2
        "30": "rain_rate_16bit_30",  # change from Parsivel
        "31": "rain_rate_16bit_1200",  # change from Parsivel
        "32": "rain_accumulated_16bit",
        "33": "reflectivity_16bit",
        "34": "rain_kinetic_energy",  # only Parsivel 2
        "35": "snowfall_intensity",  # only Parsivel 2
        "60": "n_particles_all",  # only Parsivel 2
        "61": "list_particles",  # only Parsivel 2
        "90": "ND",
        "91": "VD",
        "93": "N",
    }
    return field_dict


# -----------------------------------------------------------------------------.


def get_ThiesLPM_dict():
    """
    et a dictionary containing the variable name of Thies.

    Returns
    -------
    field_dict : dictionary
        Dictionary with the variable name of Thies field numbers.
    """

    field_dict = {
<<<<<<< HEAD
        "01": "STX (start identifier)",  # To delete?
        "02": "Device address (factory setting „00“) (NN)",  # To delete?
        "03": "sensor_serial_number",
        "04": "firmware_DSP",
        "05": "Date of the sensor (tt.mm.jj)",  # Merge Date of the sensor and Time of the sensor?
        "06": "Time of the sensor (on request) (hh:mm:ss)",  # Merge Date of the sensor and Time of the sensor?
        "07": "5M SYNOP Tab.4677 (5 minutes mean value) (NN)",  # To delete?
        "08": "5M SYNOP Tab.4680 (5 minutes mean value) (NN)",  # To delete?
        "09": "5M METAR Tab.4678 (5 minutes mean value) (AAAAA)",  # To delete?
        "10": "5M Intensität [mm/h] (5 minutes mean value) (NNN.NNN)",  # To delete?
        "11": "weather_code_SYNOP_4677",
        "12": "weather_code_SYNOP_4680",
        "13": "weather_code_METAR_4678",
        "14": "1M Intensity [mm/h] total precipitation (1 minute value) (NNN.NNN)",  # rain_amount_absolute_32bit ?
        "15": "1M Intensity [mm/h] liquid precipitation (1 minute value) (NNN.NNN)",  # rain_rate_32bit or rain_rate_16bit ?
        "16": "1M Intensity [mm/h] solid precipitation (1 minute value) (NNN.NNN)",  # snowfall_intensity ?
        "17": "Precipitation amount [mm] (Reset with command „RA“) (NNNN.NN)",  # rain_accumulated_32bit or rain_accumulated_16bit ?
        "18": "mor_visibility",
        "19": "reflectivity_16bit",
        "20": "1M Measuring quality [0...100%] (1 minute value) (NNN)",  # To delete?
        "21": "1M Maximum diameter hail [mm] (1 minute value) (N.N))",  # To delete?
        "22": "sensor_status",
        "23": "Static signal (OK:0, Error:1)",  # To delete?
        "24": "Status Laser temperature (analogue) (OK:0, Error:1)",  # To delete?
        "25": "Status Laser temperature (digital) (OK:0, Error:1)",  # To delete?
        "26": "Status Laser current (analogue) (OK:0, Error:1)",  # To delete?
        "27": "Status Laser current (digital) (OK:0, Error:1)",  # To delete?
        "28": "Status Sensor supply (OK:0, Error:1)",  # To delete?
        "29": "Status Current pane heating laser head (OK:0, warning:1)",  # To delete?
        "30": "Status Current pane heating receiver head (OK:0, warning:1)",  # To delete?
        "31": "Status Temperature sensor (OK:0, warning:1)",  # To delete?
        "32": "Status Heating supply (OK:0, warning:1)",  # To delete?
        "33": "Status Current heating housing (OK:0, warning:1)",  # To delete?
        "34": "Status Current heating heads (OK:0, warning:1)",  # To delete?
        "35": "Status Current heating carriers (OK:0, warning:1)",  # To delete?
        "36": "Status Control output laser power (OK:0, warning:1)",  # To delete?
        "37": "Reserve Status ( 0)",  # To delete?
        "38": "sensor_temperature_PBC",
        "39": "Temperature of laser driver 0-80°C (NN)",  # To delete?
        "40": "Mean value laser current [1/100 mA] (NNNN)",  # sensor_heating_current ?
        "41": "Control voltage [mV] (reference value: 4010±5) (NNNN)",  # To delete?
        "42": "Optical control output [mV] (2300 … 6500) (NNNN)",  # To delete?
        "43": "Voltage sensor supply [1/10V] (NNN)",  # sensor_battery_voltage ?
        "44": "Current pane heating laser head [mA] (NNN)",  # To delete?
        "45": "Current pane heating receiver head [mA] (NNN)",  # To delete?
        "46": "sensor_temperature",
        "47": "Voltage Heating supply [1/10 V] (only 5.4110.x1.xxx, otherwise “999”) (NNN)",  # To delete?
        "48": "Current heating housing [mA] (only 5.4110.x1.xxx, otherwise “9999”) (NNNN)",  # To delete?
        "49": "Current heating heads [mA] (only 5.4110.x1.xxx, otherwise “9999”) (NNNN)",  # To delete?
        "50": "Current heating carriers [mA] (only 5.4110.x1.xxx, otherwise “9999”) (NNNN)",  # To delete?
        "51": "n_particles",
        "52": "„00000.000“ (internal data)",  # To delete?
        "53": "Number of particles < minimal speed (0.15m/s) (NNNNN)",  # To delete?
        "54": "„00000.000“ (internal data)",  # To delete?
        "55": "Number of particles > maximal speed (20m/s) (NNNNN)",  # To delete?
        "56": "„00000.000“ (internal data)",  # To delete?
        "57": "Number of particles < minimal diameter (0.15mm) (NNNNN)",  # To delete?
        "58": "„00000.000“ (internal data)",  # To delete?
        "59": "Number of particles no hydrometeor",  # To delete?
        "60": "Total volume (gross) of this class",  # To delete?
        "61": "Number of particles with unknown classification",  # To delete?
        "62": "Total volume (gross) of this class",  # To delete?
        "63": "Number of particles class 1",  # To delete?
        "64": "Total volume (gross) of class 1",  # To delete?
        "65": "Number of particles class 2",  # To delete?
        "66": "Total volume (gross) of class 2",  # To delete?
        "67": "Number of particles class 3",  # To delete?
        "68": "Total volume (gross) of class 3",  # To delete?
        "69": "Number of particles class 4",  # To delete?
        "70": "Total volume (gross) of class 4",  # To delete?
        "71": "Number of particles class 5",  # To delete?
        "72": "Total volume (gross) of class 5",  # To delete?
        "73": "Number of particles class 6",  # To delete?
        "74": "Total volume (gross) of class 6",  # To delete?
        "75": "Number of particles class 7",  # To delete?
        "76": "Total volume (gross) of class 7",  # To delete?
        "77": "Number of particles class 8",  # To delete?
        "78": "Total volume (gross) of class 8",  # To delete?
        "79": "Number of particles class 9",  # To delete?
        "80": "Total volume (gross) of class 9",  # To delete?
        "81": "Precipitation spectrum",  # To delete?
        "520": "Diameter and speed (NNN)",  # To delete?
        "521": "Checksum (AA)",  # To delete?
        "522": "CRLF",  # To delete?
        "523": "ETX (End identifier)",  # To delete?
    }

    # By the Thies documentation, there are 2 version, if the id is like “<id>TM00005”, then there are more fields
    field_dict_id_TM00005 = {
        "01": "STX (start identifier)",  # To delete?
        "02": "Device address (factory setting „00“) (NN)",  # To delete?
        "03": "sensor_serial_number",
        "04": "firmware_DSP",
        "05": "Date of the sensor (tt.mm.jj)",  # Merge Date of the sensor and Time of the sensor?
        "06": "Time of the sensor (on request) (hh:mm:ss)",  # Merge Date of the sensor and Time of the sensor?
        "07": "5M SYNOP Tab.4677 (5 minutes mean value) (NN)",  # To delete?
        "08": "5M SYNOP Tab.4680 (5 minutes mean value) (NN)",  # To delete?
        "09": "5M METAR Tab.4678 (5 minutes mean value) (AAAAA)",  # To delete?
        "10": "5M Intensität [mm/h] (5 minutes mean value) (NNN.NNN)",  # To delete?
        "11": "weather_code_SYNOP_4677",
        "12": "weather_code_SYNOP_4680",
        "13": "weather_code_METAR_4678",
        "14": "1M Intensity [mm/h] total precipitation (1 minute value) (NNN.NNN)",  # rain_amount_absolute_32bit ?
        "15": "1M Intensity [mm/h] liquid precipitation (1 minute value) (NNN.NNN)",  # rain_rate_32bit or rain_rate_16bit ?
        "16": "1M Intensity [mm/h] solid precipitation (1 minute value) (NNN.NNN)",  # snowfall_intensity ?
        "17": "Precipitation amount [mm] (Reset with command „RA“) (NNNN.NN)",  # rain_accumulated_32bit or rain_accumulated_16bit ?
        "18": "mor_visibility",
        "19": "reflectivity_16bit",
        "20": "1M Measuring quality [0...100%] (1 minute value) (NNN)",  # To delete?
        "21": "1M Maximum diameter hail [mm] (1 minute value) (N.N))",  # To delete?
        "22": "sensor_status",
        "23": "Static signal (OK:0, Error:1)",  # To delete?
        "24": "Status Laser temperature (analogue) (OK:0, Error:1)",  # To delete?
        "25": "Status Laser temperature (digital) (OK:0, Error:1)",  # To delete?
        "26": "Status Laser current (analogue) (OK:0, Error:1)",  # To delete?
        "27": "Status Laser current (digital) (OK:0, Error:1)",  # To delete?
        "28": "Status Sensor supply (OK:0, Error:1)",  # To delete?
        "29": "Status Current pane heating laser head (OK:0, warning:1)",  # To delete?
        "30": "Status Current pane heating receiver head (OK:0, warning:1)",  # To delete?
        "31": "Status Temperature sensor (OK:0, warning:1)",  # To delete?
        "32": "Status Heating supply (OK:0, warning:1)",  # To delete?
        "33": "Status Current heating housing (OK:0, warning:1)",  # To delete?
        "34": "Status Current heating heads (OK:0, warning:1)",  # To delete?
        "35": "Status Current heating carriers (OK:0, warning:1)",  # To delete?
        "36": "Status Control output laser power (OK:0, warning:1)",  # To delete?
        "37": "Reserve Status ( 0)",  # To delete?
        "38": "sensor_temperature_PBC",
        "39": "Temperature of laser driver 0-80°C (NN)",  # To delete?
        "40": "Mean value laser current [1/100 mA] (NNNN)",  # sensor_heating_current ?
        "41": "Control voltage [mV] (reference value: 4010±5) (NNNN)",  # To delete?
        "42": "Optical control output [mV] (2300 … 6500) (NNNN)",  # To delete?
        "43": "Voltage sensor supply [1/10V] (NNN)",  # sensor_battery_voltage ?
        "44": "Current pane heating laser head [mA] (NNN)",  # To delete?
        "45": "Current pane heating receiver head [mA] (NNN)",  # To delete?
        "46": "sensor_temperature",
        "47": "Voltage Heating supply [1/10 V] (only 5.4110.x1.xxx, otherwise “999”) (NNN)",  # To delete?
        "48": "Current heating housing [mA] (only 5.4110.x1.xxx, otherwise “9999”) (NNNN)",  # To delete?
        "49": "Current heating heads [mA] (only 5.4110.x1.xxx, otherwise “9999”) (NNNN)",  # To delete?
        "50": "Current heating carriers [mA] (only 5.4110.x1.xxx, otherwise “9999”) (NNNN)",  # To delete?
        "51": "n_particles",
        "52": "„00000.000“ (internal data)",  # To delete?
        "53": "Number of particles < minimal speed (0.15m/s) (NNNNN)",  # To delete?
        "54": "„00000.000“ (internal data)",  # To delete?
        "55": "Number of particles > maximal speed (20m/s) (NNNNN)",  # To delete?
        "56": "„00000.000“ (internal data)",  # To delete?
        "57": "Number of particles < minimal diameter (0.15mm) (NNNNN)",  # To delete?
        "58": "„00000.000“ (internal data)",  # To delete?
        "59": "Number of particles no hydrometeor",  # To delete?
        "60": "Total volume (gross) of this class",  # To delete?
        "61": "Number of particles with unknown classification",  # To delete?
        "62": "Total volume (gross) of this class",  # To delete?
        "63": "Number of particles class 1",  # To delete?
        "64": "Total volume (gross) of class 1",  # To delete?
        "65": "Number of particles class 2",  # To delete?
        "66": "Total volume (gross) of class 2",  # To delete?
        "67": "Number of particles class 3",  # To delete?
        "68": "Total volume (gross) of class 3",  # To delete?
        "69": "Number of particles class 4",  # To delete?
        "70": "Total volume (gross) of class 4",  # To delete?
        "71": "Number of particles class 5",  # To delete?
        "72": "Total volume (gross) of class 5",  # To delete?
        "73": "Number of particles class 6",  # To delete?
        "74": "Total volume (gross) of class 6",  # To delete?
        "75": "Number of particles class 7",  # To delete?
        "76": "Total volume (gross) of class 7",  # To delete?
        "77": "Number of particles class 8",  # To delete?
        "78": "Total volume (gross) of class 8",  # To delete?
        "79": "Number of particles class 9",  # To delete?
        "80": "Total volume (gross) of class 9",  # To delete?
        "81": "Precipitation spectrum",  # To delete?
        "520": "Diameter and speed (NNN)",  # To delete?
        "521": "Temperature [°C] (NNN.N)",  # sensor_temperature ?
        "522": "Relative Air humidity [%r.h.] (NNN.N)",  # To delete?
        "523": "Wind speed [m/s] (NN.N)",  # To delete?
        "524": "Wind direction [°] (NNN)",  # To delete?
        "525": "Checksum (AA)",  # To delete?
        "526": "CRLF",  # To delete?
        "527": "ETX (End identifier)",  # To delete?
    }

=======
                    '01': 'start_identifier',
                    '02': 'device_address',
                    '03': 'sensor_serial_number',
                    '04': 'firmware_DSP',
                    '05': 'date_sensor',
                    '06': 'time_sensor',
                    '07': 'synop_4677_5min_weather_code',
                    '08': 'synop_4680_5min_weather_code',
                    '09': 'metar_4678_5min_weather_code',
                    '10': 'intensity_total_5min',
                    '11': 'synop_4677_weather_code',
                    '12': 'synop_4680_weather_code',
                    '13': 'metar_4678_weather_code',
                    '14': 'intensity_total',
                    '15': 'intensity_liquid',
                    '16': 'intensity_solid',
                    '17': 'accum_precip',
                    '18': 'maximum_visibility',
                    '19': 'radar_reflectivity',
                    '20': 'quality_measurement',
                    '21': 'max_diameter_hail', 
                    '22': 'laser_status',
                    '23': 'static_signal',
                    '24': 'laser_temperature_analog_status',
                    '25': 'laser_temperature_digital_status',
                    '26': 'laser_current_analog_status',
                    '27': 'laser_current_digital_status',
                    '28': 'sensor_status',
                    '29': 'pane_heating_laser_head_current', 
                    '30': 'pane_heating_receiver_head_current', 
                    '31': 'temperature_sensor_status',
                    '32': 'status_heating',
                    '33': 'heating_house_current_status',
                    '34': 'heating_heads_current_status',
                    '35': 'heating_carriers_current_status',
                    '36': 'control_output_laser_power_status',
                    '37': 'reserve_status',
                    '38': 'interior_temperature',
                    '39': 'laser_temperature',
                    '40': 'mean_laser_current',
                    '41': 'control_voltage',
                    '42': 'optical_control_output', 
                    '43': 'voltage_sensor_supply',
                    '44': 'pane_heating_laser_head_current',
                    '45': 'pane_heating_receiver_head_current',
                    '46': 'ambient_temperature',
                    '47': 'voltage_heating_supply',
                    '48': 'heating_house_current',
                    '49': 'heating_heads_current',
                    '50': 'heating_carriers_current',
                    '51': 'number_particles',
                    '52': 'number_particles_internal_data',
                    '53': 'number_particles_min_speed',
                    '54': 'number_particles_min_speed_internal_data',
                    '55': 'number_particles_max_speed',
                    '56': 'number_particles_max_speed_internal_data',
                    '57': 'number_particles_min_diameter',
                    '58': 'number_particles_min_diameter_internal_data',
                    '59': 'number_particles_no_hydrometeor',
                    '60': 'number_particles_no_hydrometeor_internal_data',
                    '61': 'number_particles_no_classification',
                    '62': 'number_particles_no_classification_internal_data',
                    '63': 'number_particles_class_1',
                    '64': 'number_particles_class_1_internal_data',
                    '65': 'number_particles_class_2',
                    '66': 'number_particles_class_2_internal_data',
                    '67': 'number_particles_class_3',
                    '68': 'number_particles_class_3_internal_data',
                    '69': 'number_particles_class_4',
                    '70': 'number_particles_class_4_internal_data',
                    '71': 'number_particles_class_5',
                    '72': 'number_particles_class_5_internal_data',
                    '73': 'number_particles_class_6',
                    '74': 'number_particles_class_6_internal_data',
                    '75': 'number_particles_class_7',
                    '76': 'number_particles_class_7_internal_data',
                    '77': 'number_particles_class_8',
                    '78': 'number_particles_class_8_internal_data',
                    '79': 'number_particles_class_9',
                    '80': 'number_particles_class_9_internal_data',
                    '81': 'precipitation_spectrum',
                    '520': 'diameter_speed',
                    '521': 'checksum',
                    '522': 'crlf',
                    '523': 'etx',
        }

    
    
 
    return field_dict


 
# -----------------------------------------------------------------------------.
# def get_ThiesLPM_dict_full():
#     """
#     Get a dictionary containing the variable name and information by the Thies documentation.
 

#-----------------------------------------------------------------------------.

# def get_ThiesLPM_dict_full():
#     """
#     Get a dictionary containing the variable name and information by the Thies documentation.
   
#     Returns
#     -------'base_time', 'time_offset', 'time_bounds', 'particle_diameter_bounds', 'particle_fall_velocity_bounds', 'synop_46
#     field_dict : dictionary
#         Dictionary with all the information about the variables for thies.
#     """ 
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
    
#     # By the Thies documentation, there are 2 version, if the id is like “<id>TM00005”, then there are more fields
#     field_dict_id_TM00005 = {
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
#                     {'No': '521', 'Column': '2207-2211', 'Len': '5', 'Description': 'Temperature [°C] (NNN.N)'},
#                     {'No': '522', 'Column': '2213-2217', 'Len': '5', 'Description': 'Relative Air humidity [%r.h.] (NNN.N)'},
#                     {'No': '523', 'Column': '2219-2222', 'Len': '4', 'Description': 'Wind speed [m/s] (NN.N)'},
#                     {'No': '524', 'Column': '2224-2226', 'Len': '3', 'Description': 'Wind direction [°] (NNN)'},
#                     {'No': '525', 'Column': '2228-2229', 'Len': '2', 'Description': 'Checksum (AA)'},
#                     {'No': '526', 'Column': '2231-2232', 'Len': '2', 'Description': 'CRLF'},
#                     {'No': '527', 'Column': '2233', 'Len': '1', 'Description': 'ETX (End identifier)'},
#         }
    
#     return field_dict
 

#     Returns
#     -------
#     field_dict : dictionary
#         Dictionary with all the information about the variables for thies.
#     """
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

#     # By the Thies documentation, there are 2 version, if the id is like “<id>TM00005”, then there are more fields
#     field_dict_id_TM00005 = {
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
#                     {'No': '521', 'Column': '2207-2211', 'Len': '5', 'Description': 'Temperature [°C] (NNN.N)'},
#                     {'No': '522', 'Column': '2213-2217', 'Len': '5', 'Description': 'Relative Air humidity [%r.h.] (NNN.N)'},
#                     {'No': '523', 'Column': '2219-2222', 'Len': '4', 'Description': 'Wind speed [m/s] (NN.N)'},
#                     {'No': '524', 'Column': '2224-2226', 'Len': '3', 'Description': 'Wind direction [°] (NNN)'},
#                     {'No': '525', 'Column': '2228-2229', 'Len': '2', 'Description': 'Checksum (AA)'},
#                     {'No': '526', 'Column': '2231-2232', 'Len': '2', 'Description': 'CRLF'},
#                     {'No': '527', 'Column': '2233', 'Len': '1', 'Description': 'ETX (End identifier)'},
#         }

#     return field_dict


# -----------------------------------------------------------------------------.
def var_units_dict():
    """
    Get a dictionary containing the units of the variables

    Returns
    -------
    units : dictionary
        Dictionary with the units of the variables
    """
    # TODO BE UPDATED AND EXPANDED
    units_dict = {
        "rain_rate_32bit": "mm/h",
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
    name_dict = {
        "timestep": "Datetime object of the measurement",
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
        "datetime": "Datetime object of the measurement",
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


# -----------------------------------------------------------------------------.
def get_diameter_bin_center(sensor_name):
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_diameter_bin_center()
    elif sensor_name == "Parsivel2":
        x = get_OTT_Parsivel2_diameter_bin_center()
    elif sensor_name == "ThiesLPM":
        x = get_ThiesLPM_diameter_bin_center()
    else:
        logger.exception(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
        raise ValueError(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
    return x


def get_diameter_bin_lower(sensor_name):
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_diameter_bin_bounds()[:, 0]
    elif sensor_name == "Parsivel2":
        logger.exception(f"Not implemented {sensor_name} device")
        raise NotImplementedError
    elif sensor_name == "ThiesLPM":
        logger.exception(f"Not implemented {sensor_name} device")
        raise NotImplementedError
    else:
        logger.exception(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
        raise ValueError(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
    return x


def get_diameter_bin_upper(sensor_name):
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_diameter_bin_bounds()[:, 1]
    elif sensor_name == "Parsivel2":
        logger.exception(f"Not implemented {sensor_name} device")
        raise NotImplementedError

    elif sensor_name == "ThiesLPM":
        logger.exception(f"Not implemented {sensor_name} device")
        raise NotImplementedError
    else:
        logger.exception(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
        raise ValueError(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
    return x


def get_diameter_bin_width(sensor_name):
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_diameter_bin_width()

    elif sensor_name == "Parsivel2":
        logger.exception(f"Not implemented {sensor_name} device")
        raise NotImplementedError

    elif sensor_name == "ThiesLPM":
        x = get_ThiesLPM_diameter_bin_width()
    else:
        logger.exception(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
        raise ValueError(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
    return x


def get_velocity_bin_center(sensor_name):
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_velocity_bin_center()

    elif sensor_name == "Parsivel2":
        logger.exception(f"Not implemented {sensor_name} device")
        raise NotImplementedError

    elif sensor_name == "ThiesLPM":
        x = get_ThiesLPM_velocity_bin_center()
    else:
        logger.exception(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
        raise ValueError(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
    return x


def get_velocity_bin_lower(sensor_name):
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_velocity_bin_bounds()[:, 0]

    elif sensor_name == "Parsivel2":
        logger.exception(f"Not implemented {sensor_name} device")
        raise NotImplementedError

    elif sensor_name == "ThiesLPM":
        x = get_ThiesLPM_diameter_bin_bounds()[:, 0]
    else:
        logger.exception(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
        raise ValueError(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
    return x


def get_velocity_bin_upper(sensor_name):
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_velocity_bin_bounds()[:, 1]

    elif sensor_name == "Parsivel2":
        logger.exception(f"Not implemented {sensor_name} device")
        raise NotImplementedError

    elif sensor_name == "ThiesLPM":
        x = get_ThiesLPM_velocity_bin_bounds()[:, 1]
    else:
        logger.exception(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
        raise ValueError(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
    return x


def get_velocity_bin_width(sensor_name):
    if sensor_name == "Parsivel":
        x = get_OTT_Parsivel_velocity_bin_width()

    elif sensor_name == "Parsivel2":
        logger.exception(f"Not implemented {sensor_name} device")
        raise NotImplementedError

    elif sensor_name == "ThiesLPM":
        x = get_ThiesLPM_velocity_bin_width()
    else:
        logger.exception(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
        raise ValueError(
            f"L0 bin characteristics for sensor {sensor_name} are not yet defined"
        )
    return x


def get_raw_field_nbins(sensor_name):
    if sensor_name == "Parsivel":
        nbins_dict = {
            "FieldN": 32,
            "FieldV": 32,
            "RawData": 1024,
        }
    elif sensor_name == "Parsivel2":
        logger.exception(f"Not implemented {sensor_name} device")
        raise NotImplementedError

    elif sensor_name == "ThiesLPM":
        logger.exception(f"Not implemented {sensor_name} device")
        raise NotImplementedError
    else:
        logger.exception(
            f"Bin characteristics for sensor {sensor_name} are not yet defined"
        )
        raise ValueError(
            f"Bin characteristics for sensor {sensor_name} are not yet defined"
        )
    return nbins_dict

    # def get_var_explanations_ARM():
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
    #             "qc_precip_rate": "Quality check results on field: Precipitation intensity",
    #             "units": "1",
    #             "description": "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.",
    #             "flag_method": "bit",
    #             "bit_1_description": "Value is equal to missing_value.",
    #             "bit_1_assessment": "Bad",
    #             "bit_2_description": "Value is less than the valid_min.",
    #             "bit_2_assessment": "Bad",
    #             "bit_3_description": "Value is greater than the valid_max.",
    #             "bit_3_assessment": "Bad",
    #         },
    #         {
    #             "weather_code": "SYNOP WaWa Table 4680",
    #             "units": "1",
    #             "valid_min": 0,
    #             "valid_max": 90,
    #             "ancillary_variables": "qc_weather_code",
    #         },
    #         {
    #             "qc_weather_code": "Quality check results on field: SYNOP WaWa Table 4680",
    #             "units": "1",
    #             "description": "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.",
    #             "flag_method": "bit",
    #             "bit_1_description": "Value is equal to missing_value.",
    #             "bit_1_assessment": "Bad",
    #             "bit_2_description": "Value is less than the valid_min.",
    #             "bit_2_assessment": "Bad",
    #             "bit_3_description": "Value is greater than the valid_max.",
    #             "bit_3_assessment": "Bad",
    #         },
    #         {
    #             "equivalent_radar_reflectivity_ott": "Radar reflectivity from the manufacturer's software",
    #             "units": "dBZ",
    #             "valid_min": -60.0,
    #             "valid_max": 250.0,
    #             "ancillary_variables": "qc_equivalent_radar_reflectivity_ott",
    #         },
    #         {
    #             "qc_equivalent_radar_reflectivity_ott": "Quality check results on field: Radar reflectivity from the manufacturer's software",
    #             "units": "1",
    #             "description": "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.",
    #             "flag_method": "bit",
    #             "bit_1_description": "Value is equal to missing_value.",
    #             "bit_1_assessment": "Bad",
    #             "bit_2_description": "Value is less than the valid_min.",
    #             "bit_2_assessment": "Bad",
    #             "bit_3_description": "Value is greater than the valid_max.",
    #             "bit_3_assessment": "Bad",
    #         },
    #         {
    #             "number_detected_particles": "Number of particles detected",
    #             "units": "count",
    #             "valid_min": 0,
    #             "valid_max": 99999,
    #             "ancillary_variables": "qc_number_detected_particles",
    #         },
    #         {
    #             "qc_number_detected_particles": "Quality check results on field: Number of particles detected",
    #             "units": "1",
    #             "description": "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.",
    #             "flag_method": "bit",
    #             "bit_1_description": "Value is equal to missing_value.",
    #             "bit_1_assessment": "Bad",
    #             "bit_2_description": "Value is less than the valid_min.",
    #             "bit_2_assessment": "Bad",
    #             "bit_3_description": "Value is greater than the valid_max.",
    #             "bit_3_assessment": "Bad",
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
    #             "qc_mor_visibility": "Quality check results on field: Meteorological optical range visibility",
    #             "units": "1",
    #             "description": "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.",
    #             "flag_method": "bit",
    #             "bit_1_description": "Value is equal to missing_value.",
    #             "bit_1_assessment": "Bad",
    #             "bit_2_description": "Value is less than the valid_min.",
    #             "bit_2_assessment": "Bad",
    #             "bit_3_description": "Value is greater than the valid_max.",
    #             "bit_3_assessment": "Bad",
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
    #             "qc_snow_depth_intensity": "Quality check results on field: New snow height",
    #             "units": "1",
    #             "description": "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.",
    #             "flag_method": "bit",
    #             "bit_1_description": "Value is equal to missing_value.",
    #             "bit_1_assessment": "Bad",
    #             "bit_2_description": "Value is less than the valid_min.",
    #             "bit_2_assessment": "Bad",
    #             "bit_3_description": "Value is greater than the valid_max.",
    #             "bit_3_assessment": "Bad",
    #         },
    #         {
    #             "laserband_amplitude": "Laserband amplitude",
    #             "units": "count",
    #             "valid_min": 0,
    #             "valid_max": 99999,
    #             "ancillary_variables": "qc_laserband_amplitude",
    #         },
    #         {
    #             "qc_laserband_amplitude": "Quality check results on field: Laserband amplitude",
    #             "units": "1",
    #             "description": "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.",
    #             "flag_method": "bit",
    #             "bit_1_description": "Value is equal to missing_value.",
    #             "bit_1_assessment": "Bad",
    #             "bit_2_description": "Value is less than the valid_min.",
    #             "bit_2_assessment": "Bad",
    #             "bit_3_description": "Value is greater than the valid_max.",
    #             "bit_3_assessment": "Bad",
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
    #         {
    #             "qc_heating_current": "Quality check results on field: Heating current",
    #             "units": "1",
    #             "description": "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.",
    #             "flag_method": "bit",
    #             "bit_1_description": "Value is equal to missing_value.",
    #             "bit_1_assessment": "Bad",
    #             "bit_2_description": "Value is less than the valid_min.",
    #             "bit_2_assessment": "Bad",
    #             "bit_3_description": "Value is greater than the valid_max.",
    #             "bit_3_assessment": "Bad",
    #         },
    #         {
    #             "sensor_voltage": "Sensor voltage",
    #             "units": "V",
    #             "valid_min": 0.0,
    #             "valid_max": 99.9,
    #             "ancillary_variables": "qc_sensor_voltage",
    #         },
    #         {
    #             "qc_sensor_voltage": "Quality check results on field: Sensor voltage",
    #             "units": "1",
    #             "description": "This field contains bit packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits; a value of 0 (no bits set) indicates the data has not failed any QC tests.",
    #             "flag_method": "bit",
    #             "bit_1_description": "Value is equal to missing_value.",
    #             "bit_1_assessment": "Bad",
    #             "bit_2_description": "Value is less than the valid_min.",
    #             "bit_2_assessment": "Bad",
    #             "bit_3_description": "Value is greater than the valid_max.",
    #             "bit_3_assessment": "Bad",
    #         },
    #         {"class_size_width": "Class size width", "units": "mm"},
    #         {
    #             "fall_velocity_calculated": "Fall velocity calculated after Lhermite",
    #             "units": "m/s",
    #         },
    #         {"raw_spectrum": "Raw drop size distribution", "units": "count"},
    #         {"liquid_water_content": "Liquid water content", "units": "mm^3/m^3"},
    #         {
    #             "equivalent_radar_reflectivity": "Radar reflectivity calculated by the ingest",
    #             "units": "dBZ",
    #         },
    #         {
    #             "intercept_parameter": "Intercept parameter, assuming an ideal Marshall-Palmer type distribution",
    #             "units": "1/(m^3 mm)",
    #         },
    #         {
    #             "slope_parameter": "Slope parameter, assuming an ideal Marshall-Palmer type distribution",
    #             "units": "1/mm",
    #         },
    #         {
    #             "median_volume_diameter": "Median volume diameter, assuming an ideal Marshall-Palmer type distribution",
    #             "units": "mm",
    #         },
    #         {
    #             "liquid_water_distribution_mean": "Liquid water distribution mean, assuming an ideal Marshall-Palmer type distribution",
    #             "units": "mm",
    #         },
    #         {
    #             "number_density_drops": "Number density of drops of the diameter corresponding to a particular drop size class per unit volume",
    #             "units": "1/(m^3 mm)",
    #         },
    #         {"diameter_min": "Diameter of smallest drop observed", "units": "mm"},
    #         {"diameter_max": "Diameter of largest drop observed", "units": "mm"},
    #         {"moment1": "Moment 1 from the observed distribution", "units": "mm/m^3"},
    #         {"moment2": "Moment 2 from the observed distribution", "units": "mm^2/m^3"},
    #         {"moment3": "Moment 3 from the observed distribution", "units": "mm^3/m^3"},
    #         {"moment4": "Moment 4 from the observed distribution", "units": "mm^4/m^3"},
    #         {"moment5": "Moment 5 from the observed distribution", "units": "mm^5/m^3"},
    #         {"moment6": "Moment 6 from the observed distribution", "units": "mm^6/m^3"},
    #         {
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
    return dict_ARM_description
