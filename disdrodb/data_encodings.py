#!/usr/bin/env python3
# -*- coding: utf-8 -*-

 

def get_L0_dtype_standards(sensor_name: str) -> dict:
    from disdrodb.standards import get_L0_dtype

    # TODO: TO REFACTOR !!!!
    dtype_dict = {
        # Disdronet raspberry variables
        "epoch_time": "float32",
        "time": "M8[s]",
        "id": "uint32",
        # Datalogger variables
        "datalogger_heating_current": "float32",
        "datalogger_battery_voltage": "float32",
        "datalogger_temperature": "object",
        "datalogger_voltage": "object",
        "datalogger_error": "uint8",
        # Coords
        "latitude": "float32",
        "longitude": "float32",
        "altitude": "float32",
        # Custom fields
        "Unknow_column": "object",
        # Temp variables
        "temp": "object",
        "temp1": "object",
        "temp2": "object",
        "temp3": "object",
        "temp4": "object",
        
        "TEMPORARY": "object",
        "TO_BE_MERGE": "object",
        "TO_BE_MERGE2": "object",
        "TO_BE_PARSED": "object",
        "TO_BE_SPLITTED": "object",
        "TO_DEBUG": "object",
        "Debug_data": "object",
        "All_0": "object",
        "error_code?": "object",
        "unknow2": "object",
        "unknow3": "object",
        "unknow4": "object",
        "unknow5": "object",
        "unknow": "object",
        "unknow6": "object",
        "unknow7": "object",
        "unknow8": "object",
        "unknow9": "object",
        "power_supply_voltage": "object",
        "A_voltage2?": "object",
        "A_voltage?": "object",
        "All_nan": "object",
        "All_5000": "object",
    }
    d1 = get_L0_dtype(sensor_name=sensor_name)
    dtype_dict.update(d1)
    return dtype_dict


def get_dtype_standards_all_object(sensor_name):
    # TODO: move to dev_tools I would say... is not used by any parser right?
    dtype_dict = get_L0_dtype_standards(sensor_name=sensor_name)
    for i in dtype_dict:
        dtype_dict[i] = "object"

    return dtype_dict
