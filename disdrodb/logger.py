#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:45:37 2021

@author: kimbo
"""

#Logger

import time
import logging

loggers = {}

def log(path, logger_name):
    
    global loggers
    
    file_name = f'{path}/{time.strftime("%d-%m-%Y_%H-%M-%S")}.log'
    
    format_type = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    level = logging.DEBUG

    logger = logging.getLogger(logger_name)
    
    logger.setLevel(level)
    
    formatter = logging.Formatter(format_type)
    
    fh = logging.FileHandler(file_name)
    
    fh.setFormatter(formatter)
    
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(fh)

        
    return logger

def close_log():
    for handler in loggers:
        handler.close()
        log.removeHandler(handler)