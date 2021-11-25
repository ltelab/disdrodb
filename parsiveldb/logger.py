#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:45:37 2021

@author: kimbo
"""

#Logger

import time
import logging

def log(path):
    
    file_name = f'{time.strftime("%d-%m-%Y_%H-%M-%S")}.log'
    
    logger = logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'{path}/{file_name}', encoding='utf-8', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' , level=logging.DEBUG)
    
    return logger