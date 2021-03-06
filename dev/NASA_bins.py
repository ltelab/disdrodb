#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:53:33 2021

@author: ghiggi
"""

### NASA Parsivel bins 
# - IFloods 
https://github.com/DISDRONET/PyDSD/blob/master/pydsd/io/ParsivelReader.py 
https://github.com/DISDRONET/PyDSD/blob/master/pydsd/io/ParsivelNasaGVReader.py

NASADiamCentres = function() {
    ## Modified centre diameters for Parsivel classes used by processed NASA data.
    
    return(c(0.064,
             0.193,
             0.321,
             0.450,
             0.579,
             0.708,
             0.836,
             0.965,
             1.094,
             1.223,
             1.416,
             1.674,
             1.931,
             2.189,
             2.446,
             2.832,
             3.347,
             3.862,
             4.378,
             4.892,
             5.665,
             6.695,
             7.725,
             8.755,
             9.785,
             11.330,
             13.390,
             15.450,
             17.510,
             19.570,
             22.145,
             25.235))
}

NASADiamWidths = function() {
    return(c(0.129, 0.129, 0.129, 0.129, 0.129, 0.129, 0.129, 0.129,
             0.129, 0.129, 0.257, 0.257, 0.257, 0.257, 0.257, 0.515,
             0.515, 0.515, 0.515, 0.515, 1.030, 1.030, 1.030, 1.030,
             1.030, 2.060, 2.060, 2.060, 2.060, 2.060, 3.090, 3.090))
}

def get_OTT_Parsivel_bins():
    diameter_center = np.array([  
        0.062,
        0.187,
        0.312,
        0.437,
        0.562,
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

    