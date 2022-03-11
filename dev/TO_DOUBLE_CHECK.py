#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:21:47 2021

@author: ghiggi
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:03:47 2021

@author: ghiggi
"""
# Name 
Nd / d / drop_diameter_counts / Field_N # ???
Vd / v / drop_velocity_counts / Field_V # ???    
N / n / raw_drop_number  / Field_Raw     # ??? PCM Parsivel conditional matrix

# 'RawDrops'
# 'VolumetricDrops'
# 'CorrectedVolumetricDrops'

##----------------------------------------------------------------------------.
### Parsivel Specifications 
# Wavelength: 650 nm 
# OutputPower: 3 mW
# Beam size: 180 x 30 mm 
# Measurement surface: 5 cm2 
# Number of bins: 32 

### Parsivel2 Specifications 
# Wavelength: 650 nm 
# OutputPower: 0.2 mW
# Light strip surface (W X D): 30 x 1 mm 
# Measurement surface (W X D): 180 x 30 mm  
# Number of bins: 32 

##----------------------------------------------------------------------------.
### Differences between Parsivel and Parsivel2
# Parsivel1:
#   11:  Number of detected particles               (number_particles)
# Parsivel2: 
#   11:  Number of particle detected and validated  (number_particles)
#   60:  Number of all particles detected           (number_particles_all)

# Field N (d)
# - Parsivel1: 1/m3 mm 
# - Parsivel2: log10(1/m3 mm) 

# Field V (d)
# - Parsivel1:  ? 
# - Parsivel2: m/s 

# Raw data same? 

# Field 61 in Parsivel2? list_particles ? 

# Only in Parsivel 2 
# "34": "rain_kinetic_energy" 
# "35": "snowfall_rate"

##----------------------------------------------------------------------------.
### Terminology for Parsivel2
# temperature_PBC               # PBC = Printed circuit board
# temperature_right             # AB: confusing: right or left with respect to what?
# temperature_left              # AB: If part of the head, I suggest to use transmitter or receiver side...

##----------------------------------------------------------------------------.
### Sensor status  
# 0: "Everything ok"
# 1: "Laser protective glass is dirty, but measurement is still possible"
# 2: "Laser protective glass is dirty, partially covered. No measurement possible"
# 3: "Laser damaged. No measurement possible"

### Error code ? What is it 

### Rain amount accumulated (32 bit) vs rain amount absolute (32 bit) (02 vs 24)

### How to retrieve following attributes
# sensor_beam_width
# sensor_nominal_width
# firmware (if not logged)

##----------------------------------------------------------------------------.
### Dict for weather codes 
# http://www.czo.psu.edu/downloads/Metadataworksheets/LPM_SYNOP_METAR_key.pdf

##----------------------------------------------------------------------------.
#############################################################
### Parsivel class bin difference (Docs, Tim code, ... ) ####
#############################################################

### Velocity  center difference 
v_bounds = get_OTT_Parsivel_velocity_bin_bounds()
v_width1 = v_bounds[:,1] - v_bounds[:,0]
v_center1 = v_bounds[:,0]  + v_width1/2

v_center = get_OTT_Parsivel_velocity_bin_center()
v_width = get_OTT_Parsivel_velocity_bin_width()

np.column_stack((v_center, v_center1))
np.column_stack((v_width, v_width1))

# np.round(v_center - v_center1, 2)


### Diameter center difference 
d_bounds = get_OTT_Parsivel_diameter_bin_bounds()
d_width1 = d_bounds[:,1] - d_bounds[:,0]
d_center1 = d_bounds[:,0]  + d_width1/2

d_center = get_OTT_Parsivel_diameter_bin_center()
d_width = get_OTT_Parsivel_diameter_bin_width()

np.column_stack((d_center, d_center1))
np.column_stack((d_width, d_width1)) 

# np.round(d_center - d_center1, 4)

##----------------------------------------------------------------------------.
### Attributes to derive 
# - Years coverage
# - Total minutes
# - Total DSD minutes 
# - Total rain events
# - Other stats TBD 

##----------------------------------------------------------------------------.
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

 