#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 13:42:14 2022

@author: ghiggi
"""

### In L1, add attribute indicating if dataset contains raw_spectrum
# - Nd and Vd are derived

### In L0 check standards: drop scalars (i.e. firmware_*, and time related stuffs)

####--------------------------------------------------------------------------.
#### L1_proc.py
create_L1_dataset_from_L0
# - replace NA flags

# write_L1_to_zarr  # rechunk before writing
# - add zarr encoding


create_L1_summary_statistics
# - regularize timeseries
# - number of dry/rainy minutes
# - timebar plot with 0,>1, NA, no data rain rate (ARM STYLE)
# - timebar data quality

# TODO STANDARDS:
# - Check variables keys are same across units, explanations
# - Check bins width correspond to bounds spacing
# - Check bins center is average of bounds (raise only warning)

####--------------------------------------------------------------------------.
# reformat_ARM_LPM
# reformat_DIVEN_LPM

# metadata.py
check_metadata_compliance  # check dtype also, raise errors !

# L0_proc.py
# - Filter bad data based on sensor_status/error_code
# - check df_sanitizer_fun has only lazy and df arguments !
# - Implement file removal in check_L0_standards

# - Add DISDRODB attrs
attrs["source_data_format"] = "raw_data"
attrs["obs_type"] = "raw"  # preprocess/postprocessed
attrs["level"] = "L0"  # L0, L1, L2, ...
attrs["disdrodb_id"] = ""  # TODO

####--------------------------------------------------------------------------.
### Others
# - Copy metadata yaml to github for syncs and review by external

####--------------------------------------------------------------------------.
### netcdf savings
# if contiguous = True, zlib and fletcher must be False !
# if contiguous, complevel has effect? or only if zlib=True?
# if contiguous, shuffle is performed?

####--------------------------------------------------------------------------.
# Float 32 or Float 64 (f4, f8)
# (u)int 8 16, 32, 64   (u/i  1 2 4 8)
