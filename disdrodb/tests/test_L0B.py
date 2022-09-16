#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 22:56:20 2022

@author: ghiggi
"""
import numpy as np

# -----------------------------------------------------------------------------.
### TODO for reader developer
# Add checks to raw_drop_concentration', 'raw_drop_average_velocity', 'raw_drop_number'
# - Function that check that the divider is recognized for
# - Function that check that the length of the array is as expected
# --> If there is a supplementary divider at the beginning and the end if fails
# --> If there is a supplementary divider at the end, it can deal with that
# --> If there is a supplementary divider at the beginning, it get an erronous results
#     The first value is set to 0 (because '') and the original last one (since exceeding the expected length) will be dropped

# -----------------------------------------------------------------------------.
### Tests various strings
n_values = 5
string = ""  # ---> [0., 0., 0., 0., 0.]
string = ";;;;"  # ---> [0., 0., 0., 0., 0.]
string = ";;;;;"  # ---> [0., 0., 0., 0., 0.] # additional delimiter add the end case
string = ";;;1;"  #
string = ";;;1;;"  # additional delimiter add the end case
string = "000;000;000;000;001"
string = "000;000;000;000;001;"  # additional delimiter add the end case
format_string_array(string, n_values=n_values)

# -----------------------------------------------------------------------------.
### Test '' replacing with 0
arr = np.array(["", "1"], dtype="str")
arr = np.array(["000", "001"], dtype="<U3")
np.char.replace(arr, "", "0")  # 001 become 000010 !!!

np.char.find(arr, "")  # is not able to pinpoint empty string
arr == ""  # yes ... but this must be test because is fragile and might change in future

# Solution
arr[arr == ""] = "0"

# -----------------------------------------------------------------------------.
### Tests splitter behaviour with None
"".split(None) == []
