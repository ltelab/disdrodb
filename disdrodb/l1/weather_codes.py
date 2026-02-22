# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://SYNOP_4677_WWw.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""DISDRODB weather codes module."""

from disdrodb.l1.classification import add_hydrometeor_type_attrs
from disdrodb.utils.xarray import xr_remap_numeric_array

SYNOP_4677_WW_HYDROCLASS = {
    # Special
    -2: -2,  # missing
    -1: -1,  # noise
    0: 0,  # no precipitation
    # Drizzle (0-0.2, 0.2-0.5, >0.5 mm/h)
    51: 1,  # drizzle light      (0-0.2)
    53: 1,  # drizzle moderate   (0.2-0.5)
    55: 1,  # drizzle heavy      (>0.5)
    # Drizzle with rain
    58: 2,  # drizzle+rain light (0-0.2)
    59: 2,  # drizzle+rain mod/strong (>0.2)
    # Rain
    61: 3,  # rain light         (0-0.2)
    63: 3,  # rain moderate      (0.5-4)
    65: 3,  # rain heavy         (>4)
    # Rain or drizzle with snow
    68: 4,  # mixed light        (0-0.5)
    69: 4,  # mixed moderate     (>0.5)
    # Snow
    71: 5,  # snow light         (0-0.5)
    73: 5,  # snow moderate      (0.5-4)
    75: 5,  # snow heavy         (>4)
    # Snow grains / ice crystals
    77: 6,
    # Ice pellets (sleet)
    87: 7,  # light              (0-2.4)
    88: 7,  # moderate/heavy     (>2.4)
    # Hail
    89: 9,  # small hail         (D < 8 mm)
    90: 9,  # hail               (D ≥ 8 mm)
}

SYNOP_4680_WAWA_HYDROCLASS = {
    -2: -2,  # missing
    -1: -1,  # noise
    0: 0,  # no precipitation
    # Drizzle
    51: 1,  # light
    52: 1,  # moderate
    53: 1,  # heavy
    # Drizzle with rain
    57: 2,
    58: 2,
    # Rain
    61: 3,
    62: 3,
    63: 3,
    # Rain or drizzle with snow
    67: 4,
    68: 4,
    # Snow
    71: 5,
    72: 5,
    73: 5,
    # Snow grains
    77: 6,
    # Ice pellets
    74: 7,
    75: 7,
    76: 7,
    # Hail
    89: 9,
    90: 9,
}

METAR_4678_HYDROCLASS = {
    # No precipitation
    "NP": 0,
    # Drizzle (0-0.25, 0.25-0.5, >0.5)
    "-DZ": 1,
    "DZ": 1,
    "+DZ": 1,
    # Drizzle with rain (0-2.5, 2.5-7.6, >7.6)
    "-RADZ": 2,
    "RADZ": 2,
    "+RADZ": 2,
    # Rain (0-2.5, 2.5-7.6, >7.6)
    "-RA": 3,
    "RA": 3,
    "+RA": 3,
    # Rain or drizzle with snow
    "-RASN": 4,
    "RASN": 4,
    "+RASN": 4,
    # Snow (0-1.25, 1.25-2.5, >2.5)
    "-SN": 5,
    "SN": 5,
    "+SN": 5,
    # Snow grains
    "SG": 6,
    # Ice pellets (0-1.25, 1.25-2.5, >2.5)
    "-PL": 7,
    "PL": 7,
    "+PL": 7,
    # Graupel
    "-GS": 8,
    "GS": 8,
    "+GS": 8,
    # Hail (size dependent)
    "-GR": 9,
    "GR": 9,
}

NWS_HYDROCLASS = {
    # No precipitation
    "C": 0,
    # Drizzle (0-0.25, 0.25-0.5, >0.5)
    "L-": 1,
    "L": 1,
    "L+": 1,
    # Drizzle with rain
    "RL-": 2,
    "RL": 2,
    "RL+": 2,
    # Rain
    "R-": 3,
    "R": 3,
    "R+": 3,
    # Rain or drizzle with snow
    "RLS-": 4,
    "RLS": 4,
    "RLS+": 4,
    # Snow
    "S-": 5,
    "S": 5,
    "S+": 5,
    # Snow grains
    "SG": 6,
    # Ice pellets
    "IP": 7,
    # Graupel
    "SP": 8,
    # Hail
    "A": 9,
}

WEATHER_CODES_HYDROCLASS_DICT = {
    "weather_code_synop_4680": SYNOP_4680_WAWA_HYDROCLASS,
    "weather_code_synop_4677": SYNOP_4677_WW_HYDROCLASS,
    "weather_code_metar_4678": METAR_4678_HYDROCLASS,
    "weather_code_nws": NWS_HYDROCLASS,
    "weather_code_synop_4680_5min": SYNOP_4680_WAWA_HYDROCLASS,
    "weather_code_synop_4677_5min": SYNOP_4677_WW_HYDROCLASS,
    "weather_code_metar_4678_5min": METAR_4678_HYDROCLASS,
}


def get_hydroclass_from_weather_code(ds, weather_code):
    """Convert weather code to DISDRODB hydrometeor type using predefined mapping dictionaries."""
    if weather_code not in ds.data_vars:
        raise ValueError(f"{weather_code} is not a Dataset variable.")
    if weather_code not in WEATHER_CODES_HYDROCLASS_DICT:
        valid_weather_codes = list(WEATHER_CODES_HYDROCLASS_DICT)
        raise ValueError(f"Invalid {weather_code}. Accepted weather codes are  {valid_weather_codes}")
    mapping_dict = WEATHER_CODES_HYDROCLASS_DICT[weather_code]
    da_weather_code = ds[weather_code]
    hydromeor_type = xr_remap_numeric_array(da_weather_code, mapping_dict)
    hydromeor_type = add_hydrometeor_type_attrs(hydromeor_type)
    return hydromeor_type
