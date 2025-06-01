# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2023 DISDRODB developers
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""Attributes and encoding options for DISDRODB products."""


def get_attrs_dict():
    """Temporary attributes."""
    attrs_dict = {
        #### L1
        "drop_number": {
            "description": "Counts of drops per diameter and velocity class",
            "long_name": "Drop counts per diameter and velocity class",
            "units": "",
        },
        "drop_counts": {
            "description": "Counts of drops per diameter class",
            "long_name": "Drop counts per diameter class",
            "units": "",
        },
        "Dmin": {
            "description": "Minimum drop diameter",
            "long_name": "Minimum drop diameter",
            "units": "mm",
        },
        "Dmax": {
            "description": "Maximum drop diameter",
            "long_name": "Maximum drop diameter",
            "units": "mm",
        },
        "fall_velocity": {
            "description": "Estimated drop fall velocity per diameter class",
            "long_name": "Estimated drop fall velocity",
            "units": "m s-1",
        },
        "drop_average_velocity": {
            "description": "Average measured drop fall velocity per diameter class",
            "long_name": "Measured average drop fall velocity",
            "units": "m s-1",
        },
        "N": {
            "description": "Total number of selected drops",
            "long_name": "Total number of selected drops",
            "units": "",
        },
        "Nremoved": {
            "description": "Total number of discarded drops",
            "long_name": "Total number of discarded drops",
            "units": "",
        },
        "Nbins": {
            "description": "Number of diameter bins with drops",
            "long_name": "Number of diameter bins with drops",
            "units": "",
        },
        "Nbins_missing": {
            "description": "Number of diameter bins with no drops",
            "long_name": "Number of diameter bins with no drops",
            "units": "",
        },
        "Nbins_missing_fraction": {
            "description": "Fraction of diameter bins with no drops",
            "long_name": "Fraction of diameter bins with no drops",
            "units": "",
        },
        "Nbins_missing_consecutive": {
            "description": "Maximum number of consecutive diameter bins with no drops",
            "long_name": "Maximum number of consecutive diameter bins with no drops",
            "units": "",
        },
        #### L2
        "drop_number_concentration": {
            "description": "Number concentration of drops per diameter class per unit volume",
            "long_name": "Drop number concentration per diameter class",
            "units": "m-3 mm-1",
        },
        "drop_volume": {
            "standard_name": "",
            "units": "mm3",
            "long_name": "Volume of Drops per Diameter Class",
        },
        "drop_total_volume": {
            "standard_name": "",
            "units": "mm3",
            "long_name": "Total Volume of Drops",
        },
        "drop_relative_volume_ratio": {
            "standard_name": "",
            "units": "",
            "long_name": "Relative Volume Ratio of Drops",
        },
        "KEmin": {
            "standard_name": "",
            "units": "J",
            "long_name": "Minimum Drop Kinetic Energy",
        },
        "KEmax": {
            "standard_name": "",
            "units": "J",
            "long_name": "Maximum Drop Kinetic Energy",
        },
        "E": {
            "description": "Kinetic energy per unit rainfall depth",
            "standard_name": "",
            "units": "J m-2 mm-1",
            "long_name": "Rainfall Kinetic Energy",
        },
        "KE": {
            "standard_name": "",
            "units": "J m-2 h-1",
            "long_name": "Kinetic Energy Density Flux",
        },
        "M1": {
            "standard_name": "",
            "units": "m-3 mm",
            "long_name": "First Moment of the Drop Size Distribution",
        },
        "M2": {
            "standard_name": "",
            "units": "m-3 mm2",
            "long_name": "Second Moment of the Drop Size Distribution",
        },
        "M3": {
            "standard_name": "",
            "units": "m-3 mm3",
            "long_name": "Third Moment of the Drop Size Distribution",
        },
        "M4": {
            "standard_name": "",
            "units": "m-3 mm4",
            "long_name": "Fourth Moment of the Drop Size Distribution",
        },
        "M5": {
            "standard_name": "",
            "units": "m-3 mm5",
            "long_name": "Fifth Moment of the Drop Size Distribution",
        },
        "M6": {
            "standard_name": "",
            "units": "m-3 mm6",
            "long_name": "Sixth Moment of the Drop Size Distribution",
        },
        "Nt": {
            "standard_name": "number_concentration_of_rain_drops_in_air",
            "units": "m-3",
            "long_name": "Total Number Concentration",
        },
        "R": {
            "standard_name": "rainfall_rate",
            "units": "mm h-1",
            "long_name": "Instantaneous Rainfall Rate",
        },
        "P": {
            "standard_name": "precipitation_amount",
            "units": "mm",
            "long_name": "Rain Accumulation",
        },
        "Z": {
            "standard_name": "equivalent_reflectivity_factor",
            "units": "dBZ",
            "long_name": "Equivalent Radar Reflectivity Factor",
        },
        "W": {
            "description": "Water Mass of the Drop Size Distribution",
            "standard_name": "mass_concentration_of_liquid_water_in_air",
            "units": "g m-3",
            "long_name": "Liquid Water Content",
        },
        "D10": {
            "standard_name": "",
            "units": "mm",
            "long_name": "10th Percentile Drop Diameter",
        },
        "D50": {
            "standard_name": "median_volume_diameter",
            "units": "mm",
            "long_name": "Median Volume Drop Diameter",
        },
        "D90": {
            "standard_name": "",
            "units": "mm",
            "long_name": "90th Percentile Drop Diameter",
        },
        "Dmode": {
            "standard_name": "",
            "units": "mm",
            "long_name": "Mode Diameter of the Drop Size Distribution",
        },
        "Dm": {
            "standard_name": "Dm",
            "units": "mm",
            "long_name": "Mean Volume Diameter",
        },
        "sigma_m": {
            "standard_name": "",
            "units": "mm",
            "long_name": "Standard Deviation of Mass Spectrum",
        },
        "Nw": {
            "standard_name": "normalized_intercept_parameter",
            "units": "mm-1 m-3",  # TODO
            "long_name": "Normalized Intercept Parameter of a Normalized Gamma Distribution",
        },
        "N0": {
            "standard_name": "intercept_parameter",
            "units": "mm-1 m-3",  # TODO
            "long_name": "Intercept Parameter of the Modeled Drop Size Distribution",
        },
        "mu": {
            "standard_name": "shape_parameter",
            "units": "1",  # TODO
            "long_name": "Shape Parameter of the Modeled Drop Size Distribution",
        },
        "Lambda": {
            "standard_name": "distribution_slope",
            "units": "1/mm",  # TODO
            "long_name": "Slope Parameter of the Modeled Drop Size Distribution",
        },
        "sigma": {
            "standard_name": "distribution_slope",
            "units": "1/mm",  # TODO
            "long_name": "Slope Parameter of the Modeled Lognormal Distribution",
        },
        # Radar variables
        "Zh": {
            "description": "Radar reflectivity factor at horizontal polarization",
            "long_name": "Horizontal Reflectivity",
            "units": "dBZ",
        },
        "Zdr": {
            "description": "Differential reflectivity",
            "long_name": "Differential Reflectivity",
            "units": "dB",
        },
        "rho_hv": {
            "description": "Correlation coefficient between horizontally and vertically polarized reflectivity",
            "long_name": "Copolarized Correlation Coefficient",
            "units": "",
        },
        "ldr": {
            "description": "Linear depolarization ratio",
            "long_name": "Linear Depolarization Ratio",
            "units": "dB",
        },
        "Kdp": {
            "description": "Specific differential phase",
            "long_name": "Specific Differential Phase",
            "units": "deg/km",
        },
        "Ai": {
            "description": "Specific attenuation",
            "long_name": "Specific attenuation",
            "units": "dB/km",
        },
    }
    return attrs_dict


def get_encoding_dict():
    """Temporary encoding dictionary."""
    encoding_dict = {
        "M1": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "M2": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "M3": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "M4": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "M5": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "M6": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "Nt": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "R": {
            "dtype": "uint16",
            "scale_factor": 0.01,
            "_FillValue": 65535,
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "P": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "Z": {
            "dtype": "uint16",
            "scale_factor": 0.01,
            "add_offset": -60,
            "_FillValue": 65535,
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "W": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "Dm": {
            "dtype": "uint16",
            "scale_factor": 0.001,
            "_FillValue": 65535,
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "sigma_m": {
            "dtype": "uint16",
            "scale_factor": 0.001,
            "_FillValue": 65535,
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "Dmode": {
            "dtype": "uint16",
            "scale_factor": 0.001,
            "_FillValue": 65535,
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "Nw": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "D50": {
            "dtype": "uint16",
            "scale_factor": 0.001,
            "_FillValue": 65535,
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "D10": {
            "dtype": "uint16",
            "scale_factor": 0.001,
            "_FillValue": 65535,
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "D90": {
            "dtype": "uint16",
            "scale_factor": 0.001,
            "_FillValue": 65535,
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "drop_number": {
            "dtype": "uint32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
            "_FillValue": 4294967295,
        },
        "drop_counts": {
            "dtype": "uint32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
            "_FillValue": 4294967295,
        },
        "N": {
            "dtype": "uint32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
            "_FillValue": 4294967295,
        },
        "Nremoved": {
            "dtype": "uint32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
            "_FillValue": 4294967295,
        },
        "Nbins": {
            "dtype": "uint8",
            "_FillValue": 255,
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "Dmin": {
            "dtype": "uint16",
            "scale_factor": 0.001,
            "_FillValue": 65535,
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "Dmax": {
            "dtype": "uint16",
            "scale_factor": 0.001,
            "_FillValue": 65535,
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "drop_average_velocity": {
            "dtype": "uint16",
            "scale_factor": 0.001,
            "_FillValue": 65535,
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "fall_velocity": {
            "dtype": "uint16",
            "scale_factor": 0.001,
            "_FillValue": 65535,
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "drop_number_concentration": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "drop_volume": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "drop_total_volume": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "drop_relative_volume_ratio": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "KEmin": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "KEmax": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "E": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "KE": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        # Radar variables
        "Zh": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "Zdr": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "rho_hv": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "ldr": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "Kdp": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
        "Ai": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "fletcher32": False,
            "contiguous": False,
        },
    }
    return encoding_dict
