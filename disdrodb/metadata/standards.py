#!/usr/bin/env python3

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
"""Define the DISDRODB Metadata standards.

When editing one of these variables, one need to update all YAML files in
the DISDRODB-METADATA repository
"""

# Define valid values for specific metadata keys
METADATA_VALUES = {
    "deployment_status": ["ongoing", "terminated"],
    "platform_type": ["fixed", "mobile"],
    "deployment_mode": ["land", "ship", "truck", "cable"],
    "platform_protection": ["shielded", "unshielded", ""],
}


METADATA_KEYS = [
    ## Mandatory fields
    "data_source",
    "campaign_name",
    "station_name",
    "sensor_name",
    # DISDRODB reader info
    "reader",
    "raw_data_glob_pattern",
    "raw_data_format",  # 'txt', 'netcdf'
    "measurement_interval",  # sampling_interval ? [in seconds]
    ## Deployment Info
    "deployment_status",  # 'terminated', 'ongoing'
    "deployment_mode",  # 'land', 'ship', 'truck', 'cable'
    "platform_type",  # 'fixed', 'mobile'
    "latitude",  # in degrees North
    "longitude",  # in degrees East
    "altitude",  # in meter above sea level
    # Platform info
    "platform_protection",  # 'shielded', 'unshielded'
    "platform_orientation",  # [0-360] from N (clockwise)
    ## Time info
    "time_coverage_start",  # YYYY-MM-DDTHH:MM:SS
    "time_coverage_end",  # YYYY-MM-DDTHH:MM:SS
    ## DISDRODB data url
    "disdrodb_data_url",
    ## Source
    "source",
    "source_convention",
    "source_processing_date",
    ## Description
    "title",
    "description",
    "project_name",
    "keywords",
    "summary",
    "history",
    "comment",
    "station_id",
    "location",
    "country",
    "continent",
    ## Sensor info
    "sensor_long_name",
    "sensor_manufacturer",
    "sensor_wavelength",
    "sensor_serial_number",
    "firmware_iop",
    "firmware_dsp",
    "firmware_version",
    "sensor_beam_length",
    "sensor_beam_width",
    "sensor_nominal_width",
    "calibration_sensitivity",
    "calibration_certification_date",
    "calibration_certification_url",
    ## Attribution
    "contributors",
    "authors",
    "authors_url",
    "contact",
    "contact_information",
    "acknowledgement",  # acknowledgements?
    "references",
    "documentation",
    "website",
    "institution",
    "source_repository",
    "license",
    "doi",
]
