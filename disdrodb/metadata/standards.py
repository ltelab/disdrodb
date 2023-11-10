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
"""Define DISDRODB Metadata Standards."""


def get_valid_metadata_keys() -> list:
    """Get DISDRODB valid metadata list.

    Returns
    -------
    list
        List of valid metadata keys
    """
    # NOTE: When updating one of these keys, one need to update the yaml in/at:
    # - the disdrodb-data repository
    # - disdrodb/data/DISDRODB/Raw/DATA_SOURCE/CAMPAIGN_NAME/metadata/*.yml
    # - disdrodb/tests/data/check_readers/DISDRODB/Raw/*/*/metadata/10.yml
    # - disdrodb/tests/data/test_dir_structure/DISDRODB/Raw/DATA_SOURCE/CAMPAIGN_NAME/metadata/STATION_NAME.yml
    list_attrs = [
        ## Mandatory fields
        "data_source",
        "campaign_name",
        "station_name",
        "sensor_name",
        "reader",
        "raw_data_format",  # 'txt', 'netcdf'
        "platform_type",  # 'fixed', 'mobile'
        ## DISDRODB keys
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
        ## Deployment Info
        "latitude",  # in degrees North
        "longitude",  # in degrees East
        "altitude",  # in meter above sea level
        "deployment_status",  # 'ended', 'ongoing'
        "deployment mode",  # 'land', 'ship', 'truck', 'cable'
        "platform_protection",  # 'shielded', 'unshielded'
        "platform_orientation",  # [0-360] from N (clockwise)
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
        "sensor_nominal_width",  # ?
        ## effective_measurement_area ?  # 0.54 m^2
        "measurement_interval",  # sampling_interval ? [in seconds]
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
    return list_attrs
