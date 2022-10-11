#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2022 DISDRODB developers
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

import os
import yaml


def get_attrs_standards() -> dict:
    """Get DISDRODB metadata default values.

    Returns
    -------
    dict
        Dictionary of attibutes standard
    """
    # TO REMOVE
    # - station_number
    # - disdrodb_id
    # - temporal_resolution

    # TO ADD
    # - comments
    # - acknoledgements
    # - license

    # sensor_wavelegth --> sensor_wavelength

    list_attrs = [  # Description
        "title",
        "description",
        "source",
        "history",
        "conventions",
        "campaign_name",
        "project_name",
        # Location
        "station_id",
        "station_name",
        # "station_number",TODO: REMOVE
        "location",
        "country",
        "continent",
        "latitude",
        "longitude",
        "altitude",
        "crs",
        "proj4_string",
        "EPSG",
        "latitude_unit",
        "longitude_unit",
        "altitude_unit",
        # Sensor info
        "sensor_name",
        "sensor_long_name",
        "sensor_wavelength",
        "sensor_serial_number",
        "firmware_iop",
        "firmware_dsp",
        "firmware_version",
        "sensor_beam_width",
        "sensor_nominal_width",
        # "temporal_resolution",# TODO REMOVE
        "measurement_interval",
        # Attribution
        "contributors",
        "authors",
        "institution",
        "references",
        "documentation",
        "website",
        "source_repository",
        "doi",
        "contact",
        "contact_information",
        # Source datatype
        "source_data_format",
        # DISDRO DB attrs
        "obs_type",
    ]
    attrs = {key: "" for key in list_attrs}
    # TODO: temporary attributes for EPFL development
    # attrs["sensor_name"] = "OTT_Parsivel"
    attrs["latitude"] = -9999
    attrs["longitude"] = -9999
    attrs["altitude"] = -9999
    # attrs["institution"] = "Laboratoire de Teledetection Environnementale -  Ecole Polytechnique Federale de Lausanne"
    # attrs["sensor_long_name"] = "OTT Hydromet Parsivel"
    # attrs["contact_information"] = "http://lte.epfl.ch"

    # Defaults attributes
    attrs["latitude_unit"] = "DegreesNorth"
    attrs["longitude_unit"] = "DegreesEast"
    attrs["altitude_unit"] = "MetersAboveSeaLevel"
    attrs["crs"] = "WGS84"
    attrs["EPSG"] = 4326
    attrs["proj4_string"] = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

    return attrs


# TODO: create_metadata_yml ? Decide similar pattern to create_issue<_yml>
def create_metadata(fpath: str) -> None:
    """Create default YAML metadata file.

    Parameters
    ----------
    fpath : str
        File path
    """

    attrs = get_attrs_standards()
    with open(fpath, "w") as f:
        yaml.dump(attrs, f, sort_keys=False)


def read_metadata(raw_dir: str, station_id: str) -> dict:
    """Read YAML metadata file.

    Parameters
    ----------
    raw_dir : str
        Path of the raw directory
    station_id : int
        Id of the station.

    Returns
    -------
    dict
        Dictionnary of the metadata.
    """

    metadata_fpath = os.path.join(raw_dir, "metadata", station_id + ".yml")
    with open(metadata_fpath, "r") as f:
        attrs = yaml.safe_load(f)
    return attrs


def check_metadata_compliance(raw_dir: str) -> None:
    """Check YAML metadata files compliance.

    Parameters
    ----------
    raw_dir : str
        Path of the raw directory
    """

    # TODO: MISSING CHECKS
    # - CHECK NO MISSING IMPORTANT METADATA
    # - CHECK NO ADDITIONAL METADATA OTHER THAN STANDARDS
    # - CHECK VALUE VALIDITY OF KEYS REQUIRED FOR PROCESSING
    # check_sensor_name(sensor_name=sensor_name)
    pass
    return


# def write_metadata(attrs, fpath):
#     """Write dictionary to YAML file."""
#     with open(fpath, "w") as f:
#         yaml.dump(attrs, f, sort_keys=False)
