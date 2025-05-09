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
"""Issue YAML File Reader."""
import logging

import yaml

from disdrodb.api.path import define_issue_filepath
from disdrodb.issue.checks import check_issue_dict

logger = logging.getLogger(__name__)


class NoDatesSafeLoader(yaml.SafeLoader):
    """A YAML loader that does not parse dates."""

    @classmethod
    def remove_implicit_resolver(cls, tag_to_remove):
        """
        Remove implicit resolvers for a particular tag.

        Takes care not to modify resolvers in super classes.

        We want to load datetimes as strings, not dates.
        """
        if "yaml_implicit_resolvers" not in cls.__dict__:
            cls.yaml_implicit_resolvers = cls.yaml_implicit_resolvers.copy()

        for first_letter, mappings in cls.yaml_implicit_resolvers.items():
            cls.yaml_implicit_resolvers[first_letter] = [
                (tag, regexp) for tag, regexp in mappings if tag != tag_to_remove
            ]


def _load_yaml_without_date_parsing(filepath):
    """Read a YAML file without converting automatically date string to datetime."""
    NoDatesSafeLoader.remove_implicit_resolver("tag:yaml.org,2002:timestamp")
    with open(filepath) as f:
        dictionary = yaml.load(f, Loader=NoDatesSafeLoader)
    # Return empty dictionary if nothing to be read in the file
    if isinstance(dictionary, type(None)):
        dictionary = {}
    return dictionary


def read_issue(filepath: str) -> dict:
    """Read YAML issue file.

    Parameters
    ----------
    filepath : str
        Filepath of the issue YAML.

    Returns
    -------
    dict
        Issue dictionary.
    """
    issue_dict = _load_yaml_without_date_parsing(filepath)
    issue_dict = check_issue_dict(issue_dict)
    return issue_dict


def read_station_issue(data_source, campaign_name, station_name, metadata_archive_dir=None):
    """Open the station issue YAML file into a dictionary.

    Parameters
    ----------
    data_source : str
        The name of the institution (for campaigns spanning multiple countries) or
        the name of the country (for campaigns or sensor networks within a single country).
        Must be provided in UPPER CASE.
    campaign_name : str
        The name of the campaign. Must be provided in UPPER CASE.
    station_name : str
        The name of the station.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the ``data_archive_dir`` path specified in the DISDRODB active configuration will be used.

    Returns
    -------
    issue_dict: dictionary
        The station issue dictionary

    """
    # Retrieve metadata filepath
    issue_filepath = define_issue_filepath(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=True,
    )
    issue_dict = read_issue(issue_filepath)
    return issue_dict
