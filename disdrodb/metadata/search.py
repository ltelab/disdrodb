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
"""Routines to manipulate the DISDRODB Metadata Archive."""


from disdrodb.api.path import define_metadata_filepath
from disdrodb.api.search import available_stations


def get_list_metadata(
    data_sources=None,
    campaign_names=None,
    station_names=None,
    product=None,
    available_data=False,
    raise_error_if_empty=False,
    invalid_fields_policy="raise",
    data_archive_dir=None,
    metadata_archive_dir=None,
    **product_kwargs,
):
    """
    Get station metadata filepaths.

    By default, it returns the metadata file paths of stations present in the DISDRODB Metadata Archive matching
    the filtering criteria.

    If the DISDRODB product is specified, it lists only metadata file paths of stations with the specified product
    present in the local DISDRODB Data Archive.

    Parameters
    ----------
    product : str or None, optional
        Name of the product to filter on (e.g., "RAW", "L0A", "L1").

        If the DISDRODB product is not specified (default),
        it returns the metadata file paths of stations present in the DISDRODB Metadata Archive matching
        the filtering criteria.

        If the DISDRODB product is specified, it lists only metadata file paths of stations with the specified product
        present in the local DISDRODB Data Archive.

    available_data : bool, optional

        If ``product`` is not specified:

            - if available_data is False, return metadata filepaths of stations present in the DISDRODB Metadata Archive
            - if available_data is True, return metadata filepaths of stations with data available on the
            online DISDRODB Decentralized Data Archive (i.e., stations with the disdrodb_data_url in the metadata).

        If ``product`` is specified:

            - if available_data is False, return metadata filepaths of stations where
            the product directory exists in the in the local DISDRODB Data Archive
            - if available_data is True, return metadata filepaths of stations where product data exists in the
              in the local DISDRODB Data Archive.
        The default is is False.

    data_sources : str or sequence of str, optional
        One or more data source identifiers to filter stations by.
        The name(s) must be UPPER CASE.
        If None, no filtering on data source is applied. The default is is ``None``.
    campaign_names : str or sequence of str, optional
        One or more campaign names to filter stations by.
        The name(s) must be UPPER CASE.
        If None, no filtering on campaign is applied. The default is is ``None``.
    station_names : str or sequence of str, optional
        One or more station names to include.
        If None, all stations matching other filters are considered. The default is is ``None``.
    raise_error_if_empty : bool, optional
        If True and no stations satisfy the criteria, raise a ``ValueError``.
        If False, return an empty list/tuple. The default is False.
    invalid_fields_policy : {'raise', 'warn', 'ignore'}, optional
        How to handle invalid filter values for ``data_sources``, ``campaign_names``,
        or ``station_names`` that are not present in the metadata archive:

          - 'raise' : raise a ``ValueError`` (default)
          - 'warn'  : emit a warning, then ignore invalid entries
          - 'ignore': silently drop invalid entries
    data_archive_dir : str or Path-like, optional
        Path to the root of the local DISDRODB Data Archive. Format: ``<...>/DISDRODB``
        Required only if ``product``is specified.
        If None, the``data_archive_dir`` path specified in the DISDRODB active configuration file is used.
        The default is None.
    metadata_archive_dir : str or Path-like, optional
        Path to the root of the DISDRODB Metadata Archive. Format: ``<...>/DISDRODB``
        If None, the``metadata_archive_dir`` path specified in the DISDRODB active configuratio. The default is None.
    **product_kwargs : dict, optional
        Additional arguments required for some products.
        For example, for the "L2E" product, you need to specify ``rolling`` and
        ``sample_interval``. For the "L2M" product, you need to specify also
        the ``model_name``.

    Returns
    -------
    metadata_filepaths: list
        List of metadata YAML file paths

    """
    list_info = available_stations(
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        product=product,
        available_data=available_data,
        raise_error_if_empty=raise_error_if_empty,
        invalid_fields_policy=invalid_fields_policy,
        metadata_archive_dir=metadata_archive_dir,
        data_archive_dir=data_archive_dir,
        return_tuple=True,
        **product_kwargs,
    )

    # Define metadata filepath
    metadata_filepaths = [
        define_metadata_filepath(
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        for data_source, campaign_name, station_name in list_info
    ]
    return sorted(set(metadata_filepaths))
