import os

import numpy as np

from disdrodb.api.checks import (
    check_invalid_fields_policy,
    check_product,
    check_product_kwargs,
    check_valid_fields,
)
from disdrodb.api.path import (
    define_data_dir,
    define_data_source_dir,
    define_metadata_dir,
    define_metadata_filepath,
    define_station_dir,
)
from disdrodb.configs import get_data_archive_dir, get_metadata_archive_dir
from disdrodb.utils.directories import contains_files, contains_netcdf_or_parquet_files
from disdrodb.utils.yaml import read_yaml


def get_required_product(product):
    """Determine the required product for input product processing."""
    from disdrodb import PRODUCTS_REQUIREMENTS

    # Check input
    check_product(product)
    # Determine required product
    required_product = PRODUCTS_REQUIREMENTS[product]
    return required_product


####-------------------------------------------------------------------------
#### List DISDRODB infrastructure directories


def list_data_sources(metadata_archive_dir, data_sources=None, invalid_fields_policy="raise"):
    """List data sources names in the DISDRODB Metadata Archive."""
    available_data_sources = os.listdir(os.path.join(metadata_archive_dir, "METADATA"))
    # Filter by optionally specified data_sources
    if data_sources is not None:
        available_data_sources = check_valid_fields(
            fields=data_sources,
            available_fields=available_data_sources,
            field_name="data_sources",
            invalid_fields_policy=invalid_fields_policy,
        )
    # Return the unique data_sources
    return np.unique(available_data_sources).tolist()


def _list_campaign_names(metadata_archive_dir, data_source):
    data_source_dir = define_data_source_dir(metadata_archive_dir, product="METADATA", data_source=data_source)
    campaign_names = os.listdir(data_source_dir)
    return campaign_names


def list_campaign_names(
    metadata_archive_dir,
    data_sources=None,
    campaign_names=None,
    invalid_fields_policy="raise",
    return_tuple=False,
):
    """List campaign names in the DISDRODB Metadata Archive."""
    # Retrieve available data sources
    data_sources = list_data_sources(
        metadata_archive_dir,
        data_sources=data_sources,
        invalid_fields_policy=invalid_fields_policy,
    )

    # Retrieve (data_source, campaign_name) tuples
    list_tuples = [
        (data_source, campaign_name)
        for data_source in data_sources
        for campaign_name in _list_campaign_names(metadata_archive_dir=metadata_archive_dir, data_source=data_source)
    ]
    # Filter by optionally specified campaign_names
    if campaign_names is not None:
        available_campaign_names = [campaign_name for _, campaign_name in list_tuples]
        campaign_names = check_valid_fields(
            fields=campaign_names,
            available_fields=available_campaign_names,
            field_name="campaign_names",
            invalid_fields_policy=invalid_fields_policy,
        )

        list_tuples = [
            (data_source, campaign_name)
            for data_source, campaign_name in list_tuples
            if campaign_name in campaign_names
        ]

    # If specified, return just the list of (data_source, campaign_name) tuples
    if return_tuple:
        return list_tuples

    # Otherwise just return the unique campaign names
    campaign_names = [campaign_name for _, campaign_name in list_tuples]
    campaign_names = np.unique(campaign_names).tolist()
    return campaign_names


def _list_station_names(metadata_archive_dir, data_source, campaign_name):
    metadata_dir = define_metadata_dir(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    metadata_filenames = os.listdir(metadata_dir)
    station_names = [fname.replace(".yml", "").replace(".yaml", "") for fname in metadata_filenames]
    return station_names


def list_station_names(
    metadata_archive_dir,
    data_sources=None,
    campaign_names=None,
    station_names=None,
    invalid_fields_policy="raise",
    return_tuple=False,
):
    """List station names in the DISDRODB Metadata Archive."""
    # Retrieve (data sources - campaign_names) tuples
    list_tuples = list_campaign_names(
        metadata_archive_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        invalid_fields_policy=invalid_fields_policy,
        return_tuple=True,
    )

    # Retrieve (data_source, campaign_name, station_name) tuples
    list_info = [
        (data_source, campaign_name, station_name)
        for data_source, campaign_name in list_tuples
        for station_name in _list_station_names(
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
        )
    ]
    # Filter by optionally specified station_names
    if station_names is not None:
        available_station_names = [station_name for data_source, campaign_name, station_name in list_info]
        station_names = check_valid_fields(
            fields=station_names,
            available_fields=available_station_names,
            field_name="station_names",
            invalid_fields_policy=invalid_fields_policy,
        )
        list_info = [
            (data_source, campaign_name, station_name)
            for data_source, campaign_name, station_name in list_info
            if station_name in station_names
        ]

    # If specified, return just the list of (data_source, campaign_name, station_name) tuples
    if return_tuple:
        return list_info

    # Otherwise just return the unique station_names
    station_names = [station_name for _, _, station_name in list_info]
    station_names = np.unique(station_names).tolist()
    return station_names


def _finalize_output(list_info, return_tuple):
    # - Return the (data_source, campaign_name, station_name) tuple
    if return_tuple:
        return list_info
    # - Return list with the name of the available stations
    return [info[2] for info in list_info]


def _raise_an_error_if_no_stations(list_info, raise_error_if_empty, msg):
    if len(list_info) == 0 and raise_error_if_empty:
        raise ValueError(msg)


def is_disdrodb_data_url_specified(metadata_filepath):
    """Check if the disdrodb_data_url is specified in the metadata file."""
    disdrodb_data_url = read_yaml(metadata_filepath).get("disdrodb_data_url", "")
    return isinstance(disdrodb_data_url, str) and len(disdrodb_data_url) > 1


def keep_list_info_with_disdrodb_data_url(metadata_archive_dir, list_info):
    """Keep only the stations with disdrodb_data_url specified in the metadata file."""
    list_info_with_data = []
    for data_source, campaign_name, station_name in list_info:
        # Define metadata filepath
        metadata_filepath = define_metadata_filepath(
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        # Add station if disdrodb_data_url is specified
        if is_disdrodb_data_url_specified(metadata_filepath):
            list_info_with_data.append((data_source, campaign_name, station_name))
    return list_info_with_data


def keep_list_info_elements_with_product_directory(data_archive_dir, product, list_info):
    """Keep only the stations with the product directory."""
    list_info_with_product_directory = []
    for data_source, campaign_name, station_name in list_info:
        # Define station directory
        station_dir = define_station_dir(
            data_archive_dir=data_archive_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            check_exists=False,
        )
        # Add station if product station directory exists
        if os.path.isdir(station_dir):
            list_info_with_product_directory.append((data_source, campaign_name, station_name))
    return list_info_with_product_directory


def keep_list_info_elements_with_product_data(data_archive_dir, product, list_info, **product_kwargs):
    """Keep only the stations with product data."""
    # Define file checking function
    checking_function = contains_files if product == "RAW" else contains_netcdf_or_parquet_files

    # Check presence of data for each station
    # TODO: - In parallel over stations to speed up ?
    list_info_with_product_data = []
    for data_source, campaign_name, station_name in list_info:
        data_dir = define_data_dir(
            data_archive_dir=data_archive_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            check_exists=False,
            **product_kwargs,
        )
        if checking_function(data_dir):
            list_info_with_product_data.append((data_source, campaign_name, station_name))
    return list_info_with_product_data


####-------------------------------------------------------------------------
#### DISDRODB Search Routines


def available_stations(
    product=None,
    data_sources=None,
    campaign_names=None,
    station_names=None,
    return_tuple=True,
    available_data=False,
    raise_error_if_empty=False,
    invalid_fields_policy="raise",
    data_archive_dir=None,
    metadata_archive_dir=None,
    **product_kwargs,
):
    """
    Return stations information for which metadata or product data are available on disk.

    This function queries the DISDRODB Metadata Archive and, optionally, the
    local DISDRODB Data Archive to identify stations that satisfy the specified
    filters.

    If the DISDRODB product is not specified, it lists the stations present
    in the DISDRODB Metadata Archive given the specified filtering criteria.
    If the DISDRODB product is specified, it lists the stations present
    in the local DISDRODB Data Archive given the specified filtering criteria.

    Parameters
    ----------
    product : str or None, optional
        Name of the product to filter on (e.g., "RAW", "L0A", "L1").

        If the DISDRODB product is not specified (default),
        it lists the stations present in the DISDRODB Metadata Archive given the specified filtering criteria.

        If the DISDRODB product is specified,
        it lists the stations present in the local DISDRODB Data Archive given the specified filtering criteria.
        The default is is None.

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
    available_data : bool, optional

        If ``product`` is not specified:

            - if available_data is False, return stations present in the DISDRODB Metadata Archive
            - if available_data is True, return stations with data available on the
            online DISDRODB Decentralized Data Archive (i.e., stations with the disdrodb_data_url in the metadata).

        If ``product`` is specified:

            - if available_data is False, return stations where the product directory exists in the
              in the local DISDRODB Data Archive
            - if available_data is True, return stations where product data exists in the
              in the local DISDRODB Data Archive.
        The default is is False.

    return_tuple : bool, optional
        If True, return a list of tuples ``(data_source, campaign_name, station_name)``.
        If False, return only a list of station names
        The default is True.
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
        Path to the root of the local DISDRODB Data Archive.
        Required only if ``product``is specified.
        If None, the default data archive base directory is used. Default is None.
    metadata_archive_dir : str or Path-like, optional
        Path to the root of the DISDRODB Metadata Archive.
        If None, the default metadata base directory is used. Default is None.
    **product_kwargs : dict, optional
        Additional arguments required for some products.
        For example, for the "L2E" product, you need to specify ``rolling`` and
        ``sample_interval``. For the "L2M" product, you need to specify also
        the ``model_name``.

    Returns
    -------
    list
        If ``return_tuple=True``, return a list of tuples ``(data_source, campaign_name, station_name)``.
        If ``return_tuple=True``,, return a list of station names.

    Examples
    --------
    >>> # List all stations present in the DISDRODB Metadata Archive
    >>> stations = available_stations()
    >>> # List all stations present in the online DISDRODB Data Archive
    >>> stations = available_stations(available_data=True)
    >>> # List stations with raw data available in the local DISDRODB Data Archive
    >>> raw_stations = available_stations(product="RAW", available_data=True)
    >>> # List stations of specific data sources
    >>> stations = available_stations(data_sources=["GPM", "EPFL"])
    """
    # Retrieve DISDRODB Data and Metadata Archive directories
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)
    product = check_product(product) if product is not None else None
    invalid_fields_policy = check_invalid_fields_policy(invalid_fields_policy)
    # Retrieve available stations from the Metadata Archive
    # - Raise error if no stations availables !
    list_info = list_station_names(
        metadata_archive_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        invalid_fields_policy=invalid_fields_policy,
        return_tuple=True,
    )

    # Return stations in the Metadata Archive
    if product is None and not available_data:
        _raise_an_error_if_no_stations(
            list_info,
            raise_error_if_empty=raise_error_if_empty,
            msg="No station available in the DISDRODB Metadata Archive.",
        )
        return _finalize_output(list_info, return_tuple=return_tuple)

    # Return stations in the Metadata Archive with specified disdrodb_data_url
    if product is None and available_data:
        list_info = keep_list_info_with_disdrodb_data_url(metadata_archive_dir, list_info)
        _raise_an_error_if_no_stations(
            list_info,
            raise_error_if_empty=raise_error_if_empty,
            msg="No station has the disdrodb_data_url specified in the metadata.",
        )
        return _finalize_output(list_info, return_tuple=return_tuple)

    # If product is specified, select stations available in the local DISDRODB Data Archive
    # - If available_data=False, search for station with the existing product directory (do not check for data)
    data_archive_dir = get_data_archive_dir(data_archive_dir)
    product = check_product(product)
    if not available_data:
        list_info = keep_list_info_elements_with_product_directory(
            data_archive_dir=data_archive_dir,
            product=product,
            list_info=list_info,
        )
        _raise_an_error_if_no_stations(
            list_info,
            raise_error_if_empty=raise_error_if_empty,
            msg=f"No station product {product} directory available in the local DISDRODB Data Archive.",
        )
        return _finalize_output(list_info, return_tuple=return_tuple)

    # - If available_data=True, search for station with product data
    product_kwargs = check_product_kwargs(product, product_kwargs)
    list_info = keep_list_info_elements_with_product_data(
        data_archive_dir=data_archive_dir,
        product=product,
        list_info=list_info,
        **product_kwargs,
    )
    product_kwargs = product_kwargs if product_kwargs else ""  # if empty, set as ""
    _raise_an_error_if_no_stations(
        list_info,
        raise_error_if_empty=raise_error_if_empty,
        msg=f"No station has {product} {product_kwargs} data available in the local DISDRODB Data Archive.",
    )
    return _finalize_output(list_info, return_tuple=return_tuple)


def available_data_sources(
    product=None,
    campaign_names=None,
    station_names=None,
    available_data=False,
    raise_error_if_empty=False,
    invalid_fields_policy="raise",
    data_archive_dir=None,
    metadata_archive_dir=None,
    **product_kwargs,
):
    """Return data sources for which stations are available."""
    list_info = available_stations(
        product=product,
        data_sources=None,
        campaign_names=campaign_names,
        station_names=station_names,
        return_tuple=True,
        available_data=available_data,
        raise_error_if_empty=raise_error_if_empty,
        invalid_fields_policy=invalid_fields_policy,
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        **product_kwargs,
    )
    data_sources = [info[0] for info in list_info]
    data_sources = np.unique(data_sources).tolist()
    return data_sources


def available_campaigns(
    product=None,
    data_sources=None,
    station_names=None,
    available_data=False,
    raise_error_if_empty=False,
    invalid_fields_policy="raise",
    data_archive_dir=None,
    metadata_archive_dir=None,
    **product_kwargs,
):
    """Return campaigns names for which stations are available."""
    list_info = available_stations(
        product=product,
        data_sources=data_sources,
        campaign_names=None,
        station_names=station_names,
        return_tuple=True,
        available_data=available_data,
        raise_error_if_empty=raise_error_if_empty,
        invalid_fields_policy=invalid_fields_policy,
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        **product_kwargs,
    )
    campaign_names = [info[1] for info in list_info]
    campaign_names = np.unique(campaign_names).tolist()
    return campaign_names
