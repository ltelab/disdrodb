import os
import pooch
import tqdm
import glob

from typing import Union

from disdrodb.api.checks import check_url
from disdrodb.api.metadata import _read_yaml_file


def get_station_local_remote_locations(yaml_file_path: str) -> tuple:
    """Return the station's local path and remote url.

    Parameters
    ----------
    yaml_file_path : str
        Path to the metadata YAML file.

    Returns
    -------
    tuple
        Tuple containing the local path and the url.
    """

    metadata_dict = _read_yaml_file(yaml_file_path)

    # Check station name

    expected_station_name = os.path.basename(yaml_file_path).replace(".yml", "")

    station_name = metadata_dict.get("station_name")

    if station_name and station_name != expected_station_name:
        return None, None

    # Get data url
    station_remote_url = metadata_dict.get("data_url")

    # Get the local path
    data_dir_path = os.path.dirname(yaml_file_path).replace("metadata", "data")
    station_dir_path = os.path.join(data_dir_path, station_name)

    return station_dir_path, station_remote_url


def _get_local_and_remote_data_directories(
    disdrodb_dir: str,
    data_sources: Union[str, list[str]] = None,
    campaign_names: Union[str, list[str]] = None,
    station_names: Union[str, list[str]] = None,
) -> list:
    """Parse the folder according to the parameters (data_source,
    campaign_name and station_name) to get all url from the config files.

    Parameters
    ----------
    disdrodb_dir : str , optional
        DisdroDB data folder path.
        Must end with DISDRODB.
    data_sources : str or list of str, optional
        Data source folder name (eg : EPFL).
        If not provided (None), all data sources will be downloaded.
        The default is data_source=None.
    campaign_names : str or list of str, optional
        Campaign name (eg :  EPFL_ROOF_2012).
        If not provided (None), all campaigns will be downloaded.
        The default is campaign_name=None.
    station_names : str or list of str, optional
        Station name.
        If not provided (None), all stations will be downloaded.
        The default is station_name=None.

    Returns
    -------
    list
        List of tuples containing the local path and the url.
    """

    # Get all config files from the metadata folders
    list_of_base_path = []
    if data_sources:
        if isinstance(data_sources, str):
            data_sources = [data_sources]
    else:
        data_sources = ["**"]
    if campaign_names:
        if isinstance(campaign_names, str):
            campaign_names = [campaign_names]
    else:
        campaign_names = ["**"]

    for data_source in data_sources:
        for campaign_name in campaign_names:
            base_path = os.path.join(disdrodb_dir, "Raw", data_source, campaign_name)
            list_of_base_path.append(base_path)

    metadata_folder_name = "metadata"

    list_metadata_folders = []

    for base_path in list_of_base_path:
        if station_names:
            if isinstance(station_names, str):
                station_names = [station_names]
            for station_name in station_names:
                metadata_path = os.path.join(base_path, "**", metadata_folder_name, f"{station_name}.yml")
                list_metadata_folders += glob.glob(metadata_path, recursive=True)
        else:
            metadata_path = os.path.join(base_path, "**", metadata_folder_name, "*.yml")
            list_metadata_folders += glob.glob(metadata_path, recursive=True)

    list_of_path_url = list()
    for yaml_file_path in list_metadata_folders:
        station_dir_path, data_url = get_station_local_remote_locations(yaml_file_path)
        if data_url is None:
            continue
        list_of_path_url.append((station_dir_path, data_url))

    return list_of_path_url


def _download_file_from_url(url: str, dir_path: str, force: bool = False) -> None:
    """Download file.

    Parameters
    ----------

    url : str
        URL of the file to download.
    dir_path : str
        Dir path where to download the file.
    force : bool, optional
        Overwrite the raw data file if already existing, by default False.

    """

    fname = os.path.basename(url)
    file_path = os.path.join(dir_path, fname)

    if not force and os.path.isfile(file_path):
        print(f"{file_path} already exists, skipping download.")
        return
    elif force and os.path.isfile(file_path):
        os.remove(file_path)

    downloader = pooch.HTTPDownloader(progressbar=True)
    pooch.retrieve(url=url, known_hash=None, path=dir_path, fname=fname, downloader=downloader, progressbar=tqdm)


def _download_station_data(url_local_path: tuple, force: bool = False) -> None:
    """Download the files based on a tuple (local path, url).
    Parameters
    ----------
    file_path : tuple
        Tuple containing the local file path and the url.
    force : bool, optional
        Overwrite the raw data file if already existing, by default False.
    """

    local_path, url = url_local_path
    if check_url(url):
        # create station folder if not existing
        os.makedirs(local_path, exist_ok=True)
        # download the file
        _download_file_from_url(url, local_path, force)
        print(f"Download {url} into {local_path}")

    else:
        print(f"URL {url} is not well formatted. Please check the URL.")


def download_disdrodb_archives(
    disdrodb_dir: str = None,
    data_sources: Union[str, list[str]] = None,
    campaign_names: Union[str, list[str]] = None,
    station_names: Union[str, list[str]] = None,
    force: bool = False,
):
    """Batch function to get all YAML files that contain
    the 'data_url' key and download the data locally.

    Parameters
    ----------
    disdrodb_dir : str, optional
        DisdroDB data folder path.
        Must end with DISDRODB.
    data_sources : str or list of str, optional
        Data source folder name (eg : EPFL).
        If not provided (None), all data sources will be downloaded.
        The default is data_source=None.
    campaign_names : str or list of str, optional
        Campaign name (eg :  EPFL_ROOF_2012).
        If not provided (None), all campaigns will be downloaded.
        The default is campaign_name=None.
    station_names : str or list of str, optional
        Station name.
        If not provided (None), all stations will be downloaded.
        The default is station_name=None.
    force : bool, optional
        If True, overwrite the already existing raw data file.
        The default is False.
    """
    list_urls_data_dir = _get_local_and_remote_data_directories(
        disdrodb_dir, data_sources, campaign_names, station_names
    )

    for url_local_path in list_urls_data_dir:
        _download_station_data(url_local_path, force)
