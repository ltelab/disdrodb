import os

try:
    import pooch
    import tqdm
except ImportError:
    raise ImportError('Please install disdrodb with "dev" dependencies to use this module.')

from disdrodb.api.checks import check_url
from disdrodb.api.io import _get_disdrodb_directory
from disdrodb.api.metadata import _read_yaml_file, get_metadata_list


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

    # Check name
    expected_name = os.path.basename(yaml_file_path).replace(".yml", "")
    name = metadata_dict.get("station_name")
    if name and name != expected_name:
        return None, None

    # Get data url
    url = metadata_dict.get("data_url")

    # Get the local path
    data_dir_path = os.path.dirname(yaml_file_path).replace("metadata", "data")
    station_dir_path = os.path.join(data_dir_path, name)

    return station_dir_path, url


def _get_local_and_remote_data_directories(
    disdrodb_dir: str,
    data_source: str = "",
    campaign_name: str = "",
    station_name: str = "",
) -> list:
    """Parse the folder according to the parameters (data_source,
    campaign_name and station_name) to get all url from the config files.

    Returns
    -------
    list
        List of tuples containing the local path and the url.
    """

    # Test if the file path is correct and exists. If not, raise an error
    _get_disdrodb_directory(disdrodb_dir, "RAW", data_source, campaign_name)

    # Get all config files from the metadata folders
    yaml_files = get_metadata_list(disdrodb_dir, data_source, campaign_name, station_name)

    # Get the list of url and local paths from the config files
    list_of_path_url = []
    for yaml_file_path in yaml_files:
        station_dir_path, data_url = get_station_local_remote_locations(yaml_file_path)
        if data_url is None:
            continue
        list_of_path_url.append((station_dir_path, data_url))

    return list_of_path_url


def download_file_from_url(url: str, dir_path: str) -> None:
    """Download file.

    Parameters
    ----------

    url : str
        URL of the file to download.
    file_name : str
        Local name of the file.
    """

    fname = os.path.basename(url)

    downloader = pooch.HTTPDownloader(progressbar=True)
    pooch.retrieve(
        url=url,
        known_hash=None,
        path=dir_path,
        fname=fname,
        downloader=downloader,
        progressbar=tqdm,
    )


def download_single_archive(url_local_path: tuple, overwrite: bool = False) -> None:
    """Download the files based on a tuple (local path, url).

    Parameters
    ----------
    file_path : tuple
        Tuple containing the local file path and the url.
    overwrite : bool, optional
        Overwrite the raw data file if already existing, by default False.

    """

    local_path, url = url_local_path
    if check_url(url):
        # create station folder if not existing
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        # download the file
        download_file_from_url(url, local_path)
        print(f"Download {url} into {local_path}")

    else:
        # raise an error if url not well formatted
        raise ValueError(f"URL {url} is not well formatted. Please check the URL.")


def download_disdrodb_archives(
    disdrodb_dir: str = None,
    data_source: str = None,
    campaign_name: str = None,
    station_name: str = None,
    overwrite: bool = False,
):
    """Batch function to get all YAML files that contain
    the 'data_url' key and download the data locally.

    Parameters
    ----------
    disdrodb_dir : str, optional
        DisdroDB data folder path.
        Must end with DISDRODB/Raw.
    data_source : str, optional
        Data source folder name (eg : EPFL).
        If not provided (None), all data sources will be downloaded.
        The default is data_source=None.
    campaign_name : str, optional
        Campaign name (eg :  EPFL_ROOF_2012).
        If not provided (None), all campaigns will be downloaded.
        The default is campaign_name=None.
    station_name : str, optional
        Station name.
        If not provided (None), all stations will be downloaded.
        The default is station_name=None.
    overwrite : bool, optional
        If True, overwrite the already existing raw data file.
        The default is False.
    """

    list_urls_data_dir = _get_local_and_remote_data_directories(disdrodb_dir, data_source, campaign_name, station_name)

    for url_local_path in list_urls_data_dir:
        download_single_archive(url_local_path, overwrite)
