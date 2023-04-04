import os
import re
import pooch
import yaml

import tqdm

from disdrodb.api import io


# function to check if a path exists
def check_path(path: str) -> None:
    """Check if a path exists.

    Parameters
    ----------
    path : str
        Path to check.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist. Please check the path.")


def get_metadata_dirs(dir_path: str) -> list:
    """Return a list containing all metadata folders paths.


    Parameters
    ----------
    dir_path : str
        Folder path.

    Returns
    -------
    list
        List of metadata folders.

    """

    return [os.path.join(root, "metadata") for root, dirs, files in os.walk(dir_path) if "metadata" in dirs]


def get_yaml_files(metadata_dirs: str) -> list:
    """Return a list containing all YAML files paths.


    Parameters
    ----------
    metadata_dirs : str
        Folder path.

    Returns
    -------
    list
        List of yaml files.
    """

    return [
        os.path.join(root, file)
        for metadata_dir in metadata_dirs
        for root, dirs, files in os.walk(metadata_dir)
        for file in files
        if file.endswith(".yml")
    ]


def get_sation_name_and_url_from_metadata(yaml_file_path: str) -> dict:
    """Return the station name and the url from the metadata folder.

    Parameters
    ----------
    yaml_file_path : str
        Path to the metadata yaml file.

    Returns
    -------
    dict
        dictionnary containing the station name and the url.
    """

    result_dict = {}
    with open(yaml_file_path) as f:
        metadata_dict = yaml.safe_load(f)

        result_dict = {k: metadata_dict[k] for k in ["data_url", "station_name"] if k in metadata_dict}

    return result_dict


def _get_remote_and_local_data_directories(
    disdrodb_dir: str = None,
    data_source: str = None,
    campaign_name: str = None,
    station_name: str = None,
) -> list:
    """Parse the folder according to the parameters (data_source,

    campaign_name and station_name) to get all url from the config files.

    Returns
    -------
    list
        List of tuples containing the local path and the url.
    """

    # raise an error if the path does not ends with "DISDRODB"
    if not disdrodb_dir.endswith("DISDRODB"):
        raise ValueError(f"The path {disdrodb_dir} does not end with DISDRODB. Please check the path.")

    # test if the file path is correct and exists. If not, raise an error
    kwargs = {}
    if data_source:
        kwargs["data_source"] = data_source
    if campaign_name:
        kwargs["campaign_name"] = campaign_name
    path_to_analyse = io._get_disdrodb_directory(disdrodb_dir, "RAW", **kwargs)

    # Get all config files from the metadata folders
    metadata_dir = get_metadata_dirs(path_to_analyse)
    yaml_files = get_yaml_files(metadata_dir)

    # Get the list of url and local paths from the config files
    list_of_path_url = []
    for yaml_file_path in yaml_files:
        station_name_url = get_sation_name_and_url_from_metadata(yaml_file_path)
        # Only if the url exists
        if station_name_url.get("data_url"):
            # Only if the station name is not specified or if the station name is the same as the config file name
            if not station_name or station_name == station_name_url.get("station_name"):
                metadata_dir_path = os.path.dirname(yaml_file_path).replace("metadata", "data")
                station_dir_path = os.path.join(metadata_dir_path, station_name_url.get("station_name"))
                data_url = station_name_url.get("data_url")
                list_of_path_url.append((station_dir_path, data_url))

    return list_of_path_url


def check_url(url: str) -> bool:
    """Check url.

    Parameters
    ----------
    url : str
        URL to check.

    Returns
    -------
    bool
        True if url well formated, False if not well formated.
    """
    regex = r"^(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$"  # noqa: E501

    if re.match(regex, url):
        return True
    else:
        return False


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
    """Download the files based on a tuple (local path , url).

    Parameters
    ----------
    file_path : tuple
        Tuple containing the local file path and the url.
    overwrite : bool, optional
        Overwrite the raw data file is already existing, by default False.

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
        # raise an error if url not well formated
        raise ValueError(f"URL {url} is not well formated. Please check the URL.")


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

    list_urls_data_dir = _get_remote_and_local_data_directories(disdrodb_dir, data_source, campaign_name, station_name)

    for url_local_path in list_urls_data_dir:
        download_single_archive(url_local_path, overwrite)
