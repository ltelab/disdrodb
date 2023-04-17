import os
import pooch
import tqdm

from typing import Union, Optional, List

from disdrodb.api.metadata import _read_yaml_file, get_list_metadata_file
from disdrodb.utils.zip import _unzip_file


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

    if station_name and str(station_name) != str(expected_station_name):
        return None, None, None

    # Get data url
    station_remote_url = metadata_dict.get("data_url")

    # Get the local path
    data_dir_path = os.path.dirname(yaml_file_path).replace("metadata", "data")

    return data_dir_path, station_name, station_remote_url


def _download_file_from_url(url: str, dir_path: str, force: bool = False) -> str:
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
        return file_path
    elif force and os.path.isfile(file_path):
        os.remove(file_path)

    downloader = pooch.HTTPDownloader(progressbar=True)
    pooch.retrieve(url=url, known_hash=None, path=dir_path, fname=fname, downloader=downloader, progressbar=tqdm)

    return file_path


def _download_station_data(metadata_fpaths: str, force: bool = False) -> None:
    """Download the station data.

    Parameters
    ----------
    metadata_fpaths : str
        Metadata file path.
    force : bool, optional
        force download, by default False

    """

    for metadata_fpath in metadata_fpaths:
        print("metadata_fpath", metadata_fpath)
        location_info = get_station_local_remote_locations(metadata_fpath)

        if None not in location_info:
            data_dir_path, station_name, data_url = location_info
            url_file_name, url_file_extension = os.path.splitext(os.path.basename(data_url))
            os.path.join(data_dir_path, url_file_name)
            temp_zip_path = _download_file_from_url(data_url, data_dir_path, force)

            if temp_zip_path and url_file_extension.endswith(".zip"):
                _unzip_file(temp_zip_path, os.path.join(data_dir_path, str(station_name)))
                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
            else:
                print(f"File extension {url_file_extension} is not supported. Should be a zip file.")


def download_disdrodb_archives(
    disdrodb_dir: str,
    data_sources: Optional[Union[str, List[str]]] = None,
    campaign_names: Optional[Union[str, List[str]]] = None,
    station_names: Optional[Union[str, List[str]]] = None,
    force: bool = False,
):
    """Get all YAML files that contain the 'data_url' key
    and download the data locally.

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

    metadata_fpath = get_list_metadata_file(disdrodb_dir, data_sources, campaign_names, station_names, False)

    _download_station_data(metadata_fpath, force)
