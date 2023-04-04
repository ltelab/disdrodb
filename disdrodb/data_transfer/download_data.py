import os
import re
import zipfile
import pooch
import yaml

import click
import tqdm


# function to check if a path exists
def check_path(path: str) -> None:
    """Check if a path exists

    Parameters
    ----------
    path : str
        Path to check

    Raises
    ------
    FileNotFoundError
        If the path does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist. Please check the path.")


def get_metadata_folders(folder_path: str) -> list:
    """Return a list containing all metadata folders paths

    Parameters
    ----------
    folder_path : str
        folder path

    Returns
    -------
    list
        list of metadata folders
    """

    return [os.path.join(root, "metadata") for root, dirs, files in os.walk(folder_path) if "metadata" in dirs]


def get_yaml_files(metadata_folders: str) -> list:
    """Return a list containing all yaml files paths

    Parameters
    ----------
    metadata_folders : str
        folder path

    Returns
    -------
    list
        list of yaml files
    """

    return [
        os.path.join(root, file)
        for metadata_folder in metadata_folders
        for root, dirs, files in os.walk(metadata_folder)
        for file in files
        if file.endswith(".yml")
    ]


def get_list_urls_local_paths(
    raw_folder: str = None,
    data_source: str = None,
    campaign_name: str = None,
    station_name: str = None,
) -> list:
    """Parse the folder according to the paramerters (data_source,
    campaign_name and station_name) to get all url from the config files.

    Returns
    -------
    list
        List of tuples containing the local path and the url.
    """
    # test if the file path is correct and exists. If not, raise an error
    check_path(raw_folder)

    # Test that the path ends with DISDRODB/Raw. If not, raise an error
    split_path = raw_folder.split(os.sep)
    if split_path[-1] != "Raw" or split_path[-2] != "DISDRODB":
        raise ValueError("The path must end with DISDRODB/Raw. Please check the path.")

    # Define the path to analyse
    path_to_analyse = raw_folder
    if data_source:
        path_to_analyse = os.path.join(path_to_analyse, data_source)
    if campaign_name:
        path_to_analyse = os.path.join(path_to_analyse, campaign_name)

    # Get all config files from the metadata folders
    metadata_folders = get_metadata_folders(path_to_analyse)
    yaml_files = get_yaml_files(metadata_folders)

    # Get the list of url and local paths from the config files
    list_of_path_url = []
    for yaml_file_path in yaml_files:
        file_content = yaml.safe_load(open(yaml_file_path))
        if "data_url" in file_content.keys():
            data_url = file_content["data_url"]
            config_station_name = os.path.basename(yaml_file_path).replace(".yml", "")
            # Only if the station name is not specified or if the station name is the same as the config file name
            if not station_name or station_name == config_station_name:
                folder_path = os.path.dirname(yaml_file_path).replace("metadata", "data")
                file_path = os.path.join(folder_path, config_station_name)
                list_of_path_url.append((file_path, data_url))

    return list_of_path_url


def check_url(url: str) -> bool:
    """Check url

    Parameters
    ----------
    url : str
        url

    Returns
    -------
    bool
        True if url well formated, False if not well formated
    """
    regex = r"^(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$"  # noqa: E501

    if re.match(regex, url):
        return True
    else:
        return False


def download_file_from_url(url: str, folder_path: str) -> None:
    """Download file

    Parameters
    ----------

    url : str
        URL of the file to download
    file_name : str
        Local name of the file
    """

    file_name = os.path.basename(url)

    downloader = pooch.HTTPDownloader(progressbar=True)
    pooch.retrieve(
        url=url,
        known_hash=None,
        path=folder_path,
        fname=file_name,
        downloader=downloader,
        progressbar=tqdm,
    )


# function to unzip one file into a folder
def unzip_file(file_path: str, dest_path: str) -> None:
    """Unzip a file into a folder

    Parameters
    ----------
    file_path : str
        Path of the file to unzip
    dest_path : str
        Path of the destination folder
    """

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dest_path)


def download_file_content(url_local_path: tuple, overwrite: bool = False, unzip: bool = False) -> None:
    """Download the files based on a tuple (local path , url).
    Unzip the file if it is a zip file and if requested.

    Parameters
    ----------
    file_path : tuple
        Tuple containing the local file path and the url
    overwrite : bool, optional
        Overwrite the raw data file is already existing, by default False
    unzip : bool, optional
        Unzip the file if it is a zip file, by default False
    """

    local_path, url = url_local_path
    if check_url(url):
        # create station folder if not existing
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        # download the file
        download_file_from_url(url, local_path)
        print(f"Download {url} into {local_path}")
        if url.endswith(".zip") and unzip:
            zip_local_path = os.path.join(local_path, os.path.basename(url))
            unzip_file(zip_local_path, local_path)
            os.remove(zip_local_path)

    else:
        # raise an error if url not well formated
        raise ValueError(f"URL {url} is not well formated. Please check the URL.")


def download_all_files(
    raw_folder: str = None,
    data_source: str = None,
    campaign_name: str = None,
    station_name: str = None,
    overwrite: bool = False,
    unzip: bool = False,
):
    """Batch function to get all yaml files that coantain
    the 'data_url' key and download the data locally.

    Parameters
    ----------
    raw_folder : str, optional
        Raw data folder path.
        Must end with DISDRODB/Raw
    data_source : str, optional
        Data source folder name (eg : EPFL).
        If not provided (None), all data sources will be downloaded.
        The default is data_source=None
    campaign_name : str, optional
        Campaign name (eg :  EPFL_ROOF_2012).
        If not provided (None), all campaigns will be downloaded.
        The default is campaign_name=None
    station_name : str, optional
        Station name.
        If not provided (None), all stations will be downloaded.
        The default is station_name=None
    overwrite : bool, optional
        If True, overwrite the already existing raw data file.
        The default is False.
    unzip : bool, optional
        If True, unzip the zip file.
        The default is False.
    """

    list_urls_local_paths = get_list_urls_local_paths(raw_folder, data_source, campaign_name, station_name)

    for url_local_path in list_urls_local_paths:
        download_file_content(url_local_path, overwrite, unzip)


@click.command()
@click.option(
    "--raw_folder",
    required=True,
    help="Raw data folder path (eg : /home/user/DISDRODB/Raw). Is compulsory.",
)
@click.option(
    "--data_source",
    help="Data source folder name (eg : EPFL). If not provided (None), all data sources will be downloaded.",
)
@click.option(
    "--campaign_name",
    help="Name of the campaign (eg :  EPFL_ROOF_2012). If not provided (None), all campaigns will be downloaded.",
)
@click.option(
    "--station_name",
    help="Station name. If not provided (None), all stations will be downloaded.",
)
@click.option("--overwrite", type=bool, help="Overwite existing file ?")
@click.option("--unzip", type=bool, help="Unzip  existing file ?")
def download_file(
    raw_folder=None,
    data_source=None,
    campaign_name=None,
    station_name=None,
    overwrite=False,
    unzip=False,
):
    download_all_files(raw_folder, data_source, campaign_name, station_name, overwrite, unzip)
