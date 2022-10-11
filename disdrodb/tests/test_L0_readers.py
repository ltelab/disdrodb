import os
import shutil
import requests
import subprocess
import zipfile
import glob
import numpy as np
import xarray as xr
from netCDF4 import Dataset


from disdrodb.L0.run_disdrodb_l0_reader import run_reader

LOCAL_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp")
GITHUB_REPO_NAME = "LTE-disdrodb-testing"
GITHUB_REPO_OWNER = "EPFL-ENAC"
GITHUB_URL = "https://github.com/EPFL-ENAC/LTE-disdrodb-testing"


def downlaod_file(url: str, local_path: str) -> None:
    """Download files.


    Parameters
    ----------
    url : str
        Url
    local_path : str
        Local path
    """
    req = requests.get(url)
    with open(local_path, "wb") as output_file:
        output_file.write(req.content)


def create_and_testing_folder_structure(path_reader: str) -> dict:
    """Create a testing folder and return available testing resources.


    Parameters
    ----------
    path_reader : str
        Reader file path.


    Returns
    -------
    dict
        Available testing resources {'data_source':['campaign']} .

    """

    list_zip_files_names = ["raw", "processed"]

    list_campaigns_paths = glob.glob(os.path.join(path_reader, "*", "*"))
    dict_available_testing_items = {}

    # Iterate through campaigns
    for campaign_path in list_campaigns_paths:
        zip_files = [
            f
            for f in os.listdir(campaign_path)
            if f in [f"{i}.zip" for i in list_zip_files_names]
        ]

        # Check if the folder contains  "raw.zip" AND "processed.zip" files
        if sorted(zip_files) == sorted([f"{i}.zip" for i in list_zip_files_names]):
            campaign_name = os.path.basename(campaign_path)
            data_source_name = os.path.basename(os.path.dirname(campaign_path))

            # Unzip files
            for item in list_zip_files_names:
                source = os.path.join(campaign_path, f"{item}.zip")
                dst_dir = os.path.join(

                    LOCAL_FOLDER,
                    "testing_files",
                    "DISDRODB",
                    item.title(),
                    data_source_name,
                    campaign_name,
                )
                with zipfile.ZipFile(source, "r") as zipObj:
                    zipObj.extractall(dst_dir)


            # Populate the dictionnary of available testing items
            if not data_source_name in dict_available_testing_items.keys():
                dict_available_testing_items[data_source_name] = []
            dict_available_testing_items[data_source_name].append(campaign_name)

    return dict_available_testing_items


def run_reader_on_test_data(data_source: str, campaign_name: str) -> None:
    """Run reader on test data.


    Parameters
    ----------
    data_source : str
        Data source.

    campaign_name : str
        Name of the campaign.

    """

    raw_dir = os.path.join(
        LOCAL_FOLDER, "testing_files", "DISDRODB", "Raw", data_source, campaign_name
    )
    processed_dir = os.path.join(
        LOCAL_FOLDER,
        "test_results",
        "DISDRODB",
        "Processed",
        data_source,
        campaign_name,
    )
    l0a_processing = True
    l0b_processing = True
    keep_l0a = True
    force = True
    verbose = True
    debugging_mode = True
    lazy = False
    single_netcdf = True

    run_reader(
        data_source,
        campaign_name,
        raw_dir,
        processed_dir,
        l0a_processing,
        l0b_processing,
        keep_l0a,
        force,
        verbose,
        debugging_mode,
        lazy,
        single_netcdf,
    )


def are_netcdf_identical(dataset_1: Dataset, dataset_2: Dataset) -> bool:

    """Check if two NetCDF are identical.


    Parameters
    ----------
    dataset_1 : Dataset
        First NetCDF
    dataset_2 : Dataset
        Second NetCDF

    Returns
    -------
    bool
        True if identical,
        False if not indentical
    """

    is_identical = True

    # Check dimension keys
    if is_identical and dataset_1.dimensions.keys() != dataset_2.dimensions.keys():
        is_identical = False

    # Check variables keys
    if is_identical and dataset_1.variables.keys() != dataset_2.variables.keys():
        is_identical = False

    # Check dimension content
    if is_identical:
        for dimension_name in dataset_2.dimensions.keys():
            dimension_result = dataset_1.variables[dimension_name][:]
            dimension_ground_thruth = dataset_2.variables[dimension_name][:]
            if not np.array_equal(dimension_ground_thruth, dimension_result):
                is_identical = False
                print(f"The dimension '{dimension_name}' does not match.")

    # Check variable content
    if is_identical:
        for variable_name in dataset_2.variables.keys():
            variable_result = dataset_1.variables[variable_name][:]
            variable_ground_truth = dataset_2.variables[variable_name][:]
            if not np.array_equal(variable_result, variable_ground_truth):
                is_identical = False
                print(f"The variable '{variable_name}' does not match")

    return is_identical


def is_reader_results_similar_to_ground_truth(
    data_source: str, campaign_name: str
) -> bool:
    """Test if the reader execution returns the same result as the ground truth.

    Parameters
    ----------
    data_source : str
        Data source.

    campaign_name : str
        Name of the campaign.


    Returns
    -------
    bool
        True if reader execution returns same result as ground truth
        False if the results of the reader execution diverge from the ground truth.
    """

    ground_truth_folder = os.path.join(
        LOCAL_FOLDER,
        "testing_files",
        "DISDRODB",
        "Processed",
        data_source,
        campaign_name,
        "L0B",
    )
    list_of_ground_truth_files_paths = glob.glob(
        os.path.join(ground_truth_folder, "*", "*.nc")
    )

    result_folder = os.path.join(
        LOCAL_FOLDER,
        "test_results",
        "DISDRODB",
        "Processed",
        data_source,
        campaign_name,
        "L0B",
    )
    list_of_test_result_files_paths = glob.glob(
        os.path.join(result_folder, "*", "*.nc")
    )

    is_reader_result_similar_to_ground_truth = True

    for ground_truth_file_path in list_of_ground_truth_files_paths:
        groud_truth_file_name = os.path.basename(ground_truth_file_path)
        ground_truth_station_name = os.path.basename(
            os.path.dirname(ground_truth_file_path)
        )
        for result_file_path in list_of_test_result_files_paths:
            result_file_name = os.path.basename(result_file_path)
            result_station_name = os.path.basename(os.path.dirname(result_file_path))
            if ground_truth_station_name == result_station_name:
                # Load datasets
                dataset_1 = Dataset(ground_truth_file_path)
                dataset_2 = Dataset(result_file_path)
                if not are_netcdf_identical(dataset_1, dataset_2):

                    is_reader_result_similar_to_ground_truth = False
                    print(f"{result_file_name} does not match {groud_truth_file_name}")

    return is_reader_result_similar_to_ground_truth


def downlaod_test_ressources() -> str:
    """Download testing resource files.


    Returns
    -------
    str
        Local path.
    """
    url = f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/archive/refs/heads/main.zip"
    path_local = os.path.join(LOCAL_FOLDER, "main.zip")
    downlaod_file(url, path_local)

    with zipfile.ZipFile(path_local, "r") as f:
        f.extractall(LOCAL_FOLDER)
    for path_folder, folder_name, list_file in os.walk(LOCAL_FOLDER):
        if path_folder.endswith("readers"):
            return path_folder


def test_readers():

    if os.path.exists(LOCAL_FOLDER):
        shutil.rmtree(LOCAL_FOLDER)
    os.mkdir(LOCAL_FOLDER)

    path_reader = downlaod_test_ressources()
    dict_available_readers = create_and_testing_folder_structure(path_reader)

    for data_source, list_campaigns in dict_available_readers.items():

        for campaign in list_campaigns:
            msg = (
                f"Start test for data source '{data_source}' and campaign '{campaign}' "
            )
            print(msg)
            run_reader_on_test_data(data_source, campaign)
            result = is_reader_results_similar_to_ground_truth(data_source, campaign)
            assert result
            msg = f"End test for data source '{data_source}' and campaign '{campaign}' "
            print(msg)


if __name__ == "__main__":
    test_readers()
