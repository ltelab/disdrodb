import glob
import os
import shutil

import pytest
from click.testing import CliRunner

# current file path
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PACKAGE_DIR, "tests", "pytest_files", "check_readers", "DISDRODB")
DATA_SOURCE = "EPFL"
CAMPAIGN_NAME = "PARSIVEL_2007"
STATION_NAME = "10"


@pytest.fixture
def remove_processed_folder(request: list) -> None:
    path_processed_folder = os.path.join(RAW_DIR, "Processed")
    if os.path.exists(path_processed_folder):
        shutil.rmtree(path_processed_folder)
    yield
    if os.path.exists(path_processed_folder):
        shutil.rmtree(path_processed_folder)


@pytest.mark.parametrize("remove_processed_folder", [()], indirect=True)
def test_run_disdrodb_l0a_station(remove_processed_folder):
    """Test the run_disdrodb_l0a_station command."""

    from disdrodb.l0.scripts.run_disdrodb_l0a_station import run_disdrodb_l0a_station

    runner = CliRunner()
    runner.invoke(run_disdrodb_l0a_station, [RAW_DIR, DATA_SOURCE, CAMPAIGN_NAME, STATION_NAME])
    list_of_l0a = glob.glob(
        os.path.join(RAW_DIR, "Processed", DATA_SOURCE, CAMPAIGN_NAME, "L0A", STATION_NAME, "*.parquet")
    )
    assert len(list_of_l0a) > 0


@pytest.mark.parametrize("remove_processed_folder", [()], indirect=True)
def test_run_disdrodb_l0a(remove_processed_folder):
    """Test the run_disdrodb_l0a command."""

    from disdrodb.l0.scripts.run_disdrodb_l0a import run_disdrodb_l0a

    runner = CliRunner()
    runner.invoke(run_disdrodb_l0a, [RAW_DIR])
    list_of_l0a = glob.glob(
        os.path.join(RAW_DIR, "Processed", DATA_SOURCE, CAMPAIGN_NAME, "L0A", STATION_NAME, "*.parquet")
    )
    assert len(list_of_l0a) > 0


@pytest.mark.parametrize("remove_processed_folder", [()], indirect=True)
def test_run_disdrodb_l0b_station(remove_processed_folder):
    """Test the run_disdrodb_l0b_station command."""
    from disdrodb.l0.scripts.run_disdrodb_l0a_station import run_disdrodb_l0a_station

    runner = CliRunner()
    runner.invoke(run_disdrodb_l0a_station, [RAW_DIR, DATA_SOURCE, CAMPAIGN_NAME, STATION_NAME])

    from disdrodb.l0.scripts.run_disdrodb_l0b_station import run_disdrodb_l0b_station

    runner.invoke(run_disdrodb_l0b_station, [RAW_DIR, DATA_SOURCE, CAMPAIGN_NAME, STATION_NAME])

    list_of_l0b = glob.glob(os.path.join(RAW_DIR, "Processed", DATA_SOURCE, CAMPAIGN_NAME, "L0B", STATION_NAME, "*.nc"))
    assert len(list_of_l0b) > 0


@pytest.mark.parametrize("remove_processed_folder", [()], indirect=True)
def test_run_disdrodb_l0_station(remove_processed_folder):
    """Test the run_disdrodb_l0_station command."""

    from disdrodb.l0.scripts.run_disdrodb_l0_station import run_disdrodb_l0_station

    runner = CliRunner()
    runner.invoke(run_disdrodb_l0_station, [RAW_DIR, DATA_SOURCE, CAMPAIGN_NAME, STATION_NAME])

    list_of_l0b = glob.glob(os.path.join(RAW_DIR, "Processed", DATA_SOURCE, CAMPAIGN_NAME, "L0B", STATION_NAME, "*.nc"))
    assert len(list_of_l0b) > 0


@pytest.mark.parametrize("remove_processed_folder", [()], indirect=True)
def test_run_disdrodb_l0b(remove_processed_folder):
    """Test the run_disdrodb_l0b command."""

    from disdrodb.l0.scripts.run_disdrodb_l0a import run_disdrodb_l0a

    runner = CliRunner()
    runner.invoke(run_disdrodb_l0a, [RAW_DIR])

    from disdrodb.l0.scripts.run_disdrodb_l0b import run_disdrodb_l0b

    runner.invoke(run_disdrodb_l0b, [RAW_DIR])
    list_of_l0b = glob.glob(os.path.join(RAW_DIR, "Processed", DATA_SOURCE, CAMPAIGN_NAME, "L0B", STATION_NAME, "*.nc"))
    assert len(list_of_l0b) > 0


@pytest.mark.parametrize("remove_processed_folder", [()], indirect=True)
def test_full_run_disdrodb_l0(remove_processed_folder):
    """Test the run_disdrodb_l0b command."""

    from disdrodb.l0.scripts.run_disdrodb_l0 import run_disdrodb_l0

    runner = CliRunner()
    runner.invoke(run_disdrodb_l0, [RAW_DIR])

    list_of_l0b = glob.glob(os.path.join(RAW_DIR, "Processed", DATA_SOURCE, CAMPAIGN_NAME, "L0B", STATION_NAME, "*.nc"))
    assert len(list_of_l0b) > 0
