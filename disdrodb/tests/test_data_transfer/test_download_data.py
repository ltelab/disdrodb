# #!/usr/bin/env python3

# # -----------------------------------------------------------------------------.
# # Copyright (c) 2021-2023 DISDRODB developers
# #
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with this program.  If not, see <http://www.gnu.org/licenses/>.
# # -----------------------------------------------------------------------------.
"""Test DISDRODB download utility."""
import os
import shutil
import subprocess

import pytest

import disdrodb
from disdrodb.api.path import define_station_dir
from disdrodb.data_transfer.download_data import (
    _download_file_from_url,
    build_webserver_wget_command,
    compute_cut_dirs,
    download_archive,
    download_station,
    download_station_data,
    download_web_server_data,
    ensure_trailing_slash,
    ensure_wget_available,
)
from disdrodb.tests.conftest import create_fake_metadata_file, create_fake_raw_data_file

TEST_ZIP_FPATH = "https://zenodo.org/records/15585935/files/station_files.zip?download=1"


class TestEnsureTrailingSlash:
    def test_adds_slash_if_missing(self):
        """Ensure trailing slash is added when missing."""
        url = "http://example.com/path"
        result = ensure_trailing_slash(url)
        assert result.endswith("/")

    def test_preserves_trailing_slash(self):
        """Ensure trailing slash is preserved."""
        url = "http://example.com/path/"
        result = ensure_trailing_slash(url)
        assert result == url


class TestComputeCutDirs:
    def test_root_url_returns_zero(self):
        """compute_cut_dirs should return 0 for root URL."""
        url = "http://example.com/"
        assert compute_cut_dirs(url) == 0

    def test_single_segment_url(self):
        """compute_cut_dirs should count one path segment."""
        url = "http://example.com/a/"
        assert compute_cut_dirs(url) == 1

    def test_multiple_segments_url(self):
        """compute_cut_dirs should count multiple path segments."""
        url = "https://example.com/a/b/c/"
        assert compute_cut_dirs(url) == 3

    def test_no_trailing_slash_behavior(self):
        """compute_cut_dirs should work even if URL lacks trailing slash."""
        url = "https://example.com/a/b/c"
        # compute_cut_dirs expects URL ending with '/', so strip and add logic inside function
        normalized = ensure_trailing_slash(url)
        assert compute_cut_dirs(normalized) == 3


class TestBuildWebserverWgetCommand:
    def test_with_force_and_verbose(self):
        """build_webserver_wget_command includes -q and --timestamping when verbose and force are True."""
        url = "http://example.com/data/"
        cut_dirs = 2
        dst_dir = "/tmp/download"
        cmd = build_webserver_wget_command(url, cut_dirs=cut_dirs, dst_dir=dst_dir, force=True, verbose=True)

        assert cmd[0] == "wget"
        assert "-q" in cmd
        assert "-r" in cmd
        assert "-np" in cmd
        assert "-nH" in cmd
        assert f"--cut-dirs={cut_dirs}" in cmd
        assert "--timestamping" in cmd
        assert "-P" in cmd
        assert dst_dir in cmd
        assert url in cmd

    def test_without_force_and_verbose(self):
        """build_webserver_wget_command omits -q and --timestamping when verbose and force are False."""
        url = "http://example.com/data/"
        cut_dirs = 1
        dst_dir = "/var/data"
        cmd = build_webserver_wget_command(url, cut_dirs=cut_dirs, dst_dir=dst_dir, force=False, verbose=False)

        assert cmd[0] == "wget"
        assert "-q" not in cmd
        assert "--timestamping" not in cmd
        assert "-r" in cmd
        assert "-np" in cmd
        assert "-nH" in cmd
        assert f"--cut-dirs={cut_dirs}" in cmd
        assert "-P" in cmd
        assert dst_dir in cmd
        assert url in cmd


class TestEnsureWgetAvailable:
    def test_wget_present(self, monkeypatch):
        """ensure_wget_available should not raise if wget is on PATH."""
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/wget")
        # Should not raise
        ensure_wget_available()

    def test_wget_missing(self, monkeypatch):
        """ensure_wget_available should raise FileNotFoundError if wget is absent."""
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        with pytest.raises(FileNotFoundError) as excinfo:
            ensure_wget_available()
        assert "WGET software was not found" in str(excinfo.value)


class TestDownloadWebServerData:
    @pytest.fixture(autouse=True)
    def patch_dependencies(self, monkeypatch):
        """Patch external dependencies for download_web_server_data."""
        # Stub ensure_wget_available
        monkeypatch.setattr(
            "disdrodb.data_transfer.download_data.ensure_wget_available",
            lambda: None,
        )

        # Stub ensure_trailing_slash to return URL with trailing slash
        monkeypatch.setattr(
            "disdrodb.data_transfer.download_data.ensure_trailing_slash",
            lambda url: url if url.endswith("/") else url + "/",
        )

        # Stub compute_cut_dirs to return a fixed value
        monkeypatch.setattr(
            "disdrodb.data_transfer.download_data.compute_cut_dirs",
            lambda url: 3,
        )

        # Stub build_webserver_wget_command to return a known command list
        monkeypatch.setattr(
            "disdrodb.data_transfer.download_data.build_webserver_wget_command",
            lambda url, cut_dirs, dst_dir, force, verbose: [
                "wget",
                "-r",
                "-np",
                "-nH",
                "--cut-dirs=3",
                "-P",
                dst_dir,
                url,
            ],
        )

        # Capture calls to os.makedirs
        calls = []
        monkeypatch.setattr(os, "makedirs", lambda path, exist_ok: calls.append((path, exist_ok)))
        self.makedirs_calls = calls

        # Prepare a placeholder for subprocess.run behavior
        self.subprocess_args = []

        def fake_run(cmd, check):
            self.subprocess_args.append((cmd, check))

        monkeypatch.setattr(subprocess, "run", fake_run)

    def test_successful_download_invokes_subprocess(self, tmp_path):
        """download_web_server_data should invoke subprocess.run with constructed command on success."""
        url = "http://example.com/data"
        dst_dir = str(tmp_path / "out")
        download_web_server_data(url, dst_dir, force=True, verbose=True)

        # os.makedirs should have been called for dst_dir
        assert (dst_dir, True) in self.makedirs_calls

        # subprocess.run should have been called once with the stubbed command
        assert len(self.subprocess_args) == 1
        cmd_called, check_flag = self.subprocess_args[0]
        assert cmd_called == ["wget", "-r", "-np", "-nH", "--cut-dirs=3", "-P", dst_dir, url + "/"]
        assert check_flag is True

    def test_subprocess_failure_raises(self, tmp_path, monkeypatch):
        """download_web_server_data should propagate CalledProcessError when subprocess.run fails."""

        # Patch subprocess.run to raise CalledProcessError
        def raise_error(cmd, check):
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, output=b"", stderr=b"error")

        monkeypatch.setattr(subprocess, "run", raise_error)

        url = "http://example.com/data"
        dst_dir = str(tmp_path / "out")
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            download_web_server_data(url, dst_dir, force=False, verbose=False)
        assert excinfo.value.returncode == 1
        assert "wget" in excinfo.value.cmd


def test_download_file_from_url(tmp_path):
    # Test download case when empty directory
    url = "https://raw.githubusercontent.com/ltelab/disdrodb/main/README.md"
    # url = "https://httpbin.org/stream-bytes/1024"
    dst_filepath = _download_file_from_url(url, tmp_path, force=False)
    assert os.path.isfile(dst_filepath)

    # Test download case when directory is not empty and force=False --> avoid download
    url = "https://raw.githubusercontent.com/ltelab/disdrodb/main/CODE_OF_CONDUCT.md"
    # url = "https://httpbin.org/stream-bytes/1025"
    with pytest.raises(ValueError):
        _download_file_from_url(url, tmp_path, force=False)

    # Test download case when directory is not empty and force=True --> it download
    url = "https://raw.githubusercontent.com/ltelab/disdrodb/main/CODE_OF_CONDUCT.md"
    # url = "https://httpbin.org/stream-bytes/1026"
    dst_filepath = _download_file_from_url(url, tmp_path, force=True)
    assert os.path.isfile(dst_filepath)


def test_download_station_data(tmp_path):
    # Define project paths
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"

    # Define metadata
    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = TEST_ZIP_FPATH

    # Create metadata file
    metadata_filepath = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        metadata_dict=metadata_dict,
    )

    # Download data
    download_station_data(metadata_filepath=metadata_filepath, data_archive_dir=data_archive_dir, force=True)

    # Define expected station directory
    station_dir = define_station_dir(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="RAW",
    )

    # Assert files in the zip file have been unzipped
    assert os.path.isfile(os.path.join(station_dir, "station_file1.txt"))
    # Assert inner zipped files are not unzipped !
    assert os.path.isfile(os.path.join(station_dir, "station_file2.zip"))
    # Assert inner directories are there
    assert os.path.isdir(os.path.join(station_dir, "2020"))
    # Assert zip file has been removed
    assert not os.path.exists(os.path.join(station_dir, "station_files.zip"))


@pytest.mark.parametrize("force", [True, False])
@pytest.mark.parametrize("disdrodb_data_url", [None, "", 1])
def test_download_without_any_remote_url(tmp_path, requests_mock, mocker, disdrodb_data_url, force):
    """Test download station data without url."""
    # Define project paths
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_archive_dir = tmp_path / "data" / "DISDRODB"

    # Create metadata file
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name = "test_station_name"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = disdrodb_data_url

    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Check download station raise error
    with pytest.raises(ValueError):
        download_station(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            force=force,
        )

    # Check download archive run
    download_archive(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_sources=data_source,
        campaign_names=campaign_name,
        station_names=station_name,
        force=force,
    )


def test_download_station_only_with_valid_metadata(tmp_path):
    """Test download of archive stations is not stopped by single stations download errors."""
    # Define project paths
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_archive_dir = tmp_path / "data" / "DISDRODB"

    # Create metadata file
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name = "test_station_name"

    metadata_dict = {}
    metadata_dict["station_name"] = "ANOTHER_STATION_NAME"
    metadata_dict["disdrodb_data_url"] = TEST_ZIP_FPATH
    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Test raise error if metadata file is not valid
    with disdrodb.config.set({"metadata_archive_dir": metadata_archive_dir}), pytest.raises(ValueError):
        download_station(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )


@pytest.mark.parametrize("force", [True, False])
def test_download_station(tmp_path, force):
    """Test download station data."""
    # Define project paths
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_archive_dir = tmp_path / "data" / "DISDRODB"

    # Create metadata file
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name = "test_station_name"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = TEST_ZIP_FPATH

    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Create raw data file
    raw_file_filepath = create_fake_raw_data_file(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    with disdrodb.config.set({"metadata_archive_dir": metadata_archive_dir}):
        # Check download_station raise error if existing data and force=False
        if not force:
            with pytest.raises(ValueError):
                download_station(
                    data_archive_dir=data_archive_dir,
                    data_source=data_source,
                    campaign_name=campaign_name,
                    station_name=station_name,
                    force=force,
                )

            # Check original raw file exists if force=False
            if not force:
                assert os.path.exists(raw_file_filepath)

        # Check download_station overwrite existing files if force=True
        else:
            download_station(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                force=force,
            )
            # Check original raw file does not exist anymore
            if force:
                assert not os.path.exists(raw_file_filepath)


@pytest.mark.parametrize("existing_data", [True, False])
@pytest.mark.parametrize("force", [True, False])
def test_download_archive(tmp_path, force, existing_data):
    """Test download station data."""
    # Define project paths
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_archive_dir = tmp_path / "data" / "DISDRODB"

    # Create metadata file
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name = "test_station_name"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = TEST_ZIP_FPATH

    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Create raw data file
    if existing_data:
        raw_file_filepath = create_fake_raw_data_file(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )

    # Check download_archive does not raise error if existing data and force=False
    with disdrodb.config.set({"metadata_archive_dir": metadata_archive_dir}):
        download_archive(
            data_archive_dir=data_archive_dir,
            data_sources=data_source,
            campaign_names=campaign_name,
            station_names=station_name,
            force=force,
        )

    # Check existing_data
    if existing_data:
        if not force:
            # Check original raw file exists if force=False
            assert os.path.exists(raw_file_filepath)
        else:
            # Check original raw file does not exist anymore if force=True
            assert not os.path.exists(raw_file_filepath)
