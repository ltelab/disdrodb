# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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
"""Test Metadata Archive Download Routine."""
import io
import urllib.request
import zipfile

import pytest

from disdrodb.metadata.download import download_metadata_archive


@pytest.fixture
def zip_bytes():
    """Provide a minimal valid DISDRODB-METADATA-main zip as bytes."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("DISDRODB-METADATA-main/DISDRODB/sample.txt", "data")
    bio.seek(0)
    return bio.getvalue()


class TestDownloadMetadataArchive:
    """Pytest test class for download_metadata_archive function."""

    def test_successful_download_and_rename(self, tmp_path, zip_bytes, monkeypatch):
        """Successfully downloads and renames the archive to target directory."""

        class MockResponse:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        monkeypatch.setattr(urllib.request, "urlopen", lambda url: MockResponse(zip_bytes))

        metadata_dir = download_metadata_archive(str(tmp_path))
        expected = tmp_path / "DISDRODB-METADATA" / "DISDRODB"
        assert metadata_dir == str(expected)
        assert expected.is_dir()

    def test_missing_extracted_directory_raises_value_error(self, tmp_path, monkeypatch):
        """Raises ValueError when extracted directory is missing."""
        bad_zip = io.BytesIO()
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("other_dir/file.txt", "data")
        bad_zip.seek(0)

        class MockResponse:
            def read(self):
                return bad_zip.getvalue()

        monkeypatch.setattr(urllib.request, "urlopen", lambda url: MockResponse())

        with pytest.raises(ValueError):
            download_metadata_archive(str(tmp_path))

    def test_existing_target_without_force_raises_file_exists_error(self, tmp_path, zip_bytes, monkeypatch):
        """Raises FileExistsError when target exists and force=False."""

        class MockResponse:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        monkeypatch.setattr(urllib.request, "urlopen", lambda url: MockResponse(zip_bytes))

        target = tmp_path / "DISDRODB-METADATA"
        target.mkdir()
        with pytest.raises(FileExistsError):
            download_metadata_archive(str(tmp_path))

    def test_existing_target_with_force_overwrites(self, tmp_path, zip_bytes, monkeypatch):
        """Overwrites existing target directory when force=True."""

        class MockResponse:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        monkeypatch.setattr(urllib.request, "urlopen", lambda url: MockResponse(zip_bytes))

        target = tmp_path / "DISDRODB-METADATA"
        target.mkdir()
        old_file = target / "old.txt"
        old_file.write_text("old")

        metadata_dir = download_metadata_archive(str(tmp_path), force=True)
        assert not old_file.exists()
        expected = tmp_path / "DISDRODB-METADATA" / "DISDRODB"
        assert metadata_dir == str(expected)
        assert expected.is_dir()
