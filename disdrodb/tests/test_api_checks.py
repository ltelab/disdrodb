import os
import pytest

from disdrodb.api import checks


def test_check_path():
    # Test a valid path
    path = os.path.abspath(__file__)
    assert checks.check_path(path) is None

    # Test an invalid path
    path = "/path/that/does/not/exist"
    with pytest.raises(FileNotFoundError):
        checks.check_path(path)


def test_check_url():
    # Test with valid URLs
    assert checks.check_url("https://www.example.com") is True
    assert checks.check_url("http://example.com/path/to/file.html?param=value") is True
    assert checks.check_url("www.example.com") is True
    assert checks.check_url("example.com") is True

    # Test with invalid URLs
    assert checks.check_url("ftp://example.com") is False
    assert checks.check_url("htp://example.com") is False
    assert checks.check_url("http://example.com/path with spaces") is False


def test_check_disdrodb_dir():
    assert checks.check_disdrodb_dir("/path/to/DISDRODB") is None

    with pytest.raises(ValueError):
        checks.check_disdrodb_dir("/path/to/DISDRO")
