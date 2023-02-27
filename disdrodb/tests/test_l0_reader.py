import inspect
import pytest
from disdrodb.l0 import l0_reader


def test_is_documented_by():
    # Uncheck function.
    # function_return = L0_processing.is_documented_by()
    assert 1 == 1


def test_reader_generic_docstring():
    # Uncheck function.
    l0_reader.reader_generic_docstring()
    assert 1 == 1


def test_get_available_readers_dict():
    # Check that at least the EPFL institution is included in the list of readers
    function_return = l0_reader.get_available_readers_dict()
    assert "EPFL" in function_return.keys()


def test_check_reader_data_source():
    # Check that at least the EPFL institution is included in the list of readers
    function_return = l0_reader._check_reader_data_source("EPFL")
    assert function_return == "EPFL"

    # Check raise error if not existing data_source
    with pytest.raises(ValueError):
        l0_reader._check_reader_data_source("epfl")

    with pytest.raises(ValueError):
        l0_reader._check_reader_data_source("dummy")


def test_check_reader_exists():
    # Check existing reader
    function_return = l0_reader.check_reader_exists("EPFL", "EPFL_ROOF_2012")
    assert function_return == "EPFL_ROOF_2012"

    # Check unexisting reader
    with pytest.raises(ValueError):
        l0_reader.check_reader_exists("EPFL", "dummy")


def test_get_reader():
    # Check that the object is a function
    function_return = l0_reader.get_reader("EPFL", "EPFL_ROOF_2012")
    assert inspect.isfunction(function_return)
