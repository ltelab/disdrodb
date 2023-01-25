import inspect
from disdrodb.L0 import L0_processing


def test_click_L0_readers_options():
    # function_return = L0_processing.click_L0_readers_options()
    assert 1 == 1


def test_run_L0():
    # Uncheck function. Tested with test_L0_readers.py
    # function_return = L0_processing.run_L0()
    assert 1 == 1


def test_get_available_readers():
    # Check that at least the EPFL institution is included in the list of readers
    function_return = L0_processing.get_available_readers()
    assert "EPFL" in function_return.keys()


def test_check_data_source():
    # Check that at least the EPFL institution is included in the list of readers
    function_return = function_return = L0_processing.check_data_source("epfl")
    assert function_return == "EPFL"


def test_get_available_readers_by_data_source():
    # Check that at least the list of EPFL readers are not empty
    function_return = L0_processing.get_available_readers_by_data_source("epfl")
    assert function_return is not None


def test_check_reader_name():
    # Check if the reader "EPFL_ROOF_2012" within EPFL
    function_return = L0_processing.check_reader_name("epfl", "EPFL_ROOF_2012")
    assert function_return == "EPFL_ROOF_2012"


def test_get_reader():
    # Check that the object is a function
    function_return = L0_processing.get_reader("epfl", "EPFL_ROOF_2012")
    assert inspect.isfunction(function_return)


def test_run_reader():
    # Uncheck function. Tested with test_L0_readers.py
    # function_return = L0_processing.run_reader()
    assert 1 == 1


def test_is_documented_by():
    # Uncheck function.
    # function_return = L0_processing.is_documented_by()
    assert 1 == 1


def test_reader_generic_docstring():
    # Uncheck function.
    function_return = L0_processing.reader_generic_docstring()
    assert 1 == 1
