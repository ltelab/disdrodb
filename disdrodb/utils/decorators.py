#!/usr/bin/env python3

# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2023 DISDRODB developers
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
"""DISDRODB decorators."""
import functools
import importlib
import uuid

import dask


def create_dask_task_name(function_name: str, name=None) -> str | None:
    """
    Create a custom dask task name.

    Parameters
    ----------
    function_name : str
        Name of the function being delayed.
    name : str, optional
        Custom name for the task (e.g., filepath or ID).
        If None, returns None so that Dask generates is own default name.

    Returns
    -------
    str | None
        Custom dask task name string if `name` is given,
        otherwise None (use Dask's default naming).
    """
    if name is None:
        return None
    return f"{function_name}.{name}-{uuid.uuid4()}"


def delayed_if_parallel(function):
    """Decorator to make the function delayed if its ``parallel`` argument is ``True``."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        # Check if it must be a delayed function
        parallel = kwargs.get("parallel")
        # If parallel is True
        if parallel:
            # Enforce verbose to be False
            kwargs["verbose"] = False
            # Define custom dask task name
            if "logs_filename" in kwargs:
                kwargs["dask_key_name"] = create_dask_task_name(
                    function_name=function.__name__,
                    name=kwargs["logs_filename"],
                )

            # Define the delayed task
            result = dask.delayed(function)(*args, **kwargs)
        else:
            # Else run the function
            result = function(*args, **kwargs)
        return result

    return wrapper


def single_threaded_if_parallel(function):
    """Decorator to make a function use a single threadon delayed if its ``parallel`` argument is ``True``."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        # Check if it must be a delayed function
        parallel = kwargs.get("parallel")
        # If parallel is True
        if parallel:
            # Call function with single thread
            # with dask.config.set(scheduler='single-threaded'):
            with dask.config.set(scheduler="synchronous"):
                result = function(*args, **kwargs)
        else:
            # Else run the function as usual
            result = function(*args, **kwargs)
        return result

    return wrapper


def check_software_availability(software, conda_package):
    """A decorator to ensure that a software package is installed.

    Parameters
    ----------
    software : str
        The package name as recognized by Python's import system.
    conda_package : str
        The package name as recognized by conda-forge.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not importlib.util.find_spec(software):
                raise ImportError(
                    f"The '{software}' package is required but not found.\n"
                    "Please install it using conda:\n"
                    f"    conda install -c conda-forge {conda_package}",
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def check_pytmatrix_availability(func):
    """Decorator to ensure that the 'pytmatrix' package is installed."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not importlib.util.find_spec("pytmatrix"):
            raise ImportError(
                "The 'pytmatrix' package is required but not found. \n"
                "Please install the following software: \n"
                "  conda install conda-forge gfortran \n"
                "  conda install conda-forge meson \n"
                "  pip install git+https://github.com/ltelab/pytmatrix-lte.git@main \n",
            )
        return func(*args, **kwargs)

    return wrapper
