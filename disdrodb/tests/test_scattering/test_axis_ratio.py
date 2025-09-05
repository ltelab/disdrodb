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
"""Test axis ratio models."""
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from disdrodb.scattering.axis_ratio import (
    available_axis_ratio_models,
    check_axis_ratio_model,
    get_axis_ratio,
    get_axis_ratio_model,
)


@pytest.mark.parametrize("model", available_axis_ratio_models())
class TestGetAxisRatio:
    """Test suite for get_axis_ratio function across all models."""

    @pytest.mark.parametrize("scalar", [0.5, 1, 3.2])
    def test_scalar_inputs(self, model, scalar):
        """Test get_axis_ratio accepts float and int scalar inputs."""
        result = get_axis_ratio(scalar, model=model)
        expected = get_axis_ratio(np.array([scalar]), model=model)[0]
        assert np.isclose(result, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("backend", ["numpy", "dask", "xarray"])
    def test_arrays_consistent_across_backends(self, model, backend):
        """Test get_axis_ratio returns identical values across numpy, dask, and xarray inputs."""
        values = np.array([0.5, 1.0, 2.5, 5.0])

        if backend == "numpy":
            input_data = values
        elif backend == "dask":
            input_data = da.from_array(values, chunks=(2,))
        elif backend == "xarray":
            input_data = xr.DataArray(values, dims="diameter")

        result = (
            get_axis_ratio(input_data, model=model).compute()
            if hasattr(input_data, "compute")
            else get_axis_ratio(input_data, model=model)
        )
        np.testing.assert_allclose(result, get_axis_ratio(values, model=model), rtol=1e-6, atol=1e-6)

    def test_nan_propagation(self, model):
        """Test get_axis_ratio returns NaN when input is NaN."""
        val = np.array([np.nan])
        result = get_axis_ratio(val, model=model)
        assert np.isnan(result), f"Expected NaN for model {model}, got {result}"


@pytest.mark.parametrize(
    ("model", "diameter", "expected"),
    [
        # Reference outputs precomputed at diameter=2.0 mm
        ("Thurai2005", 2.0, 0.93150),
        ("Thurai2007", 2.0, 0.9295128),
        ("Battaglia2010", 2.0, 0.92499),
        ("Brandes2002", 2.0, 0.93797),
        ("Pruppacher1970", 2.0, 0.906),
        ("Beard1987", 2.0, 0.92759279),
        ("Andsager1999", 2.0, 0.92759),
    ],
)
def test_get_axis_ratio_expected_values(model, diameter, expected):
    """Test each axis ratio model via get_axis_ratio returns expected reference values."""
    result = get_axis_ratio(diameter, model=model)
    assert np.isclose(result, expected, rtol=1e-6, atol=1e-5), f"Model {model} gave {result}, expected {expected}"


@pytest.mark.parametrize("model", ["InvalidModel", "", None, 123])
def test_invalid_axis_ratio_models_raise(model):
    """Test invalid model names raise ValueError."""
    with pytest.raises(ValueError):
        check_axis_ratio_model(model)


@pytest.mark.parametrize("model", available_axis_ratio_models())
def test_get_axis_ratio_model(model):
    """Test get_axis_ratio_model return a callable function."""
    func = get_axis_ratio_model(model)
    assert callable(func), f"Expected callable for model {model}, got {type(func)}"
