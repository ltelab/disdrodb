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
"""Test permittivity models."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from disdrodb.scattering.permittivity import (
    available_permittivity_models,
    check_permittivity_model,
    get_rain_refractive_index_liebe1991,
    get_rayleigh_dielectric_factor,
    get_refractive_index,
    get_refractive_index_function,
)


@pytest.mark.parametrize("model", available_permittivity_models())
class TestGetRefractiveIndex:
    """Test suite for get_refractive_index across all permittivity models."""

    @pytest.mark.parametrize("backend", ["numpy", "dask", "xarray"])
    def test_arrays_consistent_across_backends(self, model, backend):
        """Test get_refractive_index returns identical values across numpy, dask, and xarray inputs."""
        frequency = 5.6
        temps = np.array([0.5, 10.0, 20.0, 30.0])

        if backend == "numpy":
            input_data = temps
        elif backend == "dask":
            input_data = da.from_array(temps, chunks=(2,))
        elif backend == "xarray":
            input_data = xr.DataArray(temps, dims="temperature")

        result = get_refractive_index(input_data, frequency=frequency, permittivity_model=model)
        if hasattr(result, "compute"):
            result = result.compute()
        expected = get_refractive_index(temps, frequency=frequency, permittivity_model=model)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("scalar", [0, 10, 25.0, 37])
    def test_scalar_inputs(self, model, scalar):
        """Test get_refractive_index accepts float and int scalar inputs."""
        frequency = 5.6
        result = get_refractive_index(scalar, frequency=frequency, permittivity_model=model)
        expected = get_refractive_index(np.array([scalar]), frequency=frequency, permittivity_model=model)[0]
        assert np.isclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_nan_propagation(self, model):
        """Test get_refractive_index returns NaN when input temperature is NaN."""
        frequency = 5.6
        result = get_refractive_index(np.nan, frequency=frequency, permittivity_model=model)
        assert np.isnan(result), f"Expected NaN for model {model}, got {result}"

    def test_xarray_broadcasting_temperature_frequency(self, model):
        """Test get_refractive_index broadcasts temperature and frequency DataArrays to combined dims."""
        # Temperature along "temperature" dim
        temperature = np.arange(5, 10)
        temperature = xr.DataArray(temperature, dims="temperature", coords={"temperature": temperature})

        # Frequency along "frequency" dim
        frequency = np.arange(10, 12)
        frequency = xr.DataArray(frequency, dims="frequency", coords={"frequency": frequency})

        # Call function
        m = get_refractive_index(temperature=temperature, frequency=frequency, permittivity_model=model)

        # Check dimensions
        assert m.dims == ("temperature", "frequency")
        assert m.shape == (temperature.size, frequency.size)

    def test_numpy_both_inputs_size_gt1_raises(self, model):
        """Test get_refractive_index raises when both temperature and frequency are >1 with numpy arrays."""
        temp = np.array([0, 10, 20])
        freq = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="xarray.DataArray"):
            get_refractive_index(temp, frequency=freq, permittivity_model=model)


@pytest.mark.parametrize(
    ("model", "temperature", "frequency", "expected"),
    [
        # Reference outputs (computed once from current implementation at T=20°C, f=5.6 GHz)
        ("Liebe1991", 20.0, 5.6, 8.624863 + 1.29096j),
        ("Liebe1991single", 20.0, 5.6, 8.626885 + 1.28699831j),
        ("Ellison2007", 20.0, 5.6, 8.627059 + 1.298061j),
        ("Turner2016", 20.0, 5.6, 8.6217850 + 1.303982j),
    ],
)
def test_get_refractive_index_expected_values(model, temperature, frequency, expected):
    """Test each permittivity model via get_refractive_index returns expected reference values."""
    result = get_refractive_index(temperature=temperature, frequency=frequency, permittivity_model=model)
    assert np.isclose(result, expected, rtol=1e-6, atol=1e-8), f"Model {model} gave {result}, expected {expected}"


class TestModelValidityRange:
    """Test validity range enforcement for Liebe1991 refractive index model."""

    def test_temperature_below_range_raises(self):
        """Test temperature below 0°C raises ValueError."""
        with pytest.raises(ValueError):
            get_rain_refractive_index_liebe1991(temperature=-1, frequency=5.6)

    def test_temperature_above_range_raises(self):
        """Test temperature above 40°C raises ValueError."""
        with pytest.raises(ValueError):
            get_rain_refractive_index_liebe1991(temperature=50, frequency=5.6)

    def test_frequency_below_range_raises(self):
        """Test frequency below 0 GHz raises ValueError."""
        with pytest.raises(ValueError):
            get_rain_refractive_index_liebe1991(temperature=20, frequency=-1)

    def test_frequency_above_range_raises(self):
        """Test frequency above 1000 GHz raises ValueError."""
        with pytest.raises(ValueError):
            get_rain_refractive_index_liebe1991(temperature=20, frequency=2000)


class TestCheckPermittivityModel:
    """Test suite for check_permittivity_model."""

    @pytest.mark.parametrize("model", available_permittivity_models())
    def test_valid_models(self, model):
        """Test valid model names are returned unchanged."""
        assert check_permittivity_model(model) == model

    @pytest.mark.parametrize("model", ["InvalidModel", "", None, 123])
    def test_invalid_models_raise(self, model):
        """Test invalid model names raise ValueError."""
        with pytest.raises(ValueError):
            check_permittivity_model(model)


class TestGetRefractiveIndexFunction:
    """Test suite for get_refractive_index_function."""

    @pytest.mark.parametrize("model", available_permittivity_models())
    def test_returns_callable(self, model):
        """Test valid models return a callable function."""
        func = get_refractive_index_function(model)
        assert callable(func), f"Expected callable for model {model}, got {type(func)}"

    @pytest.mark.parametrize("model", available_permittivity_models())
    def test_callable_nan_propagation(self, model):
        """Test refractive index model functions propagate NaN inputs."""
        func = get_refractive_index_function(model)
        result = func(temperature=np.nan, frequency=5.6)
        assert np.isnan(result), f"Expected NaN for model {model}, got {result}"

    def test_invalid_model_raises(self):
        """Test invalid model name raises ValueError."""
        with pytest.raises(ValueError):
            get_refractive_index_function("NotAModel")


class TestRayleighDielectricFactor:
    """Test get_rayleigh_dielectric_factor function."""

    def test_known_value(self):
        """Test dielectric factor matches manual calculation for a known refractive index."""
        m = 1.33 + 0j  # refractive index of water in microwave range
        eps = m**2
        K = (eps - 1.0) / (eps + 2.0)
        expected = np.abs(K) ** 2

        result = get_rayleigh_dielectric_factor(m)
        assert np.isclose(result, expected, rtol=1e-12), f"Expected {expected}, got {result}"

    def test_array_input(self):
        """Test function handles array input correctly."""
        m_array = np.array([1.33 + 0j, 1.5 + 0j])
        result = get_rayleigh_dielectric_factor(m_array)

        expected = []
        for m in m_array:
            eps = m**2
            K = (eps - 1.0) / (eps + 2.0)
            expected.append(np.abs(K) ** 2)
        expected = np.array(expected)

        np.testing.assert_allclose(result, expected, rtol=1e-12)
