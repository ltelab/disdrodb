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
# along with this progra  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""Testing fall velocity routines."""
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION
from disdrodb.l1.fall_velocity import (
    available_raindrop_fall_velocity_models,
    check_raindrop_fall_velocity_model,
    get_raindrop_fall_velocity,
    get_raindrop_fall_velocity_from_ds,
    get_raindrop_fall_velocity_model,
)
from disdrodb.l1_env.routines import load_env_dataset
from disdrodb.tests.fake_datasets import create_template_l0c_dataset


@pytest.mark.parametrize("model", available_raindrop_fall_velocity_models())
class TestGetRaindropFallVelocity:
    """Test suite for get_raindrop_fall_velocity across all models."""

    @pytest.mark.parametrize("scalar", [0.5, 1.0, 3.2])
    def test_scalar_inputs(self, model, scalar):
        """Ensure scalars and arrays give consistent results."""
        if model == "Beard1976":
            ds_env = load_env_dataset()
            result = get_raindrop_fall_velocity(scalar, model=model, ds_env=ds_env)
            expected = get_raindrop_fall_velocity(np.array([scalar]), model=model, ds_env=ds_env)
        else:
            result = get_raindrop_fall_velocity(scalar, model=model)
            expected = get_raindrop_fall_velocity(np.array([scalar]), model=model)

        assert isinstance(result, xr.DataArray)
        assert np.isclose(result, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("backend", ["numpy", "dask", "xarray"])
    def test_arrays_consistent_across_backends(self, model, backend):
        """Check consistent results with numpy, dask, and xarray inputs."""
        values = np.array([0.5, 1.0, 2.5, 5.0])

        if backend == "numpy":
            input_data = values
        elif backend == "dask":
            input_data = da.from_array(values, chunks=(2,))
        elif backend == "xarray":
            input_data = xr.DataArray(values, dims="diameter")

        if model == "Beard1976":
            ds_env = load_env_dataset()
            result = get_raindrop_fall_velocity(input_data, model=model, ds_env=ds_env)
        else:
            result = get_raindrop_fall_velocity(input_data, model=model)

        assert isinstance(result, xr.DataArray)
        if backend == "dask":
            assert hasattr(result.data, "chunks")

        if hasattr(result, "compute"):
            result = result.compute()

        np.testing.assert_allclose(
            result,
            get_raindrop_fall_velocity(
                values,
                model=model,
                ds_env=load_env_dataset() if model == "Beard1976" else None,
            ),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_nan_input(self, model):
        """NaN inputs should propagate to outputs."""
        val = np.array([np.nan])
        if model == "Beard1976":
            ds_env = load_env_dataset()
            result = get_raindrop_fall_velocity(val, model=model, ds_env=ds_env)
        else:
            result = get_raindrop_fall_velocity(val, model=model)

        assert np.isnan(result), f"For diameter=NaN, expected NaN for model {model}, got {result}"

    def test_zero_input(self, model):
        """Zero inputs should return NaN."""
        val = np.array([0])
        if model == "Beard1976":
            ds_env = load_env_dataset()
            result = get_raindrop_fall_velocity(val, model=model, ds_env=ds_env)
        else:
            result = get_raindrop_fall_velocity(val, model=model)

        assert np.isnan(result), f"For diameter=0, expected 0 for model {model}, got {result}"

    def test_large_diameter_input(self, model):
        """Irrealistic diameter inputs (out of range) return NaN."""
        val = np.array([15, -1])
        if model == "Beard1976":
            ds_env = load_env_dataset()
            result = get_raindrop_fall_velocity(val, model=model, ds_env=ds_env)
        else:
            result = get_raindrop_fall_velocity(val, model=model)
        assert np.isnan(result).all()


@pytest.mark.parametrize(
    ("model", "diameter", "expected"),
    [
        ("Atlas1973", 2.0, 6.5476),
        ("Brandes2002", 2.0, 6.5384),  # approximate reference
        ("Uplinger1981", 2.0, 6.5999),  # approximate reference
        ("VanDijk2002", 2.0, 6.6068),  # approximate reference
    ],
)
def test_models_expected_values(model, diameter, expected):
    """Test selected models against reference values at D=2 mm."""
    result = get_raindrop_fall_velocity(diameter, model=model)
    assert np.isclose(result, expected, rtol=1e-2, atol=1e-1)


def test_beard1976_expected_values():
    """Test selected models against reference values at D=2 mm."""
    ds_env = xr.Dataset(
        {
            "sea_level_air_pressure": 101325,
            "gas_constant_dry_air": 287.0,
            "lapse_rate": 0.0065,
            "relative_humidity": 0.95,
            "temperature": 293.1,
            "water_density": 1000,
        },
        coords={
            "latitude": 46.16,
            "longitude": 8.775,
            "altitude": 0,
        },
    )

    diameter = np.arange(1, 6)
    da_fall_velocity = get_raindrop_fall_velocity(diameter, ds_env=ds_env, model="Beard1976")
    expected_values = np.array([4.01598392, 6.52857255, 8.07642473, 8.85000858, 9.123762])
    np.testing.assert_allclose(da_fall_velocity.to_numpy(), expected_values, rtol=1e-3, atol=1e-4)


def test_beard1976_model_works_without_ds_env():
    """Beard1976 model works also if ds_env not specified."""
    diameter = np.array([1.0, 2.0, 3.0])
    result = get_raindrop_fall_velocity(diameter, model="Beard1976")
    assert result.shape == diameter.shape
    assert np.all(result > 0)


@pytest.mark.parametrize("model", ["InvalidModel", "", None, 123])
def test_invalid_raindrop_fall_velocity_models_raise(model):
    """Invalid model names must raise ValueError."""
    with pytest.raises(ValueError):
        check_raindrop_fall_velocity_model(model)


@pytest.mark.parametrize("model", available_raindrop_fall_velocity_models())
def test_get_raindrop_fall_velocity_model_callable(model):
    """Ensure model getter returns a callable."""
    func = get_raindrop_fall_velocity_model(model)
    assert callable(func), f"Expected callable, got {type(func)}"


def test_get_raindrop_fall_velocity_from_ds():
    """Test get_raindrop_fall_velocity_from_ds."""
    # Create test dataset
    ds = create_template_l0c_dataset()
    ds_env = load_env_dataset(ds)

    # Run with default ds_env
    fall_velocity = get_raindrop_fall_velocity_from_ds(ds, ds_env=None, model="Beard1976")
    assert isinstance(fall_velocity, xr.DataArray)

    # Run with specified default ds_env
    fall_velocity1 = get_raindrop_fall_velocity_from_ds(ds, ds_env=ds_env, model="Beard1976")
    assert isinstance(fall_velocity, xr.DataArray)
    xr.testing.assert_allclose(fall_velocity, fall_velocity1)

    # Run with specified default ds_env
    ds_env["temperature"] = 10 + 273.15
    fall_velocity2 = get_raindrop_fall_velocity_from_ds(ds, ds_env=ds_env, model="Beard1976")
    assert isinstance(fall_velocity, xr.DataArray)
    with pytest.raises(AssertionError):
        xr.testing.assert_allclose(fall_velocity2, fall_velocity1)

    # Raise error if diameter dimension does not exists
    with pytest.raises(ValueError):
        get_raindrop_fall_velocity_from_ds(ds.drop_dims(DIAMETER_DIMENSION))


def test_beard_1976_xarray_broadcasting():
    """Test get_raindrop_fall_velocity broadcasts correctly ENV variables for Beard1976."""
    # Define model
    model = "Beard1976"

    # Define default ENV dataset
    ds_env = load_env_dataset()

    # Define diameter
    diameter = [1, 2, 3, 4]

    # Define temperature
    temperature = np.array([5, 10, 20]) + 273.15
    temperature = xr.DataArray(temperature, dims="temperature", coords={"temperature": temperature})

    # Define relative humidity
    relative_humidity = np.array([0.1, 0.5, 0.8, 1.0])
    relative_humidity = xr.DataArray(
        relative_humidity,
        dims="relative_humidity",
        coords={"relative_humidity": relative_humidity},
    )

    # Define altitude
    altitude = np.array([0, 1000, 2000, 3000])
    altitude = xr.DataArray(altitude, dims="altitude", coords={"altitude": altitude})

    # Define latitude
    latitude = np.array([0, 40, 60])
    latitude = xr.DataArray(latitude, dims="latitude", coords={"latitude": latitude})

    # Update ENV dataset
    ds_env = ds_env.drop_vars(["latitude", "altitude", "relative_humidity", "temperature"])
    ds_env["latitude"] = latitude
    ds_env["altitude"] = altitude
    ds_env["relative_humidity"] = relative_humidity
    ds_env["temperature"] = temperature

    # Compute fall velocity
    fall_velocity = get_raindrop_fall_velocity(diameter, model=model, ds_env=ds_env)

    # Check dimensions
    assert set(fall_velocity.dims) == {*list(ds_env.dims), DIAMETER_DIMENSION}
    assert np.all(fall_velocity > 0)


def test_beard_1976_with_time_varying_env_variables():
    """Test get_raindrop_fall_velocity works correctly with time-varying ENV variables."""
    # Define model
    model = "Beard1976"

    # Define default ENV dataset
    ds_env = load_env_dataset()

    # Define diameter
    diameter = [1, 2, 3, 4]

    # Define timesteps
    timesteps = pd.date_range("2023-01-01", periods=3, freq="min")

    # Define temperature
    temperature = np.array([5, 10, 20]) + 273.15
    temperature = xr.DataArray(temperature, dims="time", coords={"time": ("time", timesteps)})

    # Define relative humidity
    relative_humidity = np.array([0.1, 0.5, 0.8])
    relative_humidity = xr.DataArray(relative_humidity, dims="time", coords={"time": ("time", timesteps)})

    # Define latitude
    latitude = np.array([0, 40, 60])
    latitude = xr.DataArray(latitude, dims="time", coords={"time": ("time", timesteps)})

    # Update ENV dataset
    ds_env = ds_env.drop_vars(["latitude", "relative_humidity", "temperature"])
    ds_env["latitude"] = latitude
    ds_env["relative_humidity"] = relative_humidity
    ds_env["temperature"] = temperature

    # Compute fall velocity
    fall_velocity = get_raindrop_fall_velocity(diameter, model=model, ds_env=ds_env)

    # Check dimensions
    assert set(fall_velocity.dims) == {DIAMETER_DIMENSION, "time"}
    assert np.all(fall_velocity > 0)
