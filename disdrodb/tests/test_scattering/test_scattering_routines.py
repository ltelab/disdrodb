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
"""Test scattering routines."""
import re

import numpy as np
import pytest
import xarray as xr

from disdrodb import is_pytmatrix_available
from disdrodb.scattering import RADAR_VARIABLES, get_radar_parameters
from disdrodb.scattering.routines import (
    _check_frequency,
    available_radar_bands,
    check_radar_band,
    ensure_numerical_frequency,
    frequency_dict,
    frequency_to_wavelength,
    get_backward_geometry,
    get_forward_geometry,
    get_list_simulations_params,
    precompute_scattering_tables,
    wavelength_to_frequency,
)
from disdrodb.tests.fake_datasets import create_template_l2e_dataset


class TestRadarBands:
    """Test radar band and frequency utilities."""

    def test_available_radar_bands(self):
        """Test available_radar_bands returns all dictionary keys."""
        bands = available_radar_bands()
        assert set(bands) == set(frequency_dict.keys())

    @pytest.mark.parametrize("band", list(frequency_dict.keys()))
    def test_check_radar_band_valid(self, band):
        """Test check_radar_band returns valid band unchanged."""
        assert check_radar_band(band) == band

    @pytest.mark.parametrize("band", ["NotABand", "", None])
    def test_check_radar_band_invalid(self, band):
        """Test check_radar_band raises ValueError for invalid band."""
        with pytest.raises(ValueError):
            check_radar_band(band)

    @pytest.mark.parametrize("band", list(frequency_dict.keys()))
    def test_check_frequency_str_band(self, band):
        """Test _check_frequency converts radar band string to float frequency."""
        result = _check_frequency(band)
        assert np.isclose(result, frequency_dict[band])

    @pytest.mark.parametrize("freq", [5.4, 10, 94.05])
    def test_check_frequency_numeric(self, freq):
        """Test _check_frequency accepts numeric frequency unchanged."""
        result = _check_frequency(freq)
        assert result == freq

    def test_check_frequency_invalid_type(self):
        """Test _check_frequency raises TypeError for invalid input."""
        with pytest.raises(TypeError):
            _check_frequency({"not": "valid"})

    def test_ensure_numerical_frequency_scalars_and_list(self):
        """Test ensure_numerical_frequency handles scalars and lists."""
        # Single string band
        f1 = ensure_numerical_frequency("C")
        assert np.isclose(f1, frequency_dict["C"])

        # Single number
        f2 = ensure_numerical_frequency(5.4)
        assert np.isclose(f2, 5.4)

        # Mixed list
        f3 = ensure_numerical_frequency(["C", 9.4, "X"])
        expected = np.array([frequency_dict["C"], 9.4, frequency_dict["X"]])
        np.testing.assert_allclose(f3, expected)


@pytest.mark.parametrize("freq", [2.7, 5.4, 9.4, 35.5, 94.05])
def test_frequency_wavelength_roundtrip(freq):
    """Test wavelength-frequency conversion functions."""
    wl = frequency_to_wavelength(freq)
    freq_back = wavelength_to_frequency(wl)
    assert np.isclose(freq, freq_back, rtol=1e-12)


class TestGeometry:
    """Test backward and forward geometry definitions."""

    @pytest.mark.parametrize("angle", [0, 10, 45, 90])
    def test_backward_geometry(self, angle):
        """Test get_backward_geometry returns expected tuple."""
        theta = 90.0 - angle
        expected = (theta, 180 - theta, 0.0, 180, 0.0, 0.0)
        result = get_backward_geometry(angle)
        assert result == expected

    @pytest.mark.parametrize("angle", [0, 10, 45, 90])
    def test_forward_geometry(self, angle):
        """Test get_forward_geometry returns expected tuple."""
        theta = 90.0 - angle
        expected = (theta, theta, 0.0, 0.0, 0.0, 0.0)
        result = get_forward_geometry(angle)
        assert result == expected


@pytest.mark.skipif(not is_pytmatrix_available(), reason="pytmatrix not available")
def test_precompute_scattering_tables():
    """Test precompute_scattering_tables do not raise errors."""
    out = precompute_scattering_tables(
        frequency=[2.7, 5.6],
        num_points=[100, 200],
        diameter_max=[2.0],
        canting_angle_std=[10.0],
        axis_ratio_model=["Brandes2002"],
        permittivity_model=["Liebe1991"],
        water_temperature=[20.0],
        elevation_angle=[90.0, 80.0],
        verbose=True,
    )
    assert out is None


class TestListSimulationParams:
    """Test suite for get_list_simulations_params."""

    def test_single_value_input(self):
        """Test single values produce one parameter dict with correct entries."""
        params = get_list_simulations_params(
            frequency=5.6,
            num_points=100,
            diameter_max=5.0,
            canting_angle_std=10.0,
            axis_ratio_model="Brandes2002",
            permittivity_model="Liebe1991",
            water_temperature=20.0,
            elevation_angle=90.0,
        )
        assert isinstance(params, list)
        assert len(params) == 1
        d = params[0]
        expected_keys = {
            "frequency",
            "num_points",
            "diameter_max",
            "canting_angle_std",
            "axis_ratio_model",
            "permittivity_model",
            "water_temperature",
            "elevation_angle",
        }
        assert set(d.keys()) == expected_keys
        assert d["frequency"] == 5.6
        assert d["axis_ratio_model"] == "Brandes2002"
        assert d["permittivity_model"] == "Liebe1991"

    def test_multiple_values_cartesian_product(self):
        """Test multiple input values produce full Cartesian product of parameter combinations."""
        params = get_list_simulations_params(
            frequency=[2.7, 5.6],
            num_points=[100, 200],
            diameter_max=[2.0],
            canting_angle_std=[10.0],
            axis_ratio_model=["Brandes2002"],
            permittivity_model=["Liebe1991"],
            water_temperature=[20.0],
            elevation_angle=[90.0, 80.0],
        )
        # Expected size = 2 (freq) * 2 (num_points) * 1 * 1 * 1 * 1 * 1 * 2 = 8
        assert len(params) == 8
        freqs = [p["frequency"] for p in params]
        assert sorted(freqs) == [2.7, 2.7, 2.7, 2.7, 5.6, 5.6, 5.6, 5.6]

    def test_rounding_and_uniqueness(self):
        """Test duplicate and rounded values are collapsed into unique sorted sets."""
        params = get_list_simulations_params(
            frequency=[5.6001, 5.5999, 5.6],
            num_points=[100, 100, 100.0],
            diameter_max=[2.0, 2.0001],
            canting_angle_std=[10.0],
            axis_ratio_model=["Brandes2002"],
            permittivity_model=["Liebe1991"],
            water_temperature=[20.0, 20.0001],
            elevation_angle=[90.0],
        )
        # Expect only 1 unique combination
        assert len(params) == 1
        d = params[0]
        assert d["frequency"] == 5.6
        assert d["num_points"] == 100
        assert d["diameter_max"] == 2.0
        assert d["water_temperature"] == 20.0

    def test_invalid_axis_ratio_model_raises(self):
        """Test invalid axis ratio model raises ValueError."""
        with pytest.raises(ValueError):
            get_list_simulations_params(
                frequency=5.6,
                num_points=100,
                diameter_max=2.0,
                canting_angle_std=10.0,
                axis_ratio_model="NotAModel",
                permittivity_model="Liebe1991",
                water_temperature=20.0,
                elevation_angle=90.0,
            )

    def test_invalid_permittivity_model_raises(self):
        """Test invalid permittivity model raises ValueError."""
        with pytest.raises(ValueError):
            get_list_simulations_params(
                frequency=5.6,
                num_points=100,
                diameter_max=2.0,
                canting_angle_std=10.0,
                axis_ratio_model="Brandes2002",
                permittivity_model="NotAModel",
                water_temperature=20.0,
                elevation_angle=90.0,
            )

    def test_frequency_ordering(self):
        """Test returned frequencies are sorted ascending regardless of input order."""
        params = get_list_simulations_params(
            frequency=[35.5, 2.7, 94.05],
            num_points=100,
            diameter_max=2.0,
            canting_angle_std=10.0,
            axis_ratio_model="Brandes2002",
            permittivity_model="Liebe1991",
            water_temperature=20.0,
            elevation_angle=90.0,
        )
        freqs = [p["frequency"] for p in params]
        assert freqs == sorted(freqs)


# TODO: in future place test units pytmatrix LUTs in disdrodb software to speed up testing


@pytest.fixture(autouse=True, scope="module")
def set_scattering_table_dir(tmp_path_factory):
    """Set a scattering_table_dir for all tests in this module."""
    import disdrodb

    # Define directory
    scattering_dir = tmp_path_factory.mktemp("scattering_table")

    # Backup old value
    old_value = disdrodb.config.get("scattering_table_dir", None)

    # Set new one
    disdrodb.config.set({"scattering_table_dir": str(scattering_dir)})

    yield

    # Restore old value
    disdrodb.config.set({"scattering_table_dir": old_value})


@pytest.mark.skipif(not is_pytmatrix_available(), reason="pytmatrix not available")
class TestGetRadarParameters:

    def test_empirical_nan_psd(self):
        """Test get_radar_parameters with an empirical PSD which is all NaN."""
        ds = create_template_l2e_dataset(with_velocity=True)
        ds["drop_number_concentration"].data = np.ones(ds["drop_number_concentration"].data.shape) * np.nan
        ds_radar = get_radar_parameters(
            ds=ds,
            frequency=None,
            num_points=1024,
            diameter_max=8,
            canting_angle_std=7,
            axis_ratio_model="Thurai2007",
            permittivity_model="Turner2016",
            water_temperature=10,
            elevation_angle=0,
            parallel=False,
        )
        # Assert all radar variables are present
        assert isinstance(ds_radar, xr.Dataset)
        for var in RADAR_VARIABLES:
            assert var in ds_radar

        # Assert all radar variables are NaN
        for var in RADAR_VARIABLES:
            assert np.all(np.isnan(ds_radar[var]))

    def test_empirical_zeros_psd(self):
        """Test get_radar_parameters with an empirical PSD which is all zeros."""
        ds = create_template_l2e_dataset(with_velocity=True)
        ds["drop_number_concentration"].data = np.zeros(ds["drop_number_concentration"].data.shape)
        ds_radar = get_radar_parameters(
            ds=ds,
            frequency=None,
            num_points=1024,
            diameter_max=8,
            canting_angle_std=7,
            axis_ratio_model="Thurai2007",
            permittivity_model="Turner2016",
            water_temperature=10,
            elevation_angle=0,
            parallel=False,
        )
        # Assert all radar variables are present
        assert isinstance(ds_radar, xr.Dataset)
        for var in RADAR_VARIABLES:
            assert var in ds_radar

        # Assert all radar variables are NaN
        for var in RADAR_VARIABLES:
            assert np.all(np.isnan(ds_radar[var]))

    def test_empirical_psd_dataset(self):
        """Test get_radar_parameters with an empirical PSD dataset."""
        ds = create_template_l2e_dataset(with_velocity=True)
        ds_radar = get_radar_parameters(
            ds=ds,
            frequency=None,
            num_points=1024,
            diameter_max=8,
            canting_angle_std=7,
            axis_ratio_model="Thurai2007",
            permittivity_model="Turner2016",
            water_temperature=10,
            elevation_angle=0,
            parallel=True,
        )
        assert isinstance(ds_radar, xr.Dataset)
        for var in RADAR_VARIABLES:
            assert var in ds_radar

    def test_empirical_psd_dataset_dask_support(self):
        """Test that a Dataset with dask arrays returns lazy output and compute works."""
        ds = create_template_l2e_dataset(with_velocity=True).chunk()
        ds_radar = get_radar_parameters(
            ds=ds,
            frequency="C",
            num_points=1024,
            diameter_max=8,
            canting_angle_std=7,
            axis_ratio_model="Thurai2007",
            permittivity_model="Turner2016",
            water_temperature=10,
            elevation_angle=0,
            parallel=False,
        )
        # The output should still be a Dataset
        assert isinstance(ds_radar, xr.Dataset)
        # Should be lazy: data backed by dask array
        for var in RADAR_VARIABLES:
            assert hasattr(ds_radar[var].data, "chunks")

        # Compute should run without error
        ds_radar = ds_radar.compute()

    @pytest.mark.parametrize(
        ("param_name", "values"),
        [
            ("frequency", [4.8, 5.3]),
            ("num_points", [256, 1024]),
            ("diameter_max", [6.0, 8.0]),
            ("canting_angle_std", [5.0, 7.0]),
            ("axis_ratio_model", ["Thurai2007", "Andsager1999"]),
            ("permittivity_model", ["Turner2016", "Liebe1991"]),
            ("water_temperature", [5.0, 10.0]),
            ("elevation_angle", [0.0, 45.0]),
        ],
    )
    def test_empirical_psd_list_input_creates_dimension(self, param_name, values):
        """Check that list arguments add new dimensions in the output dataset."""
        ds = create_template_l2e_dataset(with_velocity=True)

        # Define defaults
        radar_kwargs = {
            "frequency": "C",
            "num_points": 1024,
            "diameter_max": 8,
            "canting_angle_std": 7,
            "axis_ratio_model": "Thurai2007",
            "permittivity_model": "Turner2016",
            "water_temperature": 10,
            "elevation_angle": 0,
        }
        # Overwrite the one parameter being tested with a list
        radar_kwargs[param_name] = values

        # Simulate radar variables
        ds_radar = get_radar_parameters(
            ds=ds,
            parallel=False,
            **radar_kwargs,
        )

        # Check parameter dimension is present
        assert param_name in ds_radar.dims

        # Check parameter coordinate values matches input argument
        coordinate_values = ds_radar[param_name].to_numpy().tolist()
        assert set(coordinate_values) == set(values)

    def test_model_psd_dataset(self):
        """Test get_radar_parameters with a model PSD dataset."""
        # Define L2M dataset
        n_D50 = 5
        n_Nw = 3
        D50 = xr.DataArray(np.linspace(0.5, 3.0, n_D50), dims="D50_c", name="D50_c")
        Nw = xr.DataArray(np.linspace(5e2, 2e4, n_Nw), dims="Nw_c", name="Nw_c")
        ds = xr.Dataset(coords={"D50_c": D50, "Nw_c": Nw})
        ds["D50"] = D50.expand_dims({"Nw_c": Nw})
        ds["Nw"] = Nw.expand_dims({"D50_c": D50})
        ds["mu"] = 1
        ds.attrs["disdrodb_psd_model"] = "NormalizedGammaPSD"

        # Create radar dataset
        ds_radar = get_radar_parameters(
            ds=ds,
            frequency="C",
            num_points=1024,
            diameter_max=8,
            canting_angle_std=7,
            axis_ratio_model="Thurai2007",
            permittivity_model="Turner2016",
            water_temperature=10,
            parallel=False,
        )
        assert isinstance(ds_radar, xr.Dataset)
        for var in RADAR_VARIABLES:
            assert var in ds_radar

    def test_model_psd_dataset_dask_support(self):
        """Test that a PSD Model Dataset with dask arrays returns lazy output and compute works."""
        # Define L2M dataset
        n_timesteps = 6

        D50 = xr.DataArray(np.linspace(0.5, 3.0, n_timesteps), dims="time", name="D50")
        Nw = xr.DataArray(np.linspace(5e2, 2e4, n_timesteps), dims="time", name="Nw")
        ds = xr.Dataset({"D50": D50, "Nw": Nw})
        ds["mu"] = 1
        ds.attrs["disdrodb_psd_model"] = "NormalizedGammaPSD"

        # Creaze lazy dataset
        ds = ds.chunk({"time": 2})

        # Create radar dataset
        ds_radar = get_radar_parameters(
            ds=ds,
            frequency="C",
            num_points=1024,
            diameter_max=8,
            canting_angle_std=7,
            axis_ratio_model="Thurai2007",
            permittivity_model="Turner2016",
            water_temperature=10,
            elevation_angle=0,
            parallel=False,
        )
        # The output should still be a Dataset
        assert isinstance(ds_radar, xr.Dataset)
        # Should be lazy: data backed by dask array
        for var in RADAR_VARIABLES:
            assert hasattr(ds_radar[var].data, "chunks")

        # Compute should run without error
        ds_radar = ds_radar.compute()

    @pytest.mark.parametrize(
        ("param_name", "values"),
        [
            ("frequency", ["C", "X"]),
            ("num_points", [256, 1024]),
            ("diameter_max", [6.0, 8.0]),
            ("canting_angle_std", [5.0, 7.0]),
            ("axis_ratio_model", ["Thurai2007", "Andsager1999"]),
            ("permittivity_model", ["Turner2016", "Liebe1991"]),
            ("water_temperature", [5.0, 10.0]),
            ("elevation_angle", [0.0, 45.0]),
        ],
    )
    def test_model_psd_list_input_creates_dimension(self, param_name, values):
        """Check that list arguments add new dimensions in the output dataset."""
        # Define L2M dataset
        n_timesteps = 6

        D50 = xr.DataArray(np.linspace(0.5, 3.0, n_timesteps), dims="time", name="D50")
        Nw = xr.DataArray(np.linspace(5e2, 2e4, n_timesteps), dims="time", name="Nw")
        ds = xr.Dataset({"D50": D50, "Nw": Nw})
        ds["mu"] = 1
        ds.attrs["disdrodb_psd_model"] = "NormalizedGammaPSD"

        # Define defaults
        radar_kwargs = {
            "frequency": "C",
            "num_points": 1024,
            "diameter_max": 8,
            "canting_angle_std": 7,
            "axis_ratio_model": "Thurai2007",
            "permittivity_model": "Turner2016",
            "water_temperature": 10,
            "elevation_angle": 0,
        }
        # Overwrite the one parameter being tested with a list
        radar_kwargs[param_name] = values

        # Simulate radar variables
        ds_radar = get_radar_parameters(
            ds=ds,
            parallel=False,
            **radar_kwargs,
        )

        # Check parameter dimension is present
        assert param_name in ds_radar.dims

        # Check parameter coordinate values matches input argument
        coordinate_values = ds_radar[param_name].to_numpy().tolist()
        assert set(coordinate_values) == set(values)

    def test_get_radar_parameters_invalid_dataset_raises(self):
        """Check that get_radar_parameters raises ValueError if dataset is not L2E or L2M."""
        # Create a dummy dataset with irrelevant variable
        ds = xr.Dataset(
            {"foo": (("time",), np.arange(5))},
            coords={"time": np.arange(5)},
        )

        with pytest.raises(ValueError, match="not a DISDRODB L2E or L2M product"):
            get_radar_parameters(
                ds=ds,
                frequency=5.0,  # numeric frequency in GHz
                parallel=True,
            )

    def test_missing_psd_parameters_raise_error(self):
        """Test that missing PSD model parameters in xarray Dataset raise error."""
        # Define L2M dataset
        n_timesteps = 6
        D50 = xr.DataArray(np.linspace(0.5, 3.0, n_timesteps), dims="time", name="D50")
        Nw = xr.DataArray(np.linspace(5e2, 2e4, n_timesteps), dims="time", name="Nw")
        ds = xr.Dataset({"D50": D50, "Nw": Nw})
        ds.attrs["disdrodb_psd_model"] = "NormalizedGammaPSD"

        with pytest.raises(ValueError, match=re.escape("The NormalizedGammaPSD parameters ['mu'] are not present")):
            get_radar_parameters(
                ds=ds,
                frequency=5.0,  # numeric frequency in GHz
                parallel=True,
            )
