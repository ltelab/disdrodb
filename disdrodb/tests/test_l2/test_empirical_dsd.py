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
"""Testing module for empirical DSD parameters computations."""

import os
import pytest
import xarray as xr
import numpy as np
import dask.array
from disdrodb.api.configs import available_sensor_names
from disdrodb.l2.empirical_dsd import (
    get_rain_rate_from_dsd,
    get_equivalent_reflectivity_factor,
    get_total_number_concentration,
    get_moment,
    get_mass_spectrum,
    get_liquid_water_content,
    get_quantile_volume_drop_diameter,
    get_median_volume_drop_diameter,
    get_mean_volume_drop_diameter,
    get_std_volume_drop_diameter,
    get_mode_diameter,
    get_effective_sampling_area,

)
from disdrodb import __root_path__

 

@pytest.fixture(scope="session")
def template_dataset():
    """
    Read a template NetCDF file once for all tests.
    """
    time = xr.DataArray(np.array([0, 1], dtype=float), dims="time")
    diameter_bin_center = xr.DataArray(np.array([0.2, 0.4, 0.6, 0.8]), dims="diameter_bin_center")
    diameter_bin_width = xr.DataArray(np.array([0.2, 0.2, 0.2, 0.2]), dims="diameter_bin_center")
    velocity_bin_center = xr.DataArray(np.array([0.2, 0.5, 1]), dims="velocity_bin_center")
    fall_velocity = xr.DataArray(np.array([[0.5, 1, 1.5, 2], [0.5, 1, 1.5, 2]]), dims=("time", "diameter_bin_center"))
    drop_number_concentration = xr.DataArray(
        np.array([[0, 10000, 5000, 500], [0, 10000, 5000, 500]]), dims=("time", "diameter_bin_center")
    )

    ds = xr.Dataset(
        data_vars={
            "fall_velocity": fall_velocity,
            "drop_number_concentration": drop_number_concentration,
        },
        coords={
            "time": time,
            "diameter_bin_center": diameter_bin_center,
            "diameter_bin_width": diameter_bin_width,
            "velocity_bin_center": velocity_bin_center,
        },
    )
    return ds

def _prepare_test_dataset(
    template_ds: xr.Dataset,
    scenario: str,
    array_type: str,
    variables: list,
):
    """
    Return the test dataset (with potential modifications).
    `scenario` can be "zeros", "nans", or "realistic".
    `array_type` can be "numpy" or "dask".
    """

    # Copy the template so we don't mutate the original fixture
    ds = template_ds.copy(deep=True)

    # Ensure variables is a list
    if isinstance(variables, str):
        variables = [variables]

    # Modify variables based on scenario !
    if scenario == "zeros":
        for var in variables:
            ds[var].data = np.zeros(ds[var].shape)
    elif scenario == "nans":
        for var in variables:
            ds[var].data = np.zeros(ds[var].shape) * np.nan

    # Convert the dataset to dask arrays if requested
    if array_type == "dask":
        ds = ds.chunk({"time": 1})

    # Return the dataset
    return ds


def check_datarray_type(data_array, array_type):
    assert isinstance(data_array, xr.DataArray)
    if array_type == "numpy":
        assert isinstance(data_array.data, np.ndarray)
    else:
        assert isinstance(data_array.data, dask.array.Array)


class TestGetEffectiveSamplingArea:

    @pytest.mark.parametrize("sensor_name", [s for s in available_sensor_names() if s.startswith("OTT_Parsivel")])
    def test_positive_sampling_area(self, sensor_name):
        """Test Parsivel sampling area calculation."""
        diameter = 0.001  # 1mm
        sampling_area = get_effective_sampling_area(sensor_name, diameter)
        assert isinstance(sampling_area, float), "Sampling area should be a float"
        assert sampling_area > 0, "Sampling area should be positive"
    
    @pytest.mark.parametrize("sensor_name", [s for s in available_sensor_names() if s.startswith("OTT_Parsivel")])
    def test_sampling_area_with_diameter_array(self, template_dataset, sensor_name):
        """Test Parsivel sampling area calculation."""
        diameter = template_dataset["diameter_bin_center"]/1000 # meters
        sampling_area = get_effective_sampling_area(sensor_name, diameter)
        assert isinstance(sampling_area, xr.DataArray)
        assert np.all(sampling_area.values > 0), "Sampling area should be positive"
            
    def test_invalid_sensor_name(self):
        """Test raises error for invalid sensor name."""
        with pytest.raises(ValueError):
            get_effective_sampling_area("Invalid_Sensor", diameter=0.001)
            

class TestGetRainRateFromDSD:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_rain_rate_zero_values(self, template_dataset, array_type):
        """Test returns 0 when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="zeros", array_type=array_type
        )
        output = get_rain_rate_from_dsd(
            drop_number_concentration=ds["drop_number_concentration"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )

        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with 0 as result
        assert np.all(output.values == 0), "All values should be zeros for for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_rain_rate_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="nans", array_type=array_type
        )

        output = get_rain_rate_from_dsd(
            drop_number_concentration=ds["drop_number_concentration"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with 0 as result
        assert np.all(np.isnan(output.values)), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_rain_rate_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="realistic", array_type=array_type
        )
        output = get_rain_rate_from_dsd(
            drop_number_concentration=ds["drop_number_concentration"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        np.testing.assert_allclose(output.values, np.array([1.04501938, 1.04501938]))


class TestGetEquivalentReflectivityFactor:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_equivalent_reflectivity_factor_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are 0."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="zeros", array_type=array_type
        )
        output = get_equivalent_reflectivity_factor(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with nan as result (because output is in dB)
        assert np.all(np.isnan(output.values)), "All values should be NaN for for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_equivalent_reflectivity_factor_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="nans", array_type=array_type
        )
        output = get_equivalent_reflectivity_factor(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with 0 as result
        assert np.all(np.isnan(output.values)), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_equivalent_reflectivity_factor_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="realistic", array_type=array_type
        )
        output = get_equivalent_reflectivity_factor(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        np.testing.assert_allclose(output.values, np.array([19.08819458, 19.08819458]))


class TestGetMassSpectrum:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mass_spectrum_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are 0."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="zeros", array_type=array_type
        )
        output = get_mass_spectrum(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with 0 as result
        assert np.all(output.values == 0), "All values should be zeros for for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mass_spectrum_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="nans", array_type=array_type
        )
        output = get_mass_spectrum(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with 0 as result
        assert np.all(np.isnan(output.values)), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mass_spectrum_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="realistic", array_type=array_type
        )
        output = get_mass_spectrum(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        expected_array = np.array(
            [[0.0, 0.0], [0.33510322, 0.33510322], [0.56548668, 0.56548668], [0.13404129, 0.13404129]]
        )
        np.testing.assert_allclose(output.values, expected_array)


class TestGetLiquidWaterContent:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_lwc_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are 0."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="zeros", array_type=array_type
        )
        output = get_liquid_water_content(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with 0 as result
        assert np.all(output.values == 0), "All values should be zeros for for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_lwc_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="nans", array_type=array_type
        )
        output = get_liquid_water_content(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with 0 as result
        assert np.all(np.isnan(output.values)), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_lwc_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="realistic", array_type=array_type
        )
        output = get_liquid_water_content(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        np.testing.assert_allclose(output.values, np.array([0.20692624, 0.20692624]))

class TestGetQuantileVolumeDropDiameter:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_quantile_volume_drop_diameter_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="zeros", array_type=array_type
        )
        output = get_quantile_volume_drop_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            fraction=0.5,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with NaN as result
        assert np.all(np.isnan(output.values)), "All values should be NaN for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_quantile_volume_drop_diameter_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="nans", array_type=array_type
        )
        output = get_quantile_volume_drop_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            fraction=0.5,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with NaN as result
        assert np.all(np.isnan(output.values)), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_quantile_volume_drop_diameter_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="realistic", array_type=array_type
        )
        output = get_quantile_volume_drop_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            fraction=0.5,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values - actual value depends on the data distribution in the template
        np.testing.assert_allclose(output.values, np.array([0.46444444, 0.46444444]))
    
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_median_volume_drop_diameter_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="realistic", array_type=array_type
        )
        output = get_median_volume_drop_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        np.testing.assert_allclose(output.values, np.array([0.46444444, 0.46444444]))
       
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_quantile_volume_drop_diameter_multiple_fractions(self, template_dataset, array_type):
        """Test with multiple quantile fractions."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="realistic", array_type="numpy"
        )
        output = get_quantile_volume_drop_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            fraction=[0.1, 0.25, 0.75, 0.9],
        )
        # Check dimension quantile of correct size
        assert output.sizes["quantile"] == 4
        # For different fractions, we expect valid outputs that increase with the fraction
        assert not np.any(np.isnan(output.values)), "Values should not be NaN"
        # Test expected values
        expected_values = np.array([0.26175, 0.354375, 0.55592593, 0.645625])
        np.testing.assert_allclose(output.isel(time=0).values, expected_values)
        
    def test_quantile_volume_drop_diameter_invalid_fraction(self, template_dataset):
        """Test with invalid fraction values."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="realistic", array_type="numpy"
        )
        
        # Test with fraction = 0
        with pytest.raises(ValueError):
            get_quantile_volume_drop_diameter(
                drop_number_concentration=ds["drop_number_concentration"],
                diameter=ds["diameter_bin_center"] / 1000,
                diameter_bin_width=ds["diameter_bin_width"],
                fraction=0,
            )
            
        # Test with fraction = 1
        with pytest.raises(ValueError):
            get_quantile_volume_drop_diameter(
                drop_number_concentration=ds["drop_number_concentration"],
                diameter=ds["diameter_bin_center"] / 1000,
                diameter_bin_width=ds["diameter_bin_width"],
                fraction=1,
            )


class TestGetMeanVolumeDropDiameter:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mean_volume_drop_diameter_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="zeros", array_type=array_type
        )
        # Calculate moments
        moment_3 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=3,
        )
        moment_4 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=4,
        )

        # Compute mean_volume_drop_diameter
        output = get_mean_volume_drop_diameter(
            moment_3=moment_3,
            moment_4=moment_4,
        )
          
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with NaN as result
        assert np.all(np.isnan(output.values)), "All values should be NaN for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mean_volume_drop_diameter_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="nans", array_type=array_type
        )
        # Calculate moments
        moment_3 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=3,
        )
        moment_4 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=4,
        )
    
        # Compute mean_volume_drop_diameter
        output = get_mean_volume_drop_diameter(
            moment_3=moment_3,
            moment_4=moment_4,
        )
          
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with NaN as result
        assert np.all(np.isnan(output.values)), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mean_volume_drop_diameter_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="realistic", array_type=array_type
        )
        # Calculate moments
        moment_3 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=3,
        )
        moment_4 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=4,
        )      
        # Compute mean_volume_drop_diameter
        output = get_mean_volume_drop_diameter(
            moment_3=moment_3,
            moment_4=moment_4,
        )
          
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values (these values should be calculated based on your test data)
        expected_values = np.array([0.5611336, 0.5611336])  # Replace with actual expected values
        np.testing.assert_allclose(output.values, expected_values, rtol=1e-6)
    

class TestGetStdVolumeDropDiameter:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_std_volume_drop_diameter_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="zeros", array_type=array_type
        )
        # Calculate moments
        moment_3 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=3,
        )
        moment_4 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=4,
        )
        moment_5 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=5,
        )        
        # Compute std_volume_drop_diameter
        output = get_std_volume_drop_diameter(
            moment_3=moment_3,
            moment_4=moment_4,
            moment_5=moment_5,
        )
          
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with NaN as result
        assert np.all(np.isnan(output.values)), "All values should be NaN for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_std_volume_drop_diameter_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="nans", array_type=array_type
        )
        # Calculate moments
        moment_3 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=3,
        )
        moment_4 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=4,
        )
        moment_5 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=5,
        )        
        # Compute std_volume_drop_diameter
        output = get_std_volume_drop_diameter(
            moment_3=moment_3,
            moment_4=moment_4,
            moment_5=moment_5,
        )
          
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with NaN as result
        assert np.all(np.isnan(output.values)), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_std_volume_drop_diameter_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="realistic", array_type=array_type
        )
        # Calculate moments
        moment_3 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=3,
        )
        moment_4 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=4,
        )
        moment_5 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=5,
        )        
        # Compute std_volume_drop_diameter
        output = get_std_volume_drop_diameter(
            moment_3=moment_3,
            moment_4=moment_4,
            moment_5=moment_5,
        )
          
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values (these values should be calculated based on your test data)
        expected_values = np.array([0.12894594, 0.12894594])  # Replace with actual expected values
        np.testing.assert_allclose(output.values, expected_values, rtol=1e-6)

class TestGetTotalNumberConcentration:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"]) 
    def test_total_number_concentration_zero_values(self, template_dataset, array_type):
        """Test returns 0 when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="zeros", array_type=array_type
        )
        output = get_total_number_concentration(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter_bin_width=ds["diameter_bin_width"]
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with 0 as result
        assert np.all(output.values == 0), "All values should be zeros for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_total_number_concentration_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="nans", array_type=array_type
        )
        output = get_total_number_concentration(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter_bin_width=ds["diameter_bin_width"]
        )
        # Test output type 
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with NaN as result
        assert np.all(np.isnan(output.values)), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_total_number_concentration_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="realistic", array_type=array_type
        )
        output = get_total_number_concentration(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter_bin_width=ds["diameter_bin_width"]
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values 
        np.testing.assert_allclose(output.values, np.array([3100, 3100]))
        # Check equivalent to moment 0
        output_moment_0 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000, 
            diameter_bin_width=ds["diameter_bin_width"],
            moment=0,
        )
        np.testing.assert_allclose(output.values, output_moment_0.values)
        

class TestGetMoment:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_moment_zero_values(self, template_dataset, array_type):
        """Test returns 0 when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="zeros", array_type=array_type
        )
        output = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000, 
            diameter_bin_width=ds["diameter_bin_width"],
            moment=3
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with 0 as result
        assert np.all(output.values == 0), "All values should be zeros for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_moment_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="nans", array_type=array_type
        )
        output = get_moment( 
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"], 
            moment=3
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with NaN as result
        assert np.all(np.isnan(output.values)), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_moment_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="realistic", array_type=array_type
        )
        output = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=3
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        np.testing.assert_allclose(output.values, np.array([395.2, 395.2]))

class TestGetModeDiameter:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mode_diameter_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="zeros", array_type=array_type
        )
        output = get_mode_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with NaN as result 
        assert np.all(np.isnan(output.values)), "All values should be NaN for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"]) 
    def test_mode_diameter_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="nans", array_type=array_type
        )
        output = get_mode_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with NaN as result
        assert np.all(np.isnan(output.values)), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mode_diameter_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset, variables="drop_number_concentration", scenario="realistic", array_type=array_type
        )
        output = get_mode_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values - for template dataset drop count peak is at 0.4mm
        np.testing.assert_allclose(output.values, np.array([0.4, 0.4]))

    def test_mode_diameter_when_multiple_modes(self, template_dataset):
        """Test case when there are multiple equal peaks."""
        ds = template_dataset.copy(deep=True)
        # Create dataarray with two equal peaks 
        ds["drop_number_concentration"].values = np.array([[0, 1000, 1000, 0], [0, 1000, 1000, 0]])
        output = get_mode_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"],
        )
        # Check it return diameter of first peak occurrence
        np.testing.assert_allclose(output.values, np.array([0.4, 0.4]))



