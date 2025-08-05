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
import dask.array
import numpy as np
import pytest
import xarray as xr

from disdrodb import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.api.configs import available_sensor_names
from disdrodb.l2.empirical_dsd import (
    get_bin_dimensions,
    get_drop_average_velocity,
    get_drop_number_concentration,
    get_drop_volume,
    get_effective_sampling_area,
    get_equivalent_reflectivity_factor,
    get_kinetic_energy_variables,
    get_kinetic_energy_variables_from_drop_number,
    get_liquid_water_content,
    get_liquid_water_content_from_moments,
    get_liquid_water_spectrum,
    get_mean_volume_drop_diameter,
    get_median_volume_drop_diameter,
    get_min_max_diameter,
    get_mode_diameter,
    get_moment,
    get_normalized_intercept_parameter,
    get_normalized_intercept_parameter_from_moments,
    get_quantile_volume_drop_diameter,
    get_rain_accumulation,
    get_rain_rate,
    get_rain_rate_from_drop_number,
    get_std_volume_drop_diameter,
    get_total_number_concentration,
)
from disdrodb.tests.fake_datasets import create_template_dataset


@pytest.fixture(scope="session")
def template_dataset():
    """Read a template NetCDF file once for all tests."""
    ds = create_template_dataset(with_velocity=True)
    return ds


# template_dataset = create_template_dataset(with_velocity=True)


def _prepare_test_dataset(
    template_ds: xr.Dataset,
    scenario: str,
    array_type: str,
    variables: list,
):
    """Return the test dataset (with potential modifications).

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
            if var not in ds.coords:
                ds[var].data = np.zeros(ds[var].shape)
            else:
                ds = ds.assign_coords({var: np.zeros(ds[var].shape)})

    if scenario == "nans":
        for var in variables:
            if var not in ds.coords:
                ds[var].data = np.zeros(ds[var].shape) * np.nan
            else:
                ds = ds.assign_coords({var: np.zeros(ds[var].shape) * np.nan})

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


class TestEffectiveSamplingArea:

    @pytest.mark.parametrize("sensor_name", available_sensor_names())
    def test_positive_sampling_area(self, sensor_name):
        """Test Parsivel sampling area calculation."""
        diameter = 0.001  # 1mm
        sampling_area = get_effective_sampling_area(sensor_name, diameter)
        assert isinstance(sampling_area, float), "Sampling area should be a float"
        assert sampling_area > 0, "Sampling area should be positive"

    @pytest.mark.parametrize("sensor_name", available_sensor_names())
    def test_sampling_area_with_diameter_array(self, template_dataset, sensor_name):
        """Test Parsivel sampling area calculation."""
        diameter = template_dataset["diameter_bin_center"] / 1000  # meters
        sampling_area = get_effective_sampling_area(sensor_name, diameter)
        if sensor_name in ["PARSIVEL", "PARSIVEL2", "LPM"]:
            assert isinstance(sampling_area, xr.DataArray)
            assert np.all(sampling_area.to_numpy() > 0), "Sampling area should be positive"
        else:  # IMPACT_SENSORS
            assert isinstance(sampling_area, float)
            assert sampling_area > 0, "Sampling area should be positive"

    def test_invalid_sensor_name(self):
        """Test raises error for invalid sensor name."""
        with pytest.raises(ValueError):
            get_effective_sampling_area("Invalid_Sensor", diameter=0.001)


class TestRainRate:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_rain_rate_zero_values(self, template_dataset, array_type):
        """Test returns 0 when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )
        output = get_rain_rate(
            drop_number_concentration=ds["drop_number_concentration"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )

        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with 0 as result
        assert np.all(output.to_numpy() == 0), "All values should be zeros for for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_rain_rate_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="nans",
            array_type=array_type,
        )

        output = get_rain_rate(
            drop_number_concentration=ds["drop_number_concentration"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with 0 as result
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_rain_rate_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
        )
        output = get_rain_rate(
            drop_number_concentration=ds["drop_number_concentration"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        np.testing.assert_allclose(output.to_numpy(), np.array([1.04501938, 1.04501938]))

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_same_rain_rate_results_when_same_velocity_used(self, template_dataset, array_type):
        """Test both methods give same results when using same estimated fall velocity."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="realistic",
            array_type=array_type,
        )

        # Define realistic drop_number concentration
        sampling_area = 0.005
        sample_interval = 60
        ds["drop_number_concentration"] = get_drop_number_concentration(
            drop_number=ds["drop_number"],
            velocity=ds["fall_velocity"],
            diameter_bin_width=ds["diameter_bin_width"],
            sampling_area=sampling_area,
            sample_interval=sample_interval,
        )

        # Calculate using drop_number_concentration method
        output1 = get_rain_rate(
            drop_number_concentration=ds["drop_number_concentration"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )

        # Calculate using drop_number method
        output2 = get_rain_rate_from_drop_number(
            drop_number=ds["drop_number"],
            diameter=ds["diameter_bin_center"] / 1000,
            sampling_area=sampling_area,
            sample_interval=sample_interval,
        )
        xr.testing.assert_allclose(output1, output2)

    def test_presence_velocity_dimension_in_inputs_raise_error(self, template_dataset):
        """Test raises error when velocity DataArray has the velocity dimension."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type="numpy",
        )

        # Create velocity with velocity dimension
        velocity = xr.ones_like(ds["drop_number"]) * ds["velocity_bin_center"]

        # Test raises error
        with pytest.raises(ValueError):
            get_rain_rate(
                drop_number_concentration=ds["drop_number_concentration"],
                velocity=velocity,
                diameter=ds["diameter_bin_center"] / 1000,
                diameter_bin_width=ds["diameter_bin_width"],
            )


def test_rain_accumulation():
    """Test that get_rain_accumulation computations given a known rain rate and interval."""
    # 10 mm/h for 1 hour (3600 s) should yield 10 mm
    assert get_rain_accumulation(10.0, 3600) == 10.0
    # 10 mm/h for 30 minutes (1800 s) should yield 5 mm
    assert get_rain_accumulation(10.0, 1800) == 5
    # Assert is vectorized
    rain_rates = np.array([0, 5, 10])  # mm/h
    # Sample interval of 1800 seconds (0.5 hour)
    expected = np.array([0.0, 2.5, 5.0])
    result = get_rain_accumulation(rain_rates, 1800)
    np.testing.assert_allclose(result, expected)


class TestEquivalentReflectivityFactor:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_equivalent_reflectivity_factor_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are 0."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )
        output = get_equivalent_reflectivity_factor(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with nan as result (because output is in dB)
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_equivalent_reflectivity_factor_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="nans",
            array_type=array_type,
        )
        output = get_equivalent_reflectivity_factor(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with 0 as result
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_equivalent_reflectivity_factor_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
        )
        output = get_equivalent_reflectivity_factor(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        np.testing.assert_allclose(output.to_numpy(), np.array([19.08819458, 19.08819458]))

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_equivalent_reflectivity_factor_clipped_at_minus_60(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
        )
        output = get_equivalent_reflectivity_factor(
            drop_number_concentration=ds["drop_number_concentration"] / 1000_000_000,
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        assert output.to_numpy().min() == -60


class TestMassSpectrum:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mass_spectrum_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are 0."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )
        output = get_liquid_water_spectrum(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with 0 as result
        assert np.all(output.to_numpy() == 0), "All values should be zeros for for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mass_spectrum_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="nans",
            array_type=array_type,
        )
        output = get_liquid_water_spectrum(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with 0 as result
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mass_spectrum_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
        )
        output = get_liquid_water_spectrum(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        expected_array = np.array(
            [[0.0, 0.0], [0.33510322, 0.33510322], [0.56548668, 0.56548668], [0.13404129, 0.13404129]],
        )
        np.testing.assert_allclose(output.to_numpy(), expected_array)


class TestLiquidWaterContent:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_liquid_water_content_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are 0."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )
        output = get_liquid_water_content(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with 0 as result
        assert np.all(output.to_numpy() == 0), "All values should be zeros for for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_liquid_water_content_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="nans",
            array_type=array_type,
        )
        output = get_liquid_water_content(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with 0 as result
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_liquid_water_content_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
        )
        output = get_liquid_water_content(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        np.testing.assert_allclose(output.to_numpy(), np.array([0.20692624, 0.20692624]))

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_get_liquid_water_content_from_moments(self, template_dataset, array_type):
        """Test get same values from get_liquid_water_content_from_moments."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
        )
        expected = get_liquid_water_content(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        moment_3 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=3,
        )
        output = get_liquid_water_content_from_moments(moment_3, water_density=1000)
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        np.testing.assert_allclose(output.to_numpy(), expected.to_numpy())


class TestQuantileVolumeDropDiameter:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_quantile_volume_drop_diameter_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
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
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_quantile_volume_drop_diameter_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="nans",
            array_type=array_type,
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
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_quantile_volume_drop_diameter_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
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
        np.testing.assert_allclose(output.to_numpy(), np.array([0.46444444, 0.46444444]))

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_median_volume_drop_diameter_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
        )
        output = get_median_volume_drop_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        np.testing.assert_allclose(output.to_numpy(), np.array([0.46444444, 0.46444444]))

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_quantile_output_with_drops_in_just_one_bin(self, template_dataset, array_type):
        """Test that when drops occurs in just 1 bin, the output is the diameter bin center."""
        # Create drop_number_concentration with values in just 1 bin in the middle of the array
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )
        ds["drop_number_concentration"].data[:, 2] = 10

        output = get_quantile_volume_drop_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            fraction=[0.001, 0.5, 0.999],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values - actual value depends on the data distribution in the template
        np.testing.assert_allclose(output.isel(time=0).to_numpy(), np.array([0.5002, 0.6, 0.6998]))

        # ------------------------------------------------------------------------------.
        # Create drop_number_concentration with values at the end of the array
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )
        ds["drop_number_concentration"].data[:, -1] = 10

        output = get_quantile_volume_drop_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            fraction=[0.001, 0.5, 0.999],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values - actual value depends on the data distribution in the template
        np.testing.assert_allclose(output.isel(time=0).to_numpy(), np.array([0.7002, 0.8, 0.8998]))

        # ------------------------------------------------------------------------------.
        # Create drop_number_concentration with values at the start of the array
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )
        ds["drop_number_concentration"].data[:, 0] = 10

        output = get_quantile_volume_drop_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            fraction=[0.001, 0.5, 0.999],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values - actual value depends on the data distribution in the template
        np.testing.assert_allclose(output.isel(time=0).to_numpy(), np.array([0.1002, 0.2, 0.2998]))

        # ------------------------------------------------------------------------------.

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_quantile_volume_drop_diameter_multiple_fractions(self, template_dataset, array_type):
        """Test with multiple quantile fractions."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
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
        assert not np.any(np.isnan(output.to_numpy())), "Values should not be NaN"
        # Test expected values
        expected_values = np.array([0.26175, 0.354375, 0.55592593, 0.645625])
        np.testing.assert_allclose(output.isel(time=0).to_numpy(), expected_values)

    def test_quantile_volume_drop_diameter_invalid_fraction(self, template_dataset):
        """Test with invalid fraction values."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type="numpy",
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


class TestMeanVolumeDropDiameter:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mean_volume_drop_diameter_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
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
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mean_volume_drop_diameter_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="nans",
            array_type=array_type,
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
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mean_volume_drop_diameter_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
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
        np.testing.assert_allclose(output.to_numpy(), expected_values, rtol=1e-6)

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_dm_output_with_drops_in_just_one_bin(self, template_dataset, array_type):
        """Test that when drops occurs in just 1 bin, the output is the diameter bin center."""
        # Create drop_number_concentration with values in just 1 bin
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )

        ds["drop_number_concentration"].data[:, 2] = 10

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
        expected_values = np.array([0.6, 0.6])  # Replace with actual expected values
        np.testing.assert_allclose(output.to_numpy(), expected_values, rtol=1e-6)


class TestStdVolumeDropDiameter:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_std_volume_drop_diameter_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
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
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_std_volume_drop_diameter_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="nans",
            array_type=array_type,
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
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_std_volume_drop_diameter_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
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
        np.testing.assert_allclose(output.to_numpy(), expected_values, rtol=1e-6)

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_sigmam_output_with_drops_in_just_one_bin(self, template_dataset, array_type):
        """Test that when drops occurs in just 1 bin, the output is 0."""
        # Create drop_number_concentration with values in just 1 bin
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )

        ds["drop_number_concentration"].data[:, 2] = 10

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
        expected_values = np.array([0.0, 0.0])  # Replace with actual expected values
        np.testing.assert_allclose(output.to_numpy(), expected_values, rtol=1e-6)


class TestTotalNumberConcentration:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_total_number_concentration_zero_values(self, template_dataset, array_type):
        """Test returns 0 when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )
        output = get_total_number_concentration(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with 0 as result
        assert np.all(output.to_numpy() == 0), "All values should be zeros for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_total_number_concentration_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="nans",
            array_type=array_type,
        )
        output = get_total_number_concentration(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with NaN as result
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_total_number_concentration_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
        )
        output = get_total_number_concentration(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        np.testing.assert_allclose(output.to_numpy(), np.array([3100, 3100]))
        # Check equivalenance with moment 0
        output_moment_0 = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=0,
        )
        np.testing.assert_allclose(output.to_numpy(), output_moment_0.to_numpy())


class TestMoment:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_moment_zero_values(self, template_dataset, array_type):
        """Test returns 0 when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )
        output = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=3,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with 0 as result
        assert np.all(output.to_numpy() == 0), "All values should be zeros for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_moment_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="nans",
            array_type=array_type,
        )
        output = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=3,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with NaN as result
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_moment_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
        )
        output = get_moment(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            moment=3,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        np.testing.assert_allclose(output.to_numpy(), np.array([395.2, 395.2]))


class TestGetMinMaxDiameter:
    """Test suite for the `get_min_max_diameter` function."""

    def test_single_time_partial_non_zero(self):
        """Check correct min and max diameters for a single time step."""
        # Create a 2D DataArray with one time step.
        # Data contains zeros except for a range of non-zero values in the middle.
        diameters = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        data = np.array([[0, 0, 10, 5, 0]])  # shape: (time=1, diameter_bin_center=5)

        da = xr.DataArray(
            data,
            dims=["time", "diameter_bin_center"],
            coords={"diameter_bin_center": diameters},
        )

        min_d, max_d = get_min_max_diameter(da)

        # We expect:
        # min_diameter at 0.3 (index 2)
        # max_diameter at 0.4 (index 3)
        expected_min = xr.DataArray([0.3], dims=["time"])
        expected_max = xr.DataArray([0.4], dims=["time"])

        xr.testing.assert_allclose(min_d, expected_min)
        xr.testing.assert_allclose(max_d, expected_max)

    def test_multiple_times_mixed_values(self):
        """Check correct min and max diameters for multiple time steps."""
        diameters = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        data = np.array(
            [
                [0, 2, 3, 4, 0],  # time=0 => non-zero from indices 1 to 3 => 0.2 to 0.4
                [0, 0, 0, 0, 0],  # time=1 => all zeros
                [5, 0, 0, 0, 6],  # time=2 => non-zero at indices 0 and 4 => 0.1 and 0.5
                [np.nan, np.nan, np.nan, np.nan, np.nan],  # time=3 => all nan
                [0, np.nan, 0, np.nan, 0],  # time=4 => mixture of nan and 0
                [np.nan, 2, 4, 3, np.nan],  # time=5 => mixture of nan and counts
            ],
        )

        da = xr.DataArray(
            data,
            dims=["time", "diameter_bin_center"],
            coords={"diameter_bin_center": diameters},
        )

        min_d, max_d = get_min_max_diameter(da)

        # For time=0 => min=0.2, max=0.4
        # For time=1 => all zero => NaN, NaN
        # For time=2 => min=0.1, max=0.5
        # For time=3 => all nan => NaN, NaN
        # For time=4 => some nan => min=0.2, max=0.4
        expected_min = xr.DataArray([0.2, np.nan, 0.1, np.nan, np.nan, 0.2], dims=["time"])
        expected_max = xr.DataArray([0.4, np.nan, 0.5, np.nan, np.nan, 0.4], dims=["time"])

        xr.testing.assert_allclose(min_d, expected_min)
        xr.testing.assert_allclose(max_d, expected_max)

    def test_all_zero(self):
        """Check behavior when all drop_counts values are zero."""
        diameters = np.array([1, 2, 3])
        data = np.zeros((2, 3))  # 2 time steps, all zero counts
        da = xr.DataArray(
            data,
            dims=["time", "diameter_bin_center"],
            coords={"diameter_bin_center": diameters},
        )

        min_d, max_d = get_min_max_diameter(da)

        # All are zero, so expect NaN for both min and max for each time step
        expected_min = xr.DataArray([np.nan, np.nan], dims=["time"])
        expected_max = xr.DataArray([np.nan, np.nan], dims=["time"])

        xr.testing.assert_allclose(min_d, expected_min)
        xr.testing.assert_allclose(max_d, expected_max)


class TestModeDiameter:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mode_diameter_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )
        output = get_mode_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with NaN as result
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mode_diameter_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="nans",
            array_type=array_type,
        )
        output = get_mode_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with NaN as result
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_mode_diameter_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
        )
        output = get_mode_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"],
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values - for template dataset drop count peak is at 0.4mm
        np.testing.assert_allclose(output.to_numpy(), np.array([0.4, 0.4]))

    def test_mode_diameter_when_multiple_modes(self, template_dataset):
        """Test case when there are multiple equal peaks."""
        ds = template_dataset.copy(deep=True)
        # Create dataarray with two equal peaks
        ds["drop_number_concentration"].data = np.array([[0, 1000, 1000, 0], [0, 1000, 1000, 0]])
        output = get_mode_diameter(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"],
        )
        # Check it return diameter of first peak occurrence
        np.testing.assert_allclose(output.to_numpy(), np.array([0.4, 0.4]))


class TestDropVolume:
    """Test get_drop_volume function."""

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_drop_volume_zero_values(self, template_dataset, array_type):
        """Test returns 0 when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="diameter_bin_center",
            scenario="zeros",
            array_type=array_type,
        )
        output = get_drop_volume(ds["diameter_bin_center"])
        # Test output type
        assert isinstance(output, xr.DataArray)
        # If everything is 0, we expect an array with 0 as result
        assert np.all(output.to_numpy() == 0), "All values should be zeros for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_drop_volume_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="diameter_bin_center",
            scenario="nans",
            array_type=array_type,
        )
        output = get_drop_volume(ds["diameter_bin_center"])
        # Test output type
        assert isinstance(output, xr.DataArray)
        # If everything is NaN, we expect an array with NaN as result
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_drop_volume_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="diameter_bin_center",
            scenario="realistic",
            array_type=array_type,
        )
        output = get_drop_volume(ds["diameter_bin_center"])
        # Test output type
        assert isinstance(output, xr.DataArray)
        # Test expected values - volume of sphere formula pi/6 * d^3
        expected = np.array([0.00418879, 0.03351032, 0.11309734, 0.26808257])
        np.testing.assert_allclose(output.to_numpy(), expected, rtol=1e-4)


class TestNormalizedInterceptParameter:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_normalized_intercept_parameter_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )
        # Calculate required inputs
        liquid_water_content = get_liquid_water_content(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Calculate moments for mean volume diameter
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

        # 0/0 gives NaN
        mean_volume_diameter = get_mean_volume_drop_diameter(moment_3=moment_3, moment_4=moment_4)
        output = get_normalized_intercept_parameter(
            liquid_water_content=liquid_water_content,
            mean_volume_diameter=mean_volume_diameter,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect an array with NaN as result
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_normalized_intercept_parameter_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="nans",
            array_type=array_type,
        )
        # Calculate required inputs
        liquid_water_content = get_liquid_water_content(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Calculate moments for mean volume diameter
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
        mean_volume_diameter = get_mean_volume_drop_diameter(moment_3=moment_3, moment_4=moment_4)

        output = get_normalized_intercept_parameter(
            liquid_water_content=liquid_water_content,
            mean_volume_diameter=mean_volume_diameter,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, we expect an array with NaN as result
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_normalized_intercept_parameter_realistic_values(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
        )
        # Calculate required inputs
        liquid_water_content = get_liquid_water_content(
            drop_number_concentration=ds["drop_number_concentration"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
        )
        # Calculate moments for mean volume diameter
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
        mean_volume_diameter = get_mean_volume_drop_diameter(moment_3=moment_3, moment_4=moment_4)

        output = get_normalized_intercept_parameter(
            liquid_water_content=liquid_water_content,
            mean_volume_diameter=mean_volume_diameter,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values (these values should be calculated based on your test data)
        expected_values = np.array([170075.02472902, 170075.02472902])
        np.testing.assert_allclose(output.to_numpy(), expected_values, rtol=1e-6)

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_get_normalized_intercept_parameter_from_moments(self, template_dataset, array_type):
        """Test get same values from get_normalized_intercept_parameter_from_moments."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
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
        output = get_normalized_intercept_parameter_from_moments(moment_3=moment_3, moment_4=moment_4)
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values (these values should be calculated based on your test data)
        expected_values = np.array([170075.02472902, 170075.02472902])
        np.testing.assert_allclose(output.to_numpy(), expected_values, rtol=1e-6)


class TestDropAverageVelocity:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_drop_average_velocity_zero_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="zeros",
            array_type=array_type,
        )
        output = get_drop_average_velocity(ds["drop_number"])

        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, we expect array with NaN
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_drop_average_velocity_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="nans",
            array_type=array_type,
        )
        output = get_drop_average_velocity(ds["drop_number"])

        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, expect array with NaN
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_drop_average_velocity_realistic_values(self, template_dataset, array_type):
        """Test correct values for realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="realistic",
            array_type=array_type,
        )
        output = get_drop_average_velocity(ds["drop_number"])

        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        expected = np.array(
            [[0.56666667, 0.56666667, 0.56666667, 0.56666667], [0.56666667, 0.56666667, 0.56666667, 0.56666667]],
        )
        np.testing.assert_allclose(output.to_numpy(), expected)

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_drop_average_velocity_nan_where_no_drops(self, template_dataset, array_type):
        """Test that in diameter bins with no drops, average velocity output is NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="realistic",
            array_type=array_type,
        )
        ds["drop_number"].data[:, :, -2:] = 0
        output = get_drop_average_velocity(ds["drop_number"])

        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        expected = np.array(
            [[0.56666667, 0.56666667, np.nan, np.nan], [0.56666667, 0.56666667, np.nan, np.nan]],
        )
        np.testing.assert_allclose(output.to_numpy(), expected)


class TestDropNumberConcentration:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_drop_number_concentration_zero_values(self, template_dataset, array_type):
        """Test returns 0 when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="zeros",
            array_type=array_type,
        )
        output = get_drop_number_concentration(
            drop_number=ds["drop_number"],
            velocity=ds["fall_velocity"],
            diameter_bin_width=ds["diameter_bin_width"],
            sampling_area=0.005,  # 100 cm2
            sample_interval=60,  # 1 minute
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is 0, expect array with 0
        assert np.all(output.to_numpy() == 0), "All values should be zeros for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_drop_number_concentration_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="nans",
            array_type=array_type,
        )
        output = get_drop_number_concentration(
            drop_number=ds["drop_number"],
            velocity=ds["fall_velocity"],
            diameter_bin_width=ds["diameter_bin_width"],
            sampling_area=0.005,
            sample_interval=60,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # If everything is NaN, expect array with NaN
        assert np.all(np.isnan(output.to_numpy())), "All values should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_drop_number_concentration_with_measured_velocity(self, template_dataset, array_type):
        """Test correct values for realistic case of optical disdrometers."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="realistic",
            array_type=array_type,
        )
        output = get_drop_number_concentration(
            drop_number=ds["drop_number"],
            velocity=ds["velocity_bin_center"],
            diameter_bin_width=ds["diameter_bin_width"],
            sampling_area=0.005,
            sample_interval=60,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        expected = np.array(
            [
                [133.33333333, 133.33333333, 133.33333333, 133.33333333],
                [133.33333333, 133.33333333, 133.33333333, 133.33333333],
            ],
        )
        np.testing.assert_allclose(output.to_numpy(), expected)

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_drop_number_concentration_with_estimated_fall_velocity(self, template_dataset, array_type):
        """Test correct values for realistic case of optical disdrometers."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="realistic",
            array_type=array_type,
        )
        output = get_drop_number_concentration(
            drop_number=ds["drop_number"],
            velocity=ds["fall_velocity"],
            diameter_bin_width=ds["diameter_bin_width"],
            sampling_area=0.005,
            sample_interval=60,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        expected = np.array([[100.0, 50.0, 33.33333333, 25.0], [100.0, 50.0, 33.33333333, 25.0]])
        np.testing.assert_allclose(output.to_numpy(), expected)

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_drop_number_concentration_for_impact_disdrometers(self, template_dataset, array_type):
        """Test correct values for realistic case of impact disdrometers."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="realistic",
            array_type=array_type,
        )
        ds["drop_number"] = ds["drop_number"].sum(dim=VELOCITY_DIMENSION)  # remove velocity dimension
        output = get_drop_number_concentration(
            drop_number=ds["drop_number"],
            velocity=ds["fall_velocity"],
            diameter_bin_width=ds["diameter_bin_width"],
            sampling_area=0.005,
            sample_interval=60,
        )
        # Test output type
        check_datarray_type(output, array_type=array_type)
        # Test expected values
        expected = np.array([[100.0, 50.0, 33.33333333, 25.0], [100.0, 50.0, 33.33333333, 25.0]])
        np.testing.assert_allclose(output.to_numpy(), expected)


class TestBinDimensions:
    def test_get_dimensions_optical_disdrometer(self, template_dataset):
        """Test returns correct dimensions for optical disdrometer data."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="realistic",
            array_type="numpy",
        )
        dims = get_bin_dimensions(ds["drop_number"])

        # Test returns tuple or list
        assert isinstance(dims, list)
        # Test correct dimensions for optical disdrometer
        assert dims == [DIAMETER_DIMENSION, VELOCITY_DIMENSION]

    def test_get_dimensions_impact_disdrometer(self, template_dataset):
        """Test returns correct dimensions for impact disdrometer data."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="realistic",
            array_type="numpy",
        )
        # Remove velocity dimension to simulate impact disdrometer data
        ds["drop_number"] = ds["drop_number"].sum(dim=VELOCITY_DIMENSION)
        dims = get_bin_dimensions(ds["drop_number"])

        # Test returns tuple or list
        assert isinstance(dims, list)
        # Test correct dimensions for impact disdrometer
        assert dims == [DIAMETER_DIMENSION]


class TestKineticEnergyVariables:
    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_zero_output_with_dsd_zero_values(self, template_dataset, array_type):
        """Test returns zeros when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="zeros",
            array_type=array_type,
        )
        ds_output = get_kinetic_energy_variables(
            drop_number_concentration=ds["drop_number_concentration"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            sample_interval=60,
        )
        # Test output type is xr.Dataset
        assert isinstance(ds_output, xr.Dataset)
        for var, value in ds_output.items():
            # Test each variable DataArray is of expected type
            check_datarray_type(value, array_type=array_type)
            # If everything is 0, we expect an array with 0 as result
            assert np.all(value.to_numpy() == 0), f"All values for {var} should be zeros for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_nan_output_with_dsd_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="nans",
            array_type=array_type,
        )
        ds_output = get_kinetic_energy_variables(
            drop_number_concentration=ds["drop_number_concentration"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            sample_interval=60,
        )
        # Test output type is xr.Dataset
        assert isinstance(ds_output, xr.Dataset)
        for var, value in ds_output.items():
            # Test each variable DataArray is of expected type
            check_datarray_type(value, array_type=array_type)
            # If everything is NaN, we expect an array with NaN as result
            assert np.all(np.isnan(value.to_numpy())), f"All values for {var} should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_zero_output_with_drop_number_zero_values(self, template_dataset, array_type):
        """Test returns zeros when all inputs are zero."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="zeros",
            array_type=array_type,
        )

        ds_output = get_kinetic_energy_variables_from_drop_number(
            drop_number=ds["drop_number"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            sampling_area=0.005,
            sample_interval=60,
        )
        # Test output type is xr.Dataset
        assert isinstance(ds_output, xr.Dataset)
        for var, value in ds_output.items():
            # Test each variable DataArray is of expected type
            check_datarray_type(value, array_type=array_type)
            # If everything is 0, we expect an array with 0 as result
            assert np.all(value.to_numpy() == 0), f"All values for {var} should be zeros for zero-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_nan_output_with_drop_number_nan_values(self, template_dataset, array_type):
        """Test returns NaN when all inputs are NaN."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="nans",
            array_type=array_type,
        )
        ds_output = get_kinetic_energy_variables_from_drop_number(
            drop_number=ds["drop_number"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            sampling_area=0.005,
            sample_interval=60,
        )
        # Test output type is xr.Dataset
        assert isinstance(ds_output, xr.Dataset)
        for var, value in ds_output.items():
            # Test each variable DataArray is of expected type
            check_datarray_type(value, array_type=array_type)
            # If everything is NaN, we expect an array with NaN as result
            assert np.all(np.isnan(value.to_numpy())), f"All values for {var} should be NaN for NaN-valued inputs."

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_results_with_estimated_fall_velocity(self, template_dataset, array_type):
        """Test correct values for a realistic case."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type=array_type,
        )
        ds_output = get_kinetic_energy_variables(
            drop_number_concentration=ds["drop_number_concentration"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            sample_interval=60,
        )

        # Test output type is xr.Dataset
        assert isinstance(ds_output, xr.Dataset)
        for value in ds_output.values():
            # Test each variable DataArray is of expected type
            check_datarray_type(value, array_type=array_type)

        # Check dataset returns the expected variables
        expected_variables = [
            "TKE",
            "KEF",
            "KED",
        ]
        assert all(var in ds_output for var in expected_variables)

        # Test each value is DataArray with correct type
        for value in ds_output.values():
            check_datarray_type(value, array_type=array_type)

        # Test expected values (these values should be based on the template dataset)
        expected_values = {
            "TKE": np.array([0.01989571, 0.01989571]),
            "KEF": np.array([1.19374238, 1.19374238]),
            "KED": np.array([1.14231602, 1.14231602]),
        }
        for var, expected in expected_values.items():
            np.testing.assert_allclose(ds_output[var].to_numpy(), expected, rtol=1e-6)

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_same_results_when_same_velocity_used(self, template_dataset, array_type):
        """Test both methods give same results when using same estimated fall velocity."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="realistic",
            array_type=array_type,
        )

        # Define realistic drop_number concentration
        sampling_area = 0.005
        sample_interval = 60
        ds["drop_number_concentration"] = get_drop_number_concentration(
            drop_number=ds["drop_number"],
            velocity=ds["fall_velocity"],
            diameter_bin_width=ds["diameter_bin_width"],
            sampling_area=sampling_area,
            sample_interval=sample_interval,
        )

        # Calculate using drop_number_concentration method
        ds_output1 = get_kinetic_energy_variables(
            drop_number_concentration=ds["drop_number_concentration"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            diameter_bin_width=ds["diameter_bin_width"],
            sample_interval=sample_interval,
        )

        # Calculate using drop_number method
        ds_output2 = get_kinetic_energy_variables_from_drop_number(
            drop_number=ds["drop_number"],
            velocity=ds["fall_velocity"],
            diameter=ds["diameter_bin_center"] / 1000,
            sampling_area=sampling_area,
            sample_interval=sample_interval,
        )
        xr.testing.assert_allclose(ds_output1, ds_output2)

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_results_with_measured_velocity(self, template_dataset, array_type):
        """Test results with measured fall velocity."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="realistic",
            array_type=array_type,
        )

        # Calculate using drop_number
        ds_output = get_kinetic_energy_variables_from_drop_number(
            drop_number=ds["drop_number"],
            velocity=ds["velocity_bin_center"],
            diameter=ds["diameter_bin_center"] / 1000,
            sampling_area=0.005,
            sample_interval=60,
        )
        # Check dataset returns the expected variables
        expected_variables = [
            "TKE",
            "KEF",
            "KED",
        ]
        assert all(var in ds_output for var in expected_variables)

        # Test each value is DataArray with correct type
        for value in ds_output.values():
            check_datarray_type(value, array_type=array_type)

        # Test expected values (these values should be based on the template dataset)
        expected_values = {
            "TKE": np.array([5.40353936e-05, 5.40353936e-05]),
            "KEF": np.array([0.00324212, 0.00324212]),
            "KED": np.array([0.215, 0.215]),
        }
        for var, expected in expected_values.items():
            np.testing.assert_allclose(ds_output[var].to_numpy(), expected, rtol=1e-4)

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    def test_with_velocity_method_dimension(self, template_dataset, array_type):
        """Test both methods give same results when using same estimated fall velocity."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number",
            scenario="realistic",
            array_type=array_type,
        )

        # Prepare velocity DataArray
        velocity = xr.Dataset(
            {
                "measured_velocity": xr.ones_like(ds["drop_number"]) * ds["velocity_bin_center"],
                "fall_velocity": xr.ones_like(ds["drop_number"]) * ds["fall_velocity"],
            },
        ).to_array(dim="velocity_method")

        # Calculate using drop_number
        ds_output = get_kinetic_energy_variables_from_drop_number(
            drop_number=ds["drop_number"],
            velocity=velocity,
            diameter=ds["diameter_bin_center"] / 1000,
            sampling_area=0.005,
            sample_interval=60,
        )
        # Check dataset returns the expected variables
        expected_variables = [
            "TKE",
            "KEF",
            "KED",
        ]
        assert all(var in ds_output for var in expected_variables)

        # Test each value is DataArray with correct type
        for value in ds_output.values():
            check_datarray_type(value, array_type=array_type)

        # Assert velocity method dimension
        assert "velocity_method" in ds_output.dims
        assert ds_output.sizes["velocity_method"] == 2

    def test_presence_velocity_dimension_raise_error(self, template_dataset):
        """Test raises error when velocity DataArray has the velocity dimension."""
        ds = _prepare_test_dataset(
            template_dataset,
            variables="drop_number_concentration",
            scenario="realistic",
            array_type="numpy",
        )

        # Create velocity with velocity dimension
        velocity = xr.ones_like(ds["drop_number"]) * ds["velocity_bin_center"]

        # Test raises error
        with pytest.raises(ValueError):
            get_kinetic_energy_variables(
                drop_number_concentration=ds["drop_number_concentration"],
                velocity=velocity,
                diameter=ds["diameter_bin_center"] / 1000,
                diameter_bin_width=ds["diameter_bin_width"],
                sample_interval=60,
            )
