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
"""Testing PSD models classes."""

import dask.array
import numpy as np
import pytest
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION
from disdrodb.psd.models import (
    BinnedPSD,
    ExponentialPSD,
    GammaPSD,
    LognormalPSD,
    NormalizedGammaPSD,
    available_psd_models,
    check_psd_model,
    create_psd,
    get_psd_model,
    get_psd_model_formula,
    get_required_parameters,
)


class TestPSDUtilityFunctions:
    """Test suite for PSD utility functions."""

    def test_available_psd_models(self):
        """Check the list of available PSD models."""
        models = available_psd_models()
        expected_models = ["LognormalPSD", "ExponentialPSD", "GammaPSD", "NormalizedGammaPSD"]
        assert set(models) == set(expected_models), "Mismatch in available PSD models."

    def test_check_psd_model(self):
        """Check validation of PSD model names."""
        valid_model = "LognormalPSD"
        invalid_model = "InvalidPSD"
        assert check_psd_model(valid_model) == valid_model
        with pytest.raises(ValueError):
            check_psd_model(invalid_model)

    def test_get_psd_model(self):
        """Check retrieval of PSD model classes."""
        psd_class = get_psd_model("LognormalPSD")
        assert psd_class.__name__ == "LognormalPSD"
        with pytest.raises(KeyError):
            get_psd_model("InvalidPSD")

    def test_get_psd_model_formula(self):
        """Check retrieval of PSD model formulas."""
        formula = get_psd_model_formula("LognormalPSD")
        assert callable(formula), "Formula should be callable."
        with pytest.raises(KeyError):
            get_psd_model_formula("InvalidPSD")

    def test_create_psd(self):
        """Check creation of PSD instances."""
        parameters = {"Nt": 1.0, "mu": 0.0, "sigma": 1.0}
        psd = create_psd("LognormalPSD", parameters)
        assert isinstance(psd, LognormalPSD)
        with pytest.raises(KeyError):
            create_psd("InvalidPSD", parameters)

    def test_get_required_parameters(self):
        """Check retrieval of required parameters for PSD models."""
        required_params = get_required_parameters("LognormalPSD")
        assert required_params == ["Nt", "mu", "sigma"]
        with pytest.raises(KeyError):
            get_required_parameters("InvalidPSD")


class TestXarrayPSD:
    """Test PSD functionalities with xarray parameters."""

    def test_isel_psd_subsetting(self):
        """Test isel method."""
        Nt = xr.DataArray([1.0, 2.0], dims="time")
        mu = xr.DataArray([3.0, 4.0], dims="time")
        psd = LognormalPSD(Nt=Nt, mu=mu, sigma=6.0)
        psd_subset = psd.isel(time=0)
        assert psd_subset.parameters["Nt"].size == 1
        assert psd_subset.parameters["Nt"].item() == 1
        assert psd_subset.parameters["mu"].size == 1
        assert psd_subset.parameters["mu"].item() == 3
        assert isinstance(psd_subset.parameters["sigma"], float)
        assert psd_subset.parameters["sigma"] == 6

        with pytest.raises(ValueError):
            LognormalPSD(Nt=1, mu=1, sigma=1).isel(time=0)

    def test_sel_psd_subsetting(self):
        """Test sel method."""
        Nt = xr.DataArray([1.0, 2.0], coords={"time": [1, 2]}, dims="time")
        mu = xr.DataArray([3.0, 4.0], coords={"time": [1, 2]}, dims="time")
        psd = LognormalPSD(Nt=Nt, mu=mu, sigma=6.0)
        psd_subset = psd.sel(time=1)
        assert psd_subset.parameters["Nt"].size == 1
        assert psd_subset.parameters["Nt"].item() == 1
        assert psd_subset.parameters["mu"].size == 1
        assert psd_subset.parameters["mu"].item() == 3
        assert isinstance(psd_subset.parameters["sigma"], float)
        assert psd_subset.parameters["sigma"] == 6

        with pytest.raises(ValueError):
            LognormalPSD(Nt=1, mu=1, sigma=1).sel(time=1)


class TestLognormalPSD:
    """Test suite for the LognormalPSD class."""

    def test_lognormal_scalar_parameters(self):
        """Check PSD with scalar parameters inputs produce valid PSD values."""
        psd = LognormalPSD(Nt=1.0, mu=0.0, sigma=1.0)

        # Check input scalar
        np.testing.assert_allclose(psd(1.0), 0.398942, atol=1e-5)

        # Check input numpy array
        np.testing.assert_allclose(psd(np.array([1.0, 2.0])), [0.39894228, 0.15687402], atol=1e-5)

        # Check input dask array
        np.testing.assert_allclose(psd(dask.array.from_array([1.0, 2.0])), [0.39894228, 0.15687402], atol=1e-5)

        # Check input xarray
        D = xr.DataArray([1.0, 2.0], dims=[DIAMETER_DIMENSION])
        output = psd(D)
        assert isinstance(output, xr.DataArray)
        np.testing.assert_allclose(output.to_numpy(), [0.39894228, 0.15687402], atol=1e-5)

    def test_lognormal_xarray_parameters(self):
        """Check PSD with xarray parameters inputs produce valid PSD values."""
        Nt = xr.DataArray([1.0, 2.0], dims="time")
        mu = xr.DataArray([1.0, 2.0], dims="time")
        sigma = xr.DataArray([1.0, 1.0], dims="time")
        psd = LognormalPSD(Nt=Nt, mu=mu, sigma=sigma)

        # Check input scalar
        output = psd(1.0)
        assert isinstance(output, xr.DataArray)
        assert output.sizes == {"time": 2}
        np.testing.assert_allclose(output.to_numpy(), np.array([0.24197072, 0.10798193]), atol=1e-5)

        # Check input numpy array diameters
        D = [1.0, 2.0, 3.0]
        output = psd(D)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        expected_array = np.array([[0.24197072, 0.1902978, 0.13233575], [0.10798193, 0.16984472, 0.17716858]])
        np.testing.assert_allclose(output.to_numpy(), expected_array, atol=1e-5)

        # Check input dask array diameters
        D = dask.array.from_array([1.0, 2.0, 3.0])
        output = psd(D)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        expected_array = np.array([[0.24197072, 0.1902978, 0.13233575], [0.10798193, 0.16984472, 0.17716858]])
        np.testing.assert_allclose(output.to_numpy(), expected_array, atol=1e-5)

        # Check input xarray
        D = xr.DataArray([1.0, 2.0, 3.0], dims=[DIAMETER_DIMENSION])
        output = psd(D)
        assert isinstance(output, xr.DataArray)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        expected_array = np.array([[0.24197072, 0.1902978, 0.13233575], [0.10798193, 0.16984472, 0.17716858]])
        np.testing.assert_allclose(output.to_numpy(), expected_array, atol=1e-5)

    def test_lognormal_model_accept_only_scalar_or_xarray_parameters(self):
        """Check numpy array inputs parameters are not allowed."""
        with pytest.raises(TypeError):
            LognormalPSD(Nt=[1.0, 2.0], mu=1, sigma=1)

        with pytest.raises(TypeError):
            LognormalPSD(Nt=np.array([1.0, 2.0]), mu=1, sigma=1)

        # Check mixture of scalar and DataArray is allowed
        psd = LognormalPSD(Nt=xr.DataArray([1.0, 2.0], dims="time"), mu=1, sigma=xr.DataArray([1.0, 2.0], dims="time"))
        assert isinstance(psd, LognormalPSD)

    def test_lognormal_raise_error_invalid_diameter_array(self):
        """Check invalid input diameter arrays."""
        psd = LognormalPSD(Nt=1.0, mu=0.0, sigma=1.0)

        # String input is not accepted
        with pytest.raises(TypeError):
            psd("string")
        with pytest.raises(TypeError):
            psd("1.0")

        # None input is not accepted
        with pytest.raises(TypeError):
            psd(None)

        # Empty is not accepted
        with pytest.raises(ValueError):
            psd([])
        with pytest.raises(ValueError):
            psd(())
        with pytest.raises(ValueError):
            psd(np.array([]))

        # 2-dimensional diameter array are not accepted
        with pytest.raises(ValueError):
            psd(np.ones((2, 2)))

    def test_name(self):
        """Check the 'name' property."""
        psd = LognormalPSD(Nt=1.0, mu=0.0, sigma=1.0)
        assert psd.name == "LognormalPSD"

    def test_formula_method(self):
        """Check the formula directly."""
        value = LognormalPSD.formula(D=1.0, Nt=1.0, mu=0.0, sigma=1.0)
        assert isinstance(value, float)

    def test_from_parameters(self):
        """Check PSD creation from parameters."""
        params = {"Nt": 1.0, "mu": 0.0, "sigma": 1.0}
        psd = LognormalPSD.from_parameters(params)
        assert psd.Nt == 1.0
        assert psd.mu == 0.0
        assert psd.sigma == 1.0
        assert psd.parameters == params

    def test_required_parameters(self):
        """Check the required parameters list."""
        req = LognormalPSD.required_parameters()
        assert req == ["Nt", "mu", "sigma"], "Incorrect required parameters."

    def test_parameters_summary(self):
        """Check parameters summary handling."""
        # Scalar case
        psd = LognormalPSD(Nt=2.0, mu=0.5, sigma=0.5)
        summary = psd.parameters_summary()
        assert isinstance(summary, str)
        assert "LognormalPSD" in summary, "Name missing in summary."

        # Xarray case
        psd = LognormalPSD(Nt=xr.DataArray([1.0, 1.0], dims="time"), mu=0.0, sigma=1.0)
        summary = psd.parameters_summary()
        assert isinstance(summary, str)
        assert summary == "LognormalPSD with N-d parameters \n"

    def test_eq(self):
        """Check equality of two PSD objects."""
        # Numpy
        psd1 = LognormalPSD(Nt=1.0, mu=0.0, sigma=1.0)
        psd2 = LognormalPSD(Nt=1.0, mu=0.0, sigma=1.0)
        psd3 = LognormalPSD(Nt=2.0, mu=0.0, sigma=1.0)
        assert psd1 == psd2, "Identical PSDs should be equal."
        assert psd1 != psd3, "Different PSDs should not be equal."
        # Xarray
        psd1 = LognormalPSD(Nt=xr.DataArray([1.0, 1.0], dims="time"), mu=0.0, sigma=1.0)
        psd2 = LognormalPSD(Nt=xr.DataArray([1.0, 1.0], dims="time"), mu=0.0, sigma=1.0)
        psd3 = LognormalPSD(Nt=xr.DataArray([1.0, 2.0], dims="time"), mu=0.0, sigma=1.0)
        psd4 = LognormalPSD(
            Nt=xr.DataArray([1.0, 1.0], dims="time"),
            mu=xr.DataArray([0.0, 0.0], dims="time"),
            sigma=1.0,
        )
        assert psd1 == psd2, "Identical PSDs should be equal."
        assert psd1 == psd4, "Identical PSDs should be equal."  # mu scalar vs mu xarray with equal values
        assert psd1 != psd3, "Different PSDs should not be equal."
        assert psd1 != None, "Comparison against None should return False."  # noqa: E711

        # Check with different PSD
        assert GammaPSD(N0=1.0, mu=2.0, Lambda=3.0) != LognormalPSD(Nt=1, mu=1, sigma=1)
        # Check non-sense comparison
        assert LognormalPSD(Nt=1, mu=1, sigma=1) != "dummy_test"


class TestGammaPSD:
    """Test suite for the GammaPSD class."""

    def test_gamma_scalar_parameters(self):
        """Check PSD with scalar parameters inputs produce valid PSD values."""
        psd = GammaPSD(N0=1.0, mu=2.0, Lambda=3.0)
        # Check input scalar
        np.testing.assert_allclose(psd(1.0), 0.049787, atol=1e-5)
        # Check input numpy array
        np.testing.assert_allclose(psd(np.array([1.0, 2.0])), [0.04978707, 0.00991615], atol=1e-5)
        # Check input dask array
        np.testing.assert_allclose(psd(dask.array.from_array([1.0, 2.0])), [0.04978707, 0.00991615], atol=1e-5)
        # Check input xarray
        D = xr.DataArray([1.0, 2.0], dims=[DIAMETER_DIMENSION])
        output = psd(D)
        assert isinstance(output, xr.DataArray)
        np.testing.assert_allclose(output.to_numpy(), [0.04978707, 0.00991615], atol=1e-5)

    def test_gamma_xarray_parameters(self):
        """Check PSD with xarray parameters inputs produce valid PSD values."""
        N0 = xr.DataArray([1.0, 2.0], dims="time")
        mu = xr.DataArray([1.0, 2.0], dims="time")
        Lambda = xr.DataArray([2.0, 3.0], dims="time")
        psd = GammaPSD(N0=N0, mu=mu, Lambda=Lambda)
        # Check input scalar
        output = psd(1.0)
        assert isinstance(output, xr.DataArray)
        assert output.sizes == {"time": 2}
        np.testing.assert_allclose(output.to_numpy(), np.array([0.13533528, 0.09957414]), atol=1e-5)
        # Check input numpy array diameters
        D = [1.0, 2.0, 3.0]
        output = psd(D)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        expected_array = np.array([[0.13533528, 0.03663128, 0.00743626], [0.09957414, 0.01983002, 0.00222138]])
        np.testing.assert_allclose(output.to_numpy(), expected_array, atol=1e-5)
        # Check input dask array diameters
        D = dask.array.from_array([1.0, 2.0, 3.0])
        output = psd(D)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        np.testing.assert_allclose(output.to_numpy(), expected_array, atol=1e-5)
        # Check input xarray
        D = xr.DataArray([1.0, 2.0, 3.0], dims=[DIAMETER_DIMENSION])
        output = psd(D)
        assert isinstance(output, xr.DataArray)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        np.testing.assert_allclose(output.to_numpy(), expected_array, atol=1e-5)

    def test_gamma_model_accept_only_scalar_or_xarray_parameters(self):
        """Check numpy array inputs parameters are not allowed."""
        with pytest.raises(TypeError):
            GammaPSD(N0=[1.0, 2.0], mu=1, Lambda=1)
        with pytest.raises(TypeError):
            GammaPSD(N0=np.array([1.0, 2.0]), mu=1, Lambda=1)
        # Check mixture of scalar and DataArray is allowed
        psd = GammaPSD(N0=xr.DataArray([1.0, 2.0], dims="time"), mu=1, Lambda=xr.DataArray([1.0, 2.0], dims="time"))
        assert isinstance(psd, GammaPSD)

    def test_gamma_raise_error_invalid_diameter_array(self):
        """Check invalid input diameter arrays."""
        psd = GammaPSD(N0=1.0, mu=0.0, Lambda=1.0)
        # String input is not accepted
        with pytest.raises(TypeError):
            psd("string")
        # None input is not accepted
        with pytest.raises(TypeError):
            psd(None)
        # Empty is not accepted
        with pytest.raises(ValueError):
            psd([])
        # 2-dimensional diameter array are not accepted
        with pytest.raises(ValueError):
            psd(np.ones((2, 2)))

    def test_name(self):
        """Check the 'name' property."""
        psd = GammaPSD(N0=1.0, mu=0.0, Lambda=1.0)
        assert psd.name == "GammaPSD"

    def test_formula_method(self):
        """Check the formula directly."""
        value = GammaPSD.formula(D=1.0, N0=1.0, mu=0.0, Lambda=1.0)
        assert isinstance(value, float)

    def test_from_parameters(self):
        """Check PSD creation from parameters."""
        params = {"N0": 1.0, "mu": 0.0, "Lambda": 1.0}
        psd = GammaPSD.from_parameters(params)
        assert psd.N0 == 1.0
        assert psd.mu == 0.0
        assert psd.Lambda == 1.0
        assert psd.parameters == params

    def test_required_parameters(self):
        """Check the required parameters list."""
        req = GammaPSD.required_parameters()
        assert req == ["N0", "mu", "Lambda"], "Incorrect required parameters."

    def test_parameters_summary(self):
        """Check parameters summary handling."""
        psd = GammaPSD(N0=2.0, mu=0.5, Lambda=0.5)
        summary = psd.parameters_summary()
        assert isinstance(summary, str)
        assert "GammaPSD" in summary, "Name missing in summary."

        # Xarray case
        psd = GammaPSD(N0=xr.DataArray([1.0, 1.0], dims="time"), mu=0.0, Lambda=1.0)
        summary = psd.parameters_summary()
        assert isinstance(summary, str)
        assert summary == "GammaPSD with N-d parameters \n"

    def test_eq(self):
        """Check equality of two PSD objects."""
        # Scalar
        psd1 = GammaPSD(N0=1.0, mu=0.0, Lambda=1.0)
        psd2 = GammaPSD(N0=1.0, mu=0.0, Lambda=1.0)
        psd3 = GammaPSD(N0=2.0, mu=0.0, Lambda=1.0)
        assert psd1 == psd2, "Identical PSDs should be equal."
        assert psd1 != psd3, "Different PSDs should not be equal."
        # Xarray
        psd1 = GammaPSD(N0=xr.DataArray([1.0, 1.0], dims="time"), mu=0.0, Lambda=1.0)
        psd2 = GammaPSD(N0=xr.DataArray([1.0, 1.0], dims="time"), mu=0.0, Lambda=1.0)
        psd3 = GammaPSD(N0=xr.DataArray([1.0, 2.0], dims="time"), mu=0.0, Lambda=1.0)
        psd4 = GammaPSD(N0=xr.DataArray([1.0, 1.0], dims="time"), mu=xr.DataArray([0.0, 0.0], dims="time"), Lambda=1.0)
        assert psd1 == psd2, "Identical PSDs should be equal."
        assert psd1 == psd4, "Identical PSDs should be equal."  # mu scalar vs mu xarray with equal values
        assert psd1 != psd3, "Different PSDs should not be equal."
        assert psd1 != None, "Comparison against None should return False."  # noqa: E711


class TestExponentialPSD:
    """Test suite for the ExponentialPSD class."""

    def test_exponential_scalar_parameters(self):
        """Check PSD with scalar parameters inputs produce valid PSD values."""
        psd = ExponentialPSD(N0=1.0, Lambda=2.0)
        # Check input scalar
        np.testing.assert_allclose(psd(1.0), 0.135335, atol=1e-5)
        # Check input numpy array
        np.testing.assert_allclose(psd(np.array([1.0, 2.0])), [0.13533528, 0.01831564], atol=1e-5)
        # Check input dask array
        np.testing.assert_allclose(psd(dask.array.from_array([1.0, 2.0])), [0.13533528, 0.01831564], atol=1e-5)
        # Check input xarray
        D = xr.DataArray([1.0, 2.0], dims=[DIAMETER_DIMENSION])
        output = psd(D)
        assert isinstance(output, xr.DataArray)
        np.testing.assert_allclose(output.to_numpy(), [0.13533528, 0.01831564], atol=1e-5)

    def test_exponential_xarray_parameters(self):
        """Check PSD with xarray parameters inputs produce valid PSD values."""
        N0 = xr.DataArray([1.0, 2.0], dims="time")
        Lambda = xr.DataArray([2.0, 3.0], dims="time")
        psd = ExponentialPSD(N0=N0, Lambda=Lambda)
        # Check input scalar
        output = psd(1.0)
        assert isinstance(output, xr.DataArray)
        assert output.sizes == {"time": 2}
        np.testing.assert_allclose(output.to_numpy(), np.array([0.13533528, 0.09957414]), atol=1e-5)
        # Check input numpy array diameters
        D = [1.0, 2.0, 3.0]
        output = psd(D)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        expected_array = np.array([[0.13533528, 0.01831564, 0.00247875], [0.09957414, 0.0049575, 0.00024682]])
        np.testing.assert_allclose(output.to_numpy(), expected_array, atol=1e-5)
        # Check input dask array diameters
        D = dask.array.from_array([1.0, 2.0, 3.0])
        output = psd(D)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        np.testing.assert_allclose(output.to_numpy(), expected_array, atol=1e-5)
        # Check input xarray
        D = xr.DataArray([1.0, 2.0, 3.0], dims=[DIAMETER_DIMENSION])
        output = psd(D)
        assert isinstance(output, xr.DataArray)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        np.testing.assert_allclose(output.to_numpy(), expected_array, atol=1e-5)

    def test_exponential_model_accept_only_scalar_or_xarray_parameters(self):
        """Check numpy array inputs parameters are not allowed."""
        with pytest.raises(TypeError):
            ExponentialPSD(N0=[1.0, 2.0], Lambda=1)
        with pytest.raises(TypeError):
            ExponentialPSD(N0=np.array([1.0, 2.0]), Lambda=1)
        # Check mixture of scalar and DataArray is allowed
        psd = ExponentialPSD(N0=xr.DataArray([1.0, 2.0], dims="time"), Lambda=1)
        assert isinstance(psd, ExponentialPSD)

    def test_exponential_raise_error_invalid_diameter_array(self):
        """Check invalid input diameter arrays."""
        psd = ExponentialPSD(N0=1.0, Lambda=1.0)
        # String input is not accepted
        with pytest.raises(TypeError):
            psd("string")
        # None input is not accepted
        with pytest.raises(TypeError):
            psd(None)
        # Empty is not accepted
        with pytest.raises(ValueError):
            psd([])
        # 2-dimensional diameter array are not accepted
        with pytest.raises(ValueError):
            psd(np.ones((2, 2)))

    def test_name(self):
        """Check the 'name' property."""
        psd = ExponentialPSD(N0=1.0, Lambda=1.0)
        assert psd.name == "ExponentialPSD"

    def test_formula_method(self):
        """Check the formula directly."""
        value = ExponentialPSD.formula(D=1.0, N0=1.0, Lambda=1.0)
        assert isinstance(value, float)

    def test_from_parameters(self):
        """Check PSD creation from parameters."""
        params = {"N0": 1.0, "Lambda": 1.0}
        psd = ExponentialPSD.from_parameters(params)
        assert psd.N0 == 1.0
        assert psd.Lambda == 1.0
        assert psd.parameters == params

    def test_required_parameters(self):
        """Check the required parameters list."""
        req = ExponentialPSD.required_parameters()
        assert req == ["N0", "Lambda"], "Incorrect required parameters."

    def test_parameters_summary(self):
        """Check parameters summary handling."""
        psd = ExponentialPSD(N0=2.0, Lambda=0.5)
        summary = psd.parameters_summary()
        assert isinstance(summary, str)
        assert "ExponentialPSD" in summary, "Name missing in summary."

        # Xarray case
        psd = ExponentialPSD(N0=xr.DataArray([1.0, 1.0], dims="time"), Lambda=1.0)
        summary = psd.parameters_summary()
        assert isinstance(summary, str)
        assert summary == "ExponentialPSD with N-d parameters \n"

    def test_eq(self):
        """Check equality of two PSD objects."""
        # Scalar
        psd1 = ExponentialPSD(N0=1.0, Lambda=1.0)
        psd2 = ExponentialPSD(N0=1.0, Lambda=1.0)
        psd3 = ExponentialPSD(N0=2.0, Lambda=1.0)
        assert psd1 == psd2, "Identical PSDs should be equal."
        assert psd1 != psd3, "Different PSDs should not be equal."
        # Xarray
        psd1 = ExponentialPSD(N0=xr.DataArray([1.0, 1.0], dims="time"), Lambda=1.0)
        psd2 = ExponentialPSD(N0=xr.DataArray([1.0, 1.0], dims="time"), Lambda=1.0)
        psd3 = ExponentialPSD(N0=xr.DataArray([1.0, 2.0], dims="time"), Lambda=1.0)
        psd4 = ExponentialPSD(N0=xr.DataArray([1.0, 1.0], dims="time"), Lambda=xr.DataArray([1.0, 1.0], dims="time"))
        assert psd1 == psd2, "Identical PSDs should be equal."
        assert psd1 == psd4, "Identical PSDs should be equal."  # Lambda scalar vs Lambda xarray with equal values
        assert psd1 != psd3, "Different PSDs should not be equal."
        assert psd1 != None, "Comparison against None should return False."  # noqa: E711
        assert psd1 != GammaPSD(), "Different PSDs class should not be equal."


class TestNormalizedGammaPSD:
    """Test suite for the NormalizedGammaPSD class."""

    def test_normalizedgamma_scalar_parameters(self):
        """Check PSD with scalar parameters inputs produce valid PSD values."""
        psd = NormalizedGammaPSD(Nw=1.0, D50=2.0, mu=1.0)
        # Check input scalar
        np.testing.assert_allclose(psd(1.0), 0.14816736025935, atol=1e-5)
        # Check input numpy array
        np.testing.assert_allclose(psd(np.array([1.0, 2.0])), [0.14816736, 0.02868831], atol=1e-5)
        # Check input dask array
        np.testing.assert_allclose(psd(dask.array.from_array([1.0, 2.0])), [0.14816736, 0.02868831], atol=1e-5)
        # Check input xarray
        D = xr.DataArray([1.0, 2.0], dims=[DIAMETER_DIMENSION])
        output = psd(D)
        assert isinstance(output, xr.DataArray)
        np.testing.assert_allclose(output.to_numpy(), [0.14816736, 0.02868831], atol=1e-5)

    def test_normalizedgamma_xarray_parameters(self):
        """Check PSD with xarray parameters inputs produce valid PSD values."""
        Nw = xr.DataArray([1.0, 2.0], dims="time")
        D50 = xr.DataArray([1.0, 2.0], dims="time")
        mu = xr.DataArray([1.0, 2.0], dims="time")
        psd = NormalizedGammaPSD(Nw=Nw, D50=D50, mu=mu)
        # Check input scalar
        output = psd(1.0)
        assert isinstance(output, xr.DataArray)
        assert output.sizes == {"time": 2}
        np.testing.assert_allclose(output.to_numpy(), np.array([0.02868831, 0.26887427]), atol=1e-5)
        # Check input numpy array diameters
        D = [1.0, 2.0, 3.0]
        output = psd(D)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        expected_array = np.array(
            [[2.86883073e-02, 5.37749096e-04, 7.55989420e-06], [2.68874273e-01, 6.31516037e-02, 8.34338042e-03]],
        )
        np.testing.assert_allclose(output.to_numpy(), expected_array, atol=1e-5)
        # Check input dask array diameters
        D = dask.array.from_array([1.0, 2.0, 3.0])
        output = psd(D)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        np.testing.assert_allclose(output.to_numpy(), expected_array, atol=1e-5)
        # Check input xarray
        D = xr.DataArray([1.0, 2.0, 3.0], dims=[DIAMETER_DIMENSION])
        output = psd(D)
        assert isinstance(output, xr.DataArray)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        np.testing.assert_allclose(output.to_numpy(), expected_array, atol=1e-5)

    def test_normalizedgamma_model_accept_only_scalar_or_xarray_parameters(self):
        """Check numpy array inputs parameters are not allowed."""
        with pytest.raises(TypeError):
            NormalizedGammaPSD(Nw=[1.0, 2.0], D50=1, mu=1)
        with pytest.raises(TypeError):
            NormalizedGammaPSD(Nw=np.array([1.0, 2.0]), D50=1, mu=1)
        # Check mixture of scalar and DataArray is allowed
        psd = NormalizedGammaPSD(
            Nw=xr.DataArray([1.0, 2.0], dims="time"),
            D50=1,
            mu=xr.DataArray([1.0, 2.0], dims="time"),
        )
        assert isinstance(psd, NormalizedGammaPSD)

    def test_normalizedgamma_raise_error_invalid_diameter_array(self):
        """Check invalid input diameter arrays."""
        psd = NormalizedGammaPSD(Nw=1.0, D50=1.0, mu=1.0)
        # String input is not accepted
        with pytest.raises(TypeError):
            psd("string")
        # None input is not accepted
        with pytest.raises(TypeError):
            psd(None)
        # Empty is not accepted
        with pytest.raises(ValueError):
            psd([])
        # 2-dimensional diameter array are not accepted
        with pytest.raises(ValueError):
            psd(np.ones((2, 2)))

    def test_name(self):
        """Check the 'name' property."""
        psd = NormalizedGammaPSD(Nw=1.0, D50=1.0, mu=1.0)
        assert psd.name == "NormalizedGammaPSD"

    def test_formula_method(self):
        """Check the formula directly."""
        value = NormalizedGammaPSD.formula(D=1.0, Nw=1.0, D50=1.0, mu=1.0)
        assert isinstance(value, float)

    def test_from_parameters(self):
        """Check PSD creation from parameters."""
        params = {"Nw": 1.0, "D50": 1.0, "mu": 1.0}
        psd = NormalizedGammaPSD.from_parameters(params)
        assert psd.Nw == 1.0
        assert psd.D50 == 1.0
        assert psd.mu == 1.0
        assert psd.parameters == params

    def test_required_parameters(self):
        """Check the required parameters list."""
        req = NormalizedGammaPSD.required_parameters()
        assert req == ["Nw", "D50", "mu"], "Incorrect required parameters."

    def test_parameters_summary(self):
        """Check parameters summary handling."""
        psd = NormalizedGammaPSD(Nw=2.0, D50=0.5, mu=0.5)
        summary = psd.parameters_summary()
        assert isinstance(summary, str)
        assert "NormalizedGammaPSD" in summary, "Name missing in summary."

        # Xarray case
        psd = NormalizedGammaPSD(Nw=xr.DataArray([1.0, 1.0], dims="time"), D50=1.0, mu=1.0)
        summary = psd.parameters_summary()
        assert isinstance(summary, str)
        assert summary == "NormalizedGammaPSD with N-d parameters \n"

    def test_eq(self):
        """Check equality of two PSD objects."""
        # Scalar
        psd1 = NormalizedGammaPSD(Nw=1.0, D50=1.0, mu=1.0)
        psd2 = NormalizedGammaPSD(Nw=1.0, D50=1.0, mu=1.0)
        psd3 = NormalizedGammaPSD(Nw=2.0, D50=1.0, mu=1.0)
        assert psd1 == psd2, "Identical PSDs should be equal."
        assert psd1 != psd3, "Different PSDs should not be equal."
        # Xarray
        psd1 = NormalizedGammaPSD(Nw=xr.DataArray([1.0, 1.0], dims="time"), D50=1.0, mu=1.0)
        psd2 = NormalizedGammaPSD(Nw=xr.DataArray([1.0, 1.0], dims="time"), D50=1.0, mu=1.0)
        psd3 = NormalizedGammaPSD(Nw=xr.DataArray([1.0, 2.0], dims="time"), D50=1.0, mu=1.0)
        psd4 = NormalizedGammaPSD(
            Nw=xr.DataArray([1.0, 1.0], dims="time"),
            D50=xr.DataArray([1.0, 1.0], dims="time"),
            mu=1.0,
        )
        assert psd1 == psd2, "Identical PSDs should be equal."
        assert psd1 == psd4, "Identical PSDs should be equal."  # D50 scalar vs D50 xarray with equal values
        assert psd1 != psd3, "Different PSDs should not be equal."
        assert psd1 != None, "Comparison against None should return False."  # noqa: E711


class TestBinnedPSD:
    """Test suite for the BinnedPSD class."""

    def test_binnedpsd_with_scalar_input(self):
        """Check PSD with scalar diameter inputs."""
        bin_edges = [0.0, 1.0, 2.0, 3.0]
        bin_psd = [10.0, 20.0, 30.0]
        psd = BinnedPSD(bin_edges, bin_psd)
        assert psd(0.5) == 10.0, "Scalar input within first bin failed."
        assert psd(1.5) == 20.0, "Scalar input within second bin failed."
        assert psd(2.5) == 30.0, "Scalar input within third bin failed."
        assert psd(3.5) == 0.0, "Scalar input outside bins should return 0."

        # Test behaviour at bin edges
        assert psd(0.0) == 0.0
        assert psd(3.0) == 30.0  # inclusion of Dmax

        # Test behaviour in-between edges
        assert psd(1.0) == 10  # take left bin (because step_left default interp method)
        assert psd(2.0) == 20  # take left bin (because step_left default interp method)

    def test_binnedpsd_with_array_input(self):
        """Check PSD with numpy array diameter inputs."""
        bin_edges = [0.0, 1.0, 2.0, 3.0]
        bin_psd = [10.0, 20.0, 30.0]
        psd = BinnedPSD(bin_edges, bin_psd)
        D = np.array([0.5, 1.0, 1.5, 2.5, 3.5])
        expected = np.array([10.0, 10.0, 20.0, 30.0, 0.0])
        np.testing.assert_allclose(psd(D), expected, atol=1e-5)

    def test_binnedpsd_invalid_inputs(self):
        """Check invalid inputs for BinnedPSD."""
        # Invalid bin edges and bin_psd lengths
        with pytest.raises(ValueError):
            BinnedPSD([0.0, 1.0, 2.0], [10.0, 20.0, 30.0])  # Mismatch in lengths

        # Invalid diameter input types
        psd = BinnedPSD([0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        with pytest.raises(TypeError):
            psd("invalid")  # String input
        with pytest.raises(TypeError):
            psd(None)  # None input

    def test_binnedpsd_equality(self):
        """Check equality of two BinnedPSD objects."""
        psd1 = BinnedPSD([0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        psd2 = BinnedPSD([0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        psd3 = BinnedPSD([0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 40.0])
        assert psd1 == psd2, "Identical BinnedPSD objects should be equal."
        assert psd1 != psd3, "Different BinnedPSD objects should not be equal."
