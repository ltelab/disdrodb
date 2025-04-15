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
"""Testing PSD models classes."""

import pytest
import dask.array
import numpy as np
import xarray as xr
from disdrodb.psd.models import LognormalPSD
from disdrodb import DIAMETER_DIMENSION

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
        np.testing.assert_allclose(output.to_numpy(), np.array([0.24197072, 0.10798193]),  atol=1e-5)        
        # Check input numpy array diameters 
        D = [1.0, 2.0, 3.0]
        output = psd(D)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        expected_array = np.array([[0.24197072, 0.1902978 , 0.13233575],
                                   [0.10798193, 0.16984472, 0.17716858]])
        np.testing.assert_allclose(output.to_numpy(), expected_array,  atol=1e-5)
        # Check input dask array diameters 
        D = dask.array.from_array([1.0, 2.0, 3.0])
        output = psd(D)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        expected_array = np.array([[0.24197072, 0.1902978 , 0.13233575],
                                   [0.10798193, 0.16984472, 0.17716858]])
        np.testing.assert_allclose(output.to_numpy(), expected_array,  atol=1e-5)
       # Check input xarray
        D = xr.DataArray([1.0, 2.0, 3.0], dims=[DIAMETER_DIMENSION])
        output = psd(D)
        assert isinstance(output, xr.DataArray)
        assert output.sizes == {"time": 2, DIAMETER_DIMENSION: 3}
        expected_array = np.array([[0.24197072, 0.1902978 , 0.13233575],
                                   [0.10798193, 0.16984472, 0.17716858]])
        np.testing.assert_allclose(output.to_numpy(), expected_array,  atol=1e-5)
        
    def test_lognormal_model_accept_only_scalar_or_xarray_parameters(self):
        """Check numpy array inputs parameters are not allowed."""
        with pytest.raises(TypeError): 
            LognormalPSD(Nt=[1.0, 2.0], mu=1, sigma=1)
            
        with pytest.raises(TypeError): 
            LognormalPSD(Nt=np.array([1.0, 2.0]), mu=1, sigma=1)
          
        # Check mixture of scalar and DataArray is allowed 
        psd =  LognormalPSD(Nt=xr.DataArray([1.0, 2.0], dims="time"),
                            mu=1, 
                            sigma=xr.DataArray([1.0, 2.0], dims="time"))
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
        with pytest.raises(ValueError): 
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
            psd(np.ones((2,2)))   
        
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
        assert psd.mu == 1.0
        assert psd.parameters == params

    def test_required_parameters(self):
        """Check the required parameters list."""
        req = LognormalPSD.required_parameters()
        assert req == ["Nt", "mu", "sigma"], "Incorrect required parameters."

    def test_parameters_summary(self):
        """Check parameters summary handling."""
        psd = LognormalPSD(Nt=2.0, mu=0.5, sigma=0.5)
        summary = psd.parameters_summary()
        assert isinstance(summary, str)
        assert "LognormalPSD" in summary, "Name missing in summary."

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
        psd4 = LognormalPSD(Nt=xr.DataArray([1.0, 1.0], dims="time"), mu=xr.DataArray([0.0, 0.0], dims="time"), sigma=1.0)
        assert psd1 == psd2, "Identical PSDs should be equal."
        assert psd1 == psd4, "Identical PSDs should be equal." # mu scalar vs mu xarray with equal values 
        assert psd1 != psd3, "Different PSDs should not be equal."
        