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
"""Testing PSD fitting."""

import numpy as np
import pytest
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION
from disdrodb.fall_velocity import get_rain_fall_velocity_from_ds
from disdrodb.psd.fitting import (
    _compute_target_variable_error,
    available_mom_methods,
    available_optimization,
    check_likelihood,
    check_mom_methods,
    check_optimization,
    check_optimizer,
    check_probability_method,
    check_target,
    check_transformation,
    check_truncated_likelihood,
    compute_gof_stats,
    estimate_model_parameters,
    get_exponential_parameters_Zhang2008,
)
from disdrodb.psd.models import available_psd_models, create_psd
from disdrodb.tests.fake_datasets import create_template_l2e_dataset
from disdrodb.utils.manipulations import get_diameter_coords_dict_from_bin_edges

# Define PSD test cases (psd_model, parameters)
CASES = [
    ("ExponentialPSD", {"N0": 500, "Lambda": 1}, ["GS", "ML", "MOM"]),
    ("GammaPSD", {"N0": 500, "mu": 1.0, "Lambda": 1}, ["GS", "ML", "MOM"]),
    ("NormalizedGammaPSD", {"Nw": 10_000, "D50": 1, "mu": 2}, ["GS"]),  # ML not implemented
    ("LognormalPSD", {"Nt": 800, "mu": 0, "sigma": 1}, ["GS", "ML", "MOM"]),
]


DEFAULT_ML_KWARGS = {
    "probability_method": "cdf",
    "likelihood": "multinomial",
    "truncated_likelihood": True,
    "optimizer": "Nelder-Mead",
}

# Variations: modify one at a time
ML_KWARGS_ALTERNATIVES = [
    {"probability_method": "pdf"},
    {"likelihood": "poisson"},
    {"truncated_likelihood": False},
    {"optimizer": "L-BFGS-B"},
]

ATOL = {"ML": 0.05, "GS": 0.05}


def _simulate_dataset(psd_model, parameters):
    """Helper to create a synthetic dataset with drop_number_concentration."""
    psd = create_psd(psd_model, parameters)

    diameter_bin_edges = np.linspace(0, 10)
    coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
    diameters = coords_dict["diameter_bin_center"][1]

    drop_number_concentration = psd(diameters)
    da = xr.DataArray(
        drop_number_concentration,
        dims="diameter_bin_center",
        coords=coords_dict,
    )
    ds = xr.Dataset({"drop_number_concentration": da})
    return ds, psd


class TestGSOptimization:
    """Test PSD parameter estimation using Grid Search (GS)."""

    @pytest.mark.parametrize(("psd_model", "parameters", "optimizations"), CASES)
    @pytest.mark.parametrize("transformation", ["identity", "log", "sqrt"])
    @pytest.mark.parametrize("error_order", [1, 2])
    @pytest.mark.parametrize("censoring", ["none", "left", "right", "both"])
    def test_gs_on_nd(self, psd_model, parameters, optimizations, transformation, error_order, censoring):
        """Test PSD parameter estimation using Grid Search (GS) on N(D)."""
        if "GS" not in optimizations:
            pytest.skip(f"GS not available for {psd_model}")

        optimization_kwargs = {
            "target": "ND",
            "transformation": transformation,
            "error_order": error_order,
            "censoring": censoring,
        }
        ds, psd = _simulate_dataset(psd_model, parameters)
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd.name,
            optimization="GS",
            optimization_kwargs=optimization_kwargs,
        )

        for var in ds_params.data_vars:
            np.testing.assert_allclose(
                ds_params[var].data,
                parameters[var],
                rtol=0.2,
                atol=ATOL["GS"],
                err_msg=f"GS fitting of {psd_model} using {transformation=} {error_order=} {censoring=} causes inaccurate {var}.",  # noqa
            )

    @pytest.mark.parametrize(("psd_model", "parameters", "optimizations"), CASES)
    @pytest.mark.parametrize("target", ["R", "Z", "LWC"])
    def test_gs_on_variables(self, psd_model, parameters, optimizations, target):
        """Test PSD parameter estimation using Grid Search (GS) on LWC, R and Z."""
        if "GS" not in optimizations:
            pytest.skip(f"GS not available for {psd_model}")

        if psd_model == "NormalizedGammaPSD":  # Do not find minimum by just optimizing mu
            pytest.skip(f"GS on variables skipped  for {psd_model}")

        optimization_kwargs = {
            "target": target,
        }
        ds, psd = _simulate_dataset(psd_model, parameters)
        if target == "R":
            ds["fall_velocity"] = get_rain_fall_velocity_from_ds(ds)
            V = ds["fall_velocity"].to_numpy()
        else:
            V = None

        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd.name,
            optimization="GS",
            optimization_kwargs=optimization_kwargs,
        )

        # Test relative error is close to zero
        D = ds["diameter_bin_center"].to_numpy()
        dD = ds["diameter_bin_width"].to_numpy()
        ND_obs = ds["drop_number_concentration"].to_numpy()
        ND_pred = create_psd(psd_model=ds_params.attrs["disdrodb_psd_model"], parameters=ds_params)(D).to_numpy()
        error = _compute_target_variable_error(target, ND_obs, ND_pred, D, dD, V, relative=True)
        np.testing.assert_allclose(
            error,
            0,
            atol=0.01,
            err_msg=f"GS fitting of {psd_model} using {target=} causes inaccurate {target} reproduction.",
        )

    @pytest.mark.parametrize("psd_model", available_psd_models())
    @pytest.mark.parametrize("target", ["ND", "R", "Z", "LWC"])
    def test_gs_with_zeros_nd(self, psd_model, target):
        """Test PSD parameter estimation using Grid Search (GS) when N(D) is all zeros."""
        optimization_kwargs = {
            "target": target,
        }

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.zeros(diameters.shape)
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd_model,
            optimization="GS",
            optimization_kwargs=optimization_kwargs,
        )
        for var in ds_params.data_vars:
            assert np.isnan(ds_params[var])

    @pytest.mark.parametrize("psd_model", available_psd_models())
    @pytest.mark.parametrize("target", ["ND", "R", "Z", "LWC"])
    def test_gs_with_nan_nd(self, psd_model, target):
        """Test PSD parameter estimation using Grid Search (GS) when N(D) has np.nan."""
        optimization_kwargs = {
            "target": target,
        }

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.ones(diameters.shape)
        drop_number_concentration[2] = np.nan
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd_model,
            optimization="GS",
            optimization_kwargs=optimization_kwargs,
        )
        for var in ds_params.data_vars:
            assert np.isnan(ds_params[var])

    @pytest.mark.parametrize("psd_model", available_psd_models())
    @pytest.mark.parametrize("target", ["ND", "R", "Z", "LWC"])
    def test_gs_with_inf_nd(self, psd_model, target):
        """Test PSD parameter estimation using Grid Search (GS) when N(D) has np.inf."""
        optimization_kwargs = {
            "target": target,
        }

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.ones(diameters.shape)
        drop_number_concentration[2] = np.inf
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd_model,
            optimization="GS",
            optimization_kwargs=optimization_kwargs,
        )
        for var in ds_params.data_vars:
            assert np.isnan(ds_params[var])


@pytest.mark.parametrize(("psd_model", "parameters", "optimizations"), CASES)
class TestMLOptimization:
    """Test PSD parameter estimation using Maximum Likelihood (ML)."""

    @pytest.mark.parametrize("modified_kwarg", ML_KWARGS_ALTERNATIVES)
    def test_ml_estimation(self, psd_model, parameters, optimizations, modified_kwarg):
        if "ML" not in optimizations:
            pytest.skip(f"ML not available for {psd_model}")

        optimization_kwargs = {**DEFAULT_ML_KWARGS, **modified_kwarg}

        ds, psd = _simulate_dataset(psd_model, parameters)
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd.name,
            optimization="ML",
            optimization_kwargs=optimization_kwargs,
        )

        for var in ds_params.data_vars:
            np.testing.assert_allclose(
                ds_params[var].data,
                parameters[var],
                rtol=0.2,
                atol=ATOL["ML"],
                err_msg=f"ML fitting of {psd_model} using {modified_kwarg} causes inaccurate {var}.",
            )

    def test_ml_estimation_with_mom_init(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Maximum Likelihood (ML) with MOM initialization."""
        if "ML" not in optimizations:
            pytest.skip(f"ML not available for {psd_model}")

        optimization_kwargs = {**DEFAULT_ML_KWARGS}
        mom_methods = available_mom_methods(psd_model)
        ds, psd = _simulate_dataset(psd_model, parameters)

        for init_method in mom_methods:
            optimization_kwargs["init_method"] = init_method
            ds_params = estimate_model_parameters(
                ds,
                psd_model=psd.name,
                optimization="ML",
                optimization_kwargs=optimization_kwargs,
            )
            for var in ds_params.data_vars:
                np.testing.assert_allclose(
                    ds_params[var].data,
                    parameters[var],
                    rtol=0.2,
                    atol=ATOL["ML"],
                    err_msg=f"ML fitting of {psd_model} using {init_method=} causes inaccurate {var}.",
                )

    def test_ml_with_zeros_nd(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Maximum Likelihood (ML) when N(D) is all zeros."""
        if "ML" not in optimizations:
            pytest.skip(f"ML not available for {psd_model}")

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.zeros(diameters.shape)
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(ds, psd_model=psd_model, optimization="ML")
        for var in ds_params.data_vars:
            assert np.isnan(ds_params[var])

    def test_ml_with_nan_nd(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Maximum Likelihood (ML) when N(D) has np.nan."""
        if "ML" not in optimizations:
            pytest.skip(f"ML not available for {psd_model}")

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.ones(diameters.shape)
        drop_number_concentration[2] = np.nan
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(ds, psd_model=psd_model, optimization="ML")
        for var in ds_params.data_vars:
            assert np.isnan(ds_params[var])

    def test_ml_with_inf_nd(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Maximum Likelihood (ML) when N(D) has np.inf."""
        if "ML" not in optimizations:
            pytest.skip(f"ML not available for {psd_model}")

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.ones(diameters.shape)
        drop_number_concentration[2] = np.inf
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(ds, psd_model=psd_model, optimization="ML")

        for var in ds_params.data_vars:
            assert np.isnan(ds_params[var])


@pytest.mark.parametrize(("psd_model", "parameters", "optimizations"), CASES)
class TestMOMOptimization:
    """Test PSD parameter estimation using Method of Moments (MOM)."""

    def test_mom_estimation(self, psd_model, parameters, optimizations):
        if "MOM" not in optimizations:
            pytest.skip(f"MOM not available for {psd_model}")

        mom_methods = available_mom_methods(psd_model)
        optimization_kwargs = {}
        optimization_kwargs["mom_methods"] = mom_methods

        ds, psd = _simulate_dataset(psd_model, parameters)
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd.name,
            optimization="MOM",
            optimization_kwargs=optimization_kwargs,
        )

        assert isinstance(ds_params, xr.Dataset)
        # Just test it runs ... because MOM methods do not provide good values

    def test_mom_with_zeros_nd(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Method of Moments (MOM) when N(D) is all zeros."""
        if "MOM" not in optimizations:
            pytest.skip(f"MOM not available for {psd_model}")

        mom_methods = available_mom_methods(psd_model)
        optimization_kwargs = {}
        optimization_kwargs["mom_methods"] = mom_methods

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.zeros(diameters.shape)
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd_model,
            optimization="MOM",
            optimization_kwargs=optimization_kwargs,
        )
        for var in ds_params.data_vars:
            assert np.all(np.isnan(ds_params[var]))

    def test_mom_with_nan_nd(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Method of Moments (MOM) when N(D) has np.nan."""
        if "MOM" not in optimizations:
            pytest.skip(f"MOM not available for {psd_model}")

        mom_methods = available_mom_methods(psd_model)
        optimization_kwargs = {}
        optimization_kwargs["mom_methods"] = mom_methods

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.ones(diameters.shape)
        drop_number_concentration[2] = np.nan
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd_model,
            optimization="MOM",
            optimization_kwargs=optimization_kwargs,
        )
        for var in ds_params.data_vars:
            assert np.all(np.isnan(ds_params[var]))

    def test_mom_with_inf_nd(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Method of Moments (MOM) when N(D) has np.inf."""
        if "MOM" not in optimizations:
            pytest.skip(f"MOM not available for {psd_model}")

        mom_methods = available_mom_methods(psd_model)
        optimization_kwargs = {}
        optimization_kwargs["mom_methods"] = mom_methods

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.ones(diameters.shape)
        drop_number_concentration[2] = np.inf
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd_model,
            optimization="MOM",
            optimization_kwargs=optimization_kwargs,
        )
        for var in ds_params.data_vars:
            assert np.all(np.isnan(ds_params[var]))


####-------------------------------------------------------------------------------------------------
#### Test checks


def test_available_mom_methods():
    """Test available_mom_methods."""
    mom_methods = available_mom_methods("GammaPSD")
    assert isinstance(mom_methods, list)
    assert "M234" in mom_methods

    # Assert raise error
    with pytest.raises(NotImplementedError, match="NormalizedGammaPSD"):
        available_mom_methods("NormalizedGammaPSD")


@pytest.mark.parametrize("psd_model", available_psd_models())
def test_available_optimization(psd_model):
    """Test available_optimization."""
    optimizations = available_optimization(psd_model)
    assert isinstance(optimizations, list)
    assert "GS" in optimizations


class TestCheckTarget:
    """Test suite for check_target."""

    @pytest.mark.parametrize("valid", ["ND", "R", "Z", "LWC"])
    def test_valid_targets(self, valid):
        """Valid targets should be returned unchanged."""
        assert check_target(valid) == valid

    def test_invalid_target_raises(self):
        """Invalid target should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'target'"):
            check_target("INVALID")


class TestCheckTransformation:
    """Test suite for check_transformation."""

    @pytest.mark.parametrize("valid", ["identity", "log", "sqrt"])
    def test_valid_transformations(self, valid):
        """Valid transformations should be returned unchanged."""
        assert check_transformation(valid) == valid

    def test_invalid_transformation_raises(self):
        """Invalid transformation should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'transformation'"):
            check_transformation("INVALID")


class TestCheckLikelihood:
    """Test suite for check_likelihood."""

    @pytest.mark.parametrize("valid", ["multinomial", "poisson"])
    def test_valid_likelihoods(self, valid):
        """Valid likelihoods should be returned unchanged."""
        assert check_likelihood(valid) == valid

    def test_invalid_likelihood_raises(self):
        """Invalid likelihood should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'likelihood'"):
            check_likelihood("INVALID")


class TestCheckTruncatedLikelihood:
    """Test suite for check_truncated_likelihood."""

    @pytest.mark.parametrize("valid", [True, False])
    def test_valid_bool(self, valid):
        """Valid boolean should be returned unchanged."""
        assert check_truncated_likelihood(valid) is valid

    @pytest.mark.parametrize("invalid", ["yes", 1, None, [True]])
    def test_invalid_type_raises(self, invalid):
        """Non-bool values should raise TypeError."""
        with pytest.raises(TypeError, match="Invalid 'truncated_likelihood'"):
            check_truncated_likelihood(invalid)


class TestCheckProbabilityMethod:
    """Test suite for check_probability_method."""

    @pytest.mark.parametrize("valid", ["cdf", "pdf"])
    def test_valid_methods(self, valid):
        """Valid probability methods should be returned unchanged."""
        assert check_probability_method(valid) == valid

    def test_invalid_method_raises(self):
        """Invalid probability method should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'probability_method'"):
            check_probability_method("INVALID")


class TestCheckOptimizer:
    """Test suite for check_optimizer."""

    @pytest.mark.parametrize("valid", ["Nelder-Mead", "Powell", "L-BFGS-B"])
    def test_valid_optimizers(self, valid):
        """Valid optimizers should be returned unchanged."""
        assert check_optimizer(valid) == valid

    def test_invalid_optimizer_raises(self):
        """Invalid optimizer should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'optimizer'"):
            check_optimizer("INVALID")


class TestCheckMomMethods:
    """Test suite for check_mom_methods."""

    @pytest.mark.parametrize("psd_model", available_psd_models())
    def test_valid_mom_methods(self, psd_model):
        """Valid MOM methods should be returned unchanged."""
        try:
            valid_methods = available_mom_methods(psd_model)
        except Exception:
            pytest.skip(f"No MOM methods for {psd_model}")

        for method in valid_methods:
            result = check_mom_methods(method, psd_model)
            assert result == [method]

    def test_list_input(self):
        """List of methods should be preserved."""
        psd_model = "ExponentialPSD"
        valid_methods = available_mom_methods(psd_model)
        if not valid_methods:
            pytest.skip(f"No MOM methods for {psd_model}")
        result = check_mom_methods(valid_methods, psd_model)
        assert result == valid_methods

    def test_allow_none(self):
        """'None' should be allowed when allow_none=True."""
        psd_model = "ExponentialPSD"
        result = check_mom_methods(None, psd_model, allow_none=True)
        assert result == ["None"]

    def test_invalid_method_raises(self):
        """Invalid methods should raise ValueError."""
        psd_model = "ExponentialPSD"
        with pytest.raises(ValueError, match="Unknown mom_methods"):
            check_mom_methods("INVALID", psd_model)


class TestCheckOptimization:
    """Test suite for check_optimization."""

    @pytest.mark.parametrize("valid", ["ML", "GS", "MOM"])
    def test_valid_optimizations(self, valid):
        """Valid optimizations should be returned unchanged."""
        assert check_optimization(valid) == valid

    def test_invalid_optimization_raises(self):
        """Invalid optimizations should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'optimization'"):
            check_optimization("INVALID")


####-------------------------------------------------------------------------------------------------
class TestEstimateModelParameters:
    """Test that estimate_model_parameters runs and returns expected structure."""

    def test_NormalizedGammaPSD_fitting(self):
        """Test Normalized Gamma PSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = estimate_model_parameters(ds, psd_model="NormalizedGammaPSD", optimization="GS")
        assert ds_out.attrs["disdrodb_psd_model"] == "NormalizedGammaPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_kwargs" in ds_out.attrs

        # Test raise error
        with pytest.raises(NotImplementedError, match="ML optimization is not available"):
            estimate_model_parameters(ds, psd_model="NormalizedGammaPSD", optimization="ML")

        with pytest.raises(NotImplementedError, match="MOM optimization is not available"):
            estimate_model_parameters(ds, psd_model="NormalizedGammaPSD", optimization="MOM")

    def test_GammaPSD_fitting(self):
        """Test Gamma PSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = estimate_model_parameters(ds, psd_model="GammaPSD", optimization="GS")

        ds_out = estimate_model_parameters(ds, psd_model="GammaPSD", optimization="ML")

        ds_out = estimate_model_parameters(ds, psd_model="GammaPSD", optimization="MOM")
        assert "mom_method" in ds_out.dims
        assert ds_out.attrs["disdrodb_psd_model"] == "GammaPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_kwargs" in ds_out.attrs

    def test_LognormalPSD_fitting(self):
        """Test LognormalPSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = estimate_model_parameters(ds, psd_model="LognormalPSD", optimization="GS")
        ds_out = estimate_model_parameters(ds, psd_model="LognormalPSD", optimization="ML")
        ds_out = estimate_model_parameters(ds, psd_model="LognormalPSD", optimization="MOM")

        assert ds_out.attrs["disdrodb_psd_model"] == "LognormalPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_kwargs" in ds_out.attrs

    def test_ExponentialPSD_fitting(self):
        """Test ExponentialPSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = estimate_model_parameters(ds, psd_model="ExponentialPSD", optimization="GS")
        ds_out = estimate_model_parameters(ds, psd_model="ExponentialPSD", optimization="ML")
        ds_out = estimate_model_parameters(ds, psd_model="ExponentialPSD", optimization="MOM")

        assert ds_out.attrs["disdrodb_psd_model"] == "ExponentialPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_kwargs" in ds_out.attrs

    def test_fitting_without_init_method(self):
        """Test fitting without moment initialization."""
        ds = create_template_l2e_dataset()
        ds_out = estimate_model_parameters(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_kwargs={"init_method": None},
        )
        ds_out = estimate_model_parameters(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_kwargs={"init_method": "None"},
        )
        ds_out = estimate_model_parameters(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_kwargs={"init_method": ["None"]},
        )
        assert isinstance(ds_out, xr.Dataset)

    def test_fitting_with_multiple_init_method(self):
        """Test fitting with multiple initialization methods."""
        ds = create_template_l2e_dataset()
        ds_out = estimate_model_parameters(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_kwargs={"init_method": [None, "M234", "M346"]},
        )
        assert "init_method" in ds_out.dims
        assert ds_out.sizes["init_method"] == 3

    def test_raise_error_if_missing_field(self):
        """Test raise error if drop_number_concentration variable is not available."""
        with pytest.raises(ValueError):
            estimate_model_parameters(
                ds=xr.Dataset(),
                psd_model="GammaPSD",
                optimization="ML",
                optimization_kwargs={"init_method": None},
            )


####-------------------------------------------------------------------------------------------------


def make_test_data(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
):
    """
    Generate obs, pred, and diameter_bin_width DataArrays for GoF tests.

    Parameters
    ----------
    obs_data : array_like (time, bins)
        obs values per time step and bin.
    pred_data : array_like (time, bins)
        pred values per time step and bin.

    Returns
    -------
    obs : xr.DataArray
    pred : xr.DataArray
    diameter_bin_width : xr.DataArray
    """
    # Convert inputs
    obs = np.asarray(obs_data)
    pred = np.asarray(pred_data)
    if obs.shape != pred.shape:
        raise ValueError("obs_data and pred_data must have the same shape")

    n_times, n_bins = obs.shape

    # Common coordinates (invariant across tests)
    bin_centers = np.arange(1, n_bins + 1, dtype=float)
    bin_widths = np.ones(n_bins, dtype=float)

    # Build coordinates
    diameter_bin_center = xr.DataArray(
        data=bin_centers,
        dims=[DIAMETER_DIMENSION],
        name="diameter_bin_center",
    )
    diameter_bin_width = xr.DataArray(
        data=bin_widths,
        dims=[DIAMETER_DIMENSION],
        name="diameter_bin_width",
    )
    time = xr.DataArray(
        data=np.arange(n_times),
        dims=["time"],
        coords={"time": np.arange(n_times)},
        name="time",
    )

    # Create DataArrays
    obs = xr.DataArray(
        data=obs,
        dims=["time", DIAMETER_DIMENSION],
        coords={
            "time": time,
            "diameter_bin_center": diameter_bin_center,
            "diameter_bin_width": diameter_bin_width,
        },
        name="obs_values",
    )
    pred = xr.DataArray(
        data=pred,
        dims=["time", DIAMETER_DIMENSION],
        coords={
            "time": time,
            "diameter_bin_center": diameter_bin_center,
            "diameter_bin_width": diameter_bin_width,
        },
        name="pred_values",
    )

    return obs, pred


class TestComputeGoFStats:
    """Test suite for compute_gof_stats function."""

    def test_zero_zero_case(self):
        """Test that equal (0) observed and predicted values should yield zero errors."""
        # Generate test inputs
        data = np.zeros((2, 3))
        obs, pred = make_test_data(data, data)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred).isel(time=0)

        # Assert values
        assert ds["MAE"].values == 0
        assert ds["MaxAE"].values == 0
        assert ds["PeakDiff"].values == 0
        assert ds["NtDiff"].values == 0
        assert ds["RelMaxAE"].values == 0
        assert ds["RelPeakDiff"].values == 0
        assert ds["KLDiv"].values == 0

        assert np.isnan(ds["R2"].values)  # standard deviation of arrays with unique values is 0
        assert np.isnan(ds["DmodeDiff"].values)  # for zero arrays, diameter mode is NaN

    def test_one_one_case(self):
        """Test that equal observed and predicted values should yield zero errors."""
        # Generate test inputs
        data = np.ones((2, 3))
        obs, pred = make_test_data(data, data)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred).isel(time=0)

        # Assert values
        assert np.isnan(ds["R2"].values)  # standard deviation of arrays with unique values is 0
        assert ds["MAE"].values == 0
        assert ds["MaxAE"].values == 0
        assert ds["RelMaxAE"].values == 0
        assert ds["PeakDiff"].values == 0
        assert ds["RelPeakDiff"].values == 0
        assert ds["DmodeDiff"].values == 0
        assert ds["NtDiff"].values == 0
        assert ds["KLDiv"].values == 0

    def test_one_zero_case(self):
        """Test the case where observed values are zeros."""
        # Generate test inputs
        obs = np.zeros((2, 3))
        pred = np.ones((2, 3))
        obs, pred = make_test_data(obs, pred)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred).isel(time=0)

        # Assert values
        assert np.isnan(ds["R2"].values)  # standard deviation of arrays with unique values is 0
        assert ds["MAE"].values == 1
        assert ds["MaxAE"].values == 1
        assert np.isnan(ds["RelMaxAE"].values)  # division by 0 obs_max
        assert ds["PeakDiff"].values == -1
        assert np.isnan(ds["RelPeakDiff"].values)  # division by 0 obs_max
        assert np.isnan(ds["DmodeDiff"].values)  # for zero arrays, diameter mode is NaN
        assert ds["NtDiff"].values == 3
        assert np.isnan(ds["KLDiv"].values)  # pk of zeros arrays is NaN (pk=obs/Nt_obs=0/0)

    def test_zero_one_case(self):
        """Test the case where predicted values are zeros."""
        # Generate test inputs
        obs = np.ones((2, 3))
        pred = np.zeros((2, 3))
        obs, pred = make_test_data(obs, pred)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred).isel(time=0)

        # Assert values
        assert np.isnan(ds["R2"].values)  # standard deviation of arrays with unique values is 0
        assert ds["MAE"].values == 1
        assert ds["MaxAE"].values == 1
        assert ds["RelMaxAE"].values == 1
        assert ds["PeakDiff"].values == 1
        assert ds["RelPeakDiff"].values == 1
        assert np.isnan(ds["DmodeDiff"].values)  # for zero arrays, diameter mode is NaN
        assert ds["NtDiff"].values == -3
        assert np.isnan(ds["KLDiv"].values)  # qk of zeros arrays is NaN (qk=pred/Nt_pred=0/0)

    def test_real_case(self):
        """Test the case where pred values are zeros."""
        # Generate test inputs
        obs = np.arange(0, 6).reshape(2, 3)
        pred = obs + 1
        obs, pred = make_test_data(obs, pred)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred).isel(time=0)

        # Assert values
        assert ds["MAE"].values == 1
        assert ds["MaxAE"].values == 1
        assert ds["RelMaxAE"].values == 0.5

        assert ds["PeakDiff"].values == -1
        assert ds["RelPeakDiff"].values == -0.5
        assert ds["DmodeDiff"].values == 0

        assert ds["NtDiff"].values == 3
        assert ds["KLDiv"].values == 0.19
        assert ds["R2"].values == 1

    def test_single_nan_case(self):
        """Test the case where there are NaN values."""
        # Generate test inputs
        obs = np.arange(0, 6, dtype=float).reshape(2, 3)
        pred = obs + 1
        pred[0, 2] = np.nan

        obs, pred = make_test_data(obs, pred)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred)

        # Assert values
        for var in ds.data_vars:
            assert np.isnan(ds[var].isel(time=0).values)

        # Assert values
        for var in ds.data_vars:
            assert not np.isnan(ds[var].isel(time=1).values)

    def test_all_nan_case(self):
        """Test the case where all values are NaN values."""
        # Generate test inputs
        obs = np.zeros((2, 3)) * np.nan
        pred = np.zeros((2, 3)) * np.nan
        obs, pred = make_test_data(obs, pred)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred).isel(time=0)

        # Assert values
        for var in ds.data_vars:
            assert np.isnan(ds[var].values)

    def test_dask_case(self):
        """Test that lazy dask computations are allowed."""
        # Generate test inputs
        obs = np.arange(0, 6).reshape(2, 3)
        pred = obs + 1
        obs, pred = make_test_data(obs, pred)
        obs = obs.chunk({"time": 1})
        pred = pred.chunk({"time": 1})

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred)
        for var in ds.data_vars:
            assert hasattr(ds[var].data, "chunks")
        ds = ds.compute().isel(time=0)

        # Assert values
        assert ds["MAE"].values == 1
        assert ds["MaxAE"].values == 1
        assert ds["RelMaxAE"].values == 0.5

        assert ds["PeakDiff"].values == -1
        assert ds["RelPeakDiff"].values == -0.5
        assert ds["DmodeDiff"].values == 0

        assert ds["NtDiff"].values == 3
        assert ds["KLDiv"].values == 0.19
        assert ds["R2"].values == 1

    def test_only_diameter_dimension(self):
        """Test case where only diameter dimension is present."""
        # Generate test inputs
        obs = np.arange(0, 6).reshape(2, 3)
        pred = obs + 1
        obs, pred = make_test_data(obs, pred)

        # Remove time dimension
        obs = obs.isel(time=0)
        pred = pred.isel(time=0)

        # Compute GoF statistics
        ds = compute_gof_stats(obs, pred)
        assert isinstance(ds, xr.Dataset)


####---------------------------------------------------------------------------
#### Test other functions
class TestGetExponentialParametersZhang2008:
    """Test suite for get_exponential_parameters_Zhang2008."""

    def test_known_case(self):
        """Check parameters against manual calculation for known inputs."""
        moment_l = 2.0
        moment_m = 4.0
        l, m = 0, 1  # noqa: E741
        N0, Lambda = get_exponential_parameters_Zhang2008(moment_l, moment_m, l, m)
        np.testing.assert_allclose(Lambda, 0.5)
        np.testing.assert_allclose(N0, 1.0)

    @pytest.mark.parametrize(("l", "m"), [(0, 1), (1, 2), (2, 3)])
    def test_various_moments(self, l, m):  # noqa: E741
        """Test calculation works for multiple (l, m) moment order pairs."""
        moment_l = 5.0
        moment_m = 10.0
        N0, Lambda = get_exponential_parameters_Zhang2008(moment_l, moment_m, l, m)
        assert np.isfinite(N0)
        assert np.isfinite(Lambda)
        assert N0 > 0
        assert Lambda > 0

    def test_invalid_equal_moments(self):
        """Check raises or fails gracefully when l == m."""
        moment_l = 1.0
        moment_m = 2.0
        l = m = 1  # noqa: E741
        with pytest.raises(ValueError):
            get_exponential_parameters_Zhang2008(moment_l, moment_m, l, m)
