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
"""Testing grid search utilities."""

import numpy as np
import pytest

from disdrodb.psd.grid_search import (
    CENSORING,
    DISTRIBUTION_TARGETS,
    ERROR_METRICS,
    INTEGRAL_TARGETS,
    TARGETS,
    TRANSFORMATIONS,
    apply_transformation,
    check_censoring,
    check_error_metric,
    check_target,
    check_transformation,
    check_valid_error_metric,
    compute_cost_function,
    compute_errors,
    compute_jensen_shannon_distance,
    compute_kl_divergence,
    compute_kolmogorov_smirnov_distance,
    compute_lwc,
    compute_rain_rate,
    compute_target_variable,
    compute_wasserstein_distance,
    compute_z,
    left_truncate_bins,
    normalize_errors,
    right_truncate_bins,
    truncate_bin_edges,
)


# TODOs:
# Add to targets also moments M1...M6
class TestCheckTarget:
    """Test suite for check_target."""

    @pytest.mark.parametrize("valid", TARGETS)
    def test_valid_targets(self, valid):
        """Valid targets should be returned unchanged."""
        assert check_target(valid) == valid

    def test_invalid_target_raises(self):
        """Invalid target should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'target'"):
            check_target("INVALID")


class TestCheckCensoring:
    """Test suite for check_censoring."""

    @pytest.mark.parametrize("valid", CENSORING)
    def test_valid_censoring(self, valid):
        """Valid censoring options should be returned unchanged."""
        assert check_censoring(valid) == valid

    def test_invalid_censoring_raises(self):
        """Invalid censoring should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'censoring'"):
            check_censoring("invalid_option")


class TestCheckTransformation:
    """Test suite for check_transformation."""

    @pytest.mark.parametrize("valid", TRANSFORMATIONS)
    def test_valid_transformations(self, valid):
        """Valid transformations should be returned unchanged."""
        assert check_transformation(valid) == valid

    def test_invalid_transformation_raises(self):
        """Invalid transformation should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'transformation'"):
            check_transformation("INVALID")


class TestCheckErrorMetric:
    """Test suite for check_error_metric."""

    @pytest.mark.parametrize("valid", ERROR_METRICS)
    def test_valid_error_metrics(self, valid):
        """Valid error metrics should be returned unchanged."""
        assert check_error_metric(valid) == valid

    def test_invalid_error_metric_raises(self):
        """Invalid error metric should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'error_metric'"):
            check_error_metric("INVALID_METRIC")


class TestCheckValidErrorMetric:
    """Test suite for check_valid_error_metric."""

    def test_metrics_with_distributions(self):
        """Distribution metrics should be valid for ND and H(x) targets."""
        for metric in ERROR_METRICS:
            for target in DISTRIBUTION_TARGETS:
                result = check_valid_error_metric(metric, target=target)
                assert result == metric

    def test_dist_metrics_with_integrals_raises(self):
        """Distribution metrics are not valid for Z, R, LWC targets."""
        for target in INTEGRAL_TARGETS:
            for metric in ERROR_METRICS:
                with pytest.raises(ValueError, match="error_metric should be 'None'"):
                    check_valid_error_metric(metric, target=target)

    def test_none_metric_with_integrals(self):
        """Distribution metrics are not valid for Z, R, LWC targets."""
        for target in INTEGRAL_TARGETS:
            for metric in ERROR_METRICS:
                metric = check_valid_error_metric(error_metric=None, target=target)
                assert metric is None


class TestComputeRainRate:
    """Test suite for compute_rain_rate."""

    def test_rain_rate_1d_input(self):
        """Rain rate computation should work correctly with 1D arrays."""
        ND = np.array([100, 200, 150])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        V = np.array([1.0, 2.0, 3.0])
        result = compute_rain_rate(ND, D, dD, V)
        assert isinstance(result, (float, np.floating))
        np.testing.assert_allclose(result, 0.36403, atol=1e-4)

    def test_rain_rate_2d_input(self):
        """Rain rate computation should return per-sample values for 2D arrays."""
        ND = np.array([[100, 200, 150], [50, 100, 75]])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        V = np.array([1.0, 2.0, 3.0])
        result = compute_rain_rate(ND, D, dD, V)
        assert result.shape == (2,)
        expected_result = np.array([0.36403205, 0.18201602])
        np.testing.assert_allclose(result, expected_result, atol=1e-5)

    def test_rain_rate_zero_input(self):
        """Rain rate should be zero for all-zero N(D)."""
        ND = np.zeros(3)
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        V = np.array([1.0, 2.0, 3.0])
        result = compute_rain_rate(ND, D, dD, V)
        assert result == 0.0


class TestComputeLWC:
    """Test suite for compute_lwc."""

    def test_lwc_1d_input(self):
        """LWC computation should return scalar for 1D input."""
        ND = np.array([100, 200, 150])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        result = compute_lwc(ND, D, dD)
        assert isinstance(result, (float, np.floating))
        np.testing.assert_allclose(result, 0.037633, atol=1e-5)

    def test_lwc_2d_input(self):
        """LWC computation should return per-sample values for 2D arrays."""
        ND = np.array([[100, 200, 150], [50, 100, 75]])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        result = compute_lwc(ND, D, dD)
        assert result.shape == (2,)
        expected_result = np.array([0.03763366, 0.01881683])
        np.testing.assert_allclose(result, expected_result, atol=1e-5)

    def test_lwc_custom_water_density(self):
        """LWC should scale correctly with custom water density."""
        ND = np.array([100, 200, 150])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        result1 = compute_lwc(ND, D, dD, rho_w=1000)
        result2 = compute_lwc(ND, D, dD, rho_w=2000)
        assert result2 == 2 * result1

    def test_lwc_zero_input(self):
        """LWC should be zero for all-zero N(D)."""
        ND = np.zeros(3)
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        result = compute_lwc(ND, D, dD)
        assert result == 0.0


class TestComputeZ:
    """Test suite for compute_z."""

    def test_z_1d_input(self):
        """Z computation should return scalar for 1D input."""
        ND = np.array([100, 200, 150])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        result = compute_z(ND, D, dD)
        assert isinstance(result, (float, np.floating))
        np.testing.assert_allclose(result, 22.8106, atol=1e-3)

    def test_z_2d_input(self):
        """Z computation should return per-sample values for 2D arrays."""
        ND = np.array([[100, 200, 150], [50, 100, 75]])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        result = compute_z(ND, D, dD)
        assert result.shape == (2,)
        expected_result = np.array([22.81068894, 19.80038898])
        np.testing.assert_allclose(result, expected_result, atol=1e-3)


class TestComputeTargetVariable:
    """Test suite for compute_target_variable."""

    @pytest.fixture
    def sample_data(self):
        """Provide common sample data for target variable tests."""
        ND_obs = np.array([100, 200, 150])
        ND_preds = np.array([[100, 200, 150], [50, 100, 75]])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        V = np.array([1.0, 2.0, 3.0])
        return ND_obs, ND_preds, D, dD, V

    def test_target_z(self, sample_data):
        """Target variable computation for Z should return numeric values."""
        ND_obs, ND_preds, D, dD, V = sample_data
        obs, pred = compute_target_variable("Z", ND_obs, ND_preds, D, dD, V)
        assert isinstance(obs, (float, np.floating))
        assert pred.shape == (2,)

    def test_target_r(self, sample_data):
        """Target variable computation for R should return rain rates."""
        ND_obs, ND_preds, D, dD, V = sample_data
        obs, pred = compute_target_variable("R", ND_obs, ND_preds, D, dD, V)
        assert isinstance(obs, (float, np.floating))
        assert pred.shape == (2,)

    def test_target_lwc(self, sample_data):
        """Target variable computation for LWC should return liquid water content."""
        ND_obs, ND_preds, D, dD, V = sample_data
        obs, pred = compute_target_variable("LWC", ND_obs, ND_preds, D, dD, V)
        assert isinstance(obs, (float, np.floating))
        assert pred.shape == (2,)

    def test_target_nd(self, sample_data):
        """Target variable computation for ND should return unchanged distributions."""
        ND_obs, ND_preds, D, dD, V = sample_data
        obs, pred = compute_target_variable("N(D)", ND_obs, ND_preds, D, dD, V)
        np.testing.assert_array_equal(obs, ND_obs)
        np.testing.assert_array_equal(pred, ND_preds)

    def test_target_hx(self, sample_data):
        """Target variable computation for H(x) should return unchanged distributions."""
        ND_obs, ND_preds, D, dD, V = sample_data
        obs, pred = compute_target_variable("H(x)", ND_obs, ND_preds, D, dD, V)
        np.testing.assert_array_equal(obs, ND_obs)
        np.testing.assert_array_equal(pred, ND_preds)


class TestLeftTruncateBins:
    """Test suite for left_truncate_bins."""

    def test_left_truncate_with_leading_zeros(self):
        """Left truncate should remove leading zero bins."""
        ND_obs = np.array([0, 0, 100, 200, 150])
        ND_preds = np.array([[0, 0, 100, 200, 150], [0, 0, 50, 100, 75]])
        D = np.array([0.1, 0.2, 0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        V = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        result = left_truncate_bins(ND_obs, ND_preds, D, dD, V)
        assert result is not None
        assert result[0].shape[0] == 3
        np.testing.assert_array_equal(result[0], ND_obs[2:])

    def test_left_truncate_all_zeros(self):
        """Left truncate should return None for all-zero observations."""
        ND_obs = np.zeros(5)
        ND_preds = np.zeros((2, 5))
        D = np.arange(5) * 0.1
        dD = np.ones(5) * 0.1
        V = np.ones(5)
        result = left_truncate_bins(ND_obs, ND_preds, D, dD, V)
        assert result is None

    def test_left_truncate_no_leading_zeros(self):
        """Left truncate should not truncate when no leading zeros."""
        ND_obs = np.array([100, 200, 150])
        ND_preds = np.array([[100, 200, 150], [50, 100, 75]])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        V = np.array([1.0, 2.0, 3.0])
        result = left_truncate_bins(ND_obs, ND_preds, D, dD, V)
        assert result is not None
        np.testing.assert_array_equal(result[0], ND_obs)


class TestRightTruncateBins:
    """Test suite for right_truncate_bins."""

    def test_right_truncate_with_trailing_zeros(self):
        """Right truncate should remove trailing zero bins."""
        ND_obs = np.array([100, 200, 150, 0, 0])
        ND_preds = np.array([[100, 200, 150, 0, 0], [50, 100, 75, 0, 0]])
        D = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        dD = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        V = np.array([1.0, 2.0, 3.0, 3.5, 4.0])
        result = right_truncate_bins(ND_obs, ND_preds, D, dD, V)
        assert result is not None
        assert result[0].shape[0] == 3
        np.testing.assert_array_equal(result[0], ND_obs[:3])

    def test_right_truncate_all_zeros(self):
        """Right truncate should return None for all-zero observations."""
        ND_obs = np.zeros(5)
        ND_preds = np.zeros((2, 5))
        D = np.arange(5) * 0.1
        dD = np.ones(5) * 0.1
        V = np.ones(5)
        result = right_truncate_bins(ND_obs, ND_preds, D, dD, V)
        assert result is None

    def test_right_truncate_no_trailing_zeros(self):
        """Right truncate should not truncate when no trailing zeros."""
        ND_obs = np.array([100, 200, 150])
        ND_preds = np.array([[100, 200, 150], [50, 100, 75]])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        V = np.array([1.0, 2.0, 3.0])
        result = right_truncate_bins(ND_obs, ND_preds, D, dD, V)
        assert result is not None
        np.testing.assert_array_equal(result[0], ND_obs)


class TestTruncateBinEdges:
    """Test suite for truncate_bin_edges."""

    @pytest.fixture
    def sample_data(self):
        """Provide common sample data for truncation tests."""
        ND_obs = np.array([0, 0, 100, 200, 150, 0, 0])
        ND_preds = np.array([[0, 0, 100, 200, 150, 0, 0], [0, 0, 50, 100, 75, 0, 0]])
        D = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5])
        dD = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        V = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        return ND_obs, ND_preds, D, dD, V

    def test_truncate_left_only(self, sample_data):
        """Truncate left censored should remove leading zeros."""
        ND_obs, ND_preds, D, dD, V = sample_data
        result = truncate_bin_edges(ND_obs, ND_preds, D, dD, V, left_censored=True)
        assert result is not None
        assert result[0].shape[0] == 5

    def test_truncate_right_only(self, sample_data):
        """Truncate right censored should remove trailing zeros."""
        ND_obs, ND_preds, D, dD, V = sample_data
        result = truncate_bin_edges(ND_obs, ND_preds, D, dD, V, right_censored=True)
        assert result is not None
        assert result[0].shape[0] == 5

    def test_truncate_both(self, sample_data):
        """Truncate both censored should remove leading and trailing zeros."""
        ND_obs, ND_preds, D, dD, V = sample_data
        result = truncate_bin_edges(ND_obs, ND_preds, D, dD, V, left_censored=True, right_censored=True)
        assert result is not None
        assert result[0].shape[0] == 3

    def test_truncate_no_censoring(self, sample_data):
        """Truncate with no censoring should return unchanged data."""
        ND_obs, ND_preds, D, dD, V = sample_data
        result = truncate_bin_edges(ND_obs, ND_preds, D, dD, V, left_censored=False, right_censored=False)
        assert result is not None
        np.testing.assert_array_equal(result[0], ND_obs)

    def test_truncate_all_zeros_returns_none(self):
        """Truncate should return None when all observations are zero."""
        ND_obs = np.zeros(5)
        ND_preds = np.zeros((2, 5))
        D = np.arange(5) * 0.1
        dD = np.ones(5) * 0.1
        V = np.ones(5)
        result = truncate_bin_edges(ND_obs, ND_preds, D, dD, V, left_censored=True)
        assert result is None


class TestApplyTransformation:
    """Test suite for apply_transformation."""

    def test_identity_transformation(self):
        """Identity transformation should return unchanged values."""
        obs = np.array([1, 2, 3, 4, 5])
        pred = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        obs_t, pred_t = apply_transformation(obs, pred, "identity")
        np.testing.assert_array_equal(obs_t, obs)
        np.testing.assert_array_equal(pred_t, pred)

    def test_log_transformation(self):
        """Log transformation should apply log(x+1) to handle zeros."""
        obs = np.array([0, 1, 10])
        pred = np.array([[0, 1, 10]])
        obs_t, pred_t = apply_transformation(obs, pred, "log")
        expected_obs = np.log(np.array([1, 2, 11]))
        np.testing.assert_array_almost_equal(obs_t, expected_obs)

    def test_sqrt_transformation(self):
        """Sqrt transformation should apply square root."""
        obs = np.array([0, 1, 4, 9])
        pred = np.array([[0, 1, 4, 9]])
        obs_t, pred_t = apply_transformation(obs, pred, "sqrt")
        expected_obs = np.sqrt(np.array([0, 1, 4, 9]))
        np.testing.assert_array_almost_equal(obs_t, expected_obs)

    def test_transformation_preserves_shape(self):
        """Transformation should preserve array shapes."""
        obs = np.array([1, 2, 3])
        pred = np.array([[1, 2, 3], [4, 5, 6]])
        obs_t, pred_t = apply_transformation(obs, pred, "log")
        assert obs_t.shape == obs.shape
        assert pred_t.shape == pred.shape


class TestComputeKLDivergence:
    """Test suite for compute_kl_divergence."""

    def test_kl_divergence_identical_distributions(self):
        """KL divergence should be near zero for identical distributions."""
        ND = np.array([100, 200, 150])
        dD = np.array([0.1, 0.1, 0.1])
        kl = compute_kl_divergence(ND, ND[None, :], dD)
        assert kl.shape == (1,)
        np.testing.assert_allclose(kl, 0.0)

    def test_kl_divergence_returns_array(self):
        """KL divergence should return array with one value per sample."""
        obs = np.array([100, 200, 150])
        pred = np.array([[100, 200, 150], [50, 100, 75], [1000, 2000, 3000]])
        dD = np.array([0.1, 0.1, 0.1])
        kl = compute_kl_divergence(obs, pred, dD)
        assert kl.shape == (3,)
        expected_results = np.array([0.00, 0.0, 5.66330123e-02])
        np.testing.assert_allclose(kl, expected_results, atol=1e-6)

    def test_kl_divergence_is_non_negative(self):
        """KL divergence should always be non-negative."""
        obs = np.array([100, 200, 150])
        pred = np.random.uniform(50, 250, size=(10, 3))
        dD = np.array([0.1, 0.1, 0.1])
        kl = compute_kl_divergence(obs, pred, dD)
        assert np.all(kl >= 0)


class TestComputeJensenShannonDistance:
    """Test suite for compute_jensen_shannon_distance."""

    def test_js_distance_identical_distributions(self):
        """JS distance should be zero for identical distributions."""
        ND = np.array([100, 200, 150])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        js = compute_jensen_shannon_distance(ND, ND[None, :], D, dD)
        assert js.shape == (1,)
        np.testing.assert_allclose(js, 0.0)

    def test_js_distance_returns_array(self):
        """JS distance should return array with one value per sample."""
        obs = np.array([100, 200, 150])
        pred = np.array([[100, 200, 150], [50, 100, 75], [1000, 2000, 3000]])
        dD = np.array([0.1, 0.1, 0.1])
        js = compute_jensen_shannon_distance(obs, pred, dD)
        assert js.shape == (3,)
        expected_results = np.array([0.0, 0, 0.11984403])
        np.testing.assert_allclose(js, expected_results, atol=1e-6)

    def test_js_distance_symmetry(self):
        """JS distance is symmetric: d(p,q) should equal d(q,p)."""
        obs1 = np.array([100, 200, 150])
        obs2 = np.array([50, 100, 75])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        js1 = compute_jensen_shannon_distance(obs1, obs2[None, :], D, dD)[0]
        js2 = compute_jensen_shannon_distance(obs2, obs1[None, :], D, dD)[0]
        np.testing.assert_almost_equal(js1, js2)


class TestComputeWassersteinDistance:
    """Test suite for compute_wasserstein_distance."""

    def test_wasserstein_identical_distributions(self):
        """Wasserstein distance should be zero for identical distributions."""
        ND = np.array([100, 200, 150])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        wd = compute_wasserstein_distance(ND, ND[None, :], D, dD)
        assert wd.shape == (1,)
        np.testing.assert_allclose(wd, 0.0)

    def test_wasserstein_returns_array(self):
        """Wasserstein distance should return array with one value per sample."""
        obs = np.array([100, 200, 150])
        pred = np.array([[100, 200, 150], [50, 100, 75], [2000, 4000, 3000]])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        wd = compute_wasserstein_distance(obs, pred, D, dD)
        assert wd.shape == (3,)
        expected_result = np.array([0.00000000e00, 4.18831636e-15, 2.10387263e-15])
        np.testing.assert_allclose(wd, expected_result, atol=1e-6)

        # Test bin integration is default
        wd_bin = compute_wasserstein_distance(obs, pred, D, dD, integration="bin")
        np.testing.assert_allclose(wd, wd_bin)

    def test_wasserstein_left_riemann_integration(self):
        """Wasserstein distance should support left Riemann integration method."""
        obs = np.array([100, 200, 150])
        pred = np.array([[100, 200, 150], [50, 100, 75], [1000, 2000, 3000]])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        wd = compute_wasserstein_distance(obs, pred, D, dD, integration="left_riemann")
        assert wd.shape == (3,)
        expected_result = np.array([0.0, 0, 0.111111])
        np.testing.assert_allclose(wd, expected_result, atol=1e-6)


class TestComputeKolmogorovSmirnovDistance:
    """Test suite for compute_kolmogorov_smirnov_distance."""

    def test_ks_identical_distributions(self):
        """KS distance should be zero for identical distributions."""
        ND = np.array([100, 200, 150])
        dD = np.array([0.1, 0.1, 0.1])
        ks, p = compute_kolmogorov_smirnov_distance(ND, ND[None, :], dD)
        assert ks.shape == (1,)
        np.testing.assert_allclose(ks, 0.0)
        assert p[0] > 0.99

    def test_ks_returns_arrays(self):
        """KS distance should return arrays for distance and p-value."""
        obs = np.array([100, 200, 150])
        pred = np.array([[100, 200, 150], [50, 100, 75], [3000, 2000, 1000]])
        dD = np.array([0.1, 0.1, 0.1])
        ks, p = compute_kolmogorov_smirnov_distance(obs, pred, dD)
        assert ks.shape == (3,)
        assert p.shape == (3,)
        expected_results = np.array([0.0, 0.0, 2.77777778e-01])
        np.testing.assert_allclose(ks, expected_results, atol=1e-6)

        expected_results = np.array([1, 1, 1])  # TODO
        np.testing.assert_allclose(p, expected_results, atol=1e-6)

    def test_ks_distance_bounds(self):
        """KS distance should be between 0 and 1."""
        obs = np.array([100, 200, 150])
        pred = np.random.uniform(50, 250, size=(10, 3))
        dD = np.array([0.1, 0.1, 0.1])
        ks, p = compute_kolmogorov_smirnov_distance(obs, pred, dD)
        assert np.all(ks >= 0)
        assert np.all(ks <= 1)

    def test_ks_pvalue_bounds(self):
        """KS p-value should be between 0 and 1."""
        obs = np.array([100, 200, 150])
        pred = np.random.uniform(50, 250, size=(10, 3))
        dD = np.array([0.1, 0.1, 0.1])
        ks, p = compute_kolmogorov_smirnov_distance(obs, pred, dD)
        assert np.all(p >= 0)
        assert np.all(p <= 1)


class TestComputeErrors:
    """Test suite for compute_errors."""

    @pytest.fixture
    def sample_data(self):
        """Provide common sample data for error tests."""
        obs = 10.0
        pred = np.array([10.0, 9.5, 10.5, 8.0, 12.0])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        return obs, pred, D, dD

    def test_error_scalar_obs(self, sample_data):
        """Error computation should handle scalar observations."""
        obs, pred, D, dD = sample_data
        errors = compute_errors(obs, pred, error_metric="MAE", D=D, dD=dD)
        assert errors.shape == pred.shape
        assert np.all(errors >= 0)

    def test_error_sse(self):
        """SSE error metric should compute sum of squared errors."""
        obs = np.array([1, 2, 3])
        pred = np.array([[1, 2, 3], [2, 3, 4]])
        errors = compute_errors(obs, pred, error_metric="SSE")
        expected = np.array([0, 3])
        np.testing.assert_array_almost_equal(errors, expected)

    def test_error_sae(self):
        """SAE error metric should compute sum of absolute errors."""
        obs = np.array([1, 2, 3])
        pred = np.array([[1, 2, 3], [2, 3, 4]])
        errors = compute_errors(obs, pred, error_metric="SAE")
        expected = np.array([0, 3])
        np.testing.assert_array_almost_equal(errors, expected)

    def test_error_mae(self):
        """MAE error metric should compute mean absolute error."""
        obs = np.array([1, 2, 3])
        pred = np.array([[1, 2, 3], [2, 3, 4]])
        errors = compute_errors(obs, pred, error_metric="MAE")
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(errors, expected)

    def test_error_mse(self):
        """MSE error metric should compute mean squared error."""
        obs = np.array([1, 2, 3])
        pred = np.array([[1, 2, 3], [2, 3, 4]])
        errors = compute_errors(obs, pred, error_metric="MSE")
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(errors, expected)

    def test_error_rmse(self):
        """RMSE error metric should compute root mean squared error."""
        obs = np.array([1, 2, 3])
        pred = np.array([[1, 2, 3], [2, 3, 4]])
        errors = compute_errors(obs, pred, error_metric="RMSE")
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(errors, expected)

    def test_error_relmae(self):
        """Relative MAE should handle zero division gracefully."""
        obs = np.array([1, 2, 3])
        pred = np.array([[1, 2, 3], [2, 3, 4]])
        errors = compute_errors(obs, pred, error_metric="relMAE")
        assert errors.shape == (2,)
        assert np.all(np.isfinite(errors))

    def test_error_kl_divergence(self):
        """KL divergence error metric should work with distribution data."""
        obs = np.array([100, 200, 150])
        pred = np.array([[100, 200, 150], [50, 100, 75], [1000, 2000, 3000]])
        dD = np.array([0.1, 0.1, 0.1])
        errors = compute_errors(obs, pred, error_metric="KL", dD=dD)
        assert errors.shape == (3,)
        assert np.all(errors >= 0)
        # First prediction is identical to obs, so error should be ~0
        np.testing.assert_allclose(errors[0], 0.0, atol=1e-10)

    def test_error_wasserstein_distance(self):
        """Wasserstein distance error metric should work."""
        obs = np.array([100, 200, 150])
        pred = np.array([[100, 200, 150], [50, 100, 75], [1000, 2000, 3000]])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        errors = compute_errors(obs, pred, error_metric="WD", D=D, dD=dD)
        assert errors.shape == (3,)
        assert np.all(errors >= 0)
        # First prediction is identical to obs, so error should be ~0
        np.testing.assert_allclose(errors[0], 0.0, atol=1e-10)

    def test_error_ks_statistic(self):
        """KS statistic error metric should work."""
        obs = np.array([100, 200, 150])
        pred = np.array([[100, 200, 150], [50, 100, 75], [1000, 2000, 3000]])
        dD = np.array([0.1, 0.1, 0.1])
        errors = compute_errors(obs, pred, error_metric="KS", dD=dD)
        assert errors.shape == (3,)
        assert np.all(errors >= 0)
        # First prediction is identical to obs, so KS should be ~0
        np.testing.assert_allclose(errors[0], 0.0, atol=1e-10)

    def test_error_ks_pvalue(self):
        """KS p-value error metric should work."""
        obs = np.array([100, 200, 150])
        pred = np.array([[100, 200, 150], [50, 100, 75], [1000, 2000, 3000]])
        dD = np.array([0.1, 0.1, 0.1])
        errors = compute_errors(obs, pred, error_metric="KS_pvalue", dD=dD)
        assert errors.shape == (3,)
        assert np.all(errors >= 0)
        assert np.all(errors <= 1)
        # First prediction is identical to obs, so p-value should be high (near 1)
        np.testing.assert_allclose(errors[0], 1.0, atol=1e-6)

    def test_error_js_distance(self):
        """JS distance error metric should work."""
        obs = np.array([100, 200, 150])
        pred = np.array([[100, 200, 150], [50, 100, 75], [1000, 2000, 3000]])
        D = np.array([0.5, 1.0, 1.5])
        dD = np.array([0.1, 0.1, 0.1])
        errors = compute_errors(obs, pred, error_metric="JS", D=D, dD=dD)
        assert errors.shape == (3,)
        assert np.all(errors >= 0)
        # First prediction is identical to obs, so JS distance should be ~0
        np.testing.assert_allclose(errors[0], 0.0, atol=1e-10)

    def test_error_invalid_metric_raises(self):
        """Invalid error metric should raise NotImplementedError."""
        obs = np.array([1, 2, 3])
        pred = np.array([[1, 2, 3]])
        with pytest.raises(NotImplementedError):
            compute_errors(obs, pred, error_metric="INVALID_METRIC")


class TestNormalizeErrors:
    """Test suite for normalize_errors."""

    def test_normalize_errors_disabled(self):
        """Normalization should return unchanged errors when disabled."""
        errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0])
        normalized = normalize_errors(errors, normalize_error=False)
        np.testing.assert_array_equal(normalized, errors)

    def test_normalize_errors_enabled(self):
        """Normalization should scale errors when enabled."""
        errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0])
        normalized = normalize_errors(errors, normalize_error=True)
        assert normalized.shape == errors.shape
        assert np.all(np.isfinite(normalized))
        assert not np.array_equal(normalized, errors)


class TestComputeCostFunction:
    """Test suite for compute_cost_function."""

    @pytest.fixture
    def sample_data(self):
        """Provide common sample data for cost function tests."""
        ND_obs = np.array([100, 200, 150, 50])
        ND_preds = np.array([[100, 200, 150, 50], [80, 180, 140, 60], [150, 250, 200, 30]])
        D = np.array([0.5, 1.0, 1.5, 2.0])
        dD = np.array([0.1, 0.1, 0.1, 0.1])
        V = np.array([1.0, 2.0, 3.0, 4.0])
        return ND_obs, ND_preds, D, dD, V

    def test_cost_function_returns_array(self, sample_data):
        """Cost function should return array with one error per prediction."""
        ND_obs, ND_preds, D, dD, V = sample_data
        errors = compute_cost_function(
            ND_obs,
            ND_preds,
            D,
            dD,
            V,
            target="N(D)",
            transformation="identity",
            error_metric="MAE",
            censoring="none",
        )
        assert errors.shape == (3,)
        # First prediction matches obs exactly, second and third differ
        expected_results = np.array([0.0, 15.0, 42.5])
        np.testing.assert_allclose(errors, expected_results, atol=1e-6)

    def test_cost_function_identical_prediction(self, sample_data):
        """Cost function should return ~0 error for identical prediction."""
        ND_obs, ND_preds, D, dD, V = sample_data
        errors = compute_cost_function(
            ND_obs,
            ND_preds[[0]],
            D,
            dD,
            V,  # First prediction matches obs
            target="N(D)",
            transformation="identity",
            error_metric="MAE",
            censoring="none",
        )
        np.testing.assert_allclose(errors[0], 0.0, atol=1e-10)

    def test_cost_function_all_distribution_targets(self, sample_data):
        """Cost function should work with all distribution target types."""
        ND_obs, ND_preds, D, dD, V = sample_data
        for target in DISTRIBUTION_TARGETS:
            errors = compute_cost_function(
                ND_obs,
                ND_preds,
                D,
                dD,
                V,
                target=target,
                transformation="identity",
                error_metric="MAE",
                censoring="none",
            )
            assert errors.shape == (3,)
            assert np.all(np.isfinite(errors))

    def test_cost_function_all_integral_targets(self, sample_data):
        """Cost function should work with all integral target types."""
        ND_obs, ND_preds, D, dD, V = sample_data
        for target in INTEGRAL_TARGETS:
            errors = compute_cost_function(
                ND_obs,
                ND_preds,
                D,
                dD,
                V,
                target=target,
                transformation="identity",
                censoring="none",
            )
            assert errors.shape == (3,)
            assert np.all(np.isfinite(errors))

    def test_cost_function_all_transformations(self, sample_data):
        """Cost function should work with all transformation types."""
        ND_obs, ND_preds, D, dD, V = sample_data
        for transformation in TRANSFORMATIONS:
            errors = compute_cost_function(
                ND_obs,
                ND_preds,
                D,
                dD,
                V,
                target="N(D)",
                transformation=transformation,
                error_metric="MAE",
                censoring="none",
            )
            assert errors.shape == (3,)

    def test_cost_function_all_error_metrics(self, sample_data):
        """Cost function should work with various error metrics."""
        ND_obs, ND_preds, D, dD, V = sample_data
        error_metrics = ERROR_METRICS
        for error_metric in error_metrics:
            errors = compute_cost_function(
                ND_obs,
                ND_preds,
                D,
                dD,
                V,
                target="N(D)",
                transformation="identity",
                error_metric=error_metric,
                censoring="none",
            )
            assert errors.shape == (3,)
            assert np.all(np.isfinite(errors) | np.isnan(errors))

    def test_cost_function_censoring_none(self, sample_data):
        """Cost function with no censoring should not truncate data."""
        ND_obs, ND_preds, D, dD, V = sample_data
        errors = compute_cost_function(
            ND_obs,
            ND_preds,
            D,
            dD,
            V,
            target="N(D)",
            transformation="identity",
            error_metric="MAE",
            censoring="none",
        )
        assert errors.shape == (3,)

    def test_cost_function_all_zeros_with_censoring_return_nan(self):
        """Cost function should return Inf if all-zero observations with censoring enabled."""
        ND_obs = np.zeros(5)
        ND_preds = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        D = np.arange(5) * 0.1 + 0.5
        dD = np.ones(5) * 0.1
        V = np.ones(5)
        errors = compute_cost_function(
            ND_obs,
            ND_preds,
            D,
            dD,
            V,
            target="N(D)",
            transformation="identity",
            error_metric="MAE",
            censoring="left",
        )
        assert np.all(np.isinf(errors))

    def test_cost_function_invalid_target_raises(self, sample_data):
        """Cost function should raise ValueError for invalid target."""
        ND_obs, ND_preds, D, dD, V = sample_data
        with pytest.raises(ValueError, match="Invalid 'target'"):
            compute_cost_function(
                ND_obs,
                ND_preds,
                D,
                dD,
                V,
                target="INVALID",
                transformation="identity",
                error_metric="MAE",
                censoring="none",
            )

    def test_cost_function_invalid_transformation_raises(self, sample_data):
        """Cost function should raise ValueError for invalid transformation."""
        ND_obs, ND_preds, D, dD, V = sample_data
        with pytest.raises(ValueError, match="Invalid 'transformation'"):
            compute_cost_function(
                ND_obs,
                ND_preds,
                D,
                dD,
                V,
                target="N(D)",
                transformation="INVALID",
                error_metric="MAE",
                censoring="none",
            )

    def test_cost_function_invalid_censoring_raises(self, sample_data):
        """Cost function should raise ValueError for invalid censoring."""
        ND_obs, ND_preds, D, dD, V = sample_data
        with pytest.raises(ValueError, match="Invalid 'censoring'"):
            compute_cost_function(
                ND_obs,
                ND_preds,
                D,
                dD,
                V,
                target="N(D)",
                transformation="identity",
                error_metric="MAE",
                censoring="INVALID",
            )

    def test_cost_function_invalid_error_metric_raises(self, sample_data):
        """Cost function should raise ValueError for invalid error_metric."""
        ND_obs, ND_preds, D, dD, V = sample_data
        with pytest.raises(ValueError, match="Invalid 'error_metric'"):
            compute_cost_function(
                ND_obs,
                ND_preds,
                D,
                dD,
                V,
                target="N(D)",
                transformation="identity",
                error_metric="INVALID",
                censoring="none",
            )

    def test_cost_function_different_error_metrics_produce_different_results(self, sample_data):
        """Different error metrics should produce different error values."""
        ND_obs, ND_preds, D, dD, V = sample_data
        errors_mae = compute_cost_function(
            ND_obs,
            ND_preds,
            D,
            dD,
            V,
            target="N(D)",
            transformation="identity",
            error_metric="MAE",
            censoring="none",
        )
        errors_mse = compute_cost_function(
            ND_obs,
            ND_preds,
            D,
            dD,
            V,
            target="N(D)",
            transformation="identity",
            error_metric="MSE",
            censoring="none",
        )

        # MAE and MSE should produce different values
        assert not np.allclose(errors_mae, errors_mse)
