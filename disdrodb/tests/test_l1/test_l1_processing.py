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
"""Test L1 processing function."""
import pytest
import xarray as xr

from disdrodb.l1.processing import generate_l1
from disdrodb.tests.fake_datasets import create_template_l0c_dataset


class TestGenerateL1:
    """Test suite for generate_l1 core function."""

    @pytest.mark.parametrize("with_velocity", [True, False])
    def test_returns_expected_variables(self, with_velocity):
        """Test generate_l1 returns Dataset with mandatory variables."""
        ds = create_template_l0c_dataset(with_velocity)
        ds_l1 = generate_l1(ds)

        if with_velocity:
            expected_vars = [
                "raw_drop_number",
                "qc_time",
                "qc_resampling",
                "precipitation_type",
                "hydrometeor_type",
                "n_particles",
                "n_graupel",
                "n_small_hail",
                "n_large_hail",
                "n_margin_fallers",
                "n_splashing",
                "flag_hail",
                "flag_graupel",
                "flag_noise",
                "flag_spikes",
                "flag_splashing",
                "flag_wind_artefacts",
            ]
        else:
            expected_vars = ["raw_drop_number", "qc_resampling", "qc_time"]

        for var in expected_vars:
            assert var in ds_l1, f"Missing variable {var}"

    def test_dask_allowed(self):
        """Test lazy dataset as input is allowed."""
        ds_lazy = create_template_l0c_dataset().chunk({"time": 1})
        ds_l1_lazy = generate_l1(ds_lazy)  # 0-10 dia
        for var in ds_l1_lazy.data_vars:
            if var in ds_l1_lazy:
                assert hasattr(ds_l1_lazy[var].data, "chunks")
        xr.testing.assert_allclose(ds_l1_lazy.compute(), generate_l1(create_template_l0c_dataset()))

    def test_preserves_global_attributes(self):
        """Global input attributes are preserved in output L1 dataset."""
        ds = create_template_l0c_dataset()
        ds.attrs["dummy"] = "dummy_value"
        assert ds.attrs["sensor_name"] == "PARSIVEL2"

        ds_l1 = generate_l1(ds)
        assert "dummy" in ds_l1.attrs
        assert ds_l1.attrs["sensor_name"] == "PARSIVEL2"
