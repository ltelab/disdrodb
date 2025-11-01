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
"""Testing water physics functions."""
import pytest

from disdrodb.physics.water import (
    get_pure_water_compressibility,
    get_pure_water_density,
    get_pure_water_surface_tension,
    get_water_density,
)


class TestWaterProperties:
    """Test water property calculations."""

    def test_get_pure_water_density_room_temp(self):
        """Test pure water density at room temperature."""
        density = get_pure_water_density(293.15)
        assert pytest.approx(density, abs=0.01) == 998.20

    def test_get_pure_water_density_freezing(self):
        """Test pure water density at freezing point."""
        density = get_pure_water_density(273.15)
        assert pytest.approx(density, abs=0.01) == 999.84

    def test_get_pure_water_compressibility(self):
        """Test pure water compressibility at room temperature."""
        compressibility = get_pure_water_compressibility(293.15)
        assert compressibility > 0
        assert pytest.approx(compressibility, abs=1e-11) == 4.6e-10

    def test_get_pure_water_surface_tension(self):
        """Test pure water surface tension at room temperature."""
        surface_tension = get_pure_water_surface_tension(293.15)
        assert pytest.approx(surface_tension, abs=0.001) == 0.073

    def test_get_water_density(self):
        """Test water density accounting for pressure effects."""
        density = get_water_density(
            temperature=293.15,
            air_pressure=89874,
            sea_level_air_pressure=101325,
        )
        assert density > 998
        assert density < 1000
