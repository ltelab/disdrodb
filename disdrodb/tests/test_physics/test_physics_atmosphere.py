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
"""Testing atmospheric physics functions."""
import pytest
from disdrodb.physics.atmosphere import (
    get_air_density,
    get_air_dynamic_viscosity,
    get_air_pressure_at_height,
    get_air_temperature_at_height,
    get_gravitational_acceleration,
    get_vapor_actual_pressure,
    get_vapor_actual_pressure_at_height,
    get_vapor_saturation_pressure,
)


class TestGravitationalAcceleration:
    """Test gravitational acceleration calculations."""

    def test_get_gravitational_acceleration_sea_level(self):
        """Test gravitational acceleration at sea level for 45° latitude."""
        g = get_gravitational_acceleration(latitude=45.0, altitude=0)
        assert pytest.approx(g, abs=1e-3) == 9.806

    def test_get_gravitational_acceleration_altitude(self):
        """Test gravitational acceleration decreases with altitude."""
        g = get_gravitational_acceleration(latitude=45.0, altitude=1000)
        assert pytest.approx(g, abs=1e-3) == 9.803
        assert g < get_gravitational_acceleration(latitude=45.0, altitude=0)

    def test_extreme_latitude(self):
        """Test gravitational acceleration at extreme latitudes."""
        g_north = get_gravitational_acceleration(latitude=90, altitude=0)
        g_equator = get_gravitational_acceleration(latitude=0, altitude=0)
        assert g_north > g_equator
        assert pytest.approx(g_north, abs=1e-3) == 9.832
        assert pytest.approx(g_equator, abs=1e-3) == 9.780


class TestAirPressure:
    """Test air pressure calculations."""

    def test_get_air_pressure_at_height_sea_level(self):
        """Test air pressure at sea level returns standard pressure."""
        pressure = get_air_pressure_at_height(
            altitude=0,
            latitude=45.0,
            temperature=288.15,
        )
        assert pytest.approx(pressure, abs=0.1) == 101325.0

    def test_get_air_pressure_at_height_altitude(self):
        """Test air pressure decreases with altitude."""
        pressure = get_air_pressure_at_height(
            altitude=1000,
            latitude=45.0,
            temperature=281.5,
        )
        assert pressure < 101325
        assert pytest.approx(pressure, abs=0.1) == 89872.0


class TestTemperature:
    """Test temperature calculations."""

    def test_get_air_temperature_at_height_sea_level(self):
        """Test air temperature at sea level remains unchanged."""
        temp = get_air_temperature_at_height(altitude=0, sea_level_temperature=288.15)
        assert temp == 288.15

    def test_get_air_temperature_at_height_altitude(self):
        """Test air temperature decreases with altitude at standard lapse rate."""
        temp = get_air_temperature_at_height(altitude=1000, sea_level_temperature=288.15)
        assert pytest.approx(temp, abs=0.1) == 281.65


class TestVaporPressure:
    """Test vapor pressure calculations."""

    def test_get_vapor_saturation_pressure_freezing(self):
        """Test saturation vapor pressure at freezing point."""
        esat = get_vapor_saturation_pressure(273.15)
        assert pytest.approx(esat, abs=0.1) == 611.2

    def test_get_vapor_saturation_pressure_room_temp(self):
        """Test saturation vapor pressure at room temperature."""
        esat = get_vapor_saturation_pressure(293.15)
        assert pytest.approx(esat, abs=0.1) == 2338.5

    def test_get_vapor_actual_pressure(self):
        """Test actual vapor pressure calculation."""
        vapor_pressure = get_vapor_actual_pressure(
            relative_humidity=0.7,
            temperature=293.15,
        )
        assert pytest.approx(vapor_pressure, abs=0.1) == 1636.9

    def test_get_vapor_actual_pressure_at_height(self):
        """Test vapor pressure at altitude."""
        vapor_pressure = get_vapor_actual_pressure_at_height(
            altitude=1000,
            sea_level_temperature=288.15,
            sea_level_relative_humidity=0.7,
        )
        assert vapor_pressure > 0
        assert pytest.approx(vapor_pressure, abs=1) == 706.020


class TestAirProperties:
    """Test air property calculations."""

    def test_get_air_dynamic_viscosity_room_temp(self):
        """Test air dynamic viscosity at room temperature."""
        viscosity = get_air_dynamic_viscosity(293.15)
        assert pytest.approx(viscosity, abs=1e-6) == 1.82e-5

    def test_get_air_dynamic_viscosity_freezing(self):
        """Test air dynamic viscosity at freezing point."""
        viscosity = get_air_dynamic_viscosity(273.15)
        assert pytest.approx(viscosity, abs=1e-6) == 1.72e-5

    def test_get_air_density(self):
        """Test air density calculation for moist air."""
        density = get_air_density(
            temperature=293.15,
            air_pressure=101325,
            vapor_pressure=1637,
        )
        assert pytest.approx(density, abs=0.01) == 1.20

 