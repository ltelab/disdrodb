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
"""Testing thermodynamic, atmospheric physics and hydrodynamic functions."""
import pytest

from disdrodb.l1.beard_model import (
    get_air_density,
    get_air_dynamic_viscosity,
    get_air_pressure_at_height,
    get_air_temperature_at_height,
    get_drag_coefficient,
    get_fall_velocity_beard_1976,
    get_gravitational_acceleration,
    get_pure_water_compressibility,
    get_pure_water_density,
    get_pure_water_surface_tension,
    get_raindrop_reynolds_number,
    get_vapor_actual_pressure,
    get_vapor_actual_pressure_at_height,
    get_vapor_saturation_pressure,
    get_water_density,
    retrieve_fall_velocity,
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


class TestReynoldsNumber:
    """Test Reynolds number calculations."""

    def test_get_raindrop_reynolds_number_small_drop(self):
        """Test Reynolds number for small raindrop (< 1mm)."""
        reynolds = get_raindrop_reynolds_number(
            diameter=0.0005,  # 0.5 mm
            temperature=293.15,
            air_density=1.2,
            water_density=998,
            g=9.81,
        )
        assert reynolds > 0
        assert pytest.approx(reynolds, abs=0.01) == 66.57

    def test_get_raindrop_reynolds_number_large_drop(self):
        """Test Reynolds number for large raindrop (> 1mm)."""
        reynolds = get_raindrop_reynolds_number(
            diameter=0.003,  # 3 mm
            temperature=293.15,
            air_density=1.2,
            water_density=998,
            g=9.81,
        )
        assert pytest.approx(reynolds, abs=0.01) == 1595.716


class TestDragCoefficient:
    """Test drag coefficient calculations."""

    def test_get_drag_coefficient(self):
        """Test drag coefficient calculation for falling raindrop."""
        drag_coeff = get_drag_coefficient(
            diameter=0.002,
            air_density=1.2,
            water_density=998,
            fall_velocity=6.0,
            g=9.81,
        )
        assert drag_coeff > 0
        assert pytest.approx(drag_coeff, abs=0.01) == 0.603


class TestBeardFallVelocity:
    """Test Beard fall velocity model calculations."""

    def test_get_fall_velocity_beard_1976(self):
        """Test Beard 1976 fall velocity model."""
        fall_velocity = get_fall_velocity_beard_1976(
            diameter=0.002,  # 2 mm
            temperature=293.15,
            air_density=1.2,
            water_density=998,
            g=9.81,
        )
        assert fall_velocity > 0
        assert pytest.approx(fall_velocity, abs=0.001) == 6.516

    def test_get_fall_velocity_beard_1976_with_very_small_diameter(self):
        """Test fall velocity for very small droplets."""
        fall_velocity = get_fall_velocity_beard_1976(
            diameter=1e-5,  # 10 micrometers
            temperature=293.15,
            air_density=1.2,
            water_density=998,
            g=9.81,
        )
        assert fall_velocity > 0
        assert pytest.approx(fall_velocity, abs=0.001) == 0.002

    def test_retrieve_fall_velocity_complete(self):
        """Test complete fall velocity retrieval with all atmospheric parameters."""
        fall_velocity = retrieve_fall_velocity(
            diameter=0.002,
            altitude=500,
            latitude=45.0,
            temperature=288.15,
            relative_humidity=0.7,
        )
        assert fall_velocity > 0
        assert pytest.approx(fall_velocity, abs=0.001) == 6.656

    def test_retrieve_fall_velocity_with_pressure(self):
        """Test fall velocity retrieval with specified air pressure."""
        fall_velocity = retrieve_fall_velocity(
            diameter=0.002,
            altitude=500,
            latitude=45.0,
            temperature=288.15,
            relative_humidity=0.7,
            air_pressure=101325,
        )
        assert fall_velocity > 0
        assert pytest.approx(fall_velocity, abs=0.001) == 6.5006
