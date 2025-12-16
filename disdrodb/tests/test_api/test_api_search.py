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
"""Test DISDRODB API search function."""

import os

import pytest

from disdrodb.api.search import (
    available_campaigns,
    available_data_sources,
    available_stations,
    get_required_product,
    list_campaign_names,
    list_data_sources,
    list_station_names,
    select_stations_matching_metadata_values,
)
from disdrodb.tests.conftest import create_fake_raw_data_file

# disdrodb_metadata_archive_dir = disdrodb.get_metadata_archive_dir()


def test_get_required_product():
    """Test get_required_product."""
    assert get_required_product("L0A") == "RAW"
    assert get_required_product("L0B") == "L0A"
    assert get_required_product("L0C") == "L0B"
    assert get_required_product("L1") == "L0C"
    assert get_required_product("L2E") == "L1"
    assert get_required_product("L2M") == "L2E"


class TestListDataSources:
    """Test list_data_sources."""

    def test_list_all_data_sources(self, disdrodb_metadata_archive_dir):
        """Test list_data_sources list all data sources."""
        data_sources = list_data_sources(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
        )
        assert isinstance(data_sources, list)
        assert isinstance(data_sources[0], str)
        assert "EPFL" in data_sources

    def test_list_data_sources_allow_filtering_by_data_sources(self, disdrodb_metadata_archive_dir):
        """Test list_data_sources list all data sources."""
        data_sources = list_data_sources(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_sources=["EPFL", "NETHERLANDS"],
        )
        assert isinstance(data_sources, list)
        assert isinstance(data_sources[0], str)
        assert "NETHERLANDS" in data_sources
        assert "EPFL" in data_sources
        assert len(data_sources) == 2

    def test_list_data_sources_raise_error_with_invalid_filter(self, disdrodb_metadata_archive_dir):
        """Test raise error if invalid data source filter."""
        with pytest.raises(ValueError):
            data_sources = list_data_sources(
                metadata_archive_dir=disdrodb_metadata_archive_dir,
                data_sources=["EPF", "NETHERLANDS"],
            )

        # Return only NETHERLANDS if invalid_fields_policy="ignore"
        data_sources = list_data_sources(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_sources=["EPF", "NETHERLANDS"],
            invalid_fields_policy="ignore",
        )
        assert data_sources == ["NETHERLANDS"]

    def test_list_data_sources_returns_empty_list_if_empty_metadata_archive(self):
        """Test it returns a empty list if empty metadata archive."""
        metadata_archive_dir = os.path.join("dummy", "DISDRODB")
        data_sources = list_data_sources(
            metadata_archive_dir=metadata_archive_dir,
        )
        assert data_sources == []


class TestListCampaignNames:
    """Test list_campaign_names."""

    def test_list_all_campaign_names(self, disdrodb_metadata_archive_dir):
        """Test list_campaign_names list all campaign names."""
        campaign_names = list_campaign_names(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
        )
        assert isinstance(campaign_names, list)
        assert isinstance(campaign_names[0], str)
        assert "RIETHOLZBACH_2011" in campaign_names

    def test_list_campaign_names_optionally_return_tuples(self, disdrodb_metadata_archive_dir):
        """Test list_campaign_names return list of (data_source, campaign_name) tuples."""
        list_tuples = list_campaign_names(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            return_tuple=True,
        )
        assert isinstance(list_tuples, list)
        assert isinstance(list_tuples[0], tuple)
        assert len(list_tuples[0]) == 2
        assert ("EPFL", "RIETHOLZBACH_2011") in list_tuples

    def test_list_campaign_names_allow_filtering_by_data_source(self, disdrodb_metadata_archive_dir):
        """Test list_campaign_names allow filtering by data sources."""
        campaign_names = list_campaign_names(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_sources=["EPFL"],
        )
        assert isinstance(campaign_names, list)
        assert isinstance(campaign_names[0], str)
        assert "RIETHOLZBACH_2011" in campaign_names
        assert "RADALP" not in campaign_names  # FRANCE data source
        assert len(campaign_names) > 1

    def test_list_campaign_names_allow_filtering_by_campaign_names(self, disdrodb_metadata_archive_dir):
        """Test list_campaign_names allow filtering by campaign names."""
        campaign_names = list_campaign_names(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            campaign_names=["RIETHOLZBACH_2011"],
        )
        assert isinstance(campaign_names, list)
        assert isinstance(campaign_names[0], str)
        assert "RIETHOLZBACH_2011" in campaign_names
        assert len(campaign_names) == 1

    def test_list_campaign_names_raise_error_with_invalid_filter(self, disdrodb_metadata_archive_dir):
        """Test raise error if invalid campaign name filter."""
        with pytest.raises(ValueError):
            campaign_names = list_campaign_names(
                metadata_archive_dir=disdrodb_metadata_archive_dir,
                campaign_names=["BAD_NAME", "RIETHOLZBACH_2011"],
            )

        # Return only RIETHOLZBACH_2011 if invalid_fields_policy="ignore"
        campaign_names = list_campaign_names(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            campaign_names=["BAD_NAME", "RIETHOLZBACH_2011"],
            invalid_fields_policy="ignore",
        )
        assert campaign_names == ["RIETHOLZBACH_2011"]

    def test_list_campaign_names_returns_empty_list_if_empty_metadata_archive(self):
        """Test it returns a empty list if empty metadata archive."""
        metadata_archive_dir = os.path.join("dummy", "DISDRODB")
        campaign_names = list_campaign_names(
            metadata_archive_dir=metadata_archive_dir,
        )
        assert campaign_names == []

        campaign_names = list_campaign_names(
            metadata_archive_dir=metadata_archive_dir,
            return_tuple=True,
        )
        assert campaign_names == []


class TestListStationNames:
    """Test list_station_names."""

    def test_list_all_station_names(self, disdrodb_metadata_archive_dir):
        """Test list_station_names list all campaign names."""
        station_names = list_station_names(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
        )
        assert isinstance(station_names, list)
        assert isinstance(station_names[0], str)
        assert "PAR001_Cabauw" in station_names

    def test_list_station_names_optionally_return_tuples(self, disdrodb_metadata_archive_dir):
        """Test list_station_names return list of (data_source, campaign_name, station_name) tuples."""
        list_tuples = list_station_names(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            return_tuple=True,
        )
        assert isinstance(list_tuples, list)
        assert isinstance(list_tuples[0], tuple)
        assert len(list_tuples[0]) == 3
        assert ("EPFL", "RIETHOLZBACH_2011", "60") in list_tuples

    def test_list_station_names_allow_filtering_by_data_source(self, disdrodb_metadata_archive_dir):
        """Test list_station_names allow filtering by data sources."""
        station_names = list_station_names(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_sources=["NETHERLANDS"],
        )
        assert isinstance(station_names, list)
        assert isinstance(station_names[0], str)
        assert "PAR001_Cabauw" in station_names
        assert "Carnot_PWS" not in station_names  # FRANCE data source
        assert len(station_names) > 1

    def test_list_station_names_allow_filtering_by_campaign_names(self, disdrodb_metadata_archive_dir):
        """Test list_station_names allow filtering by campaign names."""
        station_names = list_station_names(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            campaign_names=["DELFT"],
        )
        assert isinstance(station_names, list)
        assert isinstance(station_names[0], str)
        assert "PAR001_Cabauw" in station_names
        assert "THIES001_Cabauw" in station_names
        assert "Carnot_PWS" not in station_names
        assert len(station_names) > 1

    def test_list_station_names_allow_filtering_by_station_names(self, disdrodb_metadata_archive_dir):
        """Test list_station_names allow filtering by station names."""
        station_names = list_station_names(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            station_names=["PAR001_Cabauw", "PAR002_Cabauw"],
        )
        assert isinstance(station_names, list)
        assert isinstance(station_names[0], str)
        assert "PAR001_Cabauw" in station_names
        assert "THIES001_Cabauw" not in station_names
        assert len(station_names) == 2

    def test_list_station_names_raise_error_with_invalid_filter(self, disdrodb_metadata_archive_dir):
        """Test raise error if invalid station name filter."""
        with pytest.raises(ValueError):
            station_names = list_station_names(
                metadata_archive_dir=disdrodb_metadata_archive_dir,
                station_names=["BAD_NAME", "PAR001_Cabauw"],
            )

        # Return only PAR001_Cabauw if invalid_fields_policy="ignore"
        station_names = list_station_names(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            station_names=["BAD_NAME", "PAR001_Cabauw"],
            invalid_fields_policy="ignore",
        )
        assert station_names == ["PAR001_Cabauw"]

    def test_list_station_names_returns_empty_list_if_empty_metadata_archive(self):
        """Test it returns a empty list if empty metadata archive."""
        metadata_archive_dir = os.path.join("dummy", "DISDRODB")
        station_names = list_station_names(
            metadata_archive_dir=metadata_archive_dir,
        )
        assert station_names == []

        station_names = list_station_names(
            metadata_archive_dir=metadata_archive_dir,
            return_tuple=True,
        )
        assert station_names == []


class TestSelectStationsMatchingMetadataValue:
    def test_scalar_equal(self, monkeypatch):
        """It should match when metadata value equals expected scalar."""
        monkeypatch.setattr("disdrodb.api.search.define_metadata_filepath", lambda *args, **kwargs: "fakepath")
        monkeypatch.setattr("disdrodb.api.search.read_yaml", lambda _path: {"key": "A"})
        metadata_archive_dir = os.path.join("DISDRODB-METADATA", "DISDRODB")
        list_info = [("DS", "CAMP", "ST1")]

        result = select_stations_matching_metadata_values(metadata_archive_dir, list_info, {"key": "A"})
        assert result == list_info

    def test_scalar_not_equal(self, monkeypatch):
        """It should not match when metadata value does not equal expected scalar."""
        monkeypatch.setattr("disdrodb.api.search.define_metadata_filepath", lambda *args, **kwargs: "fakepath")
        monkeypatch.setattr("disdrodb.api.search.read_yaml", lambda _path: {"key": "B"})
        metadata_archive_dir = os.path.join("DISDRODB-METADATA", "DISDRODB")
        list_info = [("DS", "CAMP", "ST1")]

        result = select_stations_matching_metadata_values(metadata_archive_dir, list_info, {"key": "A"})
        assert result == []

    def test_metadata_list_scalar_value(self, monkeypatch):
        """It should match when scalar is in metadata list."""
        monkeypatch.setattr("disdrodb.api.search.define_metadata_filepath", lambda *args, **kwargs: "fakepath")
        monkeypatch.setattr("disdrodb.api.search.read_yaml", lambda _path: {"key": ["A", "B"]})
        metadata_archive_dir = os.path.join("DISDRODB-METADATA", "DISDRODB")
        list_info = [("DS", "CAMP", "ST1")]

        result = select_stations_matching_metadata_values(metadata_archive_dir, list_info, {"key": "A"})
        assert result == list_info

    def test_metadata_scalar_value_list(self, monkeypatch):
        """It should match when metadata scalar is in expected list."""
        monkeypatch.setattr("disdrodb.api.search.define_metadata_filepath", lambda *args, **kwargs: "fakepath")
        monkeypatch.setattr("disdrodb.api.search.read_yaml", lambda _path: {"key": "A"})
        metadata_archive_dir = os.path.join("DISDRODB-METADATA", "DISDRODB")
        list_info = [("DS", "CAMP", "ST1")]

        result = select_stations_matching_metadata_values(metadata_archive_dir, list_info, {"key": ["A", "C"]})
        assert result == list_info

    def test_both_lists_overlap(self, monkeypatch):
        """It should match when metadata list and expected list overlap."""
        monkeypatch.setattr("disdrodb.api.search.define_metadata_filepath", lambda *args, **kwargs: "fakepath")
        monkeypatch.setattr("disdrodb.api.search.read_yaml", lambda _path: {"key": ["A", "B"]})
        metadata_archive_dir = os.path.join("DISDRODB-METADATA", "DISDRODB")
        list_info = [("DS", "CAMP", "ST1")]

        result = select_stations_matching_metadata_values(metadata_archive_dir, list_info, {"key": ["C", "B"]})
        assert result == list_info

    def test_both_lists_no_overlap(self, monkeypatch):
        """It should not match when metadata list and expected list have no overlap."""
        monkeypatch.setattr("disdrodb.api.search.define_metadata_filepath", lambda *args, **kwargs: "fakepath")
        monkeypatch.setattr("disdrodb.api.search.read_yaml", lambda _path: {"key": ["A", "B"]})
        metadata_archive_dir = os.path.join("DISDRODB-METADATA", "DISDRODB")
        list_info = [("DS", "CAMP", "ST1")]

        result = select_stations_matching_metadata_values(metadata_archive_dir, list_info, {"key": ["C", "D"]})
        assert result == []


class TestAvailableStations:
    """Test available_stations."""

    def test_available_stations_with_empty_metadata_archive(self):
        """Test it returns an empty list if empty DISDRODB Metadata Archive."""
        metadata_archive_dir = os.path.join("dummy", "DISDRODB")
        station_names = available_stations(
            metadata_archive_dir=metadata_archive_dir,
            return_tuple=False,
        )
        assert station_names == []

        station_names = available_stations(
            metadata_archive_dir=metadata_archive_dir,
            return_tuple=True,
        )
        assert station_names == []

        stations = available_stations(
            metadata_archive_dir=metadata_archive_dir,
            available_data=True,
        )
        assert stations == []

    def test_available_stations_with_empty_data_archive(self, disdrodb_metadata_archive_dir):
        """Test it returns an empty list if empty DISDRODB Data Archive."""
        data_archive_dir = os.path.join("dummy", "DISDRODB")
        station_names = available_stations(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=False,
            return_tuple=False,
        )
        assert station_names == []

        station_names = available_stations(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=True,
            return_tuple=False,
        )
        assert station_names == []

        data_archive_dir = os.path.join("dummy", "DISDRODB")
        station_names = available_stations(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=False,
            return_tuple=True,
        )
        assert station_names == []

        station_names = available_stations(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=True,
            return_tuple=True,
        )
        assert station_names == []

    def test_available_stations_raise_error_with_empty_metadata_archive(self):
        """Test raises an error if empty DISDRODB Metadata Archive."""
        metadata_archive_dir = os.path.join("dummy", "DISDRODB")

        # No station available in the DISDRODB Metadata Archive.
        with pytest.raises(ValueError):
            available_stations(
                metadata_archive_dir=metadata_archive_dir,
                raise_error_if_empty=True,
            )

        # No station with disdrodb_data_url specified
        with pytest.raises(ValueError):
            available_stations(
                metadata_archive_dir=metadata_archive_dir,
                available_data=True,
                raise_error_if_empty=True,
            )

    def test_available_stations_raise_error_with_empty_data_archive(self, disdrodb_metadata_archive_dir):
        """Test raises an error if empty DISDRODB Data Archive."""
        data_archive_dir = os.path.join("dummy", "DISDRODB")

        with pytest.raises(ValueError):
            available_stations(
                metadata_archive_dir=disdrodb_metadata_archive_dir,
                data_archive_dir=data_archive_dir,
                product="RAW",
                available_data=False,
                raise_error_if_empty=True,
            )

        with pytest.raises(ValueError):
            available_stations(
                metadata_archive_dir=disdrodb_metadata_archive_dir,
                data_archive_dir=data_archive_dir,
                product="RAW",
                available_data=True,
                raise_error_if_empty=True,
            )

    def test_available_stations_for_download(self, disdrodb_metadata_archive_dir):
        """Test returns a list with stations with downloadable data."""
        stations = available_stations(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            available_data=True,
            raise_error_if_empty=True,
        )

        assert isinstance(stations, list)
        assert isinstance(stations[0], tuple)
        assert len(stations[0]) == 3
        assert ("EPFL", "PARADISO_2014", "10") in stations

    def test_available_stations_within_data_archive(self, tmp_path, disdrodb_metadata_archive_dir):
        """Test returns a list with stations with product directory/data in the DISDRODB data archive."""
        data_archive_dir = os.path.join(tmp_path, "DISDRODB")
        data_source = "EPFL"
        campaign_name = "PARADISO_2014"
        station_name = "10"

        stations = available_stations(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=False,
            raise_error_if_empty=False,
        )
        assert stations == []

        # Create raw data file
        raw_file_filepath = create_fake_raw_data_file(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )

        # Test it lists such station also if only product directory
        stations = available_stations(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=True,
            raise_error_if_empty=False,
        )
        assert (data_source, campaign_name, station_name) in stations
        assert len(stations) == 1

        # Now remove file
        os.remove(raw_file_filepath)
        assert not os.path.exists(raw_file_filepath)

        # Test it lists such station also if only product directory
        stations = available_stations(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=False,
            raise_error_if_empty=False,
        )
        assert (data_source, campaign_name, station_name) in stations
        assert len(stations) == 1

        # Test it does not lists such station if available_data=True
        stations = available_stations(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=True,
            raise_error_if_empty=False,
        )
        assert stations == []
        assert len(stations) == 0


class TestAvailableCampaigns:
    """Test available_campaigns."""

    def test_available_campaigns_with_empty_metadata_archive(self):
        """Test it returns an empty list if empty DISDRODB Metadata Archive."""
        metadata_archive_dir = os.path.join("dummy", "DISDRODB")
        campaign_names = available_campaigns(
            metadata_archive_dir=metadata_archive_dir,
        )
        assert campaign_names == []

        campaigns = available_campaigns(
            metadata_archive_dir=metadata_archive_dir,
            available_data=True,
        )
        assert campaigns == []

    def test_available_campaigns_with_empty_data_archive(self, disdrodb_metadata_archive_dir):
        """Test it returns an empty list if empty DISDRODB Data Archive."""
        data_archive_dir = os.path.join("dummy", "DISDRODB")
        campaign_names = available_campaigns(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=False,
        )
        assert campaign_names == []

        campaign_names = available_campaigns(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=True,
        )
        assert campaign_names == []

    def test_available_campaigns_raise_error_with_empty_metadata_archive(self):
        """Test it raise an error if empty DISDRODB Metadata Archive."""
        metadata_archive_dir = os.path.join("dummy", "DISDRODB")

        # No station available in the DISDRODB Metadata Archive.
        with pytest.raises(ValueError):
            available_campaigns(
                metadata_archive_dir=metadata_archive_dir,
                raise_error_if_empty=True,
            )

        # No station with disdrodb_data_url specified
        with pytest.raises(ValueError):
            available_campaigns(
                metadata_archive_dir=metadata_archive_dir,
                available_data=True,
                raise_error_if_empty=True,
            )

    def test_available_campaigns_raise_error_with_empty_data_archive(self, disdrodb_metadata_archive_dir):
        """Test it returns an empty list if empty DISDRODB Data Archive."""
        data_archive_dir = os.path.join("dummy", "DISDRODB")

        with pytest.raises(ValueError):
            available_campaigns(
                metadata_archive_dir=disdrodb_metadata_archive_dir,
                data_archive_dir=data_archive_dir,
                product="RAW",
                available_data=False,
                raise_error_if_empty=True,
            )

        with pytest.raises(ValueError):
            available_campaigns(
                metadata_archive_dir=disdrodb_metadata_archive_dir,
                data_archive_dir=data_archive_dir,
                product="RAW",
                available_data=True,
                raise_error_if_empty=True,
            )

    def test_available_campaigns_for_download(self, disdrodb_metadata_archive_dir):
        """Test it returns a list of campaigns with downloadable data."""
        campaigns = available_campaigns(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            available_data=True,
            raise_error_if_empty=True,
        )

        assert isinstance(campaigns, list)
        assert isinstance(campaigns[0], str)
        assert "PARADISO_2014" in campaigns

    def test_available_campaigns_within_data_archive(self, tmp_path, disdrodb_metadata_archive_dir):
        """Test it returns a list with campaigns with product directory in DISDRODB data archive."""
        data_archive_dir = os.path.join(tmp_path, "DISDRODB")
        data_source = "EPFL"
        campaign_name = "PARADISO_2014"
        station_name = "10"

        campaigns = available_campaigns(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=False,
            raise_error_if_empty=False,
        )
        assert campaigns == []

        # Create raw data file
        raw_file_filepath = create_fake_raw_data_file(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )

        # Test it lists such station also if only product directory
        campaigns = available_campaigns(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=True,
            raise_error_if_empty=False,
        )
        assert campaign_name in campaigns
        assert len(campaigns) == 1

        # Now remove file
        os.remove(raw_file_filepath)
        assert not os.path.exists(raw_file_filepath)

        # Test it lists such station also if only product directory
        campaigns = available_campaigns(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=False,
            raise_error_if_empty=False,
        )
        assert campaign_name in campaigns
        assert len(campaigns) == 1

        # Test it does not lists such station if available_data=True
        campaigns = available_campaigns(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=True,
            raise_error_if_empty=False,
        )
        assert campaigns == []
        assert len(campaigns) == 0


class TestAvailableDataSources:
    """Test available_data_sources."""

    def test_available_data_sources_with_empty_metadata_archive(self):
        """Test it returns an empty list if empty DISDRODB Metadata Archive."""
        metadata_archive_dir = os.path.join("dummy", "DISDRODB")
        data_sources = available_data_sources(
            metadata_archive_dir=metadata_archive_dir,
        )
        assert data_sources == []

        data_sources = available_data_sources(
            metadata_archive_dir=metadata_archive_dir,
            available_data=True,
        )
        assert data_sources == []

    def test_available_data_sources_with_empty_data_archive(self, disdrodb_metadata_archive_dir):
        """Test it returns an empty list if empty DISDRODB Data Archive."""
        data_archive_dir = os.path.join("dummy", "DISDRODB")
        data_sources = available_data_sources(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=False,
        )
        assert data_sources == []

        data_sources = available_data_sources(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=True,
        )
        assert data_sources == []

    def test_available_data_sources_raise_error_with_empty_metadata_archive(self):
        """Test it raise an error if empty DISDRODB Metadata Archive."""
        metadata_archive_dir = os.path.join("dummy", "DISDRODB")

        # No station available in the DISDRODB Metadata Archive.
        with pytest.raises(ValueError):
            available_data_sources(
                metadata_archive_dir=metadata_archive_dir,
                raise_error_if_empty=True,
            )

        # No station with disdrodb_data_url specified
        with pytest.raises(ValueError):
            available_data_sources(
                metadata_archive_dir=metadata_archive_dir,
                available_data=True,
                raise_error_if_empty=True,
            )

    def test_available_data_sources_raise_error_with_empty_data_archive(self, disdrodb_metadata_archive_dir):
        """Test it returns an empty list if empty DISDRODB Data Archive."""
        data_archive_dir = os.path.join("dummy", "DISDRODB")

        with pytest.raises(ValueError):
            available_data_sources(
                metadata_archive_dir=disdrodb_metadata_archive_dir,
                data_archive_dir=data_archive_dir,
                product="RAW",
                available_data=False,
                raise_error_if_empty=True,
            )

        with pytest.raises(ValueError):
            available_data_sources(
                metadata_archive_dir=disdrodb_metadata_archive_dir,
                data_archive_dir=data_archive_dir,
                product="RAW",
                available_data=True,
                raise_error_if_empty=True,
            )

    def test_available_data_sources_for_download(self, disdrodb_metadata_archive_dir):
        """Test it returns a list of data_sources with downloadable data."""
        data_sources = available_data_sources(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            available_data=True,
            raise_error_if_empty=True,
        )

        assert isinstance(data_sources, list)
        assert isinstance(data_sources[0], str)
        assert "EPFL" in data_sources

    def test_available_data_sources_within_data_archive(self, tmp_path, disdrodb_metadata_archive_dir):
        """Test it returns a list with data_sources with product directory in DISDRODB data archive."""
        data_archive_dir = os.path.join(tmp_path, "DISDRODB")
        data_source = "EPFL"
        campaign_name = "PARADISO_2014"
        station_name = "10"

        data_sources = available_data_sources(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=False,
            raise_error_if_empty=False,
        )
        assert data_sources == []

        # Create raw data file
        raw_file_filepath = create_fake_raw_data_file(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )

        # Test it lists such station also if only product directory
        data_sources = available_data_sources(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=True,
            raise_error_if_empty=False,
        )
        assert data_source in data_sources
        assert len(data_sources) == 1

        # Now remove file
        os.remove(raw_file_filepath)
        assert not os.path.exists(raw_file_filepath)

        # Test it lists such station also if only product directory
        data_sources = available_data_sources(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=False,
            raise_error_if_empty=False,
        )
        assert data_source in data_sources
        assert len(data_sources) == 1

        # Test it does not lists such station if available_data=True
        data_sources = available_data_sources(
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            product="RAW",
            available_data=True,
            raise_error_if_empty=False,
        )
        assert data_sources == []
        assert len(data_sources) == 0
