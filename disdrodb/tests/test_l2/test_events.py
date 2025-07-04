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
"""Testing events utilities."""
import numpy as np
from disdrodb.l2.routines import identify_time_partitions 
from disdrodb.l2.event import ( 
    group_timesteps_into_events, 
    remove_isolated_timesteps, 
    group_timesteps_into_event,
    get_events_info,
)


def generate_product_filename(start_time, end_time):
    """Helper to generate DISDRODB products filenames given a numpy.datetime64 start_time/end_time."""
    s = np.datetime_as_string(start_time, unit='s').replace('-', '').replace('T', '').replace(":", "")
    e = np.datetime_as_string(end_time,   unit='s').replace('-', '').replace('T', '').replace(":", "")
    return f"L1.1MIN.campaign.station.s{s}.e{e}.V0.nc"
    # 'L1.1MIN.UL.Ljubljana.s20180601120000.e20180701120000.V0.nc',

class TestGroupTimestepsIntoEvent:
    def test_basic_grouping_defaults(self):
        """Default parameters group all close timesteps into one event."""
        timesteps = np.array([
            '2022-01-01T00:00:00',
            '2022-01-01T00:30:00',
            '2022-01-01T01:00:00'
        ], dtype='datetime64[s]')
        # default intra gap=0S so each timestep is its own event, but default neighbor_min_size=0 so no isolation
        events = group_timesteps_into_event(
            timesteps,
            intra_event_max_time_gap="1H"
        )
        assert isinstance(events, list)
        # all three fall into one event since gap <=1H
        assert len(events) == 1
        ev = events[0]
        assert ev["start_time"] == np.datetime64('2022-01-01T00:00:00')
        assert ev["end_time"]   == np.datetime64('2022-01-01T01:00:00')
        assert ev["n_timesteps"] == 3
        assert ev["duration"] == np.timedelta64(60, 'm')

    def test_isolation_removal(self):
        """Isolated timesteps are removed before grouping."""
        timesteps = np.array([
            '2022-01-01T00:00:00',
            '2022-01-01T00:03:00',
            '2022-01-01T02:00:00'
        ], dtype='datetime64[s]')
        events = group_timesteps_into_event(
            timesteps,
            neighbor_min_size=1,
            neighbor_time_interval="5MIN",
            intra_event_max_time_gap="1H"
        )
        # the 02:00 point is isolated (no neighbor within 5min) and removed,
        # so only one event with the first two timesteps
        assert len(events) == 1
        ev = events[0]
        assert ev["start_time"] == np.datetime64('2022-01-01T00:00:00')
        assert ev["end_time"]   == np.datetime64('2022-01-01T00:03:00')
        assert ev["n_timesteps"] == 2

    def test_event_min_size_filter(self):
        """Events smaller than event_min_size are discarded."""
        timesteps = np.array([
            '2022-01-01T00:00:00',
            '2022-01-01T03:00:00'
        ], dtype='datetime64[s]')
        # two separate events of size 1; event_min_size=2 should filter both
        events = group_timesteps_into_event(
            timesteps,
            intra_event_max_time_gap="30MIN",
            event_min_size=2
        )
        assert events == []

    def test_event_min_duration_filter(self):
        """Events shorter than event_min_duration are discarded."""
        timesteps = np.array([
            '2022-01-01T00:00:00',
            '2022-01-01T01:00:00'
        ], dtype='datetime64[s]')
        # duration is 60m; event_min_duration=120MIN should filter it out
        events = group_timesteps_into_event(
            timesteps,
            intra_event_max_time_gap="2H",
            event_min_duration="120MIN"
        )
        assert events == []

    def test_multiple_events_split_by_gap(self):
        """Large gaps create multiple separate events."""
        timesteps = np.array([
            '2022-01-01T00:00:00',
            '2022-01-01T00:30:00',
            '2022-01-01T03:00:00',
            '2022-01-01T03:30:00'
        ], dtype='datetime64[s]')
        events = group_timesteps_into_event(
            timesteps,
            intra_event_max_time_gap="1H"
        )
        # Should split into two events: first two and last two
        assert len(events) == 2
        ev1, ev2 = events
        assert ev1["start_time"] == np.datetime64('2022-01-01T00:00:00')
        assert ev1["end_time"]   == np.datetime64('2022-01-01T00:30:00')
        assert ev2["start_time"] == np.datetime64('2022-01-01T03:00:00')
        assert ev2["end_time"]   == np.datetime64('2022-01-01T03:30:00')
        
        
class TestRemoveIsolatedTimesteps:
    def test_zero_neighbor_min_size_returns_all(self):
        """neighbor_min_size=0 should return all timesteps."""
        timesteps = np.array([
            '2022-01-01T00:00:00',
            '2022-01-01T01:00:00'
        ], dtype='datetime64[s]')
        result = remove_isolated_timesteps(timesteps, neighbor_min_size=0,
                                           neighbor_time_interval=np.timedelta64(1, 'h'))
        # order preserved and all kept
        np.testing.assert_array_equal(result, np.sort(timesteps))

    def test_neighbor_min_size_one_removes_singletons(self):
        """neighbor_min_size=1 removes timesteps without at least one neighbor."""
        base = np.datetime64('2022-01-01T00:00:00')
        timesteps = np.array([
            base,
            base + np.timedelta64(1, 'm'),
            base + np.timedelta64(10, 'm')
        ], dtype='datetime64[s]')
        # interval of 2 minutes: first two are neighbors; third is isolated
        result = remove_isolated_timesteps(timesteps, neighbor_min_size=1,
                                           neighbor_time_interval=np.timedelta64(1, 'm'))
        expected = np.array([
            base,
            base + np.timedelta64(1, 'm')
        ], dtype='datetime64[s]')
        np.testing.assert_array_equal(result, expected)

    def test_neighbor_min_size_two_keeps_only_dense(self):
        """neighbor_min_size=2 keeps only timesteps with two neighbors."""
        base = np.datetime64('2022-01-01T00:00:00')
        timesteps = np.array([
            base,
            base + np.timedelta64(1, 'm'),
            base + np.timedelta64(2, 'm'),
            base + np.timedelta64(4, 'm')
        ], dtype='datetime64[s]')
        # interval of 2 minutes: middle point has two neighbors; ends have only one
        result = remove_isolated_timesteps(timesteps, neighbor_min_size=2,
                                           neighbor_time_interval=np.timedelta64(2, 'm'))
        np.testing.assert_array_equal(result, timesteps[0:3])
        
        result = remove_isolated_timesteps(timesteps, neighbor_min_size=2,
                                           neighbor_time_interval=np.timedelta64(3, 'm'))
        np.testing.assert_array_equal(result, timesteps[0:4])
        
    def test_unsorted_input_sorted(self):
        """Input timesteps are sorted before isolation check."""
        t0 = np.datetime64('2022-01-01T02:00:00')
        t1 = np.datetime64('2022-01-01T00:00:00')
        t2 = np.datetime64('2022-01-01T01:00:00')
        timesteps = np.array([t0, t1, t2], dtype='datetime64[s]')
        # interval large so all have at least one neighbor
        result = remove_isolated_timesteps(timesteps, neighbor_min_size=1,
                                           neighbor_time_interval=np.timedelta64(1, 'h'))
        expected = np.array([t1, t2, t0], dtype='datetime64[s]')
        np.testing.assert_array_equal(result, expected)










class TestGroupTimestepsIntoEvents:
    def test_empty_input(self):
        """Empty input yields no events."""
        timesteps = np.array([], dtype='datetime64[s]')
        events = group_timesteps_into_events(timesteps, np.timedelta64(1, 'h'))
        assert events == []

    def test_single_timestep(self):
        """A single timestep yields one event containing that timestep."""
        t = np.datetime64('2022-01-01T00:00:00')
        timesteps = np.array([t], dtype='datetime64[s]')
        events = group_timesteps_into_events(timesteps, np.timedelta64(1, 'h'))
        assert len(events) == 1
        np.testing.assert_array_equal(events[0], np.array([t], dtype='datetime64[s]'))

    def test_consecutive_within_gap(self):
        """Consecutive timesteps within max gap form a single event."""
        base = np.datetime64('2022-01-01T00:00:00')
        timesteps = np.array([
            base,
            base + np.timedelta64(30, 'm'),
            base + np.timedelta64(59, 'm')
        ], dtype='datetime64[s]')
        events = group_timesteps_into_events(timesteps, np.timedelta64(1, 'h'))
        assert len(events) == 1
        np.testing.assert_array_equal(events[0], timesteps)

    def test_gap_splits_events(self):
        """Timesteps separated by more than max gap split into separate events."""
        t0 = np.datetime64('2022-01-01T00:00:00')
        t1 = t0 + np.timedelta64(30, 'm')
        t2 = t1 + np.timedelta64(2, 'h')  # gap > 1h
        timesteps = np.array([t0, t1, t2], dtype='datetime64[s]')
        events = group_timesteps_into_events(timesteps, np.timedelta64(1, 'h'))
        assert len(events) == 2
        np.testing.assert_array_equal(events[0], np.array([t0, t1], dtype='datetime64[s]'))
        np.testing.assert_array_equal(events[1], np.array([t2], dtype='datetime64[s]'))

    def test_unsorted_input(self):
        """Unsorted timesteps are sorted before grouping."""
        t0 = np.datetime64('2022-01-01T02:00:00')
        t1 = np.datetime64('2022-01-01T00:00:00')
        t2 = np.datetime64('2022-01-01T01:00:00')
        timesteps = np.array([t0, t1, t2], dtype='datetime64[s]')
        # max gap large so all fall into one event after sorting
        events = group_timesteps_into_events(timesteps, np.timedelta64(2, 'h'))
        assert len(events) == 1
        expected = np.array([t1, t2, t0], dtype='datetime64[s]')
        np.testing.assert_array_equal(events[0], expected)


class TestIdentifyTimePartitions:
    def test_none_frequency(self, monkeypatch):
        """'none' returns a single block spanning all files."""
        filepaths = [
            'L1.1MIN.UL.Ljubljana.s20170522154558.e20170731132927.V0.nc',
            'L1.1MIN.UL.Ljubljana.s20180601120000.e20180701120000.V0.nc',
        ]

        result =  identify_time_partitions(filepaths, freq='none')
        assert len(result) == 1
        block = result[0]
        assert block['start_time'] == np.datetime64('2017-05-22T15:45:58')
        assert block['end_time']   == np.datetime64('2018-07-01T12:00:00')

    def test_day_frequency(self, monkeypatch):
        """'day' returns each calendar day block overlapping any file."""
        filepaths = [
            'L1.1MIN.UL.Ljubljana.s20220501100000.e20220501120000.V0.nc',
            'L1.1MIN.UL.Ljubljana.s20220505150000.e20220505170000.V0.nc',
        ]
        result =  identify_time_partitions(filepaths, freq='day')
       
        # Expect two days: May 1 and May 5, 2022
        assert len(result) == 2
        assert result[0] == {
            'start_time': np.datetime64('2022-05-01T00:00:00'),
            'end_time':   np.datetime64('2022-05-01T23:59:59'),
        }
        assert result[1] == {
            'start_time': np.datetime64('2022-05-05T00:00:00'),
            'end_time':   np.datetime64('2022-05-05T23:59:59'),
        }

    def test_season_frequency(self, monkeypatch):
        """'season' returns DJF block starting Dec of previous year when needed."""
        filepaths = [
            'L1.1MIN.UL.Ljubljana.s20220101120000.e20220215120000.V0.nc',
        ]
        result =  identify_time_partitions(filepaths, freq='season')
        # Only the DJF season for 2022 should appear, starting 2021-12-01, ending 2022-02-28
        assert len(result) == 1
        block = result[0]
        assert block['start_time'] == np.datetime64('2021-12-01T00:00:00')
        assert block['end_time']   == np.datetime64('2022-02-28T23:59:59')
        

# Generate filenames for period on 2017-05-22T00:00:00 - 2017-05-22T00:11:00^
base = np.datetime64('2017-05-22T00:00:00')
filepaths = [
    generate_product_filename(base + np.timedelta64(i * 2, 'm'),
                              base + np.timedelta64(i * 2 + 1, 'm'))
    for i in range(6)
]

class TestGetEventsInfo:

    def test_no_overlap(self):
        """Events outside the files range should yield no results."""
        events = [{
            "start_time": np.datetime64('2017-05-22T17:30:00'),
            "end_time":   np.datetime64('2017-05-22T18:00:00'),
        }]
        out = get_events_info(events, filepaths, sample_interval=60, accumulation_interval=60, rolling=False)
        assert out == []
    
    def test_no_aggregation_case(self):
        """Test case when sample_interval== accumulation_interval."""
        events = [{
            "start_time": np.datetime64('2017-05-22T00:00:00'),
            "end_time":   np.datetime64('2017-05-22T00:02:00'),
        }]
        out = get_events_info(events, filepaths, sample_interval=60, accumulation_interval=60, rolling=False)
        assert len(out) == 1
        info = out[0]
        assert info["start_time"] == np.datetime64('2017-05-22T00:00:00')
        assert info["end_time"]   == np.datetime64('2017-05-22T00:02:00')
        assert len(info["filepaths"]) == 2
         
        # Extend end_time by 180 seconds 
        # --> Include last file based on its end time   
        out = get_events_info(events, filepaths, sample_interval=60, accumulation_interval=180, rolling=False)
        assert len(out) == 1
        info = out[0]
        assert info["start_time"] == np.datetime64('2017-05-22T00:00:00')
        assert info["end_time"]   == np.datetime64('2017-05-22T00:05:00')
        assert len(info["filepaths"]) == 3 
        assert info["filepaths"][-1] == 'L1.1MIN.campaign.station.s20170522000400.e20170522000500.V0.nc'

        
    def test_single_event_not_rolling_case(self):
        """Test case for rolling=False."""
        events = [{
            "start_time": np.datetime64('2017-05-22T00:00:00'),
            "end_time":   np.datetime64('2017-05-22T00:02:00'),
        }]
        # Extend end_time by 120 seconds  
        # --> Include last file based on its start time   
        out = get_events_info(events, filepaths, sample_interval=60, accumulation_interval=120, rolling=False)
        assert len(out) == 1
        info = out[0]
        assert info["start_time"] == np.datetime64('2017-05-22T00:00:00')
        assert info["end_time"]   == np.datetime64('2017-05-22T00:04:00')
        assert len(info["filepaths"]) == 3 
        assert info["filepaths"][-1] == 'L1.1MIN.campaign.station.s20170522000400.e20170522000500.V0.nc'
        
        # Extend end_time by 180 seconds 
        # --> Include last file based on its end time   
        out = get_events_info(events, filepaths, sample_interval=60, accumulation_interval=180, rolling=False)
        assert len(out) == 1
        info = out[0]
        assert info["start_time"] == np.datetime64('2017-05-22T00:00:00')
        assert info["end_time"]   == np.datetime64('2017-05-22T00:05:00')
        assert len(info["filepaths"]) == 3 
        assert info["filepaths"][-1] == 'L1.1MIN.campaign.station.s20170522000400.e20170522000500.V0.nc'
        
    def test_single_event_rolling_case(self):
        """Test case for rolling=True."""
        events = [{
            "start_time": np.datetime64('2017-05-22T00:00:00'),
            "end_time":   np.datetime64('2017-05-22T00:02:00'),
        }]
        # Extend end_time by 120 seconds  
        # --> Include last file based on its start time   
        out = get_events_info(events, filepaths, sample_interval=60, accumulation_interval=120, rolling=True)
        assert len(out) == 1
        info = out[0]
        assert info["start_time"] == np.datetime64('2017-05-22T00:00:00')
        assert info["end_time"]   == np.datetime64('2017-05-22T00:04:00')
        assert len(info["filepaths"]) == 3 
        assert info["filepaths"][-1] == 'L1.1MIN.campaign.station.s20170522000400.e20170522000500.V0.nc'
        
        # Extend end_time by 180 seconds 
        # --> Include last file based on its end time   
        out = get_events_info(events, filepaths, sample_interval=60, accumulation_interval=180, rolling=True)
        assert len(out) == 1
        info = out[0]
        assert info["start_time"] == np.datetime64('2017-05-22T00:00:00')
        assert info["end_time"]   == np.datetime64('2017-05-22T00:05:00')
        assert len(info["filepaths"]) == 3 
        assert info["filepaths"][-1] == 'L1.1MIN.campaign.station.s20170522000400.e20170522000500.V0.nc'
           
    def test_multiple_event(self):
        """Multiple events with forward 1-min extend map to correct file sets."""
        events = [
            {
                "start_time": np.datetime64('2017-05-22T00:00:00'),
                "end_time":   np.datetime64('2017-05-22T00:02:00'),
            },
            {
                "start_time": np.datetime64('2017-05-22T00:04:00'),
                "end_time":   np.datetime64('2017-05-22T00:05:00'),
            },
            {   # Event without files 
                "start_time": np.datetime64('2018-05-22T00:04:00'),
                "end_time":   np.datetime64('2018-05-22T00:05:00'),
            }
        ]
   
        out = get_events_info(events, filepaths, sample_interval=60, accumulation_interval=120, rolling=False)
        assert len(out) == 2
        assert out[0]["start_time"] == np.datetime64('2017-05-22T00:00:00')
        assert out[0]["end_time"]   == np.datetime64('2017-05-22T00:04:00')
        
        assert out[1]["start_time"] == np.datetime64('2017-05-22T00:04:00')
        assert out[1]["end_time"]   == np.datetime64('2017-05-22T00:07:00')

    def test_empty_events_list(self):
        """Providing no events returns an empty list."""
        out = get_events_info([], filepaths, sample_interval=60, accumulation_interval=60, rolling=False)
        assert out == []
        
    def test_empty_filepaths_list(self):
        """Providing no files always yields no overlaps."""
        events = [{
            "start_time": np.datetime64('2017-05-22T00:01:00'),
            "end_time":   np.datetime64('2017-05-22T00:01:00'),
        }]
        out = get_events_info(events, [], sample_interval=60, accumulation_interval=60, rolling=False)
        assert out == []