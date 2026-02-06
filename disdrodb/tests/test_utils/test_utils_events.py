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
"""Testing events utilities."""

import numpy as np
import pytest
import xarray as xr

from disdrodb.utils.event import (
    group_timesteps_into_event,
    group_timesteps_into_events,
    remove_isolated_timesteps,
)


class TestGroupTimestepsIntoEvent:
    def test_basic_grouping_defaults(self):
        """Default parameters group all close timesteps into one event."""
        timesteps = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
                "2022-01-01T01:00:00",
            ],
            dtype="datetime64[s]",
        )
        # default intra gap=0S so each timestep is its own event, but default neighbor_min_size=0 so no isolation
        events = group_timesteps_into_event(
            timesteps,
            event_max_time_gap="1H",
        )
        assert isinstance(events, list)
        # all three fall into one event since gap <=1H
        assert len(events) == 1
        ev = events[0]
        assert ev["start_time"] == np.datetime64("2022-01-01T00:00:00")
        assert ev["end_time"] == np.datetime64("2022-01-01T01:00:00")
        assert ev["n_timesteps"] == 3
        assert ev["duration"] == np.timedelta64(60, "m")

    def test_isolation_removal(self):
        """Isolated timesteps are removed before grouping."""
        timesteps = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:03:00",
                "2022-01-01T02:00:00",
            ],
            dtype="datetime64[s]",
        )
        events = group_timesteps_into_event(
            timesteps,
            neighbor_min_size=1,
            neighbor_time_interval="5MIN",
            event_max_time_gap="1H",
        )
        # the 02:00 point is isolated (no neighbor within 5min) and removed,
        # so only one event with the first two timesteps
        assert len(events) == 1
        ev = events[0]
        assert ev["start_time"] == np.datetime64("2022-01-01T00:00:00")
        assert ev["end_time"] == np.datetime64("2022-01-01T00:03:00")
        assert ev["n_timesteps"] == 2

    def test_event_min_size_filter(self):
        """Events smaller than event_min_size are discarded."""
        timesteps = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T03:00:00",
            ],
            dtype="datetime64[s]",
        )
        # two separate events of size 1; event_min_size=2 should filter both
        events = group_timesteps_into_event(
            timesteps,
            event_max_time_gap="30MIN",
            event_min_size=2,
        )
        assert events == []

    def test_event_min_duration_filter(self):
        """Events shorter than event_min_duration are discarded."""
        timesteps = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T01:00:00",
            ],
            dtype="datetime64[s]",
        )
        # duration is 60m; event_min_duration=120MIN should filter it out
        events = group_timesteps_into_event(
            timesteps,
            event_max_time_gap="2H",
            event_min_duration="120MIN",
        )
        assert events == []

    def test_multiple_events_split_by_gap(self):
        """Large gaps create multiple separate events."""
        timesteps = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
                "2022-01-01T03:00:00",
                "2022-01-01T03:30:00",
            ],
            dtype="datetime64[s]",
        )
        events = group_timesteps_into_event(
            timesteps,
            event_max_time_gap="1H",
        )
        # Should split into two events: first two and last two
        assert len(events) == 2
        ev1, ev2 = events
        assert ev1["start_time"] == np.datetime64("2022-01-01T00:00:00")
        assert ev1["end_time"] == np.datetime64("2022-01-01T00:30:00")
        assert ev2["start_time"] == np.datetime64("2022-01-01T03:00:00")
        assert ev2["end_time"] == np.datetime64("2022-01-01T03:30:00")


class TestRemoveIsolatedTimesteps:
    def test_zero_neighbor_min_size_returns_all(self):
        """neighbor_min_size=0 should return all timesteps."""
        timesteps = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T01:00:00",
            ],
            dtype="datetime64[s]",
        )
        result = remove_isolated_timesteps(
            timesteps,
            neighbor_min_size=0,
            neighbor_time_interval=np.timedelta64(1, "h"),
        )
        # order preserved and all kept
        np.testing.assert_array_equal(result, np.sort(timesteps))

    def test_neighbor_min_size_one_removes_singletons(self):
        """neighbor_min_size=1 removes timesteps without at least one neighbor."""
        base = np.datetime64("2022-01-01T00:00:00")
        timesteps = np.array(
            [
                base,
                base + np.timedelta64(1, "m"),
                base + np.timedelta64(10, "m"),
            ],
            dtype="datetime64[s]",
        )
        # interval of 2 minutes: first two are neighbors; third is isolated
        result = remove_isolated_timesteps(
            timesteps,
            neighbor_min_size=1,
            neighbor_time_interval=np.timedelta64(1, "m"),
        )
        expected = np.array(
            [
                base,
                base + np.timedelta64(1, "m"),
            ],
            dtype="datetime64[s]",
        )
        np.testing.assert_array_equal(result, expected)

    def test_neighbor_min_size_two_keeps_only_dense(self):
        """neighbor_min_size=2 keeps only timesteps with two neighbors."""
        base = np.datetime64("2022-01-01T00:00:00")
        timesteps = np.array(
            [
                base,
                base + np.timedelta64(1, "m"),
                base + np.timedelta64(2, "m"),
                base + np.timedelta64(4, "m"),
            ],
            dtype="datetime64[s]",
        )
        # interval of 2 minutes: middle point has two neighbors; ends have only one
        result = remove_isolated_timesteps(
            timesteps,
            neighbor_min_size=2,
            neighbor_time_interval=np.timedelta64(2, "m"),
        )
        np.testing.assert_array_equal(result, timesteps[0:3])

        result = remove_isolated_timesteps(
            timesteps,
            neighbor_min_size=2,
            neighbor_time_interval=np.timedelta64(3, "m"),
        )
        np.testing.assert_array_equal(result, timesteps[0:4])

    def test_unsorted_input_sorted(self):
        """Input timesteps are sorted before isolation check."""
        t0 = np.datetime64("2022-01-01T02:00:00")
        t1 = np.datetime64("2022-01-01T00:00:00")
        t2 = np.datetime64("2022-01-01T01:00:00")
        timesteps = np.array([t0, t1, t2], dtype="datetime64[s]")
        # interval large so all have at least one neighbor
        result = remove_isolated_timesteps(
            timesteps,
            neighbor_min_size=1,
            neighbor_time_interval=np.timedelta64(1, "h"),
        )
        expected = np.array([t1, t2, t0], dtype="datetime64[s]")
        np.testing.assert_array_equal(result, expected)


class TestGroupTimestepsIntoEvents:
    def test_empty_input(self):
        """Empty input yields no events."""
        timesteps = np.array([], dtype="datetime64[s]")
        events = group_timesteps_into_events(timesteps, np.timedelta64(1, "h"))
        assert events == []

    def test_single_timestep(self):
        """A single timestep yields one event containing that timestep."""
        t = np.datetime64("2022-01-01T00:00:00")
        timesteps = np.array([t], dtype="datetime64[s]")
        events = group_timesteps_into_events(timesteps, np.timedelta64(1, "h"))
        assert len(events) == 1
        np.testing.assert_array_equal(events[0], np.array([t], dtype="datetime64[s]"))

    def test_consecutive_within_gap(self):
        """Consecutive timesteps within max gap form a single event."""
        base = np.datetime64("2022-01-01T00:00:00")
        timesteps = np.array(
            [
                base,
                base + np.timedelta64(30, "m"),
                base + np.timedelta64(59, "m"),
            ],
            dtype="datetime64[s]",
        )
        events = group_timesteps_into_events(timesteps, np.timedelta64(1, "h"))
        assert len(events) == 1
        np.testing.assert_array_equal(events[0], timesteps)

    def test_gap_splits_events(self):
        """Timesteps separated by more than max gap split into separate events."""
        t0 = np.datetime64("2022-01-01T00:00:00")
        t1 = t0 + np.timedelta64(30, "m")
        t2 = t1 + np.timedelta64(2, "h")  # gap > 1h
        timesteps = np.array([t0, t1, t2], dtype="datetime64[s]")
        events = group_timesteps_into_events(timesteps, np.timedelta64(1, "h"))
        assert len(events) == 2
        np.testing.assert_array_equal(events[0], np.array([t0, t1], dtype="datetime64[s]"))
        np.testing.assert_array_equal(events[1], np.array([t2], dtype="datetime64[s]"))

    def test_unsorted_input(self):
        """Unsorted timesteps are sorted before grouping."""
        t0 = np.datetime64("2022-01-01T02:00:00")
        t1 = np.datetime64("2022-01-01T00:00:00")
        t2 = np.datetime64("2022-01-01T01:00:00")
        timesteps = np.array([t0, t1, t2], dtype="datetime64[s]")
        # max gap large so all fall into one event after sorting
        events = group_timesteps_into_events(timesteps, np.timedelta64(2, "h"))
        assert len(events) == 1
        expected = np.array([t1, t2, t0], dtype="datetime64[s]")
        np.testing.assert_array_equal(events[0], expected)


class TestSplitIntoEvents:
    def test_threshold_based_detection_single_event(self):
        """Basic threshold detection produces single event from consecutive high-value timesteps."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
                "2022-01-01T01:00:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [15, 20, 18])},
            coords={"time": time},
        )
        events = list(
            ds.disdrodb.split_into_events(
                variable="N",
                threshold=10,
                event_max_time_gap="2H",
                neighbor_min_size=0,
                neighbor_time_interval="5MIN",
                event_min_duration="0S",
                event_min_size=1,
            ),
        )
        assert len(events) == 1
        assert events[0].sizes["time"] == 3
        np.testing.assert_array_equal(events[0]["time"].values, time)

    def test_threshold_based_detection_multiple_events(self):
        """Threshold detection splits into multiple events when gaps exceed max time gap."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
                "2022-01-01T03:00:00",
                "2022-01-01T03:30:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [15, 20, 18, 22])},
            coords={"time": time},
        )
        events = list(
            ds.disdrodb.split_into_events(
                variable="N",
                threshold=10,
                event_max_time_gap="1H",
                neighbor_min_size=0,
                neighbor_time_interval="5MIN",
                event_min_duration="0S",
                event_min_size=1,
            ),
        )
        assert len(events) == 2
        assert events[0].sizes["time"] == 2
        assert events[1].sizes["time"] == 2

    def test_boolean_based_detection(self):
        """Boolean variable detection correctly identifies events from True values."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
                "2022-01-01T01:00:00",
                "2022-01-01T03:00:00",
                "2022-01-01T03:30:00",
                "2022-01-01T03:40:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"is_rainy": (["time"], [False, True, True, False, True, True])},
            coords={"time": time},
        )
        events = list(
            ds.disdrodb.split_into_events(
                variable="is_rainy",
                threshold=None,
                event_max_time_gap="30MIN",
                neighbor_min_size=0,
                neighbor_time_interval="5MIN",
                event_min_duration="0S",
                event_min_size=1,
            ),
        )
        # Two separate events (False in middle breaks continuity)
        assert len(events) == 2
        assert events[0].sizes["time"] == 2
        assert events[1].sizes["time"] == 2

    def test_no_events_found_below_threshold(self):
        """No events returned when all values are below threshold."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [5, 3])},
            coords={"time": time},
        )
        events = list(
            ds.disdrodb.split_into_events(
                variable="N",
                threshold=10,
                event_max_time_gap="1H",
                neighbor_min_size=0,
                neighbor_time_interval="5MIN",
                event_min_duration="0S",
                event_min_size=1,
            ),
        )
        assert len(events) == 0

    def test_no_events_found_all_false(self):
        """No events returned when boolean variable is all False."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"is_rainy": (["time"], [False, False])},
            coords={"time": time},
        )
        events = list(
            ds.disdrodb.split_into_events(
                variable="is_rainy",
                threshold=None,
                event_max_time_gap="1H",
                neighbor_min_size=0,
                neighbor_time_interval="5MIN",
                event_min_duration="0S",
                event_min_size=1,
            ),
        )
        assert len(events) == 0

    def test_event_min_size_filter(self):
        """Events with fewer timesteps than event_min_size are discarded."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
                "2022-01-01T03:00:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [15, 20, 18])},
            coords={"time": time},
        )
        events = list(
            ds.disdrodb.split_into_events(
                variable="N",
                threshold=10,
                event_max_time_gap="1H",
                event_min_size=2,
                neighbor_min_size=0,
                neighbor_time_interval="5MIN",
                event_min_duration="0S",
            ),
        )
        # First two form one event (size=2), third is isolated (size=1, filtered)
        assert len(events) == 1
        assert events[0].sizes["time"] == 2

    def test_event_min_duration_filter(self):
        """Events shorter than event_min_duration are discarded."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:10:00",
                "2022-01-01T02:00:00",
                "2022-01-01T04:00:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [15, 20, 18, 22])},
            coords={"time": time},
        )
        events = list(
            ds.disdrodb.split_into_events(
                variable="N",
                threshold=10,
                event_max_time_gap="1H",
                event_min_duration="1H",
                neighbor_min_size=0,
                neighbor_time_interval="5MIN",
                event_min_size=1,
            ),
        )
        # First event duration=10min (filtered), second two are single points (filtered)
        assert len(events) == 0

    def test_neighbor_isolation_removal(self):
        """Isolated timesteps without enough neighbors are excluded from events."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:01:00",
                "2022-01-01T02:00:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [15, 20, 18])},
            coords={"time": time},
        )
        events = list(
            ds.disdrodb.split_into_events(
                variable="N",
                threshold=10,
                neighbor_min_size=1,
                neighbor_time_interval="5MIN",
                event_min_duration="0S",
                event_max_time_gap="2H",
                event_min_size=2,
            ),
        )
        # Third timestep at 02:00 is isolated (no neighbor within 5min)
        assert len(events) == 1
        assert events[0].sizes["time"] == 2

    def test_sortby_duration_decreasing(self):
        """Events sorted by duration in decreasing order when sortby='duration'."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:10:00",
                "2022-01-01T02:00:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [15, 20, 18])},
            coords={"time": time},
        )
        events = list(
            ds.disdrodb.split_into_events(
                variable="N",
                threshold=10,
                event_max_time_gap="30MIN",
                neighbor_min_size=0,
                neighbor_time_interval="5MIN",
                event_min_duration="0S",
                event_min_size=1,
                sortby="duration",
                sortby_order="decreasing",
            ),
        )
        assert len(events) == 2
        # Longest first
        assert events[0]["duration"].item() > events[1]["duration"].item()

    def test_sortby_duration_increasing(self):
        """Events sorted by duration in increasing order when sortby_order='increasing'."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:10:00",
                "2022-01-01T02:00:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [15, 20, 18])},
            coords={"time": time},
        )
        events = list(
            ds.disdrodb.split_into_events(
                variable="N",
                threshold=10,
                event_max_time_gap="30MIN",
                neighbor_min_size=0,
                neighbor_time_interval="5MIN",
                event_min_duration="0S",
                event_min_size=1,
                sortby="duration",
                sortby_order="increasing",
            ),
        )
        assert len(events) == 2
        # Shortest first
        assert events[0]["duration"].item() < events[1]["duration"].item()

    def test_sortby_custom_callable(self):
        """Events sorted by custom callable returning scalar value."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
                "2022-01-01T02:00:00",
                "2022-01-01T02:30:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [15, 20, 10, 30])},
            coords={"time": time},
        )
        # Sort by max N in event
        sortby_func = lambda ds_event: float(ds_event["N"].max())  # noqa: E731
        events = list(
            ds.disdrodb.split_into_events(
                variable="N",
                threshold=5,
                event_max_time_gap="1H",
                neighbor_min_size=0,
                neighbor_time_interval="5MIN",
                event_min_duration="0S",
                event_min_size=1,
                sortby=sortby_func,
                sortby_order="decreasing",
            ),
        )
        assert len(events) == 2
        # Event with max N=30 should be first
        assert float(events[0]["N"].max()) == 30
        assert float(events[1]["N"].max()) == 20

    def test_invalid_non_boolean_without_threshold(self):
        """ValueError raised when threshold=None and variable is not boolean."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [15, 20])},
            coords={"time": time},
        )
        with pytest.raises(ValueError, match="must be a boolean DataArray"):
            list(
                ds.disdrodb.split_into_events(
                    variable="N",
                    threshold=None,
                    event_max_time_gap="1H",
                    neighbor_min_size=0,
                    neighbor_time_interval="5MIN",
                    event_min_duration="0S",
                    event_min_size=1,
                ),
            )

    def test_invalid_sortby_callable_non_scalar(self):
        """ValueError raised when sortby callable returns non-scalar value."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [15, 20])},
            coords={"time": time},
        )
        # Returns an array instead of scalar
        sortby_func = lambda ds_event: ds_event["N"].to_numpy()  # noqa: E731
        with pytest.raises(ValueError, match="must return a scalar value"):
            list(
                ds.disdrodb.split_into_events(
                    variable="N",
                    threshold=10,
                    event_max_time_gap="1H",
                    neighbor_min_size=0,
                    neighbor_time_interval="5MIN",
                    event_min_duration="0S",
                    event_min_size=1,
                    sortby=sortby_func,
                ),
            )

    def test_invalid_sortby_value(self):
        """ValueError raised when sortby is invalid (not None, 'duration', or callable)."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [15, 20])},
            coords={"time": time},
        )
        with pytest.raises(ValueError, match="must be None, 'duration', or a callable"):
            list(
                ds.disdrodb.split_into_events(
                    variable="N",
                    threshold=10,
                    event_max_time_gap="1H",
                    neighbor_min_size=0,
                    neighbor_time_interval="5MIN",
                    event_min_duration="0S",
                    event_min_size=1,
                    sortby="invalid_value",
                ),
            )

    def test_unsorted_dataset_is_sorted(self):
        """Dataset with unsorted time coordinate is automatically sorted before processing."""
        time = np.array(
            [
                "2022-01-01T01:00:00",
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [18, 15, 20])},
            coords={"time": time},
        )
        events = list(
            ds.disdrodb.split_into_events(
                variable="N",
                threshold=10,
                event_max_time_gap="2H",
                neighbor_min_size=0,
                neighbor_time_interval="5MIN",
                event_min_duration="0S",
                event_min_size=1,
            ),
        )
        assert len(events) == 1
        # Time should be sorted in the output
        expected_time = np.sort(time)
        np.testing.assert_array_equal(events[0]["time"].values, expected_time)

    def test_duration_attribute_added_to_events(self):
        """Each event dataset contains a 'duration' attribute with the event duration."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [15, 20])},
            coords={"time": time},
        )
        events = list(
            ds.disdrodb.split_into_events(
                variable="N",
                threshold=10,
                event_max_time_gap="1H",
                neighbor_min_size=0,
                neighbor_time_interval="5MIN",
                event_min_duration="0S",
                event_min_size=1,
            ),
        )
        assert len(events) == 1
        assert "duration" in events[0]
        assert events[0]["duration"].item() == np.timedelta64(30, "m")

    def test_empty_dataset(self):
        """Empty dataset returns no events."""
        time = np.array([], dtype="datetime64[s]")
        ds = xr.Dataset(
            {"N": (["time"], [])},
            coords={"time": time},
        )
        events = list(
            ds.disdrodb.split_into_events(
                variable="N",
                threshold=10,
                event_max_time_gap="1H",
                neighbor_min_size=0,
                neighbor_time_interval="5MIN",
                event_min_duration="0S",
                event_min_size=1,
            ),
        )
        assert len(events) == 0

    def test_sample_interval_validation(self):
        """ValueError raised when neighbor_time_interval is less than sample_interval."""
        time = np.array(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:30:00",
            ],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"N": (["time"], [15, 20])},
            coords={"time": time},
        )
        ds["sample_interval"] = 1800  # 30 minutes in seconds
        with pytest.raises(ValueError, match="must be at least equal to the dataset sample interval"):
            list(
                ds.disdrodb.split_into_events(
                    variable="N",
                    threshold=10,
                    neighbor_time_interval="10MIN",  # Less than 30min
                    event_max_time_gap="1H",
                    neighbor_min_size=0,
                    event_min_duration="0S",
                    event_min_size=1,
                ),
            )
