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
