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
"""Routines for 1D and 2D Look Up Tables (LUT)."""

import pickle

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError, cKDTree

_VALID_2D_DISTANCES = {"grid", "euclidean", "manhattan"}

### in predict()
# - if input NaN, return NaN in predict()
# - add argument max_distance(). If distance > specified, set to np.nan
#   --> allow specify tuple for x and y (univariate distance)
#   --> if one value, applied to dist currently computed
#   --> if none, do not mask


def _get_query_index(values):
    """Return a pandas index when the query input carries one."""
    if isinstance(values, pd.Index):
        return values
    index = getattr(values, "index", None)
    return index if isinstance(index, pd.Index) else None


def _discard_nan_value_rows(df, core_columns, coordinate_columns=None):
    """Discard rows with NaNs in the coordinate or core columns."""
    if coordinate_columns is None:
        coordinate_columns = []
    columns = list(dict.fromkeys([*coordinate_columns, *core_columns]))
    if len(columns) == 0:
        return df
    valid_rows = df[columns].notna().all(axis=1)
    return df.loc[valid_rows]


def _check_columns_in_dataframe(df, columns, variable_name):
    """Check that columns are available in the DataFrame."""
    if not np.all(np.isin(columns, df.columns)):
        missing = np.setdiff1d(columns, df.columns).tolist()
        raise ValueError(f"{missing} {variable_name} not found in DataFrame")


def _get_nearest_sorted_values(values, query):
    """Return nearest values from a sorted 1D array for each query value."""
    idx_upper = np.searchsorted(values, query)
    idx_upper = np.clip(idx_upper, 0, len(values) - 1)
    idx_lower = np.clip(idx_upper - 1, 0, len(values) - 1)

    lower_values = values[idx_lower]
    upper_values = values[idx_upper]
    use_upper = np.abs(query - upper_values) < np.abs(query - lower_values)
    return np.where(use_upper, upper_values, lower_values)


def _get_points_tolerance(points):
    """Return a coordinate-scaled tolerance for geometric point checks."""
    dtype = points.dtype if np.issubdtype(points.dtype, np.floating) else float
    eps = np.finfo(dtype).eps
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    return 100 * eps * scale


def _get_convex_hull(points):
    """Return a Delaunay triangulation for non-degenerate 2D points."""
    unique_points = np.unique(points, axis=0)
    if len(unique_points) < 3:
        return None
    try:
        return Delaunay(unique_points)
    except QhullError:
        return None


def _points_in_degenerate_hull(query_points, points, atol):
    """Return whether query points lie inside a degenerate 2D convex hull."""
    unique_points = np.unique(points, axis=0)
    if len(unique_points) == 0:
        return np.zeros(len(query_points), dtype=bool)
    if len(unique_points) == 1:
        return np.linalg.norm(query_points - unique_points[0], axis=1) <= atol

    center = np.mean(unique_points, axis=0)
    _, _, vh = np.linalg.svd(unique_points - center, full_matrices=False)
    direction = vh[0]

    point_proj = (unique_points - center) @ direction
    query_proj = (query_points - center) @ direction
    query_reconstructed = center + np.outer(query_proj, direction)

    on_line = np.linalg.norm(query_points - query_reconstructed, axis=1) <= atol
    within_segment = (query_proj >= np.min(point_proj) - atol) & (query_proj <= np.max(point_proj) + atol)
    return on_line & within_segment


def _points_in_convex_hull(query_points, points, hull=None):
    """Return whether query points lie inside the convex hull of points."""
    if len(query_points) == 0:
        return np.array([], dtype=bool)
    if len(points) == 0:
        return np.zeros(len(query_points), dtype=bool)
    atol = _get_points_tolerance(points)
    if hull is not None:
        return hull.find_simplex(query_points, tol=atol) >= 0
    return _points_in_degenerate_hull(query_points, points, atol=atol)


class NearestNeighbourLUT1D:
    """A 1D nearest neighbor lookup table using k-d tree.

    This class builds a k-d tree from 1D points and their associated values,
    enabling fast nearest neighbor queries for interpolation or lookup purposes.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the coordinate and values.
    x : str
        Name of the column representing the x-coordinate.
    columns : list of str, optional
        List of column names to include in the lookup table values.
        If None, all columns are included. Default is None.
    dtype : numpy.dtype, optional
        Data type for storing points and values. Default is np.float32.
    core_columns : list of str, optional
        List of column names required to contain valid values.
        Rows with NaNs in these columns or the coordinate column are discarded.
        If None, it defaults to ``columns`` and preserves the previous behavior.
        Default is None.

    Notes
    -----
    Rows with NaNs in the coordinate or core columns are discarded before
    constructing the lookup table.
    """

    def __init__(self, df, x, columns=None, dtype=np.float32, core_columns=None):
        if columns is None:
            columns = list(df.columns)
        if x in df.index.names:
            df = df.reset_index()
        if x not in df:
            raise ValueError(f"{x=} is not a column of df.")

        # Remove x from columns and core columns
        columns = list(set(columns) - {x})
        if core_columns is None:
            core_columns = columns
        core_columns = list(set(core_columns) - {x})

        # Check columns validity
        _check_columns_in_dataframe(df, columns, "columns")
        _check_columns_in_dataframe(df, core_columns, "core_columns")

        # Remove rows with core columns having NaN
        df = _discard_nan_value_rows(df, core_columns, coordinate_columns=[x])

        # Sort columns alphabetically
        columns = sorted(columns)
        core_columns = sorted(core_columns)

        self.x = x
        self.columns = columns
        self.core_columns = core_columns
        self.dtype = dtype
        self.points = df[[x]].to_numpy(dtype=dtype)
        self.values = df[columns].to_numpy(dtype=dtype)

        self.tree = cKDTree(self.points)

    # ------------------
    # Persistence
    # ------------------
    def save_lut(self, filename):
        """Save the lookup table to a file using pickle."""
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def read_lut(cls, filename):
        """Load a lookup table from a pickle file."""
        with open(filename, "rb") as f:
            return pickle.load(f)

    # ------------------
    # Query
    # ------------------
    def predict(self, x, return_distance=False, max_distance=None):
        """Query the lookup table for nearest neighbor values.

        Parameters
        ----------
        x : array-like
            x-coordinates of query points. Can be scalar or array.
        return_distance : bool, optional
            If True, include the distance to the nearest neighbor in the output.
            Default is False.
        max_distance : float, tuple of float, or None, optional
            Maximum distance threshold for valid predictions.

            - If None: no distance masking is applied.
            - If float: points with distance > max_distance are set to NaN.
            - If tuple (dx,): points are masked if |x - x_nearest| > dx.

            Default is None.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the nearest neighbor values for each query point.
            Columns include x coordinates, the lookup values, and optionally
            the distance to the nearest neighbor. Values beyond max_distance are NaN.
        """
        index = _get_query_index(x)
        x = np.atleast_1d(x).astype(float)

        invalid_mask = ~np.isfinite(x)

        n_points = len(x)
        n_columns = len(self.columns)
        data = np.full((n_points, n_columns), np.nan, dtype=self.dtype)
        dist = np.full(n_points, np.nan, dtype=self.dtype)

        valid_mask = ~invalid_mask
        if np.any(valid_mask):
            pts_valid = x[valid_mask, np.newaxis]
            dist_valid, idx_valid = self.tree.query(pts_valid)
            data_valid = self.values[idx_valid]

            if max_distance is not None:
                if isinstance(max_distance, (tuple, list)):
                    if len(max_distance) != 1:
                        raise ValueError("max_distance tuple must have exactly 1 element (dx)")
                    (dx_max,) = max_distance
                    nearest_points = self.points[idx_valid, 0]
                    distance_mask = np.abs(x[valid_mask] - nearest_points) > dx_max
                else:
                    distance_mask = dist_valid > max_distance
                data_valid[distance_mask] = np.nan

            data[valid_mask] = data_valid
            dist[valid_mask] = dist_valid

        df_out = pd.DataFrame(
            data,
            columns=self.columns,
            index=index,
        )

        df_out[self.x] = x
        df_out = df_out[[self.x, *self.columns]]

        if return_distance:
            df_out["distance"] = dist

        return df_out

    # ------------------
    # Convenience
    # ------------------
    def predict_dict(self, x):
        """Query the lookup table and return result as a dictionary."""
        if np.size(x) != 1:
            raise ValueError("predict_dict accepts only scalars")
        df = self.predict(x).iloc[0]
        return df.to_dict()

    def __len__(self):
        """Return the number of points in the lookup table."""
        return self.points.shape[0]


class NearestNeighbourLUT2D:
    """A 2D nearest neighbor lookup table using k-d tree.

    This class builds a k-d tree from 2D points and their associated values,
    enabling fast nearest neighbor queries for interpolation or lookup purposes.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the coordinates and values.
    x : str
        Name of the column representing the x-coordinate.
    y : str
        Name of the column representing the y-coordinate.
    columns : list of str, optional
        List of column names to include in the lookup table values.
        If None, all columns are included. Default is None.
    dtype : numpy.dtype, optional
        Data type for storing points and values. Default is np.float32.
    core_columns : list of str, optional
        List of column names required to contain valid values.
        Rows with NaNs in these columns or the coordinate columns are discarded.
        If None, it defaults to ``columns`` and preserves the previous behavior.
        Default is None.

    Attributes
    ----------
    x : str
        Name of the x-coordinate column.
    y : str
        Name of the y-coordinate column.
    columns : list of str
        Column names for the lookup values.
    core_columns : list of str
        Column names required to contain valid values.
    dtype : numpy.dtype
        Data type used for storage.
    points : numpy.ndarray
        Array of shape (n, 2) containing the coordinates.
    values : numpy.ndarray
        Array of shape (n, len(columns)) containing the lookup values.
    tree : scipy.spatial.cKDTree
        k-d tree for fast nearest neighbor queries.

    Raises
    ------
    ValueError
        If x or y are not columns in the DataFrame.
    ValueError
        If any of the specified columns are not found in the DataFrame.

    Notes
    -----
    Rows with NaNs in the coordinate or core columns are discarded before
    constructing the lookup table.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2], "value": [10, 20, 30]})
    >>> lut = NearestNeighbourLUT2D(df, "x", "y", columns=["value"])
    >>> lut.predict([0.5], [0.5])
    """

    def __init__(self, df, x, y, columns=None, dtype=np.float32, core_columns=None):
        if columns is None:
            columns = list(df.columns)
        if x in df.index.names:
            df = df.reset_index()
        # Check x and y
        if x not in df:
            raise ValueError(f"{x=} is not a column of df.")
        if y not in df:
            raise ValueError(f"{y=} is not a column of df.")

        # Remove x and y from columns and core columns
        columns = list(set(columns) - {x, y})
        if core_columns is None:
            core_columns = columns
        core_columns = list(set(core_columns) - {x, y})

        # Check columns
        _check_columns_in_dataframe(df, columns, "columns")
        _check_columns_in_dataframe(df, core_columns, "core_columns")

        # Remove rows with core columns having NaN
        df = _discard_nan_value_rows(df, core_columns, coordinate_columns=[x, y])

        # Sort columns alphabetically
        columns = sorted(columns)
        core_columns = sorted(core_columns)

        self.x = x
        self.y = y
        self.columns = columns
        self.core_columns = core_columns
        self.dtype = dtype
        self.points = df[[x, y]].to_numpy(dtype=dtype)
        self.values = df[columns].to_numpy(dtype=dtype)

        self.tree = cKDTree(self.points)
        self._grid_x = np.unique(self.points[:, 0])
        self._grid_y = np.unique(self.points[:, 1])
        self._grid_indices = {}
        for i, point in enumerate(self.points):
            self._grid_indices.setdefault((point[0], point[1]), i)
        self._convex_hull = _get_convex_hull(self.points)

    # ------------------
    # Persistence
    # ------------------
    def save_lut(self, filename):
        """Save the lookup table to a file using pickle.

        Parameters
        ----------
        filename : str
            Path to the file where the lookup table will be saved.

        Notes
        -----
        Uses pickle with HIGHEST_PROTOCOL for efficient serialization.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def read_lut(cls, filename):
        """Load a lookup table from a pickle file.

        Parameters
        ----------
        filename : str
            Path to the file containing the saved lookup table.

        Returns
        -------
        NearestNeighbourLUT2D
            The loaded lookup table instance.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    # ------------------
    # Query
    # ------------------
    def _query_grid(self, pts_valid):
        """Return grid nearest-neighbour distances and indices."""
        grid_x = getattr(self, "_grid_x", np.unique(self.points[:, 0]))
        grid_y = getattr(self, "_grid_y", np.unique(self.points[:, 1]))
        grid_indices = getattr(self, "_grid_indices", None)
        if grid_indices is None:
            grid_indices = {}
            for i, point in enumerate(self.points):
                grid_indices.setdefault((point[0], point[1]), i)

        nearest_x = _get_nearest_sorted_values(grid_x, pts_valid[:, 0])
        nearest_y = _get_nearest_sorted_values(grid_y, pts_valid[:, 1])

        idx = np.full(len(pts_valid), -1, dtype=int)
        for i, point in enumerate(zip(nearest_x, nearest_y, strict=True)):
            idx[i] = grid_indices.get(point, -1)

        missing_mask = idx < 0
        if np.any(missing_mask):
            snapped_points = np.column_stack([nearest_x[missing_mask], nearest_y[missing_mask]])
            _, idx[missing_mask] = self.tree.query(snapped_points)

        dist = np.full(len(pts_valid), np.nan, dtype=self.dtype)
        found_mask = idx >= 0
        if np.any(found_mask):
            nearest_points = self.points[idx[found_mask]]
            dist[found_mask] = np.linalg.norm(pts_valid[found_mask] - nearest_points, axis=1)
        return dist, idx

    def _query_out_of_bound(self, pts_valid):
        """Return True for points outside the LUT points convex hull."""
        if not hasattr(self, "_convex_hull"):
            self._convex_hull = _get_convex_hull(self.points)
        return ~_points_in_convex_hull(pts_valid, self.points, hull=self._convex_hull)

    def predict(self, x, y, return_distance=False, max_distance=None, distance="grid"):
        """Query the lookup table for nearest neighbor values.

        Parameters
        ----------
        x : array-like
            x-coordinates of query points. Can be scalar or array.
        y : array-like
            y-coordinates of query points. Can be scalar or array.
        return_distance : bool, optional
            If True, include the distance to the nearest neighbor in the output.
            Default is False.
        max_distance : float, tuple of float, or None, optional
            Maximum distance threshold for valid predictions.

            - If None: no distance masking is applied.
            - If float: points with the selected distance > max_distance are set to NaN.
            - If tuple (dx, dy): points are masked if |x - x_nearest| > dx OR |y - y_nearest| > dy.

            Default is None.
        distance : {"grid", "euclidean", "manhattan"}, optional
            Nearest-neighbour distance strategy.

            - "grid": snap x and y independently to their closest LUT coordinate.
            - "euclidean": select the closest LUT point by Euclidean distance.
            - "manhattan": select the closest LUT point by Manhattan distance.

            Default is "grid".

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the nearest neighbor values for each query point.
            Columns include x, y coordinates, an out_of_bound flag indicating
            whether the query point is outside the LUT points convex hull, the
            lookup values, and optionally the distance to the nearest neighbor.
            Values beyond max_distance are NaN.

        Raises
        ------
        ValueError
            If x and y do not have the same shape.
        ValueError
            If distance is not one of "grid", "euclidean", or "manhattan".

        Notes
        -----
        The method automatically converts scalar inputs to 1D arrays.
        Input values that are NaN or inf will produce NaN output rows.
        """
        if distance not in _VALID_2D_DISTANCES:
            raise ValueError("distance must be one of 'grid', 'euclidean', or 'manhattan'")

        # Preserve index if x/y are pandas objects
        index = _get_query_index(x)
        if index is None:
            index = _get_query_index(y)

        # Ensure floating array
        x = np.atleast_1d(x).astype(float)
        y = np.atleast_1d(y).astype(float)

        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        # Identify invalid inputs (NaN or inf)
        invalid_mask = ~np.isfinite(x) | ~np.isfinite(y)

        # Initialize output array with NaN
        n_points = len(x)
        n_columns = len(self.columns)
        data = np.full((n_points, n_columns), np.nan, dtype=self.dtype)
        dist = np.full(n_points, np.nan, dtype=self.dtype)
        out_of_bound = np.zeros(n_points, dtype=bool)

        # Only process valid points
        valid_mask = ~invalid_mask
        if np.any(valid_mask):
            pts_valid = np.column_stack([x[valid_mask], y[valid_mask]])
            out_of_bound[valid_mask] = self._query_out_of_bound(pts_valid)
            if distance == "grid":
                dist_valid, idx_valid = self._query_grid(pts_valid)
            else:
                p = 2 if distance == "euclidean" else 1
                dist_valid, idx_valid = self.tree.query(pts_valid, p=p)

            data_valid = np.full((len(pts_valid), n_columns), np.nan, dtype=self.dtype)
            found_mask = idx_valid >= 0
            if np.any(found_mask):
                data_valid[found_mask] = self.values[idx_valid[found_mask]]

            # Apply max_distance masking if specified
            if max_distance is not None:
                if isinstance(max_distance, (tuple, list)):
                    # Univariate distance masking
                    if len(max_distance) != 2:
                        raise ValueError("max_distance tuple must have exactly 2 elements (dx, dy)")
                    dx_max, dy_max = max_distance

                    # Get nearest neighbor coordinates
                    nearest_points = np.full((len(pts_valid), 2), np.nan, dtype=self.dtype)
                    if np.any(found_mask):
                        nearest_points[found_mask] = self.points[idx_valid[found_mask]]
                    dx = np.abs(x[valid_mask] - nearest_points[:, 0])
                    dy = np.abs(y[valid_mask] - nearest_points[:, 1])

                    # Mask points exceeding either threshold
                    distance_mask = (dx > dx_max) | (dy > dy_max)
                else:
                    # Selected distance masking
                    distance_mask = dist_valid > max_distance

                # Set masked values to NaN
                data_valid[distance_mask] = np.nan

            # Assign valid data to output array
            data[valid_mask] = data_valid
            dist[valid_mask] = dist_valid

        df_out = pd.DataFrame(
            data,
            columns=self.columns,
            index=index,
        )

        # Add coordinates
        df_out[self.x] = x
        df_out[self.y] = y
        df_out["out_of_bound"] = out_of_bound

        # Keep original order
        df_out = df_out[[self.x, self.y, "out_of_bound", *self.columns]]

        if return_distance:
            df_out["distance"] = dist

        return df_out

    # ------------------
    # Convenience
    # ------------------
    def predict_dict(self, x, y):
        """Query the lookup table and return result as a dictionary.

        Parameters
        ----------
        x : scalar
            x-coordinate of the query point.
        y : scalar
            y-coordinate of the query point.

        Returns
        -------
        dict
            Dictionary with column names as keys and nearest neighbor values
            as values, including x and y coordinates.

        Raises
        ------
        ValueError
            If x or y are not scalars.

        Notes
        -----
        This is a convenience method for single-point queries returning a dict
        instead of a DataFrame.
        """
        if np.size(x) != 1:
            raise ValueError("predict_dict accepts only scalars")
        df = self.predict(x, y).iloc[0]
        return df.to_dict()

    def __len__(self):
        """Return the number of points in the lookup table.

        Returns
        -------
        int
            Number of points in the lookup table.
        """
        return self.points.shape[0]
