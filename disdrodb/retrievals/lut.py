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
"""Routines for 2D Look Up Tables (LUT)."""

import pickle

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

### in predict()
# - if input NaN, return NaN in predict()
# - add argument max_distance(). If distance > specified, set to np.nan
#   --> allow specify tuple for x and y (univariate distance)
#   --> if one value, applied to dist currently computed
#   --> if none, do not mask


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

    Attributes
    ----------
    x : str
        Name of the x-coordinate column.
    y : str
        Name of the y-coordinate column.
    columns : list of str
        Column names for the lookup values.
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

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, 2], 'value': [10, 20, 30]})
    >>> lut = NearestNeighbourLUT2D(df, 'x', 'y', columns=['value'])
    >>> lut.predict([0.5], [0.5])
    """

    def __init__(self, df, x, y, columns=None, dtype=np.float32):
        if columns is None:
            columns = list(df.columns)
        if x in df.index.names:
            df = df.reset_index()
        # Check x and y
        if x not in df:
            raise ValueError(f"{x=} is not a column of df.")
        if y not in df:
            raise ValueError(f"{y=} is not a column of df.")

        # Subset columns
        if not np.all(np.isin(columns, df.columns)):
            missing = np.setdiff1d(columns, df.columns).tolist()
            raise ValueError(f"{missing} columns not found in DataFrame")

        self.x = x
        self.y = y
        self.columns = columns
        self.dtype = dtype
        self.points = df[[x, y]].to_numpy(dtype=dtype)
        self.values = df[columns].to_numpy(dtype=dtype)

        self.tree = cKDTree(self.points)

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
    def predict(self, x, y, return_distance=False, max_distance=None):
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
            - If float: points with Euclidean distance > max_distance are set to NaN.
            - If tuple (dx, dy): points are masked if |x - x_nearest| > dx OR |y - y_nearest| > dy.

            Default is None.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the nearest neighbor values for each query point.
            Columns include x, y coordinates, the lookup values, and optionally
            the distance to the nearest neighbor. Values beyond max_distance are NaN.

        Raises
        ------
        ValueError
            If x and y do not have the same shape.

        Notes
        -----
        The method automatically converts scalar inputs to 1D arrays.
        Input values that are NaN or inf will produce NaN output rows.
        """
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

        # Only process valid points
        valid_mask = ~invalid_mask
        if np.any(valid_mask):
            pts_valid = np.column_stack([x[valid_mask], y[valid_mask]])
            dist_valid, idx_valid = self.tree.query(pts_valid)
            data_valid = self.values[idx_valid]

            # Apply max_distance masking if specified
            if max_distance is not None:
                if isinstance(max_distance, (tuple, list)):
                    # Univariate distance masking
                    if len(max_distance) != 2:
                        raise ValueError("max_distance tuple must have exactly 2 elements (dx, dy)")
                    dx_max, dy_max = max_distance

                    # Get nearest neighbor coordinates
                    nearest_points = self.points[idx_valid]
                    dx = np.abs(x[valid_mask] - nearest_points[:, 0])
                    dy = np.abs(y[valid_mask] - nearest_points[:, 1])

                    # Mask points exceeding either threshold
                    distance_mask = (dx > dx_max) | (dy > dy_max)
                else:
                    # Euclidean distance masking
                    distance_mask = dist_valid > max_distance

                # Set masked values to NaN
                data_valid[distance_mask] = np.nan

            # Assign valid data to output array
            data[valid_mask] = data_valid
            dist[valid_mask] = dist_valid

        df_out = pd.DataFrame(
            data,
            columns=self.columns,
        )

        # Add coordinates
        df_out[self.x] = x
        df_out[self.y] = y

        # Keep original order
        df_out = df_out[[self.x, self.y, *self.columns]]

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
