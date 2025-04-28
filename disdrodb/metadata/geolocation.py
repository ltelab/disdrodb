#!/usr/bin/env python3

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
"""Metadata tools to verify/complete geolocation information."""
import time

import numpy as np
import requests


def infer_altitude(latitude, longitude, dem="aster30m"):
    """Infer station altitude using a Digital Elevation Model (DEM).

    This function uses the OpenTopoData API to infer the altitude of a given
    location specified by latitude and longitude.
    By default, it uses the ASTER DEM at 30m resolution.

    Parameters
    ----------
    latitude : float
        The latitude of the location for which to infer the altitude.
    longitude : float
        The longitude of the location for which to infer the altitude.
    dem : str, optional
        The DEM to use for altitude inference. Options are "aster30m" (default),
        "srtm30", and "mapzen".

    Returns
    -------
    elevation : float
        The inferred altitude of the specified location.

    Raises
    ------
    ValueError
        If the altitude retrieval fails.

    Notes
    -----
    - The OpenTopoData API has a limit of 1000 calls per day.
    - Each request can include up to 100 locations.
    - The API allows a maximum of 1 call per second.

    References
    ----------
    https://www.opentopodata.org/api/
    """
    import requests

    url = f"https://api.opentopodata.org/v1/{dem}?locations={latitude},{longitude}"
    r = requests.get(url)

    data = r.json()
    if data["status"] == "OK":
        elevation = data["results"][0]["elevation"]
    else:
        raise ValueError("Altitude retrieval failed.")
    return elevation


def infer_altitudes(lats, lons, dem="aster30m"):
    """
    Infer altitude of a given location using OpenTopoData API.

    Parameters
    ----------
    lats : list or array-like
        List or array of latitude coordinates.
    lons : list or array-like
        List or array of longitude coordinates.
    dem : str, optional
        Digital Elevation Model (DEM) to use for altitude inference.
        The default DEM is "aster30m".

    Returns
    -------
    elevations : numpy.ndarray
        Array of inferred altitudes corresponding to the input coordinates.

    Raises
    ------
    ValueError
        If the latitude and longitude arrays do not have the same length.
        If altitude retrieval fails for any block of coordinates.

    Notes
    -----
    - The OpenTopoData API has a limit of 1000 calls per day.
    - Each request can include up to 100 locations.
    - The API allows a maximum of 1 call per second.
    - The API requests are made in blocks of up to 100 coordinates,
    with a 2-second delay between requests.
    """
    # Check that lats and lons have the same length
    if len(lats) != len(lons):
        raise ValueError("Latitude and longitude arrays must have the same length.")

    # Maximum number of locations per API request
    max_locations = 100
    elevations = []

    # Total number of coordinates
    total_coords = len(lats)

    # Loop over the coordinates in blocks of max_locations
    for i in range(0, total_coords, max_locations):

        # Wait 2 seconds before another API request
        time.sleep(2)

        # Get the block of coordinates
        block_lats = lats[i : i + max_locations]
        block_lons = lons[i : i + max_locations]

        # Create the list_coords string in the format "lat1,lon1|lat2,lon2|..."
        list_coords = "|".join([f"{lat},{lon}" for lat, lon in zip(block_lats, block_lons)])

        # Define API URL
        url = f"https://api.opentopodata.org/v1/{dem}?locations={list_coords}&interpolation=nearest"

        # Retrieve info
        r = requests.get(url)
        data = r.json()

        # Parse info
        if data.get("status") == "OK":
            elevations.extend([result["elevation"] for result in data["results"]])
        else:
            raise ValueError(f"Altitude retrieval failed for block starting at index {i}.")
    elevations = np.array(elevations).astype(float)
    return elevations
