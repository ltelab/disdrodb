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
"""DISDRODB coordinates utilities."""


def add_dataset_crs_coords(xr_obj):
    """Add a CF-compliant CRS (WGS84) to an xarray.Dataset."""
    return xr_obj.assign_coords({"crs": ["WGS84"]})
    # da_crs = xr.DataArray(
    #     0,
    #     attrs={
    #         "grid_mapping_name": "latitude_longitude",
    #         "crs_wkt": (
    #             'GEOGCRS["WGS 84",'
    #             'DATUM["World Geodetic System 1984",'
    #             'ELLIPSOID["WGS 84",6378137,298.257223563]],'
    #             'PRIMEM["Greenwich",0],'
    #             "CS[ellipsoidal,2],"
    #             'AXIS["latitude",north],'
    #             'AXIS["longitude",east],'
    #             'UNIT["degree",0.0174532925199433]]'
    #         ),
    #         "epsg_code": "EPSG:4326",
    #         "semi_major_axis": 6378137.0,
    #         "inverse_flattening": 298.257223563,
    #         "longitude_of_prime_meridian": 0.0,
    #     },
    # )
    # return xr_obj.assign_coords({"crs": da_crs})
