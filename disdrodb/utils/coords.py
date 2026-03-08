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
    import xarray as xr

    crs_wkt = (
        'GEOGCRS["WGS 84",'
        'DATUM["World Geodetic System 1984",'
        'ELLIPSOID["WGS 84",6378137,298.257223563, LENGTHUNIT["metre",1]]],'
        'PRIMEM["Greenwich",0, ANGLEUNIT["degree",0.0174532925199433]],'
        "CS[ellipsoidal,2],"
        'AXIS["geodetic latitude",north, ANGLEUNIT["degree",0.0174532925199433]],'
        'AXIS["geodetic longitude",east, ANGLEUNIT["degree",0.0174532925199433]],'
        'UNIT["degree",0.0174532925199433]'
    )
    da_crs = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "crs_wkt": crs_wkt,
            "epsg_code": "EPSG:4326",
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
            "longitude_of_prime_meridian": 0.0,
        },
    )
    return xr_obj.assign_coords({"crs": da_crs})


def add_time_bnds(ds):
    """Add time bounds coordinate to xarray.Dataset."""
    import xarray as xr

    from disdrodb.utils.encoding import add_time_encoding

    if "time_bnds" in ds.coords:
        ds = ds.drop_vars("time_bnds")

    # Ensure sample_interval is timedelta64
    dt = ds["sample_interval"].astype("timedelta64[s]")

    # Define time_bnds coordinate
    start = ds["time"] - dt
    end = ds["time"]

    time_bnds = xr.concat([start, end], dim="nv").transpose("time", "nv")
    ds = ds.assign_coords(
        {"time_bnds": time_bnds},
    )

    # Add time encoding
    ds = add_time_encoding(ds, var="time_bnds")

    # Add CF attributes
    ds["time"].attrs.update({"bounds": "time_bnds"})
    ds["time_bnds"].attrs = {"long_name": "time bounds"}
    return ds
