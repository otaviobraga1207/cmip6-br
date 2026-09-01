"""Coordinate normalisation, subsetting and regridding of CMIP6 fields.

CMIP6 archives are inconsistent in ways that quietly break Brazilian workflows:
longitudes come on a 0-360 axis, latitude may run north-to-south, and the
horizontal dimensions are named ``lon``/``longitude``/``x`` depending on the
modelling centre. Everything in this module normalises first, then works.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from .grids import BBox, TargetGrid

__all__ = [
    "normalize_coords",
    "subset_bbox",
    "regrid",
    "area_weights",
    "spatial_mean",
]

_LON_NAMES = ("lon", "longitude", "long", "x", "nav_lon", "XLONG")
_LAT_NAMES = ("lat", "latitude", "y", "nav_lat", "XLAT")


def _find_dim(obj: xr.DataArray | xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in obj.coords or name in obj.dims:
            return name
    raise KeyError(
        f"Could not find any of {candidates} in {list(obj.coords)}. "
        "Rename the coordinate before calling cmip6_br."
    )


def normalize_coords(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """Return ``obj`` with coordinates named ``lon``/``lat``, lon in [-180, 180),
    and both axes sorted ascending.

    This is idempotent and safe to call on already-normalised data.
    """
    lon_name = _find_dim(obj, _LON_NAMES)
    lat_name = _find_dim(obj, _LAT_NAMES)
    renames = {}
    if lon_name != "lon":
        renames[lon_name] = "lon"
    if lat_name != "lat":
        renames[lat_name] = "lat"
    if renames:
        obj = obj.rename(renames)

    lon = obj["lon"]
    if float(lon.max()) > 180.0:
        obj = obj.assign_coords(lon=(((obj["lon"] + 180) % 360) - 180))

    obj = obj.sortby("lon").sortby("lat")
    obj["lon"].attrs.setdefault("units", "degrees_east")
    obj["lat"].attrs.setdefault("units", "degrees_north")
    return obj


def subset_bbox(
    obj: xr.DataArray | xr.Dataset, box: BBox, pad: float = 1.0
) -> xr.DataArray | xr.Dataset:
    """Subset to a bounding box, padded so that interpolation has neighbours.

    ``pad`` should be at least one source grid spacing; the default of 1 degree
    is safe for every CMIP6 resolution in common use.
    """
    obj = normalize_coords(obj)
    b = box.pad(pad)
    out = obj.sel(
        lon=slice(b.lon_min, b.lon_max),
        lat=slice(b.lat_min, b.lat_max),
    )
    if out["lon"].size == 0 or out["lat"].size == 0:
        raise ValueError(
            "Bounding box selects no grid points. Check that the source data "
            "actually covers the region."
        )
    return out


def regrid(
    obj: xr.DataArray | xr.Dataset,
    target: TargetGrid,
    method: Literal["linear", "nearest"] = "linear",
    fill_edges: bool = True,
) -> xr.DataArray | xr.Dataset:
    """Interpolate ``obj`` onto ``target``.

    Parameters
    ----------
    method
        ``"linear"`` (bilinear) for continuous fields such as temperature;
        ``"nearest"`` when the field is categorical or already at higher
        resolution than the target.
    fill_edges
        Fill target cells that fall just outside the source grid with the
        nearest source value. Coastal cells in Brazil routinely fall outside a
        coarse global grid, and leaving them as NaN silently eats the shoreline.

    Notes
    -----
    This is a point interpolation, not a conservative remapping. For
    flux-conserving remapping of precipitation onto a much coarser grid, use
    ``xesmf`` with ``method="conservative"``; for the downscaling direction
    (coarse to fine) bilinear is the standard choice and what the bias
    correction in :mod:`cmip6_br.bias` expects.
    """
    obj = normalize_coords(obj)
    out = obj.interp(
        lon=target.lon,
        lat=target.lat,
        method=method,
        kwargs={"bounds_error": False},
    )
    if fill_edges:
        nearest = obj.interp(
            lon=target.lon,
            lat=target.lat,
            method="nearest",
            kwargs={"bounds_error": False, "fill_value": None},
        )
        out = out.fillna(nearest)
    out.attrs["regrid_method"] = method
    out.attrs["regrid_target"] = target.name
    return out


def area_weights(obj: xr.DataArray | xr.Dataset) -> xr.DataArray:
    """Cosine-of-latitude weights, for honest spatial averages."""
    obj = normalize_coords(obj)
    w = np.cos(np.deg2rad(obj["lat"]))
    w.name = "area_weights"
    return w


def spatial_mean(da: xr.DataArray) -> xr.DataArray:
    """Area-weighted mean over ``lat``/``lon``."""
    da = normalize_coords(da)
    return da.weighted(area_weights(da)).mean(dim=("lat", "lon"))
