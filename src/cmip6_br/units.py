"""Unit conversions for raw CMIP6 output.

CMIP6 stores precipitation as a flux in kg m-2 s-1 and temperature in kelvin.
Every index in this package expects mm/day and degrees Celsius, and forgetting
the 86400 factor is the single most common way to produce a beautiful map of
nonsense.
"""

from __future__ import annotations

import xarray as xr

__all__ = ["to_mm_day", "to_celsius", "harmonize"]

_FLUX_UNITS = {"kg m-2 s-1", "kg/m2/s", "kg m**-2 s**-1", "mm/s"}
_KELVIN = {"k", "kelvin", "degk"}


def to_mm_day(pr: xr.DataArray, assume: str | None = None) -> xr.DataArray:
    """Convert precipitation to mm/day, using ``units`` attrs when present.

    ``assume`` forces the source units when the attribute is missing or wrong.
    """
    units = (assume or pr.attrs.get("units", "")).strip()
    if units in _FLUX_UNITS:
        out = pr * 86400.0
    elif units in {"mm", "mm/day", "mm d-1", "kg m-2", ""}:
        out = pr * 1.0
    else:
        raise ValueError(f"Unrecognised precipitation units {units!r}; pass assume=...")
    out.attrs = dict(pr.attrs)
    out.attrs["units"] = "mm/day"
    out.name = pr.name
    return out


def to_celsius(temp: xr.DataArray, assume: str | None = None) -> xr.DataArray:
    """Convert temperature to degrees Celsius."""
    units = (assume or temp.attrs.get("units", "")).strip()
    if units.lower() in _KELVIN:
        out = temp - 273.15
    elif units.lower() in {"c", "degc", "celsius", "degrees_celsius", ""}:
        out = temp * 1.0
    else:
        raise ValueError(f"Unrecognised temperature units {units!r}; pass assume=...")
    out.attrs = dict(temp.attrs)
    out.attrs["units"] = "degC"
    out.name = temp.name
    return out


def harmonize(ds: xr.Dataset) -> xr.Dataset:
    """Convert every recognised variable in a dataset in one call."""
    out = ds.copy()
    for name in ("pr", "prlr", "precipitation"):
        if name in out:
            out[name] = to_mm_day(out[name])
    for name in ("tas", "tasmax", "tasmin", "ts"):
        if name in out:
            out[name] = to_celsius(out[name])
    return out
