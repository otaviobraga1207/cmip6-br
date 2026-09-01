"""ETCCDI climate-extreme indices.

Implemented on plain xarray so they work on a station series, a single grid
cell or a full lat/lon field without changes. Definitions follow the ETCCDI /
CCl-CLIVAR-JCOMM list; ``freq`` follows pandas offset aliases (``"YS"`` for
annual, ``"MS"`` for monthly, ``"QS-DEC"`` for seasons starting in December).

All precipitation indices expect daily totals in mm; temperature indices
expect degrees Celsius. Use :func:`cmip6_br.units.to_mm_day` and
:func:`cmip6_br.units.to_celsius` on raw CMIP6 output first.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "prcptot",
    "rx1day",
    "rxnday",
    "sdii",
    "rnnmm",
    "r10mm",
    "r20mm",
    "r95p",
    "r99p",
    "cdd",
    "cwd",
    "txx",
    "tnn",
    "dtr",
    "summer_days",
    "tropical_nights",
    "all_precip_indices",
]

_WET = 1.0  # mm/day, the ETCCDI wet-day threshold


def _resample(da: xr.DataArray, freq: str):
    return da.resample(time=freq)


def _finish(out: xr.DataArray, name: str, units: str, long_name: str) -> xr.DataArray:
    out.name = name
    out.attrs.update(units=units, long_name=long_name, standard_definition="ETCCDI")
    return out


def prcptot(pr: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    """Total precipitation on wet days (>= 1 mm)."""
    out = _resample(pr.where(pr >= _WET, 0.0), freq).sum(dim="time", skipna=False)
    return _finish(out, "prcptot", "mm", "Total wet-day precipitation")


def rx1day(pr: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    """Maximum 1-day precipitation."""
    out = _resample(pr, freq).max(dim="time")
    return _finish(out, "rx1day", "mm", "Maximum 1-day precipitation")


def rxnday(pr: xr.DataArray, window: int = 5, freq: str = "YS") -> xr.DataArray:
    """Maximum consecutive ``window``-day precipitation (Rx5day by default)."""
    rolled = pr.rolling(time=window, min_periods=window).sum()
    out = _resample(rolled, freq).max(dim="time")
    return _finish(out, f"rx{window}day", "mm", f"Maximum {window}-day precipitation")


def sdii(pr: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    """Simple daily intensity index: mean precipitation on wet days."""
    wet = pr.where(pr >= _WET)
    total = _resample(wet, freq).sum(dim="time")
    count = _resample(xr.where(pr >= _WET, 1, 0), freq).sum(dim="time")
    out = total / count.where(count > 0)
    return _finish(out, "sdii", "mm/day", "Simple daily intensity index")


def rnnmm(pr: xr.DataArray, threshold: float = 10.0, freq: str = "YS") -> xr.DataArray:
    """Number of days with precipitation >= ``threshold`` mm."""
    out = _resample(xr.where(pr >= threshold, 1, 0), freq).sum(dim="time")
    return _finish(
        out, f"r{int(threshold)}mm", "days", f"Days with precipitation >= {threshold} mm"
    )


def r10mm(pr: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    """Heavy precipitation days (>= 10 mm)."""
    return rnnmm(pr, 10.0, freq)


def r20mm(pr: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    """Very heavy precipitation days (>= 20 mm)."""
    return rnnmm(pr, 20.0, freq)


def wet_day_percentile(pr: xr.DataArray, percentile: float = 95.0) -> xr.DataArray:
    """The wet-day percentile used as the reference threshold for R95p/R99p.

    Compute this over your *reference period only* (conventionally 1961-1990 or
    1981-2010) and pass it explicitly when evaluating a future period —
    otherwise the threshold moves with the climate and the index stops meaning
    anything.
    """
    wet = pr.where(pr >= _WET)
    out = wet.quantile(percentile / 100.0, dim="time")
    return out.drop_vars("quantile", errors="ignore")


def _rXXp(pr, threshold, freq, name, long_name):
    excess = pr.where((pr >= _WET) & (pr > threshold), 0.0)
    out = _resample(excess, freq).sum(dim="time", skipna=False)
    return _finish(out, name, "mm", long_name)


def r95p(
    pr: xr.DataArray, threshold: xr.DataArray | float | None = None, freq: str = "YS"
) -> xr.DataArray:
    """Precipitation falling on very wet days (above the wet-day 95th percentile)."""
    if threshold is None:
        threshold = wet_day_percentile(pr, 95.0)
    return _rXXp(pr, threshold, freq, "r95p", "Precipitation from very wet days")


def r99p(
    pr: xr.DataArray, threshold: xr.DataArray | float | None = None, freq: str = "YS"
) -> xr.DataArray:
    """Precipitation falling on extremely wet days (above the wet-day 99th percentile)."""
    if threshold is None:
        threshold = wet_day_percentile(pr, 99.0)
    return _rXXp(pr, threshold, freq, "r99p", "Precipitation from extremely wet days")


def _max_run(mask: np.ndarray) -> float:
    """Longest run of True in a 1-D boolean array (NaN-safe on all-NaN input)."""
    mask = np.asarray(mask)
    if mask.size == 0 or not np.any(np.isfinite(mask.astype(float))):
        return np.nan
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        if run > best:
            best = run
    return float(best)


def _spell(pr: xr.DataArray, condition, freq: str, name: str, long_name: str):
    mask = condition(pr)
    out = _resample(mask, freq).map(
        lambda block: xr.apply_ufunc(
            _max_run,
            block,
            input_core_dims=[["time"]],
            vectorize=True,
            output_dtypes=[float],
        )
    )
    return _finish(out, name, "days", long_name)


def cdd(pr: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    """Consecutive dry days: longest run with precipitation < 1 mm."""
    return _spell(pr, lambda x: x < _WET, freq, "cdd", "Maximum consecutive dry days")


def cwd(pr: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    """Consecutive wet days: longest run with precipitation >= 1 mm."""
    return _spell(pr, lambda x: x >= _WET, freq, "cwd", "Maximum consecutive wet days")


def txx(tasmax: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    """Warmest daily maximum temperature."""
    out = _resample(tasmax, freq).max(dim="time")
    return _finish(out, "txx", "degC", "Maximum daily maximum temperature")


def tnn(tasmin: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    """Coldest daily minimum temperature."""
    out = _resample(tasmin, freq).min(dim="time")
    return _finish(out, "tnn", "degC", "Minimum daily minimum temperature")


def dtr(tasmax: xr.DataArray, tasmin: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    """Mean diurnal temperature range."""
    out = _resample(tasmax - tasmin, freq).mean(dim="time")
    return _finish(out, "dtr", "degC", "Mean diurnal temperature range")


def summer_days(tasmax: xr.DataArray, threshold: float = 25.0, freq: str = "YS"):
    """Days with maximum temperature above ``threshold`` (SU25 by default)."""
    out = _resample(xr.where(tasmax > threshold, 1, 0), freq).sum(dim="time")
    return _finish(out, "su", "days", f"Days with Tmax > {threshold} degC")


def tropical_nights(tasmin: xr.DataArray, threshold: float = 20.0, freq: str = "YS"):
    """Nights with minimum temperature above ``threshold`` (TR20 by default)."""
    out = _resample(xr.where(tasmin > threshold, 1, 0), freq).sum(dim="time")
    return _finish(out, "tr", "days", f"Nights with Tmin > {threshold} degC")


def all_precip_indices(
    pr: xr.DataArray,
    freq: str = "YS",
    p95: xr.DataArray | float | None = None,
    p99: xr.DataArray | float | None = None,
) -> xr.Dataset:
    """Every precipitation index in one dataset.

    Pass ``p95``/``p99`` computed on the reference period when you are
    evaluating a future scenario.
    """
    out = xr.Dataset(
        {
            "prcptot": prcptot(pr, freq),
            "rx1day": rx1day(pr, freq),
            "rx5day": rxnday(pr, 5, freq),
            "sdii": sdii(pr, freq),
            "r10mm": r10mm(pr, freq),
            "r20mm": r20mm(pr, freq),
            "r95p": r95p(pr, p95, freq),
            "r99p": r99p(pr, p99, freq),
            "cdd": cdd(pr, freq),
            "cwd": cwd(pr, freq),
        }
    )
    out.attrs["definitions"] = "ETCCDI (CCl/CLIVAR/JCOMM)"
    return out
