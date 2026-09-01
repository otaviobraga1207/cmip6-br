"""Synthetic demo data, so the examples and tests run with no downloads.

The generator is not a climate model. It produces series with a Southeast
Brazil-like seasonal cycle (wet summer, dry winter) plus a deliberate model
bias -- too many drizzle days, too few intense ones -- which is exactly the
error structure quantile mapping is supposed to remove. That makes it useful
for testing, teaching and reproducing bug reports, and useless for science.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .grids import grid_for

__all__ = ["demo_precipitation", "demo_bundle"]


def _seasonal_amplitude(times: pd.DatetimeIndex) -> np.ndarray:
    """Wet season peaking in January, dry trough in July."""
    doy = times.dayofyear.to_numpy()
    return 0.35 + 0.65 * (0.5 + 0.5 * np.cos(2 * np.pi * (doy - 15) / 365.25))


def demo_precipitation(
    start: str,
    end: str,
    grid=None,
    wet_frequency: float = 0.50,
    intensity: float = 18.0,
    trend_per_decade: float = 0.0,
    seed: int = 0,
) -> xr.DataArray:
    """Daily precipitation (mm/day) on a small grid.

    ``wet_frequency`` and ``intensity`` are the two knobs that make a series
    "observation-like" or "model-like": raising the first and lowering the
    second reproduces the classic drizzle bias of a GCM.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, end, freq="D")
    grid = grid if grid is not None else grid_for("SP", 1.0)
    season = _seasonal_amplitude(times)

    lat2d, lon2d = np.meshgrid(grid.lat, grid.lon, indexing="ij")
    # A gentle south-east to north-west gradient, as over São Paulo State.
    spatial = 1.0 + 0.25 * (lat2d - grid.lat.mean()) / max(np.ptp(grid.lat), 1e-6)
    spatial += 0.15 * (lon2d - grid.lon.mean()) / max(np.ptp(grid.lon), 1e-6)

    years = (times - times[0]).days.to_numpy() / 365.25
    trend = 1.0 + trend_per_decade * years / 10.0

    shape = (times.size, grid.lat.size, grid.lon.size)
    wet = rng.random(shape) < (wet_frequency * season)[:, None, None]
    amounts = rng.gamma(
        shape=0.9,
        scale=(intensity * season)[:, None, None] * spatial[None] * trend[:, None, None],
        size=shape,
    )
    values = np.where(wet, amounts, 0.0)

    da = xr.DataArray(
        values,
        coords={"time": times, "lat": grid.lat, "lon": grid.lon},
        dims=("time", "lat", "lon"),
        name="pr",
        attrs={"units": "mm/day", "long_name": "Synthetic daily precipitation"},
    )
    return da


def demo_bundle(seed: int = 0) -> dict[str, xr.DataArray]:
    """A ready-made (obs, hist, ssp) triple sharing one grid.

    ``obs``  - 1995-2014, realistic wet-day frequency and intensity.
    ``hist`` - same period, biased: rains too often, too weakly.
    ``ssp``  - 2041-2060, same bias plus a wetting trend to be preserved.
    """
    grid = grid_for("SP", 1.0)
    obs = demo_precipitation(
        "1995-01-01", "2014-12-31", grid, wet_frequency=0.50, intensity=18.0, seed=seed
    )
    hist = demo_precipitation(
        "1995-01-01", "2014-12-31", grid, wet_frequency=0.85, intensity=9.5, seed=seed + 1
    )
    ssp = demo_precipitation(
        "2041-01-01",
        "2060-12-31",
        grid,
        wet_frequency=0.85,
        intensity=9.5,
        trend_per_decade=0.06,
        seed=seed + 2,
    )
    for name, da in (("obs", obs), ("hist", hist), ("ssp", ssp)):
        da.attrs["role"] = name
    return {"obs": obs, "hist": hist, "ssp": ssp}
