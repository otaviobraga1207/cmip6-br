"""Maps and validation figures.

Design rules this module enforces so you do not have to think about them:

* **Sequential fields get one hue, light to dark.** No rainbow, no ``jet``.
  A rainbow colormap invents boundaries where the data has none and hides real
  ones -- it is the single most common way a hazard map misleads its reader.
* **Change maps get a diverging ramp centred on zero**, with a neutral grey
  midpoint and symmetric limits. An anomaly map whose colour bar is not centred
  on zero is not an anomaly map.
* **Every colour bar is labelled with units.** A map of "precipitation" without
  mm on the bar is decoration.
* **Series are distinguished by colour *and* by line style or marker**, so the
  figures survive greyscale printing and colour-vision deficiency.

Requires the ``viz`` extra::

    pip install "cmip6-br[viz]"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "SEQUENTIAL",
    "DIVERGING",
    "SERIES",
    "map_field",
    "map_change",
    "map_panel",
    "qq_plot",
    "seasonal_cycle",
    "save_geotiff",
]

# One hue, light to dark. Validated for lightness monotonicity.
_SEQUENTIAL_STEPS = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
# Warm/cool poles with a neutral grey midpoint: dry <- 0 -> wet.
_DIVERGING_STEPS = [
    "#8c1d1c",
    "#b3302f",
    "#e34948",
    "#ee8180",
    "#f6bab9",
    "#f0efec",
    "#b7d3f6",
    "#86b6ef",
    "#3987e5",
    "#256abf",
    "#184f95",
]
#: Categorical slots for line figures. Validated all-pairs for colour-vision
#: deficiency; never extend past three without folding the rest into "other".
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
_SERIES_STYLE = (("-", "o"), ("--", "s"), ("-.", "^"))

_INK = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#dcdbd6"


def _require_mpl():
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            'Plotting needs matplotlib. Install it with: pip install "cmip6-br[viz]"'
        ) from exc
    return mpl, plt


def _colormaps():
    mpl, _ = _require_mpl()
    from matplotlib.colors import LinearSegmentedColormap

    sequential = LinearSegmentedColormap.from_list("cmip6br_seq", _SEQUENTIAL_STEPS)
    diverging = LinearSegmentedColormap.from_list("cmip6br_div", _DIVERGING_STEPS)
    sequential.set_bad("#f0efec")
    diverging.set_bad("#f0efec")
    return sequential, diverging


#: The package colormaps, built lazily so importing cmip6_br never needs matplotlib.
SEQUENTIAL = "cmip6br_seq"
DIVERGING = "cmip6br_div"


def _reduce_to_2d(da: xr.DataArray, reduce: str = "mean") -> xr.DataArray:
    """Collapse everything except lat/lon, so callers can pass a full time series."""
    extra = [d for d in da.dims if d not in ("lat", "lon")]
    if not extra:
        return da
    if reduce == "mean":
        return da.mean(dim=extra)
    if reduce == "median":
        return da.median(dim=extra)
    if reduce == "max":
        return da.max(dim=extra)
    raise ValueError(f"unknown reduce={reduce!r}")


def _label(da: xr.DataArray, units: str | None) -> str:
    units = units or da.attrs.get("units", "")
    name = da.attrs.get("long_name") or (da.name or "value")
    return f"{name} ({units})" if units else str(name)


def _style_axes(ax, da: xr.DataArray, title: str | None):
    ax.set_xlabel("longitude", color=_MUTED, fontsize=9)
    ax.set_ylabel("latitude", color=_MUTED, fontsize=9)
    ax.tick_params(colors=_MUTED, labelsize=8, length=3)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    # Degrees of longitude shrink with latitude; an equal-degree aspect stretches
    # Brazil east-west. This is the cheap correction, short of a real projection.
    mean_lat = float(np.deg2rad(da["lat"].mean()))
    ax.set_aspect(1.0 / max(np.cos(mean_lat), 0.2))
    if title:
        ax.set_title(title, color=_INK, fontsize=11, loc="left", pad=8)


def _draw_boundaries(ax, boundaries):
    """Overlay administrative limits from a GeoDataFrame or a vector file path."""
    if boundaries is None:
        return
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'Boundary overlays need geopandas. Install with: pip install "cmip6-br[geo]"'
        ) from exc
    gdf = boundaries if hasattr(boundaries, "geometry") else gpd.read_file(boundaries)
    gdf.boundary.plot(ax=ax, color=_MUTED, linewidth=0.6, zorder=5)


def map_field(
    da: xr.DataArray,
    title: str | None = None,
    units: str | None = None,
    reduce: str = "mean",
    vmin: float | None = None,
    vmax: float | None = None,
    boundaries=None,
    ax: Axes | None = None,
    robust: bool = True,
) -> Axes:
    """Map a magnitude field on a single-hue sequential ramp.

    Parameters
    ----------
    da
        A DataArray with ``lat``/``lon``. Extra dimensions (time, member) are
        collapsed with ``reduce``.
    robust
        Clip the colour limits to the 2nd-98th percentile. One freak grid cell
        should not flatten the whole map into a single colour.
    boundaries
        Optional GeoDataFrame or vector file with administrative limits to
        overlay (state borders, a basin outline).
    """
    _, plt = _require_mpl()
    sequential, _ = _colormaps()
    field = _reduce_to_2d(da, reduce)

    if vmin is None or vmax is None:
        lo, hi = (2, 98) if robust else (0, 100)
        finite = field.values[np.isfinite(field.values)]
        auto = np.percentile(finite, [lo, hi]) if finite.size else (0.0, 1.0)
        vmin = auto[0] if vmin is None else vmin
        vmax = auto[1] if vmax is None else vmax

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(
        field["lon"],
        field["lat"],
        field.values,
        cmap=sequential,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    _draw_boundaries(ax, boundaries)
    _style_axes(ax, field, title)
    bar = ax.figure.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02)
    bar.set_label(_label(da, units), color=_MUTED, fontsize=9)
    bar.ax.tick_params(colors=_MUTED, labelsize=8)
    bar.outline.set_edgecolor(_GRID)
    return ax


def map_change(
    reference: xr.DataArray,
    scenario: xr.DataArray,
    title: str | None = None,
    relative: bool = True,
    reduce: str = "mean",
    vmax: float | None = None,
    boundaries=None,
    ax: Axes | None = None,
) -> Axes:
    """Map scenario-minus-reference change on a diverging ramp centred on zero.

    ``relative=True`` maps percentage change (the usual choice for
    precipitation); ``False`` maps the absolute difference in the field's own
    units (the usual choice for temperature).

    The colour limits are always symmetric around zero, so equal wetting and
    drying get equal visual weight.
    """
    _, plt = _require_mpl()
    _, diverging = _colormaps()
    ref = _reduce_to_2d(reference, reduce)
    scen = _reduce_to_2d(scenario, reduce)

    if relative:
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = (scen / ref.where(ref != 0) - 1) * 100
        units = "%"
    else:
        delta = scen - ref
        units = reference.attrs.get("units", "")

    finite = delta.values[np.isfinite(delta.values)]
    if vmax is None:
        vmax = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
        vmax = max(vmax, 1e-6)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(
        delta["lon"],
        delta["lat"],
        delta.values,
        cmap=diverging,
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    _draw_boundaries(ax, boundaries)
    _style_axes(ax, delta, title)
    label = "change (%)" if relative else f"change ({units})" if units else "change"
    bar = ax.figure.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02)
    bar.set_label(label, color=_MUTED, fontsize=9)
    bar.ax.tick_params(colors=_MUTED, labelsize=8)
    bar.outline.set_edgecolor(_GRID)
    return ax


def map_panel(
    fields: dict[str, xr.DataArray],
    reduce: str = "mean",
    ncols: int = 2,
    shared_scale: bool = True,
    boundaries=None,
    suptitle: str | None = None,
) -> Figure:
    """Small multiples of several fields, on one shared colour scale by default.

    A shared scale is what makes panels comparable; turn it off only when the
    fields genuinely have different ranges, and say so in the caption. Each
    panel takes its units from its own ``units`` attribute, so a mixed panel
    (millimetres beside days) still labels every colour bar correctly.
    """
    _, plt = _require_mpl()
    items = list(fields.items())
    ncols = max(1, min(ncols, len(items)))
    nrows = int(np.ceil(len(items) / ncols))

    vmin = vmax = None
    if shared_scale:
        stacked = np.concatenate([_reduce_to_2d(da, reduce).values.ravel() for _, da in items])
        finite = stacked[np.isfinite(stacked)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [2, 98])

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.6 * ncols, 4.6 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, da) in zip(axes, items, strict=False):
        map_field(da, title=name, reduce=reduce, vmin=vmin, vmax=vmax, boundaries=boundaries, ax=ax)
    for ax in axes[len(items) :]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, color=_INK, fontsize=13, x=0.01, ha="left")
    return fig


def qq_plot(
    obs: xr.DataArray,
    raw: xr.DataArray,
    corrected: xr.DataArray,
    title: str = "Quantile-quantile, model vs observations",
    units: str = "mm/day",
    ax: Axes | None = None,
) -> Axes:
    """The figure that shows whether bias correction actually worked.

    Points on the 1:1 line mean the simulated distribution matches the observed
    one. The raw series bending away from the line in the upper tail is the
    classic under-representation of extremes; the corrected series should lie on
    it. This is the plot to put in front of a reviewer, not a map.
    """
    _, plt = _require_mpl()
    quantiles = np.linspace(0.01, 0.999, 120)

    def flat(da):
        values = np.asarray(da.values, dtype=float).ravel()
        return values[np.isfinite(values)]

    obs_q = np.quantile(flat(obs), quantiles)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    limit = (
        float(max(obs_q.max(), np.quantile(flat(raw), 0.999), np.quantile(flat(corrected), 0.999)))
        * 1.05
    )
    ax.plot([0, limit], [0, limit], color=_MUTED, linewidth=1.0, zorder=1)
    ax.annotate("1:1", xy=(limit * 0.86, limit * 0.9), color=_MUTED, fontsize=9)

    for (label, series), colour, (style, marker) in zip(
        (("model raw", raw), ("model corrected", corrected)),
        SERIES[1:],
        _SERIES_STYLE[1:],
        strict=False,
    ):
        ax.plot(
            obs_q,
            np.quantile(flat(series), quantiles),
            color=colour,
            linestyle=style,
            linewidth=2.0,
            marker=marker,
            markevery=15,
            markersize=5,
            label=label,
            zorder=3,
        )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel(f"observed quantile ({units})", color=_MUTED, fontsize=9)
    ax.set_ylabel(f"simulated quantile ({units})", color=_MUTED, fontsize=9)
    ax.set_title(title, color=_INK, fontsize=11, loc="left", pad=8)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(True, color=_GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(_INK)
    return ax


def seasonal_cycle(
    series: dict[str, xr.DataArray],
    title: str = "Mean seasonal cycle",
    units: str = "mm/day",
    ax: Axes | None = None,
) -> Axes:
    """Monthly means for up to three series, to check the wet/dry season shape."""
    _, plt = _require_mpl()
    if len(series) > 3:
        raise ValueError(
            "Plot at most three series; beyond that use small multiples "
            "(see map_panel) rather than more colours."
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)

    months = np.arange(1, 13)
    for (label, da), colour, (style, marker) in zip(
        series.items(), SERIES, _SERIES_STYLE, strict=False
    ):
        monthly = da.groupby("time.month").mean()
        values = (
            monthly.mean(dim=[d for d in monthly.dims if d != "month"])
            if monthly.ndim > 1
            else monthly
        )
        ax.plot(
            months,
            values.sel(month=months).values,
            color=colour,
            linestyle=style,
            linewidth=2.0,
            marker=marker,
            markersize=5,
            label=label,
        )

    ax.set_xticks(months)
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel(units, color=_MUTED, fontsize=9)
    ax.set_title(title, color=_INK, fontsize=11, loc="left", pad=8)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(_INK)
    return ax


def save_geotiff(da: xr.DataArray, path: str, reduce: str = "mean", epsg: int = 4326) -> str:
    """Write a 2-D field as a GeoTIFF, for QGIS and for anyone who does not use Python.

    ``epsg=31983`` (SIRGAS 2000 / UTM 23S) is the usual choice for São Paulo
    State deliverables; the field is reprojected on the way out.
    """
    try:
        import rioxarray  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'GeoTIFF export needs rioxarray. Install with: pip install "cmip6-br[geo]"'
        ) from exc
    field = _reduce_to_2d(da, reduce)
    field = field.rio.write_crs("EPSG:4326").rio.set_spatial_dims("lon", "lat")
    if epsg != 4326:
        field = field.rio.reproject(f"EPSG:{epsg}")
    field.rio.to_raster(path)
    return path
