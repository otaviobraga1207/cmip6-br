"""End-to-end downscaling: normalise, subset, regrid, bias-correct, index.

The pipeline exists so that the ordinary case is one call and the unusual case
is still assembled from the same public functions. Nothing here is magic --
read :func:`downscale` top to bottom and you have the whole method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import xarray as xr

from . import bias as _bias
from . import indices as _indices
from .grids import BBox, TargetGrid, bbox, regular_grid
from .regrid import regrid, subset_bbox
from .units import to_celsius, to_mm_day

__all__ = ["DownscalingConfig", "DownscalingResult", "downscale"]


@dataclass
class DownscalingConfig:
    """Everything the pipeline needs to know.

    Attributes
    ----------
    region
        Region code understood by :func:`cmip6_br.grids.bbox` (``"SP"``,
        ``"BR"``, ...) or an explicit :class:`~cmip6_br.grids.BBox`.
    resolution
        Target grid spacing in degrees. 0.1 deg (~11 km) is a common choice for
        state-level water-sector work; do not read more skill into a finer grid
        than the bias correction can actually deliver.
    variable
        ``"pr"`` for precipitation (multiplicative mapping, wet-day frequency
        adaptation) or a temperature variable (additive mapping).
    method
        ``"qdm"`` preserves the model's change signal and is the right default
        for projections; ``"eqm"`` is for historical validation.
    reference_period
        ``(start, end)`` ISO dates used to calibrate the transfer function and
        to fix the R95p/R99p thresholds.
    """

    region: str | BBox = "SP"
    resolution: float = 0.1
    variable: Literal["pr", "tas", "tasmax", "tasmin"] = "pr"
    method: Literal["qdm", "eqm"] = "qdm"
    reference_period: tuple[str, str] | None = None
    n_quantiles: int = 100
    group: Literal["month", "none"] = "month"
    convert_units: bool = True
    index_freq: str = "YS"
    extra: dict = field(default_factory=dict)

    def target_grid(self) -> TargetGrid:
        box = self.region if isinstance(self.region, BBox) else bbox(self.region)
        name = self.region if isinstance(self.region, str) else "custom"
        return regular_grid(box, self.resolution, name=str(name))


@dataclass
class DownscalingResult:
    """What came out: the corrected fields and the indices computed from them."""

    historical: xr.DataArray
    scenario: xr.DataArray | None
    indices_historical: xr.Dataset | None
    indices_scenario: xr.Dataset | None
    config: DownscalingConfig
    mapper: object

    def to_netcdf(self, prefix: str) -> list[str]:
        """Write every non-empty product to ``{prefix}_{name}.nc``."""
        written = []
        products = {
            "historical": self.historical,
            "scenario": self.scenario,
            "indices_historical": self.indices_historical,
            "indices_scenario": self.indices_scenario,
        }
        for name, obj in products.items():
            if obj is None:
                continue
            path = f"{prefix}_{name}.nc"
            obj.to_netcdf(path)
            written.append(path)
        return written


def _convert(da: xr.DataArray, variable: str) -> xr.DataArray:
    return to_mm_day(da) if variable == "pr" else to_celsius(da)


def downscale(
    obs: xr.DataArray,
    hist: xr.DataArray,
    scenario: xr.DataArray | None = None,
    config: DownscalingConfig | None = None,
    compute_indices: bool = True,
) -> DownscalingResult:
    """Run the full workflow.

    Parameters
    ----------
    obs
        Reference observations on the target grid (gridded product, or station
        data already interpolated). Must carry ``time``.
    hist
        The model's historical run, on any grid, in any units.
    scenario
        An SSP run to project. Optional -- omit it to validate the correction
        over the historical period only.

    Returns
    -------
    DownscalingResult
    """
    cfg = config or DownscalingConfig()
    grid = cfg.target_grid()
    box = cfg.region if isinstance(cfg.region, BBox) else bbox(cfg.region)

    def prepare(da):
        if da is None:
            return None
        if cfg.convert_units:
            da = _convert(da, cfg.variable)
        da = subset_bbox(da, box)
        return regrid(da, grid, method="linear")

    hist_g = prepare(hist)
    scen_g = prepare(scenario)

    obs_ref, hist_ref = obs, hist_g
    if cfg.reference_period is not None:
        start, end = cfg.reference_period
        obs_ref = obs.sel(time=slice(start, end))
        hist_ref = hist_g.sel(time=slice(start, end))

    mapper = _bias.fit(
        obs_ref,
        hist_ref,
        variable="pr" if cfg.variable == "pr" else "tas",
        method=cfg.method,
        n_quantiles=cfg.n_quantiles,
        group=cfg.group,
    )
    hist_corrected = mapper.adjust(hist_g)
    scen_corrected = mapper.adjust(scen_g) if scen_g is not None else None

    idx_hist = idx_scen = None
    if compute_indices and cfg.variable == "pr":
        p95 = _indices.wet_day_percentile(obs_ref, 95.0)
        p99 = _indices.wet_day_percentile(obs_ref, 99.0)
        idx_hist = _indices.all_precip_indices(hist_corrected, cfg.index_freq, p95, p99)
        if scen_corrected is not None:
            idx_scen = _indices.all_precip_indices(scen_corrected, cfg.index_freq, p95, p99)

    return DownscalingResult(
        historical=hist_corrected,
        scenario=scen_corrected,
        indices_historical=idx_hist,
        indices_scenario=idx_scen,
        config=cfg,
        mapper=mapper,
    )
