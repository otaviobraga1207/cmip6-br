"""cmip6-br: statistical downscaling and climate-extreme indices for Brazil.

Quick start
-----------
>>> from cmip6_br import datasets, bias, indices
>>> data = datasets.demo_bundle()
>>> mapper = bias.fit(data["obs"], data["hist"], variable="pr")
>>> corrected = mapper.adjust(data["ssp"])
>>> annual = indices.rx1day(corrected)
"""

from . import (
    bias,
    datasets,
    grids,
    indices,
    pipeline,
    plots,
    regrid,
    stations,
    units,
    validation,
)
from .bias import EQM, QDM
from .grids import BBox, TargetGrid, bbox, grid_for, regular_grid
from .pipeline import DownscalingConfig, DownscalingResult, downscale
from .regrid import normalize_coords, subset_bbox
from .stations import read_bdmep, read_bdmep_dir, stations_to_dataset
from .units import to_celsius, to_mm_day

__version__ = "0.2.0"


__all__ = [
    "__version__",
    "bias",
    "datasets",
    "grids",
    "indices",
    "pipeline",
    "plots",
    "regrid",
    "stations",
    "units",
    "validation",
    "EQM",
    "QDM",
    "BBox",
    "TargetGrid",
    "bbox",
    "grid_for",
    "regular_grid",
    "DownscalingConfig",
    "DownscalingResult",
    "downscale",
    "normalize_coords",
    "subset_bbox",
    "read_bdmep",
    "read_bdmep_dir",
    "stations_to_dataset",
    "to_celsius",
    "to_mm_day",
]
