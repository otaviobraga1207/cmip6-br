"""Target grids and regional subsetting helpers for Brazil.

The bounding boxes here are deliberately *padded* rectangles meant for cheap
subsetting of global fields before any expensive work. They are not official
IBGE boundaries. For exact administrative limits use ``geobr`` or a shapefile
together with :func:`cmip6_br.regrid.mask_to_geometry`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

__all__ = ["BBox", "TargetGrid", "BBOXES", "bbox", "grid_for", "regular_grid"]


@dataclass(frozen=True)
class BBox:
    """A geographic bounding box in decimal degrees (west, east, south, north)."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def pad(self, degrees: float) -> BBox:
        """Return a copy expanded by ``degrees`` on every side."""
        return BBox(
            self.lon_min - degrees,
            self.lon_max + degrees,
            self.lat_min - degrees,
            self.lat_max + degrees,
        )

    def contains(self, lon: float, lat: float) -> bool:
        return self.lon_min <= lon <= self.lon_max and self.lat_min <= lat <= self.lat_max


#: Padded bounding boxes for Brazil and selected states (continental territory only;
#: oceanic islands such as Fernando de Noronha and Trindade are excluded).
BBOXES: dict[str, BBox] = {
    "BR": BBox(-74.5, -32.0, -34.5, 5.5),
    "AC": BBox(-74.5, -66.0, -11.5, -7.0),
    "AM": BBox(-74.0, -55.5, -10.0, 2.5),
    "BA": BBox(-47.0, -37.0, -18.5, -8.5),
    "CE": BBox(-41.5, -37.0, -8.0, -2.5),
    "DF": BBox(-48.5, -47.0, -16.3, -15.3),
    "ES": BBox(-42.0, -39.5, -21.5, -17.5),
    "GO": BBox(-53.5, -45.5, -19.7, -12.3),
    "MA": BBox(-48.9, -41.5, -10.5, -1.0),
    "MG": BBox(-51.5, -39.5, -23.0, -14.0),
    "MS": BBox(-58.5, -50.5, -24.5, -17.0),
    "MT": BBox(-62.0, -50.0, -18.5, -7.0),
    "PA": BBox(-59.0, -46.0, -10.0, 2.8),
    "PB": BBox(-39.0, -34.5, -8.5, -6.0),
    "PE": BBox(-41.5, -34.5, -9.7, -7.0),
    "PR": BBox(-55.0, -47.8, -27.0, -22.3),
    "RJ": BBox(-45.0, -40.7, -23.5, -20.5),
    "RN": BBox(-38.8, -34.8, -6.9, -4.7),
    "RS": BBox(-58.0, -49.5, -34.0, -26.8),
    "SC": BBox(-54.0, -48.2, -29.5, -25.8),
    "SE": BBox(-38.5, -36.3, -11.7, -9.4),
    "SP": BBox(-53.5, -44.0, -25.5, -19.5),
    "TO": BBox(-50.8, -45.6, -13.6, -5.0),
}


def bbox(region: str) -> BBox:
    """Look up a padded bounding box by region code (``"BR"``, ``"SP"``, ...)."""
    key = region.upper()
    if key not in BBOXES:
        raise KeyError(
            f"Unknown region {region!r}. Known: {', '.join(sorted(BBOXES))}. "
            "Pass a cmip6_br.grids.BBox explicitly for anything else."
        )
    return BBOXES[key]


@dataclass(frozen=True)
class TargetGrid:
    """A regular lat/lon grid to downscale onto."""

    lon: np.ndarray
    lat: np.ndarray
    name: str = "target"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.lat.size, self.lon.size)

    @property
    def resolution(self) -> tuple[float, float]:
        return (
            float(np.abs(np.diff(self.lon)).mean()),
            float(np.abs(np.diff(self.lat)).mean()),
        )

    def to_dataset(self) -> xr.Dataset:
        """An empty dataset carrying the grid coordinates (handy for merges)."""
        return xr.Dataset(
            coords={
                "lat": ("lat", self.lat, {"units": "degrees_north", "axis": "Y"}),
                "lon": ("lon", self.lon, {"units": "degrees_east", "axis": "X"}),
            },
            attrs={"grid_name": self.name},
        )


def regular_grid(box: BBox, resolution: float, name: str = "target") -> TargetGrid:
    """Build a cell-centred regular grid covering ``box`` at ``resolution`` degrees.

    Cell centres are offset by half a grid spacing so that the grid *cells*
    (not their centres) tile the bounding box.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    half = resolution / 2.0
    lon = np.arange(box.lon_min + half, box.lon_max, resolution)
    lat = np.arange(box.lat_min + half, box.lat_max, resolution)
    if lon.size == 0 or lat.size == 0:
        raise ValueError("resolution is coarser than the bounding box")
    return TargetGrid(lon=lon, lat=lat, name=name)


def grid_for(region: str, resolution: float = 0.1) -> TargetGrid:
    """Convenience: a regular grid over a known region code.

    >>> grid_for("SP", 0.25).shape
    (24, 38)
    """
    return regular_grid(bbox(region), resolution, name=region.upper())
