import numpy as np
import pytest

from cmip6_br.grids import BBox, bbox, grid_for, regular_grid


def test_known_regions_are_inside_brazil():
    br = bbox("BR")
    for code in ("SP", "RJ", "AM", "RS"):
        box = bbox(code)
        assert br.lon_min <= box.lon_min < box.lon_max <= br.lon_max
        assert br.lat_min <= box.lat_min < box.lat_max <= br.lat_max


def test_sao_paulo_contains_the_capital():
    assert bbox("SP").contains(-46.63, -23.55)


def test_unknown_region_raises():
    with pytest.raises(KeyError, match="Unknown region"):
        bbox("XX")


def test_grid_spacing_and_extent():
    grid = grid_for("SP", 0.25)
    dlon, dlat = grid.resolution
    assert np.isclose(dlon, 0.25) and np.isclose(dlat, 0.25)
    box = bbox("SP")
    assert grid.lon.min() > box.lon_min and grid.lon.max() < box.lon_max


def test_pad_expands_symmetrically():
    padded = BBox(-50, -45, -25, -20).pad(2)
    assert (padded.lon_min, padded.lon_max) == (-52, -43)
    assert (padded.lat_min, padded.lat_max) == (-27, -18)


def test_resolution_coarser_than_box_raises():
    with pytest.raises(ValueError):
        regular_grid(BBox(-46, -45.9, -23, -22.9), 5.0)
