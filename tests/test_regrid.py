import numpy as np

from cmip6_br.grids import grid_for
from cmip6_br.regrid import area_weights, normalize_coords, regrid, spatial_mean, subset_bbox
from cmip6_br.units import to_mm_day


def test_normalize_renames_and_rewraps(cmip6_like):
    out = normalize_coords(cmip6_like)
    assert "lon" in out.coords and "lat" in out.coords
    assert out["lon"].min() >= -180 and out["lon"].max() < 180
    assert (np.diff(out["lat"]) > 0).all(), "latitude must end up ascending"
    assert (np.diff(out["lon"]) > 0).all()


def test_normalize_is_idempotent(cmip6_like):
    once = normalize_coords(cmip6_like)
    twice = normalize_coords(once)
    assert once.equals(twice)


def test_subset_keeps_only_the_region(cmip6_like):
    from cmip6_br.grids import bbox

    sub = subset_bbox(cmip6_like, bbox("SP"), pad=10)
    assert sub["lon"].size < cmip6_like["longitude"].size
    assert float(sub["lat"].max()) < 0  # Brazil's Southeast is entirely southern


def test_regrid_produces_the_target_shape(cmip6_like):
    from cmip6_br.grids import bbox

    grid = grid_for("SP", 0.5)
    sub = subset_bbox(to_mm_day(cmip6_like), bbox("SP"), pad=15)
    out = regrid(sub, grid)
    assert out.shape[-2:] == grid.shape
    assert np.isfinite(out).all(), "fill_edges should leave no NaN inside the box"


def test_regrid_preserves_a_constant_field(cmip6_like):
    from cmip6_br.grids import bbox

    constant = normalize_coords(cmip6_like) * 0 + 7.0
    sub = subset_bbox(constant, bbox("SP"), pad=15)
    out = regrid(sub, grid_for("SP", 0.5))
    assert np.allclose(out.values, 7.0)


def test_area_weights_shrink_towards_the_poles(cmip6_like):
    w = area_weights(cmip6_like)
    assert float(w.sel(lat=8.0, method="nearest")) > float(w.sel(lat=88.0, method="nearest"))


def test_spatial_mean_collapses_space(obs):
    out = spatial_mean(obs)
    assert out.dims == ("time",)
