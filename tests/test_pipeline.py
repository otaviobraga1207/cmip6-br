import numpy as np
import pytest

from cmip6_br.grids import grid_for
from cmip6_br.pipeline import DownscalingConfig, downscale


@pytest.fixture(scope="module")
def result(bundle):
    cfg = DownscalingConfig(
        region="SP", resolution=1.0, variable="pr", method="qdm", convert_units=False
    )
    return downscale(bundle["obs"], bundle["hist"], bundle["ssp"], cfg)


def test_outputs_land_on_the_target_grid(result):
    grid = grid_for("SP", 1.0)
    assert result.historical.shape[-2:] == grid.shape
    assert result.scenario.shape[-2:] == grid.shape


def test_indices_are_computed_for_both_periods(result):
    assert result.indices_historical is not None
    assert result.indices_scenario is not None
    assert "rx1day" in result.indices_scenario


def test_bias_is_removed_over_the_reference_period(bundle, result):
    obs_mean = float(bundle["obs"].mean())
    raw_mean = float(bundle["hist"].mean())
    corrected_mean = float(result.historical.mean())
    assert abs(corrected_mean - obs_mean) < abs(raw_mean - obs_mean)


def test_r95p_threshold_is_fixed_on_the_reference_period(result):
    """A wetter future must show up as more R95p, which only works if the
    threshold came from the reference period rather than each period's own."""
    hist_r95 = float(result.indices_historical["r95p"].mean())
    scen_r95 = float(result.indices_scenario["r95p"].mean())
    assert scen_r95 > hist_r95


def test_reference_period_subsetting(bundle):
    cfg = DownscalingConfig(
        region="SP",
        resolution=1.0,
        convert_units=False,
        reference_period=("1995-01-01", "2004-12-31"),
    )
    out = downscale(bundle["obs"], bundle["hist"], config=cfg)
    assert out.historical["time"].size == bundle["hist"]["time"].size


def test_to_netcdf_writes_every_product(result, tmp_path):
    written = result.to_netcdf(str(tmp_path / "run"))
    assert len(written) == 4
    for path in written:
        assert (tmp_path / path.split("/")[-1]).stat().st_size > 0


def test_raw_cmip6_input_flows_through_unit_conversion(cmip6_like, bundle):
    """A field in kg m-2 s-1 on a 0-360 grid must survive the whole pipeline."""
    cfg = DownscalingConfig(region="SP", resolution=1.0, variable="pr")
    out = downscale(bundle["obs"], cmip6_like, config=cfg, compute_indices=False)
    assert np.isfinite(out.historical).any()
    assert float(out.historical.max()) < 1000  # i.e. mm/day, not a raw flux
