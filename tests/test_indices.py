import numpy as np
import pandas as pd
import xarray as xr

from cmip6_br import indices


def series(values, start="2001-01-01"):
    time = pd.date_range(start, periods=len(values), freq="D")
    return xr.DataArray(np.asarray(values, dtype=float), coords={"time": time}, dims="time")


def test_rx1day_is_the_maximum():
    pr = series([0, 3, 40, 12, 0])
    assert indices.rx1day(pr).item() == 40


def test_rx5day_finds_the_wettest_window():
    pr = series([0, 10, 10, 10, 10, 10, 0, 0, 50, 0])
    assert indices.rxnday(pr, 5).item() == 50 + 0 + 0 + 10 + 10


def test_prcptot_ignores_sub_wet_day_drizzle():
    pr = series([0.5, 0.5, 20.0, 0.9])
    assert indices.prcptot(pr).item() == 20.0


def test_counts_use_the_right_thresholds():
    pr = series([9.9, 10.0, 19.9, 20.0, 25.0])
    assert indices.r10mm(pr).item() == 4
    assert indices.r20mm(pr).item() == 2


def test_sdii_is_the_wet_day_mean():
    pr = series([0.0, 10.0, 20.0, 0.2])
    assert np.isclose(indices.sdii(pr).item(), 15.0)


def test_cdd_and_cwd_measure_the_longest_run():
    pr = series([0, 0, 0, 5, 5, 0, 0, 0, 0, 8, 8, 8])
    assert indices.cdd(pr).item() == 4
    assert indices.cwd(pr).item() == 3


def test_r95p_only_counts_the_excess_days():
    pr = series([1.0] * 95 + [100.0] * 5)
    threshold = indices.wet_day_percentile(pr, 95.0)
    assert indices.r95p(pr, threshold).item() == 500.0


def test_a_fixed_threshold_lets_indices_be_compared_across_periods():
    reference = series([1.0] * 95 + [100.0] * 5, start="1991-01-01")
    future = series([1.0] * 90 + [100.0] * 10, start="2051-01-01")
    threshold = indices.wet_day_percentile(reference, 95.0)
    assert indices.r95p(future, threshold).item() > indices.r95p(reference, threshold).item()


def test_temperature_indices():
    tasmax = series([20, 26, 31, 24])
    tasmin = series([10, 18, 22, 12])
    assert indices.txx(tasmax).item() == 31
    assert indices.tnn(tasmin).item() == 10
    assert indices.summer_days(tasmax, 25).item() == 2
    assert indices.tropical_nights(tasmin, 20).item() == 1
    assert np.isclose(indices.dtr(tasmax, tasmin).item(), 9.75)


def test_all_precip_indices_on_a_grid(obs):
    ds = indices.all_precip_indices(obs)
    assert set(ds.data_vars) >= {"prcptot", "rx1day", "rx5day", "cdd", "cwd", "r95p"}
    assert ds["time"].size == 20  # one value per year
    assert ds["prcptot"].dims == ("time", "lat", "lon")
    # A São Paulo-like synthetic climate should land in a plausible range.
    assert 800 < float(ds["prcptot"].mean()) < 2500


def test_monthly_frequency_works(obs):
    monthly = indices.rx1day(obs.sel(time="2000"), freq="MS")
    assert monthly["time"].size == 12
