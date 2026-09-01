import numpy as np
import pytest

from cmip6_br import bias, validation
from cmip6_br.indices import rx1day


def test_eqm_reproduces_the_observed_distribution(obs, hist):
    mapper = bias.EQM(kind="multiplicative", wet_threshold=1.0).fit(obs, hist)
    corrected = mapper.adjust(hist)
    assert abs(float(corrected.mean()) - float(obs.mean())) < 0.15
    assert abs(float(corrected.quantile(0.99)) - float(obs.quantile(0.99))) < 3.0


def test_frequency_adaptation_fixes_the_drizzle_bias(obs, hist):
    raw_wet = float((hist >= 1).mean())
    obs_wet = float((obs >= 1).mean())
    corrected = bias.fit(obs, hist, variable="pr").adjust(hist)
    assert raw_wet - obs_wet > 0.10, "the fixture should start with a drizzle bias"
    assert abs(float((corrected >= 1).mean()) - obs_wet) < 0.03


def test_quantile_mapping_improves_the_pdf_overlap(obs, hist):
    corrected = bias.fit(obs, hist, variable="pr").adjust(hist)
    before = validation.perkins_skill_score(obs, hist)
    after = validation.perkins_skill_score(obs, corrected)
    assert after > before
    assert after > 0.95


def test_qdm_preserves_the_models_change_signal(obs, hist, ssp):
    """The whole point of QDM: the corrected scenario must keep the model's
    own change in the extremes, not inherit the observed climate wholesale."""
    raw_change = float(rx1day(ssp).mean()) / float(rx1day(hist).mean()) - 1
    mapper = bias.fit(obs, hist, variable="pr", method="qdm")
    corrected_change = (
        float(rx1day(mapper.adjust(ssp)).mean()) / float(rx1day(mapper.adjust(hist)).mean()) - 1
    )
    assert raw_change > 0.02, "the fixture should contain a wetting signal"
    assert abs(corrected_change - raw_change) < 0.6 * abs(raw_change) + 0.05


def test_additive_mapping_for_temperature():
    import pandas as pd
    import xarray as xr

    rng = np.random.default_rng(0)
    time = pd.date_range("1995-01-01", periods=3650, freq="D")
    seasonal = 20 + 6 * np.sin(2 * np.pi * time.dayofyear / 365.25)
    obs = xr.DataArray(seasonal + rng.normal(0, 2, time.size), coords={"time": time}, dims="time")
    warm = obs + 3.0  # a model with a uniform +3 degC bias
    corrected = bias.EQM(kind="additive").fit(obs, warm).adjust(warm)
    assert abs(float(corrected.mean()) - float(obs.mean())) < 0.05
    assert abs(float(corrected.std()) - float(obs.std())) < 0.05


def test_monthly_grouping_beats_a_single_transfer_function(obs, hist):
    grouped = bias.EQM(kind="multiplicative", group="month").fit(obs, hist).adjust(hist)
    pooled = bias.EQM(kind="multiplicative", group="none").fit(obs, hist).adjust(hist)
    monthly_error = lambda da: float(  # noqa: E731
        abs(da.groupby("time.month").mean() - obs.groupby("time.month").mean()).mean()
    )
    assert monthly_error(grouped) <= monthly_error(pooled)


def test_output_keeps_shape_dims_and_name(obs, hist):
    corrected = bias.fit(obs, hist, variable="pr").adjust(hist)
    assert corrected.shape == hist.shape
    assert corrected.dims == hist.dims
    assert corrected.name == hist.name
    assert "bias_correction" in corrected.attrs


def test_corrected_precipitation_is_never_negative(obs, hist):
    corrected = bias.fit(obs, hist, variable="pr").adjust(hist)
    assert float(corrected.min()) >= 0.0


def test_adjust_before_fit_raises(obs, hist):
    with pytest.raises(RuntimeError, match="fit"):
        bias.EQM().adjust(hist)


def test_wet_threshold_requires_multiplicative(obs, hist):
    with pytest.raises(ValueError, match="multiplicative"):
        bias.EQM(kind="additive", wet_threshold=1.0).fit(obs, hist)


def test_nan_input_stays_nan(obs, hist):
    holed = hist.copy()
    holed[0, 0, 0] = np.nan
    corrected = bias.EQM(kind="multiplicative").fit(obs, hist).adjust(holed)
    assert np.isnan(corrected[0, 0, 0].item())
