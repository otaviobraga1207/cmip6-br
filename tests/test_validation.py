import numpy as np

from cmip6_br import validation


def test_perfect_forecast_scores_perfectly(obs):
    assert np.isclose(float(validation.bias(obs, obs).mean()), 0.0)
    assert np.isclose(float(validation.rmse(obs, obs).mean()), 0.0)
    assert np.isclose(float(validation.kge(obs, obs).mean()), 1.0)
    assert np.isclose(validation.perkins_skill_score(obs, obs), 1.0)


def test_bias_sign_is_sim_minus_obs(obs):
    assert float(validation.bias(obs, obs + 2.0).mean()) > 0


def test_report_has_both_rows(obs, hist):
    from cmip6_br import bias

    corrected = bias.fit(obs, hist, variable="pr").adjust(hist)
    table = validation.report(obs, hist, corrected)
    assert list(table.index) == ["raw", "corrected"]
    assert abs(table.loc["corrected", "bias"]) < abs(table.loc["raw", "bias"])
    assert table.loc["corrected", "perkins_ss"] > table.loc["raw", "perkins_ss"]
