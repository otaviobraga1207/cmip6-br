import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from cmip6_br import indices, plots  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def annual(obs):
    return indices.all_precip_indices(obs)


def test_map_field_labels_the_colorbar_with_units(annual):
    ax = plots.map_field(annual["rx1day"])
    label = ax.figure.axes[-1].get_ylabel()
    assert "mm" in label and "1-day" in label


def test_map_field_collapses_the_time_dimension(annual):
    ax = plots.map_field(annual["prcptot"])
    mesh = ax.collections[0]
    assert mesh.get_array().size == annual["prcptot"].isel(time=0).size


def test_robust_limits_ignore_a_single_outlier(annual):
    spiked = annual["rx1day"].copy()
    spiked[0, 0, 0] = 1e6
    robust = plots.map_field(spiked, robust=True).collections[0].get_clim()
    plt.close("all")
    raw = plots.map_field(spiked, robust=False).collections[0].get_clim()
    assert robust[1] < raw[1] / 100


def test_change_map_is_symmetric_around_zero(annual):
    wetter = annual["rx1day"] * 1.3
    ax = plots.map_change(annual["rx1day"], wetter)
    vmin, vmax = ax.collections[0].get_clim()
    assert np.isclose(vmin, -vmax), "a diverging scale must be centred on zero"
    assert ax.figure.axes[-1].get_ylabel() == "change (%)"


def test_absolute_change_uses_the_field_units(annual):
    ax = plots.map_change(annual["rx1day"], annual["rx1day"] + 5, relative=False)
    assert "mm" in ax.figure.axes[-1].get_ylabel()


def test_change_map_direction(annual):
    """A wetter scenario must produce positive values, not just a pretty picture."""
    ax = plots.map_change(annual["rx1day"], annual["rx1day"] * 1.2)
    values = ax.collections[0].get_array()
    assert np.nanmedian(values) > 0


def test_panel_gives_one_axes_per_field_plus_colorbars(annual):
    fig = plots.map_panel(
        {"Rx1day": annual["rx1day"], "CDD": annual["cdd"], "R95p": annual["r95p"]},
        shared_scale=False,
    )
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
    assert titles == ["Rx1day", "CDD", "R95p"]


def test_panel_labels_each_colorbar_from_its_own_field(annual):
    """A panel mixing millimetres and days must not label both the same."""
    fig = plots.map_panel({"Rx1day": annual["rx1day"], "CDD": annual["cdd"]}, shared_scale=False)
    labels = [ax.get_ylabel() for ax in fig.axes if "(" in ax.get_ylabel()]
    assert any("mm" in lab for lab in labels)
    assert any("days" in lab for lab in labels)


def test_shared_scale_puts_panels_on_one_range(annual):
    fig = plots.map_panel({"a": annual["rx1day"], "b": annual["rx1day"] * 2}, shared_scale=True)
    clims = {
        ax.collections[0].get_clim()
        for ax in fig.axes
        if ax.collections and ax.get_title(loc="left")
    }
    assert len(clims) == 1


def test_qq_plot_has_a_legend_and_two_series(obs, hist):
    from cmip6_br import bias

    corrected = bias.fit(obs, hist, variable="pr").adjust(hist)
    ax = plots.qq_plot(obs, hist, corrected)
    labels = [line.get_label() for line in ax.lines if not line.get_label().startswith("_")]
    assert labels == ["model raw", "model corrected"]
    assert ax.get_legend() is not None


def test_series_are_distinguished_by_more_than_colour(obs, hist):
    """Colour alone fails in greyscale and for colour-vision deficiency."""
    from cmip6_br import bias

    corrected = bias.fit(obs, hist, variable="pr").adjust(hist)
    ax = plots.qq_plot(obs, hist, corrected)
    series = [ln for ln in ax.lines if not ln.get_label().startswith("_")]
    assert len({ln.get_linestyle() for ln in series}) == len(series)
    assert len({ln.get_marker() for ln in series}) == len(series)


def test_seasonal_cycle_has_twelve_points(obs, hist):
    ax = plots.seasonal_cycle({"observations": obs, "model raw": hist})
    assert all(len(line.get_xdata()) == 12 for line in ax.lines)


def test_seasonal_cycle_refuses_a_fourth_series(obs):
    with pytest.raises(ValueError, match="three series"):
        plots.seasonal_cycle({name: obs for name in "abcd"})


def test_palette_is_not_a_rainbow():
    """A sequential ramp must be monotonic in lightness -- that is what makes it
    readable and what a rainbow colormap is not."""
    from matplotlib.colors import to_rgb

    def luminance(hex_colour):
        r, g, b = to_rgb(hex_colour)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    steps = [luminance(c) for c in plots._SEQUENTIAL_STEPS]
    assert all(later < earlier for earlier, later in zip(steps, steps[1:], strict=False))


def test_diverging_midpoint_is_neutral():
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(plots._DIVERGING_STEPS[len(plots._DIVERGING_STEPS) // 2])
    assert max(r, g, b) - min(r, g, b) < 0.05, "the midpoint must read as 'no change'"


def test_figure_is_written_to_disk(annual, tmp_path):
    path = tmp_path / "map.png"
    plots.map_field(annual["rx1day"]).figure.savefig(path)
    assert path.stat().st_size > 5000


def test_geotiff_export(annual, tmp_path):
    rioxarray = pytest.importorskip("rioxarray")  # noqa: F841
    path = plots.save_geotiff(annual["rx1day"], str(tmp_path / "rx1day.tif"))
    assert path.endswith(".tif")

    import rioxarray as rxr

    reopened = rxr.open_rasterio(path)
    assert reopened.rio.crs is not None
