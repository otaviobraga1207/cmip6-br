import numpy as np
import pytest
import xarray as xr

from cmip6_br.units import harmonize, to_celsius, to_mm_day


def test_flux_to_mm_day():
    pr = xr.DataArray([1e-5], attrs={"units": "kg m-2 s-1"}, name="pr")
    assert np.isclose(to_mm_day(pr).item(), 0.864)
    assert to_mm_day(pr).attrs["units"] == "mm/day"


def test_mm_day_is_idempotent():
    pr = xr.DataArray([5.0], attrs={"units": "mm/day"})
    assert np.isclose(to_mm_day(to_mm_day(pr)).item(), 5.0)


def test_kelvin_to_celsius():
    tas = xr.DataArray([300.0], attrs={"units": "K"})
    assert np.isclose(to_celsius(tas).item(), 26.85)


def test_unknown_units_raise():
    with pytest.raises(ValueError, match="Unrecognised"):
        to_mm_day(xr.DataArray([1.0], attrs={"units": "furlongs"}))


def test_harmonize_dataset():
    ds = xr.Dataset(
        {
            "pr": ("time", [1e-5], {"units": "kg m-2 s-1"}),
            "tas": ("time", [300.0], {"units": "K"}),
        },
        coords={"time": [0]},
    )
    out = harmonize(ds)
    assert out["pr"].attrs["units"] == "mm/day"
    assert np.isclose(out["tas"].item(), 26.85)
