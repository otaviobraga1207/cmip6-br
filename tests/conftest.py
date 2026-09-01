import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cmip6_br.datasets import demo_bundle


@pytest.fixture(scope="session")
def bundle():
    return demo_bundle(seed=42)


@pytest.fixture(scope="session")
def obs(bundle):
    return bundle["obs"]


@pytest.fixture(scope="session")
def hist(bundle):
    return bundle["hist"]


@pytest.fixture(scope="session")
def ssp(bundle):
    return bundle["ssp"]


@pytest.fixture
def cmip6_like():
    """A tiny global field with the coordinate quirks of raw CMIP6 output:
    0-360 longitudes, descending latitude, flux units, non-standard names."""
    time = pd.date_range("2000-01-01", periods=40, freq="D")
    lon = np.arange(0.0, 360.0, 10.0)  # 0-360
    lat = np.arange(88.0, -90.0, -10.0)  # descending
    rng = np.random.default_rng(0)
    values = rng.gamma(1.0, 3e-5, size=(time.size, lat.size, lon.size))
    return xr.DataArray(
        values,
        coords={"time": time, "latitude": lat, "longitude": lon},
        dims=("time", "latitude", "longitude"),
        name="pr",
        attrs={"units": "kg m-2 s-1"},
    )
