"""Skill metrics for judging whether a bias correction actually helped."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["bias", "mae", "rmse", "pearson_r", "kge", "perkins_skill_score", "report"]


def _align(obs: xr.DataArray, sim: xr.DataArray):
    obs, sim = xr.align(obs, sim, join="inner")
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs.where(mask), sim.where(mask)


def bias(obs, sim, dim="time"):
    """Mean error (sim - obs). Positive means the model is too high."""
    obs, sim = _align(obs, sim)
    return (sim - obs).mean(dim=dim)


def mae(obs, sim, dim="time"):
    obs, sim = _align(obs, sim)
    return np.abs(sim - obs).mean(dim=dim)


def rmse(obs, sim, dim="time"):
    obs, sim = _align(obs, sim)
    return np.sqrt(((sim - obs) ** 2).mean(dim=dim))


def pearson_r(obs, sim, dim="time"):
    obs, sim = _align(obs, sim)
    return xr.corr(obs, sim, dim=dim)


def kge(obs, sim, dim="time"):
    """Kling-Gupta efficiency (Gupta et al., 2009). 1 is perfect; < -0.41 is
    worse than using the observed mean."""
    obs, sim = _align(obs, sim)
    r = xr.corr(obs, sim, dim=dim)
    alpha = sim.std(dim=dim) / obs.std(dim=dim)
    beta = sim.mean(dim=dim) / obs.mean(dim=dim)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def perkins_skill_score(obs, sim, bins: int = 40, dim: str = "time") -> float:
    """Overlap between the two probability density functions (0 to 1).

    This is the metric that tells you whether quantile mapping fixed the whole
    distribution rather than just the mean.
    """
    o = np.asarray(obs.values, dtype=float).ravel()
    s = np.asarray(sim.values, dtype=float).ravel()
    o, s = o[np.isfinite(o)], s[np.isfinite(s)]
    if o.size == 0 or s.size == 0:
        return float("nan")
    edges = np.histogram_bin_edges(np.concatenate([o, s]), bins=bins)
    po, _ = np.histogram(o, bins=edges, density=False)
    ps, _ = np.histogram(s, bins=edges, density=False)
    po = po / po.sum()
    ps = ps / ps.sum()
    return float(np.minimum(po, ps).sum())


def report(obs: xr.DataArray, raw: xr.DataArray, corrected: xr.DataArray) -> pd.DataFrame:
    """Side-by-side skill table for the raw and bias-corrected series.

    Returns a tidy DataFrame so it drops straight into a notebook or a report.
    """
    rows = []
    for label, sim in (("raw", raw), ("corrected", corrected)):
        rows.append(
            {
                "series": label,
                "bias": float(bias(obs, sim).mean()),
                "mae": float(mae(obs, sim).mean()),
                "rmse": float(rmse(obs, sim).mean()),
                "pearson_r": float(pearson_r(obs, sim).mean()),
                "kge": float(kge(obs, sim).mean()),
                "perkins_ss": perkins_skill_score(obs, sim),
            }
        )
    return pd.DataFrame(rows).set_index("series")
