"""Bias correction by quantile mapping.

Two estimators are provided:

``EQM`` -- empirical quantile mapping (Themeßl et al., 2011). Maps the
simulated distribution onto the observed one quantile by quantile. Simple and
robust for the historical period, but it *removes* part of the model's own
climate-change signal when applied to future projections, because a future
value that lies beyond the historical range is clipped back into it.

``QDM`` -- quantile delta mapping (Cannon et al., 2015). Corrects the
distribution while explicitly preserving the model's relative (or absolute)
change in each quantile. This is the estimator you want for SSP projections,
and the one used for the ETCCDI indices in :mod:`cmip6_br.indices`.

Both support monthly grouping (a separate transfer function per calendar
month), which is what makes them usable in a country whose rainfall regime
swings as hard as Brazil's between wet and dry season.

References
----------
Themeßl, M. J., Gobiet, A., & Leuprecht, A. (2011). Empirical-statistical
downscaling and error correction of daily precipitation from regional climate
models. *International Journal of Climatology*, 31(10), 1530-1544.

Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM
precipitation by quantile delta mapping. *Journal of Climate*, 28(17),
6938-6959.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import xarray as xr

__all__ = ["EQM", "QDM", "fit", "Kind"]

Kind = Literal["additive", "multiplicative"]

_MAX_RATIO = 10.0  # guards against division by a near-zero modelled quantile


def _quantile_levels(n: int) -> np.ndarray:
    """``n`` interior quantile levels, avoiding the unstable 0 and 1 endpoints."""
    if n < 5:
        raise ValueError("n_quantiles must be at least 5")
    return np.linspace(0.5 / n, 1 - 0.5 / n, n)


def _interp_1d(x, xp, fp):
    """np.interp that tolerates NaNs in the lookup tables."""
    ok = np.isfinite(xp) & np.isfinite(fp)
    if ok.sum() < 2:
        return np.full_like(np.asarray(x, dtype=float), np.nan)
    return np.interp(x, np.asarray(xp)[ok], np.asarray(fp)[ok])


def _map_eqm(x, hist_q, obs_q, qs, kind, extrapolation):
    x = np.asarray(x, dtype=float)
    tau = _interp_1d(x, hist_q, qs)
    out = _interp_1d(tau, qs, obs_q)
    if extrapolation == "delta":
        # Beyond the calibration range, carry the end-point correction forward
        # instead of flattening everything onto the last quantile.
        lo, hi = np.nanmin(hist_q), np.nanmax(hist_q)
        below, above = x < lo, x > hi
        if kind == "multiplicative":
            with np.errstate(divide="ignore", invalid="ignore"):
                f_lo = np.nanmin(obs_q) / lo if lo > 0 else 1.0
                f_hi = np.nanmax(obs_q) / hi if hi > 0 else 1.0
            out = np.where(below, x * np.clip(f_lo, 1 / _MAX_RATIO, _MAX_RATIO), out)
            out = np.where(above, x * np.clip(f_hi, 1 / _MAX_RATIO, _MAX_RATIO), out)
        else:
            out = np.where(below, x + (np.nanmin(obs_q) - lo), out)
            out = np.where(above, x + (np.nanmax(obs_q) - hi), out)
    out = np.where(np.isfinite(x), out, np.nan)
    if kind == "multiplicative":
        out = np.maximum(out, 0.0)
    return out


def _map_qdm(x, hist_q, obs_q, qs, kind, extrapolation):
    """Quantile delta mapping: correct the distribution, keep the model's change."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    out = np.full_like(x, np.nan)
    if finite.sum() < 2:
        return out
    xf = x[finite]
    # Empirical CDF of the projected period itself.
    ranks = np.argsort(np.argsort(xf))
    tau = (ranks + 0.5) / xf.size
    hist_at_tau = _interp_1d(tau, qs, hist_q)
    obs_at_tau = _interp_1d(tau, qs, obs_q)
    if kind == "multiplicative":
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.where(hist_at_tau > 0, xf / hist_at_tau, 1.0)
        delta = np.clip(np.nan_to_num(delta, nan=1.0), 1 / _MAX_RATIO, _MAX_RATIO)
        vals = np.maximum(obs_at_tau * delta, 0.0)
    else:
        vals = obs_at_tau + (xf - hist_at_tau)
    if extrapolation == "constant":
        vals = np.clip(vals, np.nanmin(obs_q), None)
    out[finite] = vals
    return out


@dataclass
class _QuantileMapper:
    """Shared fit/adjust machinery. Use :class:`EQM` or :class:`QDM`."""

    kind: Kind = "additive"
    n_quantiles: int = 100
    group: Literal["month", "none"] = "month"
    wet_threshold: float | None = None
    extrapolation: Literal["constant", "delta"] = "constant"
    _obs_q: xr.DataArray | None = field(default=None, init=False, repr=False)
    _hist_q: xr.DataArray | None = field(default=None, init=False, repr=False)
    _dry_cut: xr.DataArray | None = field(default=None, init=False, repr=False)
    _qs: np.ndarray = field(default=None, init=False, repr=False)

    _mapper = staticmethod(_map_eqm)

    # -- internals -------------------------------------------------------
    def _grouped(self, da: xr.DataArray):
        if self.group == "month":
            return da.groupby("time.month")
        return da.expand_dims(month=[0]).groupby("month")

    def _group_key(self, da: xr.DataArray):
        if self.group == "month":
            return da["time"].dt.month
        return xr.zeros_like(da["time"], dtype=int)

    def _quantiles(self, da: xr.DataArray) -> xr.DataArray:
        if self.group == "month":
            return da.groupby("time.month").quantile(self._qs, dim="time")
        out = da.quantile(self._qs, dim="time")
        return out.expand_dims(month=[0])

    # -- public API ------------------------------------------------------
    def fit(self, obs: xr.DataArray, hist: xr.DataArray) -> _QuantileMapper:
        """Learn the transfer function from observations and the historical run.

        ``obs`` and ``hist`` must both carry a ``time`` dimension and cover a
        common calibration period; they do *not* need identical time steps.
        """
        for name, da in (("obs", obs), ("hist", hist)):
            if "time" not in da.dims:
                raise ValueError(f"{name} must have a 'time' dimension")
        self._qs = _quantile_levels(self.n_quantiles)

        if self.wet_threshold is not None:
            if self.kind != "multiplicative":
                raise ValueError("wet_threshold only makes sense with kind='multiplicative'")
            obs, hist = self._frequency_adaptation(obs, hist)

        self._obs_q = self._quantiles(obs)
        self._hist_q = self._quantiles(hist)
        return self

    def _frequency_adaptation(self, obs, hist):
        """Match the modelled drizzle frequency to the observed wet-day frequency.

        Climate models rain too often and too lightly. Correcting quantiles
        without fixing that first leaves a wet bias in every dry-spell index
        (CDD in particular), so we find the simulated threshold that reproduces
        the observed dry-day fraction and zero everything below it.
        """
        wet = self.wet_threshold
        dry_frac = (
            (obs < wet).groupby("time.month").mean()
            if self.group == "month"
            else (obs < wet).mean(dim="time").expand_dims(month=[0])
        )
        dry_frac = dry_frac.clip(0.0, 0.999)

        cuts = []
        for m in np.atleast_1d(dry_frac["month"].values):
            sub = hist.sel(time=hist["time"].dt.month == m) if self.group == "month" else hist
            frac = dry_frac.sel(month=m)
            cut = xr.apply_ufunc(
                lambda x, q: np.nanquantile(x, float(q)) if np.isfinite(x).any() else np.nan,
                sub,
                frac,
                input_core_dims=[["time"], []],
                vectorize=True,
                output_dtypes=[float],
            )
            cuts.append(cut.expand_dims(month=[m]))
        self._dry_cut = xr.concat(cuts, dim="month")

        hist = self._apply_dry_cut(hist)
        obs = obs.where(obs >= wet, 0.0)
        return obs, hist

    def _apply_dry_cut(self, da: xr.DataArray) -> xr.DataArray:
        if self._dry_cut is None:
            return da
        cut = (
            self._dry_cut.sel(month=self._group_key(da))
            if self.group == "month"
            else self._dry_cut.isel(month=0)
        )
        return da.where(da >= cut, 0.0)

    def adjust(self, sim: xr.DataArray) -> xr.DataArray:
        """Apply the fitted transfer function to a simulated series."""
        if self._obs_q is None:
            raise RuntimeError("call fit() before adjust()")
        if "time" not in sim.dims:
            raise ValueError("sim must have a 'time' dimension")
        sim = self._apply_dry_cut(sim)

        pieces = []
        months = np.atleast_1d(self._obs_q["month"].values)
        for m in months:
            sub = sim.sel(time=sim["time"].dt.month == m) if self.group == "month" else sim
            if sub["time"].size == 0:
                continue
            corrected = xr.apply_ufunc(
                self.__class__._mapper,
                sub,
                self._hist_q.sel(month=m),
                self._obs_q.sel(month=m),
                input_core_dims=[["time"], ["quantile"], ["quantile"]],
                output_core_dims=[["time"]],
                kwargs={
                    "qs": self._qs,
                    "kind": self.kind,
                    "extrapolation": self.extrapolation,
                },
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            pieces.append(corrected.assign_coords(time=sub["time"]))

        out = (
            xr.concat(pieces, dim="time", coords="minimal", compat="override").sortby("time")
            if len(pieces) > 1
            else pieces[0]
        )
        out = out.transpose(*sim.dims)
        out.attrs = dict(sim.attrs)
        out.attrs["bias_correction"] = f"{self.__class__.__name__}({self.kind}, group={self.group})"
        out.name = sim.name
        return out

    def fit_adjust(self, obs, hist, sim=None) -> xr.DataArray:
        """Fit on (obs, hist) and adjust ``sim`` (defaults to ``hist`` itself)."""
        return self.fit(obs, hist).adjust(hist if sim is None else sim)


class EQM(_QuantileMapper):
    """Empirical quantile mapping. Best for validating over the historical period."""

    _mapper = staticmethod(_map_eqm)


class QDM(_QuantileMapper):
    """Quantile delta mapping. Use this for SSP projections."""

    _mapper = staticmethod(_map_qdm)


def fit(
    obs: xr.DataArray,
    hist: xr.DataArray,
    variable: Literal["pr", "tas"] = "pr",
    method: Literal["qdm", "eqm"] = "qdm",
    **kwargs,
):
    """Sensible defaults per variable.

    ``pr`` gets a multiplicative mapping with wet-day frequency adaptation at
    1 mm/day; ``tas`` (and any other temperature-like variable) gets an
    additive one.
    """
    cls = QDM if method == "qdm" else EQM
    if variable == "pr":
        defaults = {"kind": "multiplicative", "wet_threshold": 1.0}
    else:
        defaults = {"kind": "additive"}
    defaults.update(kwargs)
    return cls(**defaults).fit(obs, hist)
