"""Readers for INMET station data (BDMEP export format).

INMET's BDMEP portal (https://bdmep.inmet.gov.br) exports one CSV per station,
Latin-1 encoded, semicolon separated, comma as decimal mark, with a block of
metadata lines above the header. Nothing in the scientific Python stack reads
that file as-is, which is why every climate workflow in Brazil starts with
forty lines of throwaway parsing code. This module is those forty lines, once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["Station", "read_bdmep", "read_bdmep_dir", "stations_to_dataset"]

_HEADER_MARKERS = ("data medicao", "data")

_COLUMN_PATTERNS = [
    (r"precipitacao", "pr"),
    (r"temperatura.*maxima", "tasmax"),
    (r"temperatura.*minima", "tasmin"),
    (r"temperatura.*(media|bulbo seco)", "tas"),
    (r"umidade relativa.*media", "hurs"),
    (r"vento.*rajada", "wsgsmax"),
    (r"vento.*velocidade", "sfcwind"),
]


@dataclass(frozen=True)
class Station:
    """Station metadata parsed from the BDMEP header block."""

    code: str
    name: str
    lat: float
    lon: float
    altitude: float | None = None

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.code} {self.name} ({self.lat:.3f}, {self.lon:.3f})"


def _to_float(value: str) -> float | None:
    value = (value or "").strip().replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


def _normalize(text: str) -> str:
    text = text.lower()
    for accented, plain in (
        ("áàâã", "a"),
        ("éê", "e"),
        ("í", "i"),
        ("óôõ", "o"),
        ("ú", "u"),
        ("ç", "c"),
    ):
        for ch in accented:
            text = text.replace(ch, plain)
    return text


def _map_columns(columns) -> dict[str, str]:
    mapping = {}
    for col in columns:
        norm = _normalize(str(col))
        for pattern, target in _COLUMN_PATTERNS:
            if re.search(pattern, norm) and target not in mapping.values():
                mapping[col] = target
                break
    return mapping


def read_bdmep(path: str | Path, encoding: str = "latin-1") -> tuple[pd.DataFrame, Station]:
    """Read one BDMEP CSV.

    Returns
    -------
    (frame, station)
        ``frame`` is indexed by date with canonical CMIP-style column names
        (``pr``, ``tasmax``, ``tasmin``, ``tas``, ...). ``station`` carries the
        metadata from the header block.

    Notes
    -----
    Precipitation comes out in mm/day and temperature in degrees Celsius, i.e.
    already in the units the rest of this package expects. Missing values
    (empty cells and the ``-9999`` sentinel) become NaN.
    """
    path = Path(path)
    raw = path.read_text(encoding=encoding, errors="replace").splitlines()

    meta: dict[str, str] = {}
    header_idx = None
    for i, line in enumerate(raw[:60]):
        norm = _normalize(line)
        if norm.startswith(_HEADER_MARKERS) and ";" in line:
            header_idx = i
            break
        if ":" in line:
            key, _, value = line.partition(":")
            meta[_normalize(key).strip()] = value.strip()
    if header_idx is None:
        raise ValueError(f"{path.name}: could not find the 'Data Medicao' header row")

    frame = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        skiprows=header_idx,
        encoding=encoding,
        engine="python",
        na_values=["", " ", "null", "-9999", "-9999.0"],
    )
    frame = frame.loc[:, [c for c in frame.columns if not str(c).startswith("Unnamed")]]

    date_col = frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce", dayfirst=True)
    frame = frame.dropna(subset=[date_col]).set_index(date_col)
    frame.index.name = "time"

    mapping = _map_columns(frame.columns)
    frame = frame.rename(columns=mapping)[list(mapping.values())]
    frame = frame.apply(pd.to_numeric, errors="coerce")

    station = Station(
        code=meta.get("codigo estacao", path.stem),
        name=meta.get("nome", path.stem),
        lat=_to_float(meta.get("latitude", "")) or np.nan,
        lon=_to_float(meta.get("longitude", "")) or np.nan,
        altitude=_to_float(meta.get("altitude", "")),
    )
    return frame, station


def read_bdmep_dir(
    directory: str | Path, pattern: str = "*.csv", encoding: str = "latin-1"
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Read every BDMEP CSV in a directory.

    Returns a ``{station_code: frame}`` dict and a metadata table indexed by
    station code. Files that fail to parse are skipped with a warning rather
    than aborting a 200-station batch.
    """
    import warnings

    frames: dict[str, pd.DataFrame] = {}
    meta_rows = []
    for file in sorted(Path(directory).glob(pattern)):
        try:
            frame, station = read_bdmep(file, encoding=encoding)
        except Exception as exc:  # noqa: BLE001 - deliberately tolerant
            warnings.warn(f"skipping {file.name}: {exc}", stacklevel=2)
            continue
        frames[station.code] = frame
        meta_rows.append(
            {
                "code": station.code,
                "name": station.name,
                "lat": station.lat,
                "lon": station.lon,
                "altitude": station.altitude,
                "start": frame.index.min(),
                "end": frame.index.max(),
                "n_days": len(frame),
            }
        )
    meta = pd.DataFrame(meta_rows)
    if not meta.empty:
        meta = meta.set_index("code")
    return frames, meta


def stations_to_dataset(
    frames: dict[str, pd.DataFrame], meta: pd.DataFrame, variable: str = "pr"
) -> xr.DataArray:
    """Stack per-station series into a ``(time, station)`` DataArray.

    ``lat``/``lon`` ride along as non-dimension coordinates, so the result can
    be compared against gridded output with ``.sel(lat=..., lon=...,
    method="nearest")`` on the gridded side.
    """
    series = {}
    for code, frame in frames.items():
        if variable in frame.columns:
            series[code] = frame[variable]
    if not series:
        raise KeyError(f"no station provides variable {variable!r}")
    wide = pd.DataFrame(series)
    wide.index.name = "time"
    da = xr.DataArray(
        wide.values,
        coords={"time": wide.index, "station": list(wide.columns)},
        dims=("time", "station"),
        name=variable,
    )
    present = [c for c in wide.columns if c in meta.index]
    if present:
        da = da.assign_coords(
            lat=("station", meta.loc[present, "lat"].to_numpy()),
            lon=("station", meta.loc[present, "lon"].to_numpy()),
        )
    da.attrs["source"] = "INMET BDMEP"
    da.attrs["units"] = "mm/day" if variable == "pr" else "degC"
    return da
