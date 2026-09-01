"""The BDMEP fixture below is a faithful copy of the real export layout:
Latin-1 text, semicolons, comma decimals, metadata block, trailing separator.
"""

import numpy as np
import pytest

from cmip6_br.stations import read_bdmep, read_bdmep_dir, stations_to_dataset

_HEADER = (
    "Data Medicao;PRECIPITACAO TOTAL, DIARIO (AutoObservacao)(mm);"
    "TEMPERATURA MAXIMA, DIARIA (AUT)(\u00b0C);"
    "TEMPERATURA MINIMA, DIARIA (AUT)(\u00b0C);"
)

BDMEP = f"""\
Nome: SAO PAULO - MIRANTE
Codigo Estacao: 83781
Latitude: -23,49638888
Longitude: -46,61972222
Altitude: 785,66
Situacao: Operante
Data Inicial: 2020-01-01
Data Final: 2020-01-05
Periodicidade da consulta: Diaria
{_HEADER}
2020-01-01;12,4;28,6;19,1;
2020-01-02;0,0;30,2;20,0;
2020-01-03;;29,0;19,8;
2020-01-04;45,8;24,1;18,2;
2020-01-05;-9999;27,3;18,9;
"""


@pytest.fixture
def bdmep_file(tmp_path):
    path = tmp_path / "dados_83781_D_2020-01-01_2020-01-05.csv"
    path.write_bytes(BDMEP.encode("latin-1"))
    return path


def test_metadata_is_parsed(bdmep_file):
    _, station = read_bdmep(bdmep_file)
    assert station.code == "83781"
    assert station.name.startswith("SAO PAULO")
    assert np.isclose(station.lat, -23.4964, atol=1e-3)
    assert np.isclose(station.lon, -46.6197, atol=1e-3)
    assert np.isclose(station.altitude, 785.66)


def test_columns_are_renamed_to_cmip_names(bdmep_file):
    frame, _ = read_bdmep(bdmep_file)
    assert list(frame.columns) == ["pr", "tasmax", "tasmin"]
    assert frame.index.name == "time"


def test_decimal_comma_and_missing_values(bdmep_file):
    frame, _ = read_bdmep(bdmep_file)
    assert np.isclose(frame["pr"].iloc[0], 12.4)
    assert np.isnan(frame["pr"].iloc[2]), "empty cell must become NaN"
    assert np.isnan(frame["pr"].iloc[4]), "-9999 sentinel must become NaN"
    assert np.isclose(frame["tasmax"].max(), 30.2)


def test_directory_reader_and_stacking(tmp_path, bdmep_file):
    second = tmp_path / "dados_83782_D.csv"
    second.write_bytes(BDMEP.replace("83781", "83782").encode("latin-1"))
    frames, meta = read_bdmep_dir(tmp_path)
    assert set(frames) == {"83781", "83782"}
    assert meta.loc["83781", "n_days"] == 5

    da = stations_to_dataset(frames, meta, "pr")
    assert da.dims == ("time", "station")
    assert "lat" in da.coords and "lon" in da.coords
    assert da.attrs["units"] == "mm/day"


def test_malformed_file_is_skipped_not_fatal(tmp_path, bdmep_file):
    (tmp_path / "broken.csv").write_text("not a bdmep export at all")
    with pytest.warns(UserWarning, match="skipping"):
        frames, _ = read_bdmep_dir(tmp_path)
    assert set(frames) == {"83781"}


def test_missing_variable_raises(bdmep_file):
    frames, meta = read_bdmep_dir(bdmep_file.parent)
    with pytest.raises(KeyError, match="no station"):
        stations_to_dataset(frames, meta, "wsgsmax")
