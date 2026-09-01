import pytest

from cmip6_br.cli import main


def test_regions_lists_sao_paulo(capsys):
    assert main(["regions"]) == 0
    assert "SP" in capsys.readouterr().out


def test_demo_runs_and_reports_skill(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "observations" in out and "corrected" in out and "Rx1day change" in out


def test_indices_roundtrip(tmp_path, obs):
    src = tmp_path / "pr.nc"
    dst = tmp_path / "indices.nc"
    obs.to_netcdf(src)
    assert main(["indices", str(src), str(dst)]) == 0
    assert dst.stat().st_size > 0


def test_missing_variable_exits_cleanly(tmp_path, obs):
    src = tmp_path / "pr.nc"
    obs.to_netcdf(src)
    with pytest.raises(SystemExit, match="not found"):
        main(["indices", str(src), str(tmp_path / "o.nc"), "--variable", "nope"])


def test_map_command_writes_an_image(tmp_path, obs):
    src = tmp_path / "pr.nc"
    out = tmp_path / "map.png"
    obs.to_netcdf(src)
    assert main(["map", str(src), str(out)]) == 0
    assert out.stat().st_size > 5000


def test_map_change_command(tmp_path, obs):
    a, b = tmp_path / "a.nc", tmp_path / "b.nc"
    out = tmp_path / "change.png"
    obs.to_netcdf(a)
    (obs * 1.2).to_netcdf(b)
    assert main(["map", str(a), str(out), "--change", str(b), "--title", "change"]) == 0
    assert out.stat().st_size > 5000


def test_figures_command_writes_the_whole_set(tmp_path):
    assert main(["figures", str(tmp_path / "figs"), "--dpi", "60"]) == 0
    written = sorted((tmp_path / "figs").glob("*.png"))
    assert len(written) == 5
