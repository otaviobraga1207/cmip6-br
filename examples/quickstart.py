"""End-to-end example on synthetic data: python examples/quickstart.py

Reproduces the table in the README and writes two NetCDF files. Replace
``demo_bundle()`` with your own observations and CMIP6 files and the rest of
the script is unchanged.
"""

from cmip6_br import DownscalingConfig, datasets, downscale, validation

data = datasets.demo_bundle()
obs, hist, ssp = data["obs"], data["hist"], data["ssp"]

config = DownscalingConfig(
    region="SP",
    resolution=1.0,  # coarse so the example runs in seconds
    variable="pr",
    method="qdm",
    convert_units=False,  # the demo data is already in mm/day
)
result = downscale(obs=obs, hist=hist, scenario=ssp, config=config)

print("Skill over the historical period")
print(validation.report(obs, hist, result.historical).round(3).to_string(), "\n")

hist_idx = result.indices_historical.mean(dim=("time", "lat", "lon")).to_pandas()
scen_idx = result.indices_scenario.mean(dim=("time", "lat", "lon")).to_pandas()
change = (scen_idx / hist_idx - 1) * 100

print("Annual ETCCDI indices, corrected historical vs corrected SSP")
print(f"{'index':<10}{'historical':>12}{'scenario':>12}{'change %':>11}")
for name in hist_idx.index:
    print(f"{name:<10}{hist_idx[name]:12.1f}{scen_idx[name]:12.1f}{change[name]:+11.1f}")

written = result.to_netcdf("quickstart")
print("\nwrote:", ", ".join(written))

# --- Maps -------------------------------------------------------------------
# Needs the viz extra: pip install "cmip6-br[viz]"
try:
    import matplotlib.pyplot as plt

    from cmip6_br import plots
except ImportError:
    print("\nskipping figures (matplotlib not installed)")
else:
    hist_maps = result.indices_historical
    scen_maps = result.indices_scenario

    plots.map_field(hist_maps["rx1day"], title="Rx1day, corrected historical")
    plt.savefig("quickstart_rx1day.png", dpi=140)

    plots.map_change(
        hist_maps["rx1day"],
        scen_maps["rx1day"],
        title="Rx1day change, scenario vs historical",
    )
    plt.savefig("quickstart_rx1day_change.png", dpi=140)

    plots.qq_plot(obs, hist, result.historical)
    plt.savefig("quickstart_qq.png", dpi=140)

    print("wrote: quickstart_rx1day.png, quickstart_rx1day_change.png, quickstart_qq.png")
