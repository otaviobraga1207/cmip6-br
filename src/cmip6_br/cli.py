"""Command-line interface: ``cmip6-br <command>``."""

from __future__ import annotations

import argparse
import sys

import xarray as xr

from . import __version__, datasets, indices, validation
from .bias import fit as fit_bias
from .grids import BBOXES
from .pipeline import DownscalingConfig, downscale


def _open(path: str, variable: str) -> xr.DataArray:
    ds = xr.open_dataset(path)
    if variable not in ds:
        raise SystemExit(
            f"{path}: variable {variable!r} not found. Available: {list(ds.data_vars)}"
        )
    return ds[variable]


def cmd_regions(_args) -> int:
    for code, box in sorted(BBOXES.items()):
        print(
            f"{code:<3} lon {box.lon_min:8.2f} .. {box.lon_max:8.2f}   "
            f"lat {box.lat_min:7.2f} .. {box.lat_max:7.2f}"
        )
    return 0


def cmd_demo(args) -> int:
    """Run the whole workflow on synthetic data. No downloads, ~2 seconds."""
    data = datasets.demo_bundle()
    obs, hist, ssp = data["obs"], data["hist"], data["ssp"]
    mapper = fit_bias(obs, hist, variable="pr", method=args.method)
    corrected = mapper.adjust(hist)
    corrected_ssp = mapper.adjust(ssp)

    print("Distribution check (all values, mm/day)")
    print(f"{'series':<18}{'mean':>8}{'wet days':>10}{'p99':>8}")
    for label, da in (
        ("observations", obs),
        ("model raw", hist),
        ("model corrected", corrected),
        ("ssp raw", ssp),
        ("ssp corrected", corrected_ssp),
    ):
        print(
            f"{label:<18}{float(da.mean()):8.2f}{float((da >= 1).mean()):10.2f}"
            f"{float(da.quantile(0.99)):8.1f}"
        )
    print("\nSkill over the historical period")
    print(validation.report(obs, hist, corrected).round(3).to_string())

    change = (
        float(indices.rx1day(corrected_ssp).mean()) / float(indices.rx1day(corrected).mean()) - 1
    )
    print(f"\nRx1day change, corrected historical -> corrected scenario: {change:+.1%}")
    if args.output:
        corrected_ssp.to_netcdf(args.output)
        print(f"wrote {args.output}")
    return 0


def cmd_indices(args) -> int:
    pr = _open(args.input, args.variable)
    out = indices.all_precip_indices(pr, freq=args.freq)
    out.to_netcdf(args.output)
    print(f"wrote {args.output} ({len(out.data_vars)} indices, freq={args.freq})")
    return 0


def cmd_downscale(args) -> int:
    cfg = DownscalingConfig(
        region=args.region,
        resolution=args.resolution,
        variable=args.variable,
        method=args.method,
        reference_period=(args.ref_start, args.ref_end)
        if args.ref_start and args.ref_end
        else None,
    )
    result = downscale(
        obs=_open(args.obs, args.variable),
        hist=_open(args.hist, args.variable),
        scenario=_open(args.scenario, args.variable) if args.scenario else None,
        config=cfg,
    )
    written = result.to_netcdf(args.prefix)
    for path in written:
        print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cmip6-br",
        description="Statistical downscaling and ETCCDI indices for Brazil.",
    )
    parser.add_argument("--version", action="version", version=f"cmip6-br {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("regions", help="list the built-in bounding boxes")
    p.set_defaults(func=cmd_regions)

    p = sub.add_parser("demo", help="run the full workflow on synthetic data")
    p.add_argument("--method", choices=["qdm", "eqm"], default="qdm")
    p.add_argument("--output", help="optional NetCDF path for the corrected scenario")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("indices", help="compute ETCCDI precipitation indices")
    p.add_argument("input", help="NetCDF file with daily precipitation in mm/day")
    p.add_argument("output", help="NetCDF file to write")
    p.add_argument("--variable", default="pr")
    p.add_argument("--freq", default="YS", help="pandas offset alias (YS, MS, QS-DEC)")
    p.set_defaults(func=cmd_indices)

    p = sub.add_parser("downscale", help="regrid + bias-correct + index in one pass")
    p.add_argument("obs", help="NetCDF with reference observations on the target grid")
    p.add_argument("hist", help="NetCDF with the model historical run")
    p.add_argument("--scenario", help="NetCDF with an SSP run")
    p.add_argument("--prefix", default="cmip6br", help="output filename prefix")
    p.add_argument("--region", default="SP")
    p.add_argument("--resolution", type=float, default=0.1)
    p.add_argument("--variable", default="pr")
    p.add_argument("--method", choices=["qdm", "eqm"], default="qdm")
    p.add_argument("--ref-start", dest="ref_start")
    p.add_argument("--ref-end", dest="ref_end")
    p.set_defaults(func=cmd_downscale)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
