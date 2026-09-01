# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-01

First public release.

### Added
- `bias`: empirical quantile mapping (EQM) and quantile delta mapping (QDM),
  with monthly grouping, multiplicative and additive modes, and wet-day
  frequency adaptation for precipitation.
- `indices`: ETCCDI precipitation indices (PRCPTOT, Rx1day, RxNday, SDII,
  R10mm, R20mm, RNNmm, R95p, R99p, CDD, CWD) and basic temperature extremes
  (TXx, TNn, DTR, SU, TR), with reference-period thresholds.
- `regrid`: coordinate normalisation for raw CMIP6 output (0-360 longitudes,
  descending latitude, non-standard coordinate names), bounding-box subsetting,
  bilinear/nearest regridding with coastal edge filling, area-weighted means.
- `grids`: padded bounding boxes for Brazil and 22 states, regular target grids.
- `stations`: reader for INMET BDMEP CSV exports, including the directory
  batch reader and stacking into a `(time, station)` DataArray.
- `units`: CMIP6 flux and kelvin conversions.
- `validation`: bias, MAE, RMSE, Pearson r, KGE, Perkins skill score, and a
  raw-versus-corrected comparison table.
- `pipeline`: `downscale()` running the whole workflow from one config object.
- `cmip6-br` command-line interface with `regions`, `demo`, `indices` and
  `downscale` subcommands.
