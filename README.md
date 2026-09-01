# cmip6-br

**Statistical downscaling and climate-extreme indices for CMIP6 projections over Brazil.**

[![CI](https://github.com/OtavioBraga/cmip6-br/actions/workflows/ci.yml/badge.svg)](https://github.com/OtavioBraga/cmip6-br/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Every climate risk study in Brazil starts by rewriting the same code: unwrap the
0–360 longitudes, remember the 86400 factor, parse a Latin-1 INMET export with
comma decimals, bias-correct against stations, compute the ETCCDI indices. It
usually lives in one notebook, it is rarely reviewed, and it dies with the
project.

`cmip6-br` is that layer, written once, tested, and installable.

```bash
pip install cmip6-br            # core
pip install "cmip6-br[viz]"     # + maps and figures
pip install "cmip6-br[geo]"     # + GeoTIFF export and boundary overlays
```

## Sixty-second tour

No downloads, no credentials — the demo runs on synthetic data with a
deliberately model-like bias (rains too often, too weakly):

```bash
cmip6-br demo
```

```
Distribution check (all values, mm/day)
series                mean  wet days     p99
observations          4.13      0.30    48.0
model raw             3.69      0.47    29.7
model corrected       4.11      0.30    47.9
ssp raw               3.92      0.47    31.8
ssp corrected         4.39      0.30    51.9

Skill over the historical period
            bias    mae    rmse  pearson_r    kge  perkins_ss
raw       -0.438  5.936  11.315      0.089  0.016       0.928
corrected -0.025  6.628  13.532      0.066  0.066       0.999

Rx1day change, corrected historical -> corrected scenario: +10.7%
```

Read that table carefully, because it is the whole argument for quantile
mapping and its whole limitation in one place. The wet-day frequency goes from
0.47 to the observed 0.30, the 99th percentile from 29.7 mm to 47.9 mm against
an observed 48.0 mm, and the PDF overlap reaches 0.999. The correlation does
**not** improve — and it never will. Quantile mapping fixes *distributions*, not
the day-to-day timing of rainfall. If your application needs the right rain on
the right day, this is the wrong method.

## In Python

```python
from cmip6_br import bias, indices, datasets

data = datasets.demo_bundle()          # swap for your own DataArrays

mapper = bias.fit(data["obs"], data["hist"], variable="pr", method="qdm")
corrected = mapper.adjust(data["ssp"])

annual = indices.all_precip_indices(corrected)
print(annual["rx1day"].mean(dim=("lat", "lon")).to_pandas())
```

Or the whole workflow — subset, regrid, bias-correct, index — in one call:

```python
from cmip6_br import DownscalingConfig, downscale

config = DownscalingConfig(
    region="SP",                 # any state code, "BR", or an explicit BBox
    resolution=0.1,              # degrees, roughly 11 km
    variable="pr",
    method="qdm",                # preserves the model's change signal
    reference_period=("1995-01-01", "2014-12-31"),
)
result = downscale(obs=obs, hist=hist, scenario=ssp585, config=config)
result.to_netcdf("cantareira_ssp585")
```

## With real data

```python
import xarray as xr
from cmip6_br import DownscalingConfig, downscale, read_bdmep_dir, stations_to_dataset

# 1. Observations: INMET BDMEP exports, straight from the portal download
frames, meta = read_bdmep_dir("data/inmet/")
station_pr = stations_to_dataset(frames, meta, "pr")

# 2. Model: any CMIP6 daily file. Units and coordinates are normalised for you.
hist = xr.open_mfdataset("data/cmip6/pr_day_*_historical_*.nc")["pr"]
ssp = xr.open_mfdataset("data/cmip6/pr_day_*_ssp585_*.nc")["pr"]

result = downscale(obs=gridded_obs, hist=hist, scenario=ssp,
                   config=DownscalingConfig(region="SP", resolution=0.1))
```

Bias correction needs observations **on the target grid**. Station data must be
interpolated first — the honest options are a gridded observational product
(Xavier et al. for Brazil, or ERA5/MERGE as a lower-quality fallback), or your
own interpolation of the station network. `stations.py` gets the stations into
xarray; it does not pretend that a point is a grid cell.

## Maps

```bash
cmip6-br figures out/          # the whole figure set from the demo data
cmip6-br map indices.nc map.png --variable rx1day
cmip6-br map hist.nc change.png --variable rx1day --change ssp.nc
cmip6-br map indices.nc map.png --geotiff rx1day.tif --epsg 31983   # for QGIS
```

```python
from cmip6_br import plots

plots.map_field(result.indices_scenario["rx1day"], title="Rx1day, SSP5-8.5 2041-2060")
plots.map_change(hist_idx["rx1day"], scen_idx["rx1day"])         # % change, centred on zero
plots.map_panel({"Rx1day (mm)": ..., "CDD (days)": ...})          # small multiples
plots.qq_plot(obs, raw, corrected)                                # did the correction work?
plots.seasonal_cycle({"observations": obs, "corrected": corrected})
plots.save_geotiff(field, "rx1day.tif", epsg=31983)               # SIRGAS 2000 / UTM 23S
```

Three rules are enforced by the code rather than left to the user:

- **Magnitude fields get one hue, light to dark. Never a rainbow.** `jet` and
  friends invent boundaries the data does not have and hide the ones it does —
  a hazard map is exactly where that does damage.
- **Change maps are always centred on zero**, with a neutral grey midpoint and
  symmetric limits, so equal wetting and drying carry equal visual weight.
- **Every colour bar carries units**, taken from the field itself, so a panel
  mixing millimetres and days labels each one correctly.

Series in the line figures are separated by colour *and* line style *and*
marker, so the figures survive greyscale printing and colour-vision deficiency.
`plots.qq_plot` is the figure to put in front of a reviewer: the raw model bends
away from the 1:1 line in the upper tail, the corrected one sits on it.

Overlay administrative limits by passing any GeoDataFrame or vector file:
`map_field(..., boundaries="ottobacias_n5.gpkg")`.

## What is in the box

| Module | What it does |
|---|---|
| `bias` | EQM and QDM quantile mapping, monthly grouping, wet-day frequency adaptation |
| `indices` | ETCCDI: PRCPTOT, Rx1day, RxNday, SDII, R10mm, R20mm, R95p, R99p, CDD, CWD, TXx, TNn, DTR, SU, TR |
| `regrid` | CMIP6 coordinate normalisation, bbox subsetting, bilinear regridding with coastal edge fill |
| `grids` | Padded bounding boxes for Brazil and 22 states, regular target grids |
| `stations` | INMET BDMEP CSV reader (Latin-1, `;`, comma decimals, metadata block, `-9999`) |
| `units` | kg m⁻² s⁻¹ → mm/day, K → °C, dataset-wide `harmonize()` |
| `validation` | bias, MAE, RMSE, Pearson r, KGE, Perkins skill score, raw-vs-corrected table |
| `plots` | Maps (sequential and diverging), small multiples, Q-Q and seasonal-cycle figures, GeoTIFF export |
| `pipeline` | `downscale()` — the whole workflow from one config |

## Method notes

**EQM or QDM?** Empirical quantile mapping (Themeßl et al., 2011) maps the
model distribution onto the observed one. It is the right tool for validating
over the historical period and the wrong one for projections, because a future
value beyond the calibration range gets clipped back into it — quietly
truncating the change signal you are trying to measure. Quantile delta mapping
(Cannon et al., 2015) corrects the distribution while preserving the model's own
relative change in each quantile, and is the default here.

**Why monthly grouping?** A single transfer function fitted across the whole
year mixes the Southeast's wet summer with its dry winter and corrects neither.
Per-month transfer functions cost nothing and are the default (`group="month"`).

**Why wet-day frequency adaptation?** Climate models drizzle: they produce far
too many days with 0.1–1 mm and too few intense ones. Mapping quantiles without
fixing that first leaves a persistent wet bias in every dry-spell index — CDD
worst of all. `bias.fit(..., variable="pr")` finds the modelled threshold that
reproduces the observed dry-day fraction and zeroes everything below it
(Themeßl et al., 2012).

**Fixed thresholds for R95p/R99p.** Percentile-based indices are only
comparable across periods if the threshold comes from the reference period.
`downscale()` does this for you; if you call `indices.r95p()` directly, pass the
threshold explicitly.

## Limitations, stated plainly

- Bias correction cannot fix a model that gets the circulation wrong. It
  reshapes distributions; it does not add skill that was never there.
- The regridding is bilinear point interpolation, not conservative remapping.
  For flux-conserving remapping onto a coarser grid, use `xesmf`.
- Every method here is univariate. Inter-variable consistency (temperature and
  precipitation together) needs a multivariate method such as MBCn.
- The maps are plain lat/lon (plate carrée) with a cosine-of-latitude aspect
  correction, not a real projection. For a publication map in SIRGAS 2000 /
  UTM 23S, export a GeoTIFF and finish it in QGIS.
- A 0.1° output grid is not 11 km of real information. The added detail comes
  from the observations you calibrated against, not from the GCM.
- The bounding boxes in `grids.py` are padded rectangles for cheap subsetting,
  not official IBGE boundaries.

## References

- Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM
  precipitation by quantile delta mapping. *Journal of Climate*, 28(17),
  6938–6959.
- Themeßl, M. J., Gobiet, A., & Leuprecht, A. (2011). Empirical-statistical
  downscaling and error correction of daily precipitation from regional climate
  models. *International Journal of Climatology*, 31(10), 1530–1544.
- Themeßl, M. J., Gobiet, A., & Heinrich, G. (2012). Empirical-statistical
  downscaling and error correction of regional climate models and its impact on
  the climate change signal. *Climatic Change*, 112(2), 449–468.
- Zhang, X. et al. (2011). Indices for monitoring changes in extremes based on
  daily temperature and precipitation data. *WIREs Climate Change*, 2(6),
  851–870.
- Gupta, H. V. et al. (2009). Decomposition of the mean squared error and NSE
  performance criteria. *Journal of Hydrology*, 377(1–2), 80–91.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Readers for other Brazilian datasets
(ANA/HidroWeb, CEMADEN, MERGE/CPTEC) and validation reports from real basins are
the most useful contributions right now. Issues in Portuguese are welcome.

## License

MIT — see [LICENSE](LICENSE).

---

## Em português

`cmip6-br` faz o trabalho repetitivo de todo estudo de risco climático no
Brasil: normaliza as coordenadas do CMIP6 (longitude 0–360, latitude
descendente, unidades de fluxo), lê os CSVs do BDMEP/INMET como eles realmente
vêm (Latin-1, `;`, vírgula decimal, cabeçalho de metadados, `-9999`),
corrige viés por *quantile mapping* com função de transferência mensal e
adaptação da frequência de dias chuvosos, e calcula os índices ETCCDI.

Para os mapas: `cmip6-br figures out/` gera o conjunto completo a partir dos
dados sintéticos, e `plots.save_geotiff(campo, "rx1day.tif", epsg=31983)`
exporta em SIRGAS 2000 / UTM 23S para você terminar no QGIS.

Comece por `cmip6-br demo` — roda em segundos, sem baixar nada, e mostra o que
a correção de viés faz e o que ela não faz. Depois troque `datasets.demo_bundle()`
pelos seus dados.

Duas escolhas que valem entender antes de usar em produção:

- **QDM, não EQM, para projeções.** O *quantile mapping* empírico corta o sinal
  de mudança nos extremos ao trazer valores futuros de volta para dentro do
  intervalo de calibração. O *quantile delta mapping* preserva a mudança
  relativa de cada quantil — é o padrão aqui.
- **A correção de viés precisa de observação na grade alvo.** Dados de estação
  precisam ser interpolados antes; o módulo `stations` coloca as estações no
  xarray, mas não finge que um ponto é uma célula de grade.

Issues e pull requests em português são bem-vindos.
