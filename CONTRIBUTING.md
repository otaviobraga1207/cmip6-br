# Contributing

Contributions are welcome, in English or Portuguese — issues, questions and
pull requests all count.

## Getting set up

```bash
git clone https://github.com/OtavioBraga/cmip6-br
cd cmip6-br
pip install -e ".[dev,io]"
pytest
```

The test suite runs in about ten seconds and downloads nothing: every fixture
is synthetic, and the INMET reader is tested against an inline copy of the real
BDMEP layout. Please keep it that way — a test that needs a 4 GB NetCDF is a
test nobody runs.

## What is especially useful

- **Readers for other Brazilian datasets.** ANA/HidroWeb series, CEMADEN
  stations, Xavier's gridded product, MERGE/CPTEC. The INMET reader in
  `stations.py` is the pattern to follow.
- **Validation against real station data.** If you have run this against
  observations in your basin and it did or did not work, that is worth an issue
  even without code.
- **Additional bias-correction methods.** MBCn and other multivariate methods
  are an obvious gap.
- **Documentation in Portuguese.** The docstrings are in English so the package
  is usable outside Brazil, but worked examples in Portuguese lower the barrier
  for exactly the people who need this most.

## House rules

- `ruff check .` and `ruff format .` before opening a PR.
- New scientific behaviour needs a test that would fail without it. Tests that
  assert a *property* ("QDM preserves the model's change signal") are worth more
  than tests that assert a number someone copied from a previous run.
- Cite the paper in the docstring when you implement a published method.
- Be explicit about what a function does *not* do. The most dangerous thing in
  this field is a result that looks plausible.
