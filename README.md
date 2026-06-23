# {varguid}: Variance-Guided Regression Improving Upon OLS and ANOVA for Python

`varguid` is a Python implementation of the **stage-1** variance-guided
regression method described in:

> Liu S. and Lu M. (2026). *Variance-Guided Regression for Heteroscedastic
> Data with a Grouping-Based Extension for Nonlinear Prediction*.
> **Statistics in Medicine** 45(13-14):e70632.
> DOI: 10.1002/sim.70632.

The package implements the global linear mean-variance model from Section 2 of
the paper and mirrors the stage-1 scope of the attached R package `varGuid`
0.1.5. The grouping-based nonlinear extension from Section 3 is not included.

## Features

- Iteratively reweighted least squares for non-sparse fits.
- Iteratively reweighted lasso using scikit-learn.
- Baseline and variance-guided prediction from the same fitted result.
- Weighted least-squares and HC0-HC3 coefficient summaries for non-lasso fits.
- Matrix, pandas DataFrame, Patsy formula, and scikit-learn-style interfaces.
- Packaged `cobra2d` data and a reproducible synthetic-data generator.

## Installation

`varguid` 0.1.8 requires Python 3.12 or newer.

```bash
python -m pip install varguid
```

For local development:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
from varguid import lmv, load_cobra2d

# Packaged data from the attached R release
data = load_cobra2d()
train = data.iloc[:-200].copy()
y = train["y"]
X = train.drop(columns="y")

fit = lmv(X, y, M=10, lasso=False)

baseline_pred = fit.predict(X, model="baseline")
varguid_pred = fit.predict(X, model="varGuid")

print(fit.beta[:5])
print(baseline_pred[:3])
print(varguid_pred[:3])
print(fit.summary_frame(cov_type="HC3").head())
```

The top-level helper has the same behavior:

```python
from varguid import predict

varguid_pred = predict(fit, X, model="varGuid")
```

The R-compatible `prd()` name remains available as a deprecated alias.

## Sparse fit

```python
from varguid import lmv, load_cobra2d

data = load_cobra2d().iloc[:120]
X = data.drop(columns="y")
y = data["y"]

fit = lmv(
    X,
    y,
    M=3,
    lasso=True,
    cv_folds=5,
    rng=42,
)
pred = fit.predict(X, model="varGuid")
print(fit.beta)
print(pred[:3])
```

The lasso implementation standardizes predictors and uses shuffled,
reproducible cross-validation folds. It follows the R package's high-level
procedure but uses scikit-learn rather than `glmnet`, so sparse coefficients
are not expected to be bit-for-bit identical across languages.

The functional API follows Scientific Python SPEC 7: pass an integer or a
`numpy.random.Generator` through `rng`. The older `random_state` keyword is
accepted with a deprecation warning. `VarGuidRegressor` retains
`random_state`, as expected by scikit-learn estimators.

## Formula interface

```python
from varguid import lmv_formula, load_cobra2d

data = load_cobra2d()
fit = lmv_formula("y ~ x1 + x2 + x3 + x4 + x5", data=data, M=5)
pred = fit.predict(data.iloc[:5])
print(fit.summary())
```

No-intercept formulas are honored:

```python
fit_no_intercept = lmv_formula("y ~ 0 + x1 + x2 + x3", data=data, M=5)
```

Formula-based prediction expects a DataFrame containing the original formula
variables. Patsy reconstructs transformations and categorical encodings from
the fitted design information.

## Scikit-learn estimator

```python
from varguid import VarGuidRegressor, load_cobra2d

data = load_cobra2d()
X = data.drop(columns="y")
y = data["y"]

model = VarGuidRegressor(max_iter=5, use_lasso=False)
model.fit(X, y)
pred = model.predict(X.iloc[:5])
print(model.summary_frame(cov_type="HC1").head())
```

The estimator records `n_features_in_`, raises `NotFittedError` before fitting,
and passes scikit-learn's estimator checks. For DataFrame fits, prediction
columns are reordered to the fitted order; missing or unexpected columns are
rejected rather than silently producing incorrect predictions.

## Reproducibility and R compatibility

The non-lasso update order follows the attached R `R/irls.R` implementation,
including the model update that occurs before the convergence branch. Fitted
variance values are not clipped. A numerically exact zero-variance fit receives
a safe uniform-weight fallback instead of producing non-finite values.

The automated tests include an independent translation of the attached R
non-lasso algorithm, direct statsmodels comparisons, lasso reproducibility and
scale-invariance checks, formula tests, DataFrame schema tests, README example
execution, and the full scikit-learn estimator contract.

## Development checks

```bash
python -m pytest
ruff check src tests
ruff format --check src tests
mypy src tests
check-manifest
python -m build
python -m twine check dist/*
```

Equivalent `nox` sessions are available:

```bash
nox -s tests lint build
```

See `FIX_LOG_0.1.8.md` and `VALIDATION_LOG_0.1.8.txt` for the release audit.

## Publishing

The release workflow builds distributions in a separate job and publishes from
a protected `pypi` GitHub environment through PyPI Trusted Publishing. See
`RELEASING.md` for the one-time repository configuration and release steps.

## Citation

If you use `varguid`, please cite both the Python software and the method paper.
The root-level `CITATION.cff` file is the authoritative machine-readable
software citation. The version-specific DOI for `varguid` 0.1.8 is
https://doi.org/10.5281/zenodo.20816141.

### Python software

```bibtex
@software{wang_lu_2026_varguid_python,
  author    = {Wang, Zihao and Lu, Min},
  title     = {{varguid}: Variance-Guided Regression Improving Upon OLS and ANOVA for Python},
  version   = {0.1.8},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20816141},
  url       = {https://doi.org/10.5281/zenodo.20816141}
}
```

### Method paper

```bibtex
@article{liu_lu_2026_varguid,
  author  = {Liu, Sibei and Lu, Min},
  title   = {Variance-Guided Regression for Heteroscedastic Data With a Grouping-Based Extension for Nonlinear Prediction},
  journal = {Statistics in Medicine},
  volume  = {45},
  number  = {13-14},
  pages   = {e70632},
  year    = {2026},
  doi     = {10.1002/sim.70632}
}
```
