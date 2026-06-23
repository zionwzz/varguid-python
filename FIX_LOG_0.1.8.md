# varguid 0.1.8 fix log

Audit date: 2026-06-22

Inputs reviewed:

- Python source archive `varguid_py_0.1.7_source.zip`.
- R source package `varGuid_0.1.5.tar.gz`, especially `R/irls.R` and `R/add.R`.

## Corrections

### 1. DataFrame prediction schema

**Problem:** prediction used positional columns. Reordering the same DataFrame
columns silently changed predictions.

**Correction:** fitted DataFrame names are retained. Prediction reorders a
complete matching schema and raises a clear error for missing, additional, or
ambiguous string-equivalent names.

**Regression coverage:** `tests/test_validation.py`.

### 2. Formula intercept handling

**Problem:** `lmv_formula("y ~ 0 + x1 + x2", ...)` still inserted an intercept.

**Correction:** the Patsy design matrix determines `fit_intercept`; no-intercept
formulas now remain no-intercept during fitting, summaries, and prediction.
Design information is retained for transformed and categorical predictors.

**Regression coverage:** `tests/test_basic.py` and formula prediction tests.

Formula fitting and prediction now use Patsy's `NA_action="raise"`, preventing
missing values from silently reducing the number of fitted or predicted rows.

### 3. Non-lasso R algorithm parity

**Problem:** the Python implementation clipped fitted variance values and used a
different model update order once the coefficient-change threshold was met.

**Correction:** variance fits are no longer clipped. Candidate mean models are
assigned before the convergence branch, matching the attached `R/irls.R`
control flow. The last committed model remains the returned variance-guided fit.
An exactly zero variance fit receives a finite uniform-weight fallback.

**Regression coverage:** `tests/test_r_compatibility.py` contains an independent
translation of the attached R non-lasso procedure and exercises the
post-convergence branch.

### 4. Lasso scaling and randomness

**Problem:** predictors were sent to `LassoCV` without glmnet-like
standardization, and `random_state` had no practical effect under the previous
solver configuration.

**Correction:** mean and variance predictors are standardized using the
applicable sample weights, coefficients are mapped back to original units, and
cross-validation uses shuffled `KFold` splits generated from a deterministic
seed stream. The functional API now accepts the Scientific Python `rng`
keyword, normalizes it with `numpy.random.default_rng`, and retains
`random_state` as a deprecated alias. The scikit-learn estimator keeps its
conventional `random_state` parameter. The R default of 3 folds for at most 80
rows and 10 otherwise is retained when `cv_folds=10`.

**Limitation:** scikit-learn and `glmnet` use different solvers and lambda-grid
implementations. Sparse coefficients are reproducible within Python but are not
claimed to be bit-for-bit identical to R.

**Regression coverage:** lasso smoke, reproducibility, and unit-invariance tests.

### 5. Scikit-learn estimator contract

**Problem:** the wrapper had the wrong mixin order, raised the wrong pre-fit
exception, omitted `n_features_in_`, and did not fully validate prediction
features.

**Correction:** `VarGuidRegressor` now inherits `RegressorMixin` before
`BaseEstimator`, uses scikit-learn validation helpers, records standard fitted
attributes, and raises `NotFittedError` before fit.

**Regression coverage:** the full `sklearn.utils.estimator_checks.check_estimator`
suite plus focused wrapper tests.

### 6. Packaging, CI, and release security

The Patsy lower bound was raised from `1.0` to `1.0.1` because Patsy 1.0.0 is a
yanked release.

**Problem:** the supplied archive included old `dist/`, cache folders, bytecode,
and egg-info; configured lint/type checks failed; release instructions relied on
manual token-based uploads.

**Correction:** lint and type issues were resolved, the nox build glob was
corrected, source packaging was cleaned, and CI now tests supported Python
versions. Release publishing uses a separate least-privilege OIDC job through a
protected `pypi` environment. Third-party actions are pinned to full immutable
commit SHAs and tracked by Dependabot. A separate credential-limited job
generates and verifies SLSA build-provenance attestations before the PyPI
publishing job can run. The wheel includes a PEP 561 `py.typed` marker.

### 7. Synthetic-data compatibility

**Problem:** an early validation change rejected `n_features < 10`, whereas the
0.1.7 helper and attached R generator expand such requests to ten predictors.

**Correction:** positive feature counts below ten are again expanded to ten.
Input validation, the preferred `rng` interface, and valid negative
equicorrelation support are retained.

**Regression coverage:** `tests/test_validation.py`.

## Validation scope

The final release was validated with pytest, Ruff lint and format checks, mypy,
package build, Twine metadata checks, archive-content inspection, wheel
installation, source-distribution installation, and installed-package execution
of the documented examples. Exact commands and outputs are recorded in
`VALIDATION_LOG_0.1.8.txt`.

The available local interpreter was CPython 3.13.5. Python 3.12 and 3.14 are
configured in CI but were not available for local execution in this audit.
An R interpreter was not available locally; non-lasso parity was checked against
an independent Python translation of the attached R source rather than by
executing R itself.
