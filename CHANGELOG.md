# Changelog

All notable changes to this project are documented here.

## 0.1.8 - 2026-06-22

### Fixed

- Preserved DataFrame feature identity during prediction by reordering known
  columns and rejecting missing or unexpected columns.
- Honored no-intercept Patsy formulas such as `y ~ 0 + x1 + x2`.
- Made formula fitting and prediction reject missing values instead of silently
  dropping affected rows through Patsy.
- Matched the attached R non-lasso update order after the convergence threshold
  is reached and removed variance-prediction clipping.
- Added a safe fallback for an exactly zero fitted variance model.
- Standardized lasso features and added shuffled reproducible CV folds.
- Added the Scientific Python `rng` interface for functional lasso fits while
  retaining `random_state` as a deprecated compatibility alias.
- Corrected scikit-learn estimator behavior: mixin order, `NotFittedError`,
  `n_features_in_`, feature-name checks, and fitted attributes.
- Restored keyword compatibility for `predict(object=...)` and `prd(object=...)`.
- Corrected the nox build session so Twine receives expanded artifact paths.
- Preserved the historical `generate_cobra2d(n_features < 10)` behavior by
  expanding the generated design to ten predictors.

### Added

- Independent regression tests for the attached R non-lasso algorithm.
- Full scikit-learn estimator checks.
- README example, formula, schema-validation, lasso scale-invariance, RNG, and
  exact-fit regression tests.
- A Trusted Publishing release workflow with least-privilege permissions and
  immutable GitHub Action commit pins.
- A separate SLSA build-provenance attestation and verification job, plus
  Dependabot tracking for pinned GitHub Actions.
- A PEP 561 `py.typed` marker and typed-package classifier.
- Release, contribution, fix-log, and validation documentation.

### Changed

- Adopted the publication title `{varguid}: Variance-Guided Regression Improving Upon OLS and ANOVA for Python`.
- Updated canonical repository links to `zionwzz/varguid-python` and added machine-readable and BibTeX citation metadata.
- Updated a test annotation for compatibility with mypy 2.1.0; runtime behavior is unchanged.
- Raised the supported Python floor to 3.12 and refreshed core dependency floors
  in line with the project's Scientific Python support policy.
- CI now targets Python 3.12, 3.13, and 3.14 and separates tests, quality checks,
  distribution building, and publishing.
- Clean submitted source archives no longer contain generated caches, bytecode,
  egg-info directories, or previously built artifacts. Standard distribution metadata
  generated during an sdist build is retained as required by the build backend.

### Compatibility note

Sparse fits use scikit-learn's lasso solver. The implementation now mirrors the
R workflow more closely through standardization and CV behavior, but it is not
bit-for-bit equivalent to `glmnet`.

## 0.1.7

- Previous Python source release supplied for audit.
