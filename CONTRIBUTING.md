# Contributing

## Development environment

Create and activate a Python 3.12+ virtual environment, then install the package
and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Required checks

Run these commands from the repository root before submitting a change:

```bash
python -m pytest
ruff check src tests
ruff format --check src tests
mypy src tests
check-manifest
python -m build
python -m twine check dist/*
```

The same checks are grouped into nox sessions:

```bash
nox -s tests lint build
```

Install the local hooks once with `pre-commit install`, then run
`pre-commit run --all-files` before committing.

## Regression expectations

Changes to the fitting algorithm should include tests for:

- agreement with the update order in the attached R `R/irls.R` implementation;
- DataFrame feature-name safety;
- formula intercept behavior;
- lasso reproducibility and scaling behavior, when applicable; and
- the scikit-learn estimator contract.

Sparse fits use scikit-learn rather than R's `glmnet`; document any intentional
behavioral difference instead of asserting bit-for-bit cross-language parity.

## Releases

Update both `pyproject.toml` and `src/varguid/__init__.py`, add an entry to
`CHANGELOG.md`, and follow `RELEASING.md`. Do not upload from a developer
machine when the Trusted Publishing workflow is available.
